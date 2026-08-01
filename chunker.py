from typing import List, Dict, Any
from uuid import uuid4
import tiktoken
import re
from models import Chunk, ChunkType
from pdf_parser import LayoutElement
import config
import logging

logger = logging.getLogger(__name__)


class EnhancedContextualChunker:

    def __init__(self):
        self._encoding = None
        self.parent_size = config.CHUNK_SIZE_PARENT
        self.child_size = config.CHUNK_SIZE_CHILD
        self.overlap = config.CHUNK_OVERLAP
        self._compile_numeric_patterns()

    def _ensure_encoding(self):
        if self._encoding is None:
            self.encoding = tiktoken.get_encoding("o200k_base")
            self._encoding = self.encoding

    def _compile_numeric_patterns(self):
        self.numeric_patterns = [
            (re.compile(r'-?\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:[KMBkmb](?:illion)?)?'), 'number'),
            (re.compile(r'-?\d+(?:\.\d+)?%'), 'percentage'),
            (re.compile(r'\b(?:19|20)\d{2}\b'), 'year'),
            (re.compile(r'\b(?:Q[1-4]|FY)\s*\d{2,4}\b', re.IGNORECASE), 'fiscal_period'),
            (re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'), 'date'),
        ]

    def _extract_numbers(self, text: str) -> List[Dict[str, Any]]:
        numbers = []
        seen = set()
        for pattern, num_type in self.numeric_patterns:
            for match in pattern.finditer(text):
                v = match.group(0)
                if v not in seen:
                    seen.add(v)
                    numbers.append({"value": v, "type": num_type, "position": match.start()})
        numbers.sort(key=lambda x: x["position"])
        return numbers

    def create_chunks(self, document_id: str, elements: List[LayoutElement]) -> List[Chunk]:
        all_chunks = []
        sections = self._group_into_sections(elements)
        for section_idx, section in enumerate(sections):
            parent_chunk = self._create_parent_chunk(document_id, section, section_idx)
            all_chunks.append(parent_chunk)
            all_chunks.extend(self._create_child_chunks(document_id, parent_chunk, section))
        return all_chunks

    def _group_into_sections(self, elements: List[LayoutElement]) -> List[List[LayoutElement]]:
        sections = []
        current_section = []
        current_tokens = 0

        for element in elements:
            element_tokens = self._count_tokens(element.content)
            should_split = False

            if current_tokens + element_tokens > self.parent_size and current_section:
                should_split = True
            elif element.element_type == ChunkType.TABLE and current_section and element_tokens <= self.parent_size:
                should_split = True
            elif element.element_type == ChunkType.HEADER and current_section and current_tokens > self.parent_size * 0.5:
                should_split = True

            if should_split:
                sections.append(current_section)
                overlap_elements = self._get_overlap_elements(current_section) if self.overlap > 0 else []
                current_section = overlap_elements + [element]
                current_tokens = sum(self._count_tokens(e.content) for e in current_section)
            else:
                current_section.append(element)
                current_tokens += element_tokens

        if current_section:
            sections.append(current_section)
        return sections

    def _get_overlap_elements(self, section: List[LayoutElement]) -> List[LayoutElement]:
        overlap_elements = []
        overlap_tokens = 0
        for element in reversed(section):
            et = self._count_tokens(element.content)
            if overlap_tokens + et > self.overlap:
                break
            overlap_elements.insert(0, element)
            overlap_tokens += et
        return overlap_elements

    def _create_parent_chunk(self, document_id: str, section: List[LayoutElement], section_idx: int) -> Chunk:
        combined_text = "\n\n".join(e.content for e in section)
        first_elem = section[0]
        type_counts: Dict[str, int] = {}
        for elem in section:
            type_counts[elem.element_type.value] = type_counts.get(elem.element_type.value, 0) + 1
        primary_type = max(type_counts, key=type_counts.get)
        return Chunk(
            id=uuid4(),
            document_id=document_id,
            parent_id=None,
            text=combined_text,
            chunk_type=ChunkType(primary_type),
            format_type=first_elem.format_type,
            page_number=first_elem.page,
            bbox=first_elem.bbox,
            token_count=self._count_tokens(combined_text),
            metadata={
                "is_parent": True,
                "section_index": section_idx,
                "num_elements": len(section),
                "element_types": [e.element_type.value for e in section],
                "pages": list(set(e.page for e in section)),
                "has_table": any(e.element_type == ChunkType.TABLE for e in section),
                "has_code": any(e.element_type == ChunkType.CODE for e in section),
                "numbers": self._extract_numbers(combined_text),
            }
        )

    def _create_child_chunks(self, document_id: str, parent: Chunk, section: List[LayoutElement]) -> List[Chunk]:
        children = []
        for elem_idx, element in enumerate(section):
            if self._count_tokens(element.content) <= self.child_size:
                children.append(self._create_single_child(document_id, parent, element, elem_idx))
            else:
                children.extend(self._split_large_element(document_id, parent, element, elem_idx))
        return children

    def _create_single_child(self, document_id: str, parent: Chunk, element: LayoutElement, elem_idx: int) -> Chunk:
        return Chunk(
            id=uuid4(),
            document_id=document_id,
            parent_id=parent.id,
            text=element.content,
            chunk_type=element.element_type,
            format_type=element.format_type,
            page_number=element.page,
            bbox=element.bbox,
            token_count=self._count_tokens(element.content),
            metadata={
                "is_parent": False,
                "parent_id": str(parent.id),
                "element_index": elem_idx,
                "numbers": self._extract_numbers(element.content),
                **element.metadata
            }
        )

    def _split_large_element(self, document_id: str, parent: Chunk, element: LayoutElement, elem_idx: int) -> List[Chunk]:
        if element.element_type == ChunkType.TABLE and '|' in element.content:
            sub_chunks = self._split_table(element.content)
        else:
            sub_chunks = self._split_by_sentences(element.content)

        chunks = []
        for sub_idx, chunk_text in enumerate(sub_chunks):
            if not chunk_text.strip():
                continue
            chunks.append(Chunk(
                id=uuid4(),
                document_id=document_id,
                parent_id=parent.id,
                text=chunk_text,
                chunk_type=element.element_type,
                format_type=element.format_type,
                page_number=element.page,
                bbox=element.bbox,
                token_count=self._count_tokens(chunk_text),
                metadata={
                    "is_parent": False,
                    "parent_id": str(parent.id),
                    "element_index": elem_idx,
                    "sub_index": sub_idx,
                    "numbers": self._extract_numbers(chunk_text),
                    **element.metadata
                }
            ))
        return chunks

    def _split_table(self, table_text: str) -> List[str]:
        lines = table_text.split('\n')
        header_lines = []
        data_lines = []
        for i, line in enumerate(lines):
            if i == 0 or (i == 1 and ('|---' in line or '|-' in line)):
                header_lines.append(line)
            else:
                data_lines.append(line)

        chunks = []
        current_chunk = header_lines.copy()
        current_tokens = sum(self._count_tokens(l) for l in current_chunk)

        for line in data_lines:
            lt = self._count_tokens(line)
            if current_tokens + lt > self.child_size and len(current_chunk) > len(header_lines):
                chunks.append('\n'.join(current_chunk))
                current_chunk = header_lines.copy() + [line]
                current_tokens = sum(self._count_tokens(l) for l in current_chunk)
            else:
                current_chunk.append(line)
                current_tokens += lt

        if len(current_chunk) > len(header_lines):
            chunks.append('\n'.join(current_chunk))
        return chunks if chunks else [table_text]

    def _split_by_sentences(self, text: str) -> List[str]:
        sentences = self._split_sentences(text)
        if not sentences:
            return [text]

        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            st = self._count_tokens(sentence)
            if current_tokens + st > self.child_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                overlap_count = min(2, len(current_chunk))
                current_chunk = current_chunk[-overlap_count:] + [sentence] if self.overlap > 0 else [sentence]
                current_tokens = self._count_tokens(' '.join(current_chunk))
            else:
                current_chunk.append(sentence)
                current_tokens += st

        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        text = re.sub(r'\b(Dr|Mr|Mrs|Ms|Inc|Ltd|Co|Corp|vs|etc|e\.g|i\.e|Jr|Sr|Prof|Rev)\.\s',
                      r'\1<ABBREV> ', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+)\.(\d+)', r'\1<DECIMAL>\2', text)
        text = re.sub(r'([.!?])\s+([A-Z])', r'\1<SENT>\2', text)
        text = text.replace('<ABBREV>', '.').replace('<DECIMAL>', '.')
        sentences = text.split('<SENT>')
        merged = []
        buffer = ""
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(sent.split()) < 6 and buffer:
                buffer += " " + sent
            else:
                if buffer:
                    merged.append(buffer)
                buffer = sent
        if buffer:
            merged.append(buffer)
        return [s.strip() for s in merged if s.strip()]

    def _count_tokens(self, text: str) -> int:
        self._ensure_encoding()
        try:
            return len(self.encoding.encode(text))
        except Exception:
            return int(len(text.split()) * 1.3)


chunker = EnhancedContextualChunker()
