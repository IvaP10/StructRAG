"""Citation parsing. Pure stdlib, no model or database dependencies.

Separate from pipeline.py so this character-level state machine can be tested
without openai, numpy, or a running Qdrant.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Set

# Matches the consolidated citation the generator's system prompt asks for:
#   [[Source: report.pdf | Page: 1,3,5]]
CITATION_RE = re.compile(r'\[\[Source:\s*(.+?)\s*\|\s*Page:\s*([\d,\s]+)\s*\]\]')


def _render(sources: Dict[str, Set[int]]) -> str:
    parts = []
    for filename, pages in sorted(sources.items()):
        page_str = ','.join(str(p) for p in sorted(pages))
        parts.append(f"[Source: {filename} | Page: {page_str}]")
    return ' '.join(parts)


class CitationStreamParser:
    """Pulls [[Source: ... | Page: ...]] blocks out of a token stream.

    Citations arrive inline but are rendered once at the end. A citation can be
    split across arbitrary token boundaries, so this scans character by character
    and holds back text that might begin one.
    """

    def __init__(self):
        self._buffer = ""        # holds "[", "[[", or a partial citation
        self._inside = False     # True once "[[" has been seen
        self.sources: Dict[str, Set[int]] = defaultdict(set)

    def feed(self, text: str) -> str:
        """Consume a token; return only the text that should be shown."""
        visible: List[str] = []

        for ch in text:
            if self._inside:
                self._buffer += ch
                if self._buffer.endswith("]]"):
                    m = CITATION_RE.match(self._buffer)
                    if m:
                        self._collect(m)
                    else:
                        # Bracketed text that wasn't a citation after all —
                        # it belongs in the answer.
                        visible.append(self._buffer)
                    self._buffer = ""
                    self._inside = False
                continue

            if ch == '[' and self._buffer == "":
                self._buffer = "["
            elif ch == '[' and self._buffer == "[":
                self._buffer = "[["
                self._inside = True
            else:
                if self._buffer:
                    visible.append(self._buffer)
                    self._buffer = ""
                visible.append(ch)

        return "".join(visible)

    def finish(self) -> str:
        """Flush held-back text at end of stream.

        An unterminated "[[" is dropped — it holds no answer text.
        """
        trailing = "" if self._inside else self._buffer
        self._buffer = ""
        self._inside = False
        return trailing

    def _collect(self, match: re.Match) -> None:
        filename = match.group(1).strip()
        for page in match.group(2).split(','):
            page = page.strip()
            if page.isdigit():
                self.sources[filename].add(int(page))

    def formatted(self) -> str:
        """Render collected citations as one line, or "" if there were none."""
        return _render(self.sources) if self.sources else ""

    def as_dict(self) -> Dict[str, List[int]]:
        return {fn: sorted(pages) for fn, pages in sorted(self.sources.items())}


def consolidate_citations(text: str) -> str:
    """Merge inline citations in a complete string into one trailing line.

    Non-streaming equivalent of CitationStreamParser.
    """
    sources: Dict[str, Set[int]] = defaultdict(set)
    for m in CITATION_RE.finditer(text):
        filename = m.group(1).strip()
        for p in m.group(2).split(','):
            p = p.strip()
            if p.isdigit():
                sources[filename].add(int(p))

    cleaned = CITATION_RE.sub('', text)
    cleaned = re.sub(r'[,\s]*\.', '.', cleaned)   # drop a comma left before a period
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)
    cleaned = cleaned.strip()

    if sources:
        cleaned += '\n' + _render(sources)

    return cleaned
