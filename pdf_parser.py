from __future__ import annotations

import logging
import math
import multiprocessing as mp
import os
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF  (pip install pymupdf)
import pdfplumber  # (pip install pdfplumber)

# Docling is imported lazily inside the worker that needs it — avoids a
# ~1 s import penalty on every process start when most pages are digital.

from models import BoundingBox, ChunkType, FormatType

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class ExtractionBackend(Enum):
    PYMUPDF    = auto()
    PDFPLUMBER = auto()
    DOCLING    = auto()


class PageKind(Enum):
    DIGITAL = auto()  # clean native text layer
    MIXED   = auto()  # sparse text + images
    SCANNED = auto()  # image-only, no text layer


# ──────────────────────────────────────────────
# Value objects
# ──────────────────────────────────────────────

@dataclass
class PageProfile:
    page_no:        int
    kind:           PageKind
    text_density:   float
    has_tables:     bool
    font_ok:        bool
    best_backend:   ExtractionBackend


@dataclass
class LayoutElement:
    content:      str
    element_type: ChunkType
    page:         int
    bbox:         Optional[BoundingBox] = None
    format_type:  FormatType = FormatType.PLAIN
    backend:      ExtractionBackend = ExtractionBackend.PYMUPDF
    confidence:   float = 1.0
    metadata:     Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"LayoutElement(type={self.element_type.name}, page={self.page}, "
            f"backend={self.backend.name}, conf={self.confidence:.2f}, "
            f"len={len(self.content)})"
        )


# ──────────────────────────────────────────────
# Page profiler  (runs in the main process, fast)
# ──────────────────────────────────────────────

class PageProfiler:
    """
    One-pass diagnostic per page.  Total cost for 250 pages: ~50–80 ms.

    Key thresholds tuned for A4 / letter finance docs:
      _DIGITAL_DENSITY_THRESHOLD  — minimum chars/1000pt² for a live text layer
      _ALPHANUM_RATIO_THRESHOLD   — garbled encoding detector
      _TABLE_H_LINE_MIN           — min horizontal rules to call it a table
    """

    _DIGITAL_DENSITY_THRESHOLD = 0.04
    _ALPHANUM_RATIO_THRESHOLD   = 0.55
    _TABLE_H_LINE_MIN           = 3
    _TABLE_V_LINE_MIN           = 2

    def profile_all(self, pdf_path: str) -> List[PageProfile]:
        doc = fitz.open(pdf_path)
        try:
            return [self._profile(doc[i], i + 1) for i in range(len(doc))]
        finally:
            doc.close()

    def _profile(self, page: fitz.Page, page_no: int) -> PageProfile:
        area = max(page.rect.width * page.rect.height, 1.0)
        raw  = page.get_text("text")

        char_count   = len(raw.replace(" ", "").replace("\n", ""))
        density      = char_count / area * 1_000
        alphanum     = sum(c.isalnum() for c in raw)
        alpha_ratio  = alphanum / max(len(raw), 1)
        font_ok      = alpha_ratio >= self._ALPHANUM_RATIO_THRESHOLD

        if density < self._DIGITAL_DENSITY_THRESHOLD or not font_ok:
            kind = PageKind.SCANNED
        elif density < self._DIGITAL_DENSITY_THRESHOLD * 4:
            kind = PageKind.MIXED
        else:
            kind = PageKind.DIGITAL

        has_tables = self._cheap_table_hint(page)
        backend    = self._route(kind, has_tables)

        return PageProfile(
            page_no=page_no, kind=kind, text_density=density,
            has_tables=has_tables, font_ok=font_ok, best_backend=backend,
        )

    def _cheap_table_hint(self, page: fitz.Page) -> bool:
        """
        Counts axis-aligned rules only — much cheaper than pdfplumber's
        full table finder.  O(drawings) which is tiny on most pages.
        """
        paths  = page.get_drawings()
        h_lines = sum(
            1 for p in paths
            if p["type"] in ("l", "re")
            and abs(p["rect"].height) < 2
            and p["rect"].width > 30
        )
        v_lines = sum(
            1 for p in paths
            if p["type"] in ("l", "re")
            and abs(p["rect"].width) < 2
            and p["rect"].height > 15
        )
        return h_lines >= self._TABLE_H_LINE_MIN and v_lines >= self._TABLE_V_LINE_MIN

    @staticmethod
    def _route(kind: PageKind, has_tables: bool) -> ExtractionBackend:
        if kind == PageKind.SCANNED:
            return ExtractionBackend.DOCLING
        if has_tables:
            return ExtractionBackend.PDFPLUMBER
        return ExtractionBackend.PYMUPDF


# ──────────────────────────────────────────────
# PyMuPDF extractor  (primary — used on ALL digital pages)
# ──────────────────────────────────────────────

# Finance docs typically use 9–11 pt body, 12–14 pt section heads.
# These thresholds are tighter than the original to reduce false heading
# classifications on bold body text in tables.
_HEADING_SIZE_THRESHOLD = 13.0
_HEADING_BOLD_SIZE      = 11.5

# Pre-compile once at module level — no re-compilation per page.
_LIST_RE     = re.compile(r"^[\u2022\u2013\u2014\-\*\u25e6]|\d+[\.\)]\s")
_FOOTNOTE_RE = re.compile(r"^\s*\d{1,2}\s+[A-Z]")   # "1  See Note 3 …"


class PyMuPDFExtractor:
    """
    Extracts every text block from a page in reading order.

    Performance note:
      page.get_text("dict") returns a full block/line/span tree.
      Iterating it in Python is ~0.5–1 ms/page on a laptop.
    """

    def extract_page(self, page: fitz.Page, page_no: int) -> List[LayoutElement]:
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        elems: List[LayoutElement] = []

        for blk in blocks:
            if blk["type"] != 0:   # skip embedded image blocks
                continue

            text = self._block_text(blk)
            if not text:
                continue

            b    = blk["bbox"]
            bbox = BoundingBox(x=b[0], y=b[1], width=b[2]-b[0], height=b[3]-b[1])

            elems.append(LayoutElement(
                content      = text,
                element_type = self._infer_type(blk, text),
                page         = page_no,
                bbox         = bbox,
                format_type  = FormatType.PLAIN,
                backend      = ExtractionBackend.PYMUPDF,
                confidence   = 0.93,
            ))

        return elems

    @staticmethod
    def _block_text(blk: dict) -> str:
        lines = []
        for line in blk["lines"]:
            spans_text = " ".join(s["text"] for s in line["spans"]).strip()
            if spans_text:
                lines.append(spans_text)
        return "\n".join(lines)

    @staticmethod
    def _infer_type(blk: dict, text: str) -> ChunkType:
        sizes = [s["size"] for ln in blk["lines"] for s in ln["spans"]]
        flags = [s["flags"] for ln in blk["lines"] for s in ln["spans"]]
        if not sizes:
            return ChunkType.TEXT

        avg_size = sum(sizes) / len(sizes)
        is_bold  = any(f & 16 for f in flags)

        if avg_size >= _HEADING_SIZE_THRESHOLD or (avg_size >= _HEADING_BOLD_SIZE and is_bold):
            return ChunkType.HEADER
        if _LIST_RE.match(text):
            return ChunkType.LIST
        if _FOOTNOTE_RE.match(text):
            return ChunkType.FOOTER   # treat footnotes as footers for downstream filtering
        return ChunkType.TEXT


# ──────────────────────────────────────────────
# pdfplumber extractor  (tables only)
# ──────────────────────────────────────────────

_PLUMBER_TABLE_SETTINGS = {
    "vertical_strategy":   "lines_strict",
    "horizontal_strategy": "lines_strict",
    "snap_tolerance":       5,
    "join_tolerance":       5,
    "min_words_vertical":   1,
    "min_words_horizontal": 1,
}


class PDFPlumberExtractor:
    """
    Used ONLY when the page profiler detected table rules.

    Design: caller passes in an already-open pdfplumber PDF object so
    the open() cost is paid once per worker process, not once per page.
    """

    def extract_page(self, plumber_page, page_no: int) -> List[LayoutElement]:
        elems: List[LayoutElement] = []

        tables    = plumber_page.find_tables(_PLUMBER_TABLE_SETTINGS)
        table_bbs = []

        for t in tables:
            data = t.extract()
            if not data:
                continue
            md   = _table_to_markdown(data)
            if not md:
                continue
            bb   = t.bbox   # (x0, y0, x1, y1)
            elems.append(LayoutElement(
                content      = md,
                element_type = ChunkType.TABLE,
                page         = page_no,
                bbox         = BoundingBox(x=bb[0], y=bb[1], width=bb[2]-bb[0], height=bb[3]-bb[1]),
                format_type  = FormatType.MARKDOWN,
                backend      = ExtractionBackend.PDFPLUMBER,
                confidence   = 0.91,
            ))
            table_bbs.append(bb)

        # Non-table text — subtract table regions first
        cropped = plumber_page
        for bb in table_bbs:
            try:
                cropped = cropped.outside_bbox(bb)
            except Exception:  # nosec B110 - a bbox that will not crop is skipped, not fatal
                pass

        raw = cropped.extract_text(x_tolerance=3, y_tolerance=3) or ""
        if raw.strip():
            elems.append(LayoutElement(
                content      = raw.strip(),
                element_type = ChunkType.TEXT,
                page         = page_no,
                format_type  = FormatType.PLAIN,
                backend      = ExtractionBackend.PDFPLUMBER,
                confidence   = 0.88,
            ))

        return elems


def _table_to_markdown(table: List[List[Optional[str]]]) -> str:
    if not table or not table[0]:
        return ""
    rows = [[cell.strip() if cell else "" for cell in row] for row in table]
    hdr  = "| " + " | ".join(rows[0]) + " |"
    sep  = "| " + " | ".join("---" for _ in rows[0]) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join([hdr, sep] + body)


# ──────────────────────────────────────────────
# Docling extractor  (selective — scanned pages only)
# ──────────────────────────────────────────────

class DoclingExtractor:
    """
    Initialised lazily inside the worker process that needs it.
    For FinanceBench, this should fire on 0 or very few pages.
    """

    def __init__(self) -> None:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        opts = PdfPipelineOptions()
        opts.do_ocr             = True
        opts.do_table_structure = False   # PyMuPDF handles tables already
        self._converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )

    def extract_pages(
        self, pdf_path: str, page_nos: List[int]
    ) -> List[LayoutElement]:
        if not page_nos:
            return []
            
        import tempfile
        import os
        import fitz
        
        src_doc = fitz.open(pdf_path)
        dst_doc = fitz.open()
        
        page_indices = [p - 1 for p in page_nos]
        dst_doc.insert_pdf(src_doc, pages=page_indices)
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
            
        dst_doc.save(tmp_path)
        dst_doc.close()
        src_doc.close()
        
        elems: List[LayoutElement] = []
        try:
            result = self._converter.convert(tmp_path)
            doc = result.document
            
            for item, _ in doc.iterate_items():
                elem = _docling_item_to_element(item)
                if elem:
                    try:
                        temp_pno = item.prov[0].page_no if item.prov else 1
                        idx = temp_pno - 1
                        if 0 <= idx < len(page_nos):
                            elem.page = page_nos[idx]
                        else:
                            elem.page = page_nos[0]
                    except Exception:
                        elem.page = page_nos[0]
                    elems.append(elem)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
        return elems


_DOCLING_LABEL_MAP = {
    "TABLE":          (ChunkType.TABLE,    FormatType.MARKDOWN),
    "CODE":           (ChunkType.CODE,     FormatType.PLAIN),
    "LIST_ITEM":      (ChunkType.LIST,     FormatType.PLAIN),
    "SECTION_HEADER": (ChunkType.HEADER,   FormatType.PLAIN),
    "PAGE_HEADER":    (ChunkType.HEADER,   FormatType.PLAIN),
    "PAGE_FOOTER":    (ChunkType.FOOTER,   FormatType.PLAIN),
    "EQUATION":       (ChunkType.EQUATION, FormatType.LATEX),
    "FORMULA":        (ChunkType.EQUATION, FormatType.LATEX),
}


def _docling_item_to_element(item) -> Optional[LayoutElement]:
    label  = str(getattr(item, "label", "")).upper().split(".")[-1]
    ctype, fmt = _DOCLING_LABEL_MAP.get(label, (ChunkType.TEXT, FormatType.PLAIN))

    if ctype == ChunkType.TABLE:
        try:
            content = item.export_to_markdown()
        except Exception:
            content = getattr(item, "text", "") or ""
    elif ctype == ChunkType.EQUATION:
        latex = getattr(item, "latex", None)
        content = f"$${latex}$$" if latex else getattr(item, "text", "") or ""
    else:
        content = getattr(item, "text", "") or ""

    if not content.strip():
        return None

    try:
        pno = item.prov[0].page_no if item.prov else 1
    except Exception:
        pno = 1

    try:
        b    = item.prov[0].bbox
        bbox = BoundingBox(x=float(b.l), y=float(b.t),
                           width=float(b.r - b.l), height=float(b.b - b.t))
    except Exception:
        bbox = None

    return LayoutElement(
        content=content, element_type=ctype, page=pno, bbox=bbox,
        format_type=fmt, backend=ExtractionBackend.DOCLING, confidence=0.85,
    )


# ──────────────────────────────────────────────
# Worker function  (runs in child process)
# ──────────────────────────────────────────────

def _worker(
    pdf_path:    str,
    page_nos:    List[int],          # 1-based page numbers assigned to this worker
    profiles:    Dict[int, PageProfile],
) -> List[LayoutElement]:
    """
    Each worker opens its own fitz.Document and (optionally) a pdfplumber PDF.
    It processes all assigned pages and returns the flat element list.

    Docling is only initialised if any assigned page is SCANNED.
    """
    mupdf_ext   = PyMuPDFExtractor()
    plumber_ext = PDFPlumberExtractor()
    docling_ext = None

    fitz_doc    = fitz.open(pdf_path)

    # Open pdfplumber only if any assigned page needs tables
    need_plumber  = any(profiles[p].has_tables for p in page_nos)
    plumber_pdf   = pdfplumber.open(pdf_path) if need_plumber else None

    scanned_pages = [p for p in page_nos if profiles[p].kind == PageKind.SCANNED]
    if scanned_pages:
        try:
            docling_ext = DoclingExtractor()
        except Exception as exc:
            logger.warning("Docling unavailable: %s — scanned pages will be empty.", exc)

    elems: List[LayoutElement] = []

    try:
        # ── Digital / mixed pages ─────────────────────────────────────────
        for pno in page_nos:
            prof = profiles[pno]
            if prof.kind == PageKind.SCANNED:
                continue   # handled separately below

            fitz_page = fitz_doc[pno - 1]

            if prof.has_tables and plumber_pdf is not None:
                # pdfplumber for tables + non-table text on this page
                plumber_page = plumber_pdf.pages[pno - 1]
                page_elems   = plumber_ext.extract_page(plumber_page, pno)
                # Also run PyMuPDF for non-table text that pdfplumber might miss
                mupdf_elems  = mupdf_ext.extract_page(fitz_page, pno)
                page_elems   = _merge_page(page_elems, mupdf_elems)
            else:
                page_elems = mupdf_ext.extract_page(fitz_page, pno)

            elems.extend(page_elems)

        # ── Scanned pages via Docling ─────────────────────────────────────
        if scanned_pages and docling_ext is not None:
            elems.extend(docling_ext.extract_pages(pdf_path, scanned_pages))

    finally:
        fitz_doc.close()
        if plumber_pdf is not None:
            plumber_pdf.close()

    return elems


def _merge_page(
    plumber_elems: List[LayoutElement],
    mupdf_elems:   List[LayoutElement],
) -> List[LayoutElement]:
    """
    On table pages, keep all pdfplumber TABLE elements; fill remaining
    text from PyMuPDF (higher fidelity than pdfplumber's extract_text).

    O(1) per element: we just skip PyMuPDF TEXT blocks that spatially
    overlap any table bbox by > 50 %.
    """
    table_bboxes = [e.bbox for e in plumber_elems if e.element_type == ChunkType.TABLE and e.bbox]

    merged = [e for e in plumber_elems if e.element_type == ChunkType.TABLE]

    for e in mupdf_elems:
        if e.element_type == ChunkType.TABLE:
            continue   # pdfplumber tables win
        if e.bbox and any(_overlap(e.bbox, tb) > 0.5 for tb in table_bboxes):
            continue   # skip text inside a table region
        merged.append(e)

    return merged


def _overlap(a: BoundingBox, b: BoundingBox) -> float:
    ax1, ay1, ax2, ay2 = a.x, a.y, a.x + a.width, a.y + a.height
    bx1, by1, bx2, by2 = b.x, b.y, b.x + b.width, b.y + b.height
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    a_area = (ax2 - ax1) * (ay2 - ay1)
    return inter / a_area if a_area > 0 else 0.0


# ──────────────────────────────────────────────
# Text normaliser
# ──────────────────────────────────────────────

class TextNormaliser:
    _LIGATURE = str.maketrans({
        "\uFB00": "ff", "\uFB01": "fi", "\uFB02": "fl",
        "\uFB03": "ffi", "\uFB04": "ffl", "\uFB05": "st", "\uFB06": "st",
    })
    _CTRL     = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
    _HYPHEN   = re.compile(r"(\w)-\n(\w)")
    _SPACES   = re.compile(r"[ \t]{2,}")

    def normalise_batch(self, elements: List[LayoutElement]) -> None:
        """In-place normalisation.  Single loop — no per-element method call overhead."""
        for e in elements:
            t = e.content
            if not t:
                continue
            t = unicodedata.normalize("NFC", t)
            t = t.translate(self._LIGATURE)
            t = self._CTRL.sub("", t)
            t = self._HYPHEN.sub(r"\1\2", t)
            t = self._SPACES.sub(" ", t)
            e.content = t.strip()


# ──────────────────────────────────────────────
# Main parser
# ──────────────────────────────────────────────

class FastPDFParser:
    """
    Usage
    ─────
        parser = FastPDFParser(num_workers=4)
        total_pages, elements = parser.parse("annual_report.pdf")

    num_workers
        Defaults to os.cpu_count().  Set to 1 to disable multiprocessing
        (useful for debugging or when the PDF is small).

    Speed guide (born-digital, no scanned pages)
        workers=1 : ~1.5–2.5 s for 250 pages
        workers=4 : ~0.6–1.2 s for 250 pages
        workers=8 : ~0.4–0.8 s for 250 pages
    """

    def __init__(self, num_workers: Optional[int] = None) -> None:
        self._num_workers = num_workers or os.cpu_count() or 4
        self._profiler    = PageProfiler()
        self._normaliser  = TextNormaliser()

    def parse(self, pdf_path: str) -> Tuple[int, List[LayoutElement]]:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(pdf_path)

        # ── Profile ───────────────────────────────────────────────────────
        profiles_list = self._profiler.profile_all(pdf_path)
        total_pages   = len(profiles_list)
        profiles      = {p.page_no: p for p in profiles_list}

        logger.info("Pages: %d | workers: %d", total_pages, self._num_workers)
        _log_profile_summary(profiles_list)

        # ── Dispatch ──────────────────────────────────────────────────────
        all_elems = self._dispatch(pdf_path, profiles, total_pages)

        # ── Sort by (page, y) ─────────────────────────────────────────────
        all_elems.sort(key=lambda e: (e.page, e.bbox.y if e.bbox else 0.0))

        # ── Normalise ─────────────────────────────────────────────────────
        self._normaliser.normalise_batch(all_elems)

        # ── Drop empties ──────────────────────────────────────────────────
        final = [e for e in all_elems if e.content]

        _log_final_summary(final)
        return total_pages, final

    def _dispatch(
        self,
        pdf_path:    str,
        profiles:    Dict[int, PageProfile],
        total_pages: int,
    ) -> List[LayoutElement]:
        """
        Split pages into equal-sized batches; run each batch in a
        child process.  Uses multiprocessing.Pool.starmap.
        """
        page_nos   = list(range(1, total_pages + 1))
        n_workers  = min(self._num_workers, total_pages)
        batch_size = math.ceil(total_pages / n_workers)
        batches    = [
            page_nos[i : i + batch_size]
            for i in range(0, total_pages, batch_size)
        ]

        if n_workers == 1 or total_pages <= 8:
            # Skip multiprocessing overhead for tiny docs
            return _worker(pdf_path, page_nos, profiles)

        args = [(pdf_path, batch, profiles) for batch in batches]

        # 'spawn' context avoids forking fitz handles into child processes
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            results = pool.starmap(_worker, args)

        return [e for batch_result in results for e in batch_result]


# ──────────────────────────────────────────────
# Logging helpers
# ──────────────────────────────────────────────

def _log_profile_summary(profiles: List[PageProfile]) -> None:
    kinds: Dict[str, int] = {}
    for p in profiles:
        kinds[p.kind.name] = kinds.get(p.kind.name, 0) + 1
    table_pages = sum(1 for p in profiles if p.has_tables)
    logger.info("Page kinds: %s | table pages: %d", kinds, table_pages)


def _log_final_summary(elements: List[LayoutElement]) -> None:
    counts:   Dict[str, int] = {}
    backends: Dict[str, int] = {}
    for e in elements:
        counts[e.element_type.name]   = counts.get(e.element_type.name, 0) + 1
        backends[e.backend.name]      = backends.get(e.backend.name, 0) + 1
    logger.info(
        "Elements: %d | types: %s | backends: %s",
        len(elements), counts, backends,
    )


# ── Module-level singleton ────────────────────────────────────────────────────
parser = FastPDFParser()