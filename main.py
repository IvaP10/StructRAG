from pathlib import Path
from uuid import uuid4, UUID
import logging
import os
import sys
import time
import asyncio
from threading import Lock

from typing import Dict, Any, List, Optional

_import_start = time.time()

from pdf_parser import parser
from chunker import chunker
from embedder import embedder
from database import vector_db
from models import Chunk
from core.pipeline import stream_answer
import config

def _log_handlers() -> List[logging.Handler]:
    """Always log to stdout; add a file only when one can actually be opened.

    The container runs with a read-only application directory, so an
    unconditional FileHandler here raised PermissionError at import time — and
    because server/app.py imports process_document from this module, that broke
    the server rather than just the CLI. stdout is what container logs read
    anyway.
    """
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    log_file = os.getenv("LOG_FILE", "rag_system.log")
    if log_file:
        try:
            handlers.append(logging.FileHandler(log_file))
        except OSError as exc:
            print(f"Note: file logging disabled ({exc}). Logging to stdout only.", file=sys.stderr)

    return handlers


logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.WARNING),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=_log_handlers()
)
logger = logging.getLogger(__name__)
logger.info(f"Modules loaded in {time.time() - _import_start:.2f}s")


# ── Shared state between background ingestion and query loop ──────────────────

class IngestionState:
    """Thread-safe shared state that grows as each PDF finishes indexing."""

    def __init__(self):
        self._lock = Lock()
        self.doc_ids: List[UUID] = []
        self.chunks_metadata: List[Dict[str, Any]] = []
        self.total_pdfs = 0
        self.processed_pdfs = 0
        self.is_complete = False
        self.current_file = ""
        self.errors: List[str] = []
        self._pending_messages: List[str] = []
        self.query_active = False          # suppresses ingestion prints during streaming

    def add_document(self, doc_id: UUID, chunks_meta: List[Dict[str, Any]]):
        with self._lock:
            self.doc_ids.append(doc_id)
            self.chunks_metadata.extend(chunks_meta)
            self.processed_pdfs += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "doc_ids": list(self.doc_ids),
                "chunks_metadata": list(self.chunks_metadata),
                "processed": self.processed_pdfs,
                "total": self.total_pdfs,
                "is_complete": self.is_complete,
            }

    def post_message(self, msg: str):
        """Print immediately, or buffer if a query is streaming."""
        with self._lock:
            if self.query_active:
                self._pending_messages.append(msg)
            else:
                print(msg)

    def flush_messages(self):
        with self._lock:
            for msg in self._pending_messages:
                print(msg)
            self._pending_messages.clear()

    @property
    def status_line(self) -> str:
        with self._lock:
            if self.is_complete:
                return f"✓ All {self.total_pdfs} PDF(s) indexed ({len(self.chunks_metadata)} chunks)"
            if self.total_pdfs == 0:
                return "Initializing..."
            return f"Indexing [{self.processed_pdfs}/{self.total_pdfs}] — {self.current_file}"


# ── Document processing (unchanged logic, same helpers) ──────────────────────

def process_document(pdf_path: str, doc_id: Optional[UUID] = None) -> tuple[UUID, List[Dict[str, Any]], List[str], List[Chunk]]:
    start_time = time.time()
    if doc_id is None:
        doc_id = uuid4()

    filename = Path(pdf_path).name

    total_pages, elements = parser.parse(pdf_path)
    chunks = chunker.create_chunks(str(doc_id), elements)

    for chunk in chunks:
        chunk.metadata["source_filename"] = filename

    chunks_metadata = [
        {
            "id": str(c.id),
            "parent_id": str(c.parent_id) if c.parent_id else None,
            "text": c.text,
            "chunk_type": c.chunk_type.value,
            "format_type": c.format_type.value,
            "page_number": c.page_number,
            "bbox": c.bbox.dict() if c.bbox else None,
            "token_count": c.token_count,
            "is_parent": c.is_parent,
            "metadata": c.metadata,
        }
        for c in chunks
    ]

    all_texts = [c.text for c in chunks]

    logger.info(f"Parsed {len(chunks)} chunks from {filename} in {time.time()-start_time:.2f}s | doc_id={doc_id}")
    return doc_id, chunks_metadata, all_texts, chunks


# ── Background ingestion task ─────────────────────────────────────────────────

async def _ingest_one_pdf(pdf_path: str, state: IngestionState):
    """Process → embed → index a single PDF, then atomically add it to the queryable pool."""
    filename = Path(pdf_path).name
    state.current_file = filename

    doc_id, chunks_meta, texts, chunks = await asyncio.to_thread(
        process_document, pdf_path
    )

    # batch_size=None means "use config.EMBEDDING_BATCH_SIZE". Passing a falsy
    # value here collapses the batching logic in embedder.py into one HTTP
    # request per chunk, which is both slow and expensive.
    dense_embeddings = await asyncio.to_thread(embedder.embed_texts, texts, None)
    sparse_vectors = await asyncio.to_thread(embedder.create_sparse_vectors_batch, texts)
    await asyncio.to_thread(vector_db.index_chunks, chunks, dense_embeddings, sparse_vectors)

    # Atomic: this PDF is now queryable
    state.add_document(doc_id, chunks_meta)


async def ingest_async(input_path: str, state: IngestionState):
    """Background task — ingests all PDFs and updates shared state progressively."""
    path = Path(input_path)

    await asyncio.to_thread(vector_db.reset_collection)

    if path.is_dir():
        pdf_files = sorted(path.glob("*.pdf"))
        if not pdf_files:
            print(f"Error: No PDF files found in {input_path}")
            state.is_complete = True
            return
        state.total_pdfs = len(pdf_files)

        for pdf_file in pdf_files:
            try:
                await _ingest_one_pdf(str(pdf_file), state)
            except Exception as e:
                state.errors.append(f"{pdf_file.name}: {e}")
                logger.error(f"Failed to ingest {pdf_file.name}: {e}", exc_info=True)
                with state._lock:
                    state.processed_pdfs += 1
    else:
        state.total_pdfs = 1
        try:
            await _ingest_one_pdf(input_path, state)
        except Exception as e:
            state.errors.append(f"{path.name}: {e}")
            logger.error(f"Failed to ingest {path.name}: {e}", exc_info=True)

    state.is_complete = True


# ── Async query pipeline (no nested asyncio.run) ─────────────────────────────

async def answer_query_async(
    chunks_metadata: List[Dict[str, Any]],
    query: str,
    state: IngestionState,
    document_id: Optional[UUID] = None,
) -> Dict[str, Any]:
    """Print a streamed answer to the terminal.

    All retrieval, citation parsing and verification lives in core.pipeline so
    the CLI and the HTTP server share one implementation. This function only
    decides how events look on a terminal.
    """
    state.query_active = True
    final_payload: Dict[str, Any] = {}
    citations_line = ""

    try:
        async for event in stream_answer(
            query, chunks_metadata, document_id=document_id
        ):
            kind = event.get("type")

            if kind == "token":
                print(event["content"], end="", flush=True)
            elif kind == "refusal":
                print(event["content"], end="", flush=True)
            elif kind == "citations":
                citations_line = event["formatted"]
            elif kind == "error":
                print(f"\n  {event.get('content', 'Unknown error')}\n")
                return {}
            elif kind == "done":
                final_payload = event
    finally:
        state.query_active = False
        state.flush_messages()

    print(f"\n{citations_line}" if citations_line else "")

    if final_payload and not final_payload.get("refused"):
        print(f"confidence: {final_payload.get('confidence', 0.0):.1%}\n")
    else:
        print()

    return final_payload


# ── Interactive query loop (runs concurrently with ingestion) ─────────────────

async def interactive_mode_async(state: IngestionState):

    while True:
        try:
            snap = state.snapshot()
            if state.is_complete or snap['total'] == 0:
                prompt = "Query> "
            else:
                prompt = f"[{snap['processed']}/{snap['total']}] Query> "

            user_input = (await asyncio.to_thread(input, prompt)).strip()

            if not user_input:
                continue
            if user_input.lower() in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break
            elif user_input.lower() == 'status':
                print(f"  {state.status_line}")
                if state.errors:
                    for err in state.errors:
                        print(f"  ✗ {err}")
                print()
            elif user_input.lower() == 'stats':
                snap = state.snapshot()
                cm = snap["chunks_metadata"]
                parents = sum(1 for c in cm if c.get("is_parent"))
                children = sum(1 for c in cm if not c.get("is_parent"))
                tokens = sum(c.get("token_count", 0) for c in cm)
                pages = len(set(c.get("page_number") for c in cm))
                filenames = set(c.get("metadata", {}).get("source_filename", "?") for c in cm)
                print(f"\nDocuments: {len(snap['doc_ids'])} | Files: {', '.join(filenames)}")
                print(f"Chunks: {len(cm)} ({parents} parents, {children} children) | Tokens: {tokens:,} | Pages: {pages}")
                print(f"Ingestion: {state.status_line}\n")
            else:
                # If no docs indexed yet, wait for the first one before executing
                if not state.chunks_metadata:
                    print("⏳ Waiting for first document to finish indexing...")
                    while not state.chunks_metadata and not state.is_complete:
                        await asyncio.sleep(0.3)
                    if not state.chunks_metadata:
                        print("No documents were indexed. Cannot answer.\n")
                        continue
                    print()

                try:
                    snap = state.snapshot()
                    await answer_query_async(snap["chunks_metadata"], user_input, state)
                except Exception as e:
                    print(f"Error: {e}")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


# ── Entrypoint ────────────────────────────────────────────────────────────────

async def async_main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else input("Enter PDF file or folder path: ").strip()
    input_path = input_path.strip('"').strip("'")
    path = Path(input_path)

    if not path.exists():
        print(f"Error: Path not found: {input_path}")
        return 1

    if not path.is_dir() and path.suffix.lower() != '.pdf':
        print("Error: Unsupported path. Provide a PDF file or a folder containing PDFs.")
        return 1

    print(f"Processing: {input_path}\n")

    state = IngestionState()

    try:
        # Launch ingestion as a background task — does NOT block the query loop
        ingestion_task = asyncio.create_task(ingest_async(input_path, state))

        # Start accepting queries immediately (waits only for the first PDF)
        await interactive_mode_async(state)

        # If user quits before ingestion finishes, cancel gracefully
        if not ingestion_task.done():
            ingestion_task.cancel()
            try:
                await ingestion_task
            except asyncio.CancelledError:
                logger.info("Ingestion cancelled by user exit")
    except Exception as e:
        print(f"\nError: {e}")
        logger.error(f"Application error: {e}", exc_info=True)
        return 1

    return 0


def main():
    return asyncio.run(async_main())


if __name__ == "__main__":
    sys.exit(main())
