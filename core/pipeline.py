"""The query pipeline, with no opinion about where output goes.

Yields events instead of printing, so the CLI and the HTTP server share one
implementation. main.py prints them; server/app.py serialises them as SSE.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional, Tuple
from uuid import UUID

from core.citations import CitationStreamParser, consolidate_citations  # noqa: F401  (re-exported)
from generator import generator
from retriever import retriever

logger = logging.getLogger(__name__)

REFUSAL_MESSAGE = "I can only answer questions about the loaded documents."


def _refusal_events(start: float, reason: str, message: str = REFUSAL_MESSAGE):
    """The two events emitted when a question is declined."""
    return [
        {"type": "refusal", "content": message, "reason": reason},
        {
            "type": "done",
            "answer": message,
            "sources": [],
            "confidence": 0.0,
            "refused": True,
            "refusal_reason": reason,
            "citations": {},
            "processing_time": time.time() - start,
        },
    ]


async def stream_answer(
    query: str,
    chunks_metadata: List[Dict[str, Any]],
    *,
    document_id: Optional[UUID] = None,
    session_id: Optional[str] = None,
    usage_sink: Optional[List[Dict[str, Any]]] = None,
    intent_check: Optional[Callable[[], Awaitable[Tuple[bool, str]]]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Retrieve, generate, and verify — yielding events instead of printing.

    Event shapes:
        {"type": "token",     "content": str}    visible answer text
        {"type": "citations", "sources": {...},  "formatted": str}
        {"type": "refusal",   "content": str}    declined; no generation happened
        {"type": "error",     "content": str}
        {"type": "done",      ...verification payload, "processing_time": float}

    `usage_sink`, if provided, accumulates token-usage records so a caller can
    price the request. `session_id` scopes retrieval to one visitor's documents.

    `intent_check` runs after retrieval finds context but before generation, so
    an off-topic question is refused for the price of one embedding call rather
    than a classifier call. The CLI passes nothing and skips it.
    """
    start = time.time()

    context_data = await retriever.retrieve_context(
        query=query,
        chunks_metadata=chunks_metadata,
        document_id=document_id,
        session_id=session_id,
    )

    # Nothing in the corpus is close enough. Stop before generation, so an
    # off-topic question costs nothing.
    if context_data["metrics"].get("rejected_by_relevance_gate"):
        for event in _refusal_events(start, "no_relevant_context"):
            yield event
        return

    # Retrieval found a match, so it is now worth spending a classifier call on
    # whether the question really is a document question.
    if intent_check is not None:
        permitted, reason = await intent_check()
        if not permitted:
            logger.info(f"Query declined by intent guard: {reason}")
            for event in _refusal_events(start, "off_topic"):
                yield event
            return

    parser = CitationStreamParser()
    final_payload: Optional[Dict[str, Any]] = None

    try:
        async for raw in generator.generate_answer_stream(query, context_data, usage_sink):
            data = json.loads(raw)
            kind = data.get("type")

            if kind == "token":
                visible = parser.feed(data["content"])
                if visible:
                    yield {"type": "token", "content": visible}
            elif kind in ("verification", "error"):
                final_payload = data
    finally:
        trailing = parser.finish()
        if trailing:
            yield {"type": "token", "content": trailing}

    if parser.sources:
        yield {
            "type": "citations",
            "sources": parser.as_dict(),
            "formatted": parser.formatted(),
        }

    if final_payload and final_payload.get("type") == "error":
        yield {"type": "error", "content": final_payload.get("content", "Unknown error")}
        return

    done: Dict[str, Any] = dict(final_payload or {})
    done["type"] = "done"
    done["processing_time"] = time.time() - start
    done.setdefault("confidence", 0.0)
    done["citations"] = parser.as_dict()
    yield done
