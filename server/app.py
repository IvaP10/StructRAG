"""FastAPI wrapper around the StructRAG pipeline.

Holds the OpenAI key and enforces every limit. The frontend is untrusted — a
visitor controls it — so nothing is validated there.

Guards run cheapest first, so abuse is rejected before it costs anything:

    kill switch -> CORS -> spend cap -> session -> rate limit
    -> input validation -> relevance gate (no LLM call) -> intent guard
    -> hardened generation
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sys
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

import config
from core.pipeline import REFUSAL_MESSAGE as REFUSAL_TEXT
from core.pipeline import stream_answer
from server import guards
from server.limits import SlidingWindowLimiter, SpendLedger
from server.store import IngestJob, SessionStore

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.WARNING),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared state
# ─────────────────────────────────────────────────────────────────────────────

sessions = SessionStore(ttl_seconds=config.SESSION_TTL_SECONDS)
ledger = SpendLedger(config.DAILY_USD_CAP, config.PRICE_PER_MTOK)

query_limiter = SlidingWindowLimiter(config.MAX_QUERIES_PER_HOUR, 3600)
upload_limiter = SlidingWindowLimiter(config.MAX_UPLOADS_PER_DAY, 86400)
# Keyed by IP: a session-keyed limit would be escaped by minting a new one.
session_limiter = SlidingWindowLimiter(config.MAX_SESSIONS_PER_IP_PER_HOUR, 3600)


def _startup_checks() -> None:
    """Fail closed on a misconfigured deploy.

    Without an invite code or signing secret the app would be an open proxy to a
    paid API key, so it refuses to boot instead.
    """
    problems = []
    if not config.OPENAI_API_KEY:
        problems.append("OPENAI_API_KEY is not set.")
    if not config.INVITE_CODE:
        problems.append("INVITE_CODE is not set — the API would be open to anyone.")
    if len(config.INVITE_CODE) < 8:
        problems.append("INVITE_CODE is shorter than 8 characters — too easy to guess.")
    if not config.SESSION_SECRET:
        problems.append("SESSION_SECRET is not set — session tokens could be forged.")
    if len(config.SESSION_SECRET) < 16:
        problems.append("SESSION_SECRET is shorter than 16 characters — too weak to sign with.")
    if config.DAILY_USD_CAP <= 0:
        problems.append("DAILY_USD_CAP must be greater than zero.")
    if not config.ALLOWED_ORIGINS:
        problems.append("ALLOWED_ORIGINS is empty — no browser could call the API.")

    if problems:
        raise RuntimeError(
            "Refusing to start:\n  - " + "\n  - ".join(problems)
            + "\n\nSee key.env.example for what each value does."
        )


async def _janitor() -> None:
    """Periodically evict expired sessions and their vectors."""
    from database import vector_db

    while True:
        try:
            await asyncio.sleep(300)
            for session_id in sessions.expired_ids():
                logger.info(f"Evicting expired session {session_id}")
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(vector_db.delete_session, session_id)
                sessions.drop(session_id)
            query_limiter.evict_idle(7200)
            upload_limiter.evict_idle(172800)
            session_limiter.evict_idle(7200)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(f"Janitor pass failed: {exc}", exc_info=True)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    _startup_checks()
    logger.info(
        "StructRAG API starting | daily cap $%.2f | %d queries/hr | origins %s",
        config.DAILY_USD_CAP, config.MAX_QUERIES_PER_HOUR, config.ALLOWED_ORIGINS,
    )
    task = asyncio.create_task(_janitor())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


app = FastAPI(
    title="StructRAG API",
    version="0.2.0",
    lifespan=lifespan,
    # The OpenAPI schema maps the attack surface; three endpoints do not need it.
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

# Explicit allowlist. "*" would let any site spend the budget via a visitor's
# session.
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
    max_age=3600,
)


@app.middleware("http")
async def kill_switch(request: Request, call_next):
    """Hard stop for everything except the health probe."""
    if config.DISABLED and request.url.path != "/api/health":
        return JSONResponse(
            status_code=503,
            content={"detail": "This service is temporarily disabled."},
        )
    return await call_next(request)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    # The API returns only JSON and SSE — nothing to execute or embed.
    response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
    return response


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def client_ip(request: Request) -> str:
    """Best-effort client address.

    TLS terminates upstream, so the socket peer is the proxy and the client is
    the leftmost X-Forwarded-For entry. Spoofable: used for cost shaping, never
    as identity.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def require_session(authorization: Optional[str] = Header(default=None)) -> str:
    """FastAPI dependency: resolve a Bearer token to a session id."""
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()

    try:
        return guards.verify_session_token(token)
    except guards.SessionError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


def require_budget() -> None:
    if ledger.would_exceed():
        snapshot = ledger.snapshot()
        logger.warning(f"Daily spend cap reached: {snapshot}")
        raise HTTPException(
            status_code=503,
            detail=(
                "This demo has reached its daily usage budget. "
                "It resets at midnight UTC — please come back then."
            ),
        )


def sse(event: Dict[str, Any]) -> str:
    """Encode one Server-Sent Events frame."""
    return f"data: {json.dumps(event, default=str)}\n\n"


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

class SessionRequest(BaseModel):
    invite_code: str = Field(min_length=1, max_length=200)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)  # hard ceiling; real limit in validate_query


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health() -> Dict[str, Any]:
    """Unauthenticated liveness probe. Reveals nothing about configuration."""
    return {
        "status": "disabled" if config.DISABLED else "ok",
        "budget_exhausted": ledger.would_exceed(),
    }


@app.post("/api/session")
async def create_session(body: SessionRequest, request: Request) -> Dict[str, Any]:
    ip = client_ip(request)

    allowed, retry_after = session_limiter.check(ip)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Too many attempts. Wait a few minutes and try again.",
            headers={"Retry-After": str(retry_after)},
        )

    if not guards.check_invite_code(body.invite_code):
        logger.warning(f"Invalid invite code from {ip}")
        # Vague on purpose — do not confirm whether a code is configured.
        raise HTTPException(status_code=401, detail="That invite code is not valid.")

    token, session_id = guards.issue_session_token()
    sessions.create(session_id)
    logger.info(f"Session {session_id} opened for {ip}")

    return {
        "token": token,
        "expires_in": config.SESSION_TTL_SECONDS,
        "limits": {
            "queries_per_hour": config.MAX_QUERIES_PER_HOUR,
            "uploads_per_day": config.MAX_UPLOADS_PER_DAY,
            "max_pdf_pages": config.MAX_PDF_PAGES,
            "max_upload_mb": round(config.MAX_UPLOAD_BYTES / (1024 * 1024)),
            "max_query_chars": config.MAX_QUERY_CHARS,
        },
    }


@app.get("/api/session")
async def session_state(session_id: str = Depends(require_session)) -> Dict[str, Any]:
    session = sessions.get_or_create(session_id)
    return {
        **session.summary(),
        "queries_remaining": query_limiter.remaining(session_id),
        "uploads_remaining": upload_limiter.remaining(session_id),
        "jobs": [job.as_dict() for job in session.jobs.values()],
    }


@app.post("/api/upload")
async def upload(
    request: Request,
    file: UploadFile = File(...),
    session_id: str = Depends(require_session),
) -> Dict[str, Any]:
    require_budget()

    allowed, retry_after = upload_limiter.check(session_id)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Upload limit reached ({config.MAX_UPLOADS_PER_DAY} per day).",
            headers={"Retry-After": str(retry_after)},
        )

    # Read with a ceiling; Content-Length can be understated. One byte over the
    # limit is enough to reject.
    data = await file.read(config.MAX_UPLOAD_BYTES + 1)
    filename = guards.safe_filename(file.filename)

    try:
        info = guards.validate_pdf_bytes(data, filename)
    except guards.ValidationError as exc:
        # A rejected file cost nothing, so refund the quota slot.
        upload_limiter.refund(session_id)
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    session = sessions.get_or_create(session_id)
    job = IngestJob(job_id=uuid.uuid4().hex[:12], filename=filename, pages=info["page_count"])
    session.jobs[job.job_id] = job

    asyncio.create_task(_ingest(session_id, job, data, filename))

    logger.info(f"Session {session_id} uploading {filename} ({info['page_count']}p) as job {job.job_id}")
    return {"job_id": job.job_id, **info}


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str, session_id: str = Depends(require_session)) -> Dict[str, Any]:
    session = sessions.get_or_create(session_id)
    job = session.jobs.get(job_id)
    if job is None:
        # Scoped to the caller, so a guessed id from another session looks the
        # same as one that never existed.
        raise HTTPException(status_code=404, detail="No such job.")
    return job.as_dict()


@app.post("/api/query")
async def query(
    body: QueryRequest,
    request: Request,
    session_id: str = Depends(require_session),
) -> StreamingResponse:
    require_budget()

    allowed, retry_after = query_limiter.check(session_id)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit reached ({config.MAX_QUERIES_PER_HOUR} questions per hour).",
            headers={"Retry-After": str(retry_after)},
        )

    try:
        question = guards.validate_query(body.query)
    except guards.ValidationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    session = sessions.get_or_create(session_id)
    if not session.has_documents:
        raise HTTPException(
            status_code=400,
            detail="Upload a PDF before asking questions.",
        )

    return StreamingResponse(
        _stream_query(session_id, session.chunks_metadata, question),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # stop any reverse proxy from buffering the stream
            "Connection": "keep-alive",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Work
# ─────────────────────────────────────────────────────────────────────────────

async def _stream_query(
    session_id: str,
    chunks_metadata: List[Dict[str, Any]],
    question: str,
) -> AsyncGenerator[str, None]:
    """Run the guarded pipeline, emitting SSE frames."""
    usage: List[Dict[str, Any]] = []
    started = time.time()

    try:
        # Free regex screen, before any embedding or classifier call.
        pattern = guards.precheck_injection(question)
        if pattern:
            logger.info(f"Session {session_id} query blocked by pattern {pattern!r}")
            yield sse({
                "type": "refusal",
                "content": REFUSAL_TEXT,
                "reason": "blocked_pattern",
            })
            yield sse({"type": "done", "refused": True, "confidence": 0.0,
                       "refusal_reason": "blocked_pattern",
                       "processing_time": time.time() - started})
            return

        # Handed to the pipeline so it fires only after retrieval finds a match;
        # off-topic questions never reach it.
        async def intent_check():
            return await guards.check_intent(question, usage)

        async for event in stream_answer(
            question,
            chunks_metadata,
            session_id=session_id,
            usage_sink=usage,
            intent_check=intent_check,
        ):
            yield sse(event)

    except asyncio.CancelledError:
        # Visitor closed the tab.
        logger.info(f"Session {session_id} disconnected mid-stream")
        raise
    except Exception as exc:
        logger.error(f"Query failed for session {session_id}: {exc}", exc_info=True)
        # Generic message: exception text leaks paths and internal structure.
        yield sse({"type": "error", "content": "Something went wrong answering that. Please try again."})
    finally:
        if usage:
            charged = ledger.record(usage)
            logger.info(
                "Session %s charged $%.6f (%d calls) | day total $%.4f/%.2f",
                session_id, charged, len(usage),
                ledger.snapshot()["spent_usd"], config.DAILY_USD_CAP,
            )


async def _ingest(session_id: str, job: IngestJob, data: bytes, filename: str) -> None:
    """Parse, embed, and index one uploaded PDF into the caller's session."""
    import tempfile
    from pathlib import Path

    from database import vector_db
    from embedder import embedder

    tmp_path: Optional[str] = None
    try:
        # The parser works from a path, so the bytes have to land on disk.
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        job.status = "parsing"
        from main import process_document

        doc_id, chunks_meta, texts, chunks = await asyncio.wait_for(
            asyncio.to_thread(process_document, tmp_path),
            timeout=config.PARSE_TIMEOUT_SECONDS,
        )

        # source_filename is shown in citations; session_id scopes retrieval so
        # other visitors cannot match these chunks.
        for chunk in chunks:
            chunk.metadata["source_filename"] = filename
            chunk.metadata["session_id"] = session_id
        for meta in chunks_meta:
            meta.setdefault("metadata", {})["source_filename"] = filename
            meta["metadata"]["session_id"] = session_id

        job.chunks = len(chunks)
        job.status = "embedding"
        dense = await asyncio.to_thread(embedder.embed_texts, texts, None)

        # Uploads would otherwise be invisible to the daily cap.
        ledger.record_usd(_estimate_embedding_cost(texts))

        sparse = await asyncio.to_thread(embedder.create_sparse_vectors_batch, texts)

        job.status = "indexing"
        await asyncio.to_thread(vector_db.index_chunks, chunks, dense, sparse)

        sessions.add_document(session_id, doc_id, filename, chunks_meta)

        job.status = "ready"
        job.finished_at = time.time()
        logger.info(f"Job {job.job_id} ready: {filename}, {len(chunks)} chunks")

    except asyncio.TimeoutError:
        job.status = "failed"
        job.error = f"Parsing took longer than {config.PARSE_TIMEOUT_SECONDS}s and was stopped."
        job.finished_at = time.time()
        logger.warning(f"Job {job.job_id} timed out on {filename}")
    except Exception as exc:
        job.status = "failed"
        job.error = "This document could not be processed."
        job.finished_at = time.time()
        logger.error(f"Job {job.job_id} failed on {filename}: {exc}", exc_info=True)
    finally:
        if tmp_path:
            with contextlib.suppress(OSError):
                Path(tmp_path).unlink()


def _estimate_embedding_cost(texts: List[str]) -> float:
    """Approximate embedding spend from character count.

    ~4 characters per token, rather than re-running tiktoken. Close enough for
    the cap.
    """
    model = config.OPENAI_EMBEDDING_MODEL
    rate = config.PRICE_PER_MTOK.get(model, {"input": 0.130})["input"]
    approx_tokens = sum(len(t) for t in texts) / 4
    return approx_tokens / 1_000_000 * rate
