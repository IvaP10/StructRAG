"""Per-session document state.

The retriever needs `chunks_metadata` to expand a matched child chunk into its
parent text (retriever._build_context). The CLI keeps one list for the process;
the server keeps one per session and evicts on expiry.

In-process and single-container. With more than one replica, an upload and the
following query could land on different processes, so this would have to move
into Qdrant or Redis.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import UUID

logger = logging.getLogger(__name__)


@dataclass
class IngestJob:
    """Progress of a single upload, polled by the frontend."""
    job_id: str
    filename: str
    status: str = "queued"          # queued | parsing | embedding | indexing | ready | failed
    pages: int = 0
    chunks: int = 0
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "filename": self.filename,
            "status": self.status,
            "pages": self.pages,
            "chunks": self.chunks,
            "error": self.error,
            "elapsed_seconds": round((self.finished_at or time.time()) - self.started_at, 2),
        }


@dataclass
class SessionData:
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    chunks_metadata: List[Dict[str, Any]] = field(default_factory=list)
    doc_ids: List[UUID] = field(default_factory=list)
    filenames: List[str] = field(default_factory=list)
    jobs: Dict[str, IngestJob] = field(default_factory=dict)

    @property
    def has_documents(self) -> bool:
        return bool(self.chunks_metadata)

    def summary(self) -> Dict[str, Any]:
        return {
            "documents": len(self.doc_ids),
            "filenames": list(self.filenames),
            "chunks": len(self.chunks_metadata),
            "pages": len({c.get("page_number") for c in self.chunks_metadata}),
        }


class SessionStore:
    def __init__(self, ttl_seconds: int):
        self.ttl_seconds = ttl_seconds
        self._sessions: Dict[str, SessionData] = {}
        self._lock = threading.Lock()

    def create(self, session_id: str) -> SessionData:
        with self._lock:
            session = SessionData(session_id=session_id)
            self._sessions[session_id] = session
            return session

    def get(self, session_id: str) -> Optional[SessionData]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                session.last_seen = time.time()
            return session

    def get_or_create(self, session_id: str) -> SessionData:
        """Recreate state for a token that outlived its in-memory session.

        Tokens are self-contained, so a valid one can arrive after a restart
        cleared this store. Returns an empty session: the visitor keeps their
        seat but must re-upload.
        """
        session = self.get(session_id)
        if session is None:
            logger.info(f"Rebuilding empty session state for {session_id}")
            session = self.create(session_id)
        return session

    def add_document(
        self,
        session_id: str,
        doc_id: UUID,
        filename: str,
        chunks_metadata: List[Dict[str, Any]],
    ) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            session.doc_ids.append(doc_id)
            session.filenames.append(filename)
            session.chunks_metadata.extend(chunks_metadata)
            session.last_seen = time.time()

    def expired_ids(self) -> List[str]:
        cutoff = time.time() - self.ttl_seconds
        with self._lock:
            return [sid for sid, s in self._sessions.items() if s.last_seen < cutoff]

    def drop(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "active_sessions": len(self._sessions),
                "total_chunks": sum(len(s.chunks_metadata) for s in self._sessions.values()),
            }
