"""Shared fixtures.

Environment variables are set before any application module is imported,
because config.py reads them at import time.
"""

from __future__ import annotations

import os

# Must precede the config import. Values are obvious placeholders so a leaked
# test log cannot be mistaken for real credentials.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-a-real-key")
os.environ.setdefault("INVITE_CODE", "test-invite-code-1234")
os.environ.setdefault("SESSION_SECRET", "test-session-secret-value-32-chars")
os.environ.setdefault("ALLOWED_ORIGINS", "https://ivap10.github.io")
os.environ.setdefault("DAILY_USD_CAP", "1.00")
os.environ.setdefault("LOG_FILE", "")
os.environ.setdefault("CACHE_DIR", "/tmp/structrag-test-cache")  # noqa: S108
os.environ.setdefault("QDRANT_PATH", "")
os.environ.setdefault("DISABLED", "0")

import pytest  # noqa: E402


@pytest.fixture
def pdf_bytes():
    """A minimal but genuinely valid one-page PDF containing real text."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 100),
        "Acme Corporation Annual Report\n"
        "Total revenue for FY2023 was $5,200,000.\n"
        "Operating margin improved to 18.4 percent.",
        fontsize=11,
    )
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def multipage_pdf_bytes():
    """A valid PDF with more pages than config.MAX_PDF_PAGES allows."""
    import fitz

    import config

    doc = fitz.open()
    for i in range(config.MAX_PDF_PAGES + 5):
        page = doc.new_page()
        page.insert_text((72, 100), f"Page {i + 1} of a long document.", fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def fresh_state():
    """Reset the module-level server state between tests.

    sessions/ledger/limiters are process-global singletons, so without this a
    rate-limit test would poison whichever test ran next.
    """
    from server import app as app_module

    app_module.sessions._sessions.clear()
    app_module.query_limiter._events.clear()
    app_module.upload_limiter._events.clear()
    app_module.session_limiter._events.clear()
    app_module.ledger._spent = 0.0
    app_module.ledger._requests = 0
    yield
    app_module.sessions._sessions.clear()


@pytest.fixture
def client(fresh_state):
    """TestClient with the app's own lifespan run, so startup checks execute."""
    from fastapi.testclient import TestClient

    from server.app import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers(client):
    """A valid Bearer header from a real invite-code exchange."""
    import config

    response = client.post("/api/session", json={"invite_code": config.INVITE_CODE})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['token']}"}
