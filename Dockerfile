# StructRAG API image — this is what Render builds and runs, but it is a plain
# uvicorn container with no host-specific assumptions and runs anywhere.
#
# Build and run locally:
#   docker build -t structrag .
#   docker run --rm -p 7860:7860 --env-file key.env structrag
#
# Two-stage build keeps compilers and wheel caches out of the shipped layer:
# faster cold starts, fewer packages for a CVE to live in.

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — build wheels
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.14-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Needed to compile any package without a manylinux wheel for this platform.
RUN apt-get update && apt-get install --no-install-recommends -y \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip setuptools wheel \
    && /opt/venv/bin/pip install -r requirements.txt


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — runtime
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.14-slim AS runtime

# PYTHONDONTWRITEBYTECODE: the app directory is read-only to the runtime user.
# PYTHONUNBUFFERED: without it, logs sit in a buffer and the host's log view
# looks dead.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HOME=/home/app

# libgomp1 is required by pdfplumber's imaging path. No OCR stack: Docling would
# pull in torch and take the image from ~630 MB to ~4 GB.
RUN apt-get update && apt-get install --no-install-recommends -y \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

# Run as an unprivileged user. uid 1000 is what most container hosts assign, so
# matching it keeps the writable paths below actually writable.
RUN useradd --create-home --uid 1000 app

WORKDIR /app

# Owned by root, not writable by the runtime user, so a compromised handler
# cannot rewrite the code it is running.
COPY --chown=root:root config.py models.py chunker.py pdf_parser.py \
     embedder.py database.py retriever.py generator.py main.py ./
COPY --chown=root:root core/ ./core/
COPY --chown=root:root server/ ./server/

# The three paths the app writes to. Everything else stays read-only.
RUN mkdir -p /data/qdrant /data/cache \
    && chown -R app:app /data \
    && chmod -R 700 /data

USER app

# LOG_FILE empty: /app is root-owned and read-only to this user, so main.py's
# default of rag_system.log cannot be created. stdout is what container logs
# read anyway.
ENV QDRANT_PATH=/data/qdrant \
    CACHE_DIR=/data/cache \
    LOG_FILE="" \
    PORT=7860

EXPOSE 7860

# /api/health is the one endpoint the kill switch leaves reachable, so a
# disabled service still reports healthy rather than being restarted in a loop.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/api/health" || exit 1

# One worker: session state and rate-limit counters live in process memory
# (server/store.py), so a second worker would miss a visitor's uploads.
CMD ["sh", "-c", "exec uvicorn server.app:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers --forwarded-allow-ips='*'"]
