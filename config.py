import os
from pathlib import Path
from dotenv import load_dotenv
import logging
import warnings

warnings.filterwarnings('ignore')
for _lg in ['docling','docling_core','docling.document_converter','docling.datamodel',
            'docling.models','docling.pipeline','docling.utils','datasets',
            'httpx']:
    logging.getLogger(_lg).setLevel(logging.ERROR)

load_dotenv("key.env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documind_v2")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Embedded mode. When set, qdrant-client writes to this directory instead of
# talking to a server, so the container runs one process and needs no extra
# service or credentials. This is what the Docker image uses.
#
# Two things to know: the directory is held under an exclusive file lock, so
# only one process may open it at a time (the CLI and the server cannot share
# one), and on a free Hugging Face Space the filesystem is ephemeral — a restart
# discards it. That is intentional here, since uploaded documents should not
# outlive the session that provided them.
QDRANT_PATH = os.getenv("QDRANT_PATH", "")

CHUNK_SIZE_PARENT = int(os.getenv("CHUNK_SIZE_PARENT", 600))
CHUNK_SIZE_CHILD = int(os.getenv("CHUNK_SIZE_CHILD", 300))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 75))

RESPECT_SENTENCE_BOUNDARIES = os.getenv("RESPECT_SENTENCE_BOUNDARIES", "true").lower() == "true"
RESPECT_PARAGRAPH_BOUNDARIES = os.getenv("RESPECT_PARAGRAPH_BOUNDARIES", "true").lower() == "true"

TOP_K_INITIAL = int(os.getenv("TOP_K_INITIAL", 30))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", 15))
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", 4000))

DYNAMIC_THRESHOLD_PERCENTILE = float(os.getenv("DYNAMIC_THRESHOLD_PERCENTILE", 0.70))
DYNAMIC_THRESHOLD_MULTIPLIER = float(os.getenv("DYNAMIC_THRESHOLD_MULTIPLIER", 0.7))

RERANK_SCORE_THRESHOLD = float(os.getenv("RERANK_SCORE_THRESHOLD", 0.08))

# Absolute dense-cosine floor a query must clear to be answered at all.
# Unlike the rerank threshold this is comparable across queries, so it can
# distinguish "the corpus has no answer to this" from "these are the best of a
# weak set". Retrieval returns empty below it and the LLM is never called.
#
# Calibrating: with normalized text-embedding-3-large vectors, unrelated text
# scores roughly 0.0-0.15 and on-topic text 0.3-0.6. Raise it toward 0.35 if
# off-topic questions still get answered; lower it toward 0.20 if legitimate
# questions are being refused. Set to 0 to disable the gate entirely (which is
# what the CLI does, since there is no key to protect there).
RELEVANCE_FLOOR = float(os.getenv("RELEVANCE_FLOOR", 0.28))

RRF_K_PARAM = int(os.getenv("RRF_K_PARAM", 100))

DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", 0.70))
SPARSE_WEIGHT = float(os.getenv("SPARSE_WEIGHT", 0.30))

ENABLE_DEDUPLICATION = True
DEDUP_SIMILARITY_THRESHOLD = float(os.getenv("DEDUP_SIMILARITY_THRESHOLD", 0.90))

ENABLE_MMR = True
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", 0.8))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

EMBEDDING_MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

if "EMBEDDING_DIMENSION" in os.environ:
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION"))
else:
    EMBEDDING_DIMENSION = EMBEDDING_MODEL_DIMENSIONS.get(EMBEDDING_MODEL, 3072)

EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 2048))
OPENAI_EMBEDDING_MAX_TOKENS_PER_REQUEST = int(os.getenv("OPENAI_EMBEDDING_MAX_TOKENS_PER_REQUEST", 300000))

OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

# Optional external reranker microservice. retriever.py reads these via getattr
# and degrades to raw RRF scores when RERANKER_API_URL is empty, which is the
# default. Declared here so the names exist rather than being implicit.
RERANKER_API_URL = os.getenv("RERANKER_API_URL", "")
RERANKER_API_KEY = os.getenv("RERANKER_API_KEY", "")

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.0))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 800))

ABSTENTION_THRESHOLD = float(os.getenv("ABSTENTION_THRESHOLD", 0.20))

CONFIDENCE_CALIBRATION = {
    "use_calibration": True,
    "verification_weight": 0.30,
    "source_quality_weight": 0.25,
    "citation_quality_weight": 0.25,
    "retrieval_confidence_weight": 0.20,
}

MODE = {
    "name": "Hybrid Quality-First",
    "use_dense": True,
    "use_sparse": True,
    "use_reranker": True,
    "apply_score_threshold": False,
    "dynamic_threshold": False,
    "use_query_rewriting": False,
    "use_mmr": False,
    "mmr_lambda": 0.8,
    "adaptive_threshold": False,
    "min_rerank_score": 0.08,
    "dedup_method": "fast",
    "dedup_threshold": 0.90,
    "context_strategy": "child_with_parent_context",
    "top_k_initial": 30,
    "top_k_rerank": 15,
    "abstention_enabled": False,
    "atomic_fact_verification": True,
    "hallucination_penalty": 0.98,
    "numeric_verification": True,
    "numeric_exact_match": True,
    "extract_numeric_context": True,
    "citation_quality_weight": 0.25,
    "require_citations_for_claims": True,
    "citation_recall_threshold": 0.65,
    "rerank_by_query_similarity": True,
    "parent_context_window": 100,
    "dense_weight": 0.70,
    "sparse_weight": 0.30,
    "use_llm_listwise_rerank": False,
    "query_adaptive_weights": True,
    "numeric_boost_factor": 2.0,
}

ENABLE_EMBEDDING_CACHE = os.getenv("ENABLE_EMBEDDING_CACHE", "true").lower() == "true"
CACHE_DIR = Path(os.getenv("CACHE_DIR", ".cache"))
CACHE_DIR.mkdir(exist_ok=True)

LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING")
LOG_RETRIEVAL_SCORES = os.getenv("LOG_RETRIEVAL_SCORES", "true").lower() == "true"

PARENT_CONTEXT_WINDOW = int(os.getenv("PARENT_CONTEXT_WINDOW", 100))


# ── Server mode ───────────────────────────────────────────────────────────────
# Only consumed by server/app.py. The CLI ignores everything below.

# Passphrase visitors exchange for a session token. Empty means the server
# refuses to start, so a misconfigured deploy fails closed instead of opening
# the API to the world.
INVITE_CODE = os.getenv("INVITE_CODE", "")

# Signs session tokens. Empty means the server refuses to start.
SESSION_SECRET = os.getenv("SESSION_SECRET", "")

SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", 2 * 60 * 60))

# Kill switch — every endpoint returns 503 while set.
DISABLED = os.getenv("DISABLED", "0").strip().lower() in ("1", "true", "yes")

# No default: a deploy must name the origins allowed to call it. An empty list
# makes the server refuse to start rather than guess.
ALLOWED_ORIGINS = [
    o.strip().rstrip("/")
    for o in os.getenv("ALLOWED_ORIGINS", "").split(",")
    if o.strip()
]

# ── Spend ceiling ─────────────────────────────────────────────────────────────
# The single most important limit. Everything else is defence in depth; this is
# what actually bounds the bill. Resets at UTC midnight.
DAILY_USD_CAP = float(os.getenv("DAILY_USD_CAP", 0.70))

# USD per 1M tokens, used to estimate spend against the cap. Update if the
# pricing or the chosen models change.
PRICE_PER_MTOK = {
    "gpt-4o-mini":            {"input": 0.150, "output": 0.600},
    "text-embedding-3-large": {"input": 0.130, "output": 0.0},
    "text-embedding-3-small": {"input": 0.020, "output": 0.0},
}

# ── Rate limits ───────────────────────────────────────────────────────────────
MAX_QUERIES_PER_HOUR = int(os.getenv("MAX_QUERIES_PER_HOUR", 60))
MAX_UPLOADS_PER_DAY = int(os.getenv("MAX_UPLOADS_PER_DAY", 10))
MAX_SESSIONS_PER_IP_PER_HOUR = int(os.getenv("MAX_SESSIONS_PER_IP_PER_HOUR", 10))

# ── Input caps ────────────────────────────────────────────────────────────────
MAX_QUERY_CHARS = int(os.getenv("MAX_QUERY_CHARS", 500))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", 50))
PARSE_TIMEOUT_SECONDS = int(os.getenv("PARSE_TIMEOUT_SECONDS", 180))

# Cheap model used only by the intent guard. Deliberately the small one — it
# runs on every query that clears the relevance gate.
INTENT_GUARD_MODEL = os.getenv("INTENT_GUARD_MODEL", "gpt-4o-mini")
ENABLE_INTENT_GUARD = os.getenv("ENABLE_INTENT_GUARD", "true").lower() == "true"


# Validation runs at import. These were assert statements, which Python strips
# entirely under `python -O` — the checks would silently disappear.
_problems = []

if CHUNK_SIZE_PARENT <= CHUNK_SIZE_CHILD:
    _problems.append(f"CHUNK_SIZE_PARENT ({CHUNK_SIZE_PARENT}) must exceed CHUNK_SIZE_CHILD ({CHUNK_SIZE_CHILD}).")
if CHUNK_OVERLAP >= CHUNK_SIZE_CHILD:
    _problems.append(f"CHUNK_OVERLAP ({CHUNK_OVERLAP}) must be smaller than CHUNK_SIZE_CHILD ({CHUNK_SIZE_CHILD}).")
if abs(DENSE_WEIGHT + SPARSE_WEIGHT - 1.0) >= 0.01:
    _problems.append(f"DENSE_WEIGHT + SPARSE_WEIGHT must sum to 1.0, got {DENSE_WEIGHT + SPARSE_WEIGHT}.")
if TOP_K_RERANK > TOP_K_INITIAL:
    _problems.append(f"TOP_K_RERANK ({TOP_K_RERANK}) cannot exceed TOP_K_INITIAL ({TOP_K_INITIAL}).")
if EMBEDDING_DIMENSION <= 0:
    _problems.append(f"EMBEDDING_DIMENSION must be positive, got {EMBEDDING_DIMENSION}.")
if not 0.0 <= RELEVANCE_FLOOR <= 1.0:
    _problems.append(f"RELEVANCE_FLOOR must be between 0 and 1, got {RELEVANCE_FLOOR}.")

if _problems:
    raise ValueError("Invalid configuration:\n  - " + "\n  - ".join(_problems))

del _problems
