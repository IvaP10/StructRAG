from typing import List, Dict, Optional
import numpy as np
import re
import logging
import math
from pathlib import Path
import hashlib
import json
from collections import Counter

try:
    import config
except ImportError:
    class Config:
        OPENAI_API_KEY = ""
        OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
        EMBEDDING_BATCH_SIZE = 2048
        CACHE_DIR = Path("./cache")
        ENABLE_EMBEDDING_CACHE = True
    config = Config()

logger = logging.getLogger(__name__)

# ── Stopwords for sparse vector generation (no downloads needed) ──────────
_STOPWORDS = frozenset({
    'the','a','an','and','or','but','in','on','at','to','for','of','with',
    'from','is','are','was','were','this','that','it','its','be','been',
    'being','have','has','had','do','does','did','will','would','could',
    'should','may','might','can','shall','not','no','nor','so','if','then',
    'than','too','very','just','about','above','after','again','all','also',
    'am','any','as','because','before','between','both','by','each','few',
    'further','here','how','into','more','most','my','myself','only','other',
    'our','out','over','own','same','she','he','her','him','his','some',
    'such','there','their','them','they','these','those','through','under',
    'until','up','us','we','what','when','where','which','while','who',
    'whom','why','you','your','i','me','s','t','d','ll','re','ve','m',
})


class EnhancedEmbedder:

    def __init__(self):
        self._openai_ready = False
        self._tiktoken_ready = False
        self._embed_single_cache = {}
        self._encoding = None
        self.openai_client = None
        self.embedding_type = None
        self.dimension = None

        self.cache_dir = config.CACHE_DIR / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enable_cache = config.ENABLE_EMBEDDING_CACHE

        # IDF corpus stats for sparse vectors (built during indexing)
        self._doc_count = 0
        self._doc_freq: Counter = Counter()

    def _ensure_tiktoken(self):
        if not self._tiktoken_ready:
            import tiktoken
            self._encoding = tiktoken.get_encoding("cl100k_base")
            self._tiktoken_ready = True

    def _ensure_openai(self):
        if not self._openai_ready:
            self._init_openai_embedder()
            self._openai_ready = True

    def _init_openai_embedder(self):
        try:
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
            self.embedding_type = "openai"

            dimension_map = {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536
            }
            self.dimension = dimension_map.get(config.OPENAI_EMBEDDING_MODEL, 3072)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI embeddings: {e}")
            raise

    # ── Caching ───────────────────────────────────────────────────────────

    def _get_cache_key(self, texts: List[str]) -> str:
        content = json.dumps(texts, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def _load_cached_embeddings(self, cache_key: str) -> Optional[np.ndarray]:
        if not self.enable_cache:
            return None
        cache_path = self.cache_dir / f"{cache_key}.npy"
        if cache_path.exists():
            return np.load(cache_path)
        return None

    def _save_cached_embeddings(self, cache_key: str, embeddings: np.ndarray):
        if not self.enable_cache:
            return
        cache_path = self.cache_dir / f"{cache_key}.npy"
        np.save(cache_path, embeddings)

    # ── Dense embeddings (OpenAI API) ─────────────────────────────────────

    def embed_texts(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False
    ) -> np.ndarray:
        if not texts:
            return np.array([])

        self._ensure_openai()
        self._ensure_tiktoken()

        # Any falsy or nonsensical value falls back to the configured default.
        # A bare `is None` check let `batch_size=False` through, which made the
        # flush condition in _embed_with_openai always true and produced one API
        # request per text instead of one per batch.
        if not batch_size or batch_size < 1:
            batch_size = config.EMBEDDING_BATCH_SIZE

        target_dim = getattr(config, 'EMBEDDING_DIMENSION', None)

        def _ensure_dim(emb: np.ndarray) -> np.ndarray:
            if target_dim and emb.shape[1] > target_dim:
                emb = emb[:, :target_dim]
                norms = np.linalg.norm(emb, axis=1, keepdims=True)
                norms[norms == 0] = 1e-10
                emb = emb / norms
            return emb

        cache_key = self._get_cache_key(texts)
        cached = self._load_cached_embeddings(cache_key)
        if cached is not None:
            return _ensure_dim(cached)

        try:
            result = self._embed_with_openai(texts, batch_size)
            result = _ensure_dim(result)
            self._save_cached_embeddings(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"Embedding error: {str(e)}")
            raise

    def _embed_with_openai(self, texts: List[str], batch_size: int) -> np.ndarray:
        max_tokens_per_input = 8191
        max_tokens_per_request = 250000
        max_items_per_request = 2048

        if batch_size > max_items_per_request:
            batch_size = max_items_per_request

        truncated_texts = []
        for text in texts:
            tokens = self._encoding.encode(text)
            if len(tokens) > max_tokens_per_input:
                tokens = tokens[:max_tokens_per_input]
                text = self._encoding.decode(tokens)
            truncated_texts.append(text)

        token_counts = [len(self._encoding.encode(t)) for t in truncated_texts]

        batches = []
        current_batch = []
        current_tokens = 0

        for i, (text, tc) in enumerate(zip(truncated_texts, token_counts)):

            if (current_tokens + tc > max_tokens_per_request) or (len(current_batch) >= batch_size):
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0

            current_batch.append(text)
            current_tokens += tc

        if current_batch:
            batches.append(current_batch)

        all_embeddings = []
        logger.info(f"Embedding {len(texts)} texts in {len(batches)} batches.")

        for i, batch in enumerate(batches):
            batch_tokens = sum(len(self._encoding.encode(t)) for t in batch)
            logger.info(f"Batch {i+1}/{len(batches)}: {len(batch)} items, {batch_tokens} tokens")

            try:
                response = self.openai_client.embeddings.create(
                    model=config.OPENAI_EMBEDDING_MODEL,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error in batch {i+1}: {e}")
                raise

        embeddings = np.array(all_embeddings, dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return embeddings / norms

    def embed_single(self, text: str) -> np.ndarray:
        if text in self._embed_single_cache:
            return self._embed_single_cache[text].copy()

        vector = self.embed_texts([text], show_progress=False)[0]

        if len(self._embed_single_cache) >= 1024:
            self._embed_single_cache.pop(next(iter(self._embed_single_cache)))

        self._embed_single_cache[text] = vector.copy()
        return vector.copy()

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_single(query)

    # ── Sparse vectors (lightweight keyword-based, NO model download) ─────

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Fast regex tokenizer — lowercase alphanumeric tokens, skip stopwords."""
        return [
            w for w in re.findall(r'[a-z0-9]+(?:\.[0-9]+)?', text.lower())
            if w not in _STOPWORDS and len(w) > 1
        ]

    @staticmethod
    def _stable_hash(token: str) -> int:
        """Map a token string to a stable integer ID in [0, 2^31).
        Uses first 4 bytes of SHA-256 to avoid collisions across vocab."""
        h = hashlib.sha256(token.encode('utf-8')).digest()
        return int.from_bytes(h[:4], 'big') & 0x7FFFFFFF

    def create_sparse_vector(self, text: str, mode: str = "query") -> Dict[int, float]:
        """Create a sparse BM25-style vector from text.  Instant, no model needed."""
        tokens = self._tokenize(text)
        if not tokens:
            return {}

        tf = Counter(tokens)
        total = len(tokens)

        sparse: Dict[int, float] = {}
        for term, count in tf.items():
            # BM25-like TF saturation:  tf / (tf + 1.2)
            tf_score = count / (count + 1.2)

            # IDF component (falls back to uniform if corpus stats aren't built)
            if self._doc_count > 0:
                df = self._doc_freq.get(term, 0)
                idf = math.log(1 + (self._doc_count - df + 0.5) / (df + 0.5))
            else:
                idf = 1.0

            weight = tf_score * idf
            if weight > 0.01:
                sparse[self._stable_hash(term)] = round(weight, 4)

        return sparse

    def create_sparse_vectors_batch(self, texts: List[str], mode: str = "document") -> List[Dict[int, float]]:
        """Batch sparse vector creation.  Also builds IDF corpus stats on-the-fly."""
        # Build corpus-level document frequency for IDF
        all_token_sets = []
        for text in texts:
            tokens = self._tokenize(text)
            token_set = set(tokens)
            all_token_sets.append(token_set)
            for t in token_set:
                self._doc_freq[t] += 1
            self._doc_count += 1

        # Now generate vectors using the updated IDF stats
        results = []
        for i, text in enumerate(texts):
            results.append(self.create_sparse_vector(text, mode))
        return results

embedder = EnhancedEmbedder()
