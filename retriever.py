from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
import config
from database import vector_db
from embedder import embedder
import logging
import numpy as np
import time
import re
import asyncio
import httpx

logger = logging.getLogger(__name__)

class EnhancedRetriever:

    def __init__(self):
        # Local CrossEncoder removed. Now relies on RERANKER_API_URL
        pass

    def _extract_query_numbers(self, query: str) -> List[str]:
        patterns = [
            r'-?\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:[KMBkmb](?:illion)?)?',
            r'-?\d+(?:\.\d+)?%',
            r'\b(?:19|20)\d{2}\b',
            r'\b(?:Q[1-4]|FY)\s*\d{2,4}\b',
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        ]
        numbers = []
        for p in patterns:
            numbers.extend(re.findall(p, query, re.IGNORECASE))
        return list(set(numbers))

    def _get_adaptive_weights(self, query: str) -> Tuple[float, float]:
        q = query.lower()
        has_numbers = bool(re.search(r'\d', query))
        exact_keywords = sum(1 for kw in ['code','number','id','exact','name','date','year','price','url','github','parameter'] if kw in q)
        semantic_keywords = sum(1 for kw in ['how','why','what is','explain','describe','summary','overview','concept','purpose','role'] if kw in q)

        if has_numbers or exact_keywords > semantic_keywords:
            return 0.4, 0.6
        return config.DENSE_WEIGHT, config.SPARSE_WEIGHT

    async def retrieve_context(
        self,
        query: str,
        chunks_metadata: List[Dict[str, Any]],
        document_id: Optional[UUID] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        start_time = time.time()
        exp = config.MODE

        query_numbers = self._extract_query_numbers(query)

        candidates = await self._hybrid_search(query, document_id, session_id)

        if not candidates:
            return self._empty_context()

        if query_numbers:
            candidates = self._boost_numeric_matches(candidates, query_numbers, chunks_metadata)

        candidates, rerank_scores = await self._rerank(query, candidates, exp["top_k_rerank"])

        candidates, rerank_scores = self._adaptive_rerank_filter(candidates, rerank_scores)

        # Absolute relevance floor. _adaptive_rerank_filter is relative — it
        # keeps the best of whatever it was given, so a query with no genuine
        # match still yields chunks. This gate is what lets an off-topic
        # question ("write me a Python script") be refused before any LLM call
        # is made, which is both the cheapest and the most reliable guard
        # against the hosted app being used as a general-purpose chatbot.
        candidates, rerank_scores = self._absolute_relevance_gate(candidates, rerank_scores)
        if not candidates:
            ctx = self._empty_context()
            ctx["metrics"]["rejected_by_relevance_gate"] = True
            ctx["metrics"]["retrieval_time_ms"] = (time.time() - start_time) * 1000
            return ctx

        candidates = self._deduplicate_fast(candidates, exp["dedup_threshold"])

        context_data = self._build_context(candidates, chunks_metadata, rerank_scores[:len(candidates)])
        context_data["metrics"]["retrieval_time_ms"] = (time.time() - start_time) * 1000
        return context_data

    async def _hybrid_search(
        self,
        query: str,
        document_id: Optional[UUID] = None,
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        dense_w, sparse_w = self._get_adaptive_weights(query)

        dense_emb, sparse_vec = await asyncio.gather(
            asyncio.to_thread(embedder.embed_query, query),
            asyncio.to_thread(embedder.create_sparse_vector, query, "query")
        )

        dense_res, sparse_res = await asyncio.gather(
            asyncio.to_thread(
                vector_db.dense_search, document_id, dense_emb, config.TOP_K_INITIAL, session_id
            ),
            asyncio.to_thread(
                vector_db.sparse_search, document_id, sparse_vec, config.TOP_K_INITIAL, session_id
            )
        )

        return self._fuse_results(dense_res, sparse_res, dense_w, sparse_w)

    def _fuse_results(self, dense: List, sparse: List, dw: float, sw: float) -> List[Dict[str, Any]]:
        k = config.RRF_K_PARAM
        scores: Dict[str, float] = {}
        pool: Dict[str, Dict] = {}

        for rank, r in enumerate(dense, 1):
            cid = r["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + dw / (k + rank)
            # Preserve the raw cosine similarity before RRF overwrites "score".
            # RRF is rank-based, so its output says nothing about whether a
            # chunk is actually relevant — the top hit of a hopeless query
            # scores the same as the top hit of a perfect one. The cosine
            # similarity is an absolute measure, and it is what
            # _absolute_relevance_gate needs.
            r["dense_score"] = r.get("score", 0.0)
            pool[cid] = r

        for rank, r in enumerate(sparse, 1):
            cid = r["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + sw / (k + rank)
            r.setdefault("dense_score", 0.0)
            pool.setdefault(cid, r)

        ranked = sorted(scores, key=lambda x: scores[x], reverse=True)
        for cid in ranked:
            pool[cid]["score"] = scores[cid]
        return [pool[cid] for cid in ranked]

    def _absolute_relevance_gate(
        self, candidates: List[Dict], scores: List[float]
    ) -> Tuple[List[Dict], List[float]]:
        """Reject the whole result set when nothing is genuinely on-topic.

        Uses dense cosine similarity rather than the rerank/RRF score because
        cosine is comparable across queries. With normalized embeddings,
        unrelated text lands around 0.0-0.15 and related text around 0.3-0.6,
        so config.RELEVANCE_FLOOR sits between those bands.
        """
        floor = getattr(config, "RELEVANCE_FLOOR", 0.0)
        if floor <= 0 or not candidates:
            return candidates, scores

        best = max((c.get("dense_score", 0.0) for c in candidates), default=0.0)

        if config.LOG_RETRIEVAL_SCORES:
            logger.info(f"Relevance gate: best dense cosine {best:.4f} vs floor {floor:.4f}")

        if best < floor:
            logger.info(f"Query rejected by relevance gate (best {best:.4f} < floor {floor:.4f})")
            return [], []

        return candidates, scores

    def _boost_numeric_matches(self, candidates: List[Dict], query_numbers: List[str], chunks_metadata: List[Dict]) -> List[Dict]:
        meta_map = {c["id"]: c for c in chunks_metadata}
        boost = config.MODE["numeric_boost_factor"]
        for c in candidates:
            meta = meta_map.get(c["chunk_id"])
            if not meta:
                continue
            chunk_nums = [n["value"] for n in meta.get("metadata", {}).get("numbers", [])]
            if any(qn in chunk_nums for qn in query_numbers):
                c["score"] *= boost
                c["numeric_boost"] = True
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates

    async def _rerank(self, query: str, candidates: List[Dict], top_k: int) -> Tuple[List[Dict], List[float]]:
        if not candidates:
            return [], []
        
        # External Reranker API Integration
        api_url = getattr(config, "RERANKER_API_URL", "")
        if not api_url:
            # Fallback if no API is configured: just return candidates sorted by initial score
            sorted_cands = sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)[:top_k]
            return sorted_cands, [c.get("score", 0.0) for c in sorted_cands]

        pairs = [{"query": query, "text": c["text"]} for c in candidates]
        
        async with httpx.AsyncClient() as client:
            try:
                headers = {"Authorization": f"Bearer {config.RERANKER_API_KEY}"} if getattr(config, "RERANKER_API_KEY", "") else {}
                payload = {"pairs": pairs} # Custom payload. Edit this if using Cohere vs custom microservice
                response = await client.post(api_url, json=payload, headers=headers)
                response.raise_for_status()
                scores = response.json().get("scores", [])
            except Exception as e:
                logger.warning(f"Reranker API fallback triggered. Failed at {api_url}: {e}")
                # Fallback to initial sparse/dense scores
                scores = [c.get("score", 0.0) for c in candidates]
                
        # The reranker is an external service, so its response length is not
        # guaranteed. A short list would leave later candidates without a
        # rerank_score and the sort below would raise KeyError, so fall back to
        # the fusion scores instead.
        if len(scores) != len(candidates):
            logger.warning(
                f"Reranker returned {len(scores)} scores for {len(candidates)} candidates; "
                "using fusion scores instead."
            )
            scores = [c.get("score", 0.0) for c in candidates]

        for c, s in zip(candidates, scores, strict=True):
            c["rerank_score"] = float(s)

        sorted_cands = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
        return sorted_cands, [c["rerank_score"] for c in sorted_cands]

    def _adaptive_rerank_filter(self, candidates: List[Dict], scores: List[float]) -> Tuple[List[Dict], List[float]]:
        if not scores:
            return candidates, scores
        mean = np.mean(scores)
        std = np.std(scores)
        threshold = mean - 0.5 * std
        # strict=True: these two lists are built together and must stay aligned.
        # Without it a length mismatch would silently drop candidates instead of
        # failing.
        filtered = [(c, s) for c, s in zip(candidates, scores, strict=True) if s >= threshold]
        if len(filtered) < 3:
            return candidates[:5], scores[:5]
        fc, fs = zip(*filtered, strict=True)
        return list(fc), list(fs)

    def _deduplicate_fast(self, candidates: List[Dict], threshold: float) -> List[Dict]:
        if not candidates:
            return candidates
        keep = []
        seen_texts = []
        for c in candidates:
            text = c["text"].lower().strip()
            words = set(text.split())
            is_dup = False
            for seen in seen_texts:
                intersection = len(words & seen)
                union = len(words | seen)
                if union > 0 and intersection / union >= threshold:
                    is_dup = True
                    break
            if not is_dup:
                keep.append(c)
                seen_texts.append(words)
        return keep

    def _build_context(self, candidates: List[Dict], chunks_metadata: List[Dict], rerank_scores: List[float]) -> Dict[str, Any]:
        meta_map = {str(c["id"]): c for c in chunks_metadata}
        context_parts = []
        sources = []
        total_tokens = 0
        pw = config.PARENT_CONTEXT_WINDOW

        for candidate in candidates:
            cid = str(candidate["chunk_id"])
            meta = meta_map.get(cid)
            if not meta:
                continue
            parent_id = meta.get("parent_id")
            if parent_id:
                parent = meta_map.get(str(parent_id))
                if parent:
                    pt = parent["text"]
                    ct = meta["text"]
                    start = pt.find(ct)
                    if start >= 0:
                        expanded = pt[max(0, start - pw): min(len(pt), start + len(ct) + pw)]
                    else:
                        expanded = ct
                else:
                    expanded = meta["text"]
            else:
                expanded = meta["text"]

            page = meta.get("page_number", 1)
            source = candidate.get("source_filename", meta.get("metadata", {}).get("source_filename", "unknown"))
            context_parts.append(f"[[Source: {source} | Page: {page}]]\n{expanded}")
            sources.append({"chunk_id": cid, "text": expanded, "page": page, "source_filename": source, "score": candidate.get("rerank_score", candidate.get("score", 0.0))})
            total_tokens += meta.get("token_count", len(expanded.split()))
            if total_tokens >= config.MAX_CONTEXT_TOKENS:
                break

        all_scores = [c.get("rerank_score", c.get("score", 0.0)) for c in candidates]
        avg_r = float(np.mean(all_scores)) if all_scores else 0.0
        return {
            "context": "\n\n".join(context_parts),
            "sources": sources,
            "total_tokens": total_tokens,
            "retrieved_chunks": candidates,
            "rerank_scores": all_scores,
            "metrics": {
                "total_candidates": len(candidates),
                "final_chunks": len(candidates),
                "avg_rerank_score": avg_r,
                "max_rerank_score": float(max(all_scores)) if all_scores else 0.0,
                "min_rerank_score": float(min(all_scores)) if all_scores else 0.0,
            }
        }

    def _empty_context(self) -> Dict[str, Any]:
        return {
            "context": "", "sources": [], "total_tokens": 0,
            "retrieved_chunks": [], "rerank_scores": [],
            "metrics": {"total_candidates": 0, "final_chunks": 0, "avg_rerank_score": 0.0, "max_rerank_score": 0.0, "min_rerank_score": 0.0}
        }

retriever = EnhancedRetriever()
