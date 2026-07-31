"""Config validation and reranker robustness.

Both cover cases where the previous code failed quietly rather than loudly.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from retriever import retriever


# ── Config validation survives -O ────────────────────────────────────────────

def run_config(env_overrides: dict, optimised: bool) -> subprocess.CompletedProcess:
    """Import config in a subprocess, optionally with -O."""
    env = dict(os.environ)
    env.update({k: str(v) for k, v in env_overrides.items()})
    env["LOG_FILE"] = ""

    argv = [sys.executable]
    if optimised:
        argv.append("-O")
    argv += ["-c", "import config; print('imported')"]

    return subprocess.run(argv, capture_output=True, text=True, env=env, timeout=60)


@pytest.mark.parametrize("optimised", [False, True], ids=["normal", "python -O"])
def test_bad_chunk_sizes_are_rejected(optimised):
    """The -O case is the point: assert statements are stripped under -O, so the
    original checks silently disappeared and a broken config booted fine."""
    result = run_config({"CHUNK_SIZE_PARENT": 100, "CHUNK_SIZE_CHILD": 500}, optimised)
    assert result.returncode != 0
    assert "CHUNK_SIZE_PARENT" in result.stderr


@pytest.mark.parametrize("optimised", [False, True], ids=["normal", "python -O"])
def test_bad_weights_are_rejected(optimised):
    result = run_config({"DENSE_WEIGHT": 0.9, "SPARSE_WEIGHT": 0.9}, optimised)
    assert result.returncode != 0
    assert "sum to 1.0" in result.stderr


@pytest.mark.parametrize("optimised", [False, True], ids=["normal", "python -O"])
def test_out_of_range_relevance_floor_is_rejected(optimised):
    result = run_config({"RELEVANCE_FLOOR": 5.0}, optimised)
    assert result.returncode != 0
    assert "RELEVANCE_FLOOR" in result.stderr


def test_valid_config_imports_cleanly():
    result = run_config({}, optimised=False)
    assert result.returncode == 0, result.stderr
    assert "imported" in result.stdout


def test_error_lists_every_problem_at_once():
    """One run should surface all the mistakes, not just the first."""
    result = run_config(
        {"CHUNK_SIZE_PARENT": 100, "CHUNK_SIZE_CHILD": 500, "DENSE_WEIGHT": 0.9, "SPARSE_WEIGHT": 0.9},
        optimised=False,
    )
    assert result.returncode != 0
    assert "CHUNK_SIZE_PARENT" in result.stderr
    assert "sum to 1.0" in result.stderr


# ── Reranker length mismatch ─────────────────────────────────────────────────

def make_candidates(n):
    return [{"chunk_id": str(i), "text": f"chunk {i}", "score": 1.0 / (i + 1)} for i in range(n)]


async def test_short_reranker_response_falls_back(monkeypatch):
    """An external reranker returning too few scores used to leave candidates
    without a rerank_score, so the sort raised KeyError."""
    import config

    monkeypatch.setattr(config, "RERANKER_API_URL", "https://reranker.invalid/score", raising=False)
    monkeypatch.setattr(config, "RERANKER_API_KEY", "", raising=False)

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"scores": [0.9, 0.8]}       # only 2 for 5 candidates

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr("retriever.httpx.AsyncClient", lambda *a, **k: FakeClient())

    candidates, scores = await retriever._rerank("query", make_candidates(5), top_k=5)

    assert len(candidates) == 5
    assert len(scores) == 5
    assert all("rerank_score" in c for c in candidates)


async def test_matching_reranker_response_is_used(monkeypatch):
    import config

    monkeypatch.setattr(config, "RERANKER_API_URL", "https://reranker.invalid/score", raising=False)
    monkeypatch.setattr(config, "RERANKER_API_KEY", "", raising=False)

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"scores": [0.1, 0.9, 0.5]}

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr("retriever.httpx.AsyncClient", lambda *a, **k: FakeClient())

    candidates, scores = await retriever._rerank("query", make_candidates(3), top_k=3)

    assert scores == [0.9, 0.5, 0.1]           # sorted descending


def test_adaptive_filter_keeps_candidates_and_scores_aligned():
    candidates = make_candidates(6)
    scores = [0.9, 0.85, 0.8, 0.2, 0.1, 0.05]
    for c, s in zip(candidates, scores, strict=True):
        c["rerank_score"] = s

    filtered_candidates, filtered_scores = retriever._adaptive_rerank_filter(candidates, scores)

    assert len(filtered_candidates) == len(filtered_scores)
    for c, s in zip(filtered_candidates, filtered_scores, strict=True):
        assert c["rerank_score"] == s


def test_adaptive_filter_handles_empty_input():
    assert retriever._adaptive_rerank_filter([], []) == ([], [])
