"""Regression test for the embedding batching bug.

main.py used to call `embedder.embed_texts(texts, False)`. The second positional
parameter is batch_size, so batch_size became False, and the flush condition
`len(current_batch) >= batch_size` degenerated into `>= 0` — always true. Every
chunk was sent as its own HTTP request.

A 200-page PDF therefore fired thousands of OpenAI calls instead of one or two.
That is slow, rate-limit prone, and directly expensive, which makes it worth a
permanent test rather than a one-line fix and a hope.
"""

from __future__ import annotations

import numpy as np
import pytest

from embedder import EnhancedEmbedder


class FakeEmbeddings:
    """Stand-in for client.embeddings, counting how often it is called."""

    def __init__(self, dimension: int = 8):
        self.calls = 0
        self.batch_sizes: list[int] = []
        self.dimension = dimension

    def create(self, model: str, input: list[str]):  # noqa: A002 - matches the SDK signature
        self.calls += 1
        self.batch_sizes.append(len(input))

        class Item:
            def __init__(self, vector):
                self.embedding = vector

        class Response:
            def __init__(self, items):
                self.data = items

        return Response([Item([0.1] * self.dimension) for _ in input])


class FakeClient:
    def __init__(self):
        self.embeddings = FakeEmbeddings()


@pytest.fixture
def embedder(tmp_path, monkeypatch):
    """A real EnhancedEmbedder with the network replaced and caching disabled."""
    instance = EnhancedEmbedder()
    instance.cache_dir = tmp_path
    instance.enable_cache = False          # otherwise the second call short-circuits
    instance.openai_client = FakeClient()
    instance._openai_ready = True          # skip real client construction
    instance.dimension = 8

    monkeypatch.setattr("config.EMBEDDING_DIMENSION", 8, raising=False)
    return instance


def test_many_texts_are_sent_as_one_request(embedder):
    """The actual regression: 250 chunks must not become 250 API calls."""
    texts = [f"Chunk number {i} with some text in it." for i in range(250)]

    embedder.embed_texts(texts)

    assert embedder.openai_client.embeddings.calls == 1, (
        f"expected 1 batched request, got "
        f"{embedder.openai_client.embeddings.calls} — batching is broken again"
    )
    assert embedder.openai_client.embeddings.batch_sizes == [250]


@pytest.mark.parametrize("bad_batch_size", [False, 0, None, -1])
def test_falsy_batch_size_falls_back_to_the_configured_default(embedder, bad_batch_size):
    """The exact shape of the original bug: a falsy batch_size must not mean 'one per call'."""
    texts = [f"Text {i}" for i in range(60)]

    embedder.embed_texts(texts, bad_batch_size)

    assert embedder.openai_client.embeddings.calls == 1, (
        f"batch_size={bad_batch_size!r} produced "
        f"{embedder.openai_client.embeddings.calls} requests instead of 1"
    )


def test_an_explicit_batch_size_is_still_honoured(embedder):
    texts = [f"Text {i}" for i in range(100)]

    embedder.embed_texts(texts, 25)

    assert embedder.openai_client.embeddings.calls == 4
    assert embedder.openai_client.embeddings.batch_sizes == [25, 25, 25, 25]


def test_returned_embeddings_are_normalised(embedder):
    result = embedder.embed_texts(["one", "two", "three"])
    assert result.shape[0] == 3
    norms = np.linalg.norm(result, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_empty_input_makes_no_request(embedder):
    assert embedder.embed_texts([]).size == 0
    assert embedder.openai_client.embeddings.calls == 0
