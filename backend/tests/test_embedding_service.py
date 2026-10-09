"""The embedding provider client: batching, ordering, and width validation.

No network: the OpenAI client is replaced with a stub, so what is under test is the
request shaping around the provider rather than the provider itself.
"""

from types import SimpleNamespace

import pytest

from app.services import embedding_service


def _stub_client(recorder, width=None, returned=None):
    """A stand-in whose vectors are identifiable by the index they were built for."""
    resolved_width = width or embedding_service.EMBEDDING_DIM

    class _Embeddings:
        def create(self, model, input):
            recorder.append(list(input))
            count = len(input) if returned is None else returned
            data = [
                SimpleNamespace(index=index, embedding=[float(index)] * resolved_width)
                for index in range(count)
            ]
            # Reversed on purpose: the service must restore the caller's order, because a
            # vector attached to the wrong chunk is a silent, invisible corruption.
            return SimpleNamespace(data=list(reversed(data)))

    return SimpleNamespace(embeddings=_Embeddings())


def test_embeds_in_batches_of_at_most_ten(monkeypatch: pytest.MonkeyPatch) -> None:
    """The provider answers an oversized batch with HTTP 400, so the split is load-bearing."""
    batches: list[list[str]] = []
    monkeypatch.setattr(embedding_service, "embedding_client", lambda: _stub_client(batches))

    vectors = embedding_service.embed_texts([f"文本{index}" for index in range(23)])

    assert len(vectors) == 23
    assert [len(batch) for batch in batches] == [10, 10, 3]


def test_restores_the_provider_order(monkeypatch: pytest.MonkeyPatch) -> None:
    batches: list[list[str]] = []
    monkeypatch.setattr(embedding_service, "embedding_client", lambda: _stub_client(batches))

    vectors = embedding_service.embed_texts([f"文本{index}" for index in range(5)])

    assert [vector[0] for vector in vectors] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_returns_nothing_for_no_input(monkeypatch: pytest.MonkeyPatch) -> None:
    def _explode():  # pragma: no cover - must never be reached
        raise AssertionError("the provider must not be called for an empty input")

    monkeypatch.setattr(embedding_service, "embedding_client", _explode)
    assert embedding_service.embed_texts([]) == []


def test_reports_a_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embedding_service, "embedding_client", lambda: None)
    with pytest.raises(embedding_service.EmbeddingConfigurationError):
        embedding_service.embed_texts(["文本"])


def test_rejects_a_vector_of_the_wrong_width(monkeypatch: pytest.MonkeyPatch) -> None:
    """A narrower vector would be stored happily and then never match a query."""
    batches: list[list[str]] = []
    monkeypatch.setattr(
        embedding_service, "embedding_client", lambda: _stub_client(batches, width=512)
    )
    with pytest.raises(embedding_service.EmbeddingError, match="dimension"):
        embedding_service.embed_texts(["文本"])


def test_rejects_a_short_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fewer vectors than inputs would leave a chunk in the collection with no vector."""
    batches: list[list[str]] = []
    monkeypatch.setattr(
        embedding_service, "embedding_client", lambda: _stub_client(batches, returned=2)
    )
    with pytest.raises(embedding_service.EmbeddingError):
        embedding_service.embed_texts(["一", "二", "三"])


def test_wraps_a_provider_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Exploding:
        def create(self, model, input):
            raise RuntimeError("connection reset")

    monkeypatch.setattr(
        embedding_service, "embedding_client", lambda: SimpleNamespace(embeddings=_Exploding())
    )
    with pytest.raises(embedding_service.EmbeddingError, match="connection reset"):
        embedding_service.embed_texts(["文本"])
