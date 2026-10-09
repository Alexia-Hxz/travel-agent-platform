"""Embedding access for semantic search over uploaded travel documents.

Kept apart from :mod:`app.services.vector_service` on purpose: that module owns chunking
and the Chroma collection, while everything about talking to the embedding provider lives
here. The client settings mirror :mod:`app.agents.tools.llm` so one stalled provider call
cannot hold a request open for minutes.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_PATH)

logger = logging.getLogger(__name__)

# text-embedding-v4 is the current DashScope text embedding model. It has to keep
# producing the same width as the vectors already stored in the collection; a model
# change therefore means rebuilding the collection, which is why every stored chunk
# records the model that produced it.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v4").strip() or "text-embedding-v4"
EMBEDDING_DIM = 1024
# The provider rejects oversized batches with HTTP 400 InvalidParameter
# ("batch size is invalid, it should not be larger than 10"), measured against v4.
MAX_BATCH_SIZE = 10
# Chunks are far shorter than this; the cap only stops a pathological single line from
# being sent whole and rejected for exceeding the model's context window.
MAX_INPUT_CHARS = 4000
REQUEST_TIMEOUT = 30.0
MAX_RETRIES = 1
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class EmbeddingError(Exception):
    """The provider could not be reached, or answered with something unusable."""


class EmbeddingConfigurationError(EmbeddingError):
    """No API key, so no document can be embedded."""


def embedding_client() -> OpenAI | None:
    """Return a configured embeddings client, or None when no key is available."""
    load_dotenv(ENV_PATH)
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url=BASE_URL, timeout=REQUEST_TIMEOUT, max_retries=MAX_RETRIES)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed every text, in provider-sized batches, preserving the input order.

    Raises :class:`EmbeddingError` rather than returning a short list: a partial result
    would put documents into the collection without vectors, and they would never be
    found. The collection is created with the cosine metric, so the raw vectors are
    stored as returned and no normalisation is applied here.
    """
    if not texts:
        return []
    client = embedding_client()
    if client is None:
        raise EmbeddingConfigurationError(
            "DASHSCOPE_API_KEY is not set, so uploaded documents cannot be embedded."
        )

    prepared = [text[:MAX_INPUT_CHARS] for text in texts]
    vectors: list[list[float]] = []
    for start in range(0, len(prepared), MAX_BATCH_SIZE):
        batch = prepared[start : start + MAX_BATCH_SIZE]
        try:
            response = client.embeddings.create(model=EMBEDDING_MODEL_NAME, input=batch)
        except Exception as error:
            raise EmbeddingError(f"The embedding provider rejected the request: {error}") from error
        # The API may return items out of order; `index` is what re-establishes the
        # mapping between a vector and the chunk it belongs to.
        ordered = sorted(response.data, key=lambda item: item.index)
        if len(ordered) != len(batch):
            raise EmbeddingError(
                f"The embedding provider returned {len(ordered)} vectors for {len(batch)} inputs."
            )
        for item in ordered:
            vector = [float(value) for value in item.embedding]
            if len(vector) != EMBEDDING_DIM:
                raise EmbeddingError(
                    f"{EMBEDDING_MODEL_NAME} returned a {len(vector)}-dimension vector but the "
                    f"collection holds {EMBEDDING_DIM}-dimension vectors. Rebuild the document "
                    "index or correct EMBEDDING_DIM."
                )
            vectors.append(vector)
    return vectors
