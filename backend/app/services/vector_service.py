"""Chunking, indexing and semantic retrieval over uploaded travel documents.

Storage is a Chroma collection persisted next to the rest of the application's data
(``data/chroma``). Chroma owns the chunk text, its metadata and its vector, so a query
returns everything needed to render a citation without a second lookup.

Two properties are enforced here rather than left to callers:

* **Trip isolation.** Chroma has no partitions, so a filter is the only thing separating
  one traveller's documents from another's. ``where={"trip_id": ...}`` is applied in the
  single private helper every read goes through, so no caller can forget it.
* **No silent empty results.** A missing traveller document and a broken vector store
  look identical from the outside, and that ambiguity is what hid the previous retrieval
  attempt's failure. Every read returns a status alongside its chunks.
"""

from dataclasses import dataclass, field
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.models.trip_file import TripFile
from app.services.embedding_service import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL_NAME,
    EmbeddingError,
    embed_texts,
)


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHROMA_PATH = PROJECT_ROOT / "data" / "chroma"
COLLECTION_NAME = "trip_documents"

# A chunk has to be big enough to carry an order's details (dates, address, order
# number) and small enough that a hit points at one fact rather than half a page.
CHUNK_TARGET_CHARS = 420
CHUNK_OVERLAP_CHARS = 60
# Below this the parsed text carries no retrievable fact: an OCR pass over a blurry
# screenshot, or a PDF whose text layer is missing. Indexing it would produce a chunk
# that can never answer anything, which is a failure the UI must show rather than hide.
MIN_INDEXABLE_CHARS = 40
DEFAULT_RESULT_LIMIT = 5
MAX_RESULT_LIMIT = 20

_ZERO_WIDTH = "\u200b\u200c\u200d\ufeff"

# One client per directory. Chroma refuses a second client for a path it already holds
# with different settings, and the planner reaches this from a worker thread, so the
# cache is guarded.
_clients: dict[str, Any] = {}
_clients_lock = threading.Lock()
_client_error: str | None = None


@dataclass
class DocumentPiece:
    text: str
    char_start: int
    char_end: int
    index: int = 0


@dataclass
class IndexOutcome:
    status: str  # "indexed" | "skipped" | "failed"
    chunk_count: int = 0
    reason: str | None = None


@dataclass
class RetrievedChunk:
    chunk_id: str
    file_id: int
    source_label: str
    chunk_index: int
    text: str
    similarity: float


@dataclass
class SearchOutcome:
    # "ok" | "empty" | "unavailable" | "failed" | "skipped"
    status: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    reason: str | None = None


def chroma_path() -> Path:
    """Where the collection is persisted. Overridable so tests get a throwaway store."""
    override = os.getenv("TRAVEL_AGENT_CHROMA_PATH")
    return Path(override).resolve() if override else DEFAULT_CHROMA_PATH


def _silence_broken_telemetry() -> None:
    """Stop Chroma's PostHog telemetry from logging a failure on every single operation.

    chromadb 0.4.24 calls ``posthog.capture(user_id, event, properties)``, but the resolved
    posthog (7.x) only accepts ``capture(event, **kwargs)``. The call raises TypeError while
    binding its arguments -- before any request is built, so nothing is transmitted -- and
    chromadb logs that TypeError at ERROR level for every add, query and delete.
    ``Settings(anonymized_telemetry=False)`` does not prevent it: chromadb implements that
    setting by assigning ``posthog.disabled``, which this posthog version's module-level
    ``capture`` never consults.

    Both switches are still applied rather than only suppressing the logger, so the
    collection stays non-reporting if posthog is ever downgraded to a compatible release.
    """
    logging.getLogger("chromadb.telemetry.product.posthog").disabled = True
    try:
        import posthog

        posthog.disabled = True
    except Exception:  # pragma: no cover - posthog ships with chromadb, so this is paranoia
        logger.debug("posthog is not importable; nothing to disable.")


def _client() -> Any:
    global _client_error
    key = str(chroma_path())
    with _clients_lock:
        if key in _clients:
            return _clients[key]
        try:
            import chromadb
            from chromadb.config import Settings

            Path(key).mkdir(parents=True, exist_ok=True)
            # Before constructing the client: its very first act is a telemetry event.
            _silence_broken_telemetry()
            client = chromadb.PersistentClient(
                path=key,
                settings=Settings(anonymized_telemetry=False, allow_reset=False),
            )
            _clients[key] = client
            _client_error = None
            return client
        except Exception as error:
            _client_error = f"{type(error).__name__}: {error}"
            logger.error("Chroma is unavailable at %s: %s", key, _client_error)
            raise


def _collection() -> Any:
    """The trip-document collection, created on first use.

    ``get_collection`` is tried before ``create_collection`` because passing metadata to
    ``get_or_create_collection`` on an existing collection is rejected by some Chroma
    releases when the metadata does not match exactly.
    """
    client = _client()
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception:
        return client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def warm_up() -> tuple[bool, str | None]:
    """Create the collection at startup so a broken store fails loudly, not per request."""
    try:
        collection = _collection()
        logger.info(
            "Chroma collection %r ready at %s with %s stored chunks.",
            COLLECTION_NAME,
            chroma_path(),
            collection.count(),
        )
        return True, None
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        logger.error(
            "Chroma could not be opened, so semantic search over uploaded documents is "
            "disabled. Parsing and reading documents still work. (%s)",
            message,
        )
        return False, message


def _clean_lines(text_value: str) -> list[tuple[str, int, int]]:
    """Non-empty lines with their offsets in the normalised text.

    OCR output arrives with stray indentation and zero-width characters, which would
    otherwise be treated as content and skew every offset recorded for a citation.
    """
    normalized = text_value.replace("\r\n", "\n").replace("\r", "\n")
    for character in _ZERO_WIDTH:
        normalized = normalized.replace(character, "")
    lines: list[tuple[str, int, int]] = []
    cursor = 0
    for raw in normalized.split("\n"):
        line = raw.strip()
        if line:
            start = cursor + (len(raw) - len(raw.lstrip()))
            lines.append((line, start, start + len(line)))
        cursor += len(raw) + 1
    return lines


def split_document(
    text_value: str, target: int = CHUNK_TARGET_CHARS, overlap: int = CHUNK_OVERLAP_CHARS
) -> list[DocumentPiece]:
    """Split parsed document text into overlapping, line-aligned pieces.

    Orders and tickets are line-oriented, so pieces are built from whole lines: cutting
    mid-line is what turns "入住 2026-10-02" into a chunk that says only "入住". A line
    longer than the target is the one exception and is split with a stride that leaves
    ``overlap`` characters shared between neighbours.
    """
    lines = _clean_lines(text_value)
    if not lines:
        return []

    pieces: list[DocumentPiece] = []
    current: list[tuple[str, int, int]] = []

    def flush() -> None:
        if not current:
            return
        body = "\n".join(line for line, _, _ in current)
        pieces.append(
            DocumentPiece(text=body, char_start=current[0][1], char_end=current[-1][2], index=len(pieces))
        )

    for line, start, end in lines:
        if len(line) > target:
            flush()
            current = []
            stride = max(1, target - overlap)
            for offset in range(0, len(line), stride):
                window = line[offset : offset + target]
                if not window.strip():
                    break
                pieces.append(
                    DocumentPiece(
                        text=window,
                        char_start=start + offset,
                        char_end=start + offset + len(window),
                        index=len(pieces),
                    )
                )
                if offset + target >= len(line):
                    break
            continue
        if current and sum(len(item[0]) + 1 for item in current) + len(line) > target:
            carried = current[-1] if len(current[-1][0]) <= overlap else None
            flush()
            # The trailing line is re-issued at the head of the next piece so a fact split
            # across the boundary ("订单号" and its value) is still retrievable from one
            # of the two neighbours.
            current = [carried] if carried is not None else []
        current.append((line, start, end))
    flush()
    return pieces


def index_trip_file(session: Session, trip_file: TripFile) -> IndexOutcome:
    """(Re)build every chunk and vector for one uploaded file.

    Safe to call repeatedly: the file's previous chunks are removed first, so re-parsing
    a document never leaves two generations of the same text in the collection. The
    outcome is written back onto the file row, which is what lets the interface say
    "解析成功但未建立索引" instead of showing a document that silently answers nothing.
    """
    parsed = (trip_file.parsed_text or "").strip()
    pieces = split_document(parsed) if len(parsed) >= MIN_INDEXABLE_CHARS else []

    try:
        collection = _collection()
    except Exception as error:
        return _record_outcome(session, trip_file, "failed", 0, f"向量库不可用：{type(error).__name__}: {error}")

    _drop_file(collection, trip_file.id)
    if not pieces:
        reason = (
            f"解析出的文本只有 {len(parsed)} 字，"
            + ("不足以建立索引（可能是扫描件或截图质量太差）。" if parsed else "文件没有解析出任何文本。")
        )
        logger.info("Skipped indexing %s: %s", trip_file.original_filename, reason)
        return _record_outcome(session, trip_file, "skipped", 0, reason)

    try:
        vectors = embed_texts([piece.text for piece in pieces])
    except EmbeddingError as error:
        logger.warning("Embedding failed for %s: %s", trip_file.original_filename, error)
        return _record_outcome(session, trip_file, "failed", 0, str(error))

    try:
        collection.upsert(
            ids=[_chunk_id(trip_file.id, piece.index) for piece in pieces],
            embeddings=vectors,
            documents=[piece.text for piece in pieces],
            metadatas=[
                {
                    "trip_id": trip_file.trip_id,
                    "file_id": trip_file.id,
                    "chunk_index": piece.index,
                    "source": trip_file.original_filename,
                    "char_start": piece.char_start,
                    "char_end": piece.char_end,
                    "embedding_model": EMBEDDING_MODEL_NAME,
                }
                for piece in pieces
            ],
        )
    except Exception as error:
        # The vectors were embedded successfully but could not be stored. Recorded rather
        # than raised: the caller is a parse request whose text is already good, and the
        # file row now says plainly that it is not searchable.
        message = f"写入向量库失败：{type(error).__name__}: {error}"
        logger.warning("Indexing %s failed at write time: %s", trip_file.original_filename, error)
        return _record_outcome(session, trip_file, "failed", 0, message)
    logger.info("Indexed %s chunks from %s", len(pieces), trip_file.original_filename)
    return _record_outcome(session, trip_file, "indexed", len(pieces), None)


def _record_outcome(
    session: Session, trip_file: TripFile, status: str, chunks: int, reason: str | None
) -> IndexOutcome:
    trip_file.index_status = status
    trip_file.index_error = reason
    trip_file.indexed_chunks = chunks
    trip_file.indexed_at = datetime.utcnow() if status == "indexed" else None
    session.commit()
    return IndexOutcome(status, chunk_count=chunks, reason=reason)


def _chunk_id(file_id: int, index: int) -> str:
    return f"file-{file_id}-chunk-{index}"


def _drop_file(collection: Any, file_id: int) -> int:
    existing = collection.get(where={"file_id": file_id})
    removed = len(existing.get("ids") or [])
    if removed:
        collection.delete(where={"file_id": file_id})
    return removed


def delete_file_chunks(file_id: int) -> int:
    """Drop every chunk belonging to one file. Returns how many were removed.

    Called before the file row is deleted. A failure here is allowed to propagate: the
    alternative is a deleted document whose text is still retrievable, which is worse
    than a delete that reports it could not complete.
    """
    collection = _collection()
    return _drop_file(collection, file_id)


def delete_trip_chunks(trip_id: int) -> int:
    """Drop every chunk belonging to a trip."""
    collection = _collection()
    existing = collection.get(where={"trip_id": trip_id})
    removed = len(existing.get("ids") or [])
    if removed:
        collection.delete(where={"trip_id": trip_id})
    return removed


def search_trip_documents(
    trip_id: int, query: str, limit: int = DEFAULT_RESULT_LIMIT
) -> SearchOutcome:
    """Semantic search inside one trip's documents. Never raises: callers are request paths.

    The ``where`` filter is applied here and nowhere else, so a result set can only ever
    contain chunks belonging to ``trip_id``.
    """
    term = (query or "").strip()
    if not term:
        return SearchOutcome("skipped", reason="查询内容为空。")
    try:
        collection = _collection()
        stored = collection.get(where={"trip_id": trip_id})
    except Exception as error:
        return SearchOutcome("unavailable", reason=f"向量库不可用：{type(error).__name__}: {error}")

    available = len(stored.get("ids") or [])
    if not available:
        return SearchOutcome("empty", reason="这趟旅行还没有已建立索引的资料。")

    try:
        vector = embed_texts([term])[0]
    except EmbeddingError as error:
        return SearchOutcome("failed", reason=str(error))

    result = collection.query(
        query_embeddings=[vector],
        # Asking for more neighbours than the filter can supply makes Chroma log a
        # warning on every call and clamp internally; clamping here keeps logs honest.
        n_results=max(1, min(limit, MAX_RESULT_LIMIT, available)),
        where={"trip_id": trip_id},
        include=["documents", "metadatas", "distances"],
    )
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    ids = (result.get("ids") or [[]])[0]
    if not ids:
        return SearchOutcome("empty", reason="没有找到相关资料片段。")

    chunks = [
        RetrievedChunk(
            chunk_id=chunk_id,
            file_id=int(metadata.get("file_id") or 0),
            source_label=str(metadata.get("source") or "已上传资料"),
            chunk_index=int(metadata.get("chunk_index") or 0),
            text=document,
            # The collection is built with the cosine metric, so the distance is
            # ``1 - cosine similarity`` and the similarity is recovered directly.
            similarity=round(1.0 - float(distance), 4),
        )
        for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances)
    ]
    return SearchOutcome("ok", chunks=chunks)


def document_context(chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    """Render retrieved chunks as prompt material with their source kept visible."""
    return [
        {
            "source": chunk.source_label,
            "section": chunk.chunk_index + 1,
            "relevance": chunk.similarity,
            "excerpt": chunk.text,
        }
        for chunk in chunks
    ]


def knowledge_status(trip_id: int) -> dict[str, Any]:
    """What is searchable for one trip. Used by the status endpoint and the panel."""
    try:
        collection = _collection()
        stored = collection.get(where={"trip_id": trip_id}, include=["metadatas"])
        file_ids = {int(metadata.get("file_id") or 0) for metadata in (stored.get("metadatas") or [])}
        return {
            "available": True,
            "indexed_chunks": len(stored.get("ids") or []),
            "indexed_files": len(file_ids),
            "backend": "chromadb",
            "collection": COLLECTION_NAME,
            "path": str(chroma_path()),
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding_dim": EMBEDDING_DIM,
            "error": None,
        }
    except Exception as error:
        return {
            "available": False,
            "indexed_chunks": 0,
            "indexed_files": 0,
            "backend": "chromadb",
            "collection": COLLECTION_NAME,
            "path": str(chroma_path()),
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding_dim": EMBEDDING_DIM,
            "error": f"{type(error).__name__}: {error}",
        }
