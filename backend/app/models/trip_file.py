from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base

if TYPE_CHECKING:
    from app.models.trip import Trip


class TripFile(Base):
    __tablename__ = "trip_files"

    id: Mapped[int] = mapped_column(primary_key=True)
    trip_id: Mapped[int] = mapped_column(ForeignKey("trips.id"), index=True)
    original_filename: Mapped[str] = mapped_column(String(255))
    stored_filename: Mapped[str] = mapped_column(String(255), unique=True)
    content_type: Mapped[str] = mapped_column(String(100))
    file_size: Mapped[int] = mapped_column(Integer)
    file_path: Mapped[str] = mapped_column(String(500))
    parse_status: Mapped[str] = mapped_column(String(20), default="pending", server_default="pending")
    parsed_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    parsed_result: Mapped[dict[str, list[str]] | None] = mapped_column(JSON, nullable=True)
    parse_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    parsed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # Indexing is tracked per file and separately from parsing, because the two fail for
    # different reasons. A scanned page can parse into nothing (skipped), and the
    # embedding provider can be unreachable (failed) while the text is perfectly good.
    # Keeping only a "parsed" flag is what would let a document look usable while being
    # absent from search.
    index_status: Mapped[str] = mapped_column(String(20), default="pending", server_default="pending")
    index_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    indexed_chunks: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    trip: Mapped["Trip"] = relationship(back_populates="files")
