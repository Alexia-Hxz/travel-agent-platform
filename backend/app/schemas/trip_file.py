from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class TripFileRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    trip_id: int
    original_filename: str
    stored_filename: str
    content_type: str
    file_size: int
    file_path: str
    parse_status: Literal["pending", "processing", "completed", "failed"]
    parsed_text: str | None
    parsed_result: dict[str, list[str]] | None
    parse_error: str | None
    parsed_at: datetime | None
    # Whether this document can actually be found by search, and why not when it cannot.
    index_status: Literal["pending", "indexed", "skipped", "failed"]
    index_error: str | None
    indexed_chunks: int
    indexed_at: datetime | None
    created_at: datetime
