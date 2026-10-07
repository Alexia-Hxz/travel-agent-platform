from pathlib import Path
import os
from uuid import uuid4

from fastapi import HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.trip_file import TripFile


PROJECT_ROOT = Path(os.getenv("TRAVEL_AGENT_PROJECT_ROOT", Path(__file__).resolve().parents[3])).resolve()
UPLOADS_DIRECTORY = PROJECT_ROOT / "data" / "uploads"
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".pdf"}
CHUNK_SIZE = 1024 * 1024
CONTENT_TYPE_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "application/pdf": ".pdf",
}


async def _resolve_upload_extension(upload: UploadFile, original_filename: str) -> str:
    extension = Path(original_filename).suffix.lower()
    if extension in ALLOWED_EXTENSIONS:
        return extension

    content_type = (upload.content_type or "").split(";", 1)[0].strip().lower()
    if content_type in CONTENT_TYPE_EXTENSIONS:
        return CONTENT_TYPE_EXTENSIONS[content_type]

    # A few clients send application/octet-stream for non-ASCII filenames. Inspect
    # only the signature bytes, then rewind so the normal streaming loop is unchanged.
    header = await upload.read(16)
    await upload.seek(0)
    if header.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return ".webp"
    if header.startswith(b"%PDF"):
        return ".pdf"
    return extension


def list_trip_files(session: Session, trip_id: int) -> list[TripFile]:
    statement = select(TripFile).where(TripFile.trip_id == trip_id).order_by(TripFile.created_at.asc(), TripFile.id.asc())
    return list(session.scalars(statement))


def get_trip_file(session: Session, trip_id: int, file_id: int) -> TripFile | None:
    return session.scalar(select(TripFile).where(TripFile.id == file_id, TripFile.trip_id == trip_id))


async def save_trip_file(session: Session, trip_id: int, upload: UploadFile) -> TripFile:
    original_filename = Path(upload.filename or "").name
    extension = await _resolve_upload_extension(upload, original_filename)
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only JPG, JPEG, PNG, WEBP, and PDF files are supported.",
        )

    trip_directory = UPLOADS_DIRECTORY / str(trip_id)
    trip_directory.mkdir(parents=True, exist_ok=True)
    stored_filename = f"{uuid4().hex}{extension}"
    stored_path = trip_directory / stored_filename
    file_size = 0

    try:
        with stored_path.open("wb") as destination:
            while chunk := await upload.read(CHUNK_SIZE):
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="File size must not exceed 10 MiB.",
                    )
                destination.write(chunk)
    except Exception:
        stored_path.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()

    trip_file = TripFile(
        trip_id=trip_id,
        original_filename=original_filename,
        stored_filename=stored_filename,
        content_type=upload.content_type or "application/octet-stream",
        file_size=file_size,
        file_path=stored_path.relative_to(PROJECT_ROOT).as_posix(),
    )
    try:
        session.add(trip_file)
        session.commit()
        session.refresh(trip_file)
    except Exception:
        session.rollback()
        stored_path.unlink(missing_ok=True)
        raise
    return trip_file


def resolve_file_path(trip_file: TripFile) -> Path:
    resolved_path = (PROJECT_ROOT / trip_file.file_path).resolve()
    uploads_root = UPLOADS_DIRECTORY.resolve()
    if uploads_root not in resolved_path.parents:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Stored file was not found.")
    return resolved_path


def delete_trip_file(session: Session, trip_file: TripFile) -> None:
    resolve_file_path(trip_file).unlink(missing_ok=True)
    session.delete(trip_file)
    session.commit()


def delete_files_for_trip(session: Session, trip_id: int) -> None:
    for trip_file in list_trip_files(session, trip_id):
        resolve_file_path(trip_file).unlink(missing_ok=True)
        session.delete(trip_file)
    trip_directory = UPLOADS_DIRECTORY / str(trip_id)
    if trip_directory.is_dir():
        trip_directory.rmdir()
