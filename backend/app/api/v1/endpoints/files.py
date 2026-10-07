from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.db.session import get_session
from app.schemas.trip_file import TripFileRead
from app.multimodal.schemas import VisionConfirmation, VisionConfirmationResult, VisionPreview
from app.services.multimodal_service import confirm_suggestions, create_preview
from app.services.file_service import (
    delete_trip_file,
    get_trip_file,
    list_trip_files,
    resolve_file_path,
    save_trip_file,
)
from app.services.trip_service import get_trip
from app.services.file_parser import parse_trip_file


router = APIRouter(prefix="/trips/{trip_id}/files", tags=["files"])
SessionDependency = Annotated[Session, Depends(get_session)]


def require_trip(session: Session, trip_id: int) -> None:
    if get_trip(session, trip_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Trip does not exist.")


def require_trip_file(session: Session, trip_id: int, file_id: int):
    trip_file = get_trip_file(session, trip_id, file_id)
    if trip_file is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File does not exist.")
    return trip_file


@router.post("", response_model=TripFileRead, status_code=status.HTTP_201_CREATED)
async def upload_trip_file_endpoint(
    trip_id: int, session: SessionDependency, file: UploadFile = File(...)
) -> TripFileRead:
    require_trip(session, trip_id)
    return await save_trip_file(session, trip_id, file)


@router.get("", response_model=list[TripFileRead])
def list_trip_files_endpoint(trip_id: int, session: SessionDependency) -> list[TripFileRead]:
    require_trip(session, trip_id)
    return list_trip_files(session, trip_id)


@router.get("/{file_id}", response_model=TripFileRead)
def get_trip_file_endpoint(trip_id: int, file_id: int, session: SessionDependency) -> TripFileRead:
    require_trip(session, trip_id)
    return require_trip_file(session, trip_id, file_id)


@router.get("/{file_id}/download")
def download_trip_file_endpoint(trip_id: int, file_id: int, session: SessionDependency) -> FileResponse:
    require_trip(session, trip_id)
    trip_file = require_trip_file(session, trip_id, file_id)
    file_path = resolve_file_path(trip_file)
    if not file_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Stored file was not found.")
    return FileResponse(
        path=file_path,
        media_type=trip_file.content_type,
        filename=trip_file.original_filename,
    )


@router.delete("/{file_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_trip_file_endpoint(trip_id: int, file_id: int, session: SessionDependency) -> Response:
    require_trip(session, trip_id)
    delete_trip_file(session, require_trip_file(session, trip_id, file_id))
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{file_id}/parse", response_model=TripFileRead)
def parse_trip_file_endpoint(trip_id: int, file_id: int, session: SessionDependency) -> TripFileRead:
    require_trip(session, trip_id)
    return parse_trip_file(session, require_trip_file(session, trip_id, file_id))


@router.get("/{file_id}/parse", response_model=TripFileRead)
def get_parse_status_endpoint(trip_id: int, file_id: int, session: SessionDependency) -> TripFileRead:
    require_trip(session, trip_id)
    return require_trip_file(session, trip_id, file_id)


@router.get("/{file_id}/parse/result", response_model=TripFileRead)
def get_parse_result_endpoint(trip_id: int, file_id: int, session: SessionDependency) -> TripFileRead:
    require_trip(session, trip_id)
    trip_file = require_trip_file(session, trip_id, file_id)
    if trip_file.parse_status != "completed":
        raise HTTPException(status_code=409, detail="File parsing has not completed.")
    return trip_file


@router.post("/{file_id}/vision/preview", response_model=VisionPreview)
def preview_multimodal_analysis(
    trip_id: int, file_id: int, session: SessionDependency
) -> VisionPreview:
    trip = get_trip(session, trip_id)
    if trip is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Trip does not exist.")
    return create_preview(trip, require_trip_file(session, trip_id, file_id))


@router.post("/{file_id}/vision/confirm", response_model=VisionConfirmationResult, status_code=status.HTTP_201_CREATED)
def confirm_multimodal_suggestions(
    trip_id: int, file_id: int, payload: VisionConfirmation, session: SessionDependency
) -> VisionConfirmationResult:
    trip = get_trip(session, trip_id)
    if trip is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Trip does not exist.")
    require_trip_file(session, trip_id, file_id)
    return confirm_suggestions(session, trip, payload)
