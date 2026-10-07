from fastapi import HTTPException, status
from pathlib import Path
from sqlalchemy.orm import Session

from app.models.trip import Trip
from app.models.trip_file import TripFile
from app.multimodal.schemas import VisionConfirmation, VisionConfirmationResult, VisionPreview
from app.schemas.itinerary import ItineraryItemCreate
from app.schemas.place import PlaceCreate
from app.services import vision_service
from app.services.file_service import resolve_file_path
from app.services.itinerary_service import create_itinerary_item, create_place


def create_preview(trip: Trip, trip_file: TripFile) -> VisionPreview:
    try:
        content_type = trip_file.content_type
        if content_type == "application/octet-stream":
            content_type = {
                ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp",
            }.get(Path(trip_file.file_path).suffix.lower(), content_type)
        analysis = vision_service.analyze_file(resolve_file_path(trip_file), content_type)
        preview = VisionPreview(**analysis.__dict__)
        if preview.itinerary and not trip.start_date <= preview.itinerary.date <= trip.end_date:
            return preview.model_copy(
                update={
                    "itinerary": None,
                    "summary": f"{preview.summary} 订单日期不在当前旅行日期内，未生成每日行程建议。",
                }
            )
        return preview
    except vision_service.VisionConfigurationError as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Multimodal analysis is not configured.") from error
    except vision_service.VisionProviderError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)) from error


def confirm_suggestions(
    session: Session, trip: Trip, confirmation: VisionConfirmation
) -> VisionConfirmationResult:
    if not confirmation.save_place and not confirmation.save_itinerary:
        return VisionConfirmationResult()
    if confirmation.save_place and confirmation.place is None:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="A place suggestion is required.")
    if confirmation.save_itinerary and confirmation.itinerary is None:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="An itinerary suggestion is required.")
    if confirmation.save_itinerary and not trip.start_date <= confirmation.itinerary.date <= trip.end_date:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="The itinerary date is outside this trip.",
        )

    place = None
    itinerary = None
    try:
        if confirmation.save_place:
            place = create_place(session, trip.id, PlaceCreate(**confirmation.place.model_dump()))
        if confirmation.save_itinerary:
            suggestion = confirmation.itinerary
            item = create_itinerary_item(
                session,
                trip,
                ItineraryItemCreate(
                    date=suggestion.date,
                    start_time=suggestion.start_time,
                    end_time=suggestion.end_time,
                    place_id=place.id,
                    note=suggestion.note,
                    sort_order=0,
                ),
            )
            if item is None:
                raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="The itinerary date is outside this trip.")
            itinerary = item
    except HTTPException:
        session.rollback()
        raise
    return VisionConfirmationResult(place=place, itinerary=itinerary)
