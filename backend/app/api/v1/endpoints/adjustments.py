from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_session
from app.schemas.adjustment import AdjustmentApplyRead, AdjustmentApplyRequest, AdjustmentDraftRequest, AdjustmentRead
from app.services.adjustment_agent_service import (
    DayPlanDateError,
    adjustment_to_read,
    create_adjustment_draft,
)
from app.services.adjustment_service import (
    AdjustmentStateError,
    AdjustmentValidationError,
    apply_adjustment,
    get_adjustment,
    list_adjustments,
    reject_adjustment,
)
from app.services.trip_service import get_trip


router = APIRouter(prefix="/trips", tags=["itinerary-adjustments"])
SessionDependency = Annotated[Session, Depends(get_session)]


def _require_trip(session: Session, trip_id: int):
    trip = get_trip(session, trip_id)
    if trip is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="旅行项目不存在。")
    return trip


def _require_adjustment(session: Session, trip_id: int, adjustment_id: int):
    adjustment = get_adjustment(session, trip_id, adjustment_id)
    if adjustment is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="行程调整草案不存在。")
    return adjustment


@router.post("/{trip_id}/adjustments", response_model=AdjustmentRead, status_code=status.HTTP_201_CREATED)
def create_adjustment(
    trip_id: int, payload: AdjustmentDraftRequest, session: SessionDependency
) -> dict:
    trip = _require_trip(session, trip_id)
    try:
        return create_adjustment_draft(
            session,
            trip,
            payload.instruction,
            payload.include_web,
            operations=payload.operations,
            candidates=[candidate.model_dump(mode="json") for candidate in (payload.candidates or [])],
            structured_only=payload.structured_only,
            mode=payload.mode,
            target_date=payload.date,
        )
    except DayPlanDateError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)) from error
    except AdjustmentValidationError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)) from error


@router.get("/{trip_id}/adjustments", response_model=list[AdjustmentRead])
def list_adjustment_drafts(trip_id: int, session: SessionDependency) -> list[dict]:
    _require_trip(session, trip_id)
    return [adjustment_to_read(adjustment) for adjustment in list_adjustments(session, trip_id)]


@router.post("/{trip_id}/adjustments/{adjustment_id}/apply", response_model=AdjustmentApplyRead)
def apply_adjustment_draft(
    trip_id: int,
    adjustment_id: int,
    session: SessionDependency,
    payload: AdjustmentApplyRequest | None = None,
) -> dict:
    trip = _require_trip(session, trip_id)
    adjustment = _require_adjustment(session, trip_id, adjustment_id)
    try:
        items = apply_adjustment(
            session,
            trip,
            adjustment,
            payload.operation_indexes if payload else None,
            edited_operations=payload.operations if payload else None,
        )
    except AdjustmentStateError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    except AdjustmentValidationError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)) from error
    return {"id": adjustment.id, "status": "applied", "itinerary_item_ids": [item.id for item in items]}


@router.post("/{trip_id}/adjustments/{adjustment_id}/reject", response_model=AdjustmentRead)
def reject_adjustment_draft(trip_id: int, adjustment_id: int, session: SessionDependency) -> dict:
    _require_trip(session, trip_id)
    adjustment = _require_adjustment(session, trip_id, adjustment_id)
    try:
        return adjustment_to_read(reject_adjustment(session, adjustment))
    except AdjustmentStateError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
