from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.itinerary_adjustment import ItineraryAdjustment
from app.models.itinerary_item import ItineraryItem
from app.models.place import Place
from app.models.trip import Trip
from app.schemas.adjustment import AdjustmentOperation
from app.services.itinerary_service import get_itinerary_item, get_place, list_itinerary_items


class AdjustmentStateError(Exception):
    pass


class AdjustmentValidationError(Exception):
    pass


def create_pending_adjustment(
    session: Session,
    trip: Trip,
    instruction: str,
    operations: list[AdjustmentOperation],
    *,
    summary: str = "已生成行程调整草案，确认后才会写入行程。",
    sources: list[dict] | None = None,
    events: list[dict] | None = None,
    budget: dict | None = None,
    conflicts: list[dict] | None = None,
    daily_plan: list[dict] | None = None,
    final_summary: str = "",
    traveler_profile: dict | None = None,
    weather: dict | None = None,
) -> ItineraryAdjustment:
    adjustment = ItineraryAdjustment(
        trip_id=trip.id,
        instruction=instruction,
        proposal_json={
            "operations": [
                {
                    key: value
                    for key, value in operation.model_dump(mode="json").items()
                    if key != "budget_amount" or value is not None
                }
                for operation in operations
            ],
            "summary": summary,
            "sources": sources or [],
            "events": events or [],
            "budget": budget or {},
            "conflicts": conflicts or [],
            "daily_plan": daily_plan or [],
            "final_summary": final_summary,
            "traveler_profile": traveler_profile or {},
            "weather": weather or {},
        },
        status="pending",
    )
    session.add(adjustment)
    session.commit()
    session.refresh(adjustment)
    return adjustment


def list_adjustments(session: Session, trip_id: int) -> list[ItineraryAdjustment]:
    statement = (
        select(ItineraryAdjustment)
        .where(ItineraryAdjustment.trip_id == trip_id)
        .order_by(ItineraryAdjustment.created_at.desc(), ItineraryAdjustment.id.desc())
    )
    return list(session.scalars(statement))


def get_adjustment(session: Session, trip_id: int, adjustment_id: int) -> ItineraryAdjustment | None:
    return session.scalar(
        select(ItineraryAdjustment).where(
            ItineraryAdjustment.trip_id == trip_id,
            ItineraryAdjustment.id == adjustment_id,
        )
    )


def _operations(adjustment: ItineraryAdjustment, operation_indexes: list[int] | None = None) -> list[AdjustmentOperation]:
    try:
        operations = [AdjustmentOperation.model_validate(item) for item in adjustment.proposal_json["operations"]]
    except (KeyError, TypeError, ValueError) as error:
        raise AdjustmentValidationError("Adjustment operations are invalid.") from error
    if len(operations) > 48:
        raise AdjustmentValidationError("An adjustment can contain at most 48 operations.")
    if operation_indexes is not None:
        if any(index < 0 or index >= len(operations) for index in operation_indexes):
            raise AdjustmentValidationError("Selected operation index is invalid.")
        selected = set(operation_indexes)
        # Replacement proposals are emitted as adjacent remove/add pairs. Keep
        # the pair atomic even when an older client submits only one index.
        for index in range(len(operations) - 1):
            remove_operation = operations[index]
            add_operation = operations[index + 1]
            if (
                remove_operation.kind == "remove_item"
                and add_operation.kind == "add_place_and_item"
                and add_operation.note
                and ("替换" in add_operation.note or "replace" in add_operation.note.casefold())
                and index in selected
            ):
                selected.update({index, index + 1})
            elif (
                remove_operation.kind == "remove_item"
                and add_operation.kind == "add_place_and_item"
                and add_operation.note
                and ("替换" in add_operation.note or "replace" in add_operation.note.casefold())
                and index + 1 in selected
            ):
                selected.update({index, index + 1})
        operations = [operations[index] for index in sorted(selected)]
    return operations


def validate_operations(session: Session, trip: Trip, operations: list[AdjustmentOperation]) -> None:
    """Check that every operation is one this trip can actually carry out.

    Called both when a draft is proposed from client-built operations and again when it is
    applied. It is deliberately *not* called on model-built drafts: one bad suggestion out
    of ten must not throw away the other nine, which the user picks between with the
    checkboxes.
    """
    for operation in operations:
        if operation.kind == "remove_item":
            if operation.item_id is None or get_itinerary_item(session, trip.id, operation.item_id) is None:
                raise AdjustmentValidationError("The itinerary item does not belong to this trip.")
            continue
        if operation.kind == "set_budget":
            if operation.budget_amount is None or operation.budget_amount < 0:
                raise AdjustmentValidationError("The budget amount is invalid.")
            continue
        if operation.date is None or operation.start_time is None or operation.end_time is None:
            raise AdjustmentValidationError("Itinerary dates and times are required.")
        if not trip.start_date <= operation.date <= trip.end_date:
            raise AdjustmentValidationError("The itinerary date is outside the trip range.")
        if operation.end_time < operation.start_time:
            raise AdjustmentValidationError("The itinerary end time is invalid.")
        if operation.kind == "add_item":
            if operation.place_id is None or get_place(session, trip.id, operation.place_id) is None:
                raise AdjustmentValidationError("The saved place does not belong to this trip.")
        elif operation.kind == "add_place_and_item" and operation.place is None:
            raise AdjustmentValidationError("A new place is required.")
        elif operation.kind == "update_item":
            if operation.item_id is None or get_itinerary_item(session, trip.id, operation.item_id) is None:
                raise AdjustmentValidationError("The itinerary item does not belong to this trip.")


def _apply_operation(session: Session, trip: Trip, operation: AdjustmentOperation) -> None:
    if operation.kind == "set_budget":
        if operation.budget_amount is None:
            raise AdjustmentValidationError("The budget amount is required.")
        trip.budget = operation.budget_amount
        return
    if operation.kind == "remove_item":
        item = get_itinerary_item(session, trip.id, operation.item_id or 0)
        if item is None:
            raise AdjustmentValidationError("The itinerary item is no longer available.")
        session.delete(item)
        return
    if operation.kind == "update_item":
        # Retiming or moving a scheduled item must not create a second copy of its
        # place, which is why this is its own kind instead of remove + add.
        item = get_itinerary_item(session, trip.id, operation.item_id or 0)
        if item is None:
            raise AdjustmentValidationError("The itinerary item is no longer available.")
        if operation.date is not None:
            item.date = operation.date
        if operation.start_time is not None:
            item.start_time = operation.start_time
        if operation.end_time is not None:
            item.end_time = operation.end_time
        if operation.note is not None:
            item.note = operation.note
        if operation.sort_order:
            item.sort_order = operation.sort_order
        return

    place_id = operation.place_id
    if operation.kind == "add_place_and_item":
        if operation.place is None:
            raise AdjustmentValidationError("A new place is required.")
        place = Place(trip_id=trip.id, **operation.place.model_dump(mode="json"))
        session.add(place)
        session.flush()
        place_id = place.id

    if place_id is None or operation.date is None or operation.start_time is None or operation.end_time is None:
        raise AdjustmentValidationError("The itinerary operation is incomplete.")
    session.add(
        ItineraryItem(
            trip_id=trip.id,
            place_id=place_id,
            date=operation.date,
            start_time=operation.start_time,
            end_time=operation.end_time,
            note=operation.note,
            sort_order=operation.sort_order,
        )
    )


def apply_adjustment(
    session: Session,
    trip: Trip,
    adjustment: ItineraryAdjustment,
    operation_indexes: list[int] | None = None,
    edited_operations: list[AdjustmentOperation] | None = None,
) -> list[ItineraryItem]:
    if adjustment.status != "pending":
        raise AdjustmentStateError("Only pending adjustments can be applied.")
    # An edited list replaces the stored proposal entirely: it came from the same
    # preview the user was looking at, with prices or times they changed by hand.
    operations = edited_operations if edited_operations is not None else _operations(adjustment, operation_indexes)
    validate_operations(session, trip, operations)
    try:
        for operation in operations:
            _apply_operation(session, trip, operation)
        adjustment.status = "applied"
        adjustment.applied_at = datetime.utcnow()
        session.commit()
    except Exception:
        session.rollback()
        raise
    return list_itinerary_items(session, trip.id)


def reject_adjustment(session: Session, adjustment: ItineraryAdjustment) -> ItineraryAdjustment:
    if adjustment.status != "pending":
        raise AdjustmentStateError("Only pending adjustments can be rejected.")
    adjustment.status = "rejected"
    session.commit()
    session.refresh(adjustment)
    return adjustment
