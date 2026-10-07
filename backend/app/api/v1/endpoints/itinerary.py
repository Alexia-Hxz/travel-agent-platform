from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from app.db.session import get_session
from app.schemas.itinerary import (
    AutoScheduleRead,
    ItineraryItemCreate,
    ItineraryItemRead,
    ProposedItineraryItem,
    SkippedPlace,
)
from app.schemas.place import PlaceCreate, PlaceRead, PlaceUpdate
from app.services.auto_schedule import propose_schedule
from app.services.itinerary_service import (
    clear_places,
    create_itinerary_item,
    create_place,
    delete_itinerary_item,
    delete_place,
    get_itinerary_item,
    get_place,
    list_itinerary_items,
    list_places,
    update_place,
)
from app.services.trip_service import get_trip


router = APIRouter(prefix="/trips/{trip_id}", tags=["places", "itinerary"])
SessionDependency = Annotated[Session, Depends(get_session)]


def require_trip(session: Session, trip_id: int):
    trip = get_trip(session, trip_id)
    if trip is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="旅行项目不存在")
    return trip


@router.get("/places", response_model=list[PlaceRead])
def list_places_endpoint(trip_id: int, session: SessionDependency) -> list[PlaceRead]:
    require_trip(session, trip_id)
    return list_places(session, trip_id)


@router.post("/places", response_model=PlaceRead, status_code=status.HTTP_201_CREATED)
def create_place_endpoint(trip_id: int, payload: PlaceCreate, session: SessionDependency) -> PlaceRead:
    require_trip(session, trip_id)
    return create_place(session, trip_id, payload)


@router.patch("/places/{place_id}", response_model=PlaceRead)
def update_place_endpoint(
    trip_id: int, place_id: int, payload: PlaceUpdate, session: SessionDependency
) -> PlaceRead:
    """Correct a saved place's price, count or details without re-adding it."""
    require_trip(session, trip_id)
    place = get_place(session, trip_id, place_id)
    if place is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="收藏地点不存在")
    return update_place(session, place, payload)


@router.delete("/places", status_code=status.HTTP_200_OK)
def clear_places_endpoint(trip_id: int, session: SessionDependency) -> dict:
    """Remove every saved place at once, so planning can start over from a clean slate.

    Declared before ``/places/{place_id}`` for readability only: the paths cannot be
    confused, since this one has nothing where the path parameter would be.
    """
    require_trip(session, trip_id)
    removed = clear_places(session, trip_id)
    return {"removed": removed}


@router.delete("/places/{place_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_place_endpoint(trip_id: int, place_id: int, session: SessionDependency) -> Response:
    require_trip(session, trip_id)
    place = get_place(session, trip_id, place_id)
    if place is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="收藏地点不存在")
    delete_place(session, place)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/itinerary-items", response_model=list[ItineraryItemRead])
def list_itinerary_items_endpoint(trip_id: int, session: SessionDependency) -> list[ItineraryItemRead]:
    require_trip(session, trip_id)
    return list_itinerary_items(session, trip_id)


@router.post("/itinerary-items", response_model=ItineraryItemRead, status_code=status.HTTP_201_CREATED)
def create_itinerary_item_endpoint(
    trip_id: int, payload: ItineraryItemCreate, session: SessionDependency
) -> ItineraryItemRead:
    trip = require_trip(session, trip_id)
    item = create_itinerary_item(session, trip, payload)
    if item is None:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="行程日期或地点无效")
    return item


@router.post("/itinerary-items/auto-schedule", response_model=AutoScheduleRead)
def auto_schedule_endpoint(trip_id: int, session: SessionDependency) -> AutoScheduleRead:
    """Propose a day and time for every saved place that is not scheduled yet.

    Read-only: the caller applies the proposal through the normal itinerary-item
    endpoint, so there is a single validated write path.
    """
    trip = require_trip(session, trip_id)
    places = list_places(session, trip_id)
    items = list_itinerary_items(session, trip_id)
    proposal = propose_schedule(
        [
            {
                "id": place.id,
                "name": place.name,
                "type": place.type,
                "opening_hours": place.opening_hours,
                # The day the recommendation panel suggested, so ticking an item inside
                # a day group keeps it on that day.
                "preferred_date": place.preferred_date,
            }
            for place in places
        ],
        [
            {
                "place_id": item.place_id,
                "date": item.date,
                "start_time": item.start_time,
                "end_time": item.end_time,
            }
            for item in items
        ],
        trip.start_date.isoformat(),
        trip.end_date.isoformat(),
    )

    proposed = [
        ProposedItineraryItem(
            place_id=item.place_id,
            place_name=item.place_name,
            date=item.date,
            start_time=item.start_time,
            end_time=item.end_time,
            note=item.note,
        )
        for item in proposal.proposed
    ]
    if proposed:
        message = f"已为 {len(proposed)} 个收藏地点找到合适的时段。"
    elif proposal.already_scheduled:
        message = "收藏地点都已经排进行程了。"
    else:
        message = "这些收藏地点暂时找不到合适的时段，请手动安排。"
    if proposal.skipped:
        message += f" 有 {len(proposal.skipped)} 个地点放不下。"

    return AutoScheduleRead(
        trip_id=trip_id,
        proposed=proposed,
        skipped=[SkippedPlace(**entry) for entry in proposal.skipped],
        already_scheduled=proposal.already_scheduled,
        message=message,
    )


@router.delete("/itinerary-items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_itinerary_item_endpoint(trip_id: int, item_id: int, session: SessionDependency) -> Response:
    require_trip(session, trip_id)
    item = get_itinerary_item(session, trip_id, item_id)
    if item is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="行程项不存在")
    delete_itinerary_item(session, item)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
