from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.models.itinerary_item import ItineraryItem
from app.models.place import Place
from app.models.trip import Trip
from app.schemas.itinerary import ItineraryItemCreate
from app.schemas.place import PlaceCreate, PlaceUpdate


def list_places(session: Session, trip_id: int) -> list[Place]:
    return list(session.scalars(select(Place).where(Place.trip_id == trip_id).order_by(Place.id.asc())))


def create_place(session: Session, trip_id: int, payload: PlaceCreate) -> Place:
    # mode="json" is what turns the HttpUrl fields into storable strings, but it also
    # turns preferred_date into an ISO string, and the Date column needs a real date.
    data = payload.model_dump(mode="json")
    if data.get("preferred_date"):
        data["preferred_date"] = date.fromisoformat(data["preferred_date"])
    place = Place(trip_id=trip_id, **data)
    session.add(place)
    session.commit()
    session.refresh(place)
    return place


def get_place(session: Session, trip_id: int, place_id: int) -> Place | None:
    return session.scalar(select(Place).where(Place.id == place_id, Place.trip_id == trip_id))


def update_place(session: Session, place: Place, payload: PlaceUpdate) -> Place:
    # exclude_unset keeps "not sent" different from "explicitly cleared to null", and
    # mode="json" is what makes an HttpUrl storable while still needing the date fixed
    # back up (same reason as create_place).
    data = payload.model_dump(mode="json", exclude_unset=True)
    if "preferred_date" in data and data["preferred_date"]:
        data["preferred_date"] = date.fromisoformat(data["preferred_date"])
    for key, value in data.items():
        setattr(place, key, value)
    session.add(place)
    session.commit()
    session.refresh(place)
    return place


def delete_place(session: Session, place: Place) -> None:
    session.delete(place)
    session.commit()


def clear_places(session: Session, trip_id: int) -> int:
    """Delete every saved place of a trip in one go and report how many went.

    Deleting them one by one from the client would leave a half-cleared list behind
    whenever one request failed, so the whole clear is one transaction here. Itinerary
    items booked for those places go with them (``cascade="all, delete-orphan"``), which
    is why the caller reloads the calendar afterwards.
    """
    places = list_places(session, trip_id)
    for place in places:
        session.delete(place)
    session.commit()
    return len(places)


def list_itinerary_items(session: Session, trip_id: int) -> list[ItineraryItem]:
    statement = (
        select(ItineraryItem)
        .options(joinedload(ItineraryItem.place))
        .where(ItineraryItem.trip_id == trip_id)
        .order_by(ItineraryItem.date.asc(), ItineraryItem.sort_order.asc(), ItineraryItem.start_time.asc(), ItineraryItem.id.asc())
    )
    return list(session.scalars(statement))


def create_itinerary_item(
    session: Session, trip: Trip, payload: ItineraryItemCreate
) -> ItineraryItem | None:
    if not trip.start_date <= payload.date <= trip.end_date:
        return None

    place = get_place(session, trip.id, payload.place_id)
    if place is None:
        return None

    item = ItineraryItem(trip_id=trip.id, **payload.model_dump())
    session.add(item)
    session.commit()
    statement = select(ItineraryItem).options(joinedload(ItineraryItem.place)).where(ItineraryItem.id == item.id)
    return session.scalar(statement)


def get_itinerary_item(session: Session, trip_id: int, item_id: int) -> ItineraryItem | None:
    statement = (
        select(ItineraryItem)
        .options(joinedload(ItineraryItem.place))
        .where(ItineraryItem.id == item_id, ItineraryItem.trip_id == trip_id)
    )
    return session.scalar(statement)


def delete_itinerary_item(session: Session, item: ItineraryItem) -> None:
    session.delete(item)
    session.commit()
