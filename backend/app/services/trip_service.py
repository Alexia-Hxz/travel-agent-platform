from sqlalchemy import select
from sqlalchemy.orm import Session

from app.services.file_service import delete_files_for_trip
from app.models.trip import Trip
from app.schemas.trip import TripCreate


def create_trip(session: Session, payload: TripCreate, user_id: int | None = None) -> Trip:
    trip = Trip(**payload.model_dump(), user_id=user_id)
    session.add(trip)
    session.commit()
    session.refresh(trip)
    return trip


def list_trips(session: Session, user_id: int | None = None) -> list[Trip]:
    """The trips this account owns, earliest start date first.

    ``user_id=None`` means "no account is in play" and is what direct database scripts
    use. Every HTTP request passes a real id, so an endpoint can never list everyone's
    trips by accident.
    """
    statement = select(Trip).order_by(Trip.start_date.asc(), Trip.id.asc())
    if user_id is not None:
        statement = statement.where(Trip.user_id == user_id)
    return list(session.scalars(statement))


def get_trip(session: Session, trip_id: int) -> Trip | None:
    return session.get(Trip, trip_id)


def get_owned_trip(session: Session, trip_id: int, user_id: int) -> Trip | None:
    """Fetch a trip only when this account owns it.

    Every trip-scoped endpoint goes through here. "Someone else's trip" and "no such trip"
    both come back as ``None``, and the endpoints turn that into the same 404, so the API
    never confirms that an id exists under another account.
    """
    trip = session.get(Trip, trip_id)
    if trip is None or trip.user_id != user_id:
        return None
    return trip


def delete_trip(session: Session, trip: Trip) -> None:
    delete_files_for_trip(session, trip.id)
    session.delete(trip)
    session.commit()
