from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from app.api.deps import CurrentUser
from app.db.session import get_session
from app.schemas.trip import TripCreate, TripRead
from app.services.trip_service import create_trip, delete_trip, get_owned_trip, list_trips


router = APIRouter(prefix="/trips", tags=["trips"])
SessionDependency = Annotated[Session, Depends(get_session)]


@router.post("", response_model=TripRead, status_code=status.HTTP_201_CREATED)
def create_trip_endpoint(
    payload: TripCreate, session: SessionDependency, user: CurrentUser
) -> TripRead:
    return create_trip(session, payload, user.id)


@router.get("", response_model=list[TripRead])
def list_trips_endpoint(session: SessionDependency, user: CurrentUser) -> list[TripRead]:
    return list_trips(session, user.id)


@router.get("/{trip_id}", response_model=TripRead)
def get_trip_endpoint(trip_id: int, session: SessionDependency, user: CurrentUser) -> TripRead:
    trip = get_owned_trip(session, trip_id, user.id)
    if trip is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="旅行项目不存在")
    return trip


@router.delete("/{trip_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_trip_endpoint(
    trip_id: int, session: SessionDependency, user: CurrentUser
) -> Response:
    trip = get_owned_trip(session, trip_id, user.id)
    if trip is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="旅行项目不存在")
    delete_trip(session, trip)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
