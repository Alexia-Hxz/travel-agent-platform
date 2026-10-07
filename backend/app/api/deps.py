"""Shared dependencies: the database session, and the signed-in user.

``get_current_user`` is the single place that decides whether a request is allowed in.
Every other endpoint depends on it, so there is exactly one answer to "who is calling?"
and exactly one place to look when a request comes back 401.
"""

from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.db.session import get_session
from app.models.trip import Trip
from app.models.user import User
from app.services.auth_service import read_access_token
from app.services.trip_service import get_owned_trip

SessionDependency = Annotated[Session, Depends(get_session)]

# auto_error=False so a missing header reaches our own code: the answer has to be a
# 401 with a message the frontend can act on, not FastAPI's bare 403.
bearer_scheme = HTTPBearer(auto_error=False)

_UNAUTHORISED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="请先登录后再操作。",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    session: SessionDependency,
) -> User:
    """Resolve the bearer token to an account, or answer 401.

    An expired token, a token signed with an old secret, and a token for a deleted
    account are all the same 401: the client's only sensible reaction is to log in again.
    """
    if credentials is None or not credentials.credentials:
        raise _UNAUTHORISED
    user_id = read_access_token(credentials.credentials)
    if user_id is None:
        raise _UNAUTHORISED
    user = session.get(User, user_id)
    if user is None:
        raise _UNAUTHORISED
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def require_owned_trip(
    trip_id: int, session: SessionDependency, user: CurrentUser
) -> "Trip":
    """Refuse the request unless ``trip_id`` belongs to the signed-in account.

    This is attached to whole routers (see ``main.py``) rather than to individual
    endpoints. Every route under ``/trips/{trip_id}...`` therefore carries the check by
    construction: a new endpoint added to one of those routers is protected the moment it
    exists, and there is no per-endpoint line anyone can forget to write.

    It answers 404 -- not 403 -- on purpose. A 403 would confirm that the id exists under
    somebody else's account, which is information the caller has no business having.
    """
    trip = get_owned_trip(session, trip_id, user.id)
    if trip is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="旅行项目不存在")
    return trip
