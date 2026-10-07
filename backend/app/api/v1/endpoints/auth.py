from fastapi import APIRouter, HTTPException, status

from app.api.deps import CurrentUser, SessionDependency
from app.schemas.user import AuthResult, IdentifierPassword, IdentifierUpdate, UserRead
from app.services import auth_service


router = APIRouter(prefix="/auth", tags=["auth"])


def _auth_result(token: str, user, claimed: int = 0) -> AuthResult:
    return AuthResult(
        access_token=token,
        expires_in_seconds=auth_service.TOKEN_TTL_SECONDS,
        user=UserRead.model_validate(user),
        claimed_trip_count=claimed,
    )


@router.post("/register", response_model=AuthResult, status_code=status.HTTP_201_CREATED)
def register_endpoint(payload: IdentifierPassword, session: SessionDependency) -> AuthResult:
    """Create an account from a username, an email address, or a phone number.

    Registering signs you in immediately: making someone type the same password twice in
    a row is friction with no benefit.
    """
    try:
        user = auth_service.create_user(session, payload.identifier, payload.password)
    except auth_service.IdentifierTaken as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    except (auth_service.InvalidIdentifier, auth_service.InvalidPassword) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=error.message
        ) from error
    claimed = auth_service.claim_unowned_trips(session, user)
    return _auth_result(auth_service.create_access_token(user.id), user, claimed)


@router.post("/login", response_model=AuthResult)
def login_endpoint(payload: IdentifierPassword, session: SessionDependency) -> AuthResult:
    """Sign in with any of the three identifiers plus the password."""
    user = auth_service.authenticate(session, payload.identifier, payload.password)
    if user is None:
        # One message for "no such account" and "wrong password" alike: telling them
        # apart would let anyone probe which phone numbers are registered.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名/邮箱/手机号或密码不正确。",
            headers={"WWW-Authenticate": "Bearer"},
        )
    claimed = auth_service.claim_unowned_trips(session, user)
    return _auth_result(auth_service.create_access_token(user.id), user, claimed)


@router.get("/me", response_model=UserRead)
def me_endpoint(user: CurrentUser) -> UserRead:
    """Who the current token belongs to; this is how the frontend restores a session."""
    return UserRead.model_validate(user)


@router.patch("/me", response_model=UserRead)
def update_me_endpoint(
    payload: IdentifierUpdate, user: CurrentUser, session: SessionDependency
) -> UserRead:
    """Add an email address or phone number to the account that is signed in.

    Someone who registered with a username can attach the other two identifiers later,
    after which they can log in with any of them.
    """
    for field in ("email", "phone"):
        value = getattr(payload, field)
        if value is None:
            continue
        kind, candidate = _validate_for_field(field, value)
        existing = auth_service.find_by_identifier(session, candidate)
        if existing is not None and existing.id != user.id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=auth_service.taken_message(kind),
            )
        setattr(user, field, candidate)
    session.commit()
    session.refresh(user)
    return UserRead.model_validate(user)


def _validate_for_field(field: str, value: str) -> tuple[str, str]:
    """Check that the value really is the kind of identifier the field expects."""
    try:
        kind, candidate = auth_service.validate_identifier(value)
    except auth_service.InvalidIdentifier as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=error.message
        ) from error
    if kind != field:
        # Pasting a phone number into the email box must not silently create a
        # second account under the wrong column.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="邮箱格式不正确。" if field == "email" else "手机号格式不正确。",
        )
    return kind, candidate
