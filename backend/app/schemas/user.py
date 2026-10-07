from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class IdentifierPassword(BaseModel):
    """What both registration and login accept.

    ``identifier`` is deliberately vague: the server decides whether it is a username, an
    email address, or a phone number (see ``auth_service.identifier_kind``). One field
    means the login form does not have to ask the user which kind of account they made.
    """

    identifier: str = Field(min_length=3, max_length=160)
    password: str = Field(min_length=1, max_length=128)


class IdentifierUpdate(BaseModel):
    """Add or replace one of the three identifiers on the current account."""

    email: str | None = Field(default=None, max_length=160)
    phone: str | None = Field(default=None, max_length=24)


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str | None
    email: str | None
    phone: str | None
    display_name: str | None
    created_at: datetime
    last_login_at: datetime | None
    email_verified_at: datetime | None
    phone_verified_at: datetime | None


class AuthResult(BaseModel):
    """The answer to a successful registration or login."""

    access_token: str
    token_type: str = "bearer"
    expires_in_seconds: int
    user: UserRead
    # Trips that predate accounts are handed to the first account that logs in; telling
    # the user how many moved explains why their trip list was not empty on day one.
    claimed_trip_count: int = 0
