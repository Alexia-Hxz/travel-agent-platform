from datetime import datetime

from sqlalchemy import DateTime, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class User(Base):
    """One account, identified by a username, an email address, or a phone number.

    All three columns are nullable and unique, because a person registers with whichever
    one they prefer and can add the others later. The application refuses to create a row
    with none of them (see ``auth_service.create_user``); SQLite cannot express that rule
    as a column constraint, so it lives in one place in code.

    ``*_verified_at`` is deliberately already here: password login never needs it, but
    the moment a verification code is added, the row already has somewhere to record
    "this email/phone really belongs to this person" without a schema change.
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str | None] = mapped_column(String(40), unique=True, index=True, nullable=True)
    email: Mapped[str | None] = mapped_column(String(160), unique=True, index=True, nullable=True)
    phone: Mapped[str | None] = mapped_column(String(24), unique=True, index=True, nullable=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    display_name: Mapped[str | None] = mapped_column(String(60), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    email_verified_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    phone_verified_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # Kept for the (not yet built) verification codes: a wrong code repeated many times
    # must not become a free password guess, so the counter starts at zero now.
    failed_login_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
