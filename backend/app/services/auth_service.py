"""Accounts: identifiers, password hashing, and the access token.

Everything here is standard library on purpose. The project ships no JWT or password
library (``jwt``, ``jose``, ``passlib``, ``bcrypt`` are all absent), and adding one would
mean a new dependency in every image for what ``hashlib.scrypt`` and ``hmac`` already do.

Two things are worth knowing before changing this file:

* **The password hash is self-describing** (``scrypt$n$r$p$salt$hash``). If the cost
  parameters are ever raised, old rows keep verifying because the row says how it was
  hashed, and ``needs_rehash`` tells a caller it is time to upgrade one.
* **The token is signed, not encrypted.** Its payload (user id + expiry) is readable by
  anyone holding it, which is fine: it carries no personal data, and the signature is what
  stops tampering. Treat the value like a password -- the frontend keeps it in
  ``localStorage``.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import time
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.trip import Trip
from app.models.user import User

# scrypt cost. n=2**14 with r=8 needs about 16 MB per hash, which is a deliberate
# trade: slow enough to make guessing expensive, fast enough that login feels instant.
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 64

# A token that never expires is a token nobody can revoke, and this app has no server-side
# session table, so expiry is the only thing that ends a session. Thirty days suits a
# travel planner that people open every few weekends.
TOKEN_TTL_SECONDS = 30 * 24 * 60 * 60

_USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_.\-]{3,40}$")
_EMAIL_PATTERN = re.compile(r"^[^@\s]{1,64}@[^@\s.]+(\.[^@\s.]+)+$")
_PHONE_PATTERN = re.compile(r"^\+?[0-9]{6,20}$")
_MAINLAND_MOBILE_PATTERN = re.compile(r"^1[3-9][0-9]{9}$")


class AuthError(Exception):
    """Base class for the errors the endpoints turn into HTTP responses."""


class IdentifierTaken(AuthError):
    """That username, email, or phone number already belongs to an account."""


class InvalidIdentifier(AuthError):
    """The identifier is not a usable username, email address, or phone number."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidPassword(AuthError):
    """The password is too short or too long to be worth hashing."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def identifier_kind(value: str) -> str:
    """Say which of the three columns a login identifier belongs to.

    The shape decides, not the caller: ``@`` means email, digits (optionally starting
    with ``+``) mean phone, anything else is a username. That is what lets one login form
    accept all three without asking the user to pick a type first.
    """
    candidate = value.strip()
    if "@" in candidate:
        return "email"
    if _PHONE_PATTERN.match(candidate.replace(" ", "").replace("-", "")):
        return "phone"
    return "username"


def normalise_identifier(value: str) -> str:
    """Trim, and fold case for the two identifiers that are case-insensitive.

    ``Zhang@Example.com`` and ``zhang@example.com`` are the same mailbox, so storing them
    as different rows would let one person register twice and then fail to log in with the
    address they actually typed. Phone numbers lose their spaces and dashes so that
    ``138 0013 8000`` and ``13800138000`` are one account.
    """
    candidate = value.strip()
    kind = identifier_kind(candidate)
    if kind == "email":
        return candidate.lower()
    if kind == "phone":
        return candidate.replace(" ", "").replace("-", "")
    return candidate


def validate_identifier(value: str) -> tuple[str, str]:
    """Return ``(kind, normalised)`` or raise :class:`InvalidIdentifier` with a reason."""
    candidate = normalise_identifier(value)
    kind = identifier_kind(candidate)
    if kind == "email":
        if not _EMAIL_PATTERN.match(candidate):
            raise InvalidIdentifier("邮箱格式不正确，请检查后重试。")
    elif kind == "phone":
        digits = candidate.lstrip("+")
        if not _PHONE_PATTERN.match(candidate):
            raise InvalidIdentifier("手机号格式不正确，请填写 6-20 位数字（可带国家码 +86）。")
        if digits.startswith("1") and len(digits) == 11 and not _MAINLAND_MOBILE_PATTERN.match(digits):
            # 11 digits starting with 1 is a mainland mobile number in the form people
            # here actually type; refusing "12345678901" now avoids a typo becoming an
            # account nobody can log into.
            raise InvalidIdentifier("手机号格式不正确，请填写 11 位中国大陆手机号。")
    elif not _USERNAME_PATTERN.match(candidate):
        raise InvalidIdentifier("用户名只能包含字母、数字、下划线、点和连字符，长度 3-40 位。")
    return kind, candidate


def validate_password(password: str) -> str:
    """Require something long enough to be worth hashing, and short enough to bound cost."""
    if len(password) < 8:
        raise InvalidPassword("密码至少 8 位。")
    if len(password) > 128:
        raise InvalidPassword("密码最长 128 位。")
    return password


def hash_password(password: str) -> str:
    """Hash with scrypt and a fresh random salt, in a self-describing string."""
    salt = os.urandom(16)
    derived = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_SCRYPT_DKLEN,
    )
    return "$".join(
        (
            "scrypt",
            str(_SCRYPT_N),
            str(_SCRYPT_R),
            str(_SCRYPT_P),
            base64.b64encode(salt).decode("ascii"),
            base64.b64encode(derived).decode("ascii"),
        )
    )


def verify_password(password: str, stored: str) -> bool:
    """Check a password against a stored hash, without leaking where it differs."""
    try:
        scheme, n, r, p, salt_b64, hash_b64 = stored.split("$")
        if scheme != "scrypt":
            return False
        derived = hashlib.scrypt(
            password.encode("utf-8"),
            salt=base64.b64decode(salt_b64),
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(base64.b64decode(hash_b64)),
        )
    except (ValueError, TypeError):
        # A row written by an older or hand-edited build must fail closed, not crash.
        return False
    return hmac.compare_digest(derived, base64.b64decode(hash_b64))


def needs_rehash(stored: str) -> bool:
    """True when a stored hash used weaker parameters than the current ones."""
    try:
        scheme, n, r, p, _, _ = stored.split("$")
    except ValueError:
        return True
    return scheme != "scrypt" or (int(n), int(r), int(p)) != (_SCRYPT_N, _SCRYPT_R, _SCRYPT_P)


# A hash of a random value, verified against when the account does not exist, so that a
# missing account and a wrong password take the same time to answer. Without this, the
# response time itself tells an attacker which usernames are registered.
_DUMMY_HASH = hash_password(secrets.token_urlsafe(32))


def _secret_key() -> bytes:
    """The HMAC key for access tokens.

    ``AUTH_SECRET_KEY`` should be set in the environment for any deployment that matters:
    without it every container restart would invalidate nothing (the development fallback
    is stable on purpose), but the fallback is public knowledge, so anyone could mint a
    token for any account. The startup log says so out loud.
    """
    configured = os.getenv("AUTH_SECRET_KEY")
    if configured:
        return configured.encode("utf-8")
    return hashlib.sha256(b"travel-agent-development-secret").digest()


def using_development_secret() -> bool:
    """Whether tokens are signed with the built-in development key."""
    return not os.getenv("AUTH_SECRET_KEY")


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def create_access_token(user_id: int, *, now: int | None = None) -> str:
    """Sign a token that names a user and when it stops being valid."""
    issued_at = int(now if now is not None else time.time())
    payload = {"sub": user_id, "iat": issued_at, "exp": issued_at + TOKEN_TTL_SECONDS}
    body = _b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = hmac.new(_secret_key(), body.encode("ascii"), hashlib.sha256).digest()
    return f"{body}.{_b64encode(signature)}"


def read_access_token(token: str, *, now: int | None = None) -> int | None:
    """Return the user id in a valid token, or ``None`` for anything else.

    Expired, truncated, re-signed with the wrong key, or simply not a token: all of them
    are ``None``. The caller turns that into 401 and must not be told which it was.
    """
    try:
        body, signature = token.split(".")
        expected = hmac.new(_secret_key(), body.encode("ascii"), hashlib.sha256).digest()
        if not hmac.compare_digest(_b64decode(signature), expected):
            return None
        payload = json.loads(_b64decode(body))
        if int(payload["exp"]) <= int(now if now is not None else time.time()):
            return None
        return int(payload["sub"])
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def find_by_identifier(session: Session, value: str) -> User | None:
    """Look an account up by whichever of the three identifiers was typed."""
    kind, candidate = validate_identifier(value)
    column = {"username": User.username, "email": User.email, "phone": User.phone}[kind]
    return session.scalars(select(User).where(column == candidate)).first()


def create_user(session: Session, identifier: str, password: str) -> User:
    """Register an account. Raises :class:`IdentifierTaken` if it already exists."""
    kind, candidate = validate_identifier(identifier)
    validate_password(password)
    if find_by_identifier(session, candidate) is not None:
        raise IdentifierTaken(taken_message(kind))
    user = User(password_hash=hash_password(password), **{kind: candidate})
    session.add(user)
    try:
        session.commit()
    except IntegrityError as error:  # two registrations racing on the same address
        session.rollback()
        raise IdentifierTaken(taken_message(kind)) from error
    session.refresh(user)
    return user


def taken_message(kind: str) -> str:
    return {
        "email": "这个邮箱已经注册过了，直接登录即可。",
        "phone": "这个手机号已经注册过了，直接登录即可。",
        "username": "这个用户名已经被占用了，换一个吧。",
    }[kind]


def authenticate(session: Session, identifier: str, password: str) -> User | None:
    """Return the account when the identifier and password match, else ``None``."""
    try:
        user = find_by_identifier(session, identifier)
    except InvalidIdentifier:
        # A malformed identifier is a failed login, not a 422: the login form must not
        # reveal which shape of input exists in the database.
        user = None
    if user is None:
        verify_password(password, _DUMMY_HASH)
        return None
    if not verify_password(password, user.password_hash):
        user.failed_login_count = (user.failed_login_count or 0) + 1
        session.commit()
        return None
    if needs_rehash(user.password_hash):
        user.password_hash = hash_password(password)
    user.failed_login_count = 0
    user.last_login_at = datetime.now(timezone.utc).replace(tzinfo=None)
    session.commit()
    session.refresh(user)
    return user


def claim_unowned_trips(session: Session, user: User) -> int:
    """Give any trip without an owner to this account, and report how many moved.

    Trips created before accounts existed have ``user_id IS NULL``. Handing them to the
    FIRST account that logs in is what keeps the two real trips in this database
    reachable instead of invisible; every account after that only sees its own. The count
    is returned so the endpoint can tell the user what just happened.
    """
    orphans = list(session.scalars(select(Trip).where(Trip.user_id.is_(None))))
    for trip in orphans:
        trip.user_id = user.id
    if orphans:
        session.commit()
    return len(orphans)
