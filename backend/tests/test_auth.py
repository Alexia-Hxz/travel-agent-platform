"""Accounts: registration, the three ways to log in, tokens, and trip isolation.

The suite signs in as a shared account for free (see ``conftest._authenticated_app``), so
the tests here that care about authentication either send a real token or take the
override away. ``test_a_missing_token_is_refused`` is the one that proves the guard is
really there; the rest of this file would pass even if every token were ignored.
"""

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_current_user
from app.db.session import SessionLocal
from app.main import app
from app.models.user import User
from app.services import auth_service


def _client() -> TestClient:
    return TestClient(app)


def _register(client: TestClient, identifier: str, password: str = "correct-horse-battery"):
    return client.post("/api/v1/auth/register", json={"identifier": identifier, "password": password})


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:8]}"


# --------------------------------------------------------------------------------------
# Registration with each of the three identifiers
# --------------------------------------------------------------------------------------


def test_registering_with_a_username_signs_you_in() -> None:
    with _client() as client:
        response = _register(client, _unique("traveler"))

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["user"]["username"].startswith("traveler-")
    assert body["user"]["email"] is None and body["user"]["phone"] is None
    assert body["token_type"] == "bearer"
    assert body["access_token"]
    assert body["expires_in_seconds"] == auth_service.TOKEN_TTL_SECONDS


def test_registering_with_an_email_stores_the_email_not_a_username() -> None:
    email = f"{uuid4().hex[:8]}@example.com"
    with _client() as client:
        body = _register(client, email).json()

    assert body["user"]["email"] == email
    assert body["user"]["username"] is None


def test_registering_with_a_mainland_mobile_number_works() -> None:
    phone = f"138{uuid4().int % 100000000:08d}"
    with _client() as client:
        body = _register(client, phone).json()

    assert body["user"]["phone"] == phone


def test_a_password_is_never_stored_in_the_clear() -> None:
    identifier = _unique("hashed")
    password = "very-secret-password"
    with _client() as client:
        user_id = _register(client, identifier, password).json()["user"]["id"]

    with SessionLocal() as session:
        stored = session.get(User, user_id).password_hash

    assert password not in stored
    assert stored.startswith("scrypt$")
    assert auth_service.verify_password(password, stored)
    assert not auth_service.verify_password(password + "x", stored)


# --------------------------------------------------------------------------------------
# What registration refuses
# --------------------------------------------------------------------------------------


def test_a_second_registration_of_the_same_username_is_a_conflict() -> None:
    identifier = _unique("taken")
    with _client() as client:
        assert _register(client, identifier).status_code == 201
        again = _register(client, identifier)

    assert again.status_code == 409
    assert "用户名" in again.json()["detail"]


def test_the_same_email_in_a_different_case_is_still_the_same_account() -> None:
    email = f"{uuid4().hex[:8]}@Example.com"
    with _client() as client:
        assert _register(client, email).status_code == 201
        again = _register(client, email.lower())

    assert again.status_code == 409


@pytest.mark.parametrize(
    ("identifier", "expected_fragment"),
    [
        ("a!b", "用户名"),
        ("no spaces allowed", "用户名"),
        ("12345678901", "手机号"),  # eleven digits starting with 1 that are not a mobile number
        ("someone@", "邮箱"),
        ("someone@localhost", "邮箱"),
    ],
)
def test_malformed_identifiers_are_refused_with_a_reason(identifier: str, expected_fragment: str) -> None:
    with _client() as client:
        response = _register(client, identifier)

    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert isinstance(detail, str), f"expected our own message, got {detail}"
    assert expected_fragment in detail


def test_a_short_password_is_refused() -> None:
    with _client() as client:
        response = _register(client, _unique("shortpw"), "1234567")

    assert response.status_code == 422
    assert "8 位" in response.json()["detail"]


# --------------------------------------------------------------------------------------
# Logging in
# --------------------------------------------------------------------------------------


def test_login_works_with_username_email_and_phone_on_one_account() -> None:
    """One account, three identifiers: register with a username, add the others, use all three."""
    identifier = _unique("multi")
    email = f"{uuid4().hex[:8]}@example.com"
    phone = f"139{uuid4().int % 100000000:08d}"

    with _client() as client:
        token = _register(client, identifier).json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        updated = client.patch("/api/v1/auth/me", json={"email": email, "phone": phone}, headers=headers)
        assert updated.status_code == 200, updated.text
        assert updated.json()["email"] == email and updated.json()["phone"] == phone

        for typed in (identifier, email, phone):
            response = client.post(
                "/api/v1/auth/login", json={"identifier": typed, "password": "correct-horse-battery"}
            )
            assert response.status_code == 200, f"{typed}: {response.text}"
            assert response.json()["user"]["id"] == updated.json()["id"]


def test_a_wrong_password_and_an_unknown_account_answer_identically() -> None:
    identifier = _unique("wrongpw")
    with _client() as client:
        _register(client, identifier)
        wrong = client.post("/api/v1/auth/login", json={"identifier": identifier, "password": "nope-nope-nope"})
        unknown = client.post("/api/v1/auth/login", json={"identifier": _unique("ghost"), "password": "nope-nope-nope"})

    assert wrong.status_code == unknown.status_code == 401
    # Same words, so the response cannot be used to find out which accounts exist.
    assert wrong.json()["detail"] == unknown.json()["detail"]


def test_a_malformed_identifier_at_login_is_a_failed_login_not_a_validation_error() -> None:
    with _client() as client:
        response = client.post("/api/v1/auth/login", json={"identifier": "x", "password": "whatever-123"})

    assert response.status_code in {401, 422}
    if response.status_code == 401:
        assert "不正确" in response.json()["detail"]


# --------------------------------------------------------------------------------------
# Tokens
# --------------------------------------------------------------------------------------


def test_me_returns_the_account_the_token_belongs_to() -> None:
    identifier = _unique("me")
    with _client() as client:
        token = _register(client, identifier).json()["access_token"]
        response = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    assert response.json()["username"] == identifier


@pytest.mark.parametrize(
    "header",
    [
        "Bearer not-a-token",
        "Bearer ",
        "Basic abc123",
    ],
)
def test_a_useless_token_is_refused(header: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Signed-in-by-default is a test convenience; a real deployment sees these as 401."""
    monkeypatch.delitem(app.dependency_overrides, get_current_user, raising=False)

    with _client() as client:
        response = client.get("/api/v1/auth/me", headers={"Authorization": header})

    assert response.status_code == 401, f"{header!r} was accepted"


def test_editing_one_character_of_a_token_breaks_it() -> None:
    with _client() as client:
        token = _register(client, _unique("tamper")).json()["access_token"]
        tampered = ("A" if token[0] != "A" else "B") + token[1:]
        response = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {tampered}"})

    assert response.status_code == 401


def test_an_expired_token_is_refused() -> None:
    """A token minted thirty-one days ago must not work today."""
    with SessionLocal() as session:
        user = auth_service.create_user(session, _unique("expired"), "correct-horse-battery")
        stale = auth_service.create_access_token(user.id, now=0)
        user_id = user.id

    assert auth_service.read_access_token(stale) is None
    fresh = auth_service.create_access_token(user_id)
    assert auth_service.read_access_token(fresh) == user_id


def test_a_missing_token_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """The one test that removes the suite's free sign-in and checks the guard itself."""
    monkeypatch.delitem(app.dependency_overrides, get_current_user, raising=False)

    with _client() as client:
        assert client.get("/api/v1/trips").status_code == 401
        assert client.post("/api/v1/trips", json=_trip_payload()).status_code == 401
        # The health check must keep working without a token: it is how the frontend and
        # docker decide the service is up at all.
        assert client.get("/api/v1/health").status_code == 200


def test_the_openapi_schema_documents_the_new_endpoints() -> None:
    with _client() as client:
        paths = client.get("/openapi.json").json()["paths"]

    assert set(paths["/api/v1/auth/register"]) == {"post"}
    assert set(paths["/api/v1/auth/login"]) == {"post"}
    assert set(paths["/api/v1/auth/me"]) == {"get", "patch"}
    assert "security" in paths["/api/v1/trips"]["get"]


# --------------------------------------------------------------------------------------
# Two accounts must not see each other
# --------------------------------------------------------------------------------------


def _trip_payload(title: str = "隔离测试") -> dict[str, object]:
    return {
        "title": title,
        "destination": "南京",
        "start_date": "2026-12-01",
        "end_date": "2026-12-03",
        "budget": 5000,
    }


def test_a_second_account_cannot_see_this_accounts_trips(register_account) -> None:
    _, other_headers = register_account(_unique("intruder"))

    with _client() as client:
        mine = client.post("/api/v1/trips", json=_trip_payload()).json()

        # Not in the other account's list, not fetchable, not deletable.
        listed = client.get("/api/v1/trips", headers=other_headers).json()
        assert all(trip["id"] != mine["id"] for trip in listed)
        assert client.get(f"/api/v1/trips/{mine['id']}", headers=other_headers).status_code == 404
        assert client.delete(f"/api/v1/trips/{mine['id']}", headers=other_headers).status_code == 404

        # And the trip is still there for its owner.
        assert client.get(f"/api/v1/trips/{mine['id']}").status_code == 200


def test_another_account_cannot_touch_places_itinerary_or_files(register_account) -> None:
    """Every router under /trips/{trip_id} is covered by one guard in main.py."""
    _, other_headers = register_account(_unique("nosy"))

    with _client() as client:
        trip_id = client.post("/api/v1/trips", json=_trip_payload()).json()["id"]
        # `address` is required by the places API, which is why it is spelled out here.
        created = client.post(
            f"/api/v1/trips/{trip_id}/places",
            json={"name": "夫子庙", "type": "attraction", "address": "南京市秦淮区"},
        )
        assert created.status_code == 201, created.text

        for path in (
            f"/api/v1/trips/{trip_id}/places",
            f"/api/v1/trips/{trip_id}/itinerary-items",
            f"/api/v1/trips/{trip_id}/files",
            f"/api/v1/trips/{trip_id}/adjustments",
        ):
            response = client.get(path, headers=other_headers)
            assert response.status_code == 404, f"{path} leaked to another account"

        sneak = client.post(
            f"/api/v1/trips/{trip_id}/places",
            json={"name": "偷加的景点", "type": "attraction", "address": "南京市"},
            headers=other_headers,
        )
        assert sneak.status_code == 404
        # The owner's place list is unchanged.
        assert [place["name"] for place in client.get(f"/api/v1/trips/{trip_id}/places").json()] == ["夫子庙"]


def test_a_trip_created_before_accounts_existed_is_adopted_by_the_first_login() -> None:
    """The two real trips in this database have no owner; they must not vanish."""
    from datetime import date

    from app.models.trip import Trip

    with SessionLocal() as session:
        orphan = Trip(
            title="无主项目",
            destination="南京",
            start_date=date(2026, 12, 1),
            end_date=date(2026, 12, 3),
            budget=5000,
            user_id=None,
        )
        session.add(orphan)
        session.commit()
        orphan_id = orphan.id

    identifier = _unique("adopter")
    with _client() as client:
        body = _register(client, identifier).json()
        new_headers = {"Authorization": f"Bearer {body['access_token']}"}

        assert body["claimed_trip_count"] >= 1
        # Readable by the account that just adopted it...
        assert client.get(f"/api/v1/trips/{orphan_id}", headers=new_headers).status_code == 200
        # ...and no longer by anyone else.
        assert client.get(f"/api/v1/trips/{orphan_id}").status_code == 404

    with SessionLocal() as session:
        adopted = session.get(Trip, orphan_id)
        assert adopted.user_id == body["user"]["id"], "the orphan should now belong to the new account"


def test_an_identifier_can_be_attached_to_the_signed_in_account_only_once(register_account) -> None:
    other_user, _ = register_account(_unique("owner"))
    email = f"{uuid4().hex[:8]}@example.com"

    with _client() as client:
        first = client.patch("/api/v1/auth/me", json={"email": email})
        assert first.status_code == 200

    # A second account cannot steal that address.
    with _client() as client:
        token = client.post(
            "/api/v1/auth/login", json={"identifier": other_user["username"], "password": "second-password-123"}
        ).json()["access_token"]
        stolen = client.patch(
            "/api/v1/auth/me", json={"email": email}, headers={"Authorization": f"Bearer {token}"}
        )

    assert stolen.status_code == 409


def test_patching_me_rejects_an_identifier_of_the_wrong_kind() -> None:
    with _client() as client:
        response = client.patch("/api/v1/auth/me", json={"email": "13800138000"})

    assert response.status_code == 422
    assert "邮箱" in response.json()["detail"]
