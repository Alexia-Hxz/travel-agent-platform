import os
import tempfile
from pathlib import Path

# Point the app at a throwaway database BEFORE anything imports app.db.session.
# Tests used to run against the real backend/travel_agent.db, so every single test
# run permanently added fixtures ("行程测试 ab12cd34", "file upload test ...") to the
# user's own trip list: 1405 junk trips had accumulated before this was noticed.
# An operator can still opt in explicitly with TRAVEL_AGENT_TEST_DB_PATH.
_TEST_DB_PATH = os.getenv("TRAVEL_AGENT_TEST_DB_PATH") or str(
    Path(tempfile.mkdtemp(prefix="travel-agent-tests-")) / "test.db"
)
os.environ["TRAVEL_AGENT_DB_PATH"] = _TEST_DB_PATH
os.environ["CATEGORY_MATERIAL_CACHE_PATH"] = str(Path(_TEST_DB_PATH).with_name("category-material.sqlite3"))

# Live web search asks the model to search the internet (`enable_search`), which reaches
# the real network and takes about twenty seconds per lookup. It is off by default so the
# suite stays offline and fast; the tests that exercise it disable this themselves with
# `monkeypatch.delenv("DISABLE_LIVE_WEB_SEARCH")` and patch the model call underneath.
os.environ["DISABLE_LIVE_WEB_SEARCH"] = "1"

import pytest

from app.agents.tools import mealdb_images, pexels_images, place_images, recommendations


@pytest.fixture(autouse=True)
def _isolated_category_material_cache():
    """A test may exercise two calls, but no cached search material may cross tests."""
    recommendations.clear_category_material_cache()
    yield
    recommendations.clear_category_material_cache()


@pytest.fixture(autouse=True)
def _offline_place_image_lookups(monkeypatch: pytest.MonkeyPatch):
    """Keep tests deterministic and offline.

    Place image resolution would otherwise reach the real photo library for every planned
    item, which makes the suite slow and results order-dependent. Every source is stubbed
    here, so a test that wants a photo patches the source it means to exercise and that
    patch takes precedence over this default.
    """
    # The per-place page and the destination's category page are the two searches the
    # resolver makes; both go through this one function.
    monkeypatch.setattr(place_images, "search_pexels_photos", lambda query, limit=3, timeout=8.0: [])
    # Stubbing the HTTP call rather than `city_pool_photos` keeps the pool's own logic
    # running offline: an empty page is never cached, so nothing leaks between tests.
    monkeypatch.setattr(pexels_images, "search_pexels_photos", lambda query, limit=3, timeout=8.0: [])
    pexels_images.clear_pool_cache()
    # The dish library is the second food source, reached whenever the photo library has
    # nothing that matches the dish.
    monkeypatch.setattr(mealdb_images, "search_dish_photos", lambda query, limit=6: [])
    monkeypatch.setattr(mealdb_images, "area_photos", lambda area, limit=12: [])
    mealdb_images.clear_cache()
    # A candidate photo URL is probed for reachability before it is used; tests must
    # not depend on a real photo host answering. Tests that exercise rejection patch
    # ``place_images._is_reachable`` themselves, and that patch takes precedence.
    monkeypatch.setattr(place_images, "_is_reachable", lambda url: True)
    # The DNS pre-check must not touch the real resolver either. Tests that exercise
    # an unresolvable host patch ``place_images.host_resolves`` themselves.
    monkeypatch.setattr(place_images, "host_resolves", lambda host, budget=2.5: True)
    place_images.clear_image_cache()
    yield
    place_images.clear_image_cache()


@pytest.fixture(autouse=True)
def _offline_price_lookups(monkeypatch: pytest.MonkeyPatch):
    """Keep per-place price lookups offline for the same reason as image lookups.

    Tests that exercise price enrichment patch ``recommendations.search_snippets``
    themselves, and that patch takes precedence over this default.
    """
    monkeypatch.setattr(recommendations, "search_snippets", lambda query, limit=3, timeout=8.0: [])
    yield


# The account every pre-existing test runs as. Its password is generated here, never
# printed, and never typed: the override below hands the user object straight to the
# endpoints.
TEST_USER_IDENTIFIER = "pytest-user"
TEST_USER_PASSWORD = "pytest-password-not-a-real-one"


@pytest.fixture(scope="session", autouse=True)
def _authenticated_app():
    """Sign every request in as one real account.

    Endpoints now require a bearer token and 338 tests were written before accounts
    existed. Rewriting all of them to log in first would add three lines to every test
    without testing anything new, so instead this replaces ``get_current_user`` with a user
    that really exists in the throwaway database.

    It is deliberately NOT a blanket bypass: when a request carries an ``Authorization``
    header, the real dependency runs and the token is verified. That is what lets the auth
    tests sign in as a second account and prove the two cannot see each other's trips,
    while every other test keeps calling the API unsigned.
    """
    from fastapi import Depends
    from fastapi.security import HTTPAuthorizationCredentials

    from app.api.deps import bearer_scheme, get_current_user
    from app.db.session import Base, SessionLocal, engine, get_session
    from app.main import app
    from app.models.user import User
    from app.services import auth_service

    # The app creates its tables in its startup lifespan, and this fixture runs before any
    # TestClient opens one, so the throwaway database needs the schema here first.
    Base.metadata.create_all(bind=engine)

    with SessionLocal() as session:
        user = auth_service.find_by_identifier(session, TEST_USER_IDENTIFIER)
        if user is None:
            user = auth_service.create_user(session, TEST_USER_IDENTIFIER, TEST_USER_PASSWORD)
        test_user_id = user.id

    def _current_user(
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
        session=Depends(get_session),
    ) -> User:
        if credentials is not None and credentials.credentials:
            return get_current_user(credentials, session)
        signed_in = session.get(User, test_user_id)
        if signed_in is None:  # pragma: no cover - only if a test deletes the account
            raise AssertionError("the shared test account disappeared")
        return signed_in

    app.dependency_overrides[get_current_user] = _current_user
    yield
    app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture
def register_account():
    """Register an account through the real endpoint and return (user, auth headers).

    Handed to tests that need a SECOND account, so they can prove account A cannot read,
    list, or delete account B's trips while both are using real signed tokens.
    """
    from fastapi.testclient import TestClient

    from app.main import app

    def _register(identifier: str = "second-account", password: str = "second-password-123"):
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/auth/register",
                json={"identifier": identifier, "password": password},
            )
            assert response.status_code == 201, response.text
            body = response.json()
        return body["user"], {"Authorization": f"Bearer {body['access_token']}"}

    return _register
