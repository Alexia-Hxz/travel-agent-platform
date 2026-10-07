"""The test suite must never write into the user's real database.

Every test run used to create trips in ``backend/travel_agent.db`` and leave them
there. Over many runs that filled the user's trip list with 1405 fixture trips,
which is exactly what these tests prevent from happening again.
"""

import os
from pathlib import Path

from app.db import session as db_session


def test_tests_use_a_throwaway_database() -> None:
    assert Path(db_session.DATABASE_PATH).resolve() != db_session.DEFAULT_DATABASE_PATH.resolve()


def test_throwaway_database_is_not_the_projects_real_file() -> None:
    database_name = Path(db_session.DATABASE_PATH).name

    assert database_name != "travel_agent.db"
    # Either the per-run temp directory or an explicitly chosen test database.
    assert "travel-agent-tests-" in str(db_session.DATABASE_PATH) or os.getenv("TRAVEL_AGENT_TEST_DB_PATH")


def test_database_url_follows_the_override() -> None:
    assert db_session.DATABASE_URL == f"sqlite:///{Path(db_session.DATABASE_PATH).as_posix()}"


def test_the_real_database_file_is_left_alone() -> None:
    """Creating a trip in a test must not append rows to the real database file."""
    real_db = db_session.DEFAULT_DATABASE_PATH
    if not real_db.exists():
        return

    from fastapi.testclient import TestClient

    from app.main import app

    before = real_db.stat().st_size
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/trips",
            json={
                "title": "guard test",
                "destination": "Nowhere",
                "start_date": "2026-10-01",
                "end_date": "2026-10-03",
                "budget": 1000,
            },
        )
        assert response.status_code in (200, 201)

    assert real_db.stat().st_size == before
