from datetime import date
from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app


def _trip_payload() -> dict[str, object]:
    suffix = uuid4().hex[:8]
    return {
        "title": f"上海周末游 {suffix}",
        "destination": "上海",
        "start_date": "2026-10-01",
        "end_date": "2026-10-03",
        "budget": 3200,
    }


def test_create_trip_then_retrieve_and_list_it() -> None:
    with TestClient(app) as client:
        payload = _trip_payload()

        create_response = client.post("/api/v1/trips", json=payload)

        assert create_response.status_code == 201
        created = create_response.json()
        assert created["title"] == payload["title"]
        assert created["destination"] == "上海"
        assert created["budget"] == 3200
        assert created["start_date"] == date(2026, 10, 1).isoformat()

        trip_id = created["id"]
        assert client.get(f"/api/v1/trips/{trip_id}").json()["id"] == trip_id
        assert any(item["id"] == trip_id for item in client.get("/api/v1/trips").json())


def test_rejects_trip_with_end_date_before_start_date() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/trips",
            json={
                **_trip_payload(),
                "start_date": "2026-10-03",
                "end_date": "2026-10-01",
            },
        )

    assert response.status_code == 422


def test_delete_trip_makes_it_unavailable() -> None:
    with TestClient(app) as client:
        trip_id = client.post("/api/v1/trips", json=_trip_payload()).json()["id"]

        assert client.delete(f"/api/v1/trips/{trip_id}").status_code == 204
        assert client.get(f"/api/v1/trips/{trip_id}").status_code == 404


def test_create_trip_persists_traveler_composition() -> None:
    with TestClient(app) as client:
        payload = {
            **_trip_payload(),
            "adult_male_count": 1,
            "adult_female_count": 2,
            "child_male_count": 1,
            "child_female_count": 0,
        }
        response = client.post("/api/v1/trips", json=payload)

    assert response.status_code == 201
    body = response.json()
    assert body["adult_male_count"] == 1
    assert body["adult_female_count"] == 2
    assert body["child_male_count"] == 1
    assert body["child_female_count"] == 0
