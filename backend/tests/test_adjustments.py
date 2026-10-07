from datetime import date, time
from uuid import uuid4

from fastapi.testclient import TestClient

from app.db.session import SessionLocal
from app.models.trip import Trip
from app.schemas.adjustment import AdjustmentOperation
from app.schemas.place import PlaceCreate
from app.services.adjustment_service import apply_adjustment, create_pending_adjustment
from app.services.itinerary_service import list_itinerary_items, list_places
from app.main import app


def _create_trip_with_place_and_item(client: TestClient) -> tuple[int, int, int]:
    trip = client.post(
        "/api/v1/trips",
        json={
            "title": f"adjustment test {uuid4().hex[:8]}",
            "destination": "Shanghai",
            "start_date": "2026-10-01",
            "end_date": "2026-10-03",
            "budget": 3000,
        },
    ).json()
    place = client.post(
        f"/api/v1/trips/{trip['id']}/places",
        json={
            "name": "Existing Museum",
            "type": "attraction",
            "address": "People Square",
            "latitude": None,
            "longitude": None,
            "image_url": None,
            "rating": 4.6,
            "price_level": 2,
            "opening_hours": "09:00-17:00",
            "source_url": "https://example.com/museum",
        },
    ).json()
    item = client.post(
        f"/api/v1/trips/{trip['id']}/itinerary-items",
        json={
            "date": "2026-10-01",
            "start_time": "09:00:00",
            "end_time": "11:00:00",
            "place_id": place["id"],
            "note": "original item",
            "sort_order": 0,
        },
    ).json()
    return trip["id"], place["id"], item["id"]


def _new_place_operation() -> AdjustmentOperation:
    return AdjustmentOperation(
        kind="add_place_and_item",
        date=date(2026, 10, 2),
        start_time=time(14, 0),
        end_time=time(15, 30),
        note="afternoon coffee",
        sort_order=1,
        place=PlaceCreate(
            name="Test Cafe",
            type="restaurant",
            address="Nanjing Road",
            latitude=31.2304,
            longitude=121.4737,
            image_url=None,
            rating=4.5,
            price_level=2,
            opening_hours="10:00-22:00",
            source_url="https://example.com/cafe",
        ),
    )


def test_creating_a_pending_adjustment_does_not_change_trip_data() -> None:
    with TestClient(app) as client:
        trip_id, place_id, item_id = _create_trip_with_place_and_item(client)

    with SessionLocal() as session:
        trip = session.get(Trip, trip_id)
        assert trip is not None
        adjustment = create_pending_adjustment(
            session,
            trip,
            instruction="Add a cafe tomorrow afternoon",
            operations=[_new_place_operation()],
        )

        assert adjustment.status == "pending"
        assert [place.id for place in list_places(session, trip_id)] == [place_id]
        assert [item.id for item in list_itinerary_items(session, trip_id)] == [item_id]


def test_applying_a_pending_adjustment_creates_its_place_and_itinerary_item() -> None:
    with TestClient(app) as client:
        trip_id, _, _ = _create_trip_with_place_and_item(client)

    with SessionLocal() as session:
        trip = session.get(Trip, trip_id)
        assert trip is not None
        adjustment = create_pending_adjustment(
            session,
            trip,
            instruction="Add a cafe tomorrow afternoon",
            operations=[_new_place_operation()],
        )

        items = apply_adjustment(session, trip, adjustment)

        assert adjustment.status == "applied"
        assert [place.name for place in list_places(session, trip_id)] == ["Existing Museum", "Test Cafe"]
        assert items[-1].place.name == "Test Cafe"


def test_applying_a_budget_adjustment_updates_trip_budget() -> None:
    with TestClient(app) as client:
        trip_id, _, _ = _create_trip_with_place_and_item(client)

    with SessionLocal() as session:
        trip = session.get(Trip, trip_id)
        assert trip is not None
        adjustment = create_pending_adjustment(
            session,
            trip,
            instruction="总预算改为5000元",
            operations=[AdjustmentOperation(kind="set_budget", budget_amount=5000)],
        )

        apply_adjustment(session, trip, adjustment)
        session.refresh(trip)
        assert trip.budget == 5000


def test_selecting_one_side_of_a_replacement_applies_both_operations() -> None:
    with TestClient(app) as client:
        trip_id, _, item_id = _create_trip_with_place_and_item(client)

    with SessionLocal() as session:
        trip = session.get(Trip, trip_id)
        assert trip is not None
        replacement = create_pending_adjustment(
            session,
            trip,
            instruction="替换上午项目",
            operations=[
                AdjustmentOperation(kind="remove_item", item_id=item_id),
                AdjustmentOperation(
                    kind="add_place_and_item",
                    date=date(2026, 10, 1),
                    start_time=time(9, 0),
                    end_time=time(11, 0),
                    note="替换为新的推荐项目，保留原时间段",
                    sort_order=0,
                    place=PlaceCreate(name="Replacement Park", type="attraction", address="Shanghai", price_level=3),
                ),
            ],
        )

        apply_adjustment(session, trip, replacement, operation_indexes=[0])

        assert replacement.status == "applied"
        assert [item.place.name for item in list_itinerary_items(session, trip_id)] == ["Replacement Park"]
