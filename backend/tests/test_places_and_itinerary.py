from datetime import date
from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app


def _trip_payload() -> dict[str, object]:
    suffix = uuid4().hex[:8]
    return {
        "title": f"行程测试 {suffix}",
        "destination": "上海",
        "start_date": "2026-10-01",
        "end_date": "2026-10-03",
        "budget": 3200,
    }


def _place_payload() -> dict[str, object]:
    return {
        "name": "上海博物馆",
        "type": "attraction",
        "address": "上海市黄浦区人民大道201号",
        "latitude": 31.2304,
        "longitude": 121.4737,
        "image_url": "https://example.com/shanghai-museum.jpg",
        "rating": 4.8,
        "price_level": 2,
        "opening_hours": "09:00-17:00",
        "source_url": "https://example.com/shanghai-museum",
    }


def _create_trip(client: TestClient) -> int:
    return client.post("/api/v1/trips", json=_trip_payload()).json()["id"]


def test_a_saved_place_keeps_the_recommendation_reason_and_price() -> None:
    """Saving a recommendation must not throw away why it was chosen.

    Ticking a recommendation and pressing 加入收藏地点 used to store only the name and
    address, so 收藏地点 could not explain itself and the summary had nothing to say
    about why the user picked the place.
    """
    with TestClient(app) as client:
        trip_id = _create_trip(client)
        payload = {
            **_place_payload(),
            "name": "金陵饭店",
            "type": "hotel",
            "note": "位置便利，靠近夫子庙，评分高",
            "estimated_cost": 993,
        }

        created = client.post(f"/api/v1/trips/{trip_id}/places", json=payload)
        assert created.status_code == 201
        body = created.json()
        assert body["note"] == "位置便利，靠近夫子庙，评分高"
        assert body["estimated_cost"] == 993

        listed = client.get(f"/api/v1/trips/{trip_id}/places").json()
        saved = next(place for place in listed if place["name"] == "金陵饭店")
        assert saved["note"] == "位置便利，靠近夫子庙，评分高"
        assert saved["estimated_cost"] == 993


def test_a_place_without_a_reason_still_saves() -> None:
    """Places added by hand have no reason, and that must stay valid."""
    with TestClient(app) as client:
        trip_id = _create_trip(client)
        created = client.post(f"/api/v1/trips/{trip_id}/places", json=_place_payload())

        assert created.status_code == 201
        body = created.json()
        assert body["note"] is None
        assert body["estimated_cost"] is None


def test_saves_and_lists_all_supported_place_types_for_a_trip() -> None:
    with TestClient(app) as client:
        trip_id = _create_trip(client)
        saved_place_ids = []

        for place_type in ("attraction", "hotel", "restaurant", "airport", "station"):
            response = client.post(
                f"/api/v1/trips/{trip_id}/places",
                json={**_place_payload(), "name": f"{place_type} 地点", "type": place_type},
            )

            assert response.status_code == 201
            saved = response.json()
            saved_place_ids.append(saved["id"])
            assert saved["trip_id"] == trip_id
            assert saved["type"] == place_type
            assert saved["latitude"] == 31.2304
            assert saved["opening_hours"] == "09:00-17:00"

        listed = client.get(f"/api/v1/trips/{trip_id}/places")

    assert listed.status_code == 200
    assert [place["id"] for place in listed.json()] == saved_place_ids


def test_deletes_a_saved_place_without_deleting_its_trip() -> None:
    with TestClient(app) as client:
        trip_id = _create_trip(client)
        place_id = client.post(f"/api/v1/trips/{trip_id}/places", json=_place_payload()).json()["id"]

        delete_response = client.delete(f"/api/v1/trips/{trip_id}/places/{place_id}")

        assert delete_response.status_code == 204
        assert client.get(f"/api/v1/trips/{trip_id}").status_code == 200
        assert client.get(f"/api/v1/trips/{trip_id}/places").json() == []


def test_creates_and_lists_ordered_daily_itinerary_items() -> None:
    with TestClient(app) as client:
        trip_id = _create_trip(client)
        place_id = client.post(f"/api/v1/trips/{trip_id}/places", json=_place_payload()).json()["id"]

        later_response = client.post(
            f"/api/v1/trips/{trip_id}/itinerary-items",
            json={
                "date": "2026-10-02",
                "start_time": "14:00:00",
                "end_time": "16:00:00",
                "place_id": place_id,
                "note": "下午参观",
                "sort_order": 2,
            },
        )
        earlier_response = client.post(
            f"/api/v1/trips/{trip_id}/itinerary-items",
            json={
                "date": "2026-10-01",
                "start_time": "09:00:00",
                "end_time": "11:00:00",
                "place_id": place_id,
                "note": "第一站",
                "sort_order": 1,
            },
        )

        listed = client.get(f"/api/v1/trips/{trip_id}/itinerary-items")

    assert later_response.status_code == 201
    assert earlier_response.status_code == 201
    assert listed.status_code == 200
    assert [item["date"] for item in listed.json()] == [date(2026, 10, 1).isoformat(), date(2026, 10, 2).isoformat()]
    assert listed.json()[0]["place"]["name"] == "上海博物馆"
    assert listed.json()[1]["note"] == "下午参观"


def test_rejects_itinerary_item_outside_trip_dates() -> None:
    with TestClient(app) as client:
        trip_id = _create_trip(client)
        place_id = client.post(f"/api/v1/trips/{trip_id}/places", json=_place_payload()).json()["id"]

        response = client.post(
            f"/api/v1/trips/{trip_id}/itinerary-items",
            json={
                "date": "2026-10-04",
                "start_time": "09:00:00",
                "end_time": "10:00:00",
                "place_id": place_id,
                "note": None,
                "sort_order": 0,
            },
        )

    assert response.status_code == 422


def test_rejects_an_itinerary_item_using_another_trips_place() -> None:
    with TestClient(app) as client:
        source_trip_id = _create_trip(client)
        target_trip_id = _create_trip(client)
        place_id = client.post(f"/api/v1/trips/{source_trip_id}/places", json=_place_payload()).json()["id"]

        response = client.post(
            f"/api/v1/trips/{target_trip_id}/itinerary-items",
            json={
                "date": "2026-10-02",
                "start_time": "09:00:00",
                "end_time": "10:00:00",
                "place_id": place_id,
                "note": None,
                "sort_order": 0,
            },
        )

    assert response.status_code == 422


def test_deletes_an_itinerary_item() -> None:
    with TestClient(app) as client:
        trip_id = _create_trip(client)
        place_id = client.post(f"/api/v1/trips/{trip_id}/places", json=_place_payload()).json()["id"]
        item_id = client.post(
            f"/api/v1/trips/{trip_id}/itinerary-items",
            json={
                "date": "2026-10-01",
                "start_time": "09:00:00",
                "end_time": "10:00:00",
                "place_id": place_id,
                "note": None,
                "sort_order": 0,
            },
        ).json()["id"]

        delete_response = client.delete(f"/api/v1/trips/{trip_id}/itinerary-items/{item_id}")

        assert delete_response.status_code == 204
        assert client.get(f"/api/v1/trips/{trip_id}/itinerary-items").json() == []


def test_deleting_a_saved_place_removes_its_itinerary_items() -> None:
    with TestClient(app) as client:
        trip_id = _create_trip(client)
        place_id = client.post(f"/api/v1/trips/{trip_id}/places", json=_place_payload()).json()["id"]
        client.post(
            f"/api/v1/trips/{trip_id}/itinerary-items",
            json={
                "date": "2026-10-01",
                "start_time": "09:00:00",
                "end_time": "10:00:00",
                "place_id": place_id,
                "note": None,
                "sort_order": 0,
            },
        )

        delete_response = client.delete(f"/api/v1/trips/{trip_id}/places/{place_id}")

        assert delete_response.status_code == 204
        assert client.get(f"/api/v1/trips/{trip_id}/itinerary-items").json() == []


def _scheduled_place(client: TestClient, trip_id: int, name: str) -> int:
    """Save one place and book it on the first day, as the 收藏 flow does."""
    place_id = client.post(
        f"/api/v1/trips/{trip_id}/places", json={**_place_payload(), "name": name}
    ).json()["id"]
    client.post(
        f"/api/v1/trips/{trip_id}/itinerary-items",
        json={
            "date": "2026-10-01",
            "start_time": "09:00:00",
            "end_time": "10:00:00",
            "place_id": place_id,
            "note": None,
            "sort_order": 0,
        },
    )
    return place_id


def test_clearing_the_saved_places_reports_how_many_went_and_empties_the_calendar() -> None:
    """一键清空收藏 must leave nothing behind, including the days it filled in.

    The button exists so a plan can be started over from a clean slate. Clearing one
    place at a time from the client would leave the user with a half-cleared list if one
    request failed, and a day still holding the place that was just removed.
    """
    with TestClient(app) as client:
        trip_id = _create_trip(client)
        _scheduled_place(client, trip_id, "上海博物馆")
        _scheduled_place(client, trip_id, "外滩")
        assert len(client.get(f"/api/v1/trips/{trip_id}/places").json()) == 2
        assert len(client.get(f"/api/v1/trips/{trip_id}/itinerary-items").json()) == 2

        cleared = client.delete(f"/api/v1/trips/{trip_id}/places")

        assert cleared.status_code == 200
        assert cleared.json() == {"removed": 2}
        assert client.get(f"/api/v1/trips/{trip_id}/places").json() == []
        assert client.get(f"/api/v1/trips/{trip_id}/itinerary-items").json() == []
        # The trip itself is not the thing being cleared.
        assert client.get(f"/api/v1/trips/{trip_id}").status_code == 200


def test_clearing_places_leaves_another_trips_places_alone() -> None:
    with TestClient(app) as client:
        cleared_trip = _create_trip(client)
        kept_trip = _create_trip(client)
        _scheduled_place(client, cleared_trip, "上海博物馆")
        kept_place_id = _scheduled_place(client, kept_trip, "外滩")

        client.delete(f"/api/v1/trips/{cleared_trip}/places")

        kept = client.get(f"/api/v1/trips/{kept_trip}/places").json()
        assert [place["id"] for place in kept] == [kept_place_id]
        assert len(client.get(f"/api/v1/trips/{kept_trip}/itinerary-items").json()) == 1


def test_clearing_an_already_empty_saved_list_is_not_an_error() -> None:
    """Pressing the button twice, or on a trip that never saved anything, must be safe."""
    with TestClient(app) as client:
        trip_id = _create_trip(client)

        cleared = client.delete(f"/api/v1/trips/{trip_id}/places")

        assert cleared.status_code == 200
        assert cleared.json() == {"removed": 0}


def test_clearing_places_of_an_unknown_trip_is_a_404() -> None:
    with TestClient(app) as client:
        cleared = client.delete("/api/v1/trips/999999/places")

        assert cleared.status_code == 404


def test_clearing_places_does_not_shadow_a_single_place_delete() -> None:
    """``DELETE /places`` and ``DELETE /places/{id}`` have to stay distinguishable."""
    with TestClient(app) as client:
        trip_id = _create_trip(client)
        first = _scheduled_place(client, trip_id, "上海博物馆")
        second = _scheduled_place(client, trip_id, "外滩")

        single = client.delete(f"/api/v1/trips/{trip_id}/places/{first}")

        assert single.status_code == 204
        remaining = client.get(f"/api/v1/trips/{trip_id}/places").json()
        assert [place["id"] for place in remaining] == [second]
