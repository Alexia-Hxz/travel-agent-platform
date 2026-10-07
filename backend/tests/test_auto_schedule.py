"""Saved places should reach the calendar without the user typing every field.

Adding a place to a day meant opening a dialog and filling in the place, the date and
both times by hand, even though the app already knows the opening hours and what else
is booked that day.
"""

from datetime import date, time
from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app
from app.services.auto_schedule import parse_opening_window, propose_schedule


def test_opening_hours_are_read_in_the_forms_that_actually_appear() -> None:
    assert parse_opening_window("09:00-17:00") == (time(9, 0), time(17, 0))
    # An en dash, as the model and search snippets often produce.
    assert parse_opening_window("08:30–21:00") == (time(8, 30), time(21, 0))
    # Prefixed with a day range.
    assert parse_opening_window("周一至周日 09:00-17:30") == (time(9, 0), time(17, 30))
    # Nothing usable: the caller falls back to the default day window.
    assert parse_opening_window("全天") is None
    assert parse_opening_window(None) is None
    assert parse_opening_window("") is None


def test_a_window_crossing_midnight_is_not_used_as_a_day_window() -> None:
    # "22:00-02:00" cannot be modelled inside one day, so it is ignored safely.
    assert parse_opening_window("22:00-02:00") is None


def test_proposed_slots_never_overlap_each_other() -> None:
    """The first place used to be proposed the same slot as the second.

    Each place searched the original free time, because a placement was not fed back
    into the day's bookings.
    """
    places = [
        {"id": 1, "name": "中山陵", "type": "attraction", "opening_hours": "09:00-17:00"},
        {"id": 2, "name": "南京博物院", "type": "attraction", "opening_hours": "09:00-17:00"},
        {"id": 3, "name": "南京大牌檔", "type": "restaurant", "opening_hours": None},
    ]

    proposal = propose_schedule(places, [], "2026-10-10", "2026-10-10")

    assert len(proposal.proposed) == 3
    slots = sorted((item.start_time, item.end_time) for item in proposal.proposed)
    for (start, end), (next_start, _) in zip(slots, slots[1:]):
        assert end <= next_start, f"{start}-{end} overlaps {next_start}"


def test_proposed_slots_avoid_what_is_already_booked() -> None:
    places = [{"id": 1, "name": "中山陵", "type": "attraction", "opening_hours": "09:00-17:00"}]
    existing = [
        {
            "place_id": 99,
            "date": "2026-10-10",
            "start_time": time(9, 0),
            "end_time": time(12, 0),
        }
    ]

    proposal = propose_schedule(places, existing, "2026-10-10", "2026-10-10")

    assert len(proposal.proposed) == 1
    item = proposal.proposed[0]
    # The whole morning is taken, so the first free slot starts at 12:00.
    assert item.start_time == "12:00:00"
    assert item.end_time == "14:00:00"


def test_a_place_is_kept_inside_its_own_opening_hours() -> None:
    places = [{"id": 1, "name": "中山陵", "type": "attraction", "opening_hours": "13:00-16:00"}]

    proposal = propose_schedule(places, [], "2026-10-10", "2026-10-10")

    assert len(proposal.proposed) == 1
    item = proposal.proposed[0]
    assert item.start_time == "13:00:00"
    # Two hours, still inside the 13:00-16:00 window.
    assert item.end_time == "15:00:00"
    assert "13:00-16:00" in item.note


def test_an_evening_only_place_is_scheduled_in_the_evening() -> None:
    """A river cruise open 19:00-22:00 must not be dropped for being outside 9-6."""
    places = [{"id": 1, "name": "夜游秦淮河", "type": "attraction", "opening_hours": "19:00-22:00"}]

    proposal = propose_schedule(places, [], "2026-10-10", "2026-10-10")

    assert len(proposal.proposed) == 1
    assert proposal.proposed[0].start_time == "19:00:00"


def test_places_already_on_the_calendar_are_not_proposed_again() -> None:
    places = [{"id": 1, "name": "中山陵", "type": "attraction", "opening_hours": None}]
    existing = [
        {"place_id": 1, "date": "2026-10-10", "start_time": time(9, 0), "end_time": time(11, 0)}
    ]

    proposal = propose_schedule(places, existing, "2026-10-10", "2026-10-11")

    assert proposal.proposed == []
    assert proposal.already_scheduled == ["中山陵"]


def test_a_place_that_cannot_fit_is_reported_instead_of_disappearing() -> None:
    # A one-hour opening window filled by an existing booking leaves no room.
    places = [{"id": 1, "name": "限时展览", "type": "attraction", "opening_hours": "09:00-10:00"}]
    existing = [
        {"place_id": 99, "date": "2026-10-10", "start_time": time(9, 0), "end_time": time(10, 0)}
    ]

    proposal = propose_schedule(places, existing, "2026-10-10", "2026-10-10")

    assert proposal.proposed == []
    assert len(proposal.skipped) == 1
    assert proposal.skipped[0]["place_name"] == "限时展览"
    assert "没有足够空闲时段" in proposal.skipped[0]["reason"]


def test_places_are_spread_across_days_when_one_day_is_full() -> None:
    # Day one 09:00-21:00 is fully booked, so the place must move to day two.
    places = [{"id": 1, "name": "中山陵", "type": "attraction", "opening_hours": "09:00-17:00"}]
    existing = [
        {"place_id": 99, "date": "2026-10-10", "start_time": time(9, 0), "end_time": time(21, 0)}
    ]

    proposal = propose_schedule(places, existing, "2026-10-10", "2026-10-11")

    assert len(proposal.proposed) == 1
    assert proposal.proposed[0].date == "2026-10-11"


def test_a_preferred_day_is_honoured_over_the_earliest_free_day() -> None:
    """Ticking a recommendation inside a day group must keep it on that day.

    The scheduler always took the earliest day that fitted, so a place the user
    ticked under day three silently landed on day one.
    """
    places = [
        {"id": 1, "name": "夫子庙", "type": "attraction", "opening_hours": None, "preferred_date": "2026-10-12"}
    ]

    proposal = propose_schedule(places, [], "2026-10-10", "2026-10-14")

    assert proposal.proposed[0].date == "2026-10-12"
    assert "按规划建议放在这一天" in proposal.proposed[0].note


def test_a_preferred_day_that_is_full_falls_back_and_says_so() -> None:
    """A crowded preferred day must not lose the place, and must not pretend it fitted."""
    places = [
        {"id": 1, "name": "中山陵", "type": "attraction", "opening_hours": None, "preferred_date": "2026-10-12"}
    ]
    existing = [
        {"place_id": 99, "date": "2026-10-12", "start_time": time(9, 0), "end_time": time(21, 0)}
    ]

    proposal = propose_schedule(places, existing, "2026-10-10", "2026-10-14")

    assert len(proposal.proposed) == 1
    assert proposal.proposed[0].date == "2026-10-10"
    assert "2026-10-12 已排满" in proposal.proposed[0].note
    assert "改排到这一天" in proposal.proposed[0].note


def test_a_preferred_date_outside_the_trip_is_ignored() -> None:
    """A stale day (the trip was shortened) must not schedule anything outside the trip."""
    places = [
        {"id": 1, "name": "夫子庙", "type": "attraction", "opening_hours": None, "preferred_date": "2026-11-30"}
    ]

    proposal = propose_schedule(places, [], "2026-10-10", "2026-10-14")

    assert proposal.proposed[0].date == "2026-10-10"
    assert "已排满" not in proposal.proposed[0].note


def test_a_date_object_works_as_a_preferred_date() -> None:
    """The model hands over a real date, while JSON payloads arrive as ISO strings."""
    places = [
        {"id": 1, "name": "夫子庙", "type": "attraction", "opening_hours": None, "preferred_date": date(2026, 10, 13)}
    ]

    proposal = propose_schedule(places, [], "2026-10-10", "2026-10-14")

    assert proposal.proposed[0].date == "2026-10-13"


def test_without_a_preferred_day_behaviour_is_unchanged() -> None:
    places = [{"id": 1, "name": "夫子庙", "type": "attraction", "opening_hours": None}]

    proposal = propose_schedule(places, [], "2026-10-10", "2026-10-14")

    assert proposal.proposed[0].date == "2026-10-10"
    assert proposal.proposed[0].note == "自动安排（无开放时间信息）"


def _trip(client: TestClient) -> int:
    return client.post(
        "/api/v1/trips",
        json={
            "title": f"auto schedule {uuid4().hex[:8]}",
            "destination": "南京",
            "start_date": "2026-10-10",
            "end_date": "2026-10-11",
            "budget": 5000,
        },
    ).json()["id"]


def _place(client: TestClient, trip_id: int, name: str, opening_hours: str | None) -> int:
    return client.post(
        f"/api/v1/trips/{trip_id}/places",
        json={
            "name": name,
            "type": "attraction",
            "address": "南京市玄武区",
            "latitude": None,
            "longitude": None,
            "image_url": None,
            "rating": 4.5,
            "price_level": 2,
            "opening_hours": opening_hours,
            "source_url": None,
        },
    ).json()["id"]


def test_the_suggested_day_survives_saving_a_place_and_reaches_the_proposal() -> None:
    """The whole chain: a ticked recommendation carries its day all the way through.

    The recommendation panel knows which day an item was suggested for, and this is
    the piece that used to be dropped at every step — at save time, at the column,
    and at the scheduler.
    """
    with TestClient(app) as client:
        trip_id = _trip(client)
        created = client.post(
            f"/api/v1/trips/{trip_id}/places",
            json={
                "name": "夫子庙",
                "type": "attraction",
                "address": "南京市秦淮区",
                "latitude": None,
                "longitude": None,
                "image_url": None,
                "rating": None,
                "price_level": None,
                "opening_hours": None,
                "source_url": None,
                "preferred_date": "2026-10-11",
            },
        )
        assert created.status_code == 201
        assert created.json()["preferred_date"] == "2026-10-11"

        # And it is readable back, so the frontend can show and use it.
        places = client.get(f"/api/v1/trips/{trip_id}/places").json()
        assert places[0]["preferred_date"] == "2026-10-11"

        body = client.post(f"/api/v1/trips/{trip_id}/itinerary-items/auto-schedule").json()

        assert body["proposed"][0]["date"] == "2026-10-11"
        assert "按规划建议放在这一天" in body["proposed"][0]["note"]


def test_a_place_without_a_suggested_day_still_schedules_as_before() -> None:
    with TestClient(app) as client:
        trip_id = _trip(client)
        _place(client, trip_id, "中山陵", "08:30-17:00")

        body = client.post(f"/api/v1/trips/{trip_id}/itinerary-items/auto-schedule").json()

        assert body["proposed"][0]["date"] == "2026-10-10"
        assert "按规划建议" not in body["proposed"][0]["note"]


def test_the_auto_schedule_endpoint_only_proposes_and_creates_nothing() -> None:
    with TestClient(app) as client:
        trip_id = _trip(client)
        _place(client, trip_id, "中山陵", "08:30-17:00")
        _place(client, trip_id, "南京博物院", "09:00-17:00")

        response = client.post(f"/api/v1/trips/{trip_id}/itinerary-items/auto-schedule")

        assert response.status_code == 200
        body = response.json()
        assert [item["place_name"] for item in body["proposed"]] == ["中山陵", "南京博物院"]
        assert "2 个收藏地点" in body["message"]

        # A preview must not write anything: the caller applies it deliberately.
        assert client.get(f"/api/v1/trips/{trip_id}/itinerary-items").json() == []


def test_the_auto_schedule_endpoint_reports_places_that_are_already_scheduled() -> None:
    with TestClient(app) as client:
        trip_id = _trip(client)
        place_id = _place(client, trip_id, "中山陵", "08:30-17:00")
        client.post(
            f"/api/v1/trips/{trip_id}/itinerary-items",
            json={
                "date": "2026-10-10",
                "start_time": "09:00:00",
                "end_time": "11:00:00",
                "place_id": place_id,
                "note": None,
                "sort_order": 0,
            },
        )

        body = client.post(f"/api/v1/trips/{trip_id}/itinerary-items/auto-schedule").json()

        assert body["proposed"] == []
        assert body["already_scheduled"] == ["中山陵"]
        assert "都已经排进行程" in body["message"]
