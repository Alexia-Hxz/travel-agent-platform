"""Asking for places by hand: "有没有更适合亲子的".

The promise here is narrow and testable: a question comes back as *candidates*, each one a
separate operation the traveller ticks, nothing already on the trip is offered, and the trip is
not rearranged unless the traveller's own words asked for a swap.
"""
from datetime import date as Date, time
from types import SimpleNamespace

import pytest

from app.agents.tools import day_plan
from app.services import adjustment_agent_service


def _place(name: str, category: str, **extra) -> dict:
    return {"name": name, "type": category, "address": "南京 · 地址待确认", **extra}


def _fill_with(pool: dict[str, list[dict]], seen: dict | None = None, hints: dict | None = None):
    """A stand-in for the category fill that records what it was told to avoid and to find."""

    def fake_fill(destination, category, count, allowances=None, plan=None, search_hint=None, exclude=()):
        if seen is not None:
            seen[category] = list(exclude)
        if hints is not None:
            hints[category] = search_hint
        return [dict(place) for place in pool.get(category, [])]

    return fake_fill


@pytest.fixture(autouse=True)
def _quiet_pricing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pricing and photos are other modules' business: keep this file about the search."""
    monkeypatch.setattr(day_plan, "estimate_trip_costs", lambda places, *args, **kwargs: places)
    monkeypatch.setattr(day_plan, "resolve_place_images", lambda places, *args, **kwargs: places)


def test_a_question_about_children_asks_for_sights_only() -> None:
    assert day_plan.discovery_categories("有没有更适合亲子游玩的项目或者景点，给我推荐几个") == ["attraction"]
    # A question that names a meal asks for food; one that names neither asks for both rather
    # than silently guessing a kind the traveller did not want.
    assert day_plan.discovery_categories("推荐几个夜宵或者小吃") == ["restaurant"]
    assert day_plan.discovery_categories("还有什么值得去的地方") == ["attraction"]
    assert day_plan.discovery_categories("再来点") == ["attraction", "restaurant"]


def test_only_a_swap_question_asks_for_a_replacement() -> None:
    assert day_plan.asks_to_replace("把第2天下午换成亲子的地方") is True
    assert day_plan.asks_to_replace("推荐几个适合亲子的地方") is False


def test_the_travellers_own_words_become_the_search_hint_and_stay_out_of_the_way(monkeypatch) -> None:
    seen: dict = {}
    hints: dict = {}
    monkeypatch.setattr(
        day_plan,
        "_fill_category",
        _fill_with(
            {
                "attraction": [_place("已去过", "attraction"), _place("亲子乐园", "attraction"), _place("水族馆", "attraction")],
                "restaurant": [_place("亲子餐厅", "restaurant")],
            },
            seen,
            hints,
        ),
    )

    places = day_plan.plan_discovery("南京", "有没有更适合亲子的景点，给我推荐几个", ["已去过"], 4)

    names = [place["name"] for place in places]
    # A question about sights asks for sights alone, so no restaurant comes back.
    assert names == ["亲子乐园", "水族馆"], names
    # The question is handed down as the search hint, so this searches for family places
    # instead of returning the same famous handful again.
    assert hints["attraction"] == "有没有更适合亲子的景点，给我推荐几个"
    assert seen["attraction"] == ["已去过"]


def test_candidates_are_added_and_never_rearrange_the_day_unless_asked(monkeypatch) -> None:
    existing = [
        {"id": 11, "date": "2026-11-03", "start_time": "09:30:00", "end_time": "12:00:00", "type": "attraction"},
        {"id": 12, "date": "2026-11-03", "start_time": "12:00:00", "end_time": "13:00:00", "type": "restaurant"},
        {"id": 13, "date": "2026-11-04", "start_time": "09:30:00", "end_time": "12:00:00", "type": "attraction"},
    ]
    places = [_place("亲子乐园", "attraction"), _place("水族馆", "attraction")]

    added = day_plan.discovery_operations(Date(2026, 11, 3), places, existing, "南京")

    assert [operation["kind"] for operation in added] == ["add_place_and_item", "add_place_and_item"]
    # The day already holds a 09:30 sight, so the first candidate is offered the free afternoon
    # slot instead of being stacked on top of it.
    assert [operation["start_time"] for operation in added] == [time(14, 0), time(17, 0)]
    # Another day is never touched.
    assert all(operation.get("item_id") != 13 for operation in added)

    swapped = day_plan.discovery_operations(Date(2026, 11, 3), places, existing, "南京", replace=True)

    assert [operation["kind"] for operation in swapped] == ["remove_item", "add_place_and_item", "add_place_and_item"]
    # Only that day's sight is dropped: the restaurant and the other day stay.
    assert [operation["item_id"] for operation in swapped if operation["kind"] == "remove_item"] == [11]


def test_no_two_candidates_share_a_time_even_when_the_day_is_busy() -> None:
    """The traveller should not have to untangle overlapping candidates by hand."""
    existing = [
        {"id": 21, "date": "2026-11-03", "start_time": "09:30:00", "end_time": "12:00:00", "type": "attraction"},
        {"id": 22, "date": "2026-11-03", "start_time": "14:00:00", "end_time": "17:00:00", "type": "attraction"},
        {"id": 23, "date": "2026-11-03", "start_time": "12:00:00", "end_time": "13:00:00", "type": "restaurant"},
        {"id": 24, "date": "2026-11-03", "start_time": "18:00:00", "end_time": "19:30:00", "type": "restaurant"},
    ]
    places = [
        _place("亲子乐园", "attraction"),
        _place("水族馆", "attraction"),
        _place("科学馆", "attraction"),
        _place("拉面店", "restaurant"),
        _place("烧鸟店", "restaurant"),
    ]

    added = day_plan.discovery_operations(Date(2026, 11, 3), places, existing, "南京")

    windows = [(operation["start_time"], operation["end_time"]) for operation in added]
    assert len(set(windows)) == len(windows), windows
    # Nothing overlaps what the day already holds either.
    kept = {(time(9, 30), time(12, 0)), (time(14, 0), time(17, 0)), (time(12, 0), time(13, 0)), (time(18, 0), time(19, 30))}
    assert not (set(windows) & kept), windows


def test_a_question_that_names_a_day_lands_on_that_day() -> None:
    trip = SimpleNamespace(start_date=Date(2026, 11, 3), end_date=Date(2026, 11, 6))

    assert adjustment_agent_service._asked_day("把第3天下午换成室内的", trip) == Date(2026, 11, 5)
    # Outside the trip, or not named at all, is left to the caller's default.
    assert adjustment_agent_service._asked_day("第 9 天呢", trip) is None
    assert adjustment_agent_service._asked_day("推荐几个亲子的", trip) is None


def test_the_discover_mode_routes_to_the_search(monkeypatch) -> None:
    called: dict = {}

    def fake_discovery(session, trip, instruction, target_date, traveler_profile):
        called["instruction"] = instruction
        called["date"] = target_date
        return {"id": 1}

    monkeypatch.setattr(adjustment_agent_service, "_create_discovery_draft", fake_discovery)
    trip = SimpleNamespace(
        id=3,
        destination="南京",
        start_date=Date(2026, 11, 3),
        end_date=Date(2026, 11, 6),
        budget=5000,
        adult_male_count=1,
        adult_female_count=1,
        child_male_count=0,
        child_female_count=0,
    )

    result = adjustment_agent_service.create_adjustment_draft(
        None,
        trip,
        "有没有更适合亲子的",
        False,
        mode=adjustment_agent_service.DISCOVERY_MODE,
        target_date=Date(2026, 11, 4),
    )

    assert result == {"id": 1}
    assert called == {"instruction": "有没有更适合亲子的", "date": Date(2026, 11, 4)}
