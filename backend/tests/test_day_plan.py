"""A single day, arranged from what the trip already has.

The promise this feature makes is narrow and testable: a day arranged this way contains
nothing that is already on another day (or already saved on the trip), and a day that cannot
be filled is reported short instead of being padded with repeats.
"""
from datetime import date as Date, time
from types import SimpleNamespace

import pytest

from app.agents.tools import day_plan
from app.services import adjustment_agent_service


def _place(name: str, category: str, **extra) -> dict:
    return {"name": name, "type": category, "address": "南京 · 地址待确认", **extra}


def _fill_with(pool: dict[str, list[dict]], seen: dict | None = None):
    """A stand-in for the category fill that records what it was told to avoid."""

    def fake_fill(destination, category, count, allowances=None, plan=None, search_hint=None, exclude=()):
        if seen is not None:
            seen[category] = list(exclude)
        return [dict(place) for place in pool.get(category, [])]

    return fake_fill


@pytest.fixture(autouse=True)
def _quiet_pricing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pricing and photos are other modules' business: keep this file about selection."""
    monkeypatch.setattr(day_plan, "estimate_trip_costs", lambda places, *args, **kwargs: places)
    monkeypatch.setattr(day_plan, "resolve_place_images", lambda places, *args, **kwargs: places)


def test_a_single_day_never_offers_a_place_another_day_already_has(monkeypatch) -> None:
    seen: dict = {}
    monkeypatch.setattr(
        day_plan,
        "_fill_category",
        _fill_with(
            {
                "attraction": [_place("夫子庙", "attraction"), _place("老门东内蒋有记", "attraction"), _place("玄武湖", "attraction")],
                "restaurant": [_place("李记汤包店", "restaurant", meal_slot="breakfast")],
            },
            seen,
        ),
    )

    places = day_plan.plan_one_day("南京", 4, ["夫子庙", "蒋有记"])

    names = [place["name"] for place in places]
    assert names == ["李记汤包店", "玄武湖"], names
    # The other days' names are handed down to the extraction, which is what stops the model
    # from listing them again in the first place.
    assert seen["attraction"] == ["夫子庙", "蒋有记"]
    assert seen["restaurant"] == ["夫子庙", "蒋有记"]


def test_a_category_that_came_back_empty_is_searched_again_from_another_angle(monkeypatch) -> None:
    """A city whose famous places are all used up must not produce a day of meals only."""
    hints: list[str | None] = []

    def fake_fill(destination, category, count, allowances=None, plan=None, search_hint=None, exclude=()):
        hints.append(search_hint)
        if category == "attraction":
            # The first search only names places the trip already has.
            if search_hint is None:
                return [_place("江户东京博物馆", "attraction"), _place("东京国立博物馆", "attraction")]
            return [_place("清澄庭园", "attraction"), _place("旧古河庭园", "attraction")]
        return [_place("早餐店", "restaurant", meal_slot="breakfast"), _place("荞麦面", "restaurant")]

    monkeypatch.setattr(day_plan, "_fill_category", fake_fill)

    places = day_plan.plan_one_day("日本东京", 4, ["江户东京博物馆", "东京国立博物馆"])

    names = [place["name"] for place in places]
    assert "清澄庭园" in names, names
    # The second pass has to ask from a different angle, or it would search for the same
    # material again and find the same used-up names.
    assert any(hint and "小众" in hint for hint in hints), hints


def test_a_day_that_is_already_filled_is_not_searched_twice(monkeypatch) -> None:
    """The extra search is a repair, not a cost every day pays."""
    hints: list[str | None] = []

    def fake_fill(destination, category, count, allowances=None, plan=None, search_hint=None, exclude=()):
        hints.append(search_hint)
        if category == "attraction":
            return [_place("甲", "attraction"), _place("乙", "attraction"), _place("丙", "attraction")]
        return [
            _place("早餐店", "restaurant", meal_slot="breakfast"),
            _place("午餐店", "restaurant"),
            _place("晚餐店", "restaurant"),
        ]

    monkeypatch.setattr(day_plan, "_fill_category", fake_fill)

    day_plan.plan_one_day("南京", 3, [])

    assert hints == [None, None], hints


def test_the_day_asks_for_two_sights_a_breakfast_and_two_meals(monkeypatch) -> None:



    monkeypatch.setattr(
        day_plan,
        "_fill_category",
        _fill_with(
            {
                "attraction": [_place("夫子庙", "attraction"), _place("玄武湖", "attraction"), _place("总统府", "attraction")],
                "restaurant": [
                    _place("李记汤包店", "restaurant", meal_slot="breakfast"),
                    _place("南京大牌檔", "restaurant"),
                    _place("绿柳居", "restaurant"),
                    _place("马祥兴菜馆", "restaurant"),
                ],
            }
        ),
    )

    places = day_plan.plan_one_day("南京", 4, [])

    assert [place["type"] for place in places].count("attraction") == day_plan.DAY_SIGHTS
    meals = [place for place in places if place["type"] == "restaurant"]
    assert len(meals) == day_plan.DAY_MEALS
    assert day_plan._is_breakfast(meals[0])
    assert not any(day_plan._is_breakfast(meal) for meal in meals[1:])


def test_a_short_day_is_reported_rather_than_filled_with_repeats(monkeypatch) -> None:
    monkeypatch.setattr(
        day_plan,
        "_fill_category",
        _fill_with(
            {
                "attraction": [_place("夫子庙", "attraction")],
                "restaurant": [_place("李记汤包店", "restaurant", meal_slot="breakfast")],
            }
        ),
    )

    places = day_plan.plan_one_day("南京", 4, [])
    short = day_plan.shortfall_notes(places)

    assert [place["name"] for place in places] == ["李记汤包店", "夫子庙"]
    assert "景点" in short and "午餐或晚餐" in short
    assert "早餐" not in short
    # Nothing was invented to fill the day out.
    assert len(places) == 2


def test_a_category_that_fails_is_a_shorter_day_not_a_failed_draft(monkeypatch) -> None:
    def exploding_fill(destination, category, count, allowances=None, plan=None, search_hint=None, exclude=()):
        if category == "restaurant":
            raise RuntimeError("the model is unavailable")
        return [_place("夫子庙", "attraction")]

    monkeypatch.setattr(day_plan, "_fill_category", exploding_fill)

    places = day_plan.plan_one_day("南京", 4, [])

    assert [place["name"] for place in places] == ["夫子庙"]


def test_the_day_is_placed_on_the_clock_the_whole_trip_plan_uses() -> None:
    existing = [
        {"id": 11, "date": "2026-05-02", "place_name": "旧的景点"},
        {"id": 12, "date": "2026-05-03", "place_name": "另一天的景点"},
    ]
    places = [
        _place("李记汤包店", "restaurant", meal_slot="breakfast"),
        _place("夫子庙", "attraction"),
        _place("南京大牌檔", "restaurant"),
        _place("玄武湖", "attraction"),
        _place("绿柳居", "restaurant"),
    ]

    operations = day_plan.day_operations(Date(2026, 5, 2), places, existing, "南京")

    removed = [operation["item_id"] for operation in operations if operation["kind"] == "remove_item"]
    assert removed == [11]  # the other day is untouched
    added = [operation for operation in operations if operation["kind"] == "add_place_and_item"]
    assert [operation["start_time"] for operation in added] == [
        time(8, 0),
        time(9, 30),
        time(12, 0),
        time(14, 0),
        time(18, 0),
    ]
    assert all(operation["date"] == Date(2026, 5, 2) for operation in added)
    assert [operation["place"]["name"] for operation in added] == [
        "李记汤包店",
        "夫子庙",
        "南京大牌檔",
        "玄武湖",
        "绿柳居",
    ]


def test_a_relative_picture_does_not_travel_with_the_saved_place() -> None:
    """A saved place requires an absolute URL, so the bundled path is left to the card."""
    operations = day_plan.day_operations(
        Date(2026, 5, 2),
        [_place("夫子庙", "attraction", image_url="/static/place_images/attraction.svg")],
        [],
        "南京",
    )

    assert operations[0]["place"]["image_url"] is None


def _trip() -> SimpleNamespace:
    return SimpleNamespace(
        id=5,
        destination="南京",
        start_date=Date(2026, 5, 1),
        end_date=Date(2026, 5, 4),
        budget=8000,
        adult_male_count=1,
        adult_female_count=1,
        child_male_count=0,
        child_female_count=0,
    )


def _fake_place(name: str) -> SimpleNamespace:
    """Only what `_item_to_state` reads."""
    return SimpleNamespace(
        id=1,
        name=name,
        type="attraction",
        address="南京",
        image_url=None,
        price_level=None,
        source_url=None,
        estimated_cost=None,
        quantity=None,
        rating=None,
        note=None,
        preferred_date=None,
    )


def _fake_item(item_id: int, day: int, name: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=item_id,
        place_id=item_id,
        date=Date(2026, 5, day),
        start_time=time(9, 0),
        end_time=time(10, 0),
        note=None,
        sort_order=0,
        place=_fake_place(name),
    )


def test_the_day_draft_replaces_that_day_and_leaves_the_others_alone(monkeypatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        adjustment_agent_service,
        "list_itinerary_items",
        lambda session, trip_id: [_fake_item(11, 2, "旧的"), _fake_item(12, 3, "别处的")],
    )
    monkeypatch.setattr(
        adjustment_agent_service,
        "list_places",
        lambda session, trip_id: [SimpleNamespace(name="已收藏的餐厅")],
    )
    def fake_plan_day(destination, trip_days, used, profile, budget):
        captured["used"] = list(used)
        return [_place("夫子庙", "attraction"), _place("李记汤包店", "restaurant", meal_slot="breakfast")]

    monkeypatch.setattr(adjustment_agent_service, "plan_one_day", fake_plan_day)

    def fake_store(session, trip, instruction, operations, **kwargs):
        captured["operations"] = operations
        captured.update(kwargs)
        return SimpleNamespace(id=1)
    monkeypatch.setattr(adjustment_agent_service, "create_pending_adjustment", fake_store)
    monkeypatch.setattr(adjustment_agent_service, "adjustment_to_read", lambda adjustment: {"id": 1, **captured})

    draft = adjustment_agent_service.create_adjustment_draft(
        None, _trip(), "极简规划这一天", False, mode="day_plan", target_date=Date(2026, 5, 2)
    )

    # What must not be offered again: the other day's place. A saved place that no day
    # schedules is not spoken for, so it may come back -- which is what lets a day use the
    # collection instead of going empty once the city's famous sights are all saved.
    assert captured["used"] == ["别处的"]
    kinds = [operation.kind for operation in captured["operations"]]
    assert kinds.count("remove_item") == 1
    assert captured["operations"][0].item_id == 11
    assert draft["daily_plan"] == [
        {
            "date": "2026-05-02",
            "items": [
                {
                    "item_id": None,
                    "place_id": None,
                    "start_time": "08:00:00",
                    "end_time": "09:00:00",
                    "place_name": "李记汤包店",
                    "note": "当地早餐",
                    "image_url": None,
                    "type": "restaurant",
                    "changed": "added",
                },
                {
                    "item_id": None,
                    "place_id": None,
                    "start_time": "09:30:00",
                    "end_time": "12:00:00",
                    "place_name": "夫子庙",
                    "note": "上午主要游览，预留交通与拍照时间",
                    "image_url": None,
                    "type": "attraction",
                    "changed": "added",
                },
            ],
        }
    ]


def test_a_day_outside_the_trip_is_refused(monkeypatch) -> None:
    monkeypatch.setattr(adjustment_agent_service, "list_itinerary_items", lambda *args: [])
    monkeypatch.setattr(adjustment_agent_service, "list_places", lambda *args: [])

    with pytest.raises(adjustment_agent_service.DayPlanDateError):
        adjustment_agent_service.create_adjustment_draft(
            None, _trip(), "极简规划这一天", False, mode="day_plan", target_date=Date(2026, 6, 1)
        )


def test_without_the_day_mode_the_ordinary_adjustment_still_runs(monkeypatch) -> None:
    """Every existing caller sends no mode, and must keep getting the adjuster."""
    calls: list[str] = []

    class FakeGraph:
        def invoke(self, state):
            calls.append("graph")
            return {"operations": [], "summary": "s", "sources": [], "events": []}

    monkeypatch.setattr(adjustment_agent_service, "build_adjuster_graph", lambda: FakeGraph())
    monkeypatch.setattr(adjustment_agent_service, "list_places", lambda *args: [])
    monkeypatch.setattr(adjustment_agent_service, "list_itinerary_items", lambda *args: [])
    monkeypatch.setattr(
        adjustment_agent_service,
        "create_pending_adjustment",
        lambda *args, **kwargs: SimpleNamespace(id=9),
    )
    monkeypatch.setattr(adjustment_agent_service, "adjustment_to_read", lambda adjustment: {"id": 9})
    monkeypatch.setattr(
        adjustment_agent_service,
        "_create_day_plan_draft",
        lambda *args, **kwargs: calls.append("day_plan"),
    )

    adjustment_agent_service.create_adjustment_draft(None, _trip(), "把第三天换成博物馆", False)

    assert calls == ["graph"]
