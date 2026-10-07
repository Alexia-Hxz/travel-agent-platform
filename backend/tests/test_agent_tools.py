from datetime import date, time

from app.agents.tools.budget import estimate_budget
from app.agents.tools.conflicts import detect_conflicts
from app.agents.tools.web_search import search_web


def test_detect_conflicts_reports_overlapping_itinerary_items() -> None:
    items = [
        {"date": date(2026, 10, 1), "start_time": time(9), "end_time": time(11), "place_name": "Museum"},
        {"date": date(2026, 10, 1), "start_time": time(10), "end_time": time(12), "place_name": "Tower"},
    ]

    conflicts = detect_conflicts(items)

    assert len(conflicts) == 1
    assert conflicts[0].first_place == "Museum"
    assert conflicts[0].second_place == "Tower"


def test_estimate_budget_uses_price_levels_and_trip_budget() -> None:
    estimate = estimate_budget(
        places=[{"type": "hotel", "price_level": 4}, {"type": "restaurant", "price_level": 2}],
        trip_days=3,
        trip_budget=3200,
    )

    assert estimate.total_estimate > 0
    assert estimate.trip_budget == 3200
    assert estimate.remaining_budget == 3200 - estimate.total_estimate
    assert estimate.itemized_summary.endswith(f" = {estimate.total_estimate}元")
    assert "住宿" in estimate.itemized_summary or "地点" in estimate.itemized_summary


def test_estimate_budget_marks_zero_cost_items_as_free() -> None:
    estimate = estimate_budget(
        places=[
            {"type": "attraction", "name": "城市公园", "price_level": 0},
            {"type": "restaurant", "name": "免费试吃", "price_level": 0},
        ],
        trip_days=1,
        trip_budget=None,
    )

    assert "城市公园（免费）" in estimate.itemized_summary
    assert "免费试吃（免费）" in estimate.itemized_summary
    assert estimate.itemized_summary.endswith(" = 0元")


def test_estimate_budget_uses_type_defaults_when_price_level_is_missing() -> None:
    estimate = estimate_budget(
        places=[
            {"type": "hotel", "price_level": None},
            {"type": "restaurant", "price_level": None},
            {"type": "attraction", "price_level": None},
        ],
        trip_days=3,
        trip_budget=5000,
    )

    assert estimate.total_estimate == 2140
    assert estimate.remaining_budget == 2860


def test_a_recommendation_only_plan_does_not_precharge_rooms_or_meals() -> None:
    """The reported bug: the baseline already claimed a room and three meals a day.

    On a 5-day trip for two that was 1200 CNY of accommodation plus 800 of food that nobody
    had chosen, and ticking a hotel then ADDED its real price on top of the guess it was
    meant to replace. Only local transport is charged now.
    """
    estimate = estimate_budget(places=[], trip_days=5, trip_budget=8000, traveler_profile={"total_people": 2})

    assert estimate.total_estimate == 300  # 60 CNY/day of local transport, nothing else
    assert estimate.remaining_budget == 7700
    assert [line["category"] for line in estimate.breakdown] == ["transport"]
    # The typical amounts are still reported, but explicitly outside the total.
    excluded = {line["category"]: line["amount"] for line in estimate.excluded_allowances or []}
    assert excluded == {"hotel": 1200, "restaurant": 800}
    assert all("未计入总额" in line["name"] for line in estimate.excluded_allowances or [])
    assert "未计入" in estimate.note


def test_the_itemized_summary_adds_up_to_the_total() -> None:
    """Reference lines must never be part of the addition the summary shows."""
    estimate = estimate_budget(places=[], trip_days=4, trip_budget=None, traveler_profile={"total_people": 1})

    assert estimate.itemized_summary == f"市内交通基础估算（{estimate.total_estimate}元） = {estimate.total_estimate}元"
    assert "住宿" not in estimate.itemized_summary
    assert "餐饮" not in estimate.itemized_summary


def test_a_ticked_hotel_replaces_the_reference_instead_of_stacking_on_it() -> None:
    """Ticking a hotel is what used to double-count: guess plus real price."""
    empty = estimate_budget(places=[], trip_days=5, trip_budget=8000, traveler_profile={"total_people": 2})
    with_hotel = estimate_budget(
        places=[{"type": "hotel", "name": "南京金鹰国际酒店", "price_level": None, "estimated_cost": None}],
        trip_days=5,
        trip_budget=8000,
        traveler_profile={"total_people": 2},
    )

    # The saved-place path charges the hotel once, and reports no excluded allowance.
    assert with_hotel.total_estimate == 3 * 260 * 4
    assert with_hotel.excluded_allowances == []
    assert with_hotel.total_estimate != empty.total_estimate + 1200


def test_a_saved_place_is_charged_the_price_it_was_saved_with() -> None:
    """The night/meal count the user set has to survive into the saved-place budget.

    A place saved from the recommendations carries the price shown on its card -- which is
    already multiplied by how many nights or meals were asked for. Inventing a level-based
    figure instead would quietly disagree with the card the user ticked.
    """
    estimate = estimate_budget(
        places=[
            {"type": "hotel", "name": "金陵饭店", "price_level": None, "quoted_cost": 1860, "quantity": 3},
            {"type": "restaurant", "name": "南京大牌檔", "price_level": None, "quoted_cost": 480, "quantity": 3},
        ],
        trip_days=5,
        trip_budget=8000,
        traveler_profile={"total_people": 2},
    )

    assert estimate.total_estimate == 1860 + 480
    assert [line["source"] for line in estimate.breakdown] == ["quoted", "quoted"]
    assert "金陵饭店（1860元）" in estimate.itemized_summary
    # Nothing is precharged for accommodation or meals beyond the places actually saved.
    assert estimate.excluded_allowances == []


def test_a_place_without_a_quoted_price_still_uses_its_price_level() -> None:
    """The fallback must not be lost: places added by hand carry no quoted price."""
    estimate = estimate_budget(
        places=[
            {"type": "hotel", "name": "手填酒店", "price_level": 3},
            {"type": "hotel", "name": "推荐酒店", "price_level": 3, "estimated_cost": 600},
        ],
        trip_days=5,
        trip_budget=8000,
        traveler_profile={"total_people": 2},
    )

    # 3 * 260 * 4 nights by level, then the quoted price verbatim.
    assert estimate.total_estimate == 3120 + 600
    assert estimate.breakdown[0]["source"] == "level"
    assert estimate.breakdown[1]["source"] == "quoted"


def test_search_web_reports_unavailable_live_search_instead_of_falling_back(monkeypatch) -> None:
    """There is no second provider any more.

    The metered search API that used to back this function was removed after it answered
    HTTP 432 to every query: a fallback that never answers is not a fallback, it is a
    slower route to the same empty result. With the live search unavailable the call now
    reports that plainly, and the plan continues without web material.
    """
    from app.agents.tools import web_search

    monkeypatch.setattr(web_search.dashscope_search, "search_available", lambda: False)
    # No HTTP client is even imported any more: there is nothing left to call.
    assert not hasattr(web_search, "httpx")

    results, status = search_web("乐园", "上海", limit=1)

    assert results == []
    assert status == "fallback: live search is not configured"


def test_search_web_returns_the_live_answer_and_no_image_urls(monkeypatch) -> None:
    """The live search answers in prose: it has never supplied a picture, which is why
    pictures come from the photo library instead."""
    from app.agents.tools import web_search

    monkeypatch.setattr(web_search.dashscope_search, "search_available", lambda: True)
    monkeypatch.setattr(web_search.dashscope_search, "grounded_answer", lambda prompt: "上海迪士尼乐园，门票 475 元。")

    results, status = search_web("乐园", "上海", limit=1)

    assert status == "completed"
    assert results[0]["source_type"] == "web_answer"
    assert results[0]["image_url"] is None
    assert "475" in results[0]["snippet"]
