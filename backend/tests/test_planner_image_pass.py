"""The picture pass runs beside the plan instead of in front of it.

Measured on a four-day plan, searching and verifying a photo for every place costs 15-25 s.
Nothing that follows the recommendation node needs a picture — the conflict check, the
budget and the written summary all work from names and prices — so the pass now runs on its
own thread and the plan is composed as soon as that work is done.

The tests below fix the two things that matter: the node must genuinely return while the
photos are still being searched for, and the plan must come out with exactly the same
photos as before, including the bundled placeholder when a pass never finishes.
"""
import threading

import pytest

from app.agents.tools.budget import estimate_budget
from app.agents.workflows import planner


def _trip() -> dict:
    return {
        "id": 3,
        "destination": "南京",
        "start_date": "2026-05-01",
        "end_date": "2026-05-04",
        "budget": 8000,
    }


def _pool() -> list[dict]:
    return [
        {"name": "金陵饭店", "type": "hotel", "unit_price": 600},
        {"name": "南京大牌檔(德基广场店)", "type": "restaurant", "unit_price": 90},
        {"name": "夫子庙", "type": "attraction", "unit_price": 0},
    ]


def _stub_the_pipeline(monkeypatch: pytest.MonkeyPatch, items: list[dict]) -> None:
    """Every expensive step except the picture pass: this is about ordering, not content."""
    monkeypatch.setattr(planner, "budget_allowances", lambda *args, **kwargs: {})
    monkeypatch.setattr(planner, "extract_recommendations", lambda *args, **kwargs: (list(items), "completed"))
    monkeypatch.setattr(planner, "ensure_category_coverage", lambda destination, found, **kwargs: found)
    monkeypatch.setattr(planner, "enrich_prices_with_search", lambda found, destination: found)
    monkeypatch.setattr(planner, "enrich_place_facts", lambda found, destination: found)
    monkeypatch.setattr(planner, "estimate_trip_costs", lambda found, *args, **kwargs: found)
    monkeypatch.setattr(planner, "ensure_price_spread", lambda *args, **kwargs: (list(items), {}))
    monkeypatch.setattr(planner, "_generated_daily_plan", lambda state, found: [])


def _blocking_pictures(monkeypatch: pytest.MonkeyPatch) -> tuple[threading.Event, threading.Event]:
    """A picture pass that parks until the test lets it go, respecting the hotel rule."""
    started = threading.Event()
    release = threading.Event()

    def slow_pictures(places, destination, *, skip_categories=frozenset(), max_lookups=40, **kwargs):
        if max_lookups:
            started.set()
            release.wait(timeout=5)
        # Mirrors what `resolve_place_images` guarantees: every place that may have a picture
        # gets both a URL and the bundled substitute, and a skipped category gets neither.
        def pictured(place: dict) -> dict:
            if place["type"] in skip_categories:
                return dict(place, image_url=None, fallback_image_url=None)
            placeholder = f"/static/place_images/{place['type']}.svg"
            return dict(place, image_url=placeholder, fallback_image_url=placeholder)

        return [pictured(place) for place in places]

    monkeypatch.setattr(planner, "resolve_place_images", slow_pictures)
    return started, release


def _state() -> dict:
    return {"trip": _trip(), "itinerary": [], "places": [], "web_results": []}


def test_the_picture_pass_no_longer_holds_up_the_recommendation_node(monkeypatch) -> None:
    items = _pool()
    _stub_the_pipeline(monkeypatch, items)
    started, release = _blocking_pictures(monkeypatch)
    slot: dict = {}

    state = planner._recommend_places(_state(), image_slot=slot)

    assert started.wait(timeout=5), "the picture pass never started"
    assert slot.get("future") is not None
    # The node is finished while the photos are still being searched for: that is the point.
    assert not release.is_set()
    assert [place["name"] for place in state["recommended_places"]] == [place["name"] for place in items]
    assert not any(place.get("image_url") for place in state["recommended_places"])
    release.set()


def test_the_composed_plan_carries_the_photos_that_were_searched_in_parallel(monkeypatch) -> None:
    items = _pool()
    _stub_the_pipeline(monkeypatch, items)
    started, release = _blocking_pictures(monkeypatch)
    slot: dict = {}

    state = planner._recommend_places(_state(), image_slot=slot)
    assert started.wait(timeout=5)
    release.set()

    # The analysis and the summary run here, and only then is the plan composed.
    composed = planner._compose(
        {**state, "trip": _trip(), "final_summary": "总结", "budget": estimate_budget([], 4, 8000, None)},
        image_slot=slot,
    )
    places = composed["response"]["place_recommendations"]

    assert [place["name"] for place in places] == [place["name"] for place in items]
    assert [place["image_url"] for place in places] == [
        None,  # hotels carry a rating and a price instead of a picture
        "/static/place_images/restaurant.svg",
        "/static/place_images/attraction.svg",
    ]
    assert [place["fallback_image_url"] for place in places] == [
        None,
        "/static/place_images/restaurant.svg",
        "/static/place_images/attraction.svg",
    ]


def test_a_picture_pass_that_never_finishes_still_leaves_placeholders(monkeypatch, caplog) -> None:
    """Waiting forever must never be the outcome: a plan without pictures would be worse."""
    items = _pool()
    _stub_the_pipeline(monkeypatch, items)
    started, release = _blocking_pictures(monkeypatch)
    monkeypatch.setattr(planner, "IMAGE_JOIN_TIMEOUT", 0.05)
    slot: dict = {}

    state = planner._recommend_places(_state(), image_slot=slot)
    assert started.wait(timeout=5)

    with caplog.at_level("WARNING"):
        joined = planner._join_place_images(state, slot)

    assert [place["image_url"] for place in joined["recommended_places"]] == [
        None,
        "/static/place_images/restaurant.svg",
        "/static/place_images/attraction.svg",
    ]
    assert any("picture pass" in record.message for record in caplog.records)
    release.set()


def test_the_graph_still_builds_with_the_picture_slot() -> None:
    """A graph node given a ``partial`` must still be accepted by LangGraph."""
    graph = planner.build_planner_graph()

    assert graph is not None
