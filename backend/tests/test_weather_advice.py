"""A wet forecast must change what the app says, not just what it displays.

`weather_service` already computed `indoor_priority` per day, but nothing consumed
it: a rainy day rendered the same plan with no warning, so the user had to spot the
mismatch themselves.
"""

from app.agents.tools.weather_advice import (
    INDOOR,
    OUTDOOR,
    UNKNOWN,
    build_weather_advice,
    classify_place,
    weather_adjustment_request,
)


def _weather(days: list[dict]) -> dict:
    return {"destination": "南京", "status": "completed", "days": days}


def _wet_day(day_date: str = "2026-10-02", precipitation: int = 80) -> dict:
    return {
        "date": day_date,
        "condition": "降雨或恶劣天气",
        "precipitation_probability": precipitation,
        "indoor_priority": True,
    }


def _dry_day(day_date: str = "2026-10-01") -> dict:
    return {
        "date": day_date,
        "condition": "适合户外活动",
        "precipitation_probability": 10,
        "indoor_priority": False,
    }


def test_an_outdoor_place_is_recognised_by_its_name() -> None:
    assert classify_place("中山陵") == OUTDOOR
    assert classify_place("玄武湖公园") == OUTDOOR
    assert classify_place("鸡鸣寺") == OUTDOOR
    assert classify_place("明孝陵景区") == OUTDOOR


def test_an_indoor_place_wins_over_an_outdoor_keyword_in_the_same_name() -> None:
    # "西湖美术馆" contains both "湖" (outdoor) and "美术馆" (indoor): it is indoor.
    assert classify_place("西湖美术馆") == INDOOR
    assert classify_place("南京博物院") == INDOOR
    assert classify_place("先锋书店") == INDOOR


def test_roofed_place_types_are_indoor_even_without_a_keyword() -> None:
    assert classify_place("涵田城市酒店", "hotel") == INDOOR
    assert classify_place("大牌檔", "restaurant") == INDOOR


def test_an_unrecognised_attraction_is_left_undecided() -> None:
    # "总统府" carries no indoor or outdoor hint, so the app says nothing rather than
    # guessing wrong; the advice only ever lists items it can actually justify.
    assert classify_place("总统府") == UNKNOWN


def test_a_wet_day_flags_the_outdoor_items_that_are_scheduled() -> None:
    advice = build_weather_advice(
        "南京",
        _weather([_wet_day()]),
        [{"date": "2026-10-02", "items": [{"place_name": "中山陵"}, {"place_name": "南京博物院"}]}],
        [{"name": "中山陵", "type": "attraction"}, {"name": "南京博物院", "type": "attraction"}],
    )

    assert len(advice) == 1
    entry = advice[0]
    assert entry["date"] == "2026-10-02"
    assert entry["outdoor_items"] == ["中山陵"]
    assert entry["indoor_items"] == ["南京博物院"]
    assert "中山陵" in entry["message"]
    assert "降水概率 80%" in entry["message"]
    # The suggestion names the indoor alternative that is already planned.
    assert "南京博物院" in entry["suggestion"]


def test_a_dry_day_produces_no_advice() -> None:
    advice = build_weather_advice(
        "南京",
        _weather([_dry_day()]),
        [{"date": "2026-10-01", "items": [{"place_name": "中山陵"}]}],
        [{"name": "中山陵", "type": "attraction"}],
    )

    # An empty list means "nothing to worry about", not "no information".
    assert advice == []


def test_an_unavailable_forecast_produces_no_advice() -> None:
    assert build_weather_advice("南京", {"status": "unavailable", "days": []}, [], []) == []
    assert build_weather_advice("南京", None, [], []) == []


def test_a_wet_day_with_nothing_scheduled_still_warns() -> None:
    advice = build_weather_advice("南京", _weather([_wet_day()]), [], [])

    assert len(advice) == 1
    assert advice[0]["outdoor_items"] == []
    assert "还没有安排行程" in advice[0]["message"]


def test_a_wet_day_of_indoor_plans_is_reassuring_rather_than_alarming() -> None:
    advice = build_weather_advice(
        "南京",
        _weather([_wet_day()]),
        [{"date": "2026-10-02", "items": [{"place_name": "南京博物院"}]}],
        [{"name": "南京博物院", "type": "attraction"}],
    )

    assert len(advice) == 1
    assert advice[0]["outdoor_items"] == []
    assert "都在室内" in advice[0]["message"]


def test_the_scheduled_day_is_read_from_the_real_itinerary_when_present() -> None:
    """The advice must describe the actual plan, not the proposed one."""
    from app.agents.workflows.planner import _scheduled_days

    state = {
        "itinerary": [
            {"date": __import__("datetime").date(2026, 10, 2), "place_name": "中山陵"},
            {"date": __import__("datetime").date(2026, 10, 2), "place_name": "南京博物院"},
            {"date": __import__("datetime").date(2026, 10, 3), "place_name": "夫子庙"},
        ],
        "generated_daily_plan": [{"date": "2026-10-02", "items": [{"place_name": "不该出现的建议"}]}],
    }

    days = _scheduled_days(state)

    assert days[0]["date"] == "2026-10-02"
    assert [item["place_name"] for item in days[0]["items"]] == ["中山陵", "南京博物院"]
    assert days[1]["date"] == "2026-10-03"


def test_the_proposed_day_plan_is_used_when_there_is_no_itinerary() -> None:
    from app.agents.workflows.planner import _scheduled_days

    state = {"itinerary": [], "generated_daily_plan": [{"date": "2026-10-02", "items": [{"place_name": "中山陵"}]}]}

    assert _scheduled_days(state) == [{"date": "2026-10-02", "items": [{"place_name": "中山陵"}]}]


def test_the_adjustment_request_is_ready_to_send() -> None:
    advice = build_weather_advice(
        "南京",
        _weather([_wet_day()]),
        [{"date": "2026-10-02", "items": [{"place_name": "中山陵"}]}],
        [{"name": "中山陵", "type": "attraction"}],
    )

    request = weather_adjustment_request("南京", advice)

    assert "南京" in request
    assert "中山陵" in request
    assert "2026-10-02" in request
    # An empty advice list produces nothing to send.
    assert weather_adjustment_request("南京", []) == ""


def test_advice_reaches_the_plan_response(monkeypatch) -> None:
    from datetime import date, timedelta

    from fastapi.testclient import TestClient

    from app.agents.workflows import planner
    from app.main import app

    # Choose dates inside the forecast window so the real service is not consulted.
    start = date.today() + timedelta(days=1)
    end = start + timedelta(days=2)
    monkeypatch.setattr(
        planner,
        "get_weather_forecast",
        lambda destination, start_date, end_date: _weather(
            [
                dict(_wet_day(start.isoformat())),
                dict(_dry_day((start + timedelta(days=1)).isoformat())),
            ]
        ),
    )

    with TestClient(app) as client:
        trip = client.post(
            "/api/v1/trips",
            json={
                "title": "weather advice check",
                "destination": "南京",
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "budget": 5000,
            },
        ).json()
        client.post(
            f"/api/v1/trips/{trip['id']}/places",
            json={
                "name": "中山陵",
                "type": "attraction",
                "address": "南京市玄武区",
                "latitude": None,
                "longitude": None,
                "image_url": None,
                "rating": 4.7,
                "price_level": 1,
                "opening_hours": "08:30-17:00",
                "source_url": None,
            },
        )
        client.post(
            f"/api/v1/trips/{trip['id']}/itinerary-items",
            json={
                "date": start.isoformat(),
                "start_time": "09:00:00",
                "end_time": "11:00:00",
                "place_id": client.get(f"/api/v1/trips/{trip['id']}/places").json()[0]["id"],
                "note": None,
                "sort_order": 0,
            },
        )
        body = client.post(
            f"/api/v1/trips/{trip['id']}/plan",
            json={"query": "生成行程介绍与总结", "include_web": False},
        ).json()

    assert len(body["weather_advice"]) == 1
    entry = body["weather_advice"][0]
    assert entry["date"] == start.isoformat()
    assert entry["outdoor_items"] == ["中山陵"]
    # The response carries a ready-to-send request for the adjustment agent.
    assert "中山陵" in body["weather_adjustment_request"]
    # And the local summary fallback states the risk rather than hiding it.
    assert "中山陵" in body["final_summary"]
