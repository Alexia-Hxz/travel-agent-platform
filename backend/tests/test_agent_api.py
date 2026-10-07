from datetime import date, time
from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app


def _trip_payload() -> dict:
    return {
        "title": f"agent context test {uuid4().hex[:8]}",
        "destination": "上海",
        "start_date": "2026-10-01",
        "end_date": "2026-10-02",
        "budget": 2000,
    }


def _patch_extracted_recommendations(monkeypatch, items: list[dict]) -> None:
    """Replace model extraction with a fixed, deterministic recommendation list."""
    from app.agents.workflows import planner

    monkeypatch.setattr(
        planner,
        "extract_recommendations",
        lambda destination, web_results, **kwargs: (
            [{**item, "address": destination} for item in items],
            "completed",
        ),
    )


def test_plan_job_endpoints_return_polling_snapshots(monkeypatch) -> None:
    from app.api.v1.endpoints import agent as agent_endpoint

    created = {
        "job_id": "job-123",
        "trip_id": 0,
        "status": "queued",
        "phase": "queued",
        "progress": 0,
        "events": [],
        "place_recommendations": [],
        "result": None,
        "error": None,
    }

    def start(session, trip, *args, **kwargs):
        created["trip_id"] = trip.id
        return created

    monkeypatch.setattr(agent_endpoint, "start_plan_job", start)
    monkeypatch.setattr(
        agent_endpoint,
        "get_plan_job",
        lambda job_id, trip_id: {**created, "status": "running", "phase": "search", "progress": 18}
        if job_id == "job-123" and trip_id == created["trip_id"]
        else None,
    )

    with TestClient(app) as client:
        trip = client.post("/api/v1/trips", json=_trip_payload()).json()
        started = client.post(
            f"/api/v1/trips/{trip['id']}/plan-jobs",
            json={"query": "东京五日游", "include_web": True},
        )
        assert started.status_code == 202
        assert started.json()["job_id"] == "job-123"

        status_response = client.get(f"/api/v1/trips/{trip['id']}/plan-jobs/job-123")
        assert status_response.status_code == 200
        assert status_response.json()["phase"] == "search"
        assert status_response.json()["progress"] == 18

        missing = client.get(f"/api/v1/trips/{trip['id']}/plan-jobs/other")
        assert missing.status_code == 404


def test_plan_endpoint_returns_structured_deterministic_plan() -> None:
    with TestClient(app) as client:
        trip = client.post(
            "/api/v1/trips",
            json={
                "title": f"agent test {uuid4().hex[:8]}",
                "destination": "AgentTestCity",
                "start_date": "2026-10-01",
                "end_date": "2026-10-02",
                "budget": 2000,
            },
        ).json()
        place = client.post(
            f"/api/v1/trips/{trip['id']}/places",
            json={
                "name": "Agent Museum",
                "type": "attraction",
                "address": "Central Road",
                "latitude": None,
                "longitude": None,
                "image_url": None,
                "rating": 4.5,
                "price_level": 2,
                "opening_hours": "09:00-17:00",
                "source_url": "https://example.com/museum",
            },
        ).json()
        client.post(
            f"/api/v1/trips/{trip['id']}/itinerary-items",
            json={
                "date": "2026-10-01",
                "start_time": "09:00:00",
                "end_time": "11:00:00",
                "place_id": place["id"],
                "note": "morning visit",
                "sort_order": 0,
            },
        )
        response = client.post(
            f"/api/v1/trips/{trip['id']}/plan",
            json={"query": "安排一个轻松的博物馆行程", "include_web": False},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["trip_id"] == trip["id"]
    assert body["daily_plan"][0]["items"][0]["place_name"] == "Agent Museum"
    assert body["sources"][0]["url"] == "https://example.com/museum"
    assert any(event["tool"] == "conflict_checker" for event in body["events"])


def test_plan_daily_item_keeps_saved_place_image() -> None:
    with TestClient(app) as client:
        trip = client.post(
            "/api/v1/trips",
            json={
                "title": f"image test {uuid4().hex[:8]}",
                "destination": "上海",
                "start_date": "2026-10-01",
                "end_date": "2026-10-01",
                "budget": 2000,
            },
        ).json()
        place = client.post(
            f"/api/v1/trips/{trip['id']}/places",
            json={
                "name": "Image Museum",
                "type": "attraction",
                "address": "Central Road",
                "image_url": "https://images.example.com/museum.jpg",
            },
        ).json()
        client.post(
            f"/api/v1/trips/{trip['id']}/itinerary-items",
            json={
                "date": "2026-10-01",
                "start_time": "09:00:00",
                "end_time": "11:00:00",
                "place_id": place["id"],
                "note": "visit",
                "sort_order": 0,
            },
        )
        response = client.post(f"/api/v1/trips/{trip['id']}/plan", json={"query": "生成规划", "include_web": False})

    assert response.status_code == 200
    assert response.json()["daily_plan"][0]["items"][0]["image_url"] == "https://images.example.com/museum.jpg"


def test_plan_includes_traveler_context_and_weather(monkeypatch) -> None:
    from app.agents.workflows import planner

    monkeypatch.setattr(
        planner,
        "get_weather_forecast",
        lambda destination, start_date, end_date: {
            "destination": destination,
            "days": [{"date": start_date, "condition": "暴雨", "indoor_priority": True}],
            "status": "completed",
        },
    )
    with TestClient(app) as client:
        trip = client.post(
            "/api/v1/trips",
            json={**_trip_payload(), "adult_male_count": 1, "adult_female_count": 1, "child_male_count": 1, "child_female_count": 1},
        ).json()
        response = client.post(f"/api/v1/trips/{trip['id']}/plan", json={"query": "生成规划", "include_web": False})

    assert response.status_code == 200
    body = response.json()
    assert body["traveler_profile"]["total_people"] == 4
    assert body["weather"]["days"][0]["indoor_priority"] is True


def test_plan_endpoint_returns_a_final_summary(monkeypatch) -> None:
    from app.agents.workflows import planner

    monkeypatch.setattr(
        planner,
        "generate_final_summary",
        lambda _context: ("这是一份综合后的最终旅行建议。", "completed"),
        raising=False,
    )
    with TestClient(app) as client:
        trip = client.post(
            "/api/v1/trips",
            json={
                "title": f"summary test {uuid4().hex[:8]}",
                "destination": "上海",
                "start_date": "2026-10-01",
                "end_date": "2026-10-02",
                "budget": 2000,
            },
        ).json()
        response = client.post(
            f"/api/v1/trips/{trip['id']}/plan",
            json={"query": "给我一份最终旅行建议", "include_web": False},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["final_summary"] == "这是一份综合后的最终旅行建议。"
    assert any(event["tool"] == "final_summary_generator" for event in body["events"])


def test_route_question_uses_saved_hotel_as_the_tavily_origin(monkeypatch) -> None:
    from app.agents.workflows import planner

    captured: dict[str, str] = {}

    def fake_search(query: str, destination: str) -> tuple[list[dict[str, object]], str]:
        captured["query"] = query
        captured["destination"] = destination
        return [
            {
                # The live search is the only source left: "web_answer" is what it returns.
                "source_type": "web_answer",
                "title": "交通检索摘要",
                "url": None,
                "snippet": "优先乘坐公共交通。",
                "score": None,
            }
        ], "completed"

    monkeypatch.setattr(planner, "search_web", fake_search)
    with TestClient(app) as client:
        trip = client.post(
            "/api/v1/trips",
            json={
                "title": f"route test {uuid4().hex[:8]}",
                "destination": "上海",
                "start_date": "2026-10-02",
                "end_date": "2026-10-06",
                "budget": 10000,
            },
        ).json()
        client.post(
            f"/api/v1/trips/{trip['id']}/places",
            json={
                "name": "初心启程酒店",
                "type": "hotel",
                "address": "浦东新区界龙大道266号4幢1层109号",
                "rating": 4.7,
            },
        )

        response = client.post(
            f"/api/v1/trips/{trip['id']}/plan",
            json={"query": "我这个酒店怎么到迪士尼性价比最高", "include_web": True},
        )

    assert response.status_code == 200
    assert "初心启程酒店" in captured["query"]
    assert "界龙大道266号" in captured["query"]
    assert "迪士尼" in captured["query"]
    assert response.json()["transport_advice"] == {
        "origin": "初心启程酒店（浦东新区界龙大道266号4幢1层109号）",
        "destination": "迪士尼",
        "recommendation": "优先乘坐公共交通。",
    }


def test_plan_from_web_recommendations_groups_places_with_images(monkeypatch) -> None:
    from app.agents.workflows import planner

    monkeypatch.setattr(
        planner,
        "search_web",
        lambda query, destination: (
            [
                {"source_type": "tavily", "title": "南京性价比酒店榜单", "url": "https://example.com/hotel", "snippet": "金陵饭店房价约600元", "score": 0.9},
                {"source_type": "tavily", "title": "夫子庙夜游攻略", "url": "https://example.com/attraction", "snippet": "夫子庙适合夜游", "score": 0.8},
                {"source_type": "tavily", "title": "南京特色美食盘点", "url": "https://example.com/food", "snippet": "南京大牌檔人均约80元", "score": 0.7},
            ],
            "completed",
        ),
    )
    _patch_extracted_recommendations(
        monkeypatch,
        [
            {"name": "金陵饭店", "type": "hotel", "price_level": 3, "image_url": None, "reason": "位置便利", "price_hint": "约600元", "latitude": 32.0416, "longitude": 118.7785},
            {"name": "南京大牌檔", "type": "restaurant", "price_level": 2, "image_url": None, "reason": "本地菜", "price_hint": "人均80元", "latitude": 32.0401, "longitude": 118.7843},
            {"name": "夫子庙", "type": "attraction", "price_level": 1, "image_url": None, "reason": "适合夜游", "price_hint": None, "latitude": 32.0224, "longitude": 118.7886},
        ],
    )
    with TestClient(app) as client:
        trip = client.post(
            "/api/v1/trips",
            json={
                "title": f"web recommendation test {uuid4().hex[:8]}",
                "destination": "南京",
                "start_date": "2026-10-10",
                "end_date": "2026-10-13",
                "budget": 8000,
                "adult_male_count": 1,
                "adult_female_count": 1,
            },
        ).json()
        response = client.post(
            f"/api/v1/trips/{trip['id']}/plan",
            json={"query": "我准备在南京玩4天3晚，你给我第一天一些推荐，最好有具体时间规划，顺便再给我推荐几个性价比高的酒店", "include_web": True},
        )

    assert response.status_code == 200
    body = response.json()
    recommendations = body["place_recommendations"]
    assert [item["name"] for item in recommendations] == ["金陵饭店", "南京大牌檔", "夫子庙"]
    assert [item["category"] for item in recommendations] == ["hotel", "restaurant", "attraction"]
    # The hotel is deliberately left without a picture: a stock photo of a different hotel
    # says nothing, so its card shows a rating and a price instead. Everything else still
    # falls back to its bundled placeholder when no real photo can be found.
    hotel = recommendations[0]
    assert hotel["image_url"] is None
    assert hotel["fallback_image_url"] is None
    missing = {
        item["name"]: item["image_url"]
        for item in recommendations
        if item["category"] != "hotel"
        and not str(item["image_url"] or "").startswith("/static/place_images/")
    }
    assert missing == {}
    assert all(item["estimated_cost"] for item in recommendations)
    assert all(item["cost_explanation"] for item in recommendations)
    # The rough position has to reach the browser, where the workspace map sketches the day's
    # route from it. `compose_plan` builds each recommendation from an explicit whitelist, so a
    # field that is not repeated there is dropped silently -- which is exactly what happened to
    # `image_credit` before. This asserts the coordinates survive that whitelist.
    assert [item["latitude"] for item in recommendations] == [32.0416, 32.0401, 32.0224]
    assert [item["longitude"] for item in recommendations] == [118.7785, 118.7843, 118.7886]
    assert body["budget"]["total_estimate"] > 0
    assert body["daily_plan"]
    items = body["daily_plan"][0]["items"]
    assert items
    # A day entry for a hotel carries no picture either, for the same reason its card does
    # not: the rule is about hotels, not about cards.
    pictureless = {item["name"] for item in recommendations if item["category"] == "hotel"}
    assert all(item["image_url"] or item["place_name"] in pictureless for item in items)
    assert "没有每日行程项" not in body["final_summary"]
    assert body["budget"]["breakdown"]
    assert "daily_estimate" in body["budget"]
    # A search-result headline must never be charged as if it were a booking.
    charged = {line["name"] for line in body["budget"]["breakdown"]}
    assert "南京性价比酒店榜单" not in charged


def test_plan_daily_entries_include_weather_for_decision_making(monkeypatch) -> None:
    from app.agents.workflows import planner

    monkeypatch.setattr(
        planner,
        "get_weather_forecast",
        lambda destination, start_date, end_date: {
            "destination": destination,
            "days": [{"date": start_date, "condition": "暴雨", "indoor_priority": True, "precipitation_probability": 90}],
            "status": "completed",
        },
    )
    with TestClient(app) as client:
        trip = client.post(
            "/api/v1/trips",
            json={"title": f"weather plan {uuid4().hex[:8]}", "destination": "南京", "start_date": "2026-10-10", "end_date": "2026-10-10", "budget": 3000},
        ).json()
        response = client.post(f"/api/v1/trips/{trip['id']}/plan", json={"query": "生成规划", "include_web": False})
    assert response.status_code == 200
    assert response.json()["weather"]["days"][0]["condition"] == "暴雨"


def test_plan_keeps_recommendations_that_have_no_search_photo(monkeypatch) -> None:
    from app.agents.workflows import planner

    monkeypatch.setattr(
        planner,
        "search_web",
        lambda query, destination: (
            [
                {"source_type": "tavily", "title": "有图景点", "url": "https://example.com/a", "snippet": "景点", "image_url": "https://images.example.com/a.jpg"},
                {"source_type": "tavily", "title": "无图景点", "url": "https://example.com/b", "snippet": "景点", "image_url": None},
            ],
            "completed",
        ),
    )
    _patch_extracted_recommendations(
        monkeypatch,
        [{"name": "无图景点", "type": "attraction", "price_level": 1, "image_url": None, "reason": None, "price_hint": None}],
    )
    with TestClient(app) as client:
        trip = client.post(
            "/api/v1/trips",
            json={"title": f"image required {uuid4().hex[:8]}", "destination": "南京", "start_date": "2026-10-10", "end_date": "2026-10-10", "budget": 3000},
        ).json()
        response = client.post(f"/api/v1/trips/{trip['id']}/plan", json={"query": "推荐景点并规划第一天", "include_web": True})

    assert response.status_code == 200
    body = response.json()
    assert [item["name"] for item in body["place_recommendations"]] == ["无图景点"]
    assert body["place_recommendations"][0]["image_url"] == "/static/place_images/attraction.svg"
    assert all(item["image_url"] for day in body["daily_plan"] for item in day["items"])


def test_plan_gives_a_saved_place_without_a_photo_an_image() -> None:
    with TestClient(app) as client:
        trip = client.post(
            "/api/v1/trips",
            json={"title": f"saved image {uuid4().hex[:8]}", "destination": "上海", "start_date": "2026-10-01", "end_date": "2026-10-01", "budget": 2000},
        ).json()
        place = client.post(
            f"/api/v1/trips/{trip['id']}/places",
            json={"name": "无图博物馆", "type": "attraction", "address": "Central Road"},
        ).json()
        client.post(
            f"/api/v1/trips/{trip['id']}/itinerary-items",
            json={"date": "2026-10-01", "start_time": "09:00:00", "end_time": "11:00:00", "place_id": place["id"], "note": "visit", "sort_order": 0},
        )
        response = client.post(f"/api/v1/trips/{trip['id']}/plan", json={"query": "生成规划", "include_web": False})

    assert response.status_code == 200
    item = response.json()["daily_plan"][0]["items"][0]
    assert item["place_name"] == "无图博物馆"
    assert item["image_url"] == "/static/place_images/attraction.svg"


def test_place_placeholder_images_are_served() -> None:
    with TestClient(app) as client:
        for category in ("hotel", "restaurant", "attraction", "generic"):
            response = client.get(f"/static/place_images/{category}.svg")
            assert response.status_code == 200
            assert "svg" in response.headers["content-type"]


def test_plan_returns_a_short_budget_summary_for_final_result() -> None:
    with TestClient(app) as client:
        trip = client.post(
            "/api/v1/trips",
            json={"title": f"budget brief {uuid4().hex[:8]}", "destination": "南京", "start_date": "2026-10-10", "end_date": "2026-10-13", "budget": 8000, "adult_male_count": 1, "adult_female_count": 1},
        ).json()
        response = client.post(f"/api/v1/trips/{trip['id']}/plan", json={"query": "推荐酒店景点美食", "include_web": False})
    assert response.status_code == 200
    body = response.json()
    assert body["budget_summary"]
    # Nothing is saved on this trip yet, so the summary covers local transport only: it must
    # NOT claim accommodation or meals, which are charged only through a chosen place. The
    # old behaviour announced 住宿约 1200 CNY before the user had picked anything, and then
    # added the price of whatever they ticked on top of that guess.
    assert "住宿" not in body["budget_summary"]
    assert "hotel" not in body["budget_summary"]
    assert "交通" in body["budget_summary"]
    assert body["budget"]["itemized_summary"].endswith("元")
    # The typical amounts are still reported, explicitly outside the total.
    excluded = {line["category"] for line in body["budget"]["excluded_allowances"]}
    assert excluded == {"hotel", "restaurant"}
    assert body["budget"]["total_estimate"] == body["budget"]["breakdown"][0]["amount"]


def test_plan_recommendation_request_has_nonzero_budget_without_web_results() -> None:
    with TestClient(app) as client:
        trip = client.post(
            "/api/v1/trips",
            json={
                "title": f"offline recommendation test {uuid4().hex[:8]}",
                "destination": "南京",
                "start_date": "2026-10-10",
                "end_date": "2026-10-13",
                "budget": 8000,
                "adult_male_count": 1,
                "adult_female_count": 1,
            },
        ).json()
        response = client.post(
            f"/api/v1/trips/{trip['id']}/plan",
            json={"query": "推荐酒店、景点和美食并规划第一天", "include_web": False},
        )

    assert response.status_code == 200
    assert response.json()["budget"]["total_estimate"] > 0
