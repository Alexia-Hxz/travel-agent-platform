"""The summary must cover the planning-agent selection and the adjustment result.

`生成行程介绍与总结` used to see only the trip's saved places and itinerary, so
whatever the user ticked in the planning agent or concluded in the conversational
adjustment was silently dropped from the final write-up.
"""

from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app


def _create_trip(client: TestClient) -> int:
    response = client.post(
        "/api/v1/trips",
        json={
            "title": f"budget brief {uuid4().hex[:8]}",
            "destination": "上海",
            "start_date": "2026-10-01",
            "end_date": "2026-10-03",
            "budget": 4000,
        },
    )
    assert response.status_code == 201
    return int(response.json()["id"])


def _plan(client: TestClient, trip_id: int, payload: dict) -> dict:
    response = client.post(f"/api/v1/trips/{trip_id}/plan", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def test_selected_recommendations_and_adjustment_are_echoed_back() -> None:
    with TestClient(app) as client:
        trip_id = _create_trip(client)
        body = _plan(
            client,
            trip_id,
            {
                "query": "生成行程介绍与总结",
                "include_web": False,
                "selected_recommendations": [
                    {"name": "金陵饭店", "category": "hotel", "estimated_cost": 1800, "reason": "位置便利"},
                    {"name": "夫子庙", "category": "attraction", "estimated_cost": 200, "reason": "夜游"},
                ],
                "adjustment_summary": "第二天改为上午夫子庙、下午休息。",
            },
        )

    handoff = body["handoff"]
    assert [item["name"] for item in handoff["selected_recommendations"]] == ["金陵饭店", "夫子庙"]
    assert handoff["selected_total"] == 2000
    assert handoff["has_adjustment_summary"] is True


def test_plan_without_a_handoff_reports_empty_material() -> None:
    with TestClient(app) as client:
        trip_id = _create_trip(client)
        body = _plan(client, trip_id, {"query": "生成行程介绍与总结", "include_web": False})

    handoff = body["handoff"]
    assert handoff["selected_recommendations"] == []
    assert handoff["selected_total"] == 0
    assert handoff["has_adjustment_summary"] is False


def test_summary_lead_names_the_selected_recommendations() -> None:
    with TestClient(app) as client:
        trip_id = _create_trip(client)
        body = _plan(
            client,
            trip_id,
            {
                "query": "生成行程介绍与总结",
                "include_web": False,
                "selected_recommendations": [
                    {"name": "金陵饭店", "category": "hotel", "estimated_cost": 1800},
                ],
            },
        )

    assert "1 个项目" in body["summary"]
    assert "1800" in body["summary"]


def test_fallback_summary_includes_the_handed_over_material(monkeypatch) -> None:
    """The local fallback is used whenever the model is unavailable, so it must too."""
    from app.agents.tools import plan_summary

    # Force the deterministic local path instead of calling the real model.
    monkeypatch.setattr(plan_summary, "_model_client", lambda: None)

    summary, status = plan_summary.generate_final_summary(
        {
            "trip": {"destination": "上海", "start_date": "2026-10-01", "end_date": "2026-10-03"},
            "saved_places": [],
            "itinerary": [],
            "budget": {"total_estimate": 0, "currency": "CNY"},
            "conflicts": [],
            "selected_recommendations": [
                {"name": "金陵饭店", "category": "hotel", "estimated_cost": 1800},
                {"name": "夫子庙", "category": "attraction", "estimated_cost": 200},
            ],
            "adjustment_summary": "第二天改为上午夫子庙、下午休息。",
        }
    )

    assert status == "fallback"
    assert "金陵饭店" in summary
    assert "夫子庙" in summary
    assert "2000" in summary
    assert "第二天改为上午夫子庙、下午休息。" in summary


def test_model_prompt_carries_the_handed_over_material(monkeypatch) -> None:
    """The model path must receive the selection and the adjustment conclusion."""
    from app.agents.tools import plan_summary

    captured: dict = {}

    class _Message:
        content = "总结正文"

    class _Choice:
        message = _Message()

    class _Response:
        choices = [_Choice()]

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _Response()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    monkeypatch.setattr(plan_summary, "_model_client", lambda: _Client())

    summary, status = plan_summary.generate_final_summary(
        {
            "trip": {"destination": "上海", "start_date": "2026-10-01", "end_date": "2026-10-03"},
            "saved_places": [],
            "itinerary": [],
            "budget": {"total_estimate": 0, "currency": "CNY"},
            "conflicts": [],
            "selected_recommendations": [{"name": "金陵饭店", "category": "hotel", "estimated_cost": 1800}],
            "adjustment_summary": "第二天改为上午夫子庙。",
        }
    )

    assert status == "completed"
    assert summary == "总结正文"
    system_prompt = captured["messages"][0]["content"]
    assert "selected_recommendations" in system_prompt
    assert "adjustment_summary" in system_prompt
    payload = captured["messages"][1]["content"]
    assert "金陵饭店" in payload
    assert "第二天改为上午夫子庙。" in payload


def test_summary_hands_every_saved_place_to_the_model(monkeypatch) -> None:
    """The summary must cover the whole collection, not just its names."""
    from app.agents.tools import plan_summary

    captured: dict = {}

    class _Message:
        content = "总结正文"

    class _Choice:
        message = _Message()

    class _Response:
        choices = [_Choice()]

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _Response()

    class _Client:
        class chat:
            completions = _Completions()

    monkeypatch.setattr(plan_summary, "_model_client", lambda: _Client())

    plan_summary.generate_final_summary(
        {
            "trip": {"destination": "南京", "start_date": "2026-10-10", "end_date": "2026-10-12"},
            "saved_place_count": 2,
            "saved_places": [
                {
                    "name": "鸡鸣寺樱花大道",
                    "type": "attraction",
                    "type_label": "景点",
                    "address": "南京市玄武区鸡鸣寺路1号",
                    "opening_hours": "09:00-17:00",
                    "rating": 4.6,
                    "estimated_cost": 200,
                },
                {
                    "name": "玄武湖假日酒店",
                    "type": "hotel",
                    "type_label": "住宿",
                    "address": "南京市玄武区环湖路",
                    "opening_hours": None,
                    "rating": 4.2,
                    "estimated_cost": 1040,
                },
            ],
            "itinerary": [],
            "budget": {"total_estimate": 1240, "currency": "CNY"},
            "conflicts": [],
        }
    )

    payload = captured["messages"][1]["content"]
    assert "鸡鸣寺樱花大道" in payload
    assert "玄武湖假日酒店" in payload
    assert "南京市玄武区鸡鸣寺路1号" in payload
    assert "09:00-17:00" in payload
    assert "1040" in payload
    # The instruction must make covering the collection mandatory.
    system_prompt = captured["messages"][0]["content"]
    assert "saved_places" in system_prompt
    assert "不得遗漏" in system_prompt


def test_fallback_summary_lists_every_saved_place_with_its_details(monkeypatch) -> None:
    from app.agents.tools import plan_summary

    monkeypatch.setattr(plan_summary, "_model_client", lambda: None)

    summary, status = plan_summary.generate_final_summary(
        {
            "trip": {"destination": "南京", "start_date": "2026-10-10", "end_date": "2026-10-12"},
            "saved_place_count": 2,
            "saved_places": [
                {
                    "name": "鸡鸣寺樱花大道",
                    "type_label": "景点",
                    "address": "南京市玄武区鸡鸣寺路1号",
                    "opening_hours": "09:00-17:00",
                    "rating": 4.6,
                    "estimated_cost": 200,
                },
                {
                    "name": "玄武湖假日酒店",
                    "type_label": "住宿",
                    "address": "南京市玄武区环湖路",
                    "estimated_cost": 0,
                },
            ],
            "itinerary": [],
            "budget": {"total_estimate": 200, "currency": "CNY"},
            "conflicts": [],
        }
    )

    assert status == "fallback"
    # Both places, with their actionable detail, and no truncation.
    assert "鸡鸣寺樱花大道" in summary
    assert "玄武湖假日酒店" in summary
    assert "南京市玄武区鸡鸣寺路1号" in summary
    assert "09:00-17:00" in summary
    assert "200元" in summary
    # A zero-cost place reads as free rather than as "0".
    assert "免费" in summary


def test_handoff_reports_the_saved_collection() -> None:
    with TestClient(app) as client:
        trip = client.post(
            "/api/v1/trips",
            json={
                "title": f"summary test {uuid4().hex[:8]}",
                "destination": "南京",
                "start_date": "2026-10-10",
                "end_date": "2026-10-12",
                "budget": 6000,
            },
        ).json()
        for name in ("鸡鸣寺樱花大道", "玄武湖假日酒店"):
            response = client.post(
                f"/api/v1/trips/{trip['id']}/places",
                json={
                    "name": name,
                    "type": "attraction",
                    "address": "南京市玄武区",
                    "latitude": None,
                    "longitude": None,
                    "image_url": None,
                    "rating": 4.5,
                    "price_level": 2,
                    "opening_hours": "09:00-17:00",
                    "source_url": None,
                },
            )
            assert response.status_code == 201

        response = client.post(
            f"/api/v1/trips/{trip['id']}/plan",
            json={"query": "生成行程介绍与总结", "include_web": False},
        )

    assert response.status_code == 200
    handoff = response.json()["handoff"]
    assert handoff["saved_place_count"] == 2
    assert handoff["saved_place_names"] == ["鸡鸣寺樱花大道", "玄武湖假日酒店"]
    assert "2 个地点已全部纳入本次总结" in response.json()["summary"]


def test_handoff_tolerates_a_blank_name_and_counts_missing_costs_as_zero() -> None:
    with TestClient(app) as client:
        trip_id = _create_trip(client)
        body = _plan(
            client,
            trip_id,
            {
                "query": "生成行程介绍与总结",
                "include_web": False,
                "selected_recommendations": [
                    {"name": "   ", "category": "hotel", "estimated_cost": 999},
                    {"name": "外滩", "category": "attraction", "estimated_cost": None},
                ],
            },
        )

    handoff = body["handoff"]
    assert [item["name"] for item in handoff["selected_recommendations"]] == ["外滩"]
    assert handoff["selected_total"] == 0
