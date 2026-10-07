from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app
from app.agents.workflows import adjuster


def _create_trip(client: TestClient, destination: str = "Shanghai") -> int:
    return client.post(
        "/api/v1/trips",
        json={
            "title": f"agent adjustment {uuid4().hex[:8]}",
            "destination": destination,
            "start_date": "2026-10-01",
            "end_date": "2026-10-03",
            "budget": 3000,
        },
    ).json()["id"]


def _create_place(client: TestClient, trip_id: int, name: str = "Agent Museum") -> int:
    return client.post(
        f"/api/v1/trips/{trip_id}/places",
        json={
            "name": name,
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
    ).json()["id"]


def test_adjustment_preview_uses_fallback_and_does_not_write_until_apply(monkeypatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "")
    with TestClient(app) as client:
        trip_id = _create_trip(client)
        place_id = _create_place(client, trip_id)

        preview = client.post(
            f"/api/v1/trips/{trip_id}/adjustments",
            json={"instruction": "2026-10-02 14:00-16:00 安排 Agent Museum", "include_web": False},
        )

        assert preview.status_code == 201
        body = preview.json()
        assert body["status"] == "pending"
        assert body["operations"] == [
            {
                "kind": "add_item",
                "item_id": None,
                "place_id": place_id,
                "place": None,
                "date": "2026-10-02",
                "start_time": "14:00:00",
                "end_time": "16:00:00",
                "note": "由行程调整 Agent 建议",
                "sort_order": 0,
            }
        ]
        assert client.get(f"/api/v1/trips/{trip_id}/itinerary-items").json() == []

        applied = client.post(f"/api/v1/trips/{trip_id}/adjustments/{body['id']}/apply")

        assert applied.status_code == 200
        assert applied.json()["status"] == "applied"
        assert client.get(f"/api/v1/trips/{trip_id}/itinerary-items").json()[0]["place_id"] == place_id


def test_rejected_adjustment_cannot_be_applied_or_read_from_another_trip(monkeypatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "")
    with TestClient(app) as client:
        trip_id = _create_trip(client)
        _create_place(client, trip_id)
        other_trip_id = _create_trip(client, destination="Hangzhou")
        preview = client.post(
            f"/api/v1/trips/{trip_id}/adjustments",
            json={"instruction": "2026-10-02 14:00-16:00 安排 Agent Museum", "include_web": False},
        ).json()

        hidden = client.post(f"/api/v1/trips/{other_trip_id}/adjustments/{preview['id']}/apply")
        rejected = client.post(f"/api/v1/trips/{trip_id}/adjustments/{preview['id']}/reject")
        reapplied = client.post(f"/api/v1/trips/{trip_id}/adjustments/{preview['id']}/apply")

        assert hidden.status_code == 404
        assert rejected.status_code == 200
        assert rejected.json()["status"] == "rejected"
        assert reapplied.status_code == 409


def test_adjustment_search_query_focuses_on_requested_theme(monkeypatch) -> None:
    captured = {}

    def fake_search(query, destination):
        captured["query"] = query
        return [], "completed"

    monkeypatch.setattr(adjuster, "search_web", fake_search)
    state = {
        "trip": {"destination": "Shanghai"},
        "instruction": "10月3号下午和晚上有什么乐园项目推荐，晚饭在乐园内解决，必吃必喝",
        "include_web": True,
        "events": [],
    }
    adjuster._search_live_web(state)
    assert captured["query"] == "乐园项目 晚餐 必吃必喝"


def test_adjustment_fallback_parses_partial_date_and_builds_day_plan(monkeypatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "")
    with TestClient(app) as client:
        trip_id = _create_trip(client)
        _create_place(client, trip_id, "Shanghai Disneyland")
        response = client.post(
            f"/api/v1/trips/{trip_id}/adjustments",
            json={"instruction": "10月3号下午和晚上待到晚上8点，推荐乐园项目和必吃必喝", "include_web": False},
        )
        assert response.status_code == 201
        body = response.json()
        # The preview now shows every day of the trip, so a day with no plan still
        # appears (with no items) instead of the response starting at the requested day.
        assert [day["date"] for day in body["daily_plan"]] == ["2026-10-01", "2026-10-02", "2026-10-03"]
        scheduled = [day for day in body["daily_plan"] if day["items"]]
        assert len(scheduled) == 1
        assert scheduled[0]["date"] == "2026-10-03"
        assert scheduled[0]["items"][-1]["end_time"] == "20:00:00"
        # Everything in it was just added, and says so.
        assert {item["changed"] for item in scheduled[0]["items"]} == {"added"}
        assert body["final_summary"]


def test_adjustment_fallback_builds_half_hour_recommendations_after_dinner(monkeypatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "")
    with TestClient(app) as client:
        trip_id = _create_trip(client)
        response = client.post(
            f"/api/v1/trips/{trip_id}/adjustments",
            json={"instruction": "10月3号下午和晚上待到晚上8点，推荐乐园项目和晚饭后安排", "include_web": False},
        )

        assert response.status_code == 201
        body = response.json()
        items = next(day for day in body["daily_plan"] if day["items"])["items"]
        assert len(items) == 12
        assert any(item["start_time"] == "19:00:00" for item in items)
        assert items[-1]["end_time"] == "20:00:00"
        assert len(body["operations"]) == len(items)


def test_adjustment_apply_can_apply_only_selected_operations(monkeypatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "")
    with TestClient(app) as client:
        trip_id = _create_trip(client)
        draft = client.post(
            f"/api/v1/trips/{trip_id}/adjustments",
            json={"instruction": "10月3号下午和晚上待到晚上8点，推荐乐园项目和晚饭后安排", "include_web": False},
        ).json()

        applied = client.post(
            f"/api/v1/trips/{trip_id}/adjustments/{draft['id']}/apply",
            json={"operation_indexes": [0, 8]},
        )

        assert applied.status_code == 200
        assert len(applied.json()["itinerary_item_ids"]) == 2
        assert len(client.get(f"/api/v1/trips/{trip_id}/itinerary-items").json()) == 2


def test_explicitly_replaces_unwanted_time_slot_with_another_recommendation(monkeypatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "")
    state = {
        "trip": {"destination": "上海迪士尼", "start_date": "2026-10-01", "end_date": "2026-10-03"},
        "instruction": "我不想玩15:00–15:30的「小熊维尼历险记」+「爱丽丝梦游仙境迷宫」，请把这个项目去了并且给我推荐一个别的",
        "places": [
            {"id": 10, "name": "小熊维尼历险记+爱丽丝梦游仙境迷宫", "type": "attraction", "address": "上海迪士尼"},
        ],
        "itinerary": [
            {
                "id": 88,
                "date": "2026-10-03",
                "start_time": "15:00:00",
                "end_time": "15:30:00",
                "place_name": "小熊维尼历险记+爱丽丝梦游仙境迷宫",
            },
        ],
        "web_results": [{"title": "加勒比海盗——沉落宝藏之战", "snippet": "沉浸式室内项目"}],
    }

    operations = adjuster._fallback_operations(state)

    assert [operation.kind for operation in operations] == ["remove_item", "add_place_and_item"]
    assert operations[0].item_id == 88
    assert operations[1].place is not None
    assert operations[1].place.name == "加勒比海盗——沉落宝藏之战"
    assert operations[1].date.isoformat() == "2026-10-03"
    assert operations[1].start_time.isoformat() == "15:00:00"
    assert operations[1].end_time.isoformat() == "15:30:00"
    state["operations"] = operations
    summary = adjuster._final_summary(state, [{"date": "2026-10-03", "items": [{"place_name": operations[1].place.name}]}], [])
    # The summary names both halves of the swap, so the user can check it did what they
    # meant before applying.
    assert "删除 1 个行程项" in summary
    assert "小熊维尼历险记+爱丽丝梦游仙境迷宫" in summary
    assert "新增" in summary
    assert "加勒比海盗" in summary


def test_explicit_replacement_rule_takes_priority_over_model_output(monkeypatch) -> None:
    state = {
        "trip": {"destination": "上海迪士尼", "start_date": "2026-10-01", "end_date": "2026-10-03"},
        "instruction": "我不想玩15:00–15:30的「小熊维尼历险记」，请把这个项目去了并且给我推荐一个别的",
        "places": [],
        "itinerary": [{"id": 88, "date": "2026-10-03", "start_time": "15:00:00", "end_time": "15:30:00", "place_name": "小熊维尼历险记"}],
        "web_results": [{"title": "加勒比海盗——沉落宝藏之战", "snippet": "沉浸式室内项目"}],
        "events": [],
    }
    monkeypatch.setattr(
        adjuster,
        "_model_operations",
        lambda _: [adjuster.AdjustmentOperation(kind="add_place_and_item", date="2026-10-03", start_time="10:00", end_time="11:00", place={"name": "错误项目", "type": "attraction", "address": "上海迪士尼"})],
    )

    result = adjuster._parse_operations(state)

    assert [operation.kind for operation in result["operations"]] == ["remove_item", "add_place_and_item"]
    assert result["operations"][0].item_id == 88
    assert result["operations"][1].place.name == "加勒比海盗——沉落宝藏之战"


def test_budget_request_creates_a_set_budget_operation(monkeypatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "")
    state = {
        "trip": {"destination": "上海", "start_date": "2026-10-01", "end_date": "2026-10-03", "budget": 3000},
        "instruction": "请把这次旅行总预算调整到5000元，并保持现有行程",
        "places": [],
        "itinerary": [],
       "web_results": [],
    }

    operations = adjuster._fallback_operations(state)

    assert len(operations) == 1
    assert operations[0].kind == "set_budget"
    assert operations[0].budget_amount == 5000


def test_non_positive_budget_instruction_is_ignored(monkeypatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "")

    # A zero budget would render as "0 CNY" and zero out a real trip budget.
    assert adjuster._budget_operation("把这次旅行的预算调整为 0 元") is None
    assert adjuster._budget_operation("预算改成0") is None
    assert adjuster._budget_operation("预算调整为 5000 元").budget_amount == 5000


def test_budget_change_is_included_in_adjustment_summary(monkeypatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "")
    state = {
        "trip": {"destination": "上海", "start_date": "2026-10-01", "end_date": "2026-10-03", "budget": 3000},
        "instruction": "总预算改为5000元",
        "operations": [adjuster.AdjustmentOperation(kind="set_budget", budget_amount=5000)],
    }
    summary = adjuster._final_summary(state, [], [])
    assert "5000" in summary


def test_replacement_replaces_old_item_cost_before_recalculating_budget(monkeypatch) -> None:
    """The budget follows the itinerary, not the saved-place collection.

    The old model priced every saved place, so removing one scheduled item barely
    moved the total. It now sums the items the preview actually contains: the
    replacement (level 2 attraction) plus the transport baseline for one day.
    """
    monkeypatch.setenv("DASHSCOPE_API_KEY", "")
    state = {
        "trip": {"destination": "上海迪士尼", "start_date": "2026-10-01", "end_date": "2026-10-01", "budget": 3000},
        "instruction": "我不想玩15:00–15:30的「旧项目」，请替换一个别的",
        "places": [{"id": 1, "name": "旧项目", "type": "attraction", "address": "园区", "price_level": 1}],
        "itinerary": [{"id": 9, "date": "2026-10-01", "start_time": "15:00:00", "end_time": "15:30:00", "place_name": "旧项目", "sort_order": 0}],
        "web_results": [{"title": "新项目", "snippet": "推荐", "price_level": 2}],
        "events": [],
    }
    state["operations"] = adjuster._replacement_operations(state)
    result = adjuster._validate_preview(state)
    assert result["budget"]["total_estimate"] == 60 + 200
    assert [line["name"] for line in result["budget"]["breakdown"]] == ["市内交通基础估算", "新项目"]
    assert result["budget"]["breakdown"][1]["source"] == "level"


def test_replacement_handles_multiple_matching_slots() -> None:
    state = {
        "trip": {"destination": "上海迪士尼", "start_date": "2026-10-01", "end_date": "2026-10-03"},
        "instruction": "请删除15:00-15:30的「旧项目」并替换成其他项目",
        "places": [],
        "itinerary": [
            {"id": 1, "date": "2026-10-02", "start_time": "15:00:00", "end_time": "15:30:00", "place_name": "旧项目", "sort_order": 0},
            {"id": 2, "date": "2026-10-03", "start_time": "15:00:00", "end_time": "15:30:00", "place_name": "旧项目", "sort_order": 1},
        ],
        "web_results": [{"title": "新项目", "price_level": 2}],
    }
    operations = adjuster._replacement_operations(state)
    assert [operation.kind for operation in operations] == ["remove_item", "add_place_and_item", "remove_item", "add_place_and_item"]
    assert [operations[index].item_id for index in (0, 2)] == [1, 2]
    assert [operations[index].date.isoformat() for index in (1, 3)] == ["2026-10-02", "2026-10-03"]


def test_emergency_request_builds_return_hotel_meal_and_indoor_timeline() -> None:
    state = {
        "trip": {"destination": "上海", "start_date": "2026-10-01", "end_date": "2026-10-03"},
        "instruction": "10月3号暴雨无法出行，请改成回酒店并安排室内活动和晚餐",
        "places": [{"id": 5, "name": "上海酒店", "type": "hotel", "address": "浦东新区", "price_level": 3}],
        "itinerary": [{"id": 8, "date": "2026-10-03", "start_time": "14:00:00", "end_time": "16:00:00", "place_name": "户外景点", "sort_order": 0}],
        "weather": {"days": [{"date": "2026-10-03", "indoor_priority": True}]},
    }
    operations = adjuster._emergency_operations(state)
    names = [operation.place.name for operation in operations if operation.kind == "add_place_and_item" and operation.place]
    assert operations[0].kind == "remove_item"
    assert "返回酒店交通" in names
    assert "酒店外卖晚餐" in names
    assert any("室内" in name for name in names)


def test_operations_sent_by_the_editor_skip_the_model(monkeypatch) -> None:
    """Structured edits are authoritative: the model must not get a say in them."""
    calls = {"count": 0}

    def explode(_: object) -> list:
        calls["count"] += 1
        raise AssertionError("the model must not be consulted when operations are supplied")

    monkeypatch.setattr(adjuster, "_model_operations", explode)
    state = {
        "trip": {"destination": "南京", "start_date": "2026-11-05", "end_date": "2026-11-06", "budget": 8000},
        "instruction": "删除第一天的项目",
        "places": [],
        "itinerary": [
            {"id": 31, "date": "2026-11-05", "start_time": "09:00:00", "end_time": "11:00:00", "place_name": "南京博物院", "sort_order": 0}
        ],
        "events": [],
        "requested_operations": [adjuster.AdjustmentOperation(kind="remove_item", item_id=31)],
        "structured_only": True,
    }

    result = adjuster._parse_operations(state)

    assert calls["count"] == 0
    assert [operation.kind for operation in result["operations"]] == ["remove_item"]


def test_a_typed_request_is_merged_with_the_clicks_and_never_dropped(monkeypatch) -> None:
    """Ticking a delete AND typing a sentence must both survive into the draft.

    Sending the clicks used to replace the model call outright, so a sentence the user had
    typed next to their clicks was silently ignored.
    """
    monkeypatch.setattr(
        adjuster,
        "_model_operations",
        lambda _: [
            # The model repeats the deletion the user already made by clicking...
            adjuster.AdjustmentOperation(kind="remove_item", item_id=31),
            # ...and adds the thing the sentence asked for.
            adjuster.AdjustmentOperation(
                kind="add_place_and_item",
                date="2026-11-06",
                start_time="14:00",
                end_time="16:00",
                place={"name": "室内博物馆", "type": "attraction", "address": "南京"},
            ),
        ],
    )
    state = {
        "trip": {"destination": "南京", "start_date": "2026-11-05", "end_date": "2026-11-06", "budget": 8000},
        "instruction": "把第二天下午换成室内的项目",
        "places": [],
        "itinerary": [
            {"id": 31, "date": "2026-11-05", "start_time": "09:00:00", "end_time": "11:00:00", "place_name": "南京博物院", "sort_order": 0}
        ],
        "events": [],
        "requested_operations": [adjuster.AdjustmentOperation(kind="remove_item", item_id=31)],
        "structured_only": False,
    }

    result = adjuster._parse_operations(state)

    kinds = [operation.kind for operation in result["operations"]]
    # The click comes first, the model's duplicate is dropped, its new contribution is kept.
    assert kinds == ["remove_item", "add_place_and_item"]
    assert result["operations"][0].item_id == 31
    assert result["operations"][1].place.name == "室内博物馆"


def test_a_model_addition_cannot_land_on_a_slot_the_user_just_filled(monkeypatch) -> None:
    monkeypatch.setattr(
        adjuster,
        "_model_operations",
        lambda _: [
            adjuster.AdjustmentOperation(
                kind="add_place_and_item",
                date="2026-11-06",
                start_time="14:00",
                end_time="16:00",
                place={"name": "AI 挑的项目", "type": "attraction", "address": "南京"},
            )
        ],
    )
    mine = adjuster.AdjustmentOperation(
        kind="add_place_and_item",
        date="2026-11-06",
        start_time="14:00",
        end_time="16:00",
        place={"name": "我自己选的项目", "type": "attraction", "address": "南京"},
    )
    state = {
        "trip": {"destination": "南京", "start_date": "2026-11-05", "end_date": "2026-11-06", "budget": 8000},
        "instruction": "帮我安排第二天下午",
        "places": [],
        "itinerary": [],
        "events": [],
        "requested_operations": [mine],
        "structured_only": False,
    }

    result = adjuster._parse_operations(state)

    assert [operation.place.name for operation in result["operations"]] == ["我自己选的项目"]


def test_removing_a_scheduled_item_drops_its_cost_from_the_budget() -> None:
    """The user's complaint: deleting a trip item did not remove its money."""
    state = {
        "trip": {"destination": "南京", "start_date": "2026-11-05", "end_date": "2026-11-05", "budget": 8000},
        "instruction": "删掉这一项",
        "places": [],
        "itinerary": [
            {
                "id": 41,
                "place_id": 7,
                "date": "2026-11-05",
                "start_time": "09:00:00",
                "end_time": "11:00:00",
                "place_name": "南京博物院",
                "type": "attraction",
                "quoted_cost": 0,
                "sort_order": 0,
            },
            {
                "id": 42,
                "place_id": 8,
                "date": "2026-11-05",
                "start_time": "12:00:00",
                "end_time": "13:00:00",
                "place_name": "南京大牌檔",
                "type": "restaurant",
                "quoted_cost": 480,
                "quantity": 2,
                "sort_order": 1,
            },
        ],
        "events": [],
        "operations": [adjuster.AdjustmentOperation(kind="remove_item", item_id=42)],
        "traveler_profile": {"total_people": 2},
    }

    result = adjuster._validate_preview(state)

    # transport 60 + the free museum; the 480 restaurant is gone, not merely hidden.
    assert result["budget"]["total_estimate"] == 60
    assert [line["name"] for line in result["budget"]["breakdown"]] == ["市内交通基础估算", "南京博物院"]


def test_an_added_item_can_carry_the_price_the_user_typed() -> None:
    """Custom input: the user's own place and its price drive the estimate."""
    state = {
        "trip": {"destination": "南京", "start_date": "2026-11-05", "end_date": "2026-11-05", "budget": 8000},
        "instruction": "加一个我想吃的小店",
        "places": [],
        "itinerary": [],
        "events": [],
        "traveler_profile": {"total_people": 2},
        "operations": [
            adjuster.AdjustmentOperation(
                kind="add_place_and_item",
                date="2026-11-05",
                start_time="18:00",
                end_time="19:30",
                place={
                    "name": "我自己找的鸭血粉丝汤",
                    "type": "restaurant",
                    "address": "南京 · 地址待确认",
                    "estimated_cost": 96,
                    "quantity": 1,
                },
            )
        ],
    }

    result = adjuster._validate_preview(state)

    assert result["budget"]["total_estimate"] == 60 + 96
    row = result["budget"]["breakdown"][1]
    assert row["name"] == "我自己找的鸭血粉丝汤"
    # A price the user set is quoted, never re-derived from a price level.
    assert row["source"] == "quoted"
    assert row["amount"] == 96


def test_a_budget_phrase_does_not_swallow_the_rest_of_the_sentence(monkeypatch) -> None:
    """A compound request must not lose its non-budget half.

    The deterministic budget shortcut used to win outright, so "把 14:00-16:00 的博物馆换成
    室内的，并把总预算改成 6000 元" came back as a lone set_budget: no error, no mention,
    just half the request gone.
    """
    monkeypatch.setattr(
        adjuster,
        "_model_operations",
        lambda _: [
            adjuster.AdjustmentOperation(kind="set_budget", budget_amount=6000),
            adjuster.AdjustmentOperation(
                kind="update_item", item_id=77, date="2026-12-03", start_time="14:00", end_time="16:00"
            ),
        ],
    )
    state = {
        "trip": {"destination": "南京", "start_date": "2026-12-01", "end_date": "2026-12-05", "budget": 8000},
        "instruction": "把 12月3日 14:00-16:00 的博物馆换成一个室内项目，并把这次旅行总预算改成 6000 元",
        "places": [],
        "itinerary": [
            {
                "id": 77,
                "date": "2026-12-03",
                "start_time": "14:00:00",
                "end_time": "16:00:00",
                "place_name": "六朝博物馆",
                "sort_order": 0,
            }
        ],
        "events": [],
    }

    result = adjuster._parse_operations(state)
    kinds = [operation.kind for operation in result["operations"]]

    # The explicit budget stays, the model's duplicate is dropped, and the other half of
    # the sentence survives.
    assert kinds == ["set_budget", "update_item"]
    assert result["operations"][0].budget_amount == 6000
    assert result["operations"][1].item_id == 77


def test_a_plain_budget_request_still_takes_the_fast_path(monkeypatch) -> None:
    """The opposite of the test above: a budget-only sentence must not pay for an LLM call."""

    def explode(_: object) -> list:
        raise AssertionError("a budget-only sentence must not reach the model")

    monkeypatch.setattr(adjuster, "_model_operations", explode)
    state = {
        "trip": {"destination": "南京", "start_date": "2026-12-01", "end_date": "2026-12-05", "budget": 8000},
        "instruction": "请把这次旅行总预算调整到 6000 元",
        "places": [],
        "itinerary": [],
        "events": [],
    }

    result = adjuster._parse_operations(state)

    assert [operation.kind for operation in result["operations"]] == ["set_budget"]


def test_the_preview_shows_every_day_and_marks_what_moved() -> None:
    state = {
        "trip": {"destination": "南京", "start_date": "2026-11-05", "end_date": "2026-11-07", "budget": 8000},
        "instruction": "把博物院挪到第三天",
        "places": [],
        "itinerary": [
            {
                "id": 51,
                "place_id": 9,
                "date": "2026-11-05",
                "start_time": "09:00:00",
                "end_time": "11:00:00",
                "place_name": "南京博物院",
                "type": "attraction",
                "price_level": 0,
                "sort_order": 0,
            }
        ],
        "events": [],
        "operations": [
            adjuster.AdjustmentOperation(
                kind="update_item", item_id=51, date="2026-11-07", start_time="14:00", end_time="16:00"
            )
        ],
    }

    result = adjuster._validate_preview(state)

    assert [day["date"] for day in result["daily_plan"]] == ["2026-11-05", "2026-11-06", "2026-11-07"]
    moved_day = next(day for day in result["daily_plan"] if day["items"])
    assert moved_day["date"] == "2026-11-07"
    assert moved_day["items"][0]["changed"] == "moved"
    assert moved_day["items"][0]["start_time"] == "14:00:00"
    # Moving an item must not create a second copy of it.
    assert sum(len(day["items"]) for day in result["daily_plan"]) == 1


def test_editor_can_apply_a_price_it_edited_in_the_preview(monkeypatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "")
    with TestClient(app) as client:
        trip_id = _create_trip(client)
        draft = client.post(
            f"/api/v1/trips/{trip_id}/adjustments",
            json={
                "instruction": "新增一个项目",
                "operations": [
                    {
                        "kind": "add_place_and_item",
                        "date": "2026-10-02",
                        "start_time": "18:00",
                        "end_time": "19:30",
                        "place": {
                            "name": "自助晚餐",
                            "type": "restaurant",
                            "address": "上海 · 地址待确认",
                            "estimated_cost": 300,
                            "quantity": 1,
                        },
                    }
                ],
            },
        )
        assert draft.status_code == 201
        body = draft.json()
        assert body["daily_plan"]
        assert body["budget"]["total_estimate"] == 60 * 3 + 300

        # The user then edits the price in the preview and applies that instead.
        applied = client.post(
            f"/api/v1/trips/{trip_id}/adjustments/{body['id']}/apply",
            json={
                "operations": [
                    {
                        "kind": "add_place_and_item",
                        "date": "2026-10-02",
                        "start_time": "18:00",
                        "end_time": "19:30",
                        "place": {
                            "name": "自助晚餐",
                            "type": "restaurant",
                            "address": "上海 · 地址待确认",
                            "estimated_cost": 188,
                            "quantity": 1,
                        },
                    }
                ]
            },
        )

        assert applied.status_code == 200
        places = client.get(f"/api/v1/trips/{trip_id}/places").json()
        added = next(place for place in places if place["name"] == "自助晚餐")
        assert added["estimated_cost"] == 188


def test_a_clicked_operation_outside_the_trip_is_refused_when_proposed(monkeypatch) -> None:
    """The panel previews and prices these operations, so an impossible one must be
    refused immediately instead of failing later, after the user pressed apply."""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "")
    with TestClient(app) as client:
        trip_id = _create_trip(client)

        response = client.post(
            f"/api/v1/trips/{trip_id}/adjustments",
            json={
                "instruction": "新增一个越界项目",
                "structured_only": True,
                "operations": [
                    {
                        "kind": "add_place_and_item",
                        "date": "2027-01-01",
                        "start_time": "10:00",
                        "end_time": "11:00",
                        "place": {"name": "越界项目", "type": "attraction", "address": "上海"},
                    }
                ],
            },
        )

        assert response.status_code == 422
        # And no draft was left behind for the panel to pick up.
        assert client.get(f"/api/v1/trips/{trip_id}/adjustments").json() == []


def test_an_edited_operation_is_still_validated(monkeypatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "")
    with TestClient(app) as client:
        trip_id = _create_trip(client)
        draft = client.post(
            f"/api/v1/trips/{trip_id}/adjustments",
            json={
                "instruction": "新增一个项目",
                "operations": [
                    {
                        "kind": "add_place_and_item",
                        "date": "2026-10-02",
                        "start_time": "18:00",
                        "end_time": "19:30",
                        "place": {"name": "自助晚餐", "type": "restaurant", "address": "上海 · 地址待确认"},
                    }
                ],
            },
        ).json()

        # A date outside the trip is rejected even when it comes from the editor.
        outside = client.post(
            f"/api/v1/trips/{trip_id}/adjustments/{draft['id']}/apply",
            json={
                "operations": [
                    {
                        "kind": "add_place_and_item",
                        "date": "2027-01-01",
                        "start_time": "18:00",
                        "end_time": "19:30",
                        "place": {"name": "越界项目", "type": "restaurant", "address": "上海 · 地址待确认"},
                    }
                ]
            },
        )

        assert outside.status_code == 422
        assert client.get(f"/api/v1/trips/{trip_id}/itinerary-items").json() == []


def test_saved_place_price_and_count_can_be_corrected(monkeypatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "")
    with TestClient(app) as client:
        trip_id = _create_trip(client)
        place_id = _create_place(client, trip_id)

        response = client.patch(
            f"/api/v1/trips/{trip_id}/places/{place_id}",
            json={"estimated_cost": 780, "quantity": 3, "price_level": None},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["estimated_cost"] == 780
        assert body["quantity"] == 3
        assert body["price_level"] is None
        # Fields that were not sent keep their value.
        assert body["name"] == "Agent Museum"
        assert body["rating"] == 4.6

        other_trip = _create_trip(client, destination="Hangzhou")
        assert client.patch(f"/api/v1/trips/{other_trip}/places/{place_id}", json={"quantity": 2}).status_code == 404
