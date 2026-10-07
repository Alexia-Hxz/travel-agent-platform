from typing import Any

from app.agents.tools import llm, plan_summary


def test_the_summary_call_disables_thinking_and_bounds_itself(monkeypatch) -> None:
    """Thinking on this call cost nearly a minute and produced less text.

    Measured on one request: thinking on answered in 28.8s with 340 visible characters,
    thinking off in 11.2s with 820 -- the reasoning was spending the output budget. A real
    plan's larger context then ran past the shared 30s timeout twice (61.2s measured in a
    full run) and silently fell back to the local template, so the written summary never
    reached anyone. Every other model call in the project already disables thinking.
    """
    seen: list[dict[str, Any]] = []

    class _Completions:
        @staticmethod
        def create(**kwargs: Any):
            seen.append(kwargs)
            message = type("_Message", (), {"content": "这是一段模型写出的总结。"})()
            choice = type("_Choice", (), {"message": message, "finish_reason": "stop"})()
            return type("_Response", (), {"choices": [choice]})()

    client = type("_Client", (), {"chat": type("_Chat", (), {"completions": _Completions()})()})()
    monkeypatch.setattr(plan_summary, "_model_client", lambda: client)

    summary, status = plan_summary.generate_final_summary(_context())

    assert status == "completed"
    assert summary == "这是一段模型写出的总结。"
    assert seen[0]["extra_body"] == {"enable_thinking": False}
    # Bounded, and bounded below the shared chat timeout: this is the last step of a plan and
    # must not be able to hold it open for a minute waiting for an answer available locally.
    assert plan_summary.SUMMARY_TIMEOUT < llm.REQUEST_TIMEOUT
    assert plan_summary.SUMMARY_TIMEOUT * (1 + plan_summary.SUMMARY_MAX_RETRIES) < 60


def test_a_failed_summary_call_is_logged_before_falling_back(monkeypatch, caplog) -> None:
    """The fallback is designed, but it used to be silent: 61s of waiting left no trace."""

    class _Boom:
        class chat:  # noqa: N801 - stands in for the SDK's nested namespace
            class completions:  # noqa: N801
                @staticmethod
                def create(**kwargs: Any):
                    raise RuntimeError("provider exploded")

    monkeypatch.setattr(plan_summary, "_model_client", lambda: _Boom())

    with caplog.at_level("WARNING"):
        _summary, status = plan_summary.generate_final_summary(_context())

    assert status == "fallback"
    assert any("summary" in record.message for record in caplog.records)


def test_final_summary_falls_back_to_a_contextual_chinese_summary_without_a_model_key(monkeypatch) -> None:
    monkeypatch.setattr(plan_summary, "_model_client", lambda: None)

    summary, status = plan_summary.generate_final_summary(
        {
            "trip": {
                "destination": "上海",
                "start_date": "2026-10-01",
                "end_date": "2026-10-02",
            },
            "saved_places": [{"name": "初心启程酒店", "type": "hotel"}],
            "itinerary": [],
            "transport_advice": {
                "origin": "初心启程酒店",
                "destination": "迪士尼",
                "recommendation": "优先乘坐公共交通。",
            },
            "budget": {"total_estimate": 600, "currency": "CNY", "remaining_budget": 1400},
            "conflicts": [],
        }
    )

    assert status == "fallback"
    assert "上海" in summary
    assert "初心启程酒店" in summary
    assert "优先乘坐公共交通。" in summary
    assert "600 CNY" in summary


def _context(**overrides):
    context = {
        "trip": {"destination": "南京", "start_date": "2026-10-10", "end_date": "2026-10-14"},
        "saved_places": [],
        "itinerary": [],
        "budget": {"total_estimate": 1000, "currency": "CNY"},
        "conflicts": [],
    }
    context.update(overrides)
    return context


def test_fallback_summary_lists_the_scheduled_itinerary_day_by_day(monkeypatch) -> None:
    """每日行程 is the single source of truth, so it is named, not just counted.

    The old line only said "当前已安排 1 个行程项", which told the traveller nothing
    about what was actually planned for which day.
    """
    monkeypatch.setattr(plan_summary, "_model_client", lambda: None)

    summary, _ = plan_summary.generate_final_summary(
        _context(
            itinerary=[
                {
                    "date": "2026-10-11",
                    "start_time": "09:00:00",
                    "end_time": "11:00:00",
                    "place_name": "南京博物院",
                    "note": None,
                },
                {
                    "date": "2026-10-10",
                    "start_time": "14:00:00",
                    "end_time": "16:00:00",
                    "place_name": "夫子庙",
                    "note": "夜游秦淮河",
                },
            ]
        )
    )

    assert "这是本次确定的安排" in summary
    # Both entries are named, with their day and time window, in date order.
    assert "2026-10-10 14:00-16:00 夫子庙（夜游秦淮河）" in summary
    assert "2026-10-11 09:00-11:00 南京博物院" in summary
    assert summary.index("夫子庙") < summary.index("南京博物院")


def test_fallback_summary_calls_ticked_recommendations_candidates(monkeypatch) -> None:
    """Ticked recommendations are not scheduled until they reach 每日行程."""
    monkeypatch.setattr(plan_summary, "_model_client", lambda: None)

    summary, _ = plan_summary.generate_final_summary(
        _context(
            selected_recommendations=[{"name": "金陵饭店", "estimated_cost": 1600}],
        )
    )

    assert "金陵饭店" in summary
    assert "还只是候选" in summary
    # The old wording claimed they were already folded into the plan.
    assert "已纳入本次总结的安排建议" not in summary


def test_fallback_summary_says_the_itinerary_is_empty_instead_of_promoting_a_draft(monkeypatch) -> None:
    """A draft day must not be described as though it were the agreed plan."""
    monkeypatch.setattr(plan_summary, "_model_client", lambda: None)

    summary, _ = plan_summary.generate_final_summary(
        _context(
            itinerary=[],
            generated_daily_plan=[
                {"date": "2026-10-10", "items": [{"place_name": "夫子庙"}, {"place_name": "金陵饭店"}]}
            ],
        )
    )

    assert "每日行程目前还是空的" in summary
    assert "只是建议草稿" in summary
    assert "确认排入每日行程后才会成为确定安排" in summary

