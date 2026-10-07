"""The category budget framework the planning panel leads with.

The user's complaint was that the panel answered the wrong question: "我需要让我知道大概的价格…
显示的不是真正的预估预算，是还没有选择项目时的大概预算，可以根据我的真正预算来写（大概占我
真正预算的80%），比如酒店大概多少钱，景点大概多少钱，美食大概多少钱". So this file pins three
things: the framework is derived from the trip's own budget, it lands on 80% of it, and the rows
add up to exactly that figure. It also pins what the framework is NOT: it must stay separate from
the exact sums (saved places, ticked recommendations) that the same panel still shows.
"""

from datetime import date

import pytest

from app.agents.tools import budget_plan
from app.agents.tools.budget import estimate_budget
from app.agents.tools.budget_plan import (
    DEFAULT_ALLOCATION_SHARES,
    PLANNED_BUDGET_RATIO,
    build_budget_plan,
    format_budget_plan_summary,
    planned_total,
)
from app.agents.workflows import planner

# A realistic model answer for a six-day trip for two: amounts that already sum to 24000.
MODEL_ANSWER = {"hotel": 11000, "restaurant": 6200, "attraction": 3600, "transport": 2000, "other": 1200}


def _plan(
    budget: object = 30000,
    days: int = 6,
    people: int = 2,
    shares: dict | None = None,
    destination: str = "日本东京",
) -> dict:
    return build_budget_plan(budget, days, destination, {"total_people": people}, model_shares=shares)


@pytest.fixture(autouse=True)
def _never_call_the_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test in this file may reach the network.

    The default answer is "the model could not answer", so a test that says nothing about the model
    gets the deterministic default shares; the model's own path is exercised by passing ``shares``
    or by patching this function inside the test that cares.
    """
    monkeypatch.setattr(budget_plan, "ask_the_model_for_shares", lambda *args, **kwargs: None)


def _amounts(plan: dict) -> dict[str, int]:
    return {row["category"]: row["amount"] for row in plan["categories"]}


# --------------------------------------------------------------------------------------
# The 80% framework
# --------------------------------------------------------------------------------------


def test_the_framework_plans_to_spend_eighty_percent_of_the_trip_budget() -> None:
    plan = _plan()

    assert plan["status"] == "estimated"
    assert plan["trip_budget"] == 30000
    assert plan["planned_total"] == 24000
    assert plan["reserve_amount"] == 6000
    assert plan["planned_ratio"] == PLANNED_BUDGET_RATIO == 0.8


def test_the_rows_add_up_to_the_planned_total() -> None:
    """A table whose rows do not sum to its own total is worse than a rounded one."""
    plan = _plan()

    assert sum(_amounts(plan).values()) == plan["planned_total"] == 24000


@pytest.mark.parametrize("budget", [1, 7, 99, 333, 1234, 8000, 19999, 30000, 123456, 999999])
def test_the_rows_add_up_for_any_budget(budget: int) -> None:
    plan = _plan(budget=budget, days=5, people=3)

    assert sum(_amounts(plan).values()) == plan["planned_total"]
    assert plan["planned_total"] == int(round(budget * 0.8))


def test_the_categories_are_the_ones_the_traveller_asked_about() -> None:
    """酒店 / 景点 / 美食 were named explicitly; transport and shopping are the honest extras."""
    plan = _plan()

    assert [row["category"] for row in plan["categories"]] == [
        "hotel",
        "restaurant",
        "attraction",
        "transport",
        "other",
    ]
    labels = " ".join(row["label"] for row in plan["categories"])
    assert "酒店" in labels and "美食" in labels and "景点" in labels


def test_a_trip_without_a_budget_gets_no_framework_at_all() -> None:
    """Better to ask for the budget than to invent an allocation for money never mentioned."""
    for budget in (None, 0, -5, "abc", ""):
        plan = _plan(budget=budget)

        assert plan["status"] == "unavailable"
        assert plan["categories"] == []
        assert plan["planned_total"] == 0
        assert plan["source"] == "none"
        assert "总预算" in plan["note"]


def test_planned_total_is_eighty_percent_and_none_without_a_budget() -> None:
    assert planned_total(30000) == 24000
    assert planned_total("30000") == 24000
    assert planned_total(30000.0) == 24000
    assert planned_total(None) is None
    assert planned_total(0) is None
    assert planned_total("abc") is None


# --------------------------------------------------------------------------------------
# The model's answer, and when it is thrown away
# --------------------------------------------------------------------------------------


def test_a_sane_model_answer_is_used_as_given() -> None:
    plan = _plan(shares=MODEL_ANSWER)

    assert plan["source"] == "model"
    # The model's own proportions survive; only the rounding is ours.
    assert _amounts(plan) == MODEL_ANSWER
    assert sum(_amounts(plan).values()) == 24000


def test_percentages_work_as_well_as_amounts() -> None:
    """A model asked for 45/25/15/10/5 must land on the same split as one answering in yuan."""
    from_percent = _plan(shares={"hotel": 45, "restaurant": 25, "attraction": 15, "transport": 10, "other": 5})
    from_amounts = _plan(
        shares={"hotel": 10800, "restaurant": 6000, "attraction": 3600, "transport": 2400, "other": 1200}
    )

    assert from_percent["source"] == "model"
    assert _amounts(from_percent) == _amounts(from_amounts)


def test_an_answer_that_parks_a_third_of_the_money_in_shopping_is_thrown_away() -> None:
    """The answer this guard exists for: 31% 其他 on a four-day city break."""
    plan = _plan(
        budget=8000,
        days=4,
        people=1,
        shares={"hotel": 1200, "restaurant": 1800, "attraction": 800, "transport": 600, "other": 2000},
    )

    assert plan["source"] == "default"
    assert _amounts(plan)["other"] < 1000
    assert _amounts(plan)["hotel"] > _amounts(plan)["other"]


def test_an_incomplete_answer_is_thrown_away_rather_than_re_weighted() -> None:
    """Filling the missing categories from the defaults would silently change what the model said."""
    plan = _plan(days=5, people=3, shares={"hotel": 4000, "restaurant": 2000, "attraction": 1000})

    assert plan["source"] == "default"


def test_the_defaults_are_used_when_the_model_cannot_answer() -> None:
    plan = _plan(budget=30000, days=6, people=2, shares=None)

    assert plan["source"] == "default"
    expected = int(round(30000 * PLANNED_BUDGET_RATIO))
    assert plan["planned_total"] == expected
    assert sum(_amounts(plan).values()) == expected
    # The default framework is the documented one, so the fallback is never a surprise.
    shares = {row["category"]: row["share_percent"] for row in plan["categories"]}
    assert shares == {"hotel": 45, "restaurant": 25, "attraction": 15, "transport": 10, "other": 5}


def test_an_answer_from_the_model_is_actually_looked_up_and_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """Proves the wiring: with no shares handed in, the module asks the model itself."""
    seen: dict[str, object] = {}

    def _answer(destination, days, people, trip_budget, amount_to_allocate):
        seen.update(
            destination=destination,
            days=days,
            people=people,
            trip_budget=trip_budget,
            amount_to_allocate=amount_to_allocate,
        )
        return dict(MODEL_ANSWER)

    monkeypatch.setattr(budget_plan, "ask_the_model_for_shares", _answer)

    plan = _plan(budget=30000, days=6, people=2, shares=None)

    assert plan["source"] == "model"
    assert _amounts(plan) == MODEL_ANSWER
    # The model is told the destination, the shape of the trip, and exactly how much to split.
    assert seen == {
        "destination": "日本东京",
        "days": 6,
        "people": 2,
        "trip_budget": 30000,
        "amount_to_allocate": 24000,
    }


def test_a_broken_model_answer_is_not_allowed_to_fail_the_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    """The framework is a convenience. Whatever the provider does, the plan still gets a budget."""

    def _explode(*args, **kwargs):
        raise RuntimeError("the provider is down")

    monkeypatch.setattr(budget_plan, "ask_the_model_for_shares", _explode)

    plan = _plan(budget=30000, days=6, people=2, shares=None)

    assert plan["status"] == "estimated"
    assert plan["source"] == "default"
    assert sum(_amounts(plan).values()) == 24000


# --------------------------------------------------------------------------------------
# Day trips, and the per-night / per-person reading
# --------------------------------------------------------------------------------------


def test_a_day_trip_does_not_pay_for_a_hotel_it_never_uses() -> None:
    """The fallback used to put 720 CNY on accommodation for a same-day trip while the same row's
    note said "当日往返，可不安排住宿"."""
    plan = _plan(budget=2000, days=1, people=2)
    amounts = _amounts(plan)

    assert amounts["hotel"] == 0
    assert amounts["restaurant"] > 0 and amounts["attraction"] > 0
    assert sum(amounts.values()) == 1600


def test_no_hotel_on_a_day_trip_is_a_good_answer_and_a_bad_one_on_a_week() -> None:
    no_hotel = {"hotel": 0, "restaurant": 800, "attraction": 700, "transport": 400, "other": 100}
    assert budget_plan._model_answer_is_sane(no_hotel, nights=0) is True
    assert budget_plan._model_answer_is_sane(no_hotel, nights=5) is False


def test_every_row_explains_what_one_unit_costs() -> None:
    """The user asked for "大概的价格"; a number nobody can sanity-check is not that."""
    plan = _plan(shares=MODEL_ANSWER)
    units = {row["category"]: row["unit"] for row in plan["categories"]}

    assert units["hotel"] == "约 2200 CNY/晚 × 5 晚"
    # 6200 over 6 days and 2 people, not over 6 days alone: the label says 每人每天.
    assert units["restaurant"] == "约 517 CNY/人/天 × 2 人 × 6 天"
    assert units["attraction"] == "约 1800 CNY/人 × 2 人（全程门票）"
    assert units["transport"] == "约 333 CNY/天 × 6 天"
    assert "购物" in units["other"]


def test_the_note_says_which_number_this_is() -> None:
    plan = _plan()

    assert "30000" in plan["note"]
    assert "24000" in plan["note"]
    assert "还没有挑选具体项目" in plan["note"]
    assert "编辑每日行程" in plan["note"]


def test_the_summary_line_names_the_categories() -> None:
    summary = format_budget_plan_summary(_plan())

    assert "酒店" in summary and "美食" in summary
    assert "24000" in summary


def test_a_missing_framework_has_no_summary_line() -> None:
    assert format_budget_plan_summary(None) == ""
    assert format_budget_plan_summary({"status": "unavailable"}) == ""


# --------------------------------------------------------------------------------------
# It must never be confused with the exact sums
# --------------------------------------------------------------------------------------


def test_the_framework_is_not_the_saved_place_total() -> None:
    """The two numbers answer different questions and must stay different objects."""
    saved_total = estimate_budget([], 6, 30000, None)
    plan = _plan()

    assert saved_total.total_estimate != plan["planned_total"]
    # Nothing ticked: the exact total is the transport baseline, and the framework is 80%.
    assert plan["planned_total"] == 24000
    assert saved_total.total_estimate < 24000


def test_nothing_about_the_framework_leaks_into_the_exact_total() -> None:
    """The saved-place estimate must not gain a category framework's amounts."""
    saved_total = estimate_budget([], 6, 30000, None)

    assert saved_total.excluded_allowances
    assert all(line["amount"] > 0 for line in saved_total.excluded_allowances)
    # Those reference amounts are the old fixed guess (300 a night / 80 a person a day) and are
    # deliberately unrelated to the framework, which is why the panel no longer leads with them.
    assert not any("总预算" in line["name"] for line in saved_total.excluded_allowances)


# --------------------------------------------------------------------------------------
# Wiring: the planner node and the composed response
# --------------------------------------------------------------------------------------


def _trip() -> dict:
    return {
        "id": 7,
        "destination": "日本东京",
        "start_date": "2026-09-20",
        "end_date": "2026-09-25",
        "budget": 30000,
    }


def test_the_planner_node_builds_the_framework_from_the_trip_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(planner, "get_weather_forecast", lambda *args, **kwargs: {"status": "fallback", "message": "无"})
    monkeypatch.setattr(budget_plan, "ask_the_model_for_shares", lambda *args, **kwargs: None)

    result = planner._analyze({"trip": _trip(), "places": [], "itinerary": []})

    plan = result["budget_plan"]
    assert plan["status"] == "estimated"
    assert plan["trip_budget"] == 30000
    assert plan["planned_total"] == 24000
    assert plan["days"] == 6
    # The node reports what it did, in the same event log the UI shows.
    assert any(event["tool"] == "budget_planner" for event in result["events"])


def test_the_planner_node_does_not_fail_the_plan_without_a_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(planner, "get_weather_forecast", lambda *args, **kwargs: {"status": "fallback", "message": "无"})
    trip = {**_trip(), "budget": None}

    result = planner._analyze({"trip": trip, "places": [], "itinerary": []})

    assert result["budget_plan"]["status"] == "unavailable"
    assert not any(event["tool"] == "budget_planner" for event in result["events"])


def test_the_composed_response_carries_the_framework() -> None:
    """The response dict is an explicit whitelist: a field added upstream but not repeated here
    silently never reaches the API (Pydantic then fills in its defaults)."""
    plan = _plan(shares=MODEL_ANSWER)
    state = {
        "trip": _trip(),
        "final_summary": "总结",
        "budget": estimate_budget([], 6, 30000, None),
        "budget_plan": plan,
    }

    response = planner._compose(state)["response"]

    assert response["budget_plan"]["planned_total"] == 24000
    assert response["budget_plan"]["trip_budget"] == 30000
    assert len(response["budget_plan"]["categories"]) == 5
    # And the exact sum is still reported separately, unchanged.
    assert response["budget"]["total_estimate"] == state["budget"].total_estimate


def test_the_response_schema_exposes_the_framework() -> None:
    from app.schemas.agent import BudgetPlanRead, TravelPlanRead

    assert "budget_plan" in TravelPlanRead.model_fields
    read = BudgetPlanRead(**{**_plan(shares=MODEL_ANSWER), "days": 6, "nights": 5, "people": 2})
    assert read.status == "estimated"
    assert read.planned_total == 24000
    assert [row.category for row in read.categories][0] == "hotel"
    assert read.categories[0].unit.startswith("约 ")


def test_the_trip_dates_are_what_decide_the_nights() -> None:
    """Six days is five nights; the row's own reading must say so."""
    plan = _plan(days=6)

    assert plan["days"] == 6
    assert plan["nights"] == 5
    assert plan["categories"][0]["unit"].endswith("× 5 晚")
    assert date(2026, 9, 20) and plan["people"] == 2


def test_the_default_shares_are_the_documented_ones() -> None:
    shares = {key: share for key, _label, share in DEFAULT_ALLOCATION_SHARES}

    assert shares == {"hotel": 0.45, "restaurant": 0.25, "attraction": 0.15, "transport": 0.10, "other": 0.05}
    assert sum(shares.values()) == pytest.approx(1.0)
