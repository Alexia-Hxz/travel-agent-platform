"""The generated daily plan must cover the whole trip with a different day every day.

The user's complaint was concrete: "我总不能旅游三四天全吃一家店逛同一个景点吧", so the
guarantees pinned here are that no sight or restaurant is offered twice anywhere in the
plan, and that every day still gets a breakfast, a lunch and a dinner.
"""

from app.agents.workflows import planner


def _trip(start: str = "2026-10-10", end: str = "2026-10-14") -> dict:
    return {"destination": "南京", "start_date": start, "end_date": end, "budget": 8000}


def _pool(days: int = 5) -> list[dict]:
    """Enough candidates for `days` days: two sights and three distinct meals a day."""
    places: list[dict] = [{"name": "宽渡酒店", "type": "hotel", "estimated_cost": 800, "meal_slot": None}]
    for index in range(1, days * 2 + 1):
        places.append({"name": f"景点{index}", "type": "attraction", "estimated_cost": 0, "meal_slot": None})
    for index in range(1, days + 1):
        places.append(
            {"name": f"早餐店{index}", "type": "restaurant", "estimated_cost": 20, "meal_slot": "breakfast"}
        )
    for index in range(1, days * 2 + 1):
        places.append(
            {"name": f"正餐店{index}", "type": "restaurant", "estimated_cost": 60, "meal_slot": "dinner"}
        )
    return places


def _plan(state_overrides: dict | None = None, recommended: list[dict] | None = None):
    state = {"query": "规划南京5天4晚的游玩", "trip": _trip()}
    state.update(state_overrides or {})
    return planner._generated_daily_plan(state, recommended if recommended is not None else _pool())


def _names(day: dict, note: str) -> list[str]:
    return [item["place_name"] for item in day["items"] if item["note"] == note]


def test_the_plan_covers_every_day_of_the_trip() -> None:
    """A 5-day trip produced exactly one day before this, however long it was."""
    plan = _plan()

    assert [day["date"] for day in plan] == [
        "2026-10-10",
        "2026-10-11",
        "2026-10-12",
        "2026-10-13",
        "2026-10-14",
    ]


def test_a_single_day_trip_produces_a_single_day() -> None:
    assert len(_plan({"trip": _trip("2026-10-10", "2026-10-10")}, _pool(1))) == 1


def test_no_place_is_offered_twice_across_the_whole_trip() -> None:
    """The central guarantee: a four-day trip must not eat at one restaurant throughout."""
    plan = _plan()
    every = [item["place_name"] for day in plan for item in day["items"]]

    assert len(every) == len(set(every)), every


def test_every_day_visits_different_sights() -> None:
    plan = _plan()
    sights = [name for day in plan for name in _names(day, "上午主要游览，预留交通与拍照时间")]

    assert len(sights) == len(set(sights))
    assert len(sights) == 5, sights


def test_every_day_has_its_own_breakfast_lunch_and_dinner() -> None:
    plan = _plan()
    breakfasts = [name for day in plan for name in _names(day, "当地早餐")]
    lunches = [name for day in plan for name in _names(day, "当地特色午餐")]
    dinners = [name for day in plan for name in _names(day, "晚餐")]

    assert len(breakfasts) == len(set(breakfasts)) == 5, breakfasts
    assert len(lunches) == len(set(lunches)) == 5, lunches
    assert len(dinners) == len(set(dinners)) == 5, dinners
    # And no venue is shared between the three meals either.
    assert not (set(breakfasts) & set(lunches))
    assert not (set(breakfasts) & set(dinners))
    assert not (set(lunches) & set(dinners))


def test_two_sights_a_day_are_both_offered_and_both_differ() -> None:
    plan = _plan()
    for day in plan:
        morning = _names(day, "上午主要游览，预留交通与拍照时间")
        afternoon = _names(day, "下午继续游览，按现场人流灵活调整")
        assert len(morning) == 1 and len(afternoon) == 1, day["date"]
        assert morning[0] != afternoon[0]


def test_item_times_never_overlap_within_a_day() -> None:
    for day in _plan():
        windows = sorted((item["start_time"], item["end_time"]) for item in day["items"])
        for (_, previous_end), (next_start, _) in zip(windows, windows[1:]):
            assert previous_end <= next_start, f"{day['date']} overlaps at {previous_end} > {next_start}"


def test_a_venue_under_a_longer_name_counts_as_already_used() -> None:
    """`蒋有记` and `老门东内蒋有记` are one restaurant, not two.

    The model returned both, and the plan served the same place for breakfast and again
    for lunch on the same day.
    """
    recommended = [
        {"name": "玄武湖景区", "type": "attraction", "estimated_cost": 0},
        {"name": "蒋有记", "type": "restaurant", "estimated_cost": 20, "meal_slot": "breakfast"},
        {"name": "老门东内蒋有记", "type": "restaurant", "estimated_cost": 60, "meal_slot": "dinner"},
        {"name": "某菜馆", "type": "restaurant", "estimated_cost": 60, "meal_slot": "dinner"},
    ]
    plan = _plan({"trip": _trip("2026-10-10", "2026-10-10")}, recommended)

    names = [item["place_name"] for item in plan[0]["items"]]
    assert "蒋有记" in names
    assert "老门东内蒋有记" not in names
    # The genuinely different venue is still used.
    assert "某菜馆" in names


def test_same_venue_recognises_one_place_under_two_names() -> None:
    assert planner._same_venue("蒋有记", "老门东内蒋有记")
    assert planner._same_venue("蒋有记", "蒋有记")
    assert planner._same_venue("Luk Yu Tea House", "luk yu tea house")


def test_same_venue_keeps_numbered_candidates_apart() -> None:
    """`景点1` and `景点10` are two candidates, not one venue under a longer name."""
    assert not planner._same_venue("景点1", "景点10")
    assert not planner._same_venue("早餐店1", "早餐店12")
    assert not planner._same_venue("玄武湖景区", "南京博物院")
    assert not planner._same_venue("", "某菜馆")


def test_a_hotel_name_containing_a_sight_name_does_not_block_the_sight() -> None:
    """`夫子庙步行街轻居酒店` must not consume the sight `夫子庙`.

    Comparing names across kinds meant the hotel silently removed a whole attraction
    from the trip: only seven of the eight sights were ever used.
    """
    recommended = [
        {"name": "夫子庙步行街轻居酒店", "type": "hotel", "estimated_cost": 2800},
        {"name": "夫子庙", "type": "attraction", "estimated_cost": 0},
        {"name": "玄武湖景区", "type": "attraction", "estimated_cost": 0},
    ]
    plan = _plan({"trip": _trip("2026-10-10", "2026-10-10")}, recommended)

    names = [item["place_name"] for item in plan[0]["items"]]
    assert "夫子庙" in names
    assert "玄武湖景区" in names


def test_a_street_matching_a_sight_visited_that_day_is_not_served_as_dinner() -> None:
    """`夫子庙` in the morning and `夫子庙步行街` at dinner is the same place twice."""
    recommended = [
        {"name": "夫子庙", "type": "attraction", "estimated_cost": 0},
        {"name": "夫子庙步行街", "type": "restaurant", "estimated_cost": 80, "meal_slot": "dinner"},
        {"name": "某菜馆", "type": "restaurant", "estimated_cost": 60, "meal_slot": "dinner"},
    ]
    plan = _plan({"trip": _trip("2026-10-10", "2026-10-10")}, recommended)

    names = [item["place_name"] for item in plan[0]["items"]]
    assert "夫子庙" in names
    assert "夫子庙步行街" not in names
    assert "某菜馆" in names


def test_a_restaurant_inside_a_named_hotel_is_still_served() -> None:
    """A restaurant whose name embeds a hotel is a place to eat, not the hotel again."""
    recommended = [
        {"name": "金陵饭店", "type": "hotel", "estimated_cost": 2000},
        {"name": "金陵饭店（芳芳法国餐厅）", "type": "restaurant", "estimated_cost": 300, "meal_slot": "dinner"},
    ]
    plan = _plan({"trip": _trip("2026-10-10", "2026-10-10")}, recommended)

    names = [item["place_name"] for item in plan[0]["items"]]
    assert "金陵饭店（芳芳法国餐厅）" in names


def test_the_hotel_is_billed_for_the_stay_and_says_so() -> None:
    """The room rate covers every night, so day one must not imply it is a daily cost."""
    first_day = _plan()[0]
    hotel_items = [item for item in first_day["items"] if item["place_name"] == "宽渡酒店"]

    assert len(hotel_items) == 1
    assert "全程 4 晚的合计" in hotel_items[0]["note"]
    assert "不是当天消费" in hotel_items[0]["note"]


def test_the_hotel_only_appears_on_the_first_day() -> None:
    for day in _plan()[1:]:
        assert all(item["place_name"] != "宽渡酒店" for item in day["items"])


def test_a_pool_too_small_for_the_trip_omits_slots_and_says_why() -> None:
    """Repeating is worse than admitting the pool is too small.

    Two restaurants cannot feed a five-day trip, so the later days lose their meals
    rather than serving the same two places over and over.
    """
    recommended = [
        {"name": "玄武湖景区", "type": "attraction", "estimated_cost": 0},
        {"name": "南京博物院", "type": "attraction", "estimated_cost": 0},
        {"name": "蓮湖糕團店", "type": "restaurant", "estimated_cost": 300, "meal_slot": "breakfast"},
        {"name": "韓復興", "type": "restaurant", "estimated_cost": 525, "meal_slot": "lunch"},
    ]
    plan = _plan(recommended=recommended)
    every = [item["place_name"] for day in plan for item in day["items"]]

    # Nothing is repeated anywhere.
    assert len(every) == len(set(every)), every
    # Day one gets both sights and the two meals that exist.
    assert _names(plan[0], "当地早餐") == ["蓮湖糕團店"]
    assert _names(plan[0], "当地特色午餐") == ["韓復興"]
    assert sorted(_names(plan[0], "上午主要游览，预留交通与拍照时间")) == ["玄武湖景区"]
    # Dinner is what day one is missing, and it says so instead of reusing a venue.
    assert "晚餐" in plan[0]["shortage"]
    assert "请再多生成一次规划建议补充候选" in plan[0]["shortage"]
    # The later days have nothing left to draw on and stay visible with a reason.
    assert len(plan) == 5
    assert plan[4]["items"] == []
    assert plan[4]["shortage"] is not None


def test_a_shortage_names_the_meals_that_are_missing() -> None:
    recommended = [{"name": "玄武湖景区", "type": "attraction", "estimated_cost": 0}]
    plan = _plan(recommended=recommended)

    # Day one takes the only sight; the meals are what it cannot fill.
    assert _names(plan[0], "上午主要游览，预留交通与拍照时间") == ["玄武湖景区"]
    assert "早餐" in plan[0]["shortage"]
    assert "午餐" in plan[0]["shortage"]
    assert "晚餐" in plan[0]["shortage"]
    # Every day of the trip is still listed, so nothing silently disappears.
    assert len(plan) == 5
    assert plan[1]["items"] == []
    assert "景点" in plan[1]["shortage"]


def test_a_complete_pool_reports_no_shortage() -> None:
    assert all(day["shortage"] is None for day in _plan())


def test_breakfast_falls_back_to_general_options_when_none_are_labelled() -> None:
    """A morning slot must still be filled when the model labelled nothing as breakfast."""
    recommended = [
        {"name": "玄武湖景区", "type": "attraction", "estimated_cost": 0},
        {"name": "南京博物院", "type": "attraction", "estimated_cost": 0},
        {"name": "某菜馆", "type": "restaurant", "estimated_cost": 60, "meal_slot": "any"},
        {"name": "另一菜馆", "type": "restaurant", "estimated_cost": 60, "meal_slot": "any"},
    ]
    plan = _plan(recommended=recommended)

    assert _names(plan[0], "当地早餐") == ["某菜馆"]


def test_no_plan_is_generated_when_the_query_is_unrelated() -> None:
    assert _plan({"query": "南京的天气怎么样"}) == []


def test_no_plan_is_generated_without_recommendations() -> None:
    assert _plan(recommended=[]) == []


def test_an_unusable_date_range_yields_no_plan_instead_of_crashing() -> None:
    assert _plan({"trip": {"destination": "南京", "start_date": "not-a-date", "end_date": "2026-10-14"}}) == []
    assert _plan({"trip": {"destination": "南京", "start_date": "2026-10-14", "end_date": "2026-10-10"}}) == []


def test_only_attractions_still_produce_a_usable_day() -> None:
    recommended = [
        {"name": "玄武湖景区", "type": "attraction", "estimated_cost": 0},
        {"name": "南京博物院", "type": "attraction", "estimated_cost": 0},
    ]
    plan = _plan(recommended=recommended)

    assert len(plan) == 5
    assert [item["place_name"] for item in plan[0]["items"]] == ["玄武湖景区", "南京博物院"]
    # The days with no sights left say so rather than repeating the two above.
    assert plan[1]["items"] == []
    assert "景点" in plan[1]["shortage"]
