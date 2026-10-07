"""Tests for the destination price bands, the local defaults and the tier spread.

The complaint these exist for, in one line: a **sushi restaurant in Tokyo was shown at 53 CNY
per person**. 53 CNY is about 1,100 yen -- a conveyor-belt lunch, not a sit-down sushi dinner.
Nothing in the pipeline knew the difference, because "restaurant" was one category with one
default (80 CNY, a Chinese mid-range meal) and one flat sanity range (10-2000).

So the tests come in three groups:
* the bands themselves -- are the numbers plausible, per country, per kind of place;
* the gate that uses them -- a price below the local floor is raised, labelled, and never
  silently mixed in with quoted prices;
* the tier spread -- the budget has to produce cheap AND expensive-but-good options, and a
  thin tier has to be looked for by name rather than hoped for.
"""

from __future__ import annotations

import pytest

from app.agents.tools import meal_prices
from app.agents.tools.recommendations import (
    budget_allowances,
    estimate_trip_costs,
    extract_recommendations,
)


def _restaurant(name: str, unit_price: int | None, hint: str | None = None, **extra: object) -> dict:
    item = {
        "name": name,
        "type": "restaurant",
        "address": "日本东京",
        "price_hint": hint,
        "unit_price": unit_price,
        "price_unit": "person_meal",
    }
    item.update(extra)
    return item


# --------------------------------------------------------------------------------------
# The bands.
# --------------------------------------------------------------------------------------
def test_a_sushi_dinner_in_japan_costs_more_than_a_noodle_lunch():
    sushi = meal_prices.meal_band("日本5日游", "大和寿司（本店）")
    ramen = meal_prices.meal_band("日本5日游", "一兰拉面（新宿本店）")
    assert sushi.low > ramen.low
    assert sushi.typical > ramen.typical
    # A proper sushi dinner in Tokyo starts around 3,500 yen, which is ~168 CNY; the 53 CNY the
    # card used to show is the price of a conveyor-belt lunch, not a meal here.
    assert sushi.low >= 120
    # A ramen bowl is 700-1,500 yen: ~34-72 CNY, and it does fill you up.
    assert 30 <= ramen.typical <= 70


def test_a_conveyor_belt_meal_is_priced_from_the_chains_own_plate_prices():
    # 6-10 plates at 110-210 yen each is 1,000-2,300 yen, so ~55 CNY is a typical filling kaiten
    # meal and a price in the 50s is a real price -- it is only wrong on a *sushi restaurant*.
    kaiten = meal_prices.meal_band("日本5日游", "スシロー 浅草店")
    assert 25 <= kaiten.low <= 40
    assert 45 <= kaiten.typical <= 65
    assert kaiten.typical < meal_prices.meal_band("日本5日游", "大和寿司（本店）").typical


def test_the_kind_of_place_decides_the_price_not_just_the_category():
    assert meal_prices.meal_kind("スシロー 浅草店") == "sushi_kaiten"
    assert meal_prices.meal_kind("数寄屋橋次郎") == "sushi_fine"
    assert meal_prices.meal_kind("大和寿司（本店）") == "sushi_mid"
    assert meal_prices.meal_kind("一兰拉面（新宿本店）") == "ramen_noodles"
    assert meal_prices.meal_kind("鸟贵族 新宿店") == "izakaya_bar"
    assert meal_prices.meal_kind("夫子庙小吃") == "street_food"
    assert meal_prices.meal_kind("Blue Bottle Coffee 清澄白河") == "cafe_dessert"
    assert meal_prices.meal_kind("某家没有线索的店") == meal_prices.GENERIC_KIND


def test_the_more_specific_kind_wins():
    # "回転寿司" must not be read as the general "寿司" row, and a Michelin note must beat it too.
    assert meal_prices.meal_kind("回転寿司 函館市場") == "sushi_kaiten"
    assert meal_prices.meal_kind("鮨 さいとう（米其林）") == "sushi_fine"


@pytest.mark.parametrize(
    "destination",
    ["日本5日游", "东京", "大阪", "北海道", "日本", "韩国首尔", "曼谷", "巴黎", "纽约", "伦敦"],
)
def test_every_real_destination_has_its_own_bands(destination: str):
    band = meal_prices.meal_band(destination, "某餐厅")
    assert band.low <= band.typical <= band.high
    assert band.low > 0


def test_destination_country_survives_a_trip_title_and_a_bare_city():
    assert meal_prices.destination_country("日本5日游") == "日本"
    assert meal_prices.destination_country("东京") == "日本"
    assert meal_prices.destination_country("大阪") == "日本"
    assert meal_prices.destination_country("南京") == "中国"


def test_an_unlisted_country_is_priced_by_its_price_level_not_by_china():
    # Iceland has no row of its own; it is one of the most expensive places to eat in Europe.
    expensive = meal_prices.meal_band("冰岛", "某某")
    cheap = meal_prices.meal_band("柬埔寨", "某某")
    china = meal_prices.meal_band("南京", "某某")
    assert expensive.typical > china.typical > cheap.typical
    assert meal_prices.country_key("冰岛") == "level:expensive"
    assert meal_prices.country_key("柬埔寨") == "level:cheap"
    assert meal_prices.hotel_band("冰岛").typical > meal_prices.hotel_band("南京").typical


def test_an_unknown_destination_falls_back_to_the_chinese_mid_range():
    # The historical behaviour, kept for the case where nothing can be told about the place: the
    # Chinese row is the last resort. Its measured typical meal is 60 CNY (a composite of Beijing,
    # Shanghai, Nanjing and Chengdu).
    assert meal_prices.meal_band(None, "某某").typical == meal_prices._FALLBACK_MEALS["generic"].typical
    assert meal_prices.default_unit_price(None, "restaurant") == 60
    assert meal_prices.country_key(None) is None


def test_the_hotel_and_ticket_bands_move_with_the_destination():
    assert meal_prices.hotel_band("日本5日游").typical > meal_prices.hotel_band("曼谷").typical
    assert meal_prices.hotel_band("苏黎世").typical > meal_prices.hotel_band("曼谷").typical
    # A museum can be free, and the band says so: the low end is 0 for every destination.
    assert meal_prices.attraction_band("日本5日游").low == 0
    assert meal_prices.attraction_band("纽约").typical > meal_prices.attraction_band("曼谷").typical


def test_the_default_is_local_and_beats_the_old_flat_eighty():
    # 5,500 yen is 264 CNY: the sourced Tokyo sushi-restaurant figure, not the old flat 80.
    assert meal_prices.default_unit_price("日本5日游", "restaurant", "大和寿司（本店）") == 264
    # An unknown kind of place in Japan is the generic row, 1,600 yen ~ 77 CNY -- near the old
    # flat 80 but measured rather than assumed. The kind of place is what moves the number most.
    assert meal_prices.default_unit_price("日本5日游", "restaurant") == 77
    assert meal_prices.default_unit_price("日本5日游", "restaurant", "大和寿司（本店）") > 80
    # A Tokyo hotel night is 22,000 yen ~ 1,056 CNY, against the old flat 400.
    assert meal_prices.default_unit_price("日本5日游", "hotel") > 400
    # China is a measured row too now: a plain Nanjing meal is 45 CNY and a hotel night 550.
    assert meal_prices.default_unit_price("南京", "restaurant", "某小馆") == 45
    assert meal_prices.default_unit_price("南京", "hotel") == 550


def test_the_prompt_tells_the_model_what_a_meal_costs_there():
    hint = meal_prices.price_hint_for_prompt("日本5日游")
    assert "回转/平价寿司" in hint
    assert "正经寿司店" in hint
    assert "高级寿司" in hint
    # And that a single dish is not a meal.
    assert "一个人吃一顿正常份量" in hint
    assert meal_prices.price_hint_for_prompt(None) == ""


# --------------------------------------------------------------------------------------
# The gate that uses them.
# --------------------------------------------------------------------------------------
def test_a_fifty_three_yuan_sushi_dinner_in_tokyo_is_raised_and_says_so():
    items = estimate_trip_costs(
        [_restaurant("大和寿司（本店）", 53, "人均 1100 日元")], 6, {"total_people": 2}, 30000,
        destination="日本东京",
    )
    card = items[0]
    # 5,500 yen (the sourced mid-range sushi dinner in Tokyo) is 264 CNY; the floor for a proper
    # sushi restaurant is 3,500 yen = 168 CNY. Either way 53 CNY is not a meal here.
    assert card["unit_price"] == 168
    assert card["price_adjusted"] is True
    assert "常见人均下限" in card["cost_source"]
    # The adjustment is never confused with a quoted price.
    assert "按检索到的价格估算" not in card["cost_source"]


def test_a_genuine_cheap_meal_is_left_alone():
    items = estimate_trip_costs(
        [
            _restaurant("一兰拉面（新宿本店）", 58, "人均 1200 日元"),
            _restaurant("夫子庙小吃", 15, "人均 15 元", address="南京"),
        ],
        6,
        {"total_people": 2},
        30000,
        destination="日本东京",
    )
    assert items[0]["unit_price"] == 58
    assert items[0]["price_adjusted"] is False
    assert items[1]["unit_price"] == 15
    assert items[1]["price_adjusted"] is False


def test_a_conveyor_belt_price_is_left_exactly_as_it_is():
    items = estimate_trip_costs(
        [_restaurant("スシロー 浅草店", 53, "人均 1100 日元")], 6, {"total_people": 2}, 30000,
        destination="日本东京",
    )
    # 53 CNY is 1,100 yen, which is 6-10 plates at a conveyor-belt chain: a real, filling price.
    # Nothing is corrected, which is the whole point of pricing by kind of place rather than by
    # country alone.
    assert items[0]["unit_price"] == 53
    assert items[0]["price_adjusted"] is False
    assert items[0]["cost_source"] == "按检索到的价格估算"


def test_an_expensive_omakase_is_kept_rather_than_replaced_by_the_default():
    # 25,000 yen is 1,200 CNY: above the flat 2,000 bound? No -- but it is above the old
    # mid-range default, and the point is that a real high price survives the sanity check.
    items = estimate_trip_costs(
        [_restaurant("数寄屋橋次郎", 1200, "人均 25000 日元")], 6, {"total_people": 2}, 30000,
        destination="日本东京",
    )
    assert items[0]["unit_price"] == 1200
    assert "按检索到的价格估算" in items[0]["cost_source"]


def test_a_free_attraction_is_still_free():
    items = estimate_trip_costs(
        [{"name": "上野公园", "type": "attraction", "address": "日本东京", "price_hint": "免费", "unit_price": 0, "price_unit": "ticket", "is_free": True}],
        6,
        {"total_people": 2},
        30000,
        destination="日本东京",
    )
    assert items[0]["unit_price"] == 0
    assert items[0]["cost_source"] == "资料显示免费"


def test_the_old_flat_default_never_reaches_a_foreign_card():
    items = estimate_trip_costs(
        [
            _restaurant("没有价格的寿司店", None),
            {"name": "没有价格的酒店", "type": "hotel", "address": "日本东京", "price_hint": None, "unit_price": None, "price_unit": "night"},
        ],
        6,
        {"total_people": 2},
        30000,
        destination="日本东京",
    )
    restaurants = [item for item in items if item["type"] == "restaurant"][0]
    hotels = [item for item in items if item["type"] == "hotel"][0]
    assert restaurants["unit_price"] == 264
    assert hotels["unit_price"] == 1056
    assert restaurants["cost_source"] == "按当地中档常见价估算"


def test_the_cost_gate_still_works_without_a_destination_argument():
    # Every existing caller passes no destination; that path must stay the documented one. The
    # item has no address of its own either, so nothing can be told about where it is.
    items = estimate_trip_costs(
        [{"name": "某某", "type": "restaurant", "price_hint": None, "unit_price": None, "price_unit": "person_meal"}],
        3,
        {"total_people": 1},
        3000,
    )
    assert items[0]["unit_price"] == 60


# --------------------------------------------------------------------------------------
# The tiers: cheap, middle, and expensive-but-good.
# --------------------------------------------------------------------------------------
def test_the_budget_decides_the_three_price_points():
    plan = meal_prices.tier_plan(300, floor=144)
    assert plan is not None
    assert plan.allowance == 300
    assert plan.cheap_max < plan.allowance < plan.mid_max
    assert plan.splurge_min > plan.mid_max
    assert plan.splurge_max > plan.splurge_min
    assert plan.tier_of(100) == meal_prices.TIER_CHEAP
    assert plan.tier_of(300) == meal_prices.TIER_MID
    assert plan.tier_of(600) == meal_prices.TIER_SPLURGE
    # Past the ceiling it is not "worth paying more", it is simply unaffordable.
    assert plan.tier_of(5000) == meal_prices.TIER_OVER


def test_a_bigger_budget_moves_the_expensive_tier_up():
    small = meal_prices.tier_plan(120, floor=58)
    large = meal_prices.tier_plan(600, floor=144)
    assert large.splurge_min > small.splurge_min
    assert large.cheap_max > small.cheap_max


def test_no_allowance_means_no_plan_and_no_tiers():
    assert meal_prices.tier_plan(0) is None
    assert meal_prices.tier_plan(None) is None  # type: ignore[arg-type]
    assert meal_prices.tier_brief(None) is None


def test_the_tier_brief_asks_for_both_ends():
    brief = meal_prices.tier_brief(meal_prices.tier_plan(300, floor=144), "restaurant")
    assert brief is not None
    assert "三个档次" in brief
    assert "平价档" in brief and "高档档" in brief
    assert "不要只给便宜的，也不要只给贵的" in brief


def test_every_card_carries_its_tier():
    items = estimate_trip_costs(
        [
            _restaurant("一兰拉面（新宿本店）", 58, "人均 1200 日元"),
            _restaurant("大和寿司（本店）", 240, "人均 5000 日元"),
            _restaurant("数寄屋橋次郎", 1200, "人均 25000 日元"),
        ],
        6,
        {"total_people": 2},
        30000,
        destination="日本东京",
    )
    labels = {item["name"]: item["price_tier_label"] for item in items}
    assert labels["一兰拉面（新宿本店）"] == "平价"
    assert labels["大和寿司（本店）"] in ("中档", "值得多花钱")
    assert labels["数寄屋橋次郎"] == "超出预算"


def test_an_item_inside_its_tier_is_never_flagged_over_budget():
    """The 超预算 flag and the tier badge must not contradict each other on one card.

    Seen on a live Tokyo plan: 筑地市场玉寿司 at 182 CNY against a 166 CNY meal allowance came
    back labelled 中档 ("affordable middle") and flagged over budget in the same breath. The
    ceiling is now the top of the tier the category still calls affordable, so the two agree.
    """
    allowance = budget_allowances(30000, 6, {"total_people": 2})["restaurant"]
    plan = meal_prices.tier_plan(allowance, floor=0)
    assert plan is not None
    items = estimate_trip_costs(
        [
            _restaurant("略高于人均的寿司店", plan.allowance + 10, None),
            _restaurant("刚好在中档上限", plan.mid_max, None),
        ],
        6,
        {"total_people": 2},
        30000,
        destination="日本东京",
    )
    for item in items:
        assert item["price_tier"] == meal_prices.TIER_MID
        assert item["over_budget"] is False
        assert item["budget_note"] == ""


def test_the_deliberate_splurge_is_not_called_over_budget_and_anything_past_it_is():
    allowance = budget_allowances(30000, 6, {"total_people": 2})["restaurant"]
    plan = meal_prices.tier_plan(allowance, floor=0)
    assert plan is not None
    items = estimate_trip_costs(
        [
            _restaurant("值得多花钱的寿司", plan.splurge_min, None),
            _restaurant("贵得离谱的店", plan.splurge_max + 500, None),
        ],
        6,
        {"total_people": 2},
        30000,
        destination="日本东京",
    )
    by_name = {item["name"]: item for item in items}
    assert by_name["值得多花钱的寿司"]["over_budget"] is False
    assert "值得多花钱" in by_name["值得多花钱的寿司"]["budget_note"]
    assert by_name["贵得离谱的店"]["over_budget"] is True
    assert "超出预算参考" in by_name["贵得离谱的店"]["budget_note"]


def test_the_spread_is_measured_and_the_missing_tier_is_named():
    plan = meal_prices.tier_plan(300, floor=144)
    items = [_restaurant(f"便宜店{n}", 100) for n in range(3)]
    counts = meal_prices.ensure_tier_spread(items, plan, "restaurant")
    assert counts[meal_prices.TIER_CHEAP] == 3
    assert counts[meal_prices.TIER_SPLURGE] == 0
    missing = meal_prices.missing_tiers(counts, 3)
    assert meal_prices.TIER_SPLURGE in missing
    assert "米其林" in meal_prices.tier_search_hint(meal_prices.TIER_SPLURGE)


def test_the_missing_tier_search_actually_appends_only_that_tier(monkeypatch: pytest.MonkeyPatch):
    """The backfill is the part that makes the mix real, so it is tested against a stub search."""
    from app.agents.tools import recommendations

    calls: list[tuple[str, str]] = []

    def fake_fill(destination, category, shortfall, allowances=None, plan=None, search_hint=None):
        calls.append((category, search_hint or ""))
        if "米其林" in (search_hint or ""):
            # The splurge band for this trip is 450-625 CNY per person, so a "find" has to sit
            # inside it to count: a 40,000 yen omakase is a different (unaffordable) statement.
            return [
                {"name": "鮨 さいとう", "type": "restaurant", "address": destination,
                 "price_hint": "人均 12000 日元", "unit_price": 560, "price_unit": "person_meal"},
                # And a mid-range place that this search did not ask for: it must be dropped.
                {"name": "普通寿司店", "type": "restaurant", "address": destination,
                 "price_hint": "人均 4000 日元", "unit_price": 192, "price_unit": "person_meal"},
            ]
        return []

    monkeypatch.setattr(recommendations, "_fill_category", fake_fill)
    items = [_restaurant(f"便宜店{n}", 100) for n in range(4)]
    grown, counts = recommendations.ensure_price_spread(
        "日本东京", items, 30000, 6, {"total_people": 2}
    )
    assert [item["name"] for item in grown][-1] == "鮨 さいとう"
    assert "普通寿司店" not in [item["name"] for item in grown]
    assert counts[meal_prices.TIER_SPLURGE] == 1
    assert any("米其林" in hint for _category, hint in calls)


def test_a_splurge_find_beyond_the_band_is_not_kept_as_a_splurge(monkeypatch: pytest.MonkeyPatch):
    """A 1,920 CNY dinner is not "one splurge" on a 250 CNY-per-meal budget, so it is refused."""
    from app.agents.tools import recommendations

    monkeypatch.setattr(
        recommendations,
        "_fill_category",
        lambda *args, **kwargs: [
            {"name": "贵得不像话", "type": "restaurant", "address": "日本东京",
             "price_hint": "人均 40000 日元", "unit_price": 1920, "price_unit": "person_meal"}
        ],
    )
    items = [_restaurant(f"便宜店{n}", 100) for n in range(4)]
    grown, counts = recommendations.ensure_price_spread("日本东京", items, 30000, 6, {"total_people": 2})
    assert "贵得不像话" not in [item["name"] for item in grown]
    assert counts[meal_prices.TIER_SPLURGE] == 0


def test_no_tier_backfill_without_a_budget(monkeypatch: pytest.MonkeyPatch):
    from app.agents.tools import recommendations

    def fail_fill(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("nothing may be searched when the trip has no budget")

    monkeypatch.setattr(recommendations, "_fill_category", fail_fill)
    items = [_restaurant("某餐厅", 100)]
    grown, counts = recommendations.ensure_price_spread("日本东京", items, None, 6, None)
    assert grown == items
    assert counts == {}


def test_the_extraction_is_told_the_local_bands_and_the_tiers(monkeypatch: pytest.MonkeyPatch):
    """Both prompts that build the pool must carry the price reality, not just the focused one."""
    from app.agents.tools import recommendations

    seen: list[str] = []

    class FakeCompletions:
        def create(self, **kwargs):
            seen.append(kwargs["messages"][0]["content"])

            class _Message:
                content = "[]"

            class _Choice:
                message = _Message()
                finish_reason = "stop"

            class _Response:
                choices = [_Choice()]

            return _Response()

    class FakeClient:
        chat = type("_Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(recommendations, "model_client", lambda: FakeClient())
    recommendations._extract_with_model(
        "日本东京", [{"title": "t", "snippet": "x" * 60}], {"restaurant": 166}, {"restaurant": 4}
    )
    assert seen and "回转/平价寿司" in seen[0]
    assert "三个档次" in seen[0]

    seen.clear()
    recommendations._extract_category_with_model(
        "日本东京",
        "restaurant",
        [{"title": "t", "snippet": "x" * 60}],
        {"restaurant": 166},
        4,
        "dinner",
        (),
        meal_prices.tier_plan(166, floor=144),
    )
    assert seen and "正经寿司店" in seen[0]
    assert "三个档次" in seen[0]


def test_a_destination_with_no_table_still_produces_a_list_without_a_model(monkeypatch: pytest.MonkeyPatch):
    """The offline path must not raise on the new pricing code."""
    from app.agents.tools import recommendations

    monkeypatch.setattr(recommendations, "model_client", lambda: None)
    items, status = extract_recommendations(
        "冰岛雷克雅未克",
        [{"title": "Reykjavik", "snippet": "Fish and chips at Icelandic Street Food for 1,500 ISK. " * 4}],
    )
    assert status in ("fallback", "completed", "skipped")
    assert isinstance(items, list)
