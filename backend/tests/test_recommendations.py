import json

from app.agents.tools import recommendations


GUIDE_RESULTS = [
    {
        "source_type": "tavily",
        "title": "南京旅游攻略",
        "url": "https://example.com/guide",
        "snippet": "夫子庙适合夜游秦淮河。南京大牌檔(德基广场店)人均约80元。金陵饭店房价约600元。",
    }
]


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str, finish_reason: str | None = None) -> None:
        self.message = _FakeMessage(content)
        self.finish_reason = finish_reason


class _FakeResponse:
    def __init__(self, content: str, finish_reason: str | None = None) -> None:
        self.choices = [_FakeChoice(content, finish_reason)]


class _FakeCompletions:
    def __init__(self, content: str, finish_reason: str | None = None) -> None:
        self.content = content
        self.finish_reason = finish_reason
        self.calls = 0
        self.last_kwargs = None

    def create(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        return _FakeResponse(self.content, self.finish_reason)


def test_all_recommendation_extraction_calls_disable_thinking(monkeypatch) -> None:
    array_client = _client_returning([])
    monkeypatch.setattr(recommendations, "model_client", lambda: array_client)
    recommendations._extract_with_model("东京", [{"title": "资料", "snippet": "内容" * 40}])
    assert array_client.chat.completions.last_kwargs["extra_body"] == {"enable_thinking": False}

    recommendations._extract_category_with_model(
        "东京", "attraction", [{"title": "资料", "snippet": "内容" * 40}]
    )
    assert array_client.chat.completions.last_kwargs["extra_body"] == {"enable_thinking": False}

    object_client = _client_returning({})
    monkeypatch.setattr(recommendations, "model_client", lambda: object_client)
    recommendations._extract_prices_with_model(
        "东京", [{"name": "浅草寺", "category": "attraction", "material": "免费"}]
    )
    assert object_client.chat.completions.last_kwargs["extra_body"] == {"enable_thinking": False}


class _FakeChat:
    def __init__(self, content: str, finish_reason: str | None = None) -> None:
        self.completions = _FakeCompletions(content, finish_reason)


class _FakeClient:
    """Minimal stand-in for the OpenAI chat client."""

    def __init__(self, content: str, finish_reason: str | None = None) -> None:
        self.chat = _FakeChat(content, finish_reason)


def _client_returning(payload: object) -> _FakeClient:
    return _FakeClient(json.dumps(payload, ensure_ascii=False))


def test_ensure_category_coverage_backfills_a_missing_category(monkeypatch) -> None:
    """A user asking for hotels, food and sights must get all three."""
    monkeypatch.setattr(
        recommendations,
        "search_snippets",
        lambda query, limit=5, timeout=8.0: ["和平饭店位于外滩旁，房价约 1200 元每晚，是上海经典老牌酒店。"],
    )
    monkeypatch.setattr(
        recommendations,
        "model_client",
        lambda: _client_returning([{"name": "和平饭店", "category": "hotel", "reason": "外滩旁", "unit_price": 1200}]),
    )

    items = [{"name": "外滩", "type": "attraction", "unit_price": None}]
    filled = recommendations.ensure_category_coverage("上海", items)

    categories = [item["type"] for item in filled]
    assert categories.count("attraction") == 1
    assert "hotel" in categories
    hotel = next(item for item in filled if item["type"] == "hotel")
    assert hotel["name"] == "和平饭店"
    assert hotel["price_unit"] == "night"
    assert hotel["unit_price"] == 1200


def test_ensure_category_coverage_does_nothing_when_all_categories_are_present(monkeypatch) -> None:
    def _fail(*args, **kwargs):
        raise AssertionError("no search or model call is needed when every category is covered")

    monkeypatch.setattr(recommendations, "search_snippets", _fail)
    monkeypatch.setattr(recommendations, "model_client", _fail)

    items = [
        {"name": "某酒店", "type": "hotel"},
        {"name": "某餐厅", "type": "restaurant"},
        {"name": "某景点", "type": "attraction"},
    ]
    assert recommendations.ensure_category_coverage("上海", items) == items


class _PerCategoryCompletions:
    """Answer each focused extraction differently, keyed by the asked category."""

    def __init__(self, by_category: dict[str, list]) -> None:
        self.by_category = by_category
        self.asked: list[str] = []

    def create(self, **kwargs):
        payload = json.loads(kwargs["messages"][1]["content"])
        category = payload.get("category")
        self.asked.append(category)
        return _FakeResponse(json.dumps(self.by_category.get(category, []), ensure_ascii=False))


def _per_category_client(by_category: dict[str, list]) -> _FakeClient:
    client = _FakeClient("[]")
    client.chat.completions = _PerCategoryCompletions(by_category)
    return client


def test_ensure_category_coverage_trusts_the_searched_category_over_the_model_label(monkeypatch) -> None:
    monkeypatch.setattr(
        recommendations,
        "search_snippets",
        lambda query, limit=5, timeout=8.0: ["外滩源是外滩附近的景点，适合傍晚漫步和拍照打卡。"],
    )
    # Asked for sights, the model still labels the result a restaurant; the card
    # must land in the sights group anyway.
    client = _per_category_client({"attraction": [{"name": "外滩源", "category": "restaurant", "reason": "适合漫步"}]})
    monkeypatch.setattr(recommendations, "model_client", lambda: client)

    filled = recommendations.ensure_category_coverage("上海", [{"name": "某酒店", "type": "hotel"}])

    # Sights are asked for once. Restaurants are gathered in more than one batch — one for
    # main meals and one for breakfast — so they may appear more than once here.
    assert client.chat.completions.asked.count("attraction") == 1
    assert "restaurant" in client.chat.completions.asked
    added = [item for item in filled if item["name"] == "外滩源"]
    assert len(added) == 1
    assert added[0]["type"] == "attraction"


def test_ensure_category_coverage_never_adds_the_same_place_twice(monkeypatch) -> None:
    monkeypatch.setattr(
        recommendations,
        "search_snippets",
        lambda query, limit=5, timeout=8.0: ["外滩源是外滩附近的景点，适合傍晚漫步和拍照打卡。"],
    )
    # Both focused lookups return the same place; it must appear only once.
    client = _per_category_client(
        {
            "restaurant": [{"name": "外滩源", "category": "restaurant", "reason": "适合漫步"}],
            "attraction": [{"name": "外滩源", "category": "attraction", "reason": "适合漫步"}],
        }
    )
    monkeypatch.setattr(recommendations, "model_client", lambda: client)

    filled = recommendations.ensure_category_coverage("上海", [{"name": "某酒店", "type": "hotel"}])

    assert [item["name"] for item in filled].count("外滩源") == 1


def test_ensure_category_coverage_skips_categories_without_material(monkeypatch) -> None:
    monkeypatch.setattr(recommendations, "search_snippets", lambda query, limit=5, timeout=8.0: [])
    monkeypatch.setattr(recommendations, "model_client", lambda: _client_returning([{"name": "x", "category": "hotel"}]))

    items = [{"name": "外滩", "type": "attraction"}]
    filled = recommendations.ensure_category_coverage("上海", items)

    assert [item["type"] for item in filled] == ["attraction"]


def test_extract_recommendations_keeps_grounded_places_only(monkeypatch) -> None:
    monkeypatch.setattr(
        recommendations,
        "model_client",
        lambda: _client_returning(
            [
                {"name": "夫子庙", "category": "attraction", "reason": "适合夜游秦淮河", "price_hint": None},
                {"name": "南京大牌檔(德基广场店)", "category": "restaurant", "reason": "本地菜", "price_hint": "人均约80元"},
                {"name": "金陵饭店", "category": "hotel", "reason": "位置便利", "price_hint": "约600元"},
                {"name": "不存在的神秘酒店", "category": "hotel", "reason": "编造内容", "price_hint": None},
            ]
        ),
    )

    items, status = recommendations.extract_recommendations("南京", GUIDE_RESULTS)

    assert status == "completed"
    assert [item["name"] for item in items] == ["金陵饭店", "南京大牌檔(德基广场店)", "夫子庙"]
    assert [item["type"] for item in items] == ["hotel", "restaurant", "attraction"]
    assert items[0]["price_hint"] == "约600元"
    assert items[2]["reason"] == "适合夜游秦淮河"


def test_extract_recommendations_parses_a_fenced_json_array(monkeypatch) -> None:
    monkeypatch.setattr(
        recommendations,
        "model_client",
        lambda: _FakeClient(
            '```json\n[{"name": "夫子庙", "category": "attraction", "reason": "夜游", "price_hint": null}]\n```'
        ),
    )

    items, status = recommendations.extract_recommendations("南京", GUIDE_RESULTS)

    assert status == "completed"
    assert [item["name"] for item in items] == ["夫子庙"]


def test_parse_json_array_keeps_the_places_that_arrived_before_a_cut_off(monkeypatch) -> None:
    """A truncated answer must not cost the whole list.

    Measured in a four-day plan: the mixed call was stopped by the output ceiling
    (finish_reason=length, 7229 characters) with no closing bracket, every place was
    discarded, and the plan fell back to keyword rules — which surfaced page and exhibition
    titles as place names and left every place without a grounded price.
    """
    truncated = (
        '[{"name": "宽窄巷子", "category": "attraction", "reason": "老成都街巷", "price_hint": null},'
        ' {"name": "马旺子川小馆", "category": "restaurant", "reason": "招牌[凉拌鸡]有名", "price_hint": "人均约150元"},'
        ' {"name": "博舍酒店", "category": "hotel", "reason": "位置好", "price_hint": "约2200元"},'
        ' {"name": "锦里古街", "category": "attraction", "reason": "夜游'
    )

    parsed = recommendations._parse_json_array(truncated)

    assert [item["name"] for item in parsed] == ["宽窄巷子", "马旺子川小馆", "博舍酒店"]
    # The bracket inside a value must not be mistaken for the end of an object.
    assert parsed[1]["reason"] == "招牌[凉拌鸡]有名"


def test_parse_json_array_gives_up_when_no_place_is_complete() -> None:
    assert recommendations._parse_json_array('[{"name": "宽窄巷子", "reason": "老成') is None
    assert recommendations._parse_json_array("这不是 JSON") is None
    assert recommendations._parse_json_array("") is None


def test_a_cut_off_extraction_is_recovered_and_logged(monkeypatch, caplog) -> None:
    """Recovering silently would hide a trip that is thinner than the prompt asked for."""
    monkeypatch.setattr(
        recommendations,
        "model_client",
        lambda: _FakeClient(
            '[{"name": "夫子庙", "category": "attraction", "reason": "夜游秦淮河", "price_hint": null},'
            ' {"name": "南京大牌檔(德基广场店)", "category": "restaurant", "reason": "本地菜", "price_hint": "人均约80元"},'
            ' {"name": "金陵饭店", "category": "hotel", "reason": "位置便',
            "length",
        ),
    )

    with caplog.at_level("WARNING"):
        items, status = recommendations.extract_recommendations("南京", GUIDE_RESULTS)

    assert status == "completed"
    assert [item["name"] for item in items] == ["南京大牌檔(德基广场店)", "夫子庙"]
    assert any("cut off" in record.message for record in caplog.records)


def test_the_mixed_ceiling_stays_above_the_measured_truncation() -> None:
    """Do not lower this back: 3000 cut the list roughly in half.

    `_count_hint` asks the mixed call for up to 27 places (12 sights, 12 restaurants, 3
    hotels), and a measured four-day plan had its answer stopped at the ceiling after 7229
    characters with no closing bracket. The salvage in `_parse_json_array` keeps such a run
    usable, but a half-length list is still the wrong answer to send.
    """
    measured_tokens_that_truncated = 3000

    assert recommendations.MIXED_MAX_TOKENS >= measured_tokens_that_truncated * 2


def test_extract_recommendations_caps_each_category(monkeypatch) -> None:
    payload = [
        {"name": name, "category": "hotel", "reason": "住宿", "price_hint": None}
        for name in ("金陵饭店", "中心大酒店", "夫子庙客栈", "秦淮宾馆")
    ]
    material = [
        {
            "source_type": "tavily",
            "title": "南京住宿推荐",
            "url": "https://example.com/stay",
            "snippet": "金陵饭店、中心大酒店、夫子庙客栈、秦淮宾馆都值得考虑，价格区间较大。",
        }
    ]
    monkeypatch.setattr(recommendations, "model_client", lambda: _client_returning(payload))

    items, status = recommendations.extract_recommendations("南京", material, limit_per_category=2)

    assert status == "completed"
    assert len(items) == 2


def test_extract_recommendations_skips_the_model_without_grounded_material(monkeypatch) -> None:
    def failing_client():
        raise AssertionError("the model must not be called without material")

    monkeypatch.setattr(recommendations, "model_client", failing_client)

    items, status = recommendations.extract_recommendations("南京", [])

    assert items == []
    assert status == "skipped"


def test_extract_recommendations_falls_back_to_keyword_rules_without_a_model(monkeypatch) -> None:
    monkeypatch.setattr(recommendations, "model_client", lambda: None)

    items, status = recommendations.extract_recommendations(
        "上海",
        [
            {
                "source_type": "tavily",
                "title": "外滩",
                "url": "https://example.com/bund",
                "snippet": "外滩夜景非常值得一看，建议晚上前往，人流集中在入夜前后。",
            }
        ],
    )

    assert status == "fallback"
    assert [item["name"] for item in items] == ["外滩"]
    assert items[0]["type"] == "attraction"


def test_the_keyword_fallback_rejects_a_page_title_as_a_place_name(monkeypatch) -> None:
    """With the model unavailable the fallback is the live path, so it must not lie.

    These titles were really rendered as recommendation cards when the account ran
    out of credit: one card was named "Tavily 联网检索摘要", and another described a
    whole article rather than a place.
    """
    monkeypatch.setattr(recommendations, "model_client", lambda: None)

    items, status = recommendations.extract_recommendations(
        "南京",
        [
            {
                "source_type": "tavily",
                "title": "Tavily 联网检索摘要",
                "url": "https://example.com/a",
                "snippet": "南京住宿选择较多，夫子庙一带酒店集中。",
            },
            {
                "source_type": "tavily",
                "title": "南京旅游必去景点推荐（交通+住宿+美食）- 南京本地宝",
                "url": "https://example.com/b",
                "snippet": "夫子庙、中山陵都是必去景点。",
            },
            {
                "source_type": "tavily",
                "title": "南京旅游景点攻略_南京打卡/必去景点大全/排名/推荐【携程攻略】",
                "url": "https://example.com/c",
                "snippet": "南京博物院值得一看。",
            },
            {
                "source_type": "tavily",
                "title": "南京住宿推薦｜新街口+秦淮河夫子廟步行可達10間精選酒店",
                "url": "https://example.com/d",
                "snippet": "新街口一带住宿方便。",
            },
            {
                "source_type": "tavily",
                "title": "南京 的酒店 | Holiday Inn 南京水游城假日酒店 酒店",
                "url": "https://example.com/e",
                "snippet": "水游城附近酒店较多。",
            },
        ],
    )

    assert status == "fallback"
    assert items == []


def test_extract_recommendations_trusts_an_empty_model_result_over_titles(monkeypatch) -> None:
    monkeypatch.setattr(recommendations, "model_client", lambda: _client_returning([]))

    items, status = recommendations.extract_recommendations("南京", GUIDE_RESULTS)

    assert items == []
    assert status == "completed"


def test_extract_recommendations_reads_a_per_unit_price(monkeypatch) -> None:
    monkeypatch.setattr(
        recommendations,
        "model_client",
        lambda: _client_returning(
            [
                {"name": "金陵饭店", "category": "hotel", "reason": "位置便利", "price_hint": "约600元",
                 "unit_price": 600, "price_unit": "night"},
                {"name": "南京大牌檔(德基广场店)", "category": "restaurant", "reason": "本地菜", "price_hint": "人均约80元",
                 "unit_price": "80元", "price_unit": "person_meal"},
                {"name": "夫子庙", "category": "attraction", "reason": "夜游", "price_hint": None,
                 "unit_price": "免费", "price_unit": "ticket"},
            ]
        ),
    )

    items, _ = recommendations.extract_recommendations("南京", GUIDE_RESULTS)

    assert items[0]["unit_price"] == 600
    assert items[0]["price_unit"] == "night"
    assert items[1]["unit_price"] == 80
    assert items[1]["price_unit"] == "person_meal"
    # "免费" carries no usable number; it must not become a bogus price.
    assert items[2]["unit_price"] is None
    assert items[2]["price_unit"] == "ticket"


def test_estimate_trip_costs_scales_to_trip_length_and_party() -> None:
    """One hotel night and one meal per place, not the whole trip's worth of both.

    A restaurant used to be charged ``unit_price * days * people`` -- a four-day trip for two
    quoted one restaurant at 640 CNY as if every meal were eaten there. Ticking a
    restaurant is one meal (the card's 吃几次 control makes it more), and a hotel is one
    night (the 住几晚 control makes it more), because a trip may change hotels.
    """
    items = recommendations.estimate_trip_costs(
        [
            {"name": "酒店", "type": "hotel", "unit_price": 600, "price_unit": "night"},
            {"name": "餐厅", "type": "restaurant", "unit_price": 80, "price_unit": "person_meal"},
            {"name": "景点", "type": "attraction", "unit_price": 100, "price_unit": "ticket"},
        ],
        trip_days=4,
        traveler_profile={"total_people": 2},
    )

    # 1 night, 1 meal for 2 people, 2 tickets -- the attraction is unchanged by design.
    assert [item["estimated_cost"] for item in items] == [600, 160, 200]
    assert "约 600 元/晚 × 1 晚 = 600 元" in items[0]["cost_explanation"]
    assert "按检索到的价格估算" in items[0]["cost_explanation"]
    assert "约 80 元/人/餐 × 2 人 × 1 次 = 160 元" in items[1]["cost_explanation"]


def test_every_item_carries_the_count_the_card_edits() -> None:
    items = recommendations.estimate_trip_costs(
        [
            {"name": "酒店", "type": "hotel", "unit_price": 600, "price_unit": "night"},
            {"name": "餐厅", "type": "restaurant", "unit_price": 80, "price_unit": "person_meal"},
            {"name": "景点", "type": "attraction", "unit_price": 100, "price_unit": "ticket"},
        ],
        trip_days=5,
        traveler_profile={"total_people": 2},
    )

    hotel, restaurant, attraction = items
    assert (hotel["quantity"], hotel["quantity_unit"], hotel["quantity_max"]) == (1, "晚", 4)
    assert (restaurant["quantity"], restaurant["quantity_unit"], restaurant["quantity_max"]) == (1, "次", 5)
    # An attraction has no count: the user asked for it to stay exactly as it was.
    assert attraction["quantity"] == 1
    assert attraction["quantity_unit"] is None
    assert attraction["estimated_cost"] == 200
    # The card rebuilds the arithmetic from these, so the source travels separately
    # instead of having to be parsed back out of the explanation string.
    assert hotel["cost_source"] == "按检索到的价格估算"


def test_the_count_multiplies_the_way_the_card_computes_it() -> None:
    """The numbers the card multiplies must match the server's own arithmetic."""
    items = recommendations.estimate_trip_costs(
        [
            {"name": "酒店", "type": "hotel", "unit_price": 620, "price_unit": "night"},
            {"name": "餐厅", "type": "restaurant", "unit_price": 80, "price_unit": "person_meal"},
        ],
        trip_days=5,
        traveler_profile={"total_people": 2},
    )

    hotel, restaurant = items
    assert hotel["unit_price"] * 3 == 1860  # three nights
    assert restaurant["unit_price"] * 2 * 3 == 480  # two people, three meals


def test_estimate_trip_costs_marks_default_prices_as_estimates() -> None:
    items = recommendations.estimate_trip_costs(
        [{"name": "无价景点", "type": "attraction", "unit_price": None, "price_unit": "ticket"}],
        trip_days=2,
        traveler_profile={"total_people": 1},
    )

    # The fallback is now the destination's own mid-range figure. With no destination at all it
    # is the Chinese row (a domestic museum ticket is around 60 CNY, not the old flat 100).
    assert items[0]["unit_price"] == 60
    assert items[0]["estimated_cost"] == 60
    assert "按当地中档常见价估算" in items[0]["cost_explanation"]


# --- budget awareness (the trip's own budget must reach the recommendation) -------


def test_budget_allowances_derives_ceilings_from_the_trip_budget() -> None:
    # 8000 CNY, 5 days / 4 nights, 3 people: the real trip that was offered a
    # 3840 and a 4800 hotel (one hotel = 60% of the whole budget).
    allowances = recommendations.budget_allowances(8000, 5, {"total_people": 3})

    # lodging: 40% of 8000 spread over 4 nights
    assert allowances["hotel"] == 800
    # meals: 30% of 8000 over 5 days x 3 meals x 3 people
    assert allowances["restaurant"] == 53
    # tickets: 20% of 8000 per person
    assert allowances["attraction"] == 533


def test_budget_allowances_is_empty_without_a_budget() -> None:
    for budget in (None, 0, -100, "not a number"):
        assert recommendations.budget_allowances(budget, 5, {"total_people": 2}) == {}


def test_budget_allowances_never_divides_by_zero() -> None:
    # A one-day trip has no nights; the ceiling must still be a positive number.
    allowances = recommendations.budget_allowances(1000, 1, {"total_people": 0})

    assert allowances["hotel"] >= 1
    assert allowances["restaurant"] >= 1
    assert allowances["attraction"] >= 1


def test_estimate_trip_costs_flags_and_demotes_over_budget_hotels() -> None:
    """The user's exact complaint: 3840 and 4800 hotels against an 8000 budget."""
    items = recommendations.estimate_trip_costs(
        [
            {"name": "贵酒店", "type": "hotel", "unit_price": 1200, "price_unit": "night"},
            {"name": "合适酒店", "type": "hotel", "unit_price": 400, "price_unit": "night"},
        ],
        trip_days=5,
        traveler_profile={"total_people": 3},
        trip_budget=8000,
    )

    # Nothing is dropped: both are still offered, but the affordable one comes first.
    assert [item["name"] for item in items] == ["合适酒店", "贵酒店"]
    by_name = {item["name"]: item for item in items}
    assert by_name["贵酒店"]["over_budget"] is True
    assert by_name["贵酒店"]["budget_limit"] == 800
    assert "超出预算参考" in by_name["贵酒店"]["budget_note"]
    assert by_name["合适酒店"]["over_budget"] is False
    assert by_name["合适酒店"]["budget_note"] == ""
    # The cost itself is unchanged and still explained honestly: one night, which is what
    # the card quotes before the traveller says how many nights they want.
    assert by_name["贵酒店"]["estimated_cost"] == 1200


def test_estimate_trip_costs_never_flags_without_a_budget() -> None:
    items = recommendations.estimate_trip_costs(
        [{"name": "贵酒店", "type": "hotel", "unit_price": 5000, "price_unit": "night"}],
        trip_days=5,
        traveler_profile={"total_people": 3},
    )

    assert items[0]["over_budget"] is False
    assert "budget_limit" not in items[0]


def test_budget_hint_states_every_ceiling_and_encourages_affordable_choices() -> None:
    hint = recommendations._budget_hint(recommendations.budget_allowances(8000, 5, {"total_people": 3}))

    assert "800 元" in hint
    assert "优先" in hint
    # An over-allowance option may still be listed, so the model is not told to drop it.
    assert "也可以给出" in hint
    assert recommendations._budget_hint({}) is None


# --- meal coverage: a plan has to cover a day's three meals -----------------------


def test_classify_meal_slot_prefers_a_recognised_declaration() -> None:
    assert recommendations.classify_meal_slot("某某饭店", "breakfast") == "breakfast"
    assert recommendations.classify_meal_slot("某某饭店", "晚餐") == "dinner"
    assert recommendations.classify_meal_slot("某某饭店", "午餐") == "lunch"


def test_classify_meal_slot_falls_back_to_the_shop_name() -> None:
    assert recommendations.classify_meal_slot("李记汤包店") == "breakfast"
    assert recommendations.classify_meal_slot("老门东豆浆铺") == "breakfast"
    assert recommendations.classify_meal_slot("南京大牌檔") == "any"


def test_classify_meal_slot_ignores_an_unknown_declaration() -> None:
    # A junk label must not become a bogus slot; the name still decides.
    assert recommendations.classify_meal_slot("李记汤包店", "brunch-ish") == "breakfast"
    assert recommendations.classify_meal_slot("大牌檔", "brunch-ish") == "any"


def test_normalize_item_labels_meal_slots_for_restaurants_only() -> None:
    text = "南京大牌檔(德基广场店)是本地菜馆，李记汤包店早上营业。夫子庙适合夜游秦淮河。"

    restaurant = recommendations._normalize_item(
        {"name": "李记汤包店", "category": "restaurant"}, text, "南京"
    )
    attraction = recommendations._normalize_item(
        {"name": "夫子庙", "category": "attraction"}, text, "南京"
    )

    assert restaurant["meal_slot"] == "breakfast"
    assert attraction["meal_slot"] is None


def test_normalize_item_keeps_a_rough_position_for_the_map() -> None:
    # The workspace map only sketches the route, so a rough position is enough -- but a
    # place the map can draw has to arrive with both halves of the pair.
    text = "夫子庙适合夜游秦淮河。"
    item = recommendations._normalize_item(
        {"name": "夫子庙", "category": "attraction", "latitude": 32.0224, "longitude": 118.7886},
        text,
        "南京",
    )

    assert item["latitude"] == 32.0224
    assert item["longitude"] == 118.7886


def test_normalize_item_drops_a_position_it_cannot_trust() -> None:
    """A wrong pin is worse than no pin, so every unusable shape becomes null."""
    text = "夫子庙适合夜游秦淮河。"
    cases = [
        # Only half an answer: a latitude with no longitude is not a position.
        {"latitude": 32.0224, "longitude": None},
        {"latitude": None, "longitude": 118.7886},
        {"latitude": 32.0224},
        # Impossible coordinates.
        {"latitude": 91.0, "longitude": 118.7886},
        {"latitude": 32.0224, "longitude": 181.0},
        # The shapes a model actually emits when it does not know.
        {"latitude": "不确定", "longitude": "不确定"},
        {"latitude": "", "longitude": ""},
        {"latitude": True, "longitude": True},
    ]

    for case in cases:
        item = recommendations._normalize_item(
            {"name": "夫子庙", "category": "attraction", **case}, text, "南京"
        )
        assert item is not None, case
        assert item["latitude"] is None, case
        assert item["longitude"] is None, case


def test_category_limits_give_restaurants_room_for_a_whole_day() -> None:
    assert recommendations.CATEGORY_LIMITS["restaurant"] >= 3
    assert recommendations.CATEGORY_LIMITS["restaurant"] > recommendations.CATEGORY_LIMITS["hotel"]


def test_extract_recommendations_keeps_more_restaurants_than_hotels(monkeypatch) -> None:
    payload = [
        {"name": f"酒店{index}", "category": "hotel", "reason": "住宿"} for index in range(6)
    ] + [
        {"name": f"餐厅{index}", "category": "restaurant", "reason": "吃饭"} for index in range(6)
    ]
    material = [
        {
            "source_type": "tavily",
            "title": "南京吃住",
            "url": "https://example.com/all",
            "snippet": "酒店0 酒店1 酒店2 酒店3 酒店4 酒店5 餐厅0 餐厅1 餐厅2 餐厅3 餐厅4 餐厅5 都很好。",
        }
    ]
    monkeypatch.setattr(recommendations, "model_client", lambda: _client_returning(payload))

    items, _ = recommendations.extract_recommendations("南京", material)

    hotels = [item for item in items if item["type"] == "hotel"]
    restaurants = [item for item in items if item["type"] == "restaurant"]
    assert len(hotels) == recommendations.CATEGORY_LIMITS["hotel"]
    assert len(restaurants) == recommendations.CATEGORY_LIMITS["restaurant"]


def test_extract_recommendations_still_honours_a_uniform_limit(monkeypatch) -> None:
    payload = [
        {"name": f"餐厅{index}", "category": "restaurant", "reason": "吃饭"} for index in range(5)
    ]
    material = [
        {
            "source_type": "tavily",
            "title": "南京美食",
            "url": "https://example.com/food",
            "snippet": "餐厅0 餐厅1 餐厅2 餐厅3 餐厅4 都值得去。",
        }
    ]
    monkeypatch.setattr(recommendations, "model_client", lambda: _client_returning(payload))

    items, _ = recommendations.extract_recommendations("南京", material, limit_per_category=2)

    assert len([item for item in items if item["type"] == "restaurant"]) == 2


# --- price grounding: a number with nothing behind it is not a price --------------
#
# Real defect, caught end-to-end after the budget work: once the prompt carried a
# budget ceiling ("酒店每晚不超过约 800 元；每餐每人不超过约 53 元"), the model wrote
# those exact ceilings back as unit_price for hotels and restaurants even though the
# material quoted no price at all, and the cards then claimed "按检索到的价格估算".


def test_is_price_grounded_accepts_a_number_from_the_material() -> None:
    assert recommendations._is_price_grounded(600, "约600元", "金陵饭店房价约600元。") is True
    assert recommendations._is_price_grounded(600, None, "金陵饭店房价 600 元起。") is True


def test_is_price_grounded_ignores_thousands_separators() -> None:
    assert recommendations._is_price_grounded(1200, "¥1,200", "标间 ¥1,200 一晚。") is True


def test_is_price_grounded_rejects_a_number_that_is_nowhere_in_the_text() -> None:
    # The 800 ceiling was nowhere in the material, yet the model returned it as a price.
    assert recommendations._is_price_grounded(800, None, "夫子庙附近住宿选择较多，价格区间较大。") is False
    assert recommendations._is_price_grounded(53, None, "狮子桥美食街小吃云集。") is False


def test_is_price_grounded_rejects_a_number_absent_from_a_present_price_hint() -> None:
    # price_hint "免费" cannot justify a 70 CNY ticket.
    assert recommendations._is_price_grounded(70, "免费", "夫子庙秦淮风光带免费开放。") is False


def test_grounded_unit_price_drops_an_ungrounded_price() -> None:
    price, is_free = recommendations._grounded_unit_price(53, None, "restaurant", "小吃很多。")

    assert price is None
    assert is_free is False


def test_grounded_unit_price_keeps_a_grounded_price() -> None:
    price, is_free = recommendations._grounded_unit_price(40, "成人票40元/人", "attraction", "成人票40元/人。")

    assert price == 40
    assert is_free is False


def test_grounded_unit_price_reads_a_free_attraction_as_zero() -> None:
    for hint in ("免费", "免票", "无需门票"):
        price, is_free = recommendations._grounded_unit_price(100, hint, "attraction", f"南京博物院{hint}开放。")
        assert price == 0
        assert is_free is True


def test_free_is_not_applied_to_hotels_and_restaurants() -> None:
    price, is_free = recommendations._grounded_unit_price(300, "免费", "hotel", "某酒店介绍。")

    assert is_free is False


def test_normalize_item_rejects_a_price_the_material_never_mentioned() -> None:
    text = "夫子庙附近住宿选择较多。狮子桥美食街小吃云集。"

    item = recommendations._normalize_item(
        {"name": "狮子桥美食街", "category": "restaurant", "unit_price": 53, "price_hint": None},
        text,
        "南京",
    )

    assert item["unit_price"] is None


def test_normalize_item_keeps_a_price_the_material_quotes() -> None:
    text = "南京大牌檔人均约80元。夫子庙适合夜游。"

    item = recommendations._normalize_item(
        {"name": "南京大牌檔", "category": "restaurant", "unit_price": 80, "price_hint": "人均约80元"},
        text,
        "南京",
    )

    assert item["unit_price"] == 80


def test_estimate_trip_costs_charges_nothing_for_a_free_attraction() -> None:
    items = recommendations.estimate_trip_costs(
        [{"name": "南京博物院", "type": "attraction", "unit_price": 0, "is_free": True}],
        trip_days=2,
        traveler_profile={"total_people": 3},
    )

    assert items[0]["estimated_cost"] == 0
    assert "资料显示免费" in items[0]["cost_explanation"]


def test_enrich_prices_does_not_re_lookup_a_free_place(monkeypatch) -> None:
    calls: list[str] = []

    def fake_lookup(name, category, destination):
        calls.append(name)
        return ""

    monkeypatch.setattr(recommendations, "_lookup_price_text", fake_lookup)

    recommendations.enrich_prices_with_search(
        [{"name": "南京博物院", "type": "attraction", "unit_price": 0, "is_free": True}], "南京"
    )

    assert calls == []




def test_a_price_lookup_uses_its_own_measured_timeout(monkeypatch) -> None:
    """The bound a price lookup passes must be the measured one.

    The 8.0 the callers carried was written for the metered search provider that was later
    removed, and it was ignored anyway. Measured on this provider, six real lookups ran
    2.8s to 9.6s (median 4.2s), so 8s still discarded one real price in six, while the
    inherited 120s planning bound is far too long for one line.
    """
    seen: list[float] = []

    def fake_search_snippets(query, limit=3, timeout=8.0):
        seen.append(timeout)
        return ["人均 150 元"]

    monkeypatch.setattr(recommendations, "search_snippets", fake_search_snippets)

    text = recommendations._lookup_price_text("马旺子川小馆", "restaurant", "成都")

    assert seen == [recommendations.PRICE_LOOKUP_TIMEOUT]
    assert 8.0 < recommendations.PRICE_LOOKUP_TIMEOUT < recommendations.PRICE_PHASE_BUDGET
    assert text == "人均 150 元"


def test_price_lookups_run_wider_than_the_category_searches(monkeypatch) -> None:
    """The price phase has its own budget, so it gets its own worker count.

    It used to share `LOOKUP_WORKERS` (4) with the category searches, and with a 30s budget
    and a dozen unpriced places the phase ran out of clock. That showed up as places with
    default estimates rather than as a slow plan, which is why it was easy to miss.
    """
    import threading
    import time

    running = 0
    peak = 0
    lock = threading.Lock()

    def fake_lookup(name, category, destination):
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        time.sleep(0.3)
        with lock:
            running -= 1
        return ""

    monkeypatch.setattr(recommendations, "_lookup_price_text", fake_lookup)
    monkeypatch.setattr(recommendations, "model_client", lambda: None)
    monkeypatch.setattr(recommendations, "LOOKUP_WORKERS", 4)

    widest = recommendations.PRICE_LOOKUP_WORKERS
    items = [
        {"name": f"地点{index}", "type": "attraction", "unit_price": None} for index in range(widest)
    ]

    recommendations.enrich_prices_with_search(items, "南京")

    assert widest > 4
    assert peak == widest


def test_enrich_prices_fills_missing_prices_from_search(monkeypatch) -> None:
    monkeypatch.setattr(recommendations, "search_snippets", lambda query, limit=3, timeout=8.0: [f"snippet for {query}"])
    monkeypatch.setattr(
        recommendations, "model_client", lambda: _client_returning({"金陵饭店": 688, "夫子庙": "免费"})
    )
    items = [
        {"name": "金陵饭店", "type": "hotel", "unit_price": None},
        {"name": "夫子庙", "type": "attraction", "unit_price": None},
    ]

    enriched = recommendations.enrich_prices_with_search(items, "南京")

    assert enriched[0]["unit_price"] == 688
    # "免费" carries no usable number, so the item stays unpriced for the default path.
    assert enriched[1]["unit_price"] is None


def test_enrich_prices_skips_items_that_already_have_a_price(monkeypatch) -> None:
    def _fail(*args, **kwargs):
        raise AssertionError("search must not run when every price is already known")

    monkeypatch.setattr(recommendations, "search_snippets", _fail)
    monkeypatch.setattr(recommendations, "model_client", _fail)
    items = [{"name": "金陵饭店", "type": "hotel", "unit_price": 600}]

    enriched = recommendations.enrich_prices_with_search(items, "南京")

    assert enriched[0]["unit_price"] == 600


def test_enrich_prices_survives_a_search_failure(monkeypatch) -> None:
    def _boom(*args, **kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(recommendations, "search_snippets", _boom)
    items = [{"name": "金陵饭店", "type": "hotel", "unit_price": None}]

    enriched = recommendations.enrich_prices_with_search(items, "南京")

    assert enriched[0]["unit_price"] is None


def test_enrich_prices_ignores_model_prices_for_unknown_places(monkeypatch) -> None:
    monkeypatch.setattr(recommendations, "search_snippets", lambda query, limit=3, timeout=8.0: ["some material"])
    monkeypatch.setattr(recommendations, "model_client", lambda: _client_returning({"凭空捏造的酒店": 999}))
    items = [{"name": "金陵饭店", "type": "hotel", "unit_price": None}]

    enriched = recommendations.enrich_prices_with_search(items, "南京")

    assert enriched[0]["unit_price"] is None


def test_implausible_prices_are_rejected(monkeypatch) -> None:
    # A hotel phone number read as a nightly rate would otherwise blow up the budget.
    monkeypatch.setattr(recommendations, "search_snippets", lambda query, limit=3, timeout=8.0: ["some material"])
    monkeypatch.setattr(
        recommendations,
        "model_client",
        lambda: _client_returning({"上海半岛酒店": 2637230, "上海东方佘山索菲特大酒店": 925}),
    )
    items = [
        {"name": "上海半岛酒店", "type": "hotel", "unit_price": None},
        {"name": "上海东方佘山索菲特大酒店", "type": "hotel", "unit_price": None},
    ]

    enriched = recommendations.enrich_prices_with_search(items, "上海")

    assert enriched[0]["unit_price"] is None
    assert enriched[1]["unit_price"] == 925


def test_estimate_trip_costs_rebounds_a_stored_implausible_price() -> None:
    items = recommendations.estimate_trip_costs(
        [{"name": "上海半岛酒店", "type": "hotel", "unit_price": 2637230, "price_unit": "night"}],
        trip_days=5,
        traveler_profile={"total_people": 2},
    )

    # The stored 2,637,230 is impossible, so it falls back to the destination's own mid-range
    # figure: with no destination given that is the Chinese row, a 550 CNY hotel night.
    assert items[0]["unit_price"] == 550
    # One night, not the whole five-day stay: the count is the traveller's to set.
    assert items[0]["estimated_cost"] == 550
    assert "按当地中档常见价估算" in items[0]["cost_explanation"]


def test_estimate_trip_costs_accepts_a_plausible_restaurant_and_ticket_range() -> None:
    items = recommendations.estimate_trip_costs(
        [
            {"name": "便宜面馆", "type": "restaurant", "unit_price": 25, "price_unit": "person_meal"},
            {"name": "顶奢餐厅", "type": "restaurant", "unit_price": 1800, "price_unit": "person_meal"},
            {"name": "免费公园", "type": "attraction", "unit_price": 1, "price_unit": "ticket"},
        ],
        trip_days=2,
        traveler_profile={"total_people": 1},
    )

    # One meal each (the price itself is what is being checked here), one ticket.
    assert [item["estimated_cost"] for item in items] == [25, 1800, 1]


def test_ensure_category_coverage_respects_the_per_category_limit(monkeypatch) -> None:
    monkeypatch.setattr(
        recommendations,
        "search_snippets",
        lambda query, limit=5, timeout=8.0: ["和平饭店房价1200元，外滩茂悦大酒店房价900元，上海半岛酒店房价2500元，静安寺某酒店房价600元。"],
    )
    monkeypatch.setattr(
        recommendations,
        "model_client",
        lambda: _client_returning(
            [
                {"name": "和平饭店", "category": "hotel"},
                {"name": "外滩茂悦大酒店", "category": "hotel"},
                {"name": "上海半岛酒店", "category": "hotel"},
                {"name": "静安寺某酒店", "category": "hotel"},
            ]
        ),
    )

    filled = recommendations.ensure_category_coverage("上海", [{"name": "外滩", "type": "attraction"}])

    assert sum(1 for item in filled if item["type"] == "hotel") == 3


def test_category_limits_scale_with_the_length_of_the_trip() -> None:
    """Four sights and six restaurants cannot fill a five-day trip without repeating.

    The old fixed limits were the reason every day used the same handful of places.
    """
    three = recommendations.category_limits(3)
    five = recommendations.category_limits(5)

    # Two sights a day and three distinct meals a day, plus slack because a venue the
    # model does not label as a breakfast shop counts as a main-meal option, and because
    # a batch never returns everything it was asked for.
    assert five["attraction"] == 14
    assert five["restaurant"] == 21
    assert five["hotel"] == 3
    assert three["attraction"] == 10
    assert three["restaurant"] == 15
    # Enough for every day to differ: five breakfasts and ten main-meal venues.
    assert five["restaurant"] >= 5 + 10
    assert five["attraction"] >= 5 * 2
    # A longer trip asks for strictly more, never less.
    assert five["attraction"] > three["attraction"]
    assert five["restaurant"] > three["restaurant"]


def test_category_limits_are_bounded_and_never_zero() -> None:
    assert recommendations.category_limits(0)["attraction"] > 0
    assert recommendations.category_limits(60)["restaurant"] == recommendations.MAX_RESTAURANTS
    assert recommendations.category_limits(60)["attraction"] == recommendations.MAX_ATTRACTIONS


def test_ensure_category_coverage_tops_a_thin_category_up_to_the_target(monkeypatch) -> None:
    """A category with one item is still searched, not treated as covered.

    The historic rule only searched a category that had nothing at all, so a plan that
    returned a single sight stayed with a single sight.
    """
    monkeypatch.setattr(
        recommendations,
        "search_snippets",
        lambda query, limit=5, timeout=8.0: ["外滩、豫园、东方明珠、南京路步行街、田子坊都值得一去。"],
    )
    monkeypatch.setattr(
        recommendations,
        "model_client",
        lambda: _client_returning(
            [
                {"name": "豫园", "category": "attraction"},
                {"name": "东方明珠", "category": "attraction"},
                {"name": "南京路步行街", "category": "attraction"},
                {"name": "田子坊", "category": "attraction"},
            ]
        ),
    )
    asked: list[str] = []
    original = recommendations._extract_category_with_model

    def _spy(destination, category, material, allowances=None, wanted=None, slot=None, exclude=(), plan=None):
        asked.append(f"{category}:{wanted}")
        return original(destination, category, material, allowances, wanted, slot, exclude, plan)

    monkeypatch.setattr(recommendations, "_extract_category_with_model", _spy)

    # Hotels and restaurants already meet their targets; only the sights are thin.
    items = (
        [{"name": "外滩", "type": "attraction"}]
        + [{"name": f"酒店{index}", "type": "hotel"} for index in range(3)]
        + [{"name": f"餐厅{index}", "type": "restaurant"} for index in range(15)]
    )
    filled = recommendations.ensure_category_coverage(
        "上海",
        items,
        limits={"hotel": 3, "restaurant": 15, "attraction": 4},
    )
    attractions = [item for item in filled if item["type"] == "attraction"]

    assert len(attractions) == 4
    assert "外滩" in [item["name"] for item in attractions]
    # Only the short category was searched, and it asked for exactly what it needed.
    assert asked == ["attraction:3"]


def test_ensure_category_coverage_does_not_touch_a_category_that_is_already_full(monkeypatch) -> None:
    """A plan that already holds enough candidates pays nothing extra."""
    calls: list[str] = []
    monkeypatch.setattr(
        recommendations,
        "_fill_category",
        lambda destination, category, shortfall, allowances=None: calls.append(category) or [],
    )

    items = [
        {"name": f"景点{index}", "type": "attraction"} for index in range(4)
    ] + [
        {"name": f"餐厅{index}", "type": "restaurant"} for index in range(15)
    ] + [
        {"name": f"酒店{index}", "type": "hotel"} for index in range(3)
    ]

    filled = recommendations.ensure_category_coverage(
        "上海", items, limits={"hotel": 3, "restaurant": 15, "attraction": 4}
    )

    assert calls == []
    assert filled == items


def test_ensure_category_coverage_still_only_fills_missing_categories_without_targets(monkeypatch) -> None:
    """The historic behaviour is unchanged when no day-aware targets are given."""
    calls: list[str] = []
    monkeypatch.setattr(
        recommendations,
        "_fill_category",
        lambda destination, category, shortfall, allowances=None: calls.append(category) or [],
    )

    recommendations.ensure_category_coverage(
        "上海",
        [{"name": "外滩", "type": "attraction"}, {"name": "某酒店", "type": "hotel"}],
    )

    # One category gathered: only "restaurant" had nothing at all.
    assert calls == ["restaurant"]


def test_a_large_shortfall_is_split_into_several_small_batches() -> None:
    """Fifteen restaurants are asked for in batches, never in one oversized request.

    A single request for the whole shortfall came back as a timeout or as a flat
    rejection from the provider, so the pool is accumulated instead.
    """
    groups = recommendations._category_batch_plan("restaurant", 21)

    # Each batch stays within the per-call ceiling.
    assert all(batch <= recommendations.MAX_ITEMS_PER_CALL for _, batches in groups for batch, _ in batches)
    # The batches add up to exactly the shortfall.
    assert sum(batch for _, batches in groups for batch, _ in batches) == 21
    # Two searches, not one per batch: a search is charged per request.
    assert len(groups) == 2
    # One batch is asked for breakfast venues, and the rest for main meals.
    slots = [slot for _, batches in groups for _, slot in batches]
    assert "breakfast" in slots
    assert slots.count("main") >= 1


def test_one_search_is_reused_by_every_batch_for_the_same_category(monkeypatch) -> None:
    """Searches are charged per request, so batches for one category share a search.

    One search per batch took the search plan to its usage limit in an afternoon.
    """
    searched: list[str] = []
    calls: list[tuple[int, str | None]] = []
    monkeypatch.setattr(
        recommendations,
        "_category_material",
        lambda destination, category, hint=None: searched.append(hint)
        or [{"title": "南京", "snippet": "南京大牌档、韩复兴、马祥兴、绿柳居、安乐园、奇芳阁都值得一试。"}],
    )
    names = ["南京大牌档", "韩复兴", "马祥兴", "绿柳居", "安乐园", "奇芳阁"]
    counter = {"n": 0}

    def _fake_extract(destination, category, material, allowances=None, wanted=None, slot=None, exclude=(), plan=None):
        calls.append((wanted, slot))
        batch = names[counter["n"] : counter["n"] + wanted]
        counter["n"] += len(batch)
        return [{"name": name, "category": "restaurant"} for name in batch]

    monkeypatch.setattr(recommendations, "_extract_category_with_model", _fake_extract)

    found = recommendations._fill_category("南京", "restaurant", 26)

    # Three batches — main meals twelve and six, breakfast eight — over two searches.
    assert len(calls) == 3
    assert len(searched) == len(set(searched)) == 2
    assert len(found) == 6


def test_a_later_batch_is_told_what_the_earlier_one_already_used(monkeypatch) -> None:
    """Without this the second batch returns the same famous few places again."""
    seen_exclusions: list[tuple[str, ...]] = []
    names = ["南京大牌档", "韩复兴", "马祥兴", "绿柳居", "安乐园"]
    counter = {"n": 0}

    def _fake_extract(destination, category, material, allowances=None, wanted=None, slot=None, exclude=(), plan=None):
        seen_exclusions.append(tuple(exclude))
        name = names[counter["n"] % len(names)]
        counter["n"] += 1
        return [{"name": name, "category": "restaurant"}]

    monkeypatch.setattr(
        recommendations,
        "_category_material",
        lambda destination, category, hint=None: [
            {"title": "南京", "snippet": "南京大牌档、韩复兴、马祥兴、绿柳居、安乐园都值得一试。"}
        ],
    )
    monkeypatch.setattr(recommendations, "_extract_category_with_model", _fake_extract)

    recommendations._fill_category("南京", "restaurant", 20)

    assert seen_exclusions[0] == ()
    # The second batch of the same search knows what the first one produced.
    assert seen_exclusions[1] == ("南京大牌档",)


def test_batches_stop_when_a_search_runs_out_of_new_places(monkeypatch) -> None:
    """A batch that adds nothing ends its search, instead of asking over and over."""
    calls: list[int] = []
    monkeypatch.setattr(
        recommendations,
        "_category_material",
        lambda destination, category, hint=None: [{"title": "南京", "snippet": "南京大牌档的盐水鸭值得一试。"}],
    )

    def _same_answer(destination, category, material, allowances=None, wanted=None, slot=None, exclude=(), plan=None):
        calls.append(1)
        return [{"name": "南京大牌档", "category": "restaurant"}]

    monkeypatch.setattr(recommendations, "_extract_category_with_model", _same_answer)

    found = recommendations._fill_category("南京", "restaurant", 20)

    assert len(found) == 1
    # One attempt per search group, with the no-progress batch ending each of them.
    assert len(calls) == 3


def test_batches_never_push_a_category_past_its_target(monkeypatch) -> None:
    """Each batch is a separate call, so their results are capped together."""
    monkeypatch.setattr(
        recommendations,
        "_fill_category",
        lambda destination, category, shortfall, allowances=None: [
            {"name": f"{category}-{index}", "type": category} for index in range(shortfall + 5)
        ],
    )

    recommendation = recommendations.ensure_category_coverage(
        "南京", [], limits={"hotel": 0, "restaurant": 3, "attraction": 0}
    )

    # The category answered with more than it needed; only the shortfall is kept.
    assert len(recommendation) == 3


def test_the_extraction_asks_for_enough_of_each_category(monkeypatch) -> None:
    """The count has to reach the model, or the answer stays as thin as it was."""
    captured: dict = {}

    class _SpyCompletions(_FakeCompletions):
        def create(self, **kwargs):
            captured.update(kwargs)
            return super().create(**kwargs)

    class _SpyClient:
        def __init__(self) -> None:
            self.chat = _FakeChat("[]")
            self.chat.completions = _SpyCompletions("[]")

    monkeypatch.setattr(recommendations, "model_client", _SpyClient)

    recommendations._extract_with_model(
        "南京",
        [{"title": "南京", "snippet": "夫子庙、中山陵、南京博物院都值得一去。"}],
        None,
        {"hotel": 3, "restaurant": 15, "attraction": 10},
    )

    system = captured["messages"][0]["content"]
    # The single mixed call covers every category at once, capped at one batch each; the
    # rest of the pool is gathered by the per-category lookups, so a huge number is never
    # demanded at once.
    assert "景点 10 个" in system
    assert "餐厅 12 家" in system
    assert "酒店 3 家" in system
    assert "不得重复同一个地点" in system
    # The shared client's 30-second bound is too short for a list answer: the reply was
    # discarded and the plan fell back to keyword rules without a trace.
    assert captured["timeout"] == recommendations.EXTRACTION_TIMEOUT
    assert captured["timeout"] > 30
    # Room for the whole mixed answer. 1600 truncated it (finish_reason=length) and the
    # parse failure silently downgraded the plan to keyword rules.
    assert captured["max_tokens"] == recommendations.MIXED_MAX_TOKENS
    assert captured["max_tokens"] >= 3000
    assert '"needed"' in captured["messages"][1]["content"]


def test_a_batch_answer_has_room_without_being_asked_for_a_huge_list() -> None:
    """A batch asks for a few places, and its ceiling stays below the mixed one."""
    assert recommendations.BATCH_MAX_TOKENS >= 1600
    assert recommendations.BATCH_MAX_TOKENS < recommendations.MIXED_MAX_TOKENS
    # Measured against this provider: eight, ten and twelve places all came back complete
    # and in under 25 seconds, twice each; fifteen was rejected outright.
    assert recommendations.MAX_ITEMS_PER_CALL == 12
    assert recommendations.category_limits(10)["restaurant"] > recommendations.MAX_ITEMS_PER_CALL


def test_restaurant_batches_state_which_meal_they_are_for() -> None:
    """The daily plan needs one breakfast venue and two main-meal venues per day.

    An unguided search returns mostly main-meal restaurants, so one batch asks
    specifically for breakfast shops and is told to label them as such.
    """
    batches = recommendations._category_batch_plan("restaurant", 18)
    slots = [slot for _, group in batches for _, slot in group]

    assert "breakfast" in slots
    assert slots.count("main") >= 1
    assert "meal_slot 一律填 breakfast" in recommendations.SLOT_INSTRUCTIONS["breakfast"]
    assert "不要早餐店" in recommendations.SLOT_INSTRUCTIONS["main"]


def test_a_five_day_trip_costs_four_searches_not_one_per_batch(monkeypatch) -> None:
    """The search plan is charged per request, so the count is asserted, not assumed.

    One search per batch took the plan to its usage limit in an afternoon of testing:
    nine searches were issued for a single five-day plan request.
    """
    searched: list[str] = []
    model_calls: list[str] = []

    def _material(destination, category, hint=None):
        searched.append(f"{category}:{hint}")
        return [{"title": "南京", "snippet": "南京大牌档、夫子庙、明孝陵、金陵饭店都值得一试。"}]

    def _extract(destination, category, material, allowances=None, wanted=None, slot=None, exclude=(), plan=None):
        model_calls.append(category)
        return []

    monkeypatch.setattr(recommendations, "_category_material", _material)
    monkeypatch.setattr(recommendations, "_extract_category_with_model", _extract)

    limits = recommendations.category_limits(5)
    recommendations.ensure_category_coverage("南京", [], limits=limits)

    # One search per distinct hint: two for restaurants (main meals and breakfast), one
    # for sights and one for hotels — plus the plan's own search, this is five requests.
    assert len(searched) == 4, searched
    # The batches themselves are model calls, which are not charged the same way.
    assert len(model_calls) == 6, model_calls


def test_the_extraction_prompts_ask_for_chinese_place_names() -> None:
    """One venue came back as `咸亨酒店` one day and `Xian Heng Restaurant` the next.

    The plan then sent the traveller to the same restaurant twice, because a string
    comparison cannot see that two scripts name the same place.
    """
    assert "中文原名" in recommendations._SYSTEM_PROMPT
    assert "中文原名" in recommendations._CATEGORY_TAIL


def test_material_prefers_chinese_pages_but_keeps_english_when_that_is_all_there_is() -> None:
    """A name read off an English page came back in English, duplicating the venue.

    `咸亨酒店` was served for lunch and `Xian Heng Restaurant` for dinner on one day: the
    same restaurant, named once per language, and no string comparison could see it.
    """
    blocks = [
        {"title": "Nanjing travel guide", "snippet": "Confucius Temple and Ming City Wall are highlights."},
        {"title": "南京旅游攻略", "snippet": "夫子庙、明城墙、中山陵都值得一去。"},
    ]

    kept = recommendations._prefer_chinese(blocks)

    assert kept == [blocks[1]]
    # An English-only search is still better than no material at all.
    assert recommendations._prefer_chinese([blocks[0]]) == [blocks[0]]
    assert recommendations._prefer_chinese([]) == []


def test_material_without_any_chinese_keeps_every_block() -> None:
    blocks = [
        {"title": "Nanjing guide", "snippet": "Confucius Temple is worth a visit."},
        {"title": "Nanjing food", "snippet": "Try the salted duck."},
    ]

    assert recommendations._prefer_chinese(blocks) == blocks


def test_estimate_trip_costs_treats_a_single_day_trip_as_one_night() -> None:
    items = recommendations.estimate_trip_costs(
        [{"name": "酒店", "type": "hotel", "unit_price": 500, "price_unit": "night"}],
        trip_days=1,
        traveler_profile={"total_people": 0},
    )

    assert items[0]["estimated_cost"] == 500
    assert "× 1 晚" in items[0]["cost_explanation"]
