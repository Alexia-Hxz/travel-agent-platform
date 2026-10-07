"""Hotel ratings/prices and restaurant signature dishes.

The user asked for two changes to the cards: hotels should carry no picture, just a rating
and a price, and a food card should show the dish rather than the shop. Both facts come from
one batched live search, so these tests pin the parsing rules, the bounds that keep a phone
number out of a price, and the promise that a failed lookup is never fatal.
"""

from typing import Any

import pytest

from app.agents.tools import dashscope_search, place_facts
from app.agents.tools import place_images


@pytest.fixture(autouse=True)
def _enable_live_search(monkeypatch: pytest.MonkeyPatch):
    """Live search is off for the whole suite; these tests stub its answer instead."""
    monkeypatch.delenv("DISABLE_LIVE_WEB_SEARCH", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")


def _answers(monkeypatch: pytest.MonkeyPatch, answer: str | None):
    asked: list[str] = []

    def _grounded(query: str, timeout: float | None = None) -> str | None:
        asked.append(query)
        return answer

    monkeypatch.setattr(dashscope_search, "grounded_answer", _grounded)
    return asked


def test_a_hotel_rating_and_price_are_read_from_the_answer(monkeypatch) -> None:
    _answers(monkeypatch, '{"南京涵碧楼酒店": {"rating": 4.7, "price": 620}}')
    items = [{"name": "南京涵碧楼酒店", "type": "hotel"}]

    place_facts.enrich_place_facts(items, "南京")

    assert items[0]["rating"] == 4.7
    assert items[0]["unit_price"] == 620
    assert items[0]["price_unit"] == "night"
    assert items[0]["price_hint"] == "约 620 元/晚"


def test_a_restaurant_gets_its_signature_dish_and_a_search_term(monkeypatch) -> None:
    _answers(
        monkeypatch,
        '{"马祥兴菜馆": {"dish": "松鼠鳜鱼", "dish_query": "sweet and sour mandarin fish"}}',
    )
    items = [{"name": "马祥兴菜馆", "type": "restaurant"}]

    place_facts.enrich_place_facts(items, "南京")

    assert items[0]["signature_dish"] == "松鼠鳜鱼"
    assert items[0]["dish_query"] == "sweet and sour mandarin fish"


def test_a_review_count_is_not_mistaken_for_a_rating() -> None:
    """An implausible number is a parsing accident, not a fact."""
    assert place_facts._clean_rating("4.7") == 4.7
    assert place_facts._clean_rating(4.75) == 4.8
    assert place_facts._clean_rating("12860") is None  # a review count
    assert place_facts._clean_rating("0") is None
    assert place_facts._clean_rating("") is None
    assert place_facts._clean_rating(None) is None


def test_a_phone_number_is_not_mistaken_for_a_price() -> None:
    assert place_facts._clean_hotel_price("620") == 620
    assert place_facts._clean_hotel_price("620 元/晚") == 620
    assert place_facts._clean_hotel_price("025-88886666") is None
    assert place_facts._clean_hotel_price("3") is None
    assert place_facts._clean_hotel_price("999999") is None
    assert place_facts._clean_hotel_price(None) is None


def test_a_sentence_is_not_a_dish_name() -> None:
    assert place_facts._clean_dish("盐水鸭") == "盐水鸭"
    assert place_facts._clean_dish("这家的盐水鸭非常有名，值得一试再来一次") is None
    assert place_facts._clean_dish("salted duck") is None
    assert place_facts._clean_dish("") is None


def test_an_answer_wrapped_in_prose_is_still_read(monkeypatch) -> None:
    """The model often fences its JSON or introduces it with a sentence."""
    _answers(
        monkeypatch,
        '好的，查询结果如下：\n```json\n{"绿柳居": {"dish": "素菜包", "dish_query": "vegetarian bun"}}\n```',
    )
    items = [{"name": "绿柳居", "type": "restaurant"}]

    place_facts.enrich_place_facts(items, "南京")

    assert items[0]["signature_dish"] == "素菜包"


def test_an_unusable_answer_leaves_the_item_untouched(monkeypatch) -> None:
    for answer in (None, "", "我查不到这些信息。", "[]"):
        _answers(monkeypatch, answer)
        items = [{"name": "南京涵碧楼酒店", "type": "hotel"}]

        place_facts.enrich_place_facts(items, "南京")

        assert "rating" not in items[0]
        assert "unit_price" not in items[0]


def test_a_failing_lookup_is_never_fatal(monkeypatch) -> None:
    def _boom(query: str, timeout: float | None = None) -> str:
        raise RuntimeError("model exploded")

    monkeypatch.setattr(dashscope_search, "grounded_answer", _boom)
    items = [
        {"name": "南京涵碧楼酒店", "type": "hotel"},
        {"name": "马祥兴菜馆", "type": "restaurant"},
    ]

    place_facts.enrich_place_facts(items, "南京")

    assert "rating" not in items[0]
    assert "signature_dish" not in items[1]


def test_a_facts_batch_bounds_its_own_live_search(monkeypatch) -> None:
    """One stalled lookup must not hold an otherwise finished plan open.

    This phase has no wall clock of its own and used to inherit the 120s planning search
    timeout, for a rating the reader sees at a glance. A ten-item batch measured 9.4s for
    hotels and 11.3s for dishes, so the bound is generous. It is passed to the call itself
    rather than enforced by walking away from a thread, which is what keeps the writes to a
    recommendation finished before the phase returns.
    """
    seen: list[float | None] = []

    def _grounded(query: str, timeout: float | None = None) -> str | None:
        seen.append(timeout)
        return "{}"

    monkeypatch.setattr(dashscope_search, "grounded_answer", _grounded)

    place_facts.enrich_place_facts([{"name": "南京涵碧楼酒店", "type": "hotel"}], "南京")

    assert seen == [place_facts.FACTS_CALL_TIMEOUT]
    assert place_facts.FACTS_CALL_TIMEOUT < dashscope_search.SEARCH_TIMEOUT


def test_large_lists_are_split_into_batches(monkeypatch) -> None:
    """One answer is only reliable up to about ten items (round-8 measurements)."""
    asked = _answers(monkeypatch, "{}")
    items = [{"name": f"餐厅{i}", "type": "restaurant"} for i in range(23)]

    place_facts.enrich_place_facts(items, "南京")

    assert len(asked) == 3
    assert place_facts.FACTS_BATCH == 10


def test_the_live_search_switch_is_honoured(monkeypatch) -> None:
    """With live search off, nothing is asked and nothing is invented."""
    monkeypatch.setenv("DISABLE_LIVE_WEB_SEARCH", "1")
    asked = _answers(monkeypatch, '{"南京涵碧楼酒店": {"rating": 4.7, "price": 620}}')
    items = [{"name": "南京涵碧楼酒店", "type": "hotel"}]

    place_facts.enrich_place_facts(items, "南京")

    assert asked == []
    assert "rating" not in items[0]


def test_a_hotel_is_never_given_a_picture(monkeypatch) -> None:
    """The whole point: no stock photo of a different hotel."""
    monkeypatch.setattr(
        place_images, "search_pexels_photos", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no lookup"))
    )
    place_images.clear_image_cache()

    resolved = place_images.resolve_place_images(
        [{"name": "南京涵碧楼酒店", "type": "hotel"}],
        "南京",
        skip_categories=frozenset({"hotel"}),
    )

    assert resolved[0]["image_url"] is None
    assert resolved[0]["fallback_image_url"] is None
    assert resolved[0]["image_credit"] is None


def test_a_restaurant_is_searched_by_its_dish_not_its_name(monkeypatch) -> None:
    """A picture of the shopfront (or of a street in another city) is not a food photo."""
    asked: list[str] = []

    def _search(query, limit=15, timeout=8.0):
        asked.append(query)
        return []

    monkeypatch.setattr(place_images, "search_pexels_photos", _search)
    monkeypatch.setattr(place_images, "_is_reachable", lambda url: True)
    monkeypatch.setattr(place_images, "host_resolves", lambda host, budget=2.5: True)
    place_images.clear_image_cache()

    place_images.resolve_place_images(
        [
            {
                "name": "马祥兴菜馆",
                "type": "restaurant",
                "signature_dish": "松鼠鳜鱼",
                "dish_query": "sweet and sour mandarin fish",
            }
        ],
        "南京",
    )

    assert asked == ["sweet and sour mandarin fish food"]


def test_a_restaurant_without_a_dish_still_avoids_its_own_name(monkeypatch) -> None:
    """Nothing is known about the dish, so the city's food pool is used instead."""
    per_place: list[str] = []
    pool: list[tuple[str, str]] = []

    monkeypatch.setattr(
        place_images, "search_pexels_photos", lambda query, limit=15, timeout=8.0: (per_place.append(query), [])[1]
    )
    monkeypatch.setattr(
        place_images.pexels_images,
        "city_pool_photos",
        lambda destination, hint: (pool.append((destination, hint)), [])[1],
    )
    monkeypatch.setattr(place_images, "_is_reachable", lambda url: True)
    monkeypatch.setattr(place_images, "host_resolves", lambda host, budget=2.5: True)
    place_images.clear_image_cache()

    place_images.resolve_place_images([{"name": "马祥兴菜馆", "type": "restaurant"}], "南京")

    # The shop's own name is never searched, and the fallback is the city's FOOD pool.
    assert per_place == []
    assert pool == [("南京", "food")]


def test_an_attraction_is_unaffected(monkeypatch) -> None:
    """The picture rules for sights are already right and must not change."""
    photo = {
        "url": "https://images.pexels.com/photos/5/pexels-photo-5.jpeg",
        "photographer": "Cheng Shi Song",
        "page_url": "https://www.pexels.com/photo/5/",
    }
    monkeypatch.setattr(place_images, "search_pexels_photos", lambda query, limit=15, timeout=8.0: [photo])
    monkeypatch.setattr(place_images, "_is_reachable", lambda url: True)
    monkeypatch.setattr(place_images, "host_resolves", lambda host, budget=2.5: True)
    place_images.clear_image_cache()

    resolved = place_images.resolve_place_images(
        [{"name": "夫子庙", "type": "attraction"}], "南京", skip_categories=frozenset({"hotel"})
    )

    assert resolved[0]["image_url"] == photo["url"]
    assert resolved[0]["image_credit"] == "Photo by Cheng Shi Song on Pexels"


def test_the_response_whitelist_keeps_the_new_fields() -> None:
    """Anything missing from the response whitelist never reaches the client."""
    import inspect

    from app.agents.workflows import planner

    source = inspect.getsource(planner._compose)

    assert '"rating"' in source
    assert '"signature_dish"' in source
    # The price tier and the "this price was corrected" flag are computed in
    # estimate_trip_costs and read by the planning panel's food cards. This test exists
    # because they were added to the pipeline and NOT to this whitelist: a live plan showed
    # 19 food recommendations with no tier at all, and Pydantic quietly filled in nulls.
    assert '"price_tier"' in source
    assert '"price_tier_label"' in source
    assert '"price_adjusted"' in source
