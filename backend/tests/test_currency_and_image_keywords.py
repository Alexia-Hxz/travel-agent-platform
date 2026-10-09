"""Prices in foreign material, and the English keywords that fix the pictures.

Two measured defects are pinned here.

**Currency.** A real Tokyo plan stored a museum ticket quoted as "常设展成人630日元（约32元
人民币）" as ``unit_price`` **630**, and the cheapest restaurants fell to the 80 CNY Chinese
default because their real yen prices were outside the CNY bounds. Nothing in the project
converted anything: the prompt told the model that ``unit_price`` "只能取自资料中出现的价
格", so it faithfully copied yen into a field everything downstream reads as CNY.

**Image keywords.** The photo library is indexed in English. Searching a Chinese name as-is
returned a photo of Osaka Castle for three different Tokyo museums, and a tonkotsu ramen
query returned snow-covered horse saddles, so the keyword the model now supplies for each
place is what the picture search is built on.
"""

import pytest

from app.agents.tools import currency, mealdb_images, pexels_images, recommendations

# Captured before the autouse offline fixture replaces the network call, so a test that
# wants to exercise the real parsing can put it back.
_REAL_DISH_SEARCH = mealdb_images.search_dish_photos


# --- currency: reading a foreign price ---------------------------------------------------


@pytest.mark.parametrize(
    ("destination", "stated_cny", "raw", "hint", "expected"),
    [
        # The source printed its own conversion: that one wins, because it is a quote.
        ("日本东京", 32, 630, "常设展成人630日元（约32元人民币）", 32),
        ("日本东京", 51, 1000, "成人1,000日元（约51元）", 51),
        ("日本东京", 245, 4800, "平日成人4,800日元（约245元）", 245),
        # No conversion in the text: the rate table supplies one.
        ("日本东京", None, 3800, "3,800日元～", 182),
        ("日本东京", None, 1480, "1,480日元", 71),
        ("日本东京", None, 2800, "¥2,800/人", 134),
        ("首尔", None, 12000, "12,000韩元", 61),
        ("曼谷", None, 250, "250泰铢", 50),
        ("巴黎", None, 18, "€18", 140),
        ("纽约", None, 25, "US$25 per person", 178),
        ("香港", None, 88, "HK$88", 80),
        ("台北", None, 150, "NT$150", 33),
    ],
)
def test_a_foreign_price_becomes_cny(
    destination: str, stated_cny: int | None, raw: int, hint: str, expected: int
) -> None:
    assert currency.explicit_cny_amount(hint) == stated_cny
    converted = currency.convert_foreign_number(raw, hint, destination)
    assert converted is not None
    assert converted[0] == expected


@pytest.mark.parametrize(
    ("destination", "raw", "hint"),
    [
        # A price already in CNY must never be converted a second time.
        ("日本东京", 2150, "约 2150 元/晚"),
        ("南京", 150, "人均 ¥150"),
        ("南京", 80, "人均约 80 元"),
        # Two CNY figures in one hint: the model's own single price stands.
        ("南京", 120, "门票 100 元，学生 50 元"),
        # No currency wording and a number too small to be anything but CNY.
        ("日本东京", 120, "人均 120"),
        # A destination whose currency is not recognised is left alone.
        ("上海", 300, "人均 300"),
    ],
)
def test_a_price_that_is_already_cny_is_left_alone(destination: str, raw: int, hint: str) -> None:
    assert currency.convert_foreign_number(raw, hint, destination) is None


def test_the_destination_decides_which_currency_a_bare_symbol_means() -> None:
    """"¥" is written for both the yen and the renminbi."""
    assert currency.hint_currency("¥2,800/人", "日本东京") == "JPY"
    assert currency.hint_currency("¥150", "南京") == "CNY"
    # Both currencies in one hint: the destination's own price is the one being quoted.
    assert currency.hint_currency("630日元（约32元人民币）", "日本东京") == "JPY"
    # For a Chinese destination the renminbi figure is the one that applies.
    assert currency.hint_currency("630日元（约32元人民币）", "南京") == "CNY"


def test_the_bare_yuan_inside_a_yen_word_is_not_a_second_currency() -> None:
    """A naive substring scan reads "日元" as "JPY" plus "CNY" and then picks the wrong one."""
    assert currency.hint_currency("12,000韩元") == "KRW"
    assert currency.hint_currency("1,480日元") == "JPY"


def test_a_static_rate_table_covers_the_destinations_in_use() -> None:
    assert currency.currency_for_destination("日本东京") == "JPY"
    assert currency.currency_for_destination("南京") == "CNY"
    assert currency.currency_for_destination("首尔") == "KRW"
    assert currency.is_foreign("日本东京") is True
    assert currency.is_foreign("南京") is False
    assert currency.RATES_AS_OF  # the table says how old it is


def test_the_prompt_tells_the_model_the_rate() -> None:
    hint = currency.prompt_hint("日本东京")
    assert "日元" in hint and "0.0480" in hint and "换算" in hint
    # A Chinese destination needs no such instruction.
    assert currency.prompt_hint("南京") == ""


# --- currency: the bounds check ----------------------------------------------------------


def test_a_yen_figure_too_large_for_cny_is_read_as_yen() -> None:
    """1,480 is inside the CNY per-meal bounds, so nothing else would catch it."""
    assert recommendations._plausible_unit_price(1480, "restaurant", "日本东京") == 71
    assert recommendations._plausible_unit_price(4800, "attraction", "日本东京") == 230
    # The same number for a Chinese destination is a real CNY price and stays untouched.
    assert recommendations._plausible_unit_price(1480, "restaurant", "南京") == 1480


def test_a_yen_hotel_rate_outside_the_cny_ceiling_is_converted_not_dropped() -> None:
    """30,000 exceeds the CNY nightly ceiling, and dropping it lost the price entirely."""
    from app.agents.tools import place_facts

    assert place_facts._clean_hotel_price(30000, "日本东京") == 1440
    assert place_facts._clean_hotel_price(620, "南京") == 620
    # Still a real guard: a phone number is not a price in any currency.
    assert place_facts._clean_hotel_price(13800138000, "日本东京") is None


def test_a_grounded_foreign_price_is_converted_and_still_grounded() -> None:
    """Grounding runs on the number the material wrote, before conversion.

    Checking the converted value instead would reject every foreign price, because 32 is
    not in "630日元". Checking nothing would let a fabricated price through.
    """
    price, is_free = recommendations._grounded_unit_price(
        630, "常设展成人630日元（约32元人民币）", "attraction", "常设展成人630日元（约32元人民币）", "日本东京"
    )
    assert (price, is_free) == (32, False)

    price, _ = recommendations._grounded_unit_price(
        3800, "3,800日元～", "restaurant", "3,800日元～", "日本东京"
    )
    assert price == 182

    # A number that appears nowhere in the material is still refused.
    price, _ = recommendations._grounded_unit_price(999, None, "restaurant", "资料里没有价格", "日本东京")
    assert price is None


def test_a_free_attraction_is_still_free_in_a_foreign_country() -> None:
    price, is_free = recommendations._grounded_unit_price(
        0, "免费入场", "attraction", "免费入场", "日本东京"
    )
    assert (price, is_free) == (0, True)


@pytest.mark.parametrize(
    ("hint", "destination", "expected"),
    [
        ("常设展成人620日元", "日本东京", (620, "JPY")),
        ("一般展期成人2,000日元", "日本东京", (2000, "JPY")),
        ("¥598元/人（12,400日元）", "日本东京", (598, "CNY")),
        ("人均 ¥150", "南京", (150, "CNY")),
        ("门票 100 元，学生 50 元", "南京", (100, "CNY")),
        ("12,000韩元", "首尔", (12000, "KRW")),
        ("HK$88", "香港", (88, "HKD")),
        ("免费入场", "南京", None),
    ],
)
def test_the_amount_marked_in_the_price_text_is_readable(hint, destination, expected) -> None:
    """The source's own price line is the most reliable place to read a price from."""
    assert currency.marked_amount(hint, destination) == expected


def test_the_sources_own_cny_figure_is_not_converted_a_second_time() -> None:
    """Measured: "¥598元/人（12,400日元）" was stored as 29 -- 598 put through the yen rate."""
    price, _ = recommendations._grounded_unit_price(
        598, "¥598元/人（12,400日元）", "attraction", "¥598元/人（12,400日元）", "日本东京"
    )
    assert price == 598


def test_a_price_is_read_from_the_material_when_the_model_gives_none() -> None:
    """Measured: several museums had a yen hint and no model figure, so the card showed the
    100 CNY category default -- a price the material never stated."""
    price, _ = recommendations._grounded_unit_price(
        None, "常设展成人600日元", "attraction", "常设展成人600日元", "日本东京"
    )
    assert price == 29

    price, _ = recommendations._grounded_unit_price(
        None, "一般展期成人2,000日元", "attraction", "一般展期成人2,000日元", "日本东京"
    )
    assert price == 96

    # A restaurant, where the default (80) hid the real price entirely.
    price, _ = recommendations._grounded_unit_price(
        None, "1,480日元（咖喱饭）", "restaurant", "1,480日元（咖喱饭）", "日本东京"
    )
    assert price == 71


def test_a_model_figure_that_is_not_the_price_in_the_text_is_replaced() -> None:
    """Measured: a 620 yen museum ticket came back as ``1`` -- the number of tickets, not
    the price -- and "1" is inside the CNY bounds, so only the text can correct it."""
    price, _ = recommendations._grounded_unit_price(
        1, "常设展成人620日元", "attraction", "常设展成人620日元（约30元人民币）", "日本东京"
    )
    assert price == 30


def test_the_material_price_wins_for_a_chinese_destination_too() -> None:
    """The same rule, without conversion: a figure the text does not contain is replaced."""
    price, _ = recommendations._grounded_unit_price(
        120, "门票 100 元，学生 50 元", "attraction", "门票 100 元，学生 50 元", "南京"
    )
    assert price == 100


# --- image keywords ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Sensoji Temple", "Sensoji Temple"),
        ("tonkotsu ramen", "tonkotsu ramen"),
        ("  Tokyo   National Museum  ", "Tokyo National Museum"),
        # A Chinese keyword is discarded rather than passed to an English index.
        ("浅草寺", None),
        ("东京国立博物馆 museum", None),
        ("", None),
        (None, None),
        ("null", None),
        ("ab", None),
        ("x" * 80, None),
    ],
)
def test_only_an_english_image_keyword_is_kept(value, expected) -> None:
    assert recommendations._clean_image_query(value) == expected


def test_a_photo_is_judged_by_its_own_description() -> None:
    """Pexels describes every photo, and that description is what relevance is checked on."""
    photo = {
        "url": "https://images.pexels.com/a.jpeg",
        "alt": "delicious ramen bowl in tokyo booth",
        "page_url": "https://www.pexels.com/photo/delicious-ramen-bowl-in-tokyo-booth-37014726/",
    }
    assert pexels_images.page_slug(photo["page_url"]) == "delicious ramen bowl in tokyo booth"
    required = ["tonkotsu", "ramen"]
    assert pexels_photo_score(photo, required) > 0


def pexels_photo_score(photo: dict, required: list[str]) -> int:
    from app.agents.tools import place_images

    return place_images._photo_score(photo, required, ["tokyo"])


def test_an_off_topic_photo_scores_zero() -> None:
    """The measured failures, as fixtures: these must never be accepted for these cards."""
    from app.agents.tools import place_images

    saddles = {"alt": "snow covered horse saddles on wooden rack", "page_url": None}
    peking_duck = {"alt": "street vendor cooking peking duck in beijing", "page_url": None}
    osaka_castle = {"alt": "aerial view of osaka castle grounds and golden shachihoko", "page_url": None}

    assert place_images._photo_score(saddles, ["tonkotsu", "ramen"], []) == 0
    assert place_images._photo_score(peking_duck, ["yakitori"], []) == 0
    # The English name carries the city, which is what rejects the wrong-city photo.
    assert place_images._photo_score(osaka_castle, ["tokyo", "national", "museum"], []) == 0


def test_words_that_describe_every_photo_are_not_used_for_verification() -> None:
    """"food" appears in every food photo's description and cannot tell them apart."""
    from app.agents.tools import place_images

    assert place_images._keywords("sushi food") == ["sushi"]
    assert place_images._keywords("japanese cuisine dish") == ["japanese"]
    # A Chinese keyword yields nothing to verify against, which is the honest answer.
    assert place_images._keywords("浅草寺") == []


def test_a_food_placeholder_matches_the_kind_of_dish() -> None:
    from app.agents.tools import place_images

    assert place_images.fallback_image("restaurant", "豚骨拉面").endswith("restaurant_noodle.svg")
    assert place_images.fallback_image("restaurant", "江户前寿司").endswith("restaurant_sushi.svg")
    assert place_images.fallback_image("restaurant", "和牛烧肉").endswith("restaurant_grill.svg")
    assert place_images.fallback_image("restaurant", "抹茶冰淇淋").endswith("restaurant_dessert.svg")
    assert place_images.fallback_image("restaurant", "鳗鱼饭").endswith("restaurant_rice.svg")
    # An unrecognised dish keeps the generic restaurant icon rather than guessing.
    assert place_images.fallback_image("restaurant", "招牌套餐").endswith("restaurant.svg")
    assert place_images.fallback_image("attraction").endswith("attraction.svg")


def test_every_cuisine_placeholder_exists_on_disk() -> None:
    """A missing file would hand the browser a broken image, which is worse than no card."""
    from pathlib import Path

    from app.agents.tools import place_images

    root = Path(place_images.__file__).resolve().parents[2] / "static"
    for path in list(place_images.CATEGORY_IMAGES.values()) + list(place_images.CUISINE_IMAGES.values()):
        assert (root / path.replace("/static/", "")).is_file(), path


# --- the dish library --------------------------------------------------------------------


def test_the_dish_library_parses_a_meal_into_a_photo_record(monkeypatch) -> None:
    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "meals": [
                    {
                        "idMeal": "52894",
                        "strMeal": "Tonkatsu pork",
                        "strMealThumb": "https://www.themealdb.com/images/media/meals/x.jpg",
                    }
                ]
            }

    monkeypatch.setattr(mealdb_images.httpx, "get", lambda *a, **k: _Response())
    monkeypatch.setattr(mealdb_images, "search_dish_photos", _REAL_DISH_SEARCH)
    mealdb_images.clear_cache()

    photos = mealdb_images.search_dish_photos("katsu")

    assert len(photos) == 1
    assert photos[0]["url"].endswith("x.jpg")
    assert photos[0]["alt"] == "tonkatsu pork"
    assert photos[0]["source"] == "mealdb"
    # Not a Pexels photo, so it must not be credited to Pexels or its photographer.
    assert photos[0]["photographer"] is None
    assert mealdb_images.credit_line(photos[0]) == "Tonkatsu pork · TheMealDB"


def test_the_dish_library_reports_no_hit_instead_of_raising(monkeypatch) -> None:
    class _Empty:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"meals": None}

    monkeypatch.setattr(mealdb_images.httpx, "get", lambda *a, **k: _Empty())
    monkeypatch.setattr(mealdb_images, "search_dish_photos", _REAL_DISH_SEARCH)
    mealdb_images.clear_cache()

    assert mealdb_images.search_dish_photos("unagi") == []
    # A miss is not cached, so a later lookup can still find the dish.
    assert mealdb_images._cache == {}


def test_a_cuisine_is_named_for_the_destinations_in_use() -> None:
    assert mealdb_images.area_for_destination("日本东京") == "Japanese"
    assert mealdb_images.area_for_destination("南京") == "Chinese"
    assert mealdb_images.area_for_destination("首尔") == "Korean"
    assert mealdb_images.area_for_destination("冰岛雷克雅未克") is None


# --- per-place price lookup ---------------------------------------------------------------


def test_a_one_line_price_answer_is_not_thrown_away(monkeypatch) -> None:
    """Measured: 18 per-place price lookups, 18 discards, all because of a 40-character floor.

    The floor is right for planning material, where a paragraph is expected. A price answer
    is one line, and discarding it dropped every Tokyo restaurant to the Chinese default.
    """
    from app.agents.tools import dashscope_search, web_search

    monkeypatch.setattr(dashscope_search, "search_available", lambda: True)
    monkeypatch.setattr(
        dashscope_search, "_ask", lambda prompt, max_tokens=None, timeout=None: ("东京国立博物馆常设展 630 日元。", "stop")
    )

    # The planning call still refuses a 15-character answer...
    assert dashscope_search.grounded_answer("门票多少") is None
    # ...and the price call accepts it, because a number is what it needs.
    assert web_search.search_snippets("东京国立博物馆 门票价格") == ["东京国立博物馆常设展 630 日元。"]


def test_a_price_lookup_without_a_number_is_no_evidence(monkeypatch) -> None:
    """An answer with no figure in it cannot support a price, however long it is."""
    from app.agents.tools import dashscope_search, web_search

    monkeypatch.setattr(dashscope_search, "search_available", lambda: True)
    monkeypatch.setattr(
        dashscope_search,
        "_ask",
        lambda prompt, max_tokens=None, timeout=None: ("抱歉，我无法联网确认这家的具体价格，建议您查看官网。", "stop"),
    )

    assert web_search.search_snippets("某餐厅 人均") == []
