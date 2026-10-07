"""What eating out actually costs, per destination and per kind of place.

Why this exists
---------------
The recommendation pipeline had one flat idea of a meal: ``DEFAULT_UNIT_PRICES`` said a
restaurant costs 80 CNY per person, and the sanity bounds allowed anything from 10 to 2000.
Both are China-shaped. Measured complaint from a real Tokyo trip: a **sushi restaurant** was
shown at 53 CNY per person. 53 CNY is about 1,100 yen -- a conveyor-belt sushi lunch, not a
sit-down sushi dinner, and certainly not enough to fill anyone up. The number was not invented:
the material quoted something (a single dish, a lunch set, an old price) and nothing in the
system knew that a sushi shop in Japan cannot serve a full dinner for that.

So the pipeline needs to know three things it never did:

1. **What kind of place this is** -- a ramen counter, a conveyor-belt sushi chain, a proper sushi
   restaurant, an omakase counter, an izakaya, a street stall. "Restaurant" is one category with
   a hundred price levels inside it.
2. **What that kind costs at this destination**, as a low/typical/high band per person, in CNY.
3. **How to use the band honestly**: guide the model, replace the flat default with a local one,
   and *raise* a price that is below the cheapest realistic full meal for that kind -- while
   saying so on the card, because an adjusted number must never look like a quoted one.

The bands are data, not a model call: a static table per country, so pricing works offline and
cannot fail. They are travel-budget estimates (``AS_OF``), not accounting.

Everything here is in CNY. The table was written from local-currency prices at the static rates
in ``currency.RATES_TO_CNY``, and each row keeps its source note.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agents.tools.price_bands import Band, _band
from app.services import destination_service

# When the numbers were gathered. Shown next to any adjusted price.
AS_OF = "2026-09"


# --------------------------------------------------------------------------------------
# Kinds of place to eat, most specific first.
# --------------------------------------------------------------------------------------
# Matched against the venue name, the quoted price text and the recommendation reason, in this
# order, because the earlier patterns are the more specific ones: "回転寿司" must not be read as
# "寿司", and a "米其林" note must beat the plain "restaurant" wording.
_KIND_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "sushi_kaiten",
        ("回転寿司", "回转寿司", "kaiten", "conveyor", "スシロー", "sushiro", "くら寿司", "藏寿司",
         "kura sushi", "genki sushi", "元气寿司", "寿司郎", "はま寿司", "hamazushi", "争鲜"),
    ),
    (
        "sushi_fine",
        ("omakase", "おまかせ", "鮨", "数寄屋橋次郎", "sukiyabashi", "寿司之神", "jiro", "米其林",
         "michelin", "江户前", "怀石", "懐石", "kaiseki", "割烹"),
    ),
    ("sushi_mid", ("寿司", "sushi", "鮓", "すし", "sushi bar", "板前")),
    (
        "bbq_grill",
        ("烧肉", "焼肉", "yakiniku", "烤肉", "bbq", "barbecue", "和牛", "wagyu", "牛角", "叙々苑",
         "五花肉", "삼겹살", "korean bbq", "烤肉店", "铁板烧", "teppanyaki", "牛排", "steak"),
    ),
    (
        "izakaya_bar",
        ("居酒屋", "izakaya", "酒馆", "小酒馆", "串烧", "串焼", "焼き鳥", "yakitori", "烤鸟",
         "大排档", "夜市bar", "pub", "酒吧", "bar &", "tapas", "鸟贵族", "鳥貴族", "和民",
         "watami", "塚田農場"),
    ),
    (
        "hotpot",
        ("火锅", "hot pot", "hotpot", "shabu", "しゃぶ", "涮", "打边炉", "sukiyaki", "스키야키",
         "汤锅", "麻辣烫", "串串"),
    ),
    (
        "seafood",
        ("海鲜", "seafood", "蟹", "crab", "牡蛎", "oyster", "龙虾", "lobster", "鱼市", "水产",
         "seafood market", "海鲜市场"),
    ),
    (
        "cafe_dessert",
        ("咖啡", "cafe", "coffee", "甜品", "dessert", "蛋糕", "bakery", "面包", "茶室", "抹茶",
         "冰淇淋", "刨冰", "点心", "pastry", "starbucks", "星巴克", "茶饮", "奶茶"),
    ),
    (
        "ramen_noodles",
        ("拉面", "ラーメン", "ramen", "面馆", "麺", "そば", "soba", "乌冬", "うどん", "udon",
         "米线", "河粉", "pho", "云吞", "馄饨", "noodle", "拌面", "刀削面", "牛肉面", "冷面",
         "naengmyeon", "국수"),
    ),
    (
        "breakfast",
        ("早餐", "breakfast", "早点", "包子", "汤包", "小笼", "豆浆", "油条", "粥", "茶餐厅",
         "brunch", "早茶", "煎饼", "烧饼", "肠粉"),
    ),
    (
        "fast_food",
        ("麦当劳", "mcdonald", "肯德基", "kfc", "burger", "汉堡", "subway", "赛百味", "便利店",
         "convenience", "lawson", "7-eleven", "7-11", "familymart", "全家", "快餐", "吉野家",
         "松屋", "すき家", "sukiya", "food court", "美食广场", "快餐店"),
    ),
    (
        "street_food",
        ("小吃", "street food", "路边摊", "摊", "夜市", "market stall", "庙口", "屋台", "yatai",
         "food stall", "小吃街", "排档"),
    ),
    (
        "fine_dining",
        ("tasting menu", "fine dining", "高级餐厅", "法餐", "法式", "米其林餐厅", "三星", "套餐料理",
         "怀石料理", "分子料理", "fusion", "主厨"),
    ),
    (
        "casual_local",
        ("定食", "teishoku", "食堂", "家常", "餐厅", "菜馆", "饭店", "小馆", "家常菜", "set meal",
         "restaurant", "料理", "kitchen", "diner", "bistro", "trattoria", "餐馆", "菜"),
    ),
)
# The fallback kind: a normal mid-range sit-down meal.
GENERIC_KIND = "generic"


def meal_kind(name: str | None, price_hint: Any = None, reason: str | None = None) -> str:
    """Which kind of place this is, from its name, its quoted price and its reason line."""
    text = " ".join(str(part or "") for part in (name, price_hint, reason)).casefold()
    if not text.strip():
        return GENERIC_KIND
    for kind, keywords in _KIND_KEYWORDS:
        if any(keyword.casefold() in text for keyword in keywords):
            return kind
    return GENERIC_KIND


# --------------------------------------------------------------------------------------
# The bands, per destination. CNY per person for one meal, per room for one night, per person
# for one ticket.
# --------------------------------------------------------------------------------------
# ``meal_price_data`` is a GENERATED module: every number there was measured from a cited source
# (chain price lists, tourism boards, hotel ADR series, Numbeo baselines) and converted to CNY
# with ``currency.RATES_TO_CNY``. Run ``scripts/build_meal_price_data.py`` to rebuild it.
#
# The measured tables cover 23 countries. Everything else is placed by price level (below), and a
# destination whose country cannot be told at all falls back to the Chinese row -- the historical
# behaviour, so a plan with no recognisable destination prices exactly as it did before.
from app.agents.tools.meal_price_data import (  # noqa: E402  (kept next to its policy)
    ATTRACTION_BANDS,
    HOTEL_BANDS,
    MEAL_BANDS,
)

# Countries with no row of their own are placed by how expensive eating out is there, so an
# unlisted destination still gets a sane number instead of a Chinese one.
_PRICE_LEVELS: dict[str, tuple[str, ...]] = {
    "expensive": (
        "冰岛", "挪威", "丹麦", "瑞典", "芬兰", "荷兰", "比利时", "奥地利", "爱尔兰", "卢森堡",
        "以色列", "卡塔尔", "科威特", "巴林", "沙特阿拉伯", "阿曼", "加拿大", "夏威夷",
    ),
    "mid": (
        "葡萄牙", "希腊", "捷克", "波兰", "匈牙利", "克罗地亚", "斯洛文尼亚", "斯洛伐克",
        "爱沙尼亚", "拉脱维亚", "立陶宛", "俄罗斯", "南非", "智利", "阿根廷", "巴西", "墨西哥",
        "哥斯达黎加", "巴拿马", "乌拉圭", "塞舌尔", "毛里求斯", "约旦", "黎巴嫩", "摩洛哥",
        "突尼斯", "格鲁吉亚", "亚美尼亚", "哈萨克斯坦", "蒙古", "菲律宾", "文莱", "马尔代夫",
    ),
    "cheap": (
        "柬埔寨", "老挝", "缅甸", "尼泊尔", "斯里兰卡", "孟加拉国", "巴基斯坦", "印度", "朝鲜",
        "玻利维亚", "秘鲁", "哥伦比亚", "厄瓜多尔", "危地马拉", "洪都拉斯", "尼加拉瓜",
        "肯尼亚", "坦桑尼亚", "埃塞俄比亚", "加纳", "尼日利亚", "塞内加尔", "乌兹别克斯坦",
        "吉尔吉斯斯坦", "塔吉克斯坦", "阿塞拜疆", "伊朗", "伊拉克", "叙利亚", "也门", "古巴",
    ),
}
# One row per price level, used only when the country has no table of its own. Scales the
# Chinese bands by roughly 2.4x / 1.5x / 0.7x.
_LEVEL_MEALS: dict[str, dict[str, Band]] = {
    "expensive": {
        "street_food": _band(30, 55, 110),
        "breakfast": _band(30, 55, 100),
        "fast_food": _band(45, 75, 120),
        "ramen_noodles": _band(60, 95, 160),
        "casual_local": _band(80, 130, 250),
        "cafe_dessert": _band(35, 65, 130),
        "sushi_kaiten": _band(100, 160, 260),
        "sushi_mid": _band(180, 300, 560),
        "sushi_fine": _band(650, 1100, 2300),
        "bbq_grill": _band(130, 220, 460),
        "izakaya_bar": _band(100, 170, 350),
        "hotpot": _band(100, 170, 350),
        "seafood": _band(160, 290, 580),
        "fine_dining": _band(420, 760, 1700),
        "generic": _band(85, 165, 380),
    },
    "mid": {
        "street_food": _band(12, 24, 50),
        "breakfast": _band(12, 22, 45),
        "fast_food": _band(28, 45, 75),
        "ramen_noodles": _band(25, 42, 75),
        "casual_local": _band(40, 75, 150),
        "cafe_dessert": _band(25, 45, 90),
        "sushi_kaiten": _band(70, 110, 180),
        "sushi_mid": _band(130, 220, 420),
        "sushi_fine": _band(480, 800, 1700),
        "bbq_grill": _band(90, 160, 330),
        "izakaya_bar": _band(75, 120, 250),
        "hotpot": _band(75, 130, 260),
        "seafood": _band(110, 200, 400),
        "fine_dining": _band(280, 520, 1200),
        "generic": _band(50, 100, 240),
    },
    "cheap": {
        "street_food": _band(4, 9, 20),
        "breakfast": _band(4, 8, 18),
        "fast_food": _band(12, 22, 40),
        "ramen_noodles": _band(9, 16, 32),
        "casual_local": _band(15, 30, 65),
        "cafe_dessert": _band(10, 20, 40),
        "sushi_kaiten": _band(40, 65, 110),
        "sushi_mid": _band(70, 120, 240),
        "sushi_fine": _band(260, 450, 950),
        "bbq_grill": _band(40, 80, 170),
        "izakaya_bar": _band(35, 60, 130),
        "hotpot": _band(35, 65, 140),
        "seafood": _band(50, 95, 200),
        "fine_dining": _band(130, 280, 650),
        "generic": _band(20, 45, 110),
    },
}
_DEFAULT_LEVEL = "mid"
# The last resort, when neither the country nor its currency can be recognised. Deliberately the
# Chinese row rather than the mid price level: it keeps the historical behaviour for a plan whose
# destination is unknown (80 CNY for a plain meal), which is what every existing caller assumed.
_FALLBACK_MEALS = MEAL_BANDS["中国"]
_FALLBACK_HOTEL = HOTEL_BANDS["中国"]
_FALLBACK_ATTRACTION = ATTRACTION_BANDS["中国"]

# Currencies that say something about the price level when the country is unknown.
_CURRENCY_LEVELS: dict[str, str] = {
    "CHF": "expensive", "ISK": "expensive", "NOK": "expensive", "SEK": "expensive",
    "DKK": "expensive", "EUR": "expensive", "GBP": "expensive", "USD": "expensive",
    "CAD": "expensive", "AUD": "expensive", "NZD": "expensive", "SGD": "expensive",
    "HKD": "expensive", "AED": "expensive", "QAR": "expensive", "ILS": "expensive",
    "KRW": "mid", "TWD": "mid", "MOP": "mid", "MYR": "mid", "TRY": "mid", "RUB": "mid",
    "ZAR": "mid", "MXN": "mid", "BRL": "mid", "CNY": "mid", "THB": "cheap",
    "VND": "cheap", "IDR": "cheap", "PHP": "cheap", "INR": "cheap", "EGP": "cheap",
    "LKR": "cheap", "KHR": "cheap", "MMK": "cheap", "NPR": "cheap", "PKR": "cheap",
    "BDT": "cheap", "UZS": "cheap", "KZT": "cheap", "IRR": "cheap", "IQD": "cheap",
}

_LEVEL_BY_COUNTRY: dict[str, str] = {
    country: level for level, countries in _PRICE_LEVELS.items() for country in countries
}


def destination_country(destination: str | None) -> str | None:
    """The country a destination text belongs to, without any network call."""
    text = str(destination or "").strip()
    if not text:
        return None
    named = destination_service.expected_country(text)
    if named:
        return named
    row = destination_service.resolve_offline(text)
    country = str((row or {}).get("country") or "").strip()
    return country or None


def country_key(destination: str | None) -> str | None:
    """The key to price a destination by: its own country row, or its price level.

    A country with no row of its own (Iceland, Cambodia, Peru...) returns ``level:expensive`` /
    ``level:mid`` / ``level:cheap`` rather than its name, because there is no table to look the
    name up in: returning the name made every unlisted country price like China.
    """
    country = destination_country(destination)
    if not country:
        return None
    if country in MEAL_BANDS:
        return country
    level = _LEVEL_BY_COUNTRY.get(country) or _price_level_by_currency(destination)
    return f"level:{level}"


def _price_level_by_currency(destination: str | None) -> str:
    from app.agents.tools import currency  # local import: currency imports nothing from here

    code = currency.currency_for_destination(destination) if destination else None
    return _CURRENCY_LEVELS.get(str(code or ""), _DEFAULT_LEVEL)


def _meals_for(destination: str | None) -> dict[str, Band]:
    key = country_key(destination)
    if key is None:
        return _FALLBACK_MEALS
    if key.startswith("level:"):
        return _LEVEL_MEALS.get(key.split(":", 1)[1], _FALLBACK_MEALS)
    return MEAL_BANDS.get(key, _FALLBACK_MEALS)


def _band_for_country(
    destination: str | None, table: dict[str, Band], level_table: dict[str, Band], fallback: Band
) -> Band:
    key = country_key(destination)
    if key is None:
        return fallback
    if key.startswith("level:"):
        return level_table.get(key.split(":", 1)[1], fallback)
    return table.get(key, fallback)


_LEVEL_HOTELS: dict[str, Band] = {
    "expensive": _band(600, 1400, 4000),
    "mid": _band(180, 480, 1600),
    "cheap": _band(80, 260, 900),
}
_LEVEL_ATTRACTIONS: dict[str, Band] = {
    "expensive": _band(0, 140, 500),
    "mid": _band(0, 60, 260),
    "cheap": _band(0, 25, 110),
}


def hotel_band(destination: str | None) -> Band:
    """What one room costs per night there."""
    return _band_for_country(destination, HOTEL_BANDS, _LEVEL_HOTELS, _FALLBACK_HOTEL)


def attraction_band(destination: str | None) -> Band:
    """What one ticket costs per person there."""
    return _band_for_country(destination, ATTRACTION_BANDS, _LEVEL_ATTRACTIONS, _FALLBACK_ATTRACTION)


def meal_band(
    destination: str | None,
    name: str | None = None,
    price_hint: Any = None,
    reason: str | None = None,
    kind: str | None = None,
) -> Band:
    """The realistic per-person spend for this kind of place at this destination."""
    table = _meals_for(destination)
    key = kind or meal_kind(name, price_hint, reason)
    band = table.get(key)
    if band is None:
        band = table.get(GENERIC_KIND)
    if band is None:
        band = _FALLBACK_MEALS[GENERIC_KIND]
    # The specialised rows are hand-written, so they are used as they are: the whole point of
    # this table is that a proper sushi restaurant in Japan does NOT inherit the floor of a
    # noodle counter. `generic` is only the fallback for a venue whose kind cannot be told.
    return band


# The kinds worth naming in a prompt, in the order a traveller meets them during a day.
PROMPT_KINDS: tuple[tuple[str, str], ...] = (
    ("breakfast", "早餐"),
    ("ramen_noodles", "面/拉面"),
    ("casual_local", "普通正餐/定食"),
    ("cafe_dessert", "咖啡甜品"),
    ("street_food", "小吃/街边"),
    ("sushi_kaiten", "回转/平价寿司"),
    ("sushi_mid", "正经寿司店"),
    ("bbq_grill", "烤肉"),
    ("izakaya_bar", "居酒屋/小酒馆"),
    ("seafood", "海鲜"),
    ("sushi_fine", "高级寿司/主厨套餐"),
    ("fine_dining", "高档餐厅"),
)


def meal_bands_text(destination: str | None, limit: int = len(PROMPT_KINDS)) -> str:
    """A compact 'what a meal costs here' list for the extraction prompt."""
    table = _meals_for(destination)
    parts: list[str] = []
    for kind, label in PROMPT_KINDS:
        band = table.get(kind)
        if band is None:
            continue
        parts.append(f"{label} {band.low}-{band.high}")
        if len(parts) >= limit:
            break
    if not parts:
        return ""
    hotel = hotel_band(destination)
    ticket = attraction_band(destination)
    return (
        "；".join(parts)
        + f"（人民币元/人/餐）。参考：当地酒店 {hotel.low}-{hotel.high} 元/晚，"
        + f"门票 {ticket.low}-{ticket.high} 元/人。"
    )


def price_hint_for_prompt(destination: str | None) -> str:
    """The instruction that makes the model price a meal like a local, not like a tourist."""
    if not str(destination or "").strip():
        # Nothing to be local about, and "目的地 None" in a prompt is worse than silence.
        return ""
    bands = meal_bands_text(destination)
    if not bands:
        return ""
    return (
        f"目的地 {destination} 的常见人均消费（{AS_OF} 估算，仅供参考行情，不是本次价格）：{bands}"
        "**unit_price 必须是「一个人吃一顿正常份量的总价」，不是一道菜、一份小食或单品的价格**："
        "资料里若只有单品价格，请按一顿的份量（主食 + 小菜/饮料）换算成整餐人均再填；"
        "资料里的价格明显低于同类型店的常见人均（例如把回转寿司的价钱用在正经寿司店上）时，"
        "按同类型店的常见人均区间下限填写，并在 price_hint 里保留资料原文。"
        "**不要因为资料便宜就把整数写成低于当地人均下限的数字。**"
    )


def default_unit_price(
    destination: str | None, category: str, name: str | None = None, price_hint: Any = None
) -> int | None:
    """The honest fallback price when the material carried none: a local mid-range figure."""
    category = str(category or "").strip().lower()
    if category == "restaurant":
        return meal_band(destination, name, price_hint).typical
    if category == "hotel":
        return hotel_band(destination).typical
    if category == "attraction":
        return attraction_band(destination).typical
    return None


def realistic_floor(
    destination: str | None, category: str, name: str | None = None, price_hint: Any = None
) -> int | None:
    """The cheapest a full meal / a room / a ticket realistically costs there.

    Only restaurants and hotels have one: a free museum is a real fact, and a 20 CNY temple
    ticket is not a mistake.
    """
    category = str(category or "").strip().lower()
    if category == "restaurant":
        return meal_band(destination, name, price_hint).low
    if category == "hotel":
        return hotel_band(destination).low
    return None


def realistic_ceiling(
    destination: str | None, category: str, name: str | None = None, price_hint: Any = None
) -> int | None:
    """The most a normal place of this kind costs there, used to widen the sanity bounds.

    A 25,000 yen omakase dinner is 1,200 CNY and a real price; the flat 2,000 CNY restaurant
    bound threw it away and replaced it with the mid-range default.
    """
    category = str(category or "").strip().lower()
    if category == "restaurant":
        return meal_band(destination, name, price_hint).high
    if category == "hotel":
        return hotel_band(destination).high
    if category == "attraction":
        return attraction_band(destination).high
    return None


def adjust_price(
    price: int, destination: str | None, category: str, name: str | None = None, price_hint: Any = None
) -> tuple[int, int | None]:
    """Raise a price that cannot buy a full meal there. Returns ``(price, floor_if_raised)``."""
    floor = realistic_floor(destination, category, name, price_hint)
    if floor is None or price >= floor:
        return price, None
    return floor, floor


# --------------------------------------------------------------------------------------
# Price tiers: the budget has to produce a mix, not a single price point.
# --------------------------------------------------------------------------------------
# The traveller asked for exactly this: recommendations that follow the budget they typed, with
# cheap options AND expensive-but-good ones. Which tier a card lands in is derived from the
# per-meal allowance the budget implies.
TIER_CHEAP = "cheap"
TIER_MID = "mid"
TIER_SPLURGE = "splurge"
# Above the splurge ceiling an item is simply not affordable, which is a different statement
# from "worth paying more" and must not be dressed up as one.
TIER_OVER = "over"
TIER_LABELS = {
    TIER_CHEAP: "平价",
    TIER_MID: "中档",
    TIER_SPLURGE: "值得多花钱",
    TIER_OVER: "超出预算",
}
# How far above the allowance the "worth paying more" tier may reach before it counts as
# simply over budget. Bounded on purpose: the point is the good expensive option, not a
# restaurant the traveller cannot pay for.
SPLURGE_MULTIPLIER = 2.5
# Share of the food pool each tier should hold, so the mix survives a thin search result.
TIER_TARGET_SHARES = {TIER_CHEAP: 0.3, TIER_MID: 0.4, TIER_SPLURGE: 0.2}


@dataclass(frozen=True)
class TierPlan:
    """The three price points a budget implies for one category."""

    allowance: int
    cheap_max: int
    mid_max: int
    splurge_min: int
    splurge_max: int

    def tier_of(self, price: int) -> str:
        if price <= self.cheap_max:
            return TIER_CHEAP
        if price <= self.mid_max:
            return TIER_MID
        if price <= self.splurge_max:
            return TIER_SPLURGE
        return TIER_OVER


def tier_plan(allowance: int, floor: int = 0) -> TierPlan | None:
    """The cheap / mid / splurge price points for a per-unit allowance from the budget."""
    try:
        value = int(allowance)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    cheap_max = max(int(floor), int(value * 0.7))
    mid_max = int(value * 1.3)
    # A splurge band that would sit below the mid band is meaningless: keep it above.
    splurge_min = max(mid_max + 1, int(value * 1.8))
    splurge_max = max(splurge_min, int(value * SPLURGE_MULTIPLIER))
    return TierPlan(
        allowance=value,
        cheap_max=cheap_max,
        mid_max=mid_max,
        splurge_min=splurge_min,
        splurge_max=splurge_max,
    )


def tier_brief(plan: TierPlan | None, category: str = "restaurant") -> str | None:
    """The instruction that turns an allowance into a three-tier mix."""
    if plan is None:
        return None
    unit = "每餐每人" if category == "restaurant" else "每晚"
    return (
        f"本行程按预算推算的{unit}可接受价位约 {plan.allowance} 元。"
        f"请给出**三个档次**的选项，三个档次都必须有："
        f"平价档（约 {plan.cheap_max} 元以下，当地人也常去的实惠店）、"
        f"中档档（约 {plan.cheap_max}-{plan.mid_max} 元，主力选择）、"
        f"高档档（约 {plan.splurge_min}-{plan.splurge_max} 元，**贵但口碑好、值得专程去吃的名店**）。"
        "**不要只给便宜的，也不要只给贵的**：平价档要能天天吃，高档档要是真正好吃的那家。"
    )


def ensure_tier_spread(items: list[dict[str, Any]], plan: TierPlan | None, category: str) -> dict[str, int]:
    """Tag every item with its tier and report how many landed in each.

    The tag travels to the card, which is what lets the interface say "cheap" or "worth paying
    more" instead of one undifferentiated list of restaurants.
    """
    counts = {TIER_CHEAP: 0, TIER_MID: 0, TIER_SPLURGE: 0}
    for item in items:
        if str(item.get("type") or "") != category:
            continue
        price = item.get("unit_price")
        if not isinstance(price, int) or price <= 0:
            continue
        tier = plan.tier_of(price) if plan is not None else TIER_MID
        item["price_tier"] = tier
        item["price_tier_label"] = TIER_LABELS.get(tier, "")
        counts[tier] = counts.get(tier, 0) + 1
    return counts


def missing_tiers(counts: dict[str, int], total: int) -> list[str]:
    """Which tiers are thin, given how many candidates the pool holds."""
    if total <= 0:
        return []
    missing = []
    for tier, share in TIER_TARGET_SHARES.items():
        wanted = max(1, int(total * share))
        if counts.get(tier, 0) < wanted:
            missing.append(tier)
    return missing


def tier_search_hint(tier: str, base: str | None = None) -> str:
    """The search wording that finds the missing tier instead of more of the same."""
    if tier == TIER_CHEAP:
        return "便宜 实惠 当地人常去 平价 人均低"
    if tier == TIER_SPLURGE:
        return "米其林 名店 高级 招牌 值得专程 好吃 排队"
    return base or "推荐 人均"
