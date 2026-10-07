"""Dish photos from TheMealDB, for the food cards a stock library cannot place.

Why a second food source exists
-------------------------------
The food card's picture is not of the restaurant -- a stock library has no photo of one
particular noodle shop -- but of the **dish**, and that changes what "correct" means. A card
reading "豚骨拉面" with a photo of a ramen bowl is right; the same card with a photo of
snow-covered horse saddles is not. Measured with English keywords, Pexels gets eight of ten
Japanese dishes right and misses tonkatsu and yakitori; TheMealDB has both ("Tonkatsu pork",
"Smoky chicken skewers") and misses the four Pexels gets right (unagi, soba, matcha,
okonomiyaki). The two libraries cover each other's gaps, which is exactly why both are used.

TheMealDB is free and needs no key. The public test key ``1`` is documented for development
use; for a commercial deployment, check their terms and use a supporter key.
"""

import logging
import threading
from typing import Any

import httpx

logger = logging.getLogger(__name__)

SEARCH_URL = "https://www.themealdb.com/api/json/v1/1/search.php"
FILTER_URL = "https://www.themealdb.com/api/json/v1/1/filter.php"
# TheMealDB asks developers to use the key "1" while evaluating, and issues a personal key
# for production. Nothing secret is involved either way, so it is not read from the env.
TEST_KEY = "1"
SEARCH_TIMEOUT = 8.0
MAX_QUERY_CHARS = 40
SOURCE_LABEL = "TheMealDB"

_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
_cache_lock = threading.Lock()


def _photo_record(meal: dict[str, Any]) -> dict[str, Any] | None:
    url = str(meal.get("strMealThumb") or "").strip()
    if not url.startswith(("http://", "https://")):
        return None
    name = " ".join(str(meal.get("strMeal") or "").split())
    meal_id = str(meal.get("idMeal") or "").strip()
    return {
        "url": url,
        "alt": name.lower(),
        "title": name or None,
        "page_url": f"https://www.themealdb.com/meal/{meal_id}" if meal_id else None,
        # Not a Pexels photo, so the Pexels credit line must not be used for it.
        "photographer": None,
        "photographer_url": None,
        "source": "mealdb",
    }


def _request(url: str, params: dict[str, str]) -> list[dict[str, Any]]:
    try:
        response = httpx.get(url, params=params, timeout=SEARCH_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError, TypeError) as error:
        logger.warning("themealdb lookup failed: %s", error)
        return []
    if not isinstance(payload, dict):
        return []
    meals = payload.get("meals")
    if not isinstance(meals, list):
        return []
    records: list[dict[str, Any]] = []
    for meal in meals:
        if not isinstance(meal, dict):
            continue
        record = _photo_record(meal)
        if record is not None:
            records.append(record)
    return records


def search_dish_photos(query: str, limit: int = 3) -> list[dict[str, Any]]:
    """Photos matching one dish name; never raises, and returns [] when nothing matches."""
    text = " ".join(str(query or "").split())
    if not text or len(text) > MAX_QUERY_CHARS:
        return []
    key = (text.casefold(), "search")
    with _cache_lock:
        cached = _cache.get(key)
    if cached is not None:
        return cached[:limit]
    records = _request(SEARCH_URL, {"s": text, "apikey": TEST_KEY})
    # Only a real hit is remembered: caching an empty answer would pin one miss onto this
    # dish for the lifetime of the process.
    if records:
        with _cache_lock:
            _cache[key] = records
    return records[:limit]


def area_photos(area: str, limit: int = 12) -> list[dict[str, Any]]:
    """Every dish the library has for a cuisine, e.g. "Japanese".

    This is the category-level fallback: when the exact dish is not in the library, a photo
    of the right cuisine is still a picture that matches the card.
    """
    text = " ".join(str(area or "").split())
    if not text:
        return []
    key = (text.casefold(), "area")
    with _cache_lock:
        cached = _cache.get(key)
    if cached is not None:
        return cached[:limit]
    records = _request(FILTER_URL, {"a": text, "apikey": TEST_KEY})
    if records:
        with _cache_lock:
            _cache[key] = records
    return records[:limit]


def credit_line(photo: dict[str, Any]) -> str | None:
    """Attribution for a TheMealDB photo."""
    title = str(photo.get("title") or "").strip()
    return f"{title} · TheMealDB" if title else SOURCE_LABEL


# The library's `area` values, matched as substrings of the destination. Only the cuisines
# it actually carries are listed: an unmatched destination simply gets no cuisine fallback.
_AREAS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("日本", "东京", "東京", "大阪", "京都", "北海道", "冲绳", "名古屋", "福冈", "神户", "奈良",
      "japan", "tokyo", "osaka", "kyoto", "hokkaido", "okinawa", "nagoya"), "Japanese"),
    (("中国", "北京", "上海", "南京", "杭州", "成都", "广州", "深圳", "西安", "重庆", "苏州",
      "china", "beijing", "shanghai", "nanjing", "hangzhou", "chengdu", "guangzhou"), "Chinese"),
    (("韩国", "首尔", "釜山", "济州", "korea", "seoul", "busan"), "Korean"),
    (("泰国", "曼谷", "清迈", "普吉", "thailand", "bangkok", "phuket"), "Thai"),
    (("越南", "河内", "胡志明", "vietnam", "hanoi"), "Vietnamese"),
    (("印度", "新德里", "孟买", "india", "delhi", "mumbai"), "Indian"),
    (("意大利", "罗马", "米兰", "威尼斯", "italy", "rome", "milan", "venice"), "Italian"),
    (("法国", "巴黎", "france", "paris"), "French"),
    (("希腊", "雅典", "greece", "athens"), "Greek"),
    (("墨西哥", "mexico"), "Mexican"),
    (("埃及", "开罗", "egypt", "cairo"), "Egyptian"),
    (("摩洛哥", "morocco"), "Moroccan"),
    (("土耳其", "伊斯坦布尔", "turkey", "istanbul"), "Turkish"),
    (("英国", "伦敦", "uk", "london"), "British"),
    (("美国", "纽约", "洛杉矶", "usa", "new york", "los angeles"), "American"),
    (("西班牙", "西班牙", "spain", "madrid", "barcelona"), "Spanish"),
    (("加拿大", "canada"), "Canadian"),
    (("荷兰", "netherlands"), "Dutch"),
    (("波兰", "poland"), "Polish"),
    (("葡萄牙", "portugal"), "Portuguese"),
    (("俄罗斯", "russia"), "Russian"),
    (("乌克兰", "ukraine"), "Ukrainian"),
    (("克罗地亚", "croatia"), "Croatian"),
    (("挪威", "norway"), "Norwegian"),
    (("牙买加", "jamaica"), "Jamaican"),
    (("肯尼亚", "kenya"), "Kenyan"),
    (("突尼斯", "tunisia"), "Tunisian"),
    (("菲律宾", "philippines", "manila"), "Filipino"),
    (("马来西亚", "malaysia", "kuala lumpur"), "Malaysian"),
)


def area_for_destination(destination: str | None) -> str | None:
    """The library's cuisine name for a destination, or None when it has no dishes there."""
    text = str(destination or "").strip().casefold()
    if not text:
        return None
    for markers, area in _AREAS:
        if any(marker in text for marker in markers):
            return area
    return None


def clear_cache() -> None:  # noqa: D401 - see below
    """Forget cached dish lookups. Intended for tests."""
    with _cache_lock:
        _cache.clear()
