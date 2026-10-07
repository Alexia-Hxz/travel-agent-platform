"""Reference facts a plan cannot derive from a photo: a hotel's rating and price, and a
restaurant's signature dish.

The user asked for the picture rules to change in two ways:

* **Hotels get no picture at all.** A stock photo of some other hotel is not information, so
  a hotel card shows its rating and its nightly price instead.
* **Restaurants show food, not the shopfront.** A picture of a generic street is worse than
  useless on a food card, so the search term becomes the signature dish.

Both come from the model's own live search (`enable_search`), which answers in Chinese with
current information and costs nothing against the search provider's quota. Answers are
batched because one answer is only reliable up to about ten items; larger requests came back
as `400 data_inspection_failed` (see the round-8 notes in PROJECT_HANDOFF.md).

Everything here is best-effort. A missing rating or dish leaves the field empty and the rest
of the plan is unaffected, and ratings are labelled as a reference value in the interface
because they come from a summary rather than from the hotel.
"""

from concurrent.futures import ThreadPoolExecutor
import json
import logging
import re
from typing import Any

from app.agents.tools import currency, dashscope_search

logger = logging.getLogger(__name__)

# Items per live-search call. The round-8 measurements put the model's reliable output at
# about twelve items; ten leaves room for the extra fields each answer now carries.
FACTS_BATCH = 10
FACTS_WORKERS = 3
# What one batch is allowed to take. Measured on this provider, a ten-hotel batch answered in
# 9.4s and a ten-restaurant batch in 11.3s, so this clears both by a wide margin. The bound
# exists for the tail: a batch call inherits the 120s search timeout, and this phase has no
# wall clock of its own, so one stalled lookup could hold an otherwise finished plan open for
# two minutes for a rating the reader sees at a glance. Bounding the call itself (rather than
# walking away from the thread) keeps the items it writes deterministic: nothing mutates a
# recommendation after this phase returns.
FACTS_CALL_TIMEOUT = 20.0
# A rating is on a 0-5 scale and the nightly price of a hotel in CNY. Anything outside these
# is a parsing accident (a phone number, a review count) rather than a fact.
MAX_RATING = 5.0
HOTEL_PRICE_MIN = 60
HOTEL_PRICE_MAX = 20000
MAX_DISH_CHARS = 14

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _parse_object(text: str) -> dict[str, Any] | None:
    """Read the first JSON object out of an answer that may wrap it in prose or a fence."""
    if not text:
        return None
    cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", str(text).strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    match = _JSON_BLOCK.search(cleaned)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _clean_rating(value: Any) -> float | None:
    try:
        rating = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if rating <= 0 or rating > MAX_RATING:
        return None
    return round(rating, 1)


def _clean_hotel_price(value: Any, destination: str | None = None) -> int | None:
    """Accept "620", "620元", "6,80" — but reject anything implausible as a nightly rate.

    A nightly rate the model copied out of foreign material ("30000" for a Tokyo hotel) is
    outside the CNY bounds; rather than throwing the price away, it is read as the local
    currency and converted, which is what the number almost always is.
    """
    digits = re.sub(r"[^\d]", "", str(value or ""))
    if not digits:
        return None
    price = int(digits)
    if price < HOTEL_PRICE_MIN or price > HOTEL_PRICE_MAX:
        if currency.is_foreign(destination):
            local = currency.currency_for_destination(destination)
            converted = currency.to_cny(price, local)
            if HOTEL_PRICE_MIN <= converted <= HOTEL_PRICE_MAX:
                return converted
        return None
    return price


def _clean_dish(value: Any) -> str | None:
    """A dish name is short and written in Chinese; anything else is not a dish name."""
    text = str(value or "").strip()
    if not text or len(text) > MAX_DISH_CHARS:
        return None
    if not any("\u4e00" <= char <= "\u9fff" for char in text):
        return None
    return text


def _chunks(items: list[str], size: int = FACTS_BATCH) -> list[list[str]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _hotel_prompt(destination: str, names: list[str]) -> str:
    return (
        f"请联网查询以下{destination}的酒店，给出每家酒店的**参考评分（满分5分，一位小数）**"
        f"和**参考房价（人民币元/晚，只写数字）**。\n"
        f"酒店列表：{json.dumps(names, ensure_ascii=False)}\n"
        "只返回一个 JSON 对象，键是酒店原名，值是 "
        '{"rating": 4.7, "price": 620}。'
        + currency.prompt_hint(destination)
        + "查不到评分或房价的酒店，对应字段写 null。不要编造，不要输出任何解释文字。"
    )


def _dish_prompt(destination: str, names: list[str]) -> str:
    return (
        f"请联网查询以下{destination}餐厅各自的**招牌菜**。\n"
        f"餐厅列表：{json.dumps(names, ensure_ascii=False)}\n"
        "只返回一个 JSON 对象，键是餐厅原名，值是 "
        '{"dish": "盐水鸭", "dish_query": "Nanjing salted duck"}。'
        "dish 用中文菜名（不超过14个字）；dish_query 是用于在图库中检索这道菜的英文关键词。"
        "查不到招牌菜的餐厅写 null。不要编造，不要输出任何解释文字。"
    )


def _ask(prompt: str) -> dict[str, Any] | None:
    """One live-search call, parsed; never raises."""
    if not dashscope_search.search_available():
        return None
    try:
        answer = dashscope_search.grounded_answer(prompt, timeout=FACTS_CALL_TIMEOUT)
    except Exception as error:  # pragma: no cover - grounded_answer already swallows
        logger.warning("place facts lookup failed: %s", error)
        return None
    parsed = _parse_object(answer or "")
    if parsed is None and answer:
        logger.warning("place facts answer was not usable JSON")
    return parsed


def _apply_hotel_facts(items: list[dict[str, Any]], destination: str) -> int:
    facts = _ask(_hotel_prompt(destination, [str(item.get("name")) for item in items]))
    if not facts:
        return 0
    applied = 0
    for item in items:
        value = facts.get(str(item.get("name")))
        if not isinstance(value, dict):
            continue
        rating = _clean_rating(value.get("rating"))
        price = _clean_hotel_price(value.get("price"), destination)
        if rating is not None:
            item["rating"] = rating
        if price is not None:
            # The nightly rate is what the budget should charge for this hotel, and it is
            # marked as quoted rather than defaulted so the cost line says where it came from.
            item["unit_price"] = price
            item["price_unit"] = "night"
            item["price_hint"] = f"约 {price} 元/晚"
            applied += 1
    return applied


def _apply_dish_facts(items: list[dict[str, Any]], destination: str) -> int:
    facts = _ask(_dish_prompt(destination, [str(item.get("name")) for item in items]))
    if not facts:
        return 0
    applied = 0
    for item in items:
        value = facts.get(str(item.get("name")))
        if not isinstance(value, dict):
            continue
        dish = _clean_dish(value.get("dish"))
        if not dish:
            continue
        item["signature_dish"] = dish
        query = str(value.get("dish_query") or "").strip()
        # The English keyword is what makes the library return the dish rather than the
        # city; a missing one still beats searching for the restaurant name.
        item["dish_query"] = query[:60] if query else dish
        applied += 1
    return applied


def enrich_place_facts(items: list[dict[str, Any]], destination: str) -> list[dict[str, Any]]:
    """Fill in hotel ratings/prices and restaurant signature dishes, in place.

    Batches run concurrently: a plan holds about twenty restaurants, which is two or three
    calls, and running them one after another would add a minute to every plan for facts the
    user reads at a glance.
    """
    hotels = [item for item in items if str(item.get("type")) == "hotel"]
    restaurants = [item for item in items if str(item.get("type")) == "restaurant"]
    jobs: list[tuple[Any, list[dict[str, Any]]]] = []
    for group, worker in ((hotels, _apply_hotel_facts), (restaurants, _apply_dish_facts)):
        by_name = {str(item.get("name")): item for item in group}
        for batch in _chunks(list(by_name)):
            jobs.append((worker, [by_name[name] for name in batch]))
    if not jobs:
        return items
    pool = ThreadPoolExecutor(max_workers=min(FACTS_WORKERS, len(jobs)))
    try:
        futures = [pool.submit(job, batch_items, destination) for job, batch_items in jobs]
        # Each call carries its own bound, so collecting them in submission order cannot wait
        # longer than a single batch -- and no thread is left writing to an item afterwards.
        for future in futures:
            try:
                future.result()
            except Exception:
                # A missing rating or dish must never break planning.
                continue
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return items
