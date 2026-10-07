"""Arrange one day, from material the trip already has.

The whole-trip planner is thorough on purpose: for every category it searches the live web,
prices what it finds, looks up facts and fills every day. That is what a first plan needs and
it is why a plan takes a few minutes.

This module answers the narrower question asked later on — "just arrange this day for me" —
and it can be quick because almost nothing has to be discovered again:

* the search material for a destination is cached per category, so a second visit to the
  same city does not search the web at all;
* only two categories matter for a day (sights and food), and they are filled at the same
  time rather than one after the other;
* prices come from the material itself, without the live price lookup or the fact pass.

Two rules shape the result, and both are deliberate:

* **Nothing already spoken for may come back.** Everything on another day, and everything
  the trip already has saved as a place, is passed down as an exclusion; the extraction is
  told not to list them and anything that comes back anyway is dropped — by exact name and
  by the same "one venue, two names" rule the daily plan uses.
* **A day that cannot be filled is reported short.** Two sights and three meals is the
  target, not a quota: a missing breakfast stays missing rather than repeating one.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import date as Date, time
from typing import Any, Iterable

from app.agents.tools.place_images import resolve_place_images
from app.agents.tools.recommendations import (
    classify_meal_slot,
    estimate_trip_costs,
    _fill_category,
)


# One day's worth, matching what the whole-trip daily plan gives a day: two sights and three
# meals. The restaurant count asks for one extra because the breakfast has to be recognised
# as such before it can be placed, and not every candidate the model returns is a breakfast.
DAY_SIGHTS = 2
DAY_MEALS = 3
_RESTAURANTS_ASKED_FOR = 4

# Where each part of the day goes. The same clock the whole-trip plan uses, so a day built
# here and a day built there read the same.
DAY_SLOTS: tuple[tuple[str, str, str, str], ...] = (
    ("breakfast", "08:00:00", "09:00:00", "当地早餐"),
    ("morning", "09:30:00", "12:00:00", "上午主要游览，预留交通与拍照时间"),
    ("lunch", "12:00:00", "13:00:00", "当地特色午餐"),
    ("afternoon", "14:00:00", "17:00:00", "下午继续游览，按现场人流灵活调整"),
    ("dinner", "18:00:00", "19:30:00", "晚餐"),
)


def _same_venue(first: str, second: str) -> bool:
    """Whether two names are one venue written at different lengths.

    Kept in step with the planner's own rule: ``蒋有记`` and ``老门东内蒋有记`` are the same
    restaurant, so a day that already has one must not be offered the other.
    """
    first_key = first.strip().casefold()
    second_key = second.strip().casefold()
    if not first_key or not second_key:
        return False
    if first_key == second_key:
        return True
    shorter, longer = sorted((first_key, second_key), key=len)
    if shorter not in longer:
        return False
    return not longer.replace(shorter, "", 1).isdigit()


def _is_unused(place: dict[str, Any], used: Iterable[str]) -> bool:
    name = str(place.get("name") or "").strip()
    if not name:
        return False
    return not any(_same_venue(name, other) for other in used)


def _pick(pool: Iterable[dict[str, Any]], count: int, taken: list[str]) -> list[dict[str, Any]]:
    """Take up to ``count`` places, remembering each name so it cannot come back twice."""
    chosen: list[dict[str, Any]] = []
    for place in pool:
        if len(chosen) >= count:
            break
        name = str(place.get("name") or "").strip()
        if not name or not _is_unused(place, taken):
            continue
        taken.append(name)
        chosen.append(place)
    return chosen


def _is_breakfast(place: dict[str, Any]) -> bool:
    return classify_meal_slot(str(place.get("name") or ""), place.get("meal_slot")) == "breakfast"


# What a day needs before a category counts as filled, and the second angle to search from when
# it is not. The first material for a city names its famous places; once those are all on other
# days (or already saved) the day would come back as meals only, which reads as a bug even though
# it is the no-repeats rule working. Searching once more from a different angle fixes that.
_WIDEN_FLOOR = {"attraction": DAY_SIGHTS, "restaurant": DAY_MEALS - 1}
_WIDEN_HINTS = {
    "attraction": "本地小众景点、新开的展馆和值得专程一去的体验，避开最知名的常规景点",
    "restaurant": "本地人常去、游客少的餐厅和小吃，避开最知名的网红店",
}


def _usable(pool: Iterable[dict[str, Any]], exclude: Iterable[str]) -> list[dict[str, Any]]:
    """The places in ``pool`` that the no-repeats rule still allows."""
    return [place for place in pool if _is_unused(place, exclude)]


def _widen_thin_categories(
    destination: str,
    found: dict[str, list[dict[str, Any]]],
    exclude: tuple[str, ...],
    allowances: dict[str, int] | None,
    plan: Any,
    traveler_profile: dict[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Search again, from a different angle, for a category the first pass could not fill."""
    thin = [
        category
        for category, floor in _WIDEN_FLOOR.items()
        if len(_usable(found.get(category) or [], exclude)) < floor
    ]
    if not thin:
        return found
    children = sum(
        int((traveler_profile or {}).get(key) or 0)
        for key in ("child_male_count", "child_female_count")
    )
    with ThreadPoolExecutor(max_workers=len(thin), thread_name_prefix="day-plan-widen") as pool:
        futures = {}
        for category in thin:
            hint = _WIDEN_HINTS[category]
            if children and category == "attraction":
                hint = f"适合带孩子的{hint}"
            futures[
                pool.submit(
                    _fill_category,
                    destination,
                    category,
                    _WIDEN_FLOOR[category] + 2,
                    allowances,
                    plan,
                    hint,
                    exclude,
                )
            ] = category
        for future, category in futures.items():
            try:
                found[category] = [*(found.get(category) or []), *future.result()]
            except Exception:  # the widened search failing leaves the first pass's answer
                continue
    return found


def plan_one_day(
    destination: str,
    trip_days: int,
    used_names: Iterable[str],
    traveler_profile: dict[str, Any] | None = None,
    budget: int | None = None,
    allowances: dict[str, int] | None = None,
    plan: Any = None,
) -> list[dict[str, Any]]:
    """Return one day's places: two sights, a breakfast, a lunch and a dinner.

    ``used_names`` is everything that must not be offered again — the other days' places and
    the trip's saved places. Fewer places come back when the material cannot fill the day;
    nothing is ever repeated to make up the numbers.
    """
    found: dict[str, list[dict[str, Any]]] = {}
    wanted = {"attraction": DAY_SIGHTS + 1, "restaurant": _RESTAURANTS_ASKED_FOR}
    exclude = tuple(sorted({str(name).strip() for name in used_names if str(name).strip()}))
    # One search and one extraction per category, both at once: the two are independent, and
    # a day is small enough that running them in turn would simply add the waits together.
    with ThreadPoolExecutor(max_workers=len(wanted), thread_name_prefix="day-plan") as pool:
        futures = {
            pool.submit(
                _fill_category,
                destination,
                category,
                count,
                allowances,
                plan,
                None,
                exclude,
            ): category
            for category, count in wanted.items()
        }
        for future, category in futures.items():
            try:
                found[category] = future.result()
            except Exception:  # a category that fails is a shorter day, not a failed draft
                found[category] = []

    # Anything the first pass could not fill is searched again from another angle, so a day in a
    # city the trip already knows well does not come back as meals only.
    found = _widen_thin_categories(destination, found, exclude, allowances, plan, traveler_profile)

    restaurants = found.get("restaurant") or []
    taken: list[str] = list(exclude)
    breakfast = _pick([place for place in restaurants if _is_breakfast(place)], 1, taken)
    sights = _pick(found.get("attraction") or [], DAY_SIGHTS, taken)
    meals = _pick([place for place in restaurants if not _is_breakfast(place)], DAY_MEALS - 1, taken)
    # Breakfast first, then the day's shape: sights and meals around it.
    chosen = [*breakfast, *sights, *meals]
    if not chosen:
        return []
    chosen = estimate_trip_costs(chosen, trip_days, traveler_profile, budget, destination=destination)
    # Hotels are not part of a day here: the trip already has one, and a stock photo of a
    # different hotel says nothing. The same rule the whole-trip plan follows.
    return resolve_place_images(chosen, destination, skip_categories=frozenset({"hotel"}))


def shortfall_notes(places: list[dict[str, Any]]) -> list[str]:
    """What a day asked for and did not get. Reported, never filled by repeating."""
    notes: list[str] = []
    sights = [place for place in places if place.get("type") == "attraction"]
    restaurants = [place for place in places if place.get("type") == "restaurant"]
    if not any(_is_breakfast(place) for place in restaurants):
        notes.append("早餐")
    if len(sights) < DAY_SIGHTS:
        notes.append("景点")
    if len([place for place in restaurants if not _is_breakfast(place)]) < DAY_MEALS - 1:
        notes.append("午餐或晚餐")
    return notes


def _place_payload(place: dict[str, Any], destination: str) -> dict[str, Any]:
    """The saved-place shape for one of the places above.

    A picture only travels with the place when it is an absolute URL: the saved-place model
    requires one, and the bundled placeholder is a path this server serves, not a link the
    client can open. The card falls back to its category picture in that case.
    """
    image = str(place.get("image_url") or "").strip()
    rating = place.get("rating")
    price_level = place.get("price_level")
    return {
        "name": str(place.get("name") or "").strip()[:160],
        "type": place.get("type") or "attraction",
        "address": (str(place.get("address") or "").strip() or f"{destination} · 地址待确认")[:300],
        "latitude": place.get("latitude"),
        "longitude": place.get("longitude"),
        "image_url": image if image.startswith(("http://", "https://")) else None,
        "rating": rating if isinstance(rating, (int, float)) and 0 <= rating <= 5 else None,
        "price_level": price_level if isinstance(price_level, int) and 1 <= price_level <= 5 else None,
        "estimated_cost": place.get("estimated_cost"),
        "quantity": place.get("quantity"),
        "note": str(place.get("reason") or "").strip() or None,
    }


def day_operations(
    target_date: Date,
    places: list[dict[str, Any]],
    existing_items: list[dict[str, Any]],
    destination: str,
) -> list[dict[str, Any]]:
    """Replace one day: drop what it had, then place the new day onto the clock.

    Nothing outside ``target_date`` is touched, which is what makes this safe to offer beside
    a plan the traveller has already arranged by hand.
    """
    stamp = target_date.isoformat()
    operations: list[dict[str, Any]] = [
        {"kind": "remove_item", "item_id": int(item["id"])}
        for item in existing_items
        if item.get("id") and str(item.get("date") or "")[:10] == stamp
    ]
    sights = [place for place in places if place.get("type") == "attraction"]
    restaurants = [place for place in places if place.get("type") == "restaurant"]
    breakfasts = [place for place in restaurants if _is_breakfast(place)]
    meals = [place for place in restaurants if not _is_breakfast(place)]
    taken: list[str] = []
    for slot, start, end, note in DAY_SLOTS:
        if slot == "breakfast":
            picked = _pick(breakfasts, 1, taken)
        elif slot in ("morning", "afternoon"):
            picked = _pick(sights, 1, taken)
        else:
            picked = _pick(meals, 1, taken)
        for place in picked:
            operations.append(
                {
                    "kind": "add_place_and_item",
                    "place": _place_payload(place, destination),
                    "date": target_date,
                    "start_time": time.fromisoformat(start),
                    "end_time": time.fromisoformat(end),
                    "note": note,
                    "sort_order": len(operations),
                }
            )
    return operations


def day_summary(places: list[dict[str, Any]], short: list[str]) -> str:
    """One line for the draft: what the day now holds, and what it could not fill."""
    names = [str(place.get("name") or "").strip() for place in places]
    line = (
        f"这一天按极简方式重新安排：{'、'.join(name for name in names if name)}。"
        if any(names)
        else "这一天暂时没有可安排的推荐地点。"
    )
    if short:
        line += f"素材里暂时没有找到{'、'.join(short)}，已如实留空而不是重复别处的推荐。"
    return line


# ---- picking places to order, by hand -------------------------------------------------------
#
# The traveller can also ask for places one question at a time ("有没有更适合亲子的"). That is
# the same search-and-extract machinery as above, pointed at their own wording instead of at a
# whole day, and it stays a *suggestion*: every candidate arrives as a separate operation the
# traveller ticks or ignores, and nothing is removed unless they asked for a swap.

# How many candidates one question gets.
DISCOVERY_TARGET = 5

# Words that say which kind of place a question is about. A question that mentions neither, or
# both, is answered with both kinds rather than silently guessing one.
_FOOD_WORDS = ("吃", "餐", "美食", "小吃", "夜宵", "早餐", "午餐", "晚餐", "咖啡", "甜点", "酒吧")
_SIGHT_WORDS = ("景点", "玩", "逛", "博物", "美术", "展", "公园", "亲子", "孩子", "购物", "拍照", "文艺",
                "温泉", "体验", "夜游", "山", "湖", "古镇", "值得去")

# Words that ask for a swap. Without one of these the draft only adds.
_REPLACE_WORDS = ("换成", "替换", "改成", "改为", "换掉", "不要", "别去", "去掉", "删除")

# Where a hand-picked candidate is suggested when the traveller did not say. Sights alternate
# through the two sight slots; a restaurant goes to the meal its own name implies when it is a
# breakfast, otherwise to lunch or dinner.
_DISCOVERY_SLOTS: dict[str, list[tuple[str, str]]] = {
    "attraction": [("09:30:00", "12:00:00"), ("14:00:00", "17:00:00"), ("17:00:00", "18:00:00")],
    "restaurant": [("12:00:00", "13:00:00"), ("18:00:00", "19:30:00"), ("13:00:00", "14:00:00"),
                   ("19:30:00", "21:00:00")],
}
_BREAKFAST_SLOT = ("08:00:00", "09:00:00")

# Half-hour windows from 08:00 to 21:30, used after the standard slots above are taken. A
# question can return more candidates than a day has sight slots or meals, and two candidates
# landing on the same time is exactly the thing the traveller then has to untangle by hand.
_SHIFT_WINDOWS: tuple[tuple[str, str], ...] = tuple(
    (f"{hour:02d}:{minute:02d}:00", f"{hour:02d}:{minute + 30:02d}:00") if minute == 0
    else (f"{hour:02d}:{minute:02d}:00", f"{hour + 1:02d}:00:00")
    for hour in range(8, 22)
    for minute in (0, 30)
)


def _candidate_windows(kind: str, breakfast: bool) -> tuple[tuple[str, str], ...]:
    """Every window a candidate of this kind may take, its usual slot first."""
    if kind == "restaurant":
        return ((_BREAKFAST_SLOT,) if breakfast else ()) + tuple(_DISCOVERY_SLOTS["restaurant"]) + _SHIFT_WINDOWS
    return tuple(_DISCOVERY_SLOTS["attraction"]) + _SHIFT_WINDOWS


def discovery_categories(question: str) -> list[str]:
    """Which kinds of place a question is asking about, sights first."""
    text = question.strip()
    food = any(word in text for word in _FOOD_WORDS)
    sights = any(word in text for word in _SIGHT_WORDS)
    if food and not sights:
        return ["restaurant"]
    if sights and not food:
        return ["attraction"]
    return ["attraction", "restaurant"]


def asks_to_replace(question: str) -> bool:
    """Whether the traveller asked for a swap rather than an addition."""
    return any(word in question for word in _REPLACE_WORDS)


def plan_discovery(
    destination: str,
    question: str,
    used_names: Iterable[str],
    trip_days: int,
    traveler_profile: dict[str, Any] | None = None,
    budget: int | None = None,
    allowances: dict[str, int] | None = None,
    plan: Any = None,
    target: int = DISCOVERY_TARGET,
) -> list[dict[str, Any]]:
    """Return up to ``target`` places that answer ``question`` and are new to the trip.

    The traveller's own words are passed down as the search hint, so a question about family
    days searches for family days instead of returning the same famous handful every time. The
    hint is part of the material cache key, so asking the same question twice is free.
    """
    categories = discovery_categories(question)
    counts: dict[str, int] = {}
    remaining = max(1, target)
    for index, category in enumerate(categories):
        share = remaining if index == len(categories) - 1 else max(2, remaining - 2)
        counts[category] = share
        remaining -= share

    exclude = tuple(sorted({str(name).strip() for name in used_names if str(name).strip()}))
    hint = question.strip()[:200] or None
    found: dict[str, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=len(counts), thread_name_prefix="discover") as pool:
        futures = {
            pool.submit(_fill_category, destination, category, count, allowances, plan, hint, exclude): category
            for category, count in counts.items()
        }
        for future, category in futures.items():
            try:
                found[category] = future.result()
            except Exception:  # one empty category is fewer candidates, not a failed search
                found[category] = []

    taken: list[str] = list(exclude)
    chosen: list[dict[str, Any]] = []
    for category in categories:
        chosen.extend(_pick(found.get(category) or [], counts[category], taken))
    chosen = chosen[:target]
    if not chosen:
        return []
    chosen = estimate_trip_costs(chosen, trip_days, traveler_profile, budget, destination=destination)
    return resolve_place_images(chosen, destination, skip_categories=frozenset({"hotel"}))


def discovery_operations(
    target_date: Date,
    places: list[dict[str, Any]],
    existing_items: list[dict[str, Any]],
    destination: str,
    replace: bool = False,
) -> list[dict[str, Any]]:
    """One additive operation per candidate, optionally swapping out that day's same-kind items.

    ``replace`` is only set when the traveller's own words asked for a swap ("换成亲子的"): the
    draft then also drops that day's items of the kinds being offered, so the day does not end
    up with both the old sight and its replacement. Without it the day is left alone and the
    candidate slots are chosen around what is already there.
    """
    stamp = target_date.isoformat()
    day_items = [
        item
        for item in existing_items
        if item.get("id") and str(item.get("date") or "")[:10] == stamp
    ]
    operations: list[dict[str, Any]] = []
    if replace:
        kinds = {str(place.get("type") or "attraction") for place in places}
        operations.extend(
            {"kind": "remove_item", "item_id": int(item["id"])}
            for item in day_items
            if str(item.get("type") or "attraction") in kinds
        )
        kept = [item for item in day_items if str(item.get("type") or "attraction") not in kinds]
    else:
        kept = day_items

    busy = {
        (str(item.get("start_time") or "")[:8], str(item.get("end_time") or "")[:8])
        for item in kept
    }
    for place in places:
        kind = str(place.get("type") or "attraction")
        windows = _candidate_windows(kind, kind == "restaurant" and _is_breakfast(place))
        # The first window this day does not already use, standard slots first. Because every
        # window taken is remembered, no two candidates -- and no candidate and an item the day
        # keeps -- can end up on the same time.
        start, end = next((window for window in windows if window not in busy), windows[-1])
        busy.add((start, end))
        operations.append(
            {
                "kind": "add_place_and_item",
                "place": _place_payload(place, destination),
                "date": target_date,
                "start_time": time.fromisoformat(start),
                "end_time": time.fromisoformat(end),
                "note": str(place.get("reason") or "").strip() or "按你的要求找到的候选，时间可以自己改",
                "sort_order": len(operations),
            }
        )
    return operations


def discovery_summary(places: list[dict[str, Any]], question: str) -> str:
    """One line for the draft: what came back, and that nothing happens without a tick."""
    names = [str(place.get("name") or "").strip() for place in places]
    if not any(names):
        return (
            f"没有找到能回答「{question.strip()[:40]}」的新地点。换一种说法、换一天，或者先让「AI 发现」"
            "为这座城市补一轮素材，再来问会更准。"
        )
    return (
        f"按你的要求找到 {len([name for name in names if name])} 个候选：{'、'.join(name for name in names if name)}。"
        "勾选想要的再点应用即可加到日程里；不勾选的不会改动你的行程。"
    )
