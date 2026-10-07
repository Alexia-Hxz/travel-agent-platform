from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BudgetEstimate:
    total_estimate: int
    trip_budget: int | None
    remaining_budget: int | None
    currency: str = "CNY"
    note: str = "根据收藏地点价格等级估算，实际费用以商家报价为准。"
    breakdown: list[dict[str, Any]] | None = None
    daily_estimate: int = 0
    itemized_summary: str = ""
    # Typical accommodation/meal money that is deliberately NOT part of total_estimate,
    # shown so a reader can see what those would cost without the total pretending the trip
    # is already paid for. See the note in estimate_budget.
    excluded_allowances: list[dict[str, Any]] | None = None


def format_itemized_summary(breakdown: list[dict[str, Any]] | None, total: int, currency: str = "CNY") -> str:
    """Render an easy-to-scan addition expression for the final plan."""
    currency_label = "元" if currency.upper() == "CNY" else currency
    lines = breakdown or [{"name": "基础支出", "amount": total}]
    parts: list[str] = []
    for line in lines:
        name = str(line.get("name") or "项目")
        amount = int(line.get("amount") or 0)
        value = "免费" if amount <= 0 else f"{amount}{currency_label}"
        parts.append(f"{name}（{value}）")
    return f"{' + '.join(parts)} = {total}{currency_label}"


def _quoted_cost(place: dict[str, Any]) -> int | None:
    """The price the user was shown for this place, if it has one.

    Saved places carry it as ``quoted_cost`` (from ``Place.estimated_cost``); a place
    straight out of the recommendations carries it as ``estimated_cost``. Either way it is
    the number the user agreed to, so it beats any figure derived from a price level.
    """
    for key in ("quoted_cost", "estimated_cost"):
        raw = place.get(key)
        if raw is None:
            continue
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            continue
    return None


# How much of a category one unit costs, before the count is applied. Kept next to
# `estimate_budget` so the two models can never drift apart.
_LEVEL_UNIT_PRICE = {"hotel": 260, "restaurant": 80, "airport": 30, "station": 30}
_DEFAULT_LEVELS = {"hotel": 3, "restaurant": 2, "airport": 1, "station": 1, "attraction": 1}
_DAILY_TRANSPORT = 60
# Reference amounts, deliberately excluded from the total (see estimate_budget).
_REFERENCE_HOTEL_PER_NIGHT = 300
_REFERENCE_MEAL_PER_PERSON_PER_DAY = 80


def _level_of(place: dict[str, Any]) -> tuple[int, bool]:
    """Price level plus whether it had to fall back to the category default."""
    raw_level = place.get("price_level")
    if raw_level is None:
        return _DEFAULT_LEVELS.get(str(place.get("type", "")), 2), True
    try:
        return max(0, min(5, int(raw_level))), False
    except (TypeError, ValueError):
        return _DEFAULT_LEVELS.get(str(place.get("type", "")), 2), True


def _quantity_of(place: dict[str, Any]) -> int:
    """How many times this place is used. Defaults to one, as the cards do."""
    raw = place.get("quantity")
    if raw is None:
        return 1
    try:
        return max(1, min(99, int(raw)))
    except (TypeError, ValueError):
        return 1


def _level_amount(place_type: str, level: int, days: int, people: int) -> int:
    if place_type == "hotel":
        return level * _LEVEL_UNIT_PRICE["hotel"] * max(1, days - 1)
    if place_type == "restaurant":
        return level * _LEVEL_UNIT_PRICE["restaurant"] * days * people
    if place_type in {"airport", "station"}:
        return level * _LEVEL_UNIT_PRICE["airport"]
    return level * 100


def _level_unit_price(place_type: str, level: int, people: int) -> int:
    """What one night / one meal / one ticket costs before the count is applied."""
    if place_type == "hotel":
        return level * _LEVEL_UNIT_PRICE["hotel"]
    if place_type == "restaurant":
        return level * _LEVEL_UNIT_PRICE["restaurant"] * people
    if place_type in {"airport", "station"}:
        return level * _LEVEL_UNIT_PRICE["airport"]
    return level * 100


def _count_unit(place_type: str) -> str | None:
    if place_type == "hotel":
        return "晚"
    if place_type == "restaurant":
        return "次"
    return None


def estimate_itinerary_budget(
    itinerary: list[dict[str, Any]],
    trip_days: int,
    trip_budget: int | None,
    traveler_profile: dict[str, Any] | None = None,
) -> BudgetEstimate:
    """Estimate the trip from the items actually scheduled, one row per item.

    This is the model the daily-itinerary editor uses, and it answers a different
    question than ``estimate_budget`` does. ``estimate_budget`` prices the *saved
    places* collection; this one prices *the itinerary itself*, so:

    - deleting a scheduled item removes its money from the total, automatically,
      because it is simply no longer in the list being summed;
    - adding an item adds its money, at whatever price the user set on it;
    - a place that carries its own quoted price is charged exactly that price (the
      same rule the recommendation cards and saved places already follow);
    - anything else falls back to the same price-level formula ``estimate_budget``
      uses, multiplied by how many times the item is used (nights / meals).

    Local transport is charged as its own visible line, because a trip needs it
    whether or not any particular place was saved.
    """
    days = max(1, trip_days)
    profile = traveler_profile or {}
    total_people = max(1, int(profile.get("total_people", 0) or 1))
    transport = _DAILY_TRANSPORT * days
    total = transport
    breakdown: list[dict[str, Any]] = [
        {"category": "transport", "name": "市内交通基础估算", "amount": transport, "source": "baseline"}
    ]
    used_defaults = False
    categories_seen: set[str] = set()
    for item in itinerary:
        place_type = str(item.get("type") or item.get("category") or "attraction")
        categories_seen.add(place_type)
        name = str(item.get("place_name") or item.get("name") or "行程项")
        unit = _count_unit(place_type)
        quoted = _quoted_cost(item)
        quantity = _quantity_of(item) if unit else 1
        if quoted is not None:
            # The price the user was shown already covers the count they chose, so the
            # total is charged verbatim and the unit price is only shown for reference.
            amount = quoted
            unit_price = round(quoted / quantity) if quantity > 1 else quoted
            source = "quoted"
        else:
            level, used_default = _level_of(item)
            used_defaults = used_defaults or used_default
            unit_price = _level_unit_price(place_type, level, total_people)
            amount = unit_price * quantity
            source = "level"
        total += amount
        breakdown.append(
            {
                "category": place_type,
                "name": name,
                "amount": amount,
                "unit_price": unit_price,
                "quantity": quantity,
                "quantity_unit": unit,
                "people": total_people if place_type == "restaurant" else None,
                "price_level": item.get("price_level"),
                "source": source,
                "date": str(item.get("date") or "")[:10] or None,
                "start_time": str(item.get("start_time") or "")[:5] or None,
                "end_time": str(item.get("end_time") or "")[:5] or None,
                "item_id": item.get("id"),
                "place_id": item.get("place_id"),
                "operation_index": item.get("operation_index"),
                "note": item.get("note"),
            }
        )
    excluded: list[dict[str, Any]] = []
    if "hotel" not in categories_seen:
        excluded.append(
            {"category": "hotel", "name": "住宿参考（未计入总额）", "amount": _REFERENCE_HOTEL_PER_NIGHT * max(1, days - 1)}
        )
    if "restaurant" not in categories_seen:
        excluded.append(
            {
                "category": "restaurant",
                "name": "餐饮参考（未计入总额）",
                "amount": _REFERENCE_MEAL_PER_PERSON_PER_DAY * days * total_people,
            }
        )
    note = "按“每日行程”里实际安排的项目逐项估算，删掉的项目不再计费；含市内交通基础估算，实际费用以商家报价为准。"
    if used_defaults:
        note += " 没有价格的项目已按类型使用中档默认值。"
    if excluded:
        note += " 行程里还没有安排的类别，其参考金额列在下方且未计入总额。"
    return BudgetEstimate(
        total_estimate=total,
        trip_budget=trip_budget,
        remaining_budget=trip_budget - total if trip_budget is not None else None,
        note=note,
        breakdown=breakdown,
        daily_estimate=round(total / days),
        itemized_summary=format_itemized_summary(breakdown, total),
        excluded_allowances=excluded,
    )


def estimate_budget(
    places: list[dict[str, Any]], trip_days: int, trip_budget: int | None, traveler_profile: dict[str, Any] | None = None
) -> BudgetEstimate:
    """Estimate a transparent budget from saved place price levels.

    A place that carries its own quoted price is charged exactly that price (which already
    includes how many nights or meals the user asked for). Everything else is estimated
    conservatively and deterministically: accommodation per night, food per trip day, and
    other places once per saved place.
    """
    days = max(1, trip_days)
    profile = traveler_profile or {}
    total_people = int(profile.get("total_people", 0))
    total = 0
    breakdown: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    used_defaults = False
    for place in places:
        place_type = str(place.get("type", ""))
        name = str(place.get("name", "地点"))
        # A place the user picked from the recommendations already carries the price they
        # were shown -- including how many nights or meals they asked for. Charging it at
        # that price is what makes the count they set actually drive the total; inventing a
        # level-based figure instead would quietly disagree with the card they ticked.
        quoted = _quoted_cost(place)
        if quoted is not None:
            total += quoted
            breakdown.append({"category": place_type, "name": name, "amount": quoted, "source": "quoted"})
            continue
        level, used_default = _level_of(place)
        used_defaults = used_defaults or used_default
        amount = _level_amount(place_type, level, days, max(1, total_people))
        total += amount
        breakdown.append({"category": place_type, "name": name, "amount": amount, "source": "level"})
    if not places:
        # A plan that only has recommendations has nothing booked yet, so the total charges
        # local transport and nothing else.
        #
        # It used to also charge a guessed room (300/night) and three guessed meals (80 per
        # person per day). That double-counted the two biggest categories: the moment the user
        # ticked a hotel or a restaurant, its real price was added ON TOP of the guess the
        # ticket was meant to replace -- the plan's baseline already claimed 1200 CNY of
        # accommodation and meals that no one had chosen. Accommodation and meals now enter
        # the total exclusively through an actual place (saved or ticked), and the typical
        # amounts are still reported as reference figures that are explicitly excluded.
        people = max(1, total_people)
        nights = max(1, days - 1)
        transport = _DAILY_TRANSPORT * days
        total = transport
        used_defaults = True
        breakdown = [
            {"category": "transport", "name": "市内交通基础估算", "amount": transport, "source": "baseline"},
        ]
        excluded = [
            {"category": "hotel", "name": "住宿参考（未计入总额）", "amount": _REFERENCE_HOTEL_PER_NIGHT * nights},
            {"category": "restaurant", "name": "餐饮参考（未计入总额）", "amount": _REFERENCE_MEAL_PER_PERSON_PER_DAY * days * people},
        ]
    remaining = trip_budget - total if trip_budget is not None else None
    note = "根据地点价格等级估算，实际费用以商家报价为准。"
    if used_defaults:
        note += " 未填写价格等级的地点已按类型使用中档默认值。"
    if excluded:
        note += " 住宿与餐饮不预设基础估算，只按你已勾选或已收藏的具体项目计入总额；下方参考金额未计入。"
    return BudgetEstimate(
        total_estimate=total,
        trip_budget=trip_budget,
        remaining_budget=remaining,
        note=note,
        breakdown=breakdown,
        daily_estimate=round(total / days),
        itemized_summary=format_itemized_summary(breakdown, total),
        excluded_allowances=excluded,
    )
