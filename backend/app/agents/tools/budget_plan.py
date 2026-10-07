"""The money framework the planning panel shows *before* anything is chosen.

That panel used to answer the wrong question. It showed a total assembled from saved places plus
whatever the traveller had ticked, and listed accommodation and meals as fixed "reference" amounts
(300 a night, 80 a person a day). Those references ignored the traveller's own budget completely: a
30,000 CNY trip to Tokyo and an 8,000 CNY domestic weekend were told the same room costs 300. What
the front of the plan should answer instead is "of the money I said I would spend, how much goes
where", which is what this module builds.

It is deliberately kept apart from the two honest sums:

* ``estimate_budget`` — what the *saved and ticked* places cost. Still shown, still exact.
* ``estimate_itinerary_budget`` — what the *scheduled* rows in the daily editor cost. That is the
  panel's authoritative number once the traveller has arranged a trip.

Both of those are sums of real prices and must never be diluted with a guess. This one is a guess,
it is derived from the trip's own budget rather than from a table of national averages, and every
surface that renders it is expected to say so.
"""

from __future__ import annotations

import json
from typing import Any

from app.agents.tools.llm import MODEL_NAME, model_client
from app.agents.tools.model_json import parse_json_object

# The share of the traveller's budget this framework plans to spend. The rest stays deliberately
# unallocated: real trips include shopping, a taxi taken in the rain, a cancelled booking, and a
# framework that claims every last yuan is a promise nobody can keep.
PLANNED_BUDGET_RATIO = 0.8

# Where the planned money goes when the model cannot answer. The order doubles as the display
# order. A bed is the single biggest line of a mid-range trip, then three meals a day, then
# tickets and local transport, then the money that has no category.
DEFAULT_ALLOCATION_SHARES: tuple[tuple[str, str, float], ...] = (
    ("hotel", "酒店 / 住宿", 0.45),
    ("restaurant", "美食 / 餐饮", 0.25),
    ("attraction", "景点门票", 0.15),
    ("transport", "市内交通", 0.10),
    ("other", "其他 / 购物机动", 0.05),
)

# A trip with no night to pay for puts the room's share into the day itself. Without this the
# fallback would charge 720 CNY of accommodation for a same-day trip while the row's own note says
# "当日往返，可不安排住宿".
DAY_TRIP_ALLOCATION_SHARES: tuple[tuple[str, str, float], ...] = (
    ("hotel", "酒店 / 住宿", 0.0),
    ("restaurant", "美食 / 餐饮", 0.40),
    ("attraction", "景点门票", 0.25),
    ("transport", "市内交通", 0.20),
    ("other", "其他 / 购物机动", 0.15),
)

# The categories handed to the model, in the same order, so its answer lines up with the display.
ALLOCATION_CATEGORIES = tuple((key, label) for key, label, _ in DEFAULT_ALLOCATION_SHARES)

# What a sane split looks like. An answer inside these bands is used as given; one outside them is
# discarded in favour of the defaults rather than printing "shopping 31%" on a city break. The
# bands are wide on purpose — they are a sanity check, not a second opinion on the destination.
ALLOCATION_RANGES: dict[str, tuple[float, float]] = {
    "hotel": (0.05, 0.60),
    "restaurant": (0.10, 0.40),
    "attraction": (0.02, 0.35),
    "transport": (0.01, 0.25),
    # The one that actually catches bad answers: money with no category must stay a rounding
    # error, because everything in this framework is supposed to be a plan.
    "other": (0.0, 0.15),
}
# A usable answer has to cover every category the framework displays, otherwise the missing ones
# would be silently re-weighted, which is how a table stops matching what the model said. A zero is
# a real answer ("no hotel on a day trip") and is accepted as long as it is inside its band.
REQUIRED_MODEL_CATEGORIES = tuple(key for key, _label in ALLOCATION_CATEGORIES)

BUDGET_PLAN_SYSTEM_PROMPT = (
    "你是旅行预算规划师。用户会给出目的地、天数、人数、总预算和一个待分配金额，"
    "请把待分配金额按类别拆开。只输出 JSON 对象，不要任何解释、不要 Markdown 代码块。"
    "每个类别都必须给出正整数金额，所有类别之和必须恰好等于待分配金额。"
    "请结合该目的地的真实行情分配（例如日本住宿贵、市内交通便宜；东南亚住宿便宜）。"
    "经验区间：住宿 30%-55%、餐饮 15%-35%、景点门票 5%-30%、市内交通 3%-20%、"
    "其他（购物机动）不要超过 15%，当日往返可以不给住宿。"
)

UNALLOCATED_NOTE = (
    "这个行程还没有填写总预算，所以暂时无法按预算给出分配建议。"
    "在项目里填上总预算后，这里会按你的预算给出酒店/美食/景点等各类别的大概金额。"
)


def _positive_int(value: Any) -> int | None:
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _people(traveler_profile: dict[str, Any] | None) -> int:
    try:
        return max(1, int((traveler_profile or {}).get("total_people") or 1))
    except (TypeError, ValueError):
        return 1


def _nights(days: int) -> int:
    """A day trip has no night to pay for, so this is allowed to be zero."""
    return max(0, days - 1)


def _budget_amount(trip_budget: Any) -> int | None:
    """The trip's own budget as a positive whole number of yuan, or None when unusable."""
    return _positive_int(trip_budget)


def planned_total(trip_budget: Any) -> int | None:
    """The amount this framework plans to spend: 80% of the trip's own budget."""
    budget = _budget_amount(trip_budget)
    if budget is None:
        return None
    planned = int(round(budget * PLANNED_BUDGET_RATIO))
    return planned if planned > 0 else None


def _shares_from_model(raw: Any) -> dict[str, float] | None:
    """Read a model answer as weights, whether it answered in amounts or in percentages.

    Zero is kept (it is a real answer, e.g. no room on a day trip); negatives and non-numbers are
    dropped. Missing categories stay missing so ``_model_answer_is_sane`` can reject the answer
    rather than silently re-weighting the ones that are there.
    """
    if not isinstance(raw, dict):
        return None
    weights: dict[str, float] = {}
    for key, _label in ALLOCATION_CATEGORIES:
        value = raw.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number >= 0:
            weights[key] = number
    return weights or None


def _model_answer_is_sane(shares: dict[str, float] | None, nights: int = 1) -> bool:
    """Whether a model split is worth showing, checked on the shares it would produce."""
    if not shares:
        return False
    if any(key not in shares for key in REQUIRED_MODEL_CATEGORIES):
        return False
    weight_sum = sum(shares.values()) or 0.0
    if weight_sum <= 0:
        return False
    for key, (low, high) in ALLOCATION_RANGES.items():
        # A day trip has no bed to pay for, so "nothing on accommodation" is the right answer there
        # and a wrong one on a five-night trip.
        if key == "hotel" and nights <= 0:
            low = 0.0
        share = shares.get(key, 0.0) / weight_sum
        if share < low - 1e-9 or share > high + 1e-9:
            return False
    return True


def ask_the_model_for_shares(
    destination: str,
    days: int,
    people: int,
    trip_budget: int,
    amount_to_allocate: int,
) -> dict[str, float] | None:
    """One small model call: how this destination should split the money. Never raises."""
    client = model_client()
    if client is None:
        return None
    payload = {
        "destination": destination,
        "days": days,
        "nights": _nights(days),
        "people": people,
        "trip_budget": trip_budget,
        "amount_to_allocate": amount_to_allocate,
        "currency": "CNY",
        "categories": [key for key, _label in ALLOCATION_CATEGORIES],
        "output_example": {"hotel": 10800, "restaurant": 6000, "attraction": 3600, "transport": 2400, "other": 1200},
    }
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": BUDGET_PLAN_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        content = response.choices[0].message.content or ""
    except Exception:
        # An estimate is a convenience: a model outage degrades it to the default shares.
        return None
    return _shares_from_model(parse_json_object(content))


def _weights(shares: dict[str, float] | None, nights: int = 1) -> dict[str, float]:
    defaults = DEFAULT_ALLOCATION_SHARES if nights > 0 else DAY_TRIP_ALLOCATION_SHARES
    weights = {key: float(share) for key, _label, share in defaults}
    if shares:
        for key, value in shares.items():
            # A zero from the model is honoured, so it must not fall back to the default share.
            if key in weights and value >= 0:
                weights[key] = float(value)
    return weights


def _split(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Split ``total`` by weight so the parts add up to exactly ``total``.

    Rounding each share on its own loses or gains a few yuan; the largest remainder method hands
    those yuan to the categories that were cut hardest, which keeps the displayed rows and the
    displayed total consistent (a table that does not add up is worse than a rounded one).
    """
    weight_sum = sum(weights.values()) or 1.0
    exact = {key: total * value / weight_sum for key, value in weights.items()}
    amounts = {key: int(value) for key, value in exact.items()}
    leftover = total - sum(amounts.values())
    if leftover > 0:
        order = sorted(exact, key=lambda key: (exact[key] - amounts[key]), reverse=True)
        for index in range(leftover):
            amounts[order[index % len(order)]] += 1
    return amounts


def _unit_note(category: str, amount: int, days: int, nights: int, people: int) -> str:
    """A per-night / per-person reading of the row, so the number can be sanity-checked."""
    if amount <= 0:
        return ""
    if category == "hotel":
        if nights <= 0:
            return "当日往返，可不安排住宿"
        return f"约 {round(amount / nights)} CNY/晚 × {nights} 晚"
    if category == "restaurant":
        return f"约 {round(amount / (days * people))} CNY/人/天 × {people} 人 × {days} 天"
    if category == "attraction":
        return f"约 {round(amount / people)} CNY/人 × {people} 人（全程门票）"
    if category == "transport":
        return f"约 {round(amount / days)} CNY/天 × {days} 天"
    return "含购物、临时打车与机动支出"


def remaining_ratio() -> int:
    """The unallocated share, as a percentage, for the note."""
    return int(round((1 - PLANNED_BUDGET_RATIO) * 100))


def build_budget_plan(
    trip_budget: Any,
    trip_days: int,
    destination: str,
    traveler_profile: dict[str, Any] | None = None,
    model_shares: dict[str, float] | None = None,
) -> dict[str, Any]:
    """The category framework the planning panel shows, derived from the trip's own budget.

    ``model_shares`` lets a caller supply the split it already obtained (tests, or a caller that
    asked the model itself); when it is None the model is asked here, and when that fails — for any
    reason, including a provider outage — the default shares are used. The rows always add up to
    exactly the planned total.
    """
    days = max(1, int(trip_days or 1))
    nights = _nights(days)
    people = _people(traveler_profile)
    budget = _budget_amount(trip_budget)
    total = planned_total(trip_budget)
    if total is None or budget is None:
        return {
            "status": "unavailable",
            "currency": "CNY",
            "trip_budget": None,
            "planned_total": 0,
            "planned_ratio": PLANNED_BUDGET_RATIO,
            "planned_ratio_percent": int(round(PLANNED_BUDGET_RATIO * 100)),
            "reserve_amount": 0,
            "source": "none",
            "days": days,
            "nights": nights,
            "people": people,
            "categories": [],
            "note": UNALLOCATED_NOTE,
            "summary": "",
        }

    shares = model_shares
    if shares is None:
        try:
            shares = ask_the_model_for_shares(destination, days, people, budget, total)
        except Exception:
            # A convenience estimate must never be able to fail a planning request.
            shares = None
    # A model answer outside the sanity bands is dropped, not massaged into something plausible.
    source = "model" if _model_answer_is_sane(shares, nights) else "default"
    if source == "default":
        shares = None
    amounts = _split(total, _weights(shares, nights))

    categories = []
    for key, label in ALLOCATION_CATEGORIES:
        amount = amounts.get(key, 0)
        categories.append(
            {
                "category": key,
                "label": label,
                "amount": amount,
                "share": round(amount / total, 4) if total else 0.0,
                "share_percent": int(round(amount / total * 100)) if total else 0,
                "unit": _unit_note(key, amount, days, nights, people),
            }
        )

    reserve = max(0, budget - total)
    note = (
        f"按你填写的总预算 {budget} CNY 估算：计划花掉约 {total} CNY（{int(round(PLANNED_BUDGET_RATIO * 100))}%），"
        f"预留约 {reserve} CNY（{remaining_ratio()}%）给购物与临时支出。"
        "这是还没有挑选具体项目时的花费框架，只用来判断钱大概花在哪一类；"
        "挑好项目之后的真实估算在“编辑每日行程”面板里。"
    )
    return {
        "status": "estimated",
        "currency": "CNY",
        "trip_budget": budget,
        "planned_total": total,
        "planned_ratio": PLANNED_BUDGET_RATIO,
        "planned_ratio_percent": int(round(PLANNED_BUDGET_RATIO * 100)),
        "reserve_amount": reserve,
        "source": source,
        "days": days,
        "nights": nights,
        "people": people,
        "categories": categories,
        "note": note,
        "summary": format_budget_plan_summary(
            {"status": "estimated", "planned_total": total, "currency": "CNY", "categories": categories}
        ),
    }


def format_budget_plan_summary(plan: dict[str, Any] | None) -> str:
    """One line for the summary prompt and the tool event log."""
    if not plan or plan.get("status") != "estimated":
        return ""
    currency = plan.get("currency") or "CNY"
    parts = [
        f"{row['label']}约 {row['amount']} {currency}"
        for row in plan.get("categories") or []
        if row.get("amount")
    ]
    if not parts:
        return ""
    return f"{'，'.join(parts)}；合计约 {plan.get('planned_total')} {currency}（按总预算估算）"
