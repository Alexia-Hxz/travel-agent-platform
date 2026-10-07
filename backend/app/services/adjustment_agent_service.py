from datetime import date as Date, time
import re
from time import monotonic as _monotonic

from sqlalchemy.orm import Session

from app.agents.tools.day_plan import (
    asks_to_replace,
    day_operations,
    day_summary,
    discovery_operations,
    discovery_summary,
    plan_discovery,
    plan_one_day,
    shortfall_notes,
)
from app.agents.tools.place_address import fill_place_addresses
from app.agents.workflows.adjuster import build_adjuster_graph
from app.models.itinerary_item import ItineraryItem
from app.models.place import Place
from app.models.trip import Trip
from app.models.itinerary_adjustment import ItineraryAdjustment
from app.schemas.adjustment import AdjustmentOperation
from app.services.adjustment_service import create_pending_adjustment, validate_operations
from app.services.itinerary_service import list_itinerary_items, list_places


# The draft mode the day button sends. Named here so the schema, the route and this service
# cannot drift apart over a string literal.
DAY_PLAN_MODE = "day_plan"

# The draft mode a question like "有没有更适合亲子的" sends: look for new places that answer it
# and offer each one as a candidate to tick, rather than editing the trip on its own.
DISCOVERY_MODE = "discover"

# "第 3 天" / "第3天下午" in the traveller's own words, so a question that names a day lands on
# that day instead of on whichever day happens to be open.
_DAY_WORD = re.compile(r"第\s*([0-9一二三四五六七八九十]{1,3})\s*天")
_DIGITS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


class DayPlanDateError(ValueError):
    """The requested day is not part of the trip, so there is nothing to arrange."""


def _place_to_state(place: Place) -> dict:
    return {
        "id": place.id,
        "name": place.name,
        "type": place.type,
        "address": place.address,
        "image_url": place.image_url,
        "price_level": place.price_level,
        "source_url": place.source_url,
        # The price the user agreed to, and how many nights/meals it covers. Without
        # these the editor could not price an item the way the card that created it did.
        "quoted_cost": place.estimated_cost,
        "quantity": place.quantity,
        "rating": place.rating,
        "note": place.note,
        "preferred_date": place.preferred_date.isoformat() if place.preferred_date else None,
    }


def _item_to_state(item: ItineraryItem) -> dict:
    place = item.place
    return {
        "id": item.id,
        "place_id": item.place_id,
        "date": item.date,
        "start_time": item.start_time,
        "end_time": item.end_time,
        "place_name": place.name,
        "image_url": place.image_url,
        "note": item.note,
        "sort_order": item.sort_order,
        "type": place.type,
        "price_level": place.price_level,
        "quoted_cost": place.estimated_cost,
        "quantity": place.quantity,
        "address": place.address,
        "source_url": place.source_url,
    }


def adjustment_to_read(adjustment: ItineraryAdjustment) -> dict:
    proposal = adjustment.proposal_json
    return {
        "id": adjustment.id,
        "trip_id": adjustment.trip_id,
        "instruction": adjustment.instruction,
        "status": adjustment.status,
        "operations": proposal.get("operations", []),
        "summary": proposal.get("summary", ""),
        "sources": proposal.get("sources", []),
        "events": proposal.get("events", []),
        "budget": proposal.get("budget", {}),
        "conflicts": proposal.get("conflicts", []),
        "daily_plan": proposal.get("daily_plan", []),
        "final_summary": proposal.get("final_summary", ""),
        "traveler_profile": proposal.get("traveler_profile", {}),
        "weather": proposal.get("weather", {}),
    }


def _clock(value) -> str:
    return value.strftime("%H:%M:%S") if isinstance(value, time) else str(value or "")


def _day_plan_preview(target_date: Date, operations: list[dict]) -> list[dict]:
    """The day as the panel shows it, in clock order.

    Only the day being replaced is reported: every other day is untouched, and listing them
    as "kept" would bury the one day the traveller actually asked about.
    """
    items = []
    for operation in operations:
        if operation.get("kind") != "add_place_and_item":
            continue
        place = operation.get("place") or {}
        items.append(
            {
                "item_id": None,
                "place_id": None,
                "start_time": _clock(operation.get("start_time")),
                "end_time": _clock(operation.get("end_time")),
                "place_name": place.get("name"),
                "note": operation.get("note"),
                "image_url": place.get("image_url"),
                "type": place.get("type"),
                "changed": "added",
            }
        )
    return [{"date": target_date.isoformat(), "items": items}]


def _create_day_plan_draft(
    session: Session,
    trip: Trip,
    instruction: str,
    target_date: Date | None,
    traveler_profile: dict,
) -> dict:
    """Arrange one day and offer it as a draft the traveller can accept.

    The day is placed on the same clock the whole-trip plan uses, and the places offered are
    ones that no other day and no saved place is already using. This is a draft, not an edit:
    nothing is written until it is applied, and applying it replaces that day alone.
    """
    trip_days = max(1, (trip.end_date - trip.start_date).days + 1)
    target = target_date or trip.start_date
    if not trip.start_date <= target <= trip.end_date:
        raise DayPlanDateError("这一天不在旅行日期范围内。")
    items = [_item_to_state(item) for item in list_itinerary_items(session, trip.id)]
    stamp = target.isoformat()
    # What is spoken for is what the *itinerary* already schedules on the other days. The saved
    # places are deliberately not part of that: the collection holds candidates the traveller
    # has not decided on, and a day is allowed to use them. Excluding the whole collection used
    # to empty a day of sights whenever it already held the city's famous ones.
    used = [
        str(item.get("place_name") or "")
        for item in items
        if str(item.get("date") or "")[:10] != stamp
    ]
    started = _monotonic()
    places = plan_one_day(trip.destination, trip_days, used, traveler_profile, trip.budget)
    # Whatever the extraction did not know the address of is looked up before the day is built,
    # so the itinerary and the exported markdown name a place the traveller can actually find.
    places = fill_place_addresses(places, trip.destination)
    shaped = day_operations(target, places, items, trip.destination)
    # Validated here rather than at the client: a malformed day must fail as a draft that
    # was never created, not as a 500 the traveller cannot act on.
    operations = [AdjustmentOperation.model_validate(operation) for operation in shaped]
    short = shortfall_notes(places)
    elapsed = round(_monotonic() - started, 1)
    events = [
        {
            "tool": "day_planner",
            "status": "completed" if places else "skipped",
            "message": (
                f"已用 {elapsed} 秒为 {stamp} 整理出 {len(places)} 个不与其他日期重复的地点。"
                if places
                else f"{stamp} 暂时没有可用的推荐素材。"
            ),
        }
    ]
    adjustment = create_pending_adjustment(
        session,
        trip,
        instruction,
        operations,
        summary=day_summary(places, short),
        sources=[],
        events=events,
        budget={},
        conflicts=[],
        daily_plan=_day_plan_preview(target, shaped),
        final_summary="",
        traveler_profile=traveler_profile,
        weather={},
    )
    return adjustment_to_read(adjustment)


def _asked_day(instruction: str, trip: Trip) -> Date | None:
    """The day the traveller named ("第 3 天"), when they named one."""
    match = _DAY_WORD.search(instruction)
    if match is None:
        return None
    token = match.group(1)
    if token.isdigit():
        offset = int(token)
    elif token == "十":
        offset = 10
    elif token.startswith("十"):
        offset = 10 + _DIGITS.get(token[1:], 0)
    elif token.endswith("十"):
        offset = _DIGITS.get(token[0], 0) * 10
    else:
        offset = _DIGITS.get(token, 0)
    if offset < 1:
        return None
    from datetime import timedelta

    day = trip.start_date + timedelta(days=offset - 1)
    return day if trip.start_date <= day <= trip.end_date else None


def _create_discovery_draft(
    session: Session,
    trip: Trip,
    instruction: str,
    target_date: Date | None,
    traveler_profile: dict,
) -> dict:
    """Look for places that answer the traveller's question and offer them as candidates.

    Every place found is a separate operation the traveller ticks or ignores, so this mode adds
    to the trip and never rearranges it on its own. It only also drops that day's items of the
    kinds on offer when the traveller's own words asked for a swap ("换成亲子的"), so a swap
    does not leave the day holding both the old sight and its replacement.
    """
    trip_days = max(1, (trip.end_date - trip.start_date).days + 1)
    target = _asked_day(instruction, trip) or target_date or trip.start_date
    if not trip.start_date <= target <= trip.end_date:
        raise DayPlanDateError("这一天不在旅行日期范围内。")
    items = [_item_to_state(item) for item in list_itinerary_items(session, trip.id)]
    # A candidate has to be new to the whole trip, the day being asked about included: a place
    # already on that day is not an answer to "find me something else".
    used = [str(place.name) for place in list_places(session, trip.id)]
    used += [str(item.get("place_name") or "") for item in items]
    started = _monotonic()
    places = plan_discovery(
        trip.destination,
        instruction,
        used,
        trip_days,
        traveler_profile,
        trip.budget,
    )
    replace = asks_to_replace(instruction)
    # The candidates go into the itinerary if the traveller ticks them, so the same rule
    # applies: no "地址待确认" reaches a saved place.
    places = fill_place_addresses(places, trip.destination)
    shaped = discovery_operations(target, places, items, trip.destination, replace=replace)
    # Validated here rather than at the client: a malformed candidate must fail as a draft that
    # was never created, not as a 500 the traveller cannot act on.
    operations = [AdjustmentOperation.model_validate(operation) for operation in shaped]
    elapsed = round(_monotonic() - started, 1)
    events = [
        {
            "tool": "place_discovery",
            "status": "completed" if places else "skipped",
            "message": (
                f"已用 {elapsed} 秒找到 {len(places)} 个符合要求且行程里还没有的地点，"
                f"建议放在 {target.isoformat()}。"
                if places
                else f"没有找到符合「{instruction.strip()[:40]}」的新地点。"
            ),
        }
    ]
    adjustment = create_pending_adjustment(
        session,
        trip,
        instruction,
        operations,
        summary=discovery_summary(places, instruction),
        sources=[],
        events=events,
        budget={},
        conflicts=[],
        daily_plan=_day_plan_preview(target, shaped),
        final_summary="",
        traveler_profile=traveler_profile,
        weather={},
    )
    return adjustment_to_read(adjustment)


def create_adjustment_draft(
    session: Session,
    trip: Trip,
    instruction: str,
    include_web: bool,
    operations: list[AdjustmentOperation] | None = None,
    candidates: list[dict] | None = None,
    structured_only: bool = False,
    mode: str | None = None,
    target_date: Date | None = None,
) -> dict:
    traveler_profile = {
        "adult_male_count": trip.adult_male_count,
        "adult_female_count": trip.adult_female_count,
        "child_male_count": trip.child_male_count,
        "child_female_count": trip.child_female_count,
    }
    traveler_profile["total_people"] = sum(traveler_profile.values())
    if mode == DAY_PLAN_MODE:
        return _create_day_plan_draft(session, trip, instruction, target_date, traveler_profile)
    if mode == DISCOVERY_MODE:
        return _create_discovery_draft(session, trip, instruction, target_date, traveler_profile)
    if operations:
        # Operations the user built by clicking must be checkable right away: the panel
        # shows a preview and a budget from them, so an impossible one has to be refused
        # here rather than surfacing as a failure when they press apply. Model-proposed
        # operations are not checked here -- the user picks between those individually.
        validate_operations(session, trip, operations)
    state = {
        "trip": {
            "id": trip.id,
            "destination": trip.destination,
            "start_date": trip.start_date.isoformat(),
            "end_date": trip.end_date.isoformat(),
            "budget": trip.budget,
        },
        "places": [_place_to_state(place) for place in list_places(session, trip.id)],
        "itinerary": [_item_to_state(item) for item in list_itinerary_items(session, trip.id)],
        "instruction": instruction,
        "include_web": include_web,
        "events": [],
        "traveler_profile": traveler_profile,
        # Operations built by the editor itself, so the model is not asked to guess at a
        # change the user already spelled out by clicking.
        "requested_operations": operations or [],
        "structured_only": structured_only,
        "candidates": [candidate for candidate in (candidates or []) if str(candidate.get("name", "")).strip()],
    }
    result = build_adjuster_graph().invoke(state)
    adjustment = create_pending_adjustment(
        session,
        trip,
        instruction,
        result.get("operations", []),
        summary=result.get("summary", ""),
        sources=result.get("sources", []),
        events=result.get("events", []),
        budget=result.get("budget", {}),
        conflicts=result.get("conflicts", []),
        daily_plan=result.get("daily_plan", []),
        final_summary=result.get("final_summary", ""),
        traveler_profile=result.get("traveler_profile", traveler_profile),
        weather=result.get("weather", {}),
    )
    return adjustment_to_read(adjustment)
