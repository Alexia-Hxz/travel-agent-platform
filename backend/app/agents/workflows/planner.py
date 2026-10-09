from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from functools import partial
import logging
import re
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.state import PlannerState
from app.agents.tools.budget import estimate_budget
from app.agents.tools.budget_plan import build_budget_plan, format_budget_plan_summary
from app.agents.tools.conflicts import detect_conflicts
from app.agents.tools.place_facts import enrich_place_facts
from app.agents.tools.place_images import resolve_place_images
from app.agents.tools.plan_summary import generate_final_summary
from app.agents.tools.recommendations import (
    CATEGORY_ORDER,
    budget_allowances,
    category_limits,
    enrich_prices_with_search,
    ensure_category_coverage,
    ensure_price_spread,
    estimate_trip_costs,
    extract_recommendations,
)
from app.agents.tools.web_search import search_web
from app.agents.tools.weather_advice import build_weather_advice, weather_adjustment_request
from app.services.weather_service import get_weather_forecast


logger = logging.getLogger(__name__)

ROUTE_KEYWORDS = ("怎么到", "如何到", "怎样到", "到", "去", "交通", "路线")

# The picture pass is pure network work — a photo search per place, then a reachability
# probe per candidate — and it used to run at the end of the recommendation node with
# everything else already finished. Measured on a four-day plan, a full set of photos costs
# 15-25 s, and nothing that follows it needs a picture: the conflict check, the budget and
# the written summary all work from names and prices. Running it on its own thread therefore
# hides it behind that work instead of adding to it, and the photos themselves are
# unchanged — the same search, the same probing, the same fallbacks.
#
# This is the wait in `compose_plan` if the pass is still running. It is deliberately well
# above the pass's own 60 s budget per call, so a slow provider can never leave a plan worse
# off than before this change: at worst it lands back on the old timing, and past this the
# plan is served with the bundled placeholders rather than waiting any longer.
IMAGE_JOIN_TIMEOUT = 180.0


def _trip_date_list(trip: dict[str, Any]) -> list[str]:
    """Every date of the trip, in order. Empty when the dates are unusable."""
    try:
        start = date.fromisoformat(str(trip["start_date"]))
        end = date.fromisoformat(str(trip["end_date"]))
    except (KeyError, ValueError):
        return []
    if end < start:
        return []
    return [(start + timedelta(days=offset)).isoformat() for offset in range((end - start).days + 1)]


def _same_venue(first: str, second: str) -> bool:
    """Whether two names are the same venue written at different lengths.

    ``蒋有记`` and ``老门东内蒋有记`` are one restaurant, so offering both on the same day
    serves the same place twice. A purely numeric difference is excluded: ``景点1`` and
    ``景点10`` are two different candidates, not one venue under a longer name.
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


def _generated_daily_plan(state: PlannerState, recommended: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Suggest an arrangement for EVERY day of the trip.

    This used to emit a single day — the trip's first date — however long the trip
    was, always built from the same first hotel, first two attractions and first
    restaurant. A 5-day request therefore got one day of suggestions, and the hotel
    it named was billed for all four nights, which read as though a single day cost
    the whole stay.

    Days are now walked in order and every place is used at most once across the whole
    trip, so no day repeats another day's sights or meals. Rotating the lists was not
    enough: with four attractions a five-day trip still visited the first one twice,
    and the same lunch turned up every day.
    """
    query = state.get("query", "")
    if not recommended or not any(word in query for word in ("推荐", "规划", "第一天", "酒店", "景点", "美食")):
        return []
    dates = _trip_date_list(state["trip"])
    if not dates:
        return []

    hotels = [place for place in recommended if place.get("type") == "hotel"]
    attractions = [place for place in recommended if place.get("type") == "attraction"]
    restaurants = [place for place in recommended if place.get("type") == "restaurant"]
    # Breakfast venues come from the places labelled as such; when the model labelled
    # none, the general options stand in rather than leaving every morning empty.
    breakfasts = [place for place in restaurants if place.get("meal_slot") == "breakfast"]
    if not breakfasts:
        breakfasts = [place for place in restaurants if place.get("meal_slot") == "any"]
    # Lunch and dinner are drawn from the venues that are not breakfast-only.
    mains = [place for place in restaurants if place.get("meal_slot") != "breakfast"] or restaurants
    nights = max(0, len(dates) - 1)

    # One deal of the whole pool: a place handed to one slot is never offered again.
    used: list[tuple[str, str]] = []

    def take(category: str, pool: list[dict[str, Any]]) -> dict[str, Any] | None:
        for candidate in pool:
            key = str(candidate.get("name") or "").strip()
            if not key:
                continue
            # Two places to visit or eat at whose names contain one another are one place
            # under two names ("蒋有记" and "老门东内蒋有记"), so the second is skipped.
            # Hotels are left out of this: they are named after the district they sit in,
            # so "夫子庙步行街轻居酒店" is no evidence that the sight "夫子庙" is used.
            if any(
                other_category != "hotel" and _same_venue(key, other_name)
                for other_category, other_name in used
            ):
                continue
            if any(other_name.casefold() == key.casefold() for _, other_name in used):
                continue
            used.append((category, key))
            return candidate
        return None

    plan: list[dict[str, Any]] = []
    for index, day in enumerate(dates):
        slots: list[tuple[str, str, dict[str, Any], str]] = []
        missing: list[str] = []

        if index == 0 and hotels:
            hotel = hotels[0]
            used.append(("hotel", str(hotel.get("name") or "").strip()))
            cost = hotel.get("estimated_cost")
            if nights and cost is not None:
                # Say plainly that the figure covers the stay, so one day does not
                # appear to cost four nights' room rate.
                note = f"办理入住；住宿费约 {cost} 元是全程 {nights} 晚的合计，不是当天消费"
            else:
                note = "办理入住或先寄存行李"
            slots.append(("13:00:00", "14:00:00", hotel, note))

        breakfast = take("restaurant", breakfasts)
        if breakfast is not None:
            slots.append(("08:00:00", "09:00:00", breakfast, "当地早餐"))
        else:
            missing.append("早餐")

        morning = take("attraction", attractions)
        if morning is not None:
            slots.append(("09:30:00", "12:00:00", morning, "上午主要游览，预留交通与拍照时间"))
        else:
            missing.append("景点")

        lunch = take("restaurant", mains)
        if lunch is not None:
            slots.append(("12:00:00", "13:00:00", lunch, "当地特色午餐"))
        else:
            missing.append("午餐")

        afternoon = take("attraction", attractions)
        if afternoon is not None:
            slots.append(("14:00:00", "17:00:00", afternoon, "下午继续游览，按现场人流灵活调整"))

        dinner = take("restaurant", mains)
        if dinner is not None:
            slots.append(("18:00:00", "19:30:00", dinner, "晚餐"))
        else:
            missing.append("晚餐")

        # A day is always reported, even when nothing could be allocated to it: dropping
        # it would make the plan quietly shorter than the trip and hide the gap.
        slots.sort(key=lambda slot: slot[0])
        plan.append(
            {
                "date": day,
                "items": [
                    {
                        "place_name": place["name"],
                        "start_time": start,
                        "end_time": end,
                        "note": note,
                        "image_url": place.get("image_url"),
                        "fallback_image_url": place.get("fallback_image_url"),
                    }
                    for start, end, place, note in slots
                ],
                # Omitted slots are stated rather than filled with a repeat, so a short
                # day is explained instead of looking like the plan forgot a meal.
                "shortage": (
                    f"推荐项目不够分配，本日缺少{'、'.join(missing)}；请再多生成一次规划建议补充候选。"
                    if missing
                    else None
                ),
            }
        )
    return plan


def _budget_summary(budget: Any) -> str:
    labels = {"hotel": "住宿", "restaurant": "餐饮", "attraction": "景点", "airport": "交通", "station": "交通", "transport": "交通"}
    totals: dict[str, int] = {}
    for line in budget.breakdown or []:
        category = labels.get(str(line.get("category")), "其他")
        totals[category] = totals.get(category, 0) + int(line.get("amount", 0))
    return "；".join(f"{category}约 {amount} {budget.currency}" for category, amount in totals.items()) or f"基础支出约 {budget.total_estimate} {budget.currency}"


def _route_target(query: str) -> str | None:
    match = re.search(r"(?:怎么|如何|怎样)?(?:到|去)\s*([^，。！？?；;]+)", query)
    if match is None:
        return None
    target = re.split(r"(?:性价比|最划算|最便宜|最快|路线|交通|怎么走|如何走)", match.group(1), maxsplit=1)[0]
    return target.strip(" ：:，。！？?；;") or None


def _route_context(state: PlannerState) -> dict[str, str] | None:
    query = state.get("query", "")
    if not any(keyword in query for keyword in ROUTE_KEYWORDS):
        return None
    hotel = next((place for place in state.get("places", []) if place.get("type") == "hotel"), None)
    destination = _route_target(query)
    if hotel is None or destination is None:
        return None
    return {"hotel_name": hotel["name"], "hotel_address": hotel["address"], "destination": destination}


def _route_search_query(context: dict[str, str]) -> str:
    return (
        f"从{context['hotel_name']}（{context['hotel_address']}）到{context['destination']} "
        "性价比最高 公共交通 地铁 公交 时间 票价"
    )


def _load_context(state: PlannerState) -> PlannerState:
    events = list(state.get("events", []))
    documents = state.get("traveler_documents") or []
    message = "已加载旅行项目、收藏地点和每日行程。"
    if documents:
        message += f"并载入 {len(documents)} 段来自旅客上传资料的依据。"
    events.append({"tool": "context_loader", "status": "completed", "message": message})
    return {**state, "events": events}


def _search_live_web(state: PlannerState) -> PlannerState:
    """Gather web material through the model's own live search.

    Named for what it does rather than for the provider it used to call: this node was
    "tavily_search" because a metered search API once backed it, and that API has been
    removed (HTTP 432 for every query). The event name is what the interface shows, so
    leaving the old name would have told the traveller their plan used a service it never
    reached.
    """
    events = list(state.get("events", []))
    if not state.get("include_web", False):
        events.append({"tool": "live_search", "status": "skipped", "message": "用户未启用联网搜索。"})
        return {**state, "web_results": [], "events": events}
    trip = state["trip"]
    route_context = _route_context(state)
    search_query = _route_search_query(route_context) if route_context else f"{state.get('query', '')} 酒店 景点 美食 图片"
    results, status = search_web(search_query, trip["destination"])
    events.append({"tool": "live_search", "status": "completed" if status == "completed" else "fallback", "message": status})
    return {**state, "web_results": results, "route_context": route_context, "events": events}


def _planned_place_entries(state: PlannerState) -> list[dict[str, Any]]:
    """Saved itinerary items, which need pictures of their own."""
    types = {
        str(place.get("name") or ""): str(place.get("type") or "attraction")
        for place in state.get("places", [])
    }
    entries: list[dict[str, Any]] = []
    for item in state.get("itinerary", []):
        name = str(item.get("place_name") or "").strip()
        if not name:
            continue
        entries.append({"name": name, "type": types.get(name, "attraction"), "image_url": item.get("image_url")})
    return entries


def _trip_days(trip: dict[str, Any]) -> int:
    try:
        start = date.fromisoformat(str(trip["start_date"]))
        end = date.fromisoformat(str(trip["end_date"]))
    except (KeyError, ValueError):
        return 1
    return max(1, (end - start).days + 1)


def _picture_pass(destination: str, recommended: list[dict[str, Any]], planned: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    """Find and verify a picture for every place. Safe to run on its own thread.

    Works on the copies it is given, so nothing it does is visible until the result is
    folded back into the state in `compose_plan`.
    """
    pictured = resolve_place_images(recommended, destination, skip_categories=frozenset({"hotel"}))
    pictured_planned = resolve_place_images(planned, destination)
    image_map: dict[str, dict[str, str]] = {}
    for entry in [*pictured_planned, *pictured]:
        name = str(entry.get("name") or "")
        image = entry.get("image_url")
        if name and image and name not in image_map:
            image_map[name] = {
                "image_url": str(image),
                "fallback_image_url": str(entry.get("fallback_image_url") or ""),
            }
    return pictured, image_map


def _start_picture_pass(
    image_slot: dict[str, Any],
    destination: str,
    recommended: list[dict[str, Any]],
    planned: list[dict[str, Any]],
) -> None:
    """Send the picture pass to its own thread and remember how to collect it."""
    # Snapshots, not the state's own dicts: the nodes that run while this is in flight read
    # the recommendation list, and neither side should be able to see the other's writes.
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="place-images")
    image_slot["pool"] = pool
    image_slot["future"] = pool.submit(
        _picture_pass,
        destination,
        [dict(entry) for entry in recommended],
        [dict(entry) for entry in planned],
    )


def _join_place_images(state: PlannerState, image_slot: dict[str, Any] | None) -> PlannerState:
    """Fold the picture pass into the state, waiting for it only if it is still running.

    Reaching here means the analysis and the written summary have already had their turn,
    so on a normal run this wait is zero. If the pass never finishes, the plan is served
    with the bundled placeholders it would have fallen back to anyway rather than without
    any pictures at all.
    """
    if not image_slot:
        return state
    future = image_slot.get("future")
    if future is None:
        return state
    try:
        pictured, image_map = future.result(timeout=IMAGE_JOIN_TIMEOUT)
    except Exception:
        logger.warning("the picture pass did not finish; serving the plan without its photos", exc_info=True)
        placeholders = resolve_place_images(
            [dict(entry) for entry in state.get("recommended_places") or []],
            state["trip"]["destination"],
            skip_categories=frozenset({"hotel"}),
            max_lookups=0,
        )
        return {**state, "recommended_places": placeholders}
    finally:
        pool = image_slot.get("pool")
        if pool is not None:
            pool.shutdown(wait=False)
    return {**state, "recommended_places": pictured, "place_image_map": image_map}


def _recommend_places(state: PlannerState, image_slot: dict[str, Any] | None = None) -> PlannerState:
    """Extract named recommendations and guarantee a picture and a cost for each."""
    trip = state["trip"]
    destination = trip["destination"]
    trip_days = _trip_days(trip)
    # Recommendations were previously chosen with no idea what the trip cost, so an
    # 8000 CNY trip was offered 3840 and 4800 hotels. These ceilings are derived from
    # the trip's own budget, told to the model, and used to flag anything still over.
    allowances = budget_allowances(trip.get("budget"), trip_days, state.get("traveler_profile"))
    # How many of each kind this trip needs so that no day repeats another day's sights
    # or meals. Without it the answer was a handful of places, and a five-day plan had
    # to put the same restaurant and the same sight on every day.
    limits = category_limits(trip_days)
    recommended, status = extract_recommendations(
        destination,
        state.get("web_results", []),
        allowances=allowances,
        limits=limits,
    )
    progress_callback = state.get("progress_callback")

    def publish_category(category: str, entries: list[dict[str, Any]]) -> None:
        if progress_callback is None:
            return
        # These cards intentionally carry no price. Final prices and budget tiers are not
        # known until every category has completed and ``estimate_trip_costs`` runs.
        progress_callback(
            {
                "kind": "category",
                "category": category,
                "recommendations": [
                    {
                        "name": entry.get("name", ""),
                        "category": category,
                        "reason": entry.get("reason"),
                        "meal_slot": entry.get("meal_slot"),
                        "image_url": entry.get("image_url"),
                        "fallback_image_url": entry.get("fallback_image_url"),
                        # Present on the early category cards too, so the map can place a
                        # stop as soon as its category finishes instead of waiting for the
                        # whole plan.
                        "latitude": entry.get("latitude"),
                        "longitude": entry.get("longitude"),
                        "price_pending": True,
                    }
                    for entry in entries
                    if entry.get("name")
                ],
            }
        )

    for category in CATEGORY_ORDER:
        initial = [entry for entry in recommended if entry.get("type") == category]
        if initial:
            publish_category(category, initial)
    if status != "skipped":
        # Hotels, food and sights should all show up, and each should hold enough
        # candidates for the whole trip, not just enough to prove the category exists.
        recommended = ensure_category_coverage(
            destination,
            recommended,
            allowances=allowances,
            limits=limits,
            on_category_ready=publish_category,
        )
    recommended = enrich_prices_with_search(recommended, destination)
    # A hotel card shows a rating and a price instead of a picture, and a restaurant card
    # shows its signature dish -- which is also the term its picture is searched by. Both
    # come from one batched live search, so this must run before the cost estimate (the
    # hotel's nightly rate is what the budget should charge) and before the picture pass.
    recommended = enrich_place_facts(recommended, destination)
    recommended = estimate_trip_costs(
        recommended, trip_days, state.get("traveler_profile"), trip.get("budget"), destination=destination
    )
    # The traveller asked for a spread: food that follows their budget AND is not one price point
    # repeated -- cheap places they can eat at every day, and the expensive one that is genuinely
    # worth it. Prices are known by now (the facts pass filled in hotel rates and dishes), so the
    # thin tiers can be seen and searched for by name.
    recommended, spread = ensure_price_spread(
        destination,
        recommended,
        trip.get("budget"),
        trip_days,
        state.get("traveler_profile"),
    )
    # New prices arrived with the backfilled cards, so the cost gate runs again over the whole
    # list: it is idempotent for the cards that already had one.
    if spread:
        recommended = estimate_trip_costs(
            recommended, trip_days, state.get("traveler_profile"), trip.get("budget"), destination=destination
        )
    # Hotels are deliberately left without a picture: a stock photo of a different hotel
    # tells the reader nothing, so the card carries its rating and price instead.
    #
    # This is the last thing the node does and nothing after it needs the result until the
    # plan is composed, so it runs on its own thread and is joined in `compose_plan`.
    # Without a slot (a direct call, as the tests make) it simply runs inline.
    planned_entries = _planned_place_entries(state)
    if image_slot is None:
        recommended, image_map = _picture_pass(destination, recommended, planned_entries)
    else:
        _start_picture_pass(image_slot, destination, recommended, planned_entries)
        image_map = {}
    messages = {
        "completed": f"已从检索资料中提取 {len(recommended)} 个具体推荐地点并配图。",
        "fallback": f"未使用大模型，已按关键词整理 {len(recommended)} 个推荐地点并配图。",
        "skipped": "检索资料不足，暂无可提取的推荐地点。",
    }
    events = list(state.get("events", []))
    events.append(
        {
            "tool": "place_recommender",
            "status": status,
            "message": messages.get(status, messages["skipped"]),
        }
    )
    return {
        **state,
        "recommended_places": recommended,
        "generated_daily_plan": _generated_daily_plan(state, recommended),
        "place_image_map": image_map,
        "events": events,
    }


def _scheduled_days(state: PlannerState) -> list[dict[str, Any]]:
    """What is planned per day, from the real itinerary when there is one.

    A trip with saved itinerary items has a definite schedule, so the weather advice
    must look at that; otherwise it falls back to the day plan the agent proposed.
    """
    itinerary = state.get("itinerary", [])
    if not itinerary:
        return state.get("generated_daily_plan") or []

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in itinerary:
        item_date = item.get("date")
        day = item_date.isoformat() if hasattr(item_date, "isoformat") else str(item_date or "")
        if not day:
            continue
        grouped.setdefault(day, []).append({"place_name": item.get("place_name")})
    return [{"date": day, "items": items} for day, items in sorted(grouped.items())]


def _analyze(state: PlannerState) -> PlannerState:
    trip = state["trip"]
    days = (date.fromisoformat(trip["end_date"]) - date.fromisoformat(trip["start_date"])).days + 1
    conflicts = detect_conflicts(state.get("itinerary", []))
    # Only saved places are charged: a recommendation is a suggestion, not a booking.
    budget = estimate_budget(state.get("places", []), days, trip.get("budget"), state.get("traveler_profile"))
    # A second, deliberately different number: what the traveller's *own budget* should roughly be
    # spent on, split by category. The planning panel shows this one because it is the only figure
    # that means anything before any place has been chosen; the two sums above and below it stay
    # exact and are never mixed into it (see app/agents/tools/budget_plan.py).
    budget_plan = build_budget_plan(
        trip.get("budget"),
        days,
        trip["destination"],
        state.get("traveler_profile"),
    )
    events = list(state.get("events", []))
    events.append({"tool": "conflict_checker", "status": "completed", "message": f"检测到 {len(conflicts)} 个时间冲突。"})
    events.append({"tool": "budget_estimator", "status": "completed", "message": "已完成预算估算。"})
    if budget_plan.get("status") == "estimated":
        events.append(
            {
                "tool": "budget_planner",
                "status": "completed",
                "message": f"已按总预算 {budget_plan['trip_budget']} CNY 给出分类预算建议（计划花掉 {budget_plan['planned_total']} CNY）。",
            }
        )
    weather = get_weather_forecast(trip["destination"], trip["start_date"], trip["end_date"])
    events.append({"tool": "weather_forecast", "status": weather.get("status", "fallback"), "message": "已查询旅行日期天气预报。" if weather.get("status") == "completed" else weather.get("message", "天气预报不可用。")})
    # Pair the forecast with what is actually scheduled, so a wet day is not just
    # displayed but acted on.
    advice = build_weather_advice(
        trip["destination"],
        weather,
        _scheduled_days(state),
        state.get("places", []),
    )
    if advice:
        events.append(
            {
                "tool": "weather_advisor",
                "status": "completed",
                "message": f"发现 {len(advice)} 天天气可能影响行程安排。",
            }
        )
    return {
        **state,
        "conflicts": conflicts,
        "budget": budget,
        "budget_plan": budget_plan,
        "weather": weather,
        "weather_advice": advice,
        "events": events,
    }


def _transport_advice(state: PlannerState) -> dict[str, str | None] | None:
    route_context = state.get("route_context")
    if route_context is None:
        return None
    answer = next(
        (
            result.get("snippet")
            for result in state.get("web_results", [])
            # "web_answer" is the model's own live search: a written answer rather than a
            # list of pages. It is the only source left now that the metered search
            # provider has been removed.
            if result.get("source_type") == "web_answer"
        ),
        None,
    )
    return {
        "origin": f"{route_context['hotel_name']}（{route_context['hotel_address']}）",
        "destination": route_context["destination"],
        "recommendation": answer or "未获得可核验的联网交通结果，请重试后再确认实时票价与时刻。",
    }


_PLACE_TYPE_LABELS = {
    "hotel": "住宿",
    "restaurant": "餐饮",
    "attraction": "景点",
    "airport": "机场",
    "station": "车站",
    "transport": "交通",
}


def _saved_places_with_costs(state: PlannerState) -> list[dict[str, Any]]:
    """The user's saved places, each tied to its own slice of the budget.

    The summary must cover the whole collection, so every place carries its type
    label, address, opening hours and the amount it contributes, instead of only
    appearing as a name inside a single addition expression.
    """
    budget = state["budget"]
    cost_by_name: dict[str, int] = {}
    for line in budget.breakdown or []:
        name = str(line.get("name") or "")
        if name and name not in cost_by_name:
            cost_by_name[name] = int(line.get("amount") or 0)

    places: list[dict[str, Any]] = []
    for place in state.get("places", []):
        name = str(place.get("name") or "")
        places.append(
            {
                "name": name,
                "type": place.get("type"),
                "type_label": _PLACE_TYPE_LABELS.get(str(place.get("type")), "其他"),
                "address": place.get("address"),
                "opening_hours": place.get("opening_hours"),
                "rating": place.get("rating"),
                "price_level": place.get("price_level"),
                "estimated_cost": cost_by_name.get(name),
                # Why the user picked it and the price they were quoted, if it came
                # from a recommendation.
                "note": place.get("note"),
                "quoted_cost": place.get("quoted_cost"),
            }
        )
    return places


def _generate_final_summary(state: PlannerState) -> PlannerState:
    budget = state["budget"]
    conflicts = state.get("conflicts", [])
    saved_places = _saved_places_with_costs(state)
    summary_context = {
        "request": state.get("query", ""),
        "trip": state["trip"],
        "saved_places": saved_places,
        "saved_place_count": len(saved_places),
        "itinerary": [
            {
                **item,
                "date": item["date"].isoformat(),
                "start_time": item["start_time"].isoformat(),
                "end_time": item["end_time"].isoformat(),
            }
            for item in state.get("itinerary", [])
        ],
        "transport_advice": _transport_advice(state),
        "budget": {
            "total_estimate": budget.total_estimate,
            "trip_budget": budget.trip_budget,
            "remaining_budget": budget.remaining_budget,
            "currency": budget.currency,
            "note": budget.note,
            "breakdown": budget.breakdown or [],
            "daily_estimate": budget.daily_estimate,
            "budget_summary": _budget_summary(budget),
            "itemized_summary": budget.itemized_summary,
            # Accommodation and meals are charged only through an actual place, so these
            # reference amounts are reported separately and are NOT part of total_estimate.
            "excluded_allowances": budget.excluded_allowances or [],
        },
        # Handed to the summary prompt as well, so the written summary and the panel agree on
        # which figure is the estimate and which is the sum.
        "budget_plan": state.get("budget_plan") or {},
        "budget_plan_summary": format_budget_plan_summary(state.get("budget_plan")),
        "conflicts": [conflict.message for conflict in conflicts],
        "weather_advice": state.get("weather_advice", []),
        "web_results": state.get("web_results", [])[:3],
        "traveler_profile": state.get("traveler_profile", {}),
        "weather": state.get("weather", {}),
        "budget_summary": _budget_summary(state["budget"]),
        "itemized_summary": state["budget"].itemized_summary,
        "generated_daily_plan": state.get("generated_daily_plan", []),
        "recommendations": state.get("recommended_places", []),
        # Material handed over by the planning and adjustment panels.
        "selected_recommendations": state.get("selected_recommendations", []),
        "adjustment_summary": state.get("adjustment_summary"),
        # The traveller's own orders, tickets and confirmations. Without them the written
        # plan could contradict a hotel the traveller has already paid for, because every
        # other input here comes from search.
        "traveler_documents": state.get("traveler_documents", []),
    }
    final_summary, result_status = generate_final_summary(summary_context)
    events = list(state.get("events", []))
    events.append(
        {
            "tool": "final_summary_generator",
            "status": result_status,
            "message": "已生成最终旅行总结。" if result_status == "completed" else "已使用本地规则生成最终旅行总结。",
        }
    )
    return {**state, "final_summary": final_summary, "events": events}


def _handoff(state: PlannerState) -> dict[str, Any]:
    """The planning/adjustment material this summary was asked to cover."""
    selected = state.get("selected_recommendations") or []
    saved = state.get("places", [])
    normalized = [
        {
            "name": str(item.get("name") or "").strip(),
            "category": str(item.get("category") or "attraction"),
            "estimated_cost": item.get("estimated_cost"),
            "reason": item.get("reason"),
        }
        for item in selected
        if str(item.get("name") or "").strip()
    ]
    return {
        "selected_recommendations": normalized,
        "selected_total": sum(int(item.get("estimated_cost") or 0) for item in normalized),
        "has_adjustment_summary": bool((state.get("adjustment_summary") or "").strip()),
        "saved_place_count": len(saved),
        "saved_place_names": [str(place.get("name") or "") for place in saved if place.get("name")],
    }


def _summary_lead(state: PlannerState, handoff: dict[str, Any]) -> str:
    destination = state["trip"]["destination"]
    parts = [f"已根据 {destination} 的收藏地点和行程生成规划。"]
    if handoff.get("saved_place_count"):
        parts.append(f"你收藏的 {handoff['saved_place_count']} 个地点已全部纳入本次总结。")
    if handoff["selected_recommendations"]:
        parts.append(
            f"本次总结已纳入你在规划建议中选中的 {len(handoff['selected_recommendations'])} 个项目"
            f"（合计约 {handoff['selected_total']} CNY）。"
        )
    if handoff["has_adjustment_summary"]:
        parts.append("对话式调整行程的结论也已合并进本次总结。")
    return "".join(parts)


def _compose(state: PlannerState, image_slot: dict[str, Any] | None = None) -> PlannerState:
    # The only place the picture pass is waited for: by now the conflict check, the budget
    # and the written summary have all been done, which is what it was running behind.
    state = _join_place_images(state, image_slot)
    itinerary = state.get("itinerary", [])
    image_map = state.get("place_image_map", {})
    if not itinerary and state.get("generated_daily_plan"):
        # This suggested plan was assembled before the picture pass finished, so its entries
        # still carry only whatever each recommendation had at that moment -- usually
        # nothing, because the pass runs in parallel precisely so it does not delay the
        # plan. The saved-itinerary branch below merges the resolved photos by place name;
        # without doing the same here, a plan built from suggestions came back with no
        # photo on any day entry while its recommendation cards all had one.
        daily_plan = [
            {
                **day,
                "items": [
                    {
                        **item,
                        "image_url": (image_map.get(item.get("place_name")) or {}).get("image_url")
                        or item.get("image_url"),
                        "fallback_image_url": (image_map.get(item.get("place_name")) or {}).get(
                            "fallback_image_url"
                        )
                        or item.get("fallback_image_url"),
                    }
                    for item in day.get("items", [])
                ],
            }
            for day in state["generated_daily_plan"]
        ]
    else:
        daily_plan = None
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in itinerary:
        images = image_map.get(item["place_name"]) or {}
        grouped.setdefault(item["date"], []).append(
            {
                "place_name": item["place_name"],
                "start_time": item["start_time"].isoformat(),
                "end_time": item["end_time"].isoformat(),
                "note": item.get("note"),
                "image_url": images.get("image_url") or item.get("image_url"),
                "fallback_image_url": images.get("fallback_image_url"),
            }
        )
    daily_plan = daily_plan or [{"date": day, "items": items} for day, items in sorted(grouped.items())]
    sources = [
        {
            "source_type": "saved_place",
            "title": place["name"],
            "url": place.get("source_url"),
            "snippet": place.get("address", ""),
            "image_url": place.get("image_url"),
        }
        for place in state.get("places", [])
    ]
    sources.extend(state.get("web_results", []))
    conflicts = [
        {
            "date": conflict.date,
            "first_place": conflict.first_place,
            "second_place": conflict.second_place,
            "message": conflict.message,
        }
        for conflict in state.get("conflicts", [])
    ]
    trip = state["trip"]
    handoff = _handoff(state)
    response = {
        "trip_id": trip["id"],
        "summary": _summary_lead(state, handoff),
        "final_summary": state["final_summary"],
        "recommendations": state.get("web_results", []),
        "place_recommendations": [
            {
                "name": place.get("name", ""),
                "category": place.get("type", "attraction"),
                "reason": place.get("reason"),
                "price_hint": place.get("price_hint"),
                "unit_price": place.get("unit_price"),
                "price_unit": place.get("price_unit"),
                "estimated_cost": place.get("estimated_cost"),
                "cost_explanation": place.get("cost_explanation"),
                # How many nights/meals this price covers, and what one of them costs:
                # the card lets the traveller change the count and does the multiplication
                # from these numbers.
                "quantity": place.get("quantity", 1),
                "quantity_unit": place.get("quantity_unit"),
                "quantity_max": place.get("quantity_max"),
                "cost_source": place.get("cost_source"),
                "image_url": place.get("image_url"),
                "fallback_image_url": place.get("fallback_image_url"),
                # This list is an explicit whitelist, so anything added to the
                # recommendation dicts must be repeated here or it silently never
                # reaches the API response (Pydantic then fills in its defaults).
                # `image_credit` is one of those: dropping it would use a Pexels photo
                # without the attribution the API requires.
                "image_credit": place.get("image_credit"),
                "image_credit_url": place.get("image_credit_url"),
                # The rough position of the place, for the workspace map's straight-line
                # sketch. Like every other field here it is repeated explicitly: this list is
                # a whitelist and anything left out of it never reaches the API response.
                "latitude": place.get("latitude"),
                "longitude": place.get("longitude"),
                "rating": place.get("rating"),
                "signature_dish": place.get("signature_dish"),
                "meal_slot": place.get("meal_slot"),
                "over_budget": bool(place.get("over_budget")),
                "budget_limit": place.get("budget_limit"),
                "budget_note": place.get("budget_note"),
                # Which tier of the traveller's own budget this card is, and whether its price
                # was corrected to the local floor. Both are computed in estimate_trip_costs and
                # both are silently lost if they are not repeated in this whitelist.
                "price_tier": place.get("price_tier"),
                "price_tier_label": place.get("price_tier_label"),
                "price_adjusted": bool(place.get("price_adjusted")),
                "is_free": bool(place.get("is_free")),            }
            for place in state.get("recommended_places", [])
        ],
        "transport_advice": _transport_advice(state),
        "daily_plan": daily_plan,
        "budget": {
            "total_estimate": state["budget"].total_estimate,
            "trip_budget": state["budget"].trip_budget,
            "remaining_budget": state["budget"].remaining_budget,
            "currency": state["budget"].currency,
            "note": state["budget"].note,
            "breakdown": state["budget"].breakdown or [],
            "daily_estimate": state["budget"].daily_estimate,
            "budget_summary": _budget_summary(state["budget"]),
            "itemized_summary": state["budget"].itemized_summary,
            # Accommodation and meals are charged only through an actual place, so these
            # reference amounts are reported separately and are NOT part of total_estimate.
            "excluded_allowances": state["budget"].excluded_allowances or [],
        },
        # What the traveller's own budget should roughly be spent on, by category. This is the
        # figure the planning panel leads with, and it is an estimate rather than a sum.
        "budget_plan": state.get("budget_plan") or {},
        "conflicts": conflicts,
        "sources": sources,
        "events": state.get("events", []),
        "traveler_profile": state.get("traveler_profile", {}),
        "weather": state.get("weather", {}),
        "weather_advice": state.get("weather_advice", []),
        "weather_adjustment_request": weather_adjustment_request(
            state["trip"]["destination"], state.get("weather_advice", [])
        ),
        "budget_summary": _budget_summary(state["budget"]),
        "handoff": handoff,
    }
    return {**state, "response": response}


def build_planner_graph():
    # One slot per built graph: `recommend_places` starts the picture pass and `compose_plan`
    # collects it, and the two nodes need somewhere to meet that is not the graph state.
    image_slot: dict[str, Any] = {}
    graph = StateGraph(PlannerState)
    graph.add_node("load_context", _load_context)
    graph.add_node("live_search", _search_live_web)
    graph.add_node("recommend_places", partial(_recommend_places, image_slot=image_slot))
    graph.add_node("analyze_budget_and_conflicts", _analyze)
    graph.add_node("generate_final_summary", _generate_final_summary)
    graph.add_node("compose_plan", partial(_compose, image_slot=image_slot))
    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "live_search")
    graph.add_edge("live_search", "recommend_places")
    graph.add_edge("recommend_places", "analyze_budget_and_conflicts")
    graph.add_edge("analyze_budget_and_conflicts", "generate_final_summary")
    graph.add_edge("generate_final_summary", "compose_plan")
    graph.add_edge("compose_plan", END)
    return graph.compile()
