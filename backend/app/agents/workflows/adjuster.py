import json
import os
import re
from datetime import date, time, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from openai import OpenAI

from app.agents.state import AdjustmentState
from app.agents.tools.budget import estimate_itinerary_budget
from app.agents.tools.conflicts import detect_conflicts
from app.agents.tools.llm import MAX_RETRIES, MODEL_NAME, REQUEST_TIMEOUT
from app.agents.tools.web_search import search_web
from app.services.weather_service import get_weather_forecast
from app.schemas.adjustment import AdjustmentOperation
from app.schemas.place import PlaceCreate


_SCHEDULE_PATTERN = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})\s+(?P<start>\d{1,2}:\d{2})\s*[-~–—至到]\s*(?P<end>\d{1,2}:\d{2})"
)
_TIME_RANGE_PATTERN = re.compile(r"(?P<start>\d{1,2}:\d{2})\s*[-~–—至到]\s*(?P<end>\d{1,2}:\d{2})")
_DEFAULT_NOTE = "由行程调整 Agent 建议"
_BUDGET_PATTERN = re.compile(r"(?:预算|花费|费用)[^0-9]{0,12}(?P<amount>\d{2,7})\s*(?:元|块|人民币|CNY|rmb)?", re.IGNORECASE)


def _focused_query(instruction: str) -> str:
    """Keep search terms about the requested activity instead of sending the full chat turn."""
    terms = []
    if any(word in instruction for word in ("乐园", "游乐园", "项目", "游玩")):
        terms.append("乐园项目")
    if any(word in instruction for word in ("晚饭", "晚餐", "吃饭", "必吃")):
        terms.append("晚餐")
    if any(word in instruction for word in ("必吃", "必喝", "美食", "饮品")):
        terms.append("必吃必喝")
    return " ".join(dict.fromkeys(terms)) or instruction[:120]


def _request_date(instruction: str, trip: dict[str, Any]) -> date | None:
    full = re.search(r"(20\d{2})[-年](\d{1,2})[-月](\d{1,2})日?", instruction)
    if full:
        return date(int(full.group(1)), int(full.group(2)), int(full.group(3)))
    partial = re.search(r"(\d{1,2})月(\d{1,2})(?:日|号)?", instruction)
    if partial:
        year = date.fromisoformat(trip["start_date"]).year
        return date(year, int(partial.group(1)), int(partial.group(2)))
    return None


def _request_end_time(instruction: str) -> time:
    match = re.search(r"(?:晚上|晚间|到|至)\s*(\d{1,2})\s*点(?:\s*(\d{1,2})\s*分)?", instruction)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        if hour < 12 and "晚上" in instruction:
            hour += 12
        return time(hour % 24, minute)
    return time(20, 0) if any(word in instruction for word in ("下午", "晚上")) else time(18, 0)


def _is_timeline_request(instruction: str) -> bool:
    return any(word in instruction for word in ("乐园", "游乐园", "游玩", "晚饭", "晚餐", "必吃", "必喝"))


def _recommendation_names(state: AdjustmentState) -> list[str]:
    names: list[str] = []
    for result in state.get("web_results", []):
        title = str(result.get("title", "")).strip()
        if title and title not in names:
            names.append(title)
    return names


def _timeline_operations(state: AdjustmentState, day: date, end: time) -> list[AdjustmentOperation]:
    """Create selectable half-hour recommendations when a request asks for a day timeline."""
    destination = state["trip"].get("destination", "目的地")
    names = _recommendation_names(state)
    slots: list[tuple[str, str, str, str, str]] = []
    for index in range(12):
        start_minutes = 14 * 60 + index * 30
        slot_end = min(start_minutes + 30, end.hour * 60 + end.minute)
        if start_minutes >= slot_end:
            break
        start = time(start_minutes // 60, start_minutes % 60)
        finish = time(slot_end // 60, slot_end % 60)
        if start >= time(18, 0) and start < time(19, 0):
            name = "乐园内晚餐与必吃必喝"
            note = "安排用餐并预留点餐、取餐和休息时间"
            place_type = "restaurant"
        elif start >= time(19, 0):
            name = "乐园夜间项目与离园前安排"
            note = "晚饭后继续游玩，按现场排队情况选择项目"
            place_type = "attraction"
        else:
            name = names[index % len(names)] if names else f"{destination}乐园推荐项目 {index + 1}"
            note = "根据联网资料筛选，建议现场确认排队时间"
            place_type = "attraction"
        slots.append((start.isoformat(), finish.isoformat(), name, note, place_type))

    operations: list[AdjustmentOperation] = []
    for index, (start, finish, name, note, place_type) in enumerate(slots):
        operations.append(
            AdjustmentOperation(
                kind="add_place_and_item",
                date=day,
                start_time=time.fromisoformat(start),
                end_time=time.fromisoformat(finish),
                note=note,
                sort_order=index,
                place=PlaceCreate(
                    name=name,
                    type=place_type,
                    address=destination,
                    price_level=2,
                ),
            )
        )
    return operations


def _is_emergency_request(instruction: str) -> bool:
    return any(
        word in instruction
        for word in ("暴雨", "大雨", "台风", "天气不好", "下雨", "无法出行", "不能去", "去不了", "景点关闭", "闭园", "交通中断", "交通停运")
    )


def _emergency_operations(state: AdjustmentState) -> list[AdjustmentOperation]:
    """Replace an unsafe outdoor day with a concrete hotel/indoor fallback timeline."""
    instruction = state["instruction"]
    if not _is_emergency_request(instruction):
        return []
    trip = state["trip"]
    requested_day = _request_date(instruction, trip)
    weather_days = [day for day in (state.get("weather") or {}).get("days", []) if day.get("indoor_priority")]
    affected_dates = [requested_day] if requested_day else [date.fromisoformat(str(day["date"])[:10]) for day in weather_days if day.get("date")]
    if not affected_dates:
        affected_dates = [date.fromisoformat(trip["start_date"])]
    affected = set(affected_dates)
    existing = [item for item in state.get("itinerary", []) if date.fromisoformat(str(item.get("date"))[:10]) in affected]
    operations: list[AdjustmentOperation] = [
        AdjustmentOperation(kind="remove_item", item_id=item["id"])
        for item in existing
        if item.get("id") is not None
    ]
    destination = trip.get("destination", "目的地")
    hotel = next((place for place in state.get("places", []) if place.get("type") == "hotel"), None)
    hotel_name = hotel.get("name") if hotel else "酒店"
    hotel_address = hotel.get("address", destination) if hotel else destination
    fallback_places = [
        ("返回酒店交通", "station", "先离开受影响区域，优先选择地铁/出租车并预留缓冲", 1),
        (f"{hotel_name}室内休息与亲子活动", "hotel", "在酒店内休息、桌游或亲子小游戏，适合成人和儿童", 1),
        ("酒店外卖晚餐", "restaurant", "根据实时营业状态选择外卖，优先配送到酒店", 2),
        ("酒店内电影、桌游与旅行整理", "hotel", "安排轻量室内活动并整理第二天行程", 1),
    ]
    for day in affected_dates:
        start_minutes = 16 * 60
        for index, (name, place_type, note, price_level) in enumerate(fallback_places):
            start = time(start_minutes // 60, start_minutes % 60)
            end = time((start_minutes + 30) // 60, (start_minutes + 30) % 60)
            operations.append(
                AdjustmentOperation(
                    kind="add_place_and_item",
                    date=day,
                    start_time=start,
                    end_time=end,
                    note=note,
                    sort_order=index,
                    place=PlaceCreate(name=name, type=place_type, address=hotel_address, price_level=price_level),
                )
            )
            start_minutes += 30
    return operations


def _load_context(state: AdjustmentState) -> AdjustmentState:
    events = list(state.get("events", []))
    documents = state.get("traveler_documents") or []
    message = "已加载旅行项目、收藏地点和当前行程。"
    if documents:
        message += f"并载入 {len(documents)} 段来自旅客上传资料的依据。"
    events.append({"tool": "context_loader", "status": "completed", "message": message})
    trip = state["trip"]
    weather = state.get("weather") or get_weather_forecast(trip["destination"], trip["start_date"], trip["end_date"])
    events.append({"tool": "weather_forecast", "status": weather.get("status", "fallback"), "message": "已查询旅行日期天气预报。"})
    return {**state, "weather": weather, "events": events}


def _search_live_web(state: AdjustmentState) -> AdjustmentState:
    """Gather web material through the model's own live search.

    Renamed from ``_search_tavily`` when the metered search provider was removed; the event
    name is what the interface shows, and it must not name a service the plan never used.
    """
    events = list(state.get("events", []))
    if not state.get("include_web", False):
        events.append({"tool": "live_search", "status": "skipped", "message": "用户未启用联网搜索。"})
        return {**state, "web_results": [], "events": events}
    trip = state["trip"]
    results, result_status = search_web(_focused_query(state["instruction"]), trip["destination"])
    events.append(
        {
            "tool": "live_search",
            "status": "completed" if result_status == "completed" else "fallback",
            "message": result_status,
        }
    )
    return {**state, "web_results": results, "events": events}


def _model_client() -> OpenAI | None:
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return None
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout=REQUEST_TIMEOUT,
        max_retries=MAX_RETRIES,
    )


def _fallback_operations(state: AdjustmentState) -> list[AdjustmentOperation]:
    budget_operation = _budget_operation(state["instruction"])
    emergency = _emergency_operations(state)
    if emergency:
        return ([budget_operation] if budget_operation is not None else []) + emergency
    replacement = _replacement_operations(state)
    if replacement:
        return ([budget_operation] if budget_operation is not None else []) + replacement
    if budget_operation is not None:
        return [budget_operation]
    match = _SCHEDULE_PATTERN.search(state["instruction"])
    request_date = _request_date(state["instruction"], state["trip"])
    if match is None and request_date is None:
        return []
    if match is None and request_date is not None and _is_timeline_request(state["instruction"]):
        return _timeline_operations(state, request_date, _request_end_time(state["instruction"]))
    instruction = state["instruction"].casefold()
    matches = [place for place in state.get("places", []) if place["name"].casefold() in instruction]
    if not matches:
        return []
    place = max(matches, key=lambda candidate: len(candidate["name"]))
    if match is not None:
        day, start, end = date.fromisoformat(match.group("date")), time.fromisoformat(match.group("start")), time.fromisoformat(match.group("end"))
    else:
        day, start, end = request_date, time(14, 0), _request_end_time(state["instruction"])
    return [
        AdjustmentOperation(
            kind="add_item",
            place_id=place["id"],
            date=day,
            start_time=start,
            end_time=end,
            note=_DEFAULT_NOTE,
            sort_order=0,
        )
    ]


def _budget_operation(instruction: str) -> AdjustmentOperation | None:
    if not any(word in instruction for word in ("预算", "花费", "费用")):
        return None
    match = _BUDGET_PATTERN.search(instruction)
    if match is None:
        match = re.search(r"(?:调整到|改为|改成|控制在|不超过)\s*(\d{2,7})\s*(?:元|块|人民币|CNY|rmb)?", instruction, re.IGNORECASE)
    if match is None:
        return None
    amount = int(match.groupdict().get("amount") or match.group(1))
    if amount <= 0:
        # A zero budget is never a meaningful instruction; refusing it keeps the
        # UI from showing "调整为 0 CNY" and from zeroing a real trip budget.
        return None
    return AdjustmentOperation(kind="set_budget", budget_amount=amount)


def _replacement_operations(state: AdjustmentState) -> list[AdjustmentOperation]:
    """Handle explicit natural-language requests to remove a time slot and replace it."""
    instruction = state["instruction"]
    if not any(word in instruction for word in ("不想", "不要", "去掉", "删除", "换一个", "换个", "替换")):
        return []
    time_match = _TIME_RANGE_PATTERN.search(instruction)
    if time_match is None:
        return []
    start = time.fromisoformat(time_match.group("start"))
    end = time.fromisoformat(time_match.group("end"))
    quoted = re.findall(r"[「『“\"]([^」』”\"]+)[」』”\"]", instruction)
    components = [
        part.strip(" +、，,/")
        for phrase in quoted
        for part in re.split(r"[+＋、，,]", phrase)
        if part.strip()
    ]
    candidates = [
        str(place.get("name", "")).strip()
        for place in state.get("places", [])
        if str(place.get("name", "")).strip()
    ] + [
        str(result.get("title", "")).strip()
        for result in state.get("web_results", [])
        if str(result.get("title", "")).strip()
    ]
    destination_text = str(state["trip"].get("destination", ""))
    if any(word in destination_text for word in ("迪士尼", "乐园", "游乐园")):
        candidates.extend(["加勒比海盗——沉落宝藏之战", "飞跃地平线", "创极速光轮"])
    matching_items = []
    for item in state.get("itinerary", []):
        item_start = item.get("start_time")
        item_end = item.get("end_time")
        item_date = str(item.get("date", ""))[:10]
        same_slot = str(item_start)[:5] == time_match.group("start") and str(item_end)[:5] == time_match.group("end")
        same_name = not components or any(component in str(item.get("place_name", "")) for component in components)
        if same_slot and same_name:
            matching_items.append(item)
    if not matching_items:
        return []
    destination = state["trip"].get("destination", "目的地")
    operations: list[AdjustmentOperation] = []
    used_names = set(components)
    for item in matching_items:
        replacement_name = next(
            (name for name in candidates if name not in used_names and not any(name in component or component in name for component in components)),
            None,
        )
        if replacement_name is None:
            replacement_name = f"{destination}备用推荐项目 {len(operations) // 2 + 1}"
        used_names.add(replacement_name)
        replacement_result = next(
            (result for result in state.get("web_results", []) if str(result.get("title", "")).strip() == replacement_name),
            None,
        )
        replacement_price_level = int(replacement_result.get("price_level", 2)) if replacement_result else 2
        operations.extend(
            [
                AdjustmentOperation(kind="remove_item", item_id=item["id"]),
                AdjustmentOperation(
                    kind="add_place_and_item",
                    date=date.fromisoformat(str(item["date"])[:10]),
                    start_time=start,
                    end_time=end,
                    note="替换为新的推荐项目，保留原时间段",
                    sort_order=int(item.get("sort_order", 0)),
                    place=PlaceCreate(
                        name=replacement_name,
                        type="attraction",
                        address=destination,
                        price_level=max(0, min(5, replacement_price_level)),
                        image_url=(replacement_result or {}).get("image_url"),
                        source_url=(replacement_result or {}).get("url"),
                    ),
                ),
            ]
        )
    return operations


def _model_operations(state: AdjustmentState) -> list[AdjustmentOperation]:
    client = _model_client()
    if client is None:
        raise RuntimeError("DASHSCOPE_API_KEY is not configured")
    prompt = {
        "instruction": state["instruction"],
        "trip": state["trip"],
        "saved_places": state.get("places", []),
        "itinerary": state.get("itinerary", []),
        "buildings_from_planning_agent": state.get("candidates", []),
        "web_results": state.get("web_results", []),
        "traveler_profile": state.get("traveler_profile", {}),
        "weather": state.get("weather", {}),
        # The traveller's own uploaded orders, tickets and confirmations, retrieved for
        # this instruction. Each entry carries the file it came from.
        "traveler_documents": state.get("traveler_documents", []),
    }
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You convert a travel itinerary adjustment request into JSON only. "
                    "Return an object with an operations array. Each operation has kind "
                    "(remove_item, add_item, add_place_and_item, update_item, or set_budget), item_id, place_id, place, "
                    "date, start_time, end_time, note, and sort_order. Use saved place IDs where possible. "
                    "For budget changes use set_budget with budget_amount only. Never invent IDs. Dates must stay inside the trip dates. "
                    "Prefer update_item (with item_id plus the new date/start_time/end_time) when the user only wants to move an "
                    "item to another day or change its time; never emit remove_item for that. "
                    "Use buildings_from_planning_agent as the first source of new places: these are the recommendations the "
                    "traveler was already shown, and they carry the agreed estimated_cost and quantity. When you add one, copy its "
                    "name, category (as type), estimated_cost and quantity into the place object rather than inventing a price. "
                    "Use the current itinerary as the only baseline and preserve unaffected dates and time slots. Support replacing multiple requested items in one response; keep each replacement in its original slot unless the user asks to replan. "
                    "If weather, a closure, or transport disruption makes outdoor travel unsafe, remove affected items and create a complete replacement timeline including return-to-hotel transport, indoor or hotel activities, meals or delivery, and estimated costs. "
                    "Use traveler_profile (adults/children and gender counts) to prefer age-appropriate, accessible activities. "
                    "traveler_documents holds excerpts of documents the traveller uploaded themselves (hotel orders, tickets, "
                    "confirmations), each with its source filename. When the request concerns a booked hotel, a flight, a ticket, "
                    "a price already paid or an order number, answer from those excerpts and name the document you used. Never "
                    "invent a booking that is not in them, and never contradict one that is. "
                    "Sources are internal context only and must never be copied as a source list into user-facing text."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False, default=str)},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    content = response.choices[0].message.content or "{}"
    payload = json.loads(content)
    raw_operations = payload.get("operations", payload if isinstance(payload, list) else [])
    if not isinstance(raw_operations, list):
        raise ValueError("The model response does not include an operation list.")
    return [AdjustmentOperation.model_validate(operation) for operation in raw_operations]


def _conflicts_with_manual(model_operation: AdjustmentOperation, manual: list[AdjustmentOperation]) -> bool:
    """Whether the model is about to redo something the user already decided by clicking."""
    if model_operation.kind == "set_budget" and any(operation.kind == "set_budget" for operation in manual):
        return True
    touched_items = {
        operation.item_id
        for operation in manual
        if operation.kind in {"remove_item", "update_item"} and operation.item_id is not None
    }
    if model_operation.kind in {"remove_item", "update_item"} and model_operation.item_id in touched_items:
        return True
    occupied = {
        (str(operation.date), str(operation.start_time))
        for operation in manual
        if operation.kind in {"add_item", "add_place_and_item"}
    }
    if model_operation.kind in {"add_item", "add_place_and_item"}:
        return (str(model_operation.date), str(model_operation.start_time)) in occupied
    return False


# Words that mean the sentence asks for more than the one shortcut that matched.
_SECOND_TASK_WORDS = ("换", "替换", "删", "去掉", "取消", "增加", "加一个", "加个", "安排", "推荐", "想吃", "想去")


def _asks_for_more_than_budget(state: AdjustmentState, instruction: str) -> bool:
    """Whether a budget-only match is leaving a second request in the sentence unread.

    The deterministic shortcuts take priority over the model, and that is usually right:
    "预算改成6000" needs no LLM. But when the budget shortcut matched, everything else in
    the sentence used to be dropped without a word -- including "把博物馆换成室内的" in
    "把 12月3日 14:00-16:00 的博物馆换成室内的，并把总预算改成 6000 元". In that case the
    model is still consulted, and its operations are appended to the explicit ones.
    """
    if _SCHEDULE_PATTERN.search(instruction):
        return True
    if any(word in instruction for word in _SECOND_TASK_WORDS):
        return True
    names = {str(place.get("name", "")) for place in state.get("places", [])}
    names |= {str(item.get("place_name", "")) for item in state.get("itinerary", [])}
    return any(len(name) >= 2 and name in instruction for name in names)


def _parse_operations(state: AdjustmentState) -> AdjustmentState:
    events = list(state.get("events", []))
    requested = list(state.get("requested_operations") or [])
    if requested and state.get("structured_only"):
        # Every edit came from a click, so there is no separate request for the model to
        # interpret -- and letting it run would let it invent edits nobody asked for.
        events.append(
            {
                "tool": "adjustment_parser",
                "status": "completed",
                "message": f"已按你在界面上选择的 {len(requested)} 项改动生成草案。",
            }
        )
        return {**state, "operations": requested, "events": events}
    if requested:
        # The user clicked some changes AND typed a request. Both are honoured: the clicks
        # are kept as they are, and the model's answer is added for everything the clicks
        # did not already decide, so the typed sentence is never silently dropped.
        try:
            from_model = _model_operations(state)
        except Exception:
            from_model = _fallback_operations(state)
        extra = [operation for operation in from_model if not _conflicts_with_manual(operation, requested)]
        events.append(
            {
                "tool": "adjustment_parser",
                "status": "completed",
                "message": f"已合并你在界面上的 {len(requested)} 项改动和文字要求里的 {len(extra)} 项改动。",
            }
        )
        return {**state, "operations": requested + extra, "events": events}
    try:
        explicit_replacement = _replacement_operations(state)
        explicit_budget = _budget_operation(state["instruction"])
        explicit_emergency = _emergency_operations(state)
        explicit_operations = ([explicit_budget] if explicit_budget is not None else []) + (explicit_emergency or explicit_replacement)
        operations = explicit_operations or _model_operations(state)
        if (
            explicit_budget is not None
            and not explicit_emergency
            and not explicit_replacement
            and _asks_for_more_than_budget(state, state["instruction"])
        ):
            # The budget phrase was only part of the request; let the model read the rest.
            try:
                extra = [
                    operation
                    for operation in _model_operations(state)
                    if not _conflicts_with_manual(operation, explicit_operations)
                ]
            except Exception:
                extra = _fallback_operations(state)
            operations = explicit_operations + extra
        request_date = _request_date(state["instruction"], state["trip"])
        if not explicit_operations and request_date and _is_timeline_request(state["instruction"]) and not _SCHEDULE_PATTERN.search(state["instruction"]):
            expected_end = _request_end_time(state["instruction"])
            expected_count = max(0, (expected_end.hour * 60 + expected_end.minute - 14 * 60) // 30)
            if len(operations) < expected_count or not any(
                operation.start_time == time(19, 0) for operation in operations
            ):
                operations = _timeline_operations(state, request_date, expected_end)
        if explicit_operations and len(operations) > len(explicit_operations):
            message = "已按明确的预算或替换要求生成草案，并用模型补全了文字里的其他改动。"
        elif explicit_operations:
            message = "已按明确的预算、删除或替换要求生成结构化草案。"
        else:
            message = "已生成结构化行程调整草案。"
        events.append({"tool": "adjustment_parser", "status": "completed", "message": message})
    except Exception as error:
        operations = _fallback_operations(state)
        events.append(
            {
                "tool": "adjustment_parser",
                "status": "fallback",
                "message": f"已使用本地规则生成草案：{type(error).__name__}。",
            }
        )
    return {**state, "operations": operations, "events": events}


def _preview_items(state: AdjustmentState) -> list[dict[str, Any]]:
    items = [dict(item) for item in state.get("itinerary", [])]
    places_by_id = {place["id"]: place for place in state.get("places", [])}
    for index, operation in enumerate(state.get("operations", [])):
        if operation.kind == "remove_item":
            items = [item for item in items if item["id"] != operation.item_id]
        elif operation.kind == "update_item":
            for item in items:
                if item["id"] != operation.item_id:
                    continue
                if operation.date is not None:
                    item["date"] = operation.date
                if operation.start_time is not None:
                    item["start_time"] = operation.start_time
                if operation.end_time is not None:
                    item["end_time"] = operation.end_time
                if operation.note is not None:
                    item["note"] = operation.note
                item["changed"] = "moved"
                item["operation_index"] = index
        elif operation.kind == "add_item" and operation.place_id is not None:
            place = places_by_id.get(operation.place_id)
            if place is not None:
                items.append(
                    {
                        "id": None,
                        "place_id": place["id"],
                        "date": operation.date,
                        "start_time": operation.start_time,
                        "end_time": operation.end_time,
                        "place_name": place["name"],
                        "image_url": place.get("image_url"),
                        "note": operation.note,
                        "sort_order": operation.sort_order,
                        "type": place.get("type"),
                        "price_level": place.get("price_level"),
                        "quoted_cost": place.get("quoted_cost"),
                        "quantity": place.get("quantity"),
                        "address": place.get("address"),
                        "changed": "added",
                        "operation_index": index,
                    }
                )
        elif operation.kind == "add_place_and_item" and operation.place is not None:
            place = operation.place
            items.append(
                {
                    "id": None,
                    "place_id": None,
                    "date": operation.date,
                    "start_time": operation.start_time,
                    "end_time": operation.end_time,
                    "place_name": place.name,
                    "image_url": str(place.image_url) if place.image_url else None,
                    "note": operation.note,
                    "sort_order": operation.sort_order,
                    "type": str(place.type),
                    "price_level": place.price_level,
                    "quoted_cost": place.estimated_cost,
                    "quantity": place.quantity,
                    "address": place.address,
                    # Which operation created this row, so the panel can let the user edit
                    # that operation's price and send it back on apply.
                    "operation_index": index,
                    "changed": "added",
                }
            )
    return items


def _validate_preview(state: AdjustmentState) -> AdjustmentState:
    trip = state["trip"]
    days = (date.fromisoformat(trip["end_date"]) - date.fromisoformat(trip["start_date"])).days + 1
    preview_items = _preview_items(state)
    conflicts = detect_conflicts(preview_items)
    budget_operation = next((operation for operation in state.get("operations", []) if operation.kind == "set_budget"), None)
    requested_budget = budget_operation.budget_amount if budget_operation is not None else trip.get("budget")
    # Priced from the itinerary that the preview actually shows, so deleting a scheduled
    # item takes its money out of the total and adding one puts it in. The old version
    # priced the whole 收藏地点 collection instead, which is why removing a day's plan
    # barely moved the number while sixteen unscheduled saved places were all charged.
    budget = estimate_itinerary_budget(preview_items, days, requested_budget, state.get("traveler_profile"))
    weather = state.get("weather") or get_weather_forecast(trip["destination"], trip["start_date"], trip["end_date"])
    events = list(state.get("events", []))
    events.append({"tool": "conflict_checker", "status": "completed", "message": f"检测到 {len(conflicts)} 个时间冲突。"})
    events.append(
        {
            "tool": "budget_estimator",
            "status": "completed",
            "message": f"已按调整后的每日行程逐项估算：共 {len(budget.breakdown or []) - 1} 个行程项。",
        }
    )
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
    daily_plan = _daily_plan(state, preview_items)
    final_summary = _final_summary(state, daily_plan, sources, budget)
    return {
        **state,
        "summary": _draft_summary(state),
        "daily_plan": daily_plan,
        "final_summary": final_summary,
        "sources": sources,
        "conflicts": [
            {
                "date": conflict.date.isoformat(),
                "first_place": conflict.first_place,
                "second_place": conflict.second_place,
                "message": conflict.message,
            }
            for conflict in conflicts
        ],
        "budget": {
            "total_estimate": budget.total_estimate,
            "trip_budget": budget.trip_budget,
            "remaining_budget": budget.remaining_budget,
            "currency": budget.currency,
            "note": budget.note,
            "breakdown": budget.breakdown or [],
            "daily_estimate": budget.daily_estimate,
            "itemized_summary": budget.itemized_summary,
            "excluded_allowances": budget.excluded_allowances or [],
        },
        "events": events,
        "weather": weather,
    }


def _draft_summary(state: AdjustmentState) -> str:
    """Say what this draft does, in terms of the itinerary the user is editing."""
    operations = state.get("operations", [])
    removed = sum(1 for operation in operations if operation.kind == "remove_item")
    added = sum(1 for operation in operations if operation.kind in {"add_item", "add_place_and_item"})
    moved = sum(1 for operation in operations if operation.kind == "update_item")
    budget_changed = any(operation.kind == "set_budget" for operation in operations)
    parts = []
    if removed:
        parts.append(f"删除 {removed} 项")
    if added:
        parts.append(f"新增 {added} 项")
    if moved:
        parts.append(f"调整 {moved} 项的时间或日期")
    if budget_changed:
        parts.append("调整旅行总预算")
    if not parts:
        return "没有识别出可执行的行程改动；请在下面直接勾选要删除、移动或新增的项目。"
    return f"草案：{'、'.join(parts)}。确认应用前不会修改已保存的地点或行程。"


def _daily_plan(state: AdjustmentState, preview_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The whole trip as it would look after applying, grouped by day.

    This used to show a single day -- the one named in the instruction -- and only when
    the instruction happened to name one. The editor now always shows every day, with
    each item flagged as kept, moved or added, which is what makes the preview usable
    for judging a change.
    """
    trip = state["trip"]
    try:
        start = date.fromisoformat(str(trip["start_date"])[:10])
        end = date.fromisoformat(str(trip["end_date"])[:10])
    except (KeyError, TypeError, ValueError):
        start = end = None
    days: list[str] = []
    if start and end and end >= start:
        cursor = start
        while cursor <= end:
            days.append(cursor.isoformat())
            cursor += timedelta(days=1)
    for item in preview_items:
        value = str(item.get("date") or "")[:10]
        if value and value not in days:
            days.append(value)
    days.sort()

    ordered = sorted(
        preview_items,
        key=lambda item: (str(item.get("date") or "")[:10], str(item.get("start_time") or "")),
    )
    plan: list[dict[str, Any]] = []
    for day in days:
        normalized = []
        for item in ordered:
            if str(item.get("date") or "")[:10] != day:
                continue
            normalized.append(
                {
                    "item_id": item.get("id"),
                    "place_id": item.get("place_id"),
                    "start_time": item.get("start_time").isoformat() if hasattr(item.get("start_time"), "isoformat") else str(item.get("start_time") or ""),
                    "end_time": item.get("end_time").isoformat() if hasattr(item.get("end_time"), "isoformat") else str(item.get("end_time") or ""),
                    "place_name": item.get("place_name", "行程安排"),
                    "note": item.get("note"),
                    "image_url": item.get("image_url"),
                    "type": item.get("type"),
                    "changed": item.get("changed") or "kept",
                }
            )
        plan.append({"date": day, "items": normalized})
    return plan


def _names_for_item_ids(state: AdjustmentState, item_ids: list[Any]) -> list[str]:
    """The names of the itinerary items being touched, for the human-readable summary."""
    wanted = {value for value in item_ids if value is not None}
    return [
        str(item.get("place_name"))
        for item in state.get("itinerary", [])
        if item.get("id") in wanted and item.get("place_name")
    ]


def _final_summary(
    state: AdjustmentState,
    daily_plan: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    budget: Any = None,
) -> str:
    destination = state["trip"].get("destination", "目的地")
    operations = state.get("operations", [])
    removed_items = [operation for operation in operations if operation.kind == "remove_item"]
    added = [
        operation.place.name
        for operation in operations
        if operation.kind == "add_place_and_item" and operation.place is not None
    ]
    moved = [operation for operation in operations if operation.kind == "update_item"]
    budget_operation = next((operation for operation in operations if operation.kind == "set_budget"), None)
    scheduled = sum(len(day["items"]) for day in daily_plan)
    money = ""
    if budget is not None:
        money = f"调整后的行程共 {scheduled} 项，逐项估算约 {budget.total_estimate} {budget.currency}"
        if budget.trip_budget is not None and budget.remaining_budget is not None:
            money += f"，相对 {budget.trip_budget} CNY 的旅行预算还剩 {budget.remaining_budget} CNY"
        money += "。"

    parts: list[str] = []
    if removed_items:
        names = _names_for_item_ids(state, [operation.item_id for operation in removed_items])
        label = f"（{'、'.join(names[:4])}）" if names else ""
        parts.append(f"删除 {len(removed_items)} 个行程项{label}")
    if added:
        parts.append(f"新增 {'、'.join(added[:4])}" + ("等" if len(added) > 4 else ""))
    if moved:
        parts.append(f"移动或改时间 {len(moved)} 项")
    if budget_operation is not None:
        parts.append(f"把旅行总预算调整为 {budget_operation.budget_amount} CNY")
    if not parts:
        return f"{destination} 的行程没有变化。可以在下面直接删除、移动或新增项目，也可以在用自然语言说明要求。"
    return f"本次调整：{'；'.join(parts)}。{money}确认应用后才会写入行程。"


def build_adjuster_graph():
    graph = StateGraph(AdjustmentState)
    graph.add_node("load_context", _load_context)
    graph.add_node("live_search", _search_live_web)
    graph.add_node("parse_operations", _parse_operations)
    graph.add_node("validate_preview", _validate_preview)
    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "live_search")
    graph.add_edge("live_search", "parse_operations")
    graph.add_edge("parse_operations", "validate_preview")
    graph.add_edge("validate_preview", END)
    return graph.compile()
