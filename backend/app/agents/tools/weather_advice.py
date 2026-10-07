"""Turn a forecast into something the itinerary can act on.

The weather service already decides, per day, whether indoor activities should be
prioritised (`indoor_priority`). Nothing used it: a rainy day still showed the same
plan with no warning, and the user had to notice the mismatch themselves.

This module pairs each day's forecast with what is actually scheduled that day, and
flags the items that are likely to be outdoors, so the app can say "day 2 is wet and
you planned a mountain" and offer to rework it.
"""

from typing import Any

# Names that usually mean an open-air visit. Chinese place names are descriptive, so
# this is a reasonable heuristic; it is a hint for the user to confirm, never a verdict.
OUTDOOR_KEYWORDS = (
    "公园",
    "山",
    "陵",
    "湖",
    "寺",
    "庙",
    "塔",
    "江",
    "河",
    "海",
    "沙滩",
    "长城",
    "园林",
    "湿地",
    "岛",
    "桥",
    "谷",
    "洞",
    "温泉",
    "古镇",
    "步行街",
    "广场",
    "街",
    "苑",
    "园",
)

# Names that are reliably under a roof. Checked before the outdoor list, because a
# name like "西湖美术馆" contains both "湖" and "美术馆" and is indoor.
INDOOR_KEYWORDS = (
    "博物馆",
    "博物院",
    "美术馆",
    "展览",
    "陈列馆",
    "纪念馆",
    "科技馆",
    "图书馆",
    "书店",
    "水族馆",
    "剧院",
    "影院",
    "电影",
    "商场",
    "购物",
    "室内",
    "温泉馆",
)

OUTDOOR = "outdoor"
INDOOR = "indoor"
UNKNOWN = "unknown"


def classify_place(name: str, place_type: str | None = None) -> str:
    """Guess whether a place is visited outdoors, indoors, or neither clearly.

    Indoor keywords are checked first: "博物馆" also contains no outdoor keyword, but
    a name such as "西湖美术馆" contains both "湖" and "美术馆" and is indoor.
    """
    text = (name or "").strip()
    if not text:
        return UNKNOWN
    if any(keyword in text for keyword in INDOOR_KEYWORDS):
        return INDOOR
    # Hotels and restaurants are roofed by definition; meals are not lost to rain.
    if place_type in {"hotel", "restaurant", "airport", "station"}:
        return INDOOR
    if any(keyword in text for keyword in OUTDOOR_KEYWORDS):
        return OUTDOOR
    if place_type == "attraction":
        # An attraction with no outdoor hint is left undecided rather than guessed.
        return UNKNOWN
    return UNKNOWN


def _items_by_date(daily_plan: list[dict[str, Any]] | None) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for day in daily_plan or []:
        if not isinstance(day, dict):
            continue
        day_date = str(day.get("date") or "")
        names = []
        for item in day.get("items") or []:
            if isinstance(item, dict):
                name = str(item.get("place_name") or "").strip()
                if name:
                    names.append(name)
        if day_date:
            grouped[day_date] = names
    return grouped


def build_weather_advice(
    destination: str,
    weather: dict[str, Any] | None,
    daily_plan: list[dict[str, Any]] | None,
    places: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """One actionable entry per day whose weather threatens the plan.

    Days that are fine produce no entry, so an empty list means "nothing to worry
    about" rather than "no information".
    """
    if not weather or weather.get("status") != "completed":
        return []

    place_types = {str(place.get("name") or ""): place.get("type") for place in places or []}
    scheduled_by_date = _items_by_date(daily_plan)

    advice: list[dict[str, Any]] = []
    for day in weather.get("days") or []:
        if not isinstance(day, dict) or not day.get("indoor_priority"):
            continue
        day_date = str(day.get("date") or "")
        scheduled = scheduled_by_date.get(day_date, [])
        outdoor = [
            name
            for name in scheduled
            if classify_place(name, place_types.get(name)) == OUTDOOR
        ]
        indoor = [
            name
            for name in scheduled
            if classify_place(name, place_types.get(name)) == INDOOR
        ]
        precipitation = day.get("precipitation_probability")
        condition = day.get("condition") or "降雨或恶劣天气"

        parts = [f"{day_date} 预计{condition}"]
        if precipitation is not None:
            parts[0] += f"（降水概率 {precipitation}%）"
        if not scheduled:
            parts.append("当天还没有安排行程，建议优先选择室内景点，并留意临近出发的天气更新。")
            suggestion = "把室内的博物馆、美术馆或商场安排在这一天，户外的行程留到天气更好的日期。"
        elif outdoor:
            parts.append(f"当天已安排：{'、'.join(scheduled)}。其中 {'、'.join(outdoor)} 偏户外，雨天体验会明显受影响。")
            suggestion = (
                f"把 {'、'.join(outdoor)} 换到天气较好的日期，当天改为{('、'.join(indoor) + ' 等室内安排') if indoor else '室内的博物馆、美术馆或商场'}。"
            )
        else:
            parts.append(f"当天已安排：{'、'.join(scheduled)}，看起来都在室内，整体不受影响。")
            suggestion = "当天安排基本不受影响，出发前再确认一次天气即可。"

        advice.append(
            {
                "date": day_date,
                "condition": condition,
                "precipitation_probability": precipitation,
                "indoor_priority": True,
                "scheduled": scheduled,
                "outdoor_items": outdoor,
                "indoor_items": indoor,
                "message": "".join(parts),
                "suggestion": suggestion,
            }
        )
    return advice


def weather_adjustment_request(destination: str, advice: list[dict[str, Any]]) -> str:
    """A ready-to-send request for the conversational adjustment agent."""
    if not advice:
        return ""
    lines = [f"请根据天气调整我在{destination}的行程："]
    for entry in advice:
        lines.append(f"- {entry['date']}：{entry['message']}")
        lines.append(f"  建议：{entry['suggestion']}")
    lines.append("请保持其他日期的安排不变，并重新计算预算。")
    return "\n".join(lines)
