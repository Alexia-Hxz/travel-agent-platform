from dataclasses import dataclass
from datetime import date, time
from typing import Any


@dataclass(frozen=True)
class ScheduleConflict:
    date: date
    first_place: str
    second_place: str
    first_start: time
    first_end: time
    second_start: time
    second_end: time
    message: str


def detect_conflicts(items: list[dict[str, Any]]) -> list[ScheduleConflict]:
    """Return overlapping items on the same day without mutating the itinerary."""
    conflicts: list[ScheduleConflict] = []
    grouped: dict[date, list[dict[str, Any]]] = {}
    for item in items:
        grouped.setdefault(item["date"], []).append(item)

    for day, day_items in grouped.items():
        ordered = sorted(day_items, key=lambda item: (item["start_time"], item["end_time"]))
        for index, first in enumerate(ordered):
            for second in ordered[index + 1 :]:
                if second["start_time"] >= first["end_time"]:
                    break
                if second["end_time"] <= first["start_time"]:
                    continue
                first_name = str(first.get("place_name", first.get("place_id", "地点")))
                second_name = str(second.get("place_name", second.get("place_id", "地点")))
                conflicts.append(
                    ScheduleConflict(
                        date=day,
                        first_place=first_name,
                        second_place=second_name,
                        first_start=first["start_time"],
                        first_end=first["end_time"],
                        second_start=second["start_time"],
                        second_end=second["end_time"],
                        message=f"{first_name} 与 {second_name} 的时间安排重叠。",
                    )
                )
    return conflicts
