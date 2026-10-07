"""Put saved places on the calendar without the user typing every field.

Adding a saved place to a day meant opening a dialog and filling in the place, the
date, and both times by hand — even though the app already knows the place's opening
hours and what else is booked that day. This proposes a schedule that respects both.
"""

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
import re
from typing import Any

# The day is planned between these hours. A place whose own opening hours fall
# outside this span (an evening-only river cruise, say) is still scheduled inside
# its own window rather than being rejected.
DAY_START = time(9, 0)
DAY_END = time(21, 0)

# How long each kind of place is expected to take.
DURATIONS: dict[str, timedelta] = {
    "attraction": timedelta(hours=2),
    "restaurant": timedelta(hours=1, minutes=30),
    "hotel": timedelta(hours=1),
    "airport": timedelta(hours=1),
    "station": timedelta(hours=1),
}
DEFAULT_DURATION = timedelta(hours=2)

# Used when a place has no usable opening hours.
DEFAULT_WINDOW = (DAY_START, DAY_END)

_TIME_RANGE = re.compile(
    r"(\d{1,2})\s*[:：]\s*(\d{2})\s*(?:-|–|—|~|～|至|到)\s*(\d{1,2})\s*[:：]\s*(\d{2})"
)


@dataclass
class ProposedItem:
    place_id: int
    place_name: str
    date: str
    start_time: str
    end_time: str
    note: str


@dataclass
class ScheduleProposal:
    proposed: list[ProposedItem] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    already_scheduled: list[str] = field(default_factory=list)


def parse_opening_window(opening_hours: str | None) -> tuple[time, time] | None:
    """Read "09:00-17:00" (and its variants) as a usable time window.

    Returns None when nothing sensible can be read, so the caller falls back to the
    default day window instead of refusing to schedule the place at all.
    """
    if not opening_hours:
        return None
    match = _TIME_RANGE.search(str(opening_hours))
    if match is None:
        return None
    try:
        start = time(int(match.group(1)), int(match.group(2)))
        end = time(int(match.group(3)), int(match.group(4)))
    except ValueError:
        return None
    if end <= start:
        # "22:00-02:00" style spans cross midnight; the day window cannot model them.
        return None
    return start, end


def _duration(place_type: str | None) -> timedelta:
    return DURATIONS.get(str(place_type or ""), DEFAULT_DURATION)


def _trip_days(start_date: str, end_date: str) -> list[str]:
    try:
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
    except (TypeError, ValueError):
        return []
    if end < start:
        return []
    return [(start + timedelta(days=offset)).isoformat() for offset in range((end - start).days + 1)]


def _busy_blocks(existing_items: list[dict[str, Any]]) -> dict[str, list[tuple[time, time]]]:
    busy: dict[str, list[tuple[time, time]]] = {}
    for item in existing_items:
        item_date = item.get("date")
        day = item_date.isoformat() if hasattr(item_date, "isoformat") else str(item_date or "")
        start = item.get("start_time")
        end = item.get("end_time")
        if not day or not isinstance(start, time) or not isinstance(end, time):
            continue
        busy.setdefault(day, []).append((start, end))
    for blocks in busy.values():
        blocks.sort()
    return busy


def _fits(candidate_start: time, duration: timedelta, busy: list[tuple[time, time]]) -> bool:
    candidate_end = _add(candidate_start, duration)
    for busy_start, busy_end in busy:
        if candidate_start < busy_end and busy_start < candidate_end:
            return False
    return True


def _add(value: time, delta: timedelta) -> time:
    # Anchored on a datetime: adding a timedelta to a date yields a date, which
    # silently loses the clock time.
    anchor = datetime(2000, 1, 1)
    return (anchor + timedelta(hours=value.hour, minutes=value.minute) + delta).time()


def _planning_window(opening: tuple[time, time] | None) -> tuple[time, time]:
    """The span a place may be scheduled in: its opening hours, clamped to the day.

    An evening-only opening (19:00-22:00) sits outside the normal 09:00-21:00
    planning span, so it is honoured as-is instead of dropping the place.
    """
    if opening is None:
        return DEFAULT_WINDOW
    window_start, window_end = opening
    lower = max(DAY_START, window_start)
    upper = min(DAY_END, window_end)
    if upper <= lower:
        return window_start, window_end
    return lower, upper


def _format(value: time) -> str:
    return value.strftime("%H:%M:%S")


def _preferred_day(place: dict[str, Any], days: list[str]) -> str | None:
    """The day the recommendation panel suggested, when it is a real day of the trip."""
    raw = place.get("preferred_date")
    if raw is None:
        return None
    value = raw.isoformat() if hasattr(raw, "isoformat") else str(raw or "")
    return value if value in days else None


def _slot_for_day(
    day: str,
    duration: timedelta,
    window_start: time,
    window_end: time,
    busy: dict[str, list[tuple[time, time]]],
    step_minutes: int,
) -> tuple[time, time] | None:
    """The first free slot on one day, or None when the day cannot take the place."""
    day_busy = list(busy.get(day, []))
    cursor = window_start
    while True:
        candidate_end = _add(cursor, duration)
        if candidate_end > window_end:
            return None
        if _fits(cursor, duration, day_busy):
            day_busy.append((cursor, candidate_end))
            day_busy.sort()
            # Publish the new booking, otherwise the next place searches the original
            # free time and is proposed the very same slot.
            busy[day] = day_busy
            return cursor, candidate_end
        cursor = _add(cursor, timedelta(minutes=step_minutes))


def propose_schedule(
    places: list[dict[str, Any]],
    existing_items: list[dict[str, Any]],
    start_date: str,
    end_date: str,
    step_minutes: int = 30,
) -> ScheduleProposal:
    """Propose a date and time slot for every saved place that has none yet.

    Places are handled in the order they were saved. A place that carries a
    `preferred_date` — the day it was suggested for in the recommendation panel — is
    tried on that day first, so ticking an item inside a day group keeps it there.
    Only when that day has no room does it fall back to the earliest day that fits,
    and the note says the day was changed. A place that cannot fit anywhere is
    reported instead of being silently dropped.
    """
    proposal = ScheduleProposal()
    days = _trip_days(start_date, end_date)
    if not days:
        return proposal

    busy = _busy_blocks(existing_items)
    scheduled_ids = {item.get("place_id") for item in existing_items}

    for place in places:
        place_id = place.get("id")
        name = str(place.get("name") or "").strip()
        if place_id is None or not name:
            continue
        if place_id in scheduled_ids:
            proposal.already_scheduled.append(name)
            continue

        duration = _duration(place.get("type"))
        opening = parse_opening_window(place.get("opening_hours"))
        window_start, window_end = _planning_window(opening)
        opening_note = (
            f"按开放时间 {_format(window_start)[:5]}-{_format(window_end)[:5]} 自动安排"
            if opening
            else "自动安排（无开放时间信息）"
        )

        # The suggested day goes first; the remaining days keep their normal order so a
        # crowded preferred day still yields a usable proposal.
        preferred = _preferred_day(place, days)
        search_order = ([preferred] if preferred else []) + [day for day in days if day != preferred]

        placed_day: str | None = None
        placed_slot: tuple[time, time] | None = None
        for day in search_order:
            slot = _slot_for_day(day, duration, window_start, window_end, busy, step_minutes)
            if slot is not None:
                placed_day, placed_slot = day, slot
                break

        if placed_day is not None and placed_slot is not None:
            if preferred and placed_day != preferred:
                note = f"{opening_note}；建议的 {preferred} 已排满，改排到这一天"
            elif preferred:
                note = f"{opening_note}；按规划建议放在这一天"
            else:
                note = opening_note
            proposal.proposed.append(
                ProposedItem(
                    place_id=int(place_id),
                    place_name=name,
                    date=placed_day,
                    start_time=_format(placed_slot[0]),
                    end_time=_format(placed_slot[1]),
                    note=note,
                )
            )
            continue

        proposal.skipped.append(
            {
                "place_id": int(place_id),
                "place_name": name,
                "reason": (
                    f"这几天在开放时间（{_format(window_start)[:5]}-{_format(window_end)[:5]}）内"
                    "没有足够空闲时段，请手动安排或减少同时段的项目。"
                    if opening
                    else "这几天没有足够空闲时段，请手动安排或减少同时段的项目。"
                ),
            }
        )

    return proposal
