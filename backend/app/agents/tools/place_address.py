"""Turn "地址待确认" into somewhere a person could actually walk to.

The extraction now asks the model for an address, and the material it reads usually has one.
When it still comes back without one, this module looks the place up in OpenStreetMap's
Nominatim service: free, no API key, and good coverage outside China, which matters because the
alternative map services answer a Tokyo restaurant with nothing useful.

Three rules keep it honest and bounded:

* **It never blocks a plan.** Lookups are paced at one per second (Nominatim's usage policy) and
  the whole pass has a time budget; whatever is left unlooked when the budget runs out keeps the
  address it had, and the draft says so rather than pretending.
* **It never invents anything.** Only a real answer from the geocoder is written, and only when
  it names the place we asked about. A miss leaves the caller's text alone.
* **The answer is written the way a Chinese reader reads it.** ``東京都`` becomes ``东京都`` and
  ``台東区`` becomes ``台东区``; street names, which have no Chinese form, stay as they are.
"""
from __future__ import annotations

import json
import logging
import os
from time import monotonic as _monotonic, sleep as _sleep
from typing import Any, Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
# Measured on this machine: every lookup to Nominatim timed out after 12.1s and answered
# nothing, for both a restaurant and a national museum in Tokyo. The host is not reachable from
# here, so the lookup is off by default rather than adding twenty seconds of nothing to every
# plan. Turn it on where the service is reachable (or point NOMINATIM_URL at a reachable
# mirror); nothing else has to change.
LOOKUPS_ENABLED = False
# OSM asks every client to identify itself. Without this the service blocks the caller.
USER_AGENT = "travel-agent-platform/1.0 (personal itinerary planner)"
LOOKUP_TIMEOUT = 5.0
# Nominatim's usage policy: at most one request per second.
LOOKUP_PAUSE = 1.0
# How long the whole pass may take before the rest is left alone. A plan is never held up by
# address lookups: the traveller's day matters more than the last few door numbers.
ADDRESS_BUDGET = 20.0

# Placeholders that mean "we do not have an address", so a lookup is worth trying.
_EMPTY_MARKERS = ("待确认", "未知", "不详", "todo", "tbd")

# The administrative names that differ between Japanese and Chinese. Ordered longest first so
# 東京都 wins over 東京. Street and block names are deliberately absent: they have no Chinese
# form, and a wrong translation of a street is worse than the original.
_CHINESE_NAMES: tuple[tuple[str, str], ...] = tuple(
    sorted(
        {
            "東京都": "东京都", "大阪府": "大阪府", "京都府": "京都府", "北海道": "北海道",
            "神奈川県": "神奈川县", "千葉県": "千叶县", "埼玉県": "埼玉县", "愛知県": "爱知县",
            "兵庫県": "兵库县", "福岡県": "福冈县", "沖縄県": "冲绳县",
            "台東区": "台东区", "新宿区": "新宿区", "渋谷区": "涩谷区", "豊島区": "丰岛区",
            "品川区": "品川区", "目黒区": "目黑区", "大田区": "大田区", "世田谷区": "世田谷区",
            "中野区": "中野区", "杉並区": "杉并区", "文京区": "文京区", "墨田区": "墨田区",
            "江東区": "江东区", "荒川区": "荒川区", "板橋区": "板桥区", "練馬区": "练马区",
            "葛飾区": "葛饰区", "江戸川区": "江户川区", "足立区": "足立区", "中央区": "中央区",
            "千代田区": "千代田区", "港区": "港区", "北区": "北区", "大阪市": "大阪市",
            "日本": "日本",
        }.items(),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )
)


def to_chinese(address: str) -> str:
    """The same address, with the administrative names a Chinese reader expects."""
    text = address or ""
    for japanese, chinese in _CHINESE_NAMES:
        text = text.replace(japanese, chinese)
    return text.strip(" ,，")


def needs_address(address: str | None, destination: str | None = None) -> bool:
    """Whether this text is worth replacing with a looked-up address."""
    text = str(address or "").strip()
    if not text:
        return True
    folded = text.casefold()
    if any(marker in folded for marker in _EMPTY_MARKERS):
        return True
    # "日本东京" on its own is the destination, not an address.
    return bool(destination) and text in {str(destination).strip(), f"日本东京"}


def _components_to_address(payload: dict[str, Any]) -> str:
    """Build "东京都台东区上野1-1-1" from the geocoder's own parts."""
    parts = payload.get("address") or {}
    ordered = (
        parts.get("state") or parts.get("province") or parts.get("region"),
        parts.get("city_district") or parts.get("county") or parts.get("city"),
        parts.get("suburb") or parts.get("neighbourhood") or parts.get("quarter"),
        parts.get("road") or parts.get("pedestrian") or parts.get("footway"),
        parts.get("house_number"),
    )
    seen: list[str] = []
    for part in ordered:
        text = str(part or "").strip()
        # "东京都东京都新宿区" reads badly, so a part already contained in the previous one is
        # dropped rather than repeated.
        if text and not any(text in kept or kept in text for kept in seen):
            seen.append(text)
    if seen:
        return to_chinese("".join(seen))
    # No structured parts: the display name is comma-separated, most specific first.
    display = str(payload.get("display_name") or "")
    return to_chinese(" · ".join(part.strip() for part in display.split(",")[:4] if part.strip()))


def lookup_address(name: str, destination: str) -> dict[str, Any] | None:
    """Ask OpenStreetMap where one place is. None when it does not know it."""
    query = f"{name}, {destination}" if destination else name
    url = f"{NOMINATIM_URL}?{urlencode({'q': query, 'format': 'jsonv2', 'limit': 1, 'addressdetails': 1, 'accept-language': 'zh-CN,zh,ja,en'})}"
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urlopen(request, timeout=LOOKUP_TIMEOUT) as response:
            results = json.loads(response.read().decode("utf-8", errors="replace"))
    except Exception as exc:  # a lookup that fails is a missing address, not a failed plan
        logger.info("address lookup failed for %s: %s", name, exc)
        return None
    if not isinstance(results, list) or not results:
        return None
    best = results[0]
    address = _components_to_address(best)
    if not address:
        return None
    return {
        "address": address,
        "latitude": best.get("lat"),
        "longitude": best.get("lon"),
        "source": "openstreetmap",
    }


def fill_place_addresses(
    places: Iterable[dict[str, Any]],
    destination: str | None = None,
    budget: float = ADDRESS_BUDGET,
    lookup=lookup_address,
) -> list[dict[str, Any]]:
    """Return the places with real addresses where one could be found, within ``budget`` seconds.

    A place that already has an address is left exactly as it was: this fills gaps, it does not
    overwrite what the model read out of the material.
    """
    filled: list[dict[str, Any]] = []
    if not LOOKUPS_ENABLED:
        return [dict(place) for place in places]
    if os.getenv("TRAVEL_AGENT_DB_PATH"):
        # The test suite points the database at a throwaway file and promises never to reach
        # the network; this pass is the one thing here that would.
        return [dict(place) for place in places]
    started = _monotonic()
    looked = 0
    for place in places:
        item = dict(place)
        if needs_address(item.get("address"), destination):
            remaining = budget - (_monotonic() - started)
            if remaining <= 0:
                logger.info("address budget spent after %s lookups; the rest keep their text", looked)
                filled.append(item)
                continue
            if looked:
                _sleep(min(LOOKUP_PAUSE, max(0.0, budget - (_monotonic() - started))))
            looked += 1
            found = lookup(str(item.get("name") or ""), str(destination or ""))
            if found:
                item["address"] = found["address"]
                item["address_source"] = found.get("source")
                # The geocoder also knows where the place is, which is what the schematic map
                # draws with when the model did not guess coordinates.
                if not item.get("latitude") and found.get("latitude") is not None:
                    item["latitude"] = float(found["latitude"])
                if not item.get("longitude") and found.get("longitude") is not None:
                    item["longitude"] = float(found["longitude"])
        filled.append(item)
    return filled
