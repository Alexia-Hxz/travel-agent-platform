"""Resolve what a traveller typed as their destination into a place a service can use.

The problem this solves
-----------------------
``weather_service`` used to hand the destination straight to Open-Meteo's geocoder. Measured
against the live service (see ``destination_data``), that works for "南京" and "Tokyo" and
fails badly for almost everything else a Chinese traveller types:

* **nothing found** for "日本东京", "日本 东京", "东京都", "大阪", "日本大阪", "北海道",
  "韩国首尔", "首尔", "泰国曼谷", "法国巴黎", "纽约", "雷克雅未克", "冰岛雷克雅未克",
  "伊斯坦布尔", "巴厘岛", "普吉岛", "胡志明市", "台北";
* **the wrong place entirely** for "东京" (a village in Jiangsu), "伦敦" (Ontario),
  "罗马" (Queensland), "米兰" (Ohio), "开罗" (Illinois), "河内" (Guangdong), "奥克兰" (California)
  and even for the English "New York" (a village in Nebraska) and "Bali" (a village in India).

So the destination is resolved here, in this order:

1. **the measured table** (``destination_data``): the spelling that the geocoder answers
   correctly, with the country it must be in;
2. **country prefix plus city**: "日本东京" -> strip 日本 -> 东京 -> Tokyo;
3. **country only**: "日本" -> its capital, because a country names no weather station;
4. **the live geocoder** for anything the table does not know, but only accepting an answer
   whose country matches the country the user named -- which is what turns a confident wrong
   answer into no answer.

Everything here except :func:`geocode_candidates` / :func:`choose_result` is offline and pure.
"""

from __future__ import annotations

import re

from app.services.destination_data import CITY_ROWS, COUNTRY_PREFIXES, DESTINATION_ALIASES

# Punctuation people put between place names, and the separators inside "东京, 日本".
_SEPARATORS = re.compile(r"[\s,，、/\\|·・—\-–~]+")
# Administrative suffixes that can be dropped when what is left is a place we know.
_SUFFIXES = ("特别市", "自治市", "都", "市", "府", "县", "区", "省", "州", "町", "村")

# The geocoder writes some countries differently from the way people do.
_COUNTRY_SYNONYMS = {
    "中国台湾": "台湾",
    "台灣": "台湾",
    "中国香港": "香港",
    "中国澳门": "澳门",
    "美利坚合众国": "美国",
    "大韩民国": "韩国",
    "日本国": "日本",
    "阿拉伯联合酋长国": "阿拉伯联合酋长国",
    "阿联酋": "阿拉伯联合酋长国",
    "英国": "英国",
    "大不列颠及北爱尔兰联合王国": "英国",
}

# City results outrank region results when the country check cannot tell them apart.
_FEATURE_RANK = {"PPLC": 0, "PPLA": 1, "PPLA2": 2, "PPLA3": 3, "PPL": 4}


def fold(value: str | None) -> str:
    """Lower-case and drop the spacing/separators that make two spellings look different."""
    return _SEPARATORS.sub("", str(value or "").strip()).casefold()


# Keys and aliases are matched without spacing, because "New York"/"new york"/"newyork" are the
# same destination to a traveller and three different strings to a dict.
_KEY_INDEX: dict[str, str] = {fold(key): key for key in CITY_ROWS}
_ALIAS_INDEX: dict[str, str] = {fold(name): key for name, key in DESTINATION_ALIASES.items()}


def normalize_country(value: str | None) -> str:
    text = str(value or "").strip()
    return _COUNTRY_SYNONYMS.get(text, text)


def _row(key: str, *, country_only: bool = False, source: str = "table") -> dict | None:
    entry = CITY_ROWS.get(key)
    if entry is None:
        return None
    latitude, longitude, label, country, geocoder_name = entry
    return {
        "label": label,
        "country": country,
        "latitude": latitude,
        "longitude": longitude,
        "geocoder_name": geocoder_name,
        "country_only": country_only,
        "source": source,
    }


def _keys_of(text: str) -> list[str]:
    """Every key this text could be: as written, without a suffix, and each of its parts."""
    candidates: list[str] = []

    def _add(value: str) -> None:
        folded = fold(value)
        if folded and folded not in candidates:
            candidates.append(folded)

    _add(text)
    for part in _SEPARATORS.split(str(text or "").strip()):
        _add(part)
    for base in list(candidates):
        for suffix in _SUFFIXES:
            if base.endswith(suffix) and len(base) > len(suffix):
                _add(base[: -len(suffix)])
    return candidates


def city_key(text: str | None) -> str | None:
    """The table key for a city name, or None when the table does not know it.

    Both the keys and the aliases are indexed without spacing, so "New York", "new york" and
    "newyork" all reach the same row, and an alias that does not point at a real row is ignored
    rather than silently resolving to nothing.
    """
    for key in _keys_of(str(text or "")):
        if key in _KEY_INDEX:
            return _KEY_INDEX[key]
    for key in _keys_of(str(text or "")):
        target = _ALIAS_INDEX.get(key)
        if target:
            resolved = _KEY_INDEX.get(fold(target))
            if resolved:
                return resolved
    return None


def strip_country_prefix(name: str | None) -> tuple[str, str | None]:
    """Split "日本东京" into ("东京", "日本"), leaving the text alone when it is not prefixed.

    Longest prefix first, so "阿拉伯联合酋长国迪拜" loses the whole country name and not "阿联酋".
    """
    text = str(name or "").strip()
    folded = fold(text)
    best: str | None = None
    for prefix in COUNTRY_PREFIXES:
        folded_prefix = fold(prefix)
        if folded.startswith(folded_prefix) and len(folded) > len(folded_prefix):
            if best is None or len(folded_prefix) > len(fold(prefix)):
                best = prefix
    if best is None:
        return text, None
    # Cut the same number of characters, ignoring the separators inside both strings.
    consumed_source = 0
    consumed_folded = 0
    folded_prefix = fold(best)
    for index, character in enumerate(text):
        if fold(character):
            consumed_folded += 1
        consumed_source = index + 1
        if consumed_folded == len(folded_prefix):
            break
    rest = text[consumed_source:].lstrip(" ,，、/-|")
    return rest, best


def country_only_key(name: str | None) -> str | None:
    """When the destination is nothing but a country, the capital is the place to forecast."""
    folded = fold(name)
    for prefix, (_country, capital) in COUNTRY_PREFIXES.items():
        if fold(prefix) == folded:
            return capital
    return None


def expected_country(name: str | None) -> str | None:
    """The country the user named, if they named one."""
    _rest, prefix = strip_country_prefix(name)
    if prefix is None:
        return None
    return COUNTRY_PREFIXES[prefix][0]


def resolve_offline(destination: str | None) -> dict | None:
    """Resolve without any network call, or return None when the table cannot."""
    text = str(destination or "").strip()
    if not text:
        return None

    capital = country_only_key(text)
    if capital is not None:
        return _row(capital, country_only=True)

    rest, prefix = strip_country_prefix(text)
    key = city_key(rest) or city_key(text)
    if key is not None:
        return _row(key)

    # A city the table does not know, inside a country it does: let the geocoder try the
    # city part, and hold it to that country.
    return None


def geocode_candidates(destination: str | None) -> list[str]:
    """The spellings to try against the live geocoder, best first."""
    text = str(destination or "").strip()
    if not text:
        return []
    rest, _prefix = strip_country_prefix(text)
    keys = _keys_of(rest) or _keys_of(text)
    candidates: list[str] = []
    for key in [rest, *keys, text]:
        value = str(key or "").strip()
        if value and value not in candidates:
            candidates.append(value)
    # A Chinese label the table knows is a better geocoder query than the name as typed.
    for key in list(candidates):
        alias = _ALIAS_INDEX.get(fold(key))
        entry = CITY_ROWS.get(alias) if alias else None
        if entry and entry[4] not in candidates:
            candidates.append(entry[4])
    return candidates[:4]


def _result_rank(item: dict) -> tuple[int, int]:
    """Biggest place first, then the most city-like.

    Population is what separates a real city from a same-country namesake: the geocoder's first
    answer for "New York" is a village in Nebraska with 7,864 people, and its first answer for
    the English "London" is in Ontario.
    """
    try:
        population = int(item.get("population") or 0)
    except (TypeError, ValueError):
        population = 0
    return (-population, _FEATURE_RANK.get(str(item.get("feature_code") or ""), 9))


def choose_result(results: list[dict], country: str | None) -> dict | None:
    """Pick the geocoder result to trust: the right country, and the biggest city in it."""
    expected = normalize_country(country) if country else None
    if expected is not None:
        matches = [item for item in results if normalize_country(item.get("country")) == expected]
    else:
        matches = list(results)
    if not matches:
        return None
    matches.sort(key=_result_rank)
    best = matches[0]
    try:
        latitude = round(float(best["latitude"]), 5)
        longitude = round(float(best["longitude"]), 5)
    except (KeyError, TypeError, ValueError):
        return None
    name = str(best.get("name") or "").strip()
    return {
        "label": name or str(best.get("admin1") or "").strip(),
        "country": str(best.get("country") or "").strip(),
        "latitude": latitude,
        "longitude": longitude,
        "geocoder_name": name,
        "country_only": False,
        "source": "geocoder",
    }
