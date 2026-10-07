"""Photos from the Pexels library: the only image source this project has.

Pexels is free, its photos are real, and it is reachable, but it is a **stock library**: a
query for one specific museum comes back with whatever the photographers titled that way.
Measured on a Tokyo plan with the code's own Chinese queries, three of eight museums were
illustrated with the same photo of Osaka Castle, a tonkotsu ramen query returned
"snow-covered horse saddles", and yakitori returned "street vendor cooking peking duck in
Beijing". Two things fix that, and both live here:

* **English keywords.** The library's index is English, so ``alt`` text and page slugs are
  English too; a Chinese query is matched loosely and lands anywhere.
* **Scoring on ``alt``/slug.** Pexels returns a described ``alt`` for every photo
  ("delicious ramen bowl in tokyo booth"). Keeping it is what lets a caller check that a
  candidate is on topic instead of taking the first result on trust.

Pexels requires attribution, so every photo is returned together with its photographer and
photo page, and ``ATTRIBUTION_LABEL`` is what the interface must show next to them. That is
a condition of using the API, not a nicety.
"""

import logging
import os
from pathlib import Path
import re
import threading
from typing import Any

import httpx
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

ENV_PATH = Path(__file__).resolve().parents[3] / ".env"
SEARCH_URL = "https://api.pexels.com/v1/search"
SEARCH_TIMEOUT = 8.0
# The API's maximum page size. A city-wide query fills it; a query for one place does not.
POOL_LIMIT = 80
# A recommendation card is wide and short, so portrait photos are excluded up front.
ORIENTATION = "landscape"
# Shown verbatim in the interface, which is how this project meets Pexels' attribution
# requirement ("show a prominent link to Pexels").
ATTRIBUTION_LABEL = "Photos provided by Pexels"
ATTRIBUTION_URL = "https://www.pexels.com"
# Card-sized renditions, best first. `original` is deliberately absent: it can be many
# megabytes and the cards render a few hundred pixels wide.
_SIZE_ORDER = ("large", "landscape", "medium", "small")

_pool_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
_pool_lock = threading.Lock()

# "Photo by X on Pexels", the library's own caption, and the same shape as the slug.
_PEXELS_SLUG = re.compile(r"/photo/([^/?#]+)")


def api_key() -> str | None:
    """Return the configured Pexels key, or None. The value is never logged."""
    load_dotenv(ENV_PATH)
    value = (os.getenv("PEXELS_API_KEY") or "").strip()
    return value or None


def pexels_available() -> bool:
    return api_key() is not None


def _best_url(item: dict[str, Any]) -> str | None:
    """Pick a card-sized rendition from one search result."""
    sources = item.get("src")
    if not isinstance(sources, dict):
        return None
    for size in _SIZE_ORDER:
        value = str(sources.get(size) or "").strip()
        if value.startswith(("http://", "https://")):
            return value
    return None


def page_slug(page_url: str | None) -> str:
    """The photographer's own words from a photo page URL.

    ``https://www.pexels.com/photo/delicious-ramen-bowl-in-tokyo-booth-37014726/`` becomes
    ``delicious ramen bowl in tokyo booth``: the same descriptive text as ``alt``, and the
    signal that lets a caller verify a candidate before showing it.
    """
    match = _PEXELS_SLUG.search(str(page_url or ""))
    if match is None:
        return ""
    slug = match.group(1)
    # Trailing "-37014726" is the photo id, not a word.
    head, _, tail = slug.rpartition("-")
    if head and tail.isdigit():
        slug = head
    return slug.replace("-", " ").strip().lower()


def search_pexels_photos(
    query: str, limit: int = 3, timeout: float = SEARCH_TIMEOUT
) -> list[dict[str, Any]]:
    """Return photo records for one query; never raises.

    Each record carries ``url``, the attribution fields, and the descriptive ``alt`` text
    (falling back to the page slug) that a caller needs to judge relevance without a second
    request.
    """
    key = api_key()
    if not key:
        return []
    try:
        response = httpx.get(
            SEARCH_URL,
            params={"query": query, "per_page": max(1, limit), "orientation": ORIENTATION},
            # Pexels authenticates with a raw key in this header, not a bearer token.
            headers={"Authorization": key},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError, TypeError) as error:
        logger.warning("pexels photo search failed: %s", error)
        return []
    if not isinstance(payload, dict):
        return []
    photos: list[dict[str, Any]] = []
    for item in payload.get("photos", []):
        if not isinstance(item, dict):
            continue
        url = _best_url(item)
        if not url:
            continue
        page_url = str(item.get("url") or "").strip() or None
        alt = " ".join(str(item.get("alt") or "").split())
        photos.append(
            {
                "url": url,
                "photographer": str(item.get("photographer") or "").strip() or None,
                "photographer_url": str(item.get("photographer_url") or "").strip() or None,
                "page_url": page_url,
                "alt": alt or page_slug(page_url),
                "source": "pexels",
            }
        )
    return photos


def credit_line(photo: dict[str, Any]) -> str | None:
    """Build "Photo by <name> on Pexels", or None when the name is missing."""
    photographer = str(photo.get("photographer") or "").strip()
    if not photographer:
        return None
    return f"Photo by {photographer} on Pexels"


def city_pool_photos(destination: str, hint: str) -> list[dict[str, Any]]:
    """A deep page for one destination + category, fetched once per process.

    A query for one specific place returns a handful of results that overlap almost
    completely with the next place's query, so "one photo per card" cannot be satisfied
    from those pages alone: measured, 21 different restaurant queries between them offered
    only 11 distinct photos, and the other 10 cards fell back to a placeholder. One
    city-wide query returns the full 80-photo page, which is plenty of room to keep handing
    out different pictures.
    """
    query = f"{destination} {hint}".strip()
    if not query:
        return []
    key = (destination, hint)
    with _pool_lock:
        cached = _pool_cache.get(key)
    if cached is not None:
        return cached
    photos = search_pexels_photos(query, limit=POOL_LIMIT)
    # Only remember a real page. Caching an empty list would pin a transient failure (or an
    # exhausted quota) onto this city for the lifetime of the process.
    if photos:
        with _pool_lock:
            _pool_cache[key] = photos
    return photos


def clear_pool_cache() -> None:  # noqa: D401 - see below
    """Forget cached city pools. Intended for tests."""
    with _pool_lock:
        _pool_cache.clear()
