"""Guarantee a picture that matches the card for every recommended place.

The goal is that the picture **matches what the card says** -- a ramen card shows ramen, a
museum card shows a museum in that city -- not that it is a photograph of that exact venue.
No reachable free source has photos of particular venues: the search provider returns no
image URLs at all, Wikimedia is unreachable from this network, and a stock library only has
what its photographers happened to shoot. Measured with the code's own queries, "matching"
was badly broken: three of eight Tokyo museums were illustrated with a photo of Osaka
Castle, a tonkotsu ramen query returned snow-covered horse saddles, and yakitori returned
Peking duck in Beijing.

Resolution order for each place:

1. an image the place already carries (a saved place or an attached result image)
2. the English keyword for this place (an English place name, or the dish's English name)
   searched in the photo library, with candidates **verified against their own description**
3. for food, the dish library (TheMealDB), which covers dishes the stock library misses
4. the destination's own category page, which is at least the right city and category
5. a bundled placeholder served under ``/static`` -- cuisine-specific for a food card

Every place ends up with a non-empty image URL, which is what the planning result promises
to the client. Lookups are cached, bounded and concurrent; this module never raises.
"""

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any
from urllib.parse import urlparse
import re
import socket
import threading
import time

import httpx

from app.agents.tools import mealdb_images, pexels_images
from app.agents.tools.pexels_images import search_pexels_photos


STATIC_IMAGE_PREFIX = "/static/place_images"
CATEGORY_IMAGES = {
    "hotel": f"{STATIC_IMAGE_PREFIX}/hotel.svg",
    "restaurant": f"{STATIC_IMAGE_PREFIX}/restaurant.svg",
    "attraction": f"{STATIC_IMAGE_PREFIX}/attraction.svg",
}
GENERIC_IMAGE = f"{STATIC_IMAGE_PREFIX}/generic.svg"
# A food card's last resort is the picture of the kind of dish it is, so even the placeholder
# matches the card: "拉面" over a noodle bowl rather than over a generic knife and fork.
CUISINE_IMAGES = {
    "noodle": f"{STATIC_IMAGE_PREFIX}/restaurant_noodle.svg",
    "rice": f"{STATIC_IMAGE_PREFIX}/restaurant_rice.svg",
    "sushi": f"{STATIC_IMAGE_PREFIX}/restaurant_sushi.svg",
    "grill": f"{STATIC_IMAGE_PREFIX}/restaurant_grill.svg",
    "dessert": f"{STATIC_IMAGE_PREFIX}/restaurant_dessert.svg",
    "snack": f"{STATIC_IMAGE_PREFIX}/restaurant_snack.svg",
}
# Dish words -> the placeholder family. Chinese and English, because a dish name is Chinese
# but a dish keyword is English.
CUISINE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("noodle", ("拉面", "面", "粉", "馄饨", "饺", "そば", "noodle", "ramen", "soba", "udon", "pasta", "pho")),
    ("rice", ("饭", "丼", "粥", "盖浇", "咖喱", "rice", "don", "curry", "bibimbap", "risotto")),
    ("sushi", ("寿司", "刺身", "生鱼", "sashimi", "sushi", "maki", "nigiri")),
    ("grill", ("烧肉", "烤肉", "烧烤", "串", "牛排", "铁板", "bbq", "barbecue", "grill", "yakiniku", "yakitori", "skewer", "steak")),
    ("dessert", ("甜", "蛋糕", "冰淇淋", "布丁", "抹茶", "糕", "dessert", "cake", "ice cream", "matcha", "pudding", "sweet")),
)
CATEGORY_ALIASES = {
    "hotel": "hotel",
    "hotels": "hotel",
    "lodging": "hotel",
    "stay": "hotel",
    "accommodation": "hotel",
    "restaurant": "restaurant",
    "food": "restaurant",
    "cafe": "restaurant",
    "drink": "restaurant",
    "dining": "restaurant",
    "attraction": "attraction",
    "attractions": "attraction",
    "sight": "attraction",
    "sights": "attraction",
    "scenic": "attraction",
}
CATEGORY_QUERY_HINTS = {
    "hotel": "酒店",
    "restaurant": "美食 餐厅",
    "attraction": "景点",
}
# Enough to cover a full plan's recommendations (18 attractions + 27 restaurants + 3
# hotels is the extraction ceiling, and real 5-day plans return 34-37). This was 12 while
# every lookup cost a metered search credit; the Pexels library answers for free within a
# 20 000/month allowance, so the limit is now only about latency.
MAX_IMAGE_LOOKUPS = 40
# Used by the price-lookup path too, where each task is a model call, so it stays low.
LOOKUP_WORKERS = 4
# The picture phase is pure network I/O and was the reason a whole plan ran out of its
# wall-clock budget: measured 29 probes needing 56.9 s of work at 4 workers, so the phase
# was cut off at exactly 40.0 s with 9 of 38 cards left on a placeholder. Probing is what
# costs (about 2 s per photo), and it parallelizes cleanly.
IMAGE_LOOKUP_WORKERS = 12
# A candidate photo is only handed to the browser after this machine confirms it
# answers. Photo CDNs regularly time out or hang: an <img> pointed at a host that
# never replies shows nothing at all and may not even fire an error event, so the
# client-side fallback never runs. Verifying here turns that into a placeholder.
IMAGE_PROBE_CANDIDATES = 3
# How deep a page the Pexels library is asked for. The first place to query a city gets the
# same best-matching photo as the next one, so a deeper page is what makes it possible to
# give every card its own picture ("no two cards alike" is enforced by `_claim_photo`).
PEXELS_CANDIDATE_LIMIT = 15
IMAGE_PROBE_TIMEOUT = 4.0
# Hard wall clock for one probe. httpx's own timeout did not bound a stalling image
# host (measured 48 s with a 4 s timeout), so the caller enforces this instead.
PROBE_BUDGET = 6.0
PROBE_USER_AGENT = "Mozilla/5.0 (compatible; travel-agent-platform/1.0)"
# Hard wall-clock budget for one place's picture lookup (search + probe). A
# host whose DNS never resolves ignored httpx's own timeout and stalled a lookup
# for 112 seconds, which pushed a whole planning request past the proxy timeout.
# This budget is enforced by the caller, so nothing inside can exceed it.
#
# Raised from 25 s together with MAX_IMAGE_LOOKUPS: a whole plan's worth of lookups
# still has to finish inside it, and a measured plan makes 2.2 s of work per place at
# 4 workers, so 40 places need roughly 22 s of wall clock. The proxy allows 600 s.
#
# Raised again from 40 s when verification was added: a place may now search two queries
# (its own keyword, then the destination's category page) plus the dish library, so the
# same 34 cards cost more wall clock. Measured in the container at 9.3 s for 30 places
# before this change, so the extra room is headroom, not a requirement.
IMAGE_PHASE_BUDGET = 60.0
# How much of a candidate image must actually arrive before it counts as usable.
# Returning True on the first non-empty chunk accepted a 500 000-byte response that
# sent 64 bytes and then stalled, and the browser was handed a URL that could never
# render. A body that promises more and stops is a failure.
IMAGE_PROBE_BYTES = 65536
# A miss is remembered for this long, and no longer. The provider's ranking changes
# and a photo may appear later; caching "no picture exists" for the lifetime of the
# process kept one restaurant on a placeholder while a working photo was available.
IMAGE_MISS_TTL_SECONDS = 600.0

_image_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
# Misses are held separately with an expiry, so a hit lookup stays a plain dict.
_miss_cache: dict[tuple[str, str, str], float] = {}
_cache_lock = threading.Lock()


def normalize_category(category: Any) -> str:
    """Map a provider or model category label onto hotel/restaurant/attraction."""
    return CATEGORY_ALIASES.get(str(category or "").strip().lower(), "")


def fallback_image(category: Any, dish: Any = None) -> str:
    """Return the bundled placeholder for a category, never an empty string.

    For a food card the placeholder is chosen from the dish, so the last resort still
    matches: a ramen card falls back to a noodle bowl rather than to a generic restaurant
    icon, which is what made an unmatched dish look like a mistake.
    """
    normalized = normalize_category(category)
    if normalized == "restaurant":
        cuisine = cuisine_of(dish)
        if cuisine is not None:
            return CUISINE_IMAGES[cuisine]
    return CATEGORY_IMAGES.get(normalized, GENERIC_IMAGE)


def cuisine_of(value: Any) -> str | None:
    """Which placeholder family a dish belongs to, or None when nothing matches."""
    text = str(value or "").strip().casefold()
    if not text:
        return None
    for cuisine, keywords in CUISINE_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return cuisine
    return None


def has_usable_image(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


# Pexels indexes an English-language library, so its category word is English while the
# query above is Chinese. This matters more than it looks: without a category word a hotel
# query ("如家精选南京新街口店 南京") is answered with whatever scenery matches the city
# name, which is why hotel cards came back showing street scenes instead of rooms.
PEXELS_QUERY_HINTS = {
    "hotel": "hotel",
    # Food, not premises: a picture of a shopfront is not what a food card is for, and the
    # city-wide pool for this category is what a restaurant falls back to.
    "restaurant": "food",
    "attraction": "landmark",
}


def _pexels_query(name: str, category: str, destination: str, dish: str | None = None,
                  english: str | None = None) -> str:
    """Build the library query for one place.

    A restaurant is searched by its signature dish rather than its own name: a picture of
    the shopfront (or, worse, of a street in some other city) tells a reader nothing about
    the food, while a picture of the dish does.

    The English keyword is preferred over the Chinese name whenever there is one, because
    the library is indexed in English -- that single difference is what replaced a photo of
    Osaka Castle on three Tokyo museums with photographs of Tokyo museums. Without an
    English keyword the Chinese name is still tried, with an English category word.
    """
    if category == "restaurant":
        keyword = (english or dish or "").strip()
        if keyword:
            return f"{keyword} {PEXELS_QUERY_HINTS.get(category, '')}".strip()
        # No dish and no keyword: the venue's name would only return a shopfront or a
        # street, so there is no useful query for this place.
        return ""
    if english:
        return f"{english} {PEXELS_QUERY_HINTS.get(category, '')}".strip()
    if category == "hotel":
        parts = [name, destination, PEXELS_QUERY_HINTS.get(category, "")]
        return " ".join(part for part in parts if part).strip()
    parts = [name, destination, PEXELS_QUERY_HINTS.get(category, "")]
    return " ".join(part for part in parts if part).strip()


# Words that describe every photo in a category and so cannot tell one candidate from
# another. Scoring on them would accept any picture at all.
_STOPWORDS = frozenset(
    {
        "food", "foods", "dish", "dishes", "meal", "meals", "cuisine", "restaurant",
        "photo", "photos", "image", "images", "picture", "pictures", "the", "a", "an",
        "of", "in", "at", "on", "and", "with", "for", "to", "by", "near",
    }
)
_WORDS = re.compile(r"[a-z]+")


def _keywords(text: str | None) -> list[str]:
    """The words a candidate photo's own description must contain to be on topic."""
    words = [word for word in _WORDS.findall(str(text or "").casefold()) if len(word) > 2]
    return [word for word in words if word not in _STOPWORDS]


def _description(photo: dict[str, Any]) -> str:
    """The words Pexels itself uses for a photo: ``alt`` text plus the page slug."""
    return f"{photo.get('alt') or ''} {pexels_images.page_slug(photo.get('page_url'))}".casefold()


def _photo_score(photo: dict[str, Any], required: list[str], bonus: list[str]) -> int:
    """How well one candidate matches what the card says.

    ``required`` words are the point of the search -- the place's English name, or the dish
    -- and ``bonus`` words (the destination) merely break ties. A candidate that matches
    none of the required words scores 0 and is rejected by the caller, which is what stops
    "snow-covered horse saddles" being used for a bowl of ramen.
    """
    description = _description(photo)
    if not description:
        return 0
    score = 3 * sum(1 for word in required if word in description)
    score += sum(1 for word in bonus if word in description)
    return score


def _claim_best(
    photos: Iterable[dict[str, Any]], required: list[str], bonus: list[str], require_match: bool
) -> dict[str, Any] | None:
    """Take the best-matching unclaimed photo from this page.

    Best first, not first: the library returns its own ranking, and the best photo for the
    ranking is regularly the wrong subject (a Chinese-temple picture for a Tokyo museum).
    """
    scored: list[tuple[int, dict[str, Any]]] = []
    for photo in photos:
        url = str(photo.get("url") or "").strip()
        if not has_usable_image(url) or _skip_bad_host(url) or _url_already_failed(url):
            continue
        score = _photo_score(photo, required, bonus)
        if required and require_match and score == 0:
            continue
        scored.append((score, photo))
    # Stable sort keeps the library's own relevance order within one score.
    scored.sort(key=lambda pair: -pair[0])
    for _, photo in scored:
        url = str(photo.get("url") or "").strip()
        if not _claim_photo(url):
            continue
        if _is_reachable(url):
            return photo
        # This exact URL is done for this pass; its host is not, because the next photo on
        # a photo CDN is regularly a working one. The claim is released so the following
        # candidate can be taken.
        _remember_failed_url(url)
        _release_photo(url)
    return None


def _photo_result(photo: dict[str, Any]) -> dict[str, Any]:
    """Shape one chosen photo into what a place stores, with the credit it must show."""
    source = str(photo.get("source") or "pexels")
    if source == "mealdb":
        credit = mealdb_images.credit_line(photo)
    else:
        credit = pexels_images.credit_line(photo)
    return {
        "url": str(photo.get("url") or "").strip(),
        "provider": source,
        "credit": credit,
        "credit_url": photo.get("page_url") or photo.get("photographer_url"),
    }


# A photo already handed to one place in this pass, so no two cards ever show the same
# picture. Guarded by a lock because the lookups run on several threads at once.
_claimed_photos: set[str] = set()
_claims_lock = threading.Lock()
# URLs that failed their probe **in this pass**. A failed URL is not offered again to the
# next place (measured: a hanging CDN was re-probed once per card, spending the whole
# picture-phase budget on one dead link), but its host is not condemned, because the next
# URL on that host is regularly a working photo.
_failed_urls: set[str] = set()


def _remember_failed_url(url: str) -> None:
    with _claims_lock:
        _failed_urls.add(url)


def _url_already_failed(url: str) -> bool:
    with _claims_lock:
        return url in _failed_urls


def _claim_photo(url: str) -> bool:
    """Reserve ``url`` for one place. False when another place already owns it."""
    with _claims_lock:
        if url in _claimed_photos:
            return False
        _claimed_photos.add(url)
        return True


def _release_photo(url: str) -> None:
    """Give a claim back: the photo turned out to be unreachable after all."""
    with _claims_lock:
        _claimed_photos.discard(url)


def _skip_bad_host(url: str) -> bool:
    """True when this URL's host is already known bad or will not resolve."""
    if host_is_known_bad(url):
        return True
    # Reject an unresolvable host without paying the resolver timeout.
    if not host_resolves(_host_of(url)):
        _mark_host_bad(url)
        return True
    return False


def _probe_once(url: str) -> bool:
    """Return True only when this URL actually serves an image right now.

    Checks the status and content type, then reads the first slice of the body.
    Headers alone are not enough: some CDNs answer instantly and then stall while
    the image itself never arrives, which leaves the browser with a blank frame.
    Any failure (timeout, DNS, HTTP error, non-image body) means the client would
    probably fail too, and the caller falls back to the bundled placeholder.

    The body is read up to ``IMAGE_PROBE_BYTES`` (or the declared length when that is
    smaller) rather than accepting the first non-empty chunk, because a response that
    sends a few bytes and then stalls never renders.
    """
    try:
        with httpx.stream(
            "GET",
            url,
            timeout=IMAGE_PROBE_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": PROBE_USER_AGENT},
        ) as response:
            if response.status_code >= 400:
                return False
            content_type = (response.headers.get("content-type") or "").split(";")[0].strip().lower()
            if content_type and not content_type.startswith("image/") and content_type != "application/octet-stream":
                return False
            declared = (response.headers.get("content-length") or "").strip()
            declared_length = int(declared) if declared.isdigit() else None
            target = IMAGE_PROBE_BYTES if declared_length is None else min(declared_length, IMAGE_PROBE_BYTES)
            received = 0
            for chunk in response.iter_bytes():
                received += len(chunk or b"")
                if received >= target:
                    return True
            if declared_length is not None:
                # The body stopped short of what it promised: truncated or stalled.
                return received >= declared_length
            return received > 0
    except Exception:
        return False


def probe_image_url(url: str) -> bool:
    """Probe an image URL under a hard wall-clock budget.

    ``IMAGE_PROBE_TIMEOUT`` alone is not a real bound: an image host that accepts
    the connection and then stalls was measured at **48 seconds** despite a 4 second
    httpx timeout. That single probe consumed a whole picture-phase budget and left
    every place queued behind it with a placeholder. The probe therefore runs in its
    own thread and the caller stops waiting after ``PROBE_BUDGET``; a probe that has
    not answered by then is treated exactly like a failure, and its host is
    remembered so no further probe is attempted.
    """
    outcome = {"ok": False}

    def _run() -> None:
        outcome["ok"] = _probe_once(url)

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(PROBE_BUDGET)
    if worker.is_alive():
        return False
    return outcome["ok"]


# Indirection point: the test suite replaces this so lookups stay offline.
_is_reachable = probe_image_url

# Hosts that already failed a probe during the current resolve pass. A hanging photo
# CDN would otherwise be re-probed for every new place, and each probe can burn a
# whole phase budget because DNS resolution is not covered by the httpx timeout.
#
# This is deliberately cleared at the start of every ``resolve_place_images()`` call.
# Keeping it for the whole process meant one dead URL blacklisted its entire host
# forever: a live photo on p8.itc.cn was then skipped without ever being probed.
_bad_hosts: set[str] = set()
_bad_hosts_lock = threading.Lock()


def _host_of(url: str) -> str:
    return urlparse(url).netloc.lower()


# A host whose name cannot be resolved can stall a probe for the operating
# system's resolver timeout, which no httpx timeout covers (www.davidwin.net was
# measured at a hard DNS failure). Resolution is therefore bounded and cached here,
# before any request is made. Note that the *biggest* offender measured in
# production, dynamic-media-cdn.tripadvisor.com, resolves fine in 0.04 s; its cost
# is a stalling HTTP body, which PROBE_BUDGET handles.
DNS_BUDGET = 2.5
_dns_state: dict[str, bool] = {}
_dns_lock = threading.Lock()


def resolve_host_bounded(host: str, budget: float = DNS_BUDGET) -> bool:
    """True when ``host`` resolves within ``budget`` seconds.

    The lookup runs in its own thread so the caller is never blocked past the
    budget. Results are cached per host: a name that resolves quickly is answered
    from memory, and one that does not is not paid for again in this process.
    """
    if not host:
        return False
    with _dns_lock:
        cached = _dns_state.get(host)
    if cached is not None:
        return cached

    resolved = {"ok": False}

    def _resolve() -> None:
        try:
            socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
            resolved["ok"] = True
        except Exception:
            resolved["ok"] = False

    worker = threading.Thread(target=_resolve, daemon=True)
    worker.start()
    worker.join(budget)
    ok = resolved["ok"] and not worker.is_alive()
    with _dns_lock:
        _dns_state[host] = ok
    return ok


# Indirection point: the test suite replaces this so lookups stay offline.
host_resolves = resolve_host_bounded


def host_is_known_bad(url: str) -> bool:
    with _bad_hosts_lock:
        return _host_of(url) in _bad_hosts


def _mark_host_bad(url: str) -> None:
    with _bad_hosts_lock:
        _bad_hosts.add(_host_of(url))


def clear_image_cache() -> None:  # noqa: D401 - see below
    """Reset the lookup cache, host memory and DNS memory. Intended for tests."""
    with _cache_lock:
        _image_cache.clear()
        _miss_cache.clear()
    with _bad_hosts_lock:
        _bad_hosts.clear()
    with _dns_lock:
        _dns_state.clear()
    with _claims_lock:
        _claimed_photos.clear()
        _failed_urls.clear()


def _remember(destination: str, name: str, category: str, value: dict[str, Any] | None) -> None:
    """Store a lookup outcome, so a slow host is not paid for on every request.

    A *hit* is cached for the life of the process; a *miss* expires, because "no
    photo exists" is a guess about a provider that keeps changing its ranking. A
    permanent negative entry left one restaurant stuck on the placeholder even
    though a fresh process found a working photo for it straight away.
    """
    key = (destination, name, category)
    with _cache_lock:
        if value is None:
            _miss_cache[key] = time.monotonic() + IMAGE_MISS_TTL_SECONDS
        else:
            _image_cache[key] = value
            _miss_cache.pop(key, None)


def _cached_image(key: tuple[str, str, str]) -> tuple[bool, dict[str, Any] | None]:
    """Return (is_cached, value) — a miss only counts as cached while unexpired."""
    with _cache_lock:
        if key in _image_cache:
            return True, _image_cache[key]
        expires_at = _miss_cache.get(key)
        if expires_at is None:
            return False, None
        if expires_at > time.monotonic():
            return True, None
        del _miss_cache[key]
        return False, None


def _mealdb_photo(required: list[str], destination: str, name: str) -> dict[str, Any] | None:
    """A dish photo from TheMealDB, for food the stock library cannot place.

    Three attempts, in order of how specific they are: the dish itself, each significant
    word of it (the library matches on whole meal names, so "tonkotsu ramen" finds nothing
    while "ramen" finds "Ramen Noodles with Boiled Egg"), then the destination's cuisine.
    The last one is the point: a photo of the right cuisine is a photo that matches the
    card, which is what this whole chain is for.
    """
    bonus = _keywords(destination)
    seen: set[str] = set()
    for term in [name, *[word for word in required if len(word) > 3]]:
        text = str(term or "").strip()
        if not text or text.casefold() in seen:
            continue
        seen.add(text.casefold())
        photos = mealdb_images.search_dish_photos(text, limit=6)
        picked = _claim_best(photos, required, bonus, require_match=True)
        if picked is not None:
            return _photo_result(picked)
    area = mealdb_images.area_for_destination(destination)
    if area:
        photos = mealdb_images.area_photos(area, limit=12)
        # Nothing in the dish search matched, so this is a cuisine-level picture: accept
        # the first unclaimed one rather than demanding the dish's own words.
        picked = _claim_best(photos, [], bonus, require_match=False)
        if picked is not None:
            return _photo_result(picked)
    return None


def _pexels_photo(
    destination: str,
    name: str,
    category: str,
    dish: str | None = None,
    english: str | None = None,
) -> dict[str, Any] | None:
    """A photo from the stock library that matches this card, or None.

    Three pages are walked, each with its own standard of proof:

    1. the place's own English keyword, where a candidate must match the keyword's words --
       this is the only page that can produce a picture of the right subject;
    2. the destination's category page for unclaimed leftovers, where any photo of that
       city and category is acceptable because it is at least the right thing in the right
       place;
    3. nothing else: a Chinese-name search is not attempted for an attraction, because that
       is the query that returned Osaka Castle.

    Two pages are needed for a second reason: two places in one city produce near-identical
    queries, so the same photo comes back for both. Measured before the city pool existed,
    38 cards ended up on 17 photos, one of them on 9 cards. Each photo is claimed as it is
    taken and a claimed one is skipped in favour of the next candidate.
    """
    hint = PEXELS_QUERY_HINTS.get(category, "")
    bonus = _keywords(destination)
    # What a candidate must say to be accepted for this card: the English keyword's own
    # words, or failing that the destination's. Verification is what stops "snow-covered
    # horse saddles on wooden rack" being used for a bowl of ramen, and what rejected the
    # photo of Osaka Castle that a Tokyo museum query came back with -- an English keyword
    # carries the city in it ("Tokyo National Museum"), so the wrong city fails the check.
    # With neither (a Chinese-only keyword for a Chinese destination) there is nothing to
    # verify against and the library's own ranking is used.
    required = _keywords(english or (dish if category == "restaurant" else None))
    query = _pexels_query(name, category, destination, dish, english)
    if query:
        page = search_pexels_photos(query, limit=PEXELS_CANDIDATE_LIMIT)
        picked = _claim_best(page, required, bonus, require_match=bool(required or bonus))
        if picked is not None:
            return _photo_result(picked)
    if hint:
        pool = pexels_images.city_pool_photos(destination, hint)
        # No verification here: these are the results for "日本东京 food", which are the
        # right city and the right category but need not mention the dish.
        picked = _claim_best(pool, [], bonus, require_match=False)
        if picked is not None:
            return _photo_result(picked)
    return None


def _lookup_image(
    destination: str,
    name: str,
    category: str,
    dish: str | None = None,
    english: str | None = None,
) -> dict[str, Any] | None:
    key = (destination, name, category)
    is_cached, cached = _cached_image(key)
    if is_cached:
        # A cached photo still has to be claimed for this pass. Two places can hold the
        # same cached photo from an earlier run, and a cache hit that skipped the claim
        # would put the duplicate straight back on the page.
        if cached is None or cached.get("provider") == "static":
            return cached
        if _claim_photo(str(cached.get("url"))):
            return cached
        cached = None
    found: dict[str, Any] | None = None
    try:
        if category == "restaurant":
            # Food is pictured by the dish, never by the venue: the provider's per-place
            # search returns shopfronts and streets, and the user asked for the food. The
            # dish library is tried first for dishes the stock library misses (tonkatsu,
            # yakitori), then the stock library by the dish's English name.
            found = _pexels_photo(destination, name, category, dish, english)
            if found is None:
                found = _mealdb_photo(_keywords(english or dish), destination, english or dish or "")
        else:
            found = _pexels_photo(destination, name, category, dish, english)
    except Exception:
        # A missing picture must never break planning.
        found = None
    _remember(destination, name, category, found)
    return found


def resolve_place_images(
    places: Iterable[dict[str, Any]],
    destination: str,
    *,
    name_key: str = "name",
    category_key: str = "type",
    image_key: str = "image_url",
    max_lookups: int = MAX_IMAGE_LOOKUPS,
    skip_categories: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """Return copies of ``places`` where every entry that should have a picture has one.

    Each entry also receives ``fallback_image_url``: the bundled placeholder for
    its category. A provider photo can still turn out to be a dead link or sit on
    a host the client cannot reach, so the client needs a known-good substitute
    to swap in rather than dropping the picture.

    A photo taken from the Pexels library also carries ``image_credit`` (the line that
    must be shown with it) and ``image_credit_url`` (its page on Pexels); both are None
    for photos that need no attribution.

    ``skip_categories`` names categories that must NOT be given a picture. Hotels are
    skipped by the caller because a photo of some other hotel is not information: a hotel
    card shows its rating and its price instead. Such an entry comes back with an empty
    ``image_url`` and no credit, which is the signal the client renders it without a
    picture.

    Host memory is scoped to this call: a host that fails here is skipped for the
    rest of this pass, then forgotten, so one dead URL cannot blacklist a host that
    serves working photos for every later request.
    """
    with _bad_hosts_lock:
        _bad_hosts.clear()
    # Photo claims are per pass, like host memory above: the next request starts with an
    # empty page and every place gets to claim again.
    with _claims_lock:
        _claimed_photos.clear()
        _failed_urls.clear()
    resolved = [dict(place) for place in places]
    pending: list[tuple[str, str]] = []
    dishes: dict[tuple[str, str], str] = {}
    englishs: dict[tuple[str, str], str] = {}
    seen: set[tuple[str, str]] = set()
    for place in resolved:
        if has_usable_image(place.get(image_key)):
            continue
        name = str(place.get(name_key) or "").strip()
        if not name:
            continue
        category = normalize_category(place.get(category_key))
        if category in skip_categories:
            continue
        key = (name, category)
        if key in seen or len(pending) >= max_lookups:
            continue
        seen.add(key)
        # A restaurant is pictured by its dish, not by its shopfront.
        dish = str(place.get("dish_query") or place.get("signature_dish") or "").strip()
        if dish:
            dishes[key] = dish
        # The English keyword is what makes the search land on the right subject; without
        # one the Chinese name is still tried, with an English category word.
        english = str(place.get("image_query") or "").strip()
        if english:
            englishs[key] = english
        pending.append(key)
    found: dict[tuple[str, str], dict[str, Any] | None] = {}
    if pending:
        pool = ThreadPoolExecutor(max_workers=min(IMAGE_LOOKUP_WORKERS, len(pending)))
        try:
            futures = {
                pool.submit(
                    _lookup_image,
                    destination,
                    name,
                    category,
                    dishes.get((name, category)),
                    englishs.get((name, category)),
                ): (name, category)
                for name, category in pending
            }
            # One wall-clock budget for the whole picture phase. A per-future wait
            # would still be paid once per place when lookups finish out of order,
            # and an httpx timeout does not cover DNS resolution: an unreachable
            # photo CDN stalled one lookup for 112 seconds here.
            done, not_done = wait(futures, timeout=IMAGE_PHASE_BUDGET)
            for future in done:
                try:
                    found[futures[future]] = future.result()
                except Exception:
                    found[futures[future]] = None
            for future in not_done:
                # A lookup cancelled by the budget says nothing about whether the
                # photo exists, so it is NOT remembered: caching it as a miss would
                # deny this place a picture for the rest of the process lifetime.
                found[futures[future]] = None
                future.cancel()
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
    for place in resolved:
        category = normalize_category(place.get(category_key))
        if category in skip_categories:
            # No picture was asked for, so nothing is credited either.
            place[image_key] = None
            place["fallback_image_url"] = None
            place["image_credit"] = None
            place["image_credit_url"] = None
            continue
        place["fallback_image_url"] = fallback_image(category, place.get("dish_query") or place.get("signature_dish") or place.get("image_query"))
        if has_usable_image(place.get(image_key)):
            place[image_key] = str(place[image_key]).strip()
            place.setdefault("image_credit", None)
            place.setdefault("image_credit_url", None)
            continue
        name = str(place.get(name_key) or "").strip()
        photo = found.get((name, category))
        place[image_key] = (photo or {}).get("url") or place["fallback_image_url"]
        # Only a Pexels photo needs crediting; a provider photo carries none.
        place["image_credit"] = (photo or {}).get("credit")
        place["image_credit_url"] = (photo or {}).get("credit_url")
    return resolved
