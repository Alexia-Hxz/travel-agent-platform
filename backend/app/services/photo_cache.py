"""Keep provider photos reachable when the browser cannot reach the provider CDN.

The planner already obtains and checks public photo URLs server-side. Only the two
known photo hosts are mirrored, with a size limit, under the existing /static mount.
The plan still carries the original photographer and Pexels attribution fields.
"""

from concurrent.futures import ThreadPoolExecutor, wait
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlencode, urlparse
import logging
import os
import uuid

import httpx
from app.agents.tools.place_images import resolve_place_images

logger = logging.getLogger(__name__)
PHOTO_DIR = Path(__file__).resolve().parents[1] / "static" / "place_images" / "cache"
STATIC_PREFIX = "/static/place_images/cache"
ALLOWED_HOSTS = {"images.pexels.com", "www.themealdb.com"}
IMAGE_TYPES = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
MAX_BYTES = 2_000_000
MAX_WORKERS = 8
PHASE_BUDGET_SECONDS = 20
PLACEHOLDER_TYPES = {"attraction", "hotel", "restaurant"}


def cache_public_photo(url: str) -> str | None:
    """Return a local static path for a known public image, or None on failure."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        return None
    digest = sha256(url.encode("utf-8")).hexdigest()
    for extension in IMAGE_TYPES.values():
        existing = PHOTO_DIR / f"{digest}{extension}"
        if existing.is_file() and existing.stat().st_size > 0:
            return f"{STATIC_PREFIX}/{existing.name}"

    temporary: Path | None = None
    try:
        PHOTO_DIR.mkdir(parents=True, exist_ok=True)
        with httpx.stream("GET", url, timeout=10.0, follow_redirects=False) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            extension = IMAGE_TYPES.get(content_type)
            if not extension:
                return None
            destination = PHOTO_DIR / f"{digest}{extension}"
            temporary = PHOTO_DIR / f".{digest}.{uuid.uuid4().hex}.tmp"
            size = 0
            with temporary.open("wb") as output:
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > MAX_BYTES:
                        return None
                    output.write(chunk)
            if size == 0:
                return None
            os.replace(temporary, destination)
            temporary = None
            return f"{STATIC_PREFIX}/{destination.name}"
    except (httpx.HTTPError, OSError, ValueError) as error:
        logger.info("photo cache unavailable for %s: %s", parsed.hostname, error)
        return None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def proxy_photo_url(url: str, category: str | None = None) -> str | None:
    """Return a same-origin proxy URL for an allowed provider image."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        return None
    normalized_category = str(category or "").strip().lower()
    if normalized_category not in PLACEHOLDER_TYPES:
        normalized_category = "generic"
    return "/api/v1/image-proxy?" + urlencode({"url": url, "category": normalized_category})


def cached_photo_path(static_url: str) -> Path | None:
    """Resolve one cache URL to its file without allowing path traversal."""
    prefix = f"{STATIC_PREFIX}/"
    if not static_url.startswith(prefix):
        return None
    name = static_url.removeprefix(prefix)
    if not name or Path(name).name != name:
        return None
    path = PHOTO_DIR / name
    return path if path.is_file() and path.stat().st_size > 0 else None


def placeholder_url(category: str | None = None) -> str:
    normalized = str(category or "").strip().lower()
    if normalized not in PLACEHOLDER_TYPES:
        normalized = "generic"
    return f"/static/place_images/{normalized}.svg"


def _recover_missing_photos(plan: dict, destination: str) -> None:
    """Retry a mostly empty gallery with broad, city-level photo matching.

    The planner's first pass uses each place's specific English query. On a large
    plan that pass can exhaust its bounded lookup time while the photo provider is
    slow. The response no longer carries those specific queries, so this one retry
    uses the destination's category pool, which was measured to cover a 27-place
    Hangzhou result in one pass. Photos remain clearly labelled as stock examples.
    """
    recommendations = plan.get("place_recommendations") or []
    picturable = [entry for entry in recommendations if isinstance(entry, dict) and entry.get("category") != "hotel"]
    missing = [entry for entry in picturable if str(entry.get("image_url") or "").startswith("/static/place_images/")]
    if len(missing) < 3 or len(missing) * 10 < len(picturable) * 7:
        return
    entries = [
        {"name": entry.get("name"), "type": entry.get("category"), "image_url": None,
         "signature_dish": entry.get("signature_dish")}
        for entry in missing
    ]
    recovered = resolve_place_images(entries, destination)
    by_name = {entry.get("name"): entry for entry in recovered}
    for entry in missing:
        replacement = by_name.get(entry.get("name")) or {}
        if not str(replacement.get("image_url") or "").startswith("https://"):
            continue
        for key in ("image_url", "fallback_image_url", "image_credit", "image_credit_url"):
            entry[key] = replacement.get(key)
    by_name = {entry.get("name"): entry for entry in recommendations if isinstance(entry, dict)}
    for day in plan.get("daily_plan") or []:
        for item in day.get("items") or []:
            recommendation = by_name.get(item.get("place_name"))
            if recommendation and str(recommendation.get("image_url") or "").startswith("https://"):
                item["image_url"] = recommendation["image_url"]


def cache_plan_photos(plan: dict, *, recover_missing: bool = False, destination: str = "") -> dict:
    """Replace provider URLs in a plan with locally served copies when possible."""
    if recover_missing:
        _recover_missing_photos(plan, destination)
    recommendations = plan.get("place_recommendations") or []
    days = plan.get("daily_plan") or []
    entries = [*recommendations, *(item for day in days for item in (day.get("items") or []))]
    urls = {entry.get("image_url") for entry in entries if isinstance(entry, dict)}
    candidates = {url for url in urls if isinstance(url, str) and urlparse(url).hostname in ALLOWED_HOSTS}
    if not candidates:
        return plan

    pool = ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(candidates)))
    try:
        futures = {pool.submit(cache_public_photo, url): url for url in candidates}
        done, pending = wait(futures, timeout=PHASE_BUDGET_SECONDS)
        local = {futures[future]: future.result() for future in done if not future.exception()}
        for future in pending:
            future.cancel()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        original = entry.get("image_url")
        if original in local and local[original]:
            entry["image_url"] = local[original]
            continue
        if isinstance(original, str):
            proxied = proxy_photo_url(original, entry.get("category") or entry.get("type"))
            if proxied:
                # Browsers never need direct access to the provider CDN. A request to
                # this URL gets one final cache attempt and an SVG fallback on failure.
                entry["image_url"] = proxied
    return plan
