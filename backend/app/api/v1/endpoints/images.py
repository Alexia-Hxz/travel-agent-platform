"""Public, same-origin image proxy backed by the permanent photo cache."""

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse, RedirectResponse, Response

from app.services.photo_cache import cache_public_photo, cached_photo_path, placeholder_url

router = APIRouter(tags=["images"])


@router.get("/image-proxy", response_class=Response)
def image_proxy(
    url: str = Query(min_length=1, max_length=2048),
    category: str | None = Query(default=None, max_length=40),
) -> Response:
    """Cache and serve an allow-listed image, or fall back without hanging the UI."""
    local_url = cache_public_photo(url)
    local_path = cached_photo_path(local_url) if local_url else None
    if local_path is not None:
        return FileResponse(local_path, headers={"Cache-Control": "public, max-age=31536000, immutable"})
    return RedirectResponse(placeholder_url(category), status_code=307)
