from pathlib import Path

from fastapi.responses import FileResponse, RedirectResponse

from app.api.v1.endpoints import images
from app.services import photo_cache


class FakeResponse:
    headers = {"content-type": "image/jpeg"}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def raise_for_status(self):
        pass

    def iter_bytes(self):
        yield b"photo-bytes"


def test_provider_photo_is_served_from_existing_static_mount(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(photo_cache, "PHOTO_DIR", tmp_path)
    calls = []

    def stream(*args, **kwargs):
        calls.append((args, kwargs))
        return FakeResponse()

    monkeypatch.setattr(photo_cache.httpx, "stream", stream)
    source = "https://images.pexels.com/photos/123/photo.jpeg"
    local = photo_cache.cache_public_photo(source)

    assert local and local.startswith("/static/place_images/cache/")
    assert (tmp_path / local.rsplit("/", 1)[-1]).read_bytes() == b"photo-bytes"
    assert photo_cache.cache_public_photo(source) == local
    assert len(calls) == 1
    assert photo_cache.cache_public_photo("https://other.example.com/private.jpg") is None
    assert len(calls) == 1


def test_plan_uses_cached_photo_without_losing_attribution(monkeypatch) -> None:
    source = "https://images.pexels.com/photos/123/photo.jpeg"
    local = "/static/place_images/cache/photo.jpg"
    monkeypatch.setattr(photo_cache, "cache_public_photo", lambda url: local)
    plan = {
        "place_recommendations": [{"name": "西湖", "image_url": source, "image_credit": "Photo by Lin on Pexels", "image_credit_url": "https://www.pexels.com/photo/123"}],
        "daily_plan": [{"date": "2026-10-02", "items": [{"place_name": "西湖", "image_url": source}]}],
    }

    photo_cache.cache_plan_photos(plan)

    assert plan["place_recommendations"][0]["image_url"] == local
    assert plan["daily_plan"][0]["items"][0]["image_url"] == local
    assert plan["place_recommendations"][0]["image_credit_url"] == "https://www.pexels.com/photo/123"


def test_plan_routes_uncached_provider_photo_through_same_origin_proxy(monkeypatch) -> None:
    source = "https://images.pexels.com/photos/123/photo.jpeg?auto=compress"
    monkeypatch.setattr(photo_cache, "cache_public_photo", lambda url: None)
    plan = {"place_recommendations": [{"category": "hotel", "image_url": source}], "daily_plan": []}

    photo_cache.cache_plan_photos(plan)

    proxied = plan["place_recommendations"][0]["image_url"]
    assert proxied.startswith("/api/v1/image-proxy?")
    assert "images.pexels.com" in proxied
    assert "category=hotel" in proxied


def test_cached_photo_path_rejects_non_cache_and_traversal_paths(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(photo_cache, "PHOTO_DIR", tmp_path)
    cached = tmp_path / "abc.jpg"
    cached.write_bytes(b"photo")

    assert photo_cache.cached_photo_path("/static/place_images/cache/abc.jpg") == cached
    assert photo_cache.cached_photo_path("/static/place_images/cache/../secret.jpg") is None
    assert photo_cache.cached_photo_path("/static/place_images/hotel.svg") is None


def test_image_proxy_serves_cache_or_redirects_to_existing_placeholder(monkeypatch, tmp_path: Path) -> None:
    cached = tmp_path / "photo.jpg"
    cached.write_bytes(b"photo")
    monkeypatch.setattr(images, "cache_public_photo", lambda url: "/static/place_images/cache/photo.jpg")
    monkeypatch.setattr(images, "cached_photo_path", lambda url: cached)

    served = images.image_proxy("https://images.pexels.com/photo.jpg", "hotel")
    assert isinstance(served, FileResponse)
    assert served.headers["cache-control"] == "public, max-age=31536000, immutable"

    monkeypatch.setattr(images, "cache_public_photo", lambda url: None)
    failed = images.image_proxy("https://images.pexels.com/missing.jpg", "restaurant")
    assert isinstance(failed, RedirectResponse)
    assert failed.headers["location"] == "/static/place_images/restaurant.svg"


def test_large_placeholder_gallery_gets_one_broad_recovery_pass(monkeypatch) -> None:
    calls = []
    def recover(entries, destination):
        calls.append((entries, destination))
        return [{**entry, "image_url": f"https://images.pexels.com/{index}.jpg", "image_credit": "Photo on Pexels", "image_credit_url": "https://www.pexels.com"} for index, entry in enumerate(entries)]
    monkeypatch.setattr(photo_cache, "resolve_place_images", recover)
    monkeypatch.setattr(photo_cache, "cache_public_photo", lambda url: "/static/place_images/cache/" + url.rsplit("/", 1)[-1])
    plan = {
        "place_recommendations": [
            {"name": f"地点{index}", "category": "attraction", "image_url": "/static/place_images/attraction.svg"}
            for index in range(3)
        ],
        "daily_plan": [{"items": [{"place_name": "地点0", "image_url": "/static/place_images/attraction.svg"}]}],
    }

    photo_cache.cache_plan_photos(plan, recover_missing=True, destination="杭州")

    assert len(calls) == 1
    assert calls[0][1] == "杭州"
    assert all(entry["image_url"].startswith("/static/place_images/cache/") for entry in plan["place_recommendations"])
    assert plan["place_recommendations"][0]["image_credit"] == "Photo on Pexels"
    assert plan["daily_plan"][0]["items"][0]["image_url"].startswith("/static/place_images/cache/")
