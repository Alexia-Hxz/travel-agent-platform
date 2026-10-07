"""The Pexels photo library, the project's only picture source.

Pexels is free, its photos are real, and using them carries an attribution obligation.
These tests pin the parts that matter: the English keyword is what gets searched, a
candidate whose own description does not match is refused, every photo arrives with the
credit it must be shown with, and that credit survives into the API response instead of
being dropped by the response whitelist.
"""

from typing import Any

import pytest

from app.agents.tools import pexels_images, place_images
from app.agents.workflows import planner

PHOTO = {
    "url": "https://images.pexels.com/photos/1234/pexels-photo-1234.jpeg?auto=compress",
    "photographer": "Zhang Wei",
    "photographer_url": "https://www.pexels.com/@zhang-wei",
    "page_url": "https://www.pexels.com/photo/temple-1234/",
    "alt": "confucius temple in nanjing",
}


def _enable_pexels(
    monkeypatch: pytest.MonkeyPatch,
    photos: list[dict[str, Any]] | None = None,
    pool: list[dict[str, Any]] | None = None,
    *,
    real_pool: bool = False,
    pool_calls: list[str] | None = None,
):
    """Point the Pexels source at fixed pages and keep the shared pool empty by default.

    With ``real_pool=True`` the real ``city_pool_photos`` runs (so its own cache is under
    test) and only the HTTP call beneath it is stubbed.
    """
    monkeypatch.setenv("PEXELS_API_KEY", "test-key")
    asked: list[str] = []

    def _search(query, limit=3, timeout=8.0):
        asked.append(query)
        return list(photos if photos is not None else [PHOTO])

    monkeypatch.setattr(place_images, "search_pexels_photos", _search)
    if real_pool:

        def _pool_search(query, limit=80, timeout=8.0):
            if pool_calls is not None:
                pool_calls.append(query)
            return list(pool or [])

        monkeypatch.setattr(pexels_images, "search_pexels_photos", _pool_search)
    else:
        monkeypatch.setattr(pexels_images, "city_pool_photos", lambda destination, hint: list(pool or []))
    pexels_images.clear_pool_cache()
    monkeypatch.setattr(place_images, "_is_reachable", lambda url: True)
    monkeypatch.setattr(place_images, "host_resolves", lambda host, budget=2.5: True)
    place_images.clear_image_cache()
    return asked


def test_the_library_is_the_only_photo_source(monkeypatch) -> None:
    """The metered search provider is gone: it returned no image URLs when it answered.

    Measured before it was removed: HTTP 432 for every query, so the per-place image step
    that used it was not a safety net but a slower path to the same placeholder, and the
    photos it did once supply were the least relevant of the sources.
    """
    from app.agents.tools import web_search

    assert not hasattr(place_images, "search_images")
    assert not hasattr(web_search, "search_images")

    asked = _enable_pexels(monkeypatch)

    photo = place_images._lookup_image("南京", "夫子庙", "attraction")

    assert photo is not None
    assert photo["provider"] == "pexels"
    assert asked == ["夫子庙 南京 landmark"]


def test_pexels_fills_in_when_the_per_place_page_has_nothing(monkeypatch) -> None:
    asked = _enable_pexels(monkeypatch, photos=[], pool=[PHOTO])

    photo = place_images._lookup_image("南京", "夫子庙", "attraction")

    assert photo is not None
    assert photo["url"] == PHOTO["url"]
    assert photo["provider"] == "pexels"
    # The credit and its link travel with the photo, so no second request is needed.
    assert photo["credit"] == "Photo by Zhang Wei on Pexels"
    assert photo["credit_url"] == PHOTO["page_url"]
    # The query states the category too: a bare place name is answered with city scenery.
    assert asked == ["夫子庙 南京 landmark"]


def test_a_photo_without_a_photographer_gets_no_invented_credit() -> None:
    """A missing name must not become "Photo by None on Pexels"."""
    assert pexels_images.credit_line({"photographer": None}) is None
    assert pexels_images.credit_line({"photographer": "  "}) is None
    assert pexels_images.credit_line({"photographer": "Li Na"}) == "Photo by Li Na on Pexels"


def test_the_resolved_place_carries_its_credit(monkeypatch) -> None:
    _enable_pexels(monkeypatch)

    resolved = place_images.resolve_place_images(
        [{"name": "夫子庙", "type": "attraction"}], "南京"
    )

    place = resolved[0]
    assert place["image_url"] == PHOTO["url"]
    assert place["image_credit"] == "Photo by Zhang Wei on Pexels"
    assert place["image_credit_url"] == PHOTO["page_url"]


def test_a_placeholder_needs_no_credit(monkeypatch) -> None:
    """Nothing was used, so nothing may be credited."""
    _enable_pexels(monkeypatch, photos=[])

    place = place_images.resolve_place_images([{"name": "夫子庙", "type": "attraction"}], "南京")[0]

    assert place["image_url"] == place["fallback_image_url"]
    assert place["image_credit"] is None
    assert place["image_credit_url"] is None


def test_the_prompt_tells_the_interface_to_credit_pexels() -> None:
    """The label is a condition of the API, so it lives next to the module that uses it."""
    assert pexels_images.ATTRIBUTION_LABEL == "Photos provided by Pexels"
    assert pexels_images.ATTRIBUTION_URL == "https://www.pexels.com"


def test_a_pexels_failure_is_never_fatal(monkeypatch) -> None:
    """A missing picture must not break planning."""
    monkeypatch.setattr(pexels_images.httpx, "get", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setenv("PEXELS_API_KEY", "test-key")

    # An unexpected error is not caught inside the module, and must not be: the caller
    # wraps the whole lookup. What matters is that the module reports a real HTTP failure
    # as "no photos" rather than raising.
    import httpx

    def _bad_request(*args, **kwargs):
        raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(pexels_images.httpx, "get", _bad_request)
    assert pexels_images.search_pexels_photos("夫子庙 南京") == []


def test_the_pexels_key_is_not_required(monkeypatch) -> None:
    """Without a key the source is simply unavailable, and the placeholder is used."""
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    monkeypatch.setattr(pexels_images, "ENV_PATH", pexels_images.Path("does-not-exist.env"))

    assert pexels_images.pexels_available() is False
    assert pexels_images.search_pexels_photos("夫子庙 南京") == []


def test_the_response_whitelist_keeps_the_credit() -> None:
    """The API response is built from an explicit whitelist.

    A field added to the recommendation dicts but forgotten there never reaches the
    client, which for a Pexels photo would mean showing it uncredited.
    """
    import inspect

    source = inspect.getsource(planner._compose)

    assert '"image_credit"' in source
    assert '"image_credit_url"' in source


def test_a_card_sized_rendition_is_chosen() -> None:
    """`original` can be many megabytes and the cards render a few hundred pixels wide."""
    item = {
        "src": {
            "original": "https://images.pexels.com/original.jpeg",
            "large": "https://images.pexels.com/large.jpeg",
            "medium": "https://images.pexels.com/medium.jpeg",
        }
    }

    assert pexels_images._best_url(item) == "https://images.pexels.com/large.jpeg"


def test_a_photo_without_a_usable_rendition_is_skipped() -> None:
    assert pexels_images._best_url({"src": {"original": "https://images.pexels.com/o.jpeg"}}) is None
    assert pexels_images._best_url({"src": "not-a-dict"}) is None
    assert pexels_images._best_url({}) is None


def test_two_places_never_share_a_photo(monkeypatch) -> None:
    """Two cards showing one picture reads as a bug even when both placements are plausible.

    This was measured on a real plan: 12 places resolved to 7 photos, one of them shared by
    4 places. The whole result page is walked and every photo is claimed as it is taken.
    """
    pool = [
        {
            "url": f"https://images.pexels.com/photos/{n}/pexels-photo-{n}.jpeg",
            "photographer": "Cheng Shi Song",
            "page_url": f"https://www.pexels.com/photo/{n}/",
        }
        for n in (1, 2, 3)
    ]
    monkeypatch.setenv("PEXELS_API_KEY", "test-key")
    # Every query returns the same ranking, which is what happens for places in one city.
    monkeypatch.setattr(place_images, "search_pexels_photos", lambda query, limit=15, timeout=8.0: list(pool))
    monkeypatch.setattr(place_images, "_is_reachable", lambda url: True)
    monkeypatch.setattr(place_images, "host_resolves", lambda host, budget=2.5: True)
    place_images.clear_image_cache()

    resolved = place_images.resolve_place_images(
        [{"name": n, "type": "attraction"} for n in ("江南贡院", "大成殿", "中山陵")], "南京"
    )

    urls = [place["image_url"] for place in resolved]
    assert len(set(urls)) == 3, urls


def test_a_cached_photo_is_not_shared_within_one_pass(monkeypatch) -> None:
    """The lookup cache outlives a pass, so a cached photo still has to be claimed per pass.

    Without that re-claim, two places that happened to resolve to one photo in an earlier
    request would keep sharing it on every later page.
    """
    photo = {
        "url": "https://images.pexels.com/photos/9/pexels-photo-9.jpeg",
        "photographer": "Li Na",
        "page_url": "https://www.pexels.com/photo/9/",
    }
    monkeypatch.setenv("PEXELS_API_KEY", "test-key")
    # Only one distinct photo exists in the library for either place.
    monkeypatch.setattr(place_images, "search_pexels_photos", lambda query, limit=15, timeout=8.0: [photo])
    monkeypatch.setattr(place_images, "_is_reachable", lambda url: True)
    monkeypatch.setattr(place_images, "host_resolves", lambda host, budget=2.5: True)
    place_images.clear_image_cache()

    # Warm the cache for one place, then ask for both in a single pass.
    place_images.resolve_place_images([{"name": "夫子庙", "type": "attraction"}], "南京")
    resolved = place_images.resolve_place_images(
        [{"name": "夫子庙", "type": "attraction"}, {"name": "江南贡院", "type": "attraction"}], "南京"
    )

    urls = [place["image_url"] for place in resolved]
    # Whoever claims it first keeps it; the other must not repeat it. Which of the two wins
    # depends on thread timing, so only the invariant is asserted.
    assert len(set(urls)) == 2, urls
    assert urls.count(photo["url"]) == 1
    placeholder = [place["fallback_image_url"] for place in resolved]
    assert sorted(urls) == sorted([photo["url"], placeholder[0]])


def test_a_place_with_no_photo_left_falls_back_to_the_placeholder(monkeypatch) -> None:
    """Running out of distinct photos gives a placeholder, never a repeated photo."""
    photo = {
        "url": "https://images.pexels.com/photos/9/pexels-photo-9.jpeg",
        "photographer": "Li Na",
        "page_url": "https://www.pexels.com/photo/9/",
    }
    monkeypatch.setenv("PEXELS_API_KEY", "test-key")
    monkeypatch.setattr(place_images, "search_pexels_photos", lambda query, limit=15, timeout=8.0: [photo])
    monkeypatch.setattr(place_images, "_is_reachable", lambda url: True)
    monkeypatch.setattr(place_images, "host_resolves", lambda host, budget=2.5: True)
    place_images.clear_image_cache()

    resolved = place_images.resolve_place_images(
        [{"name": "夫子庙", "type": "attraction"}, {"name": "江南贡院", "type": "attraction"}], "南京"
    )

    urls = [place["image_url"] for place in resolved]
    assert sorted(urls) == sorted([photo["url"], resolved[1]["fallback_image_url"]])
    assert len(set(urls)) == 2


def test_the_category_is_stated_for_the_library_in_english() -> None:
    """A hotel query with no category word is answered with city scenery.

    Measured: `如家精选南京新街口店 南京` came back with a street scene rather than a hotel,
    which is why the hotel cards looked like they had the wrong picture.
    """
    assert place_images._pexels_query("如家精选南京新街口店", "hotel", "南京") == (
        "如家精选南京新街口店 南京 hotel"
    )
    # An English keyword replaces the Chinese name outright, and the category word stays.
    assert place_images._pexels_query("浅草寺", "attraction", "日本东京", None, "Sensoji Temple") == (
        "Sensoji Temple landmark"
    )
    # A food card is searched by the dish, in English, and never by the venue's name.
    assert place_images._pexels_query("南京大牌档", "restaurant", "南京", "sweet and sour mandarin fish") == (
        "sweet and sour mandarin fish food"
    )
    # Nothing is known about the dish, so there is no useful query: the city's food page is
    # used instead of a shopfront or a street in some other city.
    assert place_images._pexels_query("南京大牌档", "restaurant", "南京") == ""
    # A place with no recognized category still gets a usable query.
    assert place_images._pexels_query("夫子庙", "", "南京") == "夫子庙 南京"


def test_the_city_pool_supplies_photos_when_a_place_page_runs_dry(monkeypatch) -> None:
    """A per-place page has almost nothing the next place's page does not also have.

    Measured: 21 different restaurant queries between them offered only 11 distinct photos,
    so 10 cards fell back to a placeholder. One city-wide query returns the full 80-photo
    page, which is what lets every card have its own picture.
    """
    shared = {
        "url": "https://images.pexels.com/photos/1/shared.jpeg",
        "photographer": "A",
        "page_url": "https://www.pexels.com/photo/1/",
    }
    pool = [
        shared,
        {
            "url": "https://images.pexels.com/photos/2/pool.jpeg",
            "photographer": "B",
            "page_url": "https://www.pexels.com/photo/2/",
        },
    ]
    _enable_pexels(monkeypatch, photos=[shared], pool=pool)

    resolved = place_images.resolve_place_images(
        [{"name": "南京大牌档", "type": "restaurant"}, {"name": "绿柳居", "type": "restaurant"}],
        "南京",
    )

    urls = [place["image_url"] for place in resolved]
    # The first card takes the shared photo from its own page; the second, finding that one
    # claimed, reaches into the city pool rather than giving up on a picture.
    assert len(set(urls)) == 2, urls
    assert all("images.pexels.com" in url for url in urls)
    assert all(place["image_credit"] for place in resolved)


def test_the_city_pool_is_asked_for_once_per_destination_and_category(monkeypatch) -> None:
    """One deep page per destination + category, not one per card."""
    pool_calls: list[str] = []
    pool = [
        {
            "url": f"https://images.pexels.com/photos/{n}/pexels-photo-{n}.jpeg",
            "photographer": "A",
            "page_url": f"https://www.pexels.com/photo/{n}/",
        }
        for n in (1, 2, 3)
    ]
    # The real city_pool_photos runs here, so its cache is what is under test.
    _enable_pexels(monkeypatch, photos=[], pool=pool, real_pool=True, pool_calls=pool_calls)

    place_images.resolve_place_images(
        [{"name": n, "type": "restaurant"} for n in ("南京大牌档", "绿柳居", "马祥兴菜馆")], "南京"
    )

    assert pool_calls == ["南京 food"]


def test_the_pool_never_returns_the_same_photo_twice_in_one_pass(monkeypatch) -> None:
    pool = [
        {
            "url": f"https://images.pexels.com/photos/{n}/pexels-photo-{n}.jpeg",
            "photographer": "A",
            "page_url": f"https://www.pexels.com/photo/{n}/",
        }
        for n in (1, 2, 3)
    ]
    _enable_pexels(monkeypatch, photos=[], pool=pool)

    resolved = place_images.resolve_place_images(
        [{"name": n, "type": "hotel"} for n in ("南京涵碧楼酒店", "南京金鹰国际酒店", "南京中心大酒店")],
        "南京",
    )

    urls = [place["image_url"] for place in resolved]
    assert len(set(urls)) == 3, urls


def test_an_unreachable_photo_is_released_for_another_place(monkeypatch) -> None:
    """A claim is given back when the probe fails, so the photo is not wasted."""
    dead = "https://images.pexels.com/photos/1/dead.jpeg"
    alive = "https://images.pexels.com/photos/2/alive.jpeg"
    monkeypatch.setenv("PEXELS_API_KEY", "test-key")
    monkeypatch.setattr(
        place_images,
        "search_pexels_photos",
        lambda query, limit=15, timeout=8.0: [
            {"url": dead, "photographer": "A", "page_url": "https://www.pexels.com/photo/1/"},
            {"url": alive, "photographer": "B", "page_url": "https://www.pexels.com/photo/2/"},
        ],
    )
    monkeypatch.setattr(place_images, "_is_reachable", lambda url: url == alive)
    monkeypatch.setattr(place_images, "host_resolves", lambda host, budget=2.5: True)
    place_images.clear_image_cache()

    resolved = place_images.resolve_place_images([{"name": "夫子庙", "type": "attraction"}], "南京")

    assert resolved[0]["image_url"] == alive
