from app.agents.tools import place_images


def _photos(*urls: str, alt: str = "") -> list[dict]:
    """Photo records in the shape the library returns them."""
    return [{"url": url, "alt": alt, "page_url": None} for url in urls]


def test_resolve_place_images_keeps_an_image_the_place_already_has(monkeypatch) -> None:
    queries: list[str] = []

    def fake_search_photos(query: str, limit: int = 3, timeout: float = 8.0) -> list[dict]:
        queries.append(query)
        return []

    monkeypatch.setattr(place_images, "search_pexels_photos", fake_search_photos)
    place_images.clear_image_cache()

    resolved = place_images.resolve_place_images(
        [{"name": "外滩", "type": "attraction", "image_url": "https://images.example.com/bund.jpg"}],
        "上海",
    )

    assert resolved[0]["image_url"] == "https://images.example.com/bund.jpg"
    assert queries == []


def test_a_place_is_searched_by_its_english_keyword_not_its_chinese_name(monkeypatch) -> None:
    """The library is indexed in English, so the English keyword is what is searched.

    This is the whole fix for relevance: the Chinese name searched as-is returned a photo of
    Osaka Castle for three separate Tokyo museums, and "豚骨拉面 food" returned snow-covered
    horse saddles.
    """
    queries: list[str] = []

    def fake_search_photos(query: str, limit: int = 15, timeout: float = 8.0) -> list[dict]:
        queries.append(query)
        return _photos(f"https://images.example.com/{len(queries)}.jpg", alt=f"{query} photo")

    monkeypatch.setattr(place_images, "search_pexels_photos", fake_search_photos)
    place_images.clear_image_cache()

    resolved = place_images.resolve_place_images(
        [
            {"name": "浅草寺", "type": "attraction", "image_query": "Sensoji Temple"},
            {"name": "东京国立博物馆", "type": "attraction", "image_query": "Tokyo National Museum"},
        ],
        "日本东京",
    )

    assert queries == ["Sensoji Temple landmark", "Tokyo National Museum landmark"]
    assert all(place["image_url"].startswith("https://images.example.com/") for place in resolved)


def test_a_candidate_that_does_not_match_the_keyword_is_rejected(monkeypatch) -> None:
    """This is what stops the absurd mismatches.

    Pexels returns its own ranking first and that ranking is regularly the wrong subject:
    the top hit for a tonkotsu ramen query was titled "snow-covered horse saddles on wooden
    rack". A candidate whose own description does not mention the dish is refused, so the
    card falls through to a placeholder instead of showing a saddle.
    """
    monkeypatch.setattr(
        place_images,
        "search_pexels_photos",
        lambda query, limit=15, timeout=8.0: _photos(
            "https://images.example.com/saddles.jpg", alt="snow covered horse saddles on wooden rack"
        ),
    )
    place_images.clear_image_cache()

    resolved = place_images.resolve_place_images(
        [
            {
                "name": "一兰拉面",
                "type": "restaurant",
                "image_query": "tonkotsu ramen",
                "signature_dish": "豚骨拉面",
            }
        ],
        "日本东京",
    )

    assert resolved[0]["image_url"] == "/static/place_images/restaurant_noodle.svg"


def test_the_best_matching_candidate_wins_over_the_first(monkeypatch) -> None:
    """A later candidate that matches the keyword beats the library's own top result."""
    monkeypatch.setattr(
        place_images,
        "search_pexels_photos",
        lambda query, limit=15, timeout=8.0: [
            {"url": "https://images.example.com/first.jpg", "alt": "a city street at dusk", "page_url": None},
            {"url": "https://images.example.com/second.jpg", "alt": "delicious ramen bowl in tokyo booth", "page_url": None},
        ],
    )
    place_images.clear_image_cache()

    resolved = place_images.resolve_place_images(
        [{"name": "一兰拉面", "type": "restaurant", "image_query": "tonkotsu ramen"}], "日本东京"
    )

    assert resolved[0]["image_url"] == "https://images.example.com/second.jpg"


def test_the_page_slug_is_used_when_alt_is_missing(monkeypatch) -> None:
    """Pexels' ``alt`` is sometimes empty; the photographer's slug says the same thing."""
    monkeypatch.setattr(
        place_images,
        "search_pexels_photos",
        lambda query, limit=15, timeout=8.0: [
            {
                "url": "https://images.example.com/ramen.jpg",
                "alt": "",
                "page_url": "https://www.pexels.com/photo/delicious-ramen-bowl-in-tokyo-booth-37014726/",
            }
        ],
    )
    place_images.clear_image_cache()

    resolved = place_images.resolve_place_images(
        [{"name": "一兰拉面", "type": "restaurant", "image_query": "tonkotsu ramen"}], "日本东京"
    )

    assert resolved[0]["image_url"] == "https://images.example.com/ramen.jpg"


def test_a_restaurant_without_a_dish_falls_to_the_city_food_page(monkeypatch) -> None:
    """No dish means no useful per-place query, so the destination's food page is used."""
    monkeypatch.setattr(
        place_images, "search_pexels_photos", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no place query"))
    )
    monkeypatch.setattr(
        place_images.pexels_images,
        "city_pool_photos",
        lambda destination, hint: _photos("https://images.example.com/tokyo-food.jpg"),
    )
    place_images.clear_image_cache()

    resolved = place_images.resolve_place_images([{"name": "某食堂", "type": "restaurant"}], "日本东京")

    assert resolved[0]["image_url"] == "https://images.example.com/tokyo-food.jpg"


def test_food_falls_back_to_the_dish_library_when_the_photo_library_misses(monkeypatch) -> None:
    """Pexels misses tonkatsu and yakitori; the dish library has both.

    Measured over ten Japanese dishes: each library covers dishes the other cannot place,
    which is why both are used rather than one replacing the other.
    """
    monkeypatch.setattr(place_images, "search_pexels_photos", lambda query, limit=15, timeout=8.0: [])
    monkeypatch.setattr(
        place_images.mealdb_images,
        "search_dish_photos",
        lambda query, limit=6: [{"url": f"https://www.themealdb.com/{query}.jpg", "alt": "tonkatsu pork",
                                 "title": "Tonkatsu pork", "page_url": None, "source": "mealdb"}],
    )
    place_images.clear_image_cache()

    resolved = place_images.resolve_place_images(
        [{"name": "某炸猪排店", "type": "restaurant", "image_query": "tonkatsu pork"}], "日本东京"
    )

    assert resolved[0]["image_url"] == "https://www.themealdb.com/tonkatsu pork.jpg"
    assert resolved[0]["image_credit"] == "Tonkatsu pork · TheMealDB"


def test_the_cuisine_page_is_the_last_resort_before_a_placeholder(monkeypatch) -> None:
    """When nothing matches the dish, a photo of the right cuisine still matches the card."""
    monkeypatch.setattr(place_images, "search_pexels_photos", lambda query, limit=15, timeout=8.0: [])
    monkeypatch.setattr(place_images.mealdb_images, "search_dish_photos", lambda query, limit=6: [])
    monkeypatch.setattr(
        place_images.mealdb_images,
        "area_photos",
        lambda area, limit=12: [{"url": "https://www.themealdb.com/japanese.jpg", "alt": "japanese gohan rice",
                                  "title": "Japanese gohan rice", "page_url": None, "source": "mealdb"}],
    )
    place_images.clear_image_cache()

    resolved = place_images.resolve_place_images(
        [{"name": "某鳗鱼饭店", "type": "restaurant", "image_query": "unagi eel bowl"}], "日本东京"
    )

    assert resolved[0]["image_url"] == "https://www.themealdb.com/japanese.jpg"


def test_resolve_place_images_falls_back_to_a_category_placeholder(monkeypatch) -> None:
    place_images.clear_image_cache()

    resolved = place_images.resolve_place_images(
        [
            {"name": "某酒店", "type": "hotel"},
            {"name": "某餐厅", "type": "restaurant"},
            {"name": "某景点", "type": "attraction"},
            {"name": "某地", "type": "unknown"},
        ],
        "上海",
    )

    assert [place["image_url"] for place in resolved] == [
        "/static/place_images/hotel.svg",
        "/static/place_images/restaurant.svg",
        "/static/place_images/attraction.svg",
        "/static/place_images/generic.svg",
    ]


def test_a_food_placeholder_matches_the_dish(monkeypatch) -> None:
    """Even the last resort says the same thing as the card."""
    place_images.clear_image_cache()

    resolved = place_images.resolve_place_images(
        [
            {"name": "某面馆", "type": "restaurant", "signature_dish": "豚骨拉面"},
            {"name": "某寿司店", "type": "restaurant", "signature_dish": "江户前寿司"},
            {"name": "某烧肉店", "type": "restaurant", "signature_dish": "和牛烧肉"},
            {"name": "某甜品店", "type": "restaurant", "signature_dish": "抹茶冰淇淋"},
            {"name": "某盖饭店", "type": "restaurant", "signature_dish": "鳗鱼饭"},
            {"name": "某小吃店", "type": "restaurant", "signature_dish": "生煎包"},
        ],
        "上海",
    )

    assert [place["fallback_image_url"] for place in resolved] == [
        "/static/place_images/restaurant_noodle.svg",
        "/static/place_images/restaurant_sushi.svg",
        "/static/place_images/restaurant_grill.svg",
        "/static/place_images/restaurant_dessert.svg",
        "/static/place_images/restaurant_rice.svg",
        "/static/place_images/restaurant.svg",
    ]


def test_resolve_place_images_normalizes_provider_category_labels(monkeypatch) -> None:
    place_images.clear_image_cache()

    resolved = place_images.resolve_place_images(
        [{"name": "某食肆", "type": "food"}, {"name": "某住宿", "type": "lodging"}],
        "上海",
    )

    assert [place["image_url"] for place in resolved] == [
        "/static/place_images/restaurant.svg",
        "/static/place_images/hotel.svg",
    ]


def test_resolve_place_images_never_raises_when_the_photo_source_fails(monkeypatch) -> None:
    def failing_search_photos(query: str, limit: int = 3, timeout: float = 8.0) -> list[dict]:
        raise RuntimeError("library down")

    monkeypatch.setattr(place_images, "search_pexels_photos", failing_search_photos)
    place_images.clear_image_cache()

    resolved = place_images.resolve_place_images(
        [{"name": "外滩", "type": "attraction", "image_query": "The Bund"}], "上海"
    )

    assert resolved[0]["image_url"] == "/static/place_images/attraction.svg"


def test_resolve_place_images_reuses_cached_lookups(monkeypatch) -> None:
    queries: list[str] = []

    def fake_search_photos(query: str, limit: int = 3, timeout: float = 8.0) -> list[dict]:
        queries.append(query)
        return _photos("https://images.example.com/place.jpg", alt="the bund shanghai")

    monkeypatch.setattr(place_images, "search_pexels_photos", fake_search_photos)
    place_images.clear_image_cache()

    for _ in range(2):
        place_images.resolve_place_images(
            [{"name": "外滩", "type": "attraction", "image_query": "The Bund"}], "上海"
        )

    assert len(queries) == 1


def test_resolve_place_images_reports_a_category_fallback_for_every_place(monkeypatch) -> None:
    monkeypatch.setattr(
        place_images,
        "search_pexels_photos",
        lambda query, limit=3, timeout=8.0: _photos("https://images.example.com/photo.jpg", alt="shanghai hotel"),
    )
    place_images.clear_image_cache()

    resolved = place_images.resolve_place_images(
        [{"name": "某酒店", "type": "hotel"}, {"name": "某景点", "type": "attraction", "image_query": "Yu Garden"}],
        "上海",
    )

    assert [place["fallback_image_url"] for place in resolved] == [
        "/static/place_images/hotel.svg",
        "/static/place_images/attraction.svg",
    ]
    # The hotel shares this pass's photo; the attraction's keyword was not in its
    # description, so it is refused the mismatched photo and keeps its placeholder.
    assert resolved[0]["image_url"] == "https://images.example.com/photo.jpg"
    assert resolved[1]["image_url"] == "/static/place_images/attraction.svg"


def test_resolve_place_images_skips_a_photo_url_that_does_not_answer(monkeypatch) -> None:
    # A CDN that hangs or 404s leaves the browser with a blank <img> that may never
    # fire an error event, so the placeholder must be chosen here instead.
    monkeypatch.setattr(
        place_images,
        "search_pexels_photos",
        lambda query, limit=3, timeout=8.0: _photos("https://dynamic-media-cdn.tripadvisor.com/dead.jpg"),
    )
    monkeypatch.setattr(place_images, "_is_reachable", lambda url: False)
    place_images.clear_image_cache()

    resolved = place_images.resolve_place_images([{"name": "新荣记", "type": "restaurant"}], "上海")

    assert resolved[0]["image_url"] == "/static/place_images/restaurant.svg"
    assert resolved[0]["fallback_image_url"] == "/static/place_images/restaurant.svg"


def test_resolve_place_images_uses_a_later_candidate_when_the_first_is_dead(monkeypatch) -> None:
    monkeypatch.setattr(
        place_images,
        "search_pexels_photos",
        lambda query, limit=3, timeout=8.0: _photos(
            "https://dead.example.com/a.jpg", "https://live.example.com/b.jpg"
        ),
    )
    monkeypatch.setattr(place_images, "_is_reachable", lambda url: "live.example.com" in url)
    place_images.clear_image_cache()

    resolved = place_images.resolve_place_images(
        [{"name": "某酒店", "type": "hotel"}], "上海"
    )

    assert resolved[0]["image_url"] == "https://live.example.com/b.jpg"


def test_is_reachable_rejects_http_errors_and_non_images(monkeypatch) -> None:
    class _Response:
        def __init__(self, status_code: int, content_type: str, chunks: list[bytes] | None = None) -> None:
            self.status_code = status_code
            self.headers = {"content-type": content_type}
            self._chunks = chunks if chunks is not None else [b"\xff\xd8\xff"]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def iter_bytes(self):
            if isinstance(self._chunks, Exception):
                raise self._chunks
            return iter(self._chunks)

    def _stream(status_code: int, content_type: str, chunks=None):
        return lambda *args, **kwargs: _Response(status_code, content_type, chunks)

    monkeypatch.setattr(place_images.httpx, "stream", _stream(200, "image/jpeg"))
    assert place_images.probe_image_url("https://ok.example.com/a.jpg") is True

    monkeypatch.setattr(place_images.httpx, "stream", _stream(404, "image/jpeg"))
    assert place_images.probe_image_url("https://gone.example.com/a.jpg") is False

    monkeypatch.setattr(place_images.httpx, "stream", _stream(200, "text/html"))
    assert place_images.probe_image_url("https://html.example.com/a.jpg") is False


def test_is_reachable_rejects_a_host_that_stalls_after_the_headers(monkeypatch) -> None:
    """Headers alone are not proof: some CDNs answer and never send the image."""

    class _StallingResponse:
        status_code = 200
        headers = {"content-type": "image/jpeg"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def iter_bytes(self):
            raise TimeoutError("read timed out")

    monkeypatch.setattr(place_images.httpx, "stream", lambda *args, **kwargs: _StallingResponse())

    assert place_images.probe_image_url("https://slow.example.com/a.jpg") is False


def test_is_reachable_rejects_an_empty_body(monkeypatch) -> None:
    class _EmptyResponse:
        status_code = 200
        headers = {"content-type": "image/jpeg"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def iter_bytes(self):
            return iter([])

    monkeypatch.setattr(place_images.httpx, "stream", lambda *args, **kwargs: _EmptyResponse())

    assert place_images.probe_image_url("https://empty.example.com/a.jpg") is False


def test_is_reachable_is_false_when_the_request_raises(monkeypatch) -> None:
    def _boom(*args, **kwargs):
        raise OSError("network unreachable")

    monkeypatch.setattr(place_images.httpx, "stream", _boom)

    assert place_images.probe_image_url("https://dead.example.com/a.jpg") is False


def test_resolve_place_images_gives_up_on_a_lookup_that_never_returns(monkeypatch) -> None:
    """A DNS-hanging host must not stall the whole planning request.

    httpx's own timeout does not cover DNS resolution: an unreachable photo CDN
    blocked a single lookup for 112 seconds, which pushed a planning request past
    the reverse proxy timeout.
    """
    import time as _time

    def _never_returns(query: str, limit: int = 3, timeout: float = 8.0) -> list[dict]:
        _time.sleep(30)
        return []

    monkeypatch.setattr(place_images, "search_pexels_photos", _never_returns)
    monkeypatch.setattr(place_images, "IMAGE_PHASE_BUDGET", 0.5)
    place_images.clear_image_cache()

    started = _time.monotonic()
    resolved = place_images.resolve_place_images([{"name": "某酒店", "type": "hotel"}], "上海")
    elapsed = _time.monotonic() - started

    assert elapsed < 10
    # The place still gets a usable placeholder rather than no picture at all.
    assert resolved[0]["image_url"] == "/static/place_images/hotel.svg"


def test_a_lookup_starved_by_the_budget_is_retried_instead_of_being_cached(monkeypatch) -> None:
    """A budget-cancelled lookup says nothing about the photo, so it must not stick.

    Caching it as a miss used to deny that place a real picture for the rest of the
    process lifetime, long after the slow host had recovered.
    """
    import time as _time

    calls: list[str] = []

    def _never_returns(query: str, limit: int = 3, timeout: float = 8.0) -> list[dict]:
        calls.append(query)
        _time.sleep(30)
        return []

    monkeypatch.setattr(place_images, "search_pexels_photos", _never_returns)
    monkeypatch.setattr(place_images, "IMAGE_PHASE_BUDGET", 0.5)
    place_images.clear_image_cache()

    for _ in range(2):
        place_images.resolve_place_images([{"name": "某酒店", "type": "hotel"}], "上海")

    # The starved lookup is not remembered, so the second pass tries again.
    assert len(calls) == 2


def test_a_completed_lookup_is_remembered(monkeypatch) -> None:
    """A real answer (including a real miss) is still cached, to avoid paying twice."""
    calls: list[str] = []

    def _no_results(query: str, limit: int = 3, timeout: float = 8.0) -> list[dict]:
        calls.append(query)
        return []

    monkeypatch.setattr(place_images, "search_pexels_photos", _no_results)
    place_images.clear_image_cache()

    for _ in range(2):
        place_images.resolve_place_images([{"name": "某酒店", "type": "hotel"}], "上海")

    assert len(calls) == 1


def test_resolve_place_images_skips_a_host_that_already_failed(monkeypatch) -> None:
    """A hanging CDN must not be re-probed for every place in the same pass.

    The same dead URL used to cost one probe per card, which spent a whole picture-phase
    budget on one link and left the later cards on a placeholder.
    """
    probes: list[str] = []

    def _probe(url: str) -> bool:
        probes.append(url)
        return False

    monkeypatch.setattr(
        place_images,
        "search_pexels_photos",
        lambda query, limit=3, timeout=8.0: _photos("https://dead-cdn.example.com/a.jpg"),
    )
    monkeypatch.setattr(place_images, "_is_reachable", _probe)
    place_images.clear_image_cache()

    # Three places in ONE pass: the host is probed once and then skipped.
    place_images.resolve_place_images(
        [{"name": f"地点{index}", "type": "hotel"} for index in range(3)], "上海"
    )

    assert len(probes) == 1


def test_resolve_place_images_forgets_a_bad_host_between_passes(monkeypatch) -> None:
    """Host memory is scoped to one pass, so a dead URL cannot blacklist a host.

    Keeping the set for the whole process meant one dead URL on p8.itc.cn caused a
    LIVE photo on that same host to be skipped without ever being probed.
    """
    probes: list[str] = []
    live = "https://p8.itc.cn/live.jpg"

    def _probe(url: str) -> bool:
        probes.append(url)
        return url == live

    monkeypatch.setattr(place_images, "_is_reachable", _probe)
    place_images.clear_image_cache()

    # Pass 1: the only candidate is a dead URL on a host that also serves live photos.
    monkeypatch.setattr(
        place_images,
        "search_pexels_photos",
        lambda query, limit=3, timeout=8.0: _photos("https://p8.itc.cn/dead.jpg"),
    )
    place_images.resolve_place_images([{"name": "某酒店", "type": "hotel"}], "上海")

    # Pass 2: the same host now offers a working photo, and it must be probed.
    monkeypatch.setattr(
        place_images,
        "search_pexels_photos",
        lambda query, limit=3, timeout=8.0: _photos(live),
    )
    resolved = place_images.resolve_place_images([{"name": "另一个酒店", "type": "hotel"}], "上海")

    assert resolved[0]["image_url"] == live


def test_a_probe_that_used_its_whole_budget_does_not_blacklist_the_host(monkeypatch) -> None:
    """A slow-but-working host must not be remembered as dead.

    One real photo needed 14.3 s while PROBE_BUDGET is 6 s. Treating that budget
    expiry as a host failure discarded working photos on that host afterwards.
    """
    probed: list[str] = []
    monkeypatch.setattr(place_images, "_is_reachable", lambda url: probed.append(url) or False)
    # Force the "ran out of budget" branch without really waiting.
    monkeypatch.setattr(place_images, "PROBE_BUDGET", 0.0)
    place_images.clear_image_cache()
    monkeypatch.setattr(
        place_images,
        "search_pexels_photos",
        lambda query, limit=3, timeout=8.0: _photos("https://slow-but-real.example.com/a.jpg"),
    )

    place_images.resolve_place_images([{"name": "某酒店", "type": "hotel"}], "上海")

    assert place_images.host_is_known_bad("https://slow-but-real.example.com/b.jpg") is False


def test_a_failed_probe_does_not_blacklist_the_whole_host(monkeypatch) -> None:
    """One dead URL on a photo CDN says nothing about the next one on the same host.

    Blacklisting the host here wasted a live photo that happened to share it: the second
    card on that CDN was skipped without ever being probed.
    """
    monkeypatch.setattr(place_images, "_is_reachable", lambda url: False)
    monkeypatch.setattr(
        place_images,
        "search_pexels_photos",
        lambda query, limit=3, timeout=8.0: _photos("https://broken.example.com/a.jpg"),
    )
    place_images.clear_image_cache()

    place_images.resolve_place_images([{"name": "某酒店", "type": "hotel"}], "上海")

    assert place_images.host_is_known_bad("https://broken.example.com/b.jpg") is False


def test_a_miss_expires_instead_of_being_cached_forever(monkeypatch) -> None:
    """A remembered "no photo exists" must not outlive the provider's own ranking.

    A permanent negative entry kept a restaurant on the placeholder while a fresh
    process found a working photo for it immediately.
    """
    place_images.clear_image_cache()

    assert place_images._lookup_image("南京", "元景宴", "attraction") is None
    # Cached as a miss, so it is not looked up again straight away...
    assert place_images._cached_image(("南京", "元景宴", "attraction")) == (True, None)
    # ...but the entry carries an expiry, so it CAN come back.
    assert ("南京", "元景宴", "attraction") in place_images._miss_cache
    assert ("南京", "元景宴", "attraction") not in place_images._image_cache


def test_an_expired_miss_is_looked_up_again(monkeypatch) -> None:
    found: list[str] = []

    def _search(query, limit=3, timeout=8.0):
        found.append(query)
        return _photos("https://prod-pics.example.com/michelin.jpg", alt="michelin star restaurant")

    monkeypatch.setattr(place_images, "search_pexels_photos", _search)
    monkeypatch.setattr(place_images, "_is_reachable", lambda url: True)
    monkeypatch.setattr(place_images, "host_resolves", lambda host, budget=2.5: True)
    place_images.clear_image_cache()

    # Record a miss, then pretend its TTL has passed.
    place_images._remember("南京", "元景宴", "restaurant", None)
    with place_images._cache_lock:
        place_images._miss_cache[("南京", "元景宴", "restaurant")] = 0.0

    result = place_images._lookup_image(
        "南京", "元景宴", "restaurant", None, "michelin star restaurant"
    )
    assert result is not None
    assert result["url"] == "https://prod-pics.example.com/michelin.jpg"
    assert len(found) == 1


def test_a_hit_is_still_cached_without_expiry(monkeypatch) -> None:
    monkeypatch.setattr(
        place_images, "search_pexels_photos", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cached"))
    )
    place_images.clear_image_cache()
    place_images._remember("南京", "金陵饭店", "hotel", {"url": "https://cache.marriott.com/a.jpg"})

    assert place_images._lookup_image("南京", "金陵饭店", "hotel") == {
        "url": "https://cache.marriott.com/a.jpg"
    }


def test_an_unresolvable_host_is_skipped_without_being_probed(monkeypatch) -> None:
    """A dead name must not burn the resolver timeout, nor block the next candidate.

    dynamic-media-cdn.tripadvisor.com stalled a probe for ~16 s because httpx
    timeouts do not cover DNS; that one hang starved the other places in the
    picture phase and left them with a placeholder.
    """
    probes: list[str] = []
    checked: list[str] = []

    def _resolves(host: str, budget: float = 2.5) -> bool:
        checked.append(host)
        return host != "dynamic-media-cdn.tripadvisor.com"

    monkeypatch.setattr(place_images, "host_resolves", _resolves)
    monkeypatch.setattr(
        place_images,
        "search_pexels_photos",
        lambda query, limit=3, timeout=8.0: _photos(
            "https://dynamic-media-cdn.tripadvisor.com/dead.jpg",
            "https://pix10.agoda.net/good.jpg",
        ),
    )
    monkeypatch.setattr(place_images, "_is_reachable", lambda url: probes.append(url) or True)
    place_images.clear_image_cache()

    resolved = place_images.resolve_place_images([{"name": "某酒店", "type": "hotel"}], "南京")

    assert resolved[0]["image_url"] == "https://pix10.agoda.net/good.jpg"
    # The dead host was never probed: it was rejected on the name alone.
    assert probes == ["https://pix10.agoda.net/good.jpg"]
    assert "dynamic-media-cdn.tripadvisor.com" in checked


def test_host_resolves_is_bounded_and_cached(monkeypatch) -> None:
    """Resolution is bounded by a wall-clock budget and remembered per host."""
    import time as _time
    import threading as _threading

    started = _threading.Event()

    def _never_resolves(host: str, port: int = 443, proto: int = 0):
        started.set()
        _time.sleep(30)
        return []

    # Exercises the real implementation, which the offline fixture replaces.
    real_resolves = place_images.resolve_host_bounded
    monkeypatch.setattr(place_images.socket, "getaddrinfo", _never_resolves)
    place_images.clear_image_cache()

    begin = _time.monotonic()
    assert real_resolves("never.example.com", budget=0.4) is False
    elapsed = _time.monotonic() - begin

    # Bounded by the budget instead of the much longer resolver timeout.
    assert elapsed < 3
    assert started.is_set()

    # The negative answer is cached, so a second call returns at once.
    begin = _time.monotonic()
    assert real_resolves("never.example.com", budget=0.4) is False
    assert _time.monotonic() - begin < 0.2


def test_probe_is_bounded_by_wall_clock_not_httpx_timeout(monkeypatch) -> None:
    """A stalling image host must not hold a worker far past the probe budget.

    dynamic-media-cdn.tripadvisor.com accepted the request and stalled for 48 s
    even though IMAGE_PROBE_TIMEOUT is 4 s, which starved the rest of the picture
    phase and left those places with a placeholder.
    """
    import time as _time

    def _never_answers(url: str) -> bool:
        _time.sleep(30)
        return True

    monkeypatch.setattr(place_images, "_probe_once", _never_answers)
    monkeypatch.setattr(place_images, "PROBE_BUDGET", 0.5)

    begin = _time.monotonic()
    assert place_images.probe_image_url("https://stalling.example.com/a.jpg") is False
    elapsed = _time.monotonic() - begin

    # Bounded by PROBE_BUDGET, not by the 30 s the probe would have taken.
    assert elapsed < 5


def test_clear_image_cache_forgets_bad_hosts(monkeypatch) -> None:
    probes: list[str] = []

    monkeypatch.setattr(
        place_images,
        "search_pexels_photos",
        lambda query, limit=3, timeout=8.0: _photos("https://dead-cdn.example.com/a.jpg"),
    )
    monkeypatch.setattr(place_images, "_is_reachable", lambda url: probes.append(url) or False)
    place_images.clear_image_cache()

    place_images.resolve_place_images([{"name": "某酒店", "type": "hotel"}], "上海")
    place_images.clear_image_cache()
    place_images.resolve_place_images([{"name": "某酒店", "type": "hotel"}], "上海")

    assert len(probes) == 2


def test_resolve_place_images_bounds_the_number_of_lookups(monkeypatch) -> None:
    queries: list[str] = []

    def fake_search_photos(query: str, limit: int = 3, timeout: float = 8.0) -> list[dict]:
        queries.append(query)
        return []

    monkeypatch.setattr(place_images, "search_pexels_photos", fake_search_photos)
    place_images.clear_image_cache()

    places = [{"name": f"地点{index}", "type": "hotel"} for index in range(20)]
    resolved = place_images.resolve_place_images(places, "上海", max_lookups=3)

    assert len(queries) == 3
    assert all(place["image_url"] for place in resolved)
