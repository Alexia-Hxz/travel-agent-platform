"""Destination resolution: the geocoder failures this table exists to prevent.

Every case below is a spelling measured against the live Open-Meteo geocoder. The Chinese
names users type either find nothing or find the wrong place -- "东京" is a village in
Jiangsu, "伦敦" is in Ontario, "罗马" is in Queensland -- and the English ones are not safe
either: "New York" is a village in Nebraska and "Bali" is a village in India. These tests pin
the behaviour so a regeneration of the table cannot quietly reintroduce any of it.
"""

from datetime import date, timedelta

import pytest

from app.services import destination_data, destination_service, weather_service


@pytest.mark.parametrize(
    ("typed", "label", "country"),
    [
        # country prefix plus city
        ("日本东京", "东京", "日本"),
        ("日本 东京", "东京", "日本"),
        ("日本大阪", "大阪", "日本"),
        ("韩国首尔", "首尔", "韩国"),
        ("泰国曼谷", "曼谷", "泰国"),
        ("法国巴黎", "巴黎", "法国"),
        ("冰岛雷克雅未克", "雷克雅未克", "冰岛"),
        ("阿联酋迪拜", "迪拜", "阿拉伯联合酋长国"),
        # the bare Chinese name, which the geocoder gets wrong or misses
        ("东京", "东京", "日本"),
        ("大阪", "大阪", "日本"),
        ("首尔", "首尔", "韩国"),
        ("曼谷", "曼谷", "泰国"),
        ("巴黎", "巴黎", "法国"),
        ("伦敦", "伦敦", "英国"),
        ("纽约", "纽约", "美国"),
        ("洛杉矶", "洛杉矶", "美国"),
        ("罗马", "罗马", "意大利"),
        ("米兰", "米兰", "意大利"),
        ("开罗", "开罗", "埃及"),
        ("河内", "河内", "越南"),
        ("胡志明市", "胡志明市", "越南"),
        ("伊斯坦布尔", "伊斯坦布尔", "土耳其"),
        ("苏黎世", "苏黎世", "瑞士"),
        ("奥克兰", "奥克兰", "新西兰"),
        ("悉尼", "悉尼", "澳大利亚"),
        ("雷克雅未克", "雷克雅未克", "冰岛"),
        # administrative suffixes
        ("东京都", "东京", "日本"),
        ("大阪市", "大阪", "日本"),
        # already-English names, including the ones the geocoder answers wrongly
        ("Tokyo", "东京", "日本"),
        ("Osaka", "大阪", "日本"),
        ("New York", "纽约", "美国"),
        ("Reykjavik", "雷克雅未克", "冰岛"),
        ("Seoul", "首尔", "韩国"),
        # domestic destinations must not be "helpfully" moved abroad
        ("南京", "南京", "中国"),
        ("上海", "上海", "中国"),
        ("北京", "北京", "中国"),
        ("三亚", "三亚", "中国"),
        ("西安", "西安", "中国"),
        ("香港", "香港", "中国"),
        ("台北", "台北", "中国台湾"),
    ],
)
def test_the_forms_users_type_resolve_to_the_right_city(typed: str, label: str, country: str) -> None:
    place = destination_service.resolve_offline(typed)

    assert place is not None, f"{typed} should resolve without the geocoder"
    assert place["label"] == label
    assert place["country"] == country
    assert place["source"] == "table"


@pytest.mark.parametrize(
    ("typed", "capital"),
    [("日本", "东京"), ("韩国", "首尔"), ("泰国", "曼谷"), ("法国", "巴黎"), ("冰岛", "雷克雅未克")],
)
def test_a_country_gets_its_capital(typed: str, capital: str) -> None:
    place = destination_service.resolve_offline(typed)

    assert place is not None
    assert place["label"] == capital
    # The weather panel says so, because a forecast for the capital is an interpretation.
    assert place["country_only"] is True


@pytest.mark.parametrize(
    ("typed", "label"),
    [("北海道", "札幌"), ("冲绳", "那霸"), ("巴厘岛", "巴厘岛"), ("圣托里尼", "圣托里尼"), ("夏威夷", "檀香山")],
)
def test_a_region_resolves_to_the_city_its_forecast_belongs_to(typed: str, label: str) -> None:
    place = destination_service.resolve_offline(typed)

    assert place is not None
    assert place["label"] == label


@pytest.mark.parametrize("typed", ["Bali", "bali island", "Hokkaido", "Hawaii", "Santorini", "Okinawa"])
def test_a_foreign_language_alias_resolves_too(typed: str) -> None:
    assert destination_service.resolve_offline(typed) is not None


def test_new_york_is_the_city_and_not_the_nebraska_village() -> None:
    """The geocoder's first answer for "New York" is a village of 7,864 people."""
    place = destination_service.resolve_offline("New York")

    assert place is not None
    assert place["latitude"] > 40.5
    assert place["longitude"] < -73.0


def test_tokyo_is_not_the_village_in_jiangsu() -> None:
    """The geocoder answers "东京" with 32.2/119.3, a village in Jiangsu."""
    place = destination_service.resolve_offline("东京")

    assert place is not None
    assert place["latitude"] > 35.0
    assert place["longitude"] > 139.0


def test_a_name_the_table_does_not_know_has_no_offline_answer() -> None:
    assert destination_service.resolve_offline("意大利五渔村") is None
    assert destination_service.resolve_offline("AgentTestCity") is None
    assert destination_service.resolve_offline("") is None
    assert destination_service.resolve_offline(None) is None


def test_the_geocoder_is_asked_about_the_city_part_and_held_to_the_named_country() -> None:
    candidates = destination_service.geocode_candidates("意大利五渔村")

    assert "五渔村" in candidates
    assert destination_service.expected_country("意大利五渔村") == "意大利"
    assert destination_service.expected_country("五渔村") is None


def test_a_country_spelled_the_geocoders_way_matches_the_users_way() -> None:
    # The user writes 阿联酋, the geocoder answers 阿拉伯联合酋长国.
    assert destination_service.normalize_country("阿联酋") == destination_service.normalize_country(
        "阿拉伯联合酋长国"
    )


def test_the_wrong_country_is_rejected_even_when_it_is_the_first_answer() -> None:
    results = [
        {"name": "东京", "country": "中国", "admin1": "江苏", "latitude": 32.20514, "longitude": 119.28653},
        {"name": "東京", "country": "日本", "admin1": "东京都", "latitude": 35.6895, "longitude": 139.69171},
    ]

    chosen = destination_service.choose_result(results, "日本")

    assert chosen is not None
    assert chosen["latitude"] == 35.6895
    assert destination_service.choose_result(results, "法国") is None


def test_the_biggest_place_wins_when_the_country_alone_cannot_decide() -> None:
    """Country matching is not enough: the Nebraska "New York" is also in 美国."""
    results = [
        {"name": "约克", "country": "美国", "latitude": 40.86807, "longitude": -97.592, "population": 7864},
        {"name": "紐約市", "country": "美国", "latitude": 40.71427, "longitude": -74.00597, "population": 8804190},
    ]

    chosen = destination_service.choose_result(results, "美国")

    assert chosen is not None
    assert chosen["label"] == "紐約市"


def test_a_city_result_beats_a_region_result_of_the_same_size() -> None:
    results = [
        {"name": "区域", "country": "日本", "latitude": 1.0, "longitude": 2.0, "feature_code": "ADM1"},
        {"name": "城市", "country": "日本", "latitude": 3.0, "longitude": 4.0, "feature_code": "PPLC"},
    ]

    chosen = destination_service.choose_result(results, "日本")

    assert chosen is not None
    assert chosen["label"] == "城市"


def test_a_result_without_coordinates_is_not_used() -> None:
    assert destination_service.choose_result([{"name": "无处"}], None) is None


def test_the_table_is_well_formed() -> None:
    """A regenerated table must not contain a row the rest of the code cannot trust."""
    assert len(destination_data.CITY_ROWS) >= 150
    for key, (latitude, longitude, label, country, geocoder_name) in destination_data.CITY_ROWS.items():
        assert -90 <= latitude <= 90, key
        assert -180 <= longitude <= 180, key
        assert label and country and geocoder_name, key
    for alias, target in destination_data.DESTINATION_ALIASES.items():
        assert target in destination_data.CITY_ROWS, f"{alias} -> {target}"
    for prefix, (country, capital) in destination_data.COUNTRY_PREFIXES.items():
        assert country, prefix
        # An empty capital is deliberate: it means no city of that country is in the table, so
        # "孟加拉国" alone has no offline answer and falls through to the geocoder.
        assert capital == "" or capital in destination_data.CITY_ROWS, f"{prefix} -> {capital}"


# --------------------------------------------------------------------- the weather service


def _forecast_only(daily: dict):
    """One response, because a destination the table knows never reaches the geocoder."""
    import json

    call = {"count": 0}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(daily).encode()

    def _open(request, timeout=None):
        call["count"] += 1
        return _Response()

    _open.calls = call
    return _open


def _daily(**overrides) -> dict:
    payload = {
        "time": ["2026-09-12", "2026-09-13"],
        "weather_code": [0, 61],
        "temperature_2m_max": [28.0, 24.0],
        "temperature_2m_min": [20.0, 19.0],
        "precipitation_probability_max": [10, 80],
        "windspeed_10m_max": [12.0, 20.0],
    }
    payload.update(overrides)
    return {"daily": payload}


def test_a_foreign_destination_is_forecast_without_asking_the_geocoder(monkeypatch) -> None:
    """"日本东京" finds nothing in the geocoder; the table answers instead."""
    near = (date.today() + timedelta(days=1)).isoformat()
    opener = _forecast_only(_daily())
    monkeypatch.setattr(weather_service, "urlopen", opener)

    def _must_not_geocode(candidate):
        raise AssertionError("the table already knows this destination")

    monkeypatch.setattr(weather_service, "_geocode", _must_not_geocode)

    result = weather_service.get_weather_forecast("日本东京", near, near)

    assert result["status"] == "completed"
    assert result["resolved_destination"] == "东京"
    assert result["resolved_country"] == "日本"
    # One call: the forecast. The geocoder was never needed.
    assert opener.calls["count"] == 1
    assert len(result["days"]) == 2


def test_a_country_says_which_city_it_used(monkeypatch) -> None:
    near = (date.today() + timedelta(days=1)).isoformat()
    monkeypatch.setattr(weather_service, "urlopen", _forecast_only(_daily()))

    result = weather_service.get_weather_forecast("日本", near, near)

    assert result["status"] == "completed"
    assert result["resolved_destination"] == "东京"
    assert "东京" in result["message"]
    assert "首都" in result["message"]


def test_a_missing_temperature_does_not_become_a_null_temperature(monkeypatch) -> None:
    """The panel printed "null–null℃" because the key existed with a null value."""
    near = (date.today() + timedelta(days=1)).isoformat()
    monkeypatch.setattr(
        weather_service,
        "urlopen",
        _forecast_only(_daily(temperature_2m_max=[28.0, None], temperature_2m_min=[20.0, None])),
    )

    result = weather_service.get_weather_forecast("上海", near, near)

    assert result["status"] == "completed"
    assert result["days"][0]["max_temperature"] == 28.0
    assert "max_temperature" not in result["days"][1]
    assert "min_temperature" not in result["days"][1]
    # The day is still reported: only the numbers are unknown.
    assert result["days"][1]["date"] == "2026-09-13"


def test_an_unknown_destination_still_asks_the_geocoder(monkeypatch) -> None:
    near = (date.today() + timedelta(days=1)).isoformat()
    asked: list[str] = []

    def _geocode(candidate):
        asked.append(candidate)
        return []

    monkeypatch.setattr(weather_service, "_geocode", _geocode)

    result = weather_service.get_weather_forecast("AgentTestCity", near, near)

    assert result["status"] == "unavailable"
    assert asked, "the geocoder should have been asked about an unknown name"
    assert "AgentTestCity" in result["message"]
