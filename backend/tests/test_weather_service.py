from datetime import date, timedelta
from urllib.error import HTTPError, URLError

import pytest

from app.services import weather_service


def _fake_urlopen(payloads: list[dict]):
    """Serve the geocoding and forecast responses in order."""
    call = {"index": 0}

    class _Response:
        def __init__(self, payload: dict) -> None:
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            import json

            return json.dumps(self._payload).encode()

    def _open(request, timeout=None):
        payload = payloads[min(call["index"], len(payloads) - 1)]
        call["index"] += 1
        return _Response(payload)

    return _open


_GEO = {"results": [{"latitude": 31.22, "longitude": 121.46, "name": "上海"}]}
_DAILY = {
    "daily": {
        "time": ["2026-09-12", "2026-09-13"],
        "weather_code": [0, 61],
        "temperature_2m_max": [28.0, 24.0],
        "temperature_2m_min": [20.0, 19.0],
        "precipitation_probability_max": [10, 80],
        "windspeed_10m_max": [12.0, 20.0],
    }
}


def test_weather_reports_a_friendly_message_beyond_the_forecast_window(monkeypatch) -> None:
    """Open-Meteo returns HTTP 400 past 16 days; the user must not see "HTTPError"."""
    far = (date.today() + timedelta(days=40)).isoformat()

    def _must_not_call(*args, **kwargs):
        raise AssertionError("no network call should be attempted beyond the window")

    monkeypatch.setattr(weather_service, "urlopen", _must_not_call)

    result = weather_service.get_weather_forecast("上海", far, far)

    assert result["status"] == "unavailable"
    assert result["days"] == []
    assert "HTTPError" not in result["message"]
    assert "16" in result["message"]
    assert far in result["message"]


def test_weather_returns_days_inside_the_window(monkeypatch) -> None:
    near = (date.today() + timedelta(days=1)).isoformat()
    # 上海 is in the destination table, so the geocoder is never asked: one call, the forecast.
    monkeypatch.setattr(weather_service, "urlopen", _fake_urlopen([_DAILY]))

    result = weather_service.get_weather_forecast("上海", near, near)

    assert result["status"] == "completed"
    assert len(result["days"]) == 2
    assert result["days"][0]["condition"] == "适合户外活动"
    # Rain code 61 and 80% precipitation must be flagged for indoor planning.
    assert result["days"][1]["indoor_priority"] is True


def test_weather_truncates_a_trip_that_crosses_the_window(monkeypatch) -> None:
    start = (date.today() + timedelta(days=1)).isoformat()
    end = (date.today() + timedelta(days=60)).isoformat()
    monkeypatch.setattr(weather_service, "urlopen", _fake_urlopen([_DAILY]))

    result = weather_service.get_weather_forecast("上海", start, end)

    assert result["status"] == "completed"
    assert "message" in result
    assert "16" in result["message"]


def test_weather_explains_an_unknown_destination(monkeypatch) -> None:
    near = (date.today() + timedelta(days=1)).isoformat()
    monkeypatch.setattr(weather_service, "urlopen", _fake_urlopen([{"results": []}]))

    result = weather_service.get_weather_forecast("AgentTestCity", near, near)

    assert result["status"] == "unavailable"
    assert "AgentTestCity" in result["message"]


def test_weather_reports_an_http_error_without_leaking_the_class_name(monkeypatch) -> None:
    near = (date.today() + timedelta(days=1)).isoformat()

    def _boom(request, timeout=None):
        raise HTTPError(request.full_url, 400, "Bad Request", {}, None)

    monkeypatch.setattr(weather_service, "urlopen", _boom)

    result = weather_service.get_weather_forecast("上海", near, near)

    assert result["status"] == "unavailable"
    assert "HTTPError" not in result["message"]
    assert "400" in result["message"]


def test_weather_reports_a_network_failure(monkeypatch) -> None:
    near = (date.today() + timedelta(days=1)).isoformat()

    def _boom(request, timeout=None):
        raise URLError("no route to host")

    monkeypatch.setattr(weather_service, "urlopen", _boom)

    result = weather_service.get_weather_forecast("上海", near, near)

    assert result["status"] == "unavailable"
    assert "URLError" not in result["message"]


@pytest.mark.parametrize("bad", ["", "not-a-date", "2026-13-45"])
def test_weather_handles_an_unparseable_date(monkeypatch, bad: str) -> None:
    monkeypatch.setattr(weather_service, "urlopen", lambda *a, **k: None)

    result = weather_service.get_weather_forecast("上海", bad, bad)

    assert result["status"] == "unavailable"
    assert result["days"] == []
