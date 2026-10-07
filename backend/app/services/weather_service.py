from datetime import date, timedelta
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
import json

from app.services import destination_service


# Open-Meteo publishes a 16 day forecast. Asking for a date beyond that returns
# HTTP 400, which used to surface to the user as "天气预报暂时不可用：HTTPError".
FORECAST_WINDOW_DAYS = 16
USER_AGENT = "travel-agent-platform/1.0"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search?name=%s&count=8&language=zh&format=json"


def _unavailable(destination: str, message: str) -> dict:
    return {"destination": destination, "days": [], "status": "unavailable", "message": message}


def _parse_day(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _geocode(candidate: str) -> list[dict]:
    """Every place the geocoder offers for one spelling; [] when it offers nothing."""
    request = Request(GEOCODING_URL % quote(candidate), headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=5) as response:
        geo = json.load(response)
    return list(geo.get("results") or [])


def _resolve(destination: str) -> dict | None:
    """Resolve the destination to a place to forecast, offline first, geocoder second.

    The offline table is measured data (see ``destination_service``): for anything it knows it
    is both faster and correct, where the geocoder happily answers "东京" with a village in
    Jiangsu and "New York" with a village in Nebraska. The geocoder is only asked about names
    the table does not know, and its answer is accepted only from the country the user named.
    """
    place = destination_service.resolve_offline(destination)
    named_country = destination_service.expected_country(destination)
    wrong_country = bool(
        place
        and named_country
        and destination_service.normalize_country(place["country"])
        != destination_service.normalize_country(named_country)
    )
    if place is not None and not wrong_country:
        return place

    for candidate in destination_service.geocode_candidates(destination):
        try:
            results = _geocode(candidate)
        except (HTTPError, URLError, TimeoutError, ValueError, TypeError):
            continue
        chosen = destination_service.choose_result(results, named_country)
        if chosen is not None:
            return chosen
    # A name that contradicts the country the user named still beats no forecast at all.
    return place


def get_weather_forecast(destination: str, start_date: str, end_date: str) -> dict:
    """Fetch a no-key Open-Meteo forecast, returning an actionable fallback on failure."""
    start = _parse_day(start_date)
    end = _parse_day(end_date) or start
    if start is None:
        return _unavailable(destination, "旅行日期格式无法识别，暂时无法查询天气。")

    today = date.today()
    window_end = today + timedelta(days=FORECAST_WINDOW_DAYS)
    if start > window_end:
        days_away = (start - today).days
        return _unavailable(
            destination,
            f"出行日期（{start_date}）距今天还有 {days_away} 天，超出天气服务能提供的未来 "
            f"{FORECAST_WINDOW_DAYS} 天预报范围。请在出发前 {FORECAST_WINDOW_DAYS} 天内重新生成规划，"
            "届时这里会显示每天的天气和室内/户外建议。",
        )

    # A trip that straddles the window still gets the days that are already known.
    clamped_end = min(end, window_end)
    truncated = clamped_end < end
    try:
        place = _resolve(destination)
        if place is None:
            return _unavailable(
                destination,
                f"天气服务没有找到「{destination}」这个地点，暂时无法查询天气。"
                "把目的地改写成单个城市名（例如「东京」或「Tokyo」）后重新生成即可。",
            )
        forecast_request = Request(
            "https://api.open-meteo.com/v1/forecast?latitude=%s&longitude=%s&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,windspeed_10m_max&timezone=auto&start_date=%s&end_date=%s"
            % (place["latitude"], place["longitude"], start.isoformat(), clamped_end.isoformat()),
            headers={"User-Agent": USER_AGENT},
        )
        with urlopen(forecast_request, timeout=5) as response:
            daily = json.load(response).get("daily", {})
    except HTTPError as error:
        return _unavailable(
            destination,
            f"天气服务拒绝了这次查询（HTTP {error.code}）。出行日期可能超出预报范围，请稍后或临近出发日期重试。",
        )
    except (URLError, TimeoutError):
        return _unavailable(destination, "无法连接天气服务（网络异常），请检查网络后重试。")
    except (ValueError, KeyError, TypeError):
        return _unavailable(destination, "天气服务返回的数据无法解析，请稍后重试。")

    days = []
    for index, day in enumerate(daily.get("time", [])):
        code = int((daily.get("weather_code") or [0])[index])
        precipitation = int((daily.get("precipitation_probability_max") or [0])[index] or 0)
        wind = float((daily.get("windspeed_10m_max") or [0])[index] or 0)
        indoor = code >= 51 or precipitation >= 60 or wind >= 45
        entry = {
            "date": day,
            "weather_code": code,
            "precipitation_probability": precipitation,
            "indoor_priority": indoor,
            "condition": "降雨或恶劣天气" if indoor else "适合户外活动",
        }
        # Open-Meteo occasionally omits a temperature; leaving the key out stops the page from
        # rendering "null–null℃" (the panel checks for a missing value, not for null).
        maximum = (daily.get("temperature_2m_max") or [None])[index]
        minimum = (daily.get("temperature_2m_min") or [None])[index]
        if maximum is not None:
            entry["max_temperature"] = maximum
        if minimum is not None:
            entry["min_temperature"] = minimum
        days.append(entry)

    if not days:
        return _unavailable(destination, "天气服务暂时没有返回可用数据，请稍后重试。")

    result = {
        "destination": destination,
        "resolved_destination": place["label"],
        "resolved_country": place["country"],
        "days": days,
        "status": "completed",
        "source": "Open-Meteo",
    }
    notes: list[str] = []
    if place.get("country_only"):
        notes.append(f"「{destination}」是一个国家，这里按首都 {place['label']} 的天气查询。")
    elif place.get("source") == "geocoder" and place["label"] and place["label"] not in destination:
        notes.append(f"天气按解析出的地点「{place['label']}」查询。")
    if truncated:
        notes.append(
            f"出行日期超出未来 {FORECAST_WINDOW_DAYS} 天预报范围，这里只显示到 {clamped_end.isoformat()} 的天气；"
            "临近出发日期重新生成可获得完整预报。"
        )
    if notes:
        result["message"] = "".join(notes)
    return result
