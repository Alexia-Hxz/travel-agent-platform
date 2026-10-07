"""Cold/warm Tokyo planner benchmark with overlap-aware HTTP timing.

Run from ``backend`` with ``uv run python scripts/benchmark_planner_cache.py``. The
instrumentation wraps ``httpx.Client.send`` exactly where provider calls leave the
process, identifies searching calls from ``enable_search: true``, and classifies them
from the question-specific tail rather than the generic grounded-answer wrapper.
"""

from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.tools import mealdb_images, pexels_images, place_images, recommendations
from app.agents.workflows.planner import build_planner_graph


def _prompt_and_search_flag(request: httpx.Request) -> tuple[str, bool]:
    try:
        body = json.loads(request.content.decode("utf-8"))
    except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
        return "", False
    messages = body.get("messages") or []
    prompt = str(messages[-1].get("content") or "") if messages else ""
    return prompt, body.get("enable_search") is True


def _provider(request: httpx.Request) -> str:
    prompt, is_search = _prompt_and_search_flag(request)
    if is_search:
        if "请列出至少" in prompt and "真实存在的地点原名" in prompt:
            return "category_material"
        if "只写事实，不要输出多余解释" in prompt:
            return "search_snippets"
        if "酒店列表：" in prompt or "餐厅列表：" in prompt:
            return "place_facts"
        if "请给出具体的景点、餐厅、酒店名称和价格等事实信息" in prompt:
            return "search_web"
        return "search_other"
    host = request.url.host or ""
    if "pexels.com" in host:
        return "pexels"
    if "open-meteo.com" in host or "geocoding-api" in host:
        return "weather"
    if request.url.path.endswith("/chat/completions"):
        return "model_without_search"
    return host or "other"


def _merged_seconds(intervals: list[tuple[float, float]]) -> float:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if not merged or start >= merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def _state(progress_callback) -> dict[str, Any]:
    return {
        "trip": {
            "id": 0,
            "destination": "日本东京",
            "start_date": "2026-10-05",
            "end_date": "2026-10-09",
            "budget": 30000,
        },
        "places": [],
        "itinerary": [],
        "query": "规划东京五日游，兼顾经典景点、本地美食和交通方便的住宿。",
        "include_web": True,
        "selected_recommendations": [],
        "adjustment_summary": None,
        "events": [],
        "traveler_profile": {
            "adult_male_count": 1,
            "adult_female_count": 1,
            "child_male_count": 0,
            "child_female_count": 0,
            "total_adults": 2,
            "total_children": 0,
            "total_people": 2,
        },
        "progress_callback": progress_callback,
    }


def main() -> None:
    os.environ.pop("DISABLE_LIVE_WEB_SEARCH", None)
    os.environ.pop("CATEGORY_MATERIAL_CACHE_FORCE_REFRESH", None)
    os.environ.pop("CATEGORY_MATERIAL_CACHE_CLEAR", None)
    benchmark_cache = Path(__file__).resolve().parents[1] / "data" / "benchmark_category_material.sqlite3"
    os.environ["CATEGORY_MATERIAL_CACHE_PATH"] = str(benchmark_cache)
    recommendations.clear_category_material_cache()

    calls: list[dict[str, Any]] = []
    calls_lock = threading.Lock()
    original_send = httpx.Client.send

    def measured_send(self, request, *args, **kwargs):
        provider = _provider(request)
        start = time.perf_counter()
        try:
            return original_send(self, request, *args, **kwargs)
        finally:
            end = time.perf_counter()
            with calls_lock:
                calls.append({"provider": provider, "start": start, "end": end})

    httpx.Client.send = measured_send
    reports = []
    try:
        for run_name in ("cold", "warm"):
            # Isolate the new category-material cache. Existing in-memory photo caches
            # must start equally cold in both runs or their hits would be misreported as
            # planner-cache speedup.
            place_images.clear_image_cache()
            pexels_images.clear_pool_cache()
            mealdb_images.clear_cache()
            calls.clear()
            first_category_at: list[float] = []
            started = time.perf_counter()

            def progress(update: dict[str, Any]) -> None:
                if update.get("kind") == "category" and not first_category_at:
                    first_category_at.append(time.perf_counter())

            result = build_planner_graph().invoke(_state(progress))
            ended = time.perf_counter()
            grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
            counts: dict[str, int] = defaultdict(int)
            for call in calls:
                grouped[call["provider"]].append((call["start"], call["end"]))
                counts[call["provider"]] += 1
            reports.append(
                {
                    "run": run_name,
                    "wall_seconds": round(ended - started, 3),
                    "first_category_seconds": round(first_category_at[0] - started, 3) if first_category_at else None,
                    "recommendation_count": len(result.get("recommended_places") or []),
                    "providers": {
                        provider: {
                            "calls": counts[provider],
                            "merged_seconds": round(_merged_seconds(intervals), 3),
                        }
                        for provider, intervals in sorted(grouped.items())
                    },
                }
            )
    finally:
        httpx.Client.send = original_send

    print(json.dumps(reports, ensure_ascii=False, indent=2))
    for path in (benchmark_cache, Path(f"{benchmark_cache}-shm"), Path(f"{benchmark_cache}-wal")):
        try:
            path.unlink(missing_ok=True)
        except PermissionError:
            # A late worker may still be releasing a Windows file handle. The benchmark
            # data is more important than deleting this dedicated, gitignored temp DB.
            pass


if __name__ == "__main__":
    main()
