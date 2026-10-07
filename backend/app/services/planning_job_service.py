"""Background planning jobs with polling-friendly stage snapshots.

The planner remains synchronous internally. Running it in a bounded worker pool keeps
that well-tested code path intact while letting the browser poll completed LangGraph
stages and category batches instead of holding one HTTP request open for minutes.
"""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
import os
import logging
import threading
import time
from typing import Any
from uuid import uuid4

from sqlalchemy.orm import Session

from app.agents.workflows.planner import build_planner_graph
from app.models.trip import Trip
from app.services.agent_service import build_plan_state
from app.services.photo_cache import cache_plan_photos


logger = logging.getLogger(__name__)
JOB_TTL_SECONDS = 60 * 60
_executor = ThreadPoolExecutor(max_workers=max(1, int(os.getenv("PLANNING_JOB_WORKERS", "2"))))
_lock = threading.RLock()


@dataclass
class PlanningJob:
    job_id: str
    trip_id: int
    status: str = "queued"
    phase: str = "queued"
    progress: int = 0
    events: list[dict[str, str]] = field(default_factory=list)
    category_recommendations: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None
    updated_at: float = field(default_factory=time.time)


_jobs: dict[str, PlanningJob] = {}

_PHASES = {
    "context_loader": ("context", 8),
    "live_search": ("search", 18),
    "place_recommender": ("recommendations", 68),
    "conflict_checker": ("analysis", 76),
    "budget_estimator": ("analysis", 80),
    "budget_planner": ("analysis", 84),
    "weather_forecast": ("weather", 88),
    "weather_advisor": ("weather", 90),
    "final_summary_generator": ("summary", 96),
}


def _cleanup_locked() -> None:
    cutoff = time.time() - JOB_TTL_SECONDS
    expired = [job_id for job_id, job in _jobs.items() if job.updated_at < cutoff]
    for job_id in expired:
        _jobs.pop(job_id, None)


def _snapshot(job: PlanningJob) -> dict[str, Any]:
    recommendations = [
        item
        for category in ("hotel", "restaurant", "attraction")
        for item in job.category_recommendations.get(category, [])
    ]
    return {
        "job_id": job.job_id,
        "trip_id": job.trip_id,
        "status": job.status,
        "phase": job.phase,
        "progress": job.progress,
        "events": deepcopy(job.events),
        "place_recommendations": deepcopy(recommendations),
        "result": deepcopy(job.result),
        "error": job.error,
    }


def _record_progress(job_id: str, update: dict[str, Any]) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        if update.get("kind") == "category":
            category = str(update.get("category") or "")
            job.category_recommendations[category] = deepcopy(update.get("recommendations") or [])
            completed = len(job.category_recommendations)
            job.phase = f"category:{category}"
            job.progress = max(job.progress, 22 + completed * 12)
        job.updated_at = time.time()


def _run(job_id: str, state: dict[str, Any]) -> None:
    with _lock:
        job = _jobs[job_id]
        job.status = "running"
        job.phase = "context"
        job.progress = 2
        job.updated_at = time.time()
    try:
        final_state: dict[str, Any] = state
        for snapshot in build_planner_graph().stream(state, stream_mode="values"):
            final_state = snapshot
            events = snapshot.get("events") or []
            with _lock:
                job = _jobs[job_id]
                job.events = deepcopy(events)
                if events:
                    phase, progress = _PHASES.get(events[-1].get("tool", ""), (job.phase, job.progress))
                    job.phase = phase
                    job.progress = max(job.progress, progress)
                job.updated_at = time.time()
        response = final_state["response"]
        with _lock:
            job = _jobs[job_id]
            job.phase = "photos"
            job.progress = 98
            job.updated_at = time.time()
        trip = state.get("trip") or {}
        result = cache_plan_photos(
            response,
            recover_missing=bool(state.get("include_web")),
            destination=str(trip.get("destination") or ""),
        )
        with _lock:
            job = _jobs[job_id]
            job.status = "completed"
            job.phase = "completed"
            job.progress = 100
            job.result = result
            job.events = deepcopy(result.get("events") or job.events)
            job.updated_at = time.time()
    except Exception:
        logger.exception("planning job %s failed", job_id)
        with _lock:
            job = _jobs.get(job_id)
            if job is not None:
                job.status = "failed"
                job.phase = "failed"
                job.error = "规划生成失败，请稍后重试。"
                job.updated_at = time.time()


def start_plan_job(
    session: Session,
    trip: Trip,
    query: str,
    include_web: bool,
    selected_recommendations: list[dict] | None = None,
    adjustment_summary: str | None = None,
) -> dict[str, Any]:
    job_id = uuid4().hex
    state = build_plan_state(
        session,
        trip,
        query,
        include_web,
        selected_recommendations=selected_recommendations,
        adjustment_summary=adjustment_summary,
        progress_callback=lambda update: _record_progress(job_id, update),
    )
    with _lock:
        _cleanup_locked()
        job = PlanningJob(job_id=job_id, trip_id=trip.id)
        _jobs[job_id] = job
        snapshot = _snapshot(job)
    _executor.submit(_run, job_id, state)
    return snapshot


def get_plan_job(job_id: str, trip_id: int) -> dict[str, Any] | None:
    with _lock:
        _cleanup_locked()
        job = _jobs.get(job_id)
        if job is None or job.trip_id != trip_id:
            return None
        return _snapshot(job)
