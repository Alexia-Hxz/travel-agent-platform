from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_session
from app.schemas.agent import PlanningJobRead, TravelPlanRead, TravelPlanRequest
from app.services.agent_service import create_plan
from app.services.planning_job_service import get_plan_job, start_plan_job
from app.services.trip_service import get_trip


router = APIRouter(prefix="/trips", tags=["travel-planning-agent"])
SessionDependency = Annotated[Session, Depends(get_session)]


@router.post("/{trip_id}/plan", response_model=TravelPlanRead)
def create_travel_plan(
    trip_id: int, payload: TravelPlanRequest, session: SessionDependency
) -> TravelPlanRead:
    trip = get_trip(session, trip_id)
    if trip is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Trip does not exist.")
    return create_plan(
        session,
        trip,
        payload.query,
        payload.include_web,
        selected_recommendations=[
            recommendation.model_dump() for recommendation in payload.selected_recommendations
        ],
        adjustment_summary=payload.adjustment_summary,
    )


@router.post("/{trip_id}/plan-jobs", response_model=PlanningJobRead, status_code=status.HTTP_202_ACCEPTED)
def create_travel_plan_job(
    trip_id: int, payload: TravelPlanRequest, session: SessionDependency
) -> PlanningJobRead:
    trip = get_trip(session, trip_id)
    if trip is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Trip does not exist.")
    return start_plan_job(
        session,
        trip,
        payload.query,
        payload.include_web,
        selected_recommendations=[
            recommendation.model_dump() for recommendation in payload.selected_recommendations
        ],
        adjustment_summary=payload.adjustment_summary,
    )


@router.get("/{trip_id}/plan-jobs/{job_id}", response_model=PlanningJobRead)
def read_travel_plan_job(
    trip_id: int, job_id: str, session: SessionDependency
) -> PlanningJobRead:
    # Resolve the trip first so the existing ownership dependency and 404 semantics apply.
    if get_trip(session, trip_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Trip does not exist.")
    job = get_plan_job(job_id, trip_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Planning job does not exist.")
    return job
