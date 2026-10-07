from datetime import date

from sqlalchemy.orm import Session

from app.agents.workflows.planner import build_planner_graph
from app.models.itinerary_item import ItineraryItem
from app.models.place import Place
from app.models.trip import Trip
from app.services.itinerary_service import list_itinerary_items, list_places
from app.services.photo_cache import cache_plan_photos


def _place_to_state(place: Place) -> dict:
    return {
        "id": place.id,
        "name": place.name,
        "type": place.type,
        "address": place.address,
        "latitude": place.latitude,
        "longitude": place.longitude,
        "image_url": place.image_url,
        "rating": place.rating,
        "price_level": place.price_level,
        # Opening hours matter to the summary ("open 09:00-17:00"), so they travel with
        # the place instead of being dropped here.
        "opening_hours": place.opening_hours,
        "source_url": place.source_url,
        # The recollection of why this place was chosen, kept from the recommendation.
        "note": getattr(place, "note", None),
        "quoted_cost": getattr(place, "estimated_cost", None),
        # How many nights/meals that quoted cost covers, so the summary can say "3 晚"
        # rather than only showing a number.
        "quantity": getattr(place, "quantity", None),
    }


def _item_to_state(item: ItineraryItem) -> dict:
    return {
        "id": item.id,
        "date": item.date,
        "start_time": item.start_time,
        "end_time": item.end_time,
        "place_name": item.place.name,
        "image_url": item.place.image_url,
        "note": item.note,
        "sort_order": item.sort_order,
    }


def create_plan(
    session: Session,
    trip: Trip,
    query: str,
    include_web: bool,
    selected_recommendations: list[dict] | None = None,
    adjustment_summary: str | None = None,
) -> dict:
    state = build_plan_state(
        session,
        trip,
        query,
        include_web,
        selected_recommendations=selected_recommendations,
        adjustment_summary=adjustment_summary,
    )
    return run_plan_state(state)


def build_plan_state(
    session: Session,
    trip: Trip,
    query: str,
    include_web: bool,
    selected_recommendations: list[dict] | None = None,
    adjustment_summary: str | None = None,
    progress_callback=None,
) -> dict:
    """Snapshot database-backed inputs before a plan runs in a worker thread."""
    places = list_places(session, trip.id)
    items = list_itinerary_items(session, trip.id)
    traveler_profile = {
        "adult_male_count": trip.adult_male_count,
        "adult_female_count": trip.adult_female_count,
        "child_male_count": trip.child_male_count,
        "child_female_count": trip.child_female_count,
    }
    traveler_profile["total_adults"] = trip.adult_male_count + trip.adult_female_count
    traveler_profile["total_children"] = trip.child_male_count + trip.child_female_count
    traveler_profile["total_people"] = traveler_profile["total_adults"] + traveler_profile["total_children"]
    state = {
        "trip": {
            "id": trip.id,
            "destination": trip.destination,
            "start_date": trip.start_date.isoformat(),
            "end_date": trip.end_date.isoformat(),
            "budget": trip.budget,
        },
        "places": [_place_to_state(place) for place in places],
        "itinerary": [_item_to_state(item) for item in items],
        "query": query,
        "include_web": include_web,
        "selected_recommendations": selected_recommendations or [],
        "adjustment_summary": adjustment_summary,
        "events": [],
        "traveler_profile": traveler_profile,
    }
    if progress_callback is not None:
        state["progress_callback"] = progress_callback
    return state


def run_plan_state(state: dict) -> dict:
    """Run a previously snapshotted planner state synchronously."""
    result = build_planner_graph().invoke(state)
    return cache_plan_photos(
        result["response"],
        recover_missing=bool(state.get("include_web")),
        destination=str(state.get("trip", {}).get("destination") or ""),
    )
