from typing import Any, Callable, TypedDict


class PlannerState(TypedDict, total=False):
    trip: dict[str, Any]
    places: list[dict[str, Any]]
    itinerary: list[dict[str, Any]]
    query: str
    include_web: bool
    web_results: list[dict[str, Any]]
    # Material retrieved from the traveller's own uploaded documents (booked hotels,
    # tickets, confirmations), with its source kept alongside the excerpt so the model
    # can attribute a statement instead of asserting it.
    traveler_documents: list[dict[str, Any]]
    route_context: dict[str, str] | None
    final_summary: str
    conflicts: list[Any]
    budget: Any
    budget_plan: dict[str, Any]
    transport_advice: dict[str, str | None] | None
    sources: list[dict[str, Any]]
    events: list[dict[str, str]]
    response: dict[str, Any]
    traveler_profile: dict[str, Any]
    weather: dict[str, Any]
    weather_advice: list[dict[str, Any]]
    recommended_places: list[dict[str, Any]]
    generated_daily_plan: list[dict[str, Any]]
    place_image_map: dict[str, dict[str, str]]
    selected_recommendations: list[dict[str, Any]]
    adjustment_summary: str | None
    progress_callback: Callable[[dict[str, Any]], None]


class AdjustmentState(TypedDict, total=False):
    trip: dict[str, Any]
    places: list[dict[str, Any]]
    itinerary: list[dict[str, Any]]
    instruction: str
    include_web: bool
    web_results: list[dict[str, Any]]
    # Retrieved from the traveller's uploaded documents, so an instruction about a
    # booked hotel or a ticket can be answered from the actual order.
    traveler_documents: list[dict[str, Any]]
    # Operations the editor built by hand (delete / retime / add). When non-empty they
    # replace the model call entirely.
    requested_operations: list[Any]
    # True when those operations are the whole request and the model should not run.
    structured_only: bool
    # Recommendations from the planning panel, so the editor proposes the places the
    # user was already shown instead of inventing new ones.
    candidates: list[dict[str, Any]]
    operations: list[Any]
    sources: list[dict[str, Any]]
    events: list[dict[str, str]]
    budget: dict[str, Any]
    conflicts: list[dict[str, Any]]
    summary: str
    final_summary: str
    daily_plan: list[dict[str, Any]]
    traveler_profile: dict[str, Any]
    weather: dict[str, Any]
