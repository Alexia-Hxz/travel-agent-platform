from datetime import date

from pydantic import BaseModel, Field, HttpUrl


class SelectedRecommendation(BaseModel):
    """A recommendation card the user ticked in the planning agent."""

    name: str = Field(min_length=1, max_length=160)
    category: str = Field(default="attraction", max_length=40)
    estimated_cost: int | None = None
    reason: str | None = Field(default=None, max_length=600)


class TravelPlanRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    include_web: bool = False
    # Carried over from the planning and adjustment panels, so the final summary can
    # cover what the user actually selected instead of only the saved records.
    selected_recommendations: list[SelectedRecommendation] = Field(default_factory=list)
    adjustment_summary: str | None = Field(default=None, max_length=4000)


class PlanItem(BaseModel):
    place_name: str
    start_time: str
    end_time: str
    note: str | None
    image_url: str | None = None
    fallback_image_url: str | None = None


class DailyPlan(BaseModel):
    date: date
    items: list[PlanItem]
    # Set when a slot could not be filled without repeating a place. The plan omits the
    # slot rather than offering the same restaurant or sight twice, so this says why the
    # day looks short instead of leaving the traveller guessing.
    shortage: str | None = None


class BudgetRead(BaseModel):
    total_estimate: int
    trip_budget: int | None
    remaining_budget: int | None
    currency: str
    note: str
    breakdown: list[dict] = Field(default_factory=list)
    daily_estimate: int = 0
    budget_summary: str = ""
    itemized_summary: str = ""
    # Typical accommodation/meal amounts that are deliberately NOT part of total_estimate:
    # those categories are charged only through an actual ticked or saved place. Shown so the
    # reader can see what they would cost without the total double-counting them.
    excluded_allowances: list[dict] = Field(default_factory=list)


class ConflictRead(BaseModel):
    date: date
    first_place: str
    second_place: str
    message: str


class PlanSource(BaseModel):
    source_type: str
    title: str
    url: HttpUrl | None = None
    snippet: str
    score: float | None = None
    image_url: str | None = None


class PlaceRecommendation(BaseModel):
    """One recommended place with its own picture, cost estimate and grouping."""

    name: str
    category: str
    reason: str | None = None
    price_hint: str | None = None
    unit_price: int | None = None
    price_unit: str | None = None
    estimated_cost: int | None = None
    cost_explanation: str | None = None
    # How many times this place is used -- nights at this hotel, meals at this restaurant --
    # and the unit that count is expressed in ("晚" / "次" / null when a count makes no sense,
    # e.g. an attraction). `estimated_cost` is the price for exactly `quantity` uses.
    quantity: int = 1
    quantity_unit: str | None = None
    quantity_max: int | None = None
    # Why the price is what it is ("按检索到的价格估算" / "按当地中档常见价估算" /
    # "资料显示免费" / "资料价低于当地同类常见人均，已按常见人均下限 X 元估计"), kept apart
    # from cost_explanation so the card can rebuild the arithmetic live.
    cost_source: str | None = None
    image_url: str | None = None
    fallback_image_url: str | None = None
    # Set only when the picture came from the Pexels library, which requires that the
    # photographer and Pexels itself are credited wherever the photo is shown.
    image_credit: str | None = None
    image_credit_url: str | None = None
    # Where the place roughly is, so the workspace map can sketch the day's route and the
    # distance between stops. It is the model's estimate rather than a geocoded address, so
    # the map labels the result as a straight-line sketch and never as navigation.
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    # Hotels carry no picture: a stock photo of a different hotel is not information, so a
    # hotel card shows its reference rating and nightly price instead.
    rating: float | None = None
    # The dish a restaurant is known for. It is shown on the card and is also what the
    # card's picture is searched for, because a photo of the food is the point of a food card.
    signature_dish: str | None = None
    # `breakfast` / `lunch` / `dinner` / `any` for restaurants, else null: lets the UI
    # show a whole day of meals instead of one undifferentiated food list.
    meal_slot: str | None = None
    # Set when the trip has a budget and this item's unit price is above what its
    # category can affordably claim. It is a label, never a filter.
    over_budget: bool = False
    budget_limit: int | None = None
    budget_note: str | None = None
    # Which price tier of the traveller's own budget this is: `cheap` / `mid` / `splurge` /
    # `over`, with its Chinese label. Recommendations are a mix by design -- food that is
    # affordable every day AND the expensive one that is worth it -- and the tier is what
    # makes that visible instead of looking like one price point repeated.
    price_tier: str | None = None
    price_tier_label: str | None = None
    # True when the card's price was raised to the local minimum for this kind of place,
    # because the material quoted less than a full meal costs there.
    price_adjusted: bool = False
    # The material said this place costs nothing to enter.
    is_free: bool = False
    # Present only on a polling snapshot. The final response never sets it: early category
    # cards must say that pricing is pending until every category has been analysed.
    price_pending: bool = False


class ToolEvent(BaseModel):
    tool: str
    status: str
    message: str


class TransportAdvice(BaseModel):
    origin: str | None
    destination: str | None
    recommendation: str | None


class PlanHandoff(BaseModel):
    """Echoes which planning and adjustment material the summary was built from."""

    selected_recommendations: list[SelectedRecommendation] = Field(default_factory=list)
    selected_total: int = 0
    has_adjustment_summary: bool = False
    saved_place_count: int = 0
    saved_place_names: list[str] = Field(default_factory=list)


class WeatherAdvice(BaseModel):
    """A day whose weather threatens what is planned, with a way to fix it."""

    date: str
    condition: str
    precipitation_probability: int | None = None
    indoor_priority: bool = True
    scheduled: list[str] = Field(default_factory=list)
    outdoor_items: list[str] = Field(default_factory=list)
    indoor_items: list[str] = Field(default_factory=list)
    message: str = ""
    suggestion: str = ""


class BudgetCategoryRead(BaseModel):
    """One row of the category framework shown before anything has been chosen."""

    category: str
    label: str
    amount: int
    share: float = 0.0
    share_percent: int = 0
    unit: str = ""


class BudgetPlanRead(BaseModel):
    """What the traveller's own budget should roughly be spent on.

    This is an *estimate* derived from the trip's budget (``trip_budget`` × 80%, split by the
    model and normalised so the rows add up exactly). It is deliberately NOT the same as
    ``budget.total_estimate`` (what the saved/ticked places cost) nor as the daily editor's
    figure (what the scheduled rows cost), which remains the authoritative real estimate.
    """

    status: str = "unavailable"
    currency: str = "CNY"
    trip_budget: int | None = None
    planned_total: int = 0
    planned_ratio: float = 0.0
    planned_ratio_percent: int = 0
    reserve_amount: int = 0
    source: str = "none"
    days: int = 0
    nights: int = 0
    people: int = 1
    categories: list[BudgetCategoryRead] = Field(default_factory=list)
    note: str = ""
    summary: str = ""


class TravelPlanRead(BaseModel):
    trip_id: int
    summary: str
    final_summary: str
    recommendations: list[PlanSource]
    place_recommendations: list[PlaceRecommendation] = Field(default_factory=list)
    transport_advice: TransportAdvice | None = None
    daily_plan: list[DailyPlan]
    budget: BudgetRead
    budget_plan: BudgetPlanRead = Field(default_factory=BudgetPlanRead)
    conflicts: list[ConflictRead]
    sources: list[PlanSource]
    events: list[ToolEvent]
    traveler_profile: dict
    weather: dict
    weather_advice: list[WeatherAdvice] = Field(default_factory=list)
    weather_adjustment_request: str = ""
    budget_summary: str = ""
    handoff: PlanHandoff = Field(default_factory=PlanHandoff)


class PlanningJobRead(BaseModel):
    job_id: str
    trip_id: int
    status: str
    phase: str
    progress: int = Field(ge=0, le=100)
    events: list[ToolEvent] = Field(default_factory=list)
    place_recommendations: list[PlaceRecommendation] = Field(default_factory=list)
    result: TravelPlanRead | None = None
    error: str | None = None
