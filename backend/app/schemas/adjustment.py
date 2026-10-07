from datetime import date as Date, time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from app.schemas.place import PlaceCreate


class AdjustmentOperation(BaseModel):
    kind: Literal["remove_item", "add_item", "add_place_and_item", "set_budget", "update_item"]
    item_id: int | None = Field(default=None, ge=1)
    place_id: int | None = Field(default=None, ge=1)
    place: PlaceCreate | None = None
    date: Date | None = None
    start_time: time | None = None
    end_time: time | None = None
    note: str | None = Field(default=None, max_length=500)
    sort_order: int = Field(default=0, ge=0)
    budget_amount: int | None = Field(default=None, gt=0, exclude_if=lambda value: value is None)

    @model_serializer(mode="wrap")
    def serialize_budget_amount(self, handler):
        data = handler(self)
        if self.budget_amount is None:
            data.pop("budget_amount", None)
        return data

    @model_validator(mode="after")
    def validate_shape(self) -> "AdjustmentOperation":
        if self.kind == "remove_item" and self.item_id is None:
            raise ValueError("remove_item requires item_id")
        if self.kind == "add_item" and self.place_id is None:
            raise ValueError("add_item requires place_id")
        if self.kind == "add_place_and_item" and self.place is None:
            raise ValueError("add_place_and_item requires place")
        if self.kind == "update_item" and self.item_id is None:
            raise ValueError("update_item requires item_id")
        if self.kind == "set_budget" and self.budget_amount is None:
            raise ValueError("set_budget requires budget_amount")
        if self.kind == "set_budget" and any(value is not None for value in (self.item_id, self.place_id, self.place, self.date, self.start_time, self.end_time)):
            raise ValueError("set_budget only accepts budget_amount")
        if self.kind not in {"remove_item", "set_budget"} and None in (self.date, self.start_time, self.end_time):
            raise ValueError("itinerary operations require date and times")
        if self.start_time is not None and self.end_time is not None and self.end_time < self.start_time:
            raise ValueError("end_time cannot be earlier than start_time")
        return self


class AdjustmentCandidate(BaseModel):
    """One place from the planning agent, offered to the adjustment as material.

    Sending the plan's own recommendations is what keeps the two panels telling the
    same story: the editor proposes places the user was already shown, with the same
    prices, instead of inventing new ones from a web search.
    """

    name: str = Field(min_length=1, max_length=160)
    category: str = Field(default="attraction", max_length=20)
    reason: str | None = Field(default=None, max_length=600)
    estimated_cost: int | None = Field(default=None, ge=0)
    unit_price: int | None = Field(default=None, ge=0)
    quantity: int | None = Field(default=None, ge=1, le=99)
    quantity_unit: str | None = Field(default=None, max_length=10)
    price_level: int | None = Field(default=None, ge=0, le=5)
    image_url: str | None = Field(default=None, max_length=600)
    signature_dish: str | None = Field(default=None, max_length=120)
    rating: float | None = Field(default=None, ge=0, le=5)


class AdjustmentDraftRequest(BaseModel):
    instruction: str = Field(min_length=1, max_length=500)
    include_web: bool = False
    # Structured editing: the panel can build the operations itself (delete an item,
    # retime one, add a place the user typed).
    operations: list[AdjustmentOperation] | None = Field(default=None, max_length=48)
    # True when `instruction` is only a human-readable echo of `operations` and the user
    # typed nothing of their own, so the model has no independent request to satisfy.
    structured_only: bool = False
    candidates: list[AdjustmentCandidate] | None = Field(default=None, max_length=60)
    # `day_plan` asks for one day to be arranged from scratch rather than for an edit to be
    # interpreted: a button beside a day sends it, with `date` naming that day. `discover`
    # asks for new places that answer the instruction itself, each offered as a candidate the
    # traveller ticks. Absent means the ordinary adjustment, so every existing caller behaves
    # exactly as before.
    mode: Literal["day_plan", "discover"] | None = None
    date: Date | None = None


class AdjustmentApplyRequest(BaseModel):
    operation_indexes: list[int] | None = Field(default=None, min_length=1)
    # The preview is editable, so the user may have changed a price, a count or a time
    # after the draft was stored. These operations, when sent, are what gets applied.
    operations: list[AdjustmentOperation] | None = Field(default=None, max_length=48)


class AdjustmentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    trip_id: int
    instruction: str
    status: Literal["pending", "applied", "rejected"]
    operations: list[AdjustmentOperation]
    summary: str
    sources: list[dict]
    events: list[dict]
    budget: dict
    conflicts: list[dict]
    daily_plan: list[dict] = Field(default_factory=list)
    final_summary: str = ""
    traveler_profile: dict = Field(default_factory=dict)
    weather: dict = Field(default_factory=dict)


class AdjustmentApplyRead(BaseModel):
    id: int
    status: Literal["applied"]
    itinerary_item_ids: list[int]
