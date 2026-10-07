from datetime import date, time

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.place import PlaceRead


class ItineraryItemCreate(BaseModel):
    date: date
    start_time: time
    end_time: time
    place_id: int
    note: str | None = Field(default=None, max_length=500)
    sort_order: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_time_range(self) -> "ItineraryItemCreate":
        if self.end_time < self.start_time:
            raise ValueError("结束时间不能早于开始时间")
        return self


class ItineraryItemRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    trip_id: int
    date: date
    start_time: time
    end_time: time
    place_id: int
    note: str | None
    sort_order: int
    place: PlaceRead


class ProposedItineraryItem(BaseModel):
    """A slot the app suggests for a saved place that is not scheduled yet."""

    place_id: int
    place_name: str
    date: str
    start_time: str
    end_time: str
    note: str


class SkippedPlace(BaseModel):
    place_id: int
    place_name: str
    reason: str


class AutoScheduleRead(BaseModel):
    trip_id: int
    proposed: list[ProposedItineraryItem] = Field(default_factory=list)
    skipped: list[SkippedPlace] = Field(default_factory=list)
    already_scheduled: list[str] = Field(default_factory=list)
    message: str = ""
