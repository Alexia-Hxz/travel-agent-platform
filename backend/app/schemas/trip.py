from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TripCreate(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    destination: str = Field(min_length=1, max_length=120)
    start_date: date
    end_date: date
    budget: int | None = Field(default=None, ge=0)
    adult_male_count: int = Field(default=0, ge=0)
    adult_female_count: int = Field(default=0, ge=0)
    child_male_count: int = Field(default=0, ge=0)
    child_female_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_date_range(self) -> "TripCreate":
        if self.end_date < self.start_date:
            raise ValueError("结束日期不能早于出发日期")
        return self


class TripRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    destination: str
    start_date: date
    end_date: date
    budget: int | None
    adult_male_count: int
    adult_female_count: int
    child_male_count: int
    child_female_count: int
    created_at: datetime
