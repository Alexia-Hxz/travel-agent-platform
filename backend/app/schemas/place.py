from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class PlaceType(StrEnum):
    ATTRACTION = "attraction"
    HOTEL = "hotel"
    RESTAURANT = "restaurant"
    AIRPORT = "airport"
    STATION = "station"


class PlaceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    type: PlaceType
    address: str = Field(min_length=1, max_length=300)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    image_url: HttpUrl | None = None
    rating: float | None = Field(default=None, ge=0, le=5)
    price_level: int | None = Field(default=None, ge=0, le=5)
    opening_hours: str | None = Field(default=None, max_length=300)
    source_url: HttpUrl | None = None
    note: str | None = Field(default=None, max_length=600)
    estimated_cost: int | None = Field(default=None, ge=0)
    # How many times this place is used (nights at a hotel, meals at a restaurant).
    # `estimated_cost` is the total for this many uses.
    quantity: int | None = Field(default=None, ge=1, le=99)
    # Which day of the trip this place is intended for. Set when a ticked
    # recommendation carries the day group it came from; optional for manual entry.
    preferred_date: date | None = None


class PlaceUpdate(BaseModel):
    """Partial edit of a saved place.

    This exists so the price and the count stop being write-once: the recommendation
    cards let you set "住几晚 / 吃几次" and the price that follows from it, but once the
    place was saved there was no way to correct either, and the editor's budget could
    not be fixed without deleting the place and adding it again.
    """

    name: str | None = Field(default=None, min_length=1, max_length=160)
    type: PlaceType | None = None
    address: str | None = Field(default=None, min_length=1, max_length=300)
    image_url: HttpUrl | None = None
    rating: float | None = Field(default=None, ge=0, le=5)
    price_level: int | None = Field(default=None, ge=0, le=5)
    opening_hours: str | None = Field(default=None, max_length=300)
    note: str | None = Field(default=None, max_length=600)
    estimated_cost: int | None = Field(default=None, ge=0)
    quantity: int | None = Field(default=None, ge=1, le=99)
    preferred_date: date | None = None


class PlaceRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    trip_id: int
    name: str
    type: PlaceType
    address: str
    latitude: float | None
    longitude: float | None
    image_url: str | None
    rating: float | None
    price_level: int | None
    opening_hours: str | None
    source_url: str | None
    note: str | None = None
    estimated_cost: int | None = None
    quantity: int | None = None
    preferred_date: date | None = None
