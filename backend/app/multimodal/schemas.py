from datetime import date
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.schemas.place import PlaceType
from app.schemas.place import PlaceRead
from app.schemas.itinerary import ItineraryItemRead


class VisionDocumentType(StrEnum):
    ATTRACTION_PHOTO = "attraction_photo"
    HOTEL_BOOKING = "hotel_booking"
    TRANSPORT_TICKET = "transport_ticket"
    MENU = "menu"
    TRAVEL_GUIDE = "travel_guide"
    OTHER = "other"


class VisionPlaceSuggestion(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    type: PlaceType
    address: str = Field(min_length=1, max_length=300)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    image_url: str | None = Field(default=None, max_length=500)
    rating: float | None = Field(default=None, ge=0, le=5)
    price_level: int | None = Field(default=None, ge=0, le=5)
    opening_hours: str | None = Field(default=None, max_length=300)
    source_url: str | None = Field(default=None, max_length=500)


class VisionItinerarySuggestion(BaseModel):
    date: date
    start_time: str = Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d(?::[0-5]\d)?$")
    end_time: str = Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d(?::[0-5]\d)?$")
    note: str | None = Field(default=None, max_length=500)


class VisionPreview(BaseModel):
    document_type: VisionDocumentType
    summary: str = Field(min_length=1, max_length=1000)
    place: VisionPlaceSuggestion | None = None
    itinerary: VisionItinerarySuggestion | None = None
    reference_number: str | None = Field(default=None, max_length=100)
    raw_text: str | None = None


class VisionConfirmation(BaseModel):
    save_place: bool = False
    save_itinerary: bool = False
    place: VisionPlaceSuggestion | None = None
    itinerary: VisionItinerarySuggestion | None = None

    @field_validator("save_itinerary")
    @classmethod
    def itinerary_requires_place(cls, value: bool, info) -> bool:
        if value and not info.data.get("save_place"):
            raise ValueError("Saving an itinerary item requires saving its suggested place.")
        return value


class VisionConfirmationResult(BaseModel):
    place: PlaceRead | None = None
    itinerary: ItineraryItemRead | None = None


DOCUMENT_TYPE_ALIASES = {
    "景点": VisionDocumentType.ATTRACTION_PHOTO,
    "景点照片": VisionDocumentType.ATTRACTION_PHOTO,
    "酒店": VisionDocumentType.HOTEL_BOOKING,
    "酒店订单": VisionDocumentType.HOTEL_BOOKING,
    "酒店预订": VisionDocumentType.HOTEL_BOOKING,
    "交通票": VisionDocumentType.TRANSPORT_TICKET,
    "交通票据": VisionDocumentType.TRANSPORT_TICKET,
    "菜单": VisionDocumentType.MENU,
    "旅行指南": VisionDocumentType.TRAVEL_GUIDE,
}
PLACE_TYPE_ALIASES = {
    "景点": PlaceType.ATTRACTION,
    "景区": PlaceType.ATTRACTION,
    "酒店": PlaceType.HOTEL,
    "饭店": PlaceType.HOTEL,
    "餐厅": PlaceType.RESTAURANT,
    "餐馆": PlaceType.RESTAURANT,
    "机场": PlaceType.AIRPORT,
    "车站": PlaceType.STATION,
    "火车站": PlaceType.STATION,
    "高铁站": PlaceType.STATION,
}


def _document_type(value: Any) -> VisionDocumentType:
    if isinstance(value, VisionDocumentType):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in VisionDocumentType._value2member_map_:
            return VisionDocumentType(normalized)
        return DOCUMENT_TYPE_ALIASES.get(value.strip(), VisionDocumentType.OTHER)
    return VisionDocumentType.OTHER


def _place_type(value: Any) -> PlaceType | None:
    if isinstance(value, PlaceType):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in PlaceType._value2member_map_:
            return PlaceType(normalized)
        return PLACE_TYPE_ALIASES.get(value.strip())
    return None


def normalize_vision_result(payload: dict[str, Any]) -> VisionPreview:
    """Convert a model response into a strict, user-reviewable preview."""
    place = payload.get("place") or None
    if isinstance(place, dict):
        place = dict(place)
        required_place_fields = {"name", "type", "address"}
        if not required_place_fields.issubset(place) or not all(place.get(field) for field in required_place_fields):
            place = None
        else:
            place["type"] = _place_type(place.get("type"))
            for field in ("latitude", "longitude", "rating"):
                value = place.get(field)
                if value is not None:
                    try:
                        place[field] = float(value)
                    except (TypeError, ValueError):
                        place[field] = None
            value = place.get("price_level")
            if value is not None:
                try:
                    place["price_level"] = int(value)
                except (TypeError, ValueError):
                    place["price_level"] = None
            try:
                place = VisionPlaceSuggestion.model_validate(place)
            except ValidationError:
                place = None
    else:
        place = None
    itinerary = payload.get("itinerary") or None
    if not isinstance(itinerary, dict) or not all(itinerary.get(field) for field in ("date", "start_time", "end_time")):
        itinerary = None
    else:
        try:
            itinerary = VisionItinerarySuggestion.model_validate(itinerary)
        except ValidationError:
            itinerary = None
    return VisionPreview(
        document_type=_document_type(payload.get("document_type")),
        summary=payload.get("summary") or "No travel details could be extracted.",
        place=place,
        itinerary=itinerary,
        reference_number=payload.get("reference_number") or None,
        raw_text=payload.get("raw_text") or None,
    )
