from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy import Date, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base

if TYPE_CHECKING:
    from app.models.itinerary_item import ItineraryItem
    from app.models.trip import Trip


class Place(Base):
    __tablename__ = "places"

    id: Mapped[int] = mapped_column(primary_key=True)
    trip_id: Mapped[int] = mapped_column(ForeignKey("trips.id"), index=True)
    name: Mapped[str] = mapped_column(String(160))
    type: Mapped[str] = mapped_column(String(20))
    address: Mapped[str] = mapped_column(String(300))
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    image_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_level: Mapped[int | None] = mapped_column(Integer, nullable=True)
    opening_hours: Mapped[str | None] = mapped_column(String(300), nullable=True)
    source_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    # Why this place was chosen (the recommendation reason) and the price the
    # recommendation quoted. Both used to be thrown away when a ticked
    # recommendation became a saved place, so 收藏地点 could no longer explain itself.
    note: Mapped[str | None] = mapped_column(String(600), nullable=True)
    estimated_cost: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # How many times this place is used: nights at this hotel, meals at this restaurant.
    # `estimated_cost` is the total for that many uses, so the pair survives a re-plan and
    # 收藏地点 can still say "3 晚" instead of just showing a number.
    quantity: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # The day the recommendation panel suggested for this place. Ticking an item in a
    # specific day group used to lose that day, and auto-scheduling then put the place
    # on whatever day happened to be free first.
    preferred_date: Mapped[date | None] = mapped_column(Date, nullable=True)

    trip: Mapped["Trip"] = relationship(back_populates="places")
    itinerary_items: Mapped[list["ItineraryItem"]] = relationship(
        back_populates="place", cascade="all, delete-orphan"
    )
