from datetime import date, datetime

from sqlalchemy import Date, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.itinerary_adjustment import ItineraryAdjustment
    from app.models.itinerary_item import ItineraryItem
    from app.models.place import Place
    from app.models.trip_file import TripFile


class Trip(Base):
    __tablename__ = "trips"

    id: Mapped[int] = mapped_column(primary_key=True)
    # Nullable on purpose: trips created before accounts existed have no owner. The first
    # account to log in adopts them (``auth_service.claim_unowned_trips``), and after that
    # every trip is created with an owner and is only visible to that account.
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=True
    )
    title: Mapped[str] = mapped_column(String(120))
    destination: Mapped[str] = mapped_column(String(120), index=True)
    start_date: Mapped[date] = mapped_column(Date)
    end_date: Mapped[date] = mapped_column(Date)
    budget: Mapped[int | None] = mapped_column(Integer, nullable=True)
    adult_male_count: Mapped[int] = mapped_column(Integer, default=0)
    adult_female_count: Mapped[int] = mapped_column(Integer, default=0)
    child_male_count: Mapped[int] = mapped_column(Integer, default=0)
    child_female_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    places: Mapped[list["Place"]] = relationship(back_populates="trip", cascade="all, delete-orphan")
    itinerary_items: Mapped[list["ItineraryItem"]] = relationship(
        back_populates="trip", cascade="all, delete-orphan"
    )
    files: Mapped[list["TripFile"]] = relationship(back_populates="trip", cascade="all, delete-orphan")
    adjustments: Mapped[list["ItineraryAdjustment"]] = relationship(
        back_populates="trip", cascade="all, delete-orphan"
    )
