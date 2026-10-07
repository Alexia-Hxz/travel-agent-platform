from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import logging
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import inspect, text

from app.api.deps import require_owned_trip
from app.api.v1.endpoints.auth import router as auth_router
from app.api.v1.endpoints.files import router as files_router
from app.api.v1.endpoints.agent import router as agent_router
from app.api.v1.endpoints.adjustments import router as adjustments_router
from app.api.v1.endpoints.itinerary import router as itinerary_router
from app.api.v1.endpoints.images import router as images_router
from app.api.v1.endpoints.trips import router as trips_router
from app.db.session import Base, engine
from app.services.auth_service import using_development_secret
from app.models import (  # noqa: F401 - imports models for SQLAlchemy metadata
    ItineraryAdjustment,
    ItineraryItem,
    Place,
    Trip,
    TripFile,
    User,
)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    Base.metadata.create_all(bind=engine)
    columns = {column["name"] for column in inspect(engine).get_columns("trip_files")}
    migrations = {"parse_status": "VARCHAR(20) NOT NULL DEFAULT 'pending'", "parsed_text": "TEXT", "parsed_result": "JSON", "parse_error": "TEXT", "parsed_at": "DATETIME"}
    with engine.begin() as connection:
        for name, definition in migrations.items():
            if name not in columns:
                connection.execute(text(f"ALTER TABLE trip_files ADD COLUMN {name} {definition}"))
        trip_columns = {column["name"] for column in inspect(engine).get_columns("trips")}
        for name in ("adult_male_count", "adult_female_count", "child_male_count", "child_female_count"):
            if name not in trip_columns:
                connection.execute(text(f"ALTER TABLE trips ADD COLUMN {name} INTEGER NOT NULL DEFAULT 0"))
        # A ticked recommendation keeps its reason and quoted price when it becomes a
        # saved place, instead of losing them at the moment of saving. It also keeps
        # the day it was suggested for, so auto-scheduling can honour that day.
        place_columns = {column["name"] for column in inspect(engine).get_columns("places")}
        place_migrations = {"note": "VARCHAR(600)", "estimated_cost": "INTEGER", "preferred_date": "DATE", "quantity": "INTEGER"}
        for name, definition in place_migrations.items():
            if name not in place_columns:
                connection.execute(text(f"ALTER TABLE places ADD COLUMN {name} {definition}"))
        # Accounts arrived after trips did. Existing trips keep user_id NULL until the
        # first account logs in and adopts them (auth_service.claim_unowned_trips), so the
        # two trips already in this database stay reachable instead of disappearing.
        if "user_id" not in trip_columns:
            connection.execute(text("ALTER TABLE trips ADD COLUMN user_id INTEGER REFERENCES users(id)"))
            connection.execute(text("CREATE INDEX IF NOT EXISTS ix_trips_user_id ON trips (user_id)"))
    if using_development_secret():
        # Loud on purpose: with the built-in key anyone who reads the source can mint a
        # token for any account. Fine for local development, not for anything else.
        logging.getLogger("uvicorn.error").warning(
            "AUTH_SECRET_KEY is not set: signing access tokens with the built-in "
            "development secret. Set AUTH_SECRET_KEY before exposing this service."
        )
    yield


app = FastAPI(
    title="智能旅行规划平台",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth_router, prefix="/api/v1")
app.include_router(images_router, prefix="/api/v1")
app.include_router(trips_router, prefix="/api/v1")
# These four routers exist only under /trips/{trip_id}/... so one guard each covers every
# route in them, including any route added later. `trips_router` above cannot use this
# because it also serves /trips itself, where there is no trip to own.
OWNED_TRIP_DEPENDENCY = [Depends(require_owned_trip)]
app.include_router(itinerary_router, prefix="/api/v1", dependencies=OWNED_TRIP_DEPENDENCY)
app.include_router(files_router, prefix="/api/v1", dependencies=OWNED_TRIP_DEPENDENCY)
app.include_router(agent_router, prefix="/api/v1", dependencies=OWNED_TRIP_DEPENDENCY)
app.include_router(adjustments_router, prefix="/api/v1", dependencies=OWNED_TRIP_DEPENDENCY)
# Bundled category placeholders guarantee every recommended place renders an image.
# The path here must stay in step with place_images.STATIC_IMAGE_PREFIX.
# This mount stays public: an <img src="..."> cannot carry an Authorization header, and
# the placeholder art is not private to anyone.
app.mount("/static", StaticFiles(directory=Path(__file__).resolve().parent / "static"), name="static")


@app.get("/api/v1/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}
