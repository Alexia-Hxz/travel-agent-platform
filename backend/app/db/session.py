from collections.abc import Generator
import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


DEFAULT_DATABASE_PATH = Path(__file__).resolve().parents[2] / "travel_agent.db"
# The test suite points this at a throwaway file. Without the override every test
# run wrote its fixtures into the real database, which slowly filled the user's
# trip list with thousands of "行程测试 ab12cd34" style entries.
DATABASE_PATH = Path(os.getenv("TRAVEL_AGENT_DB_PATH") or DEFAULT_DATABASE_PATH)
DATABASE_URL = f"sqlite:///{DATABASE_PATH.as_posix()}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def get_session() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
