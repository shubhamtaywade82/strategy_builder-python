from __future__ import annotations
from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from strategy_api.config import settings
from strategy_api.db.models import Base


def make_engine(url: Optional[str] = None):
    return create_engine(url or f"sqlite:///{settings.sqlite_path}", future=True)


def make_session_factory(engine):
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


_engine = make_engine()
Base.metadata.create_all(_engine)  # creates tables only if missing; preserves existing rows
SessionLocal = make_session_factory(_engine)


def get_session():
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()
