from __future__ import annotations
import time
from typing import Optional
from sqlalchemy import Integer, Text, Float
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _now() -> int:
    return int(time.time())


class Base(DeclarativeBase):
    pass


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[Optional[str]] = mapped_column(Text, default="qwen3.5:4b")
    created_at: Mapped[int] = mapped_column(Integer, nullable=False, default=_now)


class ResearchSession(Base):
    __tablename__ = "research_sessions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    symbol: Mapped[str] = mapped_column(Text, nullable=False)
    rr_config: Mapped[str] = mapped_column(Text, nullable=False)
    leverage: Mapped[int] = mapped_column(Integer, nullable=False, default=10)
    days: Mapped[int] = mapped_column(Integer, nullable=False, default=60)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="pending")
    result_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    best_strategy: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    win_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    profit_factor: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    expectancy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False, default=_now)
    updated_at: Mapped[int] = mapped_column(Integer, nullable=False, default=_now)


class CachedResearchRun(Base):
    __tablename__ = "cached_research_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    param_hash: Mapped[str] = mapped_column(Text, nullable=False, unique=True, index=True)
    symbol: Mapped[str] = mapped_column(Text, nullable=False)
    days: Mapped[int] = mapped_column(Integer, nullable=False)
    leverage: Mapped[float] = mapped_column(Float, nullable=False)
    horizon: Mapped[int] = mapped_column(Integer, nullable=False)
    rrs: Mapped[str] = mapped_column(Text, nullable=False)
    result_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False, default=_now)
    updated_at: Mapped[int] = mapped_column(Integer, nullable=False, default=_now)


class StrategyInsight(Base):
    __tablename__ = "strategy_insights"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(Text, nullable=False)
    strategy_name: Mapped[str] = mapped_column(Text, nullable=False)
    insight_type: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[Optional[str]] = mapped_column(Text, default="qwen3.5:4b")
    created_at: Mapped[int] = mapped_column(Integer, nullable=False, default=_now)
