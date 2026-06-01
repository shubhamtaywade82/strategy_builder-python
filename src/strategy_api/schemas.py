from __future__ import annotations
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ChatMsg(BaseModel):
    role: str
    content: str


class ChatReq(BaseModel):
    messages: List[ChatMsg]
    model: str = "qwen3.5:4b"
    temperature: float = 0.7
    sessionId: Optional[str] = None


class GenStrategyReq(BaseModel):
    symbol: str
    rr: str
    features: List[str]
    metrics: Dict[str, float]
    model: str = "qwen3.5:4b"
    sessionId: Optional[str] = None


class PriceData(BaseModel):
    currentPrice: float
    change24h: float
    high24h: float
    low24h: float
    volume24h: float


class MarketAnalysisReq(BaseModel):
    symbol: str
    priceData: PriceData
    model: str = "qwen3.5:4b"


class SaveSessionReq(BaseModel):
    sessionId: str
    symbol: str
    rrConfig: str
    leverage: int = 10
    days: int = 60
    resultJson: Optional[str] = None
    bestStrategy: Optional[str] = None
    winRate: Optional[float] = None
    profitFactor: Optional[float] = None
    expectancy: Optional[float] = None


class ResearchRunReq(BaseModel):
    symbol: str = Field("SOLUSDT", pattern=r"^[A-Z0-9]{3,12}$")
    days: int = Field(60, ge=7, le=365)
    leverage: float = Field(10, ge=1, le=125)
    horizon: int = Field(120, ge=30, le=480)
    rrs: List[str] = Field(default_factory=lambda: ["2:1"], min_length=1)
