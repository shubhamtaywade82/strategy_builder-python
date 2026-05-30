"""
binance_meta.py
Per-symbol economics for the labeling layer: funding profile + maintenance
margin rate. Network access with a loud fallback (geo-block safe). A fallback
result is stamped source="fallback" so the verdict can force NOT-VALIDATED.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import requests

log = logging.getLogger("binance_meta")

FAPI_BASE = "https://fapi.binance.com"
CACHE_DIR = Path(__file__).resolve().parent / "cache"

# Documented fallback defaults (used only when the network is blocked).
_FALLBACK_INTERVAL_H = 8.0
_FALLBACK_RATE = 0.0001
_FALLBACK_MMR = 0.005


@dataclass
class FundingProfile:
    interval_h: float
    avg_rate: float
    p90_rate: float
    anchor_hour: int
    source: str          # "api" | "cache" | "fallback"


def fetch_funding_profile(symbol: str,
                          session: Optional[requests.Session] = None,
                          base_url: str = FAPI_BASE) -> FundingProfile:
    s = session or requests.Session()
    try:
        info = s.get(f"{base_url}/fapi/v1/fundingInfo", timeout=10)
        info.raise_for_status()
        interval = _FALLBACK_INTERVAL_H
        for row in info.json():
            if row.get("symbol") == symbol:
                interval = float(row.get("fundingIntervalHours", _FALLBACK_INTERVAL_H))
                break

        hist = s.get(f"{base_url}/fapi/v1/fundingRate",
                     params={"symbol": symbol, "limit": 1000}, timeout=10)
        hist.raise_for_status()
        rates = np.array([float(r["fundingRate"]) for r in hist.json()], dtype=float)
        if rates.size == 0:
            raise ValueError("empty funding history")
        return FundingProfile(
            interval_h=interval,
            avg_rate=float(np.mean(np.abs(rates))),
            p90_rate=float(np.percentile(np.abs(rates), 90)),
            anchor_hour=0,
            source="api",
        )
    except Exception as exc:  # network blocked, parse error, empty history
        log.warning("funding fetch failed for %s (%s) -> FALLBACK defaults; "
                    "run will be NOT-VALIDATED", symbol, exc)
        return FundingProfile(_FALLBACK_INTERVAL_H, _FALLBACK_RATE, _FALLBACK_RATE,
                              0, "fallback")


def fetch_mmr(symbol: str, notional: float,
              session: Optional[requests.Session] = None,
              base_url: str = FAPI_BASE) -> Tuple[float, str]:
    """Maintenance-margin ratio for the bracket covering `notional`."""
    s = session or requests.Session()
    try:
        r = s.get(f"{base_url}/fapi/v1/leverageBracket",
                  params={"symbol": symbol}, timeout=10)
        r.raise_for_status()
        payload = r.json()
        entry = payload[0] if isinstance(payload, list) else payload
        brackets = sorted(entry["brackets"], key=lambda b: b["notionalCap"])
        for b in brackets:
            if notional <= b["notionalCap"]:
                return float(b["maintMarginRatio"]), "api"
        return float(brackets[-1]["maintMarginRatio"]), "api"
    except Exception as exc:
        log.warning("mmr fetch failed for %s (%s) -> FALLBACK %.4f",
                    symbol, exc, _FALLBACK_MMR)
        return _FALLBACK_MMR, "fallback"


def load_meta(symbol: str, notional: float = 1000.0,
              session: Optional[requests.Session] = None,
              base_url: str = FAPI_BASE) -> dict:
    """
    Combine funding + mmr into a dict and cache to disk. source is "fallback" if
    EITHER component fell back, so a degraded run cannot look validated.
    """
    prof = fetch_funding_profile(symbol, session, base_url)
    mmr, mmr_source = fetch_mmr(symbol, notional, session, base_url)
    source = "fallback" if "fallback" in (prof.source, mmr_source) else "api"
    meta = {**asdict(prof), "maint_margin_rate": mmr, "source": source,
            "symbol": symbol}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{symbol}_meta.json").write_text(json.dumps(meta, indent=2))
    return meta
