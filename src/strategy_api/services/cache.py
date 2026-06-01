from __future__ import annotations
import hashlib
import json
import time
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from strategy_api.db.models import CachedResearchRun


def compute_param_hash(symbol: str, days: int, leverage: float, horizon: int, rrs: List[str]) -> str:
    """Deterministic hash of research parameters. RR order matters because
    the player block uses rrs[0] as the primary RR."""
    payload = json.dumps({
        "symbol": symbol.upper().strip(),
        "days": days,
        "leverage": leverage,
        "horizon": horizon,
        "rrs": rrs,
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_cached_run(db: Session, param_hash: str, ttl_hours: int = 168) -> Optional[Dict[str, Any]]:
    """Return cached result dict if it exists and is younger than ttl_hours."""
    row = db.query(CachedResearchRun).filter_by(param_hash=param_hash).first()
    if row is None:
        return None
    if ttl_hours > 0:
        age_seconds = time.time() - row.created_at
        if age_seconds > ttl_hours * 3600:
            return None
    try:
        return json.loads(row.result_json)
    except json.JSONDecodeError:
        return None


def save_cached_run(
    db: Session,
    param_hash: str,
    symbol: str,
    days: int,
    leverage: float,
    horizon: int,
    rrs: List[str],
    result: Dict[str, Any],
) -> None:
    """Upsert a cached research run."""
    now = int(time.time())
    row = db.query(CachedResearchRun).filter_by(param_hash=param_hash).first()
    result_json = json.dumps(result, default=str)
    if row is None:
        row = CachedResearchRun(
            param_hash=param_hash,
            symbol=symbol,
            days=days,
            leverage=leverage,
            horizon=horizon,
            rrs=json.dumps(rrs),
            result_json=result_json,
            created_at=now,
            updated_at=now,
        )
        db.add(row)
    else:
        row.result_json = result_json
        row.updated_at = now
    db.commit()
