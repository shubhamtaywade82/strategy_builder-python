"""
Triple-Barrier Labeling with Leverage Awareness
================================================
Labels each bar for BOTH long and short independently.

Design invariant: Barrier-hit logic operates in PRICE space.
Leverage is a linear transform applied AFTER barrier resolution.

Key property: A 1% price move = 10% on 10x margin.
The label says "did price move +1% before -0.5%" — leverage doesn't change
which barrier hits first.

Based on: barrier_leverage.py + mtf_research.py
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger("triple_barrier")


@dataclass
class LevBarrierConfig:
    """Leverage-aware barrier configuration."""
    up_pct: float = 0.01
    dn_pct: float = 0.005
    max_horizon: int = 120
    round_trip_cost: float = 0.0009
    leverage: float = 10.0
    maint_margin_rate: float = 0.005
    liq_safety: float = 0.5
    funding_rate: float = 0.0001
    funding_interval_h: float = 8.0
    funding_anchor_utc_hour: int = 0

    def __post_init__(self):
        # Liquidation distance check
        liq_distance = (1.0 / self.leverage) - self.maint_margin_rate
        if liq_distance <= 0:
            raise ValueError(
                f"Leverage {self.leverage}x with mmr {self.maint_margin_rate} "
                f"gives non-positive liquidation distance"
            )
        if self.dn_pct >= self.liq_safety * liq_distance:
            raise ValueError(
                f"STOP TOO CLOSE TO LIQUIDATION: "
                f"stop={self.dn_pct:.4f}, liq_distance={liq_distance:.4f}, "
                f"safe_max={self.liq_safety * liq_distance:.4f}. "
                f"Reduce leverage or tighten stop."
            )
        self.liq_distance = liq_distance
        log.info(f"Barrier config validated: liq_distance={liq_distance:.4f} "
                 f"({self.leverage}x, safe_stop_max={self.liq_safety * liq_distance:.4f})")


def _count_funding_crossings(
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    cfg: LevBarrierConfig,
) -> int:
    """Count funding settlement instants strictly within (entry, exit]."""
    if pd.isna(entry_time) or pd.isna(exit_time) or exit_time <= entry_time:
        return 0

    step_ns = int(cfg.funding_interval_h * 3600 * 1e9)
    anchor = entry_time.normalize() + pd.Timedelta(hours=cfg.funding_anchor_utc_hour)

    while anchor > entry_time:
        anchor -= pd.Timedelta(step_ns, unit="ns")

    count = 0
    t = anchor + pd.Timedelta(step_ns, unit="ns")
    while t <= exit_time:
        if t > entry_time:
            count += 1
        t += pd.Timedelta(step_ns, unit="ns")

    return count


def _label_one_side(
    base: pd.DataFrame,
    side: int,
    cfg: LevBarrierConfig,
) -> pd.DataFrame:
    """Label every bar for a single side (long=+1 or short=-1).

    Path-dependent: walks 1m highs/lows in order to determine which
    barrier is hit FIRST.
    """
    o = base["open"].to_numpy(float)
    hi = base["high"].to_numpy(float)
    lo = base["low"].to_numpy(float)
    cl = base["close"].to_numpy(float)
    t = base["open_time"].to_numpy()
    n = len(base)

    gross_up = cfg.up_pct + cfg.round_trip_cost
    L = cfg.leverage

    idx, fill, lab, bte, pret, m_pnl = [], [], [], [], [], []

    for i in range(n - 1):
        entry = o[i + 1]
        if not np.isfinite(entry) or entry <= 0:
            continue

        # Barriers in price space
        if side == 1:
            up_b = entry * (1 + gross_up)
            dn_b = entry * (1 - cfg.dn_pct)
            liq_b = entry * (1 - cfg.liq_distance)
        else:
            up_b = entry * (1 - gross_up)
            dn_b = entry * (1 + cfg.dn_pct)
            liq_b = entry * (1 + cfg.liq_distance)

        end = min(i + 1 + cfg.max_horizon, n)
        label, b, pr = 0, cfg.max_horizon, 0.0

        for j in range(i + 1, end):
            bh, bl = hi[j], lo[j]

            if side == 1:
                hit_liq = bl <= liq_b
                hit_dn = bl <= dn_b
                hit_up = bh >= up_b
            else:
                hit_liq = bh >= liq_b
                hit_dn = bh >= dn_b
                hit_up = bl <= up_b

            # Catastrophic: gapped past stop to liquidation
            if hit_liq and not hit_dn:
                label, b, pr = 0, j - (i + 1), -cfg.liq_distance - cfg.round_trip_cost
                log.warning(f"LIQUIDATION event at bar {j} (side={side})")
                break

            # Worst case: both in same bar -> stop first
            if hit_dn and hit_up:
                label, b, pr = 0, j - (i + 1), -cfg.dn_pct - cfg.round_trip_cost
                break

            if hit_dn:
                label, b, pr = 0, j - (i + 1), -cfg.dn_pct - cfg.round_trip_cost
                break

            if hit_up:
                label, b, pr = 1, j - (i + 1), cfg.up_pct
                break
        else:
            # Timeout: mark to last close
            raw = (cl[end - 1] - entry) / entry * side
            pr = raw - cfg.round_trip_cost

        # Apply leverage + funding
        exit_bar = i + 1 + b
        funding_events = _count_funding_crossings(
            pd.Timestamp(t[i + 1]),
            pd.Timestamp(t[min(exit_bar, n - 1)]),
            cfg,
        )

        # Long pays funding when rate > 0; short receives
        funding_margin = -side * cfg.funding_rate * L * funding_events
        margin_pnl = pr * L + funding_margin

        idx.append(i)
        fill.append(entry)
        lab.append(label)
        bte.append(b)
        pret.append(pr)
        m_pnl.append(margin_pnl)

    return pd.DataFrame({
        "entry_idx": idx,
        "fill_price": fill,
        "label": lab,
        "bars_to_event": bte,
        "price_return": pret,
        "margin_pnl": m_pnl,
    })


def triple_barrier_both_sides(
    base: pd.DataFrame,
    cfg: Optional[LevBarrierConfig] = None,
) -> pd.DataFrame:
    """Label every bar for BOTH directions, plus meta best_side.

    Args:
        base: 1m OHLCV DataFrame with open_time column
        cfg: Barrier configuration

    Returns:
        DataFrame with long_*, short_*, and best_side columns
    """
    cfg = cfg or LevBarrierConfig()
    log.info(f"Labeling: up={cfg.up_pct}, dn={cfg.dn_pct}, "
             f"horizon={cfg.max_horizon}bars, leverage={cfg.leverage}x")

    longs = _label_one_side(base, +1, cfg)
    longs = longs.add_prefix("long_").rename(columns={"long_entry_idx": "entry_idx"})

    shorts = _label_one_side(base, -1, cfg)
    shorts = shorts.add_prefix("short_").rename(columns={"short_entry_idx": "entry_idx"})

    df = longs.merge(shorts, on="entry_idx", how="inner")
    df["entry_time"] = base["open_time"].to_numpy()[df["entry_idx"].to_numpy() + 1]

    # Determine best side
    def _best(r):
        lw = r["long_label"] == 1
        sw = r["short_label"] == 1
        if lw and sw:
            return 1 if r["long_bars_to_event"] <= r["short_bars_to_event"] else -1
        if lw:
            return 1
        if sw:
            return -1
        return 0

    df["best_side"] = df.apply(_best, axis=1)

    # Summary stats
    n = len(df)
    long_wins = df["long_label"].sum()
    short_wins = df["short_label"].sum()
    best_dist = df["best_side"].value_counts().to_dict()

    log.info(f"Labels: n={n}, long_wins={long_wins} ({100*long_wins/n:.1f}%), "
             f"short_wins={short_wins} ({100*short_wins/n:.1f}%), "
             f"best_side={best_dist}")

    return df


def create_target_vector(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    target_type: str = "best_side",
) -> pd.Series:
    """Create aligned target vector from features and labels.

    Args:
        features: Feature DataFrame (indexed by bar position)
        labels: Labels DataFrame with entry_idx
        target_type: 'best_side', 'long_label', or 'short_label'

    Returns:
        Series aligned to features index
    """
    if target_type == "best_side":
        target = labels.set_index("entry_idx")["best_side"]
    elif target_type == "long_label":
        target = labels.set_index("entry_idx")["long_label"]
    elif target_type == "short_label":
        target = labels.set_index("entry_idx")["short_label"]
    else:
        raise ValueError(f"Unknown target_type: {target_type}")

    # Align to features
    aligned = pd.Series(index=features.index, dtype=target.dtype)
    for idx, val in target.items():
        if idx in aligned.index:
            aligned.loc[idx] = val

    return aligned
