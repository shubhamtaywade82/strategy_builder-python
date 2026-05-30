"""
barrier_leverage.py
Both-direction, leverage-aware labeling layer on top of mtf_research.py.

Design invariant: the triple-barrier hit logic operates in PRICE space and is
identical to the single-side core. Leverage is a linear transform applied AFTER
the barrier is resolved -> PnL on margin, fee drag on margin, funding, and a
hard liquidation safety barrier. Leverage NEVER changes which barrier hits first.

Funding: interval is symbol-specific on Binance USDⓈ-M. Pull it from
  GET /fapi/v1/fundingInfo  (interval + cap)  and
  GET /fapi/v1/fundingRate  (historical realized rates).
Do not hardcode 8h. Defaults below are placeholders for the self-test only.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger("barrier_leverage")


@dataclass
class LevBarrierConfig:
    up_pct: float = 0.01            # target favorable PRICE move (nets up_pct in price after cost)
    dn_pct: float = 0.005           # stop, PRICE
    max_horizon: int = 120          # bars (120 x 1m = 2h)
    round_trip_cost: float = 0.0009 # taker in + taker out + slippage, PRICE space
    leverage: float = 10.0          # 1% price * 10 = 10% margin
    maint_margin_rate: float = 0.005  # mmr for the symbol's leverage bracket (fetch from exchangeInfo)
    liq_safety: float = 0.5         # stop must be < liq_safety * liquidation_distance

    # funding (per-symbol; fetch real values, these are test defaults)
    funding_rate: float = 0.0001    # average realized rate per settlement (fraction of notional)
    funding_interval_h: float = 8.0 # SYMBOL-SPECIFIC. fetch from fundingInfo.
    funding_anchor_utc_hour: int = 0  # first settlement hour anchor (00:00 UTC)

    def __post_init__(self) -> None:
        liq_distance = (1.0 / self.leverage) - self.maint_margin_rate
        if liq_distance <= 0:
            raise ValueError(f"leverage {self.leverage}x with mmr {self.maint_margin_rate} "
                             f"gives non-positive liquidation distance")
        if self.dn_pct >= self.liq_safety * liq_distance:
            raise ValueError(
                f"INVARIANT VIOLATION: stop {self.dn_pct:.4f} too close to liquidation "
                f"distance {liq_distance:.4f} at {self.leverage}x "
                f"(must be < {self.liq_safety * liq_distance:.4f}). Reduce leverage or stop.")
        self.liq_distance = liq_distance


def _funding_crossings(entry_time: pd.Timestamp, exit_time: pd.Timestamp, cfg: LevBarrierConfig) -> int:
    """Count funding settlement instants strictly within (entry, exit]."""
    if pd.isna(entry_time) or pd.isna(exit_time) or exit_time <= entry_time:
        return 0
    step_ns = int(cfg.funding_interval_h * 3600 * 1e9)
    anchor = entry_time.normalize() + pd.Timedelta(hours=cfg.funding_anchor_utc_hour)
    # walk anchor down/up to just before entry, then count steps up to exit
    while anchor > entry_time:
        anchor -= pd.Timedelta(step_ns, unit="ns")
    count = 0
    t = anchor + pd.Timedelta(step_ns, unit="ns")
    while t <= exit_time:
        if t > entry_time:
            count += 1
        t += pd.Timedelta(step_ns, unit="ns")
    return count


def _label_one_side(base: pd.DataFrame, side: int, cfg: LevBarrierConfig) -> pd.DataFrame:
    o = base["open"].to_numpy(float)
    hi = base["high"].to_numpy(float)
    lo = base["low"].to_numpy(float)
    cl = base["close"].to_numpy(float)
    t = base["open_time"].to_numpy()
    n = len(base)
    gross_up = cfg.up_pct + cfg.round_trip_cost
    L = cfg.leverage

    idx, fill, lab, bte, price_ret, margin_pnl = [], [], [], [], [], []

    for i in range(n - 1):
        entry = o[i + 1]
        if not np.isfinite(entry) or entry <= 0:
            continue
        if side == 1:
            up_b, dn_b = entry * (1 + gross_up), entry * (1 - cfg.dn_pct)
            liq_b = entry * (1 - cfg.liq_distance)
        else:
            up_b, dn_b = entry * (1 - gross_up), entry * (1 + cfg.dn_pct)
            liq_b = entry * (1 + cfg.liq_distance)

        end = min(i + 1 + cfg.max_horizon, n)
        label, b, pret = 0, cfg.max_horizon, 0.0
        for j in range(i + 1, end):
            bh, bl = hi[j], lo[j]
            if side == 1:
                hit_liq = bl <= liq_b           # catastrophic (should never fire if invariant holds)
                hit_dn = bl <= dn_b
                hit_up = bh >= up_b
            else:
                hit_liq = bh >= liq_b
                hit_dn = bh >= dn_b
                hit_up = bl <= up_b
            if hit_liq and not hit_dn:           # gapped past stop straight to liq
                label, b, pret = 0, j - (i + 1), -cfg.liq_distance - cfg.round_trip_cost
                break
            if hit_dn and hit_up:                # worst case: stop first
                label, b, pret = 0, j - (i + 1), -cfg.dn_pct - cfg.round_trip_cost
                break
            if hit_dn:
                label, b, pret = 0, j - (i + 1), -cfg.dn_pct - cfg.round_trip_cost
                break
            if hit_up:
                label, b, pret = 1, j - (i + 1), cfg.up_pct
                break
        else:
            raw = (cl[end - 1] - entry) / entry * side
            pret = raw - cfg.round_trip_cost

        # leverage transform + funding
        exit_bar = i + 1 + b
        funding_events = _funding_crossings(pd.Timestamp(t[i + 1]),
                                            pd.Timestamp(t[min(exit_bar, n - 1)]), cfg)
        # long pays funding when rate>0; short receives. funding% on margin = rate * L * events
        funding_margin = -side * cfg.funding_rate * L * funding_events
        m_pnl = pret * L + funding_margin

        idx.append(i); fill.append(entry); lab.append(label); bte.append(b)
        price_ret.append(pret); margin_pnl.append(m_pnl)

    return pd.DataFrame({
        "entry_idx": idx, "fill_price": fill, "label": lab, "bars_to_event": bte,
        "price_return": price_ret, "margin_pnl": margin_pnl,
    })


def triple_barrier_both_sides(base: pd.DataFrame, cfg: LevBarrierConfig = LevBarrierConfig()) -> pd.DataFrame:
    """
    Label every bar for BOTH directions independently, plus a meta best_side.
    best_side: side whose label==1; if both win, the one resolving in fewer bars;
               if neither, 0.
    """
    longs = _label_one_side(base, +1, cfg).add_prefix("long_").rename(columns={"long_entry_idx": "entry_idx"})
    shorts = _label_one_side(base, -1, cfg).add_prefix("short_").rename(columns={"short_entry_idx": "entry_idx"})
    df = longs.merge(shorts, on="entry_idx", how="inner")
    df["entry_time"] = base["open_time"].to_numpy()[df["entry_idx"].to_numpy() + 1]

    def _best(r):
        lw, sw = r["long_label"] == 1, r["short_label"] == 1
        if lw and sw:
            return 1 if r["long_bars_to_event"] <= r["short_bars_to_event"] else -1
        if lw:
            return 1
        if sw:
            return -1
        return 0
    df["best_side"] = df.apply(_best, axis=1)
    return df


# --------------------------------------------------------------------------- #
def _selftest() -> None:
    # invariant guard: 0.5% stop at 10x is valid; widening to 9% must raise
    LevBarrierConfig(dn_pct=0.005, leverage=10.0)  # ok
    try:
        LevBarrierConfig(dn_pct=0.09, leverage=10.0)
        raise AssertionError("expected liquidation-buffer invariant to fire")
    except ValueError:
        log.info("Liquidation-buffer invariant correctly rejects a near-liq stop.")

    # leverage scaling: a price win of +1% must report ~+10% margin (minus funding)
    px = pd.DataFrame({
        "open_time": pd.date_range("2026-01-01", periods=6, freq="1min", tz="UTC"),
        "open":  [100, 100, 100.3, 100.8, 101.2, 101.0],
        "high":  [100, 100.4, 100.9, 101.3, 101.5, 101.2],
        "low":   [100, 99.9, 100.2, 100.6, 101.0, 100.8],
        "close": [100, 100.3, 100.8, 101.2, 101.3, 101.0],
    })
    out = triple_barrier_both_sides(px, LevBarrierConfig(leverage=10.0, funding_rate=0.0))
    assert out.loc[0, "long_label"] == 1, "long should win"
    assert abs(out.loc[0, "long_margin_pnl"] - 0.10) < 1e-9, \
        f"10x of +1% price must be +10% margin, got {out.loc[0,'long_margin_pnl']}"
    assert out.loc[0, "short_label"] == 0, "short should lose on a rising path"
    assert out.loc[0, "best_side"] == 1
    log.info("Leverage scaling: +1%% price -> +%.1f%% margin at 10x. best_side=%d",
             out.loc[0, "long_margin_pnl"] * 100, out.loc[0, "best_side"])

    # funding sign: long pays when rate>0
    cfg_f = LevBarrierConfig(leverage=10.0, funding_rate=0.0001, funding_interval_h=8.0)
    n = _funding_crossings(pd.Timestamp("2026-01-01 07:30", tz="UTC"),
                           pd.Timestamp("2026-01-01 09:30", tz="UTC"), cfg_f)
    assert n == 1, f"expected one 08:00 crossing, got {n}"
    log.info("Funding crossing detection OK (one settlement in 07:30->09:30).")
    print("ALL LEVERAGE SELF-TESTS PASSED")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _selftest()
