"""
Triple-Barrier Labeling - Both Sides
Labels in price space; leverage applied after.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class BarrierCfg:
    up_pct: float = 0.01
    dn_pct: float = 0.005
    max_horizon: int = 120
    round_trip_cost: float = 0.0009
    leverage: float = 10.0
    maint_margin_rate: float = 0.005

    def __post_init__(self):
        liq = (1.0 / self.leverage) - self.maint_margin_rate
        assert self.dn_pct < 0.5 * liq, f"stop {self.dn_pct} too close to liq {liq}"
        self.liq_distance = liq


def label_side(base: pd.DataFrame, side: int, cfg: BarrierCfg) -> pd.DataFrame:
    o = base["open"].to_numpy(float)
    hi = base["high"].to_numpy(float)
    lo = base["low"].to_numpy(float)
    cl = base["close"].to_numpy(float)
    t = base["open_time"].to_numpy()
    n = len(base)
    gross_up = cfg.up_pct + cfg.round_trip_cost
    L = cfg.leverage

    idx, fill, lab, bte, pret, mp = [], [], [], [], [], []

    for i in range(n - 1):
        entry = o[i + 1]
        if entry <= 0:
            continue

        if side == 1:
            up_b, dn_b = entry * (1 + gross_up), entry * (1 - cfg.dn_pct)
        else:
            up_b, dn_b = entry * (1 - gross_up), entry * (1 + cfg.dn_pct)

        end = min(i + 1 + cfg.max_horizon, n)
        label, b, pr = 0, cfg.max_horizon, 0.0

        for j in range(i + 1, end):
            bh, bl = hi[j], lo[j]
            if side == 1:
                hit_dn, hit_up = bl <= dn_b, bh >= up_b
            else:
                hit_dn, hit_up = bh >= dn_b, bl <= up_b

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
            raw = (cl[end - 1] - entry) / entry * side
            pr = raw - cfg.round_trip_cost

        margin_pnl = pr * L
        idx.append(i); fill.append(entry); lab.append(label)
        bte.append(b); pret.append(pr); mp.append(margin_pnl)

    return pd.DataFrame({
        "entry_idx": idx, "fill_price": fill, "label": lab,
        "bars_to_event": bte, "price_return": pret, "margin_pnl": mp,
    })


def label_both_sides(base: pd.DataFrame, cfg: BarrierCfg) -> pd.DataFrame:
    """Label every bar for both directions + best_side."""
    longs = label_side(base, +1, cfg)
    longs = longs.add_prefix("long_").rename(columns={"long_entry_idx": "entry_idx"})

    shorts = label_side(base, -1, cfg)
    shorts = shorts.add_prefix("short_").rename(columns={"short_entry_idx": "entry_idx"})

    df = longs.merge(shorts, on="entry_idx", how="inner")
    df["entry_time"] = base["open_time"].to_numpy()[df["entry_idx"].to_numpy() + 1]

    def best(r):
        lw, sw = r["long_label"] == 1, r["short_label"] == 1
        if lw and sw:
            return 1 if r["long_bars_to_event"] <= r["short_bars_to_event"] else -1
        if lw: return 1
        if sw: return -1
        return 0

    df["best_side"] = df.apply(best, axis=1)
    return df
