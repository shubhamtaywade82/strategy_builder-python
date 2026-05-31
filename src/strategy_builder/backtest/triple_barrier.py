"""
Triple-Barrier Labeler for strategy_builder-python
===================================================
Adds to: src/strategy_builder/backtest/triple_barrier.py

Usage:
    from strategy_builder.backtest.triple_barrier import TripleBarrierLabeler, BarrierConfig
    
    cfg = BarrierConfig(up_pct=0.01, dn_pct=0.005, max_horizon=120)
    labeler = TripleBarrierLabeler(cfg)
    labels = labeler.label_both_sides(candles)  # List[Candle] from domain.py

    # Labels dict contains:
    #   long_label: 1 if +1% before -0.5%, else 0
    #   short_label: 1 if -1% before +0.5%, else 0
    #   best_side: +1 (long), -1 (short), 0 (neither)
    #   bars_to_event: how many bars until target or stop hit
"""
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

from ..domain import Candle


@dataclass
class BarrierConfig:
    up_pct: float = 0.01      # Target move (price space)
    dn_pct: float = 0.005     # Stop loss (price space)
    max_horizon: int = 120    # Max bars to hold
    round_trip_cost: float = 0.0009  # Fee-adjusted


class TripleBarrierLabeler:
    """Labels each candle for both long and short independently.
    
    Path-dependent: walks 1m highs/lows to determine which barrier
    is hit FIRST. Leverage is applied after barrier resolution.
    """

    def __init__(self, config: BarrierConfig):
        self.cfg = config

    def _label_side(self, candles: List[Candle], side: int) -> Dict[str, Any]:
        n = len(candles)
        gross_up = self.cfg.up_pct + self.cfg.round_trip_cost

        idx, labels, btes, rets = [], [], [], []

        for i in range(n - 1):
            entry = candles[i + 1].open
            if entry <= 0:
                continue

            if side == 1:  # Long
                up_b = entry * (1 + gross_up)
                dn_b = entry * (1 - self.cfg.dn_pct)
            else:  # Short
                up_b = entry * (1 - gross_up)
                dn_b = entry * (1 + self.cfg.dn_pct)

            end = min(i + 1 + self.cfg.max_horizon, n)
            label, bte, ret = 0, self.cfg.max_horizon, 0.0

            for j in range(i + 1, end):
                bh = candles[j].high
                bl = candles[j].low

                if side == 1:
                    hit_dn = bl <= dn_b
                    hit_up = bh >= up_b
                else:
                    hit_dn = bh >= dn_b
                    hit_up = bl <= up_b

                if hit_dn and hit_up:
                    label, bte, ret = 0, j - (i + 1), -self.cfg.dn_pct - self.cfg.round_trip_cost
                    break
                if hit_dn:
                    label, bte, ret = 0, j - (i + 1), -self.cfg.dn_pct - self.cfg.round_trip_cost
                    break
                if hit_up:
                    label, bte, ret = 1, j - (i + 1), self.cfg.up_pct
                    break
            else:
                raw = (candles[end - 1].close - entry) / entry * side
                ret = raw - self.cfg.round_trip_cost

            idx.append(i)
            labels.append(label)
            btes.append(bte)
            rets.append(ret)

        return {
            f"{'long' if side == 1 else 'short'}_label": labels,
            f"{'long' if side == 1 else 'short'}_bars_to_event": btes,
            f"{'long' if side == 1 else 'short'}_return": rets,
            "entry_idx": idx,
        }

    def label_both_sides(self, candles: List[Candle]) -> List[Dict[str, Any]]:
        """Label every candle for both directions + best_side."""
        long = self._label_side(candles, 1)
        short = self._label_side(candles, -1)

        results = []
        for i in range(len(long["entry_idx"])):
            ll = long["long_label"][i]
            sl = short["short_label"][i]
            lb = long["long_bars_to_event"][i]
            sb = short["short_bars_to_event"][i]

            if ll == 1 and sl == 1:
                best = 1 if lb <= sb else -1
            elif ll == 1:
                best = 1
            elif sl == 1:
                best = -1
            else:
                best = 0

            results.append({
                "entry_idx": long["entry_idx"][i],
                "long_label": ll,
                "short_label": sl,
                "best_side": best,
                "long_bars": lb,
                "short_bars": sb,
                "long_return": long["long_return"][i],
                "short_return": short["short_return"][i],
            })

        return results

    @staticmethod
    def label_stats(labels: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(labels)
        lw = sum(1 for r in labels if r["long_label"] == 1)
        sw = sum(1 for r in labels if r["short_label"] == 1)
        bw = sum(1 for r in labels if r["best_side"] != 0)
        return {
            "total": n,
            "long_wins": lw,
            "long_win_pct": round(100 * lw / n, 1) if n else 0,
            "short_wins": sw,
            "short_win_pct": round(100 * sw / n, 1) if n else 0,
            "best_side_wins": bw,
            "best_side_pct": round(100 * bw / n, 1) if n else 0,
        }
