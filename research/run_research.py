"""
run_research.py
Orchestrate the full leakage-safe research run for one Binance USD-M symbol and
emit a PASS/FAIL verdict honoring the §4 contract.

Flow: fetch meta -> load 5 TFs -> MTF features -> both-side triple-barrier
labels -> join -> purged-CV per side + probes -> verdict -> console + JSON.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

import validation as v
from binance_meta import load_meta
from mtf_research import BinanceUMKlineLoader, build_mtf_features
from barrier_leverage import triple_barrier_both_sides, LevBarrierConfig

log = logging.getLogger("run_research")

_NON_FEATURE = {
    "open_time", "close_time", "open", "high", "low", "close", "volume",
    "quote_volume", "taker_buy_base", "taker_buy_quote", "n_trades",
    "entry_time", "fill_price", "best_side",
}
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def select_feature_columns(ds: pd.DataFrame) -> List[str]:
    return [c for c in ds.columns
            if c not in _NON_FEATURE
            and not c.startswith("long_") and not c.startswith("short_")]


def build_report_dict(symbol: str, side: str, report: "v.SideReport",
                      verdict: "v.Verdict", probes: Dict, meta: Dict) -> Dict:
    return {
        "symbol": symbol,
        "side": side,
        "meta": {"source": meta["source"],
                 "interval_h": meta.get("interval_h"),
                 "maint_margin_rate": meta.get("maint_margin_rate")},
        "verdict": {"passed": verdict.passed, "reasons": verdict.reasons},
        "summary": {"mean_e_margin": report.mean_e_margin,
                    "min_e_margin": report.min_e_margin,
                    "cv_e_margin": report.cv_e_margin,
                    "mean_win_rate": report.mean_win_rate},
        "folds": [asdict(f) for f in report.folds],
        "probes": {k: (v_ if not isinstance(v_, dict)
                       else {str(kk): vv for kk, vv in v_.items()})
                   for k, v_ in probes.items()},
    }


def _config_from_meta(meta: Dict, round_trip_cost: float = 0.0009, scalp: bool = False) -> LevBarrierConfig:
    if scalp:
        return LevBarrierConfig(
            up_pct=0.004, dn_pct=0.002, max_horizon=30, leverage=10.0,
            round_trip_cost=round_trip_cost,
            maint_margin_rate=meta.get("maint_margin_rate", 0.005),
            funding_rate=meta.get("avg_rate", 0.0001),
            funding_interval_h=meta.get("interval_h", 8.0),
            funding_anchor_utc_hour=meta.get("anchor_hour", 0),
        )
    return LevBarrierConfig(
        up_pct=0.01, dn_pct=0.005, max_horizon=120, leverage=10.0,
        round_trip_cost=round_trip_cost,
        maint_margin_rate=meta.get("maint_margin_rate", 0.005),
        funding_rate=meta.get("avg_rate", 0.0001),
        funding_interval_h=meta.get("interval_h", 8.0),
        funding_anchor_utc_hour=meta.get("anchor_hour", 0),
    )


# 1m frame is fetched once per symbol and reused by the cost ladder.
_FRAME_CACHE: Dict[str, pd.DataFrame] = {}


def _base_1m(symbol: str, days: int) -> pd.DataFrame:
    base = _FRAME_CACHE.get(symbol)
    if base is None:
        loader = BinanceUMKlineLoader()
        end = int(time.time() * 1000)
        start = end - days * 86_400_000
        base = loader.fetch(symbol, "1m", start, end)
        _FRAME_CACHE[symbol] = base
    return base


def build_dataset(symbol: str, days: int, meta: Dict, scalp: bool = False) -> pd.DataFrame:
    loader = BinanceUMKlineLoader()
    end = int(time.time() * 1000)
    start = end - days * 86_400_000
    frames = {tf: loader.fetch(symbol, tf, start, end)
              for tf in ["1m", "5m", "1h", "4h", "1d"]}
    _FRAME_CACHE[symbol] = frames["1m"]
    feats = build_mtf_features(frames, base_tf="1m")
    cfg = _config_from_meta(meta, scalp=scalp)
    labels = triple_barrier_both_sides(frames["1m"], cfg)
    ds = feats.join(labels.set_index("entry_idx"), how="inner").dropna().reset_index(drop=True)
    return ds


def _cost_ladder_ds(symbol: str, days: int, meta: Dict, cost: float, scalp: bool = False) -> pd.DataFrame:
    base = _base_1m(symbol, days)
    return triple_barrier_both_sides(base, _config_from_meta(meta, cost, scalp=scalp))


def run_symbol(symbol: str, days: int = 45, round_trip_cost: float = 0.0009, scalp: bool = False) -> Dict:
    meta = load_meta(symbol)
    ds = build_dataset(symbol, days, meta, scalp=scalp)
    feature_cols = select_feature_columns(ds)
    folds = v.purged_folds(len(ds), n_folds=5, embargo=120, min_train=500)
    log.info("dataset rows=%d features=%d folds=%d source=%s",
             len(ds), len(feature_cols), len(folds), meta["source"])

    results: Dict[str, Dict] = {}
    for side in ("long", "short"):
        report = v.evaluate_side(ds, side, feature_cols, folds)
        probes = {
            "shuffle_e_margin": v.shuffle_test(ds, side, feature_cols, folds),
            "leakage": v.leakage_probe(ds, side, feature_cols, folds),
            "liquidations": v.liquidation_check(ds, side, 0.005, round_trip_cost),
            "cost_ladder": v.cost_ladder(
                lambda c: _cost_ladder_ds(symbol, days, meta, c, scalp=scalp),
                (lambda sd: (lambda d: float(d[f"{sd}_margin_pnl"].mean())))(side),
                costs=[0.0009, 0.0012, 0.0015]),
        }
        verdict = v.verdict(report, probes, meta["source"], round_trip_cost, 10.0)
        results[side] = build_report_dict(symbol, side, report, verdict, probes, meta)
        _print_side(symbol, side, report, verdict, probes)
        _print_rule(ds, side, feature_cols)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{symbol}_{time.strftime('%Y-%m-%d')}.json"
    out_path.write_text(json.dumps(results, indent=2))
    log.info("wrote %s", out_path)
    return results


def _print_side(symbol, side, report, verdict, probes):
    print("\n" + "=" * 60)
    print(f"{symbol} {side.upper()}  ->  {'PASS' if verdict.passed else 'FAIL'}")
    print("=" * 60)
    for i, f in enumerate(report.folds, 1):
        print(f"  fold {i}: trades={f.n_trades:4d} E_margin={f.e_margin:+.4f} "
              f"win={f.win_rate:.2f} theta={f.theta}")
    print(f"  mean E_margin={report.mean_e_margin:+.4f}  cv={report.cv_e_margin:.2f}  "
          f"win={report.mean_win_rate:.2f}")
    print(f"  shuffle E={probes['shuffle_e_margin']:+.4f}  "
          f"leak base={probes['leakage']['baseline_e_margin']:+.4f} "
          f"shifted={probes['leakage']['shifted_e_margin']:+.4f}  "
          f"liq={probes['liquidations']}")
    print(f"  cost ladder: {probes['cost_ladder']}")
    if not verdict.passed:
        for r in verdict.reasons:
            print(f"  - {r}")


def _print_rule(ds, side, feature_cols):
    """Surrogate tree fit on TRAIN-only rows for the human-readable rule."""
    folds = v.purged_folds(len(ds), n_folds=5, embargo=120, min_train=500)
    if not folds:
        return
    train_idx = folds[-1][0]                      # largest train window
    X = ds[feature_cols].to_numpy(float)[train_idx]
    y = ds[f"{side}_label"].to_numpy(int)[train_idx]
    if len(set(y)) < 2:
        return
    dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=0)
    dt.fit(X, y)
    print(f"\n  human-readable {side} rule (surrogate tree, train-only):")
    print(export_text(dt, feature_names=list(feature_cols)))


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SOLUSDT")
    ap.add_argument("--days", type=int, default=45)
    ap.add_argument("--scalp", action="store_true", help="Test scalp targets (0.4% TP, 0.2% SL)")
    args = ap.parse_args()
    run_symbol(args.symbol, args.days, scalp=args.scalp)


if __name__ == "__main__":
    main()
