"""
Multi-RR Grid Search Strategy Discovery
Searches across 3:1, 2:1, 1:1, 1:2, 1:3 RRs for high win-rate strategies.
"""
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, List

from .features import build_mtf_features
from .labels import BarrierCfg, label_both_sides
from .backtest import backtest
from .validation import walk_forward, shuffle_test, oos_predict, _XGB_KW
from .robustness import score_robustness


RR_CONFIGS = {
    "3:1": {"up_pct": 0.015, "dn_pct": 0.005, "label": "3:1 (1.5% / 0.5%)"},
    "2:1": {"up_pct": 0.010, "dn_pct": 0.005, "label": "2:1 (1.0% / 0.5%)"},
    "1:1": {"up_pct": 0.010, "dn_pct": 0.010, "label": "1:1 (1.0% / 1.0%)"},
    "1:2": {"up_pct": 0.005, "dn_pct": 0.010, "label": "1:2 (0.5% / 1.0%)"},
    "1:3": {"up_pct": 0.005, "dn_pct": 0.015, "label": "1:3 (0.5% / 1.5%)"},
}


def discover_for_rr(
    features: pd.DataFrame,
    data_1m: pd.DataFrame,
    rr: str,
    leverage: float = 10.0,
    cost: float = 0.0009,
    horizon: int = 120,
) -> dict:
    """Run full discovery pipeline for a single RR configuration."""
    cfg = RR_CONFIGS[rr]
    print(f"\n{'='*60}")
    print(f"RR {rr}: target={cfg['up_pct']}, stop={cfg['dn_pct']}")
    print(f"{'='*60}")

    # 1. Label
    barrier = BarrierCfg(
        up_pct=cfg["up_pct"], dn_pct=cfg["dn_pct"],
        max_horizon=horizon, round_trip_cost=cost, leverage=leverage,
    )
    labels = label_both_sides(data_1m, barrier)

    n = len(labels)
    lw = labels["long_label"].sum()
    sw = labels["short_label"].sum()
    print(f"Labels: n={n}, long_wins={lw}({100*lw/n:.1f}%), short_wins={sw}({100*sw/n:.1f}%)")

    # 2. Features + Targets
    X = features.select_dtypes(include=[np.number]).fillna(0)

    results = {}
    for side_name, target_col in [("long", "best_side"), ("short", "best_side")]:
        if side_name == "long":
            y = (labels.set_index("entry_idx")["best_side"] > 0).astype(int)
        else:
            y = (labels.set_index("entry_idx")["best_side"] < 0).astype(int)

        common = X.index.intersection(y.index)
        Xs, ys = X.loc[common], y.loc[common]
        valid = ys.notna() & (ys.nunique() > 1 if hasattr(ys, 'nunique') else True)
        Xs, ys = Xs[valid], ys[valid]
        if len(Xs) < 500 or ys.nunique() < 2:
            print(f"  {side_name}: insufficient data")
            continue

        # 3. Train one full-sample model — used ONLY for descriptive feature
        #    importance, never for performance claims. (Deterministic config.)
        pos_weight = (ys == 0).sum() / max((ys == 1).sum(), 1)
        model = xgb.XGBClassifier(scale_pos_weight=pos_weight, **_XGB_KW)
        model.fit(Xs, ys)
        in_sample_probs = model.predict_proba(Xs)[:, 1]

        # 4. Feature importance
        imp = pd.DataFrame({
            "feature": Xs.columns,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)

        # 5. Out-of-sample diagnostics FIRST — they gate everything below.
        print(f"  {side_name}: Running walk-forward validation...")
        wf = walk_forward(Xs, ys, n_folds=5, embargo=horizon)
        print(f"  {side_name}: Running shuffle test...")
        st = shuffle_test(Xs, ys, n_shuffles=10)

        # 6. Honest signals: purged walk-forward OUT-OF-SAMPLE probabilities.
        #    Backtesting these (not the in-sample fit) is what kills the
        #    96%-win-rate illusion — a memorized model can't peek at the test bars.
        print(f"  {side_name}: Generating out-of-sample predictions...")
        oos_probs = oos_predict(Xs, ys, n_folds=5, embargo=horizon)
        side_val = 1 if side_name == "long" else -1

        # Shared condition table (descriptive, from top features).
        conditions = []
        for _, row in imp.head(5).iterrows():
            feat = row["feature"]
            median_val = Xs[feat].median()
            wr_high = ys[Xs[feat] >= median_val].mean()
            conditions.append({
                "feature": feat, "operator": ">=" if wr_high > 0.5 else "<",
                "threshold": float(median_val), "importance": float(row["importance"]),
            })

        strategies = []
        for threshold in [0.55, 0.60, 0.65, 0.70]:
            # OUT-OF-SAMPLE signals (bars the model never trained on at predict time)
            oos_sel = oos_probs[oos_probs >= threshold]
            oos_signals = [{"bar_idx": int(idx), "side": side_val, "confidence": float(p)}
                           for idx, p in oos_sel.items()]
            if len(oos_signals) < 5:
                continue
            oos_bt = backtest(data_1m, oos_signals, cfg["up_pct"], cfg["dn_pct"], leverage, cost, horizon)
            oos_m = oos_bt.metrics

            # In-sample backtest kept ONLY for transparency / overfit contrast.
            is_signals = [{"bar_idx": int(idx), "side": side_val, "confidence": float(p)}
                          for idx, p in zip(Xs.index, in_sample_probs) if p >= threshold]
            is_m = backtest(data_1m, is_signals, cfg["up_pct"], cfg["dn_pct"], leverage, cost, horizon).metrics

            rob = score_robustness(wf, st, oos_m)

            strategies.append({
                "name": f"{side_name}_threshold_{threshold}",
                "description": f"{side_name.upper()} when model confidence >= {threshold} (out-of-sample)",
                "side": side_name,
                "threshold": threshold,
                "conditions": conditions,
                "metrics": oos_m,                      # OUT-OF-SAMPLE — the honest numbers
                "in_sample_metrics": is_m,             # for transparency only
                "robustness_score": rob["robustness_score"],
                "is_robust": rob["is_robust"],
                "is_viable": rob["is_robust"],          # gate = robustness, not in-sample fit
                "is_viable_in_sample": bool(is_m.get("trade_count", 0) >= 20 and is_m.get("expectancy", 0) > 0),
                "warnings": rob["warnings"],
                "n_signals": len(oos_signals),
            })

        results[side_name] = {
            # Rank by out-of-sample robustness, then OOS expectancy as tiebreak.
            "strategies": sorted(
                strategies,
                key=lambda s: (s["robustness_score"], s["metrics"].get("expectancy", 0)),
                reverse=True,
            ),
            "walk_forward": wf,
            "shuffle_test": st,
            "top_features": imp.head(15).to_dict("records"),
            "label_stats": {"total": n, "long_wins": int(lw), "short_wins": int(sw), "no_trade": int(n - lw - sw)},
        }

    return results


def run_grid_search(
    data: Dict[str, pd.DataFrame],
    symbol: str = "SOLUSDT",
    leverage: float = 10.0,
    cost: float = 0.0009,
    horizon: int = 120,
    output_path: str = "research_result.json",
) -> dict:
    """Run multi-RR grid search and save results."""
    print(f"\n{'#'*70}")
    print(f"MULTI-RR STRATEGY SEARCH: {symbol}")
    print(f"Leverage: {leverage}x | Cost: {cost} | Horizon: {horizon} bars")
    print(f"{'#'*70}")

    # Build features once
    print("\n[1] Building MTF features...")
    features = build_mtf_features(data, base_tf="1m")
    print(f"    Features: {features.shape}")

    # Search each RR
    all_results = {}
    for rr in ["3:1", "2:1", "1:1", "1:2", "1:3"]:
        try:
            result = discover_for_rr(features, data["1m"], rr, leverage, cost, horizon)
            all_results[rr] = result
        except Exception as e:
            print(f"  ERROR with RR {rr}: {e}")
            all_results[rr] = {"error": str(e)}

    # Summary
    summary = {"symbol": symbol, "rr_configs": RR_CONFIGS, "results": all_results}

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return summary
