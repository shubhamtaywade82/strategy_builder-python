import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research"))

import numpy as np
import pandas as pd
import pytest

import validation as v


# --------------------------------------------------------------------------- #
# Task 1: purged_folds
# --------------------------------------------------------------------------- #
def test_purged_folds_no_overlap_and_embargo():
    folds = v.purged_folds(n=6000, n_folds=5, embargo=120, min_train=500)
    assert len(folds) >= 1
    for train_idx, test_idx in folds:
        assert len(train_idx) >= 500
        assert min(test_idx) - max(train_idx) > 120          # embargo gap respected
        assert set(train_idx).isdisjoint(set(test_idx))      # never overlap
        assert min(train_idx) >= 0 and max(test_idx) < 6000  # in range


def test_purged_folds_skips_tiny_train():
    folds = v.purged_folds(n=2000, n_folds=5, embargo=120, min_train=500)
    for train_idx, test_idx in folds:
        assert len(train_idx) >= 500
        assert len(test_idx) > 0


# --------------------------------------------------------------------------- #
# Task 2: pick_threshold_on_train
# --------------------------------------------------------------------------- #
def test_pick_threshold_prefers_profitable_cut():
    probs = np.linspace(0.1, 0.9, 100)
    pnl = np.where(probs > 0.7, 0.10, -0.05)
    theta = v.pick_threshold_on_train(probs, pnl, grid=np.arange(0.5, 0.95, 0.05),
                                       min_trades=5)
    assert theta is not None and theta >= 0.7


def test_pick_threshold_returns_none_when_too_few_trades():
    probs = np.full(100, 0.2)
    pnl = np.full(100, 0.10)
    theta = v.pick_threshold_on_train(probs, pnl, grid=np.arange(0.5, 0.95, 0.05),
                                       min_trades=5)
    assert theta is None


# --------------------------------------------------------------------------- #
# Task 3: evaluate_side
# --------------------------------------------------------------------------- #
def _synthetic_dataset(n=3000, seed=0):
    """Feature f0 is predictive of long wins; build matching label + margin pnl."""
    rng = np.random.default_rng(seed)
    f0 = rng.normal(size=n)
    f1 = rng.normal(size=n)
    win_prob = 1.0 / (1.0 + np.exp(-3.0 * f0))     # high f0 -> likely win
    long_label = (rng.uniform(size=n) < win_prob).astype(int)
    long_margin = np.where(long_label == 1, 0.091, -0.059)   # 10x of +1%/-0.5% net
    short_label = 1 - long_label
    short_margin = np.where(short_label == 1, 0.091, -0.059)
    return pd.DataFrame({
        "f0": f0, "f1": f1,
        "long_label": long_label, "long_margin_pnl": long_margin,
        "long_price_return": np.where(long_label == 1, 0.01, -0.0059),
        "short_label": short_label, "short_margin_pnl": short_margin,
        "short_price_return": np.where(short_label == 1, 0.01, -0.0059),
    })


def test_evaluate_side_finds_positive_edge_on_predictive_data():
    ds = _synthetic_dataset()
    folds = v.purged_folds(len(ds), n_folds=4, embargo=50, min_train=300)
    report = v.evaluate_side(ds, "long", ["f0", "f1"], folds)
    assert report.side == "long"
    assert len(report.folds) == len(folds)
    assert report.mean_e_margin > 0           # predictive feature -> positive E
    for fr in report.folds:
        assert 0.0 <= fr.win_rate <= 1.0


# --------------------------------------------------------------------------- #
# Task 4: shuffle_test + leakage_probe
# --------------------------------------------------------------------------- #
def test_shuffle_collapses_edge():
    ds = _synthetic_dataset()
    folds = v.purged_folds(len(ds), n_folds=4, embargo=50, min_train=300)
    baseline = v.evaluate_side(ds, "long", ["f0", "f1"], folds).mean_e_margin
    shuffled = v.shuffle_test(ds, "long", ["f0", "f1"], folds, seed=0)
    assert shuffled < baseline                 # breaking feature->label kills edge
    # collapses toward the random-trading floor (well below the real edge).
    # Note: this synthetic's payoffs are asymmetric so the floor is slightly
    # positive; on a fair triple-barrier it collapses toward ~ -round_trip_cost.
    assert shuffled < 0.5 * baseline


def test_leakage_probe_drops_when_feature_shifted():
    ds = _synthetic_dataset()
    folds = v.purged_folds(len(ds), n_folds=4, embargo=50, min_train=300)
    res = v.leakage_probe(ds, "long", ["f0", "f1"], folds)
    assert res["shifted_e_margin"] < res["baseline_e_margin"]


# --------------------------------------------------------------------------- #
# Task 5: cost_ladder + liquidation_check
# --------------------------------------------------------------------------- #
def test_cost_ladder_is_monotonic_non_increasing():
    def rebuild_fn(cost):
        d = _synthetic_dataset()
        d["long_margin_pnl"] = d["long_margin_pnl"] - 10.0 * cost
        return d

    def evaluate_fn(d):
        return float(d["long_margin_pnl"].mean())

    ladder = v.cost_ladder(rebuild_fn, evaluate_fn, costs=[0.0009, 0.0012, 0.0015])
    es = [ladder[c] for c in [0.0009, 0.0012, 0.0015]]
    assert es[0] >= es[1] >= es[2]


def test_liquidation_check_counts_catastrophic_rows():
    df = pd.DataFrame({"long_price_return": [0.01, -0.0059, -0.095, -0.0059]})
    cnt = v.liquidation_check(df, "long", dn_pct=0.005, round_trip_cost=0.0009)
    assert cnt == 1


# --------------------------------------------------------------------------- #
# Task 6: verdict gates
# --------------------------------------------------------------------------- #
def _passing_report():
    return v.SideReport(side="long", folds=[
        v.FoldResult(50, 0.012, 0.25, 0.6),
        v.FoldResult(40, 0.010, 0.22, 0.6),
        v.FoldResult(45, 0.011, 0.24, 0.6),
    ])


def _good_probes():
    return {"shuffle_e_margin": -0.009,
            "leakage": {"baseline_e_margin": 0.011, "shifted_e_margin": 0.001},
            "liquidations": 0}


def test_verdict_passes_clean_case():
    verdict = v.verdict(_passing_report(), _good_probes(),
                        meta_source="api", round_trip_cost=0.0009, leverage=10.0)
    assert verdict.passed is True
    assert verdict.reasons == []


def test_verdict_fails_on_fallback_meta():
    verdict = v.verdict(_passing_report(), _good_probes(),
                        meta_source="fallback", round_trip_cost=0.0009, leverage=10.0)
    assert verdict.passed is False
    assert any("fallback" in r.lower() for r in verdict.reasons)


def test_verdict_fails_when_a_fold_is_negative():
    rep = _passing_report()
    rep.folds[1] = v.FoldResult(40, -0.004, 0.20, 0.6)
    verdict = v.verdict(rep, _good_probes(), "api", 0.0009, 10.0)
    assert verdict.passed is False
    assert any("fold" in r.lower() for r in verdict.reasons)


def test_verdict_fails_when_liquidation_fired():
    p = _good_probes(); p["liquidations"] = 3
    verdict = v.verdict(_passing_report(), p, "api", 0.0009, 10.0)
    assert verdict.passed is False
    assert any("liquidation" in r.lower() for r in verdict.reasons)


def test_verdict_fails_when_shuffle_does_not_collapse():
    p = _good_probes(); p["shuffle_e_margin"] = 0.010
    verdict = v.verdict(_passing_report(), p, "api", 0.0009, 10.0)
    assert verdict.passed is False
    assert any("shuffle" in r.lower() for r in verdict.reasons)


def test_verdict_fails_when_winrate_out_of_band():
    rep = v.SideReport(side="long", folds=[
        v.FoldResult(50, 0.012, 0.60, 0.6),
        v.FoldResult(40, 0.010, 0.58, 0.6),
    ])
    verdict = v.verdict(rep, _good_probes(), "api", 0.0009, 10.0)
    assert verdict.passed is False
    assert any("win" in r.lower() for r in verdict.reasons)
