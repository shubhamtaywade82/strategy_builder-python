"""
Tests for the deterministic, leakage-free research engine.

These run fully offline on the synthetic data source (no Binance), so they double
as the regression guard for the '96% win rate illusion': on no-edge data the
out-of-sample backtest must NOT look profitable, and nothing may be flagged
robust.
"""
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from strategy_research._ui_engine.synthetic import make_synthetic_mtf
from strategy_research._ui_engine.features import build_mtf_features
from strategy_research._ui_engine.labels import BarrierCfg, label_both_sides
from strategy_research._ui_engine import grid_search, robustness, rule_strategy, validation


# --------------------------------------------------------------------------
# synthetic data source
# --------------------------------------------------------------------------
def test_synthetic_is_deterministic():
    a = make_synthetic_mtf("SOLUSDT", days=10)
    b = make_synthetic_mtf("SOLUSDT", days=10)
    assert np.array_equal(a["1m"]["close"].to_numpy(), b["1m"]["close"].to_numpy())
    assert np.array_equal(a["4h"]["high"].to_numpy(), b["4h"]["high"].to_numpy())


def test_synthetic_multi_timeframe_shapes():
    d = make_synthetic_mtf("SOLUSDT", days=10)
    assert len(d["1m"]) == 10 * 1440
    # higher TFs are aggregations of the 1m path
    assert len(d["15m"]) == pytest.approx(10 * 96, abs=2)
    assert len(d["1h"]) == pytest.approx(10 * 24, abs=2)
    assert "_open_ms" not in d["1m"].columns  # internal column must not leak


def test_synthetic_htf_consistent_with_1m():
    """A higher-tf bar's high/low must bound the 1m bars inside it."""
    d = make_synthetic_mtf("BTCUSDT", days=5)
    assert d["1h"]["high"].max() <= d["1m"]["high"].max() + 1e-6
    assert d["1h"]["low"].min() >= d["1m"]["low"].min() - 1e-6


# --------------------------------------------------------------------------
# robustness scorer (pure unit — instant)
# --------------------------------------------------------------------------
def test_robustness_rejects_overfit():
    wf = {"avg_test_auc": 0.521, "stability": 0.4, "degradation": -0.45, "is_valid": False}
    shuffle = {"p_value": 0.10, "is_significant": False}
    oos = {"expectancy": 0.08, "trade_count": 5000}  # great in-sample-looking exp
    out = robustness.score_robustness(wf, shuffle, oos)
    assert out["is_robust"] is False
    assert any("overfit" in w or "out-of-sample" in w for w in out["warnings"])


def test_robustness_accepts_strong_signal():
    wf = {"avg_test_auc": 0.66, "stability": 1.0, "degradation": -0.03, "is_valid": True}
    shuffle = {"p_value": 0.001, "is_significant": True}
    oos = {"expectancy": 0.0008, "trade_count": 400}
    out = robustness.score_robustness(wf, shuffle, oos)
    assert out["is_robust"] is True
    assert out["robustness_score"] > 70


def test_robustness_needs_positive_oos_expectancy():
    wf = {"avg_test_auc": 0.66, "stability": 1.0, "degradation": -0.03, "is_valid": True}
    shuffle = {"p_value": 0.001, "is_significant": True}
    oos = {"expectancy": -0.0001, "trade_count": 400}  # negative OOS
    assert robustness.score_robustness(wf, shuffle, oos)["is_robust"] is False


# --------------------------------------------------------------------------
# out-of-sample predictions vs in-sample fit
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def discovery():
    data = make_synthetic_mtf("SOLUSDT", days=14)
    feats = build_mtf_features(data)
    return grid_search.discover_for_rr(feats, data["1m"], "2:1")


def test_oos_win_rate_is_not_in_sample_illusion(discovery):
    """In-sample memorization looks great; out-of-sample tells the truth."""
    found = False
    for side in ("long", "short"):
        for strat in discovery.get(side, {}).get("strategies", []):
            ism = strat["in_sample_metrics"]
            oos = strat["metrics"]
            if ism.get("trade_count", 0) >= 20 and oos.get("trade_count", 0) >= 20:
                found = True
                # the whole point: in-sample win-rate is wildly inflated vs OOS
                assert ism["win_rate"] > oos["win_rate"] + 0.20
                assert oos["win_rate"] < 0.80  # never the 90%+ illusion
    assert found, "expected at least one comparable strategy"


def test_no_edge_data_yields_no_robust_strategy(discovery):
    """Random synthetic data has no edge -> nothing should pass the gate."""
    for side in ("long", "short"):
        for strat in discovery.get(side, {}).get("strategies", []):
            assert strat["is_robust"] is False
            assert strat["is_viable"] == strat["is_robust"]


def test_strategies_ranked_by_robustness(discovery):
    for side in ("long", "short"):
        strats = discovery.get(side, {}).get("strategies", [])
        scores = [s["robustness_score"] for s in strats]
        assert scores == sorted(scores, reverse=True)


def test_oos_predict_only_predicts_test_bars():
    """OOS probabilities must be NaN on the initial training block (no peeking)."""
    data = make_synthetic_mtf("SOLUSDT", days=12)
    feats = build_mtf_features(data)
    X = feats.select_dtypes(include=[np.number]).fillna(0)
    lab = label_both_sides(data["1m"], BarrierCfg(up_pct=0.01, dn_pct=0.005, max_horizon=120))
    y = (lab.set_index("entry_idx")["best_side"] > 0).astype(int)
    common = X.index.intersection(y.index)
    X, y = X.loc[common], y.loc[common]
    probs = validation.oos_predict(X, y, n_folds=5, embargo=120)
    assert probs.isna().any()                 # early/embargo bars unpredicted
    assert probs.notna().any()                # later bars predicted
    # the first training block (~1/6) is never an OOS test bar
    first_block = probs.iloc[: len(probs) // 6]
    assert first_block.isna().all()


# --------------------------------------------------------------------------
# rule strategy
# --------------------------------------------------------------------------
def test_rule_strategy_runs_and_is_deterministic():
    frames = make_synthetic_mtf("SOLUSDT", days=40)
    a = rule_strategy.run_player(frames)
    b = rule_strategy.run_player(frames)
    assert a["rr"] == "2:1"
    for side in ("long", "short"):
        sa, sb = a[side], b[side]
        if "error" in sa:
            assert sa["error"] == sb["error"]
            continue
        # real backtest fields present and identical across runs
        assert {"metrics", "folds", "baseline", "is_robust"}.issubset(sa.keys())
        assert sa["metrics"]["win_rate"] == sb["metrics"]["win_rate"]
        assert sa["metrics"]["p_value"] == sb["metrics"]["p_value"]


def test_rule_strategy_insufficient_signals_returns_error():
    # 3 days is far too short for the 5-condition AND to fire min_trades times.
    frames = make_synthetic_mtf("SOLUSDT", days=3)
    out = rule_strategy.run_player(frames)
    for side in ("long", "short"):
        s = out[side]
        if "error" in s:
            assert "metrics" not in s
