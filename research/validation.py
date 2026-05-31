"""
validation.py
Pure validation layer for the MTF research pipeline. No network access.

Holds: purged walk-forward folds, train-only probability-threshold selection,
per-side evaluation, the §4 probes (shuffle / leakage / cost ladder /
liquidation), and the hard PASS/FAIL verdict.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

log = logging.getLogger("validation")

Fold = Tuple[List[int], List[int]]


def purged_folds(n: int, n_folds: int = 5, embargo: int = 120,
                 min_train: int = 500) -> List[Fold]:
    """
    Anchored, expanding-train walk-forward with an embargo gap on the train side.

    For each fold k (1..n_folds): test window = [te_start, te_end). Train =
    [0, te_start - embargo). Folds whose train window is shorter than min_train
    are skipped (never yielded empty or negative).
    """
    block = n // (n_folds + 1)
    folds: List[Fold] = []
    for k in range(1, n_folds + 1):
        te_start = block * k + embargo
        te_end = min(te_start + block, n)
        if te_start >= n or te_end <= te_start:
            break
        train_end = te_start - embargo
        if train_end < min_train:
            continue
        folds.append((list(range(0, train_end)), list(range(te_start, te_end))))
    return folds


def pick_threshold_on_train(probs: np.ndarray, pnl: np.ndarray,
                            grid: Optional[np.ndarray] = None,
                            min_trades: int = 10) -> Optional[float]:
    """
    Choose the probability threshold maximizing mean PnL on TRAIN rows only.
    Returns None if no threshold yields >= min_trades trades. Never reads test.
    """
    if grid is None:
        grid = np.arange(0.5, 0.95, 0.05)
    best_theta: Optional[float] = None
    best_e = -np.inf
    for theta in grid:
        mask = probs > theta
        if int(mask.sum()) < min_trades:
            continue
        e = float(np.mean(pnl[mask]))
        if e > best_e:
            best_e, best_theta = e, float(theta)
    return best_theta


@dataclass
class FoldResult:
    n_trades: int
    e_margin: float
    win_rate: float
    theta: Optional[float]


@dataclass
class SideReport:
    side: str
    folds: List[FoldResult] = field(default_factory=list)

    @property
    def _traded(self) -> List[FoldResult]:
        return [f for f in self.folds if f.n_trades > 0]

    @property
    def mean_e_margin(self) -> float:
        t = self._traded
        return float(np.mean([f.e_margin for f in t])) if t else 0.0

    @property
    def min_e_margin(self) -> float:
        t = self._traded
        return float(np.min([f.e_margin for f in t])) if t else 0.0

    @property
    def cv_e_margin(self) -> float:
        """std/|mean| of per-fold E_margin; inf if mean ~ 0."""
        t = self._traded
        if len(t) < 2:
            return float("inf")
        vals = np.array([f.e_margin for f in t])
        m = float(np.mean(vals))
        return float(np.std(vals) / abs(m)) if abs(m) > 1e-9 else float("inf")

    @property
    def mean_win_rate(self) -> float:
        t = self._traded
        return float(np.mean([f.win_rate for f in t])) if t else 0.0


def default_clf_factory():
    return RandomForestClassifier(n_estimators=100, max_depth=4,
                                  class_weight="balanced", random_state=0, n_jobs=-1)


def _proba_win(clf, X: np.ndarray) -> np.ndarray:
    """P(label==1) robust to a fold whose train labels are single-class."""
    classes = list(clf.classes_)
    if 1 not in classes:
        return np.zeros(len(X))
    return clf.predict_proba(X)[:, classes.index(1)]


def evaluate_side(ds: pd.DataFrame, side: str, feature_cols: List[str],
                  folds: List[Fold], clf_factory: Callable = default_clf_factory,
                  min_trades: int = 10,
                  grid: Optional[np.ndarray] = None) -> SideReport:
    """Fit per fold, pick theta on TRAIN, score margin PnL on TEST."""
    X = ds[feature_cols].to_numpy(float)
    y = ds[f"{side}_label"].to_numpy(int)
    pnl = ds[f"{side}_margin_pnl"].to_numpy(float)
    report = SideReport(side=side)
    for train_idx, test_idx in folds:
        clf = clf_factory()
        clf.fit(X[train_idx], y[train_idx])
        p_tr = _proba_win(clf, X[train_idx])
        theta = pick_threshold_on_train(p_tr, pnl[train_idx], grid, min_trades)
        if theta is None:
            report.folds.append(FoldResult(0, 0.0, 0.0, None))
            continue
        p_te = _proba_win(clf, X[test_idx])
        mask = p_te > theta
        n_tr = int(mask.sum())
        if n_tr == 0:
            report.folds.append(FoldResult(0, 0.0, 0.0, theta))
            continue
        e = float(np.mean(pnl[test_idx][mask]))
        wr = float(np.mean(y[test_idx][mask]))
        report.folds.append(FoldResult(n_tr, e, wr, theta))
    return report


def shuffle_test(ds: pd.DataFrame, side: str, feature_cols: List[str],
                 folds: List[Fold], seed: int = 0,
                 clf_factory: Callable = default_clf_factory) -> float:
    """
    Permute the (label, margin_pnl) target rows relative to features, then
    evaluate. A real edge collapses; if it survives, the CV is contaminated.
    Returns the shuffled mean E_margin.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ds))
    shuffled = ds.copy()
    for col in (f"{side}_label", f"{side}_margin_pnl"):
        shuffled[col] = ds[col].to_numpy()[perm]
    return evaluate_side(shuffled, side, feature_cols, folds, clf_factory).mean_e_margin


def leakage_probe(ds: pd.DataFrame, side: str, feature_cols: List[str],
                  folds: List[Fold],
                  clf_factory: Callable = default_clf_factory) -> Dict[str, float]:
    """
    Shift every feature forward by one bar (use stale features) and confirm
    E_margin drops vs baseline. If it does not drop, a feature is leaking.
    """
    baseline = evaluate_side(ds, side, feature_cols, folds, clf_factory).mean_e_margin
    shifted = ds.copy()
    shifted[feature_cols] = ds[feature_cols].shift(1)
    shifted = shifted.dropna(subset=feature_cols).reset_index(drop=True)
    sfolds = purged_folds(len(shifted), n_folds=len(folds), embargo=50, min_train=300)
    shifted_e = evaluate_side(shifted, side, feature_cols, sfolds, clf_factory).mean_e_margin
    return {"baseline_e_margin": baseline, "shifted_e_margin": shifted_e}


def cost_ladder(rebuild_fn: Callable[[float], pd.DataFrame],
                evaluate_fn: Callable[[pd.DataFrame], float],
                costs: List[float]) -> Dict[float, float]:
    """
    For each round-trip cost, rebuild labels and evaluate E_margin. The cost at
    which E_margin crosses zero is the real slippage tolerance.
    """
    return {float(c): float(evaluate_fn(rebuild_fn(c))) for c in costs}


def liquidation_check(labels: pd.DataFrame, side: str,
                      dn_pct: float, round_trip_cost: float) -> int:
    """
    Count rows whose realized price return is worse than the stop loss, i.e. the
    catastrophic/liquidation branch fired. Must be 0 if the stop invariant holds.
    """
    col = f"{side}_price_return"
    stop_loss = -(dn_pct + round_trip_cost)
    tol = 1e-6
    return int((labels[col].to_numpy(float) < stop_loss - tol).sum())


@dataclass
class Verdict:
    passed: bool
    reasons: List[str] = field(default_factory=list)


def verdict(report: SideReport, probes: Dict, meta_source: str,
            round_trip_cost: float, leverage: float,
            cv_max: float = 1.0,
            win_lo: float = 0.10, win_hi: float = 0.35,
            shuffle_tol: float = 0.005) -> Verdict:
    """
    Apply all §4 gates. PASS requires every gate to hold. A fallback meta source
    forces NOT-VALIDATED regardless of metrics.
    """
    reasons: List[str] = []
    traded = [f for f in report.folds if f.n_trades > 0]

    if meta_source == "fallback":
        reasons.append("meta source is fallback (network blocked) -> NOT-VALIDATED")

    if not traded:
        reasons.append("no fold produced any trades")
    else:
        if report.min_e_margin <= 0:
            reasons.append(f"a fold has E_margin <= 0 (min={report.min_e_margin:.4f})")
        if report.cv_e_margin > cv_max:
            reasons.append(f"fold E_margin variance too high (cv={report.cv_e_margin:.2f} > {cv_max})")
        if not (win_lo <= report.mean_win_rate <= win_hi):
            reasons.append(f"win rate {report.mean_win_rate:.2f} outside [{win_lo}, {win_hi}]")

    # shuffle must collapse toward -(round_trip_cost * leverage)
    expected_floor = -round_trip_cost * leverage
    shuffle_e = probes.get("shuffle_e_margin", 0.0)
    if shuffle_e > expected_floor + shuffle_tol:
        reasons.append(f"shuffle test did not collapse (E={shuffle_e:.4f} > {expected_floor + shuffle_tol:.4f})")

    leak = probes.get("leakage", {})
    if leak.get("shifted_e_margin", 0.0) >= leak.get("baseline_e_margin", 0.0):
        reasons.append("leakage probe: shifted features did not reduce E_margin")

    if probes.get("liquidations", 0) != 0:
        reasons.append(f"liquidation branch fired {probes['liquidations']} times in-sample")

    return Verdict(passed=(len(reasons) == 0), reasons=reasons)
