# MTF Research Validation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the supervised MTF research pipeline in `research/` so its
fold-by-fold expectancy honors the advisor §4 verification contract (purged CV,
shuffle test, leakage probe, cost ladder, per-symbol funding/mmr, hard verdict).

**Architecture:** Three new flat modules in `research/`. `validation.py` is
pure (no network) and holds purged folds + train-only θ selection + probes +
verdict gates. `binance_meta.py` fetches per-symbol funding + mmr with a disk
cache and a loud fallback. `run_research.py` orchestrates one symbol end-to-end
and emits a PASS/FAIL verdict + JSON. Core modules `mtf_research.py` and
`barrier_leverage.py` are untouched.

**Tech Stack:** Python 3.8, pandas 2.x, numpy, scikit-learn 1.3.2, requests,
pytest 8.3. Classifier: `RandomForestClassifier(class_weight="balanced")`
(binary, deterministic, has `predict_proba`).

---

## Conventions (read once)

- **Flat imports.** `research/` has no `__init__.py`; modules import each other
  as `from validation import ...`. Scripts and tests must put `research/` on
  `sys.path`. Every test file in this plan starts with this preamble:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research"))
```

- **Python 3.8.** Use `from __future__ import annotations` at the top of every
  new module. For runtime typing use `typing.List/Dict/Tuple/Optional` — not
  `list[int]` / `X | Y` evaluated at runtime.
- **Run tests** from repo root with the venv:
  `./venv/bin/pytest tests/test_validation.py -v`
- **Side strings** are `"long"` and `"short"`. For side `s`, the label column is
  `f"{s}_label"`, margin column `f"{s}_margin_pnl"`, price column
  `f"{s}_price_return"` (these come from `triple_barrier_both_sides`, which
  prefixes `_label_one_side` output with `long_`/`short_`).
- **Determinism:** seed all RNG. Use `np.random.default_rng(0)` and
  `random_state=0`.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `research/validation.py` | Pure: folds, θ policy, eval, probes, verdict. No network. |
| `research/binance_meta.py` | Network: funding profile + mmr, cache + fallback. |
| `research/run_research.py` | Orchestrator + CLI: dataset → eval → probes → report. |
| `tests/test_validation.py` | Unit tests for `validation.py` (synthetic data). |
| `tests/test_binance_meta.py` | Unit tests for `binance_meta.py` (mocked HTTP + fallback). |

---

## Task 1: Purged walk-forward folds

**Files:**
- Create: `research/validation.py`
- Test: `tests/test_validation.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_validation.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research"))

import numpy as np
import pandas as pd
import pytest

import validation as v


def test_purged_folds_no_overlap_and_embargo():
    folds = v.purged_folds(n=6000, n_folds=5, embargo=120, min_train=500)
    assert len(folds) >= 1
    for train_idx, test_idx in folds:
        assert len(train_idx) >= 500
        assert min(test_idx) - max(train_idx) > 120          # embargo gap respected
        assert set(train_idx).isdisjoint(set(test_idx))      # never overlap
        assert min(train_idx) >= 0 and max(test_idx) < 6000  # in range


def test_purged_folds_skips_tiny_train():
    # small n: early folds cannot meet min_train and must be skipped, never empty/negative
    folds = v.purged_folds(n=2000, n_folds=5, embargo=120, min_train=500)
    for train_idx, test_idx in folds:
        assert len(train_idx) >= 500
        assert len(test_idx) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/pytest tests/test_validation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'validation'`

- [ ] **Step 3: Write minimal implementation**

```python
# research/validation.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/pytest tests/test_validation.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add research/validation.py tests/test_validation.py
git commit -m "feat(research): purged walk-forward folds for MTF validation"
```

---

## Task 2: Train-only threshold selection

**Files:**
- Modify: `research/validation.py`
- Test: `tests/test_validation.py`

- [ ] **Step 1: Write the failing test**

```python
def test_pick_threshold_prefers_profitable_cut():
    # probs ascending; pnl positive only for the top scores -> high theta wins
    probs = np.linspace(0.1, 0.9, 100)
    pnl = np.where(probs > 0.7, 0.10, -0.05)
    theta = v.pick_threshold_on_train(probs, pnl, grid=np.arange(0.5, 0.95, 0.05),
                                       min_trades=5)
    assert theta is not None and theta >= 0.7


def test_pick_threshold_returns_none_when_too_few_trades():
    probs = np.full(100, 0.2)          # nothing crosses any grid point
    pnl = np.full(100, 0.10)
    theta = v.pick_threshold_on_train(probs, pnl, grid=np.arange(0.5, 0.95, 0.05),
                                       min_trades=5)
    assert theta is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/pytest tests/test_validation.py::test_pick_threshold_prefers_profitable_cut -v`
Expected: FAIL with `AttributeError: module 'validation' has no attribute 'pick_threshold_on_train'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to research/validation.py

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/pytest tests/test_validation.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add research/validation.py tests/test_validation.py
git commit -m "feat(research): train-only threshold selection (anti-snoop)"
```

---

## Task 3: Per-side evaluation across folds

**Files:**
- Modify: `research/validation.py`
- Test: `tests/test_validation.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/pytest tests/test_validation.py::test_evaluate_side_finds_positive_edge_on_predictive_data -v`
Expected: FAIL with `AttributeError: ... 'evaluate_side'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to research/validation.py
from sklearn.ensemble import RandomForestClassifier


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/pytest tests/test_validation.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add research/validation.py tests/test_validation.py
git commit -m "feat(research): per-side fold evaluation with train-theta"
```

---

## Task 4: Shuffle test and leakage probe

**Files:**
- Modify: `research/validation.py`
- Test: `tests/test_validation.py`

- [ ] **Step 1: Write the failing test**

```python
def test_shuffle_collapses_edge():
    ds = _synthetic_dataset()
    folds = v.purged_folds(len(ds), n_folds=4, embargo=50, min_train=300)
    baseline = v.evaluate_side(ds, "long", ["f0", "f1"], folds).mean_e_margin
    shuffled = v.shuffle_test(ds, "long", ["f0", "f1"], folds, seed=0)
    assert shuffled < baseline                 # breaking feature->label kills edge
    assert shuffled < 0.02                      # collapses toward loss/zero


def test_leakage_probe_drops_when_feature_shifted():
    ds = _synthetic_dataset()
    folds = v.purged_folds(len(ds), n_folds=4, embargo=50, min_train=300)
    res = v.leakage_probe(ds, "long", ["f0", "f1"], folds)
    assert res["shifted_e_margin"] < res["baseline_e_margin"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/pytest tests/test_validation.py::test_shuffle_collapses_edge -v`
Expected: FAIL with `AttributeError: ... 'shuffle_test'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to research/validation.py

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/pytest tests/test_validation.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add research/validation.py tests/test_validation.py
git commit -m "feat(research): shuffle test and leakage probe"
```

---

## Task 5: Cost ladder and liquidation check

**Files:**
- Modify: `research/validation.py`
- Test: `tests/test_validation.py`

- [ ] **Step 1: Write the failing test**

```python
def test_cost_ladder_is_monotonic_non_increasing():
    # rebuild_fn lowers margin pnl as cost rises; evaluate_fn returns mean pnl
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
    # stop = 0.005, cost = 0.0009 -> stop loss ~ -0.0059; -0.095 is catastrophic
    cnt = v.liquidation_check(df, "long", dn_pct=0.005, round_trip_cost=0.0009)
    assert cnt == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/pytest tests/test_validation.py::test_cost_ladder_is_monotonic_non_increasing -v`
Expected: FAIL with `AttributeError: ... 'cost_ladder'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to research/validation.py

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/pytest tests/test_validation.py -v`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add research/validation.py tests/test_validation.py
git commit -m "feat(research): cost ladder and liquidation check"
```

---

## Task 6: Hard verdict gates

**Files:**
- Modify: `research/validation.py`
- Test: `tests/test_validation.py`

- [ ] **Step 1: Write the failing test**

```python
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
    p = _good_probes(); p["shuffle_e_margin"] = 0.010   # edge survived shuffle
    verdict = v.verdict(_passing_report(), p, "api", 0.0009, 10.0)
    assert verdict.passed is False
    assert any("shuffle" in r.lower() for r in verdict.reasons)


def test_verdict_fails_when_winrate_out_of_band():
    rep = v.SideReport(side="long", folds=[
        v.FoldResult(50, 0.012, 0.60, 0.6),   # 60% win rate -> barriers/fees wrong
        v.FoldResult(40, 0.010, 0.58, 0.6),
    ])
    verdict = v.verdict(rep, _good_probes(), "api", 0.0009, 10.0)
    assert verdict.passed is False
    assert any("win" in r.lower() for r in verdict.reasons)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/pytest tests/test_validation.py::test_verdict_passes_clean_case -v`
Expected: FAIL with `AttributeError: ... 'verdict'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to research/validation.py

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/pytest tests/test_validation.py -v`
Expected: PASS (16 passed)

- [ ] **Step 5: Commit**

```bash
git add research/validation.py tests/test_validation.py
git commit -m "feat(research): hard verdict gates for the research contract"
```

---

## Task 7: Funding profile fetch with fallback

**Files:**
- Create: `research/binance_meta.py`
- Test: `tests/test_binance_meta.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_binance_meta.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research"))

import binance_meta as bm


class _FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _FakeSession:
    """Returns queued responses by URL substring; raises if asked for an unknown URL."""
    def __init__(self, routes):
        self._routes = routes

    def get(self, url, params=None, timeout=None):
        for key, resp in self._routes.items():
            if key in url:
                return resp
        raise AssertionError(f"unexpected URL {url}")


def test_funding_profile_parses_api():
    session = _FakeSession({
        "fundingInfo": _FakeResp([{"symbol": "SOLUSDT", "fundingIntervalHours": 4}]),
        "fundingRate": _FakeResp([{"fundingRate": "0.0001"}, {"fundingRate": "0.0003"},
                                  {"fundingRate": "-0.0001"}, {"fundingRate": "0.0005"}]),
    })
    prof = bm.fetch_funding_profile("SOLUSDT", session=session)
    assert prof.source == "api"
    assert prof.interval_h == 4
    assert prof.avg_rate > 0
    assert prof.p90_rate >= prof.avg_rate


def test_funding_profile_falls_back_on_network_error():
    class _Boom:
        def get(self, *a, **k):
            raise ConnectionError("geo-blocked")
    prof = bm.fetch_funding_profile("SOLUSDT", session=_Boom())
    assert prof.source == "fallback"
    assert prof.interval_h == 8.0          # documented default
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/pytest tests/test_binance_meta.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'binance_meta'`

- [ ] **Step 3: Write minimal implementation**

```python
# research/binance_meta.py
"""
binance_meta.py
Per-symbol economics for the labeling layer: funding profile + maintenance
margin rate. Network access with a loud fallback (geo-block safe). A fallback
result is stamped source="fallback" so the verdict can force NOT-VALIDATED.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import requests

log = logging.getLogger("binance_meta")

FAPI_BASE = "https://fapi.binance.com"
CACHE_DIR = Path(__file__).resolve().parent / "cache"

# Documented fallback defaults (used only when the network is blocked).
_FALLBACK_INTERVAL_H = 8.0
_FALLBACK_RATE = 0.0001
_FALLBACK_MMR = 0.005


@dataclass
class FundingProfile:
    interval_h: float
    avg_rate: float
    p90_rate: float
    anchor_hour: int
    source: str          # "api" | "cache" | "fallback"


def fetch_funding_profile(symbol: str,
                          session: Optional[requests.Session] = None,
                          base_url: str = FAPI_BASE) -> FundingProfile:
    s = session or requests.Session()
    try:
        info = s.get(f"{base_url}/fapi/v1/fundingInfo", timeout=10)
        info.raise_for_status()
        interval = _FALLBACK_INTERVAL_H
        for row in info.json():
            if row.get("symbol") == symbol:
                interval = float(row.get("fundingIntervalHours", _FALLBACK_INTERVAL_H))
                break

        hist = s.get(f"{base_url}/fapi/v1/fundingRate",
                     params={"symbol": symbol, "limit": 1000}, timeout=10)
        hist.raise_for_status()
        rates = np.array([float(r["fundingRate"]) for r in hist.json()], dtype=float)
        if rates.size == 0:
            raise ValueError("empty funding history")
        return FundingProfile(
            interval_h=interval,
            avg_rate=float(np.mean(np.abs(rates))),
            p90_rate=float(np.percentile(np.abs(rates), 90)),
            anchor_hour=0,
            source="api",
        )
    except Exception as exc:  # network blocked, parse error, empty history
        log.warning("funding fetch failed for %s (%s) -> FALLBACK defaults; "
                    "run will be NOT-VALIDATED", symbol, exc)
        return FundingProfile(_FALLBACK_INTERVAL_H, _FALLBACK_RATE, _FALLBACK_RATE,
                              0, "fallback")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/pytest tests/test_binance_meta.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add research/binance_meta.py tests/test_binance_meta.py
git commit -m "feat(research): per-symbol funding profile fetch with fallback"
```

---

## Task 8: Maintenance margin fetch and cached loader

**Files:**
- Modify: `research/binance_meta.py`
- Test: `tests/test_binance_meta.py`

- [ ] **Step 1: Write the failing test**

```python
def test_fetch_mmr_picks_bracket_for_notional():
    session = _FakeSession({
        "leverageBracket": _FakeResp([{
            "symbol": "SOLUSDT",
            "brackets": [
                {"bracket": 1, "notionalCap": 5000, "maintMarginRatio": 0.004},
                {"bracket": 2, "notionalCap": 50000, "maintMarginRatio": 0.005},
            ],
        }]),
    })
    mmr, source = bm.fetch_mmr("SOLUSDT", notional=1000, session=session)
    assert mmr == 0.004 and source == "api"


def test_fetch_mmr_falls_back():
    class _Boom:
        def get(self, *a, **k):
            raise ConnectionError("blocked")
    mmr, source = bm.fetch_mmr("SOLUSDT", notional=1000, session=_Boom())
    assert source == "fallback" and mmr == 0.005


def test_load_meta_combines_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(bm, "CACHE_DIR", tmp_path)
    session = _FakeSession({
        "fundingInfo": _FakeResp([{"symbol": "SOLUSDT", "fundingIntervalHours": 8}]),
        "fundingRate": _FakeResp([{"fundingRate": "0.0001"}]),
        "leverageBracket": _FakeResp([{"symbol": "SOLUSDT", "brackets": [
            {"bracket": 1, "notionalCap": 50000, "maintMarginRatio": 0.005}]}]),
    })
    meta = bm.load_meta("SOLUSDT", notional=1000, session=session)
    assert meta["source"] == "api"
    assert (tmp_path / "SOLUSDT_meta.json").exists()      # cached to disk
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/pytest tests/test_binance_meta.py::test_fetch_mmr_picks_bracket_for_notional -v`
Expected: FAIL with `AttributeError: ... 'fetch_mmr'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to research/binance_meta.py

def fetch_mmr(symbol: str, notional: float,
              session: Optional[requests.Session] = None,
              base_url: str = FAPI_BASE) -> Tuple[float, str]:
    """Maintenance-margin ratio for the bracket covering `notional`."""
    s = session or requests.Session()
    try:
        r = s.get(f"{base_url}/fapi/v1/leverageBracket",
                  params={"symbol": symbol}, timeout=10)
        r.raise_for_status()
        payload = r.json()
        entry = payload[0] if isinstance(payload, list) else payload
        brackets = sorted(entry["brackets"], key=lambda b: b["notionalCap"])
        for b in brackets:
            if notional <= b["notionalCap"]:
                return float(b["maintMarginRatio"]), "api"
        return float(brackets[-1]["maintMarginRatio"]), "api"
    except Exception as exc:
        log.warning("mmr fetch failed for %s (%s) -> FALLBACK %.4f",
                    symbol, exc, _FALLBACK_MMR)
        return _FALLBACK_MMR, "fallback"


def load_meta(symbol: str, notional: float = 1000.0,
              session: Optional[requests.Session] = None,
              base_url: str = FAPI_BASE) -> dict:
    """
    Combine funding + mmr into a dict and cache to disk. source is "fallback" if
    EITHER component fell back, so a degraded run cannot look validated.
    """
    prof = fetch_funding_profile(symbol, session, base_url)
    mmr, mmr_source = fetch_mmr(symbol, notional, session, base_url)
    source = "fallback" if "fallback" in (prof.source, mmr_source) else "api"
    meta = {**asdict(prof), "maint_margin_rate": mmr, "source": source,
            "symbol": symbol}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{symbol}_meta.json").write_text(json.dumps(meta, indent=2))
    return meta
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/pytest tests/test_binance_meta.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add research/binance_meta.py tests/test_binance_meta.py
git commit -m "feat(research): mmr fetch and cached combined meta loader"
```

---

## Task 9: Orchestrator and CLI

**Files:**
- Create: `research/run_research.py`
- Test: `tests/test_run_research.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_research.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research"))

import numpy as np
import pandas as pd

import run_research as rr


def test_select_feature_columns_excludes_labels_and_ohlc():
    ds = pd.DataFrame(columns=[
        "open_time", "close_time", "open", "high", "low", "close", "volume",
        "taker_buy_base", "taker_buy_quote", "n_trades", "entry_time",
        "fill_price", "best_side",
        "long_label", "long_margin_pnl", "long_price_return",
        "short_label", "short_margin_pnl", "short_price_return",
        "ltf_rvol", "1h_trend", "4h_ret",      # real features
    ])
    cols = rr.select_feature_columns(ds)
    assert set(cols) == {"ltf_rvol", "1h_trend", "4h_ret"}


def test_build_report_dict_shape():
    # minimal hand-built inputs -> stable serializable report
    import validation as v
    report = v.SideReport("long", [v.FoldResult(10, 0.01, 0.2, 0.6)])
    verdict = v.Verdict(passed=True, reasons=[])
    out = rr.build_report_dict(
        symbol="SOLUSDT", side="long", report=report, verdict=verdict,
        probes={"shuffle_e_margin": -0.009,
                "leakage": {"baseline_e_margin": 0.01, "shifted_e_margin": 0.0},
                "liquidations": 0,
                "cost_ladder": {0.0009: 0.01, 0.0012: 0.0, 0.0015: -0.01}},
        meta={"source": "api", "interval_h": 8.0, "maint_margin_rate": 0.005})
    assert out["symbol"] == "SOLUSDT"
    assert out["verdict"]["passed"] is True
    assert out["folds"][0]["e_margin"] == 0.01
    import json
    json.dumps(out)        # must be JSON-serializable
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/pytest tests/test_run_research.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'run_research'`

- [ ] **Step 3: Write minimal implementation**

```python
# research/run_research.py
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


def build_dataset(symbol: str, days: int, meta: Dict) -> "pd.DataFrame":
    loader = BinanceUMKlineLoader()
    end = int(time.time() * 1000)
    start = end - days * 86_400_000
    frames = {tf: loader.fetch(symbol, tf, start, end)
              for tf in ["1m", "15m", "1h", "4h", "1d"]}
    feats = build_mtf_features(frames, base_tf="1m")
    cfg = _config_from_meta(meta)
    labels = triple_barrier_both_sides(frames["1m"], cfg)
    ds = feats.join(labels.set_index("entry_idx"), how="inner").dropna().reset_index(drop=True)
    return ds


def _config_from_meta(meta: Dict, round_trip_cost: float = 0.0009) -> LevBarrierConfig:
    return LevBarrierConfig(
        up_pct=0.01, dn_pct=0.005, max_horizon=120, leverage=10.0,
        round_trip_cost=round_trip_cost,
        maint_margin_rate=meta.get("maint_margin_rate", 0.005),
        funding_rate=meta.get("avg_rate", 0.0001),
        funding_interval_h=meta.get("interval_h", 8.0),
        funding_anchor_utc_hour=meta.get("anchor_hour", 0),
    )


def run_symbol(symbol: str, days: int = 45, round_trip_cost: float = 0.0009) -> Dict:
    meta = load_meta(symbol)
    ds = build_dataset(symbol, days, meta)
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
                lambda c: triple_barrier_both_sides(_relabel_frame(ds), _config_from_meta(meta, c))
                          if False else _cost_ladder_ds(symbol, days, meta, c),
                lambda d: float(d[f"{side}_margin_pnl"].mean()),
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


# Cost ladder rebuilds labels at a new cost on the SAME 1m frame already fetched.
_FRAME_CACHE: Dict[str, pd.DataFrame] = {}


def _cost_ladder_ds(symbol: str, days: int, meta: Dict, cost: float) -> pd.DataFrame:
    base = _FRAME_CACHE.get(symbol)
    if base is None:
        loader = BinanceUMKlineLoader()
        end = int(time.time() * 1000)
        start = end - days * 86_400_000
        base = loader.fetch(symbol, "1m", start, end)
        _FRAME_CACHE[symbol] = base
    return triple_barrier_both_sides(base, _config_from_meta(meta, cost))


def _relabel_frame(ds):  # placeholder kept out of the active path
    return ds


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
    args = ap.parse_args()
    run_symbol(args.symbol, args.days)


if __name__ == "__main__":
    main()
```

> **Note for implementer:** the `lambda c: ... if False else _cost_ladder_ds(...)`
> in `run_symbol` is intentionally simplified during TDD — the active path is
> `_cost_ladder_ds`. When you reach Step 3, write the cost-ladder rebuild fn
> cleanly as `lambda c: _cost_ladder_ds(symbol, days, meta, c)` and delete the
> `_relabel_frame` placeholder. The test only exercises `select_feature_columns`
> and `build_report_dict`, both of which are pure.

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/pytest tests/test_run_research.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Clean up the cost-ladder lambda**

Replace the `cost_ladder` rebuild argument in `run_symbol` with the clean form
and remove `_relabel_frame`:

```python
            "cost_ladder": v.cost_ladder(
                lambda c: _cost_ladder_ds(symbol, days, meta, c),
                lambda d: float(d[f"{side}_margin_pnl"].mean()),
                costs=[0.0009, 0.0012, 0.0015]),
```

- [ ] **Step 6: Run the full suite**

Run: `./venv/bin/pytest tests/test_validation.py tests/test_binance_meta.py tests/test_run_research.py -v`
Expected: PASS (all green)

- [ ] **Step 7: Commit**

```bash
git add research/run_research.py tests/test_run_research.py
git commit -m "feat(research): orchestrator + CLI emitting PASS/FAIL verdict and JSON"
```

---

## Task 10: Retire `train_model.py` and document

**Files:**
- Delete: `research/train_model.py`
- Create: `research/README.md`

- [ ] **Step 1: Confirm `run_research.py` supersedes `train_model.py`**

Run: `./venv/bin/python -c "import sys; sys.path.insert(0,'research'); import run_research; print('ok')"`
Expected: `ok` (imports resolve)

- [ ] **Step 2: Delete the superseded file**

```bash
git rm research/train_model.py
```

- [ ] **Step 3: Write `research/README.md`**

```markdown
# MTF Research Pipeline (Binance USD-M)

Leakage-safe supervised research toward a net +1% price move (10% margin @ 10x),
both directions. NOT an execution engine.

## Modules
- `mtf_research.py`   — kline loader + leakage-safe MTF features + triple-barrier labels
- `barrier_leverage.py` — both-side, 10x leverage-aware labels + liquidation guard
- `binance_meta.py`   — per-symbol funding + mmr (network, with fallback)
- `validation.py`     — purged folds, train-only theta, probes, verdict gates
- `run_research.py`   — orchestrator + CLI

## Run
    cd research
    python run_research.py --symbol SOLUSDT --days 45

Outputs `research/output/<symbol>_<date>.json` and a console PASS/FAIL per side.
A run whose funding/mmr came from the fallback path is marked NOT-VALIDATED.

## Tests
    ./venv/bin/pytest tests/test_validation.py tests/test_binance_meta.py tests/test_run_research.py -v
```

- [ ] **Step 4: Commit**

```bash
git add research/README.md
git commit -m "docs(research): retire train_model, add pipeline README"
```

---

## Self-Review

**Spec coverage:**
- §4.1 win-rate sanity → Task 6 `verdict` win band gate ✓
- §4.2 leakage probe → Task 4 `leakage_probe` + Task 6 gate ✓
- §4.3 shuffle test → Task 4 `shuffle_test` + Task 6 gate ✓
- §4.4 fold variance → Task 3 `cv_e_margin` + Task 6 cv gate ✓
- §4.5 cost stress → Task 5 `cost_ladder` ✓
- Leverage §4.1 zero liquidations → Task 5 `liquidation_check` + Task 6 gate ✓
- Leverage §4.2 margin expectancy → Task 3 uses `{side}_margin_pnl` ✓
- Per-symbol funding/mmr → Tasks 7–8 ✓
- Fallback ⇒ NOT-VALIDATED → Task 6 + Task 8 source propagation ✓
- Train-only θ → Task 2 + Task 3 ✓
- Two binary models → Task 3 (per-side `{side}_label`) ✓
- Fold bug fix → Task 1 `purged_folds` ✓
- OOS rule extraction → Task 9 `_print_rule` (train-only) ✓

**Decisions implemented:** all four approved knobs (train-θ, two binary models,
hard gates, fallback=NOT-VALIDATED) are covered.

**Type consistency:** `SideReport`/`FoldResult`/`Verdict` defined in Task 3/6 and
used identically in Task 9. Side strings `"long"`/`"short"` and column names
`{side}_label`/`{side}_margin_pnl`/`{side}_price_return` consistent across tasks.
`load_meta` dict keys (`source`, `interval_h`, `maint_margin_rate`, `avg_rate`,
`anchor_hour`) consistent between Task 8 and `_config_from_meta` in Task 9.

**Out of scope (unchanged):** `researcher.py`, `researcher_robust.py`,
`build_mtf_edge.py`, `build_regime_switcher.py` — the old data-snooping track.
A separate decision; not touched here.
```
