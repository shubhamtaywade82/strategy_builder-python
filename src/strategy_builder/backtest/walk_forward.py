import math
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Callable, List, Optional, Generator, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score

# Strategy-level deps — imported lazily so WalkForwardValidator can be used
# in isolation (e.g. research pipeline, integration tests) without the full
# strategy_builder stack present.
try:
    from .engine import BacktestEngine
    from ..configuration import Configuration
    from ..exceptions import BacktestError
    from ..features.session_detector import SessionDetector
    from ..features.volatility_profile import VolatilityProfile
    _STRATEGY_DEPS = True
except ImportError:
    BacktestEngine = None  # type: ignore[assignment,misc]
    Configuration = None   # type: ignore[assignment,misc]
    BacktestError = RuntimeError  # type: ignore[assignment,misc]
    SessionDetector = None  # type: ignore[assignment]
    VolatilityProfile = None  # type: ignore[assignment]
    _STRATEGY_DEPS = False


# ── Strategy-level walk-forward (used by agent pipeline) ─────────────────────

class WalkForward:
    """Backtesting walk-forward validator.

    Splits candle data into folds, runs IS/OOS backtests using BacktestEngine,
    and reports stability across folds, regime slices, and a holdout.
    Used by validate_phase.py and supertrend_researcher.py.
    """

    DEFAULT_FOLDS = 5

    def __init__(self, engine: Optional[BacktestEngine] = None):
        self.engine = engine or BacktestEngine()
        self.logger = logging.getLogger(__name__)
        self.in_sample_ratio = Configuration().walk_forward_in_sample_ratio

    def run(
        self,
        strategy: Dict[str, Any],
        candles: List[Any],
        signal_generator: Callable,
        folds: int = DEFAULT_FOLDS,
        mtf_candles: Optional[Dict[str, List[Any]]] = None,
    ) -> Dict[str, Any]:
        fold_size = len(candles) // folds
        if fold_size < 100:
            raise BacktestError(f"Insufficient data for {folds} folds ({len(candles)} candles)")

        fold_results = []
        for i in range(folds):
            fold_start = i * fold_size
            fold_end = min((i + 1) * fold_size, len(candles))
            fold_candles = candles[fold_start:fold_end]
            split_idx = int(len(fold_candles) * self.in_sample_ratio)

            in_sample = fold_candles[:split_idx]
            out_of_sample = fold_candles[split_idx:]

            self.logger.info(f"Fold {i + 1}/{folds}: IS={len(in_sample)} OOS={len(out_of_sample)} candles")

            is_result = self.engine.run(strategy=strategy, candles=in_sample, mtf_candles=mtf_candles, signal_generator=signal_generator)
            oos_result = self.engine.run(strategy=strategy, candles=out_of_sample, mtf_candles=mtf_candles, signal_generator=signal_generator)

            fold_results.append({
                "fold": i + 1,
                "in_sample": is_result["metrics"],
                "out_of_sample": oos_result["metrics"],
                "is_trade_count": len(is_result["trades"]),
                "oos_trade_count": len(oos_result["trades"]),
                "degradation": self._compute_degradation(is_result["metrics"], oos_result["metrics"]),
            })

        aggregate = self._aggregate_folds(fold_results)
        stability = self._compute_stability(fold_results)

        return {
            "folds": fold_results,
            "aggregate": aggregate,
            "stability_score": stability,
            "passes_walk_forward": stability > 0.5 and aggregate["oos_expectancy"] > 0,
        }

    def session_analysis(
        self,
        strategy: Dict[str, Any],
        candles: List[Any],
        signal_generator: Callable,
        mtf_candles: Optional[Dict[str, List[Any]]] = None,
    ) -> Dict[str, Any]:
        session_groups = SessionDetector.group_by_session(candles)
        results = {}
        for session, session_candles in session_groups.items():
            if len(session_candles) < 100:
                continue
            bt = self.engine.run(strategy=strategy, candles=session_candles, mtf_candles=mtf_candles, signal_generator=signal_generator)
            results[session] = bt["metrics"]
        return results

    def anchored_holdout(
        self,
        strategy: Dict[str, Any],
        candles: List[Any],
        signal_generator: Callable,
        train_ratio: float = 0.65,
        mtf_candles: Optional[Dict[str, List[Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        warmup = Configuration().backtest_indicator_warmup
        if len(candles) < warmup + 120:
            return None

        split = min(int(len(candles) * train_ratio), len(candles) - 80)
        if split <= warmup or split >= len(candles) - 20:
            return None

        train = candles[:split]
        test = candles[split:]

        is_bt = self.engine.run(strategy=strategy, candles=train, mtf_candles=mtf_candles, signal_generator=signal_generator)
        oos_bt = self.engine.run(strategy=strategy, candles=test, mtf_candles=mtf_candles, signal_generator=signal_generator)

        return {
            "train_ratio": train_ratio,
            "train_candles": len(train),
            "test_candles": len(test),
            "in_sample": is_bt["metrics"],
            "out_of_sample": oos_bt["metrics"],
        }

    def volatility_regime_slices(
        self,
        strategy: Dict[str, Any],
        candles: List[Any],
        signal_generator: Callable,
        chunk_size: int = 400,
        mtf_candles: Optional[Dict[str, List[Any]]] = None,
    ) -> Dict[str, Any]:
        warmup = Configuration().backtest_indicator_warmup
        if len(candles) < chunk_size + warmup:
            return {"segments": [], "fraction_positive_expectancy": 0.5}

        segments = []
        step = max(chunk_size // 2, 100)
        i = 0
        while i + chunk_size <= len(candles):
            slice_data = candles[i:i + chunk_size]
            reg = VolatilityProfile.regime(slice_data)
            res = self.engine.run(strategy=strategy, candles=slice_data, mtf_candles=mtf_candles, signal_generator=signal_generator)
            exp = float(res["metrics"]["expectancy"])
            segments.append({"start_index": i, "regime": reg, "expectancy": exp, "trade_count": res["metrics"]["trade_count"]})
            i += step

        positive = sum(1 for s in segments if s["expectancy"] > 0)
        frac = round(positive / len(segments), 4) if segments else 0.5
        return {"segments": segments, "fraction_positive_expectancy": frac}

    def _compute_degradation(self, is_metrics: Dict[str, Any], oos_metrics: Dict[str, Any]) -> float:
        is_exp = is_metrics.get("expectancy", 0.0)
        if is_exp == 0:
            return 1.0
        return 1.0 - (oos_metrics.get("expectancy", 0.0) / is_exp)

    def _aggregate_folds(self, folds: List[Dict[str, Any]]) -> Dict[str, Any]:
        oos = [f["out_of_sample"] for f in folds]
        is_ = [f["in_sample"] for f in folds]
        return {
            "oos_expectancy": self._safe_mean([m["expectancy"] for m in oos]),
            "oos_win_rate": self._safe_mean([m["win_rate"] for m in oos]),
            "oos_profit_factor": self._safe_mean([m["profit_factor"] for m in oos]),
            "oos_max_drawdown": max([m["max_drawdown"] for m in oos]) if oos else 0.0,
            "oos_avg_r": self._safe_mean([m["avg_r"] for m in oos]),
            "oos_trade_count": sum([m["trade_count"] for m in oos]),
            "is_expectancy": self._safe_mean([m["expectancy"] for m in is_]),
            "is_profit_factor": self._safe_mean([m["profit_factor"] for m in is_]),
            "avg_degradation": self._safe_mean([f["degradation"] for f in folds]),
        }

    def _compute_stability(self, folds: List[Dict[str, Any]]) -> float:
        oos_expectancies = [f["out_of_sample"]["expectancy"] for f in folds]
        if not oos_expectancies:
            return 0.0
        positive_folds = sum(1 for e in oos_expectancies if e > 0)
        base_stability = positive_folds / len(oos_expectancies)
        mean = sum(oos_expectancies) / len(oos_expectancies)
        variance = sum((e - mean) ** 2 for e in oos_expectancies) / len(oos_expectancies)
        cv = math.sqrt(variance) / abs(mean) if mean != 0 else float("inf")
        return round(max(base_stability - min(cv / 2.0, 0.5), 0.0), 4)

    def _safe_mean(self, values: List[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0


# ── ML-level walk-forward (used by XGBoost/SHAP research pipeline) ───────────

@dataclass
class FoldResult:
    fold: int
    train_auc: float
    test_auc: float
    test_precision: float
    test_recall: float
    n_train: int
    n_test: int


class WalkForwardValidator:
    """Purged + embargoed walk-forward cross-validation for ML models.

    Based on Lopez de Prado's method. Used by the XGBoost/SHAP discovery
    pipeline (strategy_research) and the Kimi integration modules.
    """

    def __init__(self, n_folds: int = 5, embargo: int = 120):
        self.n_folds = n_folds
        self.embargo = embargo
        self.results: List[FoldResult] = []

    def _splits(self, n: int) -> Generator[Tuple[range, range], None, None]:
        fold_size = n // (self.n_folds + 1)
        for k in range(1, self.n_folds + 1):
            tr_end = fold_size * k
            te_start = tr_end + self.embargo
            te_end = min(te_start + fold_size, n)
            if te_start >= n:
                break
            yield range(0, tr_end), range(te_start, te_end)

    def validate(
        self,
        X,
        y,
        model_factory: Callable,
        fit_params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        n = len(X)
        self.results = []
        fit_params = fit_params or {}

        for fold_idx, (tr_idx, te_idx) in enumerate(self._splits(n)):
            X_tr = X.iloc[list(tr_idx)].fillna(0) if hasattr(X, "iloc") else X[list(tr_idx)]
            y_tr = y.iloc[list(tr_idx)] if hasattr(y, "iloc") else y[list(tr_idx)]
            X_te = X.iloc[list(te_idx)].fillna(0) if hasattr(X, "iloc") else X[list(te_idx)]
            y_te = y.iloc[list(te_idx)] if hasattr(y, "iloc") else y[list(te_idx)]

            if len(X_tr) < 100 or len(X_te) < 50:
                continue
            if len(set(y_tr)) < 2 or len(set(y_te)) < 2:
                continue

            model = model_factory()
            try:
                model.fit(X_tr, y_tr, **fit_params)
                te_prob = model.predict_proba(X_te)[:, 1]
                te_pred = model.predict(X_te)
            except Exception:
                continue

            self.results.append(FoldResult(
                fold=fold_idx + 1,
                train_auc=roc_auc_score(y_tr, model.predict_proba(X_tr)[:, 1]) if len(set(y_tr)) > 1 else 0.5,
                test_auc=roc_auc_score(y_te, te_prob) if len(set(y_te)) > 1 else 0.5,
                test_precision=precision_score(y_te, te_pred, zero_division=0),
                test_recall=recall_score(y_te, te_pred, zero_division=0),
                n_train=len(X_tr),
                n_test=len(X_te),
            ))

        return self._aggregate()

    def shuffle_test(self, X, y, model_factory: Callable, n_shuffles: int = 10) -> Dict[str, Any]:
        """Compare real AUC vs shuffled labels to detect leakage."""
        from sklearn.model_selection import train_test_split
        X_arr = X.fillna(0) if hasattr(X, "fillna") else X
        y_arr = y

        X_tr, X_te, y_tr, y_te = train_test_split(X_arr, y_arr, test_size=0.3, random_state=42, stratify=y_arr)
        model = model_factory()
        model.fit(X_tr, y_tr)
        real_auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]) if len(set(y_te)) > 1 else 0.5

        shuffled_aucs = []
        for i in range(n_shuffles):
            y_shuf = np.random.permutation(y_arr.values if hasattr(y_arr, "values") else y_arr)
            _, _, y_tr_s, y_te_s = train_test_split(X_arr, y_shuf, test_size=0.3, random_state=42 + i)
            if len(set(y_te_s)) < 2:
                continue
            m = model_factory()
            m.fit(X_tr, y_tr_s)
            shuffled_aucs.append(roc_auc_score(y_te_s, m.predict_proba(X_te)[:, 1]))

        p_value = sum(1 for a in shuffled_aucs if a >= real_auc) / len(shuffled_aucs) if shuffled_aucs else 1.0
        return {
            "real_auc": real_auc,
            "shuffled_mean": np.mean(shuffled_aucs) if shuffled_aucs else 0.5,
            "p_value": p_value,
            "is_significant": p_value < 0.05,
        }

    def _aggregate(self) -> Dict[str, Any]:
        if not self.results:
            return {"is_valid": False, "folds": 0}
        test_aucs = [r.test_auc for r in self.results]
        return {
            "is_valid": np.mean(test_aucs) > 0.52,
            "folds": len(self.results),
            "avg_test_auc": round(np.mean(test_aucs), 3),
            "min_test_auc": round(np.min(test_aucs), 3),
            "stability": round(sum(1 for a in test_aucs if a > 0.5) / len(test_aucs), 2),
            "degradation": round(np.mean(test_aucs) - np.mean([r.train_auc for r in self.results]), 3),
            "fold_results": [
                {"fold": r.fold, "test_auc": r.test_auc, "test_precision": r.test_precision,
                 "test_recall": r.test_recall, "n_train": r.n_train, "n_test": r.n_test}
                for r in self.results
            ],
        }
