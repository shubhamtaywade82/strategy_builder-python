"""
Walk-Forward Validation with Purged Cross-Validation
=====================================================
Validates strategies on out-of-sample data with leakage protection.

Key invariants:
- Purging: Remove bars around test-set labels from training
- Embargo: Gap between train and test to prevent overlap leakage
- Anchored walk-forward: Each fold adds more training data

Based on: Advances in Financial Machine Learning (López de Prado)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Generator, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

log = logging.getLogger("walkforward")


def purged_walk_forward_splits(
    n: int,
    n_folds: int = 5,
    embargo: int = 120,
) -> Generator[Tuple[range, range], None, None]:
    """Generate purged + embargoed walk-forward splits.

    Args:
        n: Total number of samples
        n_folds: Number of CV folds
        embargo: Bars to embargo between train and test

    Yields:
        (train_indices, test_indices) tuples
    """
    fold_size = n // (n_folds + 1)

    for k in range(1, n_folds + 1):
        train_end = fold_size * k
        test_start = train_end + embargo
        test_end = min(test_start + fold_size, n)

        if test_start >= n:
            break

        # Train: [0, train_end - embargo)
        # Embargo zone excluded to prevent label overlap
        train_start = 0
        train_stop = train_end - embargo // 2

        yield range(train_start, max(train_stop, 100)), range(test_start, test_end)


def anchored_walk_forward_splits(
    n: int,
    n_folds: int = 5,
    embargo: int = 120,
) -> Generator[Tuple[range, range], None, None]:
    """Anchored walk-forward: training set grows with each fold.

    More realistic for financial time series where more history is better.
    """
    fold_size = n // (n_folds + 1)

    for k in range(1, n_folds + 1):
        train_end = fold_size * k
        test_start = train_end + embargo
        test_end = min(test_start + fold_size, n)

        if test_start >= n:
            break

        yield range(0, train_end), range(test_start, test_end)


@dataclass
class FoldResult:
    """Results from a single fold."""
    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    n_train: int
    n_test: int


class WalkForwardValidator:
    """Walk-forward validation with purged CV."""

    def __init__(
        self,
        n_folds: int = 5,
        embargo: int = 120,
        anchored: bool = True,
    ):
        self.n_folds = n_folds
        self.embargo = embargo
        self.anchored = anchored
        self.results: List[FoldResult] = []

    def validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_factory,
        fit_params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Run walk-forward validation.

        Args:
            X: Feature matrix
            y: Target vector
            model_factory: Callable that returns a fresh model instance
            fit_params: Extra params for model.fit()

        Returns:
            Summary dict with fold results and aggregate metrics
        """
        n = len(X)
        splitter = anchored_walk_forward_splits if self.anchored else purged_walk_forward_splits
        self.results = []

        log.info(f"Walk-forward: n={n}, folds={self.n_folds}, embargo={self.embargo}")

        for fold_idx, (train_idx, test_idx) in enumerate(splitter(n, self.n_folds, self.embargo)):
            fold_num = fold_idx + 1

            X_train = X.iloc[list(train_idx)].fillna(0)
            y_train = y.iloc[list(train_idx)]
            X_test = X.iloc[list(test_idx)].fillna(0)
            y_test = y.iloc[list(test_idx)]

            if len(X_train) < 100 or len(X_test) < 50:
                log.warning(f"Fold {fold_num}: insufficient data, skipping")
                continue

            # Check if we have both classes
            if y_train.nunique() < 2 or y_test.nunique() < 2:
                log.warning(f"Fold {fold_num}: only one class present, skipping")
                continue

            # Train
            model = model_factory()
            fp = fit_params or {}

            try:
                model.fit(X_train, y_train, **fp)
            except Exception as e:
                log.warning(f"Fold {fold_num}: training failed: {e}")
                continue

            # Predict
            try:
                train_pred = model.predict(X_train)
                train_prob = model.predict_proba(X_train)[:, 1]
                test_pred = model.predict(X_test)
                test_prob = model.predict_proba(X_test)[:, 1]
            except Exception as e:
                log.warning(f"Fold {fold_num}: prediction failed: {e}")
                continue

            # Metrics
            train_metrics = self._compute_metrics(y_train, train_pred, train_prob)
            test_metrics = self._compute_metrics(y_test, test_pred, test_prob)

            result = FoldResult(
                fold=fold_num,
                train_start=train_idx.start,
                train_end=train_idx.stop,
                test_start=test_idx.start,
                test_end=test_idx.stop,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                n_train=len(X_train),
                n_test=len(X_test),
            )
            self.results.append(result)

            log.info(f"Fold {fold_num}: train_AUC={train_metrics.get('auc', 0):.3f}, "
                     f"test_AUC={test_metrics.get('auc', 0):.3f}, "
                     f"n_train={len(X_train)}, n_test={len(X_test)}")

        # Aggregate
        return self._aggregate()

    def _compute_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
    ) -> Dict[str, float]:
        """Compute classification metrics."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "n": len(y_true),
            "pos_rate": y_true.mean(),
        }

        if len(np.unique(y_true)) > 1:
            metrics["auc"] = roc_auc_score(y_true, y_prob)

        return metrics

    def _aggregate(self) -> Dict[str, Any]:
        """Aggregate fold results."""
        if not self.results:
            return {"status": "failed", "folds": 0}

        test_aucs = [r.test_metrics.get("auc", 0) for r in self.results]
        test_precisions = [r.test_metrics.get("precision", 0) for r in self.results]
        test_recalls = [r.test_metrics.get("recall", 0) for r in self.results]

        summary = {
            "status": "ok",
            "folds": len(self.results),
            "avg_test_auc": np.mean(test_aucs),
            "std_test_auc": np.std(test_aucs),
            "min_test_auc": np.min(test_aucs),
            "avg_test_precision": np.mean(test_precisions),
            "avg_test_recall": np.mean(test_recalls),
            "fold_results": [
                {
                    "fold": r.fold,
                    "test_auc": r.test_metrics.get("auc", 0),
                    "test_precision": r.test_metrics.get("precision", 0),
                    "test_recall": r.test_metrics.get("recall", 0),
                    "n_train": r.n_train,
                    "n_test": r.n_test,
                }
                for r in self.results
            ],
        }

        # Stability: fraction of folds with positive AUC
        positive_folds = sum(1 for a in test_aucs if a > 0.5)
        summary["stability"] = positive_folds / len(test_aucs) if test_aucs else 0
        summary["degradation"] = summary["avg_test_auc"] - np.mean(
            [r.train_metrics.get("auc", 0) for r in self.results]
        )

        log.info(f"Walk-forward summary: {summary['folds']} folds, "
                 f"avg_AUC={summary['avg_test_auc']:.3f}, "
                 f"stability={summary['stability']:.1%}")

        return summary

    def is_edge_valid(self, min_auc: float = 0.52, min_stability: float = 0.6) -> bool:
        """Check if the discovered edge is valid for trading."""
        if not self.results:
            return False

        summary = self._aggregate()

        checks = [
            (summary["avg_test_auc"] >= min_auc,
             f"AUC {summary['avg_test_auc']:.3f} < {min_auc}"),
            (summary["stability"] >= min_stability,
             f"Stability {summary['stability']:.1%} < {min_stability:.0%}"),
            (summary.get("degradation", -1) > -0.15,
             f"Degradation {summary.get('degradation', 0):.3f} too high"),
        ]

        passed = all(c[0] for c in checks)
        if not passed:
            for ok, msg in checks:
                if not ok:
                    log.warning(f"Edge invalid: {msg}")

        return passed
