"""
Purged Walk-Forward Validation for strategy_builder-python
============================================================
Adds to: src/strategy_builder/backtest/walk_forward.py

Usage:
    from strategy_builder.backtest.walk_forward import WalkForwardValidator
    from strategy_builder.backtest.triple_barrier import BarrierConfig, TripleBarrierLabeler
    
    validator = WalkForwardValidator(n_folds=5, embargo=120)
    
    # Your feature matrix X and labels y
    result = validator.validate(X, y, model_factory=lambda: XGBClassifier(...))
    
    # result = {
    #   "is_valid": True,
    #   "avg_test_auc": 0.543,
    #   "stability": 0.80,
    #   "folds": [...]
    # }
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Callable, List, Optional, Generator, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score


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
    """Purged + embargoed walk-forward cross-validation.
    
    Based on Lopez de Prado's method:
    - Training on past data, testing on future data
    - Embargo gap prevents label overlap leakage
    - Stability score: fraction of folds with positive edge
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
        """Compare real AUC vs shuffled to detect leakage."""
        from sklearn.model_selection import train_test_split
        X_arr = X.fillna(0) if hasattr(X, "fillna") else X
        y_arr = y
        
        X_tr, X_te, y_tr, y_te = train_test_split(X_arr, y_arr, test_size=0.3, random_state=42, stratify=y_arr)
        
        model = model_factory()
        model.fit(X_tr, y_tr)
        real_auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]) if len(set(y_te)) > 1 else 0.5
        
        shuffled_aucs = []
        import numpy as np
        for i in range(n_shuffles):
            y_shuf = np.random.permutation(y_arr.values if hasattr(y_arr, "values") else y_arr)
            _, _, y_tr_s, y_te_s = train_test_split(X_arr, y_shuf, test_size=0.3, random_state=42+i)
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
