"""
Walk-Forward Validation with Purged CV

Determinism: every model is trained single-threaded (``n_jobs=1``) with a fixed
``random_state``, and every random draw goes through a seeded ``np.random``
Generator. The same inputs therefore produce byte-identical AUCs and p-values.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, precision_score, recall_score

# Single global seed so walk-forward / shuffle / OOS prediction are reproducible.
SEED = 42

# Deterministic XGBoost config shared across validation models. ``hist`` +
# single thread removes the float-accumulation nondeterminism multi-threaded
# boosting otherwise introduces.
_XGB_KW = dict(
    n_estimators=100, max_depth=4, learning_rate=0.1,
    tree_method="hist", n_jobs=1, random_state=SEED, verbosity=0,
)


def anchored_splits(n: int, n_folds: int = 5, embargo: int = 120):
    fold_size = n // (n_folds + 1)
    for k in range(1, n_folds + 1):
        tr_end = fold_size * k
        te_start = tr_end + embargo
        te_end = min(te_start + fold_size, n)
        if te_start >= n:
            break
        yield range(0, tr_end), range(te_start, te_end)


def walk_forward(X: pd.DataFrame, y: pd.Series, n_folds: int = 5, embargo: int = 120) -> dict:
    n = len(X)
    fold_results = []

    for fold_idx, (tr_idx, te_idx) in enumerate(anchored_splits(n, n_folds, embargo)):
        X_tr = X.iloc[list(tr_idx)].fillna(0)
        y_tr = y.iloc[list(tr_idx)]
        X_te = X.iloc[list(te_idx)].fillna(0)
        y_te = y.iloc[list(te_idx)]

        if len(X_tr) < 100 or len(X_te) < 50 or y_tr.nunique() < 2 or y_te.nunique() < 2:
            continue

        model = xgb.XGBClassifier(
            scale_pos_weight=(y_tr == 0).sum() / max((y_tr == 1).sum(), 1),
            **_XGB_KW,
        )
        model.fit(X_tr, y_tr)

        te_prob = model.predict_proba(X_te)[:, 1]
        te_pred = model.predict(X_te)

        fold_results.append({
            "fold": fold_idx + 1,
            "train_auc": roc_auc_score(y_tr, model.predict_proba(X_tr)[:, 1]) if y_tr.nunique() > 1 else 0.5,
            "test_auc": roc_auc_score(y_te, te_prob) if y_te.nunique() > 1 else 0.5,
            "test_precision": precision_score(y_te, te_pred, zero_division=0),
            "test_recall": recall_score(y_te, te_pred, zero_division=0),
            "n_train": len(X_tr),
            "n_test": len(X_te),
        })

    if not fold_results:
        return {"folds": 0, "is_valid": False}

    test_aucs = [f["test_auc"] for f in fold_results]
    return {
        "folds": len(fold_results),
        "avg_test_auc": np.mean(test_aucs),
        "min_test_auc": np.min(test_aucs),
        "stability": sum(1 for a in test_aucs if a > 0.5) / len(test_aucs),
        "degradation": np.mean(test_aucs) - np.mean([f["train_auc"] for f in fold_results]),
        "is_valid": np.mean(test_aucs) > 0.52 and sum(1 for a in test_aucs if a > 0.5) / len(test_aucs) >= 0.6,
        "fold_results": fold_results,
    }


def oos_predict(X: pd.DataFrame, y: pd.Series, n_folds: int = 5, embargo: int = 120) -> pd.Series:
    """Out-of-sample probabilities via purged anchored walk-forward.

    For each anchored split the model is trained on the past and used to predict
    ONLY the embargoed future block. The concatenation is a leakage-free
    prediction for every bar that falls in some test window; bars never used as
    test (the initial training block and embargo gaps) stay ``NaN``.

    This is what the grid search backtests instead of in-sample predictions —
    the single change that turns the 96% illusion into an honest number.
    """
    n = len(X)
    out = pd.Series(np.nan, index=X.index, dtype=float)
    Xf = X.fillna(0)

    for tr_idx, te_idx in anchored_splits(n, n_folds, embargo):
        tr = list(tr_idx)
        te = list(te_idx)
        y_tr = y.iloc[tr]
        if len(tr) < 100 or len(te) < 1 or y_tr.nunique() < 2:
            continue
        model = xgb.XGBClassifier(
            scale_pos_weight=(y_tr == 0).sum() / max((y_tr == 1).sum(), 1),
            **_XGB_KW,
        )
        model.fit(Xf.iloc[tr], y_tr)
        out.iloc[te] = model.predict_proba(Xf.iloc[te])[:, 1]

    return out


def shuffle_test(X: pd.DataFrame, y: pd.Series, n_shuffles: int = 10) -> dict:
    """Compare real AUC vs shuffled AUC to detect leakage (seeded -> deterministic)."""
    from sklearn.model_selection import train_test_split
    rng = np.random.default_rng(SEED)
    X_tr, X_te, y_tr, y_te = train_test_split(X.fillna(0), y, test_size=0.3, random_state=SEED, stratify=y)

    model = xgb.XGBClassifier(**_XGB_KW)
    model.fit(X_tr, y_tr)
    real_auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]) if y_te.nunique() > 1 else 0.5

    shuffled_aucs = []
    for i in range(n_shuffles):
        y_shuf = pd.Series(rng.permutation(y.values), index=y.index)
        _, _, y_tr_s, y_te_s = train_test_split(X, y_shuf, test_size=0.3, random_state=SEED + i, stratify=y_shuf)
        if y_te_s.nunique() < 2:
            continue
        m = xgb.XGBClassifier(**dict(_XGB_KW, random_state=SEED + i))
        m.fit(X_tr, y_tr_s)
        shuffled_aucs.append(roc_auc_score(y_te_s, m.predict_proba(X_te)[:, 1]))

    p_value = sum(1 for a in shuffled_aucs if a >= real_auc) / len(shuffled_aucs) if shuffled_aucs else 1.0

    return {
        "real_auc": real_auc,
        "shuffled_auc_mean": np.mean(shuffled_aucs) if shuffled_aucs else 0.5,
        "shuffled_auc_std": np.std(shuffled_aucs) if shuffled_aucs else 0,
        "p_value": p_value,
        "is_significant": p_value < 0.05,
    }
