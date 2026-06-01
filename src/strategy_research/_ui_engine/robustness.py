"""
Out-of-Sample Robustness Scoring
================================
Turns the walk-forward + shuffle diagnostics into a single gate and 0-100 score
so the grid search ranks strategies by *generalization*, not in-sample fit.

This is the antidote to the "96% win rate" illusion: a strategy is only
``is_robust`` when its out-of-sample (walk-forward) backtest is profitable, its
folds are consistent, and a label-shuffle test says the signal is unlikely to be
noise. In-sample expectancy is deliberately NOT part of the gate.
"""
from __future__ import annotations

from typing import Dict, List


def score_robustness(
    wf: Dict,
    shuffle: Dict,
    oos_metrics: Dict,
    min_trades: int = 30,
) -> Dict:
    """Blend OOS diagnostics into ``{robustness_score, is_robust, warnings}``.

    Args:
        wf: output of ``validation.walk_forward`` (avg_test_auc, stability, ...).
        shuffle: output of ``validation.shuffle_test`` (p_value, is_significant).
        oos_metrics: backtest metrics computed on OUT-OF-SAMPLE signals.
        min_trades: floor on OOS trade count for a strategy to be trusted.
    """
    avg_auc = float(wf.get("avg_test_auc", 0.5) or 0.5)
    stability = float(wf.get("stability", 0.0) or 0.0)
    degradation = float(wf.get("degradation", 0.0) or 0.0)  # test_auc - train_auc (<=0)
    p_value = float(shuffle.get("p_value", 1.0) if shuffle else 1.0)
    exp = float(oos_metrics.get("expectancy", 0.0) or 0.0)
    n = int(oos_metrics.get("trade_count", 0) or 0)

    # --- 0-100 score (weights mirror the design doc) ---
    auc_pts = _clip01((avg_auc - 0.5) / 0.15) * 40        # 0.65 AUC -> full
    stab_pts = _clip01(stability) * 25                     # fraction of folds AUC>0.5
    sig_pts = (1.0 - min(p_value / 0.05, 1.0)) * 25        # p=0 -> full, p>=0.05 -> 0
    degr_pts = _clip01(1.0 + degradation / 0.20) * 10      # small gap -> full
    score = round(auc_pts + stab_pts + sig_pts + degr_pts, 1)

    wf_valid = bool(wf.get("is_valid", False))
    shuffle_sig = bool(shuffle.get("is_significant", False)) if shuffle else False
    enough_trades = n >= min_trades

    is_robust = bool(exp > 0 and wf_valid and shuffle_sig and enough_trades)

    warnings: List[str] = []
    if not is_robust:
        warnings.append("in-sample only — fails out-of-sample gate")
    if avg_auc < 0.53:
        warnings.append(f"AUC≈random ({avg_auc:.3f}) — likely overfit")
    if p_value >= 0.05:
        warnings.append(f"not significant (p={p_value:.3f})")
    if stability < 0.6:
        warnings.append(f"inconsistent across folds (stability={stability:.2f})")
    if exp <= 0:
        warnings.append("negative out-of-sample expectancy")
    if not enough_trades:
        warnings.append(f"low sample (n={n} < {min_trades})")

    return {"robustness_score": score, "is_robust": is_robust, "warnings": warnings}


def _clip01(x: float) -> float:
    return 0.0 if x < 0 else (1.0 if x > 1 else float(x))
