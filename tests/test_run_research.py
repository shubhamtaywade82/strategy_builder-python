import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research"))

import json

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
    json.dumps(out)        # must be JSON-serializable
