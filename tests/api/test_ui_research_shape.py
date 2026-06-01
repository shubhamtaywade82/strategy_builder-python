from strategy_research import ui_research

REQUIRED_SIDE_KEYS = {"strategies", "top_features", "walk_forward", "label_stats"}


def test_run_research_shape():
    # allow_synthetic=True keeps this offline + deterministic (no Binance call).
    out = ui_research.run_research("SOLUSDT", days=20, rrs=["1:1"], allow_synthetic=True)
    assert out["symbol"] == "SOLUSDT"
    assert out["data_source"] in ("binance", "synthetic")

    rr = out["results"]["1:1"]
    for side in ("long", "short"):
        assert side in rr
        if "error" in rr[side]:
            continue
        assert REQUIRED_SIDE_KEYS.issubset(rr[side].keys())
        ls = rr[side]["label_stats"]
        assert {"total", "long_wins", "short_wins", "no_trade"}.issubset(ls.keys())
        # Every strategy must now carry out-of-sample robustness fields.
        for strat in rr[side]["strategies"]:
            assert {"robustness_score", "is_robust", "warnings", "in_sample_metrics"}.issubset(strat.keys())
            assert strat["is_viable"] == strat["is_robust"]


def test_player_payload_present():
    out = ui_research.run_research("SOLUSDT", days=30, rrs=["2:1"], allow_synthetic=True)
    player = out["player"]
    assert "long" in player and "short" in player
    for side in ("long", "short"):
        s = player[side]
        # Either a real backtest (metrics + folds) or an honest error — never fabricated.
        if "error" in s:
            assert "metrics" not in s
        else:
            assert {"metrics", "folds", "is_robust", "baseline"}.issubset(s.keys())
            assert "p_value" in s["metrics"]


def test_run_research_deterministic():
    a = ui_research.run_research("SOLUSDT", days=20, rrs=["2:1"], allow_synthetic=True)
    b = ui_research.run_research("SOLUSDT", days=20, rrs=["2:1"], allow_synthetic=True)

    def sig(out):
        rr = out["results"]["2:1"]["short"]
        if "error" in rr:
            return ("err",)
        return tuple((s["name"], round(s["metrics"].get("win_rate", 0), 8),
                      round(s["metrics"].get("expectancy", 0), 10), s["robustness_score"])
                     for s in rr["strategies"])

    assert sig(a) == sig(b)
