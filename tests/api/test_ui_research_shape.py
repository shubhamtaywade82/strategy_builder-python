import pytest
from strategy_research import ui_research

REQUIRED_SIDE_KEYS = {"strategies", "top_features", "walk_forward", "label_stats"}

@pytest.mark.slow
def test_run_research_shape():
    out = ui_research.run_research("SOLUSDT", days=30, rrs=["1:1"])
    assert out["symbol"] == "SOLUSDT"
    rr = out["results"]["1:1"]
    for side in ("long", "short"):
        assert side in rr
        if "error" in rr[side]:
            continue
        assert REQUIRED_SIDE_KEYS.issubset(rr[side].keys())
        ls = rr[side]["label_stats"]
        assert {"total", "long_wins", "short_wins", "no_trade"}.issubset(ls.keys())
