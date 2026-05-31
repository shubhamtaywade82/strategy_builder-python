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


# --------------------------------------------------------------------------- #
# Task 7: funding profile
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Task 8: mmr + load_meta
# --------------------------------------------------------------------------- #
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
