import time
import logging
from typing import List, Dict, Any, Optional

class InstrumentLoader:
    def __init__(self, client=None, logger: Optional[logging.Logger] = None):
        self.client = client
        self.logger = logger or logging.getLogger(__name__)
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes
        self._cache_at = {}

    def active_instruments(self, margin_currency: str = "USDT") -> List[Dict[str, Any]]:
        cache_key = f"active_{margin_currency}"
        if self._is_fresh(cache_key):
            return self._cache[cache_key]

        if not self.client:
            return []

        try:
            raw = self.client.futures.market_data.list_active_instruments(
                margin_currency_short_names=[margin_currency]
            )
            instruments = [self._normalize_instrument(i) for i in raw]
            self._cache_it(cache_key, instruments)
            return instruments
        except Exception as e:
            self.logger.error(f"Error fetching active instruments: {e}")
            return []

    def instrument_details(self, pair: str, margin_currency: str = "USDT") -> Dict[str, Any]:
        cache_key = f"detail_{pair}_{margin_currency}"
        if self._is_fresh(cache_key):
            return self._cache[cache_key]

        if not self.client:
            return {}

        try:
            raw = self.client.futures.market_data.fetch_instrument(
                pair=pair,
                margin_currency_short_name=margin_currency
            )
            detail = self._normalize_instrument(raw)
            self._cache_it(cache_key, detail)
            return detail
        except Exception as e:
            self.logger.error(f"Error fetching instrument details for {pair}: {e}")
            return {}

    def stats(self, pair: str) -> Dict[str, Any]:
        cache_key = f"stats_{pair}"
        if self._is_fresh(cache_key):
            return self._cache[cache_key]

        if not self.client:
            return {}

        try:
            raw = self.client.futures.market_data.stats(pair=pair)
            self._cache_it(cache_key, raw)
            return raw
        except Exception as e:
            self.logger.error(f"Error fetching stats for {pair}: {e}")
            return {}

    def tradeable_pairs(self, margin_currency: str = "USDT", min_volume_usdt: float = 100000.0) -> List[str]:
        instruments = self.active_instruments(margin_currency=margin_currency)
        return [
            i["pair"] for i in instruments
            if i["status"] in ["active", "operational"]
            and float(i.get("volume_24h") or 0) >= min_volume_usdt
        ]

    def _normalize_instrument(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "pair": raw.get("symbol") or raw.get("pair"),
            "base_currency": raw.get("base_currency"),
            "quote_currency": raw.get("quote_currency"),
            "margin_currency": raw.get("margin_currency_short_name"),
            "tick_size": float(raw.get("tick_size") or 0),
            "lot_size": float(raw.get("lot_size") or 0),
            "min_quantity": float(raw.get("min_quantity") or 0),
            "max_leverage": int(raw.get("max_leverage") or 0),
            "maker_fee": float(raw.get("maker_commission_rate") or 0),
            "taker_fee": float(raw.get("taker_commission_rate") or 0),
            "status": raw.get("state") or raw.get("status"),
            "volume_24h": raw.get("turnover_usd") or raw.get("volume_24h"),
            "contract_type": raw.get("contract_type")
        }

    def _is_fresh(self, key: str) -> bool:
        return key in self._cache and key in self._cache_at and (time.time() - self._cache_at[key]) < self._cache_ttl

    def _cache_it(self, key: str, value: Any):
        self._cache[key] = value
        self._cache_at[key] = time.time()
