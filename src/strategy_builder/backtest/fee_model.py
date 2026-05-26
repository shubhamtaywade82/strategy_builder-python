from typing import Optional
from ..configuration import Configuration

class FeeModel:
    def __init__(self, fee_rate: Optional[float] = None):
        cfg = Configuration()
        self.fee_rate = fee_rate if fee_rate is not None else cfg.backtest_fee_rate

    def calculate(self, price: float, size: float) -> float:
        return abs(price * size * self.fee_rate)
