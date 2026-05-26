import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator
from ..configuration import Configuration
from ..backtest.condition_registry import ConditionRegistry

class EntryConfig(BaseModel):
    conditions: List[str]
    direction: str

    @field_validator('conditions')
    @classmethod
    def validate_conditions(cls, v):
        ConditionRegistry.validate_condition_names(v)
        return v

class ExitConfig(BaseModel):
    targets: List[float]
    partial_exits: Optional[List[float]] = None
    trail: Optional[str] = None
    time_stop_candles: Optional[int] = None

    @field_validator('targets')
    @classmethod
    def validate_targets(cls, v):
        for t in v:
            if not (0.1 <= t <= 20.0):
                raise ValueError(f"Target {t} out of range (0.1-20.0)")
        return v

class RiskConfig(BaseModel):
    stop: str
    position_sizing: str
    max_risk_percent: Optional[float] = None

    @field_validator('max_risk_percent')
    @classmethod
    def validate_risk(cls, v):
        if v and v > 3.0:
            raise ValueError(f"Max risk percent {v} exceeds 3.0% limit")
        return v

class FilterConfig(BaseModel):
    min_volume_zscore: Optional[float] = None
    min_atr_percent: Optional[float] = None
    max_atr_percent: Optional[float] = None
    required_regime: Optional[List[str]] = None
    required_structure: Optional[List[str]] = None

class StrategyCandidate(BaseModel):
    name: str
    family: str
    timeframes: List[str]
    entry: EntryConfig
    exit: ExitConfig
    risk: RiskConfig
    filters: Optional[FilterConfig] = None
    session: Optional[List[str]] = None
    invalidation: Optional[List[str]] = None
    rationale: Optional[str] = None

    @field_validator('timeframes')
    @classmethod
    def validate_tfs(cls, v):
        valid = Configuration.VALID_TIMEFRAMES
        for tf in v:
            if tf not in valid:
                raise ValueError(f"Invalid timeframe: {tf}")
        return v

class CandidateValidator:
    PARTIAL_EXIT_SUM_TOLERANCE = 0.05

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def validate(self, candidate_dict: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self._repair_partial_exits(candidate_dict)
            candidate = StrategyCandidate(**candidate_dict)
            
            # Additional logic checks not easily done in Pydantic
            errors = self._check_partial_exits(candidate.exit)
            if errors:
                return {"valid": False, "errors": errors}

            return {"valid": True, "errors": [], "candidate": candidate.model_dump()}
        except Exception as e:
            return {"valid": False, "errors": [str(e)]}

    def _repair_partial_exits(self, candidate: Dict[str, Any]):
        exit_cfg = candidate.get("exit")
        if not isinstance(exit_cfg, dict):
            return

        targets = exit_cfg.get("targets")
        parts = exit_cfg.get("partial_exits")
        if not isinstance(targets, list) or not targets or not isinstance(parts, list) or not parts:
            return

        try:
            nums = [float(p) for p in parts]
            if any(n < 0 for n in nums): return
            
            n = len(targets)
            if len(nums) != n or abs(sum(nums) - 1.0) > self.PARTIAL_EXIT_SUM_TOLERANCE:
                repaired = [1.0 / n] * n
                # Snap to 1.0
                repaired[-1] = round(1.0 - sum(repaired[:-1]), 8)
                exit_cfg["partial_exits"] = repaired
            else:
                # Just normalize to exact 1.0
                total = sum(nums)
                normalized = [round(x / total, 8) for x in nums]
                normalized[-1] = round(1.0 - sum(normalized[:-1]), 8)
                exit_cfg["partial_exits"] = normalized
        except:
            pass

    def _check_partial_exits(self, exit_cfg: ExitConfig) -> List[str]:
        if exit_cfg.partial_exits is None:
            return []

        parts = exit_cfg.partial_exits
        total = sum(parts)
        if abs(total - 1.0) > self.PARTIAL_EXIT_SUM_TOLERANCE:
            return [f"Partial exits sum to {total}, must be ~1.0"]

        if len(parts) != len(exit_cfg.targets):
            return [f"Partial exits count ({len(parts)}) != targets count ({len(exit_cfg.targets)})"]

        return []
