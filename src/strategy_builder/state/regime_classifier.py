from typing import Dict, Any, Optional
from ..domain import Regime

class RegimeClassifier:
    """
    Institutional Regime Classifier.
    Maps session metrics and price action to specific trading regimes.
    """
    
    @staticmethod
    def classify(features: Dict[str, Any]) -> Regime:
        # Extract features
        rvol = features.get("rvol", 1.0)
        volatility = features.get("volatility", {}).get("current_atr_percent", 0)
        efficiency = features.get("trend_efficiency", 0.5)
        oi_expansion = features.get("oi_delta_pct", 0)
        
        # 1. Dead Market
        if rvol < 0.7 and volatility < 0.1:
            return Regime.DEAD_MARKET
        
        # 2. Liquidation Event (Extreme volatility + high volume + low efficiency)
        if rvol > 2.5 and efficiency < 0.3:
            return Regime.LIQUIDATION_EVENT
        
        # 3. Trend Expansion (Moderate RVOL + Moderate Efficiency)
        if rvol > 1.1 and efficiency > 0.55:
            return Regime.TREND_EXPANSION
        
        # 4. Breakout Environment (RVOL rising or High)
        if rvol > 1.1:
            return Regime.BREAKOUT_ENVIRONMENT
            
        # 5. Mean Reversion (Low efficiency + high volatility)
        if efficiency < 0.4 and volatility > 1.5:
            return Regime.MEAN_REVERSION
            
        # 6. Normal Trend/Range logic
        vol_regime = features.get("volatility", {}).get("regime")
        structure = features.get("structure", {}).get("structure")
        alignment = features.get("mtf_alignment", {}).get("alignment", {})
        
        aligned_bull = alignment.get("aligned_bullish", False)
        aligned_bear = alignment.get("aligned_bearish", False)

        if vol_regime == "compression":
            return Regime.COMPRESSION
        
        if structure == "bullish" and aligned_bull:
            return Regime.TREND_UP
        elif structure == "bearish" and aligned_bear:
            return Regime.TREND_DOWN
        elif structure == "ranging":
            return Regime.RANGE
            
        return Regime.CHOP
