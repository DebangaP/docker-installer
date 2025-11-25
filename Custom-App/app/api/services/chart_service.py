"""
Chart Service
Business logic for candlestick chart data processing and technical analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging
from common.TechnicalIndicators import detect_rsi_divergence

logger = logging.getLogger(__name__)


class ChartService:
    """Service for chart data processing and technical analysis"""
    
    def __init__(self):
        """Initialize Chart Service"""
        pass
    
    def detect_rsi_divergence_for_chart(self, highs: np.ndarray, lows: np.ndarray, 
                                        closes: np.ndarray, rsi: np.ndarray, 
                                        dates: list, 
                                        lookback_periods: int = 20,
                                        min_divergence_strength: float = 0.05) -> List[Dict]:
        """
        Detect RSI divergence for candlestick chart data
        
        Args:
            highs: Array of high prices
            lows: Array of low prices
            closes: Array of closing prices
            rsi: Array of RSI values
            dates: List of date strings
            lookback_periods: Number of periods to analyze
            min_divergence_strength: Minimum divergence strength threshold
            
        Returns:
            List of divergence dictionaries
        """
        return detect_rsi_divergence(
            highs=highs,
            lows=lows,
            closes=closes,
            rsi=rsi,
            dates=dates,
            lookback_periods=lookback_periods,
            min_divergence_strength=min_divergence_strength
        )

