"""
Technical Indicators Calculator
Calculates RSI, MACD, ATR, and other technical indicators for swing trading analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging


def calculate_rsi(price_data: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI)
    
    Args:
        price_data: Series of closing prices
        period: RSI period (default: 14)
        
    Returns:
        Series of RSI values
    """
    if len(price_data) < period + 1:
        return pd.Series([np.nan] * len(price_data))
    
    delta = price_data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(price_data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence)
    
    Args:
        price_data: Series of closing prices
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line period (default: 9)
        
    Returns:
        Dictionary with 'macd', 'signal', and 'histogram' series
    """
    if len(price_data) < slow + signal:
        return {
            'macd': pd.Series([np.nan] * len(price_data)),
            'signal': pd.Series([np.nan] * len(price_data)),
            'histogram': pd.Series([np.nan] * len(price_data))
        }
    
    ema_fast = price_data.ewm(span=fast, adjust=False).mean()
    ema_slow = price_data.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR)
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of closing prices
        period: ATR period (default: 14)
        
    Returns:
        Series of ATR values
    """
    if len(high) < period + 1:
        return pd.Series([np.nan] * len(high))
    
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate ATR as moving average of TR
    atr = tr.rolling(window=period).mean()
    
    return atr


def calculate_sma(price_data: pd.Series, period: int) -> pd.Series:
    """
    Calculate Simple Moving Average (SMA)
    
    Args:
        price_data: Series of prices
        period: SMA period
        
    Returns:
        Series of SMA values
    """
    if len(price_data) < period:
        return pd.Series([np.nan] * len(price_data))
    
    return price_data.rolling(window=period).mean()


def calculate_ema(price_data: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA)
    
    Args:
        price_data: Series of prices
        period: EMA period
        
    Returns:
        Series of EMA values
    """
    if len(price_data) < period:
        return pd.Series([np.nan] * len(price_data))
    
    return price_data.ewm(span=period, adjust=False).mean()


def detect_support_resistance(price_data: pd.DataFrame, window: int = 5, lookback: int = 20) -> Dict[str, list]:
    """
    Detect support and resistance levels from price data
    
    Args:
        price_data: DataFrame with 'high', 'low', 'close' columns
        window: Window for finding local minima/maxima
        lookback: Lookback period for analysis
        
    Returns:
        Dictionary with 'support_levels' and 'resistance_levels' lists
    """
    if len(price_data) < lookback:
        return {'support_levels': [], 'resistance_levels': []}
    
    recent_data = price_data.tail(lookback)
    
    # Find local minima (potential support)
    support_levels = []
    for i in range(window, len(recent_data) - window):
        low_val = recent_data['low'].iloc[i]
        if low_val == recent_data['low'].iloc[i-window:i+window+1].min():
            support_levels.append({
                'price': float(low_val),
                'strength': 'medium',
                'type': 'local_minima'
            })
    
    # Find local maxima (potential resistance)
    resistance_levels = []
    for i in range(window, len(recent_data) - window):
        high_val = recent_data['high'].iloc[i]
        if high_val == recent_data['high'].iloc[i-window:i+window+1].max():
            resistance_levels.append({
                'price': float(high_val),
                'strength': 'medium',
                'type': 'local_maxima'
            })
    
    # Sort and remove duplicates
    support_levels = sorted(support_levels, key=lambda x: x['price'], reverse=True)
    resistance_levels = sorted(resistance_levels, key=lambda x: x['price'], reverse=False)
    
    return {
        'support_levels': support_levels[:5],  # Top 5 support levels
        'resistance_levels': resistance_levels[:5]  # Top 5 resistance levels
    }


def calculate_volume_trend(volume: pd.Series, period: int = 10) -> str:
    """
    Determine volume trend (increasing, decreasing, stable)
    
    Args:
        volume: Series of volume values
        period: Period for trend analysis
        
    Returns:
        'increasing', 'decreasing', or 'stable'
    """
    if len(volume) < period:
        return 'stable'
    
    recent_volumes = volume.tail(period)
    volume_ma = recent_volumes.mean()
    
    # Compare recent volumes to average
    recent_avg = recent_volumes.tail(5).mean()
    older_avg = recent_volumes.head(5).mean() if len(recent_volumes) >= 10 else volume_ma
    
    if recent_avg > older_avg * 1.1:
        return 'increasing'
    elif recent_avg < older_avg * 0.9:
        return 'decreasing'
    else:
        return 'stable'


def calculate_fibonacci_levels(high: float, low: float) -> Dict[str, float]:
    """
    Calculate Fibonacci retracement levels
    
    Args:
        high: Highest price
        low: Lowest price
        
    Returns:
        Dictionary with Fibonacci levels (0%, 23.6%, 38.2%, 50%, 61.8%, 100%)
    """
    diff = high - low
    levels = {
        '0%': low,
        '23.6%': low + diff * 0.236,
        '38.2%': low + diff * 0.382,
        '50%': low + diff * 0.5,
        '61.8%': low + diff * 0.618,
        '100%': high
    }
    return levels


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate On Balance Volume (OBV)
    
    OBV is a cumulative indicator that adds volume on up days and subtracts volume on down days.
    
    Args:
        close: Series of closing prices
        volume: Series of volume values
        
    Returns:
        Series of OBV values
    """
    if len(close) != len(volume):
        raise ValueError("close and volume series must have the same length")
    
    if len(close) < 2:
        return pd.Series([0.0] * len(close))
    
    # Initialize OBV with first volume value
    obv = pd.Series(index=close.index, dtype=float)
    obv.iloc[0] = volume.iloc[0]
    
    # Calculate OBV: add volume on up days, subtract on down days, keep same on equal days
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            # Up day: add volume
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            # Down day: subtract volume
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            # Equal day: keep same OBV
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv


def calculate_ad_indicator(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate Accumulation/Distribution Line (A/D Line)
    
    The A/D Line is a cumulative indicator that uses volume and price to assess whether a stock
    is being accumulated or distributed.
    
    Formula: A/D = Previous A/D + Money Flow Volume
    Money Flow Volume = ((Close - Low) - (High - Close)) / (High - Low) * Volume
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of closing prices
        volume: Series of volume values
        
    Returns:
        Series of A/D Line values
    """
    if len(high) != len(low) != len(close) != len(volume):
        raise ValueError("All series must have the same length")
    
    if len(close) < 1:
        return pd.Series([0.0] * len(close))
    
    # Initialize A/D Line
    ad_line = pd.Series(index=close.index, dtype=float)
    ad_line.iloc[0] = 0.0
    
    # Calculate Money Flow Multiplier and Money Flow Volume
    for i in range(1, len(close)):
        high_low_diff = high.iloc[i] - low.iloc[i]
        
        if high_low_diff == 0:
            # If high == low, use previous A/D value
            ad_line.iloc[i] = ad_line.iloc[i-1]
        else:
            # Calculate Money Flow Multiplier
            money_flow_multiplier = ((close.iloc[i] - low.iloc[i]) - (high.iloc[i] - close.iloc[i])) / high_low_diff
            
            # Calculate Money Flow Volume
            money_flow_volume = money_flow_multiplier * volume.iloc[i]
            
            # Accumulate A/D Line
            ad_line.iloc[i] = ad_line.iloc[i-1] + money_flow_volume
    
    return ad_line


def calculate_momentum(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Price Momentum
    
    Momentum measures the rate of change in price over a specified period.
    
    Args:
        close: Series of closing prices
        period: Period for momentum calculation (default: 14)
        
    Returns:
        Series of momentum values (price change over period)
    """
    if len(close) < period + 1:
        return pd.Series([np.nan] * len(close))
    
    # Momentum = Current Price - Price N periods ago
    momentum = close - close.shift(period)
    
    return momentum


def detect_declining_momentum(close: pd.Series, period: int = 14) -> float:
    """
    Detect declining momentum and return a score (0-100)
    
    Lower score indicates declining momentum, higher score indicates rising momentum.
    0 = strong decline, 100 = strong rise
    
    Args:
        close: Series of closing prices
        period: Period for momentum analysis (default: 14)
        
    Returns:
        Momentum score (0-100), where lower = declining momentum
    """
    if len(close) < period * 2:
        return 50.0  # Neutral if insufficient data
    
    # Calculate momentum over different periods
    recent_momentum = calculate_momentum(close, period=period)
    longer_momentum = calculate_momentum(close, period=period * 2)
    
    # Get latest values
    if len(recent_momentum) > 0 and not pd.isna(recent_momentum.iloc[-1]):
        recent_mom = recent_momentum.iloc[-1]
    else:
        return 50.0
    
    if len(longer_momentum) > 0 and not pd.isna(longer_momentum.iloc[-1]):
        longer_mom = longer_momentum.iloc[-1]
    else:
        return 50.0
    
    # Calculate percentage change
    if longer_mom != 0:
        momentum_ratio = recent_mom / longer_mom
    else:
        momentum_ratio = 1.0
    
    # Calculate rate of change (ROC) for recent period (FIXED: bounds checking)
    if len(close) > period + 1:
        # Ensure we have enough data points
        roc = ((close.iloc[-1] - close.iloc[-period-1]) / close.iloc[-period-1]) * 100
    elif len(close) > 1:
        # Fallback: use first available point
        roc = ((close.iloc[-1] - close.iloc[0]) / close.iloc[0]) * 100
    else:
        roc = 0.0
    
    # Convert to score (0-100)
    # Negative ROC = declining momentum (lower score)
    # Positive ROC = rising momentum (higher score)
    # Normalize ROC to 0-100 scale (assuming ROC ranges from -50% to +50%)
    score = 50.0 + (roc * 2.0)  # Scale ROC to score
    score = max(0.0, min(100.0, score))  # Clamp to 0-100
    
    # Adjust based on momentum ratio
    if momentum_ratio < 0.5:
        # Recent momentum is much weaker than longer-term momentum (declining)
        score = score * 0.7
    elif momentum_ratio > 1.5:
        # Recent momentum is much stronger (rising)
        score = score * 1.2
    
    score = max(0.0, min(100.0, score))  # Clamp again
    
    return float(score)
