"""
Swing Trade Scanner
Scans stocks for swing trading opportunities targeting 10-20% gains
Uses technical analysis, pattern recognition, and risk management
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import json
import psycopg2
import psycopg2.extras
from TechnicalIndicators import (
    calculate_rsi, calculate_macd, calculate_atr, 
    calculate_sma, detect_support_resistance, 
    calculate_volume_trend
)
from Boilerplate import get_db_connection


class SwingTradeScanner:
    """
    Scanner for swing trading opportunities
    Identifies stocks with patterns that could lead to 10-20% gains
    """
    
    def __init__(self, min_gain: float = 10.0, max_gain: float = 20.0, min_confidence: float = 70.0):
        """
        Initialize Swing Trade Scanner
        
        Args:
            min_gain: Minimum target gain percentage (default: 10.0)
            max_gain: Maximum target gain percentage (default: 20.0)
            min_confidence: Minimum confidence score required (default: 70.0)
        """
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.min_confidence = min_confidence
        self.lookback_days = 90
    
    def get_stocks_list(self) -> List[str]:
        """
        Get list of unique scrip_ids from rt_intraday_price table
        
        Returns:
            List of stock symbols
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get unique scrip_ids from rt_intraday_price where country = 'IN'
            cursor.execute("""
                SELECT DISTINCT scrip_id
                FROM my_schema.rt_intraday_price
                WHERE country = 'IN'
                AND scrip_id NOT IN ('BITCOIN', 'SOLANA', 'DOGE', 'ETH')
                ORDER BY scrip_id
            """)
            
            stocks = [row[0] for row in cursor.fetchall()]
            cursor.close()
            conn.close()
            
            logging.info(f"Found {len(stocks)} stocks to scan")
            return stocks
            
        except Exception as e:
            logging.error(f"Error getting stocks list: {e}")
            return []
    
    def get_price_data(self, scrip_id: str, days: int = 90) -> Optional[pd.DataFrame]:
        """
        Get OHLC price data for a stock from rt_intraday_price
        
        Args:
            scrip_id: Stock symbol
            days: Number of days to fetch
            
        Returns:
            DataFrame with OHLC data or None
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cursor.execute("""
                SELECT 
                    price_date,
                    price_open,
                    price_high,
                    price_low,
                    price_close,
                    volume
                FROM my_schema.rt_intraday_price
                WHERE scrip_id = %s
                AND country = 'IN'
                AND price_date::date >= CURRENT_DATE - make_interval(days => %s)
                ORDER BY price_date ASC
            """, (scrip_id, days))
            
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            
            if not rows or len(rows) < 50:  # Need at least 50 days for reliable analysis
                return None
            
            # Convert to DataFrame
            data = []
            for row in rows:
                price_date = row['price_date']
                if hasattr(price_date, 'strftime'):
                    date_str = price_date.strftime('%Y-%m-%d')
                else:
                    date_str = str(price_date)
                
                data.append({
                    'date': date_str,
                    'open': float(row['price_open']) if row['price_open'] else 0.0,
                    'high': float(row['price_high']) if row['price_high'] else 0.0,
                    'low': float(row['price_low']) if row['price_low'] else 0.0,
                    'close': float(row['price_close']) if row['price_close'] else 0.0,
                    'volume': float(row['volume']) if row['volume'] else 0.0
                })
            
            df = pd.DataFrame(data)
            if df.empty:
                return None
            
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logging.error(f"Error fetching price data for {scrip_id}: {e}")
            return None
    
    def calculate_indicators(self, price_data: pd.DataFrame) -> Dict:
        """
        Calculate all technical indicators
        
        Args:
            price_data: DataFrame with OHLC data
            
        Returns:
            Dictionary with all calculated indicators
        """
        if price_data is None or len(price_data) < 50:
            return {}
        
        close = price_data['close']
        high = price_data['high']
        low = price_data['low']
        volume = price_data['volume']
        
        indicators = {}
        
        try:
            # Moving Averages
            indicators['sma_20'] = calculate_sma(close, 20)
            indicators['sma_50'] = calculate_sma(close, 50)
            indicators['sma_200'] = calculate_sma(close, 200)
            
            # RSI
            indicators['rsi_14'] = calculate_rsi(close, 14)
            
            # MACD
            macd_data = calculate_macd(close, 12, 26, 9)
            indicators['macd'] = macd_data['macd']
            indicators['macd_signal'] = macd_data['signal']
            indicators['macd_histogram'] = macd_data['histogram']
            
            # ATR
            indicators['atr_14'] = calculate_atr(high, low, close, 14)
            
            # Volume Trend
            indicators['volume_trend'] = calculate_volume_trend(volume, 10)
            
            # Support/Resistance
            sr_levels = detect_support_resistance(price_data, window=5, lookback=30)
            indicators['support_levels'] = sr_levels.get('support_levels', [])
            indicators['resistance_levels'] = sr_levels.get('resistance_levels', [])
            
            # Current values (most recent)
            indicators['current_price'] = float(close.iloc[-1])
            indicators['current_sma_20'] = float(indicators['sma_20'].iloc[-1]) if not indicators['sma_20'].isna().iloc[-1] else None
            indicators['current_sma_50'] = float(indicators['sma_50'].iloc[-1]) if not indicators['sma_50'].isna().iloc[-1] else None
            indicators['current_sma_200'] = float(indicators['sma_200'].iloc[-1]) if not indicators['sma_200'].isna().iloc[-1] else None
            indicators['current_rsi'] = float(indicators['rsi_14'].iloc[-1]) if not indicators['rsi_14'].isna().iloc[-1] else None
            indicators['current_macd'] = float(indicators['macd'].iloc[-1]) if not indicators['macd'].isna().iloc[-1] else None
            indicators['current_macd_signal'] = float(indicators['macd_signal'].iloc[-1]) if not indicators['macd_signal'].isna().iloc[-1] else None
            indicators['current_atr'] = float(indicators['atr_14'].iloc[-1]) if not indicators['atr_14'].isna().iloc[-1] else None
            
        except Exception as e:
            logging.error(f"Error calculating indicators: {e}")
            return {}
        
        return indicators
    
    def detect_patterns(self, price_data: pd.DataFrame, indicators: Dict) -> List[Dict]:
        """
        Detect swing trade patterns
        
        Args:
            price_data: DataFrame with OHLC data
            indicators: Dictionary with calculated indicators
            
        Returns:
            List of detected patterns with details
        """
        patterns = []
        
        if not indicators or price_data is None or len(price_data) < 50:
            return patterns
        
        current_price = indicators.get('current_price')
        if current_price is None or current_price <= 0:
            return patterns
        
        # Get latest values
        sma_20 = indicators.get('current_sma_20')
        sma_50 = indicators.get('current_sma_50')
        sma_200 = indicators.get('current_sma_200')
        rsi = indicators.get('current_rsi')
        macd = indicators.get('current_macd')
        macd_signal = indicators.get('current_macd_signal')
        atr = indicators.get('current_atr')
        support_levels = indicators.get('support_levels', [])
        resistance_levels = indicators.get('resistance_levels', [])
        volume_trend = indicators.get('volume_trend', 'stable')
        
        # Pattern 1: Pullback to Support with Bullish Reversal
        if support_levels and rsi is not None and rsi < 40:
            nearest_support = max([s['price'] for s in support_levels if s['price'] < current_price], default=None)
            if nearest_support and abs(current_price - nearest_support) / current_price < 0.05:  # Within 5% of support
                if resistance_levels:
                    next_resistance = min([r['price'] for r in resistance_levels if r['price'] > current_price], default=None)
                    if next_resistance:
                        potential_gain = (next_resistance - current_price) / current_price * 100
                        if self.min_gain <= potential_gain <= self.max_gain:
                            patterns.append({
                                'pattern_type': 'Pullback to Support',
                                'direction': 'BUY',
                                'entry_price': current_price,
                                'target_price': next_resistance,
                                'stop_loss': nearest_support * 0.95,
                                'potential_gain_pct': potential_gain,
                                'rationale': f'Price near support ({nearest_support:.2f}), RSI oversold ({rsi:.2f}), target resistance at {next_resistance:.2f}'
                            })
        
        # Pattern 2: Breakout from Consolidation
        if resistance_levels and volume_trend == 'increasing':
            nearest_resistance = min([r['price'] for r in resistance_levels if r['price'] > current_price], default=None)
            if nearest_resistance:
                # Check if price is breaking above resistance
                recent_highs = price_data['high'].tail(5)
                if recent_highs.max() > nearest_resistance * 0.98:  # Near or breaking resistance
                    # Calculate Fibonacci extension target
                    recent_range = price_data['high'].tail(20).max() - price_data['low'].tail(20).min()
                    target_price = nearest_resistance + recent_range * 0.618  # Fibonacci extension
                    potential_gain = (target_price - current_price) / current_price * 100
                    
                    if self.min_gain <= potential_gain <= self.max_gain and rsi is not None and 40 < rsi < 70:
                        patterns.append({
                            'pattern_type': 'Breakout from Consolidation',
                            'direction': 'BUY',
                            'entry_price': current_price,
                            'target_price': target_price,
                            'stop_loss': nearest_resistance * 0.97,
                            'potential_gain_pct': potential_gain,
                            'rationale': f'Breaking above resistance ({nearest_resistance:.2f}) with volume confirmation, target at {target_price:.2f}'
                        })
        
        # Pattern 3: Golden Cross
        if sma_20 and sma_50 and sma_200:
            # Check if SMA 20 is above SMA 50 and crossing
            if sma_20 > sma_50:
                # Check recent crossing (last 5 days)
                sma_20_series = indicators['sma_20']
                sma_50_series = indicators['sma_50']
                
                if len(sma_20_series) >= 5:
                    recent_cross = any(sma_20_series.iloc[i] > sma_50_series.iloc[i] and sma_20_series.iloc[i-1] <= sma_50_series.iloc[i-1] 
                                      for i in range(len(sma_20_series)-4, len(sma_20_series)))
                    
                    if recent_cross and current_price > sma_20 and rsi is not None and 30 < rsi < 70:
                        # Target based on SMA 200 or next resistance
                        if resistance_levels:
                            target_price = min([r['price'] for r in resistance_levels if r['price'] > current_price], default=current_price * 1.15)
                        else:
                            target_price = current_price * 1.15
                        
                        potential_gain = (target_price - current_price) / current_price * 100
                        if self.min_gain <= potential_gain <= self.max_gain:
                            patterns.append({
                                'pattern_type': 'Golden Cross',
                                'direction': 'BUY',
                                'entry_price': current_price,
                                'target_price': target_price,
                                'stop_loss': sma_50 * 0.95,
                                'potential_gain_pct': potential_gain,
                                'rationale': f'Golden Cross (SMA 20 > SMA 50), bullish trend, target at {target_price:.2f}'
                            })
        
        # Pattern 4: Oversold Bounce
        if rsi is not None and rsi < 30 and support_levels:
            nearest_support = max([s['price'] for s in support_levels if s['price'] < current_price], default=None)
            if nearest_support and abs(current_price - nearest_support) / current_price < 0.03:  # Very close to support
                # Target recovery to middle band or next resistance
                if resistance_levels:
                    target_price = min([r['price'] for r in resistance_levels if r['price'] > current_price], default=current_price * 1.12)
                else:
                    target_price = current_price * 1.12
                
                potential_gain = (target_price - current_price) / current_price * 100
                if self.min_gain <= potential_gain <= self.max_gain:
                    stop_loss = nearest_support * 0.93  # Tight stop below support
                    risk = current_price - stop_loss
                    reward = target_price - current_price
                    risk_reward = reward / risk if risk > 0 else 0
                    
                    if risk_reward >= 2.0:  # Minimum 2:1 risk-reward
                        patterns.append({
                            'pattern_type': 'Oversold Bounce',
                            'direction': 'BUY',
                            'entry_price': current_price,
                            'target_price': target_price,
                            'stop_loss': stop_loss,
                            'potential_gain_pct': potential_gain,
                            'rationale': f'Oversold (RSI {rsi:.2f}) at support ({nearest_support:.2f}), bounce target at {target_price:.2f}'
                        })
        
        # Pattern 5: MACD Bullish Divergence
        if macd and macd_signal:
            # Check for bullish MACD crossover
            macd_series = indicators['macd']
            signal_series = indicators['macd_signal']
            
            if len(macd_series) >= 5:
                recent_cross = any(macd_series.iloc[i] > signal_series.iloc[i] and macd_series.iloc[i-1] <= signal_series.iloc[i-1] 
                                  for i in range(len(macd_series)-4, len(macd_series)))
                
                if recent_cross and rsi is not None and 30 < rsi < 70:
                    if resistance_levels:
                        target_price = min([r['price'] for r in resistance_levels if r['price'] > current_price], default=current_price * 1.15)
                    else:
                        target_price = current_price * 1.15
                    
                    potential_gain = (target_price - current_price) / current_price * 100
                    if self.min_gain <= potential_gain <= self.max_gain:
                        stop_loss = current_price * 0.93
                        if atr:
                            stop_loss = max(stop_loss, current_price - atr * 2)  # Use ATR-based stop
                        
                        patterns.append({
                            'pattern_type': 'MACD Bullish Crossover',
                            'direction': 'BUY',
                            'entry_price': current_price,
                            'target_price': target_price,
                            'stop_loss': stop_loss,
                            'potential_gain_pct': potential_gain,
                            'rationale': f'MACD bullish crossover, momentum increasing, target at {target_price:.2f}'
                        })
        
        # Pattern 6: Head and Shoulders
        patterns.extend(self._detect_head_and_shoulders(price_data, indicators, current_price))
        
        # Pattern 7: Double Top
        patterns.extend(self._detect_double_top(price_data, indicators, current_price))
        
        # Pattern 8: Double Bottom
        patterns.extend(self._detect_double_bottom(price_data, indicators, current_price))
        
        # Pattern 9: Cup and Handle
        patterns.extend(self._detect_cup_and_handle(price_data, indicators, current_price))
        
        # Pattern 10: Flag Pattern
        patterns.extend(self._detect_flag_pattern(price_data, indicators, current_price))
        
        # Pattern 11: Ascending Triangle
        patterns.extend(self._detect_ascending_triangle(price_data, indicators, current_price))
        
        # Pattern 12: Descending Triangle
        patterns.extend(self._detect_descending_triangle(price_data, indicators, current_price))
        
        # Pattern 13: Symmetrical Triangle
        patterns.extend(self._detect_symmetrical_triangle(price_data, indicators, current_price))
        
        # Pattern 14: Rounding Bottom
        patterns.extend(self._detect_rounding_bottom(price_data, indicators, current_price))
        
        # Pattern 15: Engulfing Candlestick
        patterns.extend(self._detect_engulfing_candlestick(price_data, indicators, current_price))
        
        return patterns
    
    def _detect_head_and_shoulders(self, price_data: pd.DataFrame, indicators: Dict, current_price: float) -> List[Dict]:
        """Detect Head and Shoulders pattern (bearish reversal)"""
        patterns = []
        if len(price_data) < 60:
            return patterns
        
        try:
            highs = price_data['high'].values
            closes = price_data['close'].values
            
            # Look for three peaks: left shoulder, head, right shoulder
            # Head should be highest, shoulders roughly equal height
            recent_data = price_data.tail(60)
            peak_indices = []
            
            # Find local maxima
            for i in range(2, len(recent_data) - 2):
                if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and 
                    highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                    peak_indices.append((i, highs[i]))
            
            if len(peak_indices) >= 3:
                # Sort by height
                peak_indices.sort(key=lambda x: x[1], reverse=True)
                
                # Check if we have a head (highest) and two shoulders
                head_idx, head_price = peak_indices[0]
                
                # Find shoulders (two peaks near head but lower)
                shoulders = [(idx, price) for idx, price in peak_indices[1:] 
                            if abs(price - head_price) / head_price < 0.15 and 
                            abs(idx - head_idx) < 40]
                
                if len(shoulders) >= 2:
                    # Check neckline (support level between shoulders)
                    neckline = min(closes[peak_indices[0][0]:])
                    if neckline and current_price < head_price * 0.95:
                        target_price = neckline * 0.92
                        potential_gain = (current_price - target_price) / current_price * 100
                        
                        if self.min_gain <= abs(potential_gain) <= self.max_gain:
                            patterns.append({
                                'pattern_type': 'Head and Shoulders',
                                'direction': 'SELL',
                                'entry_price': current_price,
                                'target_price': target_price,
                                'stop_loss': head_price * 1.02,
                                'potential_gain_pct': abs(potential_gain),
                                'rationale': f'Head and Shoulders pattern detected, bearish reversal expected'
                            })
        except Exception as e:
            logging.debug(f"Error detecting Head and Shoulders: {e}")
        
        return patterns
    
    def _detect_double_top(self, price_data: pd.DataFrame, indicators: Dict, current_price: float) -> List[Dict]:
        """Detect Double Top pattern (bearish reversal)"""
        patterns = []
        if len(price_data) < 40:
            return patterns
        
        try:
            highs = price_data['high'].values
            
            # Find two similar peaks
            recent_data = price_data.tail(40)
            peaks = []
            
            for i in range(2, len(recent_data) - 2):
                if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and 
                    highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                    peaks.append((i, highs[i]))
            
            if len(peaks) >= 2:
                # Sort by index to find two consecutive peaks
                peaks.sort(key=lambda x: x[0])
                
                for i in range(len(peaks) - 1):
                    peak1_idx, peak1_price = peaks[i]
                    peak2_idx, peak2_price = peaks[i+1]
                    
                    # Peaks should be similar in height (within 2%)
                    if abs(peak1_price - peak2_price) / max(peak1_price, peak2_price) < 0.02:
                        # Find trough between peaks (neckline)
                        trough_idx = peak1_idx + np.argmin(highs[peak1_idx:peak2_idx])
                        trough_price = highs[trough_idx]
                        
                        if trough_price < min(peak1_price, peak2_price) * 0.95:
                            target_price = trough_price
                            potential_gain = (current_price - target_price) / current_price * 100
                            
                            if self.min_gain <= abs(potential_gain) <= self.max_gain and current_price < max(peak1_price, peak2_price):
                                patterns.append({
                                    'pattern_type': 'Double Top',
                                    'direction': 'SELL',
                                    'entry_price': current_price,
                                    'target_price': target_price,
                                    'stop_loss': max(peak1_price, peak2_price) * 1.02,
                                    'potential_gain_pct': abs(potential_gain),
                                    'rationale': f'Double Top pattern at resistance {max(peak1_price, peak2_price):.2f}'
                                })
        except Exception as e:
            logging.debug(f"Error detecting Double Top: {e}")
        
        return patterns
    
    def _detect_double_bottom(self, price_data: pd.DataFrame, indicators: Dict, current_price: float) -> List[Dict]:
        """Detect Double Bottom pattern (bullish reversal)"""
        patterns = []
        if len(price_data) < 40:
            return patterns
        
        try:
            lows = price_data['low'].values
            
            # Find two similar troughs
            recent_data = price_data.tail(40)
            troughs = []
            
            for i in range(2, len(recent_data) - 2):
                if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and 
                    lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                    troughs.append((i, lows[i]))
            
            if len(troughs) >= 2:
                troughs.sort(key=lambda x: x[0])
                
                for i in range(len(troughs) - 1):
                    trough1_idx, trough1_price = troughs[i]
                    trough2_idx, trough2_price = troughs[i+1]
                    
                    # Troughs should be similar (within 2%)
                    if abs(trough1_price - trough2_price) / max(trough1_price, trough2_price) < 0.02:
                        # Find peak between troughs
                        peak_idx = trough1_idx + np.argmax(price_data['high'].values[trough1_idx:trough2_idx])
                        peak_price = price_data['high'].values[peak_idx]
                        
                        if peak_price > max(trough1_price, trough2_price) * 1.05:
                            target_price = peak_price * 1.15
                            potential_gain = (target_price - current_price) / current_price * 100
                            
                            if self.min_gain <= potential_gain <= self.max_gain and current_price > min(trough1_price, trough2_price):
                                patterns.append({
                                    'pattern_type': 'Double Bottom',
                                    'direction': 'BUY',
                                    'entry_price': current_price,
                                    'target_price': target_price,
                                    'stop_loss': min(trough1_price, trough2_price) * 0.98,
                                    'potential_gain_pct': potential_gain,
                                    'rationale': f'Double Bottom pattern at support {min(trough1_price, trough2_price):.2f}'
                                })
        except Exception as e:
            logging.debug(f"Error detecting Double Bottom: {e}")
        
        return patterns
    
    def _detect_cup_and_handle(self, price_data: pd.DataFrame, indicators: Dict, current_price: float) -> List[Dict]:
        """Detect Cup and Handle pattern (bullish continuation)"""
        patterns = []
        if len(price_data) < 50:
            return patterns
        
        try:
            closes = price_data['close'].values
            recent_data = price_data.tail(50)
            
            # Cup: U-shaped bottom over 20-40 days
            # Handle: small consolidation after cup
            cup_duration = 30
            handle_duration = 10
            
            if len(recent_data) >= cup_duration + handle_duration:
                cup_data = recent_data.head(cup_duration)
                handle_data = recent_data.tail(handle_duration)
                
                cup_start = cup_data.iloc[0]['close']
                cup_end = cup_data.iloc[-1]['close']
                cup_min = cup_data['low'].min()
                cup_max = cup_data['high'].max()
                
                # Cup should have rounded bottom (U-shape)
                # Check if middle of cup is lower than start/end
                cup_mid_low = cup_data.iloc[cup_duration // 2]['low']
                
                # Cup depth should be reasonable (5-20%)
                cup_depth = (cup_max - cup_min) / cup_max
                
                # Handle should be consolidation near cup rim
                handle_high = handle_data['high'].max()
                handle_low = handle_data['low'].min()
                
                if (cup_depth > 0.05 and cup_depth < 0.20 and
                    cup_mid_low < cup_start * 0.95 and cup_mid_low < cup_end * 0.95 and
                    abs(cup_start - cup_end) / max(cup_start, cup_end) < 0.05 and  # Cup rim roughly equal
                    handle_high < cup_max * 1.02 and handle_low > cup_max * 0.90):
                    
                    target_price = cup_max * 1.12
                    potential_gain = (target_price - current_price) / current_price * 100
                    
                    if self.min_gain <= potential_gain <= self.max_gain:
                        patterns.append({
                            'pattern_type': 'Cup and Handle',
                            'direction': 'BUY',
                            'entry_price': current_price,
                            'target_price': target_price,
                            'stop_loss': handle_low * 0.95,
                            'potential_gain_pct': potential_gain,
                            'rationale': f'Cup and Handle pattern, bullish continuation expected'
                        })
        except Exception as e:
            logging.debug(f"Error detecting Cup and Handle: {e}")
        
        return patterns
    
    def _detect_flag_pattern(self, price_data: pd.DataFrame, indicators: Dict, current_price: float) -> List[Dict]:
        """Detect Flag pattern (continuation)"""
        patterns = []
        if len(price_data) < 30:
            return patterns
        
        try:
            # Flag: strong move (pole) followed by consolidation (flag)
            recent_data = price_data.tail(30)
            pole_data = recent_data.head(15)
            flag_data = recent_data.tail(15)
            
            pole_move = (pole_data['high'].max() - pole_data['low'].min()) / pole_data['low'].min()
            flag_range = (flag_data['high'].max() - flag_data['low'].min()) / flag_data['low'].min()
            
            # Pole should be strong move (>5%), flag should be tight (<3%)
            if pole_move > 0.05 and flag_range < 0.03:
                # Check direction
                pole_direction = 1 if pole_data['close'].iloc[-1] > pole_data['close'].iloc[0] else -1
                
                if pole_direction > 0:  # Bullish flag
                    target_price = pole_data['high'].max() * 1.15
                    potential_gain = (target_price - current_price) / current_price * 100
                    
                    if self.min_gain <= potential_gain <= self.max_gain:
                        patterns.append({
                            'pattern_type': 'Flag (Bullish)',
                            'direction': 'BUY',
                            'entry_price': current_price,
                            'target_price': target_price,
                            'stop_loss': flag_data['low'].min() * 0.97,
                            'potential_gain_pct': potential_gain,
                            'rationale': f'Bullish Flag pattern, continuation expected'
                        })
                else:  # Bearish flag
                    target_price = pole_data['low'].min() * 0.85
                    potential_gain = abs((target_price - current_price) / current_price * 100)
                    
                    if self.min_gain <= potential_gain <= self.max_gain:
                        patterns.append({
                            'pattern_type': 'Flag (Bearish)',
                            'direction': 'SELL',
                            'entry_price': current_price,
                            'target_price': target_price,
                            'stop_loss': flag_data['high'].max() * 1.03,
                            'potential_gain_pct': potential_gain,
                            'rationale': f'Bearish Flag pattern, continuation expected'
                        })
        except Exception as e:
            logging.debug(f"Error detecting Flag pattern: {e}")
        
        return patterns
    
    def _detect_ascending_triangle(self, price_data: pd.DataFrame, indicators: Dict, current_price: float) -> List[Dict]:
        """Detect Ascending Triangle pattern (bullish)"""
        patterns = []
        if len(price_data) < 30:
            return patterns
        
        try:
            # Ascending triangle: horizontal resistance, rising support
            recent_data = price_data.tail(30)
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            
            # Resistance line (relatively flat)
            resistance_high = np.percentile(highs, 90)
            resistance_low = np.percentile(highs, 75)
            resistance_range = (resistance_high - resistance_low) / resistance_high
            
            # Support line (rising)
            support_points = []
            for i in range(len(lows) - 5):
                local_low = lows[i:i+5].min()
                support_points.append((i, local_low))
            
            if len(support_points) >= 3 and resistance_range < 0.03:
                # Check if support is rising
                support_slope = (support_points[-1][1] - support_points[0][1]) / len(support_points)
                
                if support_slope > 0 and current_price > resistance_high * 0.95:
                    target_price = resistance_high * 1.12
                    potential_gain = (target_price - current_price) / current_price * 100
                    
                    if self.min_gain <= potential_gain <= self.max_gain:
                        patterns.append({
                            'pattern_type': 'Ascending Triangle',
                            'direction': 'BUY',
                            'entry_price': current_price,
                            'target_price': target_price,
                            'stop_loss': support_points[-1][1] * 0.98,
                            'potential_gain_pct': potential_gain,
                            'rationale': f'Ascending Triangle pattern, bullish breakout expected'
                        })
        except Exception as e:
            logging.debug(f"Error detecting Ascending Triangle: {e}")
        
        return patterns
    
    def _detect_descending_triangle(self, price_data: pd.DataFrame, indicators: Dict, current_price: float) -> List[Dict]:
        """Detect Descending Triangle pattern (bearish)"""
        patterns = []
        if len(price_data) < 30:
            return patterns
        
        try:
            # Descending triangle: horizontal support, falling resistance
            recent_data = price_data.tail(30)
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            
            # Support line (relatively flat)
            support_low = np.percentile(lows, 10)
            support_high = np.percentile(lows, 25)
            support_range = (support_high - support_low) / support_low
            
            # Resistance line (falling)
            resistance_points = []
            for i in range(len(highs) - 5):
                local_high = highs[i:i+5].max()
                resistance_points.append((i, local_high))
            
            if len(resistance_points) >= 3 and support_range < 0.03:
                # Check if resistance is falling
                resistance_slope = (resistance_points[-1][1] - resistance_points[0][1]) / len(resistance_points)
                
                if resistance_slope < 0 and current_price < support_high * 1.05:
                    target_price = support_low * 0.88
                    potential_gain = abs((target_price - current_price) / current_price * 100)
                    
                    if self.min_gain <= potential_gain <= self.max_gain:
                        patterns.append({
                            'pattern_type': 'Descending Triangle',
                            'direction': 'SELL',
                            'entry_price': current_price,
                            'target_price': target_price,
                            'stop_loss': resistance_points[-1][1] * 1.02,
                            'potential_gain_pct': potential_gain,
                            'rationale': f'Descending Triangle pattern, bearish breakdown expected'
                        })
        except Exception as e:
            logging.debug(f"Error detecting Descending Triangle: {e}")
        
        return patterns
    
    def _detect_symmetrical_triangle(self, price_data: pd.DataFrame, indicators: Dict, current_price: float) -> List[Dict]:
        """Detect Symmetrical Triangle pattern"""
        patterns = []
        if len(price_data) < 30:
            return patterns
        
        try:
            # Symmetrical triangle: converging support and resistance
            recent_data = price_data.tail(30)
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            
            # Get peaks and troughs
            peaks = []
            troughs = []
            
            for i in range(2, len(recent_data) - 2):
                if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                    peaks.append((i, highs[i]))
                if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                    troughs.append((i, lows[i]))
            
            if len(peaks) >= 2 and len(troughs) >= 2:
                peaks.sort(key=lambda x: x[0])
                troughs.sort(key=lambda x: x[0])
                
                # Check if peaks are descending and troughs are ascending (converging)
                peak_slope = (peaks[-1][1] - peaks[0][1]) / (peaks[-1][0] - peaks[0][0]) if len(peaks) > 1 else 0
                trough_slope = (troughs[-1][1] - troughs[0][1]) / (troughs[-1][0] - troughs[0][0]) if len(troughs) > 1 else 0
                
                if peak_slope < 0 and trough_slope > 0:  # Converging
                    # Determine direction based on current price position
                    mid_price = (peaks[-1][1] + troughs[-1][1]) / 2
                    
                    if current_price > mid_price:  # Bullish breakout
                        target_price = peaks[0][1] * 1.1
                        potential_gain = (target_price - current_price) / current_price * 100
                        
                        if self.min_gain <= potential_gain <= self.max_gain:
                            patterns.append({
                                'pattern_type': 'Symmetrical Triangle (Bullish)',
                                'direction': 'BUY',
                                'entry_price': current_price,
                                'target_price': target_price,
                                'stop_loss': troughs[-1][1] * 0.98,
                                'potential_gain_pct': potential_gain,
                                'rationale': f'Symmetrical Triangle, bullish breakout expected'
                            })
                    else:  # Bearish breakdown
                        target_price = troughs[0][1] * 0.9
                        potential_gain = abs((target_price - current_price) / current_price * 100)
                        
                        if self.min_gain <= potential_gain <= self.max_gain:
                            patterns.append({
                                'pattern_type': 'Symmetrical Triangle (Bearish)',
                                'direction': 'SELL',
                                'entry_price': current_price,
                                'target_price': target_price,
                                'stop_loss': peaks[-1][1] * 1.02,
                                'potential_gain_pct': potential_gain,
                                'rationale': f'Symmetrical Triangle, bearish breakdown expected'
                            })
        except Exception as e:
            logging.debug(f"Error detecting Symmetrical Triangle: {e}")
        
        return patterns
    
    def _detect_rounding_bottom(self, price_data: pd.DataFrame, indicators: Dict, current_price: float) -> List[Dict]:
        """Detect Rounding Bottom pattern (bullish reversal)"""
        patterns = []
        if len(price_data) < 40:
            return patterns
        
        try:
            closes = price_data['close'].values
            recent_data = price_data.tail(40)
            
            # Rounding bottom: U-shaped recovery
            # Should see gradual decline, then gradual rise
            mid_point = len(recent_data) // 2
            first_half = recent_data.head(mid_point)
            second_half = recent_data.tail(len(recent_data) - mid_point)
            
            first_avg = first_half['close'].mean()
            second_avg = second_half['close'].mean()
            mid_low = recent_data['low'].min()
            
            # Check for U-shape: ends higher than start, middle is lowest
            start_price = recent_data.iloc[0]['close']
            end_price = recent_data.iloc[-1]['close']
            
            if (end_price > start_price * 1.05 and 
                mid_low < start_price * 0.90 and 
                mid_low < end_price * 0.90):
                
                target_price = end_price * 1.15
                potential_gain = (target_price - current_price) / current_price * 100
                
                if self.min_gain <= potential_gain <= self.max_gain:
                    patterns.append({
                        'pattern_type': 'Rounding Bottom',
                        'direction': 'BUY',
                        'entry_price': current_price,
                        'target_price': target_price,
                        'stop_loss': mid_low * 0.95,
                        'potential_gain_pct': potential_gain,
                        'rationale': f'Rounding Bottom pattern, bullish reversal expected'
                    })
        except Exception as e:
            logging.debug(f"Error detecting Rounding Bottom: {e}")
        
        return patterns
    
    def _detect_engulfing_candlestick(self, price_data: pd.DataFrame, indicators: Dict, current_price: float) -> List[Dict]:
        """Detect Engulfing Candlestick pattern"""
        patterns = []
        if len(price_data) < 3:
            return patterns
        
        try:
            # Need at least last 2 candles
            recent = price_data.tail(2)
            
            prev_open = recent.iloc[0]['open']
            prev_close = recent.iloc[0]['close']
            prev_high = recent.iloc[0]['high']
            prev_low = recent.iloc[0]['low']
            
            curr_open = recent.iloc[1]['open']
            curr_close = recent.iloc[1]['close']
            curr_high = recent.iloc[1]['high']
            curr_low = recent.iloc[1]['low']
            
            # Bullish Engulfing: previous bearish, current bullish and engulfs previous
            if (prev_close < prev_open and  # Previous bearish
                curr_close > curr_open and  # Current bullish
                curr_open < prev_close and  # Current opens below prev close
                curr_close > prev_open):  # Current closes above prev open
                
                body_size = curr_close - curr_open
                prev_body_size = abs(prev_close - prev_open)
                
                if body_size > prev_body_size * 1.2:  # Current body engulfs previous
                    target_price = current_price * 1.12
                    potential_gain = (target_price - current_price) / current_price * 100
                    
                    if self.min_gain <= potential_gain <= self.max_gain:
                        patterns.append({
                            'pattern_type': 'Bullish Engulfing',
                            'direction': 'BUY',
                            'entry_price': current_price,
                            'target_price': target_price,
                            'stop_loss': curr_low * 0.97,
                            'potential_gain_pct': potential_gain,
                            'rationale': f'Bullish Engulfing candlestick pattern detected'
                        })
            
            # Bearish Engulfing: previous bullish, current bearish and engulfs previous
            elif (prev_close > prev_open and  # Previous bullish
                  curr_close < curr_open and  # Current bearish
                  curr_open > prev_close and  # Current opens above prev close
                  curr_close < prev_open):  # Current closes below prev open
                
                body_size = abs(curr_close - curr_open)
                prev_body_size = prev_close - prev_open
                
                if body_size > prev_body_size * 1.2:  # Current body engulfs previous
                    target_price = current_price * 0.88
                    potential_gain = abs((target_price - current_price) / current_price * 100)
                    
                    if self.min_gain <= potential_gain <= self.max_gain:
                        patterns.append({
                            'pattern_type': 'Bearish Engulfing',
                            'direction': 'SELL',
                            'entry_price': current_price,
                            'target_price': target_price,
                            'stop_loss': curr_high * 1.03,
                            'potential_gain_pct': potential_gain,
                            'rationale': f'Bearish Engulfing candlestick pattern detected'
                        })
        except Exception as e:
            logging.debug(f"Error detecting Engulfing Candlestick: {e}")
        
        return patterns
    
    def calculate_confidence(self, pattern: Dict, indicators: Dict, price_data: pd.DataFrame) -> float:
        """
        Calculate confidence score for a pattern (0-100)
        
        Args:
            pattern: Pattern dictionary
            indicators: Technical indicators
            price_data: Price data
            
        Returns:
            Confidence score (0-100)
        """
        confidence = 0.0
        
        # Base confidence from pattern type
        pattern_types = {
            'Golden Cross': 25,
            'Pullback to Support': 20,
            'Breakout from Consolidation': 25,
            'Oversold Bounce': 20,
            'MACD Bullish Crossover': 20
        }
        confidence += pattern_types.get(pattern['pattern_type'], 15)
        
        # Multiple indicators aligning
        rsi = indicators.get('current_rsi')
        macd = indicators.get('current_macd')
        macd_signal = indicators.get('current_macd_signal')
        sma_20 = indicators.get('current_sma_20')
        sma_50 = indicators.get('current_sma_50')
        volume_trend = indicators.get('volume_trend', 'stable')
        
        if rsi is not None:
            if 30 < rsi < 70:
                confidence += 10  # RSI in healthy range
            elif 40 < rsi < 60:
                confidence += 15  # RSI in optimal range
        
        if macd and macd_signal:
            if macd > macd_signal:
                confidence += 10  # MACD bullish
            if macd > 0 and macd_signal > 0:
                confidence += 5  # Both above zero
        
        if sma_20 and sma_50 and indicators.get('current_price'):
            if sma_20 > sma_50:
                confidence += 10  # Trend aligned
            if indicators['current_price'] > sma_20:
                confidence += 5  # Price above short-term MA
        
        if volume_trend == 'increasing':
            confidence += 15  # Volume confirmation
        
        # Risk-reward ratio
        risk = pattern['entry_price'] - pattern['stop_loss']
        reward = pattern['target_price'] - pattern['entry_price']
        if risk > 0:
            rr_ratio = reward / risk
            if rr_ratio >= 3:
                confidence += 10
            elif rr_ratio >= 2:
                confidence += 5
        
        # Support/Resistance strength
        support_levels = indicators.get('support_levels', [])
        resistance_levels = indicators.get('resistance_levels', [])
        if support_levels and resistance_levels:
            confidence += 10  # Clear support/resistance levels
        
        return min(confidence, 100.0)
    
    def get_patterns_for_stock(self, scrip_id: str) -> List[str]:
        """
        Get detected patterns for a stock (simplified for holdings display)
        
        Args:
            scrip_id: Stock symbol
            
        Returns:
            List of pattern type names
        """
        patterns = []
        
        try:
            # Get price data
            price_data = self.get_price_data(scrip_id, self.lookback_days)
            if price_data is None:
                return patterns
            
            # Calculate indicators
            indicators = self.calculate_indicators(price_data)
            if not indicators:
                return patterns
            
            # Detect patterns (without confidence/gain filtering for holdings)
            detected = self.detect_patterns(price_data, indicators)
            
            # Extract unique pattern types
            pattern_types = list(set([p.get('pattern_type') for p in detected if p.get('pattern_type')]))
            return pattern_types
            
        except Exception as e:
            logging.debug(f"Error getting patterns for {scrip_id}: {e}")
            return patterns
    
    def scan_stock(self, scrip_id: str) -> List[Dict]:
        """
        Scan a single stock for swing trade opportunities
        
        Args:
            scrip_id: Stock symbol
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        try:
            # Get price data
            price_data = self.get_price_data(scrip_id, self.lookback_days)
            if price_data is None:
                return recommendations
            
            # Calculate indicators
            indicators = self.calculate_indicators(price_data)
            if not indicators:
                return recommendations
            
            # Detect patterns
            patterns = self.detect_patterns(price_data, indicators)
            
            # Calculate confidence and prepare recommendations
            for pattern in patterns:
                confidence = self.calculate_confidence(pattern, indicators, price_data)
                
                if confidence >= self.min_confidence:
                    # Calculate risk-reward ratio
                    risk = pattern['entry_price'] - pattern['stop_loss']
                    reward = pattern['target_price'] - pattern['entry_price']
                    risk_reward = reward / risk if risk > 0 else 0
                    
                    # Estimate holding period (3-10 days)
                    holding_period = min(max(int(10 - (confidence / 10)), 3), 10)
                    
                    # Helper function to convert numpy types to Python native types
                    def to_python_type(value):
                        """Convert numpy types to Python native types"""
                        if value is None:
                            return None
                        try:
                            if isinstance(value, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                                return int(value)
                            elif isinstance(value, (np.floating, np.float64, np.float32, np.float16)):
                                return float(value)
                            elif isinstance(value, np.bool_):
                                return bool(value)
                            elif hasattr(value, 'item'):
                                return float(value.item()) if isinstance(value.item(), (int, float)) else value.item()
                            else:
                                return float(value) if isinstance(value, (int, float)) else value
                        except Exception:
                            return value
                    
                    # Get and convert indicator values
                    def get_indicator(key, default=None):
                        val = indicators.get(key, default)
                        return to_python_type(val) if val is not None else default
                    
                    def get_pattern_value(key):
                        val = pattern.get(key)
                        return to_python_type(val) if val is not None else None
                    
                    entry_price = to_python_type(pattern['entry_price'])
                    target_price = to_python_type(pattern['target_price'])
                    stop_loss = to_python_type(pattern['stop_loss'])
                    current_price = get_indicator('current_price', 0)
                    
                    recommendation = {
                        'scrip_id': scrip_id,
                        'pattern_type': pattern['pattern_type'],
                        'direction': pattern['direction'],
                        'entry_price': round(float(entry_price), 2),
                        'target_price': round(float(target_price), 2),
                        'stop_loss': round(float(stop_loss), 2),
                        'current_price': round(float(current_price), 2),
                        'potential_gain_pct': round(float(to_python_type(pattern['potential_gain_pct'])), 2),
                        'risk_reward_ratio': round(float(to_python_type(risk_reward)), 2),
                        'confidence_score': round(float(to_python_type(confidence)), 2),
                        'holding_period_days': int(holding_period),
                        'sma_20': get_indicator('current_sma_20'),
                        'sma_50': get_indicator('current_sma_50'),
                        'sma_200': get_indicator('current_sma_200'),
                        'rsi_14': get_indicator('current_rsi'),
                        'macd': get_indicator('current_macd'),
                        'macd_signal': get_indicator('current_macd_signal'),
                        'atr_14': get_indicator('current_atr'),
                        'volume_trend': indicators.get('volume_trend', 'stable'),
                        'support_level': round(float(to_python_type(max([s['price'] for s in indicators.get('support_levels', [])], default=0))), 2),
                        'resistance_level': round(float(to_python_type(min([r['price'] for r in indicators.get('resistance_levels', [])], default=0))), 2),
                        'rationale': pattern['rationale'],
                        'technical_context': {
                            'indicators': {
                                'sma_20': get_indicator('current_sma_20'),
                                'sma_50': get_indicator('current_sma_50'),
                                'sma_200': get_indicator('current_sma_200'),
                                'rsi': get_indicator('current_rsi'),
                                'macd': get_indicator('current_macd'),
                                'atr': get_indicator('current_atr')
                            }
                        },
                        'filtering_criteria': {
                            'min_gain': float(self.min_gain),
                            'max_gain': float(self.max_gain),
                            'min_confidence': float(self.min_confidence),
                            'lookback_days': int(self.lookback_days)
                        }
                    }
                    
                    recommendations.append(recommendation)
            
        except Exception as e:
            logging.error(f"Error scanning stock {scrip_id}: {e}")
            import traceback
            logging.error(traceback.format_exc())
        
        return recommendations
    
    def scan_nifty(self) -> List[Dict]:
        """
        Scan Nifty for swing trade opportunities
        
        Returns:
            List of recommendations
        """
        return self.scan_stock('NIFTY50')
    
    def scan_all_stocks(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Scan all stocks for swing trade opportunities
        
        Args:
            limit: Optional limit on number of stocks to scan
            
        Returns:
            List of all recommendations
        """
        all_recommendations = []
        
        stocks = self.get_stocks_list()
        if limit:
            stocks = stocks[:limit]
        
        logging.info(f"Scanning {len(stocks)} stocks for swing trade opportunities...")
        
        for i, scrip_id in enumerate(stocks, 1):
            if i % 10 == 0:
                logging.info(f"Scanned {i}/{len(stocks)} stocks, found {len(all_recommendations)} opportunities so far")
            
            recommendations = self.scan_stock(scrip_id)
            all_recommendations.extend(recommendations)
        
        # Sort by confidence score
        all_recommendations.sort(key=lambda x: x['confidence_score'], reverse=True)
        
        logging.info(f"Scan complete. Found {len(all_recommendations)} swing trade opportunities")
        
        return all_recommendations
    
    def save_recommendations(self, recommendations: List[Dict], analysis_date: Optional[date] = None) -> bool:
        """
        Save recommendations to database
        
        Args:
            recommendations: List of recommendation dictionaries
            analysis_date: Date of analysis (default: today)
            
        Returns:
            True if successful, False otherwise
        """
        if not recommendations:
            return True
        
        if analysis_date is None:
            analysis_date = date.today()
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            def convert_numpy_types(value):
                """Convert numpy types to Python native types"""
                if value is None:
                    return None
                try:
                    if isinstance(value, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                        return int(value)
                    elif isinstance(value, (np.floating, np.float64, np.float32, np.float16)):
                        return float(value)
                    elif isinstance(value, np.bool_):
                        return bool(value)
                    elif isinstance(value, np.ndarray):
                        return value.tolist()
                    elif isinstance(value, (dict, type({}))):
                        return {k: convert_numpy_types(v) for k, v in value.items()}
                    elif isinstance(value, (list, tuple)):
                        return type(value)(convert_numpy_types(v) for v in value)
                    else:
                        return value
                except Exception:
                    # If conversion fails, try direct conversion
                    try:
                        if hasattr(value, 'item'):
                            return value.item()
                        return float(value) if isinstance(value, (int, float)) else value
                    except Exception:
                        return value
            
            rows = []
            for rec in recommendations:
                row = (
                    analysis_date,
                    rec.get('scrip_id'),
                    None,  # instrument_token (can be added later)
                    rec.get('pattern_type'),
                    rec.get('direction', 'BUY'),
                    convert_numpy_types(rec.get('entry_price')),
                    convert_numpy_types(rec.get('target_price')),
                    convert_numpy_types(rec.get('stop_loss')),
                    convert_numpy_types(rec.get('potential_gain_pct')),
                    convert_numpy_types(rec.get('risk_reward_ratio')),
                    convert_numpy_types(rec.get('confidence_score')),
                    convert_numpy_types(rec.get('holding_period_days')),
                    convert_numpy_types(rec.get('current_price')),
                    convert_numpy_types(rec.get('sma_20')),
                    convert_numpy_types(rec.get('sma_50')),
                    convert_numpy_types(rec.get('sma_200')),
                    convert_numpy_types(rec.get('rsi_14')),
                    convert_numpy_types(rec.get('macd')),
                    convert_numpy_types(rec.get('macd_signal')),
                    convert_numpy_types(rec.get('atr_14')),
                    rec.get('volume_trend'),
                    convert_numpy_types(rec.get('support_level')),
                    convert_numpy_types(rec.get('resistance_level')),
                    rec.get('rationale'),
                    json.dumps(convert_numpy_types(rec.get('technical_context', {}))),
                    json.dumps(convert_numpy_types(rec.get('diagnostics', {}))),
                    json.dumps(convert_numpy_types(rec.get('filtering_criteria', {})))
                )
                rows.append(row)
            
            # First, mark existing records for today as inactive (soft delete)
            cursor.execute("""
                UPDATE my_schema.swing_trade_suggestions
                SET status = 'REPLACED'
                WHERE run_date = %s AND status = 'ACTIVE'
            """, (analysis_date,))
            
            # Then insert new recommendations
            cursor.executemany("""
                INSERT INTO my_schema.swing_trade_suggestions (
                    analysis_date, scrip_id, instrument_token, pattern_type, direction,
                    entry_price, target_price, stop_loss, potential_gain_pct, risk_reward_ratio,
                    confidence_score, holding_period_days, current_price,
                    sma_20, sma_50, sma_200, rsi_14, macd, macd_signal, atr_14,
                    volume_trend, support_level, resistance_level, rationale,
                    technical_context, diagnostics, filtering_criteria
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s
                )
            """, rows)
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logging.info(f"Saved {len(recommendations)} swing trade recommendations to database")
            return True
            
        except Exception as e:
            logging.error(f"Error saving recommendations: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return False

