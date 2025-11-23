"""
Accumulation/Distribution Analyzer
Detects accumulation and distribution patterns using multiple technical indicators and chart patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import json
import psycopg2
import psycopg2.extras
from common.Boilerplate import get_db_connection, log_stock_price_fetch_error
from common.TechnicalIndicators import (
    calculate_obv, 
    calculate_ad_indicator, 
    detect_declining_momentum,
    calculate_volume_trend
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AccumulationDistributionAnalyzer:
    """
    Analyzes stocks for accumulation and distribution patterns
    """
    
    def __init__(self):
        """Initialize the analyzer"""
        self.conn = None
        self.cursor = None
        
    def _get_connection(self):
        """Get database connection"""
        if self.conn is None or self.conn.closed:
            self.conn = get_db_connection()
            self.cursor = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        return self.conn, self.cursor
    
    def get_price_data(self, scrip_id: str, lookback_days: int = 30) -> Optional[pd.DataFrame]:
        """
        Get OHLCV price data for a stock from rt_intraday_price
        
        Args:
            scrip_id: Stock symbol
            lookback_days: Number of days to fetch (default: 30)
            
        Returns:
            DataFrame with OHLCV data or None
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
                AND price_high IS NOT NULL
                AND price_low IS NOT NULL
                AND price_close IS NOT NULL
                AND volume IS NOT NULL
                ORDER BY price_date ASC
            """, (scrip_id, lookback_days))
            
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            
            if not rows or len(rows) < 14:  # Need at least 14 days for reliable analysis
                return None
            
            # Convert to DataFrame
            data = []
            for row in rows:
                data.append({
                    'date': row['price_date'],
                    'open': float(row['price_open']) if row['price_open'] else 0.0,
                    'high': float(row['price_high']) if row['price_high'] else 0.0,
                    'low': float(row['price_low']) if row['price_low'] else 0.0,
                    'close': float(row['price_close']) if row['price_close'] else 0.0,
                    'volume': float(row['volume']) if row['volume'] else 0.0
                })
            
            df = pd.DataFrame(data)
            df.set_index('date', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting price data for {scrip_id}: {e}")
            log_stock_price_fetch_error(scrip_id, e, "AccumulationDistributionAnalyzer.get_price_data")
            return None
    
    def _detect_head_and_shoulders(self, price_data: pd.DataFrame) -> bool:
        """
        Detect Head and Shoulders pattern (bearish reversal - distribution signal)
        
        Args:
            price_data: DataFrame with OHLC data
            
        Returns:
            True if pattern detected, False otherwise
        """
        if len(price_data) < 60:
            return False
        
        try:
            highs = price_data['high'].values
            closes = price_data['close'].values
            
            # Look for three peaks: left shoulder, head, right shoulder
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
                    current_price = closes[-1]
                    
                    if neckline and current_price < head_price * 0.95:
                        return True
        except Exception as e:
            logger.debug(f"Error detecting Head and Shoulders: {e}")
        
        return False
    
    def _detect_double_top(self, price_data: pd.DataFrame) -> bool:
        """
        Detect Double Top pattern (bearish reversal - distribution signal)
        
        Args:
            price_data: DataFrame with OHLC data
            
        Returns:
            True if pattern detected, False otherwise
        """
        if len(price_data) < 40:
            return False
        
        try:
            highs = price_data['high'].values
            recent_data = price_data.tail(40)
            peaks = []
            
            # Find local maxima
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
                            current_price = price_data['close'].iloc[-1]
                            if current_price < max(peak1_price, peak2_price):
                                return True
        except Exception as e:
            logger.debug(f"Error detecting Double Top: {e}")
        
        return False
    
    def _detect_wyckoff_distribution(self, price_data: pd.DataFrame, volume: pd.Series, obv: pd.Series) -> bool:
        """
        Detect simplified Wyckoff distribution pattern
        
        Distribution signals:
        - Price makes higher highs but volume decreases
        - Price consolidates sideways after rally with elevated volume
        - Price starts declining with increasing volume
        - OBV declining while price is flat or declining
        
        Args:
            price_data: DataFrame with OHLC data
            volume: Series of volume values
            obv: Series of OBV values
            
        Returns:
            True if distribution pattern detected, False otherwise
        """
        if len(price_data) < 20:
            return False
        
        try:
            closes = price_data['close'].values
            highs = price_data['high'].values
            recent_data = price_data.tail(20)
            
            # Check 1: Price makes higher highs but volume decreases (FIXED: non-overlapping periods)
            if len(recent_data) >= 20:
                first_half = recent_data.iloc[:10]  # First 10 days (non-overlapping)
                second_half = recent_data.iloc[10:]  # Last 10 days (non-overlapping)
                
                first_half_high = first_half['high'].max()
                second_half_high = second_half['high'].max()
                first_half_vol = first_half['volume'].mean()
                second_half_vol = second_half['volume'].mean()
                
                if second_half_high > first_half_high * 1.02 and second_half_vol < first_half_vol * 0.9:
                    return True
            
            # Check 2: Price consolidates sideways after rally with elevated volume
            if len(recent_data) >= 15:
                rally_period = recent_data.head(7)
                consolidation_period = recent_data.tail(8)
                
                rally_high = rally_period['high'].max()
                rally_low = rally_period['low'].min()
                rally_range = rally_high - rally_low
                
                consolidation_high = consolidation_period['high'].max()
                consolidation_low = consolidation_period['low'].min()
                consolidation_range = consolidation_high - consolidation_low
                
                rally_vol = rally_period['volume'].mean()
                consolidation_vol = consolidation_period['volume'].mean()
                
                # Sideways consolidation (range < 50% of rally range) with higher volume
                if (consolidation_range < rally_range * 0.5 and 
                    consolidation_vol > rally_vol * 1.1 and
                    consolidation_high < rally_high * 1.05):
                    return True
            
            # Check 3: Price stagnation/decline with volume
            if len(recent_data) >= 10:
                first_half = recent_data.head(5)
                second_half = recent_data.tail(5)
                
                first_half_close = first_half['close'].mean()
                second_half_close = second_half['close'].mean()
                second_half_vol = second_half['volume'].mean()
                first_half_vol = first_half['volume'].mean()
                
                # Price declining or stagnant with volume
                if (second_half_close <= first_half_close * 1.01 and 
                    second_half_vol > first_half_vol * 1.1):
                    return True
            
            # Check 4: OBV declining while price is flat or declining
            if len(obv) >= 10:
                recent_obv = obv.tail(10)
                recent_closes = closes[-10:]
                
                obv_trend = recent_obv.iloc[-1] - recent_obv.iloc[0]
                price_trend = recent_closes[-1] - recent_closes[0]
                
                # OBV declining while price is flat or declining
                if obv_trend < 0 and price_trend <= 0:
                    return True
                    
        except Exception as e:
            logger.debug(f"Error detecting Wyckoff distribution: {e}")
        
        return False
    
    def _detect_wyckoff_accumulation(self, price_data: pd.DataFrame, volume: pd.Series, obv: pd.Series) -> bool:
        """
        Detect simplified Wyckoff accumulation pattern
        
        Accumulation signals:
        - Price makes lower lows but volume decreases
        - Price consolidates at support with decreasing volume
        - Price starts rising with increasing volume
        - OBV rising while price is flat or rising
        
        Args:
            price_data: DataFrame with OHLC data
            volume: Series of volume values
            obv: Series of OBV values
            
        Returns:
            True if accumulation pattern detected, False otherwise
        """
        if len(price_data) < 20:
            return False
        
        try:
            closes = price_data['close'].values
            lows = price_data['low'].values
            recent_data = price_data.tail(20)
            
            # Check 1: Price makes lower lows but volume decreases (FIXED: non-overlapping periods)
            if len(recent_data) >= 20:
                first_half = recent_data.iloc[:10]  # First 10 days (non-overlapping)
                second_half = recent_data.iloc[10:]  # Last 10 days (non-overlapping)
                
                first_half_low = first_half['low'].min()
                second_half_low = second_half['low'].min()
                first_half_vol = first_half['volume'].mean()
                second_half_vol = second_half['volume'].mean()
                
                if second_half_low < first_half_low * 0.98 and second_half_vol < first_half_vol * 0.9:
                    return True
            
            # Check 2: Price consolidates at support with decreasing volume
            if len(recent_data) >= 15:
                decline_period = recent_data.head(7)
                consolidation_period = recent_data.tail(8)
                
                decline_low = decline_period['low'].min()
                consolidation_low = consolidation_period['low'].min()
                
                decline_vol = decline_period['volume'].mean()
                consolidation_vol = consolidation_period['volume'].mean()
                
                # Consolidation near support (within 2% of decline low) with decreasing volume
                if (abs(consolidation_low - decline_low) / decline_low < 0.02 and 
                    consolidation_vol < decline_vol * 0.9):
                    return True
            
            # Check 3: Price starts rising with increasing volume
            if len(recent_data) >= 10:
                first_half = recent_data.head(5)
                second_half = recent_data.tail(5)
                
                first_half_close = first_half['close'].mean()
                second_half_close = second_half['close'].mean()
                second_half_vol = second_half['volume'].mean()
                first_half_vol = first_half['volume'].mean()
                
                # Price rising with volume
                if (second_half_close > first_half_close * 1.01 and 
                    second_half_vol > first_half_vol * 1.1):
                    return True
            
            # Check 4: OBV rising while price is flat or rising
            if len(obv) >= 10:
                recent_obv = obv.tail(10)
                recent_closes = closes[-10:]
                
                obv_trend = recent_obv.iloc[-1] - recent_obv.iloc[0]
                price_trend = recent_closes[-1] - recent_closes[0]
                
                # OBV rising while price is flat or rising
                if obv_trend > 0 and price_trend >= 0:
                    return True
                    
        except Exception as e:
            logger.debug(f"Error detecting Wyckoff accumulation: {e}")
        
        return False
    
    def _detect_ad_divergence(self, price_data: pd.DataFrame, ad_line: pd.Series) -> Tuple[bool, bool]:
        """
        Detect A/D line divergences with price
        
        Bullish divergence: Price makes lower lows, A/D makes higher lows (accumulation signal)
        Bearish divergence: Price makes higher highs, A/D makes lower highs (distribution signal)
        
        Args:
            price_data: DataFrame with OHLC data
            ad_line: Series of A/D line values
            
        Returns:
            Tuple of (bullish_divergence, bearish_divergence) booleans
        """
        if len(price_data) < 20 or len(ad_line) < 20:
            return False, False
        
        try:
            recent_data = price_data.tail(20)
            recent_ad = ad_line.tail(20)
            
            # Find price peaks and A/D peaks for bearish divergence
            # Look for local maxima (peaks)
            price_highs = []
            ad_highs = []
            
            for i in range(1, len(recent_data) - 1):
                if (recent_data.iloc[i]['high'] >= recent_data.iloc[i-1]['high'] and 
                    recent_data.iloc[i]['high'] >= recent_data.iloc[i+1]['high']):
                    price_highs.append((i, recent_data.iloc[i]['high']))
            
            for i in range(1, len(recent_ad) - 1):
                if (recent_ad.iloc[i] >= recent_ad.iloc[i-1] and 
                    recent_ad.iloc[i] >= recent_ad.iloc[i+1]):
                    ad_highs.append((i, recent_ad.iloc[i]))
            
            # Bearish divergence: Price makes higher highs, A/D makes lower highs
            if len(price_highs) >= 2 and len(ad_highs) >= 2:
                # Get last two peaks
                latest_price_peak_idx, latest_price_peak = price_highs[-1]
                prev_price_peak_idx, prev_price_peak = price_highs[-2]
                
                # Find corresponding A/D peaks (closest in time)
                latest_ad_peak_idx, latest_ad_peak = ad_highs[-1]
                prev_ad_peak_idx, prev_ad_peak = ad_highs[-2]
                
                # Check if price is making higher highs while A/D is making lower highs
                if (latest_price_peak > prev_price_peak * 1.01 and 
                    latest_ad_peak < prev_ad_peak * 0.99):
                    return False, True  # Bearish divergence detected
            
            # Find price troughs and A/D troughs for bullish divergence
            price_lows = []
            ad_lows = []
            
            for i in range(1, len(recent_data) - 1):
                if (recent_data.iloc[i]['low'] <= recent_data.iloc[i-1]['low'] and 
                    recent_data.iloc[i]['low'] <= recent_data.iloc[i+1]['low']):
                    price_lows.append((i, recent_data.iloc[i]['low']))
            
            for i in range(1, len(recent_ad) - 1):
                if (recent_ad.iloc[i] <= recent_ad.iloc[i-1] and 
                    recent_ad.iloc[i] <= recent_ad.iloc[i+1]):
                    ad_lows.append((i, recent_ad.iloc[i]))
            
            # Bullish divergence: Price makes lower lows, A/D makes higher lows
            if len(price_lows) >= 2 and len(ad_lows) >= 2:
                # Get last two troughs
                latest_price_trough_idx, latest_price_trough = price_lows[-1]
                prev_price_trough_idx, prev_price_trough = price_lows[-2]
                
                # Find corresponding A/D troughs (closest in time)
                latest_ad_trough_idx, latest_ad_trough = ad_lows[-1]
                prev_ad_trough_idx, prev_ad_trough = ad_lows[-2]
                
                # Check if price is making lower lows while A/D is making higher lows
                if (latest_price_trough < prev_price_trough * 0.99 and 
                    latest_ad_trough > prev_ad_trough * 1.01):
                    return True, False  # Bullish divergence detected
                    
        except Exception as e:
            logger.debug(f"Error detecting A/D divergence: {e}")
        
        return False, False
    
    def _analyze_volume_patterns(self, price_data: pd.DataFrame, volume: pd.Series) -> Dict:
        """
        Analyze volume patterns for accumulation/distribution signals
        
        Args:
            price_data: DataFrame with OHLC data
            volume: Series of volume values
            
        Returns:
            Dictionary with volume analysis results
        """
        analysis = {
            'sideways_after_rally': False,
            'price_stagnation_with_volume': False,
            'volume_trend': 'stable'
        }
        
        try:
            if len(price_data) < 15:
                return analysis
            
            recent_data = price_data.tail(15)
            
            # Check for sideways after rally with higher volumes
            if len(recent_data) >= 10:
                rally_period = recent_data.head(5)
                sideways_period = recent_data.tail(10)
                
                rally_high = rally_period['high'].max()
                rally_low = rally_period['low'].min()
                rally_range = rally_high - rally_low
                
                sideways_high = sideways_period['high'].max()
                sideways_low = sideways_period['low'].min()
                sideways_range = sideways_high - sideways_low
                
                rally_vol = rally_period['volume'].mean()
                sideways_vol = sideways_period['volume'].mean()
                
                # Sideways consolidation (range < 50% of rally range) with higher volume
                if (sideways_range < rally_range * 0.5 and 
                    sideways_vol > rally_vol * 1.1):
                    analysis['sideways_after_rally'] = True
            
            # Check for price stagnation/decline with volume
            if len(recent_data) >= 10:
                first_half = recent_data.head(5)
                second_half = recent_data.tail(5)
                
                first_half_close = first_half['close'].mean()
                second_half_close = second_half['close'].mean()
                second_half_vol = second_half['volume'].mean()
                first_half_vol = first_half['volume'].mean()
                
                # Price declining or stagnant with volume
                if (second_half_close <= first_half_close * 1.01 and 
                    second_half_vol > first_half_vol * 1.1):
                    analysis['price_stagnation_with_volume'] = True
            
            # Calculate volume trend
            analysis['volume_trend'] = calculate_volume_trend(volume, period=10)
            
        except Exception as e:
            logger.debug(f"Error analyzing volume patterns: {e}")
        
        return analysis
    
    def _calculate_momentum_score(self, price_data: pd.DataFrame) -> float:
        """
        Calculate momentum score (0-100, lower = declining momentum)
        
        Args:
            price_data: DataFrame with OHLC data
            
        Returns:
            Momentum score (0-100)
        """
        try:
            closes = price_data['close']
            return detect_declining_momentum(closes, period=14)
        except Exception as e:
            logger.debug(f"Error calculating momentum score: {e}")
            return 50.0  # Neutral
    
    def analyze_stock(self, scrip_id: str, lookback_days: int = 30) -> Optional[Dict]:
        """
        Main analysis method to detect accumulation/distribution patterns
        
        Args:
            scrip_id: Stock symbol
            lookback_days: Number of days to analyze (default: 30)
            
        Returns:
            Dictionary with analysis results or None if error
        """
        try:
            # Get price data
            price_data = self.get_price_data(scrip_id, lookback_days)
            if price_data is None or len(price_data) < 14:
                logger.debug(f"Insufficient data for {scrip_id}")
                return None
            
            # Calculate technical indicators
            closes = price_data['close']
            highs = price_data['high']
            lows = price_data['low']
            volumes = price_data['volume']
            
            # Calculate OBV and A/D indicators
            obv = calculate_obv(closes, volumes)
            ad_line = calculate_ad_indicator(highs, lows, closes, volumes)
            
            # Get latest values
            latest_obv = float(obv.iloc[-1]) if len(obv) > 0 and not pd.isna(obv.iloc[-1]) else None
            latest_ad = float(ad_line.iloc[-1]) if len(ad_line) > 0 and not pd.isna(ad_line.iloc[-1]) else None
            
            # Detect patterns
            head_shoulders = self._detect_head_and_shoulders(price_data)
            double_top = self._detect_double_top(price_data)
            wyckoff_distribution = self._detect_wyckoff_distribution(price_data, volumes, obv)
            wyckoff_accumulation = self._detect_wyckoff_accumulation(price_data, volumes, obv)
            
            # Detect A/D divergences (ENHANCED: new signal)
            ad_bullish_div, ad_bearish_div = self._detect_ad_divergence(price_data, ad_line)
            
            # Analyze volume patterns
            volume_analysis = self._analyze_volume_patterns(price_data, volumes)
            
            # Calculate momentum score
            momentum_score = self._calculate_momentum_score(price_data)
            
            # ENHANCED: Weighted signal scoring system
            distribution_score = 0.0
            accumulation_score = 0.0
            
            # Distribution signals (with weights)
            if head_shoulders:
                distribution_score += 2.0  # Strong pattern signal
            if double_top:
                distribution_score += 2.0  # Strong pattern signal
            if wyckoff_distribution:
                distribution_score += 1.5  # Medium pattern signal
            if volume_analysis.get('sideways_after_rally'):
                distribution_score += 1.0  # Volume pattern signal
            if volume_analysis.get('price_stagnation_with_volume'):
                distribution_score += 1.0  # Volume pattern signal
            if ad_bearish_div:
                distribution_score += 1.5  # Strong divergence signal
            if momentum_score < 40:  # Declining momentum
                # Scale based on how low the momentum is (lower = stronger signal)
                distribution_score += (40 - momentum_score) / 40.0 * 1.5
            
            # Accumulation signals (with weights)
            if wyckoff_accumulation:
                accumulation_score += 1.5  # Medium pattern signal
            if ad_bullish_div:
                accumulation_score += 1.5  # Strong divergence signal
            if momentum_score > 60:  # Rising momentum
                # Scale based on how high the momentum is (higher = stronger signal)
                accumulation_score += (momentum_score - 60) / 40.0 * 1.5
            if volume_analysis.get('volume_trend') == 'increasing' and len(closes) >= 10:
                if closes.iloc[-1] > closes.iloc[-10]:
                    accumulation_score += 1.0  # Volume trend signal
            
            # Determine state with weighted threshold
            if distribution_score > accumulation_score and distribution_score >= 2.0:
                state = 'DISTRIBUTION'
                confidence = min(100.0, 50.0 + (distribution_score * 15.0))
            elif accumulation_score > distribution_score and accumulation_score >= 2.0:
                state = 'ACCUMULATION'
                confidence = min(100.0, 50.0 + (accumulation_score * 15.0))
            else:
                state = 'NEUTRAL'
                confidence = 50.0
            
            # Calculate signal counts for backward compatibility
            distribution_signals = int(distribution_score)
            accumulation_signals = int(accumulation_score)
            
            # Determine pattern detected (FIXED: align with state to prevent inconsistencies)
            pattern_detected = None
            if state == 'DISTRIBUTION':
                # For distribution state, only show distribution patterns
                if head_shoulders:
                    pattern_detected = 'Head and Shoulders'
                elif double_top:
                    pattern_detected = 'Double Top'
                elif wyckoff_distribution:
                    pattern_detected = 'Wyckoff Distribution'
                elif ad_bearish_div:
                    pattern_detected = 'A/D Bearish Divergence'
            elif state == 'ACCUMULATION':
                # For accumulation state, only show accumulation patterns
                if wyckoff_accumulation:
                    pattern_detected = 'Wyckoff Accumulation'
                elif ad_bullish_div:
                    pattern_detected = 'A/D Bullish Divergence'
            else:
                # For neutral state, check all patterns in priority order
                if head_shoulders:
                    pattern_detected = 'Head and Shoulders'
                elif double_top:
                    pattern_detected = 'Double Top'
                elif wyckoff_distribution:
                    pattern_detected = 'Wyckoff Distribution'
                elif wyckoff_accumulation:
                    pattern_detected = 'Wyckoff Accumulation'
                elif ad_bearish_div:
                    pattern_detected = 'A/D Bearish Divergence'
                elif ad_bullish_div:
                    pattern_detected = 'A/D Bullish Divergence'
            
            # Build technical context (ENHANCED: include new signals and scores)
            technical_context = {
                'distribution_signals': distribution_signals,
                'accumulation_signals': accumulation_signals,
                'distribution_score': round(distribution_score, 2),
                'accumulation_score': round(accumulation_score, 2),
                'head_shoulders': head_shoulders,
                'double_top': double_top,
                'wyckoff_distribution': wyckoff_distribution,
                'wyckoff_accumulation': wyckoff_accumulation,
                'ad_bullish_divergence': ad_bullish_div,
                'ad_bearish_divergence': ad_bearish_div,
                'volume_analysis': volume_analysis,
                'momentum_score': momentum_score
            }
            
            result = {
                'scrip_id': scrip_id,
                'state': state,
                'obv_value': latest_obv,
                'ad_value': latest_ad,
                'momentum_score': momentum_score,
                'pattern_detected': pattern_detected,
                'volume_analysis': volume_analysis,
                'confidence_score': confidence,
                'technical_context': technical_context
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing stock {scrip_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def get_current_state(self, scrip_id: str, analysis_date: date) -> Optional[Dict]:
        """
        Get current accumulation/distribution state from database
        
        Args:
            scrip_id: Stock symbol
            analysis_date: Date to get state for
            
        Returns:
            Dictionary with state information or None
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cursor.execute("""
                SELECT 
                    state,
                    start_date,
                    days_in_state,
                    obv_value,
                    ad_value,
                    momentum_score,
                    pattern_detected,
                    volume_analysis,
                    confidence_score,
                    technical_context
                FROM my_schema.accumulation_distribution
                WHERE scrip_id = %s
                AND analysis_date = %s
                ORDER BY created_at DESC
                LIMIT 1
            """, (scrip_id, analysis_date))
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result:
                return dict(result)
            return None
            
        except Exception as e:
            logger.error(f"Error getting current state for {scrip_id}: {e}")
            return None
    
    def get_days_in_state(self, scrip_id: str, current_state: str, analysis_date: date) -> int:
        """
        Calculate consecutive days in current state
        
        Args:
            scrip_id: Stock symbol
            current_state: Current state (ACCUMULATION, DISTRIBUTION, NEUTRAL)
            analysis_date: Analysis date
            
        Returns:
            Number of consecutive days in current state
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Count backwards from analysis_date until state changes
            days = 0
            check_date = analysis_date
            
            while True:
                cursor.execute("""
                    SELECT state
                    FROM my_schema.accumulation_distribution
                    WHERE scrip_id = %s
                    AND analysis_date = %s
                    ORDER BY created_at DESC
                    LIMIT 1
                """, (scrip_id, check_date))
                
                result = cursor.fetchone()
                
                if result and result['state'] == current_state:
                    days += 1
                    check_date = check_date - timedelta(days=1)
                else:
                    break
                
                # Safety limit
                if days > 365:
                    break
            
            cursor.close()
            conn.close()
            
            return days
            
        except Exception as e:
            logger.error(f"Error calculating days in state for {scrip_id}: {e}")
            return 0
    
    def save_analysis_result(self, scrip_id: str, analysis_date: date, result: Dict) -> bool:
        """
        Save analysis result to database
        
        Args:
            scrip_id: Stock symbol
            analysis_date: Analysis date
            result: Analysis result dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Get previous state to check if it changed
            cursor.execute("""
                SELECT state, start_date
                FROM my_schema.accumulation_distribution
                WHERE scrip_id = %s
                AND analysis_date < %s
                ORDER BY analysis_date DESC
                LIMIT 1
            """, (scrip_id, analysis_date))
            
            prev_result = cursor.fetchone()
            prev_state = prev_result['state'] if prev_result else None
            prev_start_date = prev_result['start_date'] if prev_result else None
            
            current_state = result.get('state', 'NEUTRAL')
            
            # Determine start_date
            if prev_state == current_state and prev_start_date:
                # State hasn't changed, keep same start_date
                start_date = prev_start_date
            else:
                # State changed, start_date is today
                start_date = analysis_date
                
                # Update history table: end previous state
                if prev_state and prev_start_date:
                    cursor.execute("""
                        UPDATE my_schema.accumulation_distribution_history
                        SET end_date = %s,
                            duration_days = %s
                        WHERE scrip_id = %s
                        AND state = %s
                        AND start_date = %s
                        AND end_date IS NULL
                    """, (
                        analysis_date - timedelta(days=1),
                        (analysis_date - prev_start_date).days,
                        scrip_id,
                        prev_state,
                        prev_start_date
                    ))
                    
                    # Insert new state into history
                    cursor.execute("""
                        INSERT INTO my_schema.accumulation_distribution_history
                        (scrip_id, state, start_date, duration_days)
                        VALUES (%s, %s, %s, NULL)
                    """, (scrip_id, current_state, start_date))
            
            # Calculate days_in_state
            days_in_state = self.get_days_in_state(scrip_id, current_state, analysis_date)
            
            # Prepare volume_analysis and technical_context as JSON
            volume_analysis_json = json.dumps(result.get('volume_analysis', {}))
            technical_context_json = json.dumps(result.get('technical_context', {}))
            
            # Insert or update current state
            cursor.execute("""
                INSERT INTO my_schema.accumulation_distribution (
                    scrip_id, analysis_date, run_date, state, start_date, days_in_state,
                    obv_value, ad_value, momentum_score, pattern_detected,
                    volume_analysis, confidence_score, technical_context
                ) VALUES (
                    %s, %s, CURRENT_DATE, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s::jsonb
                )
                ON CONFLICT (scrip_id, analysis_date)
                DO UPDATE SET
                    state = EXCLUDED.state,
                    start_date = EXCLUDED.start_date,
                    days_in_state = EXCLUDED.days_in_state,
                    obv_value = EXCLUDED.obv_value,
                    ad_value = EXCLUDED.ad_value,
                    momentum_score = EXCLUDED.momentum_score,
                    pattern_detected = EXCLUDED.pattern_detected,
                    volume_analysis = EXCLUDED.volume_analysis,
                    confidence_score = EXCLUDED.confidence_score,
                    technical_context = EXCLUDED.technical_context
            """, (
                scrip_id,
                analysis_date,
                current_state,
                start_date,
                days_in_state,
                result.get('obv_value'),
                result.get('ad_value'),
                result.get('momentum_score'),
                result.get('pattern_detected'),
                volume_analysis_json,
                result.get('confidence_score'),
                technical_context_json
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving analysis result for {scrip_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def update_state_history(self, scrip_id: str, old_state: str, new_state: str, start_date: date, end_date: date) -> bool:
        """
        Update state history when state changes
        
        Args:
            scrip_id: Stock symbol
            old_state: Previous state
            new_state: New state
            start_date: Start date of old state
            end_date: End date of old state (when state changed)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Update old state end_date
            cursor.execute("""
                UPDATE my_schema.accumulation_distribution_history
                SET end_date = %s,
                    duration_days = %s
                WHERE scrip_id = %s
                AND state = %s
                AND start_date = %s
                AND end_date IS NULL
            """, (
                end_date,
                (end_date - start_date).days,
                scrip_id,
                old_state,
                start_date
            ))
            
            # Insert new state
            cursor.execute("""
                INSERT INTO my_schema.accumulation_distribution_history
                (scrip_id, state, start_date, duration_days)
                VALUES (%s, %s, %s, NULL)
            """, (scrip_id, new_state, end_date + timedelta(days=1)))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating state history for {scrip_id}: {e}")
            return False

