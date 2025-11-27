"""
Alternate Data Service
Provides alternate data indicators like volume spikes, OBV changes, supertrend changes, and golden crosses
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional
import logging
from common.Boilerplate import get_db_connection
from common.TechnicalIndicators import (
    calculate_obv, calculate_sma, calculate_ema, calculate_rsi, 
    calculate_macd, calculate_atr, calculate_volume_trend,
    calculate_momentum, detect_declining_momentum
)
from api.utils.technical_indicators import get_latest_supertrend

logger = logging.getLogger(__name__)


class AlternateDataService:
    """Service for calculating alternate data indicators"""
    
    def __init__(self):
        """Initialize Alternate Data Service"""
        pass
    
    def get_stocks_with_increased_volume(self, min_volume_increase_pct: float = 50.0, 
                                         lookback_days: int = 5) -> List[Dict]:
        """
        Get stocks with increased volume compared to previous trading day
        
        Args:
            min_volume_increase_pct: Minimum percentage increase in volume (default: 50%)
            lookback_days: Number of days to look back for finding previous trading day (default: 5)
            
        Returns:
            List of stocks with volume increases
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get the latest trading date and previous trading date
            cursor.execute("""
                SELECT DISTINCT price_date::date as trade_date
                FROM my_schema.rt_intraday_price
                WHERE price_date::date <= CURRENT_DATE
                AND volume IS NOT NULL
                AND volume > 0
                ORDER BY price_date::date DESC
                LIMIT 2
            """)
            
            date_rows = cursor.fetchall()
            if len(date_rows) < 2:
                cursor.close()
                conn.close()
                logger.warning("Insufficient trading dates found for volume comparison")
                return []
            
            latest_date = date_rows[0][0]
            prev_date = date_rows[1][0]
            
            logger.info(f"Comparing volume: Latest date = {latest_date}, Previous date = {prev_date}")
            
            # Get stocks with increased volume using the same logic as user's query
            cursor.execute("""
                SELECT 
                    curr.scrip_id,
                    curr.price_close as current_price,
                    curr.volume as current_volume,
                    prev.volume as previous_volume,
                    100.0 * (curr.volume - prev.volume) / NULLIF(prev.volume, 0) as percent_volume_increase
                FROM (
                    SELECT scrip_id, price_close, volume
                    FROM my_schema.rt_intraday_price
                    WHERE price_date::date = %s
                    AND volume IS NOT NULL
                    AND volume > 0
                ) curr
                INNER JOIN (
                    SELECT scrip_id, volume
                    FROM my_schema.rt_intraday_price
                    WHERE price_date::date = %s
                    AND volume IS NOT NULL
                    AND volume > 0
                ) prev ON curr.scrip_id = prev.scrip_id
                INNER JOIN my_schema.master_scrips s ON curr.scrip_id = s.scrip_id
                WHERE curr.volume > prev.volume
                AND prev.volume > 0
                AND (s.scrip_group IS NULL OR UPPER(s.scrip_group) != 'INDEX')
                AND (s.sector_code IS NULL OR UPPER(s.sector_code) != 'INDEX')
                AND 100.0 * (curr.volume - prev.volume) / prev.volume >= %s
                ORDER BY percent_volume_increase DESC
            """, (latest_date, prev_date, min_volume_increase_pct))
            
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                results.append({
                    'symbol': row[0],
                    'current_price': float(row[1]) if row[1] else 0,
                    'latest_volume': float(row[2]) if row[2] else 0,
                    'previous_volume': float(row[3]) if row[3] else 0,
                    'volume_increase_pct': round(float(row[4]), 2) if row[4] else 0,
                    'latest_date': str(latest_date),
                    'previous_date': str(prev_date)
                })
            
            cursor.close()
            conn.close()
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting stocks with increased volume: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def get_stocks_with_increased_obv(self, min_obv_increase_pct: float = 20.0,
                                      lookback_days: int = 10) -> List[Dict]:
        """
        Get stocks with increased On-Balance Volume (OBV)
        
        Args:
            min_obv_increase_pct: Minimum percentage increase in OBV (default: 20%)
            lookback_days: Number of days to analyze (default: 10)
            
        Returns:
            List of stocks with OBV increases
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get all stocks from master_scrips, excluding indices
            cursor.execute("""
                SELECT DISTINCT s.scrip_id
                FROM my_schema.master_scrips s
                WHERE s.scrip_id IS NOT NULL
                AND (s.scrip_group IS NULL OR UPPER(s.scrip_group) != 'INDEX')
                AND (s.sector_code IS NULL OR UPPER(s.sector_code) != 'INDEX')
                AND EXISTS (
                    SELECT 1 
                    FROM my_schema.rt_intraday_price p
                    WHERE p.scrip_id = s.scrip_id
                    AND p.volume IS NOT NULL
                    AND p.volume > 0
                    AND p.price_date::date >= CURRENT_DATE - INTERVAL '%s days'
                )
            """, (lookback_days + 5,))
            
            stock_symbols = [row[0] for row in cursor.fetchall()]
            
            if not stock_symbols:
                cursor.close()
                conn.close()
                return []
            
            results = []
            
            for symbol in stock_symbols:
                try:
                    # Get price and volume data - use proper date casting
                    cursor.execute("""
                        SELECT 
                            price_date,
                            price_close,
                            volume
                        FROM my_schema.rt_intraday_price
                        WHERE scrip_id = %s
                        AND price_date::date >= CURRENT_DATE - INTERVAL '%s days'
                        ORDER BY price_date::date ASC
                    """, (symbol, lookback_days + 5))
                    
                    price_data = cursor.fetchall()
                    
                    if len(price_data) < lookback_days + 1:
                        continue
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(price_data, columns=['price_date', 'price_close', 'volume'])
                    df = df.sort_values('price_date')
                    
                    # Calculate OBV
                    closes = df['price_close'].astype(float)
                    volumes = df['volume'].astype(float)
                    
                    obv = calculate_obv(closes, volumes)
                    
                    if len(obv) < 2:
                        continue
                    
                    # Get latest OBV and OBV from lookback_days ago
                    latest_obv = float(obv.iloc[-1])
                    previous_obv = float(obv.iloc[-lookback_days]) if len(obv) >= lookback_days else float(obv.iloc[0])
                    
                    if previous_obv != 0:
                        obv_increase_pct = ((latest_obv - previous_obv) / abs(previous_obv)) * 100
                        
                        if obv_increase_pct >= min_obv_increase_pct:
                            results.append({
                                'symbol': symbol,
                                'latest_obv': latest_obv,
                                'previous_obv': previous_obv,
                                'obv_increase_pct': round(obv_increase_pct, 2),
                                'current_price': float(df.iloc[-1]['price_close']) if df.iloc[-1]['price_close'] else 0
                            })
                except Exception as e:
                    logger.debug(f"Error processing {symbol} for OBV increase: {e}")
                    continue
            
            cursor.close()
            conn.close()
            
            # Sort by OBV increase percentage (descending)
            results.sort(key=lambda x: x['obv_increase_pct'], reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting stocks with increased OBV: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def get_stocks_with_supertrend_change(self, lookback_days: int = 5) -> List[Dict]:
        """
        Get stocks with recent supertrend changes (bullish to bearish or vice versa)
        
        Args:
            lookback_days: Number of days to check for changes (default: 5)
            
        Returns:
            List of stocks with supertrend changes
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get all stocks from master_scrips, excluding indices
            cursor.execute("""
                SELECT DISTINCT s.scrip_id
                FROM my_schema.master_scrips s
                WHERE s.scrip_id IS NOT NULL
                AND (s.scrip_group IS NULL OR UPPER(s.scrip_group) != 'INDEX')
                AND (s.sector_code IS NULL OR UPPER(s.sector_code) != 'INDEX')
                AND EXISTS (
                    SELECT 1 
                    FROM my_schema.supertrend_values st
                        WHERE st.scrip_id = s.scrip_id
                        AND st.calculation_date::date >= CURRENT_DATE - INTERVAL '%s days'
                )
            """, (lookback_days + 2,))
            
            stock_symbols = [row[0] for row in cursor.fetchall()]
            
            if not stock_symbols:
                cursor.close()
                conn.close()
                return []
            
            results = []
            
            for symbol in stock_symbols:
                try:
                    # Get supertrend history
                    cursor.execute("""
                        SELECT 
                            calculation_date,
                            supertrend_value,
                            supertrend_direction
                        FROM my_schema.supertrend_values
                        WHERE scrip_id = %s
                        AND calculation_date >= CURRENT_DATE - INTERVAL '%s days'
                        ORDER BY calculation_date DESC
                        LIMIT %s
                    """, (symbol, lookback_days + 2, lookback_days + 2))
                    
                    supertrend_data = cursor.fetchall()
                    
                    if len(supertrend_data) < 2:
                        continue
                    
                    # Check for direction change
                    # supertrend_direction is stored as integer: 1 = BULLISH, -1 = BEARISH
                    latest_direction = supertrend_data[0][2]  # supertrend_direction
                    previous_direction = supertrend_data[-1][2] if len(supertrend_data) > 1 else None
                    
                    # Convert integer to string for comparison
                    latest_dir_str = 'BULLISH' if (latest_direction == 1 or latest_direction == 'BULLISH' or str(latest_direction) == '1') else 'BEARISH'
                    prev_dir_str = 'BULLISH' if (previous_direction == 1 or previous_direction == 'BULLISH' or str(previous_direction) == '1') else 'BEARISH' if previous_direction is not None else None
                    
                    if previous_direction is not None and latest_dir_str != prev_dir_str:
                        # Direction changed
                        change_type = latest_dir_str
                        
                        # Get current price
                        cursor.execute("""
                            SELECT price_close
                            FROM my_schema.rt_intraday_price
                            WHERE scrip_id = %s
                            ORDER BY price_date::date DESC
                            LIMIT 1
                        """, (symbol,))
                        price_row = cursor.fetchone()
                        current_price = float(price_row[0]) if price_row and price_row[0] else 0
                        
                        results.append({
                            'symbol': symbol,
                            'change_type': change_type,
                            'previous_direction': prev_dir_str,
                            'current_direction': latest_dir_str,
                            'supertrend_value': float(supertrend_data[0][1]) if supertrend_data[0][1] else 0,
                            'current_price': current_price,
                            'change_date': supertrend_data[0][0].strftime('%Y-%m-%d') if hasattr(supertrend_data[0][0], 'strftime') else str(supertrend_data[0][0])
                        })
                except Exception as e:
                    logger.debug(f"Error processing {symbol} for supertrend change: {e}")
                    continue
            
            cursor.close()
            conn.close()
            
            # Sort by change date (most recent first)
            results.sort(key=lambda x: x.get('change_date', ''), reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting stocks with supertrend change: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def get_stocks_with_golden_cross(self, fast_ma: int = 50, slow_ma: int = 200,
                                     lookback_days: int = 5) -> List[Dict]:
        """
        Get stocks with golden cross (fast MA crossing above slow MA)
        
        Args:
            fast_ma: Fast moving average period (default: 50)
            slow_ma: Slow moving average period (default: 200)
            lookback_days: Number of days to check for cross (default: 5)
            
        Returns:
            List of stocks with golden cross
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get all stocks from master_scrips, excluding indices
            cursor.execute("""
                SELECT DISTINCT s.scrip_id
                FROM my_schema.master_scrips s
                WHERE s.scrip_id IS NOT NULL
                AND (s.scrip_group IS NULL OR UPPER(s.scrip_group) != 'INDEX')
                AND (s.sector_code IS NULL OR UPPER(s.sector_code) != 'INDEX')
                AND EXISTS (
                    SELECT 1 
                    FROM my_schema.rt_intraday_price p
                    WHERE p.scrip_id = s.scrip_id
                    AND p.price_close IS NOT NULL
                    AND p.price_date::date >= CURRENT_DATE - INTERVAL '%s days'
                )
            """, (slow_ma + lookback_days + 5,))
            
            stock_symbols = [row[0] for row in cursor.fetchall()]
            
            if not stock_symbols:
                cursor.close()
                conn.close()
                return []
            
            results = []
            
            for symbol in stock_symbols:
                try:
                    # Get price data
                    cursor.execute("""
                        SELECT 
                            price_date,
                            price_close
                        FROM my_schema.rt_intraday_price
                        WHERE scrip_id = %s
                        AND price_date::date >= CURRENT_DATE - INTERVAL '%s days'
                        ORDER BY price_date::date ASC
                    """, (symbol, slow_ma + lookback_days + 5))
                    
                    price_data = cursor.fetchall()
                    
                    if len(price_data) < slow_ma + lookback_days:
                        continue
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(price_data, columns=['price_date', 'price_close'])
                    df = df.sort_values('price_date')
                    
                    closes = df['price_close'].astype(float)
                    
                    # Calculate moving averages
                    fast_ma_values = calculate_sma(closes, fast_ma)
                    slow_ma_values = calculate_sma(closes, slow_ma)
                    
                    if len(fast_ma_values) < lookback_days + 1 or len(slow_ma_values) < lookback_days + 1:
                        continue
                    
                    # Check for golden cross in recent days
                    for i in range(len(fast_ma_values) - lookback_days, len(fast_ma_values)):
                        if i < 1:
                            continue
                        
                        # Check if fast MA crossed above slow MA
                        prev_fast = fast_ma_values.iloc[i-1]
                        curr_fast = fast_ma_values.iloc[i]
                        prev_slow = slow_ma_values.iloc[i-1]
                        curr_slow = slow_ma_values.iloc[i]
                        
                        # Golden cross: fast MA was below slow MA, now above
                        if not pd.isna(prev_fast) and not pd.isna(curr_fast) and \
                           not pd.isna(prev_slow) and not pd.isna(curr_slow):
                            if prev_fast <= prev_slow and curr_fast > curr_slow:
                                # Golden cross detected
                                cross_date = df.iloc[i]['price_date']
                                
                                results.append({
                                    'symbol': symbol,
                                    'fast_ma': fast_ma,
                                    'slow_ma': slow_ma,
                                    'fast_ma_value': round(float(curr_fast), 2),
                                    'slow_ma_value': round(float(curr_slow), 2),
                                    'current_price': round(float(closes.iloc[i]), 2),
                                    'cross_date': cross_date.strftime('%Y-%m-%d') if hasattr(cross_date, 'strftime') else str(cross_date)
                                })
                                break  # Only report first cross
                except Exception as e:
                    logger.debug(f"Error processing {symbol} for golden cross: {e}")
                    continue
            
            cursor.close()
            conn.close()
            
            # Sort by cross date (most recent first)
            results.sort(key=lambda x: x.get('cross_date', ''), reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting stocks with golden cross: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def get_stocks_with_rsi_signals(self, rsi_period: int = 14,
                                    oversold_threshold: float = 30.0,
                                    overbought_threshold: float = 70.0,
                                    lookback_days: int = 10) -> List[Dict]:
        """
        Get stocks with RSI signals (oversold or overbought conditions)
        
        Args:
            rsi_period: RSI calculation period (default: 14)
            oversold_threshold: RSI below this is oversold (default: 30)
            overbought_threshold: RSI above this is overbought (default: 70)
            lookback_days: Number of days to analyze (default: 10)
            
        Returns:
            List of stocks with RSI signals
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get all stocks from master_scrips, excluding indices
            cursor.execute("""
                SELECT DISTINCT s.scrip_id
                FROM my_schema.master_scrips s
                WHERE s.scrip_id IS NOT NULL
                AND (s.scrip_group IS NULL OR UPPER(s.scrip_group) != 'INDEX')
                AND (s.sector_code IS NULL OR UPPER(s.sector_code) != 'INDEX')
                AND EXISTS (
                    SELECT 1 
                    FROM my_schema.rt_intraday_price p
                    WHERE p.scrip_id = s.scrip_id
                    AND p.price_close IS NOT NULL
                    AND p.price_date::date >= CURRENT_DATE - INTERVAL '%s days'
                )
            """, (lookback_days + rsi_period + 5,))
            
            stock_symbols = [row[0] for row in cursor.fetchall()]
            
            if not stock_symbols:
                cursor.close()
                conn.close()
                return []
            
            results = []
            
            for symbol in stock_symbols:
                try:
                    # Get price data
                    cursor.execute("""
                        SELECT 
                            price_date,
                            price_close
                        FROM my_schema.rt_intraday_price
                        WHERE scrip_id = %s
                        AND price_date::date >= CURRENT_DATE - INTERVAL '%s days'
                        ORDER BY price_date::date ASC
                    """, (symbol, lookback_days + rsi_period + 5))
                    
                    price_data = cursor.fetchall()
                    
                    if len(price_data) < rsi_period + 1:
                        continue
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(price_data, columns=['price_date', 'price_close'])
                    df = df.sort_values('price_date')
                    
                    closes = df['price_close'].astype(float)
                    
                    # Calculate RSI
                    rsi = calculate_rsi(closes, period=rsi_period)
                    
                    if len(rsi) == 0 or pd.isna(rsi.iloc[-1]):
                        continue
                    
                    latest_rsi = float(rsi.iloc[-1])
                    signal_type = None
                    
                    if latest_rsi <= oversold_threshold:
                        signal_type = 'OVERSOLD'
                    elif latest_rsi >= overbought_threshold:
                        signal_type = 'OVERBOUGHT'
                    
                    if signal_type:
                        results.append({
                            'symbol': symbol,
                            'rsi_value': round(latest_rsi, 2),
                            'signal_type': signal_type,
                            'current_price': round(float(closes.iloc[-1]), 2),
                            'rsi_period': rsi_period
                        })
                except Exception as e:
                    logger.debug(f"Error processing {symbol} for RSI signals: {e}")
                    continue
            
            cursor.close()
            conn.close()
            
            # Sort by RSI value (oversold first, then overbought)
            results.sort(key=lambda x: (x['signal_type'] == 'OVERBOUGHT', x['rsi_value']))
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting stocks with RSI signals: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def get_stocks_with_macd_crossover(self, fast: int = 12, slow: int = 26, signal: int = 9,
                                      lookback_days: int = 5) -> List[Dict]:
        """
        Get stocks with MACD crossover signals (bullish or bearish)
        
        Args:
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line period (default: 9)
            lookback_days: Number of days to check for crossover (default: 5)
            
        Returns:
            List of stocks with MACD crossovers
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get all stocks from master_scrips, excluding indices
            cursor.execute("""
                SELECT DISTINCT s.scrip_id
                FROM my_schema.master_scrips s
                WHERE s.scrip_id IS NOT NULL
                AND (s.scrip_group IS NULL OR UPPER(s.scrip_group) != 'INDEX')
                AND (s.sector_code IS NULL OR UPPER(s.sector_code) != 'INDEX')
                AND EXISTS (
                    SELECT 1 
                    FROM my_schema.rt_intraday_price p
                    WHERE p.scrip_id = s.scrip_id
                    AND p.price_close IS NOT NULL
                    AND p.price_date::date >= CURRENT_DATE - INTERVAL '%s days'
                )
            """, (slow + signal + lookback_days + 5,))
            
            stock_symbols = [row[0] for row in cursor.fetchall()]
            
            if not stock_symbols:
                cursor.close()
                conn.close()
                return []
            
            results = []
            
            for symbol in stock_symbols:
                try:
                    # Get price data
                    cursor.execute("""
                        SELECT 
                            price_date,
                            price_close
                        FROM my_schema.rt_intraday_price
                        WHERE scrip_id = %s
                        AND price_date::date >= CURRENT_DATE - INTERVAL '%s days'
                        ORDER BY price_date::date ASC
                    """, (symbol, slow + signal + lookback_days + 5))
                    
                    price_data = cursor.fetchall()
                    
                    if len(price_data) < slow + signal + lookback_days:
                        continue
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(price_data, columns=['price_date', 'price_close'])
                    df = df.sort_values('price_date')
                    
                    closes = df['price_close'].astype(float)
                    
                    # Calculate MACD
                    macd_data = calculate_macd(closes, fast=fast, slow=slow, signal=signal)
                    macd_line = macd_data['macd']
                    signal_line = macd_data['signal']
                    
                    if len(macd_line) < lookback_days + 1 or len(signal_line) < lookback_days + 1:
                        continue
                    
                    # Check for crossover in recent days
                    for i in range(len(macd_line) - lookback_days, len(macd_line)):
                        if i < 1:
                            continue
                        
                        prev_macd = macd_line.iloc[i-1]
                        curr_macd = macd_line.iloc[i]
                        prev_signal = signal_line.iloc[i-1]
                        curr_signal = signal_line.iloc[i]
                        
                        # Bullish crossover: MACD crosses above signal line
                        # Bearish crossover: MACD crosses below signal line
                        if not pd.isna(prev_macd) and not pd.isna(curr_macd) and \
                           not pd.isna(prev_signal) and not pd.isna(curr_signal):
                            
                            crossover_type = None
                            if prev_macd <= prev_signal and curr_macd > curr_signal:
                                crossover_type = 'BULLISH'
                            elif prev_macd >= prev_signal and curr_macd < curr_signal:
                                crossover_type = 'BEARISH'
                            
                            if crossover_type:
                                cross_date = df.iloc[i]['price_date']
                                
                                results.append({
                                    'symbol': symbol,
                                    'crossover_type': crossover_type,
                                    'macd_value': round(float(curr_macd), 2),
                                    'signal_value': round(float(curr_signal), 2),
                                    'histogram': round(float(curr_macd - curr_signal), 2),
                                    'current_price': round(float(closes.iloc[i]), 2),
                                    'cross_date': cross_date.strftime('%Y-%m-%d') if hasattr(cross_date, 'strftime') else str(cross_date)
                                })
                                break  # Only report first crossover
                except Exception as e:
                    logger.debug(f"Error processing {symbol} for MACD crossover: {e}")
                    continue
            
            cursor.close()
            conn.close()
            
            # Sort by cross date (most recent first)
            results.sort(key=lambda x: x.get('cross_date', ''), reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting stocks with MACD crossover: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def get_stocks_with_atr_expansion(self, min_atr_increase_pct: float = 30.0,
                                     atr_period: int = 14,
                                     lookback_days: int = 10) -> List[Dict]:
        """
        Get stocks with ATR expansion (increased volatility)
        
        Args:
            min_atr_increase_pct: Minimum ATR increase percentage (default: 30%)
            atr_period: ATR calculation period (default: 14)
            lookback_days: Number of days to compare (default: 10)
            
        Returns:
            List of stocks with ATR expansion
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get all stocks from master_scrips, excluding indices
            cursor.execute("""
                SELECT DISTINCT s.scrip_id
                FROM my_schema.master_scrips s
                WHERE s.scrip_id IS NOT NULL
                AND (s.scrip_group IS NULL OR UPPER(s.scrip_group) != 'INDEX')
                AND (s.sector_code IS NULL OR UPPER(s.sector_code) != 'INDEX')
                AND EXISTS (
                    SELECT 1 
                    FROM my_schema.rt_intraday_price p
                    WHERE p.scrip_id = s.scrip_id
                    AND p.price_high IS NOT NULL
                    AND p.price_low IS NOT NULL
                    AND p.price_close IS NOT NULL
                    AND p.price_date::date >= CURRENT_DATE - INTERVAL '%s days'
                )
            """, (lookback_days + atr_period + 5,))
            
            stock_symbols = [row[0] for row in cursor.fetchall()]
            
            if not stock_symbols:
                cursor.close()
                conn.close()
                return []
            
            results = []
            
            for symbol in stock_symbols:
                try:
                    # Get OHLC data
                    cursor.execute("""
                        SELECT 
                            price_date,
                            price_high,
                            price_low,
                            price_close
                        FROM my_schema.rt_intraday_price
                        WHERE scrip_id = %s
                        AND price_date::date >= CURRENT_DATE - INTERVAL '%s days'
                        ORDER BY price_date::date ASC
                    """, (symbol, lookback_days + atr_period + 5))
                    
                    ohlc_data = cursor.fetchall()
                    
                    if len(ohlc_data) < atr_period + lookback_days:
                        continue
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(ohlc_data, columns=['price_date', 'price_high', 'price_low', 'price_close'])
                    df = df.sort_values('price_date')
                    
                    highs = df['price_high'].astype(float)
                    lows = df['price_low'].astype(float)
                    closes = df['price_close'].astype(float)
                    
                    # Calculate ATR
                    atr = calculate_atr(highs, lows, closes, period=atr_period)
                    
                    if len(atr) < lookback_days + 1:
                        continue
                    
                    # Get latest ATR and average ATR from previous period
                    latest_atr = float(atr.iloc[-1])
                    previous_atr_avg = float(atr.iloc[-lookback_days:-1].mean()) if len(atr) >= lookback_days else float(atr.iloc[0])
                    
                    if previous_atr_avg > 0:
                        atr_increase_pct = ((latest_atr - previous_atr_avg) / previous_atr_avg) * 100
                        
                        if atr_increase_pct >= min_atr_increase_pct:
                            results.append({
                                'symbol': symbol,
                                'latest_atr': round(latest_atr, 2),
                                'previous_atr_avg': round(previous_atr_avg, 2),
                                'atr_increase_pct': round(atr_increase_pct, 2),
                                'current_price': round(float(closes.iloc[-1]), 2),
                                'atr_period': atr_period
                            })
                except Exception as e:
                    logger.debug(f"Error processing {symbol} for ATR expansion: {e}")
                    continue
            
            cursor.close()
            conn.close()
            
            # Sort by ATR increase percentage (descending)
            results.sort(key=lambda x: x['atr_increase_pct'], reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting stocks with ATR expansion: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def get_stocks_with_volume_trend(self, trend_type: str = 'increasing',
                                     lookback_days: int = 10) -> List[Dict]:
        """
        Get stocks with specific volume trend (increasing, decreasing, or stable)
        
        Args:
            trend_type: Type of trend to find ('increasing', 'decreasing', or 'stable')
            lookback_days: Number of days to analyze (default: 10)
            
        Returns:
            List of stocks with the specified volume trend
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get all stocks from master_scrips, excluding indices
            cursor.execute("""
                SELECT DISTINCT s.scrip_id
                FROM my_schema.master_scrips s
                WHERE s.scrip_id IS NOT NULL
                AND (s.scrip_group IS NULL OR UPPER(s.scrip_group) != 'INDEX')
                AND (s.sector_code IS NULL OR UPPER(s.sector_code) != 'INDEX')
                AND EXISTS (
                    SELECT 1 
                    FROM my_schema.rt_intraday_price p
                    WHERE p.scrip_id = s.scrip_id
                    AND p.volume IS NOT NULL
                    AND p.volume > 0
                    AND p.price_date::date >= CURRENT_DATE - INTERVAL '%s days'
                )
            """, (lookback_days + 5,))
            
            stock_symbols = [row[0] for row in cursor.fetchall()]
            
            if not stock_symbols:
                cursor.close()
                conn.close()
                return []
            
            results = []
            
            for symbol in stock_symbols:
                try:
                    # Get volume data
                    cursor.execute("""
                        SELECT 
                            price_date,
                            volume,
                            price_close
                        FROM my_schema.rt_intraday_price
                        WHERE scrip_id = %s
                        AND price_date::date >= CURRENT_DATE - INTERVAL '%s days'
                        ORDER BY price_date::date ASC
                    """, (symbol, lookback_days + 5))
                    
                    volume_data = cursor.fetchall()
                    
                    if len(volume_data) < lookback_days:
                        continue
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(volume_data, columns=['price_date', 'volume', 'price_close'])
                    df = df.sort_values('price_date')
                    
                    volumes = df['volume'].astype(float)
                    
                    # Calculate volume trend
                    detected_trend = calculate_volume_trend(volumes, period=lookback_days)
                    
                    if detected_trend == trend_type:
                        # Calculate trend strength
                        recent_volumes = volumes.tail(5)
                        older_volumes = volumes.head(5) if len(volumes) >= 10 else volumes.head(len(volumes) // 2)
                        
                        recent_avg = recent_volumes.mean()
                        older_avg = older_volumes.mean() if len(older_volumes) > 0 else recent_avg
                        
                        trend_strength_pct = ((recent_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0
                        
                        results.append({
                            'symbol': symbol,
                            'trend_type': detected_trend,
                            'trend_strength_pct': round(trend_strength_pct, 2),
                            'recent_volume_avg': round(recent_avg, 0),
                            'older_volume_avg': round(older_avg, 0),
                            'current_price': round(float(df.iloc[-1]['price_close']) if df.iloc[-1]['price_close'] else 0, 2)
                        })
                except Exception as e:
                    logger.debug(f"Error processing {symbol} for volume trend: {e}")
                    continue
            
            cursor.close()
            conn.close()
            
            # Sort by trend strength (descending for increasing, ascending for decreasing)
            if trend_type == 'increasing':
                results.sort(key=lambda x: x['trend_strength_pct'], reverse=True)
            elif trend_type == 'decreasing':
                results.sort(key=lambda x: x['trend_strength_pct'], reverse=False)
            else:
                results.sort(key=lambda x: abs(x['trend_strength_pct']), reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting stocks with volume trend: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def get_stocks_with_momentum_change(self, min_momentum_change_pct: float = 20.0,
                                       momentum_period: int = 14,
                                       lookback_days: int = 10) -> List[Dict]:
        """
        Get stocks with significant momentum changes
        
        Args:
            min_momentum_change_pct: Minimum momentum change percentage (default: 20%)
            momentum_period: Momentum calculation period (default: 14)
            lookback_days: Number of days to compare (default: 10)
            
        Returns:
            List of stocks with momentum changes
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get all stocks from master_scrips, excluding indices
            cursor.execute("""
                SELECT DISTINCT s.scrip_id
                FROM my_schema.master_scrips s
                WHERE s.scrip_id IS NOT NULL
                AND (s.scrip_group IS NULL OR UPPER(s.scrip_group) != 'INDEX')
                AND (s.sector_code IS NULL OR UPPER(s.sector_code) != 'INDEX')
                AND EXISTS (
                    SELECT 1 
                    FROM my_schema.rt_intraday_price p
                    WHERE p.scrip_id = s.scrip_id
                    AND p.price_close IS NOT NULL
                    AND p.price_date::date >= CURRENT_DATE - INTERVAL '%s days'
                )
            """, (momentum_period + lookback_days + 5,))
            
            stock_symbols = [row[0] for row in cursor.fetchall()]
            
            if not stock_symbols:
                cursor.close()
                conn.close()
                return []
            
            results = []
            
            for symbol in stock_symbols:
                try:
                    # Get price data
                    cursor.execute("""
                        SELECT 
                            price_date,
                            price_close
                        FROM my_schema.rt_intraday_price
                        WHERE scrip_id = %s
                        AND price_date::date >= CURRENT_DATE - INTERVAL '%s days'
                        ORDER BY price_date::date ASC
                    """, (symbol, momentum_period + lookback_days + 5))
                    
                    price_data = cursor.fetchall()
                    
                    if len(price_data) < momentum_period + lookback_days:
                        continue
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(price_data, columns=['price_date', 'price_close'])
                    df = df.sort_values('price_date')
                    
                    closes = df['price_close'].astype(float)
                    
                    # Calculate momentum
                    momentum = calculate_momentum(closes, period=momentum_period)
                    
                    if len(momentum) < lookback_days + 1:
                        continue
                    
                    # Get latest momentum and momentum from lookback_days ago
                    latest_momentum = float(momentum.iloc[-1]) if not pd.isna(momentum.iloc[-1]) else 0
                    previous_momentum = float(momentum.iloc[-lookback_days]) if len(momentum) >= lookback_days and not pd.isna(momentum.iloc[-lookback_days]) else 0
                    
                    if previous_momentum != 0:
                        momentum_change_pct = ((latest_momentum - previous_momentum) / abs(previous_momentum)) * 100
                        
                        if abs(momentum_change_pct) >= min_momentum_change_pct:
                            change_type = 'ACCELERATING' if momentum_change_pct > 0 else 'DECELERATING'
                            
                            results.append({
                                'symbol': symbol,
                                'change_type': change_type,
                                'latest_momentum': round(latest_momentum, 2),
                                'previous_momentum': round(previous_momentum, 2),
                                'momentum_change_pct': round(momentum_change_pct, 2),
                                'current_price': round(float(closes.iloc[-1]), 2),
                                'momentum_period': momentum_period
                            })
                except Exception as e:
                    logger.debug(f"Error processing {symbol} for momentum change: {e}")
                    continue
            
            cursor.close()
            conn.close()
            
            # Sort by absolute momentum change (descending)
            results.sort(key=lambda x: abs(x['momentum_change_pct']), reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting stocks with momentum change: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def get_stocks_with_rsi_divergence(self, rsi_period: int = 14,
                                       lookback_days: int = 20,
                                       min_divergence_strength: float = 0.05) -> List[Dict]:
        """
        Get stocks with RSI divergence patterns
        
        Bullish divergence: Price makes lower lows, but RSI makes higher lows (potential reversal up)
        Bearish divergence: Price makes higher highs, but RSI makes lower highs (potential reversal down)
        
        Args:
            rsi_period: RSI calculation period (default: 14)
            lookback_days: Number of days to analyze (default: 20)
            min_divergence_strength: Minimum percentage difference to consider divergence (default: 5%)
            
        Returns:
            List of stocks with RSI divergence
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get all stocks from master_scrips, excluding indices
            cursor.execute("""
                SELECT DISTINCT s.scrip_id
                FROM my_schema.master_scrips s
                WHERE s.scrip_id IS NOT NULL
                AND (s.scrip_group IS NULL OR UPPER(s.scrip_group) != 'INDEX')
                AND (s.sector_code IS NULL OR UPPER(s.sector_code) != 'INDEX')
                AND EXISTS (
                    SELECT 1 
                    FROM my_schema.rt_intraday_price p
                    WHERE p.scrip_id = s.scrip_id
                    AND p.price_close IS NOT NULL
                    AND p.price_date::date >= CURRENT_DATE - INTERVAL '%s days'
                )
            """, (lookback_days + rsi_period + 10,))
            
            stock_symbols = [row[0] for row in cursor.fetchall()]
            
            if not stock_symbols:
                cursor.close()
                conn.close()
                return []
            
            results = []
            
            for symbol in stock_symbols:
                try:
                    # Get price data with high and low
                    cursor.execute("""
                        SELECT 
                            price_date,
                            price_close,
                            price_high,
                            price_low
                        FROM my_schema.rt_intraday_price
                        WHERE scrip_id = %s
                        AND price_date::date >= CURRENT_DATE - INTERVAL '%s days'
                        ORDER BY price_date::date ASC
                    """, (symbol, lookback_days + rsi_period + 10))
                    
                    price_data = cursor.fetchall()
                    
                    if len(price_data) < rsi_period + 10:
                        continue
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(price_data, columns=['price_date', 'price_close', 'price_high', 'price_low'])
                    df = df.sort_values('price_date')
                    
                    closes = df['price_close'].astype(float)
                    highs = df['price_high'].astype(float)
                    lows = df['price_low'].astype(float)
                    
                    # Calculate RSI
                    rsi = calculate_rsi(closes, period=rsi_period)
                    
                    if len(rsi) < 10 or rsi.isna().sum() > len(rsi) * 0.5:
                        continue
                    
                    # Get recent data (last lookback_days)
                    recent_prices = closes.tail(lookback_days)
                    recent_rsi = rsi.tail(lookback_days)
                    recent_highs = highs.tail(lookback_days)
                    recent_lows = lows.tail(lookback_days)
                    
                    # Find local peaks and troughs for price and RSI
                    # For bearish divergence: look for price peaks and RSI peaks
                    # For bullish divergence: look for price troughs and RSI troughs
                    
                    divergence_type = None
                    divergence_strength = 0.0
                    
                    # Check for bearish divergence (price higher highs, RSI lower highs)
                    if len(recent_prices) >= 5:
                        # Find the two most recent significant peaks in price
                        price_peaks = []
                        rsi_peaks = []
                        
                        for i in range(2, len(recent_prices) - 2):
                            # Price peak: higher than neighbors
                            if (recent_highs.iloc[i] >= recent_highs.iloc[i-1] and 
                                recent_highs.iloc[i] >= recent_highs.iloc[i-2] and
                                recent_highs.iloc[i] >= recent_highs.iloc[i+1] and
                                recent_highs.iloc[i] >= recent_highs.iloc[i+2]):
                                price_peaks.append((i, recent_highs.iloc[i]))
                            
                            # RSI peak: higher than neighbors
                            if (recent_rsi.iloc[i] >= recent_rsi.iloc[i-1] and 
                                recent_rsi.iloc[i] >= recent_rsi.iloc[i-2] and
                                recent_rsi.iloc[i] >= recent_rsi.iloc[i+1] and
                                recent_rsi.iloc[i] >= recent_rsi.iloc[i+2]):
                                rsi_peaks.append((i, recent_rsi.iloc[i]))
                        
                        # Check for bearish divergence: price makes higher high, RSI makes lower high
                        if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
                            # Get last two peaks
                            last_price_peak = price_peaks[-1]
                            prev_price_peak = price_peaks[-2]
                            last_rsi_peak = rsi_peaks[-1]
                            prev_rsi_peak = rsi_peaks[-2]
                            
                            # Check if peaks are close in time (within 10 periods)
                            if (abs(last_price_peak[0] - last_rsi_peak[0]) <= 3 and
                                abs(prev_price_peak[0] - prev_rsi_peak[0]) <= 3):
                                
                                price_change = (last_price_peak[1] - prev_price_peak[1]) / prev_price_peak[1]
                                rsi_change = (last_rsi_peak[1] - prev_rsi_peak[1]) / prev_rsi_peak[1]
                                
                                # Bearish divergence: price up, RSI down
                                if price_change > min_divergence_strength and rsi_change < -min_divergence_strength:
                                    divergence_type = 'BEARISH'
                                    divergence_strength = abs(price_change) + abs(rsi_change)
                    
                    # Check for bullish divergence (price lower lows, RSI higher lows)
                    if divergence_type is None and len(recent_prices) >= 5:
                        # Find the two most recent significant troughs in price
                        price_troughs = []
                        rsi_troughs = []
                        
                        for i in range(2, len(recent_prices) - 2):
                            # Price trough: lower than neighbors
                            if (recent_lows.iloc[i] <= recent_lows.iloc[i-1] and 
                                recent_lows.iloc[i] <= recent_lows.iloc[i-2] and
                                recent_lows.iloc[i] <= recent_lows.iloc[i+1] and
                                recent_lows.iloc[i] <= recent_lows.iloc[i+2]):
                                price_troughs.append((i, recent_lows.iloc[i]))
                            
                            # RSI trough: lower than neighbors
                            if (recent_rsi.iloc[i] <= recent_rsi.iloc[i-1] and 
                                recent_rsi.iloc[i] <= recent_rsi.iloc[i-2] and
                                recent_rsi.iloc[i] <= recent_rsi.iloc[i+1] and
                                recent_rsi.iloc[i] <= recent_rsi.iloc[i+2]):
                                rsi_troughs.append((i, recent_rsi.iloc[i]))
                        
                        # Check for bullish divergence: price makes lower low, RSI makes higher low
                        if len(price_troughs) >= 2 and len(rsi_troughs) >= 2:
                            # Get last two troughs
                            last_price_trough = price_troughs[-1]
                            prev_price_trough = price_troughs[-2]
                            last_rsi_trough = rsi_troughs[-1]
                            prev_rsi_trough = rsi_troughs[-2]
                            
                            # Check if troughs are close in time (within 3 periods)
                            if (abs(last_price_trough[0] - last_rsi_trough[0]) <= 3 and
                                abs(prev_price_trough[0] - prev_rsi_trough[0]) <= 3):
                                
                                price_change = (last_price_trough[1] - prev_price_trough[1]) / prev_price_trough[1]
                                rsi_change = (last_rsi_trough[1] - prev_rsi_trough[1]) / prev_rsi_trough[1]
                                
                                # Bullish divergence: price down, RSI up
                                if price_change < -min_divergence_strength and rsi_change > min_divergence_strength:
                                    divergence_type = 'BULLISH'
                                    divergence_strength = abs(price_change) + abs(rsi_change)
                    
                    if divergence_type:
                        results.append({
                            'symbol': symbol,
                            'divergence_type': divergence_type,
                            'divergence_strength': round(divergence_strength * 100, 2),  # Convert to percentage
                            'current_price': round(float(closes.iloc[-1]), 2),
                            'current_rsi': round(float(recent_rsi.iloc[-1]), 2),
                            'rsi_period': rsi_period
                        })
                except Exception as e:
                    logger.debug(f"Error processing {symbol} for RSI divergence: {e}")
                    continue
            
            cursor.close()
            conn.close()
            
            # Sort by divergence strength (descending)
            results.sort(key=lambda x: x['divergence_strength'], reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting stocks with RSI divergence: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def get_all_alternate_data(self, volume_threshold: float = 50.0,
                               obv_threshold: float = 20.0,
                               lookback_days: int = 10) -> Dict:
        """
        Get all alternate data indicators
        
        Args:
            volume_threshold: Minimum volume increase percentage
            obv_threshold: Minimum OBV increase percentage
            lookback_days: Number of days to analyze
            
        Returns:
            Dictionary with all alternate data
        """
        try:
            return {
                'increased_volume': self.get_stocks_with_increased_volume(
                    min_volume_increase_pct=volume_threshold,
                    lookback_days=5
                ),
                'increased_obv': self.get_stocks_with_increased_obv(
                    min_obv_increase_pct=obv_threshold,
                    lookback_days=lookback_days
                ),
                'supertrend_change': self.get_stocks_with_supertrend_change(
                    lookback_days=5
                ),
                'golden_cross': self.get_stocks_with_golden_cross(
                    fast_ma=50,
                    slow_ma=200,
                    lookback_days=5
                ),
                'rsi_signals': self.get_stocks_with_rsi_signals(
                    rsi_period=14,
                    oversold_threshold=30.0,
                    overbought_threshold=70.0,
                    lookback_days=lookback_days
                ),
                'macd_crossover': self.get_stocks_with_macd_crossover(
                    fast=12,
                    slow=26,
                    signal=9,
                    lookback_days=5
                ),
                'atr_expansion': self.get_stocks_with_atr_expansion(
                    min_atr_increase_pct=30.0,
                    atr_period=14,
                    lookback_days=lookback_days
                ),
                'volume_trend_increasing': self.get_stocks_with_volume_trend(
                    trend_type='increasing',
                    lookback_days=lookback_days
                ),
                'momentum_change': self.get_stocks_with_momentum_change(
                    min_momentum_change_pct=20.0,
                    momentum_period=14,
                    lookback_days=lookback_days
                ),
                'rsi_divergence': self.get_stocks_with_rsi_divergence(
                    rsi_period=14,
                    lookback_days=20,
                    min_divergence_strength=0.05
                )
            }
        except Exception as e:
            logger.error(f"Error getting all alternate data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'increased_volume': [],
                'increased_obv': [],
                'supertrend_change': [],
                'golden_cross': [],
                'rsi_signals': [],
                'macd_crossover': [],
                'atr_expansion': [],
                'volume_trend_increasing': [],
                'momentum_change': [],
                'rsi_divergence': []
            }

