"""
Order Flow Analyzer
Detects trapped sellers/buyers, volume imbalances, and order flow exhaustion patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from common.Boilerplate import get_db_connection


class OrderFlowAnalyzer:
    """
    Analyze order flow patterns to detect trapped traders and exhaustion signals
    """
    
    def __init__(self, instrument_token: int = 12683010):
        """
        Initialize Order Flow Analyzer
        
        Args:
            instrument_token: Instrument token (default: 12683010 for Nifty 50 Futures)
        """
        self.instrument_token = instrument_token
    
    def analyze_order_flow(self,
                          start_time: str,
                          end_time: str,
                          analysis_date: Optional[date] = None,
                          lookback_periods: int = 20) -> Dict:
        """
        Analyze order flow for trapped traders and exhaustion patterns
        
        Args:
            start_time: Start time in HH:MM:SS format
            end_time: End time in HH:MM:SS format
            analysis_date: Analysis date (defaults to today)
            lookback_periods: Number of periods to look back for patterns
            
        Returns:
            Dictionary with order flow analysis results
        """
        if analysis_date is None:
            analysis_date = date.today()
        
        try:
            # Fetch tick data
            ticks_data = self._fetch_orderflow_data(
                start_time, end_time, analysis_date
            )
            
            if ticks_data.empty:
                return {
                    'success': False,
                    'message': 'No tick data available',
                    'missing_data': [
                        'futures_ticks data for the specified date range',
                        'buy_quantity and sell_quantity data',
                        'volume and OI data',
                        'timestamp data within the time range'
                    ]
                }
            
            # Detect trapped traders
            trapped_traders = self._detect_trapped_traders(ticks_data)
            
            # Detect volume divergences
            volume_divergences = self._detect_volume_divergences(ticks_data, lookback_periods)
            
            # Detect exhaustion patterns
            exhaustion_signals = self._detect_exhaustion_patterns(ticks_data)
            
            # Analyze buy/sell pressure
            pressure_analysis = self._analyze_pressure(ticks_data)
            
            # Identify absorption zones
            absorption_zones = self._identify_absorption_zones(ticks_data)
            
            return {
                'success': True,
                'analysis_date': analysis_date.strftime('%Y-%m-%d'),
                'time_range': f"{start_time} - {end_time}",
                'instrument_token': self.instrument_token,
                'trapped_traders': trapped_traders,
                'volume_divergences': volume_divergences,
                'exhaustion_signals': exhaustion_signals,
                'pressure_analysis': pressure_analysis,
                'absorption_zones': absorption_zones,
                'overall_sentiment': self._determine_overall_sentiment(
                    trapped_traders, volume_divergences, exhaustion_signals, pressure_analysis
                )
            }
            
        except Exception as e:
            logging.error(f"Error analyzing order flow: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'missing_data': ['Error occurred during analysis - check logs']
            }
    
    def _fetch_orderflow_data(self,
                              start_time: str,
                              end_time: str,
                              analysis_date: date) -> pd.DataFrame:
        """
        Fetch order flow data from database
        For futures, we need to get the active futures contract token
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        start_datetime = f"{analysis_date} {start_time}"
        end_datetime = f"{analysis_date} {end_time}"
        
        # Try to find the active futures contract token
        # First, try with the provided instrument_token
        # If no data found, try to find any futures contract for this date
        actual_instrument_token = self.instrument_token
        logging.info(f"Attempting to fetch order flow data for instrument_token={actual_instrument_token}, date={analysis_date}, time_range={start_time}-{end_time}")
        
        # First, check if we have data for the provided token
        test_query = """
            SELECT COUNT(*) as count
            FROM my_schema.futures_ticks
            WHERE instrument_token = %s
            AND run_date = %s
            AND timestamp >= %s
            AND timestamp <= %s
        """
        cursor.execute(test_query, (actual_instrument_token, analysis_date, start_datetime, end_datetime))
        test_result = cursor.fetchone()
        has_data = test_result[0] > 0 if test_result else False
        
        if not has_data:
            # No data for provided token, try to find any futures contract for this date
            logging.info(f"No data found for instrument_token={actual_instrument_token}, searching for any futures contract for date {analysis_date}")
            
            futures_query = """
                SELECT instrument_token
                FROM (
                    SELECT instrument_token, timestamp
                    FROM my_schema.futures_ticks
                    WHERE run_date = %s
                    AND timestamp >= %s
                    AND timestamp <= %s
                    ORDER BY timestamp DESC
                    LIMIT 1
                ) AS subquery
            """
            cursor.execute(futures_query, (analysis_date, start_datetime, end_datetime))
            futures_result = cursor.fetchone()
            if futures_result:
                actual_instrument_token = futures_result[0]
                logging.info(f"Found active futures contract token: {actual_instrument_token} for date {analysis_date}")
            else:
                # Fallback: try to get any futures token for this date (without time filter)
                cursor.execute("""
                    SELECT instrument_token
                    FROM (
                        SELECT instrument_token, timestamp
                        FROM my_schema.futures_ticks
                        WHERE run_date = %s
                        ORDER BY timestamp DESC
                        LIMIT 1
                    ) AS subquery
                """, (analysis_date,))
                fallback_result = cursor.fetchone()
                if fallback_result:
                    actual_instrument_token = fallback_result[0]
                    logging.info(f"Using fallback futures contract token: {actual_instrument_token} for date {analysis_date}")
                else:
                    # Check if ANY data exists for this date
                    cursor.execute("""
                        SELECT COUNT(*) as count, MIN(timestamp) as min_ts, MAX(timestamp) as max_ts
                        FROM my_schema.futures_ticks
                        WHERE run_date = %s
                    """, (analysis_date,))
                    check_result = cursor.fetchone()
                    if check_result and check_result[0] > 0:
                        logging.warning(f"Found {check_result[0]} futures ticks for date {analysis_date}, but none in time range {start_time}-{end_time}. Min: {check_result[1]}, Max: {check_result[2]}")
                    else:
                        logging.warning(f"No futures data found for date {analysis_date} at all")
                    conn.close()
                    return pd.DataFrame()
        
        query = """
            SELECT 
                timestamp,
                last_price,
                buy_quantity,
                sell_quantity,
                volume,
                oi,
                last_quantity
            FROM my_schema.futures_ticks
            WHERE instrument_token = %s
            AND run_date = %s
            AND timestamp >= %s
            AND timestamp <= %s
            ORDER BY timestamp ASC
        """
        
        logging.info(f"Executing query with instrument_token={actual_instrument_token}, run_date={analysis_date}, start={start_datetime}, end={end_datetime}")
        cursor.execute(query, (actual_instrument_token, analysis_date, start_datetime, end_datetime))
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            logging.warning(f"Query returned no rows for instrument_token={actual_instrument_token}, date={analysis_date}, time_range={start_time}-{end_time}")
            return pd.DataFrame()
        
        logging.info(f"Query returned {len(rows)} rows for order flow analysis")
        
        df = pd.DataFrame(rows, columns=[
            'timestamp', 'last_price', 'buy_quantity', 'sell_quantity',
            'volume', 'oi', 'last_quantity'
        ])
        
        df['buy_quantity'] = df['buy_quantity'].fillna(0)
        df['sell_quantity'] = df['sell_quantity'].fillna(0)
        df['volume'] = df['volume'].fillna(0)
        
        # Calculate price change
        df['price_change'] = df['last_price'].diff()
        df['price_change_pct'] = df['last_price'].pct_change() * 100
        
        return df
    
    def _detect_trapped_traders(self, ticks_data: pd.DataFrame) -> Dict:
        """
        Detect trapped buyers or sellers
        Trapped buyers: Price breaks below recent high with high sell volume
        Trapped sellers: Price breaks above recent low with high buy volume
        """
        if len(ticks_data) < 10:
            return {'trapped_buyers': [], 'trapped_sellers': [], 'count': 0}
        
        trapped_buyers = []
        trapped_sellers = []
        
        # Look for recent highs/lows
        window = min(20, len(ticks_data))
        
        for i in range(window, len(ticks_data)):
            recent_high = ticks_data['last_price'].iloc[i-window:i].max()
            recent_low = ticks_data['last_price'].iloc[i-window:i].min()
            current_price = ticks_data['last_price'].iloc[i]
            sell_vol = ticks_data['sell_quantity'].iloc[i]
            buy_vol = ticks_data['buy_quantity'].iloc[i]
            
            # Trapped buyers: price drops below recent high with high sell volume
            if current_price < recent_high * 0.998 and sell_vol > buy_vol * 1.5:
                trapped_buyers.append({
                    'timestamp': ticks_data['timestamp'].iloc[i],
                    'price': float(current_price),
                    'recent_high': float(recent_high),
                    'sell_volume': int(sell_vol),
                    'buy_volume': int(buy_vol),
                    'severity': 'high' if sell_vol > buy_vol * 2 else 'medium'
                })
            
            # Trapped sellers: price rises above recent low with high buy volume
            if current_price > recent_low * 1.002 and buy_vol > sell_vol * 1.5:
                trapped_sellers.append({
                    'timestamp': ticks_data['timestamp'].iloc[i],
                    'price': float(current_price),
                    'recent_low': float(recent_low),
                    'buy_volume': int(buy_vol),
                    'sell_volume': int(sell_vol),
                    'severity': 'high' if buy_vol > sell_vol * 2 else 'medium'
                })
        
        return {
            'trapped_buyers': trapped_buyers[-5:] if trapped_buyers else [],  # Last 5
            'trapped_sellers': trapped_sellers[-5:] if trapped_sellers else [],  # Last 5
            'count': len(trapped_buyers) + len(trapped_sellers),
            'trapped_buyers_count': len(trapped_buyers),
            'trapped_sellers_count': len(trapped_sellers)
        }
    
    def _detect_volume_divergences(self, 
                                   ticks_data: pd.DataFrame,
                                   lookback: int) -> Dict:
        """
        Detect volume divergences (price vs volume)
        Bullish divergence: Price falling but volume decreasing (selling exhaustion)
        Bearish divergence: Price rising but volume decreasing (buying exhaustion)
        """
        if len(ticks_data) < lookback * 2:
            return {'divergences': [], 'count': 0}
        
        divergences = []
        
        for i in range(lookback, len(ticks_data)):
            # Price trend
            recent_prices = ticks_data['last_price'].iloc[i-lookback:i]
            price_trend = recent_prices.iloc[-1] - recent_prices.iloc[0]
            
            # Volume trend
            recent_volumes = ticks_data['volume'].iloc[i-lookback:i]
            volume_trend = recent_volumes.iloc[-1] - recent_volumes.iloc[0]
            
            # Divergence detection
            if price_trend < 0 and volume_trend < 0 and abs(volume_trend) > recent_volumes.mean() * 0.3:
                # Bullish divergence: price falling, volume decreasing
                divergences.append({
                    'timestamp': ticks_data['timestamp'].iloc[i],
                    'type': 'bullish_divergence',
                    'price': float(ticks_data['last_price'].iloc[i]),
                    'price_change': float(price_trend),
                    'volume_change': float(volume_trend),
                    'signal': 'potential_reversal_up'
                })
            elif price_trend > 0 and volume_trend < 0 and abs(volume_trend) > recent_volumes.mean() * 0.3:
                # Bearish divergence: price rising, volume decreasing
                divergences.append({
                    'timestamp': ticks_data['timestamp'].iloc[i],
                    'type': 'bearish_divergence',
                    'price': float(ticks_data['last_price'].iloc[i]),
                    'price_change': float(price_trend),
                    'volume_change': float(volume_trend),
                    'signal': 'potential_reversal_down'
                })
        
        return {
            'divergences': divergences[-10:] if divergences else [],  # Last 10
            'count': len(divergences),
            'bullish_divergences': len([d for d in divergences if d['type'] == 'bullish_divergence']),
            'bearish_divergences': len([d for d in divergences if d['type'] == 'bearish_divergence'])
        }
    
    def _detect_exhaustion_patterns(self, ticks_data: pd.DataFrame) -> Dict:
        """
        Detect exhaustion patterns (climactic volume spikes)
        """
        if len(ticks_data) < 10:
            return {'exhaustion_events': [], 'count': 0}
        
        exhaustion_events = []
        
        # Calculate volume moving average
        volume_ma = ticks_data['volume'].rolling(window=10).mean()
        volume_std = ticks_data['volume'].rolling(window=10).std()
        
        for i in range(10, len(ticks_data)):
            current_volume = ticks_data['volume'].iloc[i]
            avg_volume = volume_ma.iloc[i]
            std_volume = volume_std.iloc[i]
            
            # Exhaustion: volume spike > 2 standard deviations
            if current_volume > avg_volume + 2 * std_volume and not pd.isna(avg_volume):
                price_change = ticks_data['price_change_pct'].iloc[i]
                buy_vol = ticks_data['buy_quantity'].iloc[i]
                sell_vol = ticks_data['sell_quantity'].iloc[i]
                
                if price_change > 0 and buy_vol > sell_vol:
                    # Buying exhaustion: High volume buying at rising prices suggests buying is exhausted
                    # This indicates a potential top/reversal downward
                    exhaustion_events.append({
                        'timestamp': ticks_data['timestamp'].iloc[i],
                        'type': 'buying_exhaustion',  # Changed from 'bullish_exhaustion' for clarity
                        'price': float(ticks_data['last_price'].iloc[i]),
                        'volume': int(current_volume),
                        'volume_multiple': round(current_volume / avg_volume, 2) if avg_volume > 0 else 0,
                        'signal': 'potential_top',
                        'direction': 'bearish'  # Price likely to reverse down
                    })
                elif price_change < 0 and sell_vol > buy_vol:
                    # Selling exhaustion: High volume selling at falling prices suggests selling is exhausted
                    # This indicates a potential bottom/reversal upward
                    exhaustion_events.append({
                        'timestamp': ticks_data['timestamp'].iloc[i],
                        'type': 'selling_exhaustion',  # Changed from 'bearish_exhaustion' for clarity
                        'price': float(ticks_data['last_price'].iloc[i]),
                        'volume': int(current_volume),
                        'volume_multiple': round(current_volume / avg_volume, 2) if avg_volume > 0 else 0,
                        'signal': 'potential_bottom',
                        'direction': 'bullish'  # Price likely to reverse up
                    })
        
        return {
            'exhaustion_events': exhaustion_events[-10:] if exhaustion_events else [],  # Last 10
            'count': len(exhaustion_events),
            'buying_exhaustion': len([e for e in exhaustion_events if e['type'] == 'buying_exhaustion']),
            'selling_exhaustion': len([e for e in exhaustion_events if e['type'] == 'selling_exhaustion']),
            # Keep old keys for backward compatibility
            'bullish_exhaustion': len([e for e in exhaustion_events if e['type'] == 'buying_exhaustion']),
            'bearish_exhaustion': len([e for e in exhaustion_events if e['type'] == 'selling_exhaustion'])
        }
    
    def _analyze_pressure(self, ticks_data: pd.DataFrame) -> Dict:
        """
        Analyze buy/sell pressure dynamics
        """
        if len(ticks_data) < 10:
            return {'pressure_score': 0, 'pressure_direction': 'neutral'}
        
        total_buy = int(ticks_data['buy_quantity'].sum())
        total_sell = int(ticks_data['sell_quantity'].sum())
        total_volume = total_buy + total_sell
        
        if total_volume == 0:
            return {'pressure_score': 0, 'pressure_direction': 'neutral'}
        
        buy_pressure = (total_buy / total_volume) * 100
        sell_pressure = (total_sell / total_volume) * 100
        net_pressure = buy_pressure - sell_pressure
        
        # Pressure score (-100 to +100)
        pressure_score = round(net_pressure, 2)
        
        if pressure_score > 10:
            direction = 'strong_buy'
        elif pressure_score > 5:
            direction = 'buy'
        elif pressure_score < -10:
            direction = 'strong_sell'
        elif pressure_score < -5:
            direction = 'sell'
        else:
            direction = 'neutral'
        
        return {
            'pressure_score': pressure_score,
            'pressure_direction': direction,
            'buy_pressure_pct': round(buy_pressure, 2),
            'sell_pressure_pct': round(sell_pressure, 2),
            'total_buy_volume': total_buy,
            'total_sell_volume': total_sell
        }
    
    def _identify_absorption_zones(self, ticks_data: pd.DataFrame) -> List[Dict]:
        """
        Identify absorption zones (where large orders are absorbed without price movement)
        """
        if len(ticks_data) < 20:
            return []
        
        absorption_zones = []
        
        # Look for periods with high volume but low price movement
        for i in range(10, len(ticks_data)):
            recent_data = ticks_data.iloc[i-10:i]
            
            avg_volume = recent_data['volume'].mean()
            volume_std = recent_data['volume'].std()
            price_range = recent_data['last_price'].max() - recent_data['last_price'].min()
            price_change_pct = abs(recent_data['price_change_pct'].mean())
            
            # High volume but low price movement = absorption
            if recent_data['volume'].iloc[-1] > avg_volume + volume_std and price_change_pct < 0.1:
                absorption_zones.append({
                    'timestamp': ticks_data['timestamp'].iloc[i],
                    'price': float(recent_data['last_price'].mean()),
                    'price_high': float(recent_data['last_price'].max()),
                    'price_low': float(recent_data['last_price'].min()),
                    'volume': int(recent_data['volume'].sum()),
                    'type': 'absorption_zone',
                    'strength': 'strong' if recent_data['volume'].iloc[-1] > avg_volume + 2 * volume_std else 'moderate'
                })
        
        # Return last 5 absorption zones
        return absorption_zones[-5:] if absorption_zones else []
    
    def _determine_overall_sentiment(self,
                                    trapped_traders: Dict,
                                    volume_divergences: Dict,
                                    exhaustion_signals: Dict,
                                    pressure_analysis: Dict) -> Dict:
        """
        Determine overall order flow sentiment
        """
        sentiment_score = 0
        signals = []
        
        # Trapped traders contribution
        if trapped_traders.get('trapped_buyers_count', 0) > trapped_traders.get('trapped_sellers_count', 0):
            sentiment_score -= 10
            signals.append('More trapped buyers (bearish)')
        elif trapped_traders.get('trapped_sellers_count', 0) > trapped_traders.get('trapped_buyers_count', 0):
            sentiment_score += 10
            signals.append('More trapped sellers (bullish)')
        
        # Divergences contribution
        bullish_divs = volume_divergences.get('bullish_divergences', 0)
        bearish_divs = volume_divergences.get('bearish_divergences', 0)
        if bullish_divs > bearish_divs:
            sentiment_score += 15
            signals.append(f'{bullish_divs} bullish volume divergences')
        elif bearish_divs > bullish_divs:
            sentiment_score -= 15
            signals.append(f'{bearish_divs} bearish volume divergences')
        
        # Exhaustion contribution
        buying_exh = exhaustion_signals.get('buying_exhaustion', exhaustion_signals.get('bullish_exhaustion', 0))
        selling_exh = exhaustion_signals.get('selling_exhaustion', exhaustion_signals.get('bearish_exhaustion', 0))
        bullish_exh = buying_exh  # For backward compatibility
        bearish_exh = selling_exh  # For backward compatibility
        if bullish_exh > bearish_exh:
            sentiment_score -= 5
            signals.append(f'{bullish_exh} bullish exhaustion signals')
        elif bearish_exh > bullish_exh:
            sentiment_score += 5
            signals.append(f'{bearish_exh} bearish exhaustion signals')
        
        # Pressure contribution
        pressure_score = pressure_analysis.get('pressure_score', 0)
        sentiment_score += pressure_score * 0.5
        
        # Determine sentiment
        if sentiment_score > 15:
            sentiment = 'strongly_bullish'
        elif sentiment_score > 5:
            sentiment = 'bullish'
        elif sentiment_score < -15:
            sentiment = 'strongly_bearish'
        elif sentiment_score < -5:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'sentiment_score': round(sentiment_score, 2),
            'signals': signals,
            'confidence': 'high' if abs(sentiment_score) > 15 else 'medium' if abs(sentiment_score) > 5 else 'low'
        }
