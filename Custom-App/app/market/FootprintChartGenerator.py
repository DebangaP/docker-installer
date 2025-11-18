"""
Footprint Chart Generator
Analyzes volume at price levels with buy/sell breakdown
Shows order flow patterns and volume imbalances
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from common.Boilerplate import get_db_connection


class FootprintChartGenerator:
    """
    Generate footprint chart data showing volume distribution at price levels
    Footprint charts show buy vs sell volume at each price level
    """
    
    def __init__(self, instrument_token: int = 12683010, price_bucket_size: float = 5.0):
        """
        Initialize Footprint Chart Generator
        
        Args:
            instrument_token: Instrument token (default: 12683010 for Nifty 50 Futures)
            price_bucket_size: Size of price buckets for aggregation (default: 5.0)
        """
        self.instrument_token = instrument_token
        self.price_bucket_size = price_bucket_size
    
    def generate_footprint_data(self,
                                start_time: str,
                                end_time: str,
                                analysis_date: Optional[date] = None) -> Dict:
        """
        Generate footprint chart data for a given time period
        
        Args:
            start_time: Start time in HH:MM:SS format
            end_time: End time in HH:MM:SS format
            analysis_date: Analysis date (defaults to today)
            
        Returns:
            Dictionary with footprint data: price levels, buy/sell volumes, imbalances
        """
        if analysis_date is None:
            analysis_date = date.today()
        
        try:
            # Fetch tick data with buy/sell quantities
            ticks_data = self._fetch_ticks_with_orderflow(
                start_time, end_time, analysis_date
            )
            
            if ticks_data.empty:
                logging.warning(f"No tick data found for {analysis_date} {start_time} - {end_time}")
                return {
                    'success': False,
                    'message': 'No tick data available',
                    'missing_data': [
                        'futures_ticks data for the specified date range',
                        'buy_quantity and sell_quantity data',
                        'timestamp data within the time range'
                    ],
                    'footprint_levels': [],
                    'total_volume': 0,
                    'net_buy_volume': 0,
                    'net_sell_volume': 0
                }
            
            # Aggregate volume by price level
            footprint_levels = self._aggregate_by_price_level(ticks_data)
            
            # Calculate imbalances
            imbalances = self._calculate_volume_imbalances(footprint_levels)
            
            # Identify high volume nodes (HVN) and low volume nodes (LVN)
            hvn_lvn = self._identify_hvn_lvn(footprint_levels)
            
            # Calculate delta (net buying/selling pressure)
            delta = self._calculate_delta(ticks_data)
            
            return {
                'success': True,
                'analysis_date': analysis_date.strftime('%Y-%m-%d'),
                'time_range': f"{start_time} - {end_time}",
                'instrument_token': self.instrument_token,
                'footprint_levels': footprint_levels,
                'total_volume': int(ticks_data['volume'].sum()),
                'total_buy_volume': int(ticks_data['buy_quantity'].sum()) if 'buy_quantity' in ticks_data.columns else 0,
                'total_sell_volume': int(ticks_data['sell_quantity'].sum()) if 'sell_quantity' in ticks_data.columns else 0,
                'net_buy_volume': int(ticks_data['buy_quantity'].sum() - ticks_data['sell_quantity'].sum()) if 'buy_quantity' in ticks_data.columns and 'sell_quantity' in ticks_data.columns else 0,
                'imbalances': imbalances,
                'hvn_lvn': hvn_lvn,
                'delta': delta,
                'price_range': {
                    'high': float(ticks_data['last_price'].max()),
                    'low': float(ticks_data['last_price'].min()),
                    'open': float(ticks_data['last_price'].iloc[0]),
                    'close': float(ticks_data['last_price'].iloc[-1])
                }
            }
            
        except Exception as e:
            logging.error(f"Error generating footprint data: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'missing_data': ['Error occurred during analysis - check logs'],
                'footprint_levels': []
            }
    
    def _fetch_ticks_with_orderflow(self,
                                    start_time: str,
                                    end_time: str,
                                    analysis_date: date) -> pd.DataFrame:
        """
        Fetch tick data with buy/sell quantities from database
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
        logging.info(f"Attempting to fetch footprint data for instrument_token={actual_instrument_token}, date={analysis_date}, time_range={start_time}-{end_time}")
        
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
                oi
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
        
        logging.info(f"Query returned {len(rows)} rows for footprint analysis")
        
        df = pd.DataFrame(rows, columns=[
            'timestamp', 'last_price', 'buy_quantity', 'sell_quantity', 
            'volume', 'oi'
        ])
        
        # Fill missing buy/sell quantities with 0
        df['buy_quantity'] = df['buy_quantity'].fillna(0)
        df['sell_quantity'] = df['sell_quantity'].fillna(0)
        
        return df
    
    def _aggregate_by_price_level(self, ticks_data: pd.DataFrame) -> List[Dict]:
        """
        Aggregate volume by price level (bucket prices)
        """
        # Round prices to nearest bucket size
        ticks_data['price_bucket'] = (ticks_data['last_price'] / self.price_bucket_size).round() * self.price_bucket_size
        
        # Group by price bucket
        grouped = ticks_data.groupby('price_bucket').agg({
            'volume': 'sum',
            'buy_quantity': 'sum',
            'sell_quantity': 'sum'
        }).reset_index()
        
        footprint_levels = []
        for _, row in grouped.iterrows():
            price = float(row['price_bucket'])
            buy_quantity = int(row['buy_quantity']) if pd.notna(row['buy_quantity']) else 0
            sell_quantity = int(row['sell_quantity']) if pd.notna(row['sell_quantity']) else 0
            db_volume = int(row['volume']) if pd.notna(row['volume']) else 0
            
            # In footprint analysis, buy_quantity and sell_quantity represent order book volumes
            # However, for proper footprint, we need to use the actual traded volume
            # If buy_quantity + sell_quantity is close to volume, use them directly
            # Otherwise, use volume as total and distribute proportionally
            buy_sell_sum = buy_quantity + sell_quantity
            
            if buy_sell_sum > 0 and abs(buy_sell_sum - db_volume) / max(buy_sell_sum, db_volume, 1) < 0.2:
                # buy_quantity + sell_quantity is close to volume, use them directly
                total_volume = max(db_volume, buy_sell_sum)  # Use the larger value
                buy_volume = buy_quantity
                sell_volume = sell_quantity
            elif db_volume > 0:
                # Use volume as total and distribute buy/sell proportionally
                total_volume = db_volume
                if buy_sell_sum > 0:
                    # Distribute volume proportionally based on buy/sell ratio
                    buy_ratio = buy_quantity / buy_sell_sum
                    sell_ratio = sell_quantity / buy_sell_sum
                    buy_volume = int(total_volume * buy_ratio)
                    sell_volume = int(total_volume * sell_ratio)
                else:
                    # No buy/sell data, split 50/50 (neutral)
                    buy_volume = int(total_volume / 2)
                    sell_volume = int(total_volume / 2)
            else:
                # No volume data, use buy/sell quantities as is
                total_volume = buy_sell_sum
                buy_volume = buy_quantity
                sell_volume = sell_quantity
            
            net_volume = buy_volume - sell_volume
            
            footprint_levels.append({
                'price': price,
                'total_volume': total_volume,
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'net_volume': net_volume,
                'buy_percentage': (buy_volume / total_volume * 100) if total_volume > 0 else 0,
                'sell_percentage': (sell_volume / total_volume * 100) if total_volume > 0 else 0
            })
        
        # Sort by price (descending)
        footprint_levels.sort(key=lambda x: x['price'], reverse=True)
        
        return footprint_levels
    
    def _calculate_volume_imbalances(self, footprint_levels: List[Dict]) -> Dict:
        """
        Calculate volume imbalances (significant buy/sell pressure differences)
        """
        if not footprint_levels:
            return {'imbalances': [], 'total_imbalance_zones': 0}
        
        # Calculate average volume for threshold
        avg_volume = sum(level['total_volume'] for level in footprint_levels) / len(footprint_levels)
        threshold = avg_volume * 1.5  # 50% above average
        
        imbalances = []
        for level in footprint_levels:
            if level['total_volume'] > threshold:
                imbalance_ratio = abs(level['net_volume']) / level['total_volume'] if level['total_volume'] > 0 else 0
                
                # Significant imbalance if ratio > 0.3 (30% difference)
                if imbalance_ratio > 0.3:
                    imbalances.append({
                        'price': level['price'],
                        'type': 'buy_imbalance' if level['net_volume'] > 0 else 'sell_imbalance',
                        'imbalance_ratio': round(imbalance_ratio * 100, 2),
                        'buy_volume': level['buy_volume'],
                        'sell_volume': level['sell_volume'],
                        'net_volume': level['net_volume'],
                        'strength': 'strong' if imbalance_ratio > 0.5 else 'moderate'
                    })
        
        # Sort by price (descending)
        imbalances.sort(key=lambda x: x['price'], reverse=True)
        
        return {
            'imbalances': imbalances,
            'total_imbalance_zones': len(imbalances),
            'strong_imbalances': len([i for i in imbalances if i['strength'] == 'strong'])
        }
    
    def _identify_hvn_lvn(self, footprint_levels: List[Dict]) -> Dict:
        """
        Identify High Volume Nodes (HVN) and Low Volume Nodes (LVN)
        HVN: Price levels with high volume (potential support/resistance)
        LVN: Price levels with low volume (potential breakout zones)
        """
        if not footprint_levels:
            return {'hvn': [], 'lvn': []}
        
        volumes = [level['total_volume'] for level in footprint_levels]
        avg_volume = sum(volumes) / len(volumes)
        std_volume = np.std(volumes) if len(volumes) > 1 else 0
        
        hvn_threshold = avg_volume + std_volume
        lvn_threshold = avg_volume - std_volume
        
        hvn = []
        lvn = []
        
        for level in footprint_levels:
            if level['total_volume'] >= hvn_threshold:
                hvn.append({
                    'price': level['price'],
                    'volume': level['total_volume'],
                    'significance': 'high' if level['total_volume'] >= avg_volume + 2 * std_volume else 'medium'
                })
            elif level['total_volume'] <= lvn_threshold:
                lvn.append({
                    'price': level['price'],
                    'volume': level['total_volume'],
                    'significance': 'low'
                })
        
        hvn.sort(key=lambda x: x['price'], reverse=True)
        lvn.sort(key=lambda x: x['price'], reverse=True)
        
        return {
            'hvn': hvn,
            'lvn': lvn,
            'hvn_count': len(hvn),
            'lvn_count': len(lvn)
        }
    
    def _calculate_delta(self, ticks_data: pd.DataFrame) -> Dict:
        """
        Calculate delta (cumulative net buying/selling pressure)
        Delta = Cumulative sum of (buy_quantity - sell_quantity)
        """
        if 'buy_quantity' not in ticks_data.columns or 'sell_quantity' not in ticks_data.columns:
            return {
                'cumulative_delta': 0,
                'delta_trend': 'Neutral',
                'final_delta': 0
            }
        
        ticks_data['delta'] = ticks_data['buy_quantity'] - ticks_data['sell_quantity']
        ticks_data['cumulative_delta'] = ticks_data['delta'].cumsum()
        
        final_delta = int(ticks_data['cumulative_delta'].iloc[-1])
        
        # Determine trend
        if final_delta > ticks_data['cumulative_delta'].iloc[0] + ticks_data['cumulative_delta'].std():
            trend = 'Bullish'
        elif final_delta < ticks_data['cumulative_delta'].iloc[0] - ticks_data['cumulative_delta'].std():
            trend = 'Bearish'
        else:
            trend = 'Neutral'
        
        return {
            'cumulative_delta': int(final_delta),
            'delta_trend': trend,
            'final_delta': final_delta,
            'max_delta': int(ticks_data['cumulative_delta'].max()),
            'min_delta': int(ticks_data['cumulative_delta'].min())
        }
