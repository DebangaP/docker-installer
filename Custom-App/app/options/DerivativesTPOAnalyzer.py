"""
TPO Analysis Module for Derivatives Trading
Analyzes pre-market and live market TPO profiles to extract key levels for derivatives trading
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, List
import logging
from market.CalculateTPO import TPOProfile, PostgresDataFetcher
from market.MarketBiasAnalyzer import MarketBiasAnalyzer

def convert_numpy_to_native(value):
    """
    Convert NumPy types to native Python types to avoid PostgreSQL errors
    This is critical because psycopg2/PostgreSQL interprets NumPy types as schema names
    """
    if value is None:
        return None
    
    # Handle NumPy scalar types
    try:
        if hasattr(value, 'item'):  # NumPy scalars have .item() method
            return value.item()
    except (AttributeError, ValueError):
        pass
    
    # Explicit type checks
    if isinstance(value, (np.integer, np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32)):
        return int(value)
    elif isinstance(value, (np.floating, np.float64, np.float32, np.float16)):
        return float(value)
    elif isinstance(value, np.bool_):
        return bool(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, (pd.Timestamp, datetime)):
        return value
    elif isinstance(value, str) and value.startswith('np.'):
        # Fallback: if somehow a string representation of NumPy type got through
        logging.warning(f"Found string representation of NumPy type: {value}")
        return None
    else:
        # For any other type, try to convert if it looks numeric
        try:
            if isinstance(value, (int, float, bool)):
                return value
            # Try to convert if possible
            return float(value) if '.' in str(value) else int(value)
        except (ValueError, TypeError):
            return value

class DerivativesTPOAnalyzer:
    """
    Analyzes TPO profiles for derivatives trading suggestions
    """
    
    def __init__(self, db_fetcher: PostgresDataFetcher, instrument_token: int = 256265, tick_size: float = 5.0):
        """
        Initialize Derivatives TPO Analyzer
        
        Args:
            db_fetcher: Database fetcher instance
            instrument_token: Instrument token for analysis (default: 256265 for Nifty 50)
            tick_size: Price tick size for TPO calculation
        """
        self.db_fetcher = db_fetcher
        self.instrument_token = instrument_token
        self.tick_size = tick_size
        self.market_bias_analyzer = MarketBiasAnalyzer(db_fetcher, instrument_token, tick_size)
        
    def analyze_pre_market_tpo(self, analysis_date: str = None) -> Optional[Dict]:
        """
        Analyze pre-market TPO (9:05-9:15 AM)
        
        Args:
            analysis_date: Date for analysis in 'YYYY-MM-DD' format
            
        Returns:
            Dictionary containing TPO metrics or None if no data
        """
        if not analysis_date:
            analysis_date = datetime.now().strftime('%Y-%m-%d')
        
        # Fetch pre-market data
        pre_market_data = self.db_fetcher.fetch_tick_data(
            table_name='ticks',
            instrument_token=self.instrument_token,
            start_time=f'{analysis_date} 09:05:00.000',
            end_time=f'{analysis_date} 09:15:00.000'
        )
        
        if pre_market_data.empty:
            logging.warning(f"No pre-market data found for {analysis_date}")
            return None
        
        # Calculate TPO profile
        logging.info(f"Pre-market ticks fetched: {len(pre_market_data)} for {analysis_date}")
        tpo_profile = TPOProfile(tick_size=self.tick_size)
        tpo_profile.calculate_tpo(pre_market_data)
        logging.info(f"Pre-market TPO metrics: POC={tpo_profile.poc}, VAH={tpo_profile.value_area_high}, VAL={tpo_profile.value_area_low}")
        
        # Extract key metrics - ensure all values are native Python types, not NumPy
        result = {
            'analysis_date': analysis_date,
            'instrument_token': self.instrument_token,
            'session_type': 'pre_market',
            'poc': convert_numpy_to_native(tpo_profile.poc) if tpo_profile.poc else None,
            'poc_high': convert_numpy_to_native(tpo_profile.poc_high) if tpo_profile.poc_high else None,
            'poc_low': convert_numpy_to_native(tpo_profile.poc_low) if tpo_profile.poc_low else None,
            'value_area_high': convert_numpy_to_native(tpo_profile.value_area_high) if tpo_profile.value_area_high else None,
            'value_area_low': convert_numpy_to_native(tpo_profile.value_area_low) if tpo_profile.value_area_low else None,
            'initial_balance_high': convert_numpy_to_native(tpo_profile.initial_balance_high) if tpo_profile.initial_balance_high else None,
            'initial_balance_low': convert_numpy_to_native(tpo_profile.initial_balance_low) if tpo_profile.initial_balance_low else None,
            'session_range': convert_numpy_to_native(tpo_profile.value_area_high - tpo_profile.value_area_low) if (tpo_profile.value_area_high and tpo_profile.value_area_low) else None,
            'created_at': datetime.now()
        }
        
        # Calculate confidence score based on TPO profile quality and convert
        confidence_score = self._calculate_confidence_score(tpo_profile, pre_market_data)
        result['confidence_score'] = convert_numpy_to_native(confidence_score)
        
        # Ensure all values in result dict are converted (defense in depth)
        for key, value in result.items():
            if value is not None and key != 'analysis_date' and key != 'session_type' and key != 'created_at':
                result[key] = convert_numpy_to_native(value)
        
        return result
    
    def analyze_live_market_tpo(self, analysis_date: str = None, current_time: str = None) -> Optional[Dict]:
        """
        Analyze live market TPO (9:15 AM onwards)
        
        Args:
            analysis_date: Date for analysis in 'YYYY-MM-DD' format
            current_time: Current time for live analysis (optional, defaults to now)
            
        Returns:
            Dictionary containing TPO metrics or None if no data
        """
        if not analysis_date:
            analysis_date = datetime.now().strftime('%Y-%m-%d')
        
        if not current_time:
            try:
                from pytz import timezone as _tz
                current_time = datetime.now(_tz('Asia/Kolkata')).strftime('%H:%M:%S')
                logging.info(f"Current time DerivativesTPOAnalyzer:{current_time}")
            except Exception:
                current_time = datetime.now().strftime('%H:%M:%S')
        
        # Fetch live market data (9:15 AM to current time)
        live_market_data = self.db_fetcher.fetch_tick_data(
            table_name='ticks',
            instrument_token=self.instrument_token,
            start_time=f'{analysis_date} 09:15:00.000',
            end_time=f'{analysis_date} {current_time}.000'
        )
        
        if live_market_data.empty:
            logging.warning(f"No live market data found for {analysis_date}")
            return None
        
        # Calculate TPO profile
        logging.info(f"Live-market ticks fetched: {len(live_market_data)} for {analysis_date} upto {current_time}")
        tpo_profile = TPOProfile(tick_size=self.tick_size)
        tpo_profile.calculate_tpo(live_market_data)
        logging.info(f"Live TPO metrics: POC={tpo_profile.poc}, VAH={tpo_profile.value_area_high}, VAL={tpo_profile.value_area_low}")
        
        # Extract key metrics - ensure all values are native Python types, not NumPy
        result = {
            'analysis_date': analysis_date,
            'instrument_token': self.instrument_token,
            'session_type': 'live',
            'poc': convert_numpy_to_native(tpo_profile.poc) if tpo_profile.poc else None,
            'poc_high': convert_numpy_to_native(tpo_profile.poc_high) if tpo_profile.poc_high else None,
            'poc_low': convert_numpy_to_native(tpo_profile.poc_low) if tpo_profile.poc_low else None,
            'value_area_high': convert_numpy_to_native(tpo_profile.value_area_high) if tpo_profile.value_area_high else None,
            'value_area_low': convert_numpy_to_native(tpo_profile.value_area_low) if tpo_profile.value_area_low else None,
            'initial_balance_high': convert_numpy_to_native(tpo_profile.initial_balance_high) if tpo_profile.initial_balance_high else None,
            'initial_balance_low': convert_numpy_to_native(tpo_profile.initial_balance_low) if tpo_profile.initial_balance_low else None,
            'session_range': convert_numpy_to_native(tpo_profile.value_area_high - tpo_profile.value_area_low) if (tpo_profile.value_area_high and tpo_profile.value_area_low) else None,
            'created_at': datetime.now()
        }
        
        # Calculate confidence score and convert
        confidence_score = self._calculate_confidence_score(tpo_profile, live_market_data)
        result['confidence_score'] = convert_numpy_to_native(confidence_score)
        
        # Ensure all values in result dict are converted (defense in depth)
        for key, value in result.items():
            if value is not None and key != 'analysis_date' and key != 'session_type' and key != 'created_at':
                result[key] = convert_numpy_to_native(value)
        
        return result
    
    def save_tpo_analysis(self, tpo_data: Dict):
        """
        Save TPO analysis to database
        
        Args:
            tpo_data: Dictionary containing TPO analysis metrics
        """
        conn = None
        cursor = None
        try:
            from common.Boilerplate import get_db_connection
            import numpy as np
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Convert entire tpo_data dictionary first, then extract values
            converted_tpo_data = {}
            for key, value in tpo_data.items():
                converted_tpo_data[key] = convert_numpy_to_native(value)
            
            # Prepare all values and ensure they're all native types
            analysis_date = converted_tpo_data.get('analysis_date')
            instrument_token = convert_numpy_to_native(converted_tpo_data.get('instrument_token'))
            session_type = converted_tpo_data.get('session_type')
            poc = convert_numpy_to_native(converted_tpo_data.get('poc'))
            value_area_high = convert_numpy_to_native(converted_tpo_data.get('value_area_high'))
            value_area_low = convert_numpy_to_native(converted_tpo_data.get('value_area_low'))
            initial_balance_high = convert_numpy_to_native(converted_tpo_data.get('initial_balance_high'))
            initial_balance_low = convert_numpy_to_native(converted_tpo_data.get('initial_balance_low'))
            confidence_score = convert_numpy_to_native(converted_tpo_data.get('confidence_score'))
            created_at = converted_tpo_data.get('created_at')
            
            # Verify all values are native Python types (for debugging)
            values_tuple = (
                analysis_date,
                instrument_token,
                session_type,
                poc,
                value_area_high,
                value_area_low,
                initial_balance_high,
                initial_balance_low,
                confidence_score,
                created_at
            )
            
            # Final safety check: ensure no NumPy types remain
            final_values = []
            for val in values_tuple:
                if val is not None:
                    # Check if it's a NumPy type and log a warning
                    if isinstance(val, (np.integer, np.floating, np.bool_, np.generic)):
                        logging.warning(f"Found NumPy type in final conversion: {type(val)}={val}, converting...")
                        val = convert_numpy_to_native(val)
                    # Use .item() if available (recommended for NumPy scalars)
                    elif hasattr(val, 'item'):
                        try:
                            val = val.item()
                        except (AttributeError, ValueError):
                            pass
                final_values.append(val)
            
            cursor.execute("""
                INSERT INTO my_schema.tpo_analysis (
                    analysis_date, instrument_token, session_type,
                    poc, value_area_high, value_area_low,
                    initial_balance_high, initial_balance_low, confidence_score, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (analysis_date, instrument_token, session_type) 
                DO UPDATE SET
                    poc = EXCLUDED.poc,
                    value_area_high = EXCLUDED.value_area_high,
                    value_area_low = EXCLUDED.value_area_low,
                    initial_balance_high = EXCLUDED.initial_balance_high,
                    initial_balance_low = EXCLUDED.initial_balance_low,
                    confidence_score = EXCLUDED.confidence_score,
                    created_at = EXCLUDED.created_at
            """, tuple(final_values))
            
            conn.commit()
            logging.info(f"Saved TPO analysis for {session_type} on {analysis_date}")
        except Exception as e:
            logging.error(f"Error saving TPO analysis: {e}")
            import traceback
            logging.error(traceback.format_exc())
            if conn:
                try:
                    conn.rollback()
                except Exception as rollback_error:
                    logging.error(f"Error during rollback: {rollback_error}")
        finally:
            if cursor:
                try:
                    cursor.close()
                except Exception:
                    pass
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
    
    def get_tpo_analysis(self, analysis_date: str = None, session_type: str = None) -> Optional[Dict]:
        """
        Retrieve TPO analysis from database
        
        Args:
            analysis_date: Date for analysis (defaults to today)
            session_type: 'pre_market' or 'live' (optional, returns both if None)
            
        Returns:
            Dictionary or list of dictionaries containing TPO analysis
        """
        if not analysis_date:
            analysis_date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            from common.Boilerplate import get_db_connection
            conn = get_db_connection()
            cursor = conn.cursor()
            
            if session_type:
                cursor.execute("""
                    SELECT analysis_date, instrument_token, session_type,
                           poc, value_area_high, value_area_low,
                           initial_balance_high, initial_balance_low, confidence_score, created_at
                    FROM my_schema.tpo_analysis
                    WHERE analysis_date = %s 
                    AND instrument_token = %s 
                    AND session_type = %s
                    ORDER BY created_at DESC
                    LIMIT 1
                """, (analysis_date, self.instrument_token, session_type))
            else:
                cursor.execute("""
                    SELECT analysis_date, instrument_token, session_type,
                           poc, value_area_high, value_area_low,
                           initial_balance_high, initial_balance_low, confidence_score, created_at
                    FROM my_schema.tpo_analysis
                    WHERE analysis_date = %s 
                    AND instrument_token = %s
                    ORDER BY session_type, created_at DESC
                """, (analysis_date, self.instrument_token))
            
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            
            if not rows:
                return None
            
            # Convert to dictionaries
            columns = ['analysis_date', 'instrument_token', 'session_type', 'poc',
                      'value_area_high', 'value_area_low', 'initial_balance_high',
                      'initial_balance_low', 'confidence_score', 'created_at']
            
            if len(rows) == 1 and session_type:
                return dict(zip(columns, rows[0]))
            else:
                return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            logging.error(f"Error retrieving TPO analysis: {e}")
            return None
    
    def _calculate_confidence_score(self, tpo_profile: TPOProfile, data: pd.DataFrame) -> float:
        """
        Calculate confidence score for TPO analysis (0-100)
        
        Factors:
        - Data volume (more ticks = higher confidence)
        - Value area clarity (clear VA = higher confidence)
        - POC strength (strong POC = higher confidence)
        """
        import numpy as np
        
        if data.empty or tpo_profile.tpo_data is None:
            return 0.0
        
        confidence = 0.0
        
        # Factor 1: Data volume (30 points max)
        tick_count = len(data)
        if tick_count > 1000:
            confidence += 30.0
        elif tick_count > 500:
            confidence += 20.0
        elif tick_count > 100:
            confidence += 10.0
        
        # Factor 2: Value Area clarity (40 points max)
        if tpo_profile.value_area_high and tpo_profile.value_area_low:
            va_range = float(tpo_profile.value_area_high - tpo_profile.value_area_low)
            if va_range > 0:
                # Smaller VA range (relative to session range) = higher confidence
                if 'last_price' in data.columns:
                    session_range = float(data['last_price'].max() - data['last_price'].min())
                else:
                    session_range = va_range * 2.0
                va_ratio = va_range / session_range if session_range > 0 else 1.0
                if va_ratio < 0.3:  # Tight value area
                    confidence += 40.0
                elif va_ratio < 0.5:
                    confidence += 30.0
                elif va_ratio < 0.7:
                    confidence += 20.0
        
        # Factor 3: POC strength (30 points max)
        if tpo_profile.tpo_data is not None and not tpo_profile.tpo_data.empty:
            max_tpo_count = float(tpo_profile.tpo_data['tpo_count'].max())
            total_tpos = float(tpo_profile.tpo_data['tpo_count'].sum())
            if total_tpos > 0:
                poc_strength = max_tpo_count / total_tpos
                confidence += min(30.0, poc_strength * 30.0)
        
        # Ensure return value is native Python float, not NumPy
        return float(min(100.0, confidence))

