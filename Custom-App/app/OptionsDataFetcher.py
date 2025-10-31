"""
Options Data Fetcher Utility
Provides helper functions to query options chain data from database
"""

import pandas as pd
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
import logging
from Boilerplate import get_db_connection

class OptionsDataFetcher:
    """Utility class to fetch and filter options chain data"""
    
    def __init__(self):
        """Initialize Options Data Fetcher"""
        self.db_config = None
    
    def get_options_chain(self, 
                          expiry: Optional[date] = None,
                          strike_range: Optional[Tuple[float, float]] = None,
                          option_type: Optional[str] = None,
                          min_volume: int = 0,
                          min_oi: int = 0) -> pd.DataFrame:
        """
        Get options chain data from database
        
        Args:
            expiry: Filter by expiry date (None = all expiries)
            strike_range: Tuple of (min_strike, max_strike) or None for all
            option_type: 'CE' or 'PE' or None for both
            min_volume: Minimum volume filter
            min_oi: Minimum open interest filter
            
        Returns:
            DataFrame with options chain data
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Build WHERE clause
            where_conditions = ["run_date = CURRENT_DATE"]
            params = []
            
            if expiry:
                where_conditions.append("expiry = %s")
                params.append(expiry)
            
            if strike_range:
                where_conditions.append("strike_price >= %s AND strike_price <= %s")
                params.extend([strike_range[0], strike_range[1]])
            
            if option_type:
                where_conditions.append("option_type = %s")
                params.append(option_type)
            
            where_clause = " AND ".join(where_conditions)
            
            # Get latest tick for each instrument_token (group by)
            query = f"""
                SELECT DISTINCT ON (instrument_token)
                    instrument_token,
                    tradingsymbol,
                    strike_price,
                    option_type,
                    expiry,
                    last_price,
                    volume,
                    oi,
                    average_price,
                    timestamp,
                    buy_quantity,
                    sell_quantity
                FROM my_schema.options_ticks
                WHERE {where_clause}
                ORDER BY instrument_token, timestamp DESC
            """
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                return pd.DataFrame()
            
            columns = [
                'instrument_token', 'tradingsymbol', 'strike_price', 'option_type',
                'expiry', 'last_price', 'volume', 'oi', 'average_price',
                'timestamp', 'buy_quantity', 'sell_quantity'
            ]
            
            df = pd.DataFrame(rows, columns=columns)
            
            # Apply filters
            if min_volume > 0:
                df = df[df['volume'] >= min_volume]
            
            if min_oi > 0:
                df = df[df['oi'] >= min_oi]
            
            return df
            
        except Exception as e:
            logging.error(f"Error fetching options chain: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return pd.DataFrame()
    
    def get_atm_strikes(self, current_price: float, strike_step: float = 50.0) -> Tuple[float, float]:
        """
        Get ATM (At The Money) strikes around current price
        
        Args:
            current_price: Current Nifty price
            strike_step: Strike interval (default 50 for Nifty)
            
        Returns:
            Tuple of (CE_strike, PE_strike) - ATM strikes for calls and puts
        """
        # Round to nearest strike
        atm_strike = round(current_price / strike_step) * strike_step
        
        return (atm_strike, atm_strike)
    
    def get_option_quote(self, instrument_token: int) -> Optional[Dict]:
        """
        Get latest quote for a specific option
        
        Args:
            instrument_token: Option instrument token
            
        Returns:
            Dictionary with option quote data or None
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            query = """
                SELECT 
                    instrument_token,
                    tradingsymbol,
                    strike_price,
                    option_type,
                    expiry,
                    last_price,
                    volume,
                    oi,
                    average_price,
                    buy_quantity,
                    sell_quantity,
                    timestamp
                FROM my_schema.options_ticks
                WHERE instrument_token = %s
                AND run_date = CURRENT_DATE
                ORDER BY timestamp DESC
                LIMIT 1
            """
            
            cursor.execute(query, (instrument_token,))
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return None
            
            columns = [
                'instrument_token', 'tradingsymbol', 'strike_price', 'option_type',
                'expiry', 'last_price', 'volume', 'oi', 'average_price',
                'buy_quantity', 'sell_quantity', 'timestamp'
            ]
            
            return dict(zip(columns, row))
            
        except Exception as e:
            logging.error(f"Error fetching option quote: {e}")
            return None
    
    def get_expiry_dates(self) -> List[date]:
        """
        Get all available expiry dates for NIFTY options
        
        Returns:
            List of expiry dates (sorted)
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            query = """
                SELECT DISTINCT expiry
                FROM my_schema.options_ticks
                WHERE expiry >= CURRENT_DATE
                AND run_date = CURRENT_DATE
                ORDER BY expiry ASC
            """
            
            cursor.execute(query)
            rows = cursor.fetchall()
            conn.close()
            
            return [row[0] for row in rows if row[0]]
            
        except Exception as e:
            logging.error(f"Error fetching expiry dates: {e}")
            return []
    
    def get_strikes_for_expiry(self, expiry: date, option_type: Optional[str] = None) -> List[float]:
        """
        Get all available strikes for a given expiry
        
        Args:
            expiry: Expiry date
            option_type: 'CE' or 'PE' or None for both
            
        Returns:
            List of strike prices (sorted)
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            if option_type:
                query = """
                    SELECT DISTINCT strike_price
                    FROM my_schema.options_ticks
                    WHERE expiry = %s
                    AND option_type = %s
                    AND run_date = CURRENT_DATE
                    ORDER BY strike_price ASC
                """
                cursor.execute(query, (expiry, option_type))
            else:
                query = """
                    SELECT DISTINCT strike_price
                    FROM my_schema.options_ticks
                    WHERE expiry = %s
                    AND run_date = CURRENT_DATE
                    ORDER BY strike_price ASC
                """
                cursor.execute(query, (expiry,))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [float(row[0]) for row in rows if row[0] is not None]
            
        except Exception as e:
            logging.error(f"Error fetching strikes for expiry: {e}")
            return []
    
    def get_nifty_current_price(self) -> Optional[float]:
        """
        Get current Nifty 50 price from database
        
        Returns:
            Current Nifty price or None
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get latest Nifty 50 tick (instrument_token = 256265)
            query = """
                SELECT last_price
                FROM my_schema.ticks
                WHERE instrument_token = 256265
                ORDER BY timestamp DESC
                LIMIT 1
            """
            
            cursor.execute(query)
            row = cursor.fetchone()
            conn.close()
            
            if row and row[0]:
                return float(row[0])
            
            return None
            
        except Exception as e:
            logging.error(f"Error fetching Nifty current price: {e}")
            return None

