"""
Mutual Fund NAV Fetcher Module
Fetches historical NAV data from AMFI (primary) or Yahoo Finance (fallback)
"""
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import logging
import time
from typing import List, Dict, Optional
from common.Boilerplate import get_db_connection


class MFNAVFetcher:
    """Fetches historical NAV data for Mutual Funds from AMFI or Yahoo Finance"""
    
    def __init__(self, db_config=None):
        """
        Initialize MF NAV Fetcher
        
        Args:
            db_config: Optional database configuration dict
        """
        self.db_config = db_config
        self.mftool = None
        self._init_mftool()
    
    def _init_mftool(self):
        """Initialize mftool library for AMFI data"""
        try:
            from mftool import Mftool
            self.mftool = Mftool()
            logging.info("mftool library initialized successfully")
        except ImportError:
            logging.warning("mftool library not available. Will use Yahoo Finance as fallback.")
            self.mftool = None
        except Exception as e:
            logging.warning(f"Error initializing mftool: {e}. Will use Yahoo Finance as fallback.")
            self.mftool = None
    
    def fetch_mf_scheme_code(self, mf_symbol: str) -> Optional[str]:
        """
        Map MF symbol to AMFI scheme code
        
        Args:
            mf_symbol: Mutual Fund trading symbol
            
        Returns:
            AMFI scheme code if found, None otherwise
        """
        try:
            # Try to get scheme code from database first
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT scheme_code 
                FROM my_schema.mf_nav_history 
                WHERE mf_symbol = %s AND scheme_code IS NOT NULL 
                LIMIT 1
            """, (mf_symbol,))
            
            result = cursor.fetchone()
            if result:
                cursor.close()
                conn.close()
                return result[0]
            
            cursor.close()
            conn.close()
            
            # If not in database, try to search using mftool
            if self.mftool:
                try:
                    # Search for MF by symbol/name
                    schemes = self.mftool.get_scheme_codes()
                    # Note: This returns all schemes, need to search by name
                    # For now, return None and let the fetch methods handle it
                    logging.debug(f"Could not find scheme code for {mf_symbol} in database")
                    return None
                except Exception as e:
                    logging.warning(f"Error searching scheme code for {mf_symbol}: {e}")
                    return None
            
            return None
            
        except Exception as e:
            logging.error(f"Error fetching scheme code for {mf_symbol}: {e}")
            return None
    
    def fetch_nav_from_amfi(self, scheme_code: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """
        Fetch NAV data from AMFI using mftool
        
        Args:
            scheme_code: AMFI scheme code
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            List of dictionaries with NAV data
        """
        if not self.mftool:
            logging.warning("mftool not available, cannot fetch from AMFI")
            return []
        
        try:
            # Get historical NAV from AMFI
            nav_data = self.mftool.get_scheme_historical_nav(scheme_code, start_date, end_date)
            
            if not nav_data or len(nav_data) == 0:
                logging.warning(f"No NAV data returned from AMFI for scheme {scheme_code}")
                return []
            
            # Convert to list of dictionaries
            nav_list = []
            for entry in nav_data:
                if isinstance(entry, dict):
                    nav_list.append({
                        'scheme_code': scheme_code,
                        'nav_date': entry.get('date'),
                        'nav_value': entry.get('nav', 0.0)
                    })
                elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    # Handle different formats
                    nav_list.append({
                        'scheme_code': scheme_code,
                        'nav_date': entry[0],
                        'nav_value': float(entry[1]) if entry[1] else 0.0
                    })
            
            logging.info(f"Fetched {len(nav_list)} NAV records from AMFI for scheme {scheme_code}")
            return nav_list
            
        except Exception as e:
            logging.error(f"Error fetching NAV from AMFI for scheme {scheme_code}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return []
    
    def fetch_nav_from_yahoo(self, mf_symbol: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """
        Fetch NAV data from Yahoo Finance (fallback method)
        
        Args:
            mf_symbol: Mutual Fund symbol (may need .NS suffix)
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            List of dictionaries with NAV data
        """
        try:
            # Try with .NS suffix first (NSE)
            yahoo_code = f"{mf_symbol}.NS"
            quote = yf.download(yahoo_code, start=start_date, end=end_date, progress=False)
            
            if quote.empty:
                # Try without suffix
                quote = yf.download(mf_symbol, start=start_date, end=end_date, progress=False)
            
            if quote.empty:
                logging.warning(f"No NAV data found on Yahoo Finance for {mf_symbol}")
                return []
            
            # Convert to list of dictionaries
            nav_list = []
            for date, row in quote.iterrows():
                # Use Close price as NAV
                nav_value = float(row['Close']) if 'Close' in row.index else float(row.iloc[3])
                
                nav_list.append({
                    'scheme_code': None,
                    'nav_date': date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)[:10],
                    'nav_value': nav_value
                })
            
            logging.info(f"Fetched {len(nav_list)} NAV records from Yahoo Finance for {mf_symbol}")
            return nav_list
            
        except Exception as e:
            logging.error(f"Error fetching NAV from Yahoo Finance for {mf_symbol}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return []
    
    def save_nav_to_database(self, mf_symbol: str, fund_name: str, nav_data: List[Dict]) -> int:
        """
        Save NAV data to mf_nav_history table
        
        Args:
            mf_symbol: Mutual Fund trading symbol
            fund_name: Mutual Fund name
            nav_data: List of NAV data dictionaries
            
        Returns:
            Number of records inserted/updated
        """
        if not nav_data:
            return 0
        
        try:
            # Use default connection from Boilerplate
            conn = get_db_connection()
            cursor = conn.cursor()
            
            records_inserted = 0
            
            for nav_entry in nav_data:
                try:
                    scheme_code = nav_entry.get('scheme_code')
                    nav_date = nav_entry.get('nav_date')
                    nav_value = float(nav_entry.get('nav_value', 0))
                    
                    if not nav_date or nav_value <= 0:
                        continue
                    
                    # Ensure nav_date is in correct format
                    if isinstance(nav_date, str):
                        nav_date = datetime.strptime(nav_date[:10], '%Y-%m-%d').date()
                    elif hasattr(nav_date, 'date'):
                        nav_date = nav_date.date()
                    
                    cursor.execute("""
                        INSERT INTO my_schema.mf_nav_history 
                        (mf_symbol, scheme_code, fund_name, nav_date, nav_value, updated_at)
                        VALUES (%s, %s, %s, %s, %s, NOW())
                        ON CONFLICT (mf_symbol, nav_date) 
                        DO UPDATE SET 
                            nav_value = EXCLUDED.nav_value,
                            scheme_code = COALESCE(EXCLUDED.scheme_code, mf_nav_history.scheme_code),
                            fund_name = COALESCE(EXCLUDED.fund_name, mf_nav_history.fund_name),
                            updated_at = NOW()
                    """, (mf_symbol, scheme_code, fund_name, nav_date, nav_value))
                    
                    records_inserted += 1
                    
                except Exception as e:
                    logging.warning(f"Error inserting NAV record for {mf_symbol} on {nav_entry.get('nav_date')}: {e}")
                    continue
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logging.info(f"Saved {records_inserted} NAV records for {mf_symbol}")
            return records_inserted
            
        except Exception as e:
            logging.error(f"Error saving NAV data to database for {mf_symbol}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return 0
    
    def fetch_and_save_nav(self, mf_symbol: str, fund_name: str = None, start_date: datetime = None, end_date: datetime = None, days: int = 365) -> Dict:
        """
        Fetch and save NAV data for a mutual fund
        
        Args:
            mf_symbol: Mutual Fund trading symbol
            fund_name: Mutual Fund name (optional)
            start_date: Start date (optional, defaults to days ago)
            end_date: End date (optional, defaults to today)
            days: Number of days of historical data (default: 365)
            
        Returns:
            Dictionary with success status and records count
        """
        try:
            # Set default dates
            if end_date is None:
                end_date = datetime.now().date()
            if isinstance(end_date, datetime):
                end_date = end_date.date()
            
            if start_date is None:
                start_date = end_date - timedelta(days=days)
            if isinstance(start_date, datetime):
                start_date = start_date.date()
            
            # Convert to datetime for API calls
            start_dt = datetime.combine(start_date, datetime.min.time())
            end_dt = datetime.combine(end_date, datetime.min.time()) + timedelta(days=1)
            
            result = {
                'success': False,
                'records': 0,
                'source': None,
                'message': '',
                'error': None
            }
            
            # Try AMFI first if mftool is available
            scheme_code = self.fetch_mf_scheme_code(mf_symbol)
            
            if self.mftool and scheme_code:
                nav_data = self.fetch_nav_from_amfi(scheme_code, start_dt, end_dt)
                if nav_data:
                    # Add scheme code to all entries
                    for entry in nav_data:
                        entry['scheme_code'] = scheme_code
                    
                    records = self.save_nav_to_database(mf_symbol, fund_name or mf_symbol, nav_data)
                    result['success'] = records > 0
                    result['records'] = records
                    result['source'] = 'AMFI'
                    result['message'] = f"Successfully fetched {records} records from AMFI"
                    return result
            
            # Fallback to Yahoo Finance
            logging.info(f"Trying Yahoo Finance for {mf_symbol}")
            nav_data = self.fetch_nav_from_yahoo(mf_symbol, start_dt, end_dt)
            
            if nav_data:
                records = self.save_nav_to_database(mf_symbol, fund_name or mf_symbol, nav_data)
                result['success'] = records > 0
                result['records'] = records
                result['source'] = 'Yahoo Finance'
                result['message'] = f"Successfully fetched {records} records from Yahoo Finance"
                return result
            
            result['error'] = f"No NAV data found for {mf_symbol} from any source"
            result['message'] = result['error']
            return result
            
        except Exception as e:
            logging.error(f"Error in fetch_and_save_nav for {mf_symbol}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return {
                'success': False,
                'records': 0,
                'source': None,
                'message': f"Error: {str(e)}",
                'error': str(e)
            }

