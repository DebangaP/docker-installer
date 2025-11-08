"""
Historical Options Data Generator
Generates options chain data for historical dates using:
- Historical tick data when available
- Black-Scholes formula when historical data is missing
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from common.Boilerplate import get_db_connection
from market.CalculateTPO import PostgresDataFetcher
from options.OptionsGreeksCalculator import OptionsGreeksCalculator


class HistoricalOptionsGenerator:
    """
    Generate historical options chain data for back-testing
    Uses actual historical data when available, Black-Scholes when missing
    """
    
    def __init__(self, 
                 instrument_token: int = 256265,
                 risk_free_rate: float = 0.065,
                 db_config: Optional[Dict] = None):
        """
        Initialize Historical Options Generator
        
        Args:
            instrument_token: Underlying instrument token (default: 256265 for Nifty)
            risk_free_rate: Risk-free interest rate (default: 6.5% for India)
            db_config: Database configuration dictionary (optional)
        """
        self.instrument_token = instrument_token
        self.risk_free_rate = risk_free_rate
        self.greeks_calculator = OptionsGreeksCalculator(risk_free_rate=risk_free_rate)
        
        # Get database configuration
        if db_config is None:
            import os
            db_config = {
                'host': os.getenv('PG_HOST', 'postgres'),
                'database': os.getenv('PG_DATABASE', 'mydb'),
                'user': os.getenv('PG_USER', 'postgres'),
                'password': os.getenv('PG_PASSWORD', 'postgres'),
                'port': int(os.getenv('PG_PORT', 5432))
            }
        
        self.db_fetcher = PostgresDataFetcher(**db_config)
    
    def get_historical_options_chain(self,
                                    analysis_date: date,
                                    expiry_date: Optional[date] = None,
                                    strike_range: Optional[Tuple[float, float]] = None,
                                    option_type: Optional[str] = None) -> pd.DataFrame:
        """
        Get historical options chain for a specific date
        
        Args:
            analysis_date: Date for which to get options chain
            expiry_date: Filter by expiry date (None = all expiries)
            strike_range: Tuple of (min_strike, max_strike) or None for all
            option_type: 'CE' or 'PE' or None for both
            
        Returns:
            DataFrame with historical options chain data including:
            - All standard option fields
            - is_generated: True if data was generated using Black-Scholes
            - data_source: 'historical' or 'generated'
        """
        try:
            # First, try to get actual historical options data
            historical_data = self._fetch_historical_options_data(
                analysis_date, expiry_date, strike_range, option_type
            )
            
            # Get underlying price for the date
            underlying_price = self._get_underlying_price(analysis_date)
            
            if underlying_price is None:
                logging.warning(f"No underlying price found for {analysis_date}")
                return pd.DataFrame()
            
            # If we have some historical data, use it
            if not historical_data.empty:
                # Mark as historical
                historical_data['is_generated'] = False
                historical_data['data_source'] = 'historical'
                
                # Fill in missing strikes using Black-Scholes if needed
                # (This is optional - can return just historical data)
                return historical_data
            
            # If no historical data, generate using Black-Scholes
            logging.info(f"No historical options data for {analysis_date}, generating using Black-Scholes")
            generated_data = self._generate_options_chain_black_scholes(
                analysis_date, underlying_price, expiry_date, strike_range, option_type
            )
            
            return generated_data
            
        except Exception as e:
            logging.error(f"Error getting historical options chain for {analysis_date}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return pd.DataFrame()
    
    def _fetch_historical_options_data(self,
                                      analysis_date: date,
                                      expiry_date: Optional[date] = None,
                                      strike_range: Optional[Tuple[float, float]] = None,
                                      option_type: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch actual historical options data from database
        
        Returns:
            DataFrame with historical options data or empty DataFrame
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Build WHERE clause
            where_conditions = ["run_date = %s"]
            params = [analysis_date]
            
            if expiry_date:
                where_conditions.append("expiry = %s")
                params.append(expiry_date)
            
            if strike_range:
                where_conditions.append("strike_price >= %s AND strike_price <= %s")
                params.extend([strike_range[0], strike_range[1]])
            
            if option_type:
                where_conditions.append("option_type = %s")
                params.append(option_type)
            
            where_clause = " AND ".join(where_conditions)
            
            # Get latest tick for each instrument_token for the date
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
            return df
            
        except Exception as e:
            logging.error(f"Error fetching historical options data: {e}")
            return pd.DataFrame()
    
    def _get_underlying_price(self, analysis_date: date) -> Optional[float]:
        """
        Get underlying price for a specific date
        
        Args:
            analysis_date: Date to get price for
            
        Returns:
            Underlying price or None
        """
        try:
            # Try to get price from ticks table
            start_time = f'{analysis_date} 09:15:00.000'
            end_time = f'{analysis_date} 15:30:00.000'
            
            tick_data = self.db_fetcher.fetch_tick_data(
                table_name='ticks',
                instrument_token=self.instrument_token,
                start_time=start_time,
                end_time=end_time
            )
            
            if not tick_data.empty:
                # Use last price of the day
                return float(tick_data['last_price'].iloc[-1])
            
            # If no tick data, try to get from OHLC data
            conn = get_db_connection()
            cursor = conn.cursor()
            
            query = """
                SELECT close
                FROM my_schema.rt_intraday_price
                WHERE scrip_id = (
                    SELECT id FROM my_schema.master_scrips 
                    WHERE instrument_token = %s LIMIT 1
                )
                AND price_date = %s
                ORDER BY price_date DESC
                LIMIT 1
            """
            
            cursor.execute(query, (self.instrument_token, analysis_date))
            row = cursor.fetchone()
            conn.close()
            
            if row and row[0]:
                return float(row[0])
            
            return None
            
        except Exception as e:
            logging.error(f"Error getting underlying price for {analysis_date}: {e}")
            return None
    
    def _generate_options_chain_black_scholes(self,
                                             analysis_date: date,
                                             underlying_price: float,
                                             expiry_date: Optional[date] = None,
                                             strike_range: Optional[Tuple[float, float]] = None,
                                             option_type: Optional[str] = None) -> pd.DataFrame:
        """
        Generate options chain using Black-Scholes formula
        
        Args:
            analysis_date: Analysis date
            underlying_price: Current underlying price
            expiry_date: Expiry date (if None, use nearest expiry)
            strike_range: Strike range (if None, generate around ATM)
            option_type: 'CE' or 'PE' or None for both
            
        Returns:
            DataFrame with generated options data
        """
        try:
            # Determine expiry date
            if expiry_date is None:
                # Use nearest weekly expiry (typically Thursday)
                expiry_date = self._get_nearest_expiry(analysis_date)
            
            # Calculate time to expiry
            time_to_expiry = self.greeks_calculator.calculate_time_to_expiry(
                expiry_date, analysis_date
            )
            
            if time_to_expiry <= 0:
                logging.warning(f"Expiry date {expiry_date} is in the past for {analysis_date}")
                return pd.DataFrame()
            
            # Determine strike range
            if strike_range is None:
                # Generate strikes around ATM (±10%)
                strike_step = 50.0  # Nifty strike step
                atm_strike = round(underlying_price / strike_step) * strike_step
                strike_range = (atm_strike - 1000, atm_strike + 1000)
            
            # Estimate implied volatility
            iv = self._estimate_implied_volatility(analysis_date, underlying_price)
            
            # Generate strikes
            strike_step = 50.0
            strikes = np.arange(strike_range[0], strike_range[1] + strike_step, strike_step)
            
            # Convert to list to avoid numpy iteration issues
            # Handle case where strikes might be a single value or empty
            try:
                # Ensure strikes is a numpy array
                if not isinstance(strikes, np.ndarray):
                    strikes = np.array([strikes]) if np.isscalar(strikes) else np.array(strikes)
                
                # Check if array is empty
                if strikes.size == 0:
                    # Fallback: generate a simple range
                    num_strikes = int((strike_range[1] - strike_range[0]) / strike_step) + 1
                    strikes_list = [float(strike_range[0] + i * strike_step) for i in range(num_strikes)]
                else:
                    # Convert numpy array to list of floats
                    strikes_list = [float(s) for s in strikes.flatten() if np.isfinite(s)]
                    
                    # If still empty, use fallback
                    if not strikes_list:
                        num_strikes = int((strike_range[1] - strike_range[0]) / strike_step) + 1
                        strikes_list = [float(strike_range[0] + i * strike_step) for i in range(num_strikes)]
            except (TypeError, ValueError, AttributeError) as e:
                logging.warning(f"Error converting strikes to list: {e}, using default range")
                # Fallback: generate a simple range
                try:
                    num_strikes = int((strike_range[1] - strike_range[0]) / strike_step) + 1
                    strikes_list = [float(strike_range[0] + i * strike_step) for i in range(max(1, num_strikes))]
                except:
                    # Last resort: use a simple range around ATM
                    atm_strike = round(underlying_price / strike_step) * strike_step
                    strikes_list = [atm_strike - strike_step, atm_strike, atm_strike + strike_step]
            
            # Generate options for each strike
            options_list = []
            # Ensure option_types is always a list
            if option_type is None:
                option_types = ['CE', 'PE']
            elif isinstance(option_type, str):
                option_types = [option_type]
            elif isinstance(option_type, (list, tuple)):
                option_types = list(option_type)
            else:
                option_types = ['CE', 'PE']  # Default fallback
            
            for strike in strikes_list:
                for opt_type in option_types:
                    # Calculate theoretical price using Black-Scholes
                    greeks = self.greeks_calculator.calculate_greeks(
                        spot_price=underlying_price,
                        strike_price=strike,
                        time_to_expiry=time_to_expiry,
                        implied_volatility=iv,
                        option_type=opt_type
                    )
                    
                    theoretical_price = greeks['theoretical_price']
                    
                    # Generate synthetic tradingsymbol
                    tradingsymbol = self._generate_tradingsymbol(
                        strike, opt_type, expiry_date
                    )
                    
                    options_list.append({
                        'instrument_token': self._generate_instrument_token(strike, opt_type, expiry_date),
                        'tradingsymbol': tradingsymbol,
                        'strike_price': strike,
                        'option_type': opt_type,
                        'expiry': expiry_date,
                        'last_price': theoretical_price,
                        'volume': 0,  # Generated data has no volume
                        'oi': 0,  # Generated data has no OI
                        'average_price': theoretical_price,
                        'timestamp': datetime.combine(analysis_date, datetime.min.time()),
                        'buy_quantity': 0,
                        'sell_quantity': 0,
                        'is_generated': True,
                        'data_source': 'generated',
                        'implied_volatility': iv,
                        'theoretical_price': theoretical_price
                    })
            
            df = pd.DataFrame(options_list)
            return df
            
        except Exception as e:
            logging.error(f"Error generating options chain with Black-Scholes: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return pd.DataFrame()
    
    def _get_nearest_expiry(self, analysis_date: date) -> date:
        """
        Get nearest expiry date (typically Thursday for weekly options)
        
        Args:
            analysis_date: Analysis date
            
        Returns:
            Nearest expiry date
        """
        # Find next Thursday
        days_ahead = 3 - analysis_date.weekday()  # Thursday is 3
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
        
        expiry = analysis_date + timedelta(days=days_ahead)
        
        # If expiry is too close (less than 1 day), use next week
        if (expiry - analysis_date).days < 1:
            expiry += timedelta(days=7)
        
        return expiry
    
    def _estimate_implied_volatility(self, 
                                    analysis_date: date,
                                    underlying_price: float) -> float:
        """
        Estimate implied volatility for a historical date
        
        Args:
            analysis_date: Analysis date
            underlying_price: Underlying price
            
        Returns:
            Estimated implied volatility (as decimal, e.g., 0.20 for 20%)
        """
        try:
            # Try to get historical IV from options_ticks if available
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Look for historical IV data in a date range around analysis_date
            start_date = analysis_date - timedelta(days=30)
            end_date = analysis_date + timedelta(days=1)
            
            query = """
                SELECT AVG(last_price / strike_price) as avg_iv_estimate
                FROM my_schema.options_ticks
                WHERE run_date >= %s AND run_date <= %s
                AND strike_price > 0
                AND last_price > 0
                AND ABS(strike_price - %s) / %s < 0.05  -- Within 5% of current price
                LIMIT 100
            """
            
            try:
                cursor.execute(query, (start_date, end_date, underlying_price, underlying_price))
                row = cursor.fetchone()
            except Exception as e:
                logging.debug(f"Error executing IV estimation query: {e}")
                row = None
            finally:
                conn.close()
            
            # Check if row exists and has data
            # fetchone() can return None, empty tuple (), or tuple with values
            # Handle all cases carefully to avoid tuple index out of range
            if row is not None:
                try:
                    # Check if row is a tuple/sequence and has at least one element
                    if isinstance(row, (tuple, list)):
                        row_len = len(row)
                        if row_len > 0:
                            try:
                                avg_iv_estimate_val = row[0]
                                if avg_iv_estimate_val is not None:
                                    try:
                                        avg_iv_estimate = float(avg_iv_estimate_val)
                                        if not pd.isna(avg_iv_estimate) and not pd.isinf(avg_iv_estimate) and avg_iv_estimate > 0:
                                            # Use historical average as base
                                            base_iv = avg_iv_estimate * 2.0  # Rough estimate
                                            # Normalize to reasonable range
                                            iv = max(0.10, min(0.50, base_iv))
                                            return iv
                                    except (ValueError, TypeError) as e:
                                        logging.debug(f"Error converting IV estimate to float: {e}")
                                        pass
                            except IndexError as e:
                                # This should not happen if we checked row_len > 0, but handle it anyway
                                logging.debug(f"IndexError accessing row[0]: {e}, row type: {type(row)}, row length: {row_len}")
                                pass
                        else:
                            # Empty tuple/list - no data returned
                            logging.debug(f"Row is empty tuple/list, length: {row_len}")
                    else:
                        # Row is not a tuple/list - unexpected type
                        logging.debug(f"Row is not a tuple/list, type: {type(row)}")
                except (AttributeError, TypeError) as e:
                    logging.debug(f"Error accessing IV estimate from row: {e}, row type: {type(row)}")
                    pass
            else:
                # No row returned - query returned no results
                logging.debug("IV estimation query returned no rows")
            
            # Fallback: Use historical volatility from underlying price movements
            iv_from_returns = self._calculate_historical_volatility(analysis_date)
            
            if iv_from_returns:
                return iv_from_returns
            
            # Default: Use average IV for Nifty options (typically 15-20%)
            return 0.18
            
        except Exception as e:
            logging.warning(f"Error estimating IV, using default: {e}")
            return 0.18
    
    def _calculate_historical_volatility(self, analysis_date: date, 
                                        lookback_days: int = 30) -> Optional[float]:
        """
        Calculate historical volatility from underlying price movements
        
        Args:
            analysis_date: Analysis date
            lookback_days: Number of days to look back
            
        Returns:
            Historical volatility or None
        """
        try:
            start_date = analysis_date - timedelta(days=lookback_days)
            end_date = analysis_date
            
            tick_data = self.db_fetcher.fetch_tick_data(
                table_name='ticks',
                instrument_token=self.instrument_token,
                start_time=f'{start_date} 09:15:00.000',
                end_time=f'{end_date} 15:30:00.000'
            )
            
            if tick_data.empty or len(tick_data) < 10:
                return None
            
            # Calculate daily returns
            prices = tick_data['last_price'].values
            if len(prices) < 2:
                return None
            
            # Convert to float array to avoid numpy type issues
            prices_float = np.array([float(p) for p in prices if pd.notna(p)])
            if len(prices_float) < 2:
                return None
            
            returns = np.diff(prices_float) / prices_float[:-1]
            
            # Filter out invalid returns
            returns = returns[np.isfinite(returns)]
            if len(returns) == 0:
                return None
            
            # Calculate annualized volatility
            daily_vol = np.std(returns)
            if not np.isfinite(daily_vol) or daily_vol <= 0:
                return None
            
            annualized_vol = daily_vol * np.sqrt(252)  # 252 trading days
            
            return float(annualized_vol) if np.isfinite(annualized_vol) else None
            
        except Exception as e:
            logging.warning(f"Error calculating historical volatility: {e}")
            return None
    
    def _generate_tradingsymbol(self, strike: float, option_type: str, expiry: date) -> str:
        """
        Generate synthetic tradingsymbol for generated options
        
        Args:
            strike: Strike price
            option_type: 'CE' or 'PE'
            expiry: Expiry date
            
        Returns:
            Synthetic tradingsymbol
        """
        # Format: NIFTY{expiry}{strike}{CE/PE}
        expiry_str = expiry.strftime('%d%b%Y').upper()
        return f"NIFTY{expiry_str}{int(strike)}{option_type}"
    
    def _generate_instrument_token(self, strike: float, option_type: str, expiry: date) -> int:
        """
        Generate synthetic instrument token for generated options
        
        Args:
            strike: Strike price
            option_type: 'CE' or 'PE'
            expiry: Expiry date
            
        Returns:
            Synthetic instrument token (hash-based)
        """
        # Generate a deterministic token based on strike, type, and expiry
        token_str = f"{strike}_{option_type}_{expiry}"
        return abs(hash(token_str)) % (10 ** 9)  # 9-digit token

