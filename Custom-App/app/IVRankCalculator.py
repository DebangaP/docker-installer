"""
IV Rank Calculator
Calculates Implied Volatility Rank (IV Rank) from historical IV data
IV Rank = (Current IV - Min IV) / (Max IV - Min IV) * 100
"""

import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, Optional, List
import logging
from Boilerplate import get_db_connection


class IVRankCalculator:
    """
    Calculate IV Rank from historical implied volatility data
    IV Rank helps identify when IV is high/low relative to historical range
    """
    
    def __init__(self, lookback_days: int = 252):
        """
        Initialize IV Rank Calculator
        
        Args:
            lookback_days: Number of trading days to look back (default: 252 = 1 year)
        """
        self.lookback_days = lookback_days
    
    def calculate_iv_rank(self,
                         instrument_token: int,
                         strike_price: float,
                         option_type: str,
                         expiry_date: date,
                         current_iv: float,
                         calculation_date: Optional[date] = None) -> Dict:
        """
        Calculate IV Rank for a specific option
        
        Args:
            instrument_token: Option instrument token
            strike_price: Strike price
            option_type: 'CE' or 'PE'
            expiry_date: Expiry date
            current_iv: Current implied volatility (as decimal, e.g., 0.20 for 20%)
            calculation_date: Date for calculation (defaults to today)
            
        Returns:
            Dictionary with IV Rank, IV Percentile, historical min/max, and statistics
        """
        if calculation_date is None:
            calculation_date = date.today()
        
        # Get historical IV data
        historical_iv = self.get_historical_iv(
            instrument_token, strike_price, option_type, expiry_date, calculation_date
        )
        
        if not historical_iv or len(historical_iv) < 2:
            # Not enough historical data
            return {
                'iv_rank': None,
                'iv_percentile': None,
                'current_iv': current_iv,
                'iv_percentage': current_iv * 100,
                'min_iv': None,
                'max_iv': None,
                'avg_iv': None,
                'data_points': 0,
                'lookback_days': self.lookback_days,
                'sufficient_data': False
            }
        
        # Convert to percentages for easier interpretation
        iv_percentages = [iv * 100 for iv in historical_iv]
        current_iv_percentage = current_iv * 100
        
        min_iv = min(iv_percentages)
        max_iv = max(iv_percentages)
        avg_iv = sum(iv_percentages) / len(iv_percentages)
        
        # Calculate IV Rank: (Current - Min) / (Max - Min) * 100
        if max_iv - min_iv == 0:
            iv_rank = 50.0  # If all values are the same, rank is middle
        else:
            iv_rank = ((current_iv_percentage - min_iv) / (max_iv - min_iv)) * 100.0
            iv_rank = max(0.0, min(100.0, iv_rank))  # Clamp between 0-100
        
        # Calculate IV Percentile (percentage of days with IV <= current IV)
        days_below_current = sum(1 for iv in iv_percentages if iv <= current_iv_percentage)
        iv_percentile = (days_below_current / len(iv_percentages)) * 100.0
        
        return {
            'iv_rank': round(iv_rank, 2),
            'iv_percentile': round(iv_percentile, 2),
            'current_iv': current_iv,
            'iv_percentage': round(current_iv_percentage, 2),
            'min_iv': round(min_iv, 2),
            'max_iv': round(max_iv, 2),
            'avg_iv': round(avg_iv, 2),
            'data_points': len(historical_iv),
            'lookback_days': self.lookback_days,
            'sufficient_data': True
        }
    
    def get_historical_iv(self,
                         instrument_token: int,
                         strike_price: float,
                         option_type: str,
                         expiry_date: date,
                         calculation_date: date) -> List[float]:
        """
        Retrieve historical IV data from database
        
        Args:
            instrument_token: Option instrument token
            strike_price: Strike price
            option_type: 'CE' or 'PE'
            expiry_date: Expiry date
            calculation_date: Current calculation date
            
        Returns:
            List of historical IV values (as decimals)
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            start_date = calculation_date - timedelta(days=self.lookback_days)
            
            query = """
                SELECT iv, price_date
                FROM my_schema.iv_history
                WHERE instrument_token = %s
                AND strike_price = %s
                AND option_type = %s
                AND expiry = %s
                AND price_date >= %s
                AND price_date < %s
                ORDER BY price_date ASC
            """
            
            cursor.execute(query, (instrument_token, strike_price, option_type, expiry_date, 
                                  start_date, calculation_date))
            rows = cursor.fetchall()
            conn.close()
            
            # Extract IV values
            historical_iv = [float(row[0]) for row in rows if row[0] is not None]
            
            return historical_iv
            
        except Exception as e:
            logging.error(f"Error fetching historical IV: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return []
    
    def save_iv_data(self,
                    instrument_token: int,
                    strike_price: float,
                    option_type: str,
                    expiry_date: date,
                    iv: float,
                    price_date: Optional[date] = None) -> bool:
        """
        Save IV data to database for historical tracking
        
        Args:
            instrument_token: Option instrument token
            strike_price: Strike price
            option_type: 'CE' or 'PE'
            expiry_date: Expiry date
            iv: Implied volatility (as decimal)
            price_date: Date for IV (defaults to today)
            
        Returns:
            True if successful, False otherwise
        """
        if price_date is None:
            price_date = date.today()
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Insert or update IV data
            query = """
                INSERT INTO my_schema.iv_history 
                (instrument_token, strike_price, option_type, expiry, iv, price_date, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (instrument_token, strike_price, option_type, expiry, price_date)
                DO UPDATE SET iv = EXCLUDED.iv, timestamp = CURRENT_TIMESTAMP
            """
            
            cursor.execute(query, (instrument_token, strike_price, option_type, expiry_date, iv, price_date))
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logging.error(f"Error saving IV data: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return False
    
    def get_vix_adjusted_threshold(self, current_vix: Optional[float] = None) -> Dict:
        """
        Get VIX-adjusted thresholds for IV Rank interpretation
        
        Args:
            current_vix: Current VIX level (optional)
            
        Returns:
            Dictionary with thresholds adjusted for VIX level
        """
        # Default thresholds (can be adjusted based on VIX)
        thresholds = {
            'high_iv_rank': 70.0,
            'low_iv_rank': 30.0,
            'very_high_iv_rank': 85.0,
            'very_low_iv_rank': 15.0
        }
        
        # Adjust thresholds if VIX is provided
        if current_vix is not None:
            # High VIX (> 25): Lower thresholds (easier to reach high IV rank)
            if current_vix > 25:
                thresholds['high_iv_rank'] = 60.0
                thresholds['very_high_iv_rank'] = 75.0
            # Low VIX (< 15): Higher thresholds (harder to reach high IV rank)
            elif current_vix < 15:
                thresholds['high_iv_rank'] = 80.0
                thresholds['very_high_iv_rank'] = 90.0
        
        return thresholds
    
    def interpret_iv_rank(self, iv_rank: float, vix: Optional[float] = None) -> Dict:
        """
        Interpret IV Rank value and provide trading context
        
        Args:
            iv_rank: IV Rank value (0-100)
            vix: Current VIX level (optional)
            
        Returns:
            Dictionary with interpretation and trading implications
        """
        thresholds = self.get_vix_adjusted_threshold(vix)
        
        if iv_rank >= thresholds['very_high_iv_rank']:
            interpretation = 'Very High'
            recommendation = 'Favorable for option selling strategies'
            risk_level = 'Moderate'
        elif iv_rank >= thresholds['high_iv_rank']:
            interpretation = 'High'
            recommendation = 'Good for option selling, watch for IV crush'
            risk_level = 'Low-Moderate'
        elif iv_rank <= thresholds['very_low_iv_rank']:
            interpretation = 'Very Low'
            recommendation = 'Unfavorable for selling, consider buying or waiting'
            risk_level = 'Low'
        elif iv_rank <= thresholds['low_iv_rank']:
            interpretation = 'Low'
            recommendation = 'Less attractive for selling, IV may expand'
            risk_level = 'Low'
        else:
            interpretation = 'Moderate'
            recommendation = 'Neutral - evaluate other factors'
            risk_level = 'Moderate'
        
        return {
            'iv_rank': iv_rank,
            'interpretation': interpretation,
            'recommendation': recommendation,
            'risk_level': risk_level,
            'thresholds_used': thresholds
        }
