"""
Technical indicator calculation functions.
Contains Supertrend and other technical analysis utilities.
"""
import numpy as np
import psycopg2.extras
from common.Boilerplate import get_db_connection
import logging

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError as e:
    logging.error(f"TA-Lib not available: {e}")
    TALIB_AVAILABLE = False


def calculate_supertrend(high, low, close, period=14, multiplier=3.0):
    """Calculate Supertrend indicator"""
    if not TALIB_AVAILABLE:
        logging.error("TA-Lib not available, cannot calculate Supertrend")
        return np.full(len(close), np.nan), np.zeros(len(close))
    
    try:
        # Calculate ATR
        atr = talib.ATR(high, low, close, timeperiod=period)
        
        # Calculate basic bands
        hl_avg = (high + low) / 2
        upper_band = hl_avg + (multiplier * atr)
        lower_band = hl_avg - (multiplier * atr)
        
        # Initialize arrays
        supertrend = np.full(len(close), np.nan)
        direction = np.zeros(len(close))
        
        # Initialize final upper/lower bands
        final_upper_band = upper_band.copy()
        final_lower_band = lower_band.copy()
        
        # Calculate Supertrend
        for i in range(1, len(close)):
            # Final Upper Band
            if close[i-1] <= final_upper_band[i-1]:
                final_upper_band[i] = min(upper_band[i], final_upper_band[i-1])
            else:
                final_upper_band[i] = upper_band[i]
            
            # Final Lower Band
            if close[i-1] >= final_lower_band[i-1]:
                final_lower_band[i] = max(lower_band[i], final_lower_band[i-1])
            else:
                final_lower_band[i] = lower_band[i]
            
            # Supertrend
            if i == 1:
                # Initialize first value
                supertrend[i] = final_upper_band[i] if close[i-1] <= final_upper_band[i] else final_lower_band[i]
                direction[i] = -1 if close[i-1] <= final_upper_band[i] else 1
            else:
                if close[i-1] <= supertrend[i-1]:
                    supertrend[i] = final_upper_band[i]
                    direction[i] = -1  # Down
                else:
                    supertrend[i] = final_lower_band[i]
                    direction[i] = 1  # Up
        
        return supertrend, direction
    except Exception as e:
        logging.error(f"Error calculating Supertrend: {e}")
        return np.full(len(close), np.nan), np.zeros(len(close))


def get_latest_supertrend(scrip_id: str, conn=None):
    """Get the latest supertrend value, direction, corresponding close price, and days below supertrend
    Uses the same logic as the candlestick endpoint
    Returns: (supertrend_value, direction, close_price, days_below_supertrend) or None if error
    direction: -1 if price is below supertrend, 1 if price is above supertrend
    days_below_supertrend: number of consecutive days below supertrend in the latest downtrend
    """
    print(f"---XXX--- get_latest_supertrend called for {scrip_id}")
    try:
        # Always create a new connection to avoid connection issues
        # The passed conn might be closed or in use by other queries
        print(f"---XXX--- Step 1: Creating new connection for {scrip_id}")
        conn = get_db_connection()
        should_close = True
        print(f"---XXX--- Step 2: Created new connection for {scrip_id}")
        
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        print(f"---XXX--- Step 3: Created cursor for {scrip_id}")
        
        # Get OHLC data for candlestick - same query as api_candlestick endpoint
        # Get last 90 days of data ordered by date ASC (oldest to newest) to properly calculate days below supertrend
        print(f"---XXX--- Step 4: Executing query for {scrip_id}")
        cursor.execute("""
            SELECT 
                price_high,
                price_low,
                price_close,
                price_date
            FROM my_schema.rt_intraday_price
            WHERE scrip_id = %s
            AND price_date::date >= CURRENT_DATE - make_interval(days => 90)
            AND price_high IS NOT NULL
            AND price_low IS NOT NULL
            AND price_close IS NOT NULL
            ORDER BY price_date ASC
        """, (scrip_id,))
        
        print(f"---XXX--- Step 5: Fetching rows for {scrip_id}")
        rows = cursor.fetchall()
        cursor.close()
        print(f"---XXX--- Step 6: Closed cursor for {scrip_id}")
        
        if should_close:
            conn.close()
            print(f"---XXX--- Step 7: Closed connection for {scrip_id}")
        
        print(f"---XXX--- Found {len(rows) if rows else 0} rows for {scrip_id}")
        
        if not rows or len(rows) < 14:
            print(f"---XXX--- Not enough data for {scrip_id}: {len(rows) if rows else 0} rows")
            logging.debug(f"Not enough data for supertrend calculation for {scrip_id}: {len(rows) if rows else 0} rows")
            return None
        
        # Extract arrays - same as candlestick endpoint
        highs = []
        lows = []
        closes = []
        dates = []
        
        for row in rows:
            highs.append(float(row['price_high']) if row['price_high'] else 0.0)
            lows.append(float(row['price_low']) if row['price_low'] else 0.0)
            closes.append(float(row['price_close']) if row['price_close'] else 0.0)
            dates.append(row['price_date'])
        
        # Convert to numpy arrays
        highs_array = np.array(highs)
        lows_array = np.array(lows)
        closes_array = np.array(closes)
        
        # Calculate supertrend - same as candlestick endpoint
        try:
            supertrend, supertrend_direction = calculate_supertrend(highs_array, lows_array, closes_array)
            print('---XXX--- Supertrend')
            print(supertrend)
        except Exception as calc_error:
            logging.warning(f"Error calculating supertrend for {scrip_id}: {calc_error}")
            import traceback
            logging.warning(traceback.format_exc())
            return None
        
        # Get the latest non-NaN supertrend value (last element in array)
        # Same logic as candlestick endpoint - get last value
        latest_supertrend = None
        latest_direction = None
        latest_close = None
        latest_index = None
        
        # Find the last non-NaN value
        for i in range(len(supertrend) - 1, -1, -1):
            if not np.isnan(supertrend[i]):
                latest_supertrend = float(supertrend[i])
                latest_direction = int(supertrend_direction[i])
                latest_close = float(closes_array[i])
                latest_index = i
                break
        
        if latest_supertrend is None:
            logging.debug(f"No valid supertrend value found for {scrip_id} (all NaN)")
            return None
        
        # Calculate days below supertrend (only for latest downtrend)
        # Count consecutive calendar days with direction = -1 (below supertrend) from the latest day backwards
        # This looks back at least 90 days or from when supertrend data is available
        days_below_supertrend = 0
        if latest_direction == -1:  # Currently below supertrend
            # Count backwards from latest_index, counting actual calendar days
            # Normalize dates to date objects for proper comparison
            from datetime import date as date_type
            prev_date = None
            for i in range(latest_index, -1, -1):
                if not np.isnan(supertrend_direction[i]) and int(supertrend_direction[i]) == -1:
                    current_date_obj = dates[i]
                    # Normalize to date if it's a datetime
                    if hasattr(current_date_obj, 'date'):
                        current_date_obj = current_date_obj.date()
                    elif isinstance(current_date_obj, str):
                        from datetime import datetime
                        try:
                            current_date_obj = datetime.strptime(current_date_obj.split()[0], '%Y-%m-%d').date()
                        except:
                            current_date_obj = current_date_obj
                    
                    # If this is the first iteration or dates are different, count as a day
                    if prev_date is None or current_date_obj != prev_date:
                        days_below_supertrend += 1
                        prev_date = current_date_obj
                else:
                    break  # Stop counting when we hit a day above supertrend
        
        logging.debug(f"Supertrend calculated for {scrip_id}: value={latest_supertrend}, direction={latest_direction}, close={latest_close}, days_below={days_below_supertrend}")
        return (latest_supertrend, latest_direction, latest_close, days_below_supertrend)
        
    except Exception as e:
        print(f"---XXX--- Exception in get_latest_supertrend for {scrip_id}: {e}")
        import traceback
        print(f"---XXX--- Traceback: {traceback.format_exc()}")
        logging.debug(f"Error getting supertrend for {scrip_id}: {e}")
        logging.debug(traceback.format_exc())
        return None


def calculate_obv(close, volume):
    """
    Calculate On Balance Volume (OBV) indicator.
    
    OBV is calculated as:
    - If close > previous close: OBV = previous OBV + volume
    - If close < previous close: OBV = previous OBV - volume
    - If close == previous close: OBV = previous OBV (unchanged)
    
    Args:
        close: numpy array of closing prices
        volume: numpy array of volumes
        
    Returns:
        numpy array of OBV values
    """
    try:
        obv = np.zeros(len(close))
        
        # Initialize first OBV value with first volume
        if len(close) > 0 and len(volume) > 0:
            obv[0] = volume[0]
        
        # Calculate OBV for subsequent periods
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                # Price increased, add volume
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                # Price decreased, subtract volume
                obv[i] = obv[i-1] - volume[i]
            else:
                # Price unchanged, OBV stays the same
                obv[i] = obv[i-1]
        
        return obv
    except Exception as e:
        logging.error(f"Error calculating OBV: {e}")
        return np.zeros(len(close))
