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


def get_latest_supertrend(scrip_id: str, conn=None, force_recalculate: bool = False):
    """Get the latest supertrend value, direction, corresponding close price, and days below supertrend
    First checks database for today's value. Only calculates if not found or force_recalculate=True.
    
    Returns: (supertrend_value, direction, close_price, days_below_supertrend) or None if error
    direction: -1 if price is below supertrend, 1 if price is above supertrend
    days_below_supertrend: number of consecutive days below supertrend in the latest downtrend
    """
    from datetime import date
    
    calculation_date = date.today()
    
    try:
        # Always create a new connection to avoid connection issues
        conn = get_db_connection()
        should_close = True
        
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # First, check if we have today's value in database (unless force_recalculate)
        if not force_recalculate:
            cursor.execute("""
                SELECT 
                    supertrend_value,
                    supertrend_direction,
                    close_price,
                    days_below_supertrend
                FROM my_schema.supertrend_values
                WHERE scrip_id = %s
                AND calculation_date = %s
            """, (scrip_id, calculation_date))
            
            row = cursor.fetchone()
            if row:
                # Found in database, return it
                result = (
                    float(row['supertrend_value']) if row['supertrend_value'] is not None else None,
                    int(row['supertrend_direction']) if row['supertrend_direction'] is not None else None,
                    float(row['close_price']) if row['close_price'] is not None else None,
                    int(row['days_below_supertrend']) if row['days_below_supertrend'] is not None else None
                )
                cursor.close()
                if should_close:
                    conn.close()
                logging.debug(f"Supertrend retrieved from database for {scrip_id} (date: {calculation_date})")
                return result
        
        # Not found in database or force_recalculate, need to calculate
        logging.debug(f"Calculating supertrend for {scrip_id} (not found in database or force_recalculate=True)")
        
        # Get OHLC data for candlestick - same query as api_candlestick endpoint
        # Get last 90 days of data ordered by date ASC (oldest to newest) to properly calculate days below supertrend
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
        
        rows = cursor.fetchall()
        
        if not rows or len(rows) < 14:
            logging.debug(f"Not enough data for supertrend calculation for {scrip_id}: {len(rows) if rows else 0} rows")
            cursor.close()
            if should_close:
                conn.close()
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
        except Exception as calc_error:
            logging.warning(f"Error calculating supertrend for {scrip_id}: {calc_error}")
            import traceback
            logging.warning(traceback.format_exc())
            cursor.close()
            if should_close:
                conn.close()
            return None
        
        # Get the latest non-NaN supertrend value (last element in array)
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
            cursor.close()
            if should_close:
                conn.close()
            return None
        
        # Calculate days below supertrend (only for latest downtrend)
        days_below_supertrend = 0
        if latest_direction == -1:  # Currently below supertrend
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
        
        # Store in database for future use
        try:
            cursor.execute("""
                INSERT INTO my_schema.supertrend_values 
                    (scrip_id, calculation_date, supertrend_value, supertrend_direction, close_price, days_below_supertrend, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (scrip_id, calculation_date) 
                DO UPDATE SET
                    supertrend_value = EXCLUDED.supertrend_value,
                    supertrend_direction = EXCLUDED.supertrend_direction,
                    close_price = EXCLUDED.close_price,
                    days_below_supertrend = EXCLUDED.days_below_supertrend,
                    updated_at = CURRENT_TIMESTAMP
            """, (scrip_id, calculation_date, latest_supertrend, latest_direction, latest_close, days_below_supertrend))
            conn.commit()
            logging.debug(f"Supertrend stored in database for {scrip_id} (date: {calculation_date})")
        except Exception as db_error:
            logging.warning(f"Error storing supertrend in database for {scrip_id}: {db_error}")
            conn.rollback()
        
        cursor.close()
        if should_close:
            conn.close()
        
        logging.debug(f"Supertrend calculated for {scrip_id}: value={latest_supertrend}, direction={latest_direction}, close={latest_close}, days_below={days_below_supertrend}")
        return (latest_supertrend, latest_direction, latest_close, days_below_supertrend)
        
    except Exception as e:
        logging.error(f"Error getting supertrend for {scrip_id}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals() and should_close:
            conn.close()
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
