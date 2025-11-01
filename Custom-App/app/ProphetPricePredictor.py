"""
Prophet Price Predictor
Uses Facebook Prophet to generate 30-day price predictions for stocks

the below commands need to be run in the Python container before running the script.

pip install cmdstanpy
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"


"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import json
import psycopg2
import psycopg2.extras
from Boilerplate import get_db_connection

# Try to import Prophet and cmdstanpy
try:
    from prophet import Prophet
    import cmdstanpy
    PROPHET_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Prophet or cmdstanpy not available: {e}")
    PROPHET_AVAILABLE = False
    Prophet = None

# Suppress Prophet's internal logging
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import logging as std_logging
std_logging.getLogger('prophet').setLevel(std_logging.WARNING)
std_logging.getLogger('cmdstanpy').setLevel(std_logging.WARNING)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_numpy_to_native(value):
    """
    Convert NumPy types to native Python types to avoid compatibility issues
    and database insertion errors with NumPy 2.0+
    """
    if value is None:
        return None
    
    # Handle NumPy scalar types
    try:
        if hasattr(value, 'item'):  # NumPy scalars have .item() method
            return value.item()
    except (AttributeError, ValueError):
        pass
    
    # Explicit type checks (compatible with both NumPy 1.x and 2.x)
    try:
        if isinstance(value, (np.integer, np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32)):
            return int(value)
        elif isinstance(value, (np.floating, np.float64, np.float32, np.float16)):
            return float(value)
        elif isinstance(value, np.bool_):
            return bool(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, (pd.Timestamp, datetime, date)):
            return value
        elif isinstance(value, (dict, type({}))):
            return {k: convert_numpy_to_native(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return type(value)(convert_numpy_to_native(v) for v in value)
        else:
            return value
    except Exception:
        # If conversion fails, try direct conversion
        try:
            if hasattr(value, 'item'):
                return value.item()
            return float(value) if isinstance(value, (int, float)) else value
        except Exception:
            return value


class ProphetPricePredictor:
    """
    Predicts stock prices using Facebook Prophet time series forecasting
    Generates 30-day price predictions based on historical price data
    """
    
    def __init__(self, prediction_days: int = 30):
        """
        Initialize Prophet Price Predictor
        
        Args:
            prediction_days: Number of days to predict ahead (default: 30)
        """
        self.prediction_days = prediction_days
        self.min_data_points = 60  # Minimum days of data required for reliable forecasting
    
    def get_stocks_list(self) -> List[str]:
        """
        Get list of unique scrip_ids from rt_intraday_price table
        
        Returns:
            List of stock symbols
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT DISTINCT scrip_id
                FROM my_schema.rt_intraday_price
                WHERE country = 'IN'
                AND scrip_id NOT IN ('BITCOIN', 'SOLANA', 'DOGE', 'ETH')
                ORDER BY scrip_id
            """)
            
            stocks = [row[0] for row in cursor.fetchall()]
            cursor.close()
            conn.close()
            
            logger.info(f"Found {len(stocks)} stocks to predict")
            return stocks
            
        except Exception as e:
            logger.error(f"Error getting stocks list: {e}")
            return []
    
    def get_price_data(self, scrip_id: str, days: int = 180) -> Optional[pd.DataFrame]:
        """
        Get OHLC price data for a stock from rt_intraday_price
        
        Args:
            scrip_id: Stock symbol
            days: Number of days to fetch (default: 180 for better forecasting)
            
        Returns:
            DataFrame with price data or None
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cursor.execute("""
                SELECT 
                    price_date as ds,
                    price_close as y
                FROM my_schema.rt_intraday_price
                WHERE scrip_id = %s
                AND country = 'IN'
                AND price_close IS NOT NULL
                AND price_close > 0
                AND price_date::date >= CURRENT_DATE - make_interval(days => %s)
                ORDER BY price_date ASC
            """, (scrip_id, days))
            
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            
            if not rows:
                return None
                
            if len(rows) < self.min_data_points:
                return None
            
            # Convert to DataFrame
            data = []
            for row in rows:
                try:
                    price_date = row['ds']
                    close_price = row['y']
                    
                    # Skip if price is invalid
                    if not close_price or close_price <= 0:
                        continue
                    
                    # Convert price_date to datetime (Prophet requires datetime)
                    # price_date comes as VARCHAR from database, need to parse it
                    try:
                        if isinstance(price_date, str):
                            # Try parsing common date formats
                            price_date = pd.to_datetime(price_date, format='mixed')
                        elif isinstance(price_date, date):
                            price_date = pd.Timestamp(price_date)
                        elif isinstance(price_date, pd.Timestamp):
                            pass  # Already datetime
                        else:
                            # Try converting any other type
                            price_date = pd.to_datetime(str(price_date))
                    except Exception:
                        continue
                    
                    # Ensure close_price is a float
                    try:
                        close_price = float(close_price)
                    except (ValueError, TypeError):
                        continue
                    
                    if close_price > 0:
                        data.append({
                            'ds': price_date,  # Prophet expects datetime
                            'y': close_price
                        })
                except Exception as e:
                    logger.debug(f"Error processing row for {scrip_id}: {e}")
                    continue
            
            if len(data) < self.min_data_points:
                return None
            
            df = pd.DataFrame(data)
            df = df.drop_duplicates(subset=['ds'], keep='last')
            df = df.sort_values('ds')
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting price data for {scrip_id}: {e}")
            return None
    
    def predict_price(self, scrip_id: str, current_price: Optional[float] = None, prediction_days: Optional[int] = None) -> Optional[Dict]:
        """
        Generate price prediction for a stock using Prophet
        
        Args:
            scrip_id: Stock symbol
            current_price: Current price (if None, will use latest from data)
            prediction_days: Number of days to predict ahead (default: uses self.prediction_days, which defaults to 30)
            
        Returns:
            Dictionary with prediction results or None
        """
        # Use provided prediction_days or fall back to instance default
        days_to_predict = prediction_days if prediction_days is not None else self.prediction_days
        try:
            # Get historical price data
            df = self.get_price_data(scrip_id)
            if df is None or len(df) < self.min_data_points:
                return None
            
            # Ensure 'ds' is datetime (Prophet requirement)
            if not pd.api.types.is_datetime64_any_dtype(df['ds']):
                df['ds'] = pd.to_datetime(df['ds'])
            
            # Get current price if not provided
            if current_price is None:
                current_price = float(df['y'].iloc[-1])
            
            # Prepare data for Prophet (requires 'ds' and 'y' columns)
            prophet_df = df[['ds', 'y']].copy()
            
            # Ensure no NaN or infinite values
            prophet_df = prophet_df.dropna()
            prophet_df = prophet_df[prophet_df['y'] > 0]
            prophet_df = prophet_df[np.isfinite(prophet_df['y'])]
            
            if len(prophet_df) < self.min_data_points:
                return None
            
            # Initialize and fit Prophet model
            # Disable default seasonalities that may not be applicable to stock data
            try:
                if not PROPHET_AVAILABLE or Prophet is None:
                    logger.error(f"Prophet is not available. Make sure 'prophet' and 'cmdstanpy' are installed.")
                    return None
                
                # Ensure Prophet is properly initialized with cmdstanpy backend
                model = Prophet(
                    daily_seasonality=False,
                    weekly_seasonality=True,
                    yearly_seasonality=False,
                    changepoint_prior_scale=0.05,  # More conservative changepoints for stock data
                    seasonality_prior_scale=10.0,
                    interval_width=0.80  # 80% confidence interval
                )
                
                # Fit the model (silently)
                model.fit(prophet_df)
            except AttributeError as e:
                if 'stan_backend' in str(e):
                    logger.error(f"Prophet backend error for {scrip_id}: {e}")
                    logger.error("Prophet requires cmdstanpy and cmdstan to be properly installed.")
                    logger.error("Try: pip install cmdstanpy && python -c 'import cmdstanpy; cmdstanpy.install_cmdstan()'")
                else:
                    logger.error(f"Prophet attribute error for {scrip_id}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                return None
            except ImportError as e:
                logger.error(f"Prophet import error for {scrip_id}: {e}")
                logger.error("Make sure 'prophet' and 'cmdstanpy' are installed correctly")
                return None
            except Exception as e:
                logger.error(f"Error fitting Prophet model for {scrip_id}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return None
            
            # Create future dataframe for specified number of days
            future = model.make_future_dataframe(periods=days_to_predict, freq='D')
            
            # Make predictions
            forecast = model.predict(future)
            
            # Get prediction for target date
            last_date = prophet_df['ds'].max()
            target_date = last_date + timedelta(days=days_to_predict)
            
            # Get prediction for target date (find closest date in forecast)
            future_forecast = forecast[forecast['ds'] >= last_date]
            if len(future_forecast) >= days_to_predict:
                predicted_row = future_forecast.iloc[days_to_predict - 1]
            else:
                predicted_row = future_forecast.iloc[-1]
            
            # Convert NumPy types to native Python types
            predicted_price = convert_numpy_to_native(predicted_row['yhat'])
            predicted_price = float(predicted_price) if predicted_price is not None else 0.0
            
            # Calculate percentage change
            price_change_pct = ((predicted_price - current_price) / current_price) * 100
            
            # Calculate confidence based on prediction intervals
            lower_bound = convert_numpy_to_native(predicted_row['yhat_lower'])
            lower_bound = float(lower_bound) if lower_bound is not None else 0.0
            upper_bound = convert_numpy_to_native(predicted_row['yhat_upper'])
            upper_bound = float(upper_bound) if upper_bound is not None else 0.0
            interval_range = upper_bound - lower_bound
            confidence = max(0, min(100, 100 - (interval_range / current_price * 100))) if current_price > 0 else 50
            
            # Get daily predictions for the specified number of days
            daily_predictions = []
            for i in range(min(days_to_predict, len(future_forecast))):
                row = future_forecast.iloc[i]
                daily_predictions.append({
                    'date': row['ds'].strftime('%Y-%m-%d'),
                    'predicted_price': convert_numpy_to_native(row['yhat']),
                    'lower_bound': convert_numpy_to_native(row['yhat_lower']),
                    'upper_bound': convert_numpy_to_native(row['yhat_upper'])
                })
            
            # Ensure all values are native Python types
            return {
                'scrip_id': scrip_id,
                'current_price': convert_numpy_to_native(current_price),
                'predicted_price': convert_numpy_to_native(predicted_price),
                'prediction_days': days_to_predict,
                'predicted_price_change_pct': convert_numpy_to_native(price_change_pct),
                'prediction_confidence': convert_numpy_to_native(confidence),
                'target_date': target_date.strftime('%Y-%m-%d'),
                'lower_bound': convert_numpy_to_native(lower_bound),
                'upper_bound': convert_numpy_to_native(upper_bound),
                'daily_predictions': convert_numpy_to_native(daily_predictions),
                'data_points_used': len(prophet_df)
            }
            
        except Exception as e:
            logger.error(f"Error predicting price for {scrip_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def predict_all_stocks(self, limit: Optional[int] = None, prediction_days: Optional[int] = None) -> List[Dict]:
        """
        Generate predictions for all stocks
        
        Args:
            limit: Optional limit on number of stocks to process
            prediction_days: Number of days to predict ahead (default: uses self.prediction_days)
            
        Returns:
            List of prediction dictionaries
        """
        stocks = self.get_stocks_list()
        
        if limit:
            stocks = stocks[:limit]
        
        predictions = []
        errors = []
        skipped = []
        total = len(stocks)
        
        logger.info(f"Starting Prophet predictions for {total} stocks...")
        
        for idx, stock in enumerate(stocks, 1):
            # Log progress every 10 stocks or at milestones (10%, 25%, 50%, 75%, 100%)
            if idx == 1 or idx % 10 == 0 or idx in [max(1, total//10), max(1, total//4), max(1, total//2), max(1, 3*total//4), total]:
                logger.info(f"Progress: {idx}/{total} stocks processed ({len(predictions)} successful, {len(errors)} errors)")
            
            try:
                prediction = self.predict_price(stock, prediction_days=prediction_days)
                if prediction:
                    predictions.append(prediction)
                else:
                    skipped.append(stock)
            except Exception as e:
                errors.append((stock, str(e)))
                logger.error(f"Error predicting {stock}: {e}")
                continue
        
        # Summary log
        logger.info(f"✓ Prophet predictions completed: {len(predictions)} successful, {len(skipped)} skipped, {len(errors)} errors out of {total} stocks")
        if errors:
            logger.warning(f"Stocks with errors: {', '.join([s[0] for s in errors[:10]])}{' ...' if len(errors) > 10 else ''}")
        
        return predictions
    
    def save_predictions(self, predictions: List[Dict], run_date: Optional[date] = None, prediction_days: Optional[int] = None) -> bool:
        """
        Save predictions to database
        
        Args:
            predictions: List of prediction dictionaries
            run_date: Date for the predictions (default: today)
            prediction_days: Number of days predicted (default: self.prediction_days)
            
        Returns:
            True if successful, False otherwise
        """
        if not run_date:
            run_date = date.today()
        
        if prediction_days is None:
            prediction_days = self.prediction_days
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Delete existing predictions for this run_date and prediction_days
            cursor.execute("""
                DELETE FROM my_schema.prophet_predictions
                WHERE run_date = %s AND prediction_days = %s
            """, (run_date, prediction_days))
            
            # Prepare data for bulk insert
            rows = []
            for pred in predictions:
                try:
                    prediction_details = {
                        'target_date': pred.get('target_date'),
                        'lower_bound': pred.get('lower_bound'),
                        'upper_bound': pred.get('upper_bound'),
                        'data_points_used': pred.get('data_points_used', 0),
                        'daily_predictions': pred.get('daily_predictions', [])
                    }
                    
                    # Ensure all values are native Python types before database insertion
                    # Support both 'predicted_price' (new) and 'predicted_price_30d' (old) for backward compatibility
                    predicted_price = pred.get('predicted_price') or pred.get('predicted_price_30d')
                    
                    rows.append((
                        pred['scrip_id'],
                        run_date,
                        prediction_days,
                        convert_numpy_to_native(pred.get('current_price')),
                        convert_numpy_to_native(predicted_price),
                        convert_numpy_to_native(pred.get('predicted_price_change_pct')),
                        convert_numpy_to_native(pred.get('prediction_confidence', 50.0)),
                        json.dumps(convert_numpy_to_native(prediction_details)),
                        'ACTIVE'
                    ))
                except Exception as e:
                    logger.error(f"Error preparing prediction for {pred.get('scrip_id')}: {e}")
                    continue
            
            # Bulk insert
            if rows:
                cursor.executemany("""
                    INSERT INTO my_schema.prophet_predictions 
                    (scrip_id, run_date, prediction_days, current_price, predicted_price_30d, 
                     predicted_price_change_pct, prediction_confidence, prediction_details, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (scrip_id, run_date, prediction_days) 
                    DO UPDATE SET
                        current_price = EXCLUDED.current_price,
                        predicted_price_30d = EXCLUDED.predicted_price_30d,
                        predicted_price_change_pct = EXCLUDED.predicted_price_change_pct,
                        prediction_confidence = EXCLUDED.prediction_confidence,
                        prediction_details = EXCLUDED.prediction_details,
                        status = EXCLUDED.status
                """, rows)
                
                conn.commit()
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error saving predictions: {e}")
            import traceback
            logger.error(traceback.format_exc())
            if conn:
                conn.rollback()
                conn.close()
            return False
    
    def check_predictions_exist_for_date(self, run_date: Optional[date] = None, prediction_days: Optional[int] = None) -> bool:
        """
        Check if predictions already exist for a given date and prediction_days
        
        Args:
            run_date: Date to check (default: today)
            prediction_days: Number of days to check (default: self.prediction_days)
            
        Returns:
            True if predictions exist, False otherwise
        """
        if not run_date:
            run_date = date.today()
        
        if prediction_days is None:
            prediction_days = self.prediction_days
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT COUNT(*) 
                FROM my_schema.prophet_predictions
                WHERE run_date = %s
                AND prediction_days = %s
                AND status = 'ACTIVE'
            """, (run_date, prediction_days))
            
            count = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            
            return count > 0
            
        except Exception as e:
            logger.error(f"Error checking predictions: {e}")
            return False
    
    def get_prediction_for_stock(self, scrip_id: str, run_date: Optional[date] = None) -> Optional[Dict]:
        """
        Get prediction for a specific stock
        
        Args:
            scrip_id: Stock symbol
            run_date: Date to get prediction from (default: latest)
            
        Returns:
            Prediction dictionary or None
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            if run_date:
                cursor.execute("""
                    SELECT * 
                    FROM my_schema.prophet_predictions
                    WHERE scrip_id = %s
                    AND run_date = %s
                    AND status = 'ACTIVE'
                """, (scrip_id, run_date))
            else:
                cursor.execute("""
                    SELECT * 
                    FROM my_schema.prophet_predictions
                    WHERE scrip_id = %s
                    AND status = 'ACTIVE'
                    ORDER BY run_date DESC
                    LIMIT 1
                """, (scrip_id,))
            
            row = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if row:
                return dict(row)
            return None
            
        except Exception as e:
            logger.error(f"Error getting prediction for {scrip_id}: {e}")
            return None
    
    def get_top_gainers(self, limit: int = 10, run_date: Optional[date] = None, prediction_days: Optional[int] = None) -> List[Dict]:
        """
        Get top N potential gainers based on predictions
        
        Args:
            limit: Number of top gainers to return
            run_date: Date to get predictions from (default: latest)
            prediction_days: Number of days to filter by (default: 30)
            
        Returns:
            List of prediction dictionaries sorted by predicted gain
        """
        if prediction_days is None:
            prediction_days = 30  # Default to 30-day predictions
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            if run_date:
                cursor.execute("""
                    SELECT * 
                    FROM my_schema.prophet_predictions
                    WHERE run_date = %s
                    AND prediction_days = %s
                    AND status = 'ACTIVE'
                    AND predicted_price_change_pct > 0
                    ORDER BY predicted_price_change_pct DESC
                    LIMIT %s
                """, (run_date, prediction_days, limit))
            else:
                cursor.execute("""
                    SELECT * 
                    FROM my_schema.prophet_predictions p1
                    WHERE status = 'ACTIVE'
                    AND prediction_days = %s
                    AND predicted_price_change_pct > 0
                    AND run_date = (SELECT MAX(run_date) FROM my_schema.prophet_predictions WHERE status = 'ACTIVE' AND prediction_days = %s)
                    ORDER BY predicted_price_change_pct DESC
                    LIMIT %s
                """, (prediction_days, prediction_days, limit))
            
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting top gainers: {e}")
            return []
