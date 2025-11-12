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
import os
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from common.Boilerplate import get_db_connection

# Import sentiment analysis modules
try:
    from sentiment.NewsSentimentAnalyzer import NewsSentimentAnalyzer
    from sentiment.FundamentalSentimentAnalyzer import FundamentalSentimentAnalyzer
    from sentiment.CombinedSentimentCalculator import CombinedSentimentCalculator
    from stocks.EnsemblePredictor import EnsemblePredictor
    SENTIMENT_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Sentiment analysis modules not available: {e}")
    SENTIMENT_AVAILABLE = False
    NewsSentimentAnalyzer = None
    FundamentalSentimentAnalyzer = None
    CombinedSentimentCalculator = None
    EnsemblePredictor = None

# Load environment variables
load_dotenv()

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
    
    def __init__(self, prediction_days: int = 30, enable_sentiment: bool = True):
        """
        Initialize Prophet Price Predictor
        
        Args:
            prediction_days: Number of days to predict ahead (default: 30)
            enable_sentiment: Whether to enable sentiment analysis (default: True)
        """
        self.prediction_days = prediction_days
        self.min_data_points = 60  # Minimum days of data required for reliable forecasting
        # Check if cross-validation is enabled via environment variable
        self.enable_cross_validation = os.getenv("PROPHET_ENABLE_CROSS_VALIDATION", "false").lower() == "true"
        
        # Initialize sentiment analyzers if available
        self.enable_sentiment = enable_sentiment and SENTIMENT_AVAILABLE
        if self.enable_sentiment:
            try:
                self.news_analyzer = NewsSentimentAnalyzer()
                self.fundamental_analyzer = FundamentalSentimentAnalyzer()
                self.combined_calculator = CombinedSentimentCalculator(news_weight=0.5, fundamental_weight=0.5)
                self.ensemble_predictor = EnsemblePredictor(prophet_weight=0.7, sentiment_weight=0.3)
                logger.info("Sentiment analysis modules initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize sentiment analyzers: {e}")
                self.enable_sentiment = False
        else:
            self.news_analyzer = None
            self.fundamental_analyzer = None
            self.combined_calculator = None
            self.ensemble_predictor = None
        
        logger.info(f"ProphetPricePredictor initialized: cross_validation={self.enable_cross_validation}, prediction_days={self.prediction_days}, sentiment={self.enable_sentiment}")
    
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
    
    def get_price_data(self, scrip_id: str, days: int = 140) -> Optional[pd.DataFrame]:
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
            
            # Log data quality metrics for debugging
            if len(df) > 0:
                min_price = df['y'].min()
                max_price = df['y'].max()
                latest_price = df['y'].iloc[-1]
                price_range_pct = ((max_price - min_price) / min_price * 100) if min_price > 0 else 0
                logger.debug(f"Price data quality for {scrip_id}: {len(df)} data points, price range: {min_price:.2f} - {max_price:.2f} ({price_range_pct:.1f}%), latest: {latest_price:.2f}")
                
                # Check for suspicious data patterns
                if price_range_pct > 500:  # More than 500% range might indicate stock split or bad data
                    logger.warning(f"Large price range detected for {scrip_id}: {price_range_pct:.1f}% (min={min_price:.2f}, max={max_price:.2f})")
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting price data for {scrip_id}: {e}")
            from common.Boilerplate import log_stock_price_fetch_error
            log_stock_price_fetch_error(scrip_id, e, "ProphetPricePredictor.get_price_data")
            return None
    
    def calculate_mape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error (MAPE)
        
        Args:
            actual: Array of actual values
            predicted: Array of predicted values
            
        Returns:
            MAPE value (percentage)
        """
        try:
            # Filter out zeros to avoid division by zero
            mask = actual != 0
            if not mask.any():
                return float('inf')
            
            actual_filtered = actual[mask]
            predicted_filtered = predicted[mask]
            
            mape = np.mean(np.abs((actual_filtered - predicted_filtered) / actual_filtered)) * 100
            return float(mape)
        except Exception as e:
            logger.error(f"Error calculating MAPE: {e}")
            return float('inf')
    
    def calculate_rmse(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Root Mean Squared Error (RMSE)
        
        Args:
            actual: Array of actual values
            predicted: Array of predicted values
            
        Returns:
            RMSE value
        """
        try:
            mse = np.mean((actual - predicted) ** 2)
            rmse = np.sqrt(mse)
            return float(rmse)
        except Exception as e:
            logger.error(f"Error calculating RMSE: {e}")
            return float('inf')
    
    def perform_cross_validation(self, model: Prophet, df: pd.DataFrame, 
                                initial: str = '365 days', period: str = '90 days', 
                                horizon: str = '30 days') -> Dict[str, float]:
        """
        Perform cross-validation on Prophet model
        
        Args:
            model: Fitted Prophet model
            df: Historical data DataFrame
            initial: Initial training period (default: '365 days')
            period: Period between cutoff dates (default: '90 days')
            horizon: Forecast horizon (default: '30 days')
            
        Returns:
            Dictionary with MAPE and RMSE metrics
        """
        try:
            from prophet.diagnostics import cross_validation, performance_metrics
            
            # Adjust parameters based on available data
            data_length = len(df)
            
            # Calculate minimum required days for cross-validation
            # initial + period + horizon
            initial_days = int(initial.split()[0]) if 'days' in initial else 365
            horizon_days = int(horizon.split()[0]) if 'days' in horizon else 30
            
            # Adjust initial period if data is shorter
            if data_length < initial_days + horizon_days + 30:
                # Use at least 60% of data for initial training
                adjusted_initial = max(90, int(data_length * 0.6))
                initial = f'{adjusted_initial} days'
                logger.debug(f"Adjusted initial period to {initial} due to limited data (total: {data_length} days)")
            
            # Perform cross-validation
            df_cv = cross_validation(model, initial=initial, period=period, horizon=horizon)
            
            if len(df_cv) == 0:
                logger.warning("Cross-validation returned no results")
                return {
                    'mape': float('inf'),
                    'rmse': float('inf'),
                    'cv_results_count': 0
                }
            
            # Calculate performance metrics
            df_metrics = performance_metrics(df_cv)
            
            # Extract MAPE and RMSE
            mape = float(df_metrics['mape'].mean()) if 'mape' in df_metrics.columns else float('inf')
            rmse = float(df_metrics['rmse'].mean()) if 'rmse' in df_metrics.columns else float('inf')
            
            return {
                'mape': mape,
                'rmse': rmse,
                'cv_results_count': len(df_cv)
            }
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}. Returning default metrics.")
            import traceback
            logger.debug(traceback.format_exc())
            return {
                'mape': float('inf'),
                'rmse': float('inf'),
                'cv_results_count': 0
            }
    
    def evaluate_parameters(self, df: pd.DataFrame, param_grid: List[Dict]) -> Tuple[Dict, Dict]:
        """
        Evaluate different parameter settings using cross-validation
        
        Args:
            df: Historical data DataFrame
            param_grid: List of parameter dictionaries to test
            
        Returns:
            Tuple of (best_params, best_metrics)
        """
        best_params = None
        best_metrics = {'mape': float('inf'), 'rmse': float('inf')}
        
        logger.info(f"Evaluating {len(param_grid)} parameter configurations...")
        
        for params in param_grid:
            try:
                # Create model with current parameters
                model = Prophet(
                    daily_seasonality=params.get('daily_seasonality', False),
                    weekly_seasonality=params.get('weekly_seasonality', True),
                    yearly_seasonality=params.get('yearly_seasonality', False),
                    changepoint_prior_scale=params.get('changepoint_prior_scale', 0.05),
                    changepoint_range=params.get('changepoint_range', 0.95),
                    seasonality_prior_scale=params.get('seasonality_prior_scale', 10.0),
                    interval_width=params.get('interval_width', 0.80)
                )
                
                # Fit model
                model.fit(df)
                
                # Perform cross-validation
                cv_metrics = self.perform_cross_validation(model, df)
                
                # Check if this is the best configuration
                # Use MAPE as primary metric, RMSE as secondary
                if cv_metrics['mape'] < best_metrics['mape'] or \
                   (cv_metrics['mape'] == best_metrics['mape'] and cv_metrics['rmse'] < best_metrics['rmse']):
                    best_params = params.copy()
                    best_metrics = cv_metrics.copy()
                    logger.debug(f"New best parameters: MAPE={cv_metrics['mape']:.2f}%, RMSE={cv_metrics['rmse']:.2f}")
                
            except Exception as e:
                logger.warning(f"Error evaluating parameters {params}: {e}")
                continue
        
        logger.info(f"Best parameters: {best_params}, MAPE: {best_metrics['mape']:.2f}%, RMSE: {best_metrics['rmse']:.2f}")
        return best_params, best_metrics
    
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
                logger.debug(f"Using latest price from data for {scrip_id}: {current_price:.2f} (date: {df['ds'].iloc[-1]})")
            else:
                logger.debug(f"Using provided current_price for {scrip_id}: {current_price:.2f}")
            
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
                
                # Define default Prophet model parameters
                default_model_params = {
                    'changepoint_prior_scale': 0.05,  # More conservative changepoints for stock data
                    'changepoint_range': 0.95,  # Allow changepoints in first 95% of data (default is 0.8)
                    'seasonality_prior_scale': 10.0,
                    'interval_width': 0.80  # 80% confidence interval
                }
                
                # Initialize metrics for cross-validation
                cv_metrics = None
                
                # Perform cross-validation if enabled
                # Need at least 180 days of data for meaningful cross-validation
                data_days = len(prophet_df)
                if self.enable_cross_validation:
                    if data_days >= 140:
                        logger.info(f"Cross-validation enabled for {scrip_id} (data_days={data_days}). Evaluating parameter settings...")
                    else:
                        logger.debug(f"Cross-validation enabled but insufficient data for {scrip_id}: {data_days} days < 180 days required")
                else:
                    logger.debug(f"Cross-validation disabled for {scrip_id} (data_days={data_days})")
                
                if self.enable_cross_validation and data_days >= 140:
                    
                    # Define parameter grid to test
                    param_grid = [
                        default_model_params,  # Default parameters
                        {
                            'changepoint_prior_scale': 0.01,
                            'changepoint_range': 0.95,
                            'seasonality_prior_scale': 10.0,
                            'interval_width': 0.80
                        },
                        {
                            'changepoint_prior_scale': 0.10,
                            'changepoint_range': 0.95,
                            'seasonality_prior_scale': 10.0,
                            'interval_width': 0.80
                        },
                        {
                            'changepoint_prior_scale': 0.05,
                            'changepoint_range': 0.90,
                            'seasonality_prior_scale': 10.0,
                            'interval_width': 0.80
                        },
                        {
                            'changepoint_prior_scale': 0.05,
                            'changepoint_range': 0.95,
                            'seasonality_prior_scale': 5.0,
                            'interval_width': 0.80
                        },
                        {
                            'changepoint_prior_scale': 0.05,
                            'changepoint_range': 0.95,
                            'seasonality_prior_scale': 15.0,
                            'interval_width': 0.80
                        }
                    ]
                    
                    # Evaluate parameters and get best configuration
                    best_params, best_metrics = self.evaluate_parameters(prophet_df, param_grid)
                    
                    if best_params:
                        model_params = best_params
                        cv_metrics = best_metrics
                        logger.info(f"Best parameters for {scrip_id}: MAPE={best_metrics.get('mape', 'N/A')}, RMSE={best_metrics.get('rmse', 'N/A')}")
                        logger.debug(f"cv_metrics set for {scrip_id}: {cv_metrics}")
                    else:
                        model_params = default_model_params
                        logger.warning(f"Cross-validation failed for {scrip_id} (best_params=None). Using default parameters. best_metrics={best_metrics}")
                else:
                    model_params = default_model_params
                    if self.enable_cross_validation:
                        logger.debug(f"Not enough data for cross-validation for {scrip_id}. Using default parameters.")
                
                # Ensure Prophet is properly initialized with cmdstanpy backend
                model = Prophet(
                    daily_seasonality=False,
                    weekly_seasonality=True,
                    yearly_seasonality=False,
                    changepoint_prior_scale=model_params['changepoint_prior_scale'],
                    changepoint_range=model_params['changepoint_range'],
                    seasonality_prior_scale=model_params['seasonality_prior_scale'],
                    interval_width=model_params['interval_width']
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
            
            # Sanity check: predicted price should be reasonable (within 0.1x to 10x of current price)
            # This catches data quality issues or calculation errors
            if current_price > 0 and predicted_price > 0:
                price_ratio = predicted_price / current_price
                if price_ratio < 0.1 or price_ratio > 10.0:
                    logger.error(f"WARNING: Unusual predicted price for {scrip_id}: current_price={current_price:.2f}, predicted_price={predicted_price:.2f}, ratio={price_ratio:.2f}")
                    logger.error(f"  This may indicate a data quality issue. Check historical prices for {scrip_id}")
                    # Don't fail, but log the warning
            
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
            result = {
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
                'data_points_used': len(prophet_df),
                'model_params': model_params  # Include model parameters
            }
            
            # Log prediction summary for debugging
            logger.info(f"Prophet prediction for {scrip_id}: current={current_price:.2f}, predicted={predicted_price:.2f}, change_pct={price_change_pct:.2f}%, confidence={confidence:.2f}%")
            
            # Add cross-validation metrics if available
            if cv_metrics:
                logger.debug(f"Processing cv_metrics for {scrip_id}: {cv_metrics}")
                mape_value = cv_metrics.get('mape')
                rmse_value = cv_metrics.get('rmse')
                logger.debug(f"Raw values for {scrip_id}: mape_value={mape_value} (type={type(mape_value)}), rmse_value={rmse_value} (type={type(rmse_value)})")
                
                # Convert infinity to None for storage
                if mape_value is not None and mape_value != float('inf') and mape_value != float('-inf'):
                    mape_value = float(mape_value)
                else:
                    mape_value = None
                    
                if rmse_value is not None and rmse_value != float('inf') and rmse_value != float('-inf'):
                    rmse_value = float(rmse_value)
                else:
                    rmse_value = None
                
                logger.debug(f"After conversion for {scrip_id}: mape_value={mape_value}, rmse_value={rmse_value}")
                
                # Only save cv_metrics if both values are valid (not None)
                if mape_value is not None and rmse_value is not None:
                    result['cv_metrics'] = {
                        'mape': convert_numpy_to_native(mape_value),
                        'rmse': convert_numpy_to_native(rmse_value),
                        'cv_results_count': convert_numpy_to_native(cv_metrics.get('cv_results_count', 0))
                    }
                    logger.info(f"Added cv_metrics to result for {scrip_id}: MAPE={mape_value}, RMSE={rmse_value}")
                else:
                    logger.warning(f"Skipping cv_metrics for {scrip_id}: MAPE={mape_value}, RMSE={rmse_value} (invalid values)")
            else:
                logger.debug(f"No cv_metrics for {scrip_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting price for {scrip_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def predict_all_stocks(self, limit: Optional[int] = None, prediction_days: Optional[int] = None, save_immediately: bool = True, run_date: Optional[date] = None) -> List[Dict]:
        """
        Generate predictions for all stocks
        
        Args:
            limit: Optional limit on number of stocks to process
            prediction_days: Number of days to predict ahead (default: uses self.prediction_days)
            save_immediately: If True, save each prediction immediately after calculation (default: True)
            run_date: Date for the predictions (default: today)
            
        Returns:
            List of prediction dictionaries
        """
        if not run_date:
            run_date = date.today()
        
        if prediction_days is None:
            prediction_days = self.prediction_days
        
        # If saving immediately, delete existing predictions for this run_date and prediction_days first
        if save_immediately:
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM my_schema.prophet_predictions
                    WHERE run_date = %s AND prediction_days = %s
                """, (run_date, prediction_days))
                conn.commit()
                deleted_count = cursor.rowcount
                cursor.close()
                conn.close()
                if deleted_count > 0:
                    logger.info(f"Deleted {deleted_count} existing predictions for run_date={run_date}, prediction_days={prediction_days}")
            except Exception as e:
                logger.warning(f"Error deleting existing predictions: {e}. Continuing with generation...")
        
        stocks = self.get_stocks_list()
        
        if limit:
            stocks = stocks[:limit]
        
        predictions = []
        errors = []
        skipped = []
        saved_count = 0
        save_failed_count = 0
        total = len(stocks)
        cv_enabled_count = 0
        cv_ran_count = 0
        cv_failed_count = 0
        insufficient_data_count = 0
        
        logger.info(f"Starting Prophet predictions for {total} stocks (cross_validation={self.enable_cross_validation}, save_immediately={save_immediately})...")
        
        for idx, stock in enumerate(stocks, 1):
            # Log progress every 10 stocks or at milestones (10%, 25%, 50%, 75%, 100%)
            if idx == 1 or idx % 10 == 0 or idx in [max(1, total//10), max(1, total//4), max(1, total//2), max(1, 3*total//4), total]:
                logger.info(f"Progress: {idx}/{total} stocks processed ({len(predictions)} successful, {saved_count} saved, {len(errors)} errors)")
            
            try:
                prediction = self.predict_price(stock, prediction_days=prediction_days)
                if prediction:
                    predictions.append(prediction)
                    # Track cv_metrics in predictions
                    if prediction.get('cv_metrics'):
                        cv_ran_count += 1
                    
                    # Save immediately if requested
                    if save_immediately:
                        if self.save_single_prediction(prediction, run_date=run_date, prediction_days=prediction_days):
                            saved_count += 1
                        else:
                            save_failed_count += 1
                            logger.warning(f"Failed to save prediction for {stock}")
                else:
                    skipped.append(stock)
            except Exception as e:
                errors.append((stock, str(e)))
                logger.error(f"Error predicting {stock}: {e}")
                continue
        
        # Summary log
        # Always generate Nifty50 60-day predictions
        nifty50_scrip_ids = ['NIFTY50', 'Nifty_50', 'Nifty5']
        nifty50_predicted = False
        
        for nifty_id in nifty50_scrip_ids:
            try:
                logger.info(f"Generating 60-day prediction for {nifty_id}")
                nifty_prediction = self.predict_price(nifty_id, prediction_days=60)
                if nifty_prediction:
                    if save_immediately:
                        if self.save_single_prediction(nifty_prediction, run_date=run_date, prediction_days=60):
                            saved_count += 1
                            nifty50_predicted = True
                            logger.info(f"Successfully generated and saved 60-day prediction for {nifty_id}")
                        else:
                            logger.warning(f"Failed to save 60-day prediction for {nifty_id}")
                    else:
                        predictions.append(nifty_prediction)
                        nifty50_predicted = True
                        logger.info(f"Successfully generated 60-day prediction for {nifty_id}")
                    break  # Successfully generated for one of the Nifty50 IDs
            except Exception as e:
                logger.warning(f"Error generating 60-day prediction for {nifty_id}: {e}")
                continue
        
        if not nifty50_predicted:
            logger.warning("Failed to generate Nifty50 60-day prediction")
        
        logger.info(f"✓ Prophet predictions completed: {len(predictions)} successful, {len(skipped)} skipped, {len(errors)} errors out of {total} stocks")
        if save_immediately:
            logger.info(f"  Saved: {saved_count} predictions saved immediately, {save_failed_count} save failures")
        if self.enable_cross_validation:
            logger.info(f"  Cross-validation: {cv_ran_count} predictions have cv_metrics out of {len(predictions)} successful predictions")
        if errors:
            logger.warning(f"Stocks with errors: {', '.join([s[0] for s in errors[:10]])}{' ...' if len(errors) > 10 else ''}")
        
        return predictions
    
    def save_single_prediction(self, prediction: Dict, run_date: Optional[date] = None, prediction_days: Optional[int] = None) -> bool:
        """
        Save a single prediction to database immediately
        
        Args:
            prediction: Prediction dictionary
            run_date: Date for the prediction (default: today)
            prediction_days: Number of days predicted (default: self.prediction_days)
            
        Returns:
            True if successful, False otherwise
        """
        if not run_date:
            run_date = date.today()
        
        if prediction_days is None:
            prediction_days = self.prediction_days
        
        try:
            # Validate required fields
            scrip_id = prediction.get('scrip_id')
            if not scrip_id:
                logger.warning(f"Skipping prediction: missing scrip_id")
                return False
            
            # Support both 'predicted_price' (new) and 'predicted_price_30d' (old) for backward compatibility
            predicted_price = prediction.get('predicted_price') or prediction.get('predicted_price_30d')
            if predicted_price is None:
                logger.warning(f"Skipping prediction for {scrip_id}: missing predicted_price")
                return False
            
            # Convert to native types and validate
            current_price = convert_numpy_to_native(prediction.get('current_price'))
            predicted_price = convert_numpy_to_native(predicted_price)
            predicted_price_change_pct = convert_numpy_to_native(prediction.get('predicted_price_change_pct'))
            prediction_confidence = convert_numpy_to_native(prediction.get('prediction_confidence', 50.0))
            
            # Ensure numeric values are valid
            if current_price is None or predicted_price is None:
                logger.warning(f"Skipping prediction for {scrip_id}: invalid price values (current={current_price}, predicted={predicted_price})")
                return False
            
            # Additional validation: check for unreasonable price ratios
            if current_price > 0 and predicted_price > 0:
                price_ratio = predicted_price / current_price
                if price_ratio < 0.5 or price_ratio > 5.0:
                    logger.error(f"CRITICAL: Unusual predicted_price_30d for {scrip_id}: current={current_price:.2f}, predicted={predicted_price:.2f}, ratio={price_ratio:.2f}")
                    logger.error(f"  This may indicate corrupted data. Prediction will still be saved but should be reviewed.")
            
            # Ensure prediction_confidence has a default value
            if prediction_confidence is None:
                prediction_confidence = 50.0
            
            prediction_details = {
                'target_date': prediction.get('target_date'),
                'lower_bound': prediction.get('lower_bound'),
                'upper_bound': prediction.get('upper_bound'),
                'data_points_used': prediction.get('data_points_used', 0),
                'daily_predictions': prediction.get('daily_predictions', []),
                'prediction_days': prediction.get('prediction_days', prediction_days),
                'model_params': prediction.get('model_params', {
                    'changepoint_prior_scale': 0.05,
                    'changepoint_range': 0.95,
                    'seasonality_prior_scale': 10.0,
                    'interval_width': 0.80
                })
            }
            
            # Only include cv_metrics if it exists and has valid values
            cv_metrics_from_pred = prediction.get('cv_metrics')
            if cv_metrics_from_pred:
                if isinstance(cv_metrics_from_pred, dict):
                    mape = cv_metrics_from_pred.get('mape')
                    rmse = cv_metrics_from_pred.get('rmse')
                    # Only include if both are valid (not None, not infinity)
                    if mape is not None and rmse is not None and mape != float('inf') and rmse != float('inf'):
                        prediction_details['cv_metrics'] = cv_metrics_from_pred
                        logger.debug(f"Including cv_metrics for {scrip_id}: MAPE={mape}, RMSE={rmse}")
            
            # Get database connection and save immediately
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get sentiment scores and enhanced prediction if enabled
            news_sentiment_score = None
            fundamental_sentiment_score = None
            combined_sentiment_score = None
            enhanced_predicted_price_change_pct = None
            enhanced_prediction_confidence = None
            sentiment_metadata = None
            
            if self.enable_sentiment and self.ensemble_predictor:
                try:
                    # Calculate combined sentiment
                    combined_sentiment = self.combined_calculator.calculate_combined_sentiment(
                        scrip_id, run_date
                    )
                    if combined_sentiment:
                        news_sentiment_score = combined_sentiment.get('news_sentiment_score')
                        fundamental_sentiment_score = combined_sentiment.get('fundamental_sentiment_score')
                        combined_sentiment_score = combined_sentiment.get('combined_sentiment_score')
                    
                    # Enhance prediction with sentiment
                    enhanced_prediction = self.ensemble_predictor.enhance_prediction(
                        scrip_id,
                        float(predicted_price_change_pct) if predicted_price_change_pct else 0.0,
                        float(prediction_confidence),
                        combined_sentiment_score,
                        run_date
                    )
                    
                    if enhanced_prediction:
                        enhanced_predicted_price_change_pct = enhanced_prediction.get('enhanced_predicted_price_change_pct')
                        enhanced_prediction_confidence = enhanced_prediction.get('enhanced_prediction_confidence')
                        sentiment_metadata = enhanced_prediction.get('metadata')
                except Exception as e:
                    logger.warning(f"Error calculating sentiment for {scrip_id}: {e}")
            
            insert_query = """
                INSERT INTO my_schema.prophet_predictions 
                (scrip_id, run_date, prediction_days, current_price, predicted_price_30d, 
                 predicted_price_change_pct, prediction_confidence, prediction_details, status,
                 news_sentiment_score, fundamental_sentiment_score, combined_sentiment_score,
                 enhanced_predicted_price_change_pct, enhanced_prediction_confidence, sentiment_metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (scrip_id, run_date, prediction_days) 
                DO UPDATE SET
                    current_price = EXCLUDED.current_price,
                    predicted_price_30d = EXCLUDED.predicted_price_30d,
                    predicted_price_change_pct = EXCLUDED.predicted_price_change_pct,
                    prediction_confidence = EXCLUDED.prediction_confidence,
                    prediction_details = EXCLUDED.prediction_details,
                    status = EXCLUDED.status,
                    news_sentiment_score = EXCLUDED.news_sentiment_score,
                    fundamental_sentiment_score = EXCLUDED.fundamental_sentiment_score,
                    combined_sentiment_score = EXCLUDED.combined_sentiment_score,
                    enhanced_predicted_price_change_pct = EXCLUDED.enhanced_predicted_price_change_pct,
                    enhanced_prediction_confidence = EXCLUDED.enhanced_prediction_confidence,
                    sentiment_metadata = EXCLUDED.sentiment_metadata
            """
            
            try:
                sentiment_metadata_json = json.dumps(sentiment_metadata) if sentiment_metadata else None
                
                cursor.execute(insert_query, (
                    scrip_id,
                    run_date,
                    prediction_days,
                    float(current_price) if current_price is not None else None,
                    float(predicted_price) if predicted_price is not None else None,
                    float(predicted_price_change_pct) if predicted_price_change_pct is not None else None,
                    float(prediction_confidence),
                    json.dumps(convert_numpy_to_native(prediction_details)),
                    'ACTIVE',
                    float(news_sentiment_score) if news_sentiment_score is not None else None,
                    float(fundamental_sentiment_score) if fundamental_sentiment_score is not None else None,
                    float(combined_sentiment_score) if combined_sentiment_score is not None else None,
                    float(enhanced_predicted_price_change_pct) if enhanced_predicted_price_change_pct is not None else None,
                    float(enhanced_prediction_confidence) if enhanced_prediction_confidence is not None else None,
                    sentiment_metadata_json
                ))
                conn.commit()
                logger.debug(f"Successfully saved prediction for {scrip_id}")
                return True
            except Exception as e:
                conn.rollback()
                logger.error(f"Error saving prediction for {scrip_id}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return False
            finally:
                cursor.close()
                conn.close()
                
        except Exception as e:
            logger.error(f"Error saving single prediction for {prediction.get('scrip_id', 'UNKNOWN')}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
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
            skipped_count = 0
            low_confidence_count = 0
            missing_scrip_id_count = 0
            missing_price_count = 0
            invalid_price_count = 0
            for pred in predictions:
                try:
                    # Validate required fields
                    scrip_id = pred.get('scrip_id')
                    if not scrip_id:
                        logger.warning(f"Skipping prediction: missing scrip_id")
                        skipped_count += 1
                        missing_scrip_id_count += 1
                        continue
                    
                    # Support both 'predicted_price' (new) and 'predicted_price_30d' (old) for backward compatibility
                    predicted_price = pred.get('predicted_price') or pred.get('predicted_price_30d')
                    if predicted_price is None:
                        logger.warning(f"Skipping prediction for {scrip_id}: missing predicted_price")
                        skipped_count += 1
                        missing_price_count += 1
                        continue
                    
                    # Convert to native types and validate
                    current_price = convert_numpy_to_native(pred.get('current_price'))
                    predicted_price = convert_numpy_to_native(predicted_price)
                    predicted_price_change_pct = convert_numpy_to_native(pred.get('predicted_price_change_pct'))
                    prediction_confidence = convert_numpy_to_native(pred.get('prediction_confidence', 50.0))
                    
                    # Ensure numeric values are valid
                    if current_price is None or predicted_price is None:
                        logger.warning(f"Skipping prediction for {scrip_id}: invalid price values (current={current_price}, predicted={predicted_price})")
                        skipped_count += 1
                        invalid_price_count += 1
                        continue
                    
                    # Ensure prediction_confidence has a default value
                    if prediction_confidence is None:
                        prediction_confidence = 50.0
                    
                    # Note: Removing confidence filter to save all predictions
                    # Confidence filtering can be done at query time if needed
                    # if prediction_confidence < 50.0:
                    #     logger.debug(f"Skipping prediction for {scrip_id}: confidence {prediction_confidence:.2f}% < 50%")
                    #     skipped_count += 1
                    #     low_confidence_count += 1
                    #     continue
                    
                    prediction_details = {
                        'target_date': pred.get('target_date'),
                        'lower_bound': pred.get('lower_bound'),
                        'upper_bound': pred.get('upper_bound'),
                        'data_points_used': pred.get('data_points_used', 0),
                        'daily_predictions': pred.get('daily_predictions', []),
                        'prediction_days': pred.get('prediction_days', prediction_days),
                        'model_params': pred.get('model_params', {
                            'changepoint_prior_scale': 0.05,
                            'changepoint_range': 0.95,
                            'seasonality_prior_scale': 10.0,
                            'interval_width': 0.80
                        })
                    }
                    
                    # Only include cv_metrics if it exists and has valid values
                    cv_metrics_from_pred = pred.get('cv_metrics')
                    if cv_metrics_from_pred:
                        cv_metrics_count += 1
                        if isinstance(cv_metrics_from_pred, dict):
                            mape = cv_metrics_from_pred.get('mape')
                            rmse = cv_metrics_from_pred.get('rmse')
                            # Only include if both are valid (not None, not infinity)
                            if mape is not None and rmse is not None and mape != float('inf') and rmse != float('inf'):
                                prediction_details['cv_metrics'] = cv_metrics_from_pred
                                cv_metrics_valid_count += 1
                                logger.debug(f"Including cv_metrics for {scrip_id}: MAPE={mape}, RMSE={rmse}")
                            else:
                                cv_metrics_invalid_count += 1
                                logger.debug(f"Skipping cv_metrics for {scrip_id}: MAPE={mape}, RMSE={rmse} (invalid values)")
                        else:
                            cv_metrics_invalid_count += 1
                            logger.debug(f"cv_metrics for {scrip_id} is not a dict: {type(cv_metrics_from_pred)}")
                    
                    rows.append((
                        scrip_id,
                        run_date,
                        prediction_days,
                        float(current_price) if current_price is not None else None,
                        float(predicted_price) if predicted_price is not None else None,
                        float(predicted_price_change_pct) if predicted_price_change_pct is not None else None,
                        float(prediction_confidence),
                        json.dumps(convert_numpy_to_native(prediction_details)),
                        'ACTIVE'
                    ))
                except Exception as e:
                    logger.error(f"Error preparing prediction for {pred.get('scrip_id', 'UNKNOWN')}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    skipped_count += 1
                    continue
            
            if skipped_count > 0:
                logger.warning(f"Skipped {skipped_count} predictions out of {len(predictions)} total:")
                if missing_scrip_id_count > 0:
                    logger.warning(f"  - {missing_scrip_id_count} missing scrip_id")
                if missing_price_count > 0:
                    logger.warning(f"  - {missing_price_count} missing predicted_price")
                if invalid_price_count > 0:
                    logger.warning(f"  - {invalid_price_count} invalid price values")
                if low_confidence_count > 0:
                    logger.warning(f"  - {low_confidence_count} with confidence < 50% (confidence filter is currently disabled)")
                logger.info(f"Successfully prepared {len(rows)} predictions for database insertion")
            
            # Bulk insert in batches to ensure partial data is saved even if some batches fail
            if rows:
                batch_size = 100  # Process 100 rows at a time
                total_rows = len(rows)
                successful_batches = 0
                failed_batches = 0
                total_inserted = 0
                
                insert_query = """
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
                """
                
                # Process rows in batches
                for i in range(0, total_rows, batch_size):
                    batch = rows[i:i + batch_size]
                    batch_num = (i // batch_size) + 1
                    total_batches = (total_rows + batch_size - 1) // batch_size
                    
                    try:
                        cursor.executemany(insert_query, batch)
                        conn.commit()
                        successful_batches += 1
                        total_inserted += len(batch)
                        logger.debug(f"Batch {batch_num}/{total_batches}: Successfully inserted {len(batch)} predictions")
                    except Exception as batch_error:
                        failed_batches += 1
                        logger.error(f"Batch {batch_num}/{total_batches}: Failed to insert {len(batch)} predictions: {batch_error}")
                        import traceback
                        logger.error(traceback.format_exc())
                        # Rollback this batch but continue with next batch
                        try:
                            conn.rollback()
                        except Exception:
                            pass
                        # Continue processing next batch
                        continue
                
                # Summary logging
                if successful_batches > 0:
                    logger.info(f"Successfully saved {total_inserted} predictions in {successful_batches} batch(es) out of {total_batches} total batches")
                if failed_batches > 0:
                    logger.warning(f"Failed to save {total_rows - total_inserted} predictions in {failed_batches} failed batch(es)")
                
                # Log cv_metrics summary
                logger.info(f"cv_metrics summary: {cv_metrics_count} predictions had cv_metrics, {cv_metrics_valid_count} valid, {cv_metrics_invalid_count} invalid out of {len(predictions)} total predictions")
                
                # Return True if at least some data was saved
                if total_inserted > 0:
                    return True
                else:
                    logger.error(f"Failed to save any predictions - all {total_batches} batches failed")
                    return False
            else:
                logger.warning(f"No valid rows to save out of {len(predictions)} predictions")
                if skipped_count == len(predictions):
                    return False
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error saving predictions: {e}")
            import traceback
            logger.error(traceback.format_exc())
            try:
                if 'conn' in locals() and conn:
                    conn.rollback()
                    conn.close()
            except Exception:
                pass
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
        Always includes Nifty50 if available, regardless of ranking
        
        Args:
            limit: Number of top gainers to return
            run_date: Date to get predictions from (default: latest)
            prediction_days: Number of days to filter by (default: 30)
            
        Returns:
            List of prediction dictionaries sorted by predicted gain
        """
        if prediction_days is None:
            prediction_days = 30  # Default to 30-day predictions
        
        # Common scrip_ids for Nifty50
        nifty_scrip_ids = ['NIFTY', 'NIFTY50', 'NIFTY 50', 'NIFTY-50']
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Get the run_date to use
            if run_date is None:
                cursor.execute("""
                    SELECT MAX(run_date) as max_run_date
                    FROM my_schema.prophet_predictions 
                    WHERE status = 'ACTIVE' AND prediction_days = %s
                """, (prediction_days,))
                result = cursor.fetchone()
                if result and result.get('max_run_date'):
                    run_date = result['max_run_date']
                else:
                    cursor.close()
                    conn.close()
                    return []
            
            # Get top N gainers (filter out predictions with confidence < 50%)
            # Join with master_scrips to get sector_code
            cursor.execute("""
                SELECT pp.*, ms.sector_code
                FROM my_schema.prophet_predictions pp
                LEFT JOIN my_schema.master_scrips ms ON pp.scrip_id = ms.scrip_id
                WHERE pp.run_date = %s
                AND pp.prediction_days = %s
                AND pp.status = 'ACTIVE'
                AND pp.predicted_price_change_pct > 0
                AND pp.prediction_confidence >= 50.0
                ORDER BY pp.predicted_price_change_pct DESC
                LIMIT %s
            """, (run_date, prediction_days, limit))
            
            rows = cursor.fetchall()
            top_gainers = [dict(row) for row in rows]
            
            # Check if Nifty50 is already in the list
            nifty_included = False
            nifty_prediction = None
            
            for gainer in top_gainers:
                if gainer.get('scrip_id', '').upper() in [nid.upper() for nid in nifty_scrip_ids]:
                    nifty_included = True
                    break
            
            # If Nifty50 is not in the list, try to fetch it
            if not nifty_included:
                for nifty_id in nifty_scrip_ids:
                    cursor.execute("""
                        SELECT pp.*, ms.sector_code
                        FROM my_schema.prophet_predictions pp
                        LEFT JOIN my_schema.master_scrips ms ON pp.scrip_id = ms.scrip_id
                        WHERE pp.scrip_id = %s
                        AND pp.run_date = %s
                        AND pp.prediction_days = %s
                        AND pp.status = 'ACTIVE'
                        AND pp.predicted_price_change_pct > 0
                        AND pp.prediction_confidence >= 50.0
                        LIMIT 1
                    """, (nifty_id, run_date, prediction_days))
                    
                    nifty_result = cursor.fetchone()
                    if nifty_result:
                        nifty_prediction = dict(nifty_result)
                        break
            
            # Parse prediction_details JSON and extract cv_metrics for each gainer
            for gainer in top_gainers:
                prediction_details = gainer.get('prediction_details')
                if prediction_details:
                    try:
                        # PostgreSQL JSONB might return as dict or string depending on psycopg2 version
                        if isinstance(prediction_details, str):
                            details = json.loads(prediction_details)
                        elif isinstance(prediction_details, dict):
                            details = prediction_details
                        else:
                            # Try to convert to string first
                            details = json.loads(str(prediction_details))
                        
                        # Log what's in prediction_details for debugging
                        logger.debug(f"prediction_details for {gainer.get('scrip_id')}: keys={list(details.keys()) if isinstance(details, dict) else 'not a dict'}")
                        
                        cv_metrics = details.get('cv_metrics')
                        logger.debug(f"cv_metrics for {gainer.get('scrip_id')}: {cv_metrics}, type={type(cv_metrics)}")
                        
                        if cv_metrics and isinstance(cv_metrics, dict):
                            # Handle None, infinity, or invalid values
                            mape_value = cv_metrics.get('mape')
                            rmse_value = cv_metrics.get('rmse')
                            
                            logger.debug(f"Raw cv_metrics values for {gainer.get('scrip_id')}: mape={mape_value}, rmse={rmse_value}")
                            
                            # Convert infinity strings or infinity values to None
                            if mape_value is None or mape_value == float('inf') or mape_value == float('-inf') or (isinstance(mape_value, str) and mape_value.lower() in ['inf', 'infinity', 'nan']):
                                gainer['cv_mape'] = None
                            else:
                                try:
                                    gainer['cv_mape'] = float(mape_value) if mape_value is not None else None
                                except (ValueError, TypeError):
                                    gainer['cv_mape'] = None
                            
                            if rmse_value is None or rmse_value == float('inf') or rmse_value == float('-inf') or (isinstance(rmse_value, str) and rmse_value.lower() in ['inf', 'infinity', 'nan']):
                                gainer['cv_rmse'] = None
                            else:
                                try:
                                    gainer['cv_rmse'] = float(rmse_value) if rmse_value is not None else None
                                except (ValueError, TypeError):
                                    gainer['cv_rmse'] = None
                            
                            logger.debug(f"Extracted cv_metrics for {gainer.get('scrip_id')}: MAPE={gainer['cv_mape']}, RMSE={gainer['cv_rmse']}")
                        else:
                            gainer['cv_mape'] = None
                            gainer['cv_rmse'] = None
                            logger.info(f"No cv_metrics found in prediction_details for {gainer.get('scrip_id')}. cv_metrics={cv_metrics}. This may mean predictions were generated before cross-validation was enabled. Regenerate predictions with PROPHET_ENABLE_CROSS_VALIDATION=true")
                    except (json.JSONDecodeError, Exception) as e:
                        logger.warning(f"Error parsing prediction_details for {gainer.get('scrip_id')}: {e}, type={type(prediction_details)}")
                        gainer['cv_mape'] = None
                        gainer['cv_rmse'] = None
                else:
                    gainer['cv_mape'] = None
                    gainer['cv_rmse'] = None
                    logger.debug(f"No prediction_details found for {gainer.get('scrip_id')}")
            
            cursor.close()
            conn.close()
            
            # If we found Nifty50 and it's not in the list, ensure it's included
            if nifty_prediction and not nifty_included:
                # Parse prediction_details for Nifty50
                prediction_details = nifty_prediction.get('prediction_details')
                if prediction_details:
                    try:
                        # PostgreSQL JSONB might return as dict or string depending on psycopg2 version
                        if isinstance(prediction_details, str):
                            details = json.loads(prediction_details)
                        elif isinstance(prediction_details, dict):
                            details = prediction_details
                        else:
                            # Try to convert to string first
                            details = json.loads(str(prediction_details))
                        
                        cv_metrics = details.get('cv_metrics')
                        if cv_metrics and isinstance(cv_metrics, dict):
                            # Handle None, infinity, or invalid values
                            mape_value = cv_metrics.get('mape')
                            rmse_value = cv_metrics.get('rmse')
                            
                            # Convert infinity strings or infinity values to None
                            if mape_value is None or mape_value == float('inf') or mape_value == float('-inf') or (isinstance(mape_value, str) and mape_value.lower() in ['inf', 'infinity', 'nan']):
                                nifty_prediction['cv_mape'] = None
                            else:
                                try:
                                    nifty_prediction['cv_mape'] = float(mape_value) if mape_value is not None else None
                                except (ValueError, TypeError):
                                    nifty_prediction['cv_mape'] = None
                            
                            if rmse_value is None or rmse_value == float('inf') or rmse_value == float('-inf') or (isinstance(rmse_value, str) and rmse_value.lower() in ['inf', 'infinity', 'nan']):
                                nifty_prediction['cv_rmse'] = None
                            else:
                                try:
                                    nifty_prediction['cv_rmse'] = float(rmse_value) if rmse_value is not None else None
                                except (ValueError, TypeError):
                                    nifty_prediction['cv_rmse'] = None
                    except (json.JSONDecodeError, Exception) as e:
                        logger.warning(f"Error parsing prediction_details for Nifty50: {e}, type={type(prediction_details)}")
                        nifty_prediction['cv_mape'] = None
                        nifty_prediction['cv_rmse'] = None
                else:
                    nifty_prediction['cv_mape'] = None
                    nifty_prediction['cv_rmse'] = None
                
                # Remove the lowest gainer if we have exactly 'limit' items
                if len(top_gainers) >= limit:
                    # Sort to ensure we remove the lowest
                    top_gainers.sort(key=lambda x: x.get('predicted_price_change_pct', 0), reverse=True)
                    top_gainers = top_gainers[:limit-1]  # Keep top (limit-1) items
                
                # Add Nifty50
                top_gainers.append(nifty_prediction)
                # Re-sort by predicted_price_change_pct descending
                top_gainers.sort(key=lambda x: x.get('predicted_price_change_pct', 0), reverse=True)
                logger.info(f"Included Nifty50 ({nifty_prediction.get('scrip_id')}) in top gainers")
            elif nifty_included:
                logger.debug("Nifty50 already included in top gainers")
            
            return top_gainers
            
        except Exception as e:
            logger.error(f"Error getting top gainers: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
