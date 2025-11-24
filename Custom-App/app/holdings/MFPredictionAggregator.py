"""
Mutual Fund Prediction Aggregator
Aggregates stock predictions to predict mutual fund performance
"""
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from common.Boilerplate import get_db_connection
import psycopg2
import psycopg2.extras
from holdings.MFPortfolioFetcher import MFPortfolioFetcher

logger = logging.getLogger(__name__)


class MFPredictionAggregator:
    """Aggregates stock predictions to predict mutual fund NAV changes"""
    
    def __init__(self):
        """Initialize MF Prediction Aggregator"""
        self.portfolio_fetcher = MFPortfolioFetcher()
    
    def get_mf_constituents(self, mf_symbol: str) -> List[Dict]:
        """
        Get latest portfolio holdings (constituents) for a mutual fund
        
        Args:
            mf_symbol: Mutual Fund trading symbol
            
        Returns:
            List of portfolio holdings with weights
        """
        try:
            return self.portfolio_fetcher.get_latest_portfolio(mf_symbol)
        except Exception as e:
            logging.error(f"Error getting MF constituents for {mf_symbol}: {e}")
            return []
    
    def get_stock_predictions(self, stock_symbols: List[str], prediction_days: int = 30) -> Dict[str, Dict]:
        """
        Batch fetch Prophet predictions for multiple stocks
        
        Args:
            stock_symbols: List of stock symbols
            prediction_days: Number of days for prediction (default: 30)
            
        Returns:
            Dictionary mapping stock_symbol to prediction data
        """
        if not stock_symbols:
            return {}
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Get latest predictions for all stocks
            # Note: The schema uses predicted_price_30d for 30-day predictions
            # For other prediction periods, we may need to adjust the column name
            price_column = 'predicted_price_30d' if prediction_days == 30 else 'predicted_price_30d'  # Default to 30d for now
            
            cursor.execute(f"""
                SELECT DISTINCT ON (scrip_id)
                    scrip_id,
                    {price_column} as predicted_price,
                    predicted_price_change_pct,
                    prediction_confidence,
                    run_date,
                    prediction_days
                FROM my_schema.prophet_predictions
                WHERE scrip_id = ANY(%s)
                AND prediction_days = %s
                AND status = 'ACTIVE'
                ORDER BY scrip_id, run_date DESC
            """, (stock_symbols, prediction_days))
            
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            
            predictions = {}
            for row in rows:
                stock_symbol = row['scrip_id']
                predictions[stock_symbol] = {
                    'predicted_price': float(row['predicted_price']) if row.get('predicted_price') else None,
                    'predicted_price_change_pct': float(row['predicted_price_change_pct']) if row.get('predicted_price_change_pct') else None,
                    'prediction_confidence': float(row.get('prediction_confidence')) if row.get('prediction_confidence') else None,
                    'run_date': row['run_date'],
                    'prediction_days': row['prediction_days']
                }
            
            logging.info(f"Retrieved predictions for {len(predictions)} out of {len(stock_symbols)} stocks")
            return predictions
            
        except Exception as e:
            logging.error(f"Error getting stock predictions: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return {}
    
    def get_current_nav(self, mf_symbol: str) -> Optional[float]:
        """
        Get current NAV for a mutual fund
        
        Args:
            mf_symbol: Mutual Fund trading symbol
            
        Returns:
            Current NAV value or None
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get latest NAV from mf_nav_history
            cursor.execute("""
                SELECT nav_value
                FROM my_schema.mf_nav_history
                WHERE mf_symbol = %s
                ORDER BY nav_date DESC
                LIMIT 1
            """, (mf_symbol,))
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result and result[0]:
                return float(result[0])
            
            # Fallback to mf_holdings last_price
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT last_price
                FROM my_schema.mf_holdings
                WHERE tradingsymbol = %s
                AND run_date = (SELECT MAX(run_date) FROM my_schema.mf_holdings)
                LIMIT 1
            """, (mf_symbol,))
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result and result[0]:
                return float(result[0])
            
            return None
            
        except Exception as e:
            logging.error(f"Error getting current NAV for {mf_symbol}: {e}")
            return None
    
    def aggregate_predictions(self, mf_symbol: str, prediction_days: int = 30) -> Dict:
        """
        Aggregate stock predictions to predict MF NAV change (main method)
        
        Args:
            mf_symbol: Mutual Fund trading symbol
            prediction_days: Number of days for prediction (default: 30)
            
        Returns:
            Dictionary with aggregated prediction results
        """
        try:
            # Get MF constituents
            constituents = self.get_mf_constituents(mf_symbol)
            
            if not constituents:
                return {
                    'success': False,
                    'error': f'No portfolio holdings found for {mf_symbol}',
                    'mf_symbol': mf_symbol,
                    'prediction_days': prediction_days
                }
            
            # Get stock symbols
            stock_symbols = [c['stock_symbol'] for c in constituents if c.get('stock_symbol')]
            
            if not stock_symbols:
                return {
                    'success': False,
                    'error': f'No valid stock symbols in portfolio for {mf_symbol}',
                    'mf_symbol': mf_symbol,
                    'prediction_days': prediction_days
                }
            
            # Get predictions for all stocks
            stock_predictions = self.get_stock_predictions(stock_symbols, prediction_days)
            
            # Calculate weighted averages
            total_weight = 0.0
            weighted_price_change = 0.0
            weighted_confidence = 0.0
            total_weight_with_prediction = 0.0
            
            constituent_predictions = []
            stocks_without_predictions = []
            
            for constituent in constituents:
                stock_symbol = constituent.get('stock_symbol')
                weight_pct = float(constituent.get('weight_pct', 0) or 0)
                
                if not stock_symbol or weight_pct <= 0:
                    continue
                
                total_weight += weight_pct
                
                prediction = stock_predictions.get(stock_symbol)
                
                if prediction and prediction.get('predicted_price_change_pct') is not None:
                    price_change = prediction['predicted_price_change_pct']
                    confidence = prediction.get('prediction_confidence', 0) or 0
                    
                    weighted_price_change += weight_pct * price_change
                    weighted_confidence += weight_pct * confidence
                    total_weight_with_prediction += weight_pct
                    
                    constituent_predictions.append({
                        'stock_symbol': stock_symbol,
                        'stock_name': constituent.get('stock_name', ''),
                        'weight_pct': weight_pct,
                        'predicted_price_change_pct': price_change,
                        'prediction_confidence': confidence,
                        'predicted_price': prediction.get('predicted_price')
                    })
                else:
                    stocks_without_predictions.append({
                        'stock_symbol': stock_symbol,
                        'stock_name': constituent.get('stock_name', ''),
                        'weight_pct': weight_pct
                    })
            
            # Calculate final aggregated values
            if total_weight_with_prediction > 0:
                aggregated_price_change_pct = weighted_price_change / total_weight_with_prediction
                aggregated_confidence = weighted_confidence / total_weight_with_prediction
            else:
                aggregated_price_change_pct = 0.0
                aggregated_confidence = 0.0
            
            # Get current NAV
            current_nav = self.get_current_nav(mf_symbol)
            
            # Calculate predicted NAV
            predicted_nav = None
            if current_nav and aggregated_price_change_pct is not None:
                predicted_nav = current_nav * (1 + aggregated_price_change_pct / 100.0)
            
            # Calculate coverage (percentage of portfolio with predictions)
            coverage_pct = (total_weight_with_prediction / total_weight * 100) if total_weight > 0 else 0.0
            
            return {
                'success': True,
                'mf_symbol': mf_symbol,
                'prediction_days': prediction_days,
                'current_nav': current_nav,
                'predicted_price_change_pct': aggregated_price_change_pct,
                'predicted_nav': predicted_nav,
                'prediction_confidence': aggregated_confidence,
                'coverage_pct': coverage_pct,
                'total_portfolio_weight': total_weight,
                'weight_with_predictions': total_weight_with_prediction,
                'constituents_count': len(constituents),
                'constituents_with_predictions': len(constituent_predictions),
                'constituents_without_predictions': len(stocks_without_predictions),
                'constituent_predictions': constituent_predictions,
                'stocks_without_predictions': stocks_without_predictions,
                'portfolio_date': constituents[0].get('portfolio_date') if constituents else None
            }
            
        except Exception as e:
            logging.error(f"Error aggregating predictions for {mf_symbol}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'mf_symbol': mf_symbol,
                'prediction_days': prediction_days
            }
    
    def calculate_predicted_nav_change(self, mf_symbol: str, current_nav: float, 
                                       predicted_stock_change_pct: float) -> float:
        """
        Calculate predicted NAV based on weighted stock predictions
        
        Args:
            mf_symbol: Mutual Fund trading symbol
            current_nav: Current NAV value
            predicted_stock_change_pct: Weighted average predicted stock change percentage
            
        Returns:
            Predicted NAV value
        """
        if current_nav and predicted_stock_change_pct is not None:
            return current_nav * (1 + predicted_stock_change_pct / 100.0)
        return None
    
    def get_predictions_for_all_held_mfs(self, prediction_days: int = 30) -> Dict:
        """
        Get aggregated predictions for all mutual funds currently held
        
        Args:
            prediction_days: Number of days for prediction (default: 30)
            
        Returns:
            Dictionary with results for each MF
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get all unique MF symbols from mf_holdings
            cursor.execute("""
                SELECT DISTINCT tradingsymbol, fund
                FROM my_schema.mf_holdings
                WHERE run_date = (SELECT MAX(run_date) FROM my_schema.mf_holdings)
            """)
            
            mf_list = cursor.fetchall()
            cursor.close()
            conn.close()
            
            results = {
                'total_mfs': len(mf_list),
                'successful': 0,
                'failed': 0,
                'prediction_days': prediction_days,
                'results': []
            }
            
            for mf_symbol, fund_name in mf_list:
                if not mf_symbol:
                    continue
                
                logging.info(f"Generating prediction for {mf_symbol} ({fund_name})")
                result = self.aggregate_predictions(mf_symbol, prediction_days)
                
                if result.get('success'):
                    results['successful'] += 1
                else:
                    results['failed'] += 1
                
                result['fund_name'] = fund_name
                results['results'].append(result)
            
            logging.info(f"Prediction aggregation completed: {results['successful']} successful, {results['failed']} failed")
            return results
            
        except Exception as e:
            logging.error(f"Error getting predictions for all MFs: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return {
                'total_mfs': 0,
                'successful': 0,
                'failed': 0,
                'error': str(e),
                'prediction_days': prediction_days,
                'results': []
            }

