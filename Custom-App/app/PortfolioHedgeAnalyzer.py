"""
Portfolio Hedging Module
Calculates portfolio Beta, correlation with indices, and suggests hedging strategies
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from Boilerplate import get_db_connection

class PortfolioHedgeAnalyzer:
    """
    Analyzes portfolio risk and suggests hedging strategies
    """
    
    def __init__(self, db_config: Dict = None):
        """
        Initialize Portfolio Hedge Analyzer
        
        Args:
            db_config: Database configuration dict (host, database, user, password, port)
        """
        if db_config is None:
            # Default to environment variables
            import os
            db_config = {
                'host': os.getenv('PG_HOST', 'postgres'),
                'database': os.getenv('PG_DATABASE', 'mydb'),
                'user': os.getenv('PG_USER', 'postgres'),
                'password': os.getenv('PG_PASSWORD', 'postgres'),
                'port': int(os.getenv('PG_PORT', 5432))
            }
        self.db_config = db_config
        
        # Nifty 50 instrument token
        self.nifty_token = 256265
        
    def get_current_holdings(self) -> pd.DataFrame:
        """
        Get current equity holdings from database
        
        Returns:
            DataFrame with holdings data
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    trading_symbol,
                    instrument_token,
                    quantity,
                    average_price,
                    COALESCE(last_price, close_price, 0) as current_price,
                    pnl
                FROM my_schema.holdings
                WHERE run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
                AND quantity > 0
            """)
            
            rows = cursor.fetchall()
            columns = ['trading_symbol', 'instrument_token', 'quantity', 
                      'average_price', 'current_price', 'pnl']
            
            conn.close()
            
            if not rows:
                return pd.DataFrame(columns=columns)
            
            df = pd.DataFrame(rows, columns=columns)
            df['current_value'] = df['quantity'] * df['current_price']
            df['invested_value'] = df['quantity'] * df['average_price']
            
            return df
        except Exception as e:
            logging.error(f"Error fetching holdings: {e}")
            return pd.DataFrame()
    
    def calculate_portfolio_beta(self, lookback_days: int = 60) -> Dict:
        """
        Calculate portfolio Beta with respect to Nifty 50
        
        Args:
            lookback_days: Number of days to look back for correlation
            
        Returns:
            Dictionary with beta, correlation, and other metrics
        """
        try:
            holdings = self.get_current_holdings()
            if holdings.empty:
                return {
                    'beta': 0.0,
                    'correlation': 0.0,
                    'portfolio_value': 0.0,
                    'error': 'No holdings found'
                }
            
            # Get historical price data for portfolio and Nifty
            portfolio_returns = []
            nifty_returns = []
            
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get date range
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=lookback_days)
            
            # Calculate portfolio daily returns
            dates = []
            for date in pd.date_range(start_date, end_date):
                date_str = date.strftime('%Y-%m-%d')
                
                # Get portfolio value for this date
                portfolio_value = 0
                for _, holding in holdings.iterrows():
                    cursor.execute("""
                        SELECT price_close
                        FROM my_schema.rt_intraday_price
                        WHERE scrip_id = %s
                        AND price_date = %s
                        ORDER BY price_date DESC
                        LIMIT 1
                    """, (holding['trading_symbol'], date_str))
                    result = cursor.fetchone()
                    if result and result[0]:
                        portfolio_value += holding['quantity'] * float(result[0])
                
                if portfolio_value > 0:
                    dates.append(date_str)
                    portfolio_returns.append(portfolio_value)
            
            # Get Nifty returns
            cursor.execute("""
                SELECT price_date, price_close
                FROM my_schema.rt_intraday_price
                WHERE scrip_id = 'NIFTY 50'
                AND price_date >= %s
                AND price_date <= %s
                ORDER BY price_date
            """, (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
            
            nifty_data = cursor.fetchall()
            conn.close()
            
            if len(portfolio_returns) < 10 or len(nifty_data) < 10:
                return {
                    'beta': 0.0,
                    'correlation': 0.0,
                    'portfolio_value': float(holdings['current_value'].sum()),
                    'error': 'Insufficient data for beta calculation'
                }
            
            # Convert to returns (percentage change)
            portfolio_df = pd.DataFrame({
                'date': dates[:len(portfolio_returns)],
                'value': portfolio_returns
            })
            portfolio_df['returns'] = portfolio_df['value'].pct_change()
            
            nifty_df = pd.DataFrame(nifty_data, columns=['date', 'price'])
            nifty_df['returns'] = nifty_df['price'].pct_change()
            
            # Align dates
            merged = pd.merge(
                portfolio_df[['date', 'returns']],
                nifty_df[['date', 'returns']],
                on='date',
                suffixes=('_portfolio', '_nifty')
            )
            
            merged = merged.dropna()
            
            if len(merged) < 10:
                return {
                    'beta': 0.0,
                    'correlation': 0.0,
                    'portfolio_value': float(holdings['current_value'].sum()),
                    'error': 'Insufficient aligned data'
                }
            
            # Calculate correlation
            correlation = merged['returns_portfolio'].corr(merged['returns_nifty'])
            
            # Calculate beta: Cov(Portfolio, Nifty) / Var(Nifty)
            covariance = merged['returns_portfolio'].cov(merged['returns_nifty'])
            variance_nifty = merged['returns_nifty'].var()
            
            beta = covariance / variance_nifty if variance_nifty > 0 else 0.0
            
            current_portfolio_value = float(holdings['current_value'].sum())
            
            return {
                'beta': float(beta),
                'correlation': float(correlation),
                'portfolio_value': current_portfolio_value,
                'days_analyzed': len(merged),
                'portfolio_volatility': float(merged['returns_portfolio'].std()),
                'nifty_volatility': float(merged['returns_nifty'].std())
            }
            
        except Exception as e:
            logging.error(f"Error calculating portfolio beta: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return {
                'beta': 0.0,
                'correlation': 0.0,
                'portfolio_value': 0.0,
                'error': str(e)
            }
    
    def suggest_hedge(self, target_hedge_ratio: float = 0.5) -> Dict:
        """
        Suggest hedging strategy for portfolio
        
        Args:
            target_hedge_ratio: Target hedge ratio (0.0 to 1.0), e.g., 0.5 for 50% hedge
            
        Returns:
            Dictionary with hedge suggestions
        """
        try:
            holdings = self.get_current_holdings()
            if holdings.empty:
                return {
                    'suggested_instruments': [],
                    'total_hedge_value': 0.0,
                    'error': 'No holdings found'
                }
            
            # Calculate portfolio metrics
            beta_result = self.calculate_portfolio_beta()
            
            if 'error' in beta_result:
                return {
                    'suggested_instruments': [],
                    'total_hedge_value': 0.0,
                    'error': beta_result['error']
                }
            
            portfolio_value = beta_result['portfolio_value']
            beta = beta_result['beta']
            correlation = beta_result['correlation']
            
            # Calculate unhedged exposure
            unhedged_exposure = portfolio_value * beta * (1 - target_hedge_ratio)
            
            # Get current Nifty futures price
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Try to get Nifty futures price from ticks or quotes
            cursor.execute("""
                SELECT last_price
                FROM my_schema.ticks
                WHERE instrument_token = %s
                ORDER BY timestamp DESC
                LIMIT 1
            """, (self.nifty_token,))
            
            result = cursor.fetchone()
            nifty_futures_price = None
            if result and result[0]:
                nifty_futures_price = float(result[0])
            
            conn.close()
            
            # If we can't get futures price, estimate from Nifty index
            if not nifty_futures_price or nifty_futures_price == 0:
                # Try to get from OHLC
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT price_close
                    FROM my_schema.rt_intraday_price
                    WHERE scrip_id = 'NIFTY 50'
                    AND price_date = (
                        SELECT MAX(price_date) 
                        FROM my_schema.rt_intraday_price 
                        WHERE scrip_id = 'NIFTY 50'
                    )
                    ORDER BY price_date DESC
                    LIMIT 1
                """)
                result = cursor.fetchone()
                if result and result[0]:
                    nifty_futures_price = float(result[0])
                conn.close()
            
            if not nifty_futures_price:
                return {
                    'suggested_instruments': [],
                    'total_hedge_value': 0.0,
                    'error': 'Cannot determine Nifty price'
                }
            
            # Nifty futures lot size
            lot_size = 50
            
            # Calculate required hedge quantity
            # Hedge value = Portfolio Value * Beta * Target Hedge Ratio
            hedge_value = portfolio_value * beta * target_hedge_ratio
            hedge_quantity = int((hedge_value / nifty_futures_price) / lot_size) * lot_size
            
            if hedge_quantity <= 0:
                return {
                    'suggested_instruments': [],
                    'total_hedge_value': 0.0,
                    'error': 'Insufficient exposure to hedge'
                }
            
            # Calculate margin requirement (approximately 12% of contract value)
            contract_value = hedge_quantity * nifty_futures_price
            estimated_margin = contract_value * 0.12
            
            suggestions = {
                'suggested_instruments': [
                    {
                        'instrument': 'NIFTY FUT',
                        'instrument_token': self.nifty_token,
                        'direction': 'SHORT',  # Short futures to hedge long portfolio
                        'quantity': hedge_quantity,
                        'lot_size': lot_size,
                        'entry_price': nifty_futures_price,
                        'contract_value': float(contract_value),
                        'estimated_margin': float(estimated_margin),
                        'hedge_value': float(hedge_value),
                        'rationale': f'Hedge {target_hedge_ratio*100:.0f}% of portfolio exposure (Beta: {beta:.2f})'
                    }
                ],
                'portfolio_value': float(portfolio_value),
                'portfolio_beta': float(beta),
                'portfolio_correlation': float(correlation),
                'unhedged_exposure': float(unhedged_exposure),
                'total_hedge_value': float(hedge_value),
                'target_hedge_ratio': target_hedge_ratio,
                'hedge_effectiveness_score': float(correlation * 100) if correlation else 0.0,
                'cost_of_hedging': float(estimated_margin),
                'estimated_premium': 0.0,  # Futures don't require premium
                'margin_required': float(estimated_margin)
            }
            
            return suggestions
            
        except Exception as e:
            logging.error(f"Error suggesting hedge: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return {
                'suggested_instruments': [],
                'total_hedge_value': 0.0,
                'error': str(e)
            }
    
    def calculate_var(self, confidence_level: float = 0.95, time_horizon: int = 1) -> Dict:
        """
        Calculate Value at Risk (VaR) for portfolio
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            time_horizon: Time horizon in days
            
        Returns:
            Dictionary with VaR metrics
        """
        try:
            holdings = self.get_current_holdings()
            if holdings.empty:
                return {
                    'var_95': 0.0,
                    'var_99': 0.0,
                    'error': 'No holdings found'
                }
            
            portfolio_value = float(holdings['current_value'].sum())
            
            # Calculate portfolio volatility (simplified approach)
            beta_result = self.calculate_portfolio_beta()
            portfolio_vol = beta_result.get('portfolio_volatility', 0.02)  # Default 2% daily
            
            if portfolio_vol == 0:
                portfolio_vol = 0.02  # Default assumption
            
            # VaR calculation (simplified)
            # VaR = Portfolio Value * Volatility * Z-score * sqrt(time_horizon)
            z_score_95 = 1.645  # 95% confidence
            z_score_99 = 2.326  # 99% confidence
            
            var_95 = portfolio_value * portfolio_vol * z_score_95 * np.sqrt(time_horizon)
            var_99 = portfolio_value * portfolio_vol * z_score_99 * np.sqrt(time_horizon)
            
            return {
                'var_95': float(var_95),
                'var_99': float(var_99),
                'portfolio_value': float(portfolio_value),
                'portfolio_volatility': float(portfolio_vol),
                'confidence_level': confidence_level,
                'time_horizon': time_horizon
            }
            
        except Exception as e:
            logging.error(f"Error calculating VaR: {e}")
            return {
                'var_95': 0.0,
                'var_99': 0.0,
                'error': str(e)
            }

