"""
Back-testing Performance Metrics Calculator
Calculates comprehensive performance metrics for back-tested options trades
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import date, timedelta


class BacktestMetrics:
    """
    Calculate performance metrics for back-tested options trades
    """
    
    def __init__(self):
        """Initialize Backtest Metrics Calculator"""
        pass
    
    def calculate_metrics(self, trades: List[Dict]) -> Dict:
        """
        Calculate comprehensive performance metrics from back-tested trades
        
        Args:
            trades: List of trade dictionaries with:
                - entry_date: Entry date
                - exit_date: Exit date
                - entry_price: Entry price
                - exit_price: Exit price
                - profit_loss: Profit/Loss amount
                - is_generated: Whether data was generated (True) or historical (False)
                - Other trade details
        
        Returns:
            Dictionary with performance metrics
        """
        if not trades:
            return self._empty_metrics()
        
        try:
            # Convert to DataFrame for easier calculations
            df = pd.DataFrame(trades)
            
            # Basic metrics
            total_trades = len(trades)
            profitable_trades = df[df['profit_loss'] > 0]
            losing_trades = df[df['profit_loss'] <= 0]
            
            win_count = len(profitable_trades)
            loss_count = len(losing_trades)
            win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0.0
            
            # Profit/Loss metrics - convert numpy types to Python native types
            total_profit_loss_val = df['profit_loss'].sum()
            total_profit_loss = float(total_profit_loss_val) if pd.notna(total_profit_loss_val) else 0.0
            
            avg_profit_loss_val = df['profit_loss'].mean()
            avg_profit_loss = float(avg_profit_loss_val) if pd.notna(avg_profit_loss_val) else 0.0
            
            gross_profit_val = profitable_trades['profit_loss'].sum() if not profitable_trades.empty else 0.0
            gross_profit = float(gross_profit_val) if pd.notna(gross_profit_val) else 0.0
            
            gross_loss_val = abs(losing_trades['profit_loss'].sum()) if not losing_trades.empty else 0.0
            gross_loss = float(gross_loss_val) if pd.notna(gross_loss_val) else 0.0
            
            avg_win_val = profitable_trades['profit_loss'].mean() if not profitable_trades.empty else 0.0
            avg_win = float(avg_win_val) if pd.notna(avg_win_val) else 0.0
            
            avg_loss_val = losing_trades['profit_loss'].mean() if not losing_trades.empty else 0.0
            avg_loss = float(avg_loss_val) if pd.notna(avg_loss_val) else 0.0
            
            max_profit_val = df['profit_loss'].max()
            max_profit = float(max_profit_val) if pd.notna(max_profit_val) else 0.0
            
            max_loss_val = df['profit_loss'].min()
            max_loss = float(max_loss_val) if pd.notna(max_loss_val) else 0.0
            
            # Profit factor
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
            
            # Best and worst trades
            best_trade = {}
            worst_trade = {}
            if not df.empty:
                try:
                    best_idx = df['profit_loss'].idxmax()
                    worst_idx = df['profit_loss'].idxmin()
                    best_trade = df.loc[best_idx].to_dict()
                    worst_trade = df.loc[worst_idx].to_dict()
                    # Convert numpy types to native Python types
                    for key, value in best_trade.items():
                        if hasattr(value, 'item'):  # numpy scalar
                            best_trade[key] = value.item()
                        elif isinstance(value, (np.integer, np.int64, np.int32)):
                            best_trade[key] = int(value)
                        elif isinstance(value, (np.floating, np.float64, np.float32)):
                            best_trade[key] = float(value) if not pd.isna(value) else None
                    for key, value in worst_trade.items():
                        if hasattr(value, 'item'):  # numpy scalar
                            worst_trade[key] = value.item()
                        elif isinstance(value, (np.integer, np.int64, np.int32)):
                            worst_trade[key] = int(value)
                        elif isinstance(value, (np.floating, np.float64, np.float32)):
                            worst_trade[key] = float(value) if not pd.isna(value) else None
                except Exception as e:
                    logging.warning(f"Error extracting best/worst trades: {e}")
            
            # Holding period metrics
            if 'entry_date' in df.columns and 'exit_date' in df.columns:
                try:
                    df['holding_period'] = (pd.to_datetime(df['exit_date']) - 
                                           pd.to_datetime(df['entry_date'])).dt.days
                    avg_holding_period_val = df['holding_period'].mean()
                    avg_holding_period = float(avg_holding_period_val) if pd.notna(avg_holding_period_val) else 0.0
                except Exception as e:
                    logging.warning(f"Error calculating holding period: {e}")
                    avg_holding_period = 0.0
            else:
                avg_holding_period = 0.0
            
            # Sharpe ratio - convert numpy array to list to avoid type issues
            profit_loss_values = df['profit_loss'].values
            try:
                if hasattr(profit_loss_values, 'tolist'):
                    profit_loss_list = profit_loss_values.tolist()
                else:
                    profit_loss_list = [float(v) for v in profit_loss_values if pd.notna(v)]
                sharpe_ratio = self._calculate_sharpe_ratio(np.array(profit_loss_list))
            except Exception as e:
                logging.warning(f"Error calculating Sharpe ratio: {e}")
                sharpe_ratio = 0.0
            
            # Maximum drawdown
            try:
                max_drawdown = self._calculate_max_drawdown(np.array(profit_loss_list))
            except Exception as e:
                logging.warning(f"Error calculating max drawdown: {e}")
                max_drawdown = 0.0
            
            # Data quality metrics
            if 'is_generated' in df.columns:
                try:
                    generated_count_val = df['is_generated'].sum()
                    generated_count = int(generated_count_val) if pd.notna(generated_count_val) else 0
                    historical_count = total_trades - generated_count
                    generated_percentage = (generated_count / total_trades * 100) if total_trades > 0 else 0.0
                    historical_percentage = (historical_count / total_trades * 100) if total_trades > 0 else 0.0
                except Exception as e:
                    logging.warning(f"Error calculating data quality metrics: {e}")
                    generated_count = 0
                    historical_count = total_trades
                    generated_percentage = 0.0
                    historical_percentage = 100.0
            else:
                generated_count = 0
                historical_count = total_trades
                generated_percentage = 0.0
                historical_percentage = 100.0
            
            # Confidence score based on data quality
            confidence_score = self._calculate_confidence_score(
                historical_percentage, total_trades, win_rate
            )
            
            return {
                'total_trades': total_trades,
                'win_count': win_count,
                'loss_count': loss_count,
                'win_rate': round(win_rate, 2),
                'total_profit_loss': round(total_profit_loss, 2),
                'avg_profit_loss': round(avg_profit_loss, 2),
                'gross_profit': round(gross_profit, 2),
                'gross_loss': round(gross_loss, 2),
                'avg_win': round(avg_win, 2),
                'avg_loss': round(avg_loss, 2),
                'max_profit': round(max_profit, 2),
                'max_loss': round(max_loss, 2),
                'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else float('inf'),
                'sharpe_ratio': round(sharpe_ratio, 2),
                'max_drawdown': round(max_drawdown, 2),
                'avg_holding_period': round(avg_holding_period, 1),
                'best_trade': best_trade,
                'worst_trade': worst_trade,
                'data_quality': {
                    'generated_count': generated_count,
                    'historical_count': historical_count,
                    'generated_percentage': round(generated_percentage, 2),
                    'historical_percentage': round(historical_percentage, 2)
                },
                'confidence_score': round(confidence_score, 2)
            }
            
        except Exception as e:
            logging.error(f"Error calculating back-testing metrics: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return self._empty_metrics()
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.065) -> float:
        """
        Calculate Sharpe ratio
        
        Args:
            returns: Array of returns/profit-loss values
            risk_free_rate: Risk-free rate (annual, default: 6.5%)
            
        Returns:
            Sharpe ratio
        """
        try:
            if returns is None or len(returns) == 0:
                return 0.0
            
            # Ensure returns is a numpy array
            if not isinstance(returns, np.ndarray):
                returns = np.array(returns, dtype=float)
            
            # Filter out invalid values
            returns = returns[np.isfinite(returns)]
            if len(returns) == 0:
                return 0.0
            
            # Annualize returns (assuming daily trades)
            mean_return = float(np.mean(returns))
            std_return = float(np.std(returns))
            
            if std_return == 0 or not np.isfinite(std_return):
                return 0.0
            
            # Annualize
            annualized_return = mean_return * 252  # 252 trading days
            annualized_std = std_return * np.sqrt(252)
            
            if annualized_std == 0 or not np.isfinite(annualized_std):
                return 0.0
            
            # Sharpe ratio = (Return - Risk-free rate) / Std Dev
            sharpe = (annualized_return - risk_free_rate) / annualized_std
            
            return float(sharpe) if np.isfinite(sharpe) else 0.0
        except Exception as e:
            logging.warning(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """
        Calculate maximum drawdown
        
        Args:
            returns: Array of returns/profit-loss values
        
        Returns:
            Maximum drawdown (negative value)
        """
        try:
            if returns is None or len(returns) == 0:
                return 0.0
            
            # Ensure returns is a numpy array
            if not isinstance(returns, np.ndarray):
                returns = np.array(returns, dtype=float)
            
            # Filter out invalid values
            returns = returns[np.isfinite(returns)]
            if len(returns) == 0:
                return 0.0
            
            # Calculate cumulative returns
            cumulative = np.cumsum(returns)
            
            # Calculate running maximum
            running_max = np.maximum.accumulate(cumulative)
            
            # Calculate drawdown
            drawdown = cumulative - running_max
            
            # Maximum drawdown (most negative)
            max_dd = float(np.min(drawdown))
            
            return max_dd if np.isfinite(max_dd) else 0.0
        except Exception as e:
            logging.warning(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def _calculate_confidence_score(self, 
                                   historical_percentage: float,
                                   total_trades: int,
                                   win_rate: float) -> float:
        """
        Calculate confidence score based on data quality and results
        
        Args:
            historical_percentage: Percentage of trades using historical data
            total_trades: Total number of trades
            win_rate: Win rate percentage
        
        Returns:
            Confidence score (0-100)
        """
        # Base score from data quality (0-60 points)
        data_quality_score = (historical_percentage / 100.0) * 60
        
        # Sample size score (0-20 points)
        if total_trades >= 100:
            sample_score = 20
        elif total_trades >= 50:
            sample_score = 15
        elif total_trades >= 20:
            sample_score = 10
        else:
            sample_score = 5
        
        # Win rate score (0-20 points)
        # Higher win rate = higher confidence
        if win_rate >= 60:
            win_rate_score = 20
        elif win_rate >= 50:
            win_rate_score = 15
        elif win_rate >= 40:
            win_rate_score = 10
        else:
            win_rate_score = 5
        
        total_score = data_quality_score + sample_score + win_rate_score
        
        return min(100.0, max(0.0, total_score))
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics structure"""
        return {
            'total_trades': 0,
            'win_count': 0,
            'loss_count': 0,
            'win_rate': 0.0,
            'total_profit_loss': 0.0,
            'avg_profit_loss': 0.0,
            'gross_profit': 0.0,
            'gross_loss': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'max_profit': 0.0,
            'max_loss': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'avg_holding_period': 0.0,
            'best_trade': {},
            'worst_trade': {},
            'data_quality': {
                'generated_count': 0,
                'historical_count': 0,
                'generated_percentage': 0.0,
                'historical_percentage': 0.0
            },
            'confidence_score': 0.0
        }
    
    def filter_profitable_trades(self, trades: List[Dict]) -> List[Dict]:
        """
        Filter to show only profitable trades
        
        Args:
            trades: List of trade dictionaries
        
        Returns:
            Filtered list of profitable trades
        """
        return [trade for trade in trades if trade.get('profit_loss', 0) > 0]
    
    def filter_by_min_profit(self, trades: List[Dict], min_profit: float) -> List[Dict]:
        """
        Filter trades by minimum profit threshold
        
        Args:
            trades: List of trade dictionaries
            min_profit: Minimum profit threshold
        
        Returns:
            Filtered list of trades
        """
        return [trade for trade in trades if trade.get('profit_loss', 0) >= min_profit]

