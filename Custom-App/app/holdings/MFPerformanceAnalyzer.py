"""
Mutual Fund Performance Analyzer
Calculates performance metrics comparing MF NAV to benchmark indices
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from common.Boilerplate import get_db_connection
from holdings.MFBenchmarkMapper import MFBenchmarkMapper


class MFPerformanceAnalyzer:
    """Analyzes Mutual Fund performance vs benchmarks"""
    
    def __init__(self):
        """Initialize performance analyzer"""
        self.benchmark_mapper = MFBenchmarkMapper()
    
    def get_mf_nav_data(self, mf_symbol: str, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """
        Get NAV data for a mutual fund from database
        
        Args:
            mf_symbol: Mutual Fund trading symbol
            start_date: Start date (optional)
            end_date: End date (optional, defaults to today)
            
        Returns:
            DataFrame with columns: nav_date, nav_value
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            if end_date is None:
                end_date = datetime.now().date()
            if isinstance(end_date, datetime):
                end_date = end_date.date()
            
            if start_date is None:
                start_date = end_date - timedelta(days=365)
            if isinstance(start_date, datetime):
                start_date = start_date.date()
            
            query = """
                SELECT nav_date, nav_value
                FROM my_schema.mf_nav_history
                WHERE mf_symbol = %s
                AND nav_date >= %s
                AND nav_date <= %s
                ORDER BY nav_date
            """
            
            cursor.execute(query, (mf_symbol, start_date, end_date))
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            
            if not rows:
                return pd.DataFrame()
            
            df = pd.DataFrame(rows, columns=['nav_date', 'nav_value'])
            df['nav_date'] = pd.to_datetime(df['nav_date'])
            df.set_index('nav_date', inplace=True)
            
            return df
            
        except Exception as e:
            logging.error(f"Error fetching NAV data for {mf_symbol}: {e}")
            return pd.DataFrame()
    
    def get_benchmark_data(self, benchmark_symbol: str, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """
        Get benchmark price data from database
        
        Args:
            benchmark_symbol: Benchmark symbol (e.g., 'NIFTY50')
            start_date: Start date (optional)
            end_date: End date (optional, defaults to today)
            
        Returns:
            DataFrame with columns: price_date, price_close
        """
        try:
            # Get Yahoo code from mapper
            mapping = self.benchmark_mapper.map_to_benchmark('', '')
            yahoo_code = self.benchmark_mapper._get_benchmark_yahoo_code(benchmark_symbol)
            
            # Get from master_scrips to find scrip_id
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT scrip_id FROM my_schema.master_scrips
                WHERE yahoo_code = %s OR scrip_id = %s
                LIMIT 1
            """, (yahoo_code, benchmark_symbol))
            
            result = cursor.fetchone()
            if not result:
                cursor.close()
                conn.close()
                return pd.DataFrame()
            
            scrip_id = result[0]
            
            if end_date is None:
                end_date = datetime.now().date()
            if isinstance(end_date, datetime):
                end_date = end_date.date()
            
            if start_date is None:
                start_date = end_date - timedelta(days=365)
            if isinstance(start_date, datetime):
                start_date = start_date.date()
            
            query = """
                SELECT price_date, price_close
                FROM my_schema.rt_intraday_price
                WHERE scrip_id = %s
                AND price_date >= %s
                AND price_date <= %s
                ORDER BY price_date
            """
            
            cursor.execute(query, (scrip_id, str(start_date), str(end_date)))
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            
            if not rows:
                return pd.DataFrame()
            
            df = pd.DataFrame(rows, columns=['price_date', 'price_close'])
            df['price_date'] = pd.to_datetime(df['price_date'])
            df.set_index('price_date', inplace=True)
            
            return df
            
        except Exception as e:
            logging.error(f"Error fetching benchmark data for {benchmark_symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """
        Calculate daily returns from price series
        
        Args:
            prices: Series of prices
            
        Returns:
            Series of returns
        """
        return prices.pct_change().dropna()
    
    def calculate_alpha_beta(self, mf_returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float = 0.06) -> Tuple[float, float]:
        """
        Calculate alpha and beta
        
        Args:
            mf_returns: MF returns series
            benchmark_returns: Benchmark returns series
            risk_free_rate: Annual risk-free rate (default: 6%)
            
        Returns:
            Tuple of (alpha, beta)
        """
        try:
            # Align dates
            aligned = pd.concat([mf_returns, benchmark_returns], axis=1, join='inner')
            aligned.columns = ['mf', 'benchmark']
            aligned = aligned.dropna()
            
            if len(aligned) < 2:
                return (0.0, 0.0)
            
            mf_ret = aligned['mf'].values
            bench_ret = aligned['benchmark'].values
            
            # Calculate beta using covariance/covariance
            covariance = np.cov(mf_ret, bench_ret)[0][1]
            benchmark_variance = np.var(bench_ret)
            
            if benchmark_variance == 0:
                beta = 0.0
            else:
                beta = covariance / benchmark_variance
            
            # Calculate alpha (annualized)
            # Alpha = (MF Return - Risk Free Rate) - Beta * (Benchmark Return - Risk Free Rate)
            mf_mean = np.mean(mf_ret) * 252  # Annualized
            bench_mean = np.mean(bench_ret) * 252  # Annualized
            alpha = (mf_mean - risk_free_rate) - beta * (bench_mean - risk_free_rate)
            
            return (float(alpha), float(beta))
            
        except Exception as e:
            logging.error(f"Error calculating alpha/beta: {e}")
            return (0.0, 0.0)
    
    def calculate_excess_returns(self, mf_nav: pd.Series, benchmark_price: pd.Series, periods: List[int] = [30, 90, 180, 365]) -> Dict[str, float]:
        """
        Calculate excess returns over different periods
        
        Args:
            mf_nav: MF NAV series
            benchmark_price: Benchmark price series
            periods: List of periods in days (default: [30, 90, 180, 365])
            
        Returns:
            Dictionary with period labels and excess returns
        """
        try:
            # Align dates
            aligned = pd.concat([mf_nav, benchmark_price], axis=1, join='inner')
            aligned.columns = ['mf', 'benchmark']
            aligned = aligned.dropna()
            
            if len(aligned) < 2:
                return {}
            
            results = {}
            
            for period in periods:
                if len(aligned) < period:
                    continue
                
                # Get most recent period
                period_data = aligned.tail(period)
                
                if len(period_data) < 2:
                    continue
                
                mf_start = period_data['mf'].iloc[0]
                mf_end = period_data['mf'].iloc[-1]
                bench_start = period_data['benchmark'].iloc[0]
                bench_end = period_data['benchmark'].iloc[-1]
                
                if mf_start > 0 and bench_start > 0:
                    mf_return = ((mf_end - mf_start) / mf_start) * 100
                    bench_return = ((bench_end - bench_start) / bench_start) * 100
                    excess_return = mf_return - bench_return
                    
                    if period == 30:
                        results['1M'] = excess_return
                    elif period == 90:
                        results['3M'] = excess_return
                    elif period == 180:
                        results['6M'] = excess_return
                    elif period == 365:
                        results['1Y'] = excess_return
            
            return results
            
        except Exception as e:
            logging.error(f"Error calculating excess returns: {e}")
            return {}
    
    def calculate_tracking_error(self, mf_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """
        Calculate tracking error (standard deviation of difference in returns)
        
        Args:
            mf_returns: MF returns series
            benchmark_returns: Benchmark returns series
            
        Returns:
            Tracking error (annualized)
        """
        try:
            # Align dates
            aligned = pd.concat([mf_returns, benchmark_returns], axis=1, join='inner')
            aligned.columns = ['mf', 'benchmark']
            aligned = aligned.dropna()
            
            if len(aligned) < 2:
                return 0.0
            
            # Calculate difference in returns
            diff = aligned['mf'] - aligned['benchmark']
            
            # Standard deviation (annualized)
            tracking_error = diff.std() * np.sqrt(252)
            
            return float(tracking_error)
            
        except Exception as e:
            logging.error(f"Error calculating tracking error: {e}")
            return 0.0
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.06) -> float:
        """
        Calculate Sharpe ratio
        
        Args:
            returns: Returns series
            risk_free_rate: Annual risk-free rate (default: 6%)
            
        Returns:
            Sharpe ratio
        """
        try:
            if len(returns) < 2:
                return 0.0
            
            # Annualized return
            mean_return = returns.mean() * 252
            
            # Annualized standard deviation
            std_return = returns.std() * np.sqrt(252)
            
            if std_return == 0:
                return 0.0
            
            sharpe = (mean_return - risk_free_rate) / std_return
            
            return float(sharpe)
            
        except Exception as e:
            logging.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def analyze_mf_performance(self, mf_symbol: str, fund_name: str = None, benchmark_symbol: str = None) -> Dict:
        """
        Analyze MF performance vs benchmark
        
        Args:
            mf_symbol: Mutual Fund trading symbol
            fund_name: Mutual Fund name (optional)
            benchmark_symbol: Benchmark symbol (optional, auto-detected if not provided)
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            # Get benchmark mapping
            if not benchmark_symbol:
                mapping = self.benchmark_mapper.map_to_benchmark(mf_symbol, fund_name)
                benchmark_symbol = mapping['benchmark_symbol']
            
            # Get data
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=365)
            
            mf_nav_df = self.get_mf_nav_data(mf_symbol, start_date, end_date)
            benchmark_df = self.get_benchmark_data(benchmark_symbol, start_date, end_date)
            
            if mf_nav_df.empty or benchmark_df.empty:
                return {
                    'success': False,
                    'error': 'Insufficient data for analysis',
                    'mf_symbol': mf_symbol,
                    'benchmark_symbol': benchmark_symbol
                }
            
            # Calculate returns
            mf_returns = self.calculate_returns(mf_nav_df['nav_value'])
            benchmark_returns = self.calculate_returns(benchmark_df['price_close'])
            
            # Calculate metrics
            alpha, beta = self.calculate_alpha_beta(mf_returns, benchmark_returns)
            excess_returns = self.calculate_excess_returns(mf_nav_df['nav_value'], benchmark_df['price_close'])
            tracking_error = self.calculate_tracking_error(mf_returns, benchmark_returns)
            mf_sharpe = self.calculate_sharpe_ratio(mf_returns)
            bench_sharpe = self.calculate_sharpe_ratio(benchmark_returns)
            
            return {
                'success': True,
                'mf_symbol': mf_symbol,
                'fund_name': fund_name,
                'benchmark_symbol': benchmark_symbol,
                'alpha': alpha,
                'beta': beta,
                'excess_returns': excess_returns,
                'tracking_error': tracking_error,
                'mf_sharpe': mf_sharpe,
                'benchmark_sharpe': bench_sharpe,
                'data_points': len(mf_nav_df)
            }
            
        except Exception as e:
            logging.error(f"Error analyzing MF performance for {mf_symbol}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'mf_symbol': mf_symbol,
                'benchmark_symbol': benchmark_symbol or 'Unknown'
            }

