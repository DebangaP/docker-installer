"""
Risk Metrics Calculator Module
Calculates comprehensive risk metrics for portfolio and individual holdings
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
import logging
from scipy import stats
from common.Boilerplate import get_db_connection

class RiskMetricsCalculator:
    """
    Calculate comprehensive risk metrics for portfolio and holdings
    """

    def __init__(self):
        """Initialize Risk Metrics Calculator"""
        self.risk_free_rate = 0.065  # 6.5% annual risk-free rate for India
        self.trading_days_per_year = 252

    def get_holdings_data(self) -> pd.DataFrame:
        """
        Get current holdings from database (equity + mutual funds)

        Returns:
            DataFrame with holdings data
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            # Get equity holdings
            cursor.execute("""
                SELECT
                    h.trading_symbol,
                    h.instrument_token,
                    h.quantity,
                    h.average_price,
                    COALESCE(rt.price_close, h.last_price, h.close_price, 0) as current_price,
                    h.pnl,
                    'EQUITY' as holding_type
                FROM my_schema.holdings h
                LEFT JOIN (
                    SELECT DISTINCT ON (scrip_id) scrip_id, price_close, price_date
                    FROM my_schema.rt_intraday_price
                    WHERE price_date::date <= CURRENT_DATE
                    ORDER BY scrip_id, price_date DESC
                ) rt ON h.trading_symbol = rt.scrip_id
                WHERE h.run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
                AND h.quantity > 0
            """)
            equity_rows = cursor.fetchall()

            # Get MF holdings
            cursor.execute("""
                SELECT
                    tradingsymbol as trading_symbol,
                    0 as instrument_token,
                    quantity,
                    average_price,
                    COALESCE(last_price, 0) as current_price,
                    pnl,
                    'MF' as holding_type
                FROM my_schema.mf_holdings
                WHERE run_date = (SELECT MAX(run_date) FROM my_schema.mf_holdings)
                AND quantity > 0
            """)
            mf_rows = cursor.fetchall()

            conn.close()

            # Combine holdings
            all_rows = []
            equity_columns = ['trading_symbol', 'instrument_token', 'quantity',
                            'average_price', 'current_price', 'pnl', 'holding_type']

            for row in equity_rows:
                quantity = row[2]
                avg_price = row[3]
                current_price = row[4]
                current_value = quantity * current_price
                invested_value = quantity * avg_price
                all_rows.append([row[0], row[1], row[2], row[3], row[4], row[5],
                               current_value, invested_value, row[6]])

            mf_columns = ['trading_symbol', 'instrument_token', 'quantity',
                         'average_price', 'current_price', 'pnl', 'current_value',
                         'invested_value', 'holding_type']

            for row in mf_rows:
                all_rows.append([row[0], row[1], row[2], row[3], row[4], row[5],
                               row[6] if len(row) > 6 else row[2] * row[4],
                               row[7] if len(row) > 7 else row[2] * row[3], row[6] if len(row) > 6 else 'MF'])

            combined_columns = ['trading_symbol', 'instrument_token', 'quantity',
                               'average_price', 'current_price', 'pnl', 'current_value',
                               'invested_value', 'holding_type']

            df = pd.DataFrame(all_rows, columns=combined_columns)
            df['current_value'] = pd.to_numeric(df['current_value'], errors='coerce').fillna(0)
            df['invested_value'] = pd.to_numeric(df['invested_value'], errors='coerce').fillna(0)
            df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0)
            df = df[df['quantity'] > 0]

            return df
        except Exception as e:
            logging.error(f"Error fetching holdings: {e}")
            return pd.DataFrame()

    def get_historical_prices(self, symbol: str, days: int = 252) -> pd.Series:
        """
        Get historical prices for a symbol

        Args:
            symbol: Trading symbol
            days: Number of days to look back

        Returns:
            Series of historical prices
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days + 30)  # Extra buffer

            cursor.execute("""
                SELECT price_date, price_close
                FROM my_schema.rt_intraday_price
                WHERE scrip_id = %s
                AND price_date >= %s
                AND price_date <= %s
                ORDER BY price_date
            """, (symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))

            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return pd.Series(dtype=float)

            df = pd.DataFrame(rows, columns=['date', 'price'])
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')

            # Get last price per day (in case of multiple entries)
            daily_prices = df.groupby(df.index.date)['price'].last()

            return daily_prices
        except Exception as e:
            logging.error(f"Error fetching historical prices for {symbol}: {e}")
            return pd.Series(dtype=float)

    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """
        Calculate returns from price series

        Args:
            prices: Series of prices

        Returns:
            Series of returns
        """
        return prices.pct_change().dropna()

    def calculate_var_historical(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculate Historical VaR

        Args:
            returns: Series of returns
            confidence_level: Confidence level (e.g., 0.95 for 95%)

        Returns:
            VaR value (negative number representing potential loss)
        """
        if len(returns) == 0:
            return 0.0

        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(returns, var_percentile)
        return float(var)

    def calculate_var_parametric(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculate Parametric VaR (assuming normal distribution)

        Args:
            returns: Series of returns
            confidence_level: Confidence level

        Returns:
            VaR value
        """
        if len(returns) == 0:
            return 0.0

        mean_return = returns.mean()
        std_return = returns.std()
        z_score = stats.norm.ppf(1 - confidence_level)
        var = mean_return + z_score * std_return
        return float(var)

    def calculate_var_monte_carlo(self, returns: pd.Series, confidence_level: float = 0.95,
                                 simulations: int = 10000) -> float:
        """
        Calculate Monte Carlo VaR

        Args:
            returns: Series of returns
            confidence_level: Confidence level
            simulations: Number of Monte Carlo simulations

        Returns:
            VaR value
        """
        if len(returns) == 0:
            return 0.0

        mean_return = returns.mean()
        std_return = returns.std()

        # Generate random returns
        np.random.seed(42)  # For reproducibility
        simulated_returns = np.random.normal(mean_return, std_return, simulations)

        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(simulated_returns, var_percentile)
        return float(var)

    def calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall)

        Args:
            returns: Series of returns
            confidence_level: Confidence level

        Returns:
            CVaR value
        """
        if len(returns) == 0:
            return 0.0

        var = self.calculate_var_historical(returns, confidence_level)
        # CVaR is the mean of returns below VaR threshold
        tail_returns = returns[returns <= var]

        if len(tail_returns) == 0:
            return float(var)

        cvar = tail_returns.mean()
        return float(cvar)

    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: Optional[float] = None) -> float:
        """
        Calculate Sharpe Ratio (annualized)

        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate (default: 6.5%)

        Returns:
            Sharpe ratio
        """
        if len(returns) == 0:
            return 0.0

        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate

        # Annualize returns and volatility
        mean_return = returns.mean() * self.trading_days_per_year
        std_return = returns.std() * np.sqrt(self.trading_days_per_year)

        if std_return == 0:
            return 0.0

        sharpe = (mean_return - risk_free_rate) / std_return
        return float(sharpe)

    def calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: Optional[float] = None) -> float:
        """
        Calculate Sortino Ratio (downside deviation only)

        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate

        Returns:
            Sortino ratio
        """
        if len(returns) == 0:
            return 0.0

        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate

        # Annualize mean return
        mean_return = returns.mean() * self.trading_days_per_year

        # Calculate downside deviation (only negative returns)
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float('inf') if mean_return > risk_free_rate else 0.0

        downside_std = downside_returns.std() * np.sqrt(self.trading_days_per_year)

        if downside_std == 0:
            return 0.0

        sortino = (mean_return - risk_free_rate) / downside_std
        return float(sortino)

    def calculate_max_drawdown(self, prices: pd.Series) -> Dict:
        """
        Calculate Maximum Drawdown

        Args:
            prices: Series of prices

        Returns:
            Dictionary with max drawdown, peak, trough, and recovery info
        """
        if len(prices) == 0:
            return {
                'max_drawdown': 0.0,
                'max_drawdown_pct': 0.0,
                'peak': 0.0,
                'trough': 0.0,
                'peak_date': None,
                'trough_date': None
            }

        # Calculate running maximum (peak)
        running_max = prices.expanding().max()

        # Calculate drawdown
        drawdown = (prices - running_max) / running_max

        # Find maximum drawdown
        max_dd_idx = drawdown.idxmin()
        max_dd = drawdown.min()

        # Find peak before drawdown
        peak_idx = running_max[:max_dd_idx].idxmax() if max_dd_idx != prices.index[0] else prices.idxmax()
        peak = running_max.loc[peak_idx]
        trough = prices.loc[max_dd_idx]

        return {
            'max_drawdown': float(peak - trough),
            'max_drawdown_pct': float(max_dd * 100),
            'peak': float(peak),
            'trough': float(trough),
            'peak_date': peak_idx.strftime('%Y-%m-%d') if hasattr(peak_idx, 'strftime') else str(peak_idx),
            'trough_date': max_dd_idx.strftime('%Y-%m-%d') if hasattr(max_dd_idx, 'strftime') else str(max_dd_idx)
        }

    def calculate_beta(self, stock_returns: pd.Series, market_returns: pd.Series) -> float:
        """
        Calculate Beta vs market

        Args:
            stock_returns: Stock returns series
            market_returns: Market returns series (e.g., Nifty)

        Returns:
            Beta value
        """
        if len(stock_returns) == 0 or len(market_returns) == 0:
            return 0.0

        # Align dates
        aligned = pd.concat([stock_returns, market_returns], axis=1, join='inner')
        aligned.columns = ['stock', 'market']
        aligned = aligned.dropna()

        if len(aligned) < 2:
            return 0.0

        covariance = aligned['stock'].cov(aligned['market'])
        market_variance = aligned['market'].var()

        if market_variance == 0:
            return 0.0

        beta = covariance / market_variance
        return float(beta)

    def calculate_correlation_matrix(self, holdings: pd.DataFrame, days: int = 60) -> pd.DataFrame:
        """
        Calculate correlation matrix between holdings

        Args:
            holdings: DataFrame with holdings
            days: Number of days for correlation calculation

        Returns:
            Correlation matrix DataFrame
        """
        try:
            equity_holdings = holdings[holdings['holding_type'] == 'EQUITY']

            if len(equity_holdings) == 0:
                return pd.DataFrame()

            # Get returns for each holding
            returns_dict = {}
            for _, holding in equity_holdings.iterrows():
                symbol = holding['trading_symbol']
                prices = self.get_historical_prices(symbol, days)
                if len(prices) > 10:
                    returns = self.calculate_returns(prices)
                    returns_dict[symbol] = returns

            if len(returns_dict) == 0:
                return pd.DataFrame()

            # Create DataFrame of returns
            returns_df = pd.DataFrame(returns_dict)
            returns_df = returns_df.dropna()

            if len(returns_df) < 10:
                return pd.DataFrame()

            # Calculate correlation matrix
            correlation_matrix = returns_df.corr()
            return correlation_matrix
        except Exception as e:
            logging.error(f"Error calculating correlation matrix: {e}")
            return pd.DataFrame()

    def calculate_volatility(self, returns: pd.Series, annualized: bool = True) -> float:
        """
        Calculate volatility (standard deviation of returns)

        Args:
            returns: Series of returns
            annualized: Whether to annualize the volatility

        Returns:
            Volatility value
        """
        if len(returns) == 0:
            return 0.0

        volatility = returns.std()

        if annualized:
            volatility = volatility * np.sqrt(self.trading_days_per_year)

        return float(volatility)

    def calculate_portfolio_risk_metrics(self, lookback_days: int = 252) -> Dict:
        """
        Calculate comprehensive portfolio risk metrics

        Args:
            lookback_days: Number of days to look back for calculations

        Returns:
            Dictionary with all risk metrics
        """
        try:
            holdings = self.get_holdings_data()

            if holdings.empty:
                return {
                    'error': 'No holdings found',
                    'portfolio_value': 0.0
                }

            portfolio_value = float(holdings['current_value'].sum())

            # Get portfolio returns (weighted average of holdings)
            equity_holdings = holdings[holdings['holding_type'] == 'EQUITY']

            if len(equity_holdings) == 0:
                return {
                    'error': 'No equity holdings found for risk calculation',
                    'portfolio_value': portfolio_value
                }

            # Calculate portfolio returns
            portfolio_returns_list = []
            weights = []

            for _, holding in equity_holdings.iterrows():
                symbol = holding['trading_symbol']
                weight = holding['current_value'] / portfolio_value
                prices = self.get_historical_prices(symbol, lookback_days)

                if len(prices) > 10:
                    returns = self.calculate_returns(prices)
                    portfolio_returns_list.append(returns * weight)
                    weights.append(weight)

            if len(portfolio_returns_list) == 0:
                return {
                    'error': 'Insufficient historical data',
                    'portfolio_value': portfolio_value
                }

            # Combine weighted returns
            portfolio_returns = pd.concat(portfolio_returns_list, axis=1).sum(axis=1)
            portfolio_returns = portfolio_returns.dropna()

            if len(portfolio_returns) < 10:
                return {
                    'error': 'Insufficient data for risk calculation',
                    'portfolio_value': portfolio_value
                }

            # Get market returns (Nifty)
            nifty_prices = self.get_historical_prices('NIFTY 50', lookback_days)
            nifty_returns = self.calculate_returns(nifty_prices) if len(nifty_prices) > 10 else pd.Series()

            # Calculate all metrics
            var_95_historical = self.calculate_var_historical(portfolio_returns, 0.95)
            var_99_historical = self.calculate_var_historical(portfolio_returns, 0.99)
            var_95_parametric = self.calculate_var_parametric(portfolio_returns, 0.95)
            var_99_parametric = self.calculate_var_parametric(portfolio_returns, 0.99)
            var_95_mc = self.calculate_var_monte_carlo(portfolio_returns, 0.95)
            var_99_mc = self.calculate_var_monte_carlo(portfolio_returns, 0.99)

            cvar_95 = self.calculate_cvar(portfolio_returns, 0.95)
            cvar_99 = self.calculate_cvar(portfolio_returns, 0.99)

            sharpe = self.calculate_sharpe_ratio(portfolio_returns)
            sortino = self.calculate_sortino_ratio(portfolio_returns)

            # Calculate portfolio value series for drawdown
            portfolio_values = []
            for _, holding in equity_holdings.iterrows():
                symbol = holding['trading_symbol']
                quantity = holding['quantity']
                prices = self.get_historical_prices(symbol, lookback_days)
                if len(prices) > 10:
                    values = prices * quantity
                    portfolio_values.append(values)

            if portfolio_values:
                combined_values = pd.concat(portfolio_values, axis=1).sum(axis=1)
                max_dd = self.calculate_max_drawdown(combined_values)
            else:
                max_dd = {'max_drawdown': 0.0, 'max_drawdown_pct': 0.0}

            # Beta calculation
            beta = 0.0
            if len(nifty_returns) > 10:
                beta = self.calculate_beta(portfolio_returns, nifty_returns)

            # Volatility
            volatility_30d = self.calculate_volatility(portfolio_returns.tail(30))
            volatility_60d = self.calculate_volatility(portfolio_returns.tail(60))
            volatility_90d = self.calculate_volatility(portfolio_returns.tail(90))
            volatility_annual = self.calculate_volatility(portfolio_returns)

            # Correlation matrix
            correlation_matrix = self.calculate_correlation_matrix(holdings, 60)

            return {
                'portfolio_value': portfolio_value,
                'var_95_historical': var_95_historical,
                'var_99_historical': var_99_historical,
                'var_95_parametric': var_95_parametric,
                'var_99_parametric': var_99_parametric,
                'var_95_monte_carlo': var_95_mc,
                'var_99_monte_carlo': var_99_mc,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99,
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'max_drawdown': max_dd,
                'beta': beta,
                'volatility_30d': volatility_30d,
                'volatility_60d': volatility_60d,
                'volatility_90d': volatility_90d,
                'volatility_annual': volatility_annual,
                'correlation_matrix': correlation_matrix.to_dict() if not correlation_matrix.empty else {},
                'data_points': len(portfolio_returns),
                'lookback_days': lookback_days
            }
        except Exception as e:
            logging.error(f"Error calculating portfolio risk metrics: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return {
                'error': str(e),
                'portfolio_value': 0.0
            }

    def calculate_holding_risk_metrics(self, symbol: str, lookback_days: int = 252) -> Dict:
        """
        Calculate risk metrics for a single holding

        Args:
            symbol: Trading symbol
            lookback_days: Number of days to look back

        Returns:
            Dictionary with risk metrics for the holding
        """
        try:
            prices = self.get_historical_prices(symbol, lookback_days)

            if len(prices) < 10:
                return {
                    'error': 'Insufficient historical data',
                    'symbol': symbol
                }

            returns = self.calculate_returns(prices)

            # Get market returns for beta
            nifty_prices = self.get_historical_prices('NIFTY 50', lookback_days)
            nifty_returns = self.calculate_returns(nifty_prices) if len(nifty_prices) > 10 else pd.Series()

            # Calculate metrics
            var_95 = self.calculate_var_historical(returns, 0.95)
            var_99 = self.calculate_var_historical(returns, 0.99)
            cvar_95 = self.calculate_cvar(returns, 0.95)
            sharpe = self.calculate_sharpe_ratio(returns)
            sortino = self.calculate_sortino_ratio(returns)
            max_dd = self.calculate_max_drawdown(prices)

            beta = 0.0
            if len(nifty_returns) > 10:
                beta = self.calculate_beta(returns, nifty_returns)

            volatility_30d = self.calculate_volatility(returns.tail(30))
            volatility_60d = self.calculate_volatility(returns.tail(60))
            volatility_annual = self.calculate_volatility(returns)

            return {
                'symbol': symbol,
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'max_drawdown': max_dd,
                'beta': beta,
                'volatility_30d': volatility_30d,
                'volatility_60d': volatility_60d,
                'volatility_annual': volatility_annual,
                'data_points': len(returns)
            }
        except Exception as e:
            logging.error(f"Error calculating risk metrics for {symbol}: {e}")
            return {
                'error': str(e),
                'symbol': symbol
            }
