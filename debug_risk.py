import sys
import os
sys.path.append('Custom-App')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Custom-App'))

from app.holdings.RiskMetricsCalculator import RiskMetricsCalculator

def debug_risk_metrics():
    print("Initializing RiskMetricsCalculator...")
    calculator = RiskMetricsCalculator()

    print("Fetching holdings data...")
    holdings = calculator.get_holdings_data()
    print(f"Found {len(holdings)} holdings")
    print(f"Holdings columns: {holdings.columns.tolist()}")
    print(f"Sample holdings:\n{holdings.head()}")

    if holdings.empty:
        print("No holdings found!")
        return

    equity_holdings = holdings[holdings['holding_type'] == 'EQUITY']
    print(f"Found {len(equity_holdings)} equity holdings")

    if len(equity_holdings) == 0:
        print("No equity holdings found!")
        return

    # Test historical prices for first symbol
    first_symbol = equity_holdings.iloc[0]['trading_symbol']
    print(f"\nTesting historical prices for {first_symbol}...")

    prices = calculator.get_historical_prices(first_symbol, 252)
    print(f"Found {len(prices)} price records for {first_symbol}")
    print(f"Price range: {prices.min()} to {prices.max()}")

    if len(prices) < 10:
        print("Insufficient price data!")
        return

    # Test returns calculation
    returns = calculator.calculate_returns(prices)
    print(f"Calculated {len(returns)} returns")
    print(f"Returns range: {returns.min()} to {returns.max()}")

    # Test Sharpe ratio
    sharpe = calculator.calculate_sharpe_ratio(returns)
    print(f"Sharpe ratio: {sharpe}")

    # Now test full portfolio calculation
    print("\nTesting full portfolio risk metrics...")
    try:
        result = calculator.calculate_portfolio_risk_metrics(252)
        print("Portfolio risk metrics result:")
        print(f"Portfolio value: {result.get('portfolio_value', 'N/A')}")
        print(f"Sharpe ratio: {result.get('sharpe_ratio', 'N/A')}")
        print(f"Sortino ratio: {result.get('sortino_ratio', 'N/A')}")
        print(f"VaR 95%: {result.get('var_95_historical', 'N/A')}")
        print(f"Beta: {result.get('beta', 'N/A')}")
        print(f"Data points: {result.get('data_points', 'N/A')}")
        if 'error' in result:
            print(f"Error: {result['error']}")
    except Exception as e:
        print(f"Error calculating portfolio metrics: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_risk_metrics()
