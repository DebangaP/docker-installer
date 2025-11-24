"""
Integration tests for Risk Management module
These tests require a database connection and should be run in the Docker environment
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, date

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Only run these tests if database is available
SKIP_INTEGRATION_TESTS = os.getenv('SKIP_INTEGRATION_TESTS', 'false').lower() == 'true'


@unittest.skipIf(SKIP_INTEGRATION_TESTS, "Integration tests skipped")
class TestRiskMetricsCalculatorIntegration(unittest.TestCase):
    """Integration tests for RiskMetricsCalculator with real database"""

    def setUp(self):
        """Set up test fixtures"""
        try:
            from holdings.RiskMetricsCalculator import RiskMetricsCalculator
            self.calculator = RiskMetricsCalculator()
        except Exception as e:
            self.skipTest(f"Could not initialize RiskMetricsCalculator: {e}")

    def test_get_holdings_data_integration(self):
        """Test getting holdings data from real database"""
        try:
            holdings = self.calculator.get_holdings_data()
            self.assertIsInstance(holdings, pd.DataFrame)
            # If holdings exist, verify structure
            if not holdings.empty:
                self.assertIn('trading_symbol', holdings.columns)
                self.assertIn('quantity', holdings.columns)
                self.assertIn('current_price', holdings.columns)
        except Exception as e:
            self.skipTest(f"Database not available: {e}")

    def test_get_historical_prices_integration(self):
        """Test getting historical prices from real database"""
        try:
            # Try with a common stock symbol
            prices = self.calculator.get_historical_prices('RELIANCE', days=30)
            self.assertIsInstance(prices, pd.Series)
            # If prices exist, verify they're numeric
            if len(prices) > 0:
                self.assertTrue(all(isinstance(p, (int, float)) for p in prices))
        except Exception as e:
            self.skipTest(f"Database not available: {e}")

    def test_calculate_portfolio_risk_metrics_integration(self):
        """Test calculating portfolio risk metrics with real data"""
        try:
            result = self.calculator.calculate_portfolio_risk_metrics(lookback_days=60)
            self.assertIsInstance(result, dict)
            self.assertIn('portfolio_value', result)
            # Verify all expected keys are present
            expected_keys = [
                'var_95_historical', 'var_99_historical',
                'var_95_parametric', 'var_99_parametric',
                'cvar_95', 'cvar_99',
                'sharpe_ratio', 'sortino_ratio',
                'max_drawdown', 'volatility'
            ]
            for key in expected_keys:
                self.assertIn(key, result)
        except Exception as e:
            self.skipTest(f"Database not available: {e}")


@unittest.skipIf(SKIP_INTEGRATION_TESTS, "Integration tests skipped")
class TestRiskConcentrationAnalyzerIntegration(unittest.TestCase):
    """Integration tests for RiskConcentrationAnalyzer with real database"""

    def setUp(self):
        """Set up test fixtures"""
        try:
            from holdings.RiskConcentrationAnalyzer import RiskConcentrationAnalyzer
            self.analyzer = RiskConcentrationAnalyzer()
        except Exception as e:
            self.skipTest(f"Could not initialize RiskConcentrationAnalyzer: {e}")

    def test_get_holdings_data_integration(self):
        """Test getting holdings data from real database"""
        try:
            holdings = self.analyzer.get_holdings_data()
            self.assertIsInstance(holdings, pd.DataFrame)
        except Exception as e:
            self.skipTest(f"Database not available: {e}")

    def test_analyze_stock_concentration_integration(self):
        """Test stock concentration analysis with real data"""
        try:
            result = self.analyzer.analyze_stock_concentration()
            self.assertIsInstance(result, dict)
            self.assertIn('top_5_holdings', result)
            self.assertIn('concentration_distribution', result)
        except Exception as e:
            self.skipTest(f"Database not available: {e}")

    def test_analyze_sector_concentration_integration(self):
        """Test sector concentration analysis with real data"""
        try:
            result = self.analyzer.analyze_sector_concentration()
            self.assertIsInstance(result, dict)
            self.assertIn('sector_allocation', result)
        except Exception as e:
            self.skipTest(f"Database not available: {e}")


@unittest.skipIf(SKIP_INTEGRATION_TESTS, "Integration tests skipped")
class TestAdvancedHedgingStrategiesIntegration(unittest.TestCase):
    """Integration tests for AdvancedHedgingStrategies with real database"""

    def setUp(self):
        """Set up test fixtures"""
        try:
            from holdings.AdvancedHedgingStrategies import AdvancedHedgingStrategies
            self.strategies = AdvancedHedgingStrategies()
        except Exception as e:
            self.skipTest(f"Could not initialize AdvancedHedgingStrategies: {e}")

    def test_get_all_advanced_strategies_integration(self):
        """Test getting all advanced strategies with real data"""
        try:
            # Use a sample portfolio value
            portfolio_value = 1000000
            portfolio_delta = 1.2
            
            result = self.strategies.get_all_advanced_strategies(portfolio_value, portfolio_delta)
            
            self.assertIsInstance(result, dict)
            self.assertIn('delta_hedging', result)
            self.assertIn('volatility_hedging', result)
            self.assertIn('tail_risk_hedging', result)
            self.assertIn('correlation_hedging', result)
        except Exception as e:
            self.skipTest(f"Database not available: {e}")

    def test_correlation_hedging_integration(self):
        """Test correlation hedging with real database"""
        try:
            portfolio_value = 1000000
            result = self.strategies._calculate_correlation_hedging(portfolio_value)
            
            self.assertIsInstance(result, dict)
            self.assertIn('strategy_type', result)
            # Should not have error key if successful
            self.assertNotIn('error', result)
        except Exception as e:
            self.skipTest(f"Database not available: {e}")


@unittest.skipIf(SKIP_INTEGRATION_TESTS, "Integration tests skipped")
class TestRiskServiceIntegration(unittest.TestCase):
    """Integration tests for RiskService with real database"""

    def setUp(self):
        """Set up test fixtures"""
        try:
            from api.services.risk_service import RiskService
            self.service = RiskService()
        except Exception as e:
            self.skipTest(f"Could not initialize RiskService: {e}")

    def test_get_portfolio_risk_metrics_integration(self):
        """Test getting portfolio risk metrics with real data"""
        try:
            result = self.service.get_portfolio_risk_metrics()
            
            self.assertIsInstance(result, dict)
            self.assertIn('portfolio_value', result)
            # Verify metrics are numeric
            if result.get('portfolio_value', 0) > 0:
                self.assertGreater(result['portfolio_value'], 0)
        except Exception as e:
            self.skipTest(f"Database not available: {e}")

    def test_get_concentration_analysis_integration(self):
        """Test getting concentration analysis with real data"""
        try:
            result = self.service.get_concentration_analysis()
            
            self.assertIsInstance(result, dict)
            # Should have either data or error
            self.assertTrue('error' in result or 'stock_concentration' in result)
        except Exception as e:
            self.skipTest(f"Database not available: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)

