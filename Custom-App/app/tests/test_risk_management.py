"""
Comprehensive test cases for Risk Management module
Tests for RiskMetricsCalculator, RiskConcentrationAnalyzer, AdvancedHedgingStrategies, and RiskService
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test
from holdings.RiskMetricsCalculator import RiskMetricsCalculator
from holdings.RiskConcentrationAnalyzer import RiskConcentrationAnalyzer
from holdings.AdvancedHedgingStrategies import AdvancedHedgingStrategies
from api.services.risk_service import RiskService


class TestRiskMetricsCalculator(unittest.TestCase):
    """Test cases for RiskMetricsCalculator"""

    def setUp(self):
        """Set up test fixtures"""
        self.calculator = RiskMetricsCalculator()
        self.sample_returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, 0.02])
        self.sample_prices = pd.Series([100, 101, 99, 102, 101, 98, 99, 101, 103])

    def test_calculate_returns(self):
        """Test returns calculation"""
        returns = self.calculator.calculate_returns(self.sample_prices)
        self.assertIsInstance(returns, pd.Series)
        self.assertEqual(len(returns), len(self.sample_prices) - 1)
        # First return should be (101-100)/100 = 0.01
        self.assertAlmostEqual(returns.iloc[0], 0.01, places=5)

    def test_calculate_returns_empty_series(self):
        """Test returns calculation with empty series"""
        empty_prices = pd.Series([])
        returns = self.calculator.calculate_returns(empty_prices)
        self.assertEqual(len(returns), 0)

    def test_calculate_returns_single_value(self):
        """Test returns calculation with single value"""
        single_price = pd.Series([100])
        returns = self.calculator.calculate_returns(single_price)
        self.assertEqual(len(returns), 0)

    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation"""
        sharpe = self.calculator.calculate_sharpe_ratio(self.sample_returns)
        self.assertIsInstance(sharpe, (int, float))
        # Sharpe ratio should be a real number
        self.assertFalse(np.isnan(sharpe))
        self.assertFalse(np.isinf(sharpe))

    def test_calculate_sharpe_ratio_zero_volatility(self):
        """Test Sharpe ratio with zero volatility"""
        constant_returns = pd.Series([0.01] * 10)
        sharpe = self.calculator.calculate_sharpe_ratio(constant_returns)
        # Should handle zero volatility gracefully
        self.assertIsInstance(sharpe, (int, float))

    def test_calculate_sortino_ratio(self):
        """Test Sortino ratio calculation"""
        sortino = self.calculator.calculate_sortino_ratio(self.sample_returns)
        self.assertIsInstance(sortino, (int, float))
        self.assertFalse(np.isnan(sortino))
        self.assertFalse(np.isinf(sortino))

    def test_calculate_volatility(self):
        """Test volatility calculation"""
        volatility = self.calculator.calculate_volatility(self.sample_returns)
        self.assertIsInstance(volatility, (int, float))
        self.assertGreaterEqual(volatility, 0)
        self.assertFalse(np.isnan(volatility))

    def test_calculate_var_historical(self):
        """Test historical VaR calculation"""
        var_95 = self.calculator.calculate_var_historical(self.sample_returns, confidence_level=0.95)
        var_99 = self.calculator.calculate_var_historical(self.sample_returns, confidence_level=0.99)
        
        self.assertIsInstance(var_95, (int, float))
        self.assertIsInstance(var_99, (int, float))
        # VaR should be negative (loss)
        self.assertLessEqual(var_95, 0)
        self.assertLessEqual(var_99, 0)
        # 99% VaR should be more negative than 95% VaR
        self.assertLessEqual(var_99, var_95)

    def test_calculate_var_parametric(self):
        """Test parametric VaR calculation"""
        var_95 = self.calculator.calculate_var_parametric(self.sample_returns, confidence_level=0.95)
        var_99 = self.calculator.calculate_var_parametric(self.sample_returns, confidence_level=0.99)
        
        self.assertIsInstance(var_95, (int, float))
        self.assertIsInstance(var_99, (int, float))
        self.assertLessEqual(var_95, 0)
        self.assertLessEqual(var_99, 0)
        self.assertLessEqual(var_99, var_95)

    def test_calculate_var_monte_carlo(self):
        """Test Monte Carlo VaR calculation"""
        var_95 = self.calculator.calculate_var_monte_carlo(
            self.sample_returns, 
            confidence_level=0.95,
            num_simulations=1000
        )
        
        self.assertIsInstance(var_95, (int, float))
        self.assertLessEqual(var_95, 0)
        self.assertFalse(np.isnan(var_95))

    def test_calculate_cvar(self):
        """Test Conditional VaR (CVaR) calculation"""
        cvar_95 = self.calculator.calculate_cvar(self.sample_returns, confidence_level=0.95)
        
        self.assertIsInstance(cvar_95, (int, float))
        self.assertLessEqual(cvar_95, 0)
        # CVaR should be more negative than VaR
        var_95 = self.calculator.calculate_var_historical(self.sample_returns, confidence_level=0.95)
        self.assertLessEqual(cvar_95, var_95)

    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation"""
        drawdown = self.calculator.calculate_max_drawdown(self.sample_prices)
        
        self.assertIsInstance(drawdown, (int, float))
        self.assertLessEqual(drawdown, 0)
        # Drawdown should be negative or zero

    def test_calculate_beta(self):
        """Test beta calculation"""
        stock_returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        market_returns = pd.Series([0.005, -0.01, 0.015, -0.005, 0.01])
        
        beta = self.calculator.calculate_beta(stock_returns, market_returns)
        
        self.assertIsInstance(beta, (int, float))
        self.assertFalse(np.isnan(beta))
        self.assertFalse(np.isinf(beta))

    def test_calculate_beta_insufficient_data(self):
        """Test beta calculation with insufficient data"""
        stock_returns = pd.Series([0.01])
        market_returns = pd.Series([0.005])
        
        beta = self.calculator.calculate_beta(stock_returns, market_returns)
        # Should return 0 or handle gracefully
        self.assertIsInstance(beta, (int, float))

    @patch('holdings.RiskMetricsCalculator.get_db_connection')
    def test_get_holdings_data(self, mock_db_conn):
        """Test getting holdings data from database"""
        # Mock database connection and cursor
        mock_cursor = Mock()
        mock_conn = Mock()
        mock_db_conn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        # Mock equity holdings
        mock_cursor.fetchall.side_effect = [
            [('RELIANCE', 1234, 10, 2000.0, 2100.0, 1000.0, 'EQUITY')],
            []  # MF holdings
        ]
        
        holdings = self.calculator.get_holdings_data()
        
        self.assertIsInstance(holdings, pd.DataFrame)
        mock_conn.close.assert_called()

    @patch('holdings.RiskMetricsCalculator.get_db_connection')
    def test_get_historical_prices(self, mock_db_conn):
        """Test getting historical prices"""
        mock_cursor = Mock()
        mock_conn = Mock()
        mock_db_conn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        # Mock price data
        mock_cursor.fetchall.return_value = [
            (100.0,), (101.0,), (99.0,), (102.0,)
        ]
        
        prices = self.calculator.get_historical_prices('RELIANCE', days=10)
        
        self.assertIsInstance(prices, pd.Series)
        mock_conn.close.assert_called()


class TestRiskConcentrationAnalyzer(unittest.TestCase):
    """Test cases for RiskConcentrationAnalyzer"""

    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = RiskConcentrationAnalyzer()

    @patch('holdings.RiskConcentrationAnalyzer.get_db_connection')
    def test_get_holdings_data(self, mock_db_conn):
        """Test getting holdings data"""
        mock_cursor = Mock()
        mock_conn = Mock()
        mock_db_conn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        # Mock holdings data
        mock_cursor.fetchall.return_value = [
            ('RELIANCE', 10, 2000.0, 2100.0, 1000.0, 'EQUITY', 'ENERGY'),
            ('TCS', 5, 3000.0, 3100.0, 500.0, 'EQUITY', 'IT')
        ]
        
        holdings = self.analyzer.get_holdings_data()
        
        self.assertIsInstance(holdings, pd.DataFrame)
        self.assertGreater(len(holdings), 0)

    @patch('holdings.RiskConcentrationAnalyzer.get_db_connection')
    def test_analyze_stock_concentration(self, mock_db_conn):
        """Test stock concentration analysis"""
        mock_cursor = Mock()
        mock_conn = Mock()
        mock_db_conn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        # Mock holdings data
        mock_cursor.fetchall.return_value = [
            ('RELIANCE', 10, 2000.0, 2100.0, 1000.0, 'EQUITY', 'ENERGY'),
            ('TCS', 5, 3000.0, 3100.0, 500.0, 'EQUITY', 'IT')
        ]
        
        result = self.analyzer.analyze_stock_concentration()
        
        self.assertIsInstance(result, dict)
        self.assertIn('top_5_holdings', result)
        self.assertIn('concentration_distribution', result)
        self.assertIn('total_value', result)

    @patch('holdings.RiskConcentrationAnalyzer.get_db_connection')
    def test_analyze_sector_concentration(self, mock_db_conn):
        """Test sector concentration analysis"""
        mock_cursor = Mock()
        mock_conn = Mock()
        mock_db_conn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        # Mock holdings data with sectors
        mock_cursor.fetchall.return_value = [
            ('RELIANCE', 10, 2000.0, 2100.0, 1000.0, 'EQUITY', 'ENERGY'),
            ('TCS', 5, 3000.0, 3100.0, 500.0, 'EQUITY', 'IT')
        ]
        
        result = self.analyzer.analyze_sector_concentration()
        
        self.assertIsInstance(result, dict)
        self.assertIn('sector_allocation', result)

    @patch('holdings.RiskConcentrationAnalyzer.get_db_connection')
    def test_get_comprehensive_concentration_analysis(self, mock_db_conn):
        """Test comprehensive concentration analysis"""
        mock_cursor = Mock()
        mock_conn = Mock()
        mock_db_conn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        # Mock holdings data
        mock_cursor.fetchall.return_value = [
            ('RELIANCE', 10, 2000.0, 2100.0, 1000.0, 'EQUITY', 'ENERGY'),
            ('TCS', 5, 3000.0, 3100.0, 500.0, 'EQUITY', 'IT')
        ]
        
        result = self.analyzer.get_comprehensive_concentration_analysis()
        
        self.assertIsInstance(result, dict)
        self.assertIn('stock_concentration', result)
        self.assertIn('sector_concentration', result)
        self.assertIn('summary', result)


class TestAdvancedHedgingStrategies(unittest.TestCase):
    """Test cases for AdvancedHedgingStrategies"""

    def setUp(self):
        """Set up test fixtures"""
        self.strategies = AdvancedHedgingStrategies()

    def test_calculate_delta_hedging_positive_delta(self):
        """Test delta hedging with positive delta"""
        portfolio_value = 1000000
        portfolio_delta = 1.5
        
        with patch('holdings.AdvancedHedgingStrategies.get_db_connection') as mock_db:
            mock_cursor = Mock()
            mock_conn = Mock()
            mock_db.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor
            
            # Mock futures data
            mock_cursor.fetchall.return_value = [
                (123456, 22000.0, datetime.now(), date.today())
            ]
            mock_cursor.fetchone.return_value = None  # No spot price
            
            result = self.strategies._calculate_delta_hedging(portfolio_value, portfolio_delta)
            
            self.assertIsInstance(result, dict)
            self.assertEqual(result['action'], 'sell_futures')
            self.assertIn('contracts', result)
            self.assertGreater(result['contracts'], 0)

    def test_calculate_delta_hedging_negative_delta(self):
        """Test delta hedging with negative delta"""
        portfolio_value = 1000000
        portfolio_delta = 0.5
        
        with patch('holdings.AdvancedHedgingStrategies.get_db_connection') as mock_db:
            mock_cursor = Mock()
            mock_conn = Mock()
            mock_db.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor
            
            # Mock futures data
            mock_cursor.fetchall.return_value = [
                (123456, 22000.0, datetime.now(), date.today())
            ]
            mock_cursor.fetchone.return_value = None
            
            result = self.strategies._calculate_delta_hedging(portfolio_value, portfolio_delta)
            
            self.assertIsInstance(result, dict)
            self.assertEqual(result['action'], 'buy_futures')
            self.assertIn('contracts', result)

    def test_calculate_delta_hedging_neutral_delta(self):
        """Test delta hedging with neutral delta"""
        portfolio_value = 1000000
        portfolio_delta = 1.0
        
        with patch('holdings.AdvancedHedgingStrategies.get_db_connection') as mock_db:
            mock_cursor = Mock()
            mock_conn = Mock()
            mock_db.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor
            
            result = self.strategies._calculate_delta_hedging(portfolio_value, portfolio_delta)
            
            self.assertIsInstance(result, dict)
            self.assertEqual(result['action'], 'no_action')

    def test_calculate_tail_risk_hedging(self):
        """Test tail risk hedging calculation"""
        portfolio_value = 1000000
        
        with patch('holdings.AdvancedHedgingStrategies.get_db_connection') as mock_db:
            mock_cursor = Mock()
            mock_conn = Mock()
            mock_db.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor
            
            # Mock Nifty spot price
            mock_cursor.fetchone.return_value = (22000.0,)
            
            result = self.strategies._calculate_tail_risk_hedging(portfolio_value)
            
            self.assertIsInstance(result, dict)
            self.assertIn('action', result)
            self.assertIn('strike', result)
            if result['action'] != 'no_action':
                self.assertGreater(result['contracts'], 0)

    def test_calculate_correlation_hedging(self):
        """Test correlation hedging calculation"""
        portfolio_value = 1000000
        
        with patch('holdings.AdvancedHedgingStrategies.get_db_connection') as mock_db:
            mock_cursor = Mock()
            mock_conn = Mock()
            mock_db.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor
            
            # Mock no existing gold/silver holdings
            mock_cursor.fetchone.side_effect = [None, None]
            
            result = self.strategies._calculate_correlation_hedging(portfolio_value)
            
            self.assertIsInstance(result, dict)
            self.assertIn('strategy_type', result)
            self.assertIn('action', result)

    def test_calculate_correlation_hedging_with_existing_holdings(self):
        """Test correlation hedging with existing gold/silver holdings"""
        portfolio_value = 1000000
        
        with patch('holdings.AdvancedHedgingStrategies.get_db_connection') as mock_db:
            mock_cursor = Mock()
            mock_conn = Mock()
            mock_db.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor
            
            # Mock existing gold holdings
            mock_cursor.fetchone.side_effect = [
                (10, 6000.0, 60000.0),  # Gold: 10 units, price 6000, value 60000
                None  # No silver
            ]
            
            result = self.strategies._calculate_correlation_hedging(portfolio_value)
            
            self.assertIsInstance(result, dict)
            # Should account for existing holdings
            if 'strategies' in result:
                for strategy in result['strategies']:
                    if 'existing_gold_units' in strategy:
                        self.assertEqual(strategy['existing_gold_units'], 10)

    def test_get_all_advanced_strategies(self):
        """Test getting all advanced strategies"""
        portfolio_value = 1000000
        portfolio_delta = 1.2
        
        with patch.object(self.strategies, '_calculate_delta_hedging') as mock_delta, \
             patch.object(self.strategies, '_calculate_volatility_hedging') as mock_vol, \
             patch.object(self.strategies, '_calculate_tail_risk_hedging') as mock_tail, \
             patch.object(self.strategies, '_calculate_correlation_hedging') as mock_corr:
            
            mock_delta.return_value = {'action': 'sell_futures', 'contracts': 5}
            mock_vol.return_value = {'action': 'buy_options', 'contracts': 10}
            mock_tail.return_value = {'action': 'buy_puts', 'contracts': 2}
            mock_corr.return_value = {'action': 'buy_gold', 'quantity': 10}
            
            result = self.strategies.get_all_advanced_strategies(portfolio_value, portfolio_delta)
            
            self.assertIsInstance(result, dict)
            self.assertIn('delta_hedging', result)
            self.assertIn('volatility_hedging', result)
            self.assertIn('tail_risk_hedging', result)
            self.assertIn('correlation_hedging', result)


class TestRiskService(unittest.TestCase):
    """Test cases for RiskService"""

    def setUp(self):
        """Set up test fixtures"""
        with patch('api.services.risk_service.RiskMetricsCalculator'):
            self.service = RiskService()

    @patch('api.services.risk_service.RiskMetricsCalculator')
    def test_get_portfolio_risk_metrics(self, mock_calc_class):
        """Test getting portfolio risk metrics"""
        mock_calculator = Mock()
        mock_calc_class.return_value = mock_calculator
        
        mock_calculator.calculate_portfolio_risk_metrics.return_value = {
            'portfolio_value': 1000000,
            'var_95_historical': -0.05,
            'sharpe_ratio': 1.5,
            'sortino_ratio': 2.0
        }
        
        service = RiskService()
        result = service.get_portfolio_risk_metrics()
        
        self.assertIsInstance(result, dict)
        self.assertIn('portfolio_value', result)
        self.assertIn('var_95_historical', result)

    @patch('api.services.risk_service.RiskConcentrationAnalyzer')
    def test_get_concentration_analysis(self, mock_analyzer_class):
        """Test getting concentration analysis"""
        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer
        
        mock_analyzer.get_comprehensive_concentration_analysis.return_value = {
            'stock_concentration': {'top_5_holdings': []},
            'sector_concentration': {'sector_allocation': []},
            'summary': {'overall_risk_score': 50}
        }
        
        service = RiskService()
        service.concentration_analyzer = mock_analyzer
        
        result = service.get_concentration_analysis()
        
        self.assertIsInstance(result, dict)
        self.assertIn('stock_concentration', result)

    def test_get_concentration_analysis_no_analyzer(self):
        """Test concentration analysis when analyzer is not available"""
        service = RiskService()
        service.concentration_analyzer = None
        
        result = service.get_concentration_analysis()
        
        self.assertIsInstance(result, dict)
        self.assertIn('error', result)


class TestRiskManagementAPI(unittest.TestCase):
    """Test cases for Risk Management API endpoints"""

    def setUp(self):
        """Set up test fixtures"""
        from fastapi.testclient import TestClient
        from KiteAccessToken import app
        self.client = TestClient(app)

    @patch('api.services.risk_service.RiskService')
    def test_get_risk_metrics_endpoint(self, mock_service_class):
        """Test /api/risk/metrics endpoint"""
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.get_portfolio_risk_metrics.return_value = {
            'portfolio_value': 1000000,
            'var_95_historical': -0.05
        }
        
        response = self.client.get('/api/risk/metrics')
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('portfolio_value', data)

    @patch('api.services.risk_service.RiskService')
    def test_get_concentration_endpoint(self, mock_service_class):
        """Test /api/risk/concentration endpoint"""
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.get_concentration_analysis.return_value = {
            'stock_concentration': {},
            'sector_concentration': {}
        }
        
        response = self.client.get('/api/risk/concentration')
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('stock_concentration', data)

    @patch('holdings.AdvancedHedgingStrategies.AdvancedHedgingStrategies')
    def test_get_advanced_hedging_endpoint(self, mock_strategies_class):
        """Test /api/risk/hedging/advanced endpoint"""
        mock_strategies = Mock()
        mock_strategies_class.return_value = mock_strategies
        mock_strategies.get_all_advanced_strategies.return_value = {
            'delta_hedging': {'action': 'buy_futures'},
            'correlation_hedging': {'action': 'buy_gold'}
        }
        
        response = self.client.get('/api/risk/hedging/advanced')
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('strategies', data)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""

    def test_risk_metrics_empty_holdings(self):
        """Test risk metrics with empty holdings"""
        calculator = RiskMetricsCalculator()
        
        with patch.object(calculator, 'get_holdings_data', return_value=pd.DataFrame()):
            result = calculator.calculate_portfolio_risk_metrics()
            
            self.assertIsInstance(result, dict)
            # Should handle empty holdings gracefully
            self.assertIn('portfolio_value', result)

    def test_risk_metrics_insufficient_data(self):
        """Test risk metrics with insufficient price data"""
        calculator = RiskMetricsCalculator()
        
        with patch.object(calculator, 'get_holdings_data') as mock_holdings, \
             patch.object(calculator, 'get_historical_prices', return_value=pd.Series([100, 101])):
            
            mock_holdings.return_value = pd.DataFrame({
                'trading_symbol': ['STOCK1'],
                'quantity': [10],
                'current_price': [100],
                'holding_type': ['EQUITY']
            })
            
            result = calculator.calculate_portfolio_risk_metrics()
            
            self.assertIsInstance(result, dict)
            # Should handle insufficient data gracefully

    def test_hedging_strategies_zero_portfolio_value(self):
        """Test hedging strategies with zero portfolio value"""
        strategies = AdvancedHedgingStrategies()
        
        result = strategies.get_all_advanced_strategies(0, 1.0)
        
        self.assertIsInstance(result, dict)
        # Should handle zero portfolio value

    def test_hedging_strategies_negative_delta(self):
        """Test hedging strategies with negative delta"""
        strategies = AdvancedHedgingStrategies()
        
        with patch('holdings.AdvancedHedgingStrategies.get_db_connection') as mock_db:
            mock_cursor = Mock()
            mock_conn = Mock()
            mock_db.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor
            mock_cursor.fetchall.return_value = []
            mock_cursor.fetchone.return_value = None
            
            result = strategies._calculate_delta_hedging(1000000, -0.5)
            
            self.assertIsInstance(result, dict)
            # Should handle negative delta

    def test_correlation_hedging_zero_portfolio(self):
        """Test correlation hedging with zero portfolio value"""
        strategies = AdvancedHedgingStrategies()
        
        with patch('holdings.AdvancedHedgingStrategies.get_db_connection') as mock_db:
            mock_cursor = Mock()
            mock_conn = Mock()
            mock_db.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor
            mock_cursor.fetchone.side_effect = [None, None]
            
            result = strategies._calculate_correlation_hedging(0)
            
            self.assertIsInstance(result, dict)
            # Should handle zero portfolio value


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)

