"""
Business logic layer for risk management operations
"""

import logging
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import risk calculator
try:
    from holdings.RiskMetricsCalculator import RiskMetricsCalculator
except ImportError as e:
    logger.error(f"Failed to import RiskMetricsCalculator: {e}")
    RiskMetricsCalculator = None

# Don't import RiskConcentrationAnalyzer at module level - import lazily when needed
# This avoids import errors blocking the entire service
RiskConcentrationAnalyzer = None


class RiskService:
    """
    Service layer for risk management operations
    """
    
    def __init__(self):
        """Initialize Risk Service"""
        if RiskMetricsCalculator is None:
            raise ImportError("RiskMetricsCalculator is not available")
        self.risk_calculator = RiskMetricsCalculator()
        
        # Try to initialize concentration analyzer lazily, but don't fail if it's not available
        self.concentration_analyzer = None
        self.concentration_analyzer_error = None
        self._concentration_analyzer_class = None
        self._try_import_concentration_analyzer()
    
    def _try_import_concentration_analyzer(self):
        """Try to import and initialize RiskConcentrationAnalyzer lazily"""
        if self._concentration_analyzer_class is not None:
            return  # Already tried
        
        try:
            from holdings.RiskConcentrationAnalyzer import RiskConcentrationAnalyzer
            self._concentration_analyzer_class = RiskConcentrationAnalyzer
            logger.info("Successfully imported RiskConcentrationAnalyzer class")
            
            # Try to instantiate it
            try:
                self.concentration_analyzer = RiskConcentrationAnalyzer()
                logger.info("RiskConcentrationAnalyzer initialized successfully")
            except Exception as e:
                error_msg = f"Failed to initialize RiskConcentrationAnalyzer instance: {type(e).__name__}: {str(e)}"
                logger.warning(f"{error_msg} - concentration analysis will be disabled")
                self.concentration_analyzer_error = error_msg
                self.concentration_analyzer = None
        except ImportError as e:
            error_msg = f"Failed to import RiskConcentrationAnalyzer module: {type(e).__name__}: {str(e)}"
            logger.error(f"{error_msg}")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            self.concentration_analyzer_error = error_msg
            self._concentration_analyzer_class = False  # Mark as failed
        except SyntaxError as e:
            error_msg = f"Syntax error in RiskConcentrationAnalyzer module: {e}"
            logger.error(error_msg)
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            self.concentration_analyzer_error = error_msg
            self._concentration_analyzer_class = False  # Mark as failed
        except Exception as e:
            error_msg = f"Unexpected error importing RiskConcentrationAnalyzer: {type(e).__name__}: {e}"
            logger.error(error_msg)
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            self.concentration_analyzer_error = error_msg
            self._concentration_analyzer_class = False  # Mark as failed
    
    def get_portfolio_risk_metrics(self, lookback_days: int = 252) -> Dict:
        """
        Get comprehensive portfolio risk metrics
        
        Args:
            lookback_days: Number of days to look back for calculations
            
        Returns:
            Dictionary with all risk metrics
        """
        try:
            logger.info(f"Calculating portfolio risk metrics with lookback_days={lookback_days}")
            metrics = self.risk_calculator.calculate_portfolio_risk_metrics(lookback_days)
            
            # Ensure all required fields are present with default values
            result = {
                'portfolio_value': metrics.get('portfolio_value', 0.0),
                'var_95_historical': metrics.get('var_95_historical', 0.0),
                'var_99_historical': metrics.get('var_99_historical', 0.0),
                'var_95_parametric': metrics.get('var_95_parametric', 0.0),
                'var_99_parametric': metrics.get('var_99_parametric', 0.0),
                'var_95_monte_carlo': metrics.get('var_95_monte_carlo', 0.0),
                'var_99_monte_carlo': metrics.get('var_99_monte_carlo', 0.0),
                'cvar_95': metrics.get('cvar_95', 0.0),
                'cvar_99': metrics.get('cvar_99', 0.0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
                'sortino_ratio': metrics.get('sortino_ratio', 0.0),
                'beta': metrics.get('beta', 0.0),
                'volatility_30d': metrics.get('volatility_30d', 0.0),
                'volatility_60d': metrics.get('volatility_60d', 0.0),
                'volatility_90d': metrics.get('volatility_90d', 0.0),
                'volatility_annual': metrics.get('volatility_annual', 0.0),
                'data_points': metrics.get('data_points', 0),
                'lookback_days': metrics.get('lookback_days', lookback_days)
            }
            
            # Add max drawdown if available
            max_dd = metrics.get('max_drawdown', {})
            if isinstance(max_dd, dict):
                result['max_drawdown'] = max_dd.get('max_drawdown', 0.0)
                result['max_drawdown_pct'] = max_dd.get('max_drawdown_pct', 0.0)
            else:
                result['max_drawdown'] = 0.0
                result['max_drawdown_pct'] = 0.0
            
            # Add error if present
            if 'error' in metrics:
                result['error'] = metrics['error']
                logger.warning(f"Risk metrics calculation returned error: {metrics['error']}")
            
            logger.info(f"Risk metrics calculated successfully. Portfolio value: {result['portfolio_value']}, Sharpe: {result['sharpe_ratio']}")
            return result
            
        except Exception as e:
            logger.error(f"Error getting portfolio risk metrics: {e}", exc_info=True)
            return {
                'error': str(e),
                'portfolio_value': 0.0,
                'var_95_historical': 0.0,
                'var_99_historical': 0.0,
                'var_95_parametric': 0.0,
                'var_99_parametric': 0.0,
                'var_95_monte_carlo': 0.0,
                'var_99_monte_carlo': 0.0,
                'cvar_95': 0.0,
                'cvar_99': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'beta': 0.0,
                'volatility_30d': 0.0,
                'volatility_60d': 0.0,
                'volatility_90d': 0.0,
                'volatility_annual': 0.0,
                'max_drawdown': 0.0,
                'max_drawdown_pct': 0.0,
                'data_points': 0
            }
    
    def get_concentration_analysis(self) -> Dict:
        """
        Get portfolio concentration analysis
        
        Returns:
            Dictionary with concentration metrics
        """
        try:
            # Try to import if not already attempted
            if self._concentration_analyzer_class is None:
                self._try_import_concentration_analyzer()
            
            if self.concentration_analyzer is None:
                error_msg = self.concentration_analyzer_error or 'RiskConcentrationAnalyzer is not available'
                return {
                    'error': error_msg,
                    'suggestion': 'Check Docker logs for detailed import error information. The risk metrics endpoints will still work.'
                }
            
            comprehensive = self.concentration_analyzer.get_comprehensive_concentration_analysis()
            
            # Format the response to match what the dashboard expects
            stock_conc = comprehensive.get('stock_concentration', {})
            sector_conc = comprehensive.get('sector_concentration', {})
            overall_risk = comprehensive.get('overall_risk_assessment', {})
            
            # Build summary
            summary = {
                'overall_risk_score': overall_risk.get('risk_score', 0),
                'risk_level': overall_risk.get('risk_level', 'UNKNOWN'),
                'recommendations': []
            }
            
            if stock_conc.get('alerts'):
                summary['recommendations'].extend([a.get('message', '') for a in stock_conc['alerts']])
            if sector_conc.get('alerts'):
                summary['recommendations'].extend([a.get('message', '') for a in sector_conc['alerts']])
            
            return {
                'stock_concentration': stock_conc,
                'sector_concentration': sector_conc,
                'correlation_risk': comprehensive.get('correlation_risk', {}),
                'liquidity_risk': comprehensive.get('liquidity_risk', {}),
                'summary': summary
            }
        except Exception as e:
            logger.error(f"Error getting concentration analysis: {e}", exc_info=True)
            return {'error': str(e)}
    
    def get_sector_concentration(self) -> Dict:
        """
        Get sector concentration analysis
        
        Returns:
            Dictionary with sector concentration metrics
        """
        try:
            # Try to import if not already attempted
            if self._concentration_analyzer_class is None:
                self._try_import_concentration_analyzer()
            
            if self.concentration_analyzer is None:
                error_msg = self.concentration_analyzer_error or 'RiskConcentrationAnalyzer is not available'
                return {
                    'error': error_msg,
                    'suggestion': 'Check Docker logs for detailed import error information.'
                }
            return self.concentration_analyzer.analyze_sector_concentration()
        except Exception as e:
            logger.error(f"Error getting sector concentration: {e}", exc_info=True)
            return {'error': str(e)}
    
    def get_stock_concentration(self) -> Dict:
        """
        Get stock concentration analysis
        
        Returns:
            Dictionary with stock concentration metrics
        """
        try:
            # Try to import if not already attempted
            if self._concentration_analyzer_class is None:
                self._try_import_concentration_analyzer()
            
            if self.concentration_analyzer is None:
                error_msg = self.concentration_analyzer_error or 'RiskConcentrationAnalyzer is not available'
                return {
                    'error': error_msg,
                    'suggestion': 'Check Docker logs for detailed import error information.'
                }
            return self.concentration_analyzer.analyze_stock_concentration()
        except Exception as e:
            logger.error(f"Error getting stock concentration: {e}", exc_info=True)
            return {'error': str(e)}
    
    def calculate_var(self, 
                     confidence_level: float = 0.95,
                     time_horizon: int = 1,
                     method: str = 'historical') -> Dict:
        """
        Calculate Value at Risk (VaR)
        
        Args:
            confidence_level: Confidence level (0.95 for 95%, 0.99 for 99%)
            time_horizon: Time horizon in days
            method: Calculation method ('historical', 'parametric', 'monte_carlo')
            
        Returns:
            Dictionary with VaR metrics
        """
        try:
            import numpy as np
            
            # Get portfolio returns
            holdings = self.risk_calculator.get_holdings_data()
            if holdings.empty:
                return {'error': 'No holdings found', 'var': 0.0, 'var_amount': 0.0}
            
            portfolio_value = float(holdings['current_value'].sum())
            equity_holdings = holdings[holdings['holding_type'] == 'EQUITY']
            
            if len(equity_holdings) == 0:
                return {'error': 'No equity holdings found', 'var': 0.0, 'var_amount': 0.0}
            
            # Calculate portfolio returns
            portfolio_returns_list = []
            for _, holding in equity_holdings.iterrows():
                symbol = holding['trading_symbol']
                weight = holding['current_value'] / portfolio_value
                prices = self.risk_calculator.get_historical_prices(symbol, 252)
                
                if len(prices) > 10:
                    returns = self.risk_calculator.calculate_returns(prices)
                    portfolio_returns_list.append(returns * weight)
            
            if len(portfolio_returns_list) == 0:
                return {'error': 'Insufficient historical data', 'var': 0.0, 'var_amount': 0.0}
            
            import pandas as pd
            portfolio_returns = pd.concat(portfolio_returns_list, axis=1).sum(axis=1)
            portfolio_returns = portfolio_returns.dropna()
            
            if len(portfolio_returns) < 10:
                return {'error': 'Insufficient data for VaR calculation', 'var': 0.0, 'var_amount': 0.0}
            
            # Calculate VaR based on method
            if method == 'historical':
                var = self.risk_calculator.calculate_var_historical(portfolio_returns, confidence_level)
            elif method == 'parametric':
                var = self.risk_calculator.calculate_var_parametric(portfolio_returns, confidence_level)
            elif method == 'monte_carlo':
                var = self.risk_calculator.calculate_var_monte_carlo(portfolio_returns, confidence_level)
            else:
                return {'error': f'Unknown method: {method}', 'var': 0.0, 'var_amount': 0.0}
            
            # Scale VaR for time horizon
            var_scaled = var * np.sqrt(time_horizon)
            var_amount = abs(var_scaled) * portfolio_value
            
            return {
                'var': float(var_scaled),
                'var_amount': float(var_amount),
                'var_pct': float(abs(var_scaled) * 100),
                'confidence_level': confidence_level,
                'time_horizon': time_horizon,
                'method': method,
                'portfolio_value': portfolio_value
            }
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}", exc_info=True)
            return {'error': str(e), 'var': 0.0, 'var_amount': 0.0}
    
    def get_import_status(self) -> Dict:
        """
        Get the import status of RiskConcentrationAnalyzer for debugging
        
        Returns:
            Dictionary with import status and error details
        """
        if self._concentration_analyzer_class is None:
            self._try_import_concentration_analyzer()
        
        status = {
            'concentration_analyzer_available': self.concentration_analyzer is not None,
            'import_attempted': self._concentration_analyzer_class is not False,
            'error': self.concentration_analyzer_error
        }
        
        # Try to get more diagnostic info
        if self.concentration_analyzer is None:
            import sys
            import os
            status['python_path'] = sys.path
            status['current_working_directory'] = os.getcwd()
            status['holdings_module_path'] = None
            
            # Try to find the holdings module
            try:
                import holdings
                status['holdings_module_path'] = holdings.__file__ if hasattr(holdings, '__file__') else 'unknown'
                status['holdings_module_contents'] = dir(holdings)
            except Exception as e:
                status['holdings_import_error'] = str(e)
        
        return status

