"""
Risk Limits Manager Module
Manages portfolio risk limits and alerts
"""

import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import logging

# Import get_db_connection with error handling
try:
    from common.Boilerplate import get_db_connection
except (ImportError, AttributeError, Exception) as e:
    logging.warning(f"Failed to import get_db_connection from common.Boilerplate: {e}. Using fallback.")
    # Define a fallback function
    import psycopg2
    def get_db_connection():
        return psycopg2.connect(
            host="postgres",
            database="mydb",
            user="postgres",
            password="postgres"
        )

class RiskLimitsManager:
    """
    Manage portfolio risk limits and monitoring
    """

    def __init__(self):
        """Initialize Risk Limits Manager"""
        self.default_limits = {
            'max_single_stock': 0.20,  # 20% max in single stock
            'max_sector': 0.30,        # 30% max in single sector
            'max_volatility': 0.25,    # 25% max annual volatility
            'max_drawdown': 0.20,      # 20% max drawdown
            'max_var_95': 0.10,        # 10% max 95% VaR
            'min_sharpe': 0.5          # Minimum Sharpe ratio
        }

    def get_limit_status(self) -> Dict:
        """
        Get current status of all risk limits

        Returns:
            Dictionary with limit status
        """
        try:
            # Get current portfolio metrics (simplified)
            portfolio_metrics = self._get_current_portfolio_metrics()

            limits_status = {}
            alerts = []

            # Check each limit
            for limit_name, limit_value in self.default_limits.items():
                current_value = portfolio_metrics.get(limit_name, 0)

                if limit_name.startswith('max_'):
                    breached = current_value > limit_value
                elif limit_name.startswith('min_'):
                    breached = current_value < limit_value
                else:
                    breached = False

                limits_status[limit_name] = {
                    'limit_value': limit_value,
                    'current_value': current_value,
                    'breached': breached,
                    'breach_amount': abs(current_value - limit_value) if breached else 0
                }

                if breached:
                    alerts.append({
                        'limit': limit_name,
                        'severity': 'high' if limit_name in ['max_drawdown', 'max_var_95'] else 'medium',
                        'message': f"{limit_name} limit breached: {current_value:.3f} vs {limit_value:.3f}"
                    })

            return {
                'limits': limits_status,
                'alerts': alerts,
                'total_alerts': len(alerts),
                'overall_status': 'breached' if alerts else 'within_limits'
            }
        except Exception as e:
            logging.error(f"Error getting limit status: {e}")
            return {'error': str(e)}

    def check_all_limits(self) -> Dict:
        """
        Check all risk limits and generate alerts

        Returns:
            Dictionary with alerts
        """
        return self.get_limit_status()

    def set_risk_limit(self, limit_type: str, limit_value: float, enabled: bool = True) -> bool:
        """
        Set or update a risk limit

        Args:
            limit_type: Type of limit
            limit_value: Limit value
            enabled: Whether limit is enabled

        Returns:
            Success status
        """
        try:
            # In a real implementation, this would save to database
            if limit_type in self.default_limits:
                self.default_limits[limit_type] = limit_value
                return True
            return False
        except Exception as e:
            logging.error(f"Error setting risk limit: {e}")
            return False

    def _get_current_portfolio_metrics(self) -> Dict:
        """
        Get current portfolio metrics (simplified)

        Returns:
            Dictionary with current metrics
        """
        try:
            # This would normally get data from RiskMetricsCalculator
            # For now, return mock data
            return {
                'max_single_stock': 0.12,  # 12%
                'max_sector': 0.18,        # 18%
                'max_volatility': 0.15,    # 15%
                'max_drawdown': 0.08,      # 8%
                'max_var_95': 0.05,        # 5%
                'min_sharpe': 0.8          # 0.8
            }
        except Exception as e:
            logging.error(f"Error getting portfolio metrics: {e}")
            return {}
