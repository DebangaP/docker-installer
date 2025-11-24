"""
REST API endpoints for risk metrics, concentration analysis, and hedging strategies
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional
import logging

from api.services.risk_service import RiskService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/risk", tags=["risk-management"])

# Initialize service
risk_service = RiskService()


@router.get("/metrics")
async def get_risk_metrics(
    lookback_days: int = Query(252, ge=30, le=1000, description="Number of days to look back for calculations")
):
    """
    Get comprehensive portfolio risk metrics
    
    Returns:
        Dictionary with risk metrics including:
        - VaR (Value at Risk) at 95% and 99% confidence levels
        - CVaR (Conditional VaR)
        - Sharpe Ratio
        - Sortino Ratio
        - Beta
        - Volatility metrics
        - Max Drawdown
    """
    try:
        logger.info(f"Fetching risk metrics with lookback_days={lookback_days}")
        result = risk_service.get_portfolio_risk_metrics(lookback_days)
        
        if 'error' in result and result.get('portfolio_value', 0) == 0:
            # Return the error but still return the structure with zeros
            logger.warning(f"Risk metrics calculation returned error: {result.get('error')}")
        
        return result
    except Exception as e:
        logger.error(f"Error in get_risk_metrics endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/concentration")
async def get_concentration():
    """
    Get portfolio concentration analysis
    
    Returns:
        Dictionary with concentration metrics including:
        - Sector concentration
        - Stock concentration
        - Overall risk score
    """
    try:
        return risk_service.get_concentration_analysis()
    except Exception as e:
        logger.error(f"Error in get_concentration endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/concentration/sector")
async def get_sector_concentration():
    """
    Get sector concentration analysis
    
    Returns:
        Dictionary with sector allocation and concentration metrics
    """
    try:
        return risk_service.get_sector_concentration()
    except Exception as e:
        logger.error(f"Error in get_sector_concentration endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/concentration/stock")
async def get_stock_concentration():
    """
    Get stock concentration analysis
    
    Returns:
        Dictionary with top holdings and concentration metrics
    """
    try:
        return risk_service.get_stock_concentration()
    except Exception as e:
        logger.error(f"Error in get_stock_concentration endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/var")
async def calculate_var(
    confidence_level: float = Query(0.95, ge=0.01, le=0.99, description="Confidence level (e.g., 0.95 for 95%)"),
    time_horizon: int = Query(1, ge=1, le=30, description="Time horizon in days"),
    method: str = Query("historical", description="Calculation method: historical, parametric, or monte_carlo")
):
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
        if method not in ['historical', 'parametric', 'monte_carlo']:
            raise HTTPException(status_code=400, detail="Method must be 'historical', 'parametric', or 'monte_carlo'")
        
        return risk_service.calculate_var(confidence_level, time_horizon, method)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in calculate_var endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/limits")
async def get_risk_limits():
    """
    Get current risk limits status
    
    Returns:
        Dictionary with risk limits and current status
    """
    try:
        from holdings.RiskLimitsManager import RiskLimitsManager
        manager = RiskLimitsManager()
        return manager.get_limits_status()
    except Exception as e:
        logger.error(f"Error in get_risk_limits endpoint: {e}", exc_info=True)
        return {'error': str(e)}


@router.get("/limits/check")
async def check_risk_limits():
    """
    Check if any risk limits are breached
    
    Returns:
        Dictionary with alerts and breach status
    """
    try:
        from holdings.RiskLimitsManager import RiskLimitsManager
        manager = RiskLimitsManager()
        return manager.check_limits()
    except Exception as e:
        logger.error(f"Error in check_risk_limits endpoint: {e}", exc_info=True)
        return {'error': str(e), 'total_alerts': 0, 'critical_alerts': [], 'high_alerts': []}


@router.get("/hedging/advanced")
async def get_advanced_hedging():
    """
    Get advanced hedging strategies
    
    Returns:
        Dictionary with hedging strategy recommendations
    """
    try:
        from holdings.AdvancedHedgingStrategies import AdvancedHedgingStrategies
        from holdings.RiskMetricsCalculator import RiskMetricsCalculator
        
        # Get portfolio metrics to calculate portfolio value and beta
        calculator = RiskMetricsCalculator()
        metrics = calculator.calculate_portfolio_risk_metrics()
        
        portfolio_value = metrics.get('portfolio_value', 0.0)
        portfolio_delta = metrics.get('beta', 1.0)  # Use beta as delta proxy
        
        if portfolio_value == 0:
            return {
                'error': 'No portfolio value found',
                'portfolio_value': 0.0,
                'portfolio_delta': 0.0,
                'strategies': {}
            }
        
        analyzer = AdvancedHedgingStrategies()
        strategies = analyzer.get_all_advanced_strategies(portfolio_value, portfolio_delta)
        
        # Format strategies for frontend
        formatted_strategies = {}
        for strategy_type, strategy_data in strategies.items():
            if isinstance(strategy_data, dict) and 'error' not in strategy_data:
                if strategy_type not in formatted_strategies:
                    formatted_strategies[strategy_type] = []
                
                # Handle correlation_hedging which may return multiple strategies
                if strategy_type == 'correlation_hedging' and 'strategies' in strategy_data:
                    # Multiple strategies returned
                    for strategy in strategy_data['strategies']:
                        formatted_strategies[strategy_type].append({
                            'strategy_name': strategy.get('strategy_name', strategy.get('strategy_type', strategy_type).replace('_', ' ').title()),
                            'strategy_type': strategy_type,
                            'rationale': strategy.get('rationale', ''),
                            'action': strategy.get('action', ''),
                            'expected_impact': strategy.get('expected_impact', ''),
                            'contracts': strategy.get('contracts', strategy.get('quantity', 0)),
                            'underlying': strategy.get('underlying', strategy.get('instrument', '')),
                            'instrument': strategy.get('instrument', ''),
                            'strike': strategy.get('strike', ''),
                            'option_type': strategy.get('option_type', ''),
                            'futures_price': strategy.get('futures_price'),
                            'contract_value': strategy.get('contract_value'),
                            'margin_required': strategy.get('margin_required'),
                            'estimated_premium_per_lot': strategy.get('estimated_premium_per_lot'),
                            'total_premium': strategy.get('total_premium'),
                            'existing_gold_units': strategy.get('existing_gold_units'),
                            'existing_silver_units': strategy.get('existing_silver_units'),
                            'existing_gold_value': strategy.get('existing_gold_value'),
                            'existing_silver_value': strategy.get('existing_silver_value'),
                            'current_allocation_pct': strategy.get('current_allocation_pct'),
                            'target_allocation_pct': strategy.get('target_allocation_pct')
                        })
                    # Add summary if available
                    if 'summary' in strategy_data:
                        formatted_strategies[strategy_type].append({
                            'strategy_name': 'Summary',
                            'strategy_type': strategy_type,
                            'rationale': strategy_data.get('summary', ''),
                            'action': 'info',
                            'expected_impact': ''
                        })
                else:
                    # Single strategy dict - convert to list format
                    formatted_strategies[strategy_type].append({
                        'strategy_name': strategy_data.get('strategy_name', strategy_data.get('strategy_type', strategy_type).replace('_', ' ').title()),
                        'strategy_type': strategy_type,
                        'rationale': strategy_data.get('rationale', ''),
                        'action': strategy_data.get('action', ''),
                        'expected_impact': strategy_data.get('expected_impact', ''),
                        'contracts': strategy_data.get('contracts', strategy_data.get('quantity', 0)),
                        'underlying': strategy_data.get('underlying', strategy_data.get('instrument', '')),
                        'instrument': strategy_data.get('instrument', ''),
                        'strike': strategy_data.get('strike', ''),
                        'option_type': strategy_data.get('option_type', ''),
                        'futures_price': strategy_data.get('futures_price'),
                        'contract_value': strategy_data.get('contract_value'),
                        'margin_required': strategy_data.get('margin_required'),
                        'estimated_premium_per_lot': strategy_data.get('estimated_premium_per_lot'),
                        'total_premium': strategy_data.get('total_premium'),
                        'existing_gold_units': strategy_data.get('existing_gold_units'),
                        'existing_silver_units': strategy_data.get('existing_silver_units'),
                        'existing_gold_value': strategy_data.get('existing_gold_value'),
                        'existing_silver_value': strategy_data.get('existing_silver_value'),
                        'current_allocation_pct': strategy_data.get('current_allocation_pct'),
                        'target_allocation_pct': strategy_data.get('target_allocation_pct')
                    })
        
        return {
            'portfolio_value': portfolio_value,
            'portfolio_delta': portfolio_delta,
            'strategies': formatted_strategies
        }
    except Exception as e:
        logger.error(f"Error in get_advanced_hedging endpoint: {e}", exc_info=True)
        import traceback
        logger.error(traceback.format_exc())
        return {'error': str(e), 'strategies': {}, 'portfolio_value': 0.0, 'portfolio_delta': 0.0}


@router.get("/import-status")
async def get_import_status():
    """
    Get import status for debugging
    
    Returns:
        Dictionary with import status information
    """
    try:
        return risk_service.get_import_status()
    except Exception as e:
        logger.error(f"Error in get_import_status endpoint: {e}", exc_info=True)
        return {'error': str(e)}

