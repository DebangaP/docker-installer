"""
Alternate Data Router
API endpoints for alternate data indicators
"""

from fastapi import APIRouter, Query
from typing import Optional
import logging
from api.services.alternate_data_service import AlternateDataService

router = APIRouter(prefix="/api/alternate-data", tags=["alternate-data"])
logger = logging.getLogger(__name__)


@router.get("/increased-volume")
async def get_increased_volume(
    min_volume_increase_pct: float = Query(50.0, ge=0, description="Minimum volume increase percentage"),
    lookback_days: int = Query(5, ge=1, le=30, description="Number of days to compare against")
):
    """Get stocks with increased volume"""
    try:
        service = AlternateDataService()
        results = service.get_stocks_with_increased_volume(
            min_volume_increase_pct=min_volume_increase_pct,
            lookback_days=lookback_days
        )
        return {
            "success": True,
            "count": len(results),
            "stocks": results
        }
    except Exception as e:
        logger.error(f"Error getting increased volume stocks: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "stocks": []
        }


@router.get("/increased-obv")
async def get_increased_obv(
    min_obv_increase_pct: float = Query(20.0, ge=0, description="Minimum OBV increase percentage"),
    lookback_days: int = Query(10, ge=1, le=60, description="Number of days to analyze")
):
    """Get stocks with increased On-Balance Volume (OBV)"""
    try:
        service = AlternateDataService()
        results = service.get_stocks_with_increased_obv(
            min_obv_increase_pct=min_obv_increase_pct,
            lookback_days=lookback_days
        )
        return {
            "success": True,
            "count": len(results),
            "stocks": results
        }
    except Exception as e:
        logger.error(f"Error getting increased OBV stocks: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "stocks": []
        }


@router.get("/supertrend-change")
async def get_supertrend_change(
    lookback_days: int = Query(5, ge=1, le=30, description="Number of days to check for changes")
):
    """Get stocks with recent supertrend changes"""
    try:
        service = AlternateDataService()
        results = service.get_stocks_with_supertrend_change(
            lookback_days=lookback_days
        )
        return {
            "success": True,
            "count": len(results),
            "stocks": results
        }
    except Exception as e:
        logger.error(f"Error getting supertrend change stocks: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "stocks": []
        }


@router.get("/golden-cross")
async def get_golden_cross(
    fast_ma: int = Query(50, ge=5, le=100, description="Fast moving average period"),
    slow_ma: int = Query(200, ge=50, le=500, description="Slow moving average period"),
    lookback_days: int = Query(5, ge=1, le=30, description="Number of days to check for cross")
):
    """Get stocks with golden cross (fast MA crossing above slow MA)"""
    try:
        service = AlternateDataService()
        results = service.get_stocks_with_golden_cross(
            fast_ma=fast_ma,
            slow_ma=slow_ma,
            lookback_days=lookback_days
        )
        return {
            "success": True,
            "count": len(results),
            "stocks": results
        }
    except Exception as e:
        logger.error(f"Error getting golden cross stocks: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "stocks": []
        }


@router.get("/rsi-signals")
async def get_rsi_signals(
    rsi_period: int = Query(14, ge=5, le=30, description="RSI calculation period"),
    oversold_threshold: float = Query(30.0, ge=0, le=50, description="RSI oversold threshold"),
    overbought_threshold: float = Query(70.0, ge=50, le=100, description="RSI overbought threshold"),
    lookback_days: int = Query(10, ge=1, le=60, description="Number of days to analyze")
):
    """Get stocks with RSI signals (oversold or overbought)"""
    try:
        service = AlternateDataService()
        results = service.get_stocks_with_rsi_signals(
            rsi_period=rsi_period,
            oversold_threshold=oversold_threshold,
            overbought_threshold=overbought_threshold,
            lookback_days=lookback_days
        )
        return {
            "success": True,
            "count": len(results),
            "stocks": results
        }
    except Exception as e:
        logger.error(f"Error getting RSI signals: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "stocks": []
        }


@router.get("/macd-crossover")
async def get_macd_crossover(
    fast: int = Query(12, ge=5, le=30, description="Fast EMA period"),
    slow: int = Query(26, ge=10, le=50, description="Slow EMA period"),
    signal: int = Query(9, ge=3, le=20, description="Signal line period"),
    lookback_days: int = Query(5, ge=1, le=30, description="Number of days to check for crossover")
):
    """Get stocks with MACD crossover signals"""
    try:
        service = AlternateDataService()
        results = service.get_stocks_with_macd_crossover(
            fast=fast,
            slow=slow,
            signal=signal,
            lookback_days=lookback_days
        )
        return {
            "success": True,
            "count": len(results),
            "stocks": results
        }
    except Exception as e:
        logger.error(f"Error getting MACD crossover stocks: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "stocks": []
        }


@router.get("/atr-expansion")
async def get_atr_expansion(
    min_atr_increase_pct: float = Query(30.0, ge=0, description="Minimum ATR increase percentage"),
    atr_period: int = Query(14, ge=5, le=30, description="ATR calculation period"),
    lookback_days: int = Query(10, ge=1, le=60, description="Number of days to compare")
):
    """Get stocks with ATR expansion (increased volatility)"""
    try:
        service = AlternateDataService()
        results = service.get_stocks_with_atr_expansion(
            min_atr_increase_pct=min_atr_increase_pct,
            atr_period=atr_period,
            lookback_days=lookback_days
        )
        return {
            "success": True,
            "count": len(results),
            "stocks": results
        }
    except Exception as e:
        logger.error(f"Error getting ATR expansion stocks: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "stocks": []
        }


@router.get("/volume-trend")
async def get_volume_trend(
    trend_type: str = Query("increasing", regex="^(increasing|decreasing|stable)$", description="Type of volume trend"),
    lookback_days: int = Query(10, ge=1, le=60, description="Number of days to analyze")
):
    """Get stocks with specific volume trend"""
    try:
        service = AlternateDataService()
        results = service.get_stocks_with_volume_trend(
            trend_type=trend_type,
            lookback_days=lookback_days
        )
        return {
            "success": True,
            "count": len(results),
            "stocks": results
        }
    except Exception as e:
        logger.error(f"Error getting volume trend stocks: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "stocks": []
        }


@router.get("/momentum-change")
async def get_momentum_change(
    min_momentum_change_pct: float = Query(20.0, ge=0, description="Minimum momentum change percentage"),
    momentum_period: int = Query(14, ge=5, le=50, description="Momentum calculation period"),
    lookback_days: int = Query(10, ge=1, le=60, description="Number of days to compare")
):
    """Get stocks with significant momentum changes"""
    try:
        service = AlternateDataService()
        results = service.get_stocks_with_momentum_change(
            min_momentum_change_pct=min_momentum_change_pct,
            momentum_period=momentum_period,
            lookback_days=lookback_days
        )
        return {
            "success": True,
            "count": len(results),
            "stocks": results
        }
    except Exception as e:
        logger.error(f"Error getting momentum change stocks: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "stocks": []
        }


@router.get("/rsi-divergence")
async def get_rsi_divergence(
    rsi_period: int = Query(14, ge=5, le=30, description="RSI calculation period"),
    lookback_days: int = Query(20, ge=10, le=60, description="Number of days to analyze"),
    min_divergence_strength: float = Query(0.05, ge=0.01, le=0.2, description="Minimum divergence strength (5% = 0.05)")
):
    """Get stocks with RSI divergence patterns"""
    try:
        service = AlternateDataService()
        results = service.get_stocks_with_rsi_divergence(
            rsi_period=rsi_period,
            lookback_days=lookback_days,
            min_divergence_strength=min_divergence_strength
        )
        return {
            "success": True,
            "count": len(results),
            "stocks": results
        }
    except Exception as e:
        logger.error(f"Error getting RSI divergence: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "stocks": []
        }


@router.get("/all")
async def get_all_alternate_data(
    volume_threshold: float = Query(50.0, ge=0, description="Minimum volume increase percentage"),
    obv_threshold: float = Query(20.0, ge=0, description="Minimum OBV increase percentage"),
    lookback_days: int = Query(10, ge=1, le=60, description="Number of days to analyze")
):
    """Get all alternate data indicators"""
    try:
        service = AlternateDataService()
        results = service.get_all_alternate_data(
            volume_threshold=volume_threshold,
            obv_threshold=obv_threshold,
            lookback_days=lookback_days
        )
        return {
            "success": True,
            "data": results,
            "counts": {
                "increased_volume": len(results.get('increased_volume', [])),
                "increased_obv": len(results.get('increased_obv', [])),
                "supertrend_change": len(results.get('supertrend_change', [])),
                "golden_cross": len(results.get('golden_cross', [])),
                "rsi_signals": len(results.get('rsi_signals', [])),
                "macd_crossover": len(results.get('macd_crossover', [])),
                "atr_expansion": len(results.get('atr_expansion', [])),
                "volume_trend_increasing": len(results.get('volume_trend_increasing', [])),
                "momentum_change": len(results.get('momentum_change', [])),
                "rsi_divergence": len(results.get('rsi_divergence', []))
            }
        }
    except Exception as e:
        logger.error(f"Error getting all alternate data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "data": {
                "increased_volume": [],
                "increased_obv": [],
                "supertrend_change": [],
                "golden_cross": [],
                "rsi_signals": [],
                "macd_crossover": [],
                "atr_expansion": [],
                "volume_trend_increasing": [],
                "momentum_change": [],
                "rsi_divergence": []
            }
        }

