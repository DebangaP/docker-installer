"""
API routers organized by domain.
Each router handles endpoints for a specific domain (holdings, market, options, etc.).
"""
from api.routers.holdings import router as holdings_router
from api.routers.risk_management import router as risk_router
from api.routers.alternate_data import router as alternate_data_router

__all__ = ["holdings_router", "risk_router", "alternate_data_router"]

