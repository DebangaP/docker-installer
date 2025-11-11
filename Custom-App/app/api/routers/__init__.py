"""
API routers organized by domain.
Each router handles endpoints for a specific domain (holdings, market, options, etc.).
"""
from api.routers.holdings import router as holdings_router

__all__ = ["holdings_router"]

