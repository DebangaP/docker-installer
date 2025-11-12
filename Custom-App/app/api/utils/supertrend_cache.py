"""
Supertrend cache utility for storing and retrieving pre-calculated supertrend values.
This helps avoid blocking the dashboard with expensive calculations.
"""
import logging
from typing import Optional, Dict, Tuple
from datetime import date, datetime, timedelta
from common.Boilerplate import get_db_connection
import psycopg2.extras

# In-memory cache with TTL (Time To Live)
_supertrend_cache: Dict[str, Tuple[Tuple, datetime]] = {}
CACHE_TTL_MINUTES = 30  # Cache for 30 minutes

def get_cached_supertrend(scrip_id: str) -> Optional[Tuple]:
    """
    Get cached supertrend value for a stock.
    
    Returns:
        Tuple of (supertrend_value, direction, close_price, days_below_supertrend) or None
    """
    cache_key = scrip_id.upper()
    
    if cache_key in _supertrend_cache:
        value, cached_time = _supertrend_cache[cache_key]
        # Check if cache is still valid
        if datetime.now() - cached_time < timedelta(minutes=CACHE_TTL_MINUTES):
            logging.debug(f"Using cached supertrend for {scrip_id}")
            return value
        else:
            # Cache expired, remove it
            del _supertrend_cache[cache_key]
            logging.debug(f"Cache expired for {scrip_id}, removing")
    
    return None

def set_cached_supertrend(scrip_id: str, value: Tuple):
    """
    Cache supertrend value for a stock.
    
    Args:
        scrip_id: Stock symbol
        value: Tuple of (supertrend_value, direction, close_price, days_below_supertrend)
    """
    cache_key = scrip_id.upper()
    _supertrend_cache[cache_key] = (value, datetime.now())
    logging.debug(f"Cached supertrend for {scrip_id}")

def clear_cache():
    """Clear all cached supertrend values."""
    _supertrend_cache.clear()
    logging.info("Cleared supertrend cache")

def get_cache_stats() -> Dict:
    """Get cache statistics."""
    now = datetime.now()
    valid_count = 0
    expired_count = 0
    
    for cache_key, (value, cached_time) in _supertrend_cache.items():
        if now - cached_time < timedelta(minutes=CACHE_TTL_MINUTES):
            valid_count += 1
        else:
            expired_count += 1
    
    return {
        "total_entries": len(_supertrend_cache),
        "valid_entries": valid_count,
        "expired_entries": expired_count
    }

