"""
Cache utility functions for API responses.
Provides cache header generation and cached response helpers.
"""
from datetime import datetime, timedelta
from fastapi.responses import JSONResponse
import logging


def get_cache_headers(endpoint_path: str) -> dict:
    """
    Returns Cache-Control header based on endpoint refresh frequency
    Based on crontab refresh intervals:
    - Options data: 5 minutes
    - Holdings: 5 minutes  
    - OHLC: 30 minutes
    - Derivatives suggestions: 1 minute
    - System status: 10 seconds
    - Market data (bias, order flow): 5 minutes
    - Gainers/Losers: 1 hour
    """
    # Cache durations in seconds based on refresh frequency
    cache_durations = {
        # Options data - 5 minutes (300 seconds)
        'options_latest': 300,
        'options_scanner': 300,
        'options_chain': 300,
        'options_data': 300,
        
        # Holdings - 5 minutes (300 seconds)
        'holdings': 300,
        'mf_holdings': 300,
        'today_pnl_summary': 300,
        'portfolio_history': 300,
        'holdings/patterns': 300,
        
        # Derivatives - 1 minute (60 seconds)
        'derivatives_suggestions': 60,
        'derivatives_history': 60,
        
        # Market data - 5 minutes (300 seconds)
        'market_bias': 300,
        'futures_order_flow': 300,
        'premarket_analysis': 300,
        'market_dashboard': 300,
        
        # System status - 10 seconds
        'system_status': 10,
        
        # Positions - 5 minutes
        'positions': 300,
        
        # Gainers/Losers - 30 seconds (to match polling interval)
        'gainers': 30,
        'losers': 30,
        'gainers_losers': 30,
        'top_gainers': 30,
        
        # Swing trades - 30 minutes (1800 seconds)
        'swing_trades': 1800,
        'swing_trades_nifty': 1800,
        'swing_trades_history': 1800,
        
        # Portfolio hedge - 5 minutes
        'portfolio_hedge_analysis': 300,
        
        # Sparklines and charts - 15 minutes (900 seconds)
        'sparkline': 900,
        'sparklines': 900,
        'candlestick_chart': 300,
        'candlestick': 300,
        
        # Margin - 5 minutes
        'margin_data': 300,
        'margin/available': 300,
        'margin/calculate': 300,
    }
    
    # Find matching cache duration
    for key, duration in cache_durations.items():
        if key in endpoint_path:
            # Calculate expiration time
            expires_time = datetime.now() + timedelta(seconds=duration)
            expires_str = expires_time.strftime('%a, %d %b %Y %H:%M:%S GMT')
            return {
                'Cache-Control': f'public, max-age={duration}, s-maxage={duration}, stale-while-revalidate=60',
                'Expires': expires_str
            }
    
    # Default: 5 minutes for most endpoints
    default_duration = 300
    expires_time = datetime.now() + timedelta(seconds=default_duration)
    expires_str = expires_time.strftime('%a, %d %b %Y %H:%M:%S GMT')
    return {
        'Cache-Control': 'public, max-age=300, s-maxage=300, stale-while-revalidate=60',
        'Expires': expires_str
    }


def cached_json_response(data: dict, endpoint_path: str) -> JSONResponse:
    """
    Returns a JSONResponse with appropriate cache headers
    """
    headers = get_cache_headers(endpoint_path)
    return JSONResponse(content=data, headers=headers)


def cache_get_json(key: str):
    """
    Get JSON data from Redis cache
    """
    from common.Boilerplate import redis_client
    import json
    
    try:
        cached = redis_client.get(key)
        if cached:
            if isinstance(cached, bytes):
                cached = cached.decode('utf-8')
            return json.loads(cached)
    except Exception as e:
        logging.error(f"Error getting cache for key {key}: {e}")
    return None


def cache_set_json(key: str, value: dict, ttl_seconds: int = 5):
    """
    Set JSON data in Redis cache with TTL
    """
    from common.Boilerplate import redis_client
    import json
    
    try:
        redis_client.setex(key, ttl_seconds, json.dumps(value))
    except Exception as e:
        logging.error(f"Error setting cache for key {key}: {e}")


def cache_delete_json(key: str):
    """
    Delete a specific cache key from Redis
    """
    from common.Boilerplate import redis_client
    
    try:
        redis_client.delete(key)
        logging.debug(f"Deleted cache key: {key}")
    except Exception as e:
        logging.error(f"Error deleting cache for key {key}: {e}")


def cache_delete_pattern(pattern: str):
    """
    Delete all cache keys matching a pattern from Redis
    Uses SCAN to find matching keys and deletes them
    """
    from common.Boilerplate import redis_client
    
    try:
        deleted_count = 0
        cursor = 0
        while True:
            cursor, keys = redis_client.scan(cursor, match=pattern, count=100)
            if keys:
                deleted = redis_client.delete(*keys)
                deleted_count += deleted if isinstance(deleted, int) else len(keys)
            if cursor == 0:
                break
        logging.info(f"Deleted {deleted_count} cache keys matching pattern: {pattern}")
        return deleted_count
    except Exception as e:
        logging.error(f"Error deleting cache for pattern {pattern}: {e}")
        return 0


def cache_clear_all():
    """
    Clear all application cache keys from Redis
    Uses common cache key patterns to clear all cached data
    """
    from common.Boilerplate import redis_client
    
    try:
        total_deleted = 0
        
        # Common cache key patterns used in the application
        cache_patterns = [
            "holdings:*",
            "today_pnl_summary",
            "options_*",
            "market_*",
            "gainers*",
            "losers*",
            "swing_trades*",
            "portfolio_*",
            "sparkline*",
            "candlestick*",
            "margin_*",
            "derivatives_*",
            "premarket_*",
            "futures_*",
            "mf_*",
        ]
        
        # Clear each pattern
        for pattern in cache_patterns:
            deleted = cache_delete_pattern(pattern)
            total_deleted += deleted
        
        # Also clear supertrend cache (in-memory cache)
        try:
            from api.utils.supertrend_cache import clear_cache as clear_supertrend_cache
            clear_supertrend_cache()
        except Exception as e:
            logging.warning(f"Error clearing supertrend cache: {e}")
        
        logging.info(f"Cleared all cache. Total keys deleted: {total_deleted}")
        return {
            "success": True,
            "message": f"Cleared all cache successfully",
            "keys_deleted": total_deleted
        }
    except Exception as e:
        logging.error(f"Error clearing all cache: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "keys_deleted": 0
        }