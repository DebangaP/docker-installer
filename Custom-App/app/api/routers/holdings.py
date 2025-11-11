"""
Holdings router for all holdings, positions, mutual funds, and portfolio-related endpoints.
"""
from fastapi import APIRouter, Query, Body
from typing import Optional, List
from common.Boilerplate import get_db_connection, kite
import psycopg2.extras
import logging
import json
import io
from io import StringIO
import pandas as pd
import os
from datetime import datetime, date
from fastapi.responses import StreamingResponse

# Optional imports for PDF generation (only needed for PDF download endpoints)
# Note: These imports may show IDE warnings if reportlab is not installed in the dev environment
# The try/except block ensures the module loads correctly even without reportlab
try:
    from reportlab.lib.pagesizes import letter, A4  # type: ignore
    from reportlab.lib import colors  # type: ignore
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image  # type: ignore
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle  # type: ignore
    from reportlab.lib.units import inch  # type: ignore
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    # Define dummy values to prevent errors if reportlab is not available
    letter = A4 = None
    colors = None
    SimpleDocTemplate = Table = TableStyle = Paragraph = Spacer = PageBreak = Image = None
    getSampleStyleSheet = ParagraphStyle = None
    inch = None

# Optional imports for chart generation (only needed for PDF download endpoints)
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    mdates = None

from api.utils.cache import cached_json_response, cache_get_json, cache_set_json
from api.utils.technical_indicators import get_latest_supertrend
from api.services.holdings_service import HoldingsService

router = APIRouter(prefix="/api", tags=["holdings"])

# Initialize service
holdings_service = HoldingsService()


@router.get("/holdings")
async def api_holdings(page: int = Query(1, ge=1), per_page: int = Query(10, ge=1), sort_by: str = Query("below_supertrend"), sort_dir: str = Query("desc"), search: str = Query(None)):
    """API endpoint to get paginated holdings data with sorting and GTT info
    Default sort: by Supertrend Alert (desc) - stocks below supertrend shown first
    """
    print(f"---XXX--- api_holdings called: page={page}, per_page={per_page}, sort_by={sort_by}, search={search}")
    try:
        # Small cache for hot endpoint (exclude search from cache key for real-time search)
        if not search:
            cache_key = f"holdings:{page}:{per_page}:{sort_by}:{sort_dir}"
            cached = cache_get_json(cache_key)
            if cached:
                print(f"---XXX--- Returning cached result for {cache_key}")
                return cached
        print(f"---XXX--- Not using cache, proceeding with fresh data")
        # If sorting by today_pnl, prophet_prediction_pct, below_supertrend, or accumulation_state, we need to get all holdings, enrich them, sort, then paginate
        if sort_by == 'today_pnl' or sort_by == 'prophet_prediction_pct' or sort_by == 'below_supertrend' or sort_by == 'accumulation_state':
            # Get all holdings without pagination
            holdings_info = holdings_service.get_holdings_data(page=1, per_page=10000, sort_by='trading_symbol', sort_dir='asc', search=search)
        else:
            holdings_info = holdings_service.get_holdings_data(page=page, per_page=per_page, sort_by=sort_by, sort_dir=sort_dir, search=search)
        
        # Get all active GTTs for holdings lookup
        from kite.KiteGTT import KiteGTTManager
        manager = KiteGTTManager(kite)
        all_gtts = manager.get_all_gtts()
        
        # Create a map of tradingsymbol to GTT info
        gtt_map = {}
        for gtt in all_gtts:
            symbol = gtt.get('tradingsymbol', '')
            if symbol:
                gtt_map[symbol] = {
                    'trigger_id': gtt.get('trigger_id') or gtt.get('id'),
                    'trigger_price': gtt.get('trigger_price'),
                    'order_price': gtt.get('order_price'),
                    'status': gtt.get('status', 'ACTIVE')
                }
        
        # Get today's P&L prices for all holdings
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Get today's and previous day's prices
        cursor.execute("SELECT CURRENT_DATE as today_date")
        today = cursor.fetchone()['today_date']
        today_str = today.strftime('%Y-%m-%d')
        
        cursor.execute("""
            SELECT MAX(price_date) as prev_date
            FROM my_schema.rt_intraday_price
            WHERE price_date < %s
            and scrip_id not in ('BITCOIN', 'SOLANA', 'DOGE', 'ETH')
        """, (today_str,))
        
        prev_date_result = cursor.fetchone()
        prev_date = prev_date_result['prev_date'] if prev_date_result and prev_date_result['prev_date'] else None
        
        # Create a map of today's P&L per holding by instrument_token
        today_pnl_map = {}
        
        # Always populate the map with holdings first, even if prev_date doesn't exist (will default to 0)
        # First, get all holdings to ensure we have entries for all of them
        cursor.execute("""
            SELECT instrument_token, trading_symbol, quantity
            FROM my_schema.holdings
            WHERE run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
        """)
        
        for row in cursor.fetchall():
            instrument_token = row['instrument_token']
            # Initialize with 0 values for all holdings
            today_pnl_map[instrument_token] = {
                'today_pnl': 0.0,
                'today_price': 0.0,
                'prev_price': 0.0,
                'pct_change': 0.0,
                'today_change': '0 (0%)'
            }
        
        if prev_date:
            # Convert prev_date to string format safely
            try:
                if hasattr(prev_date, 'strftime'):
                    prev_date_str = prev_date.strftime('%Y-%m-%d')
                elif isinstance(prev_date, str):
                    prev_date_str = prev_date
                else:
                    prev_date_str = str(prev_date)
            except Exception as e:
                logging.error(f"Error converting prev_date to string: {e}, prev_date={prev_date}")
                prev_date_str = None
            
            if prev_date_str:
                cursor.execute("""
                    WITH holdings_today AS (
                        SELECT instrument_token, trading_symbol, quantity, last_price
                        FROM my_schema.holdings
                        WHERE run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
                    ),
                    today_prices AS (
                        SELECT scrip_id, price_close
                        FROM my_schema.rt_intraday_price
                        WHERE price_date = %s
                    ),
                    latest_prices AS (
                        SELECT scrip_id, price_close, 
                               ROW_NUMBER() OVER (PARTITION BY scrip_id ORDER BY price_date DESC) as rn
                        FROM my_schema.rt_intraday_price
                        WHERE price_date::date <= CURRENT_DATE
                    ),
                    prev_prices AS (
                        SELECT scrip_id, price_close
                        FROM my_schema.rt_intraday_price
                        WHERE price_date = %s
                    )
                    SELECT 
                        h.instrument_token,
                        h.trading_symbol,
                        h.quantity,
                        COALESCE(h.last_price, today_p.price_close, latest_p.price_close, 0) as today_price,
                        COALESCE(prev_p.price_close, 0) as prev_price,
                        -- If today's price is null/0, P&L should be 0 regardless of prev price
                        -- If prev_price is 0, P&L should be 0 (no previous data to compare)
                        CASE 
                            WHEN (COALESCE(h.last_price, today_p.price_close, latest_p.price_close) IS NULL OR COALESCE(h.last_price, today_p.price_close, latest_p.price_close) = 0) THEN 0
                            WHEN (COALESCE(prev_p.price_close, 0) = 0) THEN 0
                            ELSE h.quantity * (COALESCE(h.last_price, today_p.price_close, latest_p.price_close, 0) - COALESCE(prev_p.price_close, 0))
                        END as today_pnl,
                        round(case when prev_p.price_close != 0 AND COALESCE(h.last_price, today_p.price_close, latest_p.price_close) IS NOT NULL AND COALESCE(h.last_price, today_p.price_close, latest_p.price_close) != 0 then 
    	                    ((COALESCE(h.last_price, today_p.price_close, latest_p.price_close, 0) - COALESCE(prev_p.price_close, 0))/COALESCE(prev_p.price_close, 0)) * 100
    	                else 0
    	                end::numeric, 2) as pct_change,
                        concat(round(CASE 
                            WHEN (COALESCE(h.last_price, today_p.price_close, latest_p.price_close) IS NULL OR COALESCE(h.last_price, today_p.price_close, latest_p.price_close) = 0) THEN 0
                            WHEN (COALESCE(prev_p.price_close, 0) = 0) THEN 0
                            ELSE h.quantity * (COALESCE(h.last_price, today_p.price_close, latest_p.price_close, 0) - COALESCE(prev_p.price_close, 0))
                        END), ' (',
                        round(case when prev_p.price_close != 0 AND COALESCE(h.last_price, today_p.price_close, latest_p.price_close) IS NOT NULL AND COALESCE(h.last_price, today_p.price_close, latest_p.price_close) != 0 then 
    	                    ((COALESCE(h.last_price, today_p.price_close, latest_p.price_close, 0) - COALESCE(prev_p.price_close, 0))/COALESCE(prev_p.price_close, 0)) * 100
    	                else 0
    	                end::numeric, 2), '%%)') as "Todays_Change"
                    FROM holdings_today h
                    LEFT JOIN today_prices today_p ON h.trading_symbol = today_p.scrip_id
                    LEFT JOIN latest_prices latest_p ON h.trading_symbol = latest_p.scrip_id AND latest_p.rn = 1
                    LEFT JOIN prev_prices prev_p ON h.trading_symbol = prev_p.scrip_id
                """, (today_str, prev_date_str))
                
                for row in cursor.fetchall():
                    instrument_token = row['instrument_token']
                    today_price = float(row['today_price']) if row['today_price'] else 0.0
                    prev_price = float(row['prev_price']) if row['prev_price'] else 0.0
                    today_pnl = float(row['today_pnl']) if row['today_pnl'] else 0.0
                    pct_change = float(row['pct_change']) if row['pct_change'] else 0.0
                    today_change_str = str(row.get("Todays_Change", '')) if row.get("Todays_Change") else ''
                    
                    # Store by instrument_token to match individual holdings
                    today_pnl_map[instrument_token] = {
                        'today_pnl': today_pnl,
                        'today_price': today_price,
                        'prev_price': prev_price,
                        'pct_change': pct_change,
                        'today_change': today_change_str
                    }
        
        # Get Prophet predictions for all holdings
        conn2 = get_db_connection()
        cursor2 = conn2.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        try:
            # Get Prophet predictions - prefer 60-day, fallback to latest available
            cursor2.execute("""
                SELECT DISTINCT ON (scrip_id) 
                    scrip_id, 
                    predicted_price_change_pct, 
                    prediction_confidence,
                    prediction_days,
                    prediction_details
                FROM my_schema.prophet_predictions
                WHERE status = 'ACTIVE'
                ORDER BY scrip_id, 
                         CASE WHEN prediction_days = 60 THEN 1 
                              WHEN prediction_days = 30 THEN 2 
                              ELSE 3 END,
                         run_date DESC
            """)
            
            predictions_rows = cursor2.fetchall()
            predictions_map = {row['scrip_id'].upper(): dict(row) for row in predictions_rows}
            logging.debug(f"Loaded {len(predictions_map)} Prophet predictions for holdings")
        except Exception as e:
            logging.error(f"Error loading Prophet predictions for holdings: {e}")
            predictions_map = {}
        finally:
            cursor2.close()
            conn2.close()
        
        conn.close()
        
        # Get accumulation/distribution data for all holdings
        from indicators.AccumulationDistributionAnalyzer import AccumulationDistributionAnalyzer
        analyzer = AccumulationDistributionAnalyzer()
        accumulation_map = {}
        
        try:
            for holding in holdings_info["holdings"]:
                symbol = holding["trading_symbol"]
                if symbol:
                    state_data = analyzer.get_current_state(symbol, date.today())
                    if state_data:
                        accumulation_map[symbol] = state_data
        except Exception as e:
            logging.debug(f"Error loading accumulation/distribution data: {e}")
            accumulation_map = {}
        
        # Convert holdings to serializable format
        holdings_list = []
        print(f"---XXX--- Starting holdings loop, total holdings: {len(holdings_info.get('holdings', []))}")
        for holding in holdings_info["holdings"]:
            symbol = holding["trading_symbol"]
            instrument_token = holding["instrument_token"]
            print(f"---XXX--- Processing holding: {symbol}")
            today_pnl_info = today_pnl_map.get(instrument_token, {'today_pnl': 0.0, 'today_price': 0.0, 'prev_price': 0.0, 'pct_change': 0.0, 'today_change': '0 (0%)'})
            
            # Get Prophet prediction for this holding
            prophet_pred = None
            prophet_conf = None
            prophet_pred_days = None
            prophet_cv_mape = None
            prophet_cv_rmse = None
            if symbol and symbol.upper() in predictions_map:
                pred = predictions_map[symbol.upper()]
                try:
                    prophet_pred = float(pred['predicted_price_change_pct']) if pred.get('predicted_price_change_pct') is not None else None
                    prophet_conf = float(pred['prediction_confidence']) if pred.get('prediction_confidence') is not None else None
                    prophet_pred_days = int(pred['prediction_days']) if pred.get('prediction_days') is not None else None
                    
                    # Extract cv_metrics from prediction_details
                    prediction_details = pred.get('prediction_details')
                    if prediction_details:
                        try:
                            # PostgreSQL JSONB might return as dict or string depending on psycopg2 version
                            if isinstance(prediction_details, str):
                                details = json.loads(prediction_details)
                            elif isinstance(prediction_details, dict):
                                details = prediction_details
                            else:
                                # Try to convert to string first
                                details = json.loads(str(prediction_details))
                            
                            cv_metrics = details.get('cv_metrics')
                            if cv_metrics and isinstance(cv_metrics, dict):
                                # Handle None, infinity, or invalid values
                                mape_value = cv_metrics.get('mape')
                                rmse_value = cv_metrics.get('rmse')
                                
                                # Convert infinity strings or infinity values to None
                                if mape_value is None or mape_value == float('inf') or mape_value == float('-inf') or (isinstance(mape_value, str) and mape_value.lower() in ['inf', 'infinity', 'nan']):
                                    prophet_cv_mape = None
                                else:
                                    try:
                                        prophet_cv_mape = float(mape_value) if mape_value is not None else None
                                    except (ValueError, TypeError):
                                        prophet_cv_mape = None
                                
                                if rmse_value is None or rmse_value == float('inf') or rmse_value == float('-inf') or (isinstance(rmse_value, str) and rmse_value.lower() in ['inf', 'infinity', 'nan']):
                                    prophet_cv_rmse = None
                                else:
                                    try:
                                        prophet_cv_rmse = float(rmse_value) if rmse_value is not None else None
                                    except (ValueError, TypeError):
                                        prophet_cv_rmse = None
                                
                                logging.debug(f"Extracted cv_metrics for {symbol}: MAPE={prophet_cv_mape}, RMSE={prophet_cv_rmse}")
                            else:
                                prophet_cv_mape = None
                                prophet_cv_rmse = None
                                logging.debug(f"No cv_metrics found in prediction_details for {symbol}, cv_metrics={cv_metrics}")
                        except (json.JSONDecodeError, Exception) as e:
                            logging.warning(f"Error parsing prediction_details for {symbol}: {e}, type={type(prediction_details)}")
                except (ValueError, TypeError):
                    prophet_pred = None
                    prophet_conf = None
                    prophet_pred_days = None
                    prophet_cv_mape = None
                    prophet_cv_rmse = None
            
            # Check if stock is below supertrend
            print(f"---XXX--- About to check supertrend for {symbol}")
            below_supertrend = False
            supertrend_value = None
            days_below_supertrend = None
            try:
                print(f"---XXX--- Calling get_latest_supertrend for {symbol}")
                # Don't pass conn - let the function create its own connection
                supertrend_result = get_latest_supertrend(symbol, conn=None)
                print(f"---XXX--- Result for {symbol}: {supertrend_result}")
                if supertrend_result is not None:
                    supertrend_value, supertrend_direction, supertrend_close, days_below_supertrend = supertrend_result
                    print(f"---XXX--- Unpacked for {symbol}: value={supertrend_value}, direction={supertrend_direction}, close={supertrend_close}, days_below={days_below_supertrend}")
                    
                    # Compare the close price used in supertrend calculation with the supertrend value
                    # This is the most accurate comparison since they're from the same calculation
                    if supertrend_close is not None and supertrend_value is not None:
                        below_supertrend = supertrend_close < supertrend_value
                    else:
                        # Fallback: use direction if close price is not available
                        # direction = -1 means price is below supertrend, direction = 1 means price is above supertrend
                        below_supertrend = (supertrend_direction == -1)
                    
                    # Also verify with latest price from database as a double-check
                    cursor.execute("""
                        SELECT price_close
                        FROM my_schema.rt_intraday_price
                        WHERE scrip_id = %s
                        AND country = 'IN'
                        AND price_date::date <= CURRENT_DATE
                        AND price_close IS NOT NULL
                        ORDER BY price_date DESC
                        LIMIT 1
                    """, (symbol,))
                    price_result = cursor.fetchone()
                    
                    if price_result and price_result['price_close']:
                        latest_close_price = float(price_result['price_close'])
                        # Double-check: if supertrend value > closing price, then stock is below supertrend
                        price_below = latest_close_price < supertrend_value
                        if below_supertrend != price_below:
                            # If there's a mismatch, use the latest price comparison (more current)
                            logging.debug(f"Supertrend comparison mismatch for {symbol}: calc_close={supertrend_close}, latest_close={latest_close_price}, st={supertrend_value}, using_latest={price_below}")
                            below_supertrend = price_below
                else:
                    logging.debug(f"Supertrend result is None for {symbol}")
            except Exception as e:
                logging.warning(f"Could not check supertrend for {symbol}: {e}")
                import traceback
                logging.warning(traceback.format_exc())
            
            # Get accumulation/distribution data
            accumulation_state = None
            days_in_accumulation_state = None
            accumulation_confidence = None
            if symbol and symbol in accumulation_map:
                acc_data = accumulation_map[symbol]
                accumulation_state = acc_data.get('state')
                days_in_accumulation_state = acc_data.get('days_in_state')
                accumulation_confidence = acc_data.get('confidence_score')
            
            print(f"---XXX--- Adding to holdings_list for {symbol}: supertrend_value={supertrend_value}, below_supertrend={below_supertrend}")
            holdings_list.append({
                "trading_symbol": symbol,
                "instrument_token": instrument_token,
                "quantity": holding["quantity"],
                "average_price": float(holding["average_price"]) if holding["average_price"] else 0.0,
                "last_price": float(holding["last_price"]) if holding["last_price"] else 0.0,
                "pnl": float(holding["pnl"]) if holding["pnl"] else 0.0,
                "invested_amount": float(holding.get("invested_amount", 0)),
                "current_amount": float(holding.get("current_amount", 0)),
                "pnl_pct_change": float(holding.get("pnl_pct_change", 0)) if holding.get("pnl_pct_change") else 0.0,
                "today_pnl": today_pnl_info['today_pnl'],
                "today_price": today_pnl_info['today_price'],
                "prev_price": today_pnl_info['prev_price'],
                "pct_change": today_pnl_info.get('pct_change', 0.0),
                "today_change": today_pnl_info.get('today_change', '0 (0%)'),
                "prophet_prediction_pct": prophet_pred,
                "prophet_confidence": prophet_conf,
                "prediction_days": prophet_pred_days,
                "prophet_cv_mape": prophet_cv_mape,
                "prophet_cv_rmse": prophet_cv_rmse,
                "below_supertrend": below_supertrend,
                "supertrend_value": supertrend_value,
                "days_below_supertrend": days_below_supertrend,
                "accumulation_state": accumulation_state,
                "days_in_accumulation_state": days_in_accumulation_state,
                "accumulation_confidence": accumulation_confidence
            })
            print(f"---XXX--- Added to holdings_list for {symbol}")
        
        # If sorting by today_pnl, prophet_prediction_pct, below_supertrend, or accumulation_state, sort the enriched list and then paginate
        if sort_by == 'today_pnl':
            sort_reverse = sort_dir.lower() == 'desc'
            holdings_list.sort(key=lambda x: x['today_pnl'], reverse=sort_reverse)
            
            # Apply pagination after sorting
            total_count = len(holdings_list)
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            holdings_list = holdings_list[start_idx:end_idx]
        elif sort_by == 'prophet_prediction_pct':
            sort_reverse = sort_dir.lower() == 'desc'
            # Sort by prophet_prediction_pct, handling None values
            holdings_list.sort(key=lambda x: x['prophet_prediction_pct'] if x['prophet_prediction_pct'] is not None else (float('-inf') if sort_reverse else float('inf')), reverse=sort_reverse)
            
            # Apply pagination after sorting
            total_count = len(holdings_list)
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            holdings_list = holdings_list[start_idx:end_idx]
        elif sort_by == 'below_supertrend':
            sort_reverse = sort_dir.lower() == 'desc'
            # Sort by below_supertrend (boolean), then by supertrend_value
            # True (below) comes first in desc order, False (above) comes first in asc order
            holdings_list.sort(key=lambda x: (
                x.get('below_supertrend', False),
                x.get('supertrend_value') if x.get('supertrend_value') is not None else (float('inf') if sort_reverse else float('-inf'))
            ), reverse=sort_reverse)
            
            # Apply pagination after sorting
            total_count = len(holdings_list)
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            holdings_list = holdings_list[start_idx:end_idx]
        elif sort_by == 'accumulation_state':
            sort_reverse = sort_dir.lower() == 'desc'
            # Sort by accumulation_state (ACCUMULATION, DISTRIBUTION, NEUTRAL, None)
            # Priority: DISTRIBUTION (highest), ACCUMULATION, NEUTRAL, None (lowest)
            def get_state_priority(state):
                if state == 'DISTRIBUTION':
                    return 3
                elif state == 'ACCUMULATION':
                    return 2
                elif state == 'NEUTRAL':
                    return 1
                else:
                    return 0
            
            holdings_list.sort(key=lambda x: (
                get_state_priority(x.get('accumulation_state')),
                x.get('days_in_accumulation_state') if x.get('days_in_accumulation_state') is not None else 0
            ), reverse=sort_reverse)
            
            # Apply pagination after sorting
            total_count = len(holdings_list)
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            holdings_list = holdings_list[start_idx:end_idx]
        else:
            total_count = holdings_info["total_count"]
        
        result = {
            "holdings": holdings_list,
            "total_count": total_count,
            "page": page,
            "per_page": per_page
        }
        cache_set_json(cache_key, result, ttl_seconds=5)
        return cached_json_response(result, "/api/holdings")
    except Exception as e:
        logging.error(f"Error fetching holdings: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return cached_json_response({"error": str(e)}, "/api/holdings")


@router.post("/refresh_holdings")
async def api_refresh_holdings():
    """API endpoint to refresh holdings from Kite API"""
    try:
        from holdings.RefreshHoldings import refresh_holdings
        refresh_holdings()
        return {"success": True, "message": "Holdings refreshed successfully"}
    except Exception as e:
        logging.error(f"Error refreshing holdings: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"success": False, "error": str(e)}


@router.get("/positions")
async def api_positions():
    """API endpoint to get latest positions data"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cursor.execute("""
            SELECT 
                fetch_timestamp, 
                position_type, 
                trading_symbol, 
                product, 
                exchange, 
                average_price, 
                pnl 
            FROM my_schema.positions 
            WHERE run_date = CURRENT_DATE
            ORDER BY trading_symbol
        """)
        
        positions = cursor.fetchall()
        conn.close()
        
        # Convert to serializable format
        positions_list = []
        for position in positions:
            # Handle fetch_timestamp (could be string or datetime)
            fetch_ts = position["fetch_timestamp"]
            if fetch_ts:
                if hasattr(fetch_ts, 'strftime'):
                    fetch_ts_str = fetch_ts.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    fetch_ts_str = str(fetch_ts)
            else:
                fetch_ts_str = ""
            
            positions_list.append({
                "fetch_timestamp": fetch_ts_str,
                "position_type": position["position_type"],
                "trading_symbol": position["trading_symbol"],
                "product": position["product"],
                "exchange": position["exchange"],
                "average_price": float(position["average_price"]) if position["average_price"] else 0.0,
                "pnl": float(position["pnl"]) if position["pnl"] else 0.0
            })
        
        return cached_json_response({"positions": positions_list}, "/api/positions")
    except Exception as e:
        logging.error(f"Error fetching positions: {e}")
        return cached_json_response({"error": str(e), "positions": []}, "/api/positions")


@router.get("/mf_holdings")
async def api_mf_holdings():
    """API endpoint to get latest MF holdings data"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cursor.execute("""
            SELECT 
                folio,
                fund,
                tradingsymbol,
                isin,
                quantity,
                average_price,
                last_price,
                invested_amount,
                current_value,
                pnl,
                net_change_percentage,
                day_change_percentage
            FROM my_schema.mf_holdings 
            WHERE run_date = (SELECT MAX(run_date) FROM my_schema.mf_holdings)
            ORDER BY fund, tradingsymbol
        """)
        
        mf_holdings = cursor.fetchall()
        conn.close()
        
        # Convert to serializable format
        mf_holdings_list = []
        for mf in mf_holdings:
            mf_holdings_list.append({
                "folio": mf["folio"],
                "fund": mf["fund"],
                "tradingsymbol": mf["tradingsymbol"],
                "isin": mf["isin"],
                "quantity": float(mf["quantity"]) if mf["quantity"] else 0.0,
                "average_price": float(mf["average_price"]) if mf["average_price"] else 0.0,
                "last_price": float(mf["last_price"]) if mf["last_price"] else 0.0,
                "invested_amount": float(mf["invested_amount"]) if mf["invested_amount"] else 0.0,
                "current_value": float(mf["current_value"]) if mf["current_value"] else 0.0,
                "pnl": float(mf["pnl"]) if mf["pnl"] else 0.0,
                "net_change_percentage": float(mf["net_change_percentage"]) if mf["net_change_percentage"] else 0.0,
                "day_change_percentage": float(mf["day_change_percentage"]) if mf["day_change_percentage"] else 0.0
            })
        
        return cached_json_response({"mf_holdings": mf_holdings_list}, "/api/mf_holdings")
        
    except Exception as e:
        logging.error(f"Error fetching MF holdings: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return cached_json_response({"error": str(e), "mf_holdings": []}, "/api/mf_holdings")


@router.post("/refresh_mf_nav")
async def api_refresh_mf_nav():
    """API endpoint to refresh MF NAV data"""
    try:
        from holdings.RefreshMFNAV import refresh_mf_nav
        result = refresh_mf_nav()
        return {
            "success": result.get('success', False),
            "message": result.get('message', 'MF NAV data refreshed'),
            "mfs_processed": result.get('mfs_processed', 0),
            "records_inserted": result.get('records_inserted', 0)
        }
    except Exception as e:
        logging.error(f"Error refreshing MF NAV: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"success": False, "error": str(e)}


@router.get("/mf_nav/{mf_symbol}")
async def api_mf_nav(
    mf_symbol: str,
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(365, description="Maximum number of records")
):
    """API endpoint to get historical NAV data for a mutual fund"""
    try:
        from datetime import datetime
        from holdings.MFNAVFetcher import MFNAVFetcher
        
        # Parse dates
        start = None
        end = None
        
        if start_date:
            start = datetime.strptime(start_date, '%Y-%m-%d').date()
        if end_date:
            end = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        query = """
            SELECT nav_date, nav_value, scheme_code, fund_name
            FROM my_schema.mf_nav_history
            WHERE mf_symbol = %s
        """
        params = [mf_symbol]
        
        if start:
            query += " AND nav_date >= %s"
            params.append(start)
        if end:
            query += " AND nav_date <= %s"
            params.append(end)
        
        query += " ORDER BY nav_date DESC LIMIT %s"
        params.append(limit)
        
        cursor.execute(query, params)
        nav_records = cursor.fetchall()
        cursor.close()
        conn.close()
        
        nav_list = []
        for record in nav_records:
            nav_list.append({
                "nav_date": record["nav_date"].strftime('%Y-%m-%d') if record["nav_date"] else None,
                "nav_value": float(record["nav_value"]) if record["nav_value"] else 0.0,
                "scheme_code": record["scheme_code"],
                "fund_name": record["fund_name"]
            })
        
        return cached_json_response({
            "success": True,
            "mf_symbol": mf_symbol,
            "nav_data": nav_list,
            "count": len(nav_list)
        }, f"/api/mf_nav/{mf_symbol}")
        
    except Exception as e:
        logging.error(f"Error fetching NAV data for {mf_symbol}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return cached_json_response({
            "success": False,
            "error": str(e),
            "mf_symbol": mf_symbol,
            "nav_data": []
        }, f"/api/mf_nav/{mf_symbol}")


@router.get("/mf_benchmark_comparison")
async def api_mf_benchmark_comparison(
    mf_symbol: Optional[str] = Query(None, description="Mutual Fund symbol (if not provided, returns for all MFs)"),
    benchmark_symbol: Optional[str] = Query(None, description="Benchmark symbol (auto-detected if not provided)")
):
    """API endpoint to get MF performance vs benchmark metrics"""
    try:
        from holdings.MFPerformanceAnalyzer import MFPerformanceAnalyzer
        
        analyzer = MFPerformanceAnalyzer()
        
        if mf_symbol:
            # Get benchmark for this specific MF
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT DISTINCT tradingsymbol, fund
                FROM my_schema.mf_holdings
                WHERE tradingsymbol = %s
                LIMIT 1
            """, (mf_symbol,))
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if not result:
                return cached_json_response({
                    "success": False,
                    "error": f"MF {mf_symbol} not found in holdings"
                }, "/api/mf_benchmark_comparison")
            
            fund_name = result[1]
            performance = analyzer.analyze_mf_performance(mf_symbol, fund_name, benchmark_symbol)
            
            return cached_json_response(performance, "/api/mf_benchmark_comparison")
        else:
            # Get comparison for all MFs
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT DISTINCT tradingsymbol, fund
                FROM my_schema.mf_holdings
                WHERE tradingsymbol IS NOT NULL AND tradingsymbol != ''
                ORDER BY tradingsymbol
            """)
            
            mf_list = cursor.fetchall()
            cursor.close()
            conn.close()
            
            comparisons = []
            for mf_symbol_item, fund_name in mf_list:
                try:
                    performance = analyzer.analyze_mf_performance(mf_symbol_item, fund_name)
                    if performance.get('success'):
                        comparisons.append(performance)
                except Exception as e:
                    logging.warning(f"Error analyzing {mf_symbol_item}: {e}")
                    continue
            
            return cached_json_response({
                "success": True,
                "comparisons": comparisons,
                "count": len(comparisons)
            }, "/api/mf_benchmark_comparison")
        
    except Exception as e:
        logging.error(f"Error fetching benchmark comparison: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return cached_json_response({
            "success": False,
            "error": str(e)
        }, "/api/mf_benchmark_comparison")


@router.get("/holdings/patterns")
async def api_holdings_patterns():
    """API endpoint to get detected patterns for all holdings"""
    try:
        from stocks.SwingTradeScanner import SwingTradeScanner
        scanner = SwingTradeScanner(min_gain=10.0, max_gain=20.0, min_confidence=70.0)
        
        # Get holdings symbols
        holdings_info = holdings_service.get_holdings_data(page=1, per_page=10000, sort_by='trading_symbol', sort_dir='asc')
        symbols = [h["trading_symbol"] for h in holdings_info.get("holdings", [])]
        
        # Get patterns for each holding
        patterns_map = {}
        for symbol in symbols:
            try:
                patterns = scanner.get_patterns_for_stock(symbol)
                if patterns:
                    patterns_map[symbol] = patterns
            except Exception as e:
                logging.debug(f"Error getting patterns for {symbol}: {e}")
                continue
        
        return cached_json_response({"success": True, "patterns": patterns_map}, "/api/holdings/patterns")
        
    except Exception as e:
        logging.error(f"Error fetching holdings patterns: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return cached_json_response({"success": False, "error": str(e), "patterns": {}}, "/api/holdings/patterns")


@router.get("/today_pnl_summary")
async def api_today_pnl_summary():
    """API endpoint to get today's P&L summary for Equity, MF, and Intraday trades"""
    try:
        cached = cache_get_json("today_pnl_summary")
        if cached:
            return cached
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Find the latest date that has actual stock data (exclude crypto symbols)
        # This prevents future-dated crypto data from skewing the results
        cursor.execute("""
            SELECT MAX(price_date::date) as latest_date
            FROM my_schema.rt_intraday_price
            WHERE country = 'IN'
            AND scrip_id NOT IN ('BITCOIN', 'SOLANA', 'DOGE', 'ETH')
        """)
        latest_result = cursor.fetchone()
        latest_date = latest_result['latest_date'] if latest_result and latest_result['latest_date'] else None
        
        if not latest_date:
            logging.warning("No stock data found in rt_intraday_price (excluding crypto), using CURRENT_DATE")
            cursor.execute("SELECT CURRENT_DATE as today_date")
            today = cursor.fetchone()['today_date']
            latest_date = today
        
        latest_date_str = latest_date.strftime('%Y-%m-%d') if hasattr(latest_date, 'strftime') else str(latest_date)
        logging.info(f"Using latest stock data date: {latest_date_str} (excluding crypto symbols)")
        
        # Get previous trading day (also excluding crypto)
        cursor.execute("""
            SELECT MAX(price_date::date) as prev_date
            FROM my_schema.rt_intraday_price
            WHERE price_date::date < %s
            AND country = 'IN'
            AND scrip_id NOT IN ('BITCOIN', 'SOLANA', 'DOGE', 'ETH')
        """, (latest_date,))
        prev_date_result = cursor.fetchone()
        prev_date = prev_date_result['prev_date'] if prev_date_result and prev_date_result['prev_date'] else None
        
        # Calculate Equity P&L - sum up individual holdings P&L
        equity_pnl = 0.0
        if prev_date:
            # Convert prev_date to string format
            prev_date_str = prev_date.strftime('%Y-%m-%d') if hasattr(prev_date, 'strftime') else str(prev_date)
            cursor.execute("""
                WITH holdings_today AS (
                    SELECT instrument_token, trading_symbol, quantity, last_price
                    FROM my_schema.holdings
                    WHERE run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
                ),
                today_prices AS (
                    SELECT scrip_id, price_close
                    FROM my_schema.rt_intraday_price
                    WHERE price_date::date = %s
                    AND country = 'IN'
                    AND scrip_id NOT IN ('BITCOIN', 'SOLANA', 'DOGE', 'ETH')
                ),
                latest_prices AS (
                    SELECT scrip_id, price_close, 
                           ROW_NUMBER() OVER (PARTITION BY scrip_id ORDER BY price_date DESC) as rn
                    FROM my_schema.rt_intraday_price
                    WHERE price_date::date <= %s
                    AND country = 'IN'
                    AND scrip_id NOT IN ('BITCOIN', 'SOLANA', 'DOGE', 'ETH')
                ),
                prev_prices AS (
                    SELECT scrip_id, price_close
                    FROM my_schema.rt_intraday_price
                    WHERE price_date::date = %s
                    AND country = 'IN'
                    AND scrip_id NOT IN ('BITCOIN', 'SOLANA', 'DOGE', 'ETH')
                )
                SELECT 
                    h.instrument_token,
                    h.trading_symbol,
                    h.quantity,
                    COALESCE(h.last_price, today_p.price_close, latest_p.price_close, 0) as today_price,
                    COALESCE(prev_p.price_close, 0) as prev_price,
                    -- If either today's or previous price is missing/0, P&L should be 0
                    CASE 
                        WHEN (COALESCE(h.last_price, today_p.price_close, latest_p.price_close) IS NULL OR COALESCE(h.last_price, today_p.price_close, latest_p.price_close) = 0) THEN 0
                        WHEN (COALESCE(prev_p.price_close, 0) = 0) THEN 0
                        ELSE h.quantity * (COALESCE(h.last_price, today_p.price_close, latest_p.price_close) - prev_p.price_close)
                    END as today_pnl
                FROM holdings_today h
                LEFT JOIN today_prices today_p ON h.trading_symbol = today_p.scrip_id
                LEFT JOIN latest_prices latest_p ON h.trading_symbol = latest_p.scrip_id AND latest_p.rn = 1
                LEFT JOIN prev_prices prev_p ON h.trading_symbol = prev_p.scrip_id
            """, (latest_date_str, latest_date_str, prev_date_str))
            
            # Sum up all individual holdings P&L
            rows = cursor.fetchall()
            logging.info(f"Found {len(rows)} holdings for P&L calculation")
            for row in rows:
                holding_pnl = float(row['today_pnl']) if row['today_pnl'] else 0.0
                equity_pnl += holding_pnl
                logging.debug(f"P&L for {row.get('trading_symbol', 'unknown')}: {holding_pnl}")
            
            logging.info(f"Total equity P&L: {equity_pnl}")
        
        # Calculate MF P&L (today's change)
        cursor.execute("""
            SELECT 
                COALESCE(SUM(day_change_percentage * invested_amount / 100), 0) as mf_pnl
            FROM my_schema.mf_holdings 
            WHERE run_date = (SELECT MAX(run_date) FROM my_schema.mf_holdings)
        """)
        mf_result = cursor.fetchone()
        mf_pnl = float(mf_result['mf_pnl']) if mf_result and mf_result['mf_pnl'] else 0.0
        
        # Calculate Intraday trades P&L (from positions)
        cursor.execute("""
            SELECT 
                COALESCE(SUM(pnl), 0) as intraday_pnl
            FROM my_schema.positions 
            WHERE run_date = (SELECT MAX(run_date) FROM my_schema.positions)
            AND position_type = 'day'
        """)
        intraday_result = cursor.fetchone()
        intraday_pnl = float(intraday_result['intraday_pnl']) if intraday_result and intraday_result['intraday_pnl'] else 0.0
        
        # Calculate overall Equity P&L (total unrealized P&L from holdings)
        cursor.execute("""
            SELECT 
                COALESCE(SUM(pnl), 0) as overall_equity_pnl
            FROM my_schema.holdings 
            WHERE run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
        """)
        equity_overall_result = cursor.fetchone()
        equity_overall_pnl = float(equity_overall_result['overall_equity_pnl']) if equity_overall_result and equity_overall_result['overall_equity_pnl'] else 0.0
        
        # Calculate overall MF P&L (total P&L from mf_holdings)
        cursor.execute("""
            SELECT 
                COALESCE(SUM(pnl), 0) as overall_mf_pnl
            FROM my_schema.mf_holdings 
            WHERE run_date = (SELECT MAX(run_date) FROM my_schema.mf_holdings)
        """)
        mf_overall_result = cursor.fetchone()
        mf_overall_pnl = float(mf_overall_result['overall_mf_pnl']) if mf_overall_result and mf_overall_result['overall_mf_pnl'] else 0.0
        
        # Calculate totals
        total_today_pnl = equity_pnl + mf_pnl + intraday_pnl
        total_overall_pnl = equity_overall_pnl + mf_overall_pnl
        
        conn.close()
        
        result = {
            "equity_pnl": equity_pnl,
            "equity_overall_pnl": equity_overall_pnl,
            "mf_pnl": mf_pnl,
            "mf_overall_pnl": mf_overall_pnl,
            "intraday_pnl": intraday_pnl,
            "total_today_pnl": total_today_pnl,
            "total_overall_pnl": total_overall_pnl
        }
        cache_set_json("today_pnl_summary", result, ttl_seconds=5)
        return cached_json_response(result, "/api/today_pnl_summary")
    except Exception as e:
        logging.error(f"Error fetching today's P&L summary: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return cached_json_response({
            "equity_pnl": 0.0,
            "equity_overall_pnl": 0.0,
            "mf_pnl": 0.0,
            "mf_overall_pnl": 0.0,
            "intraday_pnl": 0.0,
            "total_today_pnl": 0.0,
            "total_overall_pnl": 0.0,
            "error": str(e)
        }, "/api/today_pnl_summary")


@router.get("/today_pnl")
async def api_today_pnl():
    """API endpoint to get today's total P&L from holdings using rt_intraday_price"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Get today's date in YYYY-MM-DD format
        cursor.execute("SELECT CURRENT_DATE as today_date")
        today = cursor.fetchone()['today_date']
        today_str = today.strftime('%Y-%m-%d')
        
        # Get previous trading day (assuming we can find it from rt_intraday_price)
        cursor.execute("""
            SELECT MAX(price_date) as prev_date
            FROM my_schema.rt_intraday_price
            WHERE price_date < %s
        """, (today_str,))
        
        prev_date_result = cursor.fetchone()
        if not prev_date_result or not prev_date_result['prev_date']:
            logging.warning("No previous trading day found")
            return {"total_pnl": 0.0, "message": "No previous trading day data available"}
        
        prev_date = prev_date_result['prev_date']
        
        # Calculate P&L for each holding: quantity × (today_price - prev_day_price)
        cursor.execute("""
            WITH holdings_data AS (
                SELECT 
                    h.trading_symbol,
                    h.quantity,
                    h.last_price as current_price
                FROM my_schema.holdings h
                WHERE h.run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
            ),
            today_prices AS (
                SELECT 
                    scrip_id,
                    price_close
                FROM my_schema.rt_intraday_price
                WHERE price_date = %s
            ),
            prev_prices AS (
                SELECT 
                    scrip_id,
                    price_close
                FROM my_schema.rt_intraday_price
                WHERE price_date = %s
            )
            SELECT 
                COALESCE(SUM(
                    h.quantity * (COALESCE(today_p.price_close, h.current_price) - COALESCE(prev_p.price_close, 0))
                ), 0) as total_pnl
            FROM holdings_data h
            LEFT JOIN today_prices today_p ON h.trading_symbol = today_p.scrip_id
            LEFT JOIN prev_prices prev_p ON h.trading_symbol = prev_p.scrip_id
        """, (today_str, prev_date))
        
        result = cursor.fetchone()
        total_pnl = float(result['total_pnl']) if result and result['total_pnl'] is not None else 0.0
        conn.close()
        
        return {
            "total_pnl": total_pnl,
            "today_date": today_str,
            "prev_date": prev_date
        }
    except Exception as e:
        logging.error(f"Error fetching today's P&L: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"error": str(e), "total_pnl": 0.0}


@router.get("/portfolio_history")
async def api_portfolio_history(days: int = Query(30, ge=1, le=365)):
    """API endpoint to get daily portfolio values for chart"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Get daily equity balances
        cursor.execute("""
            SELECT 
                run_date,
                COALESCE(SUM(quantity * last_price), 0) as equity_value
            FROM my_schema.holdings
            WHERE run_date >= CURRENT_DATE - make_interval(days => %s)
            GROUP BY run_date
            ORDER BY run_date
        """, (days,))
        
        equity_data = cursor.fetchall()
        
        # Get daily MF balances
        cursor.execute("""
            SELECT 
                run_date,
                COALESCE(SUM(current_value), 0) as mf_value
            FROM my_schema.mf_holdings
            WHERE run_date >= CURRENT_DATE - make_interval(days => %s)
            GROUP BY run_date
            ORDER BY run_date
        """, (days,))
        
        mf_data = cursor.fetchall()
        
        # Create a map of dates to values
        equity_map = {str(row['run_date']): float(row['equity_value']) for row in equity_data}
        mf_map = {str(row['run_date']): float(row['mf_value']) for row in mf_data}
        
        # Get all unique dates
        all_dates = sorted(set(list(equity_map.keys()) + list(mf_map.keys())))
        
        # Build chart data
        chart_data = []
        for date in all_dates:
            equity_val = equity_map.get(date, 0.0)
            mf_val = mf_map.get(date, 0.0)
            total_val = equity_val + mf_val
            chart_data.append({
                "date": date,
                "equity": equity_val,
                "mutual_fund": mf_val,
                "total": total_val
            })
        
        conn.close()
        
        return cached_json_response({"portfolio_history": chart_data}, "/api/portfolio_history")
    except Exception as e:
        logging.error(f"Error fetching portfolio history: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return cached_json_response({"error": str(e), "portfolio_history": []}, "/api/portfolio_history")


@router.get("/portfolio_hedge_analysis")
async def api_portfolio_hedge_analysis(
    target_hedge_ratio: float = Query(0.5, ge=0.0, le=1.0, description="Target hedge ratio (0.0 to 1.0)"),
    strategy_type: str = Query("all", description="Strategy type filter: all, puts, calls, collars, futures, or comma-separated")
):
    """API endpoint to get portfolio hedge analysis with multiple strategies"""
    try:
        from holdings.PortfolioHedgeAnalyzer import PortfolioHedgeAnalyzer
        
        db_config = {
            'host': os.getenv('PG_HOST', 'postgres'),
            'database': os.getenv('PG_DATABASE', 'mydb'),
            'user': os.getenv('PG_USER', 'postgres'),
            'password': os.getenv('PG_PASSWORD', 'postgres'),
            'port': int(os.getenv('PG_PORT', 5432))
        }
        
        analyzer = PortfolioHedgeAnalyzer(db_config)
        
        # Calculate portfolio beta
        beta_result = analyzer.calculate_portfolio_beta()
        
        # Get hedge suggestions with strategy filtering
        hedge_suggestions = analyzer.suggest_hedge(target_hedge_ratio=target_hedge_ratio, strategy_type=strategy_type)

        # Ensure diagnostics are always included, even if missing
        if 'diagnostics' not in hedge_suggestions:
            logging.warning("hedge_suggestions missing diagnostics, adding default diagnostics")
            # Get holdings for diagnostics if missing
            try:
                holdings = analyzer.get_current_holdings()
                equity_count = len(holdings[holdings['holding_type'] == 'EQUITY']) if not holdings.empty and 'holding_type' in holdings.columns else 0
                mf_count = len(holdings[holdings['holding_type'] == 'MF']) if not holdings.empty and 'holding_type' in holdings.columns and 'MF' in holdings['holding_type'].values else 0
                equity_value = float(holdings[holdings['holding_type'] == 'EQUITY']['current_value'].sum()) if not holdings.empty and 'holding_type' in holdings.columns and 'EQUITY' in holdings['holding_type'].values else 0.0
                mf_value = float(holdings[holdings['holding_type'] == 'MF']['current_value'].sum()) if not holdings.empty and 'holding_type' in holdings.columns and 'MF' in holdings['holding_type'].values else 0.0
            except Exception as e:
                logging.error(f"Failed to compute diagnostics fallback: {e}")
                equity_count = mf_count = 0
                equity_value = mf_value = 0.0

            hedge_suggestions['diagnostics'] = {
                'holdings_count': equity_count + mf_count,
                'equity_count': equity_count,
                'mf_count': mf_count,
                'portfolio_value': float(beta_result.get('portfolio_value', equity_value + mf_value)),
                'equity_value': float(equity_value),
                'mf_value': float(mf_value),
                'beta': float(beta_result.get('beta', 0.0)),
                'correlation': float(beta_result.get('correlation', 0.0)),
                'nifty_price': 0.0,
                'expiries_available': 0,
                'futures_strategies': 0,
                'puts_strategies': 0,
                'calls_strategies': 0,
                'collars_strategies': 0
            }
        
        # Calculate VaR
        var_result = analyzer.calculate_var()
        
        return {
            "success": True,
            "portfolio_metrics": beta_result,
            "hedge_suggestions": hedge_suggestions,
            "var_metrics": var_result,
            "target_hedge_ratio": target_hedge_ratio,
            "strategy_type": strategy_type
        }
    except Exception as e:
        logging.error(f"Error generating portfolio hedge analysis: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }


# ==================== Download Endpoints ====================

@router.get("/download/holdings/excel")
async def download_holdings_excel():
    """Download holdings data as Excel file"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Get holdings with calculated fields
        cursor.execute("""
            SELECT 
                h.trading_symbol,
                h.instrument_token,
                h.quantity,
                h.average_price,
                h.last_price,
                (h.quantity * h.average_price) as invested_amount,
                (h.quantity * COALESCE(h.last_price, rt.price_close, 0)) as current_amount,
                h.pnl,
                round(case when (h.quantity * h.average_price) != 0 then
                    (h.pnl / (h.quantity * h.average_price)) * 100
                else 0
                end::numeric, 2) as pnl_pct_change
            FROM my_schema.holdings h
            LEFT JOIN (
                SELECT DISTINCT ON (scrip_id) scrip_id, price_close, price_date
                FROM my_schema.rt_intraday_price
                WHERE price_date::date <= CURRENT_DATE
                ORDER BY scrip_id, price_date DESC
            ) rt ON h.trading_symbol = rt.scrip_id
            WHERE h.run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
            ORDER BY h.trading_symbol
        """)
        
        holdings = cursor.fetchall()
        
        # Get today's P&L data
        cursor.execute("SELECT CURRENT_DATE as today_date")
        today = cursor.fetchone()['today_date']
        today_str = today.strftime('%Y-%m-%d')
        
        cursor.execute("""
            SELECT MAX(price_date) as prev_date
            FROM my_schema.rt_intraday_price
            WHERE price_date < %s
        """, (today_str,))
        
        prev_date_result = cursor.fetchone()
        prev_date = prev_date_result['prev_date'] if prev_date_result and prev_date_result['prev_date'] else None
        
        # Create today's P&L map
        today_pnl_map = {}
        for row in holdings:
            instrument_token = row.get('instrument_token')
            if instrument_token:
                today_pnl_map[instrument_token] = {
                    'today_pnl': 0.0,
                    'pct_change': 0.0
                }
        
        if prev_date:
            prev_date_str = prev_date.strftime('%Y-%m-%d') if hasattr(prev_date, 'strftime') else str(prev_date)
            cursor.execute("""
                WITH holdings_today AS (
                    SELECT h.instrument_token, h.trading_symbol, h.quantity, h.last_price
                    FROM my_schema.holdings h
                    WHERE h.run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
                ),
                today_prices AS (
                    SELECT scrip_id, price_close
                    FROM my_schema.rt_intraday_price
                    WHERE price_date = %s
                ),
                latest_prices AS (
                    SELECT scrip_id, price_close, 
                           ROW_NUMBER() OVER (PARTITION BY scrip_id ORDER BY price_date DESC) as rn
                    FROM my_schema.rt_intraday_price
                    WHERE price_date::date <= CURRENT_DATE
                ),
                prev_prices AS (
                    SELECT scrip_id, price_close
                    FROM my_schema.rt_intraday_price
                    WHERE price_date = %s
                )
                SELECT 
                    h.instrument_token,
                    COALESCE(h.last_price, today_p.price_close, latest_p.price_close, 0) as today_price,
                    COALESCE(prev_p.price_close, 0) as prev_price,
                    CASE 
                        WHEN (COALESCE(h.last_price, today_p.price_close, latest_p.price_close) IS NULL OR COALESCE(h.last_price, today_p.price_close, latest_p.price_close) = 0) THEN 0
                        WHEN (COALESCE(prev_p.price_close, 0) = 0) THEN 0
                        ELSE h.quantity * (COALESCE(h.last_price, today_p.price_close, latest_p.price_close, 0) - COALESCE(prev_p.price_close, 0))
                    END as today_pnl,
                    round(case when prev_p.price_close != 0 AND COALESCE(h.last_price, today_p.price_close, latest_p.price_close) IS NOT NULL AND COALESCE(h.last_price, today_p.price_close, latest_p.price_close) != 0 then 
                        ((COALESCE(h.last_price, today_p.price_close, latest_p.price_close, 0) - COALESCE(prev_p.price_close, 0))/COALESCE(prev_p.price_close, 0)) * 100
                    else 0
                    end::numeric, 2) as pct_change
                FROM holdings_today h
                LEFT JOIN today_prices today_p ON h.trading_symbol = today_p.scrip_id
                LEFT JOIN latest_prices latest_p ON h.trading_symbol = latest_p.scrip_id AND latest_p.rn = 1
                LEFT JOIN prev_prices prev_p ON h.trading_symbol = prev_p.scrip_id
            """, (today_str, prev_date_str))
            
            for row in cursor.fetchall():
                instrument_token = row['instrument_token']
                today_pnl_map[instrument_token] = {
                    'today_pnl': float(row['today_pnl']) if row['today_pnl'] else 0.0,
                    'pct_change': float(row['pct_change']) if row['pct_change'] else 0.0
                }
        
        # Get Prophet predictions (60-day preferred, fallback to latest)
        cursor.execute("""
            SELECT DISTINCT ON (scrip_id) scrip_id, predicted_price_change_pct, prediction_confidence, prediction_days
            FROM my_schema.prophet_predictions
            WHERE status = 'ACTIVE'
            ORDER BY scrip_id, 
                     CASE WHEN prediction_days = 60 THEN 1 ELSE 2 END,
                     run_date DESC
        """)
        
        prophet_map = {}
        for row in cursor.fetchall():
            scrip_id = row['scrip_id'].upper()
            prophet_map[scrip_id] = {
                'prediction_pct': float(row['predicted_price_change_pct']) if row['predicted_price_change_pct'] is not None else None,
                'confidence': float(row['prediction_confidence']) if row['prediction_confidence'] is not None else None,
                'prediction_days': int(row['prediction_days']) if row['prediction_days'] is not None else None
            }
        
        conn.close()
        
        # Enrich holdings data
        holdings_list = []
        for row in holdings:
            row_dict = dict(row)
            symbol = row_dict['trading_symbol']
            instrument_token = row_dict.get('instrument_token')
            
            # Add today's P&L
            today_pnl_info = today_pnl_map.get(instrument_token, {'today_pnl': 0.0, 'pct_change': 0.0})
            row_dict['today_pnl'] = today_pnl_info['today_pnl']
            row_dict['today_pnl_pct'] = today_pnl_info['pct_change']
            
            # Add Prophet predictions
            prophet_info = prophet_map.get(symbol.upper() if symbol else '', {})
            row_dict['ghost_prediction_pct'] = prophet_info.get('prediction_pct')
            row_dict['confidence'] = prophet_info.get('confidence')
            row_dict['prediction_days'] = prophet_info.get('prediction_days')
            
            holdings_list.append(row_dict)
        
        # Create DataFrame with proper column order
        df = pd.DataFrame(holdings_list)
        
        # Reorder columns for better readability
        column_order = [
            'trading_symbol', 'quantity', 'average_price', 'last_price',
            'invested_amount', 'current_amount', 'pnl', 'pnl_pct_change',
            'today_pnl', 'today_pnl_pct',
            'ghost_prediction_pct', 'confidence', 'prediction_days'
        ]
        
        # Only include columns that exist
        available_columns = [col for col in column_order if col in df.columns]
        df = df[available_columns]
        
        # Rename columns for readability
        df = df.rename(columns={
            'trading_symbol': 'Symbol',
            'quantity': 'Qty',
            'average_price': 'Avg Price',
            'last_price': 'LTP',
            'invested_amount': 'Invested Amount',
            'current_amount': 'Current Amount',
            'pnl': 'Total P&L',
            'pnl_pct_change': 'Total P&L %',
            'today_pnl': "Today's P&L",
            'today_pnl_pct': "Today's P&L %",
            'ghost_prediction_pct': 'Ghost Prediction %',
            'confidence': 'Confidence %',
            'prediction_days': 'Prediction Days'
        })
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Holdings', index=False)
        
        output.seek(0)
        
        return StreamingResponse(
            io.BytesIO(output.read()),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=holdings_{datetime.now().strftime('%Y%m%d')}.xlsx"}
        )
    except Exception as e:
        logging.error(f"Error generating Excel file: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"error": str(e)}


@router.get("/download/holdings/csv")
async def download_holdings_csv():
    """Download holdings data as CSV file - uses same logic as Excel"""
    try:
        # Use the same data fetching logic as Excel
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Get holdings with calculated fields
        cursor.execute("""
            SELECT 
                h.trading_symbol,
                h.instrument_token,
                h.quantity,
                h.average_price,
                h.last_price,
                (h.quantity * h.average_price) as invested_amount,
                (h.quantity * COALESCE(h.last_price, rt.price_close, 0)) as current_amount,
                h.pnl,
                round(case when (h.quantity * h.average_price) != 0 then
                    (h.pnl / (h.quantity * h.average_price)) * 100
                else 0
                end::numeric, 2) as pnl_pct_change
            FROM my_schema.holdings h
            LEFT JOIN (
                SELECT DISTINCT ON (scrip_id) scrip_id, price_close, price_date
                FROM my_schema.rt_intraday_price
                WHERE price_date::date <= CURRENT_DATE
                ORDER BY scrip_id, price_date DESC
            ) rt ON h.trading_symbol = rt.scrip_id
            WHERE h.run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
            ORDER BY h.trading_symbol
        """)
        
        holdings = cursor.fetchall()
        
        # Get today's P&L data
        cursor.execute("SELECT CURRENT_DATE as today_date")
        today = cursor.fetchone()['today_date']
        today_str = today.strftime('%Y-%m-%d')
        
        cursor.execute("""
            SELECT MAX(price_date) as prev_date
            FROM my_schema.rt_intraday_price
            WHERE price_date < %s
        """, (today_str,))
        
        prev_date_result = cursor.fetchone()
        prev_date = prev_date_result['prev_date'] if prev_date_result and prev_date_result['prev_date'] else None
        
        # Create today's P&L map
        today_pnl_map = {}
        for row in holdings:
            instrument_token = row.get('instrument_token')
            if instrument_token:
                today_pnl_map[instrument_token] = {
                    'today_pnl': 0.0,
                    'pct_change': 0.0
                }
        
        if prev_date:
            prev_date_str = prev_date.strftime('%Y-%m-%d') if hasattr(prev_date, 'strftime') else str(prev_date)
            cursor.execute("""
                WITH holdings_today AS (
                    SELECT h.instrument_token, h.trading_symbol, h.quantity, h.last_price
                    FROM my_schema.holdings h
                    WHERE h.run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
                ),
                today_prices AS (
                    SELECT scrip_id, price_close
                    FROM my_schema.rt_intraday_price
                    WHERE price_date = %s
                ),
                latest_prices AS (
                    SELECT scrip_id, price_close, 
                           ROW_NUMBER() OVER (PARTITION BY scrip_id ORDER BY price_date DESC) as rn
                    FROM my_schema.rt_intraday_price
                    WHERE price_date::date <= CURRENT_DATE
                ),
                prev_prices AS (
                    SELECT scrip_id, price_close
                    FROM my_schema.rt_intraday_price
                    WHERE price_date = %s
                )
                SELECT 
                    h.instrument_token,
                    COALESCE(h.last_price, today_p.price_close, latest_p.price_close, 0) as today_price,
                    COALESCE(prev_p.price_close, 0) as prev_price,
                    CASE 
                        WHEN (COALESCE(h.last_price, today_p.price_close, latest_p.price_close) IS NULL OR COALESCE(h.last_price, today_p.price_close, latest_p.price_close) = 0) THEN 0
                        WHEN (COALESCE(prev_p.price_close, 0) = 0) THEN 0
                        ELSE h.quantity * (COALESCE(h.last_price, today_p.price_close, latest_p.price_close, 0) - COALESCE(prev_p.price_close, 0))
                    END as today_pnl,
                    round(case when prev_p.price_close != 0 AND COALESCE(h.last_price, today_p.price_close, latest_p.price_close) IS NOT NULL AND COALESCE(h.last_price, today_p.price_close, latest_p.price_close) != 0 then 
                        ((COALESCE(h.last_price, today_p.price_close, latest_p.price_close, 0) - COALESCE(prev_p.price_close, 0))/COALESCE(prev_p.price_close, 0)) * 100
                    else 0
                    end::numeric, 2) as pct_change
                FROM holdings_today h
                LEFT JOIN today_prices today_p ON h.trading_symbol = today_p.scrip_id
                LEFT JOIN latest_prices latest_p ON h.trading_symbol = latest_p.scrip_id AND latest_p.rn = 1
                LEFT JOIN prev_prices prev_p ON h.trading_symbol = prev_p.scrip_id
            """, (today_str, prev_date_str))
            
            for row in cursor.fetchall():
                instrument_token = row['instrument_token']
                today_pnl_map[instrument_token] = {
                    'today_pnl': float(row['today_pnl']) if row['today_pnl'] else 0.0,
                    'pct_change': float(row['pct_change']) if row['pct_change'] else 0.0
                }
        
        # Get Prophet predictions (60-day preferred, fallback to latest)
        cursor.execute("""
            SELECT DISTINCT ON (scrip_id) scrip_id, predicted_price_change_pct, prediction_confidence, prediction_days
            FROM my_schema.prophet_predictions
            WHERE status = 'ACTIVE'
            ORDER BY scrip_id, 
                     CASE WHEN prediction_days = 60 THEN 1 ELSE 2 END,
                     run_date DESC
        """)
        
        prophet_map = {}
        for row in cursor.fetchall():
            scrip_id = row['scrip_id'].upper()
            prophet_map[scrip_id] = {
                'prediction_pct': float(row['predicted_price_change_pct']) if row['predicted_price_change_pct'] is not None else None,
                'confidence': float(row['prediction_confidence']) if row['prediction_confidence'] is not None else None,
                'prediction_days': int(row['prediction_days']) if row['prediction_days'] is not None else None
            }
        
        conn.close()
        
        # Enrich holdings data
        holdings_list = []
        for row in holdings:
            row_dict = dict(row)
            symbol = row_dict['trading_symbol']
            instrument_token = row_dict.get('instrument_token')
            
            # Add today's P&L
            today_pnl_info = today_pnl_map.get(instrument_token, {'today_pnl': 0.0, 'pct_change': 0.0})
            row_dict['today_pnl'] = today_pnl_info['today_pnl']
            row_dict['today_pnl_pct'] = today_pnl_info['pct_change']
            
            # Add Prophet predictions
            prophet_info = prophet_map.get(symbol.upper() if symbol else '', {})
            row_dict['ghost_prediction_pct'] = prophet_info.get('prediction_pct')
            row_dict['confidence'] = prophet_info.get('confidence')
            row_dict['prediction_days'] = prophet_info.get('prediction_days')
            
            holdings_list.append(row_dict)
        
        # Create DataFrame with proper column order
        df = pd.DataFrame(holdings_list)
        
        # Reorder columns for better readability
        column_order = [
            'trading_symbol', 'quantity', 'average_price', 'last_price',
            'invested_amount', 'current_amount', 'pnl', 'pnl_pct_change',
            'today_pnl', 'today_pnl_pct',
            'ghost_prediction_pct', 'confidence', 'prediction_days'
        ]
        
        # Only include columns that exist
        available_columns = [col for col in column_order if col in df.columns]
        df = df[available_columns]
        
        # Rename columns for readability
        df = df.rename(columns={
            'trading_symbol': 'Symbol',
            'quantity': 'Qty',
            'average_price': 'Avg Price',
            'last_price': 'LTP',
            'invested_amount': 'Invested Amount',
            'current_amount': 'Current Amount',
            'pnl': 'Total P&L',
            'pnl_pct_change': 'Total P&L %',
            'today_pnl': "Today's P&L",
            'today_pnl_pct': "Today's P&L %",
            'ghost_prediction_pct': 'Ghost Prediction %',
            'confidence': 'Confidence %',
            'prediction_days': 'Prediction Days'
        })
        
        output = StringIO()
        df.to_csv(output, index=False)
        
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode('utf-8')),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=holdings_{datetime.now().strftime('%Y%m%d')}.csv"}
        )
    except Exception as e:
        logging.error(f"Error generating CSV file: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"error": str(e)}


@router.get("/download/holdings/pdf")
async def download_holdings_pdf():
    """Download holdings data as PDF file"""
    if not REPORTLAB_AVAILABLE:
        return {"error": "reportlab library is not installed. Please install it to generate PDF files."}
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Get holdings with calculated fields (same logic as Excel/CSV)
        cursor.execute("""
            SELECT 
                h.trading_symbol,
                h.instrument_token,
                h.quantity,
                h.average_price,
                h.last_price,
                (h.quantity * h.average_price) as invested_amount,
                (h.quantity * COALESCE(h.last_price, rt.price_close, 0)) as current_amount,
                h.pnl,
                round(case when (h.quantity * h.average_price) != 0 then
                    (h.pnl / (h.quantity * h.average_price)) * 100
                else 0
                end::numeric, 2) as pnl_pct_change
            FROM my_schema.holdings h
            LEFT JOIN (
                SELECT DISTINCT ON (scrip_id) scrip_id, price_close, price_date
                FROM my_schema.rt_intraday_price
                WHERE price_date::date <= CURRENT_DATE
                ORDER BY scrip_id, price_date DESC
            ) rt ON h.trading_symbol = rt.scrip_id
            WHERE h.run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
            ORDER BY h.trading_symbol
        """)
        
        holdings = cursor.fetchall()
        
        # Get today's P&L data (same logic as Excel/CSV)
        cursor.execute("SELECT CURRENT_DATE as today_date")
        today = cursor.fetchone()['today_date']
        today_str = today.strftime('%Y-%m-%d')
        
        cursor.execute("""
            SELECT MAX(price_date) as prev_date
            FROM my_schema.rt_intraday_price
            WHERE price_date < %s
        """, (today_str,))
        
        prev_date_result = cursor.fetchone()
        prev_date = prev_date_result['prev_date'] if prev_date_result and prev_date_result['prev_date'] else None
        
        today_pnl_map = {}
        for row in holdings:
            instrument_token = row.get('instrument_token')
            if instrument_token:
                today_pnl_map[instrument_token] = {'today_pnl': 0.0, 'pct_change': 0.0}
        
        if prev_date:
            prev_date_str = prev_date.strftime('%Y-%m-%d') if hasattr(prev_date, 'strftime') else str(prev_date)
            cursor.execute("""
                WITH holdings_today AS (
                    SELECT h.instrument_token, h.trading_symbol, h.quantity, h.last_price
                    FROM my_schema.holdings h
                    WHERE h.run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
                ),
                today_prices AS (
                    SELECT scrip_id, price_close
                    FROM my_schema.rt_intraday_price
                    WHERE price_date = %s
                ),
                latest_prices AS (
                    SELECT scrip_id, price_close, 
                           ROW_NUMBER() OVER (PARTITION BY scrip_id ORDER BY price_date DESC) as rn
                    FROM my_schema.rt_intraday_price
                    WHERE price_date::date <= CURRENT_DATE
                ),
                prev_prices AS (
                    SELECT scrip_id, price_close
                    FROM my_schema.rt_intraday_price
                    WHERE price_date = %s
                )
                SELECT 
                    h.instrument_token,
                    COALESCE(h.last_price, today_p.price_close, latest_p.price_close, 0) as today_price,
                    COALESCE(prev_p.price_close, 0) as prev_price,
                    CASE 
                        WHEN (COALESCE(h.last_price, today_p.price_close, latest_p.price_close) IS NULL OR COALESCE(h.last_price, today_p.price_close, latest_p.price_close) = 0) THEN 0
                        WHEN (COALESCE(prev_p.price_close, 0) = 0) THEN 0
                        ELSE h.quantity * (COALESCE(h.last_price, today_p.price_close, latest_p.price_close, 0) - COALESCE(prev_p.price_close, 0))
                    END as today_pnl,
                    round(case when prev_p.price_close != 0 AND COALESCE(h.last_price, today_p.price_close, latest_p.price_close) IS NOT NULL AND COALESCE(h.last_price, today_p.price_close, latest_p.price_close) != 0 then 
                        ((COALESCE(h.last_price, today_p.price_close, latest_p.price_close, 0) - COALESCE(prev_p.price_close, 0))/COALESCE(prev_p.price_close, 0)) * 100
                    else 0
                    end::numeric, 2) as pct_change
                FROM holdings_today h
                LEFT JOIN today_prices today_p ON h.trading_symbol = today_p.scrip_id
                LEFT JOIN latest_prices latest_p ON h.trading_symbol = latest_p.scrip_id AND latest_p.rn = 1
                LEFT JOIN prev_prices prev_p ON h.trading_symbol = prev_p.scrip_id
            """, (today_str, prev_date_str))
            
            for row in cursor.fetchall():
                instrument_token = row['instrument_token']
                today_pnl_map[instrument_token] = {
                    'today_pnl': float(row['today_pnl']) if row['today_pnl'] else 0.0,
                    'pct_change': float(row['pct_change']) if row['pct_change'] else 0.0
                }
        
        # Get Prophet predictions
        cursor.execute("""
            SELECT DISTINCT ON (scrip_id) scrip_id, predicted_price_change_pct, prediction_confidence, prediction_days
            FROM my_schema.prophet_predictions
            WHERE status = 'ACTIVE'
            ORDER BY scrip_id, 
                     CASE WHEN prediction_days = 60 THEN 1 ELSE 2 END,
                     run_date DESC
        """)
        
        prophet_map = {}
        for row in cursor.fetchall():
            scrip_id = row['scrip_id'].upper()
            prophet_map[scrip_id] = {
                'prediction_pct': float(row['predicted_price_change_pct']) if row['predicted_price_change_pct'] is not None else None,
                'confidence': float(row['prediction_confidence']) if row['prediction_confidence'] is not None else None,
                'prediction_days': int(row['prediction_days']) if row['prediction_days'] is not None else None
            }
        
        conn.close()
        
        # Enrich holdings data
        holdings_list = []
        for row in holdings:
            row_dict = dict(row)
            symbol = row_dict['trading_symbol']
            instrument_token = row_dict.get('instrument_token')
            
            today_pnl_info = today_pnl_map.get(instrument_token, {'today_pnl': 0.0, 'pct_change': 0.0})
            row_dict['today_pnl'] = today_pnl_info['today_pnl']
            row_dict['today_pnl_pct'] = today_pnl_info['pct_change']
            
            prophet_info = prophet_map.get(symbol.upper() if symbol else '', {})
            row_dict['ghost_prediction_pct'] = prophet_info.get('prediction_pct')
            row_dict['confidence'] = prophet_info.get('confidence')
            row_dict['prediction_days'] = prophet_info.get('prediction_days')
            
            holdings_list.append(row_dict)
        
        # Create PDF in memory
        output = io.BytesIO()
        doc = SimpleDocTemplate(output, pagesize=letter, topMargin=0.5*inch)
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=16, textColor=colors.HexColor('#1a1a1a'), spaceAfter=12)
        
        # Calculate totals
        total_invested = sum(row['invested_amount'] for row in holdings_list)
        total_current = sum(row['current_amount'] for row in holdings_list)
        total_pnl = sum(row['pnl'] for row in holdings_list)
        total_today_pnl = sum(row['today_pnl'] for row in holdings_list)
        
        elements = []
        elements.append(Paragraph("Equity Holdings Report", title_style))
        elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Add summary box
        summary_data = [
            ['Total Invested:', f'Rs {total_invested:,.2f}'],
            ['Current Value:', f'Rs {total_current:,.2f}'],
            ['Total P&L:', f'Rs {total_pnl:,.2f}'],
            ["Today's P&L:", f'Rs {total_today_pnl:,.2f}']
        ]
        summary_table = Table(summary_data, colWidths=[2*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.grey),
            ('BACKGROUND', (1, 0), (1, -1), colors.beige),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(summary_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Create table data
        data = [['Symbol', 'Qty', 'Avg Price', 'LTP', 'Invested', 'Current', 'P&L', 'P&L %', "Today's P&L", "Today's P&L %", 'Ghost Pred %', 'Conf %', 'Pred Days']]
        for row in holdings_list:
            row_dict = dict(row)
            pnl_pct = row_dict.get('pnl_pct_change', 0) or 0
            today_pnl = row_dict.get('today_pnl', 0) or 0
            today_pnl_pct = row_dict.get('today_pnl_pct', 0) or 0
            ghost_pred = row_dict.get('ghost_prediction_pct')
            confidence = row_dict.get('confidence')
            pred_days = row_dict.get('prediction_days')
            
            data.append([
                row_dict['trading_symbol'],
                str(row_dict['quantity']),
                f"Rs {row_dict['average_price']:.2f}",
                f"Rs {row_dict['last_price']:.2f}",
                f"Rs {row_dict['invested_amount']:.2f}",
                f"Rs {row_dict['current_amount']:.2f}",
                f"Rs {row_dict['pnl']:.2f}",
                f"{pnl_pct:.2f}%" if pnl_pct else "0.00%",
                f"Rs {today_pnl:.2f}",
                f"{today_pnl_pct:.2f}%" if today_pnl_pct else "0.00%",
                f"{ghost_pred:.2f}%" if ghost_pred is not None else "N/A",
                f"{confidence:.0f}%" if confidence is not None else "N/A",
                str(pred_days) if pred_days is not None else "N/A"
            ])
        
        # Add totals row
        data.append([
            'TOTAL', '', '', '',
            f'Rs {total_invested:,.2f}',
            f'Rs {total_current:,.2f}',
            f'Rs {total_pnl:,.2f}',
            '',
            f'Rs {total_today_pnl:,.2f}',
            '', '', '', ''
        ])
        
        col_widths = [0.8*inch, 0.5*inch, 0.7*inch, 0.7*inch, 0.9*inch, 0.9*inch, 0.8*inch, 0.6*inch, 0.8*inch, 0.7*inch, 0.7*inch, 0.5*inch, 0.5*inch]
        table = Table(data, repeatRows=1, colWidths=col_widths)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -2), colors.beige),
            ('BACKGROUND', (0, -1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -2), 7),
            ('FONTSIZE', (0, -1), (-1, -1), 9),
        ]))
        
        elements.append(table)
        doc.build(elements)
        
        output.seek(0)
        return StreamingResponse(
            io.BytesIO(output.read()),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=holdings_{datetime.now().strftime('%Y%m%d')}.pdf"}
        )
    except Exception as e:
        logging.error(f"Error generating PDF: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"error": str(e)}


@router.get("/download/mf/excel")
async def download_mf_excel():
    """Download MF holdings data as Excel file"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cursor.execute("""
            SELECT 
                folio, fund, tradingsymbol, quantity, average_price, last_price,
                invested_amount, current_value, pnl,
                net_change_percentage, day_change_percentage
            FROM my_schema.mf_holdings 
            WHERE run_date = (SELECT MAX(run_date) FROM my_schema.mf_holdings)
            ORDER BY fund, tradingsymbol
        """)
        
        mf_holdings = cursor.fetchall()
        conn.close()
        
        df = pd.DataFrame([dict(row) for row in mf_holdings])
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='MF Holdings', index=False)
        
        output.seek(0)
        
        return StreamingResponse(
            io.BytesIO(output.read()),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=mf_holdings_{datetime.now().strftime('%Y%m%d')}.xlsx"}
        )
    except Exception as e:
        logging.error(f"Error generating Excel file: {e}")
        return {"error": str(e)}


@router.get("/download/mf/csv")
async def download_mf_csv():
    """Download MF holdings data as CSV file"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cursor.execute("""
            SELECT 
                folio, fund, tradingsymbol, quantity, average_price, last_price,
                invested_amount, current_value, pnl,
                net_change_percentage, day_change_percentage
            FROM my_schema.mf_holdings 
            WHERE run_date = (SELECT MAX(run_date) FROM my_schema.mf_holdings)
            ORDER BY fund, tradingsymbol
        """)
        
        mf_holdings = cursor.fetchall()
        conn.close()
        
        df = pd.DataFrame([dict(row) for row in mf_holdings])
        output = StringIO()
        df.to_csv(output, index=False)
        
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode('utf-8')),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=mf_holdings_{datetime.now().strftime('%Y%m%d')}.csv"}
        )
    except Exception as e:
        logging.error(f"Error generating CSV file: {e}")
        return {"error": str(e)}


@router.get("/download/mf/pdf")
async def download_mf_pdf():
    """Download MF holdings data as PDF file"""
    if not REPORTLAB_AVAILABLE:
        return {"error": "reportlab library is not installed. Please install it to generate PDF files."}
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cursor.execute("""
            SELECT folio, fund, tradingsymbol, invested_amount, current_value, pnl, net_change_percentage, day_change_percentage
            FROM my_schema.mf_holdings WHERE run_date = (SELECT MAX(run_date) FROM my_schema.mf_holdings) ORDER BY fund, tradingsymbol
        """)
        
        mf_holdings = cursor.fetchall()
        conn.close()
        
        output = io.BytesIO()
        doc = SimpleDocTemplate(output, pagesize=letter, topMargin=0.5*inch)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=16, textColor=colors.HexColor('#1a1a1a'), spaceAfter=12)
        
        elements = []
        elements.append(Paragraph("Mutual Fund Holdings Report", title_style))
        elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        data = [['Fund', 'Symbol', 'Invested', 'Current', 'P&L', 'Net %', 'Day %']]
        for row in mf_holdings:
            row_dict = dict(row)
            data.append([
                row_dict['fund'] or row_dict['tradingsymbol'],
                row_dict['tradingsymbol'],
                f"Rs {row_dict['invested_amount']:.2f}",
                f"Rs {row_dict['current_value']:.2f}",
                f"Rs {row_dict['pnl']:.2f}",
                f"{row_dict['net_change_percentage']:.2f}%",
                f"{row_dict['day_change_percentage']:.2f}%"
            ])
        
        table = Table(data, repeatRows=1, colWidths=[2*inch, 1.2*inch, 1.1*inch, 1.1*inch, 0.9*inch, 1*inch, 1*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 7),
        ]))
        
        elements.append(table)
        doc.build(elements)
        output.seek(0)
        
        return StreamingResponse(
            io.BytesIO(output.read()),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=mf_holdings_{datetime.now().strftime('%Y%m%d')}.pdf"}
        )
    except Exception as e:
        logging.error(f"Error generating PDF: {e}")
        return {"error": str(e)}


@router.get("/download/pnl_summary/excel")
async def download_pnl_summary_excel():
    """Download complete P&L summary with equity and MF holdings as Excel file"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cursor.execute("SELECT CURRENT_DATE as today_date")
        today = cursor.fetchone()['today_date']
        today_str = today.strftime('%Y-%m-%d')
        
        cursor.execute("""
            SELECT MAX(price_date) as prev_date FROM my_schema.rt_intraday_price WHERE price_date < %s
        """, (today_str,))
        prev_date_result = cursor.fetchone()
        prev_date = prev_date_result['prev_date'] if prev_date_result and prev_date_result['prev_date'] else None
        
        equity_today, equity_overall = 0.0, 0.0
        mf_today, mf_overall = 0.0, 0.0
        intraday = 0.0
        
        if prev_date:
            prev_date_str = prev_date.strftime('%Y-%m-%d') if hasattr(prev_date, 'strftime') else str(prev_date)
            cursor.execute("""
                WITH holdings_today AS (
                    SELECT trading_symbol, quantity FROM my_schema.holdings
                    WHERE run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
                ),
                today_prices AS (SELECT scrip_id, price_close FROM my_schema.rt_intraday_price WHERE price_date = %s),
                prev_prices AS (SELECT scrip_id, price_close FROM my_schema.rt_intraday_price WHERE price_date = %s)
                SELECT COALESCE(SUM(h.quantity * (COALESCE(today_p.price_close, 0) - COALESCE(prev_p.price_close, 0))), 0) as equity_today_pnl
                FROM holdings_today h
                LEFT JOIN today_prices today_p ON h.trading_symbol = today_p.scrip_id
                LEFT JOIN prev_prices prev_p ON h.trading_symbol = prev_p.scrip_id
            """, (today_str, prev_date_str))
            result = cursor.fetchone()
            equity_today = float(result['equity_today_pnl']) if result else 0.0
        
        cursor.execute("SELECT COALESCE(SUM(pnl), 0) as equity_overall_pnl FROM my_schema.holdings WHERE run_date = (SELECT MAX(run_date) FROM my_schema.holdings)")
        result = cursor.fetchone()
        equity_overall = float(result['equity_overall_pnl']) if result else 0.0
        
        cursor.execute("SELECT COALESCE(SUM(day_change_percentage * invested_amount / 100), 0) as mf_today_pnl FROM my_schema.mf_holdings WHERE run_date = (SELECT MAX(run_date) FROM my_schema.mf_holdings)")
        result = cursor.fetchone()
        mf_today = float(result['mf_today_pnl']) if result else 0.0
        
        cursor.execute("SELECT COALESCE(SUM(pnl), 0) as mf_overall_pnl FROM my_schema.mf_holdings WHERE run_date = (SELECT MAX(run_date) FROM my_schema.mf_holdings)")
        result = cursor.fetchone()
        mf_overall = float(result['mf_overall_pnl']) if result else 0.0
        
        cursor.execute("SELECT COALESCE(SUM(pnl), 0) as intraday_pnl FROM my_schema.positions WHERE run_date = (SELECT MAX(run_date) FROM my_schema.positions) AND position_type = 'day'")
        result = cursor.fetchone()
        intraday = float(result['intraday_pnl']) if result else 0.0
        
        summary_data = [
            {'Category': 'Equity Portfolio', "Today's P&L": equity_today, 'Overall P&L': equity_overall},
            {'Category': 'Mutual Fund Portfolio', "Today's P&L": mf_today, 'Overall P&L': mf_overall},
            {'Category': 'Intraday Trades', "Today's P&L": intraday, 'Overall P&L': '-'},
            {'Category': 'Total', "Today's P&L": equity_today + mf_today + intraday, 'Overall P&L': equity_overall + mf_overall}
        ]
        
        cursor.execute("""
            SELECT trading_symbol, quantity, average_price, last_price,
                   (quantity * average_price) as invested_amount,
                   (quantity * last_price) as current_amount, pnl
            FROM my_schema.holdings WHERE run_date = (SELECT MAX(run_date) FROM my_schema.holdings) ORDER BY trading_symbol
        """)
        equity_holdings = cursor.fetchall()
        
        cursor.execute("""
            SELECT folio, fund, tradingsymbol, quantity, average_price, last_price,
                   invested_amount, current_value, pnl, net_change_percentage, day_change_percentage
            FROM my_schema.mf_holdings WHERE run_date = (SELECT MAX(run_date) FROM my_schema.mf_holdings) ORDER BY fund, tradingsymbol
        """)
        mf_holdings = cursor.fetchall()
        conn.close()
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='P&L Summary', index=False)
            pd.DataFrame([dict(row) for row in equity_holdings]).to_excel(writer, sheet_name='Equity Holdings', index=False)
            pd.DataFrame([dict(row) for row in mf_holdings]).to_excel(writer, sheet_name='MF Holdings', index=False)
        
        output.seek(0)
        return StreamingResponse(
            io.BytesIO(output.read()),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=pnl_summary_{datetime.now().strftime('%Y%m%d')}.xlsx"}
        )
    except Exception as e:
        logging.error(f"Error generating Excel file: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"error": str(e)}


@router.get("/download/pnl_summary/pdf")
async def download_pnl_summary_pdf():
    """Download complete P&L summary with equity and MF holdings as PDF file"""
    if not REPORTLAB_AVAILABLE or not MATPLOTLIB_AVAILABLE:
        return {"error": "reportlab and/or matplotlib libraries are not installed. Please install them to generate PDF files."}
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cursor.execute("SELECT CURRENT_DATE as today_date")
        today = cursor.fetchone()['today_date']
        today_str = today.strftime('%Y-%m-%d')
        
        cursor.execute("""
            SELECT MAX(price_date) as prev_date FROM my_schema.rt_intraday_price WHERE price_date < %s
        """, (today_str,))
        prev_date_result = cursor.fetchone()
        prev_date = prev_date_result['prev_date'] if prev_date_result and prev_date_result['prev_date'] else None
        
        equity_today, equity_overall = 0.0, 0.0
        mf_today, mf_overall = 0.0, 0.0
        intraday = 0.0
        
        if prev_date:
            prev_date_str = prev_date.strftime('%Y-%m-%d') if hasattr(prev_date, 'strftime') else str(prev_date)
            cursor.execute("""
                WITH holdings_today AS (
                    SELECT trading_symbol, quantity FROM my_schema.holdings
                    WHERE run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
                ),
                today_prices AS (SELECT scrip_id, price_close FROM my_schema.rt_intraday_price WHERE price_date = %s),
                prev_prices AS (SELECT scrip_id, price_close FROM my_schema.rt_intraday_price WHERE price_date = %s)
                SELECT COALESCE(SUM(h.quantity * (COALESCE(today_p.price_close, 0) - COALESCE(prev_p.price_close, 0))), 0) as equity_today_pnl
                FROM holdings_today h
                LEFT JOIN today_prices today_p ON h.trading_symbol = today_p.scrip_id
                LEFT JOIN prev_prices prev_p ON h.trading_symbol = prev_p.scrip_id
            """, (today_str, prev_date_str))
            result = cursor.fetchone()
            equity_today = float(result['equity_today_pnl']) if result else 0.0
        
        cursor.execute("SELECT COALESCE(SUM(pnl), 0) as equity_overall_pnl FROM my_schema.holdings WHERE run_date = (SELECT MAX(run_date) FROM my_schema.holdings)")
        result = cursor.fetchone()
        equity_overall = float(result['equity_overall_pnl']) if result else 0.0
        
        cursor.execute("SELECT COALESCE(SUM(day_change_percentage * invested_amount / 100), 0) as mf_today_pnl FROM my_schema.mf_holdings WHERE run_date = (SELECT MAX(run_date) FROM my_schema.mf_holdings)")
        result = cursor.fetchone()
        mf_today = float(result['mf_today_pnl']) if result else 0.0
        
        cursor.execute("SELECT COALESCE(SUM(pnl), 0) as mf_overall_pnl FROM my_schema.mf_holdings WHERE run_date = (SELECT MAX(run_date) FROM my_schema.mf_holdings)")
        result = cursor.fetchone()
        mf_overall = float(result['mf_overall_pnl']) if result else 0.0
        
        cursor.execute("SELECT COALESCE(SUM(pnl), 0) as intraday_pnl FROM my_schema.positions WHERE run_date = (SELECT MAX(run_date) FROM my_schema.positions) AND position_type = 'day'")
        result = cursor.fetchone()
        intraday = float(result['intraday_pnl']) if result else 0.0
        
        # Get equity holdings
        cursor.execute("""
            SELECT trading_symbol, quantity, average_price, last_price,
                   (quantity * average_price) as invested_amount,
                   (quantity * last_price) as current_amount, pnl
            FROM my_schema.holdings WHERE run_date = (SELECT MAX(run_date) FROM my_schema.holdings) ORDER BY trading_symbol
        """)
        equity_holdings = cursor.fetchall()
        
        # Get MF holdings
        cursor.execute("""
            SELECT folio, fund, tradingsymbol, quantity, average_price, last_price,
                   invested_amount, current_value, pnl, net_change_percentage, day_change_percentage
            FROM my_schema.mf_holdings WHERE run_date = (SELECT MAX(run_date) FROM my_schema.mf_holdings) ORDER BY fund, tradingsymbol
        """)
        mf_holdings = cursor.fetchall()
        conn.close()
        
        # Create PDF in memory
        output = io.BytesIO()
        doc = SimpleDocTemplate(output, pagesize=A4, topMargin=0.5*inch)
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=16, textColor=colors.HexColor('#1a1a1a'), spaceAfter=12)
        
        elements = []
        elements.append(Paragraph("P&L Summary Report", title_style))
        elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        elements.append(Spacer(1, 0.3*inch))
        
        # P&L Summary Table
        summary_data = [
            ['Category', "Today's P&L", 'Overall P&L'],
            ['Equity Portfolio', f'Rs {equity_today:,.2f}', f'Rs {equity_overall:,.2f}'],
            ['Mutual Fund Portfolio', f'Rs {mf_today:,.2f}', f'Rs {mf_overall:,.2f}'],
            ['Intraday Trades', f'Rs {intraday:,.2f}', '-'],
            ['Total', f'Rs {equity_today + mf_today + intraday:,.2f}', f'Rs {equity_overall + mf_overall:,.2f}']
        ]
        summary_table = Table(summary_data, colWidths=[2.5*inch, 2*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -2), 'Helvetica'),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BACKGROUND', (0, -1), (-1, -1), colors.lightgrey),
        ]))
        elements.append(summary_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Generate Portfolio Value Chart
        try:
            cursor_hist = get_db_connection()
            cursor_history = cursor_hist.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cursor_history.execute("""
                SELECT 
                    run_date,
                    COALESCE(SUM(quantity * last_price), 0) as equity_value
                FROM my_schema.holdings
                WHERE run_date >= CURRENT_DATE - make_interval(days => 30)
                GROUP BY run_date
                ORDER BY run_date
            """)
            equity_data = cursor_history.fetchall()
            
            cursor_history.execute("""
                SELECT 
                    run_date,
                    COALESCE(SUM(current_value), 0) as mf_value
                FROM my_schema.mf_holdings
                WHERE run_date >= CURRENT_DATE - make_interval(days => 30)
                GROUP BY run_date
                ORDER BY run_date
            """)
            mf_data = cursor_history.fetchall()
            cursor_hist.close()
            
            equity_map = {str(row['run_date']): float(row['equity_value']) for row in equity_data}
            mf_map = {str(row['run_date']): float(row['mf_value']) for row in mf_data}
            
            all_dates = sorted(set(list(equity_map.keys()) + list(mf_map.keys())))
            
            if all_dates and len(all_dates) > 0:
                dates = [datetime.strptime(date, '%Y-%m-%d') for date in all_dates]
                equity_vals = [equity_map.get(date, 0.0) for date in all_dates]
                mf_vals = [mf_map.get(date, 0.0) for date in all_dates]
                total_vals = [e + m for e, m in zip(equity_vals, mf_vals)]
                
                plt.figure(figsize=(8, 4))
                ax = plt.subplot(111)
                
                ax.plot(dates, equity_vals, label='Equity', color='green', linewidth=2)
                ax.plot(dates, mf_vals, label='Mutual Fund', color='blue', linewidth=2)
                ax.plot(dates, total_vals, label='Total Portfolio', color='red', linewidth=2.5)
                
                ax.set_title('Total Portfolio Value (Last 30 Days)', fontsize=12, fontweight='bold')
                ax.set_xlabel('Date', fontsize=10)
                ax.set_ylabel('Value (₹)', fontsize=10)
                ax.legend(loc='best', fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                chart_buffer = io.BytesIO()
                plt.savefig(chart_buffer, format='png', dpi=150, bbox_inches='tight')
                chart_buffer.seek(0)
                plt.close()
                
                elements.append(Paragraph("Portfolio Value Trend", title_style))
                elements.append(Spacer(1, 0.2*inch))
                chart_img = Image(chart_buffer, width=6*inch, height=3*inch)
                elements.append(chart_img)
                elements.append(Spacer(1, 0.3*inch))
        except Exception as e:
            logging.error(f"Error generating portfolio chart: {e}")
            import traceback
            logging.error(traceback.format_exc())
        
        elements.append(PageBreak())
        
        # Equity Holdings Table
        elements.append(Paragraph("Equity Holdings", title_style))
        elements.append(Spacer(1, 0.2*inch))
        
        equity_total_invested = sum(row['invested_amount'] for row in equity_holdings)
        equity_total_current = sum(row['current_amount'] for row in equity_holdings)
        equity_total_pnl = sum(row['pnl'] for row in equity_holdings)
        
        equity_data = [['Symbol', 'Qty', 'Avg Price', 'LTP', 'Invested', 'Current', 'P&L']]
        for row in equity_holdings:
            row_dict = dict(row)
            equity_data.append([
                row_dict['trading_symbol'],
                str(row_dict['quantity']),
                f"Rs {row_dict['average_price']:.2f}",
                f"Rs {row_dict['last_price']:.2f}",
                f"Rs {row_dict['invested_amount']:.2f}",
                f"Rs {row_dict['current_amount']:.2f}",
                f"Rs {row_dict['pnl']:.2f}"
            ])
        
        equity_data.append([
            'TOTAL', '', '', '',
            f'Rs {equity_total_invested:,.2f}',
            f'Rs {equity_total_current:,.2f}',
            f'Rs {equity_total_pnl:,.2f}'
        ])
        
        equity_table = Table(equity_data, repeatRows=1, colWidths=[1.2*inch, 0.5*inch, 0.8*inch, 0.8*inch, 1*inch, 1*inch, 0.9*inch])
        equity_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -2), colors.beige),
            ('BACKGROUND', (0, -1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -2), 7),
            ('FONTSIZE', (0, -1), (-1, -1), 9),
        ]))
        elements.append(equity_table)
        elements.append(Spacer(1, 0.3*inch))
        elements.append(PageBreak())
        
        # MF Holdings Table
        elements.append(Paragraph("Mutual Fund Holdings", title_style))
        elements.append(Spacer(1, 0.2*inch))
        
        mf_total_invested = sum(row['invested_amount'] for row in mf_holdings)
        mf_total_current = sum(row['current_value'] for row in mf_holdings)
        mf_total_pnl = sum(row['pnl'] for row in mf_holdings)
        
        mf_data = [['Fund', 'Symbol', 'Qty', 'Avg Price', 'LTP', 'Invested', 'Current', 'P&L']]
        for row in mf_holdings:
            row_dict = dict(row)
            mf_data.append([
                row_dict.get('fund', '')[:20],
                row_dict.get('tradingsymbol', ''),
                str(row_dict.get('quantity', 0)),
                f"Rs {row_dict.get('average_price', 0):.2f}",
                f"Rs {row_dict.get('last_price', 0):.2f}",
                f"Rs {row_dict.get('invested_amount', 0):.2f}",
                f"Rs {row_dict.get('current_value', 0):.2f}",
                f"Rs {row_dict.get('pnl', 0):.2f}"
            ])
        
        mf_data.append([
            'TOTAL', '', '', '', '',
            f'Rs {mf_total_invested:,.2f}',
            f'Rs {mf_total_current:,.2f}',
            f'Rs {mf_total_pnl:,.2f}'
        ])
        
        mf_table = Table(mf_data, repeatRows=1, colWidths=[1.5*inch, 1*inch, 0.5*inch, 0.7*inch, 0.7*inch, 0.9*inch, 0.9*inch, 0.8*inch])
        mf_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -2), colors.beige),
            ('BACKGROUND', (0, -1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -2), 7),
            ('FONTSIZE', (0, -1), (-1, -1), 8),
        ]))
        elements.append(mf_table)
        
        doc.build(elements)
        output.seek(0)
        
        return StreamingResponse(
            io.BytesIO(output.read()),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=pnl_summary_{datetime.now().strftime('%Y%m%d')}.pdf"}
        )
    except Exception as e:
        logging.error(f"Error generating P&L Summary PDF: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"error": str(e)}


@router.post("/holdings/calculate-wyckoff")
async def calculate_wyckoff_for_selected(
    symbols: List[str] = Body(...),
    force_recalculate: bool = Body(False)
):
    """
    Calculate Wyckoff Accumulation/Distribution for selected stocks
    
    Args:
        symbols: List of trading symbols
        force_recalculate: If True, recalculate even if recent analysis exists
    
    Returns:
        Dict with results and statistics
    """
    try:
        from api.services.wyckoff_service import WyckoffService
        
        wyckoff_service = WyckoffService()
        results = wyckoff_service.calculate_for_symbols(symbols, force_recalculate)
        
        return results
        
    except Exception as e:
        logging.error(f"Error calculating Wyckoff for selected stocks: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "total": 0,
            "processed": 0,
            "results": [],
            "errors": []
        }


@router.post("/holdings/calculate-wyckoff-all")
async def calculate_wyckoff_for_all_holdings(
    force_recalculate: bool = Body(False)
):
    """
    Calculate Wyckoff Accumulation/Distribution for all holdings
    
    Args:
        force_recalculate: If True, recalculate even if recent analysis exists
    
    Returns:
        Dict with summary statistics
    """
    try:
        from api.services.wyckoff_service import WyckoffService
        
        wyckoff_service = WyckoffService()
        results = wyckoff_service.calculate_for_all_holdings(force_recalculate)
        
        return results
        
    except Exception as e:
        logging.error(f"Error calculating Wyckoff for all holdings: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "total_holdings": 0,
            "processed": 0
        }


@router.get("/holdings/wyckoff-status")
async def get_wyckoff_status():
    """
    Get status of Wyckoff analysis for all holdings
    
    Returns:
        Dict with analysis status for each holding
    """
    try:
        from api.services.wyckoff_service import WyckoffService
        
        wyckoff_service = WyckoffService()
        status = wyckoff_service.get_analysis_status()
        
        return status
        
    except Exception as e:
        logging.error(f"Error getting Wyckoff status: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "total_holdings": 0,
            "analyzed": 0,
            "not_analyzed": 0,
            "holdings_status": []
        }

