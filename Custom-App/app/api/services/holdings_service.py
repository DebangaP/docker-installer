"""
Holdings service for business logic related to holdings, positions, and portfolio management.
"""
from typing import List
from common.Boilerplate import get_db_connection
import psycopg2.extras
import logging


class HoldingsService:
    """Service class for holdings-related business logic"""
    
    def get_holdings_data(self, page: int = 1, per_page: int = 10, sort_by: str = None, sort_dir: str = "asc", search: str = None):
        """Fetch current holdings from the database with pagination and sorting."""
        try:
            conn = get_db_connection()
            # Use RealDictCursor to get results as dictionaries
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            # Build WHERE clause with search filter
            where_clause = "WHERE h.run_date = (SELECT MAX(run_date) FROM my_schema.holdings)"
            search_params = []
            if search and search.strip():
                where_clause += " AND h.trading_symbol ILIKE %s"
                search_params.append(f"%{search.strip()}%")

            # Fetch holdings for the most recent run_date
            # Get total count for pagination
            count_query = f"""
                SELECT COUNT(*) as total_count
                FROM my_schema.holdings h
                {where_clause}
            """
            cursor.execute(count_query, search_params)
            total_count = cursor.fetchone()['total_count']

            # Validate and set sort column
            valid_sort_columns = ['trading_symbol', 'invested_amount', 'current_amount', 'pnl', 'today_pnl', 'prophet_prediction_pct']
            if sort_by not in valid_sort_columns:
                sort_by = 'trading_symbol'
            
            # Validate sort direction
            if sort_dir.lower() not in ['asc', 'desc']:
                sort_dir = 'asc'

            # Calculate offset for pagination
            offset = (page - 1) * per_page

            # Fetch paginated holdings for the most recent run_date
            # Calculate invested amount and current value with coalesced prices
            # Include Prophet predictions from latest run_date
            cursor.execute("""
                SELECT 
                    h.trading_symbol,
                    h.instrument_token,
                    h.quantity,
                    h.average_price,
                    h.last_price,
                    COALESCE(h.last_price, rt.price_close) as current_price,
                    h.pnl,
                    (h.quantity * h.average_price) as invested_amount,
                    (h.quantity * COALESCE(h.last_price, rt.price_close)) as current_amount,
                    round(case when (h.quantity * h.average_price) != 0 then
                        (h.pnl / (h.quantity * h.average_price)) * 100
                    else 0
                    end::numeric, 2) as pnl_pct_change,
                    pp.predicted_price_change_pct,
                    pp.prediction_confidence,
                    (SELECT MAX(run_date) FROM my_schema.holdings) as run_date
                FROM my_schema.holdings h
                LEFT JOIN (
                    SELECT DISTINCT ON (scrip_id) scrip_id, price_close, price_date
                    FROM my_schema.rt_intraday_price
                    WHERE price_date::date <= CURRENT_DATE
                    ORDER BY scrip_id, price_date DESC
                ) rt ON h.trading_symbol = rt.scrip_id
                LEFT JOIN (
                    SELECT DISTINCT ON (scrip_id) scrip_id, predicted_price_change_pct, prediction_confidence
                    FROM my_schema.prophet_predictions
                    WHERE run_date = (SELECT MAX(run_date) FROM my_schema.prophet_predictions WHERE status = 'ACTIVE')
                    AND prediction_days = 30
                    AND status = 'ACTIVE'
                    ORDER BY scrip_id, run_date DESC
                ) pp ON h.trading_symbol = pp.scrip_id
                {where_clause}
                ORDER BY {sort_by} {sort_dir}
                LIMIT %s OFFSET %s
            """.format(where_clause=where_clause, sort_by=sort_by, sort_dir=sort_dir.upper()), search_params + [per_page, offset])
            holdings = cursor.fetchall()
            
            # Convert RealDictRow objects to regular dictionaries for Jinja2 template compatibility
            holdings_list = [dict(row) for row in holdings]
            
            conn.close()
            return {
                "holdings": holdings_list,
                "total_count": total_count,
                "page": page,
                "per_page": per_page
            }
        except Exception as e:
            logging.error(f"Error fetching holdings data: {e}")
            return {
                "holdings": [],
                "total_count": 0,
                "page": page,
                "per_page": per_page
            }

    def enrich_holdings_with_today_pnl(self, holdings_data):
        """Enrich holdings data with today's P&L from rt_intraday_price"""
        try:
            conn = get_db_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Get latest available date (not necessarily today - could be last trading day)
            cursor.execute("""
                SELECT MAX(price_date::date) as latest_date
                FROM my_schema.rt_intraday_price
            """)
            latest_result = cursor.fetchone()
            latest_date = latest_result['latest_date'] if latest_result and latest_result['latest_date'] else None
            
            if not latest_date:
                logging.warning("No data found in rt_intraday_price, returning holdings without today's P&L")
                conn.close()
                # Return holdings with default today_pnl values
                enriched_holdings = []
                for holding in holdings_data.get('holdings', []):
                    holding_dict = dict(holding)
                    holding_dict['today_pnl'] = 0.0
                    holding_dict['today_price'] = 0.0
                    holding_dict['prev_price'] = 0.0
                    holding_dict['pct_change'] = 0.0
                    holding_dict['today_change'] = '0 (0%)'
                    enriched_holdings.append(holding_dict)
                holdings_data['holdings'] = enriched_holdings
                return holdings_data
            
            # Convert latest_date to string
            latest_date_str = latest_date.strftime('%Y-%m-%d') if hasattr(latest_date, 'strftime') else str(latest_date)
            
            # Get previous trading day
            cursor.execute("""
                SELECT MAX(price_date::date) as prev_date
                FROM my_schema.rt_intraday_price
                WHERE price_date::date < %s
            """, (latest_date,))
            
            prev_date_result = cursor.fetchone()
            prev_date = prev_date_result['prev_date'] if prev_date_result and prev_date_result['prev_date'] else None
            
            logging.info(f"Using latest date: {latest_date_str}, previous date: {prev_date}")
            
            # Get today's P&L for all holdings - match by instrument_token
            today_pnl_map = {}
            
            # Always populate the map with holdings, even if prev_date doesn't exist (will default to 0)
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
                # Convert prev_date to string format
                prev_date_str = prev_date.strftime('%Y-%m-%d') if hasattr(prev_date, 'strftime') else str(prev_date)
                cursor.execute("""
                    WITH holdings_today AS (
                        SELECT instrument_token, trading_symbol, quantity, last_price
                        FROM my_schema.holdings
                        WHERE run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
                    ),
                    today_prices AS (
                        SELECT scrip_id, price_close, price_date
                        FROM my_schema.rt_intraday_price
                        WHERE price_date::date = %s
                    ),
                    latest_prices AS (
                        SELECT scrip_id, price_close, 
                               ROW_NUMBER() OVER (PARTITION BY scrip_id ORDER BY price_date DESC) as rn
                        FROM my_schema.rt_intraday_price
                        WHERE price_date::date <= %s
                    ),
                    prev_prices AS (
                        SELECT scrip_id, price_close
                        FROM my_schema.rt_intraday_price
                        WHERE price_date::date = %s
                    )
                    SELECT 
                        h.instrument_token,
                        h.trading_symbol,
                        h.quantity,
                        COALESCE(h.last_price, today_p.price_close, latest_p.price_close, 0) as today_price,
                        COALESCE(prev_p.price_close, 0) as prev_price,
                        -- If today's price is null/0, P&L should be 0 regardless of prev price
                        CASE 
                            WHEN (COALESCE(h.last_price, today_p.price_close, latest_p.price_close) IS NULL OR COALESCE(h.last_price, today_p.price_close, latest_p.price_close) = 0) THEN 0
                            ELSE h.quantity * (COALESCE(h.last_price, today_p.price_close, latest_p.price_close, 0) - COALESCE(prev_p.price_close, 0))
                        END as today_pnl,
                        round(case when prev_p.price_close != 0 AND COALESCE(h.last_price, today_p.price_close, latest_p.price_close) IS NOT NULL AND COALESCE(h.last_price, today_p.price_close, latest_p.price_close) != 0 then 
                            ((COALESCE(h.last_price, today_p.price_close, latest_p.price_close, 0) - COALESCE(prev_p.price_close, 0))/COALESCE(prev_p.price_close, 0)) * 100
                        else 0
                        end::numeric, 2) as pct_change,
                        concat(round(CASE 
                            WHEN (COALESCE(h.last_price, today_p.price_close, latest_p.price_close) IS NULL OR COALESCE(h.last_price, today_p.price_close, latest_p.price_close) = 0) THEN 0
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
                    
                """, (latest_date, latest_date, prev_date))
                
                for row in cursor.fetchall():
                    instrument_token = row['instrument_token']
                    today_price = float(row['today_price']) if row['today_price'] else 0.0
                    prev_price = float(row['prev_price']) if row['prev_price'] else 0.0
                    today_pnl = float(row['today_pnl']) if row['today_pnl'] else 0.0
                    pct_change = float(row['pct_change']) if row['pct_change'] else 0.0
                    today_change_str = str(row.get("Todays_Change", '')) if row.get("Todays_Change") else ''
                    
                    # Store by instrument_token to match with holdings
                    today_pnl_map[instrument_token] = {
                        'today_pnl': today_pnl,
                        'today_price': today_price,
                        'prev_price': prev_price,
                        'pct_change': pct_change,
                        'today_change': today_change_str
                    }
            
            # Get Prophet predictions for all holdings (60-day predictions)
            try:
                # First try to get 60-day predictions
                cursor.execute("""
                    SELECT scrip_id, predicted_price_change_pct, prediction_confidence, prediction_days
                    FROM my_schema.prophet_predictions
                    WHERE run_date = (SELECT MAX(run_date) FROM my_schema.prophet_predictions WHERE status = 'ACTIVE' AND prediction_days = 60)
                    AND prediction_days = 60
                    AND status = 'ACTIVE'
                """)
                
                predictions_rows = cursor.fetchall()
                
                # If no 60-day predictions, try to get latest predictions regardless of prediction_days
                if not predictions_rows:
                    logging.warning("No 60-day Prophet predictions found for holdings, trying latest predictions")
                    cursor.execute("""
                        SELECT scrip_id, predicted_price_change_pct, prediction_confidence, prediction_days
                        FROM my_schema.prophet_predictions pp1
                        WHERE status = 'ACTIVE'
                        AND run_date = (SELECT MAX(run_date) FROM my_schema.prophet_predictions WHERE status = 'ACTIVE')
                    """)
                    predictions_rows = cursor.fetchall()
                
                predictions_map = {row['scrip_id'].upper(): dict(row) for row in predictions_rows}
                logging.info(f"Loaded {len(predictions_map)} Prophet predictions for holdings enrichment")
                
                if predictions_map:
                    sample_ids = list(predictions_map.keys())[:5]
                    logging.debug(f"Sample Prophet prediction scrip_ids for holdings: {sample_ids}")
            except Exception as e:
                logging.error(f"Error loading Prophet predictions for holdings: {e}")
                import traceback
                logging.error(traceback.format_exc())
                predictions_map = {}
            
            conn.close()
            
            # Convert RealDictRow objects to regular dictionaries and enrich with today's P&L and Prophet predictions
            enriched_holdings = []
            matched_predictions = 0
            holdings_symbols = []
            
            for holding in holdings_data.get('holdings', []):
                # Convert RealDictRow to dict
                holding_dict = dict(holding)
                instrument_token = holding_dict.get('instrument_token')
                symbol = holding_dict.get('trading_symbol', '')
                holdings_symbols.append(symbol.upper() if symbol else '')
                
                today_pnl_info = today_pnl_map.get(instrument_token, {'today_pnl': 0.0, 'today_price': 0.0, 'prev_price': 0.0, 'pct_change': 0.0, 'today_change': '0 (0%)'})
                
                # Add today's P&L fields
                holding_dict['today_pnl'] = today_pnl_info['today_pnl']
                holding_dict['today_price'] = today_pnl_info['today_price']
                holding_dict['prev_price'] = today_pnl_info['prev_price']
                holding_dict['pct_change'] = today_pnl_info.get('pct_change', 0.0)
                holding_dict['today_change'] = today_pnl_info.get('today_change', '0 (0%)')
                
                # Get Prophet prediction for this holding
                prophet_pred = None
                prophet_conf = None
                prediction_days = None
                if symbol:
                    symbol_upper = symbol.upper()
                    if symbol_upper in predictions_map:
                        pred = predictions_map[symbol_upper]
                        try:
                            pred_pct = pred.get('predicted_price_change_pct')
                            pred_conf = pred.get('prediction_confidence')
                            pred_days = pred.get('prediction_days')
                            
                            if pred_pct is not None and not (isinstance(pred_pct, float) and (pred_pct != pred_pct)):  # Check for NaN
                                prophet_pred = float(pred_pct)
                            if pred_conf is not None and not (isinstance(pred_conf, float) and (pred_conf != pred_conf)):  # Check for NaN
                                prophet_conf = float(pred_conf)
                            if pred_days is not None:
                                prediction_days = int(pred_days)
                            
                            if prophet_pred is not None:
                                matched_predictions += 1
                        except (ValueError, TypeError) as e:
                            logging.debug(f"Error converting Prophet prediction for {symbol}: {e}")
                            prophet_pred = None
                            prophet_conf = None
                            prediction_days = None
                    else:
                        logging.debug(f"No Prophet prediction found for holding {symbol} (searched as: {symbol_upper})")
                
                # Add Prophet prediction fields
                holding_dict['prophet_prediction_pct'] = prophet_pred
                holding_dict['prophet_confidence'] = prophet_conf
                holding_dict['prediction_days'] = prediction_days
                
                enriched_holdings.append(holding_dict)
            
            holdings_data['holdings'] = enriched_holdings
            logging.info(f"Enriched {len(enriched_holdings)} holdings with today's P&L. Matched {matched_predictions} Prophet predictions.")
            return holdings_data
            
        except Exception as e:
            logging.error(f"Error enriching holdings with today's P&L: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return holdings_data
    
    def get_all_holdings_symbols(self) -> List[str]:
        """
        Get list of all trading symbols from holdings
        
        Returns:
            List of trading symbols
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cursor.execute("""
                SELECT DISTINCT trading_symbol
                FROM my_schema.holdings
                WHERE run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
                AND trading_symbol IS NOT NULL
                ORDER BY trading_symbol
            """)
            
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            
            symbols = [row['trading_symbol'] for row in rows if row['trading_symbol']]
            return symbols
            
        except Exception as e:
            logging.error(f"Error getting holdings symbols: {e}")
            return []

