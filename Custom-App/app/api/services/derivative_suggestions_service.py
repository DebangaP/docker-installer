"""
Derivative Suggestions Service
Handles saving and tracking derivative suggestions and their outcomes
"""

import logging
import json
from datetime import datetime, date
from typing import List, Dict, Optional
from common.Boilerplate import get_db_connection
import psycopg2.extras

logger = logging.getLogger(__name__)


class DerivativeSuggestionsService:
    """Service for managing derivative suggestions and tracking their outcomes"""
    
    def save_suggestions(self, suggestions: List[Dict], analysis_date: Optional[str] = None, 
                        source: str = 'TPO', diagnostics: Optional[Dict] = None) -> int:
        """
        Save derivative suggestions to database
        
        Args:
            suggestions: List of suggestion dictionaries
            analysis_date: Analysis date (YYYY-MM-DD format or date object)
            source: Source of suggestions ('TPO', 'ORDERFLOW', etc.)
            diagnostics: Additional diagnostic information
            
        Returns:
            Number of suggestions saved
        """
        if not suggestions:
            logger.warning("No suggestions provided to save")
            return 0
        
        logger.info(f"Attempting to save {len(suggestions)} suggestions to database")
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            rows = []
            
            # Parse analysis_date
            if analysis_date is None:
                analysis_date_obj = datetime.now().date()
            elif isinstance(analysis_date, str):
                try:
                    analysis_date_obj = datetime.strptime(analysis_date, '%Y-%m-%d').date()
                except:
                    analysis_date_obj = datetime.now().date()
            else:
                analysis_date_obj = analysis_date
            
            for s in suggestions:
                try:
                    # Map suggestion fields - handle different field names
                    # Suggestions may have 'derivative_type' instead of 'strategy_type'
                    strategy_type = s.get('strategy_type') or s.get('derivative_type') or 'UNKNOWN'
                    strategy_name = s.get('strategy_name') or s.get('strategy_type') or s.get('derivative_type') or 'Unknown Strategy'
                    
                    # Map instrument - could be 'instrument' or 'tradingsymbol'
                    instrument = s.get('instrument') or s.get('tradingsymbol') or s.get('trading_symbol') or None
                    
                    # Map direction - could be 'direction' or derived from 'action'
                    direction = s.get('direction')
                    if not direction and s.get('action'):
                        action = s.get('action')
                        if action == 'BUY':
                            direction = 'Bullish'
                        elif action == 'SELL':
                            direction = 'Bearish'
                        else:
                            direction = 'Neutral'
                    
                    # Parse expiry date
                    expiry_val = s.get('expiry') or s.get('expiry_date')
                    if isinstance(expiry_val, str):
                        try:
                            expiry_dt = datetime.strptime(expiry_val, '%Y-%m-%d').date()
                        except:
                            try:
                                expiry_dt = datetime.strptime(expiry_val, '%Y-%m-%d %H:%M:%S').date()
                            except:
                                expiry_dt = None
                    elif isinstance(expiry_val, date):
                        expiry_dt = expiry_val
                    else:
                        expiry_dt = None
                    
                    # Prepare tpo_context
                    tpo_context = s.get('tpo_levels_used') or s.get('tpo_context') or {}
                    if isinstance(tpo_context, dict):
                        tpo_context_json = json.dumps(tpo_context)
                    else:
                        tpo_context_json = tpo_context if tpo_context else '{}'
                    
                    # Prepare diagnostics
                    diag = diagnostics or s.get('diagnostics') or {}
                    if isinstance(diag, dict):
                        diag_json = json.dumps(diag)
                    else:
                        diag_json = diag if diag else '{}'
                    
                    # Map entry_price - could be 'entry_price' or 'entry_level'
                    entry_price = s.get('entry_price') or s.get('entry_level')
                    
                    # Map strike_price - could be 'strike_price' or 'strike'
                    strike_price = s.get('strike_price') or s.get('strike')
                    
                    # Map quantity - could be 'quantity' or 'lots' or from 'position_size'
                    quantity = s.get('quantity') or s.get('lots')
                    if not quantity and s.get('position_size'):
                        # Try to extract number from position_size like "2 lots"
                        import re
                        match = re.search(r'(\d+)', str(s.get('position_size')))
                        if match:
                            quantity = int(match.group(1))
                    
                    # Map premium fields
                    total_premium = s.get('total_premium') or s.get('estimated_premium')
                    total_premium_income = s.get('total_premium_income')
                    
                    # Map margin - could be 'margin_required', 'estimated_margin', 'margin', or 'required_margin'
                    margin_required = (s.get('margin_required') or 
                                     s.get('estimated_margin') or 
                                     s.get('margin') or
                                     s.get('required_margin'))
                    if isinstance(margin_required, dict):
                        # If it's a dict, try to extract a numeric value
                        margin_required = margin_required.get('total_margin') or margin_required.get('total') or margin_required.get('margin') or None
                    
                    # Map profit/loss fields
                    # Note: max_profit column doesn't exist in the database, using max_potential_profit instead
                    potential_profit = s.get('potential_profit') or s.get('max_potential_profit')
                    
                    rows.append((
                        analysis_date_obj,
                        source,
                        strategy_type,
                        strategy_name,
                        instrument,
                        s.get('instrument_token'),
                        direction,
                        quantity,
                        s.get('lot_size') or 50,  # Default to 50 for Nifty
                        entry_price,
                        strike_price,
                        expiry_dt,
                        total_premium,
                        total_premium_income,
                        margin_required,
                        s.get('hedge_value'),
                        s.get('coverage_percentage'),
                        float(diag.get('portfolio_value') or 0.0) if isinstance(diag, dict) else 0.0,
                        float(diag.get('beta') or 0.0) if isinstance(diag, dict) else 0.0,
                        s.get('rationale') or s.get('reason'),
                        tpo_context_json,
                        diag_json,
                        potential_profit,
                        s.get('max_potential_profit'),
                        s.get('max_loss'),
                        s.get('risk_reward_ratio'),
                        s.get('probability_of_profit'),
                        s.get('breakeven'),
                        s.get('payoff_chart'),
                        s.get('payoff_sparkline')
                    ))
                except Exception as e:
                    logger.warning(f"Error processing suggestion: {e}. Suggestion data: {json.dumps(s, default=str)}")
                    continue
            
            if rows:
                cursor.executemany(
                    """
                    INSERT INTO my_schema.derivative_suggestions (
                        analysis_date, source, strategy_type, strategy_name, instrument, instrument_token,
                        direction, quantity, lot_size, entry_price, strike_price, expiry, total_premium,
                        total_premium_income, margin_required, hedge_value, coverage_percentage,
                        portfolio_value, beta, rationale, tpo_context, diagnostics,
                        potential_profit, max_potential_profit, max_loss, risk_reward_ratio, 
                        probability_of_profit, breakeven, payoff_chart, payoff_sparkline
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    """,
                    rows
                )
                conn.commit()
                saved_count = len(rows)
                logger.info(f"Saved {saved_count} derivative suggestions to database")
            else:
                saved_count = 0
            
            cursor.close()
            conn.close()
            return saved_count
            
        except Exception as e:
            logger.error(f"Error saving derivative suggestions: {e}")
            import traceback
            logger.error(traceback.format_exc())
            logger.error(f"Suggestions data sample (first item): {json.dumps(suggestions[0] if suggestions else {}, default=str, indent=2)}")
            if 'conn' in locals():
                try:
                    conn.rollback()
                    conn.close()
                except:
                    pass
            return 0
    
    def update_suggestion_result(self, suggestion_id: int, status: str, 
                                 exit_date: Optional[str] = None,
                                 exit_price: Optional[float] = None,
                                 actual_profit: Optional[float] = None,
                                 actual_loss: Optional[float] = None,
                                 actual_pnl: Optional[float] = None,
                                 outcome: Optional[str] = None,
                                 notes: Optional[str] = None) -> bool:
        """
        Update the result/outcome of a suggestion
        
        Args:
            suggestion_id: ID of the suggestion to update
            status: New status (EXECUTED, CLOSED, EXPIRED, CANCELLED)
            exit_date: Exit date (YYYY-MM-DD format)
            exit_price: Exit price
            actual_profit: Actual profit realized
            actual_loss: Actual loss incurred
            actual_pnl: Net P&L (profit - loss)
            outcome: Outcome (SUCCESS, FAILURE, PARTIAL, BREAKEVEN)
            notes: Additional notes
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Parse exit_date
            exit_date_obj = None
            if exit_date:
                if isinstance(exit_date, str):
                    try:
                        exit_date_obj = datetime.strptime(exit_date, '%Y-%m-%d').date()
                    except:
                        pass
                elif isinstance(exit_date, date):
                    exit_date_obj = exit_date
            
            # Calculate actual_pnl if not provided but profit/loss are
            if actual_pnl is None:
                if actual_profit is not None and actual_loss is not None:
                    actual_pnl = actual_profit - actual_loss
                elif actual_profit is not None:
                    actual_pnl = actual_profit
                elif actual_loss is not None:
                    actual_pnl = -actual_loss
            
            # Determine outcome if not provided
            if outcome is None and actual_pnl is not None:
                if actual_pnl > 0:
                    outcome = 'SUCCESS'
                elif actual_pnl < 0:
                    outcome = 'FAILURE'
                else:
                    outcome = 'BREAKEVEN'
            
            # Set executed_at if status is EXECUTED and not already set
            update_fields = []
            params = []
            
            update_fields.append("status = %s")
            params.append(status)
            
            if exit_date_obj:
                update_fields.append("exit_date = %s")
                params.append(exit_date_obj)
            
            if exit_price is not None:
                update_fields.append("exit_price = %s")
                params.append(exit_price)
            
            if actual_profit is not None:
                update_fields.append("actual_profit = %s")
                params.append(actual_profit)
            
            if actual_loss is not None:
                update_fields.append("actual_loss = %s")
                params.append(actual_loss)
            
            if actual_pnl is not None:
                update_fields.append("actual_pnl = %s")
                params.append(actual_pnl)
            
            if outcome:
                update_fields.append("outcome = %s")
                params.append(outcome)
            
            if notes:
                update_fields.append("notes = %s")
                params.append(notes)
            
            # Set executed_at if status is EXECUTED
            if status == 'EXECUTED':
                update_fields.append("executed_at = COALESCE(executed_at, CURRENT_TIMESTAMP)")
            
            # Always update updated_at
            update_fields.append("updated_at = CURRENT_TIMESTAMP")
            
            params.append(suggestion_id)
            
            query = f"""
                UPDATE my_schema.derivative_suggestions
                SET {', '.join(update_fields)}
                WHERE id = %s
            """
            
            cursor.execute(query, tuple(params))
            conn.commit()
            
            if cursor.rowcount > 0:
                logger.info(f"Updated suggestion {suggestion_id} with status={status}, outcome={outcome}")
                cursor.close()
                conn.close()
                return True
            else:
                logger.warning(f"Suggestion {suggestion_id} not found for update")
                cursor.close()
                conn.close()
                return False
                
        except Exception as e:
            logger.error(f"Error updating suggestion result: {e}")
            import traceback
            logger.error(traceback.format_exc())
            if conn:
                try:
                    conn.rollback()
                    conn.close()
                except:
                    pass
            return False
    
    def get_efficacy_statistics(self, start_date: Optional[str] = None, 
                                end_date: Optional[str] = None,
                                strategy_type: Optional[str] = None,
                                source: Optional[str] = None) -> Dict:
        """
        Get efficacy statistics for suggestions
        
        Args:
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            strategy_type: Strategy type filter
            source: Source filter (TPO, ORDERFLOW, etc.)
            
        Returns:
            Dictionary with efficacy statistics
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            where_clauses = []
            params = []
            
            if start_date:
                where_clauses.append("generated_at::date >= %s")
                params.append(start_date)
            
            if end_date:
                where_clauses.append("generated_at::date <= %s")
                params.append(end_date)
            
            if strategy_type:
                where_clauses.append("strategy_type = %s")
                params.append(strategy_type.upper())
            
            if source:
                where_clauses.append("source = %s")
                params.append(source.upper())
            
            where_clause = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
            
            query = f"""
                SELECT 
                    COUNT(*) as total_suggestions,
                    COUNT(CASE WHEN status = 'PENDING' THEN 1 END) as pending,
                    COUNT(CASE WHEN status = 'EXECUTED' THEN 1 END) as executed,
                    COUNT(CASE WHEN status = 'CLOSED' THEN 1 END) as closed,
                    COUNT(CASE WHEN status = 'EXPIRED' THEN 1 END) as expired,
                    COUNT(CASE WHEN status = 'CANCELLED' THEN 1 END) as cancelled,
                    COUNT(CASE WHEN status = 'MOCKED' THEN 1 END) as mocked,
                    COUNT(CASE WHEN outcome = 'SUCCESS' THEN 1 END) as successful,
                    COUNT(CASE WHEN outcome = 'FAILURE' THEN 1 END) as failed,
                    COUNT(CASE WHEN outcome = 'PARTIAL' THEN 1 END) as partial,
                    COUNT(CASE WHEN outcome = 'BREAKEVEN' THEN 1 END) as breakeven,
                    SUM(actual_pnl) as total_pnl,
                    AVG(actual_pnl) as avg_pnl,
                    SUM(actual_profit) as total_profit,
                    SUM(actual_loss) as total_loss,
                    SUM(CASE WHEN actual_pnl > 0 THEN 1 ELSE 0 END) as profitable_count,
                    SUM(CASE WHEN actual_pnl < 0 THEN 1 ELSE 0 END) as loss_making_count,
                    AVG(CASE WHEN actual_pnl IS NOT NULL THEN actual_pnl END) as avg_pnl_closed,
                    AVG(risk_reward_ratio) as avg_risk_reward_ratio,
                    AVG(probability_of_profit) as avg_probability_of_profit
                FROM my_schema.derivative_suggestions
                {where_clause}
            """
            
            cursor.execute(query, tuple(params))
            stats = cursor.fetchone()
            
            # Calculate success rate
            total_closed = stats['closed'] + stats['expired'] if stats else 0
            total_with_outcome = (stats['successful'] or 0) + (stats['failed'] or 0) + (stats['partial'] or 0) + (stats['breakeven'] or 0) if stats else 0
            
            success_rate = None
            if total_with_outcome > 0:
                success_rate = ((stats['successful'] or 0) / total_with_outcome * 100) if stats else None
            
            result = {
                'total_suggestions': stats['total_suggestions'] if stats else 0,
                'status_breakdown': {
                    'pending': stats['pending'] if stats else 0,
                    'executed': stats['executed'] if stats else 0,
                    'closed': stats['closed'] if stats else 0,
                    'expired': stats['expired'] if stats else 0,
                    'cancelled': stats['cancelled'] if stats else 0,
                    'mocked': stats['mocked'] if stats else 0
                },
                'outcome_breakdown': {
                    'successful': stats['successful'] if stats else 0,
                    'failed': stats['failed'] if stats else 0,
                    'partial': stats['partial'] if stats else 0,
                    'breakeven': stats['breakeven'] if stats else 0
                },
                'pnl_statistics': {
                    'total_pnl': float(stats['total_pnl']) if stats and stats['total_pnl'] else 0.0,
                    'avg_pnl': float(stats['avg_pnl']) if stats and stats['avg_pnl'] else 0.0,
                    'avg_pnl_closed': float(stats['avg_pnl_closed']) if stats and stats['avg_pnl_closed'] else 0.0,
                    'total_profit': float(stats['total_profit']) if stats and stats['total_profit'] else 0.0,
                    'total_loss': float(stats['total_loss']) if stats and stats['total_loss'] else 0.0,
                    'profitable_count': stats['profitable_count'] if stats else 0,
                    'loss_making_count': stats['loss_making_count'] if stats else 0
                },
                'performance_metrics': {
                    'success_rate_pct': round(success_rate, 2) if success_rate is not None else None,
                    'avg_risk_reward_ratio': float(stats['avg_risk_reward_ratio']) if stats and stats['avg_risk_reward_ratio'] else None,
                    'avg_probability_of_profit': float(stats['avg_probability_of_profit']) if stats and stats['avg_probability_of_profit'] else None
                }
            }
            
            cursor.close()
            conn.close()
            return result
            
        except Exception as e:
            logger.error(f"Error getting efficacy statistics: {e}")
            import traceback
            logger.error(traceback.format_exc())
            if conn:
                try:
                    conn.close()
                except:
                    pass
            return {
                'total_suggestions': 0,
                'status_breakdown': {},
                'outcome_breakdown': {},
                'pnl_statistics': {},
                'performance_metrics': {}
            }

