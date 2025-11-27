"""
Theoretical P&L Calculator Service
Calculates theoretical gains/losses for derivative suggestions based on entry price, target levels, and stop loss
"""

import logging
import json
from datetime import datetime, date
from typing import List, Dict, Optional
from common.Boilerplate import get_db_connection
import psycopg2.extras
from options.DerivativesSuggestionEngine import (
    calculate_futures_profit,
    calculate_options_profit,
    calculate_straddle_profit
)

logger = logging.getLogger(__name__)


class TheoreticalPnlCalculator:
    """Service for calculating theoretical P&L for derivative suggestions"""
    
    def calculate_theoretical_pnl_for_suggestions(self, 
                                                  start_date: Optional[str] = None,
                                                  end_date: Optional[str] = None,
                                                  strategy_type: Optional[str] = None,
                                                  source: Optional[str] = None) -> Dict:
        """
        Calculate theoretical P&L for all PENDING suggestions
        
        Args:
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            strategy_type: Strategy type filter
            source: Source filter (TPO, ORDERFLOW, etc.)
            
        Returns:
            Dictionary with updated_count and suggestions_processed
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Build WHERE clause for fetching PENDING suggestions
            where_clauses = ["status = 'PENDING'"]
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
            
            where_clause = "WHERE " + " AND ".join(where_clauses)
            
            # Fetch PENDING suggestions
            query = f"""
                SELECT id, entry_price, strike_price, quantity, lot_size, strategy_type,
                       direction, instrument_token, tpo_context, diagnostics, total_premium,
                       instrument
                FROM my_schema.derivative_suggestions
                {where_clause}
                ORDER BY generated_at DESC
            """
            
            cursor.execute(query, tuple(params))
            suggestions = cursor.fetchall()
            
            updated_count = 0
            suggestions_processed = len(suggestions)
            
            logger.info(f"Processing {suggestions_processed} PENDING suggestions for theoretical P&L calculation")
            
            for suggestion in suggestions:
                try:
                    result = self._calculate_single_suggestion_pnl(suggestion, cursor)
                    if result:
                        updated_count += 1
                except Exception as e:
                    logger.warning(f"Error calculating P&L for suggestion {suggestion['id']}: {e}")
                    continue
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"Successfully calculated theoretical P&L for {updated_count} out of {suggestions_processed} suggestions")
            
            return {
                'updated_count': updated_count,
                'suggestions_processed': suggestions_processed
            }
            
        except Exception as e:
            logger.error(f"Error calculating theoretical P&L: {e}")
            import traceback
            logger.error(traceback.format_exc())
            if 'conn' in locals():
                try:
                    conn.rollback()
                    conn.close()
                except:
                    pass
            return {
                'updated_count': 0,
                'suggestions_processed': 0
            }
    
    def _calculate_single_suggestion_pnl(self, suggestion: Dict, cursor) -> bool:
        """
        Calculate P&L for a single suggestion and update database
        
        Args:
            suggestion: Suggestion record from database
            cursor: Database cursor
            
        Returns:
            True if successfully updated, False otherwise
        """
        try:
            suggestion_id = suggestion['id']
            strategy_type = suggestion.get('strategy_type', '').upper()
            
            # Get entry price - could be in entry_price field or we need to extract from diagnostics/tpo_context
            entry_price = suggestion.get('entry_price')
            if not entry_price:
                # Try to get from diagnostics or tpo_context
                diagnostics_raw = suggestion.get('diagnostics')
                tpo_context_raw = suggestion.get('tpo_context')
                
                # Parse if strings
                if isinstance(diagnostics_raw, str):
                    try:
                        diagnostics_raw = json.loads(diagnostics_raw)
                    except:
                        diagnostics_raw = {}
                if isinstance(tpo_context_raw, str):
                    try:
                        tpo_context_raw = json.loads(tpo_context_raw)
                    except:
                        tpo_context_raw = {}
                
                # Check for entry_level in diagnostics or decision_context
                if isinstance(diagnostics_raw, dict):
                    if 'entry_level' in diagnostics_raw:
                        entry_price = diagnostics_raw['entry_level']
                    elif 'decision_context' in diagnostics_raw:
                        decision_context = diagnostics_raw['decision_context']
                        if isinstance(decision_context, dict) and 'key_levels' in decision_context:
                            key_levels = decision_context['key_levels']
                            if isinstance(key_levels, dict) and 'entry_level' in key_levels:
                                entry_price = key_levels['entry_level']
                
                if not entry_price:
                    logger.warning(f"Suggestion {suggestion_id} has no entry_price or entry_level")
                    return False
            
            # Get current market price
            current_price = self._get_current_market_price(suggestion.get('instrument_token'))
            if not current_price:
                logger.warning(f"Could not get current price for suggestion {suggestion_id}, using entry price")
                current_price = entry_price
            
            # Get target levels and stop loss from diagnostics or tpo_context
            diagnostics = suggestion.get('diagnostics')
            tpo_context = suggestion.get('tpo_context')
            
            target_levels = []
            stop_loss = None
            
            # Parse diagnostics if it's a JSON string
            if isinstance(diagnostics, str):
                try:
                    diagnostics = json.loads(diagnostics)
                except:
                    diagnostics = {}
            
            # Parse tpo_context if it's a JSON string
            if isinstance(tpo_context, str):
                try:
                    tpo_context = json.loads(tpo_context)
                except:
                    tpo_context = {}
            
            # Extract target levels and stop loss
            if isinstance(diagnostics, dict):
                # Check for target_levels in diagnostics
                if 'target_levels' in diagnostics:
                    target_levels = diagnostics['target_levels']
                elif 'targets' in diagnostics:
                    target_levels = diagnostics['targets']
                
                # Check for stop_loss in diagnostics
                if 'stop_loss' in diagnostics:
                    stop_loss = diagnostics.get('stop_loss')
                
                # Also check decision_context for key_levels
                if 'decision_context' in diagnostics:
                    decision_context = diagnostics['decision_context']
                    if isinstance(decision_context, dict):
                        if 'key_levels' in decision_context:
                            key_levels = decision_context['key_levels']
                            if isinstance(key_levels, dict):
                                if 'targets' in key_levels and not target_levels:
                                    targets = key_levels['targets']
                                    if isinstance(targets, list):
                                        target_levels = targets
                                if 'stop_loss' in key_levels and not stop_loss:
                                    stop_loss = key_levels.get('stop_loss')
            
            # Also check tpo_context
            if isinstance(tpo_context, dict):
                if 'target_levels' in tpo_context and not target_levels:
                    target_levels = tpo_context['target_levels']
                if 'stop_loss' in tpo_context and not stop_loss:
                    stop_loss = tpo_context.get('stop_loss')
            
            # Determine exit price
            exit_price = self._determine_exit_price(current_price, target_levels, stop_loss, entry_price)
            exit_reason = self._get_exit_reason(current_price, target_levels, stop_loss, exit_price)
            
            # Calculate P&L based on strategy type
            pnl = self._calculate_pnl(strategy_type, suggestion, entry_price, exit_price)
            
            # Determine outcome
            if pnl > 0:
                outcome = 'SUCCESS'
            elif pnl < 0:
                outcome = 'FAILURE'
            else:
                outcome = 'BREAKEVEN'
            
            # Update suggestion record
            update_query = """
                UPDATE my_schema.derivative_suggestions
                SET status = 'MOCKED',
                    exit_price = %s,
                    exit_date = CURRENT_DATE,
                    actual_pnl = %s,
                    outcome = %s,
                    notes = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """
            
            notes = f"Theoretical calculation: Entry at ₹{entry_price:.2f}, Exit at ₹{exit_price:.2f} ({exit_reason})"
            
            cursor.execute(update_query, (
                exit_price,
                pnl,
                outcome,
                notes,
                suggestion_id
            ))
            
            return True
            
        except Exception as e:
            logger.error(f"Error calculating P&L for suggestion {suggestion.get('id')}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _get_current_market_price(self, instrument_token: Optional[int]) -> Optional[float]:
        """
        Get current market price from database
        
        Args:
            instrument_token: Instrument token
            
        Returns:
            Current price or None
        """
        if not instrument_token:
            return None
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get latest price from ticks table
            query = """
                SELECT last_price
                FROM my_schema.ticks
                WHERE instrument_token = %s
                ORDER BY timestamp DESC
                LIMIT 1
            """
            
            cursor.execute(query, (instrument_token,))
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result and result[0]:
                return float(result[0])
            
            return None
            
        except Exception as e:
            logger.warning(f"Error fetching current price for instrument {instrument_token}: {e}")
            return None
    
    def _determine_exit_price(self, current_price: float, target_levels: List, 
                              stop_loss: Optional[float], entry_price: float) -> float:
        """
        Determine exit price based on target levels, stop loss, and current price
        
        Args:
            current_price: Current market price
            target_levels: List of target levels (can be list of numbers or list of dicts with 'level' key)
            stop_loss: Stop loss price
            entry_price: Entry price
            
        Returns:
            Exit price
        """
        # Extract numeric target levels
        numeric_targets = []
        if target_levels:
            for target in target_levels:
                if isinstance(target, dict):
                    level = target.get('level') or target.get('price')
                    if level:
                        numeric_targets.append(float(level))
                elif isinstance(target, (int, float)):
                    numeric_targets.append(float(target))
        
        # Check if current price >= any target level (use highest target reached)
        if numeric_targets:
            reached_targets = [t for t in numeric_targets if current_price >= t]
            if reached_targets:
                return max(reached_targets)
        
        # Check if current price <= stop loss
        if stop_loss and current_price <= stop_loss:
            return float(stop_loss)
        
        # Otherwise use current market price
        return current_price
    
    def _get_exit_reason(self, current_price: float, target_levels: List,
                        stop_loss: Optional[float], exit_price: float) -> str:
        """
        Get reason for exit price determination
        
        Args:
            current_price: Current market price
            target_levels: List of target levels
            stop_loss: Stop loss price
            exit_price: Determined exit price
            
        Returns:
            Exit reason string
        """
        # Extract numeric target levels
        numeric_targets = []
        if target_levels:
            for target in target_levels:
                if isinstance(target, dict):
                    level = target.get('level') or target.get('price')
                    if level:
                        numeric_targets.append(float(level))
                elif isinstance(target, (int, float)):
                    numeric_targets.append(float(target))
        
        # Check if exit is at target
        if numeric_targets and exit_price in numeric_targets:
            return "target level"
        
        # Check if exit is at stop loss
        if stop_loss and abs(exit_price - stop_loss) < 0.01:
            return "stop loss"
        
        # Otherwise it's current market price
        return "current price"
    
    def _calculate_pnl(self, strategy_type: str, suggestion: Dict, 
                       entry_price: float, exit_price: float) -> float:
        """
        Calculate P&L based on strategy type
        
        Args:
            strategy_type: Strategy type (FUTURES, CALL, PUT, STRADDLE)
            suggestion: Suggestion record
            entry_price: Entry price
            exit_price: Exit price
            
        Returns:
            Calculated P&L
        """
        quantity = suggestion.get('quantity') or 1
        lot_size = suggestion.get('lot_size') or 50
        direction = suggestion.get('direction', '')
        
        # Determine action from direction or action field
        action = suggestion.get('action', 'BUY')
        if not action or action not in ['BUY', 'SELL']:
            # Fallback to direction
            if 'bearish' in direction.lower() or 'sell' in direction.lower():
                action = 'SELL'
            else:
                action = 'BUY'
        
        if strategy_type == 'FUTURES':
            return calculate_futures_profit(entry_price, exit_price, action, quantity, lot_size)
        
        elif strategy_type in ['CALL', 'PUT']:
            strike_price = suggestion.get('strike_price')
            if not strike_price:
                logger.warning(f"No strike_price for {strategy_type} suggestion {suggestion.get('id')}")
                return 0.0
            
            # Get premium from total_premium
            premium = suggestion.get('total_premium') or 0.0
            if premium > 0:
                # Premium is total, need per share
                premium_per_share = premium / (quantity * lot_size)
            else:
                premium_per_share = 0.0
            
            return calculate_options_profit(
                entry_price, exit_price, strike_price,
                premium_per_share, strategy_type, quantity, lot_size
            )
        
        elif strategy_type == 'STRADDLE':
            strike_price = suggestion.get('strike_price')
            if not strike_price:
                logger.warning(f"No strike_price for STRADDLE suggestion {suggestion.get('id')}")
                return 0.0
            
            # For straddle, we need call and put premiums
            # If we only have total_premium, split it 50/50
            total_premium = suggestion.get('total_premium') or 0.0
            if total_premium > 0:
                premium_per_share = total_premium / (quantity * lot_size * 2)  # Divided by 2 for call+put
                call_premium = premium_per_share
                put_premium = premium_per_share
            else:
                call_premium = 0.0
                put_premium = 0.0
            
            result = calculate_straddle_profit(
                entry_price, exit_price, strike_price,
                call_premium, put_premium, quantity, lot_size
            )
            
            return result.get('total_profit', 0.0)
        
        else:
            logger.warning(f"Unknown strategy type: {strategy_type}")
            return 0.0

