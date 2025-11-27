"""
Derivative Suggestions Monitor Service
Automatically monitors MOCKED suggestions for exit conditions:
- Target levels reached
- Stop loss hit
- Conflicting signals detected
Then calculates and updates P&L
"""

import logging
import json
from datetime import datetime, date
from typing import List, Dict, Optional, Tuple
from common.Boilerplate import get_db_connection
import psycopg2.extras
from api.services.theoretical_pnl_calculator import TheoreticalPnlCalculator

logger = logging.getLogger(__name__)


class DerivativeSuggestionsMonitor:
    """Service for automatically monitoring and exiting MOCKED derivative suggestions"""
    
    def __init__(self):
        self.pnl_calculator = TheoreticalPnlCalculator()
    
    def monitor_and_exit_suggestions(self) -> Dict:
        """
        Monitor all MOCKED suggestions and exit them if exit conditions are met
        
        Returns:
            Dictionary with statistics about monitoring results
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Fetch all MOCKED suggestions that haven't been exited
            query = """
                SELECT id, entry_price, strike_price, quantity, lot_size, strategy_type,
                       direction, instrument_token, instrument, tpo_context, diagnostics,
                       total_premium, generated_at, analysis_date
                FROM my_schema.derivative_suggestions
                WHERE status = 'MOCKED'
                  AND exit_date IS NULL
                  AND exit_price IS NULL
                ORDER BY generated_at DESC
            """
            
            cursor.execute(query)
            suggestions = cursor.fetchall()
            
            stats = {
                'total_checked': len(suggestions),
                'target_exits': 0,
                'stop_loss_exits': 0,
                'conflicting_signal_exits': 0,
                'current_price_exits': 0,
                'no_exit_needed': 0,
                'errors': 0
            }
            
            logger.info(f"Monitoring {stats['total_checked']} MOCKED suggestions for exit conditions")
            
            for suggestion in suggestions:
                try:
                    exit_result = self._check_exit_conditions(suggestion, cursor)
                    
                    if exit_result['should_exit']:
                        # Calculate P&L and update the suggestion
                        self._exit_suggestion(suggestion, exit_result, cursor)
                        
                        # Update statistics
                        exit_reason = exit_result['exit_reason']
                        if 'target' in exit_reason.lower():
                            stats['target_exits'] += 1
                        elif 'stop' in exit_reason.lower() or 'loss' in exit_reason.lower():
                            stats['stop_loss_exits'] += 1
                        elif 'conflict' in exit_reason.lower():
                            stats['conflicting_signal_exits'] += 1
                        else:
                            stats['current_price_exits'] += 1
                    else:
                        stats['no_exit_needed'] += 1
                        
                except Exception as e:
                    logger.error(f"Error monitoring suggestion {suggestion['id']}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    stats['errors'] += 1
                    continue
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"Monitoring complete: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error in monitor_and_exit_suggestions: {e}")
            import traceback
            logger.error(traceback.format_exc())
            if 'conn' in locals():
                try:
                    conn.rollback()
                    conn.close()
                except:
                    pass
            return {'error': str(e)}
    
    def _check_exit_conditions(self, suggestion: Dict, cursor) -> Dict:
        """
        Check if exit conditions are met for a suggestion
        
        Args:
            suggestion: Suggestion dictionary from database
            cursor: Database cursor
            
        Returns:
            Dictionary with should_exit (bool), exit_price (float), exit_reason (str)
        """
        suggestion_id = suggestion['id']
        instrument_token = suggestion.get('instrument_token')
        instrument = suggestion.get('instrument')
        direction = suggestion.get('direction', '')
        
        # Get current market price
        current_price = self._get_current_market_price(instrument_token, cursor)
        if not current_price:
            logger.warning(f"Could not get current price for suggestion {suggestion_id}, skipping")
            return {'should_exit': False, 'exit_price': None, 'exit_reason': 'No current price available'}
        
        # Extract entry price
        entry_price = suggestion.get('entry_price')
        if not entry_price:
            # Try to extract from diagnostics or tpo_context
            entry_price = self._extract_entry_price(suggestion)
        
        if not entry_price:
            logger.warning(f"Could not determine entry price for suggestion {suggestion_id}, skipping")
            return {'should_exit': False, 'exit_price': None, 'exit_reason': 'No entry price available'}
        
        # Extract target levels and stop loss
        target_levels, stop_loss = self._extract_targets_and_stop_loss(suggestion)
        
        # Check 1: Conflicting signals (opposite direction for same instrument)
        conflicting = self._check_conflicting_signals(suggestion, cursor)
        if conflicting:
            logger.info(f"Suggestion {suggestion_id} has conflicting signals, exiting")
            return {
                'should_exit': True,
                'exit_price': current_price,
                'exit_reason': f'Conflicting signal detected: {conflicting}'
            }
        
        # Check 2: Target reached
        if target_levels:
            reached_targets = []
            for target in target_levels:
                if isinstance(target, dict):
                    target_value = target.get('level') or target.get('price')
                else:
                    target_value = target
                
                if target_value:
                    try:
                        target_float = float(target_value)
                        # For bullish: check if price >= target, for bearish: check if price <= target
                        if direction.upper() in ['BULLISH', 'BUY']:
                            if current_price >= target_float:
                                reached_targets.append(target_float)
                        else:  # Bearish/SELL
                            if current_price <= target_float:
                                reached_targets.append(target_float)
                    except (ValueError, TypeError):
                        continue
            
            if reached_targets:
                # Use highest target reached for bullish, lowest for bearish
                if direction.upper() in ['BULLISH', 'BUY']:
                    exit_price = max(reached_targets)
                else:
                    exit_price = min(reached_targets)
                
                logger.info(f"Suggestion {suggestion_id} target reached at {exit_price}")
                return {
                    'should_exit': True,
                    'exit_price': exit_price,
                    'exit_reason': f'Target level reached: {exit_price}'
                }
        
        # Check 3: Stop loss hit
        if stop_loss:
            try:
                stop_loss_float = float(stop_loss)
                # For bullish: exit if price <= stop loss, for bearish: exit if price >= stop loss
                if direction.upper() in ['BULLISH', 'BUY']:
                    if current_price <= stop_loss_float:
                        logger.info(f"Suggestion {suggestion_id} stop loss hit at {stop_loss_float}")
                        return {
                            'should_exit': True,
                            'exit_price': stop_loss_float,
                            'exit_reason': f'Stop loss hit: {stop_loss_float}'
                        }
                else:  # Bearish/SELL
                    if current_price >= stop_loss_float:
                        logger.info(f"Suggestion {suggestion_id} stop loss hit at {stop_loss_float}")
                        return {
                            'should_exit': True,
                            'exit_price': stop_loss_float,
                            'exit_reason': f'Stop loss hit: {stop_loss_float}'
                        }
            except (ValueError, TypeError):
                pass
        
        # No exit condition met - use current price for calculation (optional)
        # For now, we'll only exit when conditions are met
        return {
            'should_exit': False,
            'exit_price': current_price,
            'exit_reason': 'No exit condition met'
        }
    
    def _get_current_market_price(self, instrument_token: Optional[int], cursor) -> Optional[float]:
        """Get current market price from ticks table"""
        if not instrument_token:
            return None
        
        try:
            query = """
                SELECT last_price
                FROM my_schema.ticks
                WHERE instrument_token = %s
                ORDER BY timestamp DESC
                LIMIT 1
            """
            cursor.execute(query, (instrument_token,))
            result = cursor.fetchone()
            if result and result.get('last_price'):
                return float(result['last_price'])
        except Exception as e:
            logger.warning(f"Error fetching current price for instrument_token {instrument_token}: {e}")
        
        return None
    
    def _extract_entry_price(self, suggestion: Dict) -> Optional[float]:
        """Extract entry price from various fields"""
        entry_price = suggestion.get('entry_price')
        if entry_price:
            try:
                return float(entry_price)
            except (ValueError, TypeError):
                pass
        
        # Try to extract from diagnostics or tpo_context
        diagnostics = suggestion.get('diagnostics')
        if isinstance(diagnostics, str):
            try:
                diagnostics = json.loads(diagnostics)
            except:
                diagnostics = {}
        
        if isinstance(diagnostics, dict):
            decision_context = diagnostics.get('decision_context', {})
            if isinstance(decision_context, dict):
                key_levels = decision_context.get('key_levels', {})
                if isinstance(key_levels, dict):
                    entry = key_levels.get('entry_level') or key_levels.get('entry_price')
                    if entry:
                        try:
                            return float(entry)
                        except (ValueError, TypeError):
                            pass
        
        return None
    
    def _extract_targets_and_stop_loss(self, suggestion: Dict) -> Tuple[List, Optional[float]]:
        """Extract target levels and stop loss from suggestion"""
        target_levels = []
        stop_loss = None
        
        # Try to extract from diagnostics
        diagnostics = suggestion.get('diagnostics')
        if isinstance(diagnostics, str):
            try:
                diagnostics = json.loads(diagnostics)
            except:
                diagnostics = {}
        
        if isinstance(diagnostics, dict):
            decision_context = diagnostics.get('decision_context', {})
            if isinstance(decision_context, dict):
                key_levels = decision_context.get('key_levels', {})
                if isinstance(key_levels, dict):
                    # Extract targets
                    targets = key_levels.get('target_levels') or key_levels.get('targets') or []
                    if isinstance(targets, list):
                        target_levels = targets
                    elif targets:
                        target_levels = [targets]
                    
                    # Extract stop loss
                    stop_loss = key_levels.get('stop_loss') or key_levels.get('stop_loss_level')
        
        # Try to extract from tpo_context
        tpo_context = suggestion.get('tpo_context')
        if isinstance(tpo_context, str):
            try:
                tpo_context = json.loads(tpo_context)
            except:
                tpo_context = {}
        
        if isinstance(tpo_context, dict) and not target_levels:
            target_levels = tpo_context.get('target_levels') or tpo_context.get('targets') or []
            if not isinstance(target_levels, list):
                target_levels = [target_levels] if target_levels else []
            stop_loss = stop_loss or tpo_context.get('stop_loss')
        
        return target_levels, stop_loss
    
    def _check_conflicting_signals(self, suggestion: Dict, cursor) -> Optional[str]:
        """
        Check if there are conflicting signals (opposite direction for same instrument)
        
        Returns:
            String describing the conflict, or None if no conflict
        """
        suggestion_id = suggestion['id']
        instrument = suggestion.get('instrument')
        direction = suggestion.get('direction', '').upper()
        instrument_token = suggestion.get('instrument_token')
        
        if not instrument and not instrument_token:
            return None
        
        # Determine opposite direction
        opposite_direction = None
        if direction in ['BULLISH', 'BUY']:
            opposite_direction = ['BEARISH', 'SELL']
        elif direction in ['BEARISH', 'SELL']:
            opposite_direction = ['BULLISH', 'BUY']
        else:
            return None
        
        # Check for conflicting suggestions
        where_clauses = ["status = 'MOCKED'", "id != %s"]
        params = [suggestion_id]
        
        if instrument_token:
            where_clauses.append("instrument_token = %s")
            params.append(instrument_token)
        elif instrument:
            where_clauses.append("instrument = %s")
            params.append(instrument)
        else:
            return None
        
        # Check for opposite direction
        direction_conditions = []
        for opp_dir in opposite_direction:
            direction_conditions.append(f"UPPER(direction) = '{opp_dir}'")
        
        if direction_conditions:
            where_clauses.append(f"({' OR '.join(direction_conditions)})")
        
        query = f"""
            SELECT id, direction, generated_at
            FROM my_schema.derivative_suggestions
            WHERE {' AND '.join(where_clauses)}
            ORDER BY generated_at DESC
            LIMIT 1
        """
        
        cursor.execute(query, tuple(params))
        conflicting = cursor.fetchone()
        
        if conflicting:
            return f"Opposite direction suggestion {conflicting['id']} ({conflicting['direction']}) found"
        
        return None
    
    def _exit_suggestion(self, suggestion: Dict, exit_result: Dict, cursor):
        """
        Exit a suggestion by calculating P&L and updating the record
        """
        suggestion_id = suggestion['id']
        exit_price = exit_result['exit_price']
        exit_reason = exit_result['exit_reason']
        
        # Calculate P&L using the theoretical P&L calculator
        try:
            # Get entry price
            entry_price = suggestion.get('entry_price') or self._extract_entry_price(suggestion)
            if not entry_price:
                logger.warning(f"Cannot calculate P&L for suggestion {suggestion_id}: no entry price")
                return
            
            # Calculate P&L
            pnl_result = self.pnl_calculator._calculate_pnl(
                strategy_type=suggestion.get('strategy_type', ''),
                suggestion=suggestion,
                entry_price=float(entry_price),
                exit_price=float(exit_price)
            )
            
            # Handle different return types (float for most, dict for STRADDLE)
            if isinstance(pnl_result, dict):
                actual_pnl = pnl_result.get('total_profit', 0.0)
            else:
                actual_pnl = float(pnl_result) if pnl_result else 0.0
            
            # Determine outcome
            if actual_pnl > 0:
                outcome = 'SUCCESS'
            elif actual_pnl < 0:
                outcome = 'FAILURE'
            else:
                outcome = 'BREAKEVEN'
            
            # Update the suggestion
            update_query = """
                UPDATE my_schema.derivative_suggestions
                SET exit_price = %s,
                    exit_date = CURRENT_DATE,
                    actual_pnl = %s,
                    outcome = %s,
                    notes = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """
            
            notes = f"Theoretical exit: {exit_reason}. Entry: {entry_price}, Exit: {exit_price}"
            
            cursor.execute(update_query, (
                exit_price,
                actual_pnl,
                outcome,
                notes,
                suggestion_id
            ))
            
            logger.info(f"Exited suggestion {suggestion_id}: {exit_reason}, P&L: {actual_pnl}, Outcome: {outcome}")
            
        except Exception as e:
            logger.error(f"Error exiting suggestion {suggestion_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())


def monitor_derivative_suggestions():
    """
    Standalone function to run monitoring (for use in cron jobs)
    """
    monitor = DerivativeSuggestionsMonitor()
    result = monitor.monitor_and_exit_suggestions()
    return result

