"""
Advanced Hedging Strategies Module
Provides advanced hedging strategy suggestions and calculations
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
from common.Boilerplate import get_db_connection

class AdvancedHedgingStrategies:
    """
    Advanced hedging strategy suggestions and calculations
    """

    def __init__(self):
        """Initialize Advanced Hedging Strategies"""
        self.risk_free_rate = 0.065

    def get_all_advanced_strategies(self, portfolio_value: float, portfolio_delta: float = 0.0) -> Dict:
        """
        Get all advanced hedging strategy suggestions

        Args:
            portfolio_value: Current portfolio value
            portfolio_delta: Portfolio delta (beta)

        Returns:
            Dictionary with all strategies
        """
        try:
            strategies = {
                'delta_hedging': self._calculate_delta_hedging(portfolio_value, portfolio_delta),
                'volatility_hedging': self._calculate_volatility_hedging(portfolio_value),
                'tail_risk_hedging': self._calculate_tail_risk_hedging(portfolio_value),
                'correlation_hedging': self._calculate_correlation_hedging(portfolio_value)
            }

            return strategies
        except Exception as e:
            logging.error(f"Error getting advanced strategies: {e}")
            return {}

    def _calculate_delta_hedging(self, portfolio_value: float, portfolio_delta: float) -> Dict:
        """
        Calculate delta hedging strategy with actual futures contracts

        Args:
            portfolio_value: Portfolio value
            portfolio_delta: Portfolio delta

        Returns:
            Delta hedging strategy with specific futures contract recommendations
        """
        try:
            # Fetch actual futures contracts from database
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get latest futures prices for available contracts
            # Common Nifty futures: NIFTY25OCTFUT, NIFTY25NOVFUT, NIFTY25DECFUT
            cursor.execute("""
                SELECT DISTINCT ON (instrument_token)
                    instrument_token,
                    last_price,
                    timestamp,
                    run_date
                FROM my_schema.futures_ticks
                WHERE run_date = (SELECT MAX(run_date) FROM my_schema.futures_ticks)
                ORDER BY instrument_token, timestamp DESC
            """)
            
            futures_data = cursor.fetchall()
            cursor.close()
            conn.close()
            
            # Get Nifty spot price for fallback
            nifty_spot_price = 22000  # Default fallback
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT last_price
                    FROM my_schema.ticks
                    WHERE instrument_token = 256265
                    ORDER BY timestamp DESC
                    LIMIT 1
                """)
                spot_row = cursor.fetchone()
                if spot_row and spot_row[0]:
                    nifty_spot_price = float(spot_row[0])
                cursor.close()
                conn.close()
            except Exception as e:
                logging.warning(f"Could not fetch Nifty spot price: {e}")
            
            nifty_futures_lot = 50  # Nifty futures lot size
            
            # Find the nearest month futures contract (prefer current month, then next month)
            selected_futures = None
            selected_price = nifty_spot_price  # Fallback to spot
            
            if futures_data:
                # Sort by timestamp (most recent first) and pick the first one
                # In practice, you might want to pick based on expiry date, but for now use most recent
                selected_futures = futures_data[0]
                selected_price = float(selected_futures[1]) if selected_futures[1] else nifty_spot_price
                
                # Try to get instrument name from master_scrips or Kite API
                futures_symbol = None
                try:
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    # Try scrip_id first (more reliable than scrip_name)
                    cursor.execute("""
                        SELECT scrip_id, scrip_name
                        FROM my_schema.master_scrips
                        WHERE instrument_token = %s
                        LIMIT 1
                    """, (selected_futures[0],))
                    scrip_row = cursor.fetchone()
                    if scrip_row:
                        # Prefer scrip_id over scrip_name
                        futures_symbol = scrip_row[0] if scrip_row[0] else (scrip_row[1] if scrip_row[1] else None)
                    cursor.close()
                    conn.close()
                except Exception as e:
                    logging.debug(f"Could not get futures symbol from master_scrips: {e}")
                
                # If not found in database, try Kite API
                if not futures_symbol:
                    try:
                        from common.Boilerplate import kite
                        instruments = kite.instruments('NFO')
                        for inst in instruments:
                            if inst.get('instrument_token') == selected_futures[0]:
                                futures_symbol = inst.get('tradingsymbol') or inst.get('name')
                                break
                    except Exception as e:
                        logging.debug(f"Could not get futures symbol from Kite API: {e}")
                
                # Final fallback
                if not futures_symbol:
                    futures_symbol = f"NIFTY FUT"
            else:
                futures_symbol = "NIFTY FUT"
            
            if portfolio_delta > 1.0:
                # Need to hedge positive delta (sell futures)
                hedge_ratio = portfolio_delta - 1.0
                futures_contracts = int((portfolio_value * hedge_ratio) / (selected_price * nifty_futures_lot))
                
                if futures_contracts <= 0:
                    return {
                        'strategy_type': 'delta_hedging',
                        'action': 'no_action',
                        'rationale': f"Portfolio delta ({portfolio_delta:.2f}) exceeds 1.0, but hedge amount is too small to execute",
                        'expected_impact': "No change needed"
                    }

                # Calculate margin requirement
                contract_value = selected_price * nifty_futures_lot * futures_contracts
                margin_required = contract_value * 0.12  # ~12% margin for futures

                return {
                    'strategy_type': 'delta_hedging',
                    'strategy_name': 'Delta Hedge - Sell Futures',
                    'action': 'sell_futures',
                    'contracts': futures_contracts,
                    'underlying': 'NIFTY',
                    'instrument': futures_symbol,
                    'futures_price': selected_price,
                    'contract_value': contract_value,
                    'margin_required': margin_required,
                    'rationale': f"Portfolio delta ({portfolio_delta:.2f}) exceeds 1.0. Sell {futures_contracts} lots of {futures_symbol} at ₹{selected_price:,.2f} per unit to hedge.",
                    'expected_impact': f"Reduce portfolio delta to ~1.0. Contract value: ₹{contract_value:,.0f}, Margin required: ₹{margin_required:,.0f}"
                }
            elif portfolio_delta < 1.0:
                # Need to hedge negative delta (buy futures)
                hedge_ratio = 1.0 - portfolio_delta
                futures_contracts = int((portfolio_value * hedge_ratio) / (selected_price * nifty_futures_lot))
                
                if futures_contracts <= 0:
                    return {
                        'strategy_type': 'delta_hedging',
                        'action': 'no_action',
                        'rationale': f"Portfolio delta ({portfolio_delta:.2f}) below 1.0, but hedge amount is too small to execute",
                        'expected_impact': "No change needed"
                    }

                # Calculate margin requirement
                contract_value = selected_price * nifty_futures_lot * futures_contracts
                margin_required = contract_value * 0.12  # ~12% margin for futures

                return {
                    'strategy_type': 'delta_hedging',
                    'strategy_name': 'Delta Hedge - Buy Futures',
                    'action': 'buy_futures',
                    'contracts': futures_contracts,
                    'underlying': 'NIFTY',
                    'instrument': futures_symbol,
                    'futures_price': selected_price,
                    'contract_value': contract_value,
                    'margin_required': margin_required,
                    'rationale': f"Portfolio delta ({portfolio_delta:.2f}) below 1.0. Buy {futures_contracts} lots of {futures_symbol} at ₹{selected_price:,.2f} per unit to hedge.",
                    'expected_impact': f"Increase portfolio delta to ~1.0. Contract value: ₹{contract_value:,.0f}, Margin required: ₹{margin_required:,.0f}"
                }
            else:
                return {
                    'strategy_type': 'delta_hedging',
                    'action': 'no_action',
                    'rationale': f"Portfolio delta ({portfolio_delta:.2f}) is already neutral",
                    'expected_impact': "No change needed"
                }
        except Exception as e:
            logging.error(f"Error calculating delta hedging: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return {'error': str(e)}

    def _calculate_volatility_hedging(self, portfolio_value: float) -> Dict:
        """
        Calculate volatility hedging strategy

        Args:
            portfolio_value: Portfolio value

        Returns:
            Volatility hedging strategy
        """
        try:
            # Suggest VIX-based hedging
            vix_price = 18  # Approximate VIX price
            india_vix_contracts = int(portfolio_value * 0.05 / (vix_price * 1000))  # 5% allocation

            return {
                'strategy_type': 'volatility_hedging',
                'action': 'buy_options',
                'instrument': 'INDIA VIX',
                'contracts': india_vix_contracts,
                'option_type': 'call',
                'strike': 'ATM',
                'rationale': f"Buy {india_vix_contracts} India VIX calls to hedge against volatility spikes",
                'expected_impact': "Protect against market volatility increases"
            }
        except Exception as e:
            logging.error(f"Error calculating volatility hedging: {e}")
            return {'error': str(e)}

    def _calculate_tail_risk_hedging(self, portfolio_value: float) -> Dict:
        """
        Calculate tail risk hedging strategy

        Args:
            portfolio_value: Portfolio value

        Returns:
            Tail risk hedging strategy
        """
        try:
            # Get current Nifty price from database
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT last_price
                FROM my_schema.ticks
                WHERE instrument_token = 256265
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            spot_row = cursor.fetchone()
            cursor.close()
            conn.close()
            
            nifty_price = float(spot_row[0]) if spot_row and spot_row[0] else 22000  # Fallback price
            
            # Calculate tail risk hedge
            # Use 2-3% of portfolio value for tail risk protection
            hedge_allocation_pct = 0.025  # 2.5% of portfolio
            hedge_value = portfolio_value * hedge_allocation_pct
            
            # 15% out-of-the-money put strike
            put_strike = int(nifty_price * 0.85)
            
            # Estimate put premium (roughly 1-2% of strike for 15% OTM puts)
            # This is a rough estimate - in practice, you'd fetch from options chain
            estimated_put_premium = put_strike * 0.015  # ~1.5% of strike
            
            # Calculate number of lots needed
            # Each lot = 50 shares, so contract value = premium * 50
            contract_value = estimated_put_premium * 50
            contracts = int(hedge_value / contract_value) if contract_value > 0 else 0
            
            # Ensure minimum of 1 contract if portfolio is large enough
            if contracts == 0 and hedge_value > 5000:  # If hedge value > ₹5000, suggest at least 1 contract
                contracts = 1
            
            if contracts <= 0:
                return {
                    'strategy_type': 'tail_risk_hedging',
                    'action': 'no_action',
                    'rationale': f"Portfolio value too small for effective tail risk hedging. Minimum hedge value required: ₹{contract_value:,.0f}",
                    'expected_impact': "No change needed"
                }
            
            total_premium = contracts * contract_value
            
            return {
                'strategy_type': 'tail_risk_hedging',
                'strategy_name': 'Tail Risk Protection - Put Options',
                'action': 'buy_puts',
                'underlying': 'NIFTY',
                'contracts': contracts,
                'strike': put_strike,
                'option_type': 'PUT',
                'estimated_premium_per_lot': estimated_put_premium * 50,
                'total_premium': total_premium,
                'rationale': f"Buy {contracts} lots of Nifty {put_strike} PUT options (15% OTM) to protect against severe market downturns. Estimated premium: ₹{total_premium:,.0f} ({hedge_allocation_pct*100:.1f}% of portfolio).",
                'expected_impact': f"Limit downside risk in extreme market conditions. Protection for ~15% decline below {put_strike}."
            }
        except Exception as e:
            logging.error(f"Error calculating tail risk hedging: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return {'error': str(e)}

    def _calculate_correlation_hedging(self, portfolio_value: float) -> Dict:
        """
        Calculate correlation hedging strategy, taking into account existing gold and silver holdings

        Args:
            portfolio_value: Portfolio value

        Returns:
            Correlation hedging strategy (may include multiple recommendations)
        """
        try:
            # Target allocation: 4% of portfolio in precious metals (gold + silver)
            target_allocation_pct = 0.04
            target_value = portfolio_value * target_allocation_pct
            
            # Get existing gold and silver holdings
            conn = get_db_connection()
            cursor = conn.cursor()
            
            existing_gold_units = 0
            existing_gold_value = 0.0
            existing_silver_units = 0
            existing_silver_value = 0.0
            gold_current_price = 0.0
            silver_current_price = 0.0
            
            # Fetch existing SETFGOLD holdings
            cursor.execute("""
                SELECT 
                    h.quantity,
                    COALESCE(rt.price_close, h.last_price, h.close_price, 0) as current_price,
                    (h.quantity * COALESCE(rt.price_close, h.last_price, h.close_price, 0)) as current_value
                FROM my_schema.holdings h
                LEFT JOIN (
                    SELECT DISTINCT ON (scrip_id) scrip_id, price_close, price_date
                    FROM my_schema.rt_intraday_price
                    WHERE price_date::date <= CURRENT_DATE
                    ORDER BY scrip_id, price_date DESC
                ) rt ON h.trading_symbol = rt.scrip_id
                WHERE h.trading_symbol = 'SETFGOLD'
                AND h.run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
                AND h.quantity > 0
            """)
            gold_row = cursor.fetchone()
            if gold_row:
                existing_gold_units = gold_row[0] or 0
                gold_current_price = gold_row[1] or 0.0
                existing_gold_value = gold_row[2] or 0.0
            
            # Fetch existing SILVERIETF holdings
            cursor.execute("""
                SELECT 
                    h.quantity,
                    COALESCE(rt.price_close, h.last_price, h.close_price, 0) as current_price,
                    (h.quantity * COALESCE(rt.price_close, h.last_price, h.close_price, 0)) as current_value
                FROM my_schema.holdings h
                LEFT JOIN (
                    SELECT DISTINCT ON (scrip_id) scrip_id, price_close, price_date
                    FROM my_schema.rt_intraday_price
                    WHERE price_date::date <= CURRENT_DATE
                    ORDER BY scrip_id, price_date DESC
                ) rt ON h.trading_symbol = rt.scrip_id
                WHERE h.trading_symbol = 'SILVERIETF'
                AND h.run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
                AND h.quantity > 0
            """)
            silver_row = cursor.fetchone()
            if silver_row:
                existing_silver_units = silver_row[0] or 0
                silver_current_price = silver_row[1] or 0.0
                existing_silver_value = silver_row[2] or 0.0
            
            cursor.close()
            conn.close()
            
            # Calculate total existing precious metals value
            total_existing_value = existing_gold_value + existing_silver_value
            remaining_target = target_value - total_existing_value
            
            # Get current prices for recommendations (use fetched prices or fallback)
            gold_price = gold_current_price if gold_row and gold_current_price > 0 else 6000  # Approximate gold price per unit
            silver_price = silver_current_price if silver_row and silver_current_price > 0 else 80  # Approximate silver price per unit
            
            strategies = []
            
            # If we already have enough precious metals, suggest maintaining or slight adjustment
            if total_existing_value >= target_value * 0.95:  # Within 5% of target
                strategies.append({
                    'strategy_type': 'correlation_hedging',
                    'strategy_name': 'Maintain Precious Metals Allocation',
                    'action': 'maintain',
                    'instrument': 'SETFGOLD/SILVERIETF',
                    'rationale': f"Current precious metals allocation ({total_existing_value/portfolio_value*100:.2f}%) is near target ({target_allocation_pct*100}%). Maintain current positions.",
                    'expected_impact': "Maintain correlation hedge with equity markets",
                    'existing_gold_units': int(existing_gold_units) if existing_gold_units else 0,
                    'existing_silver_units': int(existing_silver_units) if existing_silver_units else 0,
                    'existing_gold_value': float(existing_gold_value) if existing_gold_value else 0.0,
                    'existing_silver_value': float(existing_silver_value) if existing_silver_value else 0.0,
                    'current_allocation_pct': float((total_existing_value / portfolio_value * 100)) if portfolio_value > 0 else 0.0,
                    'target_allocation_pct': float(target_allocation_pct * 100)
                })
            elif remaining_target > portfolio_value * 0.01:  # Need to add at least 1% more
                # Suggest adding gold (preferred for correlation hedging)
                gold_needed_value = remaining_target * 0.7  # 70% of remaining in gold
                gold_units_needed = int(gold_needed_value / gold_price) if gold_price > 0 else 0
                
                if gold_units_needed > 0:
                    strategies.append({
                        'strategy_type': 'correlation_hedging',
                        'strategy_name': 'Add Gold ETF',
                        'action': 'buy_gold',
                        'instrument': 'SETFGOLD',
                        'quantity': gold_units_needed,
                        'rationale': f"Add {gold_units_needed} units of SETFGOLD. Current: {existing_gold_units} units (₹{existing_gold_value:,.0f}). Target: {target_allocation_pct*100}% allocation.",
                        'expected_impact': "Increase correlation hedge against equity markets",
                        'existing_gold_units': int(existing_gold_units) if existing_gold_units else 0,
                        'existing_gold_value': float(existing_gold_value) if existing_gold_value else 0.0,
                        'target_allocation_pct': float(target_allocation_pct * 100)
                    })
                
                # Suggest adding silver (30% of remaining)
                silver_needed_value = remaining_target * 0.3  # 30% of remaining in silver
                silver_units_needed = int(silver_needed_value / silver_price) if silver_price > 0 else 0
                
                if silver_units_needed > 0:
                    strategies.append({
                        'strategy_type': 'correlation_hedging',
                        'strategy_name': 'Add Silver ETF',
                        'action': 'buy_silver',
                        'instrument': 'SILVERIETF',
                        'quantity': silver_units_needed,
                        'rationale': f"Add {silver_units_needed} units of SILVERIETF. Current: {existing_silver_units} units (₹{existing_silver_value:,.0f}). Target: {target_allocation_pct*100}% allocation.",
                        'expected_impact': "Diversify correlation hedge with silver exposure",
                        'existing_silver_units': int(existing_silver_units) if existing_silver_units else 0,
                        'existing_silver_value': float(existing_silver_value) if existing_silver_value else 0.0,
                        'target_allocation_pct': float(target_allocation_pct * 100)
                    })
            else:
                # Very close to target, suggest maintaining
                strategies.append({
                    'strategy_type': 'correlation_hedging',
                    'strategy_name': 'Maintain Precious Metals',
                    'action': 'maintain',
                    'instrument': 'SETFGOLD/SILVERIETF',
                    'rationale': f"Current allocation ({total_existing_value/portfolio_value*100:.2f}%) is close to target ({target_allocation_pct*100}%). Gold: {existing_gold_units} units, Silver: {existing_silver_units} units.",
                    'expected_impact': "Maintain correlation hedge",
                    'existing_gold_units': int(existing_gold_units) if existing_gold_units else 0,
                    'existing_silver_units': int(existing_silver_units) if existing_silver_units else 0,
                    'existing_gold_value': float(existing_gold_value) if existing_gold_value else 0.0,
                    'existing_silver_value': float(existing_silver_value) if existing_silver_value else 0.0,
                    'current_allocation_pct': float((total_existing_value / portfolio_value * 100)) if portfolio_value > 0 else 0.0,
                    'target_allocation_pct': float(target_allocation_pct * 100)
                })
            
            # Return single strategy dict if only one, or list if multiple
            if len(strategies) == 1:
                return strategies[0]
            elif len(strategies) > 1:
                # Return as a list of strategies
                return {
                    'strategy_type': 'correlation_hedging',
                    'strategies': strategies,
                    'summary': f"Current precious metals: Gold {existing_gold_units} units (₹{existing_gold_value:,.0f}), Silver {existing_silver_units} units (₹{existing_silver_value:,.0f}). Total: {total_existing_value/portfolio_value*100:.2f}% of portfolio."
                }
            else:
                # Fallback if no strategies generated
                return {
                    'strategy_type': 'correlation_hedging',
                    'action': 'no_action',
                    'rationale': "Unable to calculate correlation hedging strategy",
                    'expected_impact': "No change"
                }
                
        except Exception as e:
            logging.error(f"Error calculating correlation hedging: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return {'error': str(e)}
