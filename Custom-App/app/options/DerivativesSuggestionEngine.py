"""
Derivatives Trade Suggestion Engine
Analyzes TPO levels, order flow, market bias, options data, and tick data to suggest futures and options trades
"""

import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
from options.DerivativesTPOAnalyzer import DerivativesTPOAnalyzer
from market.OrderFlowAnalyzer import OrderFlowAnalyzer
from options.OptionsDataFetcher import OptionsDataFetcher
from market.CalculateTPO import PostgresDataFetcher
from common.Boilerplate import get_db_connection
import pandas as pd
import numpy as np

def calculate_futures_margin(entry_price: float, lot_size: int = 50) -> Dict:
    """
    Calculate margin requirement for Nifty futures
    
    Args:
        entry_price: Entry price for futures
        lot_size: Lot size (default: 50 for Nifty)
        
    Returns:
        Dictionary with margin breakdown
    """
    # Contract value = Price * Lot Size * 1 (multiplier for Nifty futures)
    contract_value = entry_price * lot_size
    
    # Nifty futures margin is typically around 10-15% of contract value
    # SPAN margin ~8-10%, Exposure margin ~3-5%
    # Using conservative estimate of 12% for total margin
    margin_percentage = 0.12
    total_margin = contract_value * margin_percentage
    
    return {
        'contract_value': contract_value,
        'span_margin': contract_value * 0.085,  # ~8.5% SPAN
        'exposure_margin': contract_value * 0.035,  # ~3.5% Exposure
        'total_margin': total_margin,
        'margin_percentage': margin_percentage * 100
    }

def calculate_options_margin(strike_price: float, premium: float, lot_size: int = 50, is_long: bool = True) -> Dict:
    """
    Calculate margin/premium requirement for options
    
    Args:
        strike_price: Strike price of the option
        premium: Premium per share
        lot_size: Lot size (default: 50 for Nifty)
        is_long: Whether buying (True) or selling (False) the option
        
        Returns:
        Dictionary with margin/premium breakdown
    """
    premium_per_lot = premium * lot_size
    
    if is_long:
        # For long options (buying), margin is just the premium paid
        return {
            'premium_per_lot': premium_per_lot,
            'total_required': premium_per_lot,
            'margin_type': 'Premium'
        }
    else:
        # For short options (selling), margin is much higher (SPAN + Exposure)
        contract_value = strike_price * lot_size
        margin_percentage = 0.15  # Higher margin for short positions
        total_margin = max(contract_value * margin_percentage, premium_per_lot * 3)
        return {
            'premium_received': premium_per_lot,
            'span_margin': contract_value * 0.10,
            'exposure_margin': contract_value * 0.05,
            'total_required': total_margin,
            'margin_type': 'SPAN + Exposure'
        }

def calculate_futures_profit(entry_price: float, exit_price: float, action: str, num_lots: int, lot_size: int = 50) -> float:
    """
    Calculate potential profit for futures trade
    
    Args:
        entry_price: Entry price
        exit_price: Exit price (target level)
        action: 'BUY' or 'SELL'
        num_lots: Number of lots
        lot_size: Lot size (default: 50 for Nifty)
        
    Returns:
        Potential profit in rupees
    """
    if action == 'BUY':
        # Long futures: profit = (exit - entry) * lot_size * num_lots
        profit = (exit_price - entry_price) * lot_size * num_lots
    elif action == 'SELL':
        # Short futures: profit = (entry - exit) * lot_size * num_lots
        profit = (entry_price - exit_price) * lot_size * num_lots
    else:
        return 0.0
    return round(profit, 2)

def calculate_options_profit(entry_price: float, exit_price: float, strike_price: float, 
                           premium_paid: float, option_type: str, num_lots: int, lot_size: int = 50) -> float:
    """
    Calculate potential profit for options trade
    
    Args:
        entry_price: Entry price (current market price when buying option)
        exit_price: Exit price (target level)
        strike_price: Strike price of the option
        premium_paid: Premium paid per share
        option_type: 'CALL' or 'PUT'
        num_lots: Number of lots
        lot_size: Lot size (default: 50 for Nifty)
        
    Returns:
        Potential profit in rupees
    """
    if option_type == 'CALL':
        # CALL: intrinsic value at exit = max(0, exit_price - strike_price)
        intrinsic_value = max(0, exit_price - strike_price)
        # Profit = (intrinsic_value - premium_paid) * lot_size * num_lots
        profit = (intrinsic_value - premium_paid) * lot_size * num_lots
    elif option_type == 'PUT':
        # PUT: intrinsic value at exit = max(0, strike_price - exit_price)
        intrinsic_value = max(0, strike_price - exit_price)
        # Profit = (intrinsic_value - premium_paid) * lot_size * num_lots
        profit = (intrinsic_value - premium_paid) * lot_size * num_lots
    else:
        return 0.0
    return round(profit, 2)

def calculate_straddle_profit(entry_price: float, exit_price: float, strike_price: float,
                            call_premium: float, put_premium: float, num_lots: int, lot_size: int = 50) -> Dict:
    """
    Calculate potential profit for straddle (CALL + PUT)
    
    Args:
        entry_price: Entry price (current market price)
        exit_price: Exit price (target level)
        strike_price: Strike price (same for both CALL and PUT)
        call_premium: Premium paid for CALL option per share
        put_premium: Premium paid for PUT option per share
        num_lots: Number of lots (per leg)
        lot_size: Lot size (default: 50 for Nifty)
        
    Returns:
        Dictionary with call_profit, put_profit, and total_profit
    """
    # Calculate CALL profit
    call_intrinsic = max(0, exit_price - strike_price)
    call_profit = (call_intrinsic - call_premium) * lot_size * num_lots
    
    # Calculate PUT profit
    put_intrinsic = max(0, strike_price - exit_price)
    put_profit = (put_intrinsic - put_premium) * lot_size * num_lots
    
    # Total profit is sum of both legs (one will be 0 or negative, other will be positive)
    total_profit = call_profit + put_profit
    
    return {
        'call_profit': round(call_profit, 2),
        'put_profit': round(put_profit, 2),
        'total_profit': round(total_profit, 2)
    }

class DerivativesSuggestionEngine:
    """
    Generates derivatives trading suggestions based on TPO analysis, order flow, market bias, options data, and tick data
    """
    
    def __init__(self, tpo_analyzer: DerivativesTPOAnalyzer, 
                 instrument_token: int = 256265,
                 futures_instrument_token: int = 12683010):
        """
        Initialize suggestion engine
        
        Args:
            tpo_analyzer: DerivativesTPOAnalyzer instance
            instrument_token: Nifty spot instrument token (default: 256265)
            futures_instrument_token: Nifty futures instrument token (default: 12683010)
        """
        self.tpo_analyzer = tpo_analyzer
        self.instrument_token = instrument_token
        self.futures_instrument_token = futures_instrument_token
        
        # Initialize analyzers
        self.order_flow_analyzer = OrderFlowAnalyzer(instrument_token=futures_instrument_token)
        self.options_fetcher = OptionsDataFetcher()
        self.db_fetcher = tpo_analyzer.db_fetcher if hasattr(tpo_analyzer, 'db_fetcher') else None
    
    def generate_suggestions(self, analysis_date: str = None, current_price: float = None, suggestion_type: str = 'tpo') -> List[Dict]:
        """
        Generate derivatives trading suggestions based on TPO analysis or Order flow analysis
        
        Args:
            analysis_date: Date for analysis
            current_price: Current market price (optional, will be fetched if not provided)
            suggestion_type: 'tpo' for TPO-based suggestions, 'orderflow' for Order flow-based suggestions
            
        Returns:
            List of suggestion dictionaries
        """
        if suggestion_type.lower() == 'orderflow':
            return self.generate_order_flow_suggestions(analysis_date, current_price)
        else:
            return self.generate_tpo_suggestions(analysis_date, current_price)
    
    def generate_tpo_suggestions(self, analysis_date: str = None, current_price: float = None) -> List[Dict]:
        """
        Generate derivatives trading suggestions based on TPO analysis
        
        Args:
            analysis_date: Date for analysis
            current_price: Current market price (optional, will be fetched if not provided)
            
        Returns:
            List of suggestion dictionaries
        """
        if not analysis_date:
            analysis_date = datetime.now().strftime('%Y-%m-%d')
        
        suggestions = []
        
        # Get pre-market and live TPO analysis
        pre_market_tpo = self.tpo_analyzer.get_tpo_analysis(analysis_date, 'pre_market')
        live_tpo = self.tpo_analyzer.get_tpo_analysis(analysis_date, 'live')
        
        # If not in database, analyze now
        if not pre_market_tpo:
            pre_market_tpo_dict = self.tpo_analyzer.analyze_pre_market_tpo(analysis_date)
            if pre_market_tpo_dict:
                self.tpo_analyzer.save_tpo_analysis(pre_market_tpo_dict)
                pre_market_tpo = pre_market_tpo_dict
        
        if not live_tpo:
            live_tpo_dict = self.tpo_analyzer.analyze_live_market_tpo(analysis_date)
            if live_tpo_dict:
                self.tpo_analyzer.save_tpo_analysis(live_tpo_dict)
                live_tpo = live_tpo_dict
        
        if not pre_market_tpo and not live_tpo:
            logging.warning("No TPO data available for suggestions")
            return suggestions
        
        # Use live TPO if available, otherwise pre-market
        tpo_data = live_tpo if live_tpo else pre_market_tpo
        
        if isinstance(tpo_data, list):
            tpo_data = tpo_data[0]  # Take first if list
        
        if not tpo_data:
            logging.info("TPO data missing after analysis (both live and pre-market)")
            return suggestions
        
        # Get current price if not provided
        if not current_price and tpo_data.get('poc'):
            current_price = tpo_data.get('poc')
        
        if not current_price:
            logging.warning("Unable to determine current price for suggestions")
            return suggestions
        
        # Debug: log TPO metrics and current price
        logging.info(
            f"TPO metrics for suggestions - POC={tpo_data.get('poc')}, VAH={tpo_data.get('value_area_high')}, VAL={tpo_data.get('value_area_low')}, current_price={current_price}"
        )
        
        # Fetch additional data sources
        order_flow_data = self._fetch_order_flow_data(analysis_date)
        market_bias_data = self._fetch_market_bias_data(analysis_date)
        options_data = self._fetch_options_data(analysis_date, current_price)
        futures_data = self._fetch_futures_data(analysis_date)
        tick_momentum = self._analyze_tick_momentum(analysis_date)
        
        # Combine all signals into a unified context
        market_context = {
            'tpo_data': tpo_data,
            'pre_market_tpo': pre_market_tpo,
            'order_flow': order_flow_data,
            'market_bias': market_bias_data,
            'options_data': options_data,
            'futures_data': futures_data,
            'tick_momentum': tick_momentum,
            'current_price': current_price
        }

        # Generate futures suggestions with enhanced context
        futures_suggestions = self._generate_futures_suggestions_enhanced(market_context)
        suggestions.extend(futures_suggestions)
        
        # Generate options suggestions with enhanced context
        options_suggestions = self._generate_options_suggestions_enhanced(market_context)
        suggestions.extend(options_suggestions)
        
        # Apply order flow filtering
        filtered_suggestions, filtering_stats = self._filter_conflicting_order_flow_signals(suggestions, order_flow_data)
        
        # Add generation diagnostics to filtering stats
        filtering_stats['futures_generated'] = len(futures_suggestions)
        filtering_stats['options_generated'] = len(options_suggestions)
        filtering_stats['generation_issue'] = len(suggestions) == 0
        
        # Add decision logic and context to each suggestion
        for suggestion in filtered_suggestions:
            direction = 'Bullish' if suggestion.get('action') == 'BUY' else 'Bearish' if suggestion.get('action') == 'SELL' else 'Neutral'
            
            # Build decision logic
            decision_logic = self._build_decision_logic(market_context, suggestion, direction)
            suggestion['decision_logic'] = decision_logic
            
            # Add context information
            decision_context = self._build_decision_context(market_context, suggestion, direction)
            suggestion['decision_context'] = decision_context
        
        logging.info(f"Suggestions generated: futures={len(futures_suggestions)}, options={len(options_suggestions)}, initial={len(suggestions)}, filtered={len(filtered_suggestions)}")
        
        if len(suggestions) == 0:
            logging.warning("No suggestions generated. Possible reasons: missing TPO data, invalid price, or market conditions don't meet criteria.")
        
        # Store filtering stats in a class variable for API access
        self._last_filtering_stats = filtering_stats
        
        return filtered_suggestions
    
    def _generate_futures_suggestions(self, tpo_data: Dict, current_price: float, pre_market_tpo: Optional[Dict] = None) -> List[Dict]:
        """
        Generate futures trading suggestions
        
        Args:
            tpo_data: TPO analysis data (live or pre-market)
            current_price: Current market price
            pre_market_tpo: Pre-market TPO data for comparison
            
        Returns:
            List of futures suggestions
        """
        suggestions = []
        
        poc = tpo_data.get('poc')
        vah = tpo_data.get('value_area_high')
        val = tpo_data.get('value_area_low')
        ib_high = tpo_data.get('initial_balance_high')
        ib_low = tpo_data.get('initial_balance_low')
        
        if not all([poc, vah, val]):
            return suggestions
        
        # Determine bias from TPO structure
        price_vs_poc = current_price - poc
        price_vs_vah = current_price - vah
        price_vs_val = current_price - val
        
        # Calculate position sizing based on Value Area width
        va_width = vah - val
        position_size_multiplier = max(1.0, min(3.0, 100.0 / max(10.0, va_width)))  # Adjust based on VA width
        num_lots = int(50 * position_size_multiplier)
        
        # Bullish scenario: Price above POC, approaching or above VAH
        if price_vs_poc > 0 and (price_vs_vah >= -20 or price_vs_vah > 0):
            entry_level = max(current_price, poc)
            margin_info = calculate_futures_margin(entry_level, num_lots)
            
            # Calculate potential profit for each target level - only include profitable targets
            target_levels_with_profit = []
            max_profit = 0.0
            lot_size = 50  # Nifty lot size
            
            for target in [
                {'level': vah, 'type': 'VAH', 'probability': 'High'},
                {'level': vah + (va_width * 0.3), 'type': 'Extension', 'probability': 'Medium'}
            ]:
                # For BUY, target must be above entry to be profitable
                if target['level'] > entry_level:
                    profit = calculate_futures_profit(entry_level, target['level'], 'BUY', num_lots, lot_size)
                    if profit > 0:  # Only include targets with positive profit
                        target['potential_profit'] = profit
                        target_levels_with_profit.append(target)
                        max_profit = max(max_profit, profit)
            
            # Only add suggestion if we have at least one profitable target
            if target_levels_with_profit and max_profit > 0:
                suggestion = {
                    'instrument': 'NIFTY',
                    'derivative_type': 'FUTURES',
                    'action': 'BUY',
                    'entry_level': entry_level,
                    'stop_loss': ib_low if ib_low and ib_low < val else val - (va_width * 0.1),
                    'target_levels': target_levels_with_profit,
                    'position_size': f"{num_lots} lots",
                    'max_potential_profit': round(max_profit, 2),
                    'confidence_score': tpo_data.get('confidence_score', 0),
                    'rationale': f"Price above POC ({poc:.2f}) and near/above VAH ({vah:.2f}). Bullish structure suggests upward continuation.",
                    'tpo_levels_used': {
                        'poc': poc,
                        'vah': vah,
                        'val': val,
                        'ib_low': ib_low
                    },
                    'required_margin': margin_info,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Adjust if pre-market showed different structure
                if pre_market_tpo and isinstance(pre_market_tpo, dict):
                    pm_vah = pre_market_tpo.get('value_area_high')
                    if pm_vah and vah > pm_vah:
                        suggestion['rationale'] += " Live market VAH above pre-market VAH - strengthening bullish bias."
                        suggestion['confidence_score'] = min(100, suggestion['confidence_score'] + 5)
                
                suggestions.append(suggestion)
        
        # Bearish scenario: Price below POC, approaching or below VAL
        elif price_vs_poc < 0 and (price_vs_val <= 20 or price_vs_val < 0):
            entry_level = min(current_price, poc)
            margin_info = calculate_futures_margin(entry_level, num_lots)
            
            # Calculate potential profit for each target level - only include profitable targets
            target_levels_with_profit = []
            max_profit = 0.0
            lot_size = 50  # Nifty lot size
            
            for target in [
                {'level': val, 'type': 'VAL', 'probability': 'High'},
                {'level': val - (va_width * 0.3), 'type': 'Extension', 'probability': 'Medium'}
            ]:
                # For SELL, target must be below entry to be profitable
                if target['level'] < entry_level:
                    profit = calculate_futures_profit(entry_level, target['level'], 'SELL', num_lots, lot_size)
                    if profit > 0:  # Only include targets with positive profit
                        target['potential_profit'] = profit
                        target_levels_with_profit.append(target)
                        max_profit = max(max_profit, profit)
            
            # Only add suggestion if we have at least one profitable target
            if target_levels_with_profit and max_profit > 0:
                suggestion = {
                    'instrument': 'NIFTY',
                    'derivative_type': 'FUTURES',
                    'action': 'SELL',
                    'entry_level': entry_level,
                    'stop_loss': ib_high if ib_high and ib_high > vah else vah + (va_width * 0.1),
                    'target_levels': target_levels_with_profit,
                    'position_size': f"{num_lots} lots",
                    'max_potential_profit': round(max_profit, 2),
                    'confidence_score': tpo_data.get('confidence_score', 0),
                    'rationale': f"Price below POC ({poc:.2f}) and near/below VAL ({val:.2f}). Bearish structure suggests downward continuation.",
                    'tpo_levels_used': {
                        'poc': poc,
                        'vah': vah,
                        'val': val,
                        'ib_high': ib_high
                    },
                    'required_margin': margin_info,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Adjust if pre-market showed different structure
                if pre_market_tpo and isinstance(pre_market_tpo, dict):
                    pm_val = pre_market_tpo.get('value_area_low')
                    if pm_val and val < pm_val:
                        suggestion['rationale'] += " Live market VAL below pre-market VAL - strengthening bearish bias."
                        suggestion['confidence_score'] = min(100, suggestion['confidence_score'] + 5)
                
                suggestions.append(suggestion)
        
        # Neutral/Mean Reversion: Price near POC, in Value Area
        elif abs(price_vs_poc) <= (va_width * 0.1):
            # Suggest straddle/strangle if in middle of value area
            suggestion = {
                'instrument': 'NIFTY',
                'derivative_type': 'FUTURES',
                'action': 'WAIT',
                'entry_level': poc,
                'stop_loss': None,
                'target_levels': [
                    {'level': vah, 'type': 'VAH', 'probability': 'Medium'},
                    {'level': val, 'type': 'VAL', 'probability': 'Medium'}
                ],
                'position_size': 'Monitor for breakout',
                'confidence_score': tpo_data.get('confidence_score', 0) * 0.7,  # Lower confidence for neutral
                'rationale': f"Price near POC ({poc:.2f}) within Value Area. Wait for breakout above VAH ({vah:.2f}) or below VAL ({val:.2f}).",
                'tpo_levels_used': {
                    'poc': poc,
                    'vah': vah,
                    'val': val
                },
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            suggestions.append(suggestion)
        
        return suggestions
    
    def _generate_options_suggestions(self, tpo_data: Dict, current_price: float, pre_market_tpo: Optional[Dict] = None) -> List[Dict]:
        """
        Generate options trading suggestions
        
        Args:
            tpo_data: TPO analysis data
            current_price: Current market price
            pre_market_tpo: Pre-market TPO data for comparison
            
        Returns:
            List of options suggestions
        """
        suggestions = []
        
        poc = tpo_data.get('poc')
        vah = tpo_data.get('value_area_high')
        val = tpo_data.get('value_area_low')
        
        if not all([poc, vah, val]):
            return suggestions
        
        va_width = vah - val
        
        # Determine suggested strikes based on TPO levels
        # Round to nearest 50 for Nifty (strike interval)
        def round_to_strike(price):
            return round(price / 50) * 50
        
        # Bullish Call Options
        if current_price > poc:
            call_strike = round_to_strike(vah)
            put_strike = round_to_strike(val)
            
            # Estimate premium (typically 1-3% of strike for ATM/OTM options)
            estimated_premium_pct = 0.02  # 2% conservative estimate
            estimated_premium = current_price * estimated_premium_pct
            num_lots = 2  # Default for options
            lot_size = 50  # Nifty lot size
            
            margin_info = calculate_options_margin(call_strike, estimated_premium, num_lots, is_long=True)
            
            # Calculate potential profit for each target level - only include profitable targets
            target_levels_with_profit = []
            max_profit = 0.0
            
            for target in [
                {'level': round_to_strike(vah + va_width * 0.3), 'type': 'Extension', 'probability': 'Medium'}
            ]:
                # For CALL, target must be above strike to be profitable
                if target['level'] > call_strike:
                    profit = calculate_options_profit(current_price, target['level'], call_strike, 
                                                       estimated_premium, 'CALL', num_lots, lot_size)
                    if profit > 0:  # Only include targets with positive profit
                        target['potential_profit'] = profit
                        target_levels_with_profit.append(target)
                        max_profit = max(max_profit, profit)
            
            # Only add suggestion if we have at least one profitable target
            if target_levels_with_profit and max_profit > 0:
                suggestion = {
                    'instrument': 'NIFTY',
                    'derivative_type': 'CALL',
                    'action': 'BUY',
                    'strike_price': call_strike,
                    'entry_level': current_price,
                    'stop_loss': None,  # Options are limited risk
                    'target_levels': target_levels_with_profit,
                    'position_size': f'{num_lots} lots (based on risk)',
                    'max_potential_profit': round(max_profit, 2),
                    'confidence_score': tpo_data.get('confidence_score', 0),
                    'rationale': f"Price above POC, bullish structure. Buy ATM/OTM CALLs near VAH strike ({call_strike}). Target extension above VAH.",
                    'tpo_levels_used': {
                        'poc': poc,
                        'vah': vah,
                        'suggested_strike': call_strike
                    },
                    'required_margin': margin_info,
                    'estimated_premium': estimated_premium,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                suggestions.append(suggestion)
        
        # Bearish Put Options
        elif current_price < poc:
            call_strike = round_to_strike(vah)
            put_strike = round_to_strike(val)
            
            # Estimate premium (typically 1-3% of strike for ATM/OTM options)
            estimated_premium_pct = 0.02  # 2% conservative estimate
            estimated_premium = current_price * estimated_premium_pct
            num_lots = 2  # Default for options
            lot_size = 50  # Nifty lot size
            
            margin_info = calculate_options_margin(put_strike, estimated_premium, num_lots, is_long=True)
            
            # Calculate potential profit for each target level - only include profitable targets
            target_levels_with_profit = []
            max_profit = 0.0
            
            for target in [
                {'level': round_to_strike(val - va_width * 0.3), 'type': 'Extension', 'probability': 'Medium'}
            ]:
                # For PUT, target must be below strike to be profitable
                if target['level'] < put_strike:
                    profit = calculate_options_profit(current_price, target['level'], put_strike, 
                                                       estimated_premium, 'PUT', num_lots, lot_size)
                    if profit > 0:  # Only include targets with positive profit
                        target['potential_profit'] = profit
                        target_levels_with_profit.append(target)
                        max_profit = max(max_profit, profit)
            
            # Only add suggestion if we have at least one profitable target
            if target_levels_with_profit and max_profit > 0:
                suggestion = {
                    'instrument': 'NIFTY',
                    'derivative_type': 'PUT',
                    'action': 'BUY',
                    'strike_price': put_strike,
                    'entry_level': current_price,
                    'stop_loss': None,  # Options are limited risk
                    'target_levels': target_levels_with_profit,
                    'position_size': f'{num_lots} lots (based on risk)',
                    'max_potential_profit': round(max_profit, 2),
                    'confidence_score': tpo_data.get('confidence_score', 0),
                    'rationale': f"Price below POC, bearish structure. Buy ATM/OTM PUTs near VAL strike ({put_strike}). Target extension below VAL.",
                    'tpo_levels_used': {
                        'poc': poc,
                        'val': val,
                        'suggested_strike': put_strike
                    },
                    'required_margin': margin_info,
                    'estimated_premium': estimated_premium,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                suggestions.append(suggestion)
        
        # Straddle/Strangle for neutral scenarios
        if abs(current_price - poc) <= (va_width * 0.15):
            atm_strike = round_to_strike(current_price)
            upper_strike = round_to_strike(vah)
            lower_strike = round_to_strike(val)
            
            # Estimate premium for both CALL and PUT
            estimated_premium_pct = 0.02
            estimated_premium = current_price * estimated_premium_pct
            num_lots = 1  # 1 lot of each
            lot_size = 50  # Nifty lot size
            
            # Calculate margin for straddle (1 CALL + 1 PUT)
            call_margin = calculate_options_margin(atm_strike, estimated_premium, num_lots, is_long=True)
            put_margin = calculate_options_margin(atm_strike, estimated_premium, num_lots, is_long=True)
            total_straddle_margin = call_margin['total_required'] + put_margin['total_required']
            
            # Calculate potential profit for each target level - only include profitable targets
            target_levels_with_profit = []
            max_profit = 0.0
            
            for target in [
                {'level': upper_strike, 'type': 'VAH', 'probability': 'Medium'},
                {'level': lower_strike, 'type': 'VAL', 'probability': 'Medium'}
            ]:
                # For straddle, target must be significantly away from strike to be profitable
                # (one leg needs to make enough profit to cover the loss on the other leg + both premiums)
                straddle_profit = calculate_straddle_profit(current_price, target['level'], atm_strike,
                                                             estimated_premium, estimated_premium, num_lots, lot_size)
                
                # Only include targets with positive total profit
                if straddle_profit['total_profit'] > 0:
                    # For straddle, profit depends on which leg is in the money
                    # If target is above strike, CALL leg profits; if below, PUT leg profits
                    if target['level'] > atm_strike:
                        # CALL leg in the money
                        target['potential_profit'] = straddle_profit['total_profit']
                        target['call_profit'] = straddle_profit['call_profit']
                        target['put_profit'] = straddle_profit['put_profit']
                    else:
                        # PUT leg in the money
                        target['potential_profit'] = straddle_profit['total_profit']
                        target['call_profit'] = straddle_profit['call_profit']
                        target['put_profit'] = straddle_profit['put_profit']
                    target_levels_with_profit.append(target)
                    max_profit = max(max_profit, straddle_profit['total_profit'])
            
            # Only add suggestion if we have at least one profitable target
            if target_levels_with_profit and max_profit > 0:
                suggestion = {
                    'instrument': 'NIFTY',
                    'derivative_type': 'STRADDLE',
                    'action': 'BUY',
                    'strike_price': f"CALL {atm_strike} + PUT {atm_strike}",
                    'entry_level': current_price,
                    'stop_loss': None,
                    'target_levels': target_levels_with_profit,
                    'position_size': '1 straddle (1 CALL + 1 PUT)',
                    'max_potential_profit': round(max_profit, 2),
                    'confidence_score': tpo_data.get('confidence_score', 0) * 0.6,
                    'rationale': f"Price consolidating near POC. Buy ATM straddle for breakout play. Target VAH ({upper_strike}) or VAL ({lower_strike}).",
                    'tpo_levels_used': {
                        'poc': poc,
                        'vah': vah,
                        'val': val,
                        'atm_strike': atm_strike
                    },
                    'required_margin': {
                        'total_required': total_straddle_margin,
                        'margin_type': 'Premium (CALL + PUT)',
                        'call_premium': call_margin['total_required'],
                        'put_premium': put_margin['total_required']
                    },
                    'estimated_premium': estimated_premium * 2,  # Both CALL and PUT
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                suggestions.append(suggestion)
        
        return suggestions
    
    def _fetch_order_flow_data(self, analysis_date: str) -> Dict:
        """
        Fetch order flow analysis data and extract detailed signals
        
        Returns:
            Dictionary with detailed order flow signals:
            - exhaustion_signals: List of exhaustion events with prices and volume multiples
            - trapped_traders: Buyers/sellers with specific price levels and severity
            - volume_divergences: Bullish/bearish divergences with price levels
            - pressure_analysis: Granular pressure scores (Strong_buy, Buy, Neutral, Sell, Strong_sell)
            - absorption_zones: Price ranges where large orders are absorbed
        """
        try:
            from datetime import date as date_type
            analysis_date_obj = datetime.strptime(analysis_date, '%Y-%m-%d').date() if isinstance(analysis_date, str) else analysis_date
            
            # Analyze order flow for the current session
            order_flow = self.order_flow_analyzer.analyze_order_flow(
                start_time="09:15:00",
                end_time="15:30:00",
                analysis_date=analysis_date_obj,
                lookback_periods=20
            )
            
            if not order_flow.get('success'):
                return {}
            
            # Extract detailed signals
            exhaustion_signals = order_flow.get('exhaustion_signals', {})
            trapped_traders = order_flow.get('trapped_traders', {})
            volume_divergences = order_flow.get('volume_divergences', {})
            pressure_analysis = order_flow.get('pressure_analysis', {})
            absorption_zones = order_flow.get('absorption_zones', [])
            
            # Return enhanced order flow data with detailed signals
            return {
                'success': True,
                'exhaustion_signals': {
                    'buying_exhaustion': exhaustion_signals.get('buying_exhaustion', exhaustion_signals.get('bullish_exhaustion', 0)),
                    'selling_exhaustion': exhaustion_signals.get('selling_exhaustion', exhaustion_signals.get('bearish_exhaustion', 0)),
                    'events': exhaustion_signals.get('exhaustion_events', []),
                    'count': exhaustion_signals.get('count', 0)
                },
                'trapped_traders': {
                    'trapped_buyers': trapped_traders.get('trapped_buyers', []),
                    'trapped_sellers': trapped_traders.get('trapped_sellers', []),
                    'trapped_buyers_count': trapped_traders.get('trapped_buyers_count', 0),
                    'trapped_sellers_count': trapped_traders.get('trapped_sellers_count', 0),
                    'count': trapped_traders.get('count', 0)
                },
                'volume_divergences': {
                    'bullish_divergences': volume_divergences.get('bullish_divergences', 0),
                    'bearish_divergences': volume_divergences.get('bearish_divergences', 0),
                    'divergences': volume_divergences.get('divergences', []),
                    'count': volume_divergences.get('count', 0)
                },
                'pressure_analysis': {
                    'pressure_direction': pressure_analysis.get('pressure_direction', 'Neutral'),
                    'pressure_score': pressure_analysis.get('pressure_score', 0),
                    'buy_pressure_pct': pressure_analysis.get('buy_pressure_pct', 50.0),
                    'sell_pressure_pct': pressure_analysis.get('sell_pressure_pct', 50.0),
                    'total_buy_volume': pressure_analysis.get('total_buy_volume', 0),
                    'total_sell_volume': pressure_analysis.get('total_sell_volume', 0)
                },
                'absorption_zones': absorption_zones,
                'overall_sentiment': order_flow.get('overall_sentiment', {}),
                # Keep original structure for backward compatibility
                'original': order_flow
            }
        except Exception as e:
            logging.warning(f"Error fetching order flow data: {e}")
            return {}
    
    def _filter_conflicting_order_flow_signals(self, suggestions: List[Dict], order_flow: Dict) -> Tuple[List[Dict], Dict]:
        """
        Filter out suggestions that conflict with order flow signals
        
        Args:
            suggestions: List of suggestion dictionaries
            order_flow: Order flow analysis data
            
        Returns:
            Tuple of (filtered_suggestions, filtering_stats)
        """
        if not order_flow.get('success'):
            return suggestions, {
                'initial_count': len(suggestions),
                'filtered_exhaustion': 0,
                'filtered_sentiment_conflict': 0,
                'filtered_pressure_conflict': 0,
                'filtered_weak_confidence': 0,
                'final_count': len(suggestions)
            }
        
        filtering_stats = {
            'initial_count': len(suggestions),
            'filtered_exhaustion': 0,
            'filtered_sentiment_conflict': 0,
            'filtered_pressure_conflict': 0,
            'filtered_weak_confidence': 0,
            'final_count': 0
        }
        
        filtered_suggestions = []
        exhaustion_signals = order_flow.get('exhaustion_signals', {})
        pressure_analysis = order_flow.get('pressure_analysis', {})
        overall_sentiment = order_flow.get('overall_sentiment', {})
        
        for suggestion in suggestions:
            action = suggestion.get('action', '')
            direction = 'Bullish' if action == 'BUY' else 'Bearish' if action == 'SELL' else 'Neutral'
            confidence = suggestion.get('confidence_score', 0)
            
            should_filter = False
            filter_reason = None
            
            # Filter 1: Exhaustion signals conflict
            if action == 'BUY':
                buying_exhaustion_count = exhaustion_signals.get('buying_exhaustion', 0)
                if buying_exhaustion_count > 0:
                    should_filter = True
                    filter_reason = 'buying_exhaustion'
                    filtering_stats['filtered_exhaustion'] += 1
            elif action == 'SELL':
                selling_exhaustion_count = exhaustion_signals.get('selling_exhaustion', 0)
                if selling_exhaustion_count > 0:
                    should_filter = True
                    filter_reason = 'selling_exhaustion'
                    filtering_stats['filtered_exhaustion'] += 1
            
            # Filter 2: Order flow sentiment strongly conflicts
            if not should_filter:
                sentiment = overall_sentiment.get('sentiment', 'Neutral')
                if direction == 'Bullish' and sentiment in ['Bearish', 'Strongly_Bearish']:
                    if confidence < 50:  # Only filter weak signals
                        should_filter = True
                        filter_reason = 'sentiment_conflict'
                        filtering_stats['filtered_sentiment_conflict'] += 1
                elif direction == 'Bearish' and sentiment in ['Bullish', 'Strongly_Bullish']:
                    if confidence < 50:  # Only filter weak signals
                        should_filter = True
                        filter_reason = 'sentiment_conflict'
                        filtering_stats['filtered_sentiment_conflict'] += 1
            
            # Filter 3: Pressure analysis strongly conflicts
            if not should_filter:
                pressure_direction = pressure_analysis.get('pressure_direction', 'Neutral')
                if direction == 'Bullish' and pressure_direction in ['Strong_sell', 'Sell']:
                    if confidence < 30:  # Filter weak signals when pressure strongly conflicts
                        should_filter = True
                        filter_reason = 'pressure_conflict'
                        filtering_stats['filtered_pressure_conflict'] += 1
                elif direction == 'Bearish' and pressure_direction in ['Strong_buy', 'Buy']:
                    if confidence < 30:  # Filter weak signals when pressure strongly conflicts
                        should_filter = True
                        filter_reason = 'pressure_conflict'
                        filtering_stats['filtered_pressure_conflict'] += 1
            
            # Filter 4: Very weak confidence (below threshold)
            if not should_filter and confidence < 20:
                should_filter = True
                filter_reason = 'weak_confidence'
                filtering_stats['filtered_weak_confidence'] += 1
            
            if not should_filter:
                filtered_suggestions.append(suggestion)
            else:
                # Add filter reason to suggestion for debugging (optional)
                suggestion['_filtered'] = True
                suggestion['_filter_reason'] = filter_reason
        
        filtering_stats['final_count'] = len(filtered_suggestions)
        return filtered_suggestions, filtering_stats
    
    def _fetch_market_bias_data(self, analysis_date: str) -> Dict:
        """Fetch market bias analysis data"""
        try:
            if hasattr(self.tpo_analyzer, 'market_bias_analyzer'):
                bias_analyzer = self.tpo_analyzer.market_bias_analyzer
                # Get combined bias analysis
                pre_market_bias = bias_analyzer.analyze_pre_market_bias(analysis_date)
                real_time_bias = bias_analyzer.analyze_real_time_bias(analysis_date)
                
                return {
                    'pre_market_bias': pre_market_bias,
                    'real_time_bias': real_time_bias,
                    'combined_bias': bias_analyzer._combine_bias_analyses(pre_market_bias, real_time_bias) if hasattr(bias_analyzer, '_combine_bias_analyses') else {}
                }
            return {}
        except Exception as e:
            logging.warning(f"Error fetching market bias data: {e}")
            return {}
    
    def _fetch_options_data(self, analysis_date: str, current_price: float) -> Dict:
        """Fetch options chain data for analysis"""
        try:
            # Get options chain around current price (±5% range)
            strike_range = (current_price * 0.95, current_price * 1.05)
            options_chain = self.options_fetcher.get_options_chain(
                expiry=None,  # Get all expiries
                strike_range=strike_range,
                option_type=None,  # Both CE and PE
                min_volume=100,
                min_oi=1000
            )
            
            if options_chain.empty:
                return {}
            
            # Analyze options data
            call_oi = options_chain[options_chain['option_type'] == 'CE']['oi'].sum() if 'option_type' in options_chain.columns else 0
            put_oi = options_chain[options_chain['option_type'] == 'PE']['oi'].sum() if 'option_type' in options_chain.columns else 0
            pcr = put_oi / call_oi if call_oi > 0 else 0  # Put-Call Ratio
            
            # Find max OI strikes
            max_oi_call = options_chain[options_chain['option_type'] == 'CE'].nlargest(1, 'oi') if 'option_type' in options_chain.columns else pd.DataFrame()
            max_oi_put = options_chain[options_chain['option_type'] == 'PE'].nlargest(1, 'oi') if 'option_type' in options_chain.columns else pd.DataFrame()
            
            return {
                'options_chain': options_chain,
                'put_call_ratio': pcr,
                'call_oi': call_oi,
                'put_oi': put_oi,
                'max_oi_call_strike': float(max_oi_call['strike_price'].iloc[0]) if not max_oi_call.empty else None,
                'max_oi_put_strike': float(max_oi_put['strike_price'].iloc[0]) if not max_oi_put.empty else None,
                'total_options': len(options_chain)
            }
        except Exception as e:
            logging.warning(f"Error fetching options data: {e}")
            return {}
    
    def _fetch_futures_data(self, analysis_date: str) -> Dict:
        """Fetch futures tick data for momentum analysis"""
        try:
            if not self.db_fetcher:
                return {}
            
            # Fetch recent futures data (last 30 minutes)
            futures_df = self.db_fetcher.fetch_tick_data(
                table_name='futures_ticks',
                instrument_token=self.futures_instrument_token,
                start_time=f'{analysis_date} 09:15:00.000',
                end_time=f'{analysis_date} 15:30:00.000'
            )
            
            if futures_df.empty:
                return {}
            
            # Calculate momentum indicators
            if 'last_price' in futures_df.columns and len(futures_df) > 1:
                prices = futures_df['last_price'].values
                recent_prices = prices[-20:] if len(prices) >= 20 else prices
                
                price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] * 100 if len(recent_prices) > 0 else 0
                volatility = np.std(recent_prices) / np.mean(recent_prices) * 100 if len(recent_prices) > 1 else 0
                
                # Volume analysis
                if 'volume' in futures_df.columns:
                    recent_volume = futures_df['volume'].tail(20).sum() if len(futures_df) >= 20 else futures_df['volume'].sum()
                    avg_volume = futures_df['volume'].mean() if 'volume' in futures_df.columns else 0
                    volume_ratio = recent_volume / (avg_volume * 20) if avg_volume > 0 else 1.0
                else:
                    volume_ratio = 1.0
                
                return {
                    'current_price': float(prices[-1]) if len(prices) > 0 else None,
                    'price_change_pct': float(price_change),
                    'volatility_pct': float(volatility),
                    'volume_ratio': float(volume_ratio),
                    'trend': 'Bullish' if price_change > 0.1 else 'Bearish' if price_change < -0.1 else 'Neutral'
                }
            
            return {}
        except Exception as e:
            logging.warning(f"Error fetching futures data: {e}")
            return {}
    
    def _analyze_tick_momentum(self, analysis_date: str) -> Dict:
        """Analyze Nifty tick data for momentum and trend strength"""
        try:
            if not self.db_fetcher:
                return {}
            
            # Fetch recent tick data (last hour)
            tick_df = self.db_fetcher.fetch_tick_data(
                table_name='ticks',
                instrument_token=self.instrument_token,
                start_time=f'{analysis_date} 09:15:00.000',
                end_time=f'{analysis_date} 15:30:00.000'
            )
            
            if tick_df.empty or 'last_price' not in tick_df.columns:
                return {}
            
            prices = tick_df['last_price'].values
            if len(prices) < 10:
                return {}
            
            # Calculate momentum indicators
            recent_prices = prices[-30:] if len(prices) >= 30 else prices
            
            # Price momentum
            short_ma = np.mean(recent_prices[-5:]) if len(recent_prices) >= 5 else recent_prices[-1]
            long_ma = np.mean(recent_prices[-20:]) if len(recent_prices) >= 20 else np.mean(recent_prices)
            momentum = (short_ma - long_ma) / long_ma * 100 if long_ma > 0 else 0
            
            # Trend strength
            price_changes = np.diff(recent_prices)
            up_moves = np.sum(price_changes > 0)
            down_moves = np.sum(price_changes < 0)
            trend_strength = abs(up_moves - down_moves) / len(price_changes) * 100 if len(price_changes) > 0 else 0
            
            # Volatility
            volatility = np.std(recent_prices) / np.mean(recent_prices) * 100 if len(recent_prices) > 1 else 0
            
            return {
                'momentum_pct': float(momentum),
                'trend_strength': float(trend_strength),
                'volatility_pct': float(volatility),
                'trend_direction': 'Bullish' if momentum > 0.05 else 'Bearish' if momentum < -0.05 else 'Neutral',
                'current_price': float(prices[-1])
            }
        except Exception as e:
            logging.warning(f"Error analyzing tick momentum: {e}")
            return {}
    
    def _calculate_enhanced_confidence(self, market_context: Dict, base_confidence: float, direction: str) -> float:
        """
        Calculate enhanced confidence score by combining multiple signals
        
        Args:
            market_context: Dictionary containing all market data
            base_confidence: Base confidence from TPO (0-100)
            direction: 'Bullish' or 'Bearish'
            
        Returns:
            Enhanced confidence score (0-100)
        """
        confidence = base_confidence
        
        # Order flow signals (max +15 points)
        order_flow = market_context.get('order_flow', {})
        if order_flow.get('success'):
            sentiment = order_flow.get('overall_sentiment', {})
            sentiment_type = sentiment.get('sentiment', 'Neutral')
            
            if direction == 'Bullish' and sentiment_type == 'Bullish':
                confidence += 10
            elif direction == 'Bearish' and sentiment_type == 'Bearish':
                confidence += 10
            elif direction == 'Bullish' and sentiment_type == 'Bearish':
                confidence -= 5
            elif direction == 'Bearish' and sentiment_type == 'Bullish':
                confidence -= 5
            
            # Trapped traders signal
            trapped = order_flow.get('trapped_traders', {})
            if direction == 'Bullish' and trapped.get('trapped_sellers_count', 0) > 0:
                confidence += 5
            elif direction == 'Bearish' and trapped.get('trapped_buyers_count', 0) > 0:
                confidence += 5
            
            # Granular pressure analysis adjustments
            pressure_analysis = order_flow.get('pressure_analysis', {})
            pressure_direction = pressure_analysis.get('pressure_direction', 'Neutral')
            
            if pressure_direction == 'Strong_buy':
                if direction == 'Bullish':
                    confidence += 10
                elif direction == 'Bearish':
                    confidence -= 10
            elif pressure_direction == 'Buy':
                if direction == 'Bullish':
                    confidence += 5
                elif direction == 'Bearish':
                    confidence -= 5
            elif pressure_direction == 'Strong_sell':
                if direction == 'Bearish':
                    confidence += 10
                elif direction == 'Bullish':
                    confidence -= 10
            elif pressure_direction == 'Sell':
                if direction == 'Bearish':
                    confidence += 5
                elif direction == 'Bullish':
                    confidence -= 5
            
            # Exhaustion signal impact: -15 points when exhaustion conflicts with direction
            exhaustion_signals = order_flow.get('exhaustion_signals', {})
            if direction == 'Bullish' and exhaustion_signals.get('buying_exhaustion', 0) > 0:
                confidence -= 15
            elif direction == 'Bearish' and exhaustion_signals.get('selling_exhaustion', 0) > 0:
                confidence -= 15
        
        # Market bias signals (max +20 points)
        market_bias = market_context.get('market_bias', {})
        combined_bias = market_bias.get('combined_bias', {})
        bias_score = combined_bias.get('bias_score', 0)
        bias_direction = combined_bias.get('bias_direction', 'Neutral')
        
        if direction == 'Bullish' and bias_direction == 'Bullish':
            confidence += min(20, abs(bias_score) * 0.5)
        elif direction == 'Bearish' and bias_direction == 'Bearish':
            confidence += min(20, abs(bias_score) * 0.5)
        elif direction == 'Bullish' and bias_direction == 'Bearish':
            confidence -= min(10, abs(bias_score) * 0.3)
        elif direction == 'Bearish' and bias_direction == 'Bullish':
            confidence -= min(10, abs(bias_score) * 0.3)
        
        # Options data signals (max +15 points)
        options_data = market_context.get('options_data', {})
        pcr = options_data.get('put_call_ratio', 1.0)
        
        if direction == 'Bullish':
            # High PCR (>1.2) suggests bearish sentiment, which could mean bullish reversal
            if pcr > 1.2:
                confidence += 5
            elif pcr < 0.8:
                confidence -= 5
        elif direction == 'Bearish':
            # Low PCR (<0.8) suggests bullish sentiment, which could mean bearish reversal
            if pcr < 0.8:
                confidence += 5
            elif pcr > 1.2:
                confidence -= 5
        
        # Futures momentum signals (max +10 points)
        futures_data = market_context.get('futures_data', {})
        futures_trend = futures_data.get('trend', 'Neutral')
        futures_change = futures_data.get('price_change_pct', 0)
        
        if direction == 'Bullish' and futures_trend == 'Bullish' and futures_change > 0.2:
            confidence += 10
        elif direction == 'Bearish' and futures_trend == 'Bearish' and futures_change < -0.2:
            confidence += 10
        elif direction == 'Bullish' and futures_trend == 'Bearish':
            confidence -= 5
        elif direction == 'Bearish' and futures_trend == 'Bullish':
            confidence -= 5
        
        # Tick momentum signals (max +10 points)
        tick_momentum = market_context.get('tick_momentum', {})
        tick_direction = tick_momentum.get('trend_direction', 'Neutral')
        momentum_pct = tick_momentum.get('momentum_pct', 0)
        
        if direction == 'Bullish' and tick_direction == 'Bullish' and momentum_pct > 0.1:
            confidence += 10
        elif direction == 'Bearish' and tick_direction == 'Bearish' and momentum_pct < -0.1:
            confidence += 10
        elif direction == 'Bullish' and tick_direction == 'Bearish':
            confidence -= 5
        elif direction == 'Bearish' and tick_direction == 'Bullish':
            confidence -= 5
        
        return max(0, min(100, confidence))
    
    def _generate_futures_suggestions_enhanced(self, market_context: Dict) -> List[Dict]:
        """
        Generate futures trading suggestions with enhanced context from all data sources
        """
        tpo_data = market_context.get('tpo_data', {})
        current_price = market_context.get('current_price')
        pre_market_tpo = market_context.get('pre_market_tpo')
        
        if not tpo_data or not current_price:
            return []
        
        # Use existing logic but with enhanced confidence and rationale
        suggestions = self._generate_futures_suggestions(tpo_data, current_price, pre_market_tpo)
        
        # Get order flow data for stop loss enhancement
        order_flow = market_context.get('order_flow', {})
        absorption_zones = order_flow.get('absorption_zones', []) if order_flow.get('success') else []
        trapped_traders = order_flow.get('trapped_traders', {}) if order_flow.get('success') else {}
        
        # Enhance each suggestion with additional signals
        for suggestion in suggestions:
            direction = 'Bullish' if suggestion.get('action') == 'BUY' else 'Bearish' if suggestion.get('action') == 'SELL' else 'Neutral'
            entry_level = suggestion.get('entry_level', current_price)
            original_stop_loss = suggestion.get('stop_loss')
            
            # Improve stop loss placement using absorption zones and trapped trader levels
            if direction == 'Bullish' and original_stop_loss:
                # For BUY: Set stop loss below nearest absorption zone or trapped buyer level
                best_stop_loss = original_stop_loss
                stop_loss_reason = "TPO-based (IB/VAL)"
                
                # Check absorption zones below entry
                for zone in absorption_zones:
                    zone_low = zone.get('price_low') or zone.get('price')
                    if zone_low and zone_low < entry_level and zone_low < original_stop_loss:
                        # Use absorption zone as stop loss if it's closer to entry (tighter stop)
                        if zone_low > best_stop_loss or best_stop_loss == original_stop_loss:
                            best_stop_loss = zone_low - 5  # 5 points below zone for safety
                            stop_loss_reason = f"Absorption zone at {zone_low:.2f}"
                
                # Check trapped buyer levels
                trapped_buyers = trapped_traders.get('trapped_buyers', [])
                for trapped in trapped_buyers:
                    trapped_price = trapped.get('futures_price') or trapped.get('price')
                    if trapped_price and trapped_price < entry_level and trapped_price < original_stop_loss:
                        if trapped_price > best_stop_loss or best_stop_loss == original_stop_loss:
                            best_stop_loss = trapped_price - 5  # 5 points below trapped level
                            stop_loss_reason = f"Trapped buyer level at {trapped_price:.2f}"
                
                suggestion['stop_loss'] = best_stop_loss
                if best_stop_loss != original_stop_loss:
                    suggestion['rationale'] += f" Stop loss adjusted to {stop_loss_reason}."
            
            elif direction == 'Bearish' and original_stop_loss:
                # For SELL: Set stop loss above nearest absorption zone or trapped seller level
                best_stop_loss = original_stop_loss
                stop_loss_reason = "TPO-based (IB/VAH)"
                
                # Check absorption zones above entry
                for zone in absorption_zones:
                    zone_high = zone.get('price_high') or zone.get('price')
                    if zone_high and zone_high > entry_level and zone_high > original_stop_loss:
                        # Use absorption zone as stop loss if it's closer to entry (tighter stop)
                        if zone_high < best_stop_loss or best_stop_loss == original_stop_loss:
                            best_stop_loss = zone_high + 5  # 5 points above zone for safety
                            stop_loss_reason = f"Absorption zone at {zone_high:.2f}"
                
                # Check trapped seller levels
                trapped_sellers = trapped_traders.get('trapped_sellers', [])
                for trapped in trapped_sellers:
                    trapped_price = trapped.get('futures_price') or trapped.get('price')
                    if trapped_price and trapped_price > entry_level and trapped_price > original_stop_loss:
                        if trapped_price < best_stop_loss or best_stop_loss == original_stop_loss:
                            best_stop_loss = trapped_price + 5  # 5 points above trapped level
                            stop_loss_reason = f"Trapped seller level at {trapped_price:.2f}"
                
                suggestion['stop_loss'] = best_stop_loss
                if best_stop_loss != original_stop_loss:
                    suggestion['rationale'] += f" Stop loss adjusted to {stop_loss_reason}."
            
            # Update confidence with enhanced calculation
            base_confidence = suggestion.get('confidence_score', 0)
            enhanced_confidence = self._calculate_enhanced_confidence(market_context, base_confidence, direction)
            suggestion['confidence_score'] = round(enhanced_confidence, 2)
            
            # Enhance rationale with insights from all sources
            rationale = suggestion.get('rationale', '')
            enhanced_rationale = self._build_enhanced_rationale(market_context, rationale, direction)
            suggestion['rationale'] = enhanced_rationale
            
            # Add additional context with comprehensive order flow details
            order_flow = market_context.get('order_flow', {})
            suggestion['market_signals'] = {
                'order_flow_sentiment': order_flow.get('overall_sentiment', {}).get('sentiment', 'Neutral'),
                'market_bias': market_context.get('market_bias', {}).get('combined_bias', {}).get('bias_direction', 'Neutral'),
                'put_call_ratio': market_context.get('options_data', {}).get('put_call_ratio', 1.0),
                'futures_trend': market_context.get('futures_data', {}).get('trend', 'Neutral'),
                'tick_momentum': market_context.get('tick_momentum', {}).get('trend_direction', 'Neutral')
            }
            
            # Add comprehensive order flow details
            if order_flow.get('success'):
                suggestion['order_flow_details'] = {
                    'exhaustion_signals': order_flow.get('exhaustion_signals', {}).get('events', []),
                    'trapped_traders': {
                        'buyers': order_flow.get('trapped_traders', {}).get('trapped_buyers', []),
                        'sellers': order_flow.get('trapped_traders', {}).get('trapped_sellers', [])
                    },
                    'volume_divergences': order_flow.get('volume_divergences', {}).get('divergences', []),
                    'pressure_details': {
                        'direction': order_flow.get('pressure_analysis', {}).get('pressure_direction', 'Neutral'),
                        'score': order_flow.get('pressure_analysis', {}).get('pressure_score', 0),
                        'buy_pct': order_flow.get('pressure_analysis', {}).get('buy_pressure_pct', 50.0),
                        'sell_pct': order_flow.get('pressure_analysis', {}).get('sell_pressure_pct', 50.0)
                    },
                    'absorption_zones': order_flow.get('absorption_zones', [])
                }
        
        return suggestions
    
    def _generate_options_suggestions_enhanced(self, market_context: Dict) -> List[Dict]:
        """
        Generate options trading suggestions with enhanced context from all data sources
        """
        tpo_data = market_context.get('tpo_data', {})
        current_price = market_context.get('current_price')
        pre_market_tpo = market_context.get('pre_market_tpo')
        options_data = market_context.get('options_data', {})
        
        if not tpo_data or not current_price:
            return []
        
        # Use existing logic but with enhanced strike selection and confidence
        suggestions = self._generate_options_suggestions(tpo_data, current_price, pre_market_tpo)
        
        # Enhance strike selection using order flow and options data
        order_flow = market_context.get('order_flow', {})
        trapped_traders = order_flow.get('trapped_traders', {}) if order_flow.get('success') else {}
        exhaustion_signals = order_flow.get('exhaustion_signals', {}) if order_flow.get('success') else {}
        absorption_zones = order_flow.get('absorption_zones', []) if order_flow.get('success') else []
        
        if options_data.get('options_chain') is not None and not options_data['options_chain'].empty:
            for suggestion in suggestions:
                original_strike = suggestion.get('strike_price', 0)
                strike_adjusted = False
                
                if suggestion.get('derivative_type') == 'CALL':
                    # Adjust CALL strike based on trapped seller levels
                    trapped_sellers = trapped_traders.get('trapped_sellers', [])
                    if trapped_sellers:
                        # Use the most recent trapped seller price as reference
                        latest_trapped_seller = trapped_sellers[-1] if trapped_sellers else None
                        if latest_trapped_seller:
                            trapped_price = latest_trapped_seller.get('futures_price') or latest_trapped_seller.get('price')
                            if trapped_price and abs(trapped_price - original_strike) < 200:
                                # Round to nearest 50 (Nifty strike interval)
                                adjusted_strike = round(trapped_price / 50) * 50
                                if abs(adjusted_strike - original_strike) < 100:
                                    suggestion['strike_price'] = adjusted_strike
                                    suggestion['rationale'] += f" Strike adjusted to trapped seller level ({adjusted_strike})."
                                    strike_adjusted = True
                    
                    # Consider exhaustion signal prices
                    if not strike_adjusted:
                        exhaustion_events = exhaustion_signals.get('events', [])
                        buying_exhaustion_events = [e for e in exhaustion_events if e.get('type') == 'buying_exhaustion']
                        if buying_exhaustion_events:
                            exhaustion_price = buying_exhaustion_events[-1].get('futures_price') or buying_exhaustion_events[-1].get('price')
                            if exhaustion_price and abs(exhaustion_price - original_strike) < 200:
                                adjusted_strike = round(exhaustion_price / 50) * 50
                                if abs(adjusted_strike - original_strike) < 100:
                                    suggestion['strike_price'] = adjusted_strike
                                    suggestion['rationale'] += f" Strike adjusted to buying exhaustion level ({adjusted_strike})."
                                    strike_adjusted = True
                    
                    # Use absorption zone boundaries as strike reference
                    if not strike_adjusted and absorption_zones:
                        # Use upper boundary of absorption zone
                        latest_zone = absorption_zones[-1] if absorption_zones else None
                        if latest_zone:
                            zone_high = latest_zone.get('price_high') or latest_zone.get('price')
                            if zone_high and abs(zone_high - original_strike) < 200:
                                adjusted_strike = round(zone_high / 50) * 50
                                if abs(adjusted_strike - original_strike) < 100:
                                    suggestion['strike_price'] = adjusted_strike
                                    suggestion['rationale'] += f" Strike adjusted to absorption zone upper boundary ({adjusted_strike})."
                                    strike_adjusted = True
                    
                    # Fallback to max OI strikes if available
                    if not strike_adjusted:
                        max_oi_strike = options_data.get('max_oi_call_strike')
                        if max_oi_strike and abs(max_oi_strike - original_strike) < 100:
                            suggestion['strike_price'] = max_oi_strike
                            suggestion['rationale'] += f" Strike adjusted to max OI level ({max_oi_strike})."
                
                elif suggestion.get('derivative_type') == 'PUT':
                    # Adjust PUT strike based on trapped buyer levels
                    trapped_buyers = trapped_traders.get('trapped_buyers', [])
                    if trapped_buyers:
                        # Use the most recent trapped buyer price as reference
                        latest_trapped_buyer = trapped_buyers[-1] if trapped_buyers else None
                        if latest_trapped_buyer:
                            trapped_price = latest_trapped_buyer.get('futures_price') or latest_trapped_buyer.get('price')
                            if trapped_price and abs(trapped_price - original_strike) < 200:
                                # Round to nearest 50 (Nifty strike interval)
                                adjusted_strike = round(trapped_price / 50) * 50
                                if abs(adjusted_strike - original_strike) < 100:
                                    suggestion['strike_price'] = adjusted_strike
                                    suggestion['rationale'] += f" Strike adjusted to trapped buyer level ({adjusted_strike})."
                                    strike_adjusted = True
                    
                    # Consider exhaustion signal prices
                    if not strike_adjusted:
                        exhaustion_events = exhaustion_signals.get('events', [])
                        selling_exhaustion_events = [e for e in exhaustion_events if e.get('type') == 'selling_exhaustion']
                        if selling_exhaustion_events:
                            exhaustion_price = selling_exhaustion_events[-1].get('futures_price') or selling_exhaustion_events[-1].get('price')
                            if exhaustion_price and abs(exhaustion_price - original_strike) < 200:
                                adjusted_strike = round(exhaustion_price / 50) * 50
                                if abs(adjusted_strike - original_strike) < 100:
                                    suggestion['strike_price'] = adjusted_strike
                                    suggestion['rationale'] += f" Strike adjusted to selling exhaustion level ({adjusted_strike})."
                                    strike_adjusted = True
                    
                    # Use absorption zone boundaries as strike reference
                    if not strike_adjusted and absorption_zones:
                        # Use lower boundary of absorption zone
                        latest_zone = absorption_zones[-1] if absorption_zones else None
                        if latest_zone:
                            zone_low = latest_zone.get('price_low') or latest_zone.get('price')
                            if zone_low and abs(zone_low - original_strike) < 200:
                                adjusted_strike = round(zone_low / 50) * 50
                                if abs(adjusted_strike - original_strike) < 100:
                                    suggestion['strike_price'] = adjusted_strike
                                    suggestion['rationale'] += f" Strike adjusted to absorption zone lower boundary ({adjusted_strike})."
                                    strike_adjusted = True
                    
                    # Fallback to max OI strikes if available
                    if not strike_adjusted:
                        max_oi_strike = options_data.get('max_oi_put_strike')
                        if max_oi_strike and abs(max_oi_strike - original_strike) < 100:
                            suggestion['strike_price'] = max_oi_strike
                            suggestion['rationale'] += f" Strike adjusted to max OI level ({max_oi_strike})."
        
        # Enhance each suggestion with additional signals
        for suggestion in suggestions:
            option_type = suggestion.get('derivative_type', '')
            direction = 'Bullish' if option_type == 'CALL' else 'Bearish' if option_type == 'PUT' else 'Neutral'
            
            # Update confidence with enhanced calculation
            base_confidence = suggestion.get('confidence_score', 0)
            enhanced_confidence = self._calculate_enhanced_confidence(market_context, base_confidence, direction)
            suggestion['confidence_score'] = round(enhanced_confidence, 2)
            
            # Enhance rationale
            rationale = suggestion.get('rationale', '')
            enhanced_rationale = self._build_enhanced_rationale(market_context, rationale, direction)
            suggestion['rationale'] = enhanced_rationale
            
            # Add additional context with comprehensive order flow details
            order_flow = market_context.get('order_flow', {})
            suggestion['market_signals'] = {
                'order_flow_sentiment': order_flow.get('overall_sentiment', {}).get('sentiment', 'Neutral'),
                'market_bias': market_context.get('market_bias', {}).get('combined_bias', {}).get('bias_direction', 'Neutral'),
                'put_call_ratio': market_context.get('options_data', {}).get('put_call_ratio', 1.0),
                'futures_trend': market_context.get('futures_data', {}).get('trend', 'Neutral'),
                'tick_momentum': market_context.get('tick_momentum', {}).get('trend_direction', 'Neutral')
            }
            
            # Add comprehensive order flow details
            if order_flow.get('success'):
                suggestion['order_flow_details'] = {
                    'exhaustion_signals': order_flow.get('exhaustion_signals', {}).get('events', []),
                    'trapped_traders': {
                        'buyers': order_flow.get('trapped_traders', {}).get('trapped_buyers', []),
                        'sellers': order_flow.get('trapped_traders', {}).get('trapped_sellers', [])
                    },
                    'volume_divergences': order_flow.get('volume_divergences', {}).get('divergences', []),
                    'pressure_details': {
                        'direction': order_flow.get('pressure_analysis', {}).get('pressure_direction', 'Neutral'),
                        'score': order_flow.get('pressure_analysis', {}).get('pressure_score', 0),
                        'buy_pct': order_flow.get('pressure_analysis', {}).get('buy_pressure_pct', 50.0),
                        'sell_pct': order_flow.get('pressure_analysis', {}).get('sell_pressure_pct', 50.0)
                    },
                    'absorption_zones': order_flow.get('absorption_zones', [])
                }
        
        return suggestions
    
    def _build_enhanced_rationale(self, market_context: Dict, base_rationale: str, direction: str) -> str:
        """
        Build enhanced rationale by combining insights from all data sources
        """
        rationale_parts = [base_rationale]
        
        # Order flow insights with detailed information
        order_flow = market_context.get('order_flow', {})
        if order_flow.get('success'):
            sentiment = order_flow.get('overall_sentiment', {})
            sentiment_type = sentiment.get('sentiment', 'Neutral')
            
            if sentiment_type == direction:
                rationale_parts.append(f"Order flow confirms {direction.lower()} sentiment.")
            
            # Exhaustion signal details
            exhaustion_signals = order_flow.get('exhaustion_signals', {})
            exhaustion_events = exhaustion_signals.get('events', [])
            if exhaustion_events:
                relevant_exhaustion = [e for e in exhaustion_events if 
                                     (direction == 'Bullish' and e.get('type') == 'selling_exhaustion') or
                                     (direction == 'Bearish' and e.get('type') == 'buying_exhaustion')]
                if relevant_exhaustion:
                    latest_exhaustion = relevant_exhaustion[-1]
                    exhaustion_price = latest_exhaustion.get('futures_price') or latest_exhaustion.get('price')
                    volume_multiple = latest_exhaustion.get('volume_multiple', 0)
                    if exhaustion_price:
                        rationale_parts.append(f"{latest_exhaustion.get('type', 'exhaustion').replace('_', ' ').title()} detected at {exhaustion_price:.2f} (volume {volume_multiple}x avg) - potential reversal.")
            
            # Trapped trader details
            trapped = order_flow.get('trapped_traders', {})
            if direction == 'Bullish' and trapped.get('trapped_sellers_count', 0) > 0:
                trapped_sellers = trapped.get('trapped_sellers', [])
                if trapped_sellers:
                    latest_trapped = trapped_sellers[-1]
                    trapped_price = latest_trapped.get('futures_price') or latest_trapped.get('price')
                    severity = latest_trapped.get('severity', 'medium')
                    rationale_parts.append(f"Trapped sellers detected ({trapped.get('trapped_sellers_count', 0)} instances, latest at {trapped_price:.2f}, {severity} severity) - potential bullish reversal.")
            elif direction == 'Bearish' and trapped.get('trapped_buyers_count', 0) > 0:
                trapped_buyers = trapped.get('trapped_buyers', [])
                if trapped_buyers:
                    latest_trapped = trapped_buyers[-1]
                    trapped_price = latest_trapped.get('futures_price') or latest_trapped.get('price')
                    severity = latest_trapped.get('severity', 'medium')
                    rationale_parts.append(f"Trapped buyers detected ({trapped.get('trapped_buyers_count', 0)} instances, latest at {trapped_price:.2f}, {severity} severity) - potential bearish reversal.")
            
            # Volume divergence information
            volume_divergences = order_flow.get('volume_divergences', {})
            divergences = volume_divergences.get('divergences', [])
            if divergences:
                relevant_divergences = [d for d in divergences if 
                                      (direction == 'Bullish' and d.get('type') == 'bullish_divergence') or
                                      (direction == 'Bearish' and d.get('type') == 'bearish_divergence')]
                if relevant_divergences:
                    latest_div = relevant_divergences[-1]
                    div_price = latest_div.get('futures_price') or latest_div.get('price')
                    if div_price:
                        rationale_parts.append(f"{latest_div.get('type', 'divergence').replace('_', ' ').title()} detected at {div_price:.2f} - potential reversal signal.")
            
            # Pressure analysis details
            pressure_analysis = order_flow.get('pressure_analysis', {})
            pressure_direction = pressure_analysis.get('pressure_direction', 'Neutral')
            pressure_score = pressure_analysis.get('pressure_score', 0)
            buy_pct = pressure_analysis.get('buy_pressure_pct', 50.0)
            sell_pct = pressure_analysis.get('sell_pressure_pct', 50.0)
            if pressure_direction != 'Neutral':
                rationale_parts.append(f"Pressure analysis: {pressure_direction} (score: {pressure_score:.1f}, Buy: {buy_pct:.1f}%, Sell: {sell_pct:.1f}%).")
            
            # Absorption zone information when relevant
            absorption_zones = order_flow.get('absorption_zones', [])
            if absorption_zones:
                latest_zone = absorption_zones[-1]
                zone_price = latest_zone.get('price')
                zone_high = latest_zone.get('price_high')
                zone_low = latest_zone.get('price_low')
                strength = latest_zone.get('strength', 'moderate')
                if zone_price:
                    if zone_high and zone_low:
                        rationale_parts.append(f"Absorption zone detected: {zone_low:.2f}-{zone_high:.2f} ({strength} strength) - large orders being absorbed.")
                    else:
                        rationale_parts.append(f"Absorption zone detected at {zone_price:.2f} ({strength} strength) - large orders being absorbed.")
        
        # Market bias insights
        market_bias = market_context.get('market_bias', {})
        combined_bias = market_bias.get('combined_bias', {})
        bias_direction = combined_bias.get('bias_direction', 'Neutral')
        bias_strength = combined_bias.get('bias_strength', 'Weak')
        
        if bias_direction == direction:
            rationale_parts.append(f"Market bias is {bias_strength.lower()} {bias_direction.lower()}.")
        elif bias_direction != 'Neutral':
            rationale_parts.append(f"Market bias is {bias_direction.lower()}, which may conflict with {direction.lower()} setup.")
        
        # Options data insights
        options_data = market_context.get('options_data', {})
        pcr = options_data.get('put_call_ratio', 1.0)
        if pcr > 1.2:
            rationale_parts.append(f"High Put-Call Ratio ({pcr:.2f}) suggests bearish sentiment - watch for reversal.")
        elif pcr < 0.8:
            rationale_parts.append(f"Low Put-Call Ratio ({pcr:.2f}) suggests bullish sentiment - watch for reversal.")
        
        # Futures momentum insights
        futures_data = market_context.get('futures_data', {})
        futures_trend = futures_data.get('trend', 'Neutral')
        futures_change = futures_data.get('price_change_pct', 0)
        if futures_trend == direction and abs(futures_change) > 0.2:
            rationale_parts.append(f"Futures showing {futures_trend.lower()} momentum ({futures_change:+.2f}%).")
        
        # Tick momentum insights
        tick_momentum = market_context.get('tick_momentum', {})
        tick_direction = tick_momentum.get('trend_direction', 'Neutral')
        momentum_pct = tick_momentum.get('momentum_pct', 0)
        if tick_direction == direction and abs(momentum_pct) > 0.1:
            rationale_parts.append(f"Tick momentum confirms {tick_direction.lower()} trend ({momentum_pct:+.2f}%).")
        
        return " ".join(rationale_parts)
    
    def _build_decision_logic(self, market_context: Dict, suggestion: Dict, direction: str) -> Dict:
        """
        Build decision logic explaining how the suggestion was generated
        
        Returns:
            Dictionary with decision logic breakdown
        """
        tpo_data = market_context.get('tpo_data', {})
        current_price = market_context.get('current_price')
        order_flow = market_context.get('order_flow', {})
        
        poc = tpo_data.get('poc')
        vah = tpo_data.get('value_area_high')
        val = tpo_data.get('value_area_low')
        
        # TPO Logic
        price_vs_poc = current_price - poc if poc else 0
        price_vs_vah = current_price - vah if vah else 0
        price_vs_val = current_price - val if val else 0
        
        tpo_logic = f"Price ({current_price:.2f}) vs POC ({poc:.2f}): {price_vs_poc:+.2f}. "
        if direction == 'Bullish':
            tpo_logic += f"Price above POC and near/above VAH ({vah:.2f}) suggests bullish continuation."
        elif direction == 'Bearish':
            tpo_logic += f"Price below POC and near/below VAL ({val:.2f}) suggests bearish continuation."
        else:
            tpo_logic += f"Price near POC within Value Area suggests neutral/consolidation."
        
        # Entry Level Logic
        entry_level = suggestion.get('entry_level', current_price)
        entry_logic = f"Entry level set at {entry_level:.2f} based on "
        if entry_level == current_price:
            entry_logic += "current market price."
        elif entry_level == poc:
            entry_logic += f"POC level ({poc:.2f})."
        else:
            entry_logic += f"TPO analysis (current: {current_price:.2f}, POC: {poc:.2f})."
        
        # Stop Loss Logic
        stop_loss = suggestion.get('stop_loss')
        stop_loss_logic = "Stop loss not applicable (options are limited risk)."
        if stop_loss:
            stop_loss_logic = f"Stop loss set at {stop_loss:.2f} based on "
            if 'absorption zone' in suggestion.get('rationale', '').lower():
                stop_loss_logic += "absorption zone from order flow analysis."
            elif 'trapped' in suggestion.get('rationale', '').lower():
                stop_loss_logic += "trapped trader level from order flow analysis."
            else:
                ib_high = tpo_data.get('initial_balance_high')
                ib_low = tpo_data.get('initial_balance_low')
                if direction == 'Bullish' and ib_low:
                    stop_loss_logic += f"IB low ({ib_low:.2f}) or VAL ({val:.2f})."
                elif direction == 'Bearish' and ib_high:
                    stop_loss_logic += f"IB high ({ib_high:.2f}) or VAH ({vah:.2f})."
                else:
                    stop_loss_logic += "TPO-based levels (IB/VAL/VAH)."
        
        # Target Selection Logic
        target_levels = suggestion.get('target_levels', [])
        target_logic = "Targets selected based on "
        if target_levels:
            target_types = [t.get('type') for t in target_levels]
            if 'VAH' in target_types:
                target_logic += f"VAH ({vah:.2f}) for high probability target. "
            if 'VAL' in target_types:
                target_logic += f"VAL ({val:.2f}) for high probability target. "
            if 'Extension' in target_types:
                va_width = vah - val if vah and val else 0
                if direction == 'Bullish':
                    target_logic += f"Extension above VAH ({vah + va_width * 0.3:.2f}) for medium probability target."
                else:
                    target_logic += f"Extension below VAL ({val - va_width * 0.3:.2f}) for medium probability target."
        else:
            target_logic += "TPO value area levels."
        
        # Strike Selection Logic (for options)
        strike_logic = None
        if suggestion.get('derivative_type') in ['CALL', 'PUT']:
            strike_price = suggestion.get('strike_price')
            strike_logic = f"Strike selected at {strike_price} based on "
            if 'trapped' in suggestion.get('rationale', '').lower():
                strike_logic += "trapped trader level from order flow."
            elif 'exhaustion' in suggestion.get('rationale', '').lower():
                strike_logic += "exhaustion signal price from order flow."
            elif 'absorption' in suggestion.get('rationale', '').lower():
                strike_logic += "absorption zone boundary from order flow."
            elif 'max OI' in suggestion.get('rationale', ''):
                strike_logic += "maximum open interest level."
            else:
                if direction == 'Bullish':
                    strike_logic += f"VAH level ({vah:.2f}) rounded to nearest strike."
                else:
                    strike_logic += f"VAL level ({val:.2f}) rounded to nearest strike."
        
        # Confidence Calculation Breakdown
        base_confidence = suggestion.get('confidence_score', 0)  # This is already enhanced
        confidence_breakdown = {
            'base_tpo_confidence': tpo_data.get('confidence_score', 0),
            'order_flow_adjustment': 0,
            'market_bias_adjustment': 0,
            'options_data_adjustment': 0,
            'futures_momentum_adjustment': 0,
            'tick_momentum_adjustment': 0,
            'final_confidence': base_confidence
        }
        
        # Calculate adjustments (simplified - actual calculation is in _calculate_enhanced_confidence)
        if order_flow.get('success'):
            sentiment = order_flow.get('overall_sentiment', {}).get('sentiment', 'Neutral')
            if (direction == 'Bullish' and sentiment == 'Bullish') or (direction == 'Bearish' and sentiment == 'Bearish'):
                confidence_breakdown['order_flow_adjustment'] = 10
            elif (direction == 'Bullish' and sentiment in ['Bearish', 'Strongly_Bearish']) or (direction == 'Bearish' and sentiment in ['Bullish', 'Strongly_Bullish']):
                confidence_breakdown['order_flow_adjustment'] = -5
        
        # Filters Applied
        filters_applied = []
        if suggestion.get('_filtered'):
            filters_applied.append(f"Filtered out: {suggestion.get('_filter_reason', 'unknown')}")
        
        return {
            'tpo_logic': tpo_logic,
            'entry_level_logic': entry_logic,
            'stop_loss_logic': stop_loss_logic,
            'target_selection_logic': target_logic,
            'strike_selection_logic': strike_logic,
            'confidence_breakdown': confidence_breakdown,
            'filters_applied': filters_applied
        }
    
    def _build_decision_context(self, market_context: Dict, suggestion: Dict, direction: str) -> Dict:
        """
        Build decision context with market condition, price position, signal alignment, etc.
        
        Returns:
            Dictionary with decision context
        """
        tpo_data = market_context.get('tpo_data', {})
        current_price = market_context.get('current_price')
        order_flow = market_context.get('order_flow', {})
        
        poc = tpo_data.get('poc')
        vah = tpo_data.get('value_area_high')
        val = tpo_data.get('value_area_low')
        
        # Market Condition
        price_vs_poc = current_price - poc if poc else 0
        if price_vs_poc > 0:
            market_condition = 'Bullish'
        elif price_vs_poc < 0:
            market_condition = 'Bearish'
        else:
            market_condition = 'Neutral'
        
        # Price Position
        price_position = f"Current price ({current_price:.2f}) is "
        if poc:
            if current_price > vah:
                price_position += f"above VAH ({vah:.2f}) - strong bullish position."
            elif current_price > poc:
                price_position += f"above POC ({poc:.2f}) but below VAH ({vah:.2f}) - moderate bullish position."
            elif current_price > val:
                price_position += f"between VAL ({val:.2f}) and POC ({poc:.2f}) - neutral position."
            else:
                price_position += f"below VAL ({val:.2f}) - bearish position."
        
        # Signal Alignment
        signal_alignment = {
            'supporting_signals': [],
            'conflicting_signals': [],
            'neutral_signals': []
        }
        
        # Order flow sentiment
        if order_flow.get('success'):
            sentiment = order_flow.get('overall_sentiment', {}).get('sentiment', 'Neutral')
            if sentiment == direction or (sentiment == 'Strongly_Bullish' and direction == 'Bullish') or (sentiment == 'Strongly_Bearish' and direction == 'Bearish'):
                signal_alignment['supporting_signals'].append(f"Order flow sentiment: {sentiment}")
            elif (sentiment in ['Bearish', 'Strongly_Bearish'] and direction == 'Bullish') or (sentiment in ['Bullish', 'Strongly_Bullish'] and direction == 'Bearish'):
                signal_alignment['conflicting_signals'].append(f"Order flow sentiment: {sentiment}")
            else:
                signal_alignment['neutral_signals'].append(f"Order flow sentiment: {sentiment}")
        
        # Market bias
        market_bias = market_context.get('market_bias', {})
        combined_bias = market_bias.get('combined_bias', {})
        bias_direction = combined_bias.get('bias_direction', 'Neutral')
        if bias_direction == direction:
            signal_alignment['supporting_signals'].append(f"Market bias: {bias_direction}")
        elif bias_direction != 'Neutral':
            signal_alignment['conflicting_signals'].append(f"Market bias: {bias_direction}")
        
        # Risk Factors
        risk_factors = []
        if order_flow.get('success'):
            exhaustion_signals = order_flow.get('exhaustion_signals', {})
            if direction == 'Bullish' and exhaustion_signals.get('buying_exhaustion', 0) > 0:
                risk_factors.append("Buying exhaustion detected - potential top/reversal risk")
            elif direction == 'Bearish' and exhaustion_signals.get('selling_exhaustion', 0) > 0:
                risk_factors.append("Selling exhaustion detected - potential bottom/reversal risk")
        
        # Key Levels
        key_levels = {
            'poc': poc,
            'vah': vah,
            'val': val,
            'current_price': current_price,
            'entry_level': suggestion.get('entry_level'),
            'stop_loss': suggestion.get('stop_loss'),
            'targets': [t.get('level') for t in suggestion.get('target_levels', [])]
        }
        
        # Add trapped trader levels
        if order_flow.get('success'):
            trapped_traders = order_flow.get('trapped_traders', {})
            trapped_buyer_prices = [t.get('futures_price') or t.get('price') for t in trapped_traders.get('trapped_buyers', [])]
            trapped_seller_prices = [t.get('futures_price') or t.get('price') for t in trapped_traders.get('trapped_sellers', [])]
            if trapped_buyer_prices:
                key_levels['trapped_buyer_levels'] = trapped_buyer_prices
            if trapped_seller_prices:
                key_levels['trapped_seller_levels'] = trapped_seller_prices
        
        # Add absorption zones
        if order_flow.get('success'):
            absorption_zones = order_flow.get('absorption_zones', [])
            if absorption_zones:
                key_levels['absorption_zones'] = [
                    {'low': z.get('price_low') or z.get('price'), 'high': z.get('price_high') or z.get('price')}
                    for z in absorption_zones
                ]
        
        return {
            'market_condition': market_condition,
            'price_position': price_position,
            'signal_alignment': signal_alignment,
            'risk_factors': risk_factors,
            'key_levels': key_levels
        }
    
    def generate_order_flow_suggestions(self, analysis_date: str = None, current_price: float = None) -> List[Dict]:
        """
        Generate derivatives trading suggestions based on Order Flow analysis
        
        Args:
            analysis_date: Date for analysis
            current_price: Current market price (optional, will be fetched if not provided)
            
        Returns:
            List of suggestion dictionaries
        """
        if not analysis_date:
            analysis_date = datetime.now().strftime('%Y-%m-%d')
        
        suggestions = []
        
        # Fetch order flow data (primary signal source)
        order_flow_data = self._fetch_order_flow_data(analysis_date)
        
        if not order_flow_data.get('success'):
            logging.warning("No order flow data available for suggestions")
            return suggestions
        
        # Get current price if not provided
        if not current_price:
            # Try to get from order flow data or database
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT last_price FROM my_schema.futures_ticks
                    WHERE instrument_token = %s
                    ORDER BY timestamp DESC LIMIT 1
                """, (self.futures_instrument_token,))
                result = cursor.fetchone()
                if result:
                    current_price = float(result[0])
                conn.close()
            except Exception as e:
                logging.warning(f"Could not fetch current price: {e}")
        
        if not current_price:
            logging.warning("Unable to determine current price for order flow suggestions")
            return suggestions
        
        # Fetch additional data sources for context
        market_bias_data = self._fetch_market_bias_data(analysis_date)
        options_data = self._fetch_options_data(analysis_date, current_price)
        futures_data = self._fetch_futures_data(analysis_date)
        tick_momentum = self._analyze_tick_momentum(analysis_date)
        
        # Get TPO data for reference (but not primary signal)
        pre_market_tpo = self.tpo_analyzer.get_tpo_analysis(analysis_date, 'pre_market')
        live_tpo = self.tpo_analyzer.get_tpo_analysis(analysis_date, 'live')
        tpo_data = live_tpo if live_tpo else pre_market_tpo
        if isinstance(tpo_data, list):
            tpo_data = tpo_data[0] if tpo_data else {}
        
        # Combine all signals into a unified context (order flow is primary)
        market_context = {
            'tpo_data': tpo_data or {},
            'pre_market_tpo': pre_market_tpo,
            'order_flow': order_flow_data,
            'market_bias': market_bias_data,
            'options_data': options_data,
            'futures_data': futures_data,
            'tick_momentum': tick_momentum,
            'current_price': current_price,
            'suggestion_type': 'orderflow'
        }
        
        # Generate suggestions based on order flow signals
        futures_suggestions = self._generate_order_flow_futures_suggestions(market_context)
        suggestions.extend(futures_suggestions)
        
        # Generate options suggestions based on order flow
        options_suggestions = self._generate_order_flow_options_suggestions(market_context)
        suggestions.extend(options_suggestions)
        
        # Apply filtering (less aggressive for order flow-based)
        filtered_suggestions, filtering_stats = self._filter_order_flow_suggestions(suggestions, order_flow_data)
        
        # Add generation diagnostics to filtering stats
        filtering_stats['futures_generated'] = len(futures_suggestions)
        filtering_stats['options_generated'] = len(options_suggestions)
        filtering_stats['generation_issue'] = len(suggestions) == 0
        filtering_stats['suggestion_type'] = 'orderflow'
        
        # Add decision logic and context to each suggestion
        for suggestion in filtered_suggestions:
            direction = 'Bullish' if suggestion.get('action') == 'BUY' else 'Bearish' if suggestion.get('action') == 'SELL' else 'Neutral'
            
            # Build decision logic
            decision_logic = self._build_order_flow_decision_logic(market_context, suggestion, direction)
            suggestion['decision_logic'] = decision_logic
            
            # Add context information
            decision_context = self._build_order_flow_decision_context(market_context, suggestion, direction)
            suggestion['decision_context'] = decision_context
        
        logging.info(f"Order flow suggestions generated: futures={len(futures_suggestions)}, options={len(options_suggestions)}, initial={len(suggestions)}, filtered={len(filtered_suggestions)}")
        
        if len(suggestions) == 0:
            logging.warning("No order flow suggestions generated. Possible reasons: missing order flow data, invalid price, or market conditions don't meet criteria.")
        
        # Store filtering stats in a class variable for API access
        self._last_filtering_stats = filtering_stats
        
        return filtered_suggestions
    
    def _generate_order_flow_futures_suggestions(self, market_context: Dict) -> List[Dict]:
        """
        Generate futures trading suggestions based on order flow signals
        """
        suggestions = []
        order_flow = market_context.get('order_flow', {})
        current_price = market_context.get('current_price')
        
        if not order_flow.get('success') or not current_price:
            return suggestions
        
        # Get order flow signals
        pressure_analysis = order_flow.get('pressure_analysis', {})
        trapped_traders = order_flow.get('trapped_traders', {})
        exhaustion_signals = order_flow.get('exhaustion_signals', {})
        absorption_zones = order_flow.get('absorption_zones', [])
        overall_sentiment = order_flow.get('overall_sentiment', {})
        
        pressure_direction = pressure_analysis.get('pressure_direction', 'Neutral')
        sentiment = overall_sentiment.get('sentiment', 'Neutral')
        
        # Use TPO data for reference levels if available
        tpo_data = market_context.get('tpo_data', {})
        poc = tpo_data.get('poc') if tpo_data else None
        vah = tpo_data.get('value_area_high') if tpo_data else None
        val = tpo_data.get('value_area_low') if tpo_data else None
        
        # Calculate position sizing
        num_lots = 50  # Default
        lot_size = 50
        
        # Generate BUY suggestions based on order flow
        if pressure_direction in ['Strong_buy', 'Buy'] or sentiment in ['Bullish', 'Strongly_Bullish']:
            # Check for trapped sellers (bullish signal)
            trapped_sellers = trapped_traders.get('trapped_sellers', [])
            if trapped_sellers or pressure_direction in ['Strong_buy', 'Buy']:
                entry_level = current_price
                
                # Determine targets based on absorption zones or TPO levels
                target_levels = []
                if absorption_zones:
                    # Use upper boundary of latest absorption zone
                    latest_zone = absorption_zones[-1]
                    zone_high = latest_zone.get('price_high') or latest_zone.get('price')
                    if zone_high and zone_high > entry_level:
                        profit = calculate_futures_profit(entry_level, zone_high, 'BUY', num_lots, lot_size)
                        if profit > 0:
                            target_levels.append({
                                'level': zone_high,
                                'type': 'Absorption Zone',
                                'probability': 'High',
                                'potential_profit': profit
                            })
                
                # Use VAH if available and higher than entry
                if vah and vah > entry_level and not any(t.get('level') == vah for t in target_levels):
                    profit = calculate_futures_profit(entry_level, vah, 'BUY', num_lots, lot_size)
                    if profit > 0:
                        target_levels.append({
                            'level': vah,
                            'type': 'VAH (TPO Reference)',
                            'probability': 'Medium',
                            'potential_profit': profit
                        })
                
                # Use trapped seller levels as targets
                for trapped in trapped_sellers[:3]:  # Top 3 trapped seller levels
                    trapped_price = trapped.get('futures_price') or trapped.get('price')
                    if trapped_price and trapped_price > entry_level:
                        profit = calculate_futures_profit(entry_level, trapped_price, 'BUY', num_lots, lot_size)
                        if profit > 0:
                            target_levels.append({
                                'level': trapped_price,
                                'type': 'Trapped Seller Level',
                                'probability': 'High',
                                'potential_profit': profit
                            })
                
                if target_levels:
                    max_profit = max(t.get('potential_profit', 0) for t in target_levels)
                    margin_info = calculate_futures_margin(entry_level, num_lots)
                    
                    # Determine stop loss
                    stop_loss = None
                    if absorption_zones:
                        latest_zone = absorption_zones[-1]
                        zone_low = latest_zone.get('price_low') or latest_zone.get('price')
                        if zone_low and zone_low < entry_level:
                            stop_loss = zone_low - 5
                    if not stop_loss and val:
                        stop_loss = val - 10
                    elif not stop_loss:
                        stop_loss = entry_level - (entry_level * 0.01)  # 1% stop loss
                    
                    suggestion = {
                        'instrument': 'NIFTY',
                        'derivative_type': 'FUTURES',
                        'action': 'BUY',
                        'entry_level': entry_level,
                        'stop_loss': stop_loss,
                        'target_levels': target_levels,
                        'position_size': f"{num_lots} lots",
                        'max_potential_profit': round(max_profit, 2),
                        'confidence_score': self._calculate_order_flow_confidence(order_flow, 'Bullish'),
                        'rationale': self._build_order_flow_rationale(order_flow, 'Bullish', trapped_sellers),
                        'order_flow_signals': {
                            'pressure_direction': pressure_direction,
                            'sentiment': sentiment,
                            'trapped_sellers_count': len(trapped_sellers),
                            'exhaustion_signals': exhaustion_signals.get('selling_exhaustion', 0)
                        },
                        'required_margin': margin_info,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'source': 'orderflow'
                    }
                    suggestions.append(suggestion)
        
        # Generate SELL suggestions based on order flow
        if pressure_direction in ['Strong_sell', 'Sell'] or sentiment in ['Bearish', 'Strongly_Bearish']:
            # Check for trapped buyers (bearish signal)
            trapped_buyers = trapped_traders.get('trapped_buyers', [])
            if trapped_buyers or pressure_direction in ['Strong_sell', 'Sell']:
                entry_level = current_price
                
                # Determine targets based on absorption zones or TPO levels
                target_levels = []
                if absorption_zones:
                    # Use lower boundary of latest absorption zone
                    latest_zone = absorption_zones[-1]
                    zone_low = latest_zone.get('price_low') or latest_zone.get('price')
                    if zone_low and zone_low < entry_level:
                        profit = calculate_futures_profit(entry_level, zone_low, 'SELL', num_lots, lot_size)
                        if profit > 0:
                            target_levels.append({
                                'level': zone_low,
                                'type': 'Absorption Zone',
                                'probability': 'High',
                                'potential_profit': profit
                            })
                
                # Use VAL if available and lower than entry
                if val and val < entry_level and not any(t.get('level') == val for t in target_levels):
                    profit = calculate_futures_profit(entry_level, val, 'SELL', num_lots, lot_size)
                    if profit > 0:
                        target_levels.append({
                            'level': val,
                            'type': 'VAL (TPO Reference)',
                            'probability': 'Medium',
                            'potential_profit': profit
                        })
                
                # Use trapped buyer levels as targets
                for trapped in trapped_buyers[:3]:  # Top 3 trapped buyer levels
                    trapped_price = trapped.get('futures_price') or trapped.get('price')
                    if trapped_price and trapped_price < entry_level:
                        profit = calculate_futures_profit(entry_level, trapped_price, 'SELL', num_lots, lot_size)
                        if profit > 0:
                            target_levels.append({
                                'level': trapped_price,
                                'type': 'Trapped Buyer Level',
                                'probability': 'High',
                                'potential_profit': profit
                            })
                
                if target_levels:
                    max_profit = max(t.get('potential_profit', 0) for t in target_levels)
                    margin_info = calculate_futures_margin(entry_level, num_lots)
                    
                    # Determine stop loss
                    stop_loss = None
                    if absorption_zones:
                        latest_zone = absorption_zones[-1]
                        zone_high = latest_zone.get('price_high') or latest_zone.get('price')
                        if zone_high and zone_high > entry_level:
                            stop_loss = zone_high + 5
                    if not stop_loss and vah:
                        stop_loss = vah + 10
                    elif not stop_loss:
                        stop_loss = entry_level + (entry_level * 0.01)  # 1% stop loss
                    
                    suggestion = {
                        'instrument': 'NIFTY',
                        'derivative_type': 'FUTURES',
                        'action': 'SELL',
                        'entry_level': entry_level,
                        'stop_loss': stop_loss,
                        'target_levels': target_levels,
                        'position_size': f"{num_lots} lots",
                        'max_potential_profit': round(max_profit, 2),
                        'confidence_score': self._calculate_order_flow_confidence(order_flow, 'Bearish'),
                        'rationale': self._build_order_flow_rationale(order_flow, 'Bearish', trapped_buyers),
                        'order_flow_signals': {
                            'pressure_direction': pressure_direction,
                            'sentiment': sentiment,
                            'trapped_buyers_count': len(trapped_buyers),
                            'exhaustion_signals': exhaustion_signals.get('buying_exhaustion', 0)
                        },
                        'required_margin': margin_info,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'source': 'orderflow'
                    }
                    suggestions.append(suggestion)
        
        return suggestions
    
    def _generate_order_flow_options_suggestions(self, market_context: Dict) -> List[Dict]:
        """
        Generate options trading suggestions based on order flow signals
        """
        suggestions = []
        order_flow = market_context.get('order_flow', {})
        current_price = market_context.get('current_price')
        options_data = market_context.get('options_data', {})
        
        if not order_flow.get('success') or not current_price:
            return suggestions
        
        # Get order flow signals
        pressure_analysis = order_flow.get('pressure_analysis', {})
        trapped_traders = order_flow.get('trapped_traders', {})
        overall_sentiment = order_flow.get('overall_sentiment', {})
        
        pressure_direction = pressure_analysis.get('pressure_direction', 'Neutral')
        sentiment = overall_sentiment.get('sentiment', 'Neutral')
        
        # Round to nearest 50 for Nifty strikes
        def round_to_strike(price):
            return round(price / 50) * 50
        
        num_lots = 2
        lot_size = 50
        estimated_premium_pct = 0.02
        estimated_premium = current_price * estimated_premium_pct
        
        # Generate CALL suggestions
        if pressure_direction in ['Strong_buy', 'Buy'] or sentiment in ['Bullish', 'Strongly_Bullish']:
            trapped_sellers = trapped_traders.get('trapped_sellers', [])
            if trapped_sellers:
                # Use trapped seller level as strike reference
                latest_trapped = trapped_sellers[-1]
                trapped_price = latest_trapped.get('futures_price') or latest_trapped.get('price')
                call_strike = round_to_strike(trapped_price) if trapped_price else round_to_strike(current_price)
            else:
                call_strike = round_to_strike(current_price)
            
            # Determine targets
            target_levels = []
            if trapped_sellers:
                for trapped in trapped_sellers[:2]:
                    trapped_price = trapped.get('futures_price') or trapped.get('price')
                    if trapped_price and trapped_price > call_strike:
                        profit = calculate_options_profit(current_price, trapped_price, call_strike, 
                                                           estimated_premium, 'CALL', num_lots, lot_size)
                        if profit > 0:
                            target_levels.append({
                                'level': trapped_price,
                                'type': 'Trapped Seller Level',
                                'probability': 'High',
                                'potential_profit': profit
                            })
            
            if target_levels:
                max_profit = max(t.get('potential_profit', 0) for t in target_levels)
                margin_info = calculate_options_margin(call_strike, estimated_premium, num_lots, is_long=True)
                
                suggestion = {
                    'instrument': 'NIFTY',
                    'derivative_type': 'CALL',
                    'action': 'BUY',
                    'strike_price': call_strike,
                    'entry_level': current_price,
                    'stop_loss': None,
                    'target_levels': target_levels,
                    'position_size': f'{num_lots} lots',
                    'max_potential_profit': round(max_profit, 2),
                    'confidence_score': self._calculate_order_flow_confidence(order_flow, 'Bullish'),
                    'rationale': f"Order flow shows {pressure_direction.lower()} pressure and {sentiment.lower()} sentiment. Trapped sellers detected ({len(trapped_sellers)} instances) - bullish reversal potential.",
                    'order_flow_signals': {
                        'pressure_direction': pressure_direction,
                        'sentiment': sentiment,
                        'trapped_sellers_count': len(trapped_sellers)
                    },
                    'required_margin': margin_info,
                    'estimated_premium': estimated_premium,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'source': 'orderflow'
                }
                suggestions.append(suggestion)
        
        # Generate PUT suggestions
        if pressure_direction in ['Strong_sell', 'Sell'] or sentiment in ['Bearish', 'Strongly_Bearish']:
            trapped_buyers = trapped_traders.get('trapped_buyers', [])
            if trapped_buyers:
                # Use trapped buyer level as strike reference
                latest_trapped = trapped_buyers[-1]
                trapped_price = latest_trapped.get('futures_price') or latest_trapped.get('price')
                put_strike = round_to_strike(trapped_price) if trapped_price else round_to_strike(current_price)
            else:
                put_strike = round_to_strike(current_price)
            
            # Determine targets
            target_levels = []
            if trapped_buyers:
                for trapped in trapped_buyers[:2]:
                    trapped_price = trapped.get('futures_price') or trapped.get('price')
                    if trapped_price and trapped_price < put_strike:
                        profit = calculate_options_profit(current_price, trapped_price, put_strike, 
                                                           estimated_premium, 'PUT', num_lots, lot_size)
                        if profit > 0:
                            target_levels.append({
                                'level': trapped_price,
                                'type': 'Trapped Buyer Level',
                                'probability': 'High',
                                'potential_profit': profit
                            })
            
            if target_levels:
                max_profit = max(t.get('potential_profit', 0) for t in target_levels)
                margin_info = calculate_options_margin(put_strike, estimated_premium, num_lots, is_long=True)
                
                suggestion = {
                    'instrument': 'NIFTY',
                    'derivative_type': 'PUT',
                    'action': 'BUY',
                    'strike_price': put_strike,
                    'entry_level': current_price,
                    'stop_loss': None,
                    'target_levels': target_levels,
                    'position_size': f'{num_lots} lots',
                    'max_potential_profit': round(max_profit, 2),
                    'confidence_score': self._calculate_order_flow_confidence(order_flow, 'Bearish'),
                    'rationale': f"Order flow shows {pressure_direction.lower()} pressure and {sentiment.lower()} sentiment. Trapped buyers detected ({len(trapped_buyers)} instances) - bearish reversal potential.",
                    'order_flow_signals': {
                        'pressure_direction': pressure_direction,
                        'sentiment': sentiment,
                        'trapped_buyers_count': len(trapped_buyers)
                    },
                    'required_margin': margin_info,
                    'estimated_premium': estimated_premium,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'source': 'orderflow'
                }
                suggestions.append(suggestion)
        
        return suggestions
    
    def _calculate_order_flow_confidence(self, order_flow: Dict, direction: str) -> float:
        """Calculate confidence score based on order flow signals"""
        confidence = 50.0  # Base confidence
        
        pressure_analysis = order_flow.get('pressure_analysis', {})
        pressure_direction = pressure_analysis.get('pressure_direction', 'Neutral')
        pressure_score = pressure_analysis.get('pressure_score', 0)
        
        if direction == 'Bullish':
            if pressure_direction == 'Strong_buy':
                confidence += 25
            elif pressure_direction == 'Buy':
                confidence += 15
            elif pressure_direction == 'Strong_sell':
                confidence -= 20
            elif pressure_direction == 'Sell':
                confidence -= 10
        elif direction == 'Bearish':
            if pressure_direction == 'Strong_sell':
                confidence += 25
            elif pressure_direction == 'Sell':
                confidence += 15
            elif pressure_direction == 'Strong_buy':
                confidence -= 20
            elif pressure_direction == 'Buy':
                confidence -= 10
        
        # Adjust based on pressure score
        confidence += pressure_score * 0.1
        
        # Trapped traders boost confidence
        trapped_traders = order_flow.get('trapped_traders', {})
        if direction == 'Bullish' and trapped_traders.get('trapped_sellers_count', 0) > 0:
            confidence += 10
        elif direction == 'Bearish' and trapped_traders.get('trapped_buyers_count', 0) > 0:
            confidence += 10
        
        return max(0, min(100, confidence))
    
    def _build_order_flow_rationale(self, order_flow: Dict, direction: str, trapped_list: List) -> str:
        """Build rationale based on order flow signals"""
        rationale_parts = []
        
        pressure_analysis = order_flow.get('pressure_analysis', {})
        pressure_direction = pressure_analysis.get('pressure_direction', 'Neutral')
        overall_sentiment = order_flow.get('overall_sentiment', {})
        sentiment = overall_sentiment.get('sentiment', 'Neutral')
        
        rationale_parts.append(f"Order flow analysis shows {pressure_direction.lower()} pressure with {sentiment.lower()} sentiment.")
        
        if trapped_list:
            if direction == 'Bullish':
                rationale_parts.append(f"Trapped sellers detected ({len(trapped_list)} instances) indicating potential bullish reversal.")
            else:
                rationale_parts.append(f"Trapped buyers detected ({len(trapped_list)} instances) indicating potential bearish reversal.")
        
        exhaustion_signals = order_flow.get('exhaustion_signals', {})
        if direction == 'Bullish' and exhaustion_signals.get('selling_exhaustion', 0) > 0:
            rationale_parts.append("Selling exhaustion detected - bullish reversal signal.")
        elif direction == 'Bearish' and exhaustion_signals.get('buying_exhaustion', 0) > 0:
            rationale_parts.append("Buying exhaustion detected - bearish reversal signal.")
        
        return " ".join(rationale_parts)
    
    def _filter_order_flow_suggestions(self, suggestions: List[Dict], order_flow: Dict) -> Tuple[List[Dict], Dict]:
        """Filter order flow suggestions (less aggressive than TPO filtering)"""
        if not order_flow.get('success'):
            return suggestions, {
                'initial_count': len(suggestions),
                'filtered_exhaustion': 0,
                'filtered_sentiment_conflict': 0,
                'filtered_pressure_conflict': 0,
                'filtered_weak_confidence': 0,
                'final_count': len(suggestions)
            }
        
        filtering_stats = {
            'initial_count': len(suggestions),
            'filtered_exhaustion': 0,
            'filtered_sentiment_conflict': 0,
            'filtered_pressure_conflict': 0,
            'filtered_weak_confidence': 0,
            'final_count': 0
        }
        
        filtered_suggestions = []
        exhaustion_signals = order_flow.get('exhaustion_signals', {})
        
        for suggestion in suggestions:
            action = suggestion.get('action', '')
            confidence = suggestion.get('confidence_score', 0)
            should_filter = False
            
            # Only filter if exhaustion strongly conflicts (more lenient than TPO)
            if action == 'BUY' and exhaustion_signals.get('buying_exhaustion', 0) > 2:
                should_filter = True
                filtering_stats['filtered_exhaustion'] += 1
            elif action == 'SELL' and exhaustion_signals.get('selling_exhaustion', 0) > 2:
                should_filter = True
                filtering_stats['filtered_exhaustion'] += 1
            
            # Filter very weak confidence
            if not should_filter and confidence < 30:
                should_filter = True
                filtering_stats['filtered_weak_confidence'] += 1
            
            if not should_filter:
                filtered_suggestions.append(suggestion)
        
        filtering_stats['final_count'] = len(filtered_suggestions)
        return filtered_suggestions, filtering_stats
    
    def _build_order_flow_decision_logic(self, market_context: Dict, suggestion: Dict, direction: str) -> Dict:
        """Build decision logic for order flow-based suggestions"""
        order_flow = market_context.get('order_flow', {})
        current_price = market_context.get('current_price')
        
        pressure_analysis = order_flow.get('pressure_analysis', {})
        trapped_traders = order_flow.get('trapped_traders', {})
        
        entry_logic = f"Entry level set at {suggestion.get('entry_level', current_price):.2f} based on current market price and order flow signals."
        
        stop_loss_logic = "Stop loss not applicable (options are limited risk)."
        if suggestion.get('stop_loss'):
            stop_loss_logic = f"Stop loss set at {suggestion.get('stop_loss'):.2f} based on absorption zones or TPO reference levels."
        
        target_logic = "Targets selected based on trapped trader levels and absorption zones from order flow analysis."
        
        return {
            'entry_level_logic': entry_logic,
            'stop_loss_logic': stop_loss_logic,
            'target_selection_logic': target_logic,
            'primary_signal': 'Order Flow',
            'confidence_breakdown': {
                'base_order_flow_confidence': suggestion.get('confidence_score', 0),
                'final_confidence': suggestion.get('confidence_score', 0)
            }
        }
    
    def _build_order_flow_decision_context(self, market_context: Dict, suggestion: Dict, direction: str) -> Dict:
        """Build decision context for order flow-based suggestions"""
        order_flow = market_context.get('order_flow', {})
        current_price = market_context.get('current_price')
        
        pressure_analysis = order_flow.get('pressure_analysis', {})
        trapped_traders = order_flow.get('trapped_traders', {})
        absorption_zones = order_flow.get('absorption_zones', [])
        
        return {
            'market_condition': pressure_analysis.get('pressure_direction', 'Neutral'),
            'price_position': f"Current price: {current_price:.2f}",
            'signal_alignment': {
                'supporting_signals': [f"Order flow pressure: {pressure_analysis.get('pressure_direction', 'Neutral')}"],
                'conflicting_signals': [],
                'neutral_signals': []
            },
            'risk_factors': [],
            'key_levels': {
                'current_price': current_price,
                'entry_level': suggestion.get('entry_level'),
                'stop_loss': suggestion.get('stop_loss'),
                'targets': [t.get('level') for t in suggestion.get('target_levels', [])],
                'trapped_levels': [t.get('futures_price') or t.get('price') for t in 
                                  (trapped_traders.get('trapped_buyers', []) if direction == 'Bearish' 
                                   else trapped_traders.get('trapped_sellers', []))],
                'absorption_zones': absorption_zones
            }
        }
