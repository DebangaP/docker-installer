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
    
    def generate_suggestions(self, analysis_date: str = None, current_price: float = None) -> List[Dict]:
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
        
        logging.info(f"Suggestions generated: futures={len(futures_suggestions)}, options={len(options_suggestions)}, total={len(suggestions)}")
        return suggestions
    
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
        """Fetch order flow analysis data"""
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
            return order_flow if order_flow.get('success') else {}
        except Exception as e:
            logging.warning(f"Error fetching order flow data: {e}")
            return {}
    
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
            if direction == 'Bullish' and trapped.get('trapped_sellers', 0) > 0:
                confidence += 5
            elif direction == 'Bearish' and trapped.get('trapped_buyers', 0) > 0:
                confidence += 5
        
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
        
        # Enhance each suggestion with additional signals
        for suggestion in suggestions:
            direction = 'Bullish' if suggestion.get('action') == 'BUY' else 'Bearish' if suggestion.get('action') == 'SELL' else 'Neutral'
            
            # Update confidence with enhanced calculation
            base_confidence = suggestion.get('confidence_score', 0)
            enhanced_confidence = self._calculate_enhanced_confidence(market_context, base_confidence, direction)
            suggestion['confidence_score'] = round(enhanced_confidence, 2)
            
            # Enhance rationale with insights from all sources
            rationale = suggestion.get('rationale', '')
            enhanced_rationale = self._build_enhanced_rationale(market_context, rationale, direction)
            suggestion['rationale'] = enhanced_rationale
            
            # Add additional context
            suggestion['market_signals'] = {
                'order_flow_sentiment': market_context.get('order_flow', {}).get('overall_sentiment', {}).get('sentiment', 'Neutral'),
                'market_bias': market_context.get('market_bias', {}).get('combined_bias', {}).get('bias_direction', 'Neutral'),
                'put_call_ratio': market_context.get('options_data', {}).get('put_call_ratio', 1.0),
                'futures_trend': market_context.get('futures_data', {}).get('trend', 'Neutral'),
                'tick_momentum': market_context.get('tick_momentum', {}).get('trend_direction', 'Neutral')
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
        
        # Enhance strike selection using options data
        if options_data.get('options_chain') is not None and not options_data['options_chain'].empty:
            for suggestion in suggestions:
                # Use max OI strikes if available
                if suggestion.get('derivative_type') == 'CALL':
                    max_oi_strike = options_data.get('max_oi_call_strike')
                    if max_oi_strike and abs(max_oi_strike - suggestion.get('strike_price', 0)) < 100:
                        suggestion['strike_price'] = max_oi_strike
                        suggestion['rationale'] += f" Strike adjusted to max OI level ({max_oi_strike})."
                elif suggestion.get('derivative_type') == 'PUT':
                    max_oi_strike = options_data.get('max_oi_put_strike')
                    if max_oi_strike and abs(max_oi_strike - suggestion.get('strike_price', 0)) < 100:
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
            
            # Add additional context
            suggestion['market_signals'] = {
                'order_flow_sentiment': market_context.get('order_flow', {}).get('overall_sentiment', {}).get('sentiment', 'Neutral'),
                'market_bias': market_context.get('market_bias', {}).get('combined_bias', {}).get('bias_direction', 'Neutral'),
                'put_call_ratio': market_context.get('options_data', {}).get('put_call_ratio', 1.0),
                'futures_trend': market_context.get('futures_data', {}).get('trend', 'Neutral'),
                'tick_momentum': market_context.get('tick_momentum', {}).get('trend_direction', 'Neutral')
            }
        
        return suggestions
    
    def _build_enhanced_rationale(self, market_context: Dict, base_rationale: str, direction: str) -> str:
        """
        Build enhanced rationale by combining insights from all data sources
        """
        rationale_parts = [base_rationale]
        
        # Order flow insights
        order_flow = market_context.get('order_flow', {})
        if order_flow.get('success'):
            sentiment = order_flow.get('overall_sentiment', {})
            sentiment_type = sentiment.get('sentiment', 'Neutral')
            
            if sentiment_type == direction:
                rationale_parts.append(f"Order flow confirms {direction.lower()} sentiment.")
            
            trapped = order_flow.get('trapped_traders', {})
            if direction == 'Bullish' and trapped.get('trapped_sellers', 0) > 0:
                rationale_parts.append(f"Trapped sellers detected ({trapped.get('trapped_sellers', 0)} instances) - potential bullish reversal.")
            elif direction == 'Bearish' and trapped.get('trapped_buyers', 0) > 0:
                rationale_parts.append(f"Trapped buyers detected ({trapped.get('trapped_buyers', 0)} instances) - potential bearish reversal.")
        
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
