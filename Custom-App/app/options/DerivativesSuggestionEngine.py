"""
Derivatives Trade Suggestion Engine
Analyzes TPO levels to suggest futures and options trades
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
from options.DerivativesTPOAnalyzer import DerivativesTPOAnalyzer

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
    Generates derivatives trading suggestions based on TPO analysis
    """
    
    def __init__(self, tpo_analyzer: DerivativesTPOAnalyzer):
        """
        Initialize suggestion engine
        
        Args:
            tpo_analyzer: DerivativesTPOAnalyzer instance
        """
        self.tpo_analyzer = tpo_analyzer
    
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

        # Generate futures suggestions
        futures_suggestions = self._generate_futures_suggestions(tpo_data, current_price, pre_market_tpo)
        suggestions.extend(futures_suggestions)
        
        # Generate options suggestions
        options_suggestions = self._generate_options_suggestions(tpo_data, current_price, pre_market_tpo)
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
