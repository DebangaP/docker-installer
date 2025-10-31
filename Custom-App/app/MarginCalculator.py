"""
Margin Calculator Module
Calculates margin requirements for futures, options, and equity positions
"""

import logging
from typing import Dict, Optional
from Boilerplate import get_db_connection, kite

class MarginCalculator:
    """
    Calculate margin requirements for various instruments
    """
    
    def __init__(self):
        """Initialize Margin Calculator"""
        pass
    
    def calculate_futures_margin(self, instrument_token: int, quantity: int, 
                                 entry_price: float, product: str = 'MIS') -> Dict:
        """
        Calculate margin requirement for futures
        
        Args:
            instrument_token: Instrument token
            quantity: Quantity (in lots)
            entry_price: Entry price
            product: Product type (MIS/CNC/NRML)
            
        Returns:
            Dictionary with margin breakdown
        """
        try:
            # Get lot size from instrument (default: 50 for Nifty)
            lot_size = 50  # Default for Nifty/Bank Nifty
            
            # Try to get from Kite API
            try:
                instruments = kite.instruments('NFO')
                instrument_info = [inst for inst in instruments if inst['instrument_token'] == instrument_token]
                if instrument_info:
                    lot_size = instrument_info[0].get('lot_size', 50)
            except Exception as e:
                logging.warning(f"Could not fetch lot size from Kite: {e}, using default 50")
            
            # Total quantity in shares
            total_quantity = quantity * lot_size
            
            # Contract value
            contract_value = entry_price * total_quantity
            
            # Margin calculation (approximate percentages)
            # SPAN margin: ~8-10% of contract value
            # Exposure margin: ~3-5% of contract value
            # Total: ~12-15% for MIS, ~20-25% for NRML
            
            if product == 'MIS':
                span_margin_pct = 0.085  # 8.5%
                exposure_margin_pct = 0.035  # 3.5%
            elif product == 'NRML':
                span_margin_pct = 0.12  # 12%
                exposure_margin_pct = 0.08  # 8%
            else:  # CNC
                span_margin_pct = 0.10
                exposure_margin_pct = 0.05
            
            span_margin = contract_value * span_margin_pct
            exposure_margin = contract_value * exposure_margin_pct
            total_margin = span_margin + exposure_margin
            
            return {
                'instrument_token': instrument_token,
                'quantity': quantity,
                'lot_size': lot_size,
                'total_quantity': total_quantity,
                'entry_price': float(entry_price),
                'contract_value': float(contract_value),
                'span_margin': float(span_margin),
                'exposure_margin': float(exposure_margin),
                'total_margin': float(total_margin),
                'product': product,
                'margin_percentage': float((total_margin / contract_value) * 100)
            }
            
        except Exception as e:
            logging.error(f"Error calculating futures margin: {e}")
            return {
                'error': str(e),
                'total_margin': 0.0
            }
    
    def calculate_options_margin(self, instrument_token: int, quantity: int,
                                strike_price: float, premium: float,
                                option_type: str = 'CE', is_long: bool = True,
                                product: str = 'MIS') -> Dict:
        """
        Calculate margin/premium requirement for options
        
        Args:
            instrument_token: Instrument token
            quantity: Quantity (in lots)
            strike_price: Strike price
            premium: Premium per share
            option_type: 'CE' (Call) or 'PE' (Put)
            is_long: True for buying, False for selling
            product: Product type (MIS/CNC/NRML)
            
        Returns:
            Dictionary with margin/premium breakdown
        """
        try:
            # Get lot size
            lot_size = 50  # Default
            
            try:
                instruments = kite.instruments('NFO')
                instrument_info = [inst for inst in instruments if inst['instrument_token'] == instrument_token]
                if instrument_info:
                    lot_size = instrument_info[0].get('lot_size', 50)
            except Exception as e:
                logging.warning(f"Could not fetch lot size from Kite: {e}")
            
            total_quantity = quantity * lot_size
            premium_per_lot = premium * lot_size
            total_premium = premium_per_lot * quantity
            
            if is_long:
                # Buying options: Pay premium only
                return {
                    'instrument_token': instrument_token,
                    'quantity': quantity,
                    'lot_size': lot_size,
                    'total_quantity': total_quantity,
                    'strike_price': float(strike_price),
                    'premium_per_share': float(premium),
                    'premium_per_lot': float(premium_per_lot),
                    'total_premium': float(total_premium),
                    'option_type': option_type,
                    'direction': 'LONG',
                    'product': product,
                    'margin_type': 'Premium',
                    'total_required': float(total_premium)
                }
            else:
                # Selling options: Requires margin
                contract_value = strike_price * total_quantity
                
                # Margin for short options (higher)
                if product == 'MIS':
                    span_margin = contract_value * 0.15  # 15%
                    exposure_margin = contract_value * 0.05  # 5%
                else:  # NRML
                    span_margin = contract_value * 0.20  # 20%
                    exposure_margin = contract_value * 0.08  # 8%
                
                total_margin = span_margin + exposure_margin
                # Premium received (credit)
                premium_received = total_premium
                
                return {
                    'instrument_token': instrument_token,
                    'quantity': quantity,
                    'lot_size': lot_size,
                    'total_quantity': total_quantity,
                    'strike_price': float(strike_price),
                    'premium_per_share': float(premium),
                    'premium_per_lot': float(premium_per_lot),
                    'premium_received': float(premium_received),
                    'contract_value': float(contract_value),
                    'span_margin': float(span_margin),
                    'exposure_margin': float(exposure_margin),
                    'total_margin': float(total_margin),
                    'total_required': float(total_margin),  # Net requirement (margin - premium received)
                    'option_type': option_type,
                    'direction': 'SHORT',
                    'product': product,
                    'margin_type': 'SPAN + Exposure',
                    'net_required_after_premium': float(max(0, total_margin - premium_received))
                }
                
        except Exception as e:
            logging.error(f"Error calculating options margin: {e}")
            return {
                'error': str(e),
                'total_required': 0.0
            }
    
    def get_available_margin(self) -> Dict:
        """
        Get available margin from database
        
        Returns:
            Dictionary with available margin details
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    margin_type,
                    net,
                    available_cash,
                    available_live_balance,
                    utilised_span,
                    utilised_exposure,
                    utilised_option_premium
                FROM my_schema.margins
                WHERE run_date = (SELECT MAX(run_date) FROM my_schema.margins)
                AND enabled IS TRUE
            """)
            
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                return {
                    'available_cash': 0.0,
                    'available_live_balance': 0.0,
                    'total_utilised': 0.0,
                    'net': 0.0
                }
            
            total_cash = sum(row[2] for row in results if row[2] is not None)
            total_live_balance = sum(row[3] for row in results if row[3] is not None)
            total_utilised = sum(
                (row[4] or 0) + (row[5] or 0) + (row[6] or 0) 
                for row in results
            )
            total_net = sum(row[1] for row in results if row[1] is not None)
            
            return {
                'available_cash': float(total_cash),
                'available_live_balance': float(total_live_balance),
                'total_utilised': float(total_utilised),
                'net': float(total_net),
                'utilization_percentage': float((total_utilised / total_net * 100) if total_net > 0 else 0)
            }
            
        except Exception as e:
            logging.error(f"Error fetching available margin: {e}")
            return {
                'available_cash': 0.0,
                'available_live_balance': 0.0,
                'total_utilised': 0.0,
                'net': 0.0,
                'error': str(e)
            }
    
    def check_margin_sufficiency(self, required_margin: float) -> Dict:
        """
        Check if sufficient margin is available
        
        Args:
            required_margin: Required margin amount
            
        Returns:
            Dictionary with margin check results
        """
        try:
            available = self.get_available_margin()
            
            available_margin = available.get('available_live_balance', 0.0)
            is_sufficient = available_margin >= required_margin
            
            return {
                'required_margin': float(required_margin),
                'available_margin': float(available_margin),
                'is_sufficient': is_sufficient,
                'shortfall': float(max(0, required_margin - available_margin)),
                'utilization_after_trade': float(
                    ((available.get('total_utilised', 0) + required_margin) / available.get('net', 1)) * 100
                    if available.get('net', 0) > 0 else 0
                )
            }
            
        except Exception as e:
            logging.error(f"Error checking margin sufficiency: {e}")
            return {
                'required_margin': float(required_margin),
                'available_margin': 0.0,
                'is_sufficient': False,
                'error': str(e)
            }

