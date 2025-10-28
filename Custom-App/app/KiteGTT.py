"""
Kite GTT (Good Till Triggered) Management Module

This module provides functionality to manage GTT orders for Zerodha Kite Connect API.
Includes methods to add, modify, cancel GTT orders, and bulk operations for holdings.
"""

from Boilerplate import *
import logging
import json
from typing import Optional, Dict, List, Any
from datetime import datetime


class KiteGTTManager:
    """Manager class for Kite GTT (Good Till Triggered) orders"""
    
    def __init__(self, kite_instance=None):
        """
        Initialize GTT Manager
        
        Args:
            kite_instance: KiteConnect instance (defaults to global kite)
        """
        self.kite = kite_instance or kite
        self.logger = logging.getLogger(__name__)
    
    def _save_gtt_transaction(self, gtt_data, status='ACTIVE', error_message=None, is_cancel=False):
        """
        Save GTT transaction to database
        
        Args:
            gtt_data: Dictionary containing GTT data
            status: Current status of the GTT
            error_message: Error message if any
            is_cancel: Whether this is a cancellation operation
            
        Returns:
            Transaction ID if successful, None otherwise
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            if is_cancel and gtt_data.get('trigger_id'):
                # Update existing record to mark as cancelled
                cursor.execute("""
                    UPDATE my_schema.gtt_transactions
                    SET status = 'CANCELLED',
                        cancelled_at = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP,
                        notes = %s
                    WHERE trigger_id = %s AND status = 'ACTIVE'
                    RETURNING id
                """, (f"Cancelled at {datetime.now()}", gtt_data['trigger_id']))
                
                result = cursor.fetchone()
                if result:
                    conn.commit()
                    return result[0]
            else:
                # Insert new GTT transaction
                cursor.execute("""
                    INSERT INTO my_schema.gtt_transactions (
                        trigger_id, trading_symbol, exchange, instrument_token, quantity,
                        trigger_price, last_price, order_price, order_type, transaction_type,
                        product, status, gtt_type, response, error_message, notes
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    ) RETURNING id
                """, (
                    gtt_data.get('trigger_id'),
                    gtt_data.get('trading_symbol'),
                    gtt_data.get('exchange'),
                    gtt_data.get('instrument_token'),
                    gtt_data.get('quantity'),
                    gtt_data.get('trigger_price'),
                    gtt_data.get('last_price'),
                    gtt_data.get('order_price'),
                    gtt_data.get('order_type', 'LIMIT'),
                    gtt_data.get('transaction_type', 'SELL'),
                    gtt_data.get('product', 'CNC'),
                    status,
                    gtt_data.get('gtt_type', 'SINGLE'),
                    json.dumps(gtt_data.get('response')) if gtt_data.get('response') else None,
                    error_message,
                    gtt_data.get('notes')
                ))
                
                result = cursor.fetchone()
                conn.commit()
                return result[0] if result else None
                
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to save GTT transaction: {str(e)}")
            return None
    
    def add_gtt(
        self,
        tradingsymbol: str,
        exchange: str,
        quantity: int,
        trigger_price: float,
        last_price: float,
        order_price: float,
        transaction_type: str = kite.TRANSACTION_TYPE_SELL,
        product: str = kite.PRODUCT_CNC,
        order_type: str = kite.ORDER_TYPE_LIMIT
    ) -> Optional[Dict[str, Any]]:
        """
        Add a new GTT (Good Till Triggered) order
        
        Args:
            tradingsymbol: Trading symbol (e.g., "RELIANCE")
            exchange: Exchange name (kite.EXCHANGE_NSE, kite.EXCHANGE_BSE, etc.)
            quantity: Number of shares
            trigger_price: Trigger price for the GTT
            last_price: Current market price
            order_price: Limit price for the stop-loss order
            transaction_type: SELL or BUY (default: SELL)
            product: PRODUCT_CNC (delivery) or PRODUCT_MIS (intraday) (default: PRODUCT_CNC)
            order_type: ORDER_TYPE_LIMIT, ORDER_TYPE_MARKET, etc. (default: ORDER_TYPE_LIMIT)
            
        Returns:
            GTT response dictionary with trigger_id or None if failed
        """
        try:
            orders = [{
                "transaction_type": transaction_type,
                "quantity": quantity,
                "order_type": order_type,
                "product": product,
                "price": order_price
            }]
            
            gtt_response = self.kite.place_gtt(
                trigger_type=kite.GTT_TYPE_SINGLE,
                tradingsymbol=tradingsymbol,
                exchange=exchange,
                trigger_values=[trigger_price],
                last_price=last_price,
                orders=orders
            )
            
            self.logger.info(f"GTT added successfully for {tradingsymbol}: {gtt_response}")
            
            # Save transaction to database
            gtt_data = {
                'trigger_id': gtt_response.get('trigger_id') or gtt_response.get('id'),
                'trading_symbol': tradingsymbol,
                'exchange': exchange,
                'instrument_token': None,  # Can be fetched if needed
                'quantity': quantity,
                'trigger_price': trigger_price,
                'last_price': last_price,
                'order_price': order_price,
                'order_type': order_type,
                'transaction_type': transaction_type,
                'product': product,
                'gtt_type': 'SINGLE',
                'response': gtt_response,
                'notes': f"GTT created for {tradingsymbol}"
            }
            self._save_gtt_transaction(gtt_data, status='ACTIVE')
            
            return gtt_response
            
        except Exception as e:
            self.logger.error(f"Failed to add GTT for {tradingsymbol}: {str(e)}")
            
            # Save failed transaction
            gtt_data = {
                'trading_symbol': tradingsymbol,
                'exchange': exchange,
                'quantity': quantity,
                'trigger_price': trigger_price,
                'last_price': last_price,
                'order_price': order_price,
                'transaction_type': transaction_type,
                'product': product
            }
            self._save_gtt_transaction(gtt_data, status='FAILED', error_message=str(e))
            
            return None
    
    def modify_gtt(
        self,
        trigger_id: int,
        tradingsymbol: str,
        exchange: str,
        quantity: int,
        trigger_price: float,
        last_price: float,
        order_price: float,
        transaction_type: str = kite.TRANSACTION_TYPE_SELL,
        product: str = kite.PRODUCT_CNC,
        order_type: str = kite.ORDER_TYPE_LIMIT
    ) -> Optional[Dict[str, Any]]:
        """
        Modify an existing GTT order
        
        Args:
            trigger_id: Existing GTT trigger ID
            tradingsymbol: Trading symbol
            exchange: Exchange name
            quantity: Number of shares
            trigger_price: New trigger price
            last_price: Current market price
            order_price: Limit price for the stop-loss order
            transaction_type: SELL or BUY
            product: PRODUCT_CNC or PRODUCT_MIS
            order_type: ORDER_TYPE_LIMIT, ORDER_TYPE_MARKET, etc.
            
        Returns:
            Modified GTT response dictionary or None if failed
        """
        try:
            # Cancel existing GTT first
            self.cancel_gtt(trigger_id)
            
            # Add new GTT with updated parameters
            return self.add_gtt(
                tradingsymbol=tradingsymbol,
                exchange=exchange,
                quantity=quantity,
                trigger_price=trigger_price,
                last_price=last_price,
                order_price=order_price,
                transaction_type=transaction_type,
                product=product,
                order_type=order_type
            )
            
        except Exception as e:
            self.logger.error(f"Failed to modify GTT {trigger_id}: {str(e)}")
            return None
    
    def cancel_gtt(self, trigger_id: int) -> bool:
        """
        Cancel a specific GTT order by trigger ID
        
        Args:
            trigger_id: GTT trigger ID to cancel
            
        Returns:
            True if cancelled successfully, False otherwise
        """
        try:
            response = self.kite.delete_gtt(trigger_id)
            self.logger.info(f"GTT {trigger_id} cancelled successfully: {response}")
            
            # Update transaction status in database
            self._save_gtt_transaction({'trigger_id': trigger_id}, status='CANCELLED', is_cancel=True)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cancel GTT {trigger_id}: {str(e)}")
            
            # Update transaction status to show cancellation failure
            self._save_gtt_transaction(
                {'trigger_id': trigger_id}, 
                status='CANCELLATION_FAILED', 
                error_message=str(e),
                is_cancel=True
            )
            
            return False
    
    def cancel_all_gtts(self) -> Dict[str, int]:
        """
        Cancel all active GTT orders
        
        Returns:
            Dictionary with counts: {'success': int, 'failed': int}
        """
        try:
            # Get all active GTT orders
            gtt_orders = self.kite.gtts()
            
            success_count = 0
            failed_count = 0
            
            for gtt_order in gtt_orders:
                trigger_id = gtt_order.get('trigger_id') or gtt_order.get('id')
                if trigger_id:
                    if self.cancel_gtt(trigger_id):
                        success_count += 1
                    else:
                        failed_count += 1
            
            result = {
                'success': success_count,
                'failed': failed_count,
                'total': len(gtt_orders)
            }
            
            self.logger.info(f"Cancelled all GTTs: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to cancel all GTTs: {str(e)}")
            return {'success': 0, 'failed': 0, 'total': 0}
    
    def add_gtt_for_all_holdings(
        self,
        stop_loss_percentage: float = 5.0,
        exchange: str = kite.EXCHANGE_NSE,
        product: str = kite.PRODUCT_CNC,
        overwrite_existing: bool = False
    ) -> Dict[str, Any]:
        """
        Add GTT orders for all current holdings with a stop-loss
        
        Args:
            stop_loss_percentage: Stop-loss percentage (default: 5.0 for 5% loss)
            exchange: Exchange name (default: NSE)
            product: Product type (default: PRODUCT_CNC for delivery holdings)
            overwrite_existing: Whether to overwrite existing GTTs (default: False)
            
        Returns:
            Dictionary with results: {'success': list, 'failed': list, 'skipped': list}
        """
        try:
            # Get current holdings
            holdings = self.kite.holdings()
            
            results = {
                'success': [],
                'failed': [],
                'skipped': [],
                'total': len(holdings)
            }
            
            # Get existing GTTs if overwrite_existing is False
            existing_gtts = {}
            if not overwrite_existing:
                try:
                    gtt_list = self.kite.gtts()
                    for gtt in gtt_list:
                        symbol = gtt.get('tradingsymbol', '')
                        if symbol:
                            existing_gtts[symbol] = gtt.get('trigger_id') or gtt.get('id')
                except Exception as e:
                    self.logger.warning(f"Could not fetch existing GTTs: {e}")
            
            for holding in holdings:
                try:
                    symbol = holding.get('tradingsymbol')
                    quantity = holding.get('quantity', 0)
                    last_price = holding.get('last_price', 0)
                    average_price = holding.get('average_price', 0)
                    
                    if not symbol or quantity <= 0 or last_price <= 0:
                        results['skipped'].append({
                            'symbol': symbol,
                            'reason': 'Invalid holding data'
                        })
                        continue
                    
                    # Check if GTT already exists
                    if symbol in existing_gtts and not overwrite_existing:
                        results['skipped'].append({
                            'symbol': symbol,
                            'reason': 'GTT already exists'
                        })
                        continue
                    
                    # Calculate stop-loss trigger price
                    trigger_price = last_price * (1 - stop_loss_percentage / 100)
                    
                    # Set order price (limit order, 0.1% below trigger)
                    order_price = trigger_price * 0.999
                    
                    # Add GTT
                    gtt_response = self.add_gtt(
                        tradingsymbol=symbol,
                        exchange=exchange,
                        quantity=quantity,
                        trigger_price=round(trigger_price, 2),
                        last_price=last_price,
                        order_price=round(order_price, 2),
                        transaction_type=kite.TRANSACTION_TYPE_SELL,
                        product=product,
                        order_type=kite.ORDER_TYPE_LIMIT
                    )
                    
                    if gtt_response:
                        results['success'].append({
                            'symbol': symbol,
                            'trigger_id': gtt_response.get('trigger_id') or gtt_response.get('id'),
                            'trigger_price': trigger_price
                        })
                    else:
                        results['failed'].append({
                            'symbol': symbol,
                            'reason': 'Failed to add GTT'
                        })
                        
                except Exception as e:
                    self.logger.error(f"Error processing holding {symbol}: {str(e)}")
                    results['failed'].append({
                        'symbol': symbol,
                        'reason': str(e)
                    })
            
            self.logger.info(f"Bulk GTT operation completed: {len(results['success'])} success, "
                           f"{len(results['failed'])} failed, {len(results['skipped'])} skipped")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to add GTT for all holdings: {str(e)}")
            return {'success': [], 'failed': [], 'skipped': [], 'total': 0}
    
    def get_all_gtts(self) -> List[Dict[str, Any]]:
        """
        Get all active GTT orders
        
        Returns:
            List of GTT order dictionaries
        """
        try:
            gtt_list = self.kite.gtts()
            self.logger.info(f"Fetched {len(gtt_list)} GTT orders")
            return gtt_list
            
        except Exception as e:
            self.logger.error(f"Failed to fetch GTT orders: {str(e)}")
            return []
    
    def get_gtt_by_symbol(self, tradingsymbol: str) -> List[Dict[str, Any]]:
        """
        Get all GTT orders for a specific trading symbol
        
        Args:
            tradingsymbol: Trading symbol to search for
            
        Returns:
            List of matching GTT orders
        """
        try:
            all_gtts = self.get_all_gtts()
            matching_gtts = [
                gtt for gtt in all_gtts 
                if gtt.get('tradingsymbol', '').upper() == tradingsymbol.upper()
            ]
            return matching_gtts
            
        except Exception as e:
            self.logger.error(f"Failed to fetch GTT for {tradingsymbol}: {str(e)}")
            return []


# Convenience functions for quick access
def add_gtt_stop_loss(symbol, exchange, quantity, last_price, stop_loss_pct=5.0):
    """Quick function to add a stop-loss GTT"""
    manager = KiteGTTManager()
    trigger_price = last_price * (1 - stop_loss_pct / 100)
    order_price = trigger_price * 0.999
    return manager.add_gtt(
        tradingsymbol=symbol,
        exchange=exchange,
        quantity=quantity,
        trigger_price=round(trigger_price, 2),
        last_price=last_price,
        order_price=round(order_price, 2)
    )


def add_stop_loss_for_all_holdings(stop_loss_pct=5.0):
    """Quick function to add stop-loss GTTs for all holdings"""
    manager = KiteGTTManager()
    return manager.add_gtt_for_all_holdings(stop_loss_percentage=stop_loss_pct)


def cancel_all_gtts():
    """Quick function to cancel all GTT orders"""
    manager = KiteGTTManager()
    return manager.cancel_all_gtts()


if __name__ == "__main__":
    # Example usage
    manager = KiteGTTManager()
    
    # Example: Add GTT for a single stock
    response = manager.add_gtt(
        tradingsymbol="RELIANCE",
        exchange=kite.EXCHANGE_NSE,
        quantity=1,
        trigger_price=2500,
        last_price=2600,
        order_price=2495
    )
    print("GTT Response:", response)
    
    # Example: Add stop-loss GTTs for all holdings
    results = manager.add_gtt_for_all_holdings(stop_loss_percentage=5.0)
    print("Bulk GTT Results:", results)
