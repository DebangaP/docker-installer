"""
Background task for calculating Wyckoff Accumulation/Distribution for holdings.
This runs asynchronously to avoid blocking the dashboard.
"""
import logging
from typing import List, Optional
from datetime import date
from api.services.wyckoff_service import WyckoffService

def calculate_wyckoff_for_holdings(symbols: Optional[List[str]] = None, force_recalculate: bool = False, lookback_days: int = 30):
    """
    Background task to calculate Wyckoff Accumulation/Distribution for holdings.
    
    Args:
        symbols: Optional list of specific symbols to calculate. If None, calculates for all holdings.
        force_recalculate: If True, recalculate even if recent analysis exists
        lookback_days: Number of days to analyze (default: 30)
    """
    try:
        logging.info(f"[Background] Starting Wyckoff calculation for holdings")
        
        wyckoff_service = WyckoffService()
        
        if symbols is None:
            # Calculate for all holdings
            logging.info(f"[Background] Calculating Wyckoff for all holdings")
            results = wyckoff_service.calculate_for_all_holdings(force_recalculate, lookback_days)
        else:
            # Calculate for specific symbols
            logging.info(f"[Background] Calculating Wyckoff for {len(symbols)} symbols")
            results = wyckoff_service.calculate_for_symbols(symbols, force_recalculate, lookback_days)
        
        if results.get('success'):
            processed = results.get('processed', 0)
            success_count = results.get('success_count', 0)
            failed_count = results.get('failed_count', 0)
            
            if symbols is None:
                accumulation_count = results.get('accumulation_count', 0)
                distribution_count = results.get('distribution_count', 0)
                neutral_count = results.get('neutral_count', 0)
                logging.info(f"[Background] Wyckoff calculation completed: {processed} processed, {success_count} successful, {failed_count} failed")
                logging.info(f"[Background] Results: {accumulation_count} accumulation, {distribution_count} distribution, {neutral_count} neutral")
            else:
                logging.info(f"[Background] Wyckoff calculation completed: {processed} processed, {success_count} successful, {failed_count} failed")
        else:
            error = results.get('error', 'Unknown error')
            logging.error(f"[Background] Wyckoff calculation failed: {error}")
            
    except Exception as e:
        logging.error(f"[Background] Error in Wyckoff calculation task: {e}")
        import traceback
        logging.error(traceback.format_exc())

