"""
Wyckoff Accumulation/Distribution Service
Business logic for Wyckoff analysis calculations
"""

from typing import Dict, List, Optional
from datetime import date, datetime
import logging
from common.Boilerplate import get_db_connection
import psycopg2.extras
from indicators.AccumulationDistributionAnalyzer import AccumulationDistributionAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WyckoffService:
    """Service class for Wyckoff Accumulation/Distribution analysis"""
    
    def __init__(self):
        """Initialize the service"""
        self.analyzer = AccumulationDistributionAnalyzer()
    
    def calculate_for_symbols(self, symbols: List[str], force_recalculate: bool = False, lookback_days: int = 30) -> Dict:
        """
        Calculate Wyckoff analysis for a list of stock symbols
        
        Args:
            symbols: List of trading symbols
            force_recalculate: If True, recalculate even if recent analysis exists
            lookback_days: Number of days to analyze (default: 30)
            
        Returns:
            Dictionary with results and statistics
        """
        results = {
            'success': True,
            'total': len(symbols),
            'processed': 0,
            'success_count': 0,
            'failed_count': 0,
            'results': [],
            'errors': []
        }
        
        analysis_date = date.today()
        
        for symbol in symbols:
            try:
                # Check if analysis already exists (unless force_recalculate)
                if not force_recalculate:
                    existing = self.analyzer.get_current_state(symbol, analysis_date)
                    if existing:
                        results['processed'] += 1
                        results['results'].append({
                            'symbol': symbol,
                            'state': existing.get('state'),
                            'confidence': existing.get('confidence_score'),
                            'days_in_state': existing.get('days_in_state'),
                            'status': 'cached',
                            'message': 'Using existing analysis'
                        })
                        continue
                
                # Perform analysis
                logger.info(f"Analyzing {symbol}...")
                result = self.analyzer.analyze_stock(symbol, lookback_days=lookback_days)
                
                if result:
                    # Save to database
                    success = self.analyzer.save_analysis_result(symbol, analysis_date, result)
                    
                    if success:
                        results['processed'] += 1
                        results['success_count'] += 1
                        results['results'].append({
                            'symbol': symbol,
                            'state': result.get('state'),
                            'confidence': result.get('confidence_score'),
                            'days_in_state': result.get('days_in_state', 0),
                            'pattern_detected': result.get('pattern_detected'),
                            'status': 'success'
                        })
                    else:
                        results['processed'] += 1
                        results['failed_count'] += 1
                        results['errors'].append({
                            'symbol': symbol,
                            'error': 'Failed to save analysis result'
                        })
                else:
                    results['processed'] += 1
                    results['failed_count'] += 1
                    results['errors'].append({
                        'symbol': symbol,
                        'error': 'Insufficient data for analysis'
                    })
                    
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                results['processed'] += 1
                results['failed_count'] += 1
                results['errors'].append({
                    'symbol': symbol,
                    'error': str(e)
                })
        
        return results
    
    def calculate_for_all_holdings(self, force_recalculate: bool = False, lookback_days: int = 30) -> Dict:
        """
        Calculate Wyckoff analysis for all holdings
        
        Args:
            force_recalculate: If True, recalculate even if recent analysis exists
            lookback_days: Number of days to analyze (default: 30)
            
        Returns:
            Dictionary with summary statistics
        """
        # Get all holdings symbols
        from api.services.holdings_service import HoldingsService
        holdings_service = HoldingsService()
        symbols = holdings_service.get_all_holdings_symbols()
        
        if not symbols:
            return {
                'success': False,
                'error': 'No holdings found',
                'total_holdings': 0
            }
        
        # Calculate for all symbols
        results = self.calculate_for_symbols(symbols, force_recalculate, lookback_days)
        
        # Add summary statistics
        accumulation_count = sum(1 for r in results['results'] if r.get('state') == 'ACCUMULATION')
        distribution_count = sum(1 for r in results['results'] if r.get('state') == 'DISTRIBUTION')
        neutral_count = sum(1 for r in results['results'] if r.get('state') == 'NEUTRAL')
        
        results['total_holdings'] = len(symbols)
        results['accumulation_count'] = accumulation_count
        results['distribution_count'] = distribution_count
        results['neutral_count'] = neutral_count
        
        return results
    
    def get_analysis_status(self) -> Dict:
        """
        Get status of Wyckoff analysis for all holdings
        
        Returns:
            Dictionary with analysis status for each holding
        """
        try:
            # Get all holdings
            from api.services.holdings_service import HoldingsService
            holdings_service = HoldingsService()
            symbols = holdings_service.get_all_holdings_symbols()
            
            if not symbols:
                return {
                    'success': True,
                    'total_holdings': 0,
                    'analyzed': 0,
                    'not_analyzed': 0,
                    'holdings_status': []
                }
            
            # Get latest analysis date
            conn = get_db_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cursor.execute("""
                SELECT MAX(analysis_date) as last_analysis_date
                FROM my_schema.accumulation_distribution
            """)
            
            result = cursor.fetchone()
            last_analysis_date = result['last_analysis_date'] if result and result['last_analysis_date'] else None
            
            # Get status for each holding
            holdings_status = []
            analyzed_count = 0
            not_analyzed_count = 0
            
            analysis_date = date.today()
            
            for symbol in symbols:
                state_data = self.analyzer.get_current_state(symbol, analysis_date)
                
                if state_data:
                    analyzed_count += 1
                    holdings_status.append({
                        'symbol': symbol,
                        'has_analysis': True,
                        'last_analysis_date': str(state_data.get('start_date', analysis_date)),
                        'state': state_data.get('state'),
                        'confidence': state_data.get('confidence_score'),
                        'days_in_state': state_data.get('days_in_state')
                    })
                else:
                    not_analyzed_count += 1
                    holdings_status.append({
                        'symbol': symbol,
                        'has_analysis': False,
                        'last_analysis_date': None,
                        'state': None
                    })
            
            cursor.close()
            conn.close()
            
            return {
                'success': True,
                'total_holdings': len(symbols),
                'analyzed': analyzed_count,
                'not_analyzed': not_analyzed_count,
                'last_analysis_date': str(last_analysis_date) if last_analysis_date else None,
                'holdings_status': holdings_status
            }
            
        except Exception as e:
            logger.error(f"Error getting analysis status: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_missing_analyses(self) -> List[str]:
        """
        Get list of holdings that don't have analysis
        
        Returns:
            List of trading symbols without analysis
        """
        try:
            status = self.get_analysis_status()
            if not status.get('success'):
                return []
            
            missing = [
                h['symbol'] for h in status.get('holdings_status', [])
                if not h.get('has_analysis', False)
            ]
            
            return missing
            
        except Exception as e:
            logger.error(f"Error getting missing analyses: {e}")
            return []

