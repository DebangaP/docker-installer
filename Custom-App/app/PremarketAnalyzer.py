"""
Premarket TPO Profile Analysis Module
Enhanced analysis with overnight data comparison, support/resistance zones, and directional context
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, Optional, List, Tuple
import logging
from CalculateTPO import TPOProfile, PostgresDataFetcher
from MarketBiasAnalyzer import MarketBiasAnalyzer


def convert_numpy_to_native(value):
    """Convert NumPy types to native Python types"""
    if value is None:
        return None
    
    try:
        if hasattr(value, 'item'):
            return value.item()
    except (AttributeError, ValueError):
        pass
    
    if isinstance(value, (np.integer, np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32)):
        return int(value)
    elif isinstance(value, (np.floating, np.float64, np.float32, np.float16)):
        return float(value)
    elif isinstance(value, np.bool_):
        return bool(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, (pd.Timestamp, datetime)):
        return value
    else:
        try:
            if isinstance(value, (int, float, bool)):
                return value
            return float(value) if '.' in str(value) else int(value)
        except (ValueError, TypeError):
            return value


class PremarketAnalyzer:
    """
    Enhanced Premarket TPO Profile Analysis
    Provides comprehensive day-start directional context and auction structure basis
    """
    
    def __init__(self, db_fetcher: PostgresDataFetcher, instrument_token: int = 256265, tick_size: float = 5.0):
        """
        Initialize Premarket Analyzer
        
        Args:
            db_fetcher: Database fetcher instance
            instrument_token: Instrument token for analysis (default: 256265 for Nifty 50)
            tick_size: Price tick size for TPO calculation
        """
        self.db_fetcher = db_fetcher
        self.instrument_token = instrument_token
        self.tick_size = tick_size
        self.market_bias_analyzer = MarketBiasAnalyzer(db_fetcher, instrument_token, tick_size)
    
    def get_prior_day_market_structure(self, analysis_date: str) -> Optional[Dict]:
        """
        Fetch prior day's market structure (POC, VA, IB) from database
        
        Args:
            analysis_date: Current analysis date
            
        Returns:
            Dictionary with prior day structure or None
        """
        try:
            from Boilerplate import get_db_connection
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get previous trading day
            prev_date = (datetime.strptime(analysis_date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
            
            # Try to get market structure from market_structure table
            cursor.execute("""
                SELECT poc, vah, val, ib_high, ib_low, overnight_high, overnight_low
                FROM my_schema.market_structure
                WHERE session_date = %s AND instrument_token = %s
                ORDER BY timestamp DESC
                LIMIT 1
            """, (prev_date, self.instrument_token))
            
            row = cursor.fetchone()
            
            if row:
                result = {
                    'poc': float(row[0]) if row[0] else None,
                    'vah': float(row[1]) if row[1] else None,
                    'val': float(row[2]) if row[2] else None,
                    'ib_high': float(row[3]) if row[3] else None,
                    'ib_low': float(row[4]) if row[4] else None,
                    'overnight_high': float(row[5]) if row[5] else None,
                    'overnight_low': float(row[6]) if row[6] else None
                }
                cursor.close()
                conn.close()
                return result
            
            # Fallback: Try to get from TPO analysis table
            cursor.execute("""
                SELECT poc, value_area_high, value_area_low, initial_balance_high, initial_balance_low
                FROM my_schema.tpo_analysis
                WHERE analysis_date = %s AND instrument_token = %s AND session_type = 'live'
                ORDER BY created_at DESC
                LIMIT 1
            """, (prev_date, self.instrument_token))
            
            row = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if row:
                return {
                    'poc': float(row[0]) if row[0] else None,
                    'vah': float(row[1]) if row[1] else None,
                    'val': float(row[2]) if row[2] else None,
                    'ib_high': float(row[3]) if row[3] else None,
                    'ib_low': float(row[4]) if row[4] else None,
                    'overnight_high': None,
                    'overnight_low': None
                }
            
            return None
            
        except Exception as e:
            logging.error(f"Error fetching prior day structure: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return None
    
    def get_overnight_data(self, analysis_date: str) -> Dict:
        """
        Get overnight high/low from pre-market ticks
        
        Args:
            analysis_date: Current analysis date
            
        Returns:
            Dictionary with overnight_high and overnight_low
        """
        try:
            # Fetch pre-market ticks (before 9:15 AM on analysis date)
            # Overnight period: Prior day close (15:30) to current day 9:15 AM
            prev_date = (datetime.strptime(analysis_date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
            
            # Get ticks from prior day close onwards
            overnight_start = f'{prev_date} 15:30:00.000'
            overnight_end = f'{analysis_date} 09:15:00.000'
            
            overnight_data = self.db_fetcher.fetch_tick_data(
                table_name='ticks',
                instrument_token=self.instrument_token,
                start_time=overnight_start,
                end_time=overnight_end
            )
            
            if overnight_data.empty:
                # Fallback: Just get pre-market data (9:05-9:15)
                pre_market_data = self.db_fetcher.fetch_tick_data(
                    table_name='ticks',
                    instrument_token=self.instrument_token,
                    start_time=f'{analysis_date} 09:05:00.000',
                    end_time=f'{analysis_date} 09:15:00.000'
                )
                
                if not pre_market_data.empty and 'last_price' in pre_market_data.columns:
                    return {
                        'overnight_high': float(pre_market_data['last_price'].max()),
                        'overnight_low': float(pre_market_data['last_price'].min()),
                        'overnight_open': float(pre_market_data['last_price'].iloc[0]) if len(pre_market_data) > 0 else None,
                        'overnight_close': float(pre_market_data['last_price'].iloc[-1]) if len(pre_market_data) > 0 else None
                    }
                return {'overnight_high': None, 'overnight_low': None, 'overnight_open': None, 'overnight_close': None}
            
            if 'last_price' in overnight_data.columns:
                return {
                    'overnight_high': float(overnight_data['last_price'].max()),
                    'overnight_low': float(overnight_data['last_price'].min()),
                    'overnight_open': float(overnight_data['last_price'].iloc[0]) if len(overnight_data) > 0 else None,
                    'overnight_close': float(overnight_data['last_price'].iloc[-1]) if len(overnight_data) > 0 else None
                }
            
            return {'overnight_high': None, 'overnight_low': None, 'overnight_open': None, 'overnight_close': None}
            
        except Exception as e:
            logging.error(f"Error fetching overnight data: {e}")
            return {'overnight_high': None, 'overnight_low': None, 'overnight_open': None, 'overnight_close': None}
    
    def calculate_support_resistance_zones(self, tpo_profile: TPOProfile, prior_day: Optional[Dict], overnight: Dict) -> Dict:
        """
        Calculate key support and resistance zones from pre-market TPO
        
        Args:
            tpo_profile: Pre-market TPO profile
            prior_day: Prior day market structure
            overnight: Overnight data
            
        Returns:
            Dictionary with support/resistance zones
        """
        zones = {
            'support_levels': [],
            'resistance_levels': [],
            'pivot_levels': [],
            'target_zones': []
        }
        
        if tpo_profile.tpo_data is None or tpo_profile.tpo_data.empty:
            return zones
        
        # POC as key pivot
        if tpo_profile.poc:
            zones['pivot_levels'].append({
                'price': convert_numpy_to_native(tpo_profile.poc),
                'type': 'POC',
                'strength': 'High',
                'description': 'Point of Control - Most traded price'
            })
        
        # Value Area edges as support/resistance
        if tpo_profile.value_area_low:
            zones['support_levels'].append({
                'price': convert_numpy_to_native(tpo_profile.value_area_low),
                'type': 'VAL',
                'strength': 'High',
                'description': 'Value Area Low - Key support level'
            })
        
        if tpo_profile.value_area_high:
            zones['resistance_levels'].append({
                'price': convert_numpy_to_native(tpo_profile.value_area_high),
                'type': 'VAH',
                'strength': 'High',
                'description': 'Value Area High - Key resistance level'
            })
        
        # Initial Balance edges
        if tpo_profile.initial_balance_low:
            zones['support_levels'].append({
                'price': convert_numpy_to_native(tpo_profile.initial_balance_low),
                'type': 'IBL',
                'strength': 'Medium',
                'description': 'Initial Balance Low - Early session support'
            })
        
        if tpo_profile.initial_balance_high:
            zones['resistance_levels'].append({
                'price': convert_numpy_to_native(tpo_profile.initial_balance_high),
                'type': 'IBH',
                'strength': 'Medium',
                'description': 'Initial Balance High - Early session resistance'
            })
        
        # Overnight levels
        if overnight.get('overnight_high'):
            zones['resistance_levels'].append({
                'price': overnight['overnight_high'],
                'type': 'Overnight High',
                'strength': 'Medium',
                'description': 'Overnight session high - Potential resistance'
            })
        
        if overnight.get('overnight_low'):
            zones['support_levels'].append({
                'price': overnight['overnight_low'],
                'type': 'Overnight Low',
                'strength': 'Medium',
                'description': 'Overnight session low - Potential support'
            })
        
        # Prior day levels for context
        if prior_day:
            if prior_day.get('poc'):
                zones['pivot_levels'].append({
                    'price': prior_day['poc'],
                    'type': 'Prior POC',
                    'strength': 'High',
                    'description': 'Previous day POC - Key reference level'
                })
            
            if prior_day.get('vah'):
                zones['resistance_levels'].append({
                    'price': prior_day['vah'],
                    'type': 'Prior VAH',
                    'strength': 'Medium',
                    'description': 'Previous day Value Area High'
                })
            
            if prior_day.get('val'):
                zones['support_levels'].append({
                    'price': prior_day['val'],
                    'type': 'Prior VAL',
                    'strength': 'Medium',
                    'description': 'Previous day Value Area Low'
                })
        
        # Sort and remove duplicates (keep highest strength)
        def sort_zones(zone_list):
            seen_prices = {}
            for zone in zone_list:
                price = zone['price']
                strength_val = {'High': 3, 'Medium': 2, 'Low': 1}.get(zone['strength'], 0)
                
                if price not in seen_prices or strength_val > seen_prices[price]['strength_val']:
                    seen_prices[price] = {
                        'zone': zone,
                        'strength_val': strength_val
                    }
            
            return [v['zone'] for v in sorted(seen_prices.values(), key=lambda x: x['zone']['price'], reverse=True)]
        
        zones['support_levels'] = sort_zones(zones['support_levels'])
        zones['resistance_levels'] = sort_zones(zones['resistance_levels'])
        zones['pivot_levels'] = sort_zones(zones['pivot_levels'])
        
        # Calculate target zones based on VA and overnight range
        if tpo_profile.value_area_low and tpo_profile.value_area_high:
            va_range = tpo_profile.value_area_high - tpo_profile.value_area_low
            zones['target_zones'] = [
                {
                    'price': convert_numpy_to_native(tpo_profile.value_area_low - va_range * 0.382),
                    'type': 'Support Target',
                    'description': 'Fibonacci extension below VAL'
                },
                {
                    'price': convert_numpy_to_native(tpo_profile.value_area_high + va_range * 0.382),
                    'type': 'Resistance Target',
                    'description': 'Fibonacci extension above VAH'
                }
            ]
        
        return zones
    
    def analyze_gap_vs_prior_day(self, overnight: Dict, prior_day: Optional[Dict], premarket_poc: Optional[float]) -> Dict:
        """
        Analyze gap between overnight/premarket and prior day structure
        
        Args:
            overnight: Overnight data
            prior_day: Prior day structure
            premarket_poc: Pre-market POC
            
        Returns:
            Gap analysis dictionary
        """
        gap_analysis = {
            'gap_type': 'No Gap',
            'gap_size': 0.0,
            'gap_percentage': 0.0,
            'inventory_position': 'Neutral',
            'gap_description': ''
        }
        
        if not prior_day or not prior_day.get('vah') or not prior_day.get('val'):
            return gap_analysis
        
        prior_va_mid = (prior_day['vah'] + prior_day['val']) / 2
        prior_poc = prior_day.get('poc', prior_va_mid)
        
        # Determine reference price (use premarket POC if available, otherwise overnight close)
        reference_price = None
        if premarket_poc:
            reference_price = premarket_poc
        elif overnight.get('overnight_close'):
            reference_price = overnight['overnight_close']
        elif overnight.get('overnight_high') and overnight.get('overnight_low'):
            reference_price = (overnight['overnight_high'] + overnight['overnight_low']) / 2
        
        if not reference_price:
            return gap_analysis
        
        gap_size = reference_price - prior_day['poc'] if prior_day.get('poc') else reference_price - prior_va_mid
        gap_percentage = (gap_size / prior_poc * 100) if prior_poc > 0 else 0.0
        
        # Determine gap type
        if gap_percentage > 0.3:
            gap_analysis['gap_type'] = 'Gap Up'
            if gap_percentage > 1.0:
                gap_analysis['gap_type'] = 'Strong Gap Up'
            gap_analysis['inventory_position'] = 'Net Long (Risk of Liquidation)'
        elif gap_percentage < -0.3:
            gap_analysis['gap_type'] = 'Gap Down'
            if gap_percentage < -1.0:
                gap_analysis['gap_type'] = 'Strong Gap Down'
            gap_analysis['inventory_position'] = 'Net Short (Risk of Liquidation)'
        else:
            gap_analysis['gap_type'] = 'No Gap / Within Range'
            gap_analysis['inventory_position'] = 'Balanced'
        
        gap_analysis['gap_size'] = convert_numpy_to_native(abs(gap_size))
        gap_analysis['gap_percentage'] = convert_numpy_to_native(abs(gap_percentage))
        
        # Check if gap is within prior day VA
        if prior_day.get('val') and prior_day.get('vah'):
            if prior_day['val'] <= reference_price <= prior_day['vah']:
                gap_analysis['gap_description'] = f"Opening within prior day Value Area ({prior_day['val']:.2f} - {prior_day['vah']:.2f})"
            elif reference_price > prior_day['vah']:
                gap_analysis['gap_description'] = f"Opening above prior day Value Area High ({prior_day['vah']:.2f})"
            else:
                gap_analysis['gap_description'] = f"Opening below prior day Value Area Low ({prior_day['val']:.2f})"
        
        return gap_analysis
    
    def generate_comprehensive_premarket_analysis(self, analysis_date: str = None) -> Dict:
        """
        Generate comprehensive pre-market analysis with all enhancements
        
        Args:
            analysis_date: Date for analysis in 'YYYY-MM-DD' format
            
        Returns:
            Dictionary containing comprehensive pre-market analysis
        """
        if not analysis_date:
            analysis_date = datetime.now().strftime('%Y-%m-%d')
        
        # Fetch pre-market data
        pre_market_data = self.db_fetcher.fetch_tick_data(
            table_name='ticks',
            instrument_token=self.instrument_token,
            start_time=f'{analysis_date} 09:05:00.000',
            end_time=f'{analysis_date} 09:15:00.000'
        )
        
        if pre_market_data.empty:
            logging.warning(f"No pre-market data found for {analysis_date}")
            return {
                'success': False,
                'error': 'No pre-market data available',
                'analysis_date': analysis_date,
                'missing_data': [
                    'ticks data for 9:05 AM - 9:15 AM time period',
                    'price data (open, high, low, close)',
                    'volume data for TPO calculation',
                    'timestamp data within the pre-market window'
                ]
            }
        
        # Calculate pre-market TPO profile
        tpo_profile = TPOProfile(tick_size=self.tick_size)
        tpo_profile.calculate_tpo(pre_market_data)
        
        # Get prior day structure
        prior_day = self.get_prior_day_market_structure(analysis_date)
        
        # Get overnight data
        overnight = self.get_overnight_data(analysis_date)
        
        # Calculate support/resistance zones
        zones = self.calculate_support_resistance_zones(tpo_profile, prior_day, overnight)
        
        # Analyze gap
        gap_analysis = self.analyze_gap_vs_prior_day(overnight, prior_day, convert_numpy_to_native(tpo_profile.poc))
        
        # Get market bias
        bias_analysis = self.market_bias_analyzer.analyze_pre_market_bias(analysis_date)
        
        # Generate directional context summary
        directional_context = self._generate_directional_context(
            tpo_profile, prior_day, overnight, gap_analysis, bias_analysis
        )
        
        # Compile comprehensive result
        result = {
            'success': True,
            'analysis_date': analysis_date,
            'instrument_token': self.instrument_token,
            'timestamp': datetime.now().isoformat(),
            
            # Pre-market TPO metrics
            'premarket_tpo': {
                'poc': convert_numpy_to_native(tpo_profile.poc),
                'poc_high': convert_numpy_to_native(tpo_profile.poc_high),
                'poc_low': convert_numpy_to_native(tpo_profile.poc_low),
                'value_area_high': convert_numpy_to_native(tpo_profile.value_area_high),
                'value_area_low': convert_numpy_to_native(tpo_profile.value_area_low),
                'initial_balance_high': convert_numpy_to_native(tpo_profile.initial_balance_high),
                'initial_balance_low': convert_numpy_to_native(tpo_profile.initial_balance_low),
                'session_range': convert_numpy_to_native(
                    tpo_profile.value_area_high - tpo_profile.value_area_low
                ) if (tpo_profile.value_area_high and tpo_profile.value_area_low) else None
            },
            
            # Prior day comparison
            'prior_day_structure': prior_day,
            
            # Overnight data
            'overnight_data': overnight,
            
            # Gap analysis
            'gap_analysis': gap_analysis,
            
            # Support/Resistance zones
            'key_zones': zones,
            
            # Market bias
            'market_bias': {
                'bias_score': convert_numpy_to_native(bias_analysis.get('bias_score', 0.0)),
                'bias_direction': bias_analysis.get('bias_direction', 'Neutral'),
                'bias_strength': bias_analysis.get('bias_strength', 'Weak')
            },
            
            # Directional context
            'directional_context': directional_context,
            
            # Auction structure basis
            'auction_structure': {
                'value_area_range': convert_numpy_to_native(
                    tpo_profile.value_area_high - tpo_profile.value_area_low
                ) if (tpo_profile.value_area_high and tpo_profile.value_area_low) else None,
                'poc_strength': self._calculate_poc_strength(tpo_profile),
                'balance_quality': self._assess_balance_quality(tpo_profile)
            }
        }
        
        return result
    
    def _generate_directional_context(self, tpo_profile: TPOProfile, prior_day: Optional[Dict], 
                                     overnight: Dict, gap_analysis: Dict, bias_analysis: Dict) -> Dict:
        """
        Generate directional context summary for day-start
        
        Args:
            tpo_profile: Pre-market TPO profile
            prior_day: Prior day structure
            overnight: Overnight data
            gap_analysis: Gap analysis
            bias_analysis: Market bias analysis
            
        Returns:
            Directional context dictionary
        """
        context = {
            'opening_bias': 'Neutral',
            'key_levels_to_watch': [],
            'opening_expectations': [],
            'risk_factors': []
        }
        
        # Determine opening bias from gap and bias analysis
        bias_direction = bias_analysis.get('bias_direction', 'Neutral')
        gap_type = gap_analysis.get('gap_type', 'No Gap')
        
        if gap_type.startswith('Gap Up') or (gap_type == 'No Gap' and bias_direction == 'Bullish'):
            context['opening_bias'] = 'Bullish'
        elif gap_type.startswith('Gap Down') or (gap_type == 'No Gap' and bias_direction == 'Bearish'):
            context['opening_bias'] = 'Bearish'
        
        # Key levels to watch
        if tpo_profile.value_area_high:
            context['key_levels_to_watch'].append({
                'level': convert_numpy_to_native(tpo_profile.value_area_high),
                'type': 'Premarket VAH',
                'significance': 'Break above suggests bullish continuation'
            })
        
        if tpo_profile.value_area_low:
            context['key_levels_to_watch'].append({
                'level': convert_numpy_to_native(tpo_profile.value_area_low),
                'type': 'Premarket VAL',
                'significance': 'Break below suggests bearish continuation'
            })
        
        if prior_day and prior_day.get('poc'):
            context['key_levels_to_watch'].append({
                'level': prior_day['poc'],
                'type': 'Prior Day POC',
                'significance': 'Key reference level from previous session'
            })
        
        # Opening expectations
        if gap_analysis.get('inventory_position', '') == 'Net Long (Risk of Liquidation)':
            context['opening_expectations'].append('Watch for liquidation risk - gap may fill')
            context['risk_factors'].append('Strong gap up creates liquidation risk')
        elif gap_analysis.get('inventory_position', '') == 'Net Short (Risk of Liquidation)':
            context['opening_expectations'].append('Watch for short covering - gap may fill')
            context['risk_factors'].append('Strong gap down creates short covering risk')
        
        if bias_direction == 'Bullish':
            context['opening_expectations'].append('Pre-market bias suggests bullish opening')
        elif bias_direction == 'Bearish':
            context['opening_expectations'].append('Pre-market bias suggests bearish opening')
        
        return context
    
    def _calculate_poc_strength(self, tpo_profile: TPOProfile) -> str:
        """Calculate POC strength indicator"""
        if not tpo_profile.tpo_data or tpo_profile.tpo_data.empty:
            return 'Weak'
        
        max_tpo = float(tpo_profile.tpo_data['tpo_count'].max())
        total_tpos = float(tpo_profile.tpo_data['tpo_count'].sum())
        
        if total_tpos == 0:
            return 'Weak'
        
        poc_ratio = max_tpo / total_tpos
        
        if poc_ratio > 0.15:
            return 'Strong'
        elif poc_ratio > 0.10:
            return 'Moderate'
        else:
            return 'Weak'
    
    def _assess_balance_quality(self, tpo_profile: TPOProfile) -> str:
        """Assess quality of balance in pre-market"""
        if not tpo_profile.value_area_high or not tpo_profile.value_area_low:
            return 'Unknown'
        
        va_range = tpo_profile.value_area_high - tpo_profile.value_area_low
        poc = tpo_profile.poc
        
        if not poc:
            return 'Unknown'
        
        # Check if POC is well-centered in VA
        va_center = (tpo_profile.value_area_high + tpo_profile.value_area_low) / 2
        poc_offset = abs(poc - va_center) / va_range if va_range > 0 else 0
        
        if poc_offset < 0.15:
            return 'Well Balanced'
        elif poc_offset < 0.30:
            return 'Moderately Balanced'
        else:
            return 'Unbalanced'
