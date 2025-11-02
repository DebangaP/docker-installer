"""
Micro Level Detector
Identifies critical price levels from order flow and volume analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from common.Boilerplate import get_db_connection
from market.FootprintChartGenerator import FootprintChartGenerator


class MicroLevelDetector:
    """
    Detect critical micro price levels from footprint and order flow data
    """
    
    def __init__(self, instrument_token: int = 256265, price_bucket_size: float = 5.0):
        """
        Initialize Micro Level Detector
        
        Args:
            instrument_token: Instrument token (default: 256265 for Nifty 50)
            price_bucket_size: Size of price buckets (default: 5.0)
        """
        self.instrument_token = instrument_token
        self.price_bucket_size = price_bucket_size
        self.footprint_gen = FootprintChartGenerator(instrument_token, price_bucket_size)
    
    def detect_critical_levels(self,
                               start_time: str,
                               end_time: str,
                               analysis_date: Optional[date] = None) -> Dict:
        """
        Detect critical price levels for tactical entries/exits
        
        Args:
            start_time: Start time in HH:MM:SS format
            end_time: End time in HH:MM:SS format
            analysis_date: Analysis date (defaults to today)
            
        Returns:
            Dictionary with critical levels: support, resistance, pivot points
        """
        if analysis_date is None:
            analysis_date = date.today()
        
        try:
            # Get footprint data
            footprint_data = self.footprint_gen.generate_footprint_data(
                start_time, end_time, analysis_date
            )
            
            if not footprint_data.get('success'):
                return {
                    'success': False,
                    'message': 'Could not generate footprint data',
                    'missing_data': footprint_data.get('missing_data', [
                        'footprint data for critical level detection',
                        'volume distribution data',
                        'price level data'
                    ]),
                    'levels': []
                }
            
            # Identify levels from footprint
            support_levels = self._identify_support_levels(footprint_data)
            resistance_levels = self._identify_resistance_levels(footprint_data)
            pivot_levels = self._identify_pivot_points(footprint_data)
            
            # Combine and rank levels
            all_levels = self._combine_and_rank_levels(
                support_levels, resistance_levels, pivot_levels, footprint_data
            )
            
            return {
                'success': True,
                'analysis_date': analysis_date.strftime('%Y-%m-%d'),
                'time_range': f"{start_time} - {end_time}",
                'instrument_token': self.instrument_token,
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'pivot_levels': pivot_levels,
                'all_critical_levels': all_levels,
                'current_price': footprint_data.get('price_range', {}).get('close', 0),
                'price_range': footprint_data.get('price_range', {})
            }
            
        except Exception as e:
            logging.error(f"Error detecting critical levels: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'missing_data': ['Error occurred during analysis - check logs'],
                'levels': []
            }
    
    def _identify_support_levels(self, footprint_data: Dict) -> List[Dict]:
        """
        Identify support levels from footprint data
        Support = High volume nodes (HVN) with significant buy volume
        """
        footprint_levels = footprint_data.get('footprint_levels', [])
        current_price = footprint_data.get('price_range', {}).get('close', 0)
        
        if not footprint_levels or current_price == 0:
            return []
        
        support_levels = []
        hvn_list = footprint_data.get('hvn_lvn', {}).get('hvn', [])
        
        for level in footprint_levels:
            price = level['price']
            
            # Support: below current price, high volume, more buy than sell
            if price < current_price and level['total_volume'] > 0:
                buy_sell_ratio = level['buy_volume'] / level['sell_volume'] if level['sell_volume'] > 0 else 2.0
                
                # Check if it's an HVN
                is_hvn = any(abs(hvn['price'] - price) < self.price_bucket_size for hvn in hvn_list)
                
                # Support criteria: high volume, buy pressure, or HVN
                if (is_hvn or buy_sell_ratio > 1.3 or level['total_volume'] > 
                    sum(l['total_volume'] for l in footprint_levels) / len(footprint_levels) * 1.5):
                    
                    strength = 'strong' if (is_hvn and buy_sell_ratio > 1.5) else 'medium' if (is_hvn or buy_sell_ratio > 1.3) else 'weak'
                    
                    support_levels.append({
                        'price': price,
                        'strength': strength,
                        'volume': level['total_volume'],
                        'buy_volume': level['buy_volume'],
                        'sell_volume': level['sell_volume'],
                        'buy_sell_ratio': round(buy_sell_ratio, 2),
                        'distance_from_current': round((current_price - price) / current_price * 100, 2),
                        'is_hvn': is_hvn,
                        'type': 'support'
                    })
        
        # Sort by price (descending) and take top 5
        support_levels.sort(key=lambda x: x['price'], reverse=True)
        return support_levels[:5]
    
    def _identify_resistance_levels(self, footprint_data: Dict) -> List[Dict]:
        """
        Identify resistance levels from footprint data
        Resistance = High volume nodes (HVN) with significant sell volume
        """
        footprint_levels = footprint_data.get('footprint_levels', [])
        current_price = footprint_data.get('price_range', {}).get('close', 0)
        
        if not footprint_levels or current_price == 0:
            return []
        
        resistance_levels = []
        hvn_list = footprint_data.get('hvn_lvn', {}).get('hvn', [])
        
        for level in footprint_levels:
            price = level['price']
            
            # Resistance: above current price, high volume, more sell than buy
            if price > current_price and level['total_volume'] > 0:
                sell_buy_ratio = level['sell_volume'] / level['buy_volume'] if level['buy_volume'] > 0 else 2.0
                
                # Check if it's an HVN
                is_hvn = any(abs(hvn['price'] - price) < self.price_bucket_size for hvn in hvn_list)
                
                # Resistance criteria: high volume, sell pressure, or HVN
                if (is_hvn or sell_buy_ratio > 1.3 or level['total_volume'] > 
                    sum(l['total_volume'] for l in footprint_levels) / len(footprint_levels) * 1.5):
                    
                    strength = 'strong' if (is_hvn and sell_buy_ratio > 1.5) else 'medium' if (is_hvn or sell_buy_ratio > 1.3) else 'weak'
                    
                    resistance_levels.append({
                        'price': price,
                        'strength': strength,
                        'volume': level['total_volume'],
                        'buy_volume': level['buy_volume'],
                        'sell_volume': level['sell_volume'],
                        'sell_buy_ratio': round(sell_buy_ratio, 2),
                        'distance_from_current': round((price - current_price) / current_price * 100, 2),
                        'is_hvn': is_hvn,
                        'type': 'resistance'
                    })
        
        # Sort by price (ascending) and take top 5
        resistance_levels.sort(key=lambda x: x['price'], reverse=False)
        return resistance_levels[:5]
    
    def _identify_pivot_points(self, footprint_data: Dict) -> List[Dict]:
        """
        Identify pivot points (price levels where order flow reverses)
        """
        footprint_levels = footprint_data.get('footprint_levels', [])
        current_price = footprint_data.get('price_range', {}).get('close', 0)
        imbalances = footprint_data.get('imbalances', {}).get('imbalances', [])
        
        if not footprint_levels or current_price == 0:
            return []
        
        pivot_levels = []
        
        # Pivot points are often at imbalance zones
        for imbalance in imbalances:
            price = imbalance['price']
            distance = abs(price - current_price) / current_price * 100
            
            # Consider pivots within 2% of current price
            if distance < 2.0:
                pivot_levels.append({
                    'price': price,
                    'strength': imbalance.get('strength', 'medium'),
                    'type': 'pivot',
                    'imbalance_type': imbalance.get('type', 'unknown'),
                    'distance_from_current': round(distance, 2),
                    'volume': sum(l['total_volume'] for l in footprint_levels 
                                 if abs(l['price'] - price) < self.price_bucket_size)
                })
        
        # Also identify price levels with balanced volume (potential reversal points)
        avg_volume = sum(l['total_volume'] for l in footprint_levels) / len(footprint_levels) if footprint_levels else 0
        
        for level in footprint_levels:
            price = level['price']
            distance = abs(price - current_price) / current_price * 100
            
            # Balanced volume levels near current price (potential pivot)
            if (distance < 1.5 and 
                0.8 < level['buy_percentage'] / 100 < 1.2 and  # Balanced buy/sell
                level['total_volume'] > avg_volume * 0.8):
                
                # Check if not already added from imbalances
                if not any(abs(p['price'] - price) < self.price_bucket_size for p in pivot_levels):
                    pivot_levels.append({
                        'price': price,
                        'strength': 'medium',
                        'type': 'pivot',
                        'imbalance_type': 'balanced',
                        'distance_from_current': round(distance, 2),
                        'volume': level['total_volume'],
                        'buy_percentage': level.get('buy_percentage', 50)
                    })
        
        # Sort by distance from current price
        pivot_levels.sort(key=lambda x: x['distance_from_current'])
        return pivot_levels[:5]
    
    def _combine_and_rank_levels(self,
                                  support_levels: List[Dict],
                                  resistance_levels: List[Dict],
                                  pivot_levels: List[Dict],
                                  footprint_data: Dict) -> List[Dict]:
        """
        Combine all levels and rank by significance
        """
        current_price = footprint_data.get('price_range', {}).get('close', 0)
        
        all_levels = []
        
        # Add support levels
        for level in support_levels:
            all_levels.append({
                **level,
                'significance_score': self._calculate_significance_score(level, 'support', current_price)
            })
        
        # Add resistance levels
        for level in resistance_levels:
            all_levels.append({
                **level,
                'significance_score': self._calculate_significance_score(level, 'resistance', current_price)
            })
        
        # Add pivot levels
        for level in pivot_levels:
            all_levels.append({
                **level,
                'significance_score': self._calculate_significance_score(level, 'pivot', current_price)
            })
        
        # Sort by significance score (descending)
        all_levels.sort(key=lambda x: x['significance_score'], reverse=True)
        
        return all_levels[:10]  # Top 10 most significant levels
    
    def _calculate_significance_score(self, level: Dict, level_type: str, current_price: float) -> float:
        """
        Calculate significance score for a price level (0-100)
        """
        score = 0.0
        
        # Volume contribution (40%)
        volume_factor = min(1.0, level.get('volume', 0) / 1000000.0)  # Normalize to 1M
        score += volume_factor * 40.0
        
        # Strength contribution (30%)
        strength_map = {'strong': 1.0, 'medium': 0.6, 'weak': 0.3}
        strength_score = strength_map.get(level.get('strength', 'weak'), 0.3)
        score += strength_score * 30.0
        
        # Distance from current price (20%) - closer levels are more significant
        distance = level.get('distance_from_current', 100)
        if distance < 0.5:
            distance_score = 1.0
        elif distance < 1.0:
            distance_score = 0.8
        elif distance < 1.5:
            distance_score = 0.6
        else:
            distance_score = 0.4
        score += distance_score * 20.0
        
        # HVN contribution (10%)
        if level.get('is_hvn', False):
            score += 10.0
        
        return round(score, 2)
    
    def get_tactical_levels_for_scanner(self,
                                       scanner_candidates: List[Dict],
                                       footprint_data: Dict) -> Dict:
        """
        Get tactical entry/exit levels aligned with options scanner candidates
        Uses footprint data directly instead of re-calculating
        """
        if not scanner_candidates or not footprint_data.get('success'):
            return {'tactical_levels': [], 'aligned_candidates': []}
        
        # Extract critical levels from footprint data
        footprint_levels = footprint_data.get('footprint_levels', [])
        current_price = footprint_data.get('price_range', {}).get('close', 0)
        imbalances = footprint_data.get('imbalances', {}).get('imbalances', [])
        hvn_list = footprint_data.get('hvn_lvn', {}).get('hvn', [])
        
        # Identify support/resistance from footprint
        support_levels = []
        resistance_levels = []
        
        for level in footprint_levels:
            price = level['price']
            if price < current_price and level['total_volume'] > 0:
                buy_sell_ratio = level['buy_volume'] / level['sell_volume'] if level['sell_volume'] > 0 else 2.0
                is_hvn = any(abs(hvn['price'] - price) < self.price_bucket_size for hvn in hvn_list)
                if is_hvn or buy_sell_ratio > 1.3:
                    support_levels.append({
                        'price': price,
                        'strength': 'strong' if (is_hvn and buy_sell_ratio > 1.5) else 'medium',
                        'distance_from_current': round((current_price - price) / current_price * 100, 2),
                        'type': 'support'
                    })
            elif price > current_price and level['total_volume'] > 0:
                sell_buy_ratio = level['sell_volume'] / level['buy_volume'] if level['buy_volume'] > 0 else 2.0
                is_hvn = any(abs(hvn['price'] - price) < self.price_bucket_size for hvn in hvn_list)
                if is_hvn or sell_buy_ratio > 1.3:
                    resistance_levels.append({
                        'price': price,
                        'strength': 'strong' if (is_hvn and sell_buy_ratio > 1.5) else 'medium',
                        'distance_from_current': round((price - current_price) / current_price * 100, 2),
                        'type': 'resistance'
                    })
        
        support_levels.sort(key=lambda x: x['price'], reverse=True)
        resistance_levels.sort(key=lambda x: x['price'], reverse=False)
        
        aligned_candidates = []
        
        for candidate in scanner_candidates[:5]:  # Top 5 scanner candidates
            strike = candidate.get('strike_price', 0)
            
            # Find nearest critical levels
            nearest_support = None
            nearest_resistance = None
            
            for level in support_levels[:5]:
                if strike > level['price']:
                    if nearest_support is None or abs(strike - level['price']) < abs(strike - nearest_support['price']):
                        nearest_support = level
            
            for level in resistance_levels[:5]:
                if strike < level['price']:
                    if nearest_resistance is None or abs(strike - level['price']) < abs(strike - nearest_resistance['price']):
                        nearest_resistance = level
            
            aligned_candidates.append({
                **candidate,
                'nearest_support': nearest_support,
                'nearest_resistance': nearest_resistance,
                'has_confirmation': nearest_support is not None or nearest_resistance is not None
            })
        
        tactical_levels = (support_levels[:3] + resistance_levels[:3] + 
                          [{'price': imb['price'], 'type': 'pivot', 'strength': imb.get('strength', 'medium')} 
                           for imb in imbalances[:2]])
        
        return {
            'tactical_levels': tactical_levels,
            'aligned_candidates': aligned_candidates,
            'confirmation_rate': len([c for c in aligned_candidates if c['has_confirmation']]) / len(aligned_candidates) * 100 if aligned_candidates else 0
        }
