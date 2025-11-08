"""
Market Bias and Key Levels Analyzer
Analyzes pre-market and real-time TPO data to generate market bias and key levels
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import defaultdict
import logging
from typing import Dict, List, Tuple, Optional
from market.CalculateTPO import TPOProfile, PostgresDataFetcher, RealTimeTPOProfile
import json

def convert_numpy_types(obj):
    """
    Convert numpy types to native Python types for JSON serialization
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

class MarketBiasAnalyzer:
    """
    Analyzes market bias and key levels based on TPO profiles
    """
    
    def __init__(self, db_fetcher: PostgresDataFetcher, instrument_token: int = 256265, tick_size: float = 5.0):
        """
        Initialize Market Bias Analyzer
        
        Args:
            db_fetcher: Database fetcher instance
            instrument_token: Instrument token for analysis
            tick_size: Price tick size for TPO calculation
        """
        self.db_fetcher = db_fetcher
        self.instrument_token = instrument_token
        self.tick_size = tick_size
        
        # Market bias indicators
        self.bias_score = 0.0  # -100 to +100, negative=bearish, positive=bullish
        self.bias_strength = "Neutral"  # Weak, Moderate, Strong
        self.bias_direction = "Neutral"  # Bullish, Bearish, Neutral
        
        # Key levels
        self.key_levels = {
            'support_levels': [],
            'resistance_levels': [],
            'pivot_points': {},
            'value_area_levels': {},
            'initial_balance_levels': {},
            'single_prints': [],
            'poor_highs_lows': {}
        }
        
        # Market structure analysis
        self.market_structure = {
            'day_type': "Normal",
            'opening_type': "Open Auction",
            'session_range': 0.0,
            'value_area_range': 0.0,
            'poc_strength': 0.0
        }
        
        # Analysis metadata
        self.analysis_timestamp = None
        self.pre_market_profile = None
        self.real_time_profile = None
        self.long_term_profiles = []  # Store multiple daily TPO profiles for long-term analysis
        
    def analyze_pre_market_bias(self, analysis_date: str = None) -> Dict:
        """
        Analyze pre-market TPO data for bias indicators
        
        Args:
            analysis_date: Date for analysis in 'YYYY-MM-DD' format
            
        Returns:
            Dictionary containing pre-market bias analysis
        """
        if not analysis_date:
            analysis_date = datetime.now().strftime('%Y-%m-%d')
        
        # Fetch pre-market data (9:05am to 9:15am IST)
        pre_market_data = self.db_fetcher.fetch_tick_data(
            table_name='ticks',
            instrument_token=self.instrument_token,
            start_time=f'{analysis_date} 09:05:00.000 +0530',
            end_time=f'{analysis_date} 09:15:00.000 +0530'
        )
        
        if pre_market_data.empty:
            return {
                'bias_score': 0.0,
                'bias_direction': 'Neutral',
                'bias_strength': 'Weak',
                'key_levels': {},
                'analysis_quality': 'No Data'
            }
        
        # Calculate pre-market TPO profile
        pre_market_tpo = TPOProfile(tick_size=self.tick_size)
        pre_market_tpo.calculate_tpo(pre_market_data)
        self.pre_market_profile = pre_market_tpo
        
        # Analyze pre-market bias
        bias_analysis = self._calculate_pre_market_bias(pre_market_tpo, pre_market_data)
        
        return bias_analysis
    
    def analyze_real_time_bias(self, analysis_date: str = None) -> Dict:
        """
        Analyze real-time TPO data for bias indicators
        
        Args:
            analysis_date: Date for analysis in 'YYYY-MM-DD' format
            
        Returns:
            Dictionary containing real-time bias analysis
        """
        if not analysis_date:
            analysis_date = datetime.now().strftime('%Y-%m-%d')
        
        # For historical analysis, use the full market session (9:15am to 3:30pm IST)
        # For live analysis, use current time as end time
        if analysis_date == datetime.now().strftime('%Y-%m-%d'):
            # Live mode - use current time
            end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.000 +0530')
        else:
            # Historical mode - use full market session
            end_time = f'{analysis_date} 15:30:00.000 +0530'
        
        real_time_data = self.db_fetcher.fetch_tick_data(
            table_name='ticks',
            instrument_token=self.instrument_token,
            start_time=f'{analysis_date} 09:15:00.000 +0530',
            end_time=end_time
        )
        
        if real_time_data.empty:
            return {
                'bias_score': 0.0,
                'bias_direction': 'Neutral',
                'bias_strength': 'Weak',
                'key_levels': {},
                'analysis_quality': 'No Data'
            }
        
        # Calculate real-time TPO profile
        real_time_tpo = TPOProfile(tick_size=self.tick_size)
        real_time_tpo.calculate_tpo(real_time_data)
        self.real_time_profile = real_time_tpo
        
        # Analyze real-time bias
        bias_analysis = self._calculate_real_time_bias(real_time_tpo, real_time_data)
        
        return bias_analysis
    
    def analyze_long_term_tpo_bias(self, num_days: int = 20, analysis_date: str = None) -> Dict:
        """
        Analyze longer-duration TPO profiles for directional bias
        
        Args:
            num_days: Number of trading days to analyze (default: 20)
            analysis_date: Reference date for analysis in 'YYYY-MM-DD' format
            
        Returns:
            Dictionary containing long-term TPO bias analysis
        """
        if not analysis_date:
            analysis_date = datetime.now().strftime('%Y-%m-%d')
        
        # Fetch trading days data
        trading_days = self.db_fetcher.fetch_trading_days_data(
            table_name='ticks',
            instrument_token=self.instrument_token,
            num_days=num_days,
            market_start_time="09:15",
            market_end_time="15:30"
        )
        
        if not trading_days or len(trading_days) < 5:
            return {
                'bias_score': 0.0,
                'bias_direction': 'Neutral',
                'bias_strength': 'Weak',
                'bias_factors': [],
                'analysis_quality': 'Insufficient Data'
            }
        
        # Calculate TPO profiles for each day
        daily_profiles = []
        for trading_date, day_data in trading_days:
            if day_data.empty:
                continue
            
            tpo_profile = TPOProfile(tick_size=self.tick_size)
            tpo_profile.calculate_tpo(day_data)
            
            if tpo_profile.tpo_data is not None and not tpo_profile.tpo_data.empty:
                current_price = day_data['last_price'].iloc[-1] if not day_data.empty else None
                daily_profiles.append({
                    'date': trading_date,
                    'tpo_profile': tpo_profile,
                    'current_price': current_price,
                    'poc': tpo_profile.poc,
                    'value_area_high': tpo_profile.value_area_high,
                    'value_area_low': tpo_profile.value_area_low,
                    'poc_high': tpo_profile.poc_high,
                    'poc_low': tpo_profile.poc_low
                })
        
        if len(daily_profiles) < 5:
            return {
                'bias_score': 0.0,
                'bias_direction': 'Neutral',
                'bias_strength': 'Weak',
                'bias_factors': [],
                'analysis_quality': 'Insufficient Data'
            }
        
        self.long_term_profiles = daily_profiles
        
        # Analyze long-term bias using the rules
        bias_score = 0.0
        bias_factors = []
        
        # Rule 1: POC Movement Analysis
        poc_movement_score = self._analyze_poc_movement(daily_profiles)
        bias_score += poc_movement_score
        if abs(poc_movement_score) > 5:
            direction = "Bullish" if poc_movement_score > 0 else "Bearish"
            bias_factors.append(f"POC Movement ({direction}): {poc_movement_score:.1f}")
        
        # Rule 2: Value Area Movement Analysis
        va_movement_score = self._analyze_value_area_movement(daily_profiles)
        bias_score += va_movement_score
        if abs(va_movement_score) > 5:
            direction = "Bullish" if va_movement_score > 0 else "Bearish"
            bias_factors.append(f"Value Area Movement ({direction}): {va_movement_score:.1f}")
        
        # Rule 3: Distribution Shape Analysis
        distribution_score = self._analyze_distribution_shape(daily_profiles)
        bias_score += distribution_score
        if abs(distribution_score) > 5:
            direction = "Bullish" if distribution_score > 0 else "Bearish"
            bias_factors.append(f"Distribution Shape ({direction}): {distribution_score:.1f}")
        
        # Rule 4: Control Location Relative to Price
        control_location_score = self._analyze_control_location(daily_profiles)
        bias_score += control_location_score
        if abs(control_location_score) > 5:
            direction = "Bullish" if control_location_score > 0 else "Bearish"
            bias_factors.append(f"Control Location ({direction}): {control_location_score:.1f}")
        
        # Rule 5: Bear Trap Detection
        bear_trap_analysis = self._detect_bear_traps(trading_days, daily_profiles)
        if bear_trap_analysis['potential_bear_traps']:
            bias_factors.append(f"Bear Trap Detection: {len(bear_trap_analysis['potential_bear_traps'])} potential trap(s) found")
        
        # Determine bias direction and strength
        bias_direction = "Neutral"
        bias_strength = "Weak"
        
        if bias_score > 15:
            bias_direction = "Bullish"
            bias_strength = "Strong" if bias_score > 30 else "Moderate"
        elif bias_score < -15:
            bias_direction = "Bearish"
            bias_strength = "Strong" if bias_score < -30 else "Moderate"
        
        return {
            'bias_score': bias_score,
            'bias_direction': bias_direction,
            'bias_strength': bias_strength,
            'bias_factors': bias_factors,
            'num_days_analyzed': len(daily_profiles),
            'analysis_quality': 'Good' if len(daily_profiles) >= 10 else 'Fair',
            'bear_trap_analysis': bear_trap_analysis
        }
    
    def _analyze_poc_movement(self, daily_profiles: List[Dict]) -> float:
        """
        Analyze POC movement over sessions
        Higher POC trend = bullish, Lower POC trend = bearish
        
        Returns:
            Score from -30 to +30
        """
        if len(daily_profiles) < 3:
            return 0.0
        
        pocs = [p['poc'] for p in daily_profiles if p['poc'] is not None]
        
        if len(pocs) < 3:
            return 0.0
        
        # Calculate POC trend (linear regression slope)
        x = np.arange(len(pocs))
        y = np.array(pocs)
        
        # Remove NaN values
        valid_mask = ~np.isnan(y)
        if valid_mask.sum() < 3:
            return 0.0
        
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        if len(x_valid) < 3:
            return 0.0
        
        # Calculate slope
        slope = np.polyfit(x_valid, y_valid, 1)[0]
        
        # Normalize by average POC to get percentage change
        avg_poc = np.mean(y_valid)
        if avg_poc > 0:
            poc_change_pct = (slope / avg_poc) * 100
            
            # Scale to -30 to +30 score
            # Positive slope (POC moving higher) = bullish
            # Negative slope (POC moving lower) = bearish
            score = poc_change_pct * 3  # Scale factor
            return np.clip(score, -30, 30)
        
        return 0.0
    
    def _analyze_value_area_movement(self, daily_profiles: List[Dict]) -> float:
        """
        Analyze Value Area movement over sessions
        Upward movement or expansion upward = bullish
        Downward movement = bearish
        
        Returns:
            Score from -30 to +30
        """
        if len(daily_profiles) < 3:
            return 0.0
        
        va_highs = [p['value_area_high'] for p in daily_profiles 
                   if p['value_area_high'] is not None]
        va_lows = [p['value_area_low'] for p in daily_profiles 
                  if p['value_area_low'] is not None]
        
        if len(va_highs) < 3 or len(va_lows) < 3:
            return 0.0
        
        # Analyze VA High movement
        x = np.arange(len(va_highs))
        y_high = np.array(va_highs)
        y_low = np.array(va_lows)
        
        # Remove NaN values
        valid_mask = ~(np.isnan(y_high) | np.isnan(y_low))
        if valid_mask.sum() < 3:
            return 0.0
        
        x_valid = x[valid_mask]
        y_high_valid = y_high[valid_mask]
        y_low_valid = y_low[valid_mask]
        
        if len(x_valid) < 3:
            return 0.0
        
        # Calculate VA center movement
        va_centers = (y_high_valid + y_low_valid) / 2
        slope = np.polyfit(x_valid, va_centers, 1)[0]
        
        # Calculate VA expansion (size change)
        va_sizes = y_high_valid - y_low_valid
        size_slope = np.polyfit(x_valid, va_sizes, 1)[0]
        
        # Normalize by average VA center
        avg_va_center = np.mean(va_centers)
        if avg_va_center > 0:
            va_movement_pct = (slope / avg_va_center) * 100
            
            # Expansion upward (higher VA high) = bullish
            expansion_score = 0
            if size_slope > 0 and y_high_valid[-1] > y_high_valid[0]:
                expansion_score = 10  # VA expanding upward
            elif size_slope < 0 and y_low_valid[-1] < y_low_valid[0]:
                expansion_score = -10  # VA contracting downward
            
            # Scale to -30 to +30 score
            score = va_movement_pct * 2 + expansion_score
            return np.clip(score, -30, 30)
        
        return 0.0
    
    def _analyze_distribution_shape(self, daily_profiles: List[Dict]) -> float:
        """
        Analyze distribution shape
        Balanced/bell-shaped = neutral
        Trend day with pronounced distribution = directional strength
        
        Returns:
            Score from -30 to +30
        """
        if len(daily_profiles) < 5:
            return 0.0
        
        # Analyze recent profiles for trend day characteristics
        recent_profiles = daily_profiles[-5:]
        
        trend_scores = []
        for profile_data in recent_profiles:
            tpo_profile = profile_data['tpo_profile']
            if tpo_profile.tpo_data is None or tpo_profile.tpo_data.empty:
                continue
            
            prices = tpo_profile.tpo_data['price'].values
            counts = tpo_profile.tpo_data['tpo_count'].values
            
            if len(counts) < 3:
                continue
            
            # Calculate distribution skewness
            poc_idx = np.argmax(counts)
            poc_price = prices[poc_idx]
            
            # Count TPOs above and below POC
            upper_tpos = counts[prices > poc_price].sum()
            lower_tpos = counts[prices < poc_price].sum()
            total_tpos = counts.sum()
            
            if total_tpos > 0:
                # Skewness: positive = bullish (more TPOs above POC), negative = bearish
                skewness = (upper_tpos - lower_tpos) / total_tpos
                
                # Distribution concentration (how focused vs balanced)
                max_count = counts.max()
                concentration = max_count / total_tpos
                
                # Trend day = high concentration + high skewness
                if concentration > 0.3:  # More than 30% of TPOs at one level
                    # Strong trend day
                    trend_score = skewness * 20 * concentration
                    trend_scores.append(trend_score)
                else:
                    # Balanced distribution = neutral
                    trend_scores.append(0)
        
        if trend_scores:
            avg_trend_score = np.mean(trend_scores)
            return np.clip(avg_trend_score * 10, -30, 30)
        
        return 0.0
    
    def _analyze_control_location(self, daily_profiles: List[Dict]) -> float:
        """
        Analyze control location relative to price
        Trading mostly above control areas = bullish
        Trading mostly below control areas = bearish
        
        Returns:
            Score from -30 to +30
        """
        if len(daily_profiles) < 5:
            return 0.0
        
        above_control_count = 0
        below_control_count = 0
        total_count = 0
        
        for profile_data in daily_profiles:
            poc = profile_data['poc']
            current_price = profile_data['current_price']
            
            if poc is not None and current_price is not None and not np.isnan(poc) and not np.isnan(current_price):
                total_count += 1
                if current_price > poc:
                    above_control_count += 1
                elif current_price < poc:
                    below_control_count += 1
        
        if total_count == 0:
            return 0.0
        
        # Calculate percentage above/below control
        above_pct = above_control_count / total_count
        below_pct = below_control_count / total_count
        
        # Score: positive if mostly above, negative if mostly below
        # Strong signal if >70% on one side
        if above_pct >= 0.7:
            score = 30  # Strong bullish
        elif above_pct >= 0.6:
            score = 20  # Moderate bullish
        elif above_pct >= 0.55:
            score = 10  # Weak bullish
        elif below_pct >= 0.7:
            score = -30  # Strong bearish
        elif below_pct >= 0.6:
            score = -20  # Moderate bearish
        elif below_pct >= 0.55:
            score = -10  # Weak bearish
        else:
            # Balanced around control
            score = (above_pct - below_pct) * 20
        
        return np.clip(score, -30, 30)
    
    def _detect_bear_traps(self, trading_days: List[Tuple], daily_profiles: List[Dict]) -> Dict:
        """
        Detect potential bear traps using Nifty tick data
        
        Logic:
        1. If price breaks below known support level
        2. If breakdown occurs on high volume
        3. Watch subsequent candles (1-3 bars later)
        4. If price quickly returns above support, especially with a bullish candle
        5. If overall trend is still bullish (higher highs/lows intact)
        6. Flag as potential bear trap
        
        Args:
            trading_days: List of (date, tick_data) tuples
            daily_profiles: List of daily TPO profiles
            
        Returns:
            Dictionary containing bear trap analysis
        """
        try:
            potential_bear_traps = []
            
            if len(trading_days) < 5 or len(daily_profiles) < 5:
                return {
                    'potential_bear_traps': [],
                    'total_traps_detected': 0,
                    'analysis_quality': 'Insufficient Data'
                }
            
            # Convert tick data to daily OHLC bars
            daily_bars = []
            for trading_date, day_data in trading_days:
                if day_data.empty:
                    continue
                
                # Calculate daily OHLC from tick data
                daily_open = day_data['last_price'].iloc[0] if not day_data.empty else None
                daily_high = day_data['last_price'].max() if not day_data.empty else None
                daily_low = day_data['last_price'].min() if not day_data.empty else None
                daily_close = day_data['last_price'].iloc[-1] if not day_data.empty else None
                
                # Calculate volume (sum of all volume if available)
                daily_volume = 0
                if 'volume' in day_data.columns:
                    daily_volume = day_data['volume'].sum() if day_data['volume'].notna().any() else 0
                elif 'last_quantity' in day_data.columns:
                    daily_volume = day_data['last_quantity'].sum() if day_data['last_quantity'].notna().any() else 0
                
                if daily_open and daily_high and daily_low and daily_close:
                    daily_bars.append({
                        'date': trading_date,
                        'open': daily_open,
                        'high': daily_high,
                        'low': daily_low,
                        'close': daily_close,
                        'volume': daily_volume
                    })
            
            if len(daily_bars) < 5:
                return {
                    'potential_bear_traps': [],
                    'total_traps_detected': 0,
                    'analysis_quality': 'Insufficient Data'
                }
            
            # Identify support levels from TPO profiles (value_area_low, poc_low, etc.)
            support_levels = []
            for profile_data in daily_profiles:
                if profile_data.get('value_area_low'):
                    support_levels.append({
                        'level': profile_data['value_area_low'],
                        'type': 'Value Area Low',
                        'date': profile_data['date']
                    })
                if profile_data.get('poc_low'):
                    support_levels.append({
                        'level': profile_data['poc_low'],
                        'type': 'POC Low',
                        'date': profile_data['date']
                    })
            
            # Sort support levels by price (ascending)
            support_levels.sort(key=lambda x: x['level'])
            
            # Calculate average volume for volume comparison
            volumes = [bar['volume'] for bar in daily_bars if bar['volume'] > 0]
            avg_volume = np.mean(volumes) if volumes else 0
            
            # Check for bear traps
            for i in range(len(daily_bars) - 3):  # Need at least 3 bars ahead
                current_bar = daily_bars[i]
                current_date = current_bar['date']
                
                # Find relevant support levels (from previous days)
                relevant_supports = [s for s in support_levels 
                                  if s['date'] < current_date and s['level'] is not None]
                
                if not relevant_supports:
                    continue
                
                # Get the most recent support level
                most_recent_support = max(relevant_supports, key=lambda x: x['date'])
                support_level = most_recent_support['level']
                
                # Check if price breaks below support
                if current_bar['low'] < support_level:
                    # Check if breakdown occurs on high volume
                    breakdown_volume = current_bar['volume']
                    is_high_volume = breakdown_volume > (avg_volume * 1.2) if avg_volume > 0 else False
                    
                    if is_high_volume:
                        # Watch subsequent candles (1-3 bars later)
                        for j in range(1, min(4, len(daily_bars) - i)):  # Check next 1-3 bars
                            future_bar = daily_bars[i + j]
                            
                            # Check if price quickly returns above support
                            if future_bar['close'] > support_level:
                                # Check if it's a bullish candle (close > open)
                                is_bullish_candle = future_bar['close'] > future_bar['open']
                                
                                # Check if overall trend is still bullish (higher highs/lows intact)
                                # Look at the last 5-10 bars before the breakdown
                                lookback_start = max(0, i - 10)
                                lookback_bars = daily_bars[lookback_start:i]
                                
                                if len(lookback_bars) >= 3:
                                    # Check for higher highs and higher lows
                                    recent_highs = [b['high'] for b in lookback_bars[-5:]]
                                    recent_lows = [b['low'] for b in lookback_bars[-5:]]
                                    
                                    # Simple trend check: recent highs and lows are generally increasing
                                    if len(recent_highs) >= 3:
                                        # Check if trend is bullish (more highs than lows in recent period)
                                        high_trend = recent_highs[-1] > recent_highs[0] if len(recent_highs) >= 2 else False
                                        low_trend = recent_lows[-1] > recent_lows[0] if len(recent_lows) >= 2 else False
                                        
                                        is_bullish_trend = high_trend or low_trend
                                        
                                        # Flag as potential bear trap
                                        if is_bullish_trend:
                                            trap_details = {
                                                'date': current_date,
                                                'support_level': support_level,
                                                'support_type': most_recent_support['type'],
                                                'breakdown_low': current_bar['low'],
                                                'breakdown_volume': breakdown_volume,
                                                'recovery_date': future_bar['date'],
                                                'recovery_close': future_bar['close'],
                                                'recovery_bars_later': j,
                                                'is_bullish_candle': is_bullish_candle,
                                                'confidence': 'High' if is_bullish_candle and j <= 2 else 'Medium'
                                            }
                                            potential_bear_traps.append(trap_details)
                                            break  # Found a trap, move to next bar
            
            return {
                'potential_bear_traps': potential_bear_traps,
                'total_traps_detected': len(potential_bear_traps),
                'analysis_quality': 'Good' if len(daily_bars) >= 10 else 'Fair'
            }
            
        except Exception as e:
            logging.error(f"Error detecting bear traps: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return {
                'potential_bear_traps': [],
                'total_traps_detected': 0,
                'analysis_quality': 'Error',
                'error': str(e)
            }
    
    def generate_comprehensive_analysis(self, analysis_date: str = None) -> Dict:
        """
        Generate comprehensive market bias and key levels analysis
        
        Args:
            analysis_date: Date for analysis in 'YYYY-MM-DD' format
            
        Returns:
            Dictionary containing comprehensive analysis
        """
        if not analysis_date:
            analysis_date = datetime.now().strftime('%Y-%m-%d')
        
        self.analysis_timestamp = datetime.now()
        
        # Analyze pre-market bias
        pre_market_analysis = self.analyze_pre_market_bias(analysis_date)
        
        # Analyze real-time bias
        real_time_analysis = self.analyze_real_time_bias(analysis_date)
        
        # Analyze long-term TPO bias (20 days)
        long_term_analysis = self.analyze_long_term_tpo_bias(num_days=20, analysis_date=analysis_date)
        
        # Combine analyses for comprehensive bias determination
        comprehensive_bias = self._combine_bias_analyses(pre_market_analysis, real_time_analysis, long_term_analysis)
        
        # Calculate key levels
        key_levels = self._calculate_key_levels()
        
        # Determine market structure
        market_structure = self._analyze_market_structure()
        
        # Generate trading recommendations
        trading_recommendations = self._generate_trading_recommendations(comprehensive_bias, key_levels)
        
        # Convert numpy types to native Python types for JSON serialization
        result = {
            'analysis_timestamp': self.analysis_timestamp.isoformat(),
            'analysis_date': analysis_date,
            'instrument_token': int(self.instrument_token),
            'pre_market_analysis': convert_numpy_types(pre_market_analysis),
            'real_time_analysis': convert_numpy_types(real_time_analysis),
            'long_term_analysis': convert_numpy_types(long_term_analysis),
            'comprehensive_bias': convert_numpy_types(comprehensive_bias),
            'key_levels': convert_numpy_types(key_levels),
            'market_structure': convert_numpy_types(market_structure),
            'trading_recommendations': convert_numpy_types(trading_recommendations)
        }
        
        return result
    
    def _calculate_pre_market_bias(self, tpo_profile: TPOProfile, data: pd.DataFrame) -> Dict:
        """
        Calculate bias indicators from pre-market TPO profile
        
        Args:
            tpo_profile: Pre-market TPO profile
            data: Pre-market tick data
            
        Returns:
            Dictionary containing pre-market bias analysis
        """
        if tpo_profile.tpo_data is None or tpo_profile.tpo_data.empty:
            return {
                'bias_score': 0.0,
                'bias_direction': 'Neutral',
                'bias_strength': 'Weak',
                'key_levels': {},
                'analysis_quality': 'No TPO Data'
            }
        
        bias_score = 0.0
        bias_factors = []
        
        # Factor 1: POC position relative to session range
        if tpo_profile.poc is not None:
            session_high = data['last_price'].max()
            session_low = data['last_price'].min()
            session_range = session_high - session_low
            
            if session_range > 0:
                poc_position = (tpo_profile.poc - session_low) / session_range
                # POC above 60% = bullish bias, below 40% = bearish bias
                poc_factor = (poc_position - 0.5) * 40  # Scale to -20 to +20
                bias_score += poc_factor
                bias_factors.append(f"POC Position: {poc_factor:.1f}")
        
        # Factor 2: Value Area asymmetry
        if tpo_profile.value_area_high is not None and tpo_profile.value_area_low is not None:
            va_range = tpo_profile.value_area_high - tpo_profile.value_area_low
            if va_range > 0:
                va_center = (tpo_profile.value_area_high + tpo_profile.value_area_low) / 2
                session_center = (data['last_price'].max() + data['last_price'].min()) / 2
                va_asymmetry = (va_center - session_center) / va_range * 20  # Scale to -20 to +20
                bias_score += va_asymmetry
                bias_factors.append(f"VA Asymmetry: {va_asymmetry:.1f}")
        
        # Factor 3: Initial Balance breakout potential
        if tpo_profile.initial_balance_high is not None and tpo_profile.initial_balance_low is not None:
            ib_range = tpo_profile.initial_balance_high - tpo_profile.initial_balance_low
            session_range = data['last_price'].max() - data['last_price'].min()
            
            if session_range > 0:
                ib_ratio = ib_range / session_range
                # Narrow IB suggests breakout potential
                ib_factor = (0.3 - ib_ratio) * 30  # Scale to -9 to +9
                bias_score += ib_factor
                bias_factors.append(f"IB Breakout Potential: {ib_factor:.1f}")
        
        # Factor 4: Price action momentum
        if len(data) > 1:
            price_change = data['last_price'].iloc[-1] - data['last_price'].iloc[0]
            price_range = data['last_price'].max() - data['last_price'].min()
            
            if price_range > 0:
                momentum_factor = (price_change / price_range) * 30  # Scale to -30 to +30
                bias_score += momentum_factor
                bias_factors.append(f"Price Momentum: {momentum_factor:.1f}")
        
        # Determine bias direction and strength
        bias_direction = "Neutral"
        bias_strength = "Weak"
        
        if bias_score > 15:
            bias_direction = "Bullish"
            bias_strength = "Strong" if bias_score > 30 else "Moderate"
        elif bias_score < -15:
            bias_direction = "Bearish"
            bias_strength = "Strong" if bias_score < -30 else "Moderate"
        
        return {
            'bias_score': bias_score,
            'bias_direction': bias_direction,
            'bias_strength': bias_strength,
            'bias_factors': bias_factors,
            'key_levels': {
                'poc': tpo_profile.poc,
                'value_area_high': tpo_profile.value_area_high,
                'value_area_low': tpo_profile.value_area_low,
                'initial_balance_high': tpo_profile.initial_balance_high,
                'initial_balance_low': tpo_profile.initial_balance_low
            },
            'analysis_quality': 'Good'
        }
    
    def _calculate_real_time_bias(self, tpo_profile: TPOProfile, data: pd.DataFrame) -> Dict:
        """
        Calculate bias indicators from real-time TPO profile
        
        Args:
            tpo_profile: Real-time TPO profile
            data: Real-time tick data
            
        Returns:
            Dictionary containing real-time bias analysis
        """
        if tpo_profile.tpo_data is None or tpo_profile.tpo_data.empty:
            return {
                'bias_score': 0.0,
                'bias_direction': 'Neutral',
                'bias_strength': 'Weak',
                'key_levels': {},
                'analysis_quality': 'No TPO Data'
            }
        
        bias_score = 0.0
        bias_factors = []
        
        # Factor 1: Current price vs POC
        if tpo_profile.poc is not None and not data.empty:
            current_price = data['last_price'].iloc[-1]
            poc_distance = (current_price - tpo_profile.poc) / tpo_profile.poc * 100
            poc_factor = poc_distance * 2  # Scale to -20 to +20
            bias_score += poc_factor
            bias_factors.append(f"Price vs POC: {poc_factor:.1f}")
        
        # Factor 2: Value Area acceptance/rejection
        if tpo_profile.value_area_high is not None and tpo_profile.value_area_low is not None:
            if not data.empty:
                current_price = data['last_price'].iloc[-1]
                if current_price > tpo_profile.value_area_high:
                    va_factor = 15  # Above VA = bullish
                    bias_factors.append("Above Value Area: +15")
                elif current_price < tpo_profile.value_area_low:
                    va_factor = -15  # Below VA = bearish
                    bias_factors.append("Below Value Area: -15")
                else:
                    va_factor = 0  # Within VA = neutral
                    bias_factors.append("Within Value Area: 0")
                bias_score += va_factor
        
        # Factor 3: TPO distribution analysis
        if tpo_profile.tpo_data is not None:
            tpo_counts = tpo_profile.tpo_data['tpo_count'].values
            total_tpos = tpo_counts.sum()
            
            if total_tpos > 0:
                # Calculate TPO distribution skewness
                prices = tpo_profile.tpo_data['price'].values
                poc_idx = np.argmax(tpo_counts)
                poc_price = prices[poc_idx]
                
                upper_tpos = tpo_counts[prices > poc_price].sum()
                lower_tpos = tpo_counts[prices < poc_price].sum()
                
                if upper_tpos + lower_tpos > 0:
                    distribution_skew = (upper_tpos - lower_tpos) / (upper_tpos + lower_tpos) * 20
                    bias_score += distribution_skew
                    bias_factors.append(f"TPO Distribution: {distribution_skew:.1f}")
        
        # Factor 4: Session progression analysis
        if len(data) > 10:  # Need sufficient data points
            # Analyze price progression throughout the session
            session_length = len(data)
            first_quarter = data['last_price'].iloc[:session_length//4].mean()
            last_quarter = data['last_price'].iloc[-session_length//4:].mean()
            
            if first_quarter > 0:
                progression_factor = (last_quarter - first_quarter) / first_quarter * 100 * 2
                bias_score += progression_factor
                bias_factors.append(f"Session Progression: {progression_factor:.1f}")
        
        # Factor 5: Volume profile analysis (if available)
        if 'volume' in data.columns and not data.empty:
            recent_volume = data['volume'].tail(10).mean()
            early_volume = data['volume'].head(10).mean()
            
            if early_volume > 0:
                volume_trend = (recent_volume - early_volume) / early_volume * 100
                volume_factor = volume_trend * 0.5  # Scale down volume impact
                bias_score += volume_factor
                bias_factors.append(f"Volume Trend: {volume_factor:.1f}")
        
        # Determine bias direction and strength
        bias_direction = "Neutral"
        bias_strength = "Weak"
        
        if bias_score > 20:
            bias_direction = "Bullish"
            bias_strength = "Strong" if bias_score > 40 else "Moderate"
        elif bias_score < -20:
            bias_direction = "Bearish"
            bias_strength = "Strong" if bias_score < -40 else "Moderate"
        
        return {
            'bias_score': bias_score,
            'bias_direction': bias_direction,
            'bias_strength': bias_strength,
            'bias_factors': bias_factors,
            'key_levels': {
                'poc': tpo_profile.poc,
                'value_area_high': tpo_profile.value_area_high,
                'value_area_low': tpo_profile.value_area_low,
                'initial_balance_high': tpo_profile.initial_balance_high,
                'initial_balance_low': tpo_profile.initial_balance_low
            },
            'analysis_quality': 'Good'
        }
    
    def _combine_bias_analyses(self, pre_market: Dict, real_time: Dict, long_term: Dict = None) -> Dict:
        """
        Combine pre-market, real-time, and long-term bias analyses
        
        Args:
            pre_market: Pre-market bias analysis
            real_time: Real-time bias analysis
            long_term: Long-term TPO bias analysis (optional)
            
        Returns:
            Combined bias analysis
        """
        # Weight the analyses (pre-market 20%, real-time 40%, long-term 40%)
        pre_market_weight = 0.2
        real_time_weight = 0.4
        long_term_weight = 0.4
        
        if long_term is None or long_term.get('analysis_quality') == 'Insufficient Data':
            # Fallback to original weighting if long-term analysis is not available
            pre_market_weight = 0.3
            real_time_weight = 0.7
            long_term_weight = 0.0
        
        combined_score = (pre_market['bias_score'] * pre_market_weight + 
                         real_time['bias_score'] * real_time_weight)
        
        if long_term_weight > 0 and long_term:
            combined_score += long_term['bias_score'] * long_term_weight
        
        # Determine combined direction and strength
        bias_direction = "Neutral"
        bias_strength = "Weak"
        
        if combined_score > 15:
            bias_direction = "Bullish"
            bias_strength = "Strong" if combined_score > 30 else "Moderate"
        elif combined_score < -15:
            bias_direction = "Bearish"
            bias_strength = "Strong" if combined_score < -30 else "Moderate"
        
        # Combine bias factors
        combined_factors = []
        combined_factors.extend([f"Pre-market: {factor}" for factor in pre_market.get('bias_factors', [])])
        combined_factors.extend([f"Real-time: {factor}" for factor in real_time.get('bias_factors', [])])
        if long_term and long_term.get('bias_factors'):
            combined_factors.extend([f"Long-term TPO: {factor}" for factor in long_term.get('bias_factors', [])])
        
        return {
            'bias_score': combined_score,
            'bias_direction': bias_direction,
            'bias_strength': bias_strength,
            'bias_factors': combined_factors,
            'pre_market_weight': pre_market_weight,
            'real_time_weight': real_time_weight,
            'long_term_weight': long_term_weight,
            'includes_long_term': long_term_weight > 0
        }
    
    def _calculate_key_levels(self) -> Dict:
        """
        Calculate key support and resistance levels
        
        Returns:
            Dictionary containing key levels
        """
        key_levels = {
            'support_levels': [],
            'resistance_levels': [],
            'pivot_points': {},
            'value_area_levels': {},
            'initial_balance_levels': {},
            'single_prints': [],
            'poor_highs_lows': {}
        }
        
        # Use real-time profile if available, otherwise pre-market
        profile = self.real_time_profile if self.real_time_profile else self.pre_market_profile
        
        if profile is None or profile.tpo_data is None:
            return key_levels
        
        # Calculate support and resistance levels
        if profile.tpo_data is not None and not profile.tpo_data.empty:
            prices = profile.tpo_data['price'].values
            tpo_counts = profile.tpo_data['tpo_count'].values
            
            # Find significant TPO clusters (local maxima)
            for i in range(1, len(tpo_counts) - 1):
                if (tpo_counts[i] > tpo_counts[i-1] and 
                    tpo_counts[i] > tpo_counts[i+1] and 
                    tpo_counts[i] >= 3):  # Minimum TPO count for significance
                    
                    if prices[i] < profile.poc:
                        key_levels['support_levels'].append({
                            'price': prices[i],
                            'strength': tpo_counts[i],
                            'type': 'TPO Cluster'
                        })
                    else:
                        key_levels['resistance_levels'].append({
                            'price': prices[i],
                            'strength': tpo_counts[i],
                            'type': 'TPO Cluster'
                        })
        
        # Add POC as key level
        if profile.poc is not None:
            key_levels['pivot_points']['poc'] = {
                'price': profile.poc,
                'strength': 'High',
                'type': 'Point of Control'
            }
        
        # Add Value Area levels
        if profile.value_area_high is not None:
            key_levels['value_area_levels']['vah'] = {
                'price': profile.value_area_high,
                'strength': 'High',
                'type': 'Value Area High'
            }
        
        if profile.value_area_low is not None:
            key_levels['value_area_levels']['val'] = {
                'price': profile.value_area_low,
                'strength': 'High',
                'type': 'Value Area Low'
            }
        
        # Add Initial Balance levels
        if profile.initial_balance_high is not None:
            key_levels['initial_balance_levels']['ibh'] = {
                'price': profile.initial_balance_high,
                'strength': 'Medium',
                'type': 'Initial Balance High'
            }
        
        if profile.initial_balance_low is not None:
            key_levels['initial_balance_levels']['ibl'] = {
                'price': profile.initial_balance_low,
                'strength': 'Medium',
                'type': 'Initial Balance Low'
            }
        
        # Find single prints (weak levels)
        if profile.tpo_data is not None:
            single_prints = profile.tpo_data[profile.tpo_data['tpo_count'] == 1]
            for _, row in single_prints.iterrows():
                key_levels['single_prints'].append({
                    'price': row['price'],
                    'strength': 'Weak',
                    'type': 'Single Print'
                })
        
        return key_levels
    
    def _analyze_market_structure(self) -> Dict:
        """
        Analyze market structure based on TPO profiles
        
        Returns:
            Dictionary containing market structure analysis
        """
        structure = {
            'day_type': "Normal",
            'opening_type': "Open Auction",
            'session_range': 0.0,
            'value_area_range': 0.0,
            'poc_strength': 0.0,
            'distribution_type': "Single Distribution"
        }
        
        profile = self.real_time_profile if self.real_time_profile else self.pre_market_profile
        
        if profile is None or profile.tpo_data is None:
            return structure
        
        # Determine day type based on TPO distribution
        if profile.tpo_data is not None and not profile.tpo_data.empty:
            tpo_counts = profile.tpo_data['tpo_count'].values
            prices = profile.tpo_data['price'].values
            
            # Check for double distribution
            peaks = []
            for i in range(1, len(tpo_counts) - 1):
                if (tpo_counts[i] > tpo_counts[i-1] and 
                    tpo_counts[i] > tpo_counts[i+1] and 
                    tpo_counts[i] >= 3):
                    peaks.append(prices[i])
            
            if len(peaks) > 1:
                structure['distribution_type'] = "Double Distribution"
                structure['day_type'] = "Double Distribution"
            
            # Calculate session range
            if len(prices) > 0:
                structure['session_range'] = prices.max() - prices.min()
            
            # Calculate value area range
            if profile.value_area_high is not None and profile.value_area_low is not None:
                structure['value_area_range'] = profile.value_area_high - profile.value_area_low
            
            # Calculate POC strength
            if profile.poc is not None:
                max_tpo_count = tpo_counts.max()
                total_tpos = tpo_counts.sum()
                if total_tpos > 0:
                    structure['poc_strength'] = (max_tpo_count / total_tpos) * 100
        
        return structure
    
    def _generate_trading_recommendations(self, bias: Dict, key_levels: Dict) -> Dict:
        """
        Generate trading recommendations based on bias and key levels
        
        Args:
            bias: Market bias analysis
            key_levels: Key levels analysis
            
        Returns:
            Dictionary containing trading recommendations
        """
        recommendations = {
            'primary_bias': bias['bias_direction'],
            'bias_strength': bias['bias_strength'],
            'key_levels_to_watch': [],
            'entry_signals': [],
            'exit_signals': [],
            'risk_level': 'Medium',
            'position_sizing': 'Standard'
        }
        
        # Determine key levels to watch
        if key_levels['pivot_points'].get('poc'):
            recommendations['key_levels_to_watch'].append({
                'level': key_levels['pivot_points']['poc']['price'],
                'type': 'POC',
                'importance': 'Critical'
            })
        
        if key_levels['value_area_levels'].get('vah'):
            recommendations['key_levels_to_watch'].append({
                'level': key_levels['value_area_levels']['vah']['price'],
                'type': 'VAH',
                'importance': 'High'
            })
        
        if key_levels['value_area_levels'].get('val'):
            recommendations['key_levels_to_watch'].append({
                'level': key_levels['value_area_levels']['val']['price'],
                'type': 'VAL',
                'importance': 'High'
            })
        
        # Generate entry signals based on bias
        if bias['bias_direction'] == 'Bullish':
            recommendations['entry_signals'].append({
                'signal': 'Long',
                'condition': 'Price breaks above VAH with volume',
                'confidence': bias['bias_strength']
            })
        elif bias['bias_direction'] == 'Bearish':
            recommendations['entry_signals'].append({
                'signal': 'Short',
                'condition': 'Price breaks below VAL with volume',
                'confidence': bias['bias_strength']
            })
        
        # Generate exit signals
        if bias['bias_direction'] == 'Bullish':
            recommendations['exit_signals'].append({
                'signal': 'Exit Long',
                'condition': 'Price fails to hold above POC',
                'confidence': 'High'
            })
        elif bias['bias_direction'] == 'Bearish':
            recommendations['exit_signals'].append({
                'signal': 'Exit Short',
                'condition': 'Price fails to hold below POC',
                'confidence': 'High'
            })
        
        # Adjust risk level based on bias strength
        if bias['bias_strength'] == 'Strong':
            recommendations['risk_level'] = 'High'
            recommendations['position_sizing'] = 'Aggressive'
        elif bias['bias_strength'] == 'Weak':
            recommendations['risk_level'] = 'Low'
            recommendations['position_sizing'] = 'Conservative'
        
        return recommendations
    
    def plot_bias_analysis(self, analysis: Dict, save_path: str = None) -> str:
        """
        Plot comprehensive bias analysis
        
        Args:
            analysis: Comprehensive analysis dictionary
            save_path: Path to save the plot
            
        Returns:
            Base64 encoded image string
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Bias Score Visualization
        bias_score = analysis['comprehensive_bias']['bias_score']
        colors = ['red' if bias_score < -15 else 'orange' if bias_score < 15 else 'green']
        
        ax1.bar(['Market Bias'], [bias_score], color=colors, alpha=0.7)
        ax1.axhline(y=15, color='green', linestyle='--', alpha=0.5, label='Bullish Threshold')
        ax1.axhline(y=-15, color='red', linestyle='--', alpha=0.5, label='Bearish Threshold')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.set_ylabel('Bias Score')
        ax1.set_title('Market Bias Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Key Levels
        if analysis['key_levels']['pivot_points']:
            levels = []
            labels = []
            for level_type, level_data in analysis['key_levels']['pivot_points'].items():
                levels.append(level_data['price'])
                labels.append(f"{level_type.upper()}: {level_data['price']:.2f}")
            
            ax2.barh(labels, levels, alpha=0.7, color='blue')
            ax2.set_xlabel('Price Level')
            ax2.set_title('Key Pivot Points')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Support and Resistance Levels
        support_levels = analysis['key_levels']['support_levels']
        resistance_levels = analysis['key_levels']['resistance_levels']
        
        if support_levels or resistance_levels:
            all_levels = []
            all_labels = []
            all_colors = []
            
            for level in support_levels:
                all_levels.append(level['price'])
                all_labels.append(f"S: {level['price']:.2f}")
                all_colors.append('green')
            
            for level in resistance_levels:
                all_levels.append(level['price'])
                all_labels.append(f"R: {level['price']:.2f}")
                all_colors.append('red')
            
            if all_levels:
                ax3.barh(all_labels, all_levels, color=all_colors, alpha=0.7)
                ax3.set_xlabel('Price Level')
                ax3.set_title('Support & Resistance Levels')
                ax3.grid(True, alpha=0.3)
        
        # Plot 4: Trading Recommendations
        recommendations = analysis['trading_recommendations']
        ax4.text(0.1, 0.9, f"Primary Bias: {recommendations['primary_bias']}", 
                transform=ax4.transAxes, fontsize=12, fontweight='bold')
        ax4.text(0.1, 0.8, f"Bias Strength: {recommendations['bias_strength']}", 
                transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.7, f"Risk Level: {recommendations['risk_level']}", 
                transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.6, f"Position Sizing: {recommendations['position_sizing']}", 
                transform=ax4.transAxes, fontsize=12)
        
        ax4.set_title('Trading Recommendations')
        ax4.axis('off')
        
        plt.suptitle(f"Market Bias Analysis - {analysis['analysis_date']}", 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Convert to base64
        import io
        import base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"


# Example usage
if __name__ == "__main__":
    # Configuration
    DB_CONFIG = {
        'host': 'localhost',
        'database': 'mydb',
        'user': 'postgres',
        'password': 'postgres',
        'port': 5434
    }
    
    # Initialize components
    db_fetcher = PostgresDataFetcher(**DB_CONFIG)
    bias_analyzer = MarketBiasAnalyzer(db_fetcher, instrument_token=256265, tick_size=5.0)
    
    # Generate comprehensive analysis
    analysis = bias_analyzer.generate_comprehensive_analysis("2025-01-20")
    
    # Print results
    print("=== MARKET BIAS ANALYSIS ===")
    print(f"Analysis Date: {analysis['analysis_date']}")
    print(f"Comprehensive Bias Score: {analysis['comprehensive_bias']['bias_score']:.2f}")
    print(f"Bias Direction: {analysis['comprehensive_bias']['bias_direction']}")
    print(f"Bias Strength: {analysis['comprehensive_bias']['bias_strength']}")
    
    print("\n=== KEY LEVELS ===")
    for level_type, levels in analysis['key_levels'].items():
        if levels:
            print(f"{level_type}: {levels}")
    
    print("\n=== TRADING RECOMMENDATIONS ===")
    recommendations = analysis['trading_recommendations']
    print(f"Primary Bias: {recommendations['primary_bias']}")
    print(f"Risk Level: {recommendations['risk_level']}")
    print(f"Position Sizing: {recommendations['position_sizing']}")
    
    # Generate plot
    plot_image = bias_analyzer.plot_bias_analysis(analysis)
    print(f"\nPlot generated successfully (Base64 length: {len(plot_image)})")
