"""
Intraday Options Trading Suggestions Engine
Generates 15-30 minute timeframe options trading suggestions based on:
- Intraday POC changes
- Futures and Options order flow
- Real-time market structure
"""

import logging
import pandas as pd
import numpy as np
import os
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from market.CalculateTPO import TPOProfile, PostgresDataFetcher
from market.OrderFlowAnalyzer import OrderFlowAnalyzer
from options.OptionsDataFetcher import OptionsDataFetcher
from options.OptionsStrategies import OptionsStrategies
from common.Boilerplate import get_db_connection


class IntradayOptionsSuggestions:
    """
    Generate intraday options trading suggestions based on POC changes and order flow
    """
    
    def __init__(self, 
                 instrument_token: int = 256265,
                 futures_token: int = 12683010,
                 tick_size: float = 5.0,
                 timeframe_minutes: int = 15,
                 db_config: Optional[Dict] = None):
        """
        Initialize Intraday Options Suggestions Engine
        
        Args:
            instrument_token: Nifty spot instrument token (default: 256265)
            futures_token: Nifty futures instrument token (default: 12683010)
            tick_size: Price tick size for TPO calculation (default: 5.0)
            timeframe_minutes: Timeframe for analysis in minutes (default: 15)
            db_config: Database configuration dictionary (optional, will use env vars if not provided)
        """
        self.instrument_token = instrument_token
        self.futures_token = futures_token
        self.tick_size = tick_size
        self.timeframe_minutes = timeframe_minutes
        
        # Get database configuration
        if db_config is None:
            db_config = {
                'host': os.getenv('PG_HOST', 'postgres'),
                'database': os.getenv('PG_DATABASE', 'mydb'),
                'user': os.getenv('PG_USER', 'postgres'),
                'password': os.getenv('PG_PASSWORD', 'postgres'),
                'port': int(os.getenv('PG_PORT', 5432))
            }
        
        # Initialize components
        self.db_fetcher = PostgresDataFetcher(**db_config)
        self.order_flow_analyzer = OrderFlowAnalyzer(instrument_token=futures_token)
        self.options_fetcher = OptionsDataFetcher()
        self.strategies_generator = OptionsStrategies(risk_free_rate=0.065)
        self._backtester = None  # Lazy initialization to avoid circular import
        
        # Store previous POC for change detection
        self.previous_poc = None
        self.previous_poc_time = None
    
    def generate_suggestions(self, 
                           analysis_date: Optional[date] = None,
                           current_time: Optional[str] = None) -> Dict:
        """
        Generate intraday options trading suggestions
        
        Args:
            analysis_date: Analysis date (defaults to today)
            current_time: Current time in HH:MM:SS format (defaults to now)
            
        Returns:
            Dictionary containing trading suggestions
        """
        if analysis_date is None:
            analysis_date = date.today()
        
        if current_time is None:
            from pytz import timezone as _tz
            current_time = datetime.now(_tz('Asia/Kolkata')).strftime('%H:%M:%S')
        
        try:
            # Step 1: Get current intraday POC
            current_poc_data = self._get_current_intraday_poc(analysis_date, current_time)
            
            if not current_poc_data or current_poc_data.get('poc') is None:
                return {
                    'success': False,
                    'message': 'Insufficient data for POC calculation',
                    'suggestions': []
                }
            
            current_poc = current_poc_data['poc']
            current_price = current_poc_data.get('current_price')
            
            # Step 2: Detect POC change
            poc_change = self._detect_poc_change(current_poc, current_poc_data.get('timestamp'))
            
            # Step 3: Analyze order flow (futures and options)
            order_flow_analysis = self._analyze_order_flow(analysis_date, current_time)
            
            # Step 4: Get options chain data
            options_chain = self._get_relevant_options(current_price, current_poc)
            
            # Step 5: Generate short-term trading suggestions (15-30 min)
            short_term_suggestions = self._generate_trade_suggestions(
                poc_change=poc_change,
                current_price=current_price,
                current_poc=current_poc,
                order_flow=order_flow_analysis,
                options_chain=options_chain,
                analysis_date=analysis_date,
                timeframe_days=None  # Short-term
            )
            
            # Step 6: Generate longer-term trading suggestions (3-5 days)
            # Note: timeframe_days can be passed via generate_suggestions if needed
            long_term_suggestions = self._generate_long_term_suggestions(
                current_price=current_price,
                current_poc=current_poc,
                poc_change=poc_change,
                order_flow=order_flow_analysis,
                options_chain=options_chain,
                analysis_date=analysis_date,
                timeframe_days=None  # Will default to 3, can be overridden
            )
            
            return {
                'success': True,
                'analysis_date': analysis_date.strftime('%Y-%m-%d'),
                'current_time': current_time,
                'current_poc': current_poc,
                'current_price': current_price,
                'poc_change': poc_change,
                'order_flow_sentiment': order_flow_analysis.get('overall_sentiment', 'Neutral'),
                'short_term_suggestions': short_term_suggestions,
                'long_term_suggestions': long_term_suggestions,
                'timeframe_minutes': self.timeframe_minutes
            }
            
        except Exception as e:
            logging.error(f"Error generating intraday options suggestions: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'suggestions': []
            }
    
    def _get_current_intraday_poc(self, 
                                  analysis_date: date,
                                  current_time: str) -> Optional[Dict]:
        """
        Get current intraday POC from recent tick data
        
        Args:
            analysis_date: Analysis date
            current_time: Current time in HH:MM:SS format
            
        Returns:
            Dictionary with POC data or None
        """
        try:
            # Calculate time window (last N minutes)
            time_window_start = (datetime.strptime(current_time, '%H:%M:%S') - 
                               timedelta(minutes=self.timeframe_minutes)).strftime('%H:%M:%S')
            
            # Fetch tick data for the time window
            tick_data = self.db_fetcher.fetch_tick_data(
                table_name='ticks',
                instrument_token=self.instrument_token,
                start_time=f'{analysis_date} {time_window_start}.000',
                end_time=f'{analysis_date} {current_time}.000'
            )
            
            if tick_data.empty:
                return None
            
            # Calculate TPO profile for this window
            tpo_profile = TPOProfile(tick_size=self.tick_size)
            tpo_profile.calculate_tpo(tick_data)
            
            if tpo_profile.poc is None:
                return None
            
            current_price = tick_data['last_price'].iloc[-1] if not tick_data.empty else None
            
            return {
                'poc': tpo_profile.poc,
                'poc_low': tpo_profile.poc_low,
                'poc_high': tpo_profile.poc_high,
                'value_area_high': tpo_profile.value_area_high,
                'value_area_low': tpo_profile.value_area_low,
                'current_price': current_price,
                'timestamp': current_time,
                'tick_count': len(tick_data)
            }
            
        except Exception as e:
            logging.error(f"Error getting current intraday POC: {e}")
            return None
    
    def _detect_poc_change(self, current_poc: float, current_time: str) -> Dict:
        """
        Detect POC change from previous reading
        
        Args:
            current_poc: Current POC value
            current_time: Current time
            
        Returns:
            Dictionary with POC change analysis
        """
        if self.previous_poc is None:
            # First reading, no change detected
            self.previous_poc = current_poc
            self.previous_poc_time = current_time
            return {
                'change': 0.0,
                'change_pct': 0.0,
                'direction': 'None',
                'strength': 'None',
                'is_first_reading': True
            }
        
        # Calculate change
        poc_change = current_poc - self.previous_poc
        poc_change_pct = (poc_change / self.previous_poc * 100) if self.previous_poc > 0 else 0.0
        
        # Determine direction and strength
        direction = 'Up' if poc_change > 0 else 'Down' if poc_change < 0 else 'Neutral'
        
        # Strength based on percentage change
        abs_change_pct = abs(poc_change_pct)
        if abs_change_pct > 0.3:
            strength = 'Strong'
        elif abs_change_pct > 0.15:
            strength = 'Moderate'
        elif abs_change_pct > 0.05:
            strength = 'Weak'
        else:
            strength = 'None'
        
        # Update previous values
        self.previous_poc = current_poc
        self.previous_poc_time = current_time
        
        return {
            'change': poc_change,
            'change_pct': poc_change_pct,
            'direction': direction,
            'strength': strength,
            'is_first_reading': False,
            'previous_poc': self.previous_poc,
            'time_elapsed_minutes': self._calculate_time_diff(self.previous_poc_time, current_time)
        }
    
    def _calculate_time_diff(self, time1: str, time2: str) -> float:
        """Calculate time difference in minutes"""
        try:
            t1 = datetime.strptime(time1, '%H:%M:%S')
            t2 = datetime.strptime(time2, '%H:%M:%S')
            diff = (t2 - t1).total_seconds() / 60.0
            return max(0, diff)  # Ensure non-negative
        except:
            return 0.0
    
    def _analyze_order_flow(self, 
                           analysis_date: date,
                           current_time: str) -> Dict:
        """
        Analyze order flow from futures and options
        
        Args:
            analysis_date: Analysis date
            current_time: Current time
            
        Returns:
            Dictionary with order flow analysis
        """
        try:
            # Calculate time window
            time_window_start = (datetime.strptime(current_time, '%H:%M:%S') - 
                               timedelta(minutes=self.timeframe_minutes)).strftime('%H:%M:%S')
            
            # Analyze futures order flow
            futures_order_flow = self.order_flow_analyzer.analyze_order_flow(
                start_time=time_window_start,
                end_time=current_time,
                analysis_date=analysis_date,
                lookback_periods=10
            )
            
            # Analyze options order flow (from options_ticks)
            options_order_flow = self._analyze_options_order_flow(
                analysis_date, time_window_start, current_time
            )
            
            # Combine analysis
            combined_sentiment = self._combine_order_flow_sentiment(
                futures_order_flow, options_order_flow
            )
            
            return {
                'futures_flow': futures_order_flow,
                'options_flow': options_order_flow,
                'overall_sentiment': combined_sentiment,
                'time_window': f"{time_window_start} - {current_time}"
            }
            
        except Exception as e:
            logging.error(f"Error analyzing order flow: {e}")
            return {
                'futures_flow': {},
                'options_flow': {},
                'overall_sentiment': 'Neutral',
                'error': str(e)
            }
    
    def _analyze_options_order_flow(self,
                                   analysis_date: date,
                                   start_time: str,
                                   end_time: str) -> Dict:
        """
        Analyze options order flow from options_ticks table
        
        Args:
            analysis_date: Analysis date
            start_time: Start time
            end_time: End time
            
        Returns:
            Dictionary with options order flow analysis
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get options tick data with buy/sell quantities
            query = """
                SELECT 
                    option_type,
                    SUM(buy_quantity) as total_buy_qty,
                    SUM(sell_quantity) as total_sell_qty,
                    SUM(volume) as total_volume,
                    AVG(last_price) as avg_price
                FROM my_schema.options_ticks
                WHERE run_date = %s
                AND timestamp >= %s
                AND timestamp <= %s
                AND buy_quantity IS NOT NULL
                AND sell_quantity IS NOT NULL
                GROUP BY option_type
            """
            
            start_datetime = f"{analysis_date} {start_time}"
            end_datetime = f"{analysis_date} {end_time}"
            
            cursor.execute(query, (analysis_date, start_datetime, end_datetime))
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                return {
                    'ce_buy_pressure': 0,
                    'ce_sell_pressure': 0,
                    'pe_buy_pressure': 0,
                    'pe_sell_pressure': 0,
                    'net_sentiment': 'Neutral'
                }
            
            # Analyze buy/sell pressure by option type
            ce_buy = 0
            ce_sell = 0
            pe_buy = 0
            pe_sell = 0
            
            for row in rows:
                option_type = row[0]
                buy_qty = row[1] or 0
                sell_qty = row[2] or 0
                
                if option_type == 'CE':
                    ce_buy += buy_qty
                    ce_sell += sell_qty
                elif option_type == 'PE':
                    pe_buy += buy_qty
                    pe_sell += sell_qty
            
            # Calculate net sentiment
            ce_net = ce_buy - ce_sell
            pe_net = pe_buy - pe_sell
            
            if ce_net > pe_net and ce_net > 0:
                net_sentiment = 'Bullish'
            elif pe_net > ce_net and pe_net > 0:
                net_sentiment = 'Bearish'
            else:
                net_sentiment = 'Neutral'
            
            return {
                'ce_buy_pressure': ce_buy,
                'ce_sell_pressure': ce_sell,
                'pe_buy_pressure': pe_buy,
                'pe_sell_pressure': pe_sell,
                'ce_net': ce_net,
                'pe_net': pe_net,
                'net_sentiment': net_sentiment
            }
            
        except Exception as e:
            logging.error(f"Error analyzing options order flow: {e}")
            return {
                'ce_buy_pressure': 0,
                'ce_sell_pressure': 0,
                'pe_buy_pressure': 0,
                'pe_sell_pressure': 0,
                'net_sentiment': 'Neutral',
                'error': str(e)
            }
    
    def _combine_order_flow_sentiment(self,
                                     futures_flow: Dict,
                                     options_flow: Dict) -> str:
        """
        Combine futures and options order flow sentiment
        
        Args:
            futures_flow: Futures order flow analysis
            options_flow: Options order flow analysis
            
        Returns:
            Combined sentiment string
        """
        futures_sentiment = futures_flow.get('overall_sentiment', 'Neutral')
        options_sentiment = options_flow.get('net_sentiment', 'Neutral')
        
        # Combine sentiments
        if futures_sentiment == 'Bullish' and options_sentiment == 'Bullish':
            return 'Strong Bullish'
        elif futures_sentiment == 'Bearish' and options_sentiment == 'Bearish':
            return 'Strong Bearish'
        elif futures_sentiment == 'Bullish' or options_sentiment == 'Bullish':
            return 'Bullish'
        elif futures_sentiment == 'Bearish' or options_sentiment == 'Bearish':
            return 'Bearish'
        else:
            return 'Neutral'
    
    def _get_relevant_options(self,
                             current_price: float,
                             current_poc: float) -> pd.DataFrame:
        """
        Get relevant options around current price and POC
        
        Args:
            current_price: Current spot price
            current_poc: Current POC
            
        Returns:
            DataFrame with relevant options
        """
        try:
            # Get options within 2% of current price
            price_range = current_price * 0.02
            strike_range = (current_price - price_range, current_price + price_range)
            
            # Get options chain
            options_chain = self.options_fetcher.get_options_chain(
                expiry=None,  # Get all expiries
                strike_range=strike_range,
                option_type=None,  # Both CE and PE
                min_volume=100,  # Minimum volume filter
                min_oi=1000  # Minimum OI filter
            )
            
            if options_chain.empty:
                return pd.DataFrame()
            
            # Sort by proximity to POC and current price
            options_chain['distance_from_poc'] = abs(options_chain['strike_price'] - current_poc)
            options_chain['distance_from_price'] = abs(options_chain['strike_price'] - current_price)
            
            # Sort by distance from POC first, then by volume
            options_chain = options_chain.sort_values(
                by=['distance_from_poc', 'volume'],
                ascending=[True, False]
            )
            
            return options_chain.head(20)  # Return top 20 options
            
        except Exception as e:
            logging.error(f"Error getting relevant options: {e}")
            return pd.DataFrame()
    
    def _generate_trade_suggestions(self,
                                   poc_change: Dict,
                                   current_price: float,
                                   current_poc: float,
                                   order_flow: Dict,
                                   options_chain: pd.DataFrame,
                                   analysis_date: date,
                                   timeframe_days: Optional[int] = None) -> List[Dict]:
        """
        Generate trading suggestions based on POC change and order flow
        
        Args:
            poc_change: POC change analysis
            order_flow: Order flow analysis
            options_chain: Relevant options chain
            analysis_date: Analysis date
            
        Returns:
            List of trading suggestions
        """
        suggestions = []
        
        if options_chain.empty:
            return suggestions
        
        # Determine trade direction based on POC change and order flow
        poc_direction = poc_change.get('direction', 'Neutral')
        poc_strength = poc_change.get('strength', 'None')
        order_flow_sentiment = order_flow.get('overall_sentiment', 'Neutral')
        
        # Generate suggestions based on signals
        if poc_direction == 'Up' and poc_strength in ['Moderate', 'Strong']:
            # POC moving up - bullish signal
            if order_flow_sentiment in ['Bullish', 'Strong Bullish']:
                # Strong bullish signal - suggest CE options
                suggestions.extend(self._generate_ce_suggestions(
                    options_chain, current_price, current_poc, 'Strong Bullish', analysis_date, timeframe_days
                ))
        
        elif poc_direction == 'Down' and poc_strength in ['Moderate', 'Strong']:
            # POC moving down - bearish signal
            if order_flow_sentiment in ['Bearish', 'Strong Bearish']:
                # Strong bearish signal - suggest PE options
                suggestions.extend(self._generate_pe_suggestions(
                    options_chain, current_price, current_poc, 'Strong Bearish', timeframe_days, analysis_date
                ))
        
        # Add neutral/range-bound suggestions if POC is stable
        if poc_direction == 'Neutral' or poc_strength == 'None':
            # Range-bound market - suggest straddle/strangle strategies
            suggestions.extend(self._generate_range_suggestions(
                options_chain, current_price, current_poc, timeframe_days=timeframe_days
            ))
        
        # Generate multi-leg strategies based on market conditions
        multi_leg_suggestions = self._generate_multi_leg_strategies(
            options_chain, current_price, current_poc, poc_change, order_flow
        )
        suggestions.extend(multi_leg_suggestions)
        
        # Sort suggestions by confidence/priority
        suggestions.sort(key=lambda x: x.get('confidence_score', 0), reverse=True)
        
        return suggestions[:10]  # Return top 10 suggestions (including multi-leg)
    
    def _get_backtester(self):
        """Lazy initialization of OptionsBacktester to avoid circular import"""
        if self._backtester is None:
            from options.OptionsBacktester import OptionsBacktester
            self._backtester = OptionsBacktester()
        return self._backtester
    
    def _calculate_payoff_metrics(self, suggestion: Dict, current_price: float, analysis_date: date) -> Dict:
        """
        Calculate payoff metrics for a suggestion using OptionsBacktester
        
        Args:
            suggestion: Trading suggestion dictionary
            current_price: Current underlying price
            analysis_date: Analysis date
            
        Returns:
            Dictionary with payoff metrics
        """
        try:
            # Get required fields from suggestion
            entry_price = suggestion.get('entry_price', 0)
            strike = suggestion.get('strike_price', 0)
            option_type = suggestion.get('option_type', 'CE')
            expiry = suggestion.get('expiry')
            
            if not entry_price or not strike or not option_type:
                return {
                    'max_profit': 0.0,
                    'max_loss': -entry_price if entry_price > 0 else 0.0,
                    'risk_reward_ratio': 0.0,
                    'breakeven': 0.0,
                    'probability_of_profit': 0.0,
                    'payoff_chart': None,
                    'payoff_sparkline': None
                }
            
            # Use backtester's _calculate_trade_metrics method
            # For suggestions, we use analysis_date as entry_date and expiry as exit_date
            # If expiry is not available, use a default time period
            if expiry:
                if isinstance(expiry, str):
                    try:
                        expiry_date = datetime.strptime(expiry, '%Y-%m-%d').date()
                    except:
                        expiry_date = analysis_date + timedelta(days=7)  # Default 7 days
                elif isinstance(expiry, date):
                    expiry_date = expiry
                else:
                    expiry_date = analysis_date + timedelta(days=7)
            else:
                expiry_date = analysis_date + timedelta(days=7)
            
            # Calculate metrics using backtester
            # We need to create a temporary suggestion dict for the backtester
            temp_suggestion = {
                'strike_price': strike,
                'option_type': option_type,
                'expiry': expiry_date
            }
            
            # Use the backtester's internal method to calculate metrics
            # We'll use the underlying price calculation from backtester
            backtester = self._get_backtester()
            metrics = backtester._calculate_trade_metrics(
                suggestion=temp_suggestion,
                entry_date=analysis_date,
                exit_date=expiry_date,
                entry_price=entry_price,
                exit_price=entry_price  # For suggestions, we use entry price as exit price for calculation
            )
            
            return {
                'max_profit': metrics.get('max_profit', 0.0),
                'max_loss': metrics.get('max_loss', 0.0),
                'risk_reward_ratio': metrics.get('risk_reward_ratio', 0.0),
                'breakeven': metrics.get('breakeven', 0.0),
                'probability_of_profit': metrics.get('probability_of_profit', 0.0),
                'payoff_chart': metrics.get('payoff_chart'),
                'payoff_sparkline': metrics.get('payoff_sparkline')
            }
            
        except Exception as e:
            logging.warning(f"Error calculating payoff metrics: {e}")
            return {
                'max_profit': 0.0,
                'max_loss': -suggestion.get('entry_price', 0) if suggestion.get('entry_price', 0) > 0 else 0.0,
                'risk_reward_ratio': 0.0,
                'breakeven': 0.0,
                'probability_of_profit': 0.0,
                'payoff_chart': None,
                'payoff_sparkline': None
            }
    
    def _generate_ce_suggestions(self,
                                options_chain: pd.DataFrame,
                                current_price: float,
                                current_poc: float,
                                sentiment: str,
                                analysis_date: Optional[date] = None,
                                timeframe_days: Optional[int] = None) -> List[Dict]:
        """Generate Call option suggestions"""
        suggestions = []
        
        # Filter for CE options
        ce_options = options_chain[options_chain['option_type'] == 'CE'].head(5)
        
        for _, option in ce_options.iterrows():
            strike = float(option['strike_price']) if pd.notna(option['strike_price']) else 0
            premium = float(option['last_price']) if pd.notna(option['last_price']) else 0
            volume = int(option['volume']) if 'volume' in option and pd.notna(option['volume']) else 0
            oi = int(option['oi']) if 'oi' in option and pd.notna(option['oi']) else 0
            tradingsymbol = str(option['tradingsymbol']) if 'tradingsymbol' in option and pd.notna(option['tradingsymbol']) else ''
            expiry = option['expiry'] if 'expiry' in option and pd.notna(option['expiry']) else None
            
            # Calculate target and stop loss based on timeframe
            if timeframe_days:
                # Longer-term: larger targets
                target_price = current_price * (1.02 if timeframe_days == 3 else 1.03)  # 2-3% target
                stop_loss_price = current_price * (0.98 if timeframe_days == 3 else 0.97)  # 2-3% stop loss
                target_premium = premium * (1.5 if timeframe_days == 3 else 1.8)  # 50-80% premium increase
                stop_loss_premium = premium * 0.7  # 30% premium decrease
            else:
                # Short-term: smaller targets
                target_price = current_price * 1.0075  # 0.75% target
                stop_loss_price = current_price * 0.995  # 0.5% stop loss
                target_premium = premium * 1.2  # 20% premium increase
                stop_loss_premium = premium * 0.8  # 20% premium decrease
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                strike, current_poc, current_price, volume, oi, 'CE', sentiment
            )
            
            suggestion = {
                'symbol': tradingsymbol,
                'option_type': 'CE',
                'strike_price': strike,
                'current_premium': premium,
                'entry_price': premium,
                'target_price': target_premium,
                'stop_loss': stop_loss_premium,
                'target_underlying': target_price,
                'stop_loss_underlying': stop_loss_price,
                'timeframe': f'{self.timeframe_minutes} minutes' if timeframe_days is None else f'{timeframe_days} days',
                'rationale': f'POC moving up with {sentiment} order flow. CE option at {strike} strike.' + (f' ({timeframe_days}-day hold)' if timeframe_days else ''),
                'confidence_score': confidence_score,
                'confidence': 'High' if confidence_score > 70 else 'Medium' if confidence_score > 50 else 'Low',
                'volume': volume,
                'oi': oi,
                'expiry': expiry
            }
            
            # Calculate payoff metrics
            payoff_metrics = self._calculate_payoff_metrics(
                suggestion, current_price, analysis_date if analysis_date else date.today()
            )
            suggestion.update(payoff_metrics)
            
            suggestions.append(suggestion)
        
        return suggestions
    
    def _generate_pe_suggestions(self,
                                options_chain: pd.DataFrame,
                                current_price: float,
                                current_poc: float,
                                sentiment: str,
                                timeframe_days: Optional[int] = None,
                                analysis_date: Optional[date] = None) -> List[Dict]:
        """Generate Put option suggestions"""
        suggestions = []
        
        # Filter for PE options
        pe_options = options_chain[options_chain['option_type'] == 'PE'].head(5)
        
        for _, option in pe_options.iterrows():
            strike = float(option['strike_price']) if pd.notna(option['strike_price']) else 0
            premium = float(option['last_price']) if pd.notna(option['last_price']) else 0
            volume = int(option['volume']) if 'volume' in option and pd.notna(option['volume']) else 0
            oi = int(option['oi']) if 'oi' in option and pd.notna(option['oi']) else 0
            tradingsymbol = str(option['tradingsymbol']) if 'tradingsymbol' in option and pd.notna(option['tradingsymbol']) else ''
            expiry = option['expiry'] if 'expiry' in option and pd.notna(option['expiry']) else None
            
            # Calculate target and stop loss based on timeframe
            if timeframe_days:
                # Longer-term: larger targets
                target_price = current_price * (0.98 if timeframe_days == 3 else 0.97)  # 2-3% target down
                stop_loss_price = current_price * (1.02 if timeframe_days == 3 else 1.03)  # 2-3% stop loss up
                target_premium = premium * (1.5 if timeframe_days == 3 else 1.8)  # 50-80% premium increase
                stop_loss_premium = premium * 0.7  # 30% premium decrease
            else:
                # Short-term: smaller targets
                target_price = current_price * 0.9925  # 0.75% target down
                stop_loss_price = current_price * 1.005  # 0.5% stop loss up
                target_premium = premium * 1.2  # 20% premium increase
                stop_loss_premium = premium * 0.8  # 20% premium decrease
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                strike, current_poc, current_price, volume, oi, 'PE', sentiment
            )
            
            suggestion = {
                'symbol': tradingsymbol,
                'option_type': 'PE',
                'strike_price': strike,
                'current_premium': premium,
                'entry_price': premium,
                'target_price': target_premium,
                'stop_loss': stop_loss_premium,
                'target_underlying': target_price,
                'stop_loss_underlying': stop_loss_price,
                'timeframe': f'{self.timeframe_minutes} minutes' if timeframe_days is None else f'{timeframe_days} days',
                'rationale': f'POC moving down with {sentiment} order flow. PE option at {strike} strike.' + (f' ({timeframe_days}-day hold)' if timeframe_days else ''),
                'confidence_score': confidence_score,
                'confidence': 'High' if confidence_score > 70 else 'Medium' if confidence_score > 50 else 'Low',
                'volume': volume,
                'oi': oi,
                'expiry': expiry
            }
            
            # Calculate payoff metrics
            payoff_metrics = self._calculate_payoff_metrics(
                suggestion, current_price, analysis_date if analysis_date else date.today()
            )
            suggestion.update(payoff_metrics)
            
            suggestions.append(suggestion)
        
        return suggestions
    
    def _generate_range_suggestions(self,
                                   options_chain: pd.DataFrame,
                                   current_price: float,
                                   current_poc: float,
                                   timeframe_days: Optional[int] = None) -> List[Dict]:
        """Generate range-bound/neutral suggestions"""
        suggestions = []
        
        # For range-bound markets, suggest ATM straddles or strangles
        # Find ATM options
        atm_ce = options_chain[
            (options_chain['option_type'] == 'CE') & 
            (abs(options_chain['strike_price'] - current_price) < current_price * 0.01)
        ].head(1)
        
        atm_pe = options_chain[
            (options_chain['option_type'] == 'PE') & 
            (abs(options_chain['strike_price'] - current_price) < current_price * 0.01)
        ].head(1)
        
        if not atm_ce.empty and not atm_pe.empty:
            ce_option = atm_ce.iloc[0]
            pe_option = atm_pe.iloc[0]
            
            ce_premium = float(ce_option['last_price']) if pd.notna(ce_option['last_price']) else 0
            pe_premium = float(pe_option['last_price']) if pd.notna(pe_option['last_price']) else 0
            total_premium = ce_premium + pe_premium
            
            ce_symbol = str(ce_option['tradingsymbol']) if 'tradingsymbol' in ce_option and pd.notna(ce_option['tradingsymbol']) else ''
            pe_symbol = str(pe_option['tradingsymbol']) if 'tradingsymbol' in pe_option and pd.notna(pe_option['tradingsymbol']) else ''
            ce_volume = int(ce_option['volume']) if 'volume' in ce_option and pd.notna(ce_option['volume']) else 0
            pe_volume = int(pe_option['volume']) if 'volume' in pe_option and pd.notna(pe_option['volume']) else 0
            expiry = ce_option['expiry'] if 'expiry' in ce_option and pd.notna(ce_option['expiry']) else None
            
            suggestions.append({
                'symbol': f"{ce_symbol} + {pe_symbol}",
                'option_type': 'Straddle',
                'strike_price': current_price,
                'current_premium': total_premium,
                'entry_price': total_premium,
                'target_price': total_premium * 1.3,  # 30% target
                'stop_loss': total_premium * 0.7,  # 30% stop loss
                'timeframe': f'{self.timeframe_minutes} minutes' if timeframe_days is None else f'{timeframe_days} days',
                'rationale': 'Range-bound market. ATM Straddle strategy for volatility breakout.' + (f' ({timeframe_days}-day hold)' if timeframe_days else ''),
                'confidence_score': 50,
                'confidence': 'Medium',
                'volume': min(ce_volume, pe_volume),
                'expiry': expiry
            })
        
        return suggestions
    
    def _calculate_confidence_score(self,
                                   strike: float,
                                   poc: float,
                                   current_price: float,
                                   volume: int,
                                   oi: int,
                                   option_type: str,
                                   sentiment: str) -> float:
        """
        Calculate confidence score for a suggestion
        
        Args:
            strike: Strike price
            poc: Current POC
            current_price: Current price
            volume: Option volume
            oi: Open interest
            option_type: 'CE' or 'PE'
            sentiment: Order flow sentiment
            
        Returns:
            Confidence score (0-100)
        """
        score = 50.0  # Base score
        
        # Strike proximity to POC (closer is better)
        distance_from_poc = abs(strike - poc)
        poc_distance_pct = (distance_from_poc / poc * 100) if poc > 0 else 100
        if poc_distance_pct < 0.5:
            score += 20
        elif poc_distance_pct < 1.0:
            score += 10
        
        # Strike proximity to current price (ATM is good)
        distance_from_price = abs(strike - current_price)
        price_distance_pct = (distance_from_price / current_price * 100) if current_price > 0 else 100
        if price_distance_pct < 0.5:
            score += 15
        elif price_distance_pct < 1.0:
            score += 8
        
        # Volume and OI (higher is better for liquidity)
        if volume > 10000:
            score += 10
        elif volume > 5000:
            score += 5
        
        if oi > 50000:
            score += 10
        elif oi > 25000:
            score += 5
        
        # Sentiment alignment
        if sentiment in ['Strong Bullish', 'Strong Bearish']:
            score += 10
        elif sentiment in ['Bullish', 'Bearish']:
            score += 5
        
        return min(100, max(0, score))
    
    def _generate_long_term_suggestions(self,
                                       current_price: float,
                                       current_poc: float,
                                       poc_change: Dict,
                                       order_flow: Dict,
                                       options_chain: pd.DataFrame,
                                       analysis_date: date,
                                       timeframe_days: Optional[int] = None) -> List[Dict]:
        """
        Generate longer-term options trading suggestions (3-5 days)
        
        Args:
            current_price: Current underlying price
            current_poc: Current POC
            poc_change: POC change analysis
            order_flow: Order flow analysis
            options_chain: Relevant options chain
            analysis_date: Analysis date
            
        Returns:
            List of longer-term trading suggestions
        """
        suggestions = []
        
        if options_chain.empty:
            return suggestions
        
        try:
            # Get daily POC data for longer-term analysis
            daily_poc_data = self._get_daily_poc_data(analysis_date, num_days=5)
            
            if not daily_poc_data:
                return suggestions
            
            # Analyze longer-term trend
            long_term_trend = self._analyze_long_term_trend(daily_poc_data, current_poc)
            
            # Determine trade direction based on longer-term trend
            trend_direction = long_term_trend.get('direction', 'Neutral')
            trend_strength = long_term_trend.get('strength', 'None')
            order_flow_sentiment = order_flow.get('overall_sentiment', 'Neutral')
            
            # Generate longer-term suggestions (1, 3, or 5 days timeframe)
            if timeframe_days is None:
                timeframe_days = 3  # Default to 3 days
            elif timeframe_days not in [1, 3, 5]:
                timeframe_days = 3  # Fallback to 3 if invalid
            
            # Generate suggestions based on longer-term signals
            if trend_direction == 'Up' and trend_strength in ['Moderate', 'Strong']:
                # Longer-term bullish trend - suggest CE options
                if order_flow_sentiment in ['Bullish', 'Strong Bullish']:
                    suggestions.extend(self._generate_ce_suggestions(
                        options_chain, current_price, current_poc, 'Strong Bullish', timeframe_days=timeframe_days
                    ))
            
            elif trend_direction == 'Down' and trend_strength in ['Moderate', 'Strong']:
                # Longer-term bearish trend - suggest PE options
                if order_flow_sentiment in ['Bearish', 'Strong Bearish']:
                    suggestions.extend(self._generate_pe_suggestions(
                        options_chain, current_price, current_poc, 'Strong Bearish', timeframe_days=timeframe_days
                    ))
            
            # Generate longer-term multi-leg strategies
            multi_leg_suggestions = self._generate_long_term_multi_leg_strategies(
                options_chain, current_price, current_poc, long_term_trend, order_flow, timeframe_days
            )
            suggestions.extend(multi_leg_suggestions)
            
            # Sort suggestions by confidence/priority
            suggestions.sort(key=lambda x: x.get('confidence_score', 0), reverse=True)
            
        except Exception as e:
            logging.error(f"Error generating long-term suggestions: {e}")
            import traceback
            logging.error(traceback.format_exc())
        
        return suggestions[:5]  # Return top 5 longer-term suggestions
    
    def _get_daily_poc_data(self, analysis_date: date, num_days: int = 5) -> Optional[List[Dict]]:
        """
        Get daily POC data for longer-term analysis
        
        Args:
            analysis_date: Analysis date
            num_days: Number of days to analyze
            
        Returns:
            List of daily POC data or None
        """
        try:
            daily_poc_list = []
            
            for i in range(num_days):
                target_date = analysis_date - timedelta(days=i)
                
                # Get full day's tick data
                day_data = self.db_fetcher.fetch_tick_data(
                    table_name='ticks',
                    instrument_token=self.instrument_token,
                    start_time=f'{target_date} 09:15:00.000 +0530',
                    end_time=f'{target_date} 15:30:00.000 +0530'
                )
                
                if day_data.empty:
                    continue
                
                # Calculate TPO profile for the day
                tpo_profile = TPOProfile(tick_size=self.tick_size)
                tpo_profile.calculate_tpo(day_data)
                
                if tpo_profile.poc is not None:
                    current_price = day_data['last_price'].iloc[-1] if not day_data.empty else None
                    daily_poc_list.append({
                        'date': target_date,
                        'poc': tpo_profile.poc,
                        'value_area_high': tpo_profile.value_area_high,
                        'value_area_low': tpo_profile.value_area_low,
                        'current_price': current_price
                    })
            
            return daily_poc_list if daily_poc_list else None
            
        except Exception as e:
            logging.error(f"Error getting daily POC data: {e}")
            return None
    
    def _analyze_long_term_trend(self, daily_poc_data: List[Dict], current_poc: float) -> Dict:
        """
        Analyze longer-term trend from daily POC data
        
        Args:
            daily_poc_data: List of daily POC data
            current_poc: Current POC
            
        Returns:
            Dictionary with trend analysis
        """
        if not daily_poc_data or len(daily_poc_data) < 3:
            return {
                'direction': 'Neutral',
                'strength': 'None',
                'trend_score': 0.0
            }
        
        # Sort by date (oldest first)
        daily_poc_data.sort(key=lambda x: x['date'])
        
        # Calculate POC trend
        pocs = [d['poc'] for d in daily_poc_data]
        
        # Calculate linear regression slope
        x = np.arange(len(pocs))
        y = np.array(pocs)
        
        if len(x) >= 2:
            slope = np.polyfit(x, y, 1)[0]
            avg_poc = np.mean(y)
            
            if avg_poc > 0:
                trend_pct = (slope / avg_poc) * 100
                
                # Determine direction and strength
                if trend_pct > 0.2:
                    direction = 'Up'
                    strength = 'Strong' if trend_pct > 0.5 else 'Moderate'
                elif trend_pct < -0.2:
                    direction = 'Down'
                    strength = 'Strong' if trend_pct < -0.5 else 'Moderate'
                else:
                    direction = 'Neutral'
                    strength = 'None'
                
                return {
                    'direction': direction,
                    'strength': strength,
                    'trend_score': trend_pct,
                    'slope': slope
                }
        
        return {
            'direction': 'Neutral',
            'strength': 'None',
            'trend_score': 0.0
        }
    
    def _generate_long_term_multi_leg_strategies(self,
                                                options_chain: pd.DataFrame,
                                                current_price: float,
                                                current_poc: float,
                                                long_term_trend: Dict,
                                                order_flow: Dict,
                                                timeframe_days: int) -> List[Dict]:
        """
        Generate longer-term multi-leg strategies (3-5 days)
        
        Args:
            options_chain: Options chain data
            current_price: Current underlying price
            current_poc: Current POC
            long_term_trend: Long-term trend analysis
            order_flow: Order flow analysis
            timeframe_days: Timeframe in days (3-5)
            
        Returns:
            List of longer-term multi-leg strategy suggestions
        """
        suggestions = []
        
        if options_chain.empty:
            return suggestions
        
        try:
            # Get ATM and nearby strikes
            atm_strikes = self._get_atm_strikes(options_chain, current_price)
            
            if not atm_strikes:
                return suggestions
            
            trend_direction = long_term_trend.get('direction', 'Neutral')
            trend_strength = long_term_trend.get('strength', 'None')
            order_flow_sentiment = order_flow.get('overall_sentiment', 'Neutral')
            
            # Generate strategies based on longer-term trend
            if trend_direction == 'Up' and trend_strength in ['Moderate', 'Strong']:
                # Longer-term bullish - suggest bull spreads, protective puts
                suggestions.extend(self._generate_bullish_strategies(
                    options_chain, current_price, atm_strikes, timeframe_days
                ))
            
            elif trend_direction == 'Down' and trend_strength in ['Moderate', 'Strong']:
                # Longer-term bearish - suggest bear spreads, covered calls
                suggestions.extend(self._generate_bearish_strategies(
                    options_chain, current_price, atm_strikes, timeframe_days
                ))
            
            # Range-bound strategies for longer-term
            if trend_direction == 'Neutral' or trend_strength == 'None':
                suggestions.extend(self._generate_range_strategies(
                    options_chain, current_price, atm_strikes, timeframe_days
                ))
        
        except Exception as e:
            logging.error(f"Error generating long-term multi-leg strategies: {e}")
        
        return suggestions
    
    def _generate_multi_leg_strategies(self,
                                      options_chain: pd.DataFrame,
                                      current_price: float,
                                      current_poc: float,
                                      poc_change: Dict,
                                      order_flow: Dict) -> List[Dict]:
        """
        Generate multi-leg options strategies
        
        Args:
            options_chain: Options chain data
            current_price: Current underlying price
            current_poc: Current POC
            poc_change: POC change analysis
            order_flow: Order flow analysis
            
        Returns:
            List of multi-leg strategy suggestions
        """
        suggestions = []
        
        if options_chain.empty:
            return suggestions
        
        poc_direction = poc_change.get('direction', 'Neutral')
        poc_strength = poc_change.get('strength', 'None')
        order_flow_sentiment = order_flow.get('overall_sentiment', 'Neutral')
        
        # Get ATM and nearby strikes
        atm_strikes = self._get_atm_strikes(options_chain, current_price)
        
        if not atm_strikes:
            return suggestions
        
        # Generate strategies based on market conditions
        if poc_direction == 'Up' and poc_strength in ['Moderate', 'Strong']:
            # Bullish market - suggest bull spreads, protective puts
            suggestions.extend(self._generate_bullish_strategies(
                options_chain, current_price, atm_strikes
            ))
        
        elif poc_direction == 'Down' and poc_strength in ['Moderate', 'Strong']:
            # Bearish market - suggest bear spreads, protective calls
            suggestions.extend(self._generate_bearish_strategies(
                options_chain, current_price, atm_strikes
            ))
        
        # Range-bound strategies (iron condor, straddle, collar)
        if poc_direction == 'Neutral' or poc_strength == 'None':
            suggestions.extend(self._generate_range_strategies(
                options_chain, current_price, atm_strikes
            ))
        
        return suggestions
    
    def _get_atm_strikes(self, options_chain: pd.DataFrame, current_price: float) -> Dict:
        """Get ATM and nearby strikes"""
        # Find strikes closest to current price
        ce_options = options_chain[options_chain['option_type'] == 'CE'].copy()
        pe_options = options_chain[options_chain['option_type'] == 'PE'].copy()
        
        if ce_options.empty or pe_options.empty:
            return {}
        
        # Find ATM strikes
        ce_options['distance'] = abs(ce_options['strike_price'] - current_price)
        pe_options['distance'] = abs(pe_options['strike_price'] - current_price)
        
        atm_ce = ce_options.nsmallest(3, 'distance')
        atm_pe = pe_options.nsmallest(3, 'distance')
        
        # Get strikes at different distances
        strikes = {
            'atm': current_price,
            'otm_ce': float(atm_ce['strike_price'].max()) if not atm_ce.empty else None,
            'itm_ce': float(atm_ce['strike_price'].min()) if not atm_ce.empty else None,
            'otm_pe': float(atm_pe['strike_price'].min()) if not atm_pe.empty else None,
            'itm_pe': float(atm_pe['strike_price'].max()) if not atm_pe.empty else None,
        }
        
        return strikes
    
    def _generate_bullish_strategies(self,
                                    options_chain: pd.DataFrame,
                                    current_price: float,
                                    strikes: Dict,
                                    timeframe_days: Optional[int] = None) -> List[Dict]:
        """Generate bullish multi-leg strategies"""
        suggestions = []
        
        try:
            # Bull Call Spread
            if strikes.get('itm_ce') and strikes.get('otm_ce'):
                itm_ce_df = options_chain[
                    (options_chain['option_type'] == 'CE') & 
                    (options_chain['strike_price'] == strikes['itm_ce'])
                ]
                otm_ce_df = options_chain[
                    (options_chain['option_type'] == 'CE') & 
                    (options_chain['strike_price'] == strikes['otm_ce'])
                ]
                
                if not itm_ce_df.empty and not otm_ce_df.empty:
                    itm_ce = itm_ce_df.iloc[0]
                    otm_ce = otm_ce_df.iloc[0]
                    
                    strategy = self.strategies_generator.generate_vertical_spread(
                        current_price=current_price,
                        lower_strike=strikes['itm_ce'],
                        upper_strike=strikes['otm_ce'],
                        lower_premium=float(itm_ce['last_price']) if pd.notna(itm_ce['last_price']) else 0,
                        upper_premium=float(otm_ce['last_price']) if pd.notna(otm_ce['last_price']) else 0,
                        option_type='CE',
                        is_bullish=True
                    )
                    strategy['confidence_score'] = 70
                    strategy['confidence'] = 'High'
                    suggestions.append(strategy)
            
            # Protective Put
            if strikes.get('itm_pe'):
                itm_pe_df = options_chain[
                    (options_chain['option_type'] == 'PE') & 
                    (options_chain['strike_price'] == strikes['itm_pe'])
                ]
                
                if not itm_pe_df.empty:
                    itm_pe = itm_pe_df.iloc[0]
                    
                    strategy = self.strategies_generator.generate_protective_put(
                        current_price=current_price,
                        strike=strikes['itm_pe'],
                        premium=float(itm_pe['last_price']) if pd.notna(itm_pe['last_price']) else 0,
                        stock_price=current_price
                    )
                    strategy['confidence_score'] = 65
                    strategy['confidence'] = 'Medium'
                    suggestions.append(strategy)
        
        except Exception as e:
            logging.error(f"Error generating bullish strategies: {e}")
        
        return suggestions
    
    def _generate_bearish_strategies(self,
                                    options_chain: pd.DataFrame,
                                    current_price: float,
                                    strikes: Dict,
                                    timeframe_days: Optional[int] = None) -> List[Dict]:
        """Generate bearish multi-leg strategies"""
        suggestions = []
        
        try:
            # Bear Put Spread
            if strikes.get('itm_pe') and strikes.get('otm_pe'):
                itm_pe_df = options_chain[
                    (options_chain['option_type'] == 'PE') & 
                    (options_chain['strike_price'] == strikes['itm_pe'])
                ]
                otm_pe_df = options_chain[
                    (options_chain['option_type'] == 'PE') & 
                    (options_chain['strike_price'] == strikes['otm_pe'])
                ]
                
                if not itm_pe_df.empty and not otm_pe_df.empty:
                    itm_pe = itm_pe_df.iloc[0]
                    otm_pe = otm_pe_df.iloc[0]
                    
                    strategy = self.strategies_generator.generate_vertical_spread(
                        current_price=current_price,
                        lower_strike=strikes['otm_pe'],
                        upper_strike=strikes['itm_pe'],
                        lower_premium=float(otm_pe['last_price']) if pd.notna(otm_pe['last_price']) else 0,
                        upper_premium=float(itm_pe['last_price']) if pd.notna(itm_pe['last_price']) else 0,
                        option_type='PE',
                        is_bullish=False
                    )
                    strategy['confidence_score'] = 70
                    strategy['confidence'] = 'High'
                    suggestions.append(strategy)
            
            # Covered Call
            if strikes.get('otm_ce'):
                otm_ce_df = options_chain[
                    (options_chain['option_type'] == 'CE') & 
                    (options_chain['strike_price'] == strikes['otm_ce'])
                ]
                
                if not otm_ce_df.empty:
                    otm_ce = otm_ce_df.iloc[0]
                    
                    strategy = self.strategies_generator.generate_covered_call(
                        current_price=current_price,
                        strike=strikes['otm_ce'],
                        premium=float(otm_ce['last_price']) if pd.notna(otm_ce['last_price']) else 0,
                        stock_price=current_price
                    )
                    strategy['confidence_score'] = 65
                    strategy['confidence'] = 'Medium'
                    suggestions.append(strategy)
        
        except Exception as e:
            logging.error(f"Error generating bearish strategies: {e}")
        
        return suggestions
    
    def _generate_range_strategies(self,
                                  options_chain: pd.DataFrame,
                                  current_price: float,
                                  strikes: Dict,
                                  timeframe_days: Optional[int] = None) -> List[Dict]:
        """Generate range-bound multi-leg strategies"""
        suggestions = []
        
        try:
            # Iron Condor
            if (strikes.get('itm_pe') and strikes.get('otm_pe') and 
                strikes.get('otm_ce') and strikes.get('itm_ce')):
                
                # Get options for iron condor
                put_lower_df = options_chain[
                    (options_chain['option_type'] == 'PE') & 
                    (options_chain['strike_price'] == strikes['itm_pe'])
                ]
                put_upper_df = options_chain[
                    (options_chain['option_type'] == 'PE') & 
                    (options_chain['strike_price'] == strikes['otm_pe'])
                ]
                call_lower_df = options_chain[
                    (options_chain['option_type'] == 'CE') & 
                    (options_chain['strike_price'] == strikes['otm_ce'])
                ]
                call_upper_df = options_chain[
                    (options_chain['option_type'] == 'CE') & 
                    (options_chain['strike_price'] == strikes['itm_ce'])
                ]
                
                if all([not put_lower_df.empty, not put_upper_df.empty, 
                       not call_lower_df.empty, not call_upper_df.empty]):
                    put_lower = put_lower_df.iloc[0]
                    put_upper = put_upper_df.iloc[0]
                    call_lower = call_lower_df.iloc[0]
                    call_upper = call_upper_df.iloc[0]
                    
                    strategy = self.strategies_generator.generate_iron_condor(
                        current_price=current_price,
                        put_lower_strike=strikes['itm_pe'],
                        put_upper_strike=strikes['otm_pe'],
                        call_lower_strike=strikes['otm_ce'],
                        call_upper_strike=strikes['itm_ce'],
                        put_lower_premium=float(put_lower['last_price']) if pd.notna(put_lower['last_price']) else 0,
                        put_upper_premium=float(put_upper['last_price']) if pd.notna(put_upper['last_price']) else 0,
                        call_lower_premium=float(call_lower['last_price']) if pd.notna(call_lower['last_price']) else 0,
                        call_upper_premium=float(call_upper['last_price']) if pd.notna(call_upper['last_price']) else 0
                    )
                    strategy['confidence_score'] = 75
                    strategy['confidence'] = 'High'
                    suggestions.append(strategy)
            
            # Straddle
            if strikes.get('atm'):
                atm_ce_df = options_chain[
                    (options_chain['option_type'] == 'CE') & 
                    (abs(options_chain['strike_price'] - strikes['atm']) < strikes['atm'] * 0.01)
                ]
                atm_pe_df = options_chain[
                    (options_chain['option_type'] == 'PE') & 
                    (abs(options_chain['strike_price'] - strikes['atm']) < strikes['atm'] * 0.01)
                ]
                
                if not atm_ce_df.empty and not atm_pe_df.empty:
                    atm_ce = atm_ce_df.iloc[0]
                    atm_pe = atm_pe_df.iloc[0]
                    
                    strategy = self.strategies_generator.generate_straddle(
                        current_price=current_price,
                        strike=strikes['atm'],
                        call_premium=float(atm_ce['last_price']) if pd.notna(atm_ce['last_price']) else 0,
                        put_premium=float(atm_pe['last_price']) if pd.notna(atm_pe['last_price']) else 0,
                        is_long=False  # Short straddle for range-bound
                    )
                    strategy['confidence_score'] = 70
                    strategy['confidence'] = 'High'
                    suggestions.append(strategy)
            
            # Protective Collar
            if strikes.get('itm_pe') and strikes.get('otm_ce'):
                itm_pe_df = options_chain[
                    (options_chain['option_type'] == 'PE') & 
                    (options_chain['strike_price'] == strikes['itm_pe'])
                ]
                otm_ce_df = options_chain[
                    (options_chain['option_type'] == 'CE') & 
                    (options_chain['strike_price'] == strikes['otm_ce'])
                ]
                
                if not itm_pe_df.empty and not otm_ce_df.empty:
                    itm_pe = itm_pe_df.iloc[0]
                    otm_ce = otm_ce_df.iloc[0]
                    
                    strategy = self.strategies_generator.generate_protective_collar(
                        current_price=current_price,
                        put_strike=strikes['itm_pe'],
                        call_strike=strikes['otm_ce'],
                        put_premium=float(itm_pe['last_price']) if pd.notna(itm_pe['last_price']) else 0,
                        call_premium=float(otm_ce['last_price']) if pd.notna(otm_ce['last_price']) else 0,
                        stock_price=current_price
                    )
                    strategy['confidence_score'] = 68
                    strategy['confidence'] = 'Medium'
                    suggestions.append(strategy)
        
        except Exception as e:
            logging.error(f"Error generating range strategies: {e}")
        
        return suggestions

