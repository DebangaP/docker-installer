"""
Options Back-testing Engine
Core back-testing system for options trading strategies
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from options.HistoricalOptionsGenerator import HistoricalOptionsGenerator
from options.BacktestMetrics import BacktestMetrics
from options.IntradayOptionsSuggestions import IntradayOptionsSuggestions
from options.OptionsStrategies import OptionsStrategies
from options.OptionsGreeksCalculator import OptionsGreeksCalculator
from common.Boilerplate import get_db_connection


class OptionsBacktester:
    """
    Back-testing engine for options trading strategies
    """
    
    def __init__(self,
                 instrument_token: int = 256265,
                 futures_token: int = 12683010,
                 risk_free_rate: float = 0.065,
                 db_config: Optional[Dict] = None):
        """
        Initialize Options Back-tester
        
        Args:
            instrument_token: Underlying instrument token (default: 256265 for Nifty)
            futures_token: Futures instrument token (default: 12683010)
            risk_free_rate: Risk-free interest rate (default: 6.5%)
            db_config: Database configuration dictionary (optional)
        """
        self.instrument_token = instrument_token
        self.futures_token = futures_token
        self.risk_free_rate = risk_free_rate
        
        # Initialize components
        self.historical_generator = HistoricalOptionsGenerator(
            instrument_token=instrument_token,
            risk_free_rate=risk_free_rate,
            db_config=db_config
        )
        self.metrics_calculator = BacktestMetrics()
        self.greeks_calculator = OptionsGreeksCalculator(risk_free_rate=risk_free_rate)
        self.strategies_generator = OptionsStrategies(risk_free_rate=risk_free_rate)
        
        # Initialize suggestions engine for generating signals
        if db_config is None:
            import os
            db_config = {
                'host': os.getenv('PG_HOST', 'postgres'),
                'database': os.getenv('PG_DATABASE', 'mydb'),
                'user': os.getenv('PG_USER', 'postgres'),
                'password': os.getenv('PG_PASSWORD', 'postgres'),
                'port': int(os.getenv('PG_PORT', 5432))
            }
        
        self.suggestions_engine = IntradayOptionsSuggestions(
            instrument_token=instrument_token,
            futures_token=futures_token,
            db_config=db_config
        )
    
    def run_backtest(self,
                    start_date: date,
                    end_date: date,
                    strategy_type: str = 'all',
                    show_only_profitable: bool = False,
                    min_profit: Optional[float] = None,
                    timeframe_minutes: int = 15) -> Dict:
        """
        Run back-testing for a date range
        
        Args:
            start_date: Start date for back-testing
            end_date: End date for back-testing
            strategy_type: 'all', 'single', 'spread', 'multi_leg'
            show_only_profitable: Filter to show only profitable trades
            min_profit: Minimum profit threshold (optional)
            timeframe_minutes: Timeframe for analysis (default: 15 minutes)
            
        Returns:
            Dictionary with back-testing results including:
            - trades: List of trades
            - metrics: Performance metrics
            - summary: Summary statistics
        """
        try:
            logging.info(f"Starting back-test from {start_date} to {end_date}")
            
            # Generate list of trading dates
            trading_dates = self._get_trading_dates(start_date, end_date)
            
            if not trading_dates:
                return {
                    'success': False,
                    'error': 'No trading dates found in range',
                    'trades': [],
                    'metrics': self.metrics_calculator._empty_metrics()
                }
            
            # Run back-testing for each date
            all_trades = []
            
            for trade_date in trading_dates:
                try:
                    # Generate suggestions for this date
                    suggestions = self._generate_suggestions_for_date(
                        trade_date, timeframe_minutes
                    )
                    
                    # Simulate trades based on suggestions
                    trades = self._simulate_trades(
                        trade_date, suggestions, end_date
                    )
                    
                    all_trades.extend(trades)
                    
                except Exception as e:
                    logging.warning(f"Error processing date {trade_date}: {e}")
                    continue
            
            # Filter trades if needed
            if show_only_profitable:
                all_trades = self.metrics_calculator.filter_profitable_trades(all_trades)
            
            if min_profit is not None:
                all_trades = self.metrics_calculator.filter_by_min_profit(all_trades, min_profit)
            
            # Calculate metrics
            metrics = self.metrics_calculator.calculate_metrics(all_trades)
            
            # Prepare summary
            summary = {
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'total_days': len(trading_dates),
                'strategy_type': strategy_type,
                'filters_applied': {
                    'show_only_profitable': show_only_profitable,
                    'min_profit': min_profit
                }
            }
            
            return {
                'success': True,
                'trades': all_trades,
                'metrics': metrics,
                'summary': summary
            }
            
        except Exception as e:
            logging.error(f"Error running back-test: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'trades': [],
                'metrics': self.metrics_calculator._empty_metrics()
            }
    
    def _get_trading_dates(self, start_date: date, end_date: date) -> List[date]:
        """
        Get list of trading dates (excluding weekends)
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of trading dates
        """
        dates = []
        current_date = start_date
        
        while current_date <= end_date:
            # Exclude weekends (Saturday=5, Sunday=6)
            if current_date.weekday() < 5:
                dates.append(current_date)
            current_date += timedelta(days=1)
        
        return dates
    
    def _generate_suggestions_for_date(self,
                                      trade_date: date,
                                      timeframe_minutes: int) -> List[Dict]:
        """
        Generate trading suggestions for a historical date
        
        Args:
            trade_date: Date to generate suggestions for
            timeframe_minutes: Timeframe for analysis
            
        Returns:
            List of trading suggestions
        """
        try:
            # Get historical options chain for the date
            options_chain = self.historical_generator.get_historical_options_chain(
                analysis_date=trade_date
            )
            
            if options_chain.empty:
                return []
            
            # Get underlying price for the date
            underlying_price = self.historical_generator._get_underlying_price(trade_date)
            
            if underlying_price is None:
                return []
            
            # Generate suggestions using the suggestions engine
            # We'll simulate the POC and order flow analysis
            suggestions = self._generate_suggestions_from_chain(
                options_chain, underlying_price, trade_date
            )
            
            return suggestions
            
        except Exception as e:
            logging.warning(f"Error generating suggestions for {trade_date}: {e}")
            return []
    
    def _generate_suggestions_from_chain(self,
                                        options_chain: pd.DataFrame,
                                        underlying_price: float,
                                        trade_date: date) -> List[Dict]:
        """
        Generate trading suggestions from options chain
        
        Args:
            options_chain: Historical options chain
            underlying_price: Underlying price
            trade_date: Trade date
            
        Returns:
            List of trading suggestions
        """
        suggestions = []
        
        try:
            # Filter for ATM and nearby strikes
            price_range = underlying_price * 0.02  # ±2%
            strike_range = (underlying_price - price_range, underlying_price + price_range)
            
            relevant_options = options_chain[
                (options_chain['strike_price'] >= strike_range[0]) &
                (options_chain['strike_price'] <= strike_range[1])
            ]
            
            if relevant_options.empty:
                return []
            
            # Ensure we have a DataFrame, not a Series
            if isinstance(relevant_options, pd.Series):
                relevant_options = relevant_options.to_frame().T
            
            # Generate simple suggestions (CE and PE options)
            for idx, option in relevant_options.head(10).iterrows():
                try:
                    # Convert numpy types to Python native types to avoid iteration issues
                    # Use .item() for numpy scalars, or float() for regular values
                    strike_val = option.get('strike_price') if 'strike_price' in option else None
                    if strike_val is not None and pd.notna(strike_val):
                        if hasattr(strike_val, 'item'):  # numpy scalar
                            strike = float(strike_val.item())
                        elif isinstance(strike_val, (np.integer, np.int64, np.int32)):
                            strike = float(int(strike_val))
                        else:
                            strike = float(strike_val)
                    else:
                        strike = 0.0
                    
                    premium_val = option.get('last_price') if 'last_price' in option else None
                    if premium_val is not None and pd.notna(premium_val):
                        if hasattr(premium_val, 'item'):  # numpy scalar
                            premium = float(premium_val.item())
                        elif isinstance(premium_val, (np.integer, np.int64, np.int32, np.floating, np.float64)):
                            premium = float(premium_val)
                        else:
                            premium = float(premium_val)
                    else:
                        premium = 0.0
                    
                    option_type_val = option.get('option_type') if 'option_type' in option else None
                    if option_type_val is not None and pd.notna(option_type_val):
                        option_type = str(option_type_val)
                    else:
                        option_type = ''
                    
                    expiry_val = option.get('expiry') if 'expiry' in option else None
                    expiry = expiry_val if expiry_val is not None and pd.notna(expiry_val) else None
                    
                    is_generated_val = option.get('is_generated', False) if 'is_generated' in option else False
                    if is_generated_val is not None and pd.notna(is_generated_val):
                        if hasattr(is_generated_val, 'item'):  # numpy scalar
                            is_generated = bool(is_generated_val.item())
                        elif isinstance(is_generated_val, (np.bool_, bool)):
                            is_generated = bool(is_generated_val)
                        else:
                            is_generated = bool(is_generated_val)
                    else:
                        is_generated = False
                    
                    data_source_val = option.get('data_source', 'unknown') if 'data_source' in option else 'unknown'
                    if data_source_val is not None and pd.notna(data_source_val):
                        data_source = str(data_source_val)
                    else:
                        data_source = 'unknown'
                    
                    # Calculate target and stop loss
                    if option_type == 'CE':
                        target_price = underlying_price * 1.01  # 1% target
                        stop_loss_price = underlying_price * 0.99  # 1% stop loss
                        target_premium = premium * 1.2  # 20% premium increase
                        stop_loss_premium = premium * 0.8  # 20% premium decrease
                    else:  # PE
                        target_price = underlying_price * 0.99  # 1% target down
                        stop_loss_price = underlying_price * 1.01  # 1% stop loss up
                        target_premium = premium * 1.2  # 20% premium increase
                        stop_loss_premium = premium * 0.8  # 20% premium decrease
                    
                    # Convert expiry to string if it's a date
                    expiry_str = None
                    if expiry:
                        if isinstance(expiry, date):
                            expiry_str = expiry.strftime('%Y-%m-%d')
                        elif isinstance(expiry, str):
                            expiry_str = expiry
                        else:
                            try:
                                expiry_str = str(expiry)
                            except:
                                expiry_str = None
                    
                    suggestions.append({
                        'symbol': str(option.get('tradingsymbol', '')) if pd.notna(option.get('tradingsymbol', '')) else '',
                        'option_type': str(option_type),
                        'strike_price': float(strike),
                        'current_premium': float(premium),
                        'entry_price': float(premium),
                        'target_price': float(target_premium),
                        'stop_loss': float(stop_loss_premium),
                        'target_underlying': float(target_price),
                        'stop_loss_underlying': float(stop_loss_price),
                        'expiry': expiry_str,
                        'is_generated': bool(is_generated),
                        'data_source': str(data_source),
                        'confidence_score': float(50.0)
                    })
                except Exception as e:
                    logging.warning(f"Error creating suggestion for option: {e}")
                    import traceback
                    logging.debug(traceback.format_exc())
                    continue
            
            return suggestions
            
        except Exception as e:
            logging.error(f"Error generating suggestions from chain: {e}")
            return []
    
    def _simulate_trades(self,
                        entry_date: date,
                        suggestions: List[Dict],
                        end_date: date) -> List[Dict]:
        """
        Simulate trades based on suggestions
        
        Args:
            entry_date: Entry date
            suggestions: List of trading suggestions
            end_date: Maximum exit date
            
        Returns:
            List of simulated trades
        """
        trades = []
        
        for suggestion in suggestions:
            try:
                # Determine exit date (target, stop-loss, or expiry)
                exit_date, exit_price, exit_reason = self._determine_exit(
                    entry_date, suggestion, end_date
                )
                
                if exit_date is None:
                    continue
                
                # Calculate profit/loss
                entry_price = suggestion.get('entry_price', 0)
                profit_loss = exit_price - entry_price
                
                # Determine if data was generated
                is_generated = suggestion.get('is_generated', False)
                data_source = suggestion.get('data_source', 'unknown')
                
                # Calculate additional metrics
                trade_metrics = self._calculate_trade_metrics(
                    suggestion, entry_date, exit_date, entry_price, exit_price
                )
                
                trade = {
                    'entry_date': entry_date.strftime('%Y-%m-%d'),
                    'exit_date': exit_date.strftime('%Y-%m-%d'),
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'profit_loss': profit_loss,
                    'symbol': suggestion.get('symbol', ''),
                    'option_type': suggestion.get('option_type', ''),
                    'strike_price': suggestion.get('strike_price', 0),
                    'expiry': suggestion.get('expiry', None),
                    'exit_reason': exit_reason,
                    'is_generated': is_generated,
                    'data_source': data_source,
                    'holding_period': (exit_date - entry_date).days,
                    **trade_metrics  # Add all calculated metrics
                }
                
                trades.append(trade)
                
            except Exception as e:
                logging.warning(f"Error simulating trade: {e}")
                continue
        
        return trades
    
    def _determine_exit(self,
                       entry_date: date,
                       suggestion: Dict,
                       max_date: date) -> Tuple[Optional[date], float, str]:
        """
        Determine exit date, price, and reason
        
        Args:
            entry_date: Entry date
            suggestion: Trading suggestion
            max_date: Maximum exit date
            
        Returns:
            Tuple of (exit_date, exit_price, exit_reason)
        """
        try:
            expiry = suggestion.get('expiry')
            target_price = suggestion.get('target_price', 0)
            stop_loss = suggestion.get('stop_loss', 0)
            entry_price = suggestion.get('entry_price', 0)
            
            # Convert expiry to date if it's a string
            expiry_date = None
            if expiry:
                if isinstance(expiry, str):
                    try:
                        from datetime import datetime
                        expiry_date = datetime.strptime(expiry, '%Y-%m-%d').date()
                    except (ValueError, TypeError):
                        try:
                            expiry_date = datetime.fromisoformat(expiry).date()
                        except (ValueError, TypeError):
                            logging.warning(f"Could not parse expiry date: {expiry}")
                            expiry_date = None
                elif isinstance(expiry, date):
                    expiry_date = expiry
                else:
                    try:
                        # Try to convert other types
                        expiry_date = expiry if hasattr(expiry, 'year') else None
                    except:
                        expiry_date = None
            
            # Check expiry first
            if expiry_date and isinstance(expiry_date, date):
                if expiry_date <= max_date:
                    # Exit at expiry
                    # Calculate intrinsic value at expiry
                    underlying_price = self.historical_generator._get_underlying_price(expiry_date)
                    if underlying_price:
                        option_type = suggestion.get('option_type', 'CE')
                        strike = suggestion.get('strike_price', 0)
                        
                        if option_type == 'CE':
                            intrinsic = max(0, underlying_price - strike)
                        else:  # PE
                            intrinsic = max(0, strike - underlying_price)
                        
                        return expiry_date, intrinsic, 'Expired'
            
            # Check target and stop-loss
            # For simplicity, check daily
            current_date = entry_date + timedelta(days=1)
            
            while current_date <= max_date:
                underlying_price = self.historical_generator._get_underlying_price(current_date)
                
                if underlying_price:
                    # Calculate option price at current date
                    option_price = self._calculate_option_price_at_date(
                        current_date, suggestion, underlying_price
                    )
                    
                    # Check stop-loss
                    if stop_loss > 0 and option_price <= stop_loss:
                        return current_date, stop_loss, 'Stop Loss Hit'
                    
                    # Check target
                    if target_price > 0 and option_price >= target_price:
                        return current_date, target_price, 'Target Achieved'
                
                current_date += timedelta(days=1)
            
            # If no exit triggered, exit at max_date or expiry
            if expiry_date:
                exit_date = min(max_date, expiry_date)
                if exit_date == expiry_date:
                    # Exit at expiry
                    underlying_price = self.historical_generator._get_underlying_price(exit_date)
                    if underlying_price:
                        option_type = suggestion.get('option_type', 'CE')
                        strike = suggestion.get('strike_price', 0)
                        if option_type == 'CE':
                            intrinsic = max(0, underlying_price - strike)
                        else:  # PE
                            intrinsic = max(0, strike - underlying_price)
                        return exit_date, intrinsic, 'Expired'
                    else:
                        return exit_date, entry_price, 'Expired (No Price Data)'
                else:
                    # Exit at max_date (before expiry)
                    exit_date = max_date
                    underlying_price = self.historical_generator._get_underlying_price(exit_date)
                    if underlying_price:
                        option_price = self._calculate_option_price_at_date(
                            exit_date, suggestion, underlying_price
                        )
                        return exit_date, option_price, 'End of Backtest Period'
                    else:
                        return exit_date, entry_price, 'End of Backtest Period (No Price Data)'
            else:
                # No expiry, exit at max_date
                exit_date = max_date
                underlying_price = self.historical_generator._get_underlying_price(exit_date)
                if underlying_price:
                    option_price = self._calculate_option_price_at_date(
                        exit_date, suggestion, underlying_price
                    )
                    return exit_date, option_price, 'End of Backtest Period'
                else:
                    return exit_date, entry_price, 'End of Backtest Period (No Price Data)'
            
            # Fallback: exit at entry price
            return exit_date, entry_price, 'No Exit Triggered'
            
        except Exception as e:
            logging.warning(f"Error determining exit: {e}")
            return None, 0.0, 'Error Determining Exit'
    
    def _calculate_option_price_at_date(self,
                                       current_date: date,
                                       suggestion: Dict,
                                       underlying_price: float) -> float:
        """
        Calculate option price at a specific date
        
        Args:
            current_date: Current date
            suggestion: Trading suggestion
            underlying_price: Current underlying price
            
        Returns:
            Option price
        """
        try:
            strike = suggestion.get('strike_price', 0)
            option_type = suggestion.get('option_type', 'CE')
            expiry = suggestion.get('expiry')
            
            if expiry is None:
                return suggestion.get('entry_price', 0)
            
            # Convert expiry to date if it's a string
            expiry_date = None
            if expiry:
                if isinstance(expiry, str):
                    try:
                        from datetime import datetime
                        expiry_date = datetime.strptime(expiry, '%Y-%m-%d').date()
                    except (ValueError, TypeError):
                        try:
                            expiry_date = datetime.fromisoformat(expiry).date()
                        except (ValueError, TypeError):
                            logging.warning(f"Could not parse expiry date in _calculate_option_price_at_date: {expiry}")
                            return suggestion.get('entry_price', 0)
                elif isinstance(expiry, date):
                    expiry_date = expiry
                else:
                    try:
                        # Try to convert other types
                        expiry_date = expiry if hasattr(expiry, 'year') else None
                    except:
                        expiry_date = None
            
            if expiry_date is None:
                return suggestion.get('entry_price', 0)
            
            # Calculate time to expiry
            time_to_expiry = self.greeks_calculator.calculate_time_to_expiry(
                expiry_date, current_date
            )
            
            if time_to_expiry <= 0:
                # At or past expiry, return intrinsic value
                if option_type == 'CE':
                    return max(0, underlying_price - strike)
                else:  # PE
                    return max(0, strike - underlying_price)
            
            # Estimate IV (use historical or default)
            iv = self.historical_generator._estimate_implied_volatility(
                current_date, underlying_price
            )
            
            # Calculate theoretical price using Black-Scholes
            greeks = self.greeks_calculator.calculate_greeks(
                spot_price=underlying_price,
                strike_price=strike,
                time_to_expiry=time_to_expiry,
                implied_volatility=iv,
                option_type=option_type
            )
            
            return greeks['theoretical_price']
            
        except Exception as e:
            logging.warning(f"Error calculating option price: {e}")
            return suggestion.get('entry_price', 0)
    
    def _calculate_trade_metrics(self,
                                 suggestion: Dict,
                                 entry_date: date,
                                 exit_date: date,
                                 entry_price: float,
                                 exit_price: float) -> Dict:
        """
        Calculate additional metrics for a trade:
        - Payoff chart
        - Payoff chart sparkline
        - Probability of Profit
        - Max Profit
        - Max Loss
        - Risk/Reward ratio
        - Breakeven
        - Estimated Margin/Premium
        
        Args:
            suggestion: Trading suggestion
            entry_date: Entry date
            exit_date: Exit date
            entry_price: Entry price
            exit_price: Exit price
            
        Returns:
            Dictionary with calculated metrics
        """
        try:
            from options.OptionsStrategies import OptionsStrategies
            import numpy as np
            import math
            
            strike = suggestion.get('strike_price', 0)
            option_type = suggestion.get('option_type', 'CE')
            expiry = suggestion.get('expiry')
            underlying_price = self.historical_generator._get_underlying_price(entry_date)
            
            if not underlying_price or strike <= 0:
                return self._get_default_metrics(entry_price)
            
            # Convert expiry to date if string
            expiry_date = None
            if expiry:
                if isinstance(expiry, str):
                    try:
                        from datetime import datetime
                        expiry_date = datetime.strptime(expiry, '%Y-%m-%d').date()
                    except:
                        pass
                elif isinstance(expiry, date):
                    expiry_date = expiry
            
            # Calculate time to expiry at entry
            if expiry_date:
                time_to_expiry = self.greeks_calculator.calculate_time_to_expiry(
                    expiry_date, entry_date
                )
            else:
                time_to_expiry = 0.1  # Default to 10 days
            
            # Estimate IV
            iv = self.historical_generator._estimate_implied_volatility(entry_date, underlying_price)
            
            # Generate price range for payoff calculation
            price_range = np.linspace(
                max(0, underlying_price * 0.7),
                underlying_price * 1.3,
                200
            )
            
            # Calculate payoffs at different underlying prices
            payoffs = []
            for price in price_range:
                if time_to_expiry <= 0:
                    # At expiry
                    if option_type == 'CE':
                        intrinsic = max(0, price - strike)
                    else:
                        intrinsic = max(0, strike - price)
                    payoff = intrinsic - entry_price
                else:
                    # Before expiry - use Black-Scholes
                    greeks = self.greeks_calculator.calculate_greeks(
                        spot_price=price,
                        strike_price=strike,
                        time_to_expiry=time_to_expiry,
                        implied_volatility=iv,
                        option_type=option_type
                    )
                    option_price = greeks['theoretical_price']
                    payoff = option_price - entry_price
                payoffs.append(payoff)
            
            payoffs = np.array(payoffs)
            
            # Calculate metrics
            max_profit = float(np.max(payoffs))
            max_loss = float(np.min(payoffs))
            
            # Breakeven calculation
            if option_type == 'CE':
                breakeven = strike + entry_price
            else:  # PE
                breakeven = strike - entry_price
            
            # Risk/Reward ratio
            risk = abs(max_loss) if max_loss < 0 else entry_price
            reward = max_profit if max_profit > 0 else 0
            risk_reward_ratio = reward / risk if risk > 0 else 0.0
            
            # Probability of Profit
            prob_of_profit = self._calculate_probability_of_profit(
                underlying_price, strike, option_type, iv, time_to_expiry, entry_price
            )
            
            # Estimated Margin/Premium (for single options, margin is typically the premium)
            estimated_margin = entry_price
            estimated_premium = entry_price
            
            # Generate payoff chart
            payoff_chart = self._generate_payoff_chart(
                price_range, payoffs, underlying_price, strike, breakeven
            )
            
            # Generate sparkline (smaller version)
            sparkline = self._generate_payoff_sparkline(price_range, payoffs)
            
            return {
                'max_profit': max_profit,
                'max_loss': max_loss,
                'risk_reward_ratio': risk_reward_ratio,
                'breakeven': breakeven,
                'probability_of_profit': prob_of_profit,
                'estimated_margin': estimated_margin,
                'estimated_premium': estimated_premium,
                'payoff_chart': payoff_chart,
                'payoff_sparkline': sparkline
            }
            
        except Exception as e:
            logging.warning(f"Error calculating trade metrics: {e}")
            return self._get_default_metrics(entry_price)
    
    def _get_default_metrics(self, entry_price: float) -> Dict:
        """Return default metrics when calculation fails"""
        return {
            'max_profit': 0.0,
            'max_loss': -entry_price if entry_price > 0 else 0.0,
            'risk_reward_ratio': 0.0,
            'breakeven': 0.0,
            'probability_of_profit': 0.0,
            'estimated_margin': entry_price,
            'estimated_premium': entry_price,
            'payoff_chart': None,
            'payoff_sparkline': None
        }
    
    def _calculate_probability_of_profit(self,
                                        current_price: float,
                                        strike: float,
                                        option_type: str,
                                        iv: float,
                                        time_to_expiry: float,
                                        entry_price: float) -> float:
        """
        Calculate probability of profit using Black-Scholes model
        """
        try:
            from scipy.stats import norm
        except ImportError:
            import math
            def norm_cdf(x):
                return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
            norm = type('norm', (), {'cdf': norm_cdf})()
        
        try:
            if time_to_expiry <= 0:
                # At expiry, calculate probability based on breakeven
                if option_type == 'CE':
                    breakeven = strike + entry_price
                    if current_price >= breakeven:
                        return 1.0
                    else:
                        return 0.0
                else:  # PE
                    breakeven = strike - entry_price
                    if current_price <= breakeven:
                        return 1.0
                    else:
                        return 0.0
            
            # Calculate d1 and d2 for breakeven price
            if option_type == 'CE':
                breakeven = strike + entry_price
            else:
                breakeven = strike - entry_price
            
            if iv <= 0 or time_to_expiry <= 0:
                return 0.5  # Default to 50% if invalid inputs
            
            sqrt_T = math.sqrt(time_to_expiry)
            d1 = (math.log(current_price / breakeven) + (self.risk_free_rate + (iv ** 2) / 2.0) * time_to_expiry) / (iv * sqrt_T)
            d2 = d1 - iv * sqrt_T
            
            if option_type == 'CE':
                # Probability that price > breakeven (profit)
                # N(d2) gives probability that price < breakeven, so 1 - N(d2) gives probability > breakeven
                prob = 1.0 - norm.cdf(d2)
            else:  # PE
                # Probability that price < breakeven (profit)
                # N(d2) gives probability that price < breakeven
                prob = norm.cdf(d2)
            
            prob_value = float(prob) if not (math.isnan(prob) or math.isinf(prob)) else 0.5
            # Return as percentage (0-100)
            return prob_value * 100.0
            
        except Exception as e:
            logging.debug(f"Error calculating probability of profit: {e}")
            return 50.0  # Return 50% as default
    
    def _generate_payoff_chart(self,
                              price_range: np.ndarray,
                              payoffs: np.ndarray,
                              current_price: float,
                              strike: float,
                              breakeven: float) -> Optional[str]:
        """
        Generate payoff chart as base64 encoded image
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            import io
            import base64
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot payoff curve
            ax.plot(price_range, payoffs, 'b-', linewidth=2, label='Payoff')
            ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
            ax.axvline(x=current_price, color='g', linestyle='--', linewidth=1, label='Current Price')
            ax.axvline(x=strike, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Strike')
            if breakeven > 0:
                ax.axvline(x=breakeven, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Breakeven')
            
            ax.fill_between(price_range, 0, payoffs, where=(payoffs >= 0), alpha=0.3, color='green', label='Profit Zone')
            ax.fill_between(price_range, 0, payoffs, where=(payoffs < 0), alpha=0.3, color='red', label='Loss Zone')
            
            ax.set_xlabel('Underlying Price')
            ax.set_ylabel('Payoff')
            ax.set_title('Payoff Chart')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Save to base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            logging.warning(f"Error generating payoff chart: {e}")
            return None
    
    def _generate_payoff_sparkline(self,
                                   price_range: np.ndarray,
                                   payoffs: np.ndarray) -> Optional[str]:
        """
        Generate small sparkline version of payoff chart
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import io
            import base64
            
            fig, ax = plt.subplots(figsize=(3, 1.5))
            
            # Plot simplified payoff curve
            ax.plot(price_range, payoffs, 'b-', linewidth=1.5)
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
            ax.fill_between(price_range, 0, payoffs, where=(payoffs >= 0), alpha=0.3, color='green')
            ax.fill_between(price_range, 0, payoffs, where=(payoffs < 0), alpha=0.3, color='red')
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            
            # Save to base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=50, bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            logging.warning(f"Error generating payoff sparkline: {e}")
            return None
    
    def _sanitize_for_json(self, obj):
        """
        Recursively sanitize data for JSON serialization
        Replace NaN, Infinity, and -Infinity with None or 0
        """
        import math
        import numpy as np
        
        if isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._sanitize_for_json(item) for item in obj]
        elif isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            val = float(obj)
            if math.isnan(val) or math.isinf(val):
                return None
            return val
        elif hasattr(obj, 'item'):  # numpy scalar
            val = obj.item()
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                return None
            return val
        else:
            return obj
    
    def get_existing_backtest(self,
                              start_date: date,
                              end_date: date,
                              strategy_type: str = 'all',
                              show_only_profitable: bool = False,
                              min_profit: Optional[float] = None,
                              timeframe_minutes: int = 15) -> Optional[Dict]:
        """
        Check if a backtest with the same parameters already exists in the database
        
        Args:
            start_date: Start date for back-testing
            end_date: End date for back-testing
            strategy_type: Strategy type
            show_only_profitable: Filter for profitable trades
            min_profit: Minimum profit threshold
            timeframe_minutes: Timeframe in minutes
            
        Returns:
            Dictionary with existing backtest results if found, None otherwise
        """
        try:
            from common.Boilerplate import get_db_connection
            import json
            from datetime import datetime
            
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Query for existing backtest with matching parameters
            query = """
                SELECT 
                    id, backtest_name, start_date, end_date, strategy_type, timeframe_minutes,
                    show_only_profitable, min_profit, total_trades, win_count, loss_count,
                    win_rate, total_profit_loss, avg_profit_loss, gross_profit, gross_loss,
                    avg_win, avg_loss, max_profit, max_loss, profit_factor, sharpe_ratio,
                    max_drawdown, avg_holding_period, confidence_score, data_quality, metrics, summary,
                    created_at
                FROM my_schema.options_backtest_results
                WHERE start_date = %s 
                AND end_date = %s
                AND (strategy_type = %s OR (strategy_type IS NULL AND %s = 'all'))
                AND (timeframe_minutes = %s OR (timeframe_minutes IS NULL AND %s = 15))
                AND show_only_profitable = %s
                AND (min_profit = %s OR (min_profit IS NULL AND %s IS NULL))
                ORDER BY created_at DESC
                LIMIT 1
            """
            
            cursor.execute(query, (
                start_date, end_date, strategy_type, strategy_type,
                timeframe_minutes, timeframe_minutes,
                show_only_profitable, min_profit, min_profit
            ))
            
            row = cursor.fetchone()
            
            if not row:
                conn.close()
                return None
            
            # Get trades for this backtest
            trades_query = """
                SELECT 
                    entry_date, exit_date, symbol, option_type, strike_price, expiry,
                    entry_price, exit_price, profit_loss, exit_reason, holding_period,
                    is_generated, data_source, trade_details
                FROM my_schema.options_backtest_trades
                WHERE backtest_result_id = %s
                ORDER BY entry_date, exit_date
            """
            
            cursor.execute(trades_query, (row[0],))
            trade_rows = cursor.fetchall()
            conn.close()
            
            # Parse trades
            trades = []
            for trade_row in trade_rows:
                trade = json.loads(trade_row[13]) if trade_row[13] else {}
                # Ensure all fields are present
                if not trade:
                    trade = {
                        'entry_date': trade_row[0].strftime('%Y-%m-%d') if trade_row[0] else None,
                        'exit_date': trade_row[1].strftime('%Y-%m-%d') if trade_row[1] else None,
                        'symbol': trade_row[2],
                        'option_type': trade_row[3],
                        'strike_price': float(trade_row[4]) if trade_row[4] else 0.0,
                        'expiry': trade_row[5].strftime('%Y-%m-%d') if trade_row[5] else None,
                        'entry_price': float(trade_row[6]) if trade_row[6] else 0.0,
                        'exit_price': float(trade_row[7]) if trade_row[7] else 0.0,
                        'profit_loss': float(trade_row[8]) if trade_row[8] else 0.0,
                        'exit_reason': trade_row[9],
                        'holding_period': trade_row[10],
                        'is_generated': trade_row[11],
                        'data_source': trade_row[12]
                    }
                trades.append(trade)
            
            # Parse metrics and summary
            metrics = json.loads(row[26]) if row[26] else {}
            summary = json.loads(row[27]) if row[27] else {}
            data_quality = json.loads(row[25]) if row[25] else {}
            
            # Build results dictionary
            results = {
                'success': True,
                'backtest_id': row[0],
                'saved': True,
                'from_cache': True,
                'trades': trades,
                'metrics': metrics,
                'summary': summary,
                'data_quality': data_quality,
                'created_at': row[28].isoformat() if row[28] else None
            }
            
            return results
            
        except Exception as e:
            logging.warning(f"Error retrieving existing backtest: {e}")
            try:
                if conn:
                    conn.close()
            except:
                pass
            return None
    
    def save_backtest_results(self,
                             backtest_name: Optional[str] = None,
                             results: Dict = None) -> Optional[int]:
        """
        Save back-testing results to database
        
        Args:
            backtest_name: Optional name for the back-test
            results: Results dictionary from run_backtest()
            
        Returns:
            Back-test result ID if successful, None otherwise
        """
        if not results or not results.get('success'):
            return None
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            metrics = results.get('metrics', {})
            summary = results.get('summary', {})
            trades = results.get('trades', [])
            
            # Sanitize data for JSON serialization
            metrics = self._sanitize_for_json(metrics)
            summary = self._sanitize_for_json(summary)
            trades = self._sanitize_for_json(trades)
            
            # Insert main back-test result
            insert_result_query = """
                INSERT INTO my_schema.options_backtest_results (
                    backtest_name, start_date, end_date, strategy_type, timeframe_minutes,
                    show_only_profitable, min_profit, total_trades, win_count, loss_count,
                    win_rate, total_profit_loss, avg_profit_loss, gross_profit, gross_loss,
                    avg_win, avg_loss, max_profit, max_loss, profit_factor, sharpe_ratio,
                    max_drawdown, avg_holding_period, confidence_score, data_quality, metrics, summary
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                ) RETURNING id
            """
            
            from datetime import datetime
            start_date = datetime.strptime(summary.get('start_date', ''), '%Y-%m-%d').date()
            end_date = datetime.strptime(summary.get('end_date', ''), '%Y-%m-%d').date()
            
            import json
            import math
            
            # Ensure all float values are valid (not NaN or Infinity)
            def safe_float(val, default=0.0):
                if val is None:
                    return default if default is not None else None
                if isinstance(val, float):
                    if math.isnan(val) or math.isinf(val):
                        return default if default is not None else None
                    return val
                try:
                    fval = float(val)
                    if math.isnan(fval) or math.isinf(fval):
                        return default if default is not None else None
                    return fval
                except (ValueError, TypeError):
                    return default if default is not None else None
            
            cursor.execute(insert_result_query, (
                backtest_name or f"Back-test {start_date} to {end_date}",
                start_date,
                end_date,
                summary.get('strategy_type'),
                None,  # timeframe_minutes - can be added if needed
                summary.get('filters_applied', {}).get('show_only_profitable', False),
                summary.get('filters_applied', {}).get('min_profit'),
                metrics.get('total_trades', 0),
                metrics.get('win_count', 0),
                metrics.get('loss_count', 0),
                safe_float(metrics.get('win_rate', 0.0)),
                safe_float(metrics.get('total_profit_loss', 0.0)),
                safe_float(metrics.get('avg_profit_loss', 0.0)),
                safe_float(metrics.get('gross_profit', 0.0)),
                safe_float(metrics.get('gross_loss', 0.0)),
                safe_float(metrics.get('avg_win', 0.0)),
                safe_float(metrics.get('avg_loss', 0.0)),
                safe_float(metrics.get('max_profit', 0.0)),
                safe_float(metrics.get('max_loss', 0.0)),
                safe_float(metrics.get('profit_factor'), None) if metrics.get('profit_factor') is not None else None,
                safe_float(metrics.get('sharpe_ratio', 0.0)),
                safe_float(metrics.get('max_drawdown', 0.0)),
                safe_float(metrics.get('avg_holding_period', 0.0)),
                safe_float(metrics.get('confidence_score', 0.0)),
                json.dumps(metrics.get('data_quality', {})),
                json.dumps(metrics),
                json.dumps(summary)
            ))
            
            backtest_result_id = cursor.fetchone()[0]
            
            # Insert individual trades
            if trades:
                insert_trade_query = """
                    INSERT INTO my_schema.options_backtest_trades (
                        backtest_result_id, entry_date, exit_date, symbol, option_type,
                        strike_price, expiry, entry_price, exit_price, profit_loss,
                        exit_reason, holding_period, is_generated, data_source, trade_details
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """
                
                trade_data = []
                for trade in trades:
                    entry_date = datetime.strptime(trade.get('entry_date', ''), '%Y-%m-%d').date()
                    exit_date = datetime.strptime(trade.get('exit_date', ''), '%Y-%m-%d').date()
                    expiry = None
                    if trade.get('expiry'):
                        if isinstance(trade['expiry'], str):
                            expiry = datetime.strptime(trade['expiry'], '%Y-%m-%d').date()
                        else:
                            expiry = trade['expiry']
                    
                    # Sanitize trade data for JSON
                    sanitized_trade = self._sanitize_for_json(trade)
                    
                    trade_data.append((
                        backtest_result_id,
                        entry_date,
                        exit_date,
                        trade.get('symbol'),
                        trade.get('option_type'),
                        safe_float(trade.get('strike_price'), 0.0),
                        expiry,
                        safe_float(trade.get('entry_price'), 0.0),
                        safe_float(trade.get('exit_price'), 0.0),
                        safe_float(trade.get('profit_loss'), 0.0),
                        trade.get('exit_reason'),
                        trade.get('holding_period'),
                        trade.get('is_generated', False),
                        trade.get('data_source', 'unknown'),
                        json.dumps(sanitized_trade)
                    ))
                
                from psycopg2.extras import execute_batch
                execute_batch(cursor, insert_trade_query, trade_data)
            
            conn.commit()
            conn.close()
            
            logging.info(f"Saved back-test results with ID: {backtest_result_id}")
            return backtest_result_id
            
        except Exception as e:
            logging.error(f"Error saving back-test results: {e}")
            import traceback
            logging.error(traceback.format_exc())
            try:
                if conn:
                    conn.rollback()
                    conn.close()
            except:
                pass
            return None

