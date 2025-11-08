"""
Multi-Leg Options Strategies
Generates multi-leg options strategies and their pay-off graphs
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import base64
from typing import Dict, List, Optional, Tuple
from datetime import date, datetime, timedelta
try:
    from scipy.stats import norm
except ImportError:
    # Fallback to manual normal CDF if scipy is not available
    import math
    def norm_cdf(x):
        """Manual normal CDF approximation"""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    norm = type('norm', (), {'cdf': norm_cdf})()
from options.OptionsGreeksCalculator import OptionsGreeksCalculator


class OptionsStrategies:
    """
    Generate multi-leg options strategies and their pay-off graphs with Greeks
    """
    
    def __init__(self, risk_free_rate: float = 0.065):
        """
        Initialize Options Strategies Generator
        
        Args:
            risk_free_rate: Risk-free interest rate (default: 6.5% for India)
        """
        self.greeks_calculator = OptionsGreeksCalculator(risk_free_rate=risk_free_rate)
        self.risk_free_rate = risk_free_rate
    
    def generate_vertical_spread(self,
                                 current_price: float,
                                 lower_strike: float,
                                 upper_strike: float,
                                 lower_premium: float,
                                 upper_premium: float,
                                 option_type: str = 'CE',
                                 is_bullish: bool = True) -> Dict:
        """
        Generate Vertical Spread strategy
        
        Args:
            current_price: Current underlying price
            lower_strike: Lower strike price
            upper_strike: Upper strike price
            lower_premium: Premium for lower strike option
            upper_premium: Premium for upper strike option
            option_type: 'CE' or 'PE'
            is_bullish: True for bull spread, False for bear spread
            
        Returns:
            Dictionary with strategy details and pay-off graph
        """
        # Calculate net premium
        if is_bullish:
            # Bull spread: Buy lower strike, sell upper strike
            net_premium = lower_premium - upper_premium
        else:
            # Bear spread: Sell lower strike, buy upper strike
            net_premium = upper_premium - lower_premium
        
        # Calculate pay-off at different underlying prices
        price_range = np.linspace(
            min(lower_strike, upper_strike) * 0.8,
            max(lower_strike, upper_strike) * 1.2,
            200
        )
        
        payoffs = []
        for price in price_range:
            if option_type == 'CE':
                if is_bullish:
                    # Bull call spread
                    long_payoff = max(0, price - lower_strike) - lower_premium
                    short_payoff = upper_premium - max(0, price - upper_strike)
                else:
                    # Bear call spread
                    long_payoff = upper_premium - max(0, price - upper_strike)
                    short_payoff = max(0, price - lower_strike) - lower_premium
            else:
                if is_bullish:
                    # Bull put spread
                    long_payoff = max(0, lower_strike - price) - lower_premium
                    short_payoff = upper_premium - max(0, upper_strike - price)
                else:
                    # Bear put spread
                    long_payoff = upper_premium - max(0, upper_strike - price)
                    short_payoff = max(0, lower_strike - price) - lower_premium
            
            total_payoff = long_payoff + short_payoff
            payoffs.append(total_payoff)
        
        # Generate pay-off graph
        graph_image = self._plot_payoff_graph(
            price_range, payoffs, current_price,
            f"{'Bull' if is_bullish else 'Bear'} {option_type} Vertical Spread",
            lower_strike, upper_strike
        )
        
        # Calculate max profit and loss
        max_profit = max(payoffs)
        max_loss = min(payoffs)
        
        # Calculate breakeven
        if option_type == 'CE':
            if is_bullish:
                breakeven = lower_strike + net_premium
            else:
                breakeven = upper_strike - net_premium
        else:
            if is_bullish:
                breakeven = lower_strike - net_premium
            else:
                breakeven = upper_strike + net_premium
        
        return {
            'strategy_type': f"{'Bull' if is_bullish else 'Bear'} {option_type} Vertical Spread",
            'lower_strike': lower_strike,
            'upper_strike': upper_strike,
            'lower_premium': lower_premium,
            'upper_premium': upper_premium,
            'net_premium': net_premium,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'breakeven': breakeven,
            'current_price': current_price,
            'payoff_graph': graph_image,
            'description': f"{'Bull' if is_bullish else 'Bear'} {option_type} Vertical Spread: Buy {lower_strike} {option_type} @ {lower_premium:.2f}, Sell {upper_strike} {option_type} @ {upper_premium:.2f}"
        }
    
    def generate_protective_put(self,
                                current_price: float,
                                strike: float,
                                premium: float,
                                stock_price: float = None) -> Dict:
        """
        Generate Protective Put strategy (Long Stock + Long Put)
        
        Args:
            current_price: Current underlying price
            strike: Put strike price
            premium: Put premium
            stock_price: Stock purchase price (defaults to current_price)
            
        Returns:
            Dictionary with strategy details and pay-off graph
        """
        if stock_price is None:
            stock_price = current_price
        
        # Calculate pay-off at different underlying prices
        price_range = np.linspace(strike * 0.7, strike * 1.3, 200)
        
        payoffs = []
        for price in price_range:
            # Stock payoff: price - purchase_price
            stock_payoff = price - stock_price
            # Put payoff: max(0, strike - price) - premium
            put_payoff = max(0, strike - price) - premium
            total_payoff = stock_payoff + put_payoff
            payoffs.append(total_payoff)
        
        # Generate pay-off graph
        graph_image = self._plot_payoff_graph(
            price_range, payoffs, current_price,
            "Protective Put (Long Stock + Long Put)",
            strike, None
        )
        
        # Calculate max profit and loss
        max_profit = max(payoffs)
        max_loss = min(payoffs)
        breakeven = stock_price + premium
        
        return {
            'strategy_type': 'Protective Put',
            'strike': strike,
            'premium': premium,
            'stock_price': stock_price,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'breakeven': breakeven,
            'current_price': current_price,
            'payoff_graph': graph_image,
            'description': f"Protective Put: Long Stock @ {stock_price:.2f} + Long Put {strike} @ {premium:.2f}"
        }
    
    def generate_covered_call(self,
                              current_price: float,
                              strike: float,
                              premium: float,
                              stock_price: float = None) -> Dict:
        """
        Generate Covered Call strategy (Long Stock + Short Call)
        
        Args:
            current_price: Current underlying price
            strike: Call strike price
            premium: Call premium
            stock_price: Stock purchase price (defaults to current_price)
            
        Returns:
            Dictionary with strategy details and pay-off graph
        """
        if stock_price is None:
            stock_price = current_price
        
        # Calculate pay-off at different underlying prices
        price_range = np.linspace(strike * 0.7, strike * 1.3, 200)
        
        payoffs = []
        for price in price_range:
            # Stock payoff: price - purchase_price
            stock_payoff = price - stock_price
            # Short call payoff: premium - max(0, price - strike)
            call_payoff = premium - max(0, price - strike)
            total_payoff = stock_payoff + call_payoff
            payoffs.append(total_payoff)
        
        # Generate pay-off graph
        graph_image = self._plot_payoff_graph(
            price_range, payoffs, current_price,
            "Covered Call (Long Stock + Short Call)",
            strike, None
        )
        
        # Calculate max profit and loss
        max_profit = max(payoffs)
        max_loss = min(payoffs)
        breakeven = stock_price - premium
        
        return {
            'strategy_type': 'Covered Call',
            'strike': strike,
            'premium': premium,
            'stock_price': stock_price,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'breakeven': breakeven,
            'current_price': current_price,
            'payoff_graph': graph_image,
            'description': f"Covered Call: Long Stock @ {stock_price:.2f} + Short Call {strike} @ {premium:.2f}"
        }
    
    def generate_protective_collar(self,
                                    current_price: float,
                                    put_strike: float,
                                    call_strike: float,
                                    put_premium: float,
                                    call_premium: float,
                                    stock_price: float = None) -> Dict:
        """
        Generate Protective Collar strategy (Long Stock + Long Put + Short Call)
        
        Args:
            current_price: Current underlying price
            put_strike: Put strike price
            call_strike: Call strike price
            put_premium: Put premium
            call_premium: Call premium
            stock_price: Stock purchase price (defaults to current_price)
            
        Returns:
            Dictionary with strategy details and pay-off graph
        """
        if stock_price is None:
            stock_price = current_price
        
        # Calculate pay-off at different underlying prices
        price_range = np.linspace(
            min(put_strike, call_strike) * 0.8,
            max(put_strike, call_strike) * 1.2,
            200
        )
        
        payoffs = []
        for price in price_range:
            # Stock payoff: price - purchase_price
            stock_payoff = price - stock_price
            # Put payoff: max(0, put_strike - price) - put_premium
            put_payoff = max(0, put_strike - price) - put_premium
            # Short call payoff: call_premium - max(0, price - call_strike)
            call_payoff = call_premium - max(0, price - call_strike)
            total_payoff = stock_payoff + put_payoff + call_payoff
            payoffs.append(total_payoff)
        
        # Generate pay-off graph
        graph_image = self._plot_payoff_graph(
            price_range, payoffs, current_price,
            "Protective Collar (Long Stock + Long Put + Short Call)",
            put_strike, call_strike
        )
        
        # Calculate max profit and loss
        max_profit = max(payoffs)
        max_loss = min(payoffs)
        net_premium = call_premium - put_premium
        breakeven = stock_price - net_premium
        
        return {
            'strategy_type': 'Protective Collar',
            'put_strike': put_strike,
            'call_strike': call_strike,
            'put_premium': put_premium,
            'call_premium': call_premium,
            'net_premium': net_premium,
            'stock_price': stock_price,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'breakeven': breakeven,
            'current_price': current_price,
            'payoff_graph': graph_image,
            'description': f"Protective Collar: Long Stock @ {stock_price:.2f} + Long Put {put_strike} @ {put_premium:.2f} + Short Call {call_strike} @ {call_premium:.2f}"
        }
    
    def generate_iron_condor(self,
                            current_price: float,
                            put_lower_strike: float,
                            put_upper_strike: float,
                            call_lower_strike: float,
                            call_upper_strike: float,
                            put_lower_premium: float,
                            put_upper_premium: float,
                            call_lower_premium: float,
                            call_upper_premium: float) -> Dict:
        """
        Generate Iron Condor strategy
        
        Args:
            current_price: Current underlying price
            put_lower_strike: Lower put strike (long)
            put_upper_strike: Upper put strike (short)
            call_lower_strike: Lower call strike (short)
            call_upper_strike: Upper call strike (long)
            put_lower_premium: Lower put premium
            put_upper_premium: Upper put premium
            call_lower_premium: Lower call premium
            call_upper_premium: Upper call premium
            
        Returns:
            Dictionary with strategy details and pay-off graph
        """
        # Calculate net premium received
        net_premium = (put_upper_premium + call_lower_premium) - (put_lower_premium + call_upper_premium)
        
        # Calculate pay-off at different underlying prices
        price_range = np.linspace(
            put_lower_strike * 0.8,
            call_upper_strike * 1.2,
            200
        )
        
        payoffs = []
        for price in price_range:
            # Long put payoff: max(0, put_lower_strike - price) - put_lower_premium
            long_put_payoff = max(0, put_lower_strike - price) - put_lower_premium
            # Short put payoff: put_upper_premium - max(0, put_upper_strike - price)
            short_put_payoff = put_upper_premium - max(0, put_upper_strike - price)
            # Short call payoff: call_lower_premium - max(0, price - call_lower_strike)
            short_call_payoff = call_lower_premium - max(0, price - call_lower_strike)
            # Long call payoff: max(0, price - call_upper_strike) - call_upper_premium
            long_call_payoff = max(0, price - call_upper_strike) - call_upper_premium
            
            total_payoff = long_put_payoff + short_put_payoff + short_call_payoff + long_call_payoff
            payoffs.append(total_payoff)
        
        # Generate pay-off graph
        graph_image = self._plot_payoff_graph(
            price_range, payoffs, current_price,
            "Iron Condor",
            put_lower_strike, call_upper_strike,
            [put_upper_strike, call_lower_strike]
        )
        
        # Calculate max profit and loss
        max_profit = max(payoffs)
        max_loss = min(payoffs)
        
        # Breakeven points
        breakeven_lower = put_upper_strike - net_premium
        breakeven_upper = call_lower_strike + net_premium
        
        return {
            'strategy_type': 'Iron Condor',
            'put_lower_strike': put_lower_strike,
            'put_upper_strike': put_upper_strike,
            'call_lower_strike': call_lower_strike,
            'call_upper_strike': call_upper_strike,
            'put_lower_premium': put_lower_premium,
            'put_upper_premium': put_upper_premium,
            'call_lower_premium': call_lower_premium,
            'call_upper_premium': call_upper_premium,
            'net_premium': net_premium,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'breakeven_lower': breakeven_lower,
            'breakeven_upper': breakeven_upper,
            'current_price': current_price,
            'payoff_graph': graph_image,
            'description': f"Iron Condor: Long Put {put_lower_strike} @ {put_lower_premium:.2f}, Short Put {put_upper_strike} @ {put_upper_premium:.2f}, Short Call {call_lower_strike} @ {call_lower_premium:.2f}, Long Call {call_upper_strike} @ {call_upper_premium:.2f}"
        }
    
    def generate_straddle(self,
                          current_price: float,
                          strike: float,
                          call_premium: float,
                          put_premium: float,
                          is_long: bool = True) -> Dict:
        """
        Generate Straddle strategy
        
        Args:
            current_price: Current underlying price
            strike: Strike price (same for both call and put)
            call_premium: Call premium
            put_premium: Put premium
            is_long: True for long straddle, False for short straddle
            
        Returns:
            Dictionary with strategy details and pay-off graph
        """
        total_premium = call_premium + put_premium
        
        # Calculate pay-off at different underlying prices
        price_range = np.linspace(strike * 0.7, strike * 1.3, 200)
        
        payoffs = []
        for price in price_range:
            call_payoff = max(0, price - strike) - call_premium
            put_payoff = max(0, strike - price) - put_premium
            
            if is_long:
                total_payoff = call_payoff + put_payoff
            else:
                # Short straddle: reverse the payoffs
                total_payoff = -(call_payoff + put_payoff)
            
            payoffs.append(total_payoff)
        
        # Generate pay-off graph
        graph_image = self._plot_payoff_graph(
            price_range, payoffs, current_price,
            f"{'Long' if is_long else 'Short'} Straddle",
            strike, None
        )
        
        # Calculate max profit and loss
        max_profit = max(payoffs)
        max_loss = min(payoffs)
        
        if is_long:
            breakeven_lower = strike - total_premium
            breakeven_upper = strike + total_premium
        else:
            breakeven_lower = strike - total_premium
            breakeven_upper = strike + total_premium
        
        return {
            'strategy_type': f"{'Long' if is_long else 'Short'} Straddle",
            'strike': strike,
            'call_premium': call_premium,
            'put_premium': put_premium,
            'total_premium': total_premium,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'breakeven_lower': breakeven_lower,
            'breakeven_upper': breakeven_upper,
            'current_price': current_price,
            'payoff_graph': graph_image,
            'description': f"{'Long' if is_long else 'Short'} Straddle: {'Buy' if is_long else 'Sell'} Call {strike} @ {call_premium:.2f} + {'Buy' if is_long else 'Sell'} Put {strike} @ {put_premium:.2f}"
        }
    
    def _plot_payoff_graph(self,
                          price_range: np.ndarray,
                          payoffs: List[float],
                          current_price: float,
                          title: str,
                          strike1: float,
                          strike2: Optional[float] = None,
                          additional_strikes: Optional[List[float]] = None) -> str:
        """
        Plot pay-off graph with profitable regions in green and loss-making regions in red
        
        Args:
            price_range: Array of underlying prices
            payoffs: Array of pay-offs corresponding to prices
            current_price: Current underlying price
            title: Graph title
            strike1: First strike price
            strike2: Second strike price (optional)
            additional_strikes: Additional strike prices (optional)
            
        Returns:
            Base64 encoded image string
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Convert payoffs to numpy array
        payoffs_array = np.array(payoffs)
        
        # Plot the pay-off curve
        ax.plot(price_range, payoffs_array, 'k-', linewidth=2.5, label='Pay-off')
        
        # Fill profitable regions (above zero) in green
        ax.fill_between(price_range, payoffs_array, 0, 
                       where=(payoffs_array >= 0), 
                       color='green', alpha=0.3, label='Profit')
        
        # Fill loss-making regions (below zero) in red
        ax.fill_between(price_range, payoffs_array, 0, 
                       where=(payoffs_array < 0), 
                       color='red', alpha=0.3, label='Loss')
        
        # Add horizontal line at zero (breakeven)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        # Add vertical line at current price
        ax.axvline(x=current_price, color='blue', linestyle='--', linewidth=2, 
                  alpha=0.7, label=f'Current Price: {current_price:.2f}')
        
        # Add vertical lines at strike prices
        if strike1:
            ax.axvline(x=strike1, color='red', linestyle='--', linewidth=1.5, 
                      alpha=0.6, label=f'Strike: {strike1:.2f}')
        
        if strike2:
            ax.axvline(x=strike2, color='red', linestyle='--', linewidth=1.5, 
                      alpha=0.6, label=f'Strike: {strike2:.2f}')
        
        if additional_strikes:
            for strike in additional_strikes:
                ax.axvline(x=strike, color='orange', linestyle='--', linewidth=1, 
                          alpha=0.5, label=f'Strike: {strike:.2f}')
        
        # Set labels and title
        ax.set_xlabel('Underlying Price', fontsize=12, fontweight='bold')
        ax.set_ylabel('Profit/Loss', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Format y-axis to show values in thousands
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.1f}k' if abs(x) >= 1000 else f'{x:.0f}'))
        
        # Add legend
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        
        # Set background color
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buffer.seek(0)
        plt.close()
        
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{image_base64}"
    
    def generate_payoff_chart_with_greeks(self,
                                         strategy_legs: List[Dict],
                                         current_price: float,
                                         expiry_date: date,
                                         implied_volatility: float = 0.20,
                                         lot_size: int = 50,
                                         current_date: Optional[date] = None) -> Dict:
        """
        Generate enhanced payoff chart with Greeks for a multi-leg options strategy
        
        Args:
            strategy_legs: List of strategy legs, each with:
                - 'action': 'BUY' or 'SELL'
                - 'quantity': Number of lots
                - 'option_type': 'CE' or 'PE'
                - 'strike': Strike price
                - 'premium': Premium paid/received
                - 'expiry': Expiry date
            current_price: Current underlying price
            expiry_date: Expiry date for the strategy
            implied_volatility: Implied volatility (default: 20%)
            lot_size: Lot size (default: 50 for Nifty)
            current_date: Current date (defaults to today)
            
        Returns:
            Dictionary with payoff chart, Greeks, and strategy metrics
        """
        if current_date is None:
            current_date = date.today()
        
        # Calculate time to expiry
        days_to_expiry = (expiry_date - current_date).days
        time_to_expiry_years = max(days_to_expiry / 365.0, 0.0001)
        
        # Calculate price range for chart
        strikes = [leg['strike'] for leg in strategy_legs]
        min_strike = min(strikes) if strikes else current_price
        max_strike = max(strikes) if strikes else current_price
        price_range = np.linspace(min_strike * 0.7, max_strike * 1.3, 300)
        
        # Calculate payoff at expiry and current payoff
        payoff_at_expiry = []
        current_payoff = []
        total_greeks = {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }
        total_premium = 0.0
        net_credit = 0.0
        
        for leg in strategy_legs:
            action = leg.get('action', 'BUY')
            quantity = leg.get('quantity', 1)
            option_type = leg.get('option_type', 'CE')
            strike = leg.get('strike', 0)
            premium = leg.get('premium', 0)
            multiplier = 1 if action == 'BUY' else -1
            leg_multiplier = multiplier * quantity * lot_size
            
            # Calculate premium paid/received
            premium_total = premium * quantity * lot_size
            if action == 'BUY':
                total_premium += premium_total
                net_credit -= premium_total
            else:
                net_credit += premium_total
            
            # Calculate Greeks for this leg
            try:
                greeks = self.greeks_calculator.calculate_greeks(
                    spot_price=current_price,
                    strike_price=strike,
                    time_to_expiry=time_to_expiry_years,
                    implied_volatility=implied_volatility,
                    option_type=option_type,
                    option_price=premium
                )
                
                # Aggregate Greeks (weighted by quantity and lot size)
                total_greeks['delta'] += greeks['delta'] * leg_multiplier
                total_greeks['gamma'] += greeks['gamma'] * leg_multiplier
                total_greeks['theta'] += greeks['theta'] * leg_multiplier
                total_greeks['vega'] += greeks['vega'] * leg_multiplier
                total_greeks['rho'] += greeks['rho'] * leg_multiplier
            except Exception as e:
                logging.warning(f"Error calculating Greeks for leg {leg}: {e}")
        
        # Calculate payoffs for each price in range
        for price in price_range:
            expiry_payoff = 0.0
            current_payoff_val = 0.0
            
            for leg in strategy_legs:
                action = leg.get('action', 'BUY')
                quantity = leg.get('quantity', 1)
                option_type = leg.get('option_type', 'CE')
                strike = leg.get('strike', 0)
                premium = leg.get('premium', 0)
                multiplier = 1 if action == 'BUY' else -1
                leg_multiplier = multiplier * quantity * lot_size
                
                # Payoff at expiry
                if option_type == 'CE':
                    intrinsic = max(0, price - strike)
                else:  # PE
                    intrinsic = max(0, strike - price)
                
                expiry_payoff += (intrinsic - premium) * leg_multiplier
                
                # Current payoff (with time value)
                try:
                    # Calculate theoretical price at current price level
                    time_to_expiry_at_price = time_to_expiry_years
                    greeks_at_price = self.greeks_calculator.calculate_greeks(
                        spot_price=price,
                        strike_price=strike,
                        time_to_expiry=time_to_expiry_at_price,
                        implied_volatility=implied_volatility,
                        option_type=option_type
                    )
                    theoretical_price = greeks_at_price['theoretical_price']
                    current_payoff_val += (theoretical_price - premium) * leg_multiplier
                except Exception as e:
                    # Fallback to intrinsic value if Greeks calculation fails
                    current_payoff_val += (intrinsic - premium) * leg_multiplier
            
            payoff_at_expiry.append(expiry_payoff)
            current_payoff.append(current_payoff_val)
        
        # Calculate strategy metrics
        max_profit = max(payoff_at_expiry)
        max_loss = min(payoff_at_expiry)
        
        # Find breakeven points
        breakevens = []
        for i in range(len(price_range) - 1):
            if (payoff_at_expiry[i] <= 0 and payoff_at_expiry[i+1] > 0) or \
               (payoff_at_expiry[i] >= 0 and payoff_at_expiry[i+1] < 0):
                # Linear interpolation to find exact breakeven
                x1, y1 = price_range[i], payoff_at_expiry[i]
                x2, y2 = price_range[i+1], payoff_at_expiry[i+1]
                if y2 != y1:
                    breakeven = x1 - y1 * (x2 - x1) / (y2 - y1)
                    breakevens.append(breakeven)
        
        # Calculate probability of profit
        prob_of_profit = self._calculate_probability_of_profit(
            current_price, strikes, implied_volatility, time_to_expiry_years, payoff_at_expiry, price_range
        )
        
        # Calculate standard deviation bands
        std_dev = current_price * implied_volatility * np.sqrt(time_to_expiry_years)
        std_bands = {
            '-2σ': current_price - 2 * std_dev,
            '-1σ': current_price - std_dev,
            '+1σ': current_price + std_dev,
            '+2σ': current_price + 2 * std_dev
        }
        
        # Generate enhanced chart
        chart_image = self._plot_enhanced_payoff_graph(
            price_range=price_range,
            payoff_at_expiry=payoff_at_expiry,
            current_payoff=current_payoff,
            current_price=current_price,
            strikes=strikes,
            std_bands=std_bands,
            breakevens=breakevens,
            title="Options Strategy Payoff Chart"
        )
        
        return {
            'payoff_graph': chart_image,
            'greeks': {
                'positional_delta': round(total_greeks['delta'], 2),
                'gamma': round(total_greeks['gamma'], 4),
                'theta': round(total_greeks['theta'], 2),
                'vega': round(total_greeks['vega'], 2),
                'rho': round(total_greeks['rho'], 2)
            },
            'metrics': {
                'max_profit': round(max_profit, 2),
                'max_loss': round(max_loss, 2),
                'breakevens': [round(be, 2) for be in breakevens],
                'prob_of_profit': round(prob_of_profit, 2),
                'total_premium': round(total_premium, 2),
                'net_credit': round(net_credit, 2),
                'estimated_margin_premium': round(abs(net_credit) if net_credit < 0 else total_premium, 2)
            },
            'current_price': current_price,
            'expiry_date': expiry_date.strftime('%Y-%m-%d'),
            'days_to_expiry': days_to_expiry,
            'implied_volatility': implied_volatility * 100  # As percentage
        }
    
    def _calculate_probability_of_profit(self,
                                        current_price: float,
                                        strikes: List[float],
                                        iv: float,
                                        time_to_expiry: float,
                                        payoffs: List[float],
                                        price_range: np.ndarray) -> float:
        """
        Calculate probability of profit using normal distribution
        """
        try:
            # Find prices where payoff > 0
            profitable_prices = []
            for i, payoff in enumerate(payoffs):
                if payoff > 0:
                    profitable_prices.append(price_range[i])
            
            if not profitable_prices:
                return 0.0
            
            # Calculate mean and std dev for price distribution
            mean = current_price
            std_dev = current_price * iv * np.sqrt(time_to_expiry)
            
            if std_dev == 0:
                return 0.0
            
            # Calculate probability using normal distribution
            # For each profitable price range, calculate probability
            prob = 0.0
            for i in range(len(price_range) - 1):
                if payoffs[i] > 0:
                    # Probability that price falls in this range
                    prob_lower = norm.cdf((price_range[i] - mean) / std_dev)
                    prob_upper = norm.cdf((price_range[i+1] - mean) / std_dev)
                    prob += prob_upper - prob_lower
            
            return prob * 100  # As percentage
        except Exception as e:
            logging.warning(f"Error calculating probability of profit: {e}")
            return 0.0
    
    def calculate_historical_payoff(self,
                                   strategy_legs: List[Dict],
                                   entry_date: date,
                                   exit_date: date,
                                   entry_underlying_price: float,
                                   exit_underlying_price: float,
                                   implied_volatility: float = 0.20,
                                   lot_size: int = 50) -> Dict:
        """
        Calculate payoff for a historical trade
        
        Args:
            strategy_legs: List of strategy legs
            entry_date: Entry date
            exit_date: Exit date
            entry_underlying_price: Underlying price at entry
            exit_underlying_price: Underlying price at exit
            implied_volatility: Implied volatility (default: 20%)
            lot_size: Lot size (default: 50)
            
        Returns:
            Dictionary with historical payoff details
        """
        try:
            # Calculate time to expiry at entry and exit
            expiry_dates = [leg.get('expiry') for leg in strategy_legs if leg.get('expiry')]
            if not expiry_dates:
                return {'error': 'No expiry dates in strategy legs'}
            
            expiry_date = expiry_dates[0]  # Use first expiry
            
            entry_time_to_expiry = self.greeks_calculator.calculate_time_to_expiry(
                expiry_date, entry_date
            )
            exit_time_to_expiry = self.greeks_calculator.calculate_time_to_expiry(
                expiry_date, exit_date
            )
            
            # Calculate entry and exit payoffs
            entry_payoff = 0.0
            exit_payoff = 0.0
            total_premium_paid = 0.0
            
            for leg in strategy_legs:
                action = leg.get('action', 'BUY')
                quantity = leg.get('quantity', 1)
                option_type = leg.get('option_type', 'CE')
                strike = leg.get('strike', 0)
                premium = leg.get('premium', 0)
                multiplier = 1 if action == 'BUY' else -1
                leg_multiplier = multiplier * quantity * lot_size
                
                # Entry premium
                premium_total = premium * quantity * lot_size
                if action == 'BUY':
                    total_premium_paid += premium_total
                    entry_payoff -= premium_total
                else:
                    entry_payoff += premium_total
                
                # Calculate entry option price
                entry_greeks = self.greeks_calculator.calculate_greeks(
                    spot_price=entry_underlying_price,
                    strike_price=strike,
                    time_to_expiry=entry_time_to_expiry,
                    implied_volatility=implied_volatility,
                    option_type=option_type,
                    option_price=premium
                )
                entry_option_price = entry_greeks['theoretical_price']
                
                # Calculate exit option price
                if exit_time_to_expiry <= 0:
                    # At or past expiry, use intrinsic value
                    if option_type == 'CE':
                        exit_option_price = max(0, exit_underlying_price - strike)
                    else:  # PE
                        exit_option_price = max(0, strike - exit_underlying_price)
                else:
                    exit_greeks = self.greeks_calculator.calculate_greeks(
                        spot_price=exit_underlying_price,
                        strike_price=strike,
                        time_to_expiry=exit_time_to_expiry,
                        implied_volatility=implied_volatility,
                        option_type=option_type
                    )
                    exit_option_price = exit_greeks['theoretical_price']
                
                # Calculate payoff for this leg
                if action == 'BUY':
                    exit_payoff += (exit_option_price - premium) * leg_multiplier
                else:  # SELL
                    exit_payoff += (premium - exit_option_price) * leg_multiplier
            
            # Net profit/loss
            net_pnl = exit_payoff - entry_payoff
            
            return {
                'entry_date': entry_date.strftime('%Y-%m-%d'),
                'exit_date': exit_date.strftime('%Y-%m-%d'),
                'entry_underlying_price': entry_underlying_price,
                'exit_underlying_price': exit_underlying_price,
                'entry_payoff': round(entry_payoff, 2),
                'exit_payoff': round(exit_payoff, 2),
                'net_pnl': round(net_pnl, 2),
                'total_premium_paid': round(total_premium_paid, 2),
                'holding_period_days': (exit_date - entry_date).days,
                'return_pct': round((net_pnl / total_premium_paid * 100) if total_premium_paid > 0 else 0, 2)
            }
            
        except Exception as e:
            logging.error(f"Error calculating historical payoff: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return {'error': str(e)}
    
    def generate_historical_payoff_chart(self,
                                        strategy_legs: List[Dict],
                                        entry_date: date,
                                        exit_date: date,
                                        entry_underlying_price: float,
                                        exit_underlying_price: float,
                                        implied_volatility: float = 0.20,
                                        lot_size: int = 50) -> str:
        """
        Generate payoff chart for historical trade
        
        Args:
            strategy_legs: List of strategy legs
            entry_date: Entry date
            exit_date: Exit date
            entry_underlying_price: Underlying price at entry
            exit_underlying_price: Underlying price at exit
            implied_volatility: Implied volatility
            lot_size: Lot size
            
        Returns:
            Base64 encoded chart image
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            # Calculate payoff at different price points
            price_range = np.linspace(
                min(entry_underlying_price, exit_underlying_price) * 0.9,
                max(entry_underlying_price, exit_underlying_price) * 1.1,
                200
            )
            
            expiry_dates = [leg.get('expiry') for leg in strategy_legs if leg.get('expiry')]
            expiry_date = expiry_dates[0] if expiry_dates else exit_date
            
            entry_time_to_expiry = self.greeks_calculator.calculate_time_to_expiry(
                expiry_date, entry_date
            )
            
            payoffs = []
            for price in price_range:
                payoff = 0.0
                for leg in strategy_legs:
                    action = leg.get('action', 'BUY')
                    quantity = leg.get('quantity', 1)
                    option_type = leg.get('option_type', 'CE')
                    strike = leg.get('strike', 0)
                    premium = leg.get('premium', 0)
                    multiplier = 1 if action == 'BUY' else -1
                    leg_multiplier = multiplier * quantity * lot_size
                    
                    # Calculate option price at this price level
                    if entry_time_to_expiry <= 0:
                        if option_type == 'CE':
                            option_price = max(0, price - strike)
                        else:
                            option_price = max(0, strike - price)
                    else:
                        greeks = self.greeks_calculator.calculate_greeks(
                            spot_price=price,
                            strike_price=strike,
                            time_to_expiry=entry_time_to_expiry,
                            implied_volatility=implied_volatility,
                            option_type=option_type
                        )
                        option_price = greeks['theoretical_price']
                    
                    if action == 'BUY':
                        payoff += (option_price - premium) * leg_multiplier
                    else:
                        payoff += (premium - option_price) * leg_multiplier
                
                payoffs.append(payoff)
            
            # Create chart
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(price_range, payoffs, 'b-', linewidth=2, label='Payoff')
            ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
            ax.axvline(x=entry_underlying_price, color='g', linestyle='--', linewidth=1, label='Entry Price')
            ax.axvline(x=exit_underlying_price, color='r', linestyle='--', linewidth=1, label='Exit Price')
            
            ax.set_xlabel('Underlying Price')
            ax.set_ylabel('Payoff')
            ax.set_title(f'Historical Payoff Chart ({entry_date} to {exit_date})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Save to buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            plt.close()
            
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            logging.error(f"Error generating historical payoff chart: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return ""
    
    def _plot_enhanced_payoff_graph(self,
                                   price_range: np.ndarray,
                                   payoff_at_expiry: List[float],
                                   current_payoff: List[float],
                                   current_price: float,
                                   strikes: List[float],
                                   std_bands: Dict[str, float],
                                   breakevens: List[float],
                                   title: str) -> str:
        """
        Plot enhanced payoff chart with current payoff, expiry payoff, and standard deviation bands
        """
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Convert to numpy arrays
        payoff_expiry = np.array(payoff_at_expiry)
        payoff_current = np.array(current_payoff)
        
        # Plot standard deviation bands (light grey vertical bands)
        for band_name, band_price in std_bands.items():
            if band_price > price_range[0] and band_price < price_range[-1]:
                color = 'lightgrey' if '2' in band_name else 'lightgreen'
                alpha = 0.2 if '2' in band_name else 0.3
                ax.axvline(x=band_price, color=color, linestyle=':', linewidth=1, alpha=alpha)
                ax.text(band_price, ax.get_ylim()[1] * 0.95, band_name, 
                       ha='center', va='top', fontsize=9, color='gray', rotation=90)
        
        # Fill profit/loss regions for expiry payoff
        ax.fill_between(price_range, payoff_expiry, 0, 
                       where=(payoff_expiry >= 0), 
                       color='green', alpha=0.2, label='Profit at Expiry')
        ax.fill_between(price_range, payoff_expiry, 0, 
                       where=(payoff_expiry < 0), 
                       color='red', alpha=0.2, label='Loss at Expiry')
        
        # Plot payoff at expiry (solid orange line)
        ax.plot(price_range, payoff_expiry, 'orange', linewidth=2.5, 
               label='Payoff at Expiry', alpha=0.9)
        
        # Plot current payoff (dashed blue line)
        ax.plot(price_range, payoff_current, 'b--', linewidth=2, 
               label='Current Payoff', alpha=0.8)
        
        # Add horizontal line at zero (breakeven)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        # Add vertical line at current price (green dashed)
        ax.axvline(x=current_price, color='green', linestyle='--', linewidth=2.5, 
                  alpha=0.8, label=f'Current Price: {current_price:.2f}')
        
        # Add vertical lines at strike prices
        for strike in strikes:
            ax.axvline(x=strike, color='red', linestyle='--', linewidth=1.5, 
                      alpha=0.6)
        
        # Add breakeven markers
        for be in breakevens:
            ax.axvline(x=be, color='orange', linestyle=':', linewidth=1, 
                      alpha=0.5)
            ax.plot(be, 0, 'o', color='orange', markersize=6, alpha=0.7)
        
        # Set labels and title
        ax.set_xlabel('Underlying Price', fontsize=13, fontweight='bold')
        ax.set_ylabel('Profit/Loss (₹)', fontsize=13, fontweight='bold')
        ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Format y-axis to show values in thousands
        ax.yaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, p: f'{x/1000:.1f}k' if abs(x) >= 1000 else f'{x:.0f}'
        ))
        
        # Add legend
        ax.legend(loc='best', fontsize=10, framealpha=0.9, ncol=2)
        
        # Set background color
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buffer.seek(0)
        plt.close()
        
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{image_base64}"

