"""
Options Greeks Calculator using Black-Scholes Model
Calculates Delta, Gamma, Theta, Vega, and Rho for options
"""

import math
from typing import Dict, Optional, Tuple
from datetime import datetime, date
import logging


class OptionsGreeksCalculator:
    """
    Calculate option Greeks using Black-Scholes model
    """
    
    def __init__(self, risk_free_rate: float = 0.065):
        """
        Initialize Greeks Calculator
        
        Args:
            risk_free_rate: Risk-free interest rate (default: 6.5% for India)
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_greeks(self,
                        spot_price: float,
                        strike_price: float,
                        time_to_expiry: float,
                        implied_volatility: float,
                        option_type: str,
                        option_price: Optional[float] = None) -> Dict:
        """
        Calculate all Greeks for an option using Black-Scholes
        
        Args:
            spot_price: Current spot price (S)
            strike_price: Strike price (K)
            time_to_expiry: Time to expiry in years (T)
            implied_volatility: Implied volatility as decimal (e.g., 0.20 for 20%)
            option_type: 'CE' for Call or 'PE' for Put
            option_price: Option premium (optional, calculated if not provided)
            
        Returns:
            Dictionary with Greeks: delta, gamma, theta, vega, rho, and theoretical_price
        """
        if time_to_expiry <= 0:
            time_to_expiry = 0.0001  # Minimum value to avoid division by zero
        
        # Calculate d1 and d2
        d1, d2 = self._calculate_d1_d2(spot_price, strike_price, time_to_expiry, 
                                      implied_volatility, self.risk_free_rate)
        
        # Calculate option price using Black-Scholes if not provided
        if option_price is None:
            theoretical_price = self._calculate_option_price(
                spot_price, strike_price, time_to_expiry, 
                implied_volatility, option_type, d1, d2
            )
        else:
            theoretical_price = option_price
        
        # Calculate Greeks
        delta = self._calculate_delta(spot_price, strike_price, time_to_expiry, 
                                     implied_volatility, option_type, d1)
        
        gamma = self._calculate_gamma(spot_price, time_to_expiry, implied_volatility, d1)
        
        theta = self._calculate_theta(spot_price, strike_price, time_to_expiry, 
                                     implied_volatility, option_type, d1, d2)
        
        vega = self._calculate_vega(spot_price, time_to_expiry, d1)
        
        rho = self._calculate_rho(spot_price, strike_price, time_to_expiry, 
                                  implied_volatility, option_type, d2)
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta / 365.0,  # Theta per day
            'vega': vega / 100.0,    # Vega per 1% change in IV
            'rho': rho / 100.0,      # Rho per 1% change in interest rate
            'theoretical_price': theoretical_price,
            'd1': d1,
            'd2': d2
        }
    
    def _calculate_d1_d2(self, S: float, K: float, T: float, sigma: float, r: float) -> Tuple[float, float]:
        """
        Calculate d1 and d2 for Black-Scholes
        
        d1 = (ln(S/K) + (r + σ²/2)*T) / (σ*√T)
        d2 = d1 - σ*√T
        """
        sqrt_T = math.sqrt(T)
        if sqrt_T == 0 or sigma == 0:
            return 0.0, 0.0
        
        d1 = (math.log(S / K) + (r + (sigma ** 2) / 2.0) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        
        return d1, d2
    
    def _calculate_option_price(self, S: float, K: float, T: float, sigma: float, 
                                option_type: str, d1: float, d2: float) -> float:
        """
        Calculate theoretical option price using Black-Scholes
        """
        r = self.risk_free_rate
        N_d1 = self._normal_cdf(d1)
        N_d2 = self._normal_cdf(d2)
        N_neg_d1 = self._normal_cdf(-d1)
        N_neg_d2 = self._normal_cdf(-d2)
        
        if option_type.upper() == 'CE':
            # Call option: S*N(d1) - K*e^(-r*T)*N(d2)
            price = S * N_d1 - K * math.exp(-r * T) * N_d2
        else:  # PE
            # Put option: K*e^(-r*T)*N(-d2) - S*N(-d1)
            price = K * math.exp(-r * T) * N_neg_d2 - S * N_neg_d1
        
        return max(0.0, price)  # Price cannot be negative
    
    def _calculate_delta(self, S: float, K: float, T: float, sigma: float, 
                        option_type: str, d1: float) -> float:
        """
        Calculate Delta (rate of change of option price with respect to spot)
        
        Call Delta = N(d1)
        Put Delta = N(d1) - 1
        """
        N_d1 = self._normal_cdf(d1)
        
        if option_type.upper() == 'CE':
            return N_d1
        else:  # PE
            return N_d1 - 1.0
    
    def _calculate_gamma(self, S: float, T: float, sigma: float, d1: float) -> float:
        """
        Calculate Gamma (rate of change of delta with respect to spot)
        
        Gamma = N'(d1) / (S * σ * √T)
        where N'(d1) is the PDF of standard normal distribution
        """
        sqrt_T = math.sqrt(T)
        if S == 0 or sigma == 0 or sqrt_T == 0:
            return 0.0
        
        # Standard normal PDF: (1/√(2π)) * e^(-d1²/2)
        pdf_d1 = (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * d1 * d1)
        
        gamma = pdf_d1 / (S * sigma * sqrt_T)
        return gamma
    
    def _calculate_theta(self, S: float, K: float, T: float, sigma: float, 
                        option_type: str, d1: float, d2: float) -> float:
        """
        Calculate Theta (rate of change of option price with respect to time)
        
        Returns theta per year (will be converted to per day in main function)
        """
        r = self.risk_free_rate
        sqrt_T = math.sqrt(T)
        if sqrt_T == 0:
            return 0.0
        
        # Standard normal PDF
        pdf_d1 = (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * d1 * d1)
        N_d2 = self._normal_cdf(d2)
        N_neg_d2 = self._normal_cdf(-d2)
        
        if option_type.upper() == 'CE':
            # Call Theta
            theta = (-(S * pdf_d1 * sigma) / (2.0 * sqrt_T) 
                    - r * K * math.exp(-r * T) * N_d2)
        else:  # PE
            # Put Theta
            theta = (-(S * pdf_d1 * sigma) / (2.0 * sqrt_T) 
                    + r * K * math.exp(-r * T) * N_neg_d2)
        
        return theta
    
    def _calculate_vega(self, S: float, T: float, d1: float) -> float:
        """
        Calculate Vega (rate of change of option price with respect to volatility)
        
        Returns vega per unit change (will be converted to per 1% in main function)
        """
        sqrt_T = math.sqrt(T)
        if sqrt_T == 0:
            return 0.0
        
        # Standard normal PDF
        pdf_d1 = (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * d1 * d1)
        
        vega = S * pdf_d1 * sqrt_T
        return vega
    
    def _calculate_rho(self, S: float, K: float, T: float, sigma: float, 
                      option_type: str, d2: float) -> float:
        """
        Calculate Rho (rate of change of option price with respect to interest rate)
        
        Returns rho per unit change (will be converted to per 1% in main function)
        """
        r = self.risk_free_rate
        N_d2 = self._normal_cdf(d2)
        N_neg_d2 = self._normal_cdf(-d2)
        
        if option_type.upper() == 'CE':
            rho = K * T * math.exp(-r * T) * N_d2
        else:  # PE
            rho = -K * T * math.exp(-r * T) * N_neg_d2
        
        return rho
    
    def _normal_cdf(self, x: float) -> float:
        """
        Cumulative Distribution Function for standard normal distribution
        Using approximation formula (Abramowitz and Stegun approximation)
        """
        # Approximation: erf(x/sqrt(2)) / 2 + 0.5
        return 0.5 * (1.0 + self._erf(x / math.sqrt(2.0)))
    
    def _erf(self, x: float) -> float:
        """
        Error function approximation
        """
        # Approximation using Taylor series
        a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
        p = 0.3275911
        
        sign = 1.0
        if x < 0:
            sign = -1.0
            x = -x
        
        # A&S formula 7.1.26
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
        
        return sign * y
    
    def calculate_time_to_expiry(self, expiry_date: date, current_date: Optional[date] = None) -> float:
        """
        Calculate time to expiry in years
        
        Args:
            expiry_date: Option expiry date
            current_date: Current date (defaults to today)
            
        Returns:
            Time to expiry in years
        """
        if current_date is None:
            current_date = date.today()
        
        # Calculate days to expiry
        days_to_expiry = (expiry_date - current_date).days
        
        # For options, typically use trading days (252) or calendar days (365)
        # Using 365 for simplicity
        years_to_expiry = days_to_expiry / 365.0
        
        # Ensure non-negative
        return max(0.0, years_to_expiry)
    
    def calculate_implied_volatility_from_price(self,
                                                spot_price: float,
                                                strike_price: float,
                                                time_to_expiry: float,
                                                option_price: float,
                                                option_type: str,
                                                initial_guess: float = 0.20,
                                                max_iterations: int = 100,
                                                tolerance: float = 0.0001) -> Optional[float]:
        """
        Calculate implied volatility from option price using Newton-Raphson method
        
        Args:
            spot_price: Current spot price
            strike_price: Strike price
            time_to_expiry: Time to expiry in years
            option_price: Current option premium
            option_type: 'CE' or 'PE'
            initial_guess: Initial IV guess (default: 20%)
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            
        Returns:
            Implied volatility or None if calculation fails
        """
        iv = initial_guess
        
        for i in range(max_iterations):
            # Calculate option price with current IV
            d1, d2 = self._calculate_d1_d2(spot_price, strike_price, time_to_expiry, iv, self.risk_free_rate)
            theoretical_price = self._calculate_option_price(
                spot_price, strike_price, time_to_expiry, iv, option_type, d1, d2
            )
            
            # Calculate vega (derivative of price w.r.t. IV)
            vega = self._calculate_vega(spot_price, time_to_expiry, d1)
            
            # Check convergence
            price_diff = theoretical_price - option_price
            
            if abs(price_diff) < tolerance:
                return iv
            
            # Avoid division by zero
            if abs(vega) < 1e-10:
                break
            
            # Newton-Raphson update: IV_new = IV_old - (price_diff / vega)
            iv_new = iv - price_diff / (vega / 100.0)  # Adjust for vega scaling
            
            # Ensure reasonable bounds (0.001 to 5.0 = 0.1% to 500%)
            iv_new = max(0.001, min(5.0, iv_new))
            
            # Check if stuck
            if abs(iv_new - iv) < 1e-8:
                break
            
            iv = iv_new
        
        # If didn't converge, return best guess (silently, no warning)
        return iv if abs(price_diff) < 0.01 else None
