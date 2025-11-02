"""
Options Chain Scanner
Advanced filtering and scanning for options selling strategies
Includes Greeks calculation, IV Rank, liquidity scoring, and strategy-specific filters
"""

import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from options.OptionsDataFetcher import OptionsDataFetcher
from options.OptionsGreeksCalculator import OptionsGreeksCalculator
from options.IVRankCalculator import IVRankCalculator
from common.Boilerplate import get_db_connection


class OptionsScanner:
    """
    Advanced Options Chain Scanner
    Filters and scores options for selling strategies based on multiple criteria
    """
    
    def __init__(self, risk_free_rate: float = 0.065, lookback_days: int = 252):
        """
        Initialize Options Scanner
        
        Args:
            risk_free_rate: Risk-free interest rate (default: 6.5% for India)
            lookback_days: Days to look back for IV Rank (default: 252 trading days)
        """
        self.data_fetcher = OptionsDataFetcher()
        self.greeks_calculator = OptionsGreeksCalculator(risk_free_rate=risk_free_rate)
        self.iv_rank_calculator = IVRankCalculator(lookback_days=lookback_days)
        self.risk_free_rate = risk_free_rate
    
    def scan_options_chain(self,
                          expiry: Optional[date] = None,
                          strike_range: Optional[Tuple[float, float]] = None,
                          option_type: Optional[str] = None,
                          strategy_type: str = 'covered_call',
                          min_iv_rank: float = 50.0,
                          max_iv_rank: float = 100.0,
                          min_liquidity_score: float = 0.5,
                          min_volume: int = 0,
                          min_oi: int = 0,
                          max_days_to_expiry: int = 60,
                          min_days_to_expiry: int = 7,
                          min_delta: Optional[float] = None,
                          max_delta: Optional[float] = None,
                          current_spot: Optional[float] = None) -> List[Dict]:
        """
        Scan and filter options chain with advanced criteria
        
        Args:
            expiry: Expiry date filter (None = all)
            strike_range: Strike price range filter
            option_type: 'CE' or 'PE' or None for both
            strategy_type: Strategy type ('covered_call', 'cash_secured_put', 'iron_condor', 'strangle', etc.)
            min_iv_rank: Minimum IV Rank (0-100)
            max_iv_rank: Maximum IV Rank (0-100)
            min_liquidity_score: Minimum liquidity score (0-1)
            min_volume: Minimum daily volume
            min_oi: Minimum open interest
            max_days_to_expiry: Maximum days to expiry
            min_days_to_expiry: Minimum days to expiry
            min_delta: Minimum delta filter
            max_delta: Maximum delta filter
            current_spot: Current spot price (auto-fetched if None)
            
        Returns:
            List of filtered and scored option candidates
        """
        try:
            # Get current spot price if not provided
            if current_spot is None:
                current_spot = self.data_fetcher.get_nifty_current_price()
                if current_spot is None:
                    logging.warning("Could not fetch current spot price")
                    return []
            
            # Fetch options chain
            options_chain = self.data_fetcher.get_options_chain(
                expiry=expiry,
                strike_range=strike_range,
                option_type=option_type,
                min_volume=min_volume,
                min_oi=min_oi
            )
            
            if options_chain.empty:
                logging.warning(f"No options data found for given criteria. Expiry: {expiry}, Option Type: {option_type}, Min Volume: {min_volume}, Min OI: {min_oi}")
                return []
            
            logging.info(f"Fetched {len(options_chain)} options from database for scanning")
            
            # Get expiry dates to calculate days to expiry
            expiries = options_chain['expiry'].unique() if 'expiry' in options_chain.columns else []
            
            # Enrich options with Greeks, IV Rank, and liquidity metrics
            enriched_options = []
            
            for _, option in options_chain.iterrows():
                try:
                    # Calculate time to expiry
                    expiry_date = option['expiry']
                    if isinstance(expiry_date, str):
                        expiry_date = datetime.strptime(expiry_date, '%Y-%m-%d').date()
                    
                    days_to_expiry = (expiry_date - date.today()).days
                    
                    # Filter by days to expiry
                    if days_to_expiry < min_days_to_expiry or days_to_expiry > max_days_to_expiry:
                        continue
                    
                    time_to_expiry_years = days_to_expiry / 365.0
                    
                    # Estimate implied volatility from option price if not available
                    # We'll need to back-calculate IV from option price
                    option_price = option.get('last_price') or option.get('average_price', 0)
                    
                    if not option_price or option_price <= 0:
                        logging.debug(f"Skipping option {option.get('tradingsymbol', 'Unknown')} - invalid price: {option_price}")
                        continue
                    
                    strike_price = float(option['strike_price'])
                    opt_type = option['option_type']
                    
                    # Estimate IV from option price (using iterative method)
                    # For now, use a default IV if we can't calculate
                    # In production, you'd want to fetch IV from market data or calculate it
                    estimated_iv = self._estimate_iv_from_price(
                        current_spot, strike_price, time_to_expiry_years, 
                        option_price, opt_type
                    )
                    
                    if estimated_iv is None or estimated_iv <= 0:
                        # Use a fallback IV estimate if calculation fails
                        estimated_iv = 0.20  # Default 20% IV
                        logging.debug(f"Using fallback IV for {option.get('tradingsymbol', 'Unknown')}: {estimated_iv}")
                    
                    # Calculate Greeks
                    greeks = self.greeks_calculator.calculate_greeks(
                        spot_price=current_spot,
                        strike_price=strike_price,
                        time_to_expiry=time_to_expiry_years,
                        implied_volatility=estimated_iv,
                        option_type=opt_type,
                        option_price=option_price
                    )
                    
                    # Calculate IV Rank
                    iv_rank_data = self.iv_rank_calculator.calculate_iv_rank(
                        instrument_token=int(option['instrument_token']),
                        strike_price=strike_price,
                        option_type=opt_type,
                        expiry_date=expiry_date,
                        current_iv=estimated_iv
                    )
                    
                    # Calculate liquidity score
                    liquidity_score = self._calculate_liquidity_score(option)
                    
                    # Calculate bid-ask spread estimate
                    bid_ask_spread = self._estimate_bid_ask_spread(option, option_price)
                    
                    # Apply filters
                    # Only filter by IV Rank if it's available
                    if iv_rank_data.get('iv_rank') is not None:
                        if iv_rank_data['iv_rank'] < min_iv_rank or iv_rank_data['iv_rank'] > max_iv_rank:
                            logging.debug(f"Skipping {option.get('tradingsymbol', 'Unknown')} - IV Rank {iv_rank_data['iv_rank']:.1f} outside range [{min_iv_rank}, {max_iv_rank}]")
                            continue
                    # If IV Rank is not available, we still include it (no historical data yet)
                    
                    if liquidity_score < min_liquidity_score:
                        logging.debug(f"Skipping {option.get('tradingsymbol', 'Unknown')} - Liquidity score {liquidity_score:.2f} < {min_liquidity_score}")
                        continue
                    
                    if min_delta is not None and abs(greeks['delta']) < abs(min_delta):
                        continue
                    
                    if max_delta is not None and abs(greeks['delta']) > abs(max_delta):
                        continue
                    
                    # Strategy-specific filtering
                    if not self._passes_strategy_filter(option, greeks, iv_rank_data, strategy_type, current_spot):
                        continue
                    
                    # Calculate overall score
                    overall_score = self._calculate_overall_score(
                        option, greeks, iv_rank_data, liquidity_score, strategy_type
                    )
                    
                    # Compile enriched option data
                    enriched_option = {
                        'instrument_token': int(option['instrument_token']),
                        'tradingsymbol': option.get('tradingsymbol', ''),
                        'strike_price': strike_price,
                        'option_type': opt_type,
                        'expiry': expiry_date.strftime('%Y-%m-%d') if isinstance(expiry_date, date) else str(expiry_date),
                        'days_to_expiry': days_to_expiry,
                        'last_price': float(option_price),
                        'volume': int(option.get('volume', 0)),
                        'oi': int(option.get('oi', 0)),
                        'current_spot': current_spot,
                        'moneyness': self._calculate_moneyness(current_spot, strike_price, opt_type),
                        
                        # Greeks
                        'delta': round(greeks['delta'], 4),
                        'gamma': round(greeks['gamma'], 6),
                        'theta': round(greeks['theta'], 4),
                        'vega': round(greeks['vega'], 4),
                        'rho': round(greeks['rho'], 4),
                        'theoretical_price': round(greeks['theoretical_price'], 2),
                        
                        # IV Data
                        'implied_volatility': round(estimated_iv * 100, 2),  # As percentage
                        'iv_rank': iv_rank_data.get('iv_rank'),
                        'iv_percentile': iv_rank_data.get('iv_percentile'),
                        'iv_interpretation': self.iv_rank_calculator.interpret_iv_rank(
                            iv_rank_data.get('iv_rank', 50), None
                        ) if iv_rank_data.get('iv_rank') else None,
                        
                        # Liquidity Metrics
                        'liquidity_score': round(liquidity_score, 2),
                        'bid_ask_spread': round(bid_ask_spread, 2),
                        'spread_percentage': round((bid_ask_spread / option_price * 100) if option_price > 0 else 0, 2),
                        
                        # Overall Score
                        'overall_score': round(overall_score, 2),
                        
                        # Strategy Suitability
                        'strategy_suitability': self._get_strategy_suitability(
                            option, greeks, iv_rank_data, strategy_type
                        )
                    }
                    
                    enriched_options.append(enriched_option)
                    
                except Exception as e:
                    logging.error(f"Error processing option {option.get('tradingsymbol', 'Unknown')}: {e}")
                    continue
            
            # Sort by overall score (descending)
            enriched_options.sort(key=lambda x: x['overall_score'], reverse=True)
            
            logging.info(f"Scanner completed: {len(enriched_options)} options passed all filters out of {len(options_chain)} total")
            
            return enriched_options
            
        except Exception as e:
            logging.error(f"Error scanning options chain: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return []
    
    def _estimate_iv_from_price(self,
                               spot: float,
                               strike: float,
                               time_to_expiry: float,
                               option_price: float,
                               option_type: str,
                               initial_guess: float = 0.20) -> Optional[float]:
        """
        Estimate implied volatility from option price
        
        Args:
            spot: Current spot price
            strike: Strike price
            time_to_expiry: Time to expiry in years
            option_price: Option premium
            option_type: 'CE' or 'PE'
            initial_guess: Initial IV guess
            
        Returns:
            Estimated IV or None if calculation fails
        """
        if option_price <= 0 or time_to_expiry <= 0:
            return None
        
        # Use Greeks calculator's IV calculation method
        try:
            iv = self.greeks_calculator.calculate_implied_volatility_from_price(
                spot_price=spot,
                strike_price=strike,
                time_to_expiry=time_to_expiry,
                option_price=option_price,
                option_type=option_type,
                initial_guess=initial_guess
            )
            return iv
        except Exception as e:
            logging.debug(f"IV calculation failed: {e}, using default estimate")
            # Fallback: Use a simple heuristic based on moneyness
            moneyness = spot / strike if strike > 0 else 1.0
            if option_type == 'CE':
                # For calls, higher moneyness = higher IV typically needed
                estimated_iv = min(0.50, max(0.05, (option_price / spot) * 2.0))
            else:
                # For puts
                estimated_iv = min(0.50, max(0.05, (option_price / spot) * 2.0))
            return estimated_iv
    
    def _calculate_liquidity_score(self, option: pd.Series) -> float:
        """
        Calculate liquidity score (0-1) based on volume, OI, and spread
        
        Args:
            option: Option row from DataFrame
            
        Returns:
            Liquidity score (0-1, higher is more liquid)
        """
        volume = float(option.get('volume', 0))
        oi = float(option.get('oi', 0))
        price = float(option.get('last_price', 0) or option.get('average_price', 0))
        
        # Normalize volume and OI (assuming typical ranges)
        # Adjust these thresholds based on your market
        volume_score = min(1.0, volume / 10000.0)  # Max score at 10k+ volume
        oi_score = min(1.0, oi / 100000.0)  # Max score at 100k+ OI
        
        # Combine scores (weighted average)
        liquidity_score = (volume_score * 0.6 + oi_score * 0.4)
        
        return liquidity_score
    
    def _estimate_bid_ask_spread(self, option: pd.Series, option_price: float) -> float:
        """
        Estimate bid-ask spread
        
        Args:
            option: Option row
            option_price: Current option price
            
        Returns:
            Estimated bid-ask spread
        """
        # If we have buy/sell quantity, estimate spread
        buy_qty = option.get('buy_quantity', 0) or 0
        sell_qty = option.get('sell_quantity', 0) or 0
        
        # Simple heuristic: if price is low, use percentage-based estimate
        if option_price < 1.0:
            spread = option_price * 0.05  # 5% for low-priced options
        elif option_price < 10.0:
            spread = option_price * 0.02  # 2% for medium-priced options
        else:
            spread = option_price * 0.01  # 1% for high-priced options
        
        # Adjust based on liquidity
        total_qty = buy_qty + sell_qty
        if total_qty > 0:
            # More liquidity = tighter spread
            liquidity_factor = min(1.0, total_qty / 50000.0)
            spread = spread * (1.0 - liquidity_factor * 0.5)  # Reduce spread by up to 50%
        
        return max(0.01, spread)  # Minimum spread of 0.01
    
    def _calculate_moneyness(self, spot: float, strike: float, option_type: str) -> str:
        """
        Calculate moneyness classification
        
        Args:
            spot: Current spot price
            strike: Strike price
            option_type: 'CE' or 'PE'
            
        Returns:
            Moneyness classification (ITM, ATM, OTM)
        """
        if option_type == 'CE':
            if strike < spot * 0.98:
                return 'ITM'
            elif strike > spot * 1.02:
                return 'OTM'
            else:
                return 'ATM'
        else:  # PE
            if strike > spot * 1.02:
                return 'ITM'
            elif strike < spot * 0.98:
                return 'OTM'
            else:
                return 'ATM'
    
    def _passes_strategy_filter(self,
                              option: pd.Series,
                              greeks: Dict,
                              iv_rank_data: Dict,
                              strategy_type: str,
                              current_spot: float) -> bool:
        """
        Check if option passes strategy-specific filters
        
        Args:
            option: Option data
            greeks: Greeks data
            iv_rank_data: IV Rank data
            strategy_type: Strategy type
            current_spot: Current spot price
            
        Returns:
            True if passes filter, False otherwise
        """
        strike = float(option['strike_price'])
        opt_type = option['option_type']
        iv_rank = iv_rank_data.get('iv_rank')
        delta = greeks.get('delta', 0)
        
        if strategy_type == 'covered_call':
            # For covered calls: OTM calls with high IV rank
            return (opt_type == 'CE' and 
                   strike > current_spot and 
                   (iv_rank is None or iv_rank >= 60))
        
        elif strategy_type == 'cash_secured_put':
            # For cash-secured puts: OTM puts with high IV rank
            return (opt_type == 'PE' and 
                   strike < current_spot and 
                   (iv_rank is None or iv_rank >= 60))
        
        elif strategy_type == 'iron_condor':
            # For iron condors: Need both CE and PE, typically OTM
            return True  # Will be handled in pairs
        
        elif strategy_type == 'strangle':
            # For strangles: OTM both CE and PE with high IV rank
            if opt_type == 'CE':
                return strike > current_spot and (iv_rank is None or iv_rank >= 70)
            else:  # PE
                return strike < current_spot and (iv_rank is None or iv_rank >= 70)
        
        elif strategy_type == 'straddle':
            # For straddles: ATM with high IV rank
            return (abs(strike - current_spot) / current_spot < 0.02 and 
                   (iv_rank is None or iv_rank >= 80))
        
        elif strategy_type == 'vertical_spread':
            # For vertical spreads: Moderate delta, high IV rank
            return (0.20 <= abs(delta) <= 0.50 and 
                   (iv_rank is None or iv_rank >= 60))
        
        else:
            # Default: no specific filter
            return True
    
    def _calculate_overall_score(self,
                                option: pd.Series,
                                greeks: Dict,
                                iv_rank_data: Dict,
                                liquidity_score: float,
                                strategy_type: str) -> float:
        """
        Calculate overall suitability score (0-100)
        
        Args:
            option: Option data
            greeks: Greeks data
            iv_rank_data: IV Rank data
            liquidity_score: Liquidity score
            strategy_type: Strategy type
            
        Returns:
            Overall score (0-100)
        """
        score = 0.0
        
        # IV Rank component (40% weight)
        iv_rank = iv_rank_data.get('iv_rank', 50)
        if iv_rank is not None:
            # Higher IV Rank is better for selling
            score += (iv_rank / 100.0) * 40.0
        
        # Liquidity component (30% weight)
        score += liquidity_score * 30.0
        
        # Greeks component (20% weight)
        # Prefer higher theta (time decay) for selling
        theta = abs(greeks.get('theta', 0))
        theta_score = min(1.0, theta / 0.10)  # Normalize theta
        score += theta_score * 20.0
        
        # Strategy-specific bonus (10% weight)
        strategy_bonus = self._get_strategy_bonus(option, greeks, iv_rank_data, strategy_type)
        score += strategy_bonus * 10.0
        
        return min(100.0, score)
    
    def _get_strategy_bonus(self,
                           option: pd.Series,
                           greeks: Dict,
                           iv_rank_data: Dict,
                           strategy_type: str) -> float:
        """
        Get strategy-specific bonus score
        
        Args:
            option: Option data
            greeks: Greeks data
            iv_rank_data: IV Rank data
            strategy_type: Strategy type
            
        Returns:
            Bonus score (0-1)
        """
        strike = float(option['strike_price'])
        opt_type = option['option_type']
        
        # This would need current spot price for full implementation
        # For now, return a basic bonus
        
        iv_rank = iv_rank_data.get('iv_rank', 50)
        if iv_rank is not None and iv_rank >= 70:
            return 1.0  # High IV rank = full bonus
        elif iv_rank is not None and iv_rank >= 50:
            return 0.5  # Moderate IV rank = half bonus
        
        return 0.0
    
    def _get_strategy_suitability(self,
                                 option: pd.Series,
                                 greeks: Dict,
                                 iv_rank_data: Dict,
                                 strategy_type: str) -> Dict:
        """
        Get strategy suitability assessment
        
        Args:
            option: Option data
            greeks: Greeks data
            iv_rank_data: IV Rank data
            strategy_type: Strategy type
            
        Returns:
            Suitability dictionary
        """
        iv_rank = iv_rank_data.get('iv_rank', 50)
        theta = greeks.get('theta', 0)
        
        if iv_rank is None:
            suitability = 'Unknown'
        elif iv_rank >= 70:
            suitability = 'Excellent'
        elif iv_rank >= 50:
            suitability = 'Good'
        elif iv_rank >= 30:
            suitability = 'Fair'
        else:
            suitability = 'Poor'
        
        return {
            'suitability': suitability,
            'iv_rank': iv_rank,
            'primary_factors': [
                f"IV Rank: {iv_rank:.1f}" if iv_rank else "IV Rank: N/A",
                f"Theta: ₹{theta:.4f}/day",
                f"Liquidity Score: {self._calculate_liquidity_score(option):.2f}"
            ]
        }
