"""
Portfolio Hedging Module
Calculates portfolio Beta, correlation with indices, and suggests hedging strategies
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import json
from typing import Dict, List, Optional
import logging
from common.Boilerplate import get_db_connection
from options.OptionsDataFetcher import OptionsDataFetcher
from common.MarginCalculator import MarginCalculator

class PortfolioHedgeAnalyzer:
    """
    Analyzes portfolio risk and suggests hedging strategies
    """
    
    def __init__(self, db_config: Dict = None):
        """
        Initialize Portfolio Hedge Analyzer
        
        Args:
            db_config: Database configuration dict (host, database, user, password, port)
        """
        if db_config is None:
            # Default to environment variables
            import os
            db_config = {
                'host': os.getenv('PG_HOST', 'postgres'),
                'database': os.getenv('PG_DATABASE', 'mydb'),
                'user': os.getenv('PG_USER', 'postgres'),
                'password': os.getenv('PG_PASSWORD', 'postgres'),
                'port': int(os.getenv('PG_PORT', 5432))
            }
        self.db_config = db_config
        
        # Nifty 50 instrument token
        self.nifty_token = 256265
        
        # Initialize helpers
        self.options_fetcher = OptionsDataFetcher()
        self.margin_calculator = MarginCalculator()
        
    def get_current_holdings(self) -> pd.DataFrame:
        """
        Get current holdings (equity + mutual funds) from database
        
        Returns:
            DataFrame with holdings data (equity and MF combined)
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # First, check what data exists in the database for debugging
            cursor.execute("""
                SELECT COUNT(*) as total_count, 
                       COUNT(CASE WHEN quantity > 0 THEN 1 END) as positive_qty_count,
                       MAX(run_date) as latest_run_date,
                       MIN(run_date) as earliest_run_date
                FROM my_schema.holdings
            """)
            equity_stats = cursor.fetchone()
            logging.info(f"Equity holdings stats: {equity_stats}")
            
            cursor.execute("""
                SELECT COUNT(*) as total_count, 
                       COUNT(CASE WHEN quantity > 0 THEN 1 END) as positive_qty_count,
                       MAX(run_date) as latest_run_date,
                       MIN(run_date) as earliest_run_date
                FROM my_schema.mf_holdings
            """)
            mf_stats = cursor.fetchone()
            logging.info(f"MF holdings stats: {mf_stats}")
            
            # Get equity holdings - use the same pattern as the working holdings API
            # Get the most recent run_date and fetch holdings from that date only
            # First check if MAX(run_date) exists
            cursor.execute("SELECT MAX(run_date) as max_run_date FROM my_schema.holdings")
            max_equity_date = cursor.fetchone()[0]
            logging.info(f"Latest equity holdings run_date: {max_equity_date}")
            
            if max_equity_date:
                # Use the same pattern as the working holdings API - LEFT JOIN with rt_intraday_price
                cursor.execute("""
                    SELECT 
                        h.trading_symbol,
                        h.instrument_token,
                        h.quantity,
                        h.average_price,
                        COALESCE(h.last_price, rt.price_close, h.close_price, 0) as current_price,
                        h.pnl,
                        'EQUITY' as holding_type
                    FROM my_schema.holdings h
                    LEFT JOIN (
                        SELECT DISTINCT ON (scrip_id) scrip_id, price_close, price_date
                        FROM my_schema.rt_intraday_price
                        WHERE price_date::date <= CURRENT_DATE
                        ORDER BY scrip_id, price_date DESC
                    ) rt ON h.trading_symbol = rt.scrip_id
                    WHERE h.run_date = %s
                    AND h.quantity > 0
                """, (max_equity_date,))
                equity_rows = cursor.fetchall()
            else:
                equity_rows = []
                logging.warning("No equity holdings run_date found - table may be empty")
            
            logging.info(f"Equity holdings fetched: {len(equity_rows)} rows")
            
            # Filter out any null quantities (shouldn't happen due to SQL filter, but just in case)
            equity_rows = [row for row in equity_rows if row[2] and row[2] > 0]
            
            equity_columns = ['trading_symbol', 'instrument_token', 'quantity', 
                            'average_price', 'current_price', 'pnl', 'holding_type']
            
            # Get mutual fund holdings - use the same pattern as the working holdings API
            # Get the most recent run_date and fetch holdings from that date only
            # First check if MAX(run_date) exists
            cursor.execute("SELECT MAX(run_date) as max_run_date FROM my_schema.mf_holdings")
            max_mf_date = cursor.fetchone()[0]
            logging.info(f"Latest MF holdings run_date: {max_mf_date}")
            
            if max_mf_date:
                cursor.execute("""
                    SELECT 
                        tradingsymbol as trading_symbol,
                        0 as instrument_token,  -- MF holdings don't have instrument_token
                        quantity,
                        average_price,
                        COALESCE(last_price, 0) as current_price,
                        pnl,
                        COALESCE(current_value, quantity * COALESCE(last_price, 0)) as current_value,
                        COALESCE(invested_amount, quantity * average_price) as invested_value,
                        'MF' as holding_type
                    FROM my_schema.mf_holdings
                    WHERE run_date = %s
                    AND quantity > 0
                """, (max_mf_date,))
                mf_rows = cursor.fetchall()
            else:
                mf_rows = []
                logging.warning("No MF holdings run_date found - table may be empty")
            
            logging.info(f"MF holdings fetched: {len(mf_rows)} rows")
            
            # Filter out any null quantities (shouldn't happen due to SQL filter, but just in case)
            mf_rows = [row for row in mf_rows if row[2] and row[2] > 0]
            
            mf_columns = ['trading_symbol', 'instrument_token', 'quantity', 
                         'average_price', 'current_price', 'pnl', 'current_value', 'invested_value', 'holding_type']
            
            conn.close()
            
            logging.info(f"Found {len(equity_rows)} equity holdings and {len(mf_rows)} MF holdings before combining")
            
            # Combine equity and MF holdings
            all_rows = []
            if equity_rows:
                logging.info(f"Processing {len(equity_rows)} equity rows...")
                # Add current_value and invested_value columns for equity holdings
                for row in equity_rows:
                    # row[0]=trading_symbol, row[1]=instrument_token, row[2]=quantity, 
                    # row[3]=average_price, row[4]=current_price, row[5]=pnl, row[6]=holding_type
                    quantity = row[2]
                    avg_price = row[3]
                    current_price = row[4]
                    current_value = quantity * current_price
                    invested_value = quantity * avg_price
                    # Add current_value and invested_value
                    equity_row = [row[0], row[1], row[2], row[3], row[4], row[5], current_value, invested_value, row[6]]
                    all_rows.append(equity_row)
                    if len(all_rows) <= 3:  # Log first few rows for debugging
                        logging.info(f"Added equity row: {equity_row}")
            
            if mf_rows:
                logging.info(f"Processing {len(mf_rows)} MF rows...")
                # MF rows already have current_value and invested_value
                for row in mf_rows:
                    # row[0]=trading_symbol, row[1]=instrument_token, row[2]=quantity, 
                    # row[3]=average_price, row[4]=current_price, row[5]=pnl, 
                    # row[6]=current_value, row[7]=invested_value, row[8]=holding_type
                    mf_row = [row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8]]
                    all_rows.append(mf_row)
                    if len(all_rows) - len(equity_rows) <= 3:  # Log first few MF rows for debugging
                        logging.info(f"Added MF row: {mf_row}")
            
            logging.info(f"Total all_rows count after combining: {len(all_rows)}")
            
            if not all_rows:
                logging.warning(f"No holdings found after combining. Equity stats: {equity_stats}, MF stats: {mf_stats}")
                logging.warning(f"Equity rows found: {len(equity_rows)}, MF rows found: {len(mf_rows)}")
                return pd.DataFrame(columns=['trading_symbol', 'instrument_token', 'quantity', 
                                           'average_price', 'current_price', 'pnl', 'current_value', 
                                           'invested_value', 'holding_type'])
            
            # Create DataFrame from combined holdings
            combined_columns = ['trading_symbol', 'instrument_token', 'quantity', 
                               'average_price', 'current_price', 'pnl', 'current_value', 
                               'invested_value', 'holding_type']
            df = pd.DataFrame(all_rows, columns=combined_columns)
            logging.info(f"DataFrame created with {len(df)} rows, columns: {df.columns.tolist()}")
            
            if len(df) > 0:
                logging.info(f"Sample DataFrame data before filtering: quantity range: [{df['quantity'].min()}, {df['quantity'].max()}], current_value range: [{df['current_value'].min()}, {df['current_value'].max()}]")
                logging.info(f"Sample rows: {df.head(3).to_dict('records')}")
            
            # Ensure current_value and invested_value are numeric
            df['current_value'] = pd.to_numeric(df['current_value'], errors='coerce').fillna(0)
            df['invested_value'] = pd.to_numeric(df['invested_value'], errors='coerce').fillna(0)
            df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0)
            
            logging.info(f"After numeric conversion: {len(df)} rows")
            
            # Filter out rows with zero quantity only
            # Note: We keep holdings even if current_value is 0 (price might not be available yet)
            before_filter = len(df)
            df = df[df['quantity'] > 0]
            logging.info(f"After quantity > 0 filter: {len(df)} rows (filtered out {before_filter - len(df)} rows)")
            
            # Log holdings with zero current_value for debugging
            zero_value_holdings = df[df['current_value'] == 0]
            if len(zero_value_holdings) > 0:
                logging.warning(f"Found {len(zero_value_holdings)} holdings with zero current_value (missing price data): {zero_value_holdings[['trading_symbol', 'quantity', 'current_price']].to_dict('records')}")
            
            logging.info(f"Final holdings DataFrame: {len(df)} rows")
            if len(df) > 0:
                total_value = df['current_value'].sum()
                equity_count = len(df[df['holding_type'] == 'EQUITY'])
                mf_count = len(df[df['holding_type'] == 'MF'])
                equity_value = df[df['holding_type'] == 'EQUITY']['current_value'].sum()
                mf_value = df[df['holding_type'] == 'MF']['current_value'].sum()
                logging.info(f"Total portfolio value: ₹{total_value:,.2f} (Equity: {equity_count} holdings, ₹{equity_value:,.2f}; MF: {mf_count} holdings, ₹{mf_value:,.2f})")
                logging.info(f"Sample holdings: {df[['trading_symbol', 'quantity', 'current_value', 'holding_type']].head(5).to_dict('records')}")
            else:
                logging.warning(f"No holdings in final DataFrame. Raw counts - Equity: {len(equity_rows)}, MF: {len(mf_rows)}")
            
            return df
        except Exception as e:
            logging.error(f"Error fetching holdings: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return pd.DataFrame()
    
    def calculate_portfolio_beta(self, lookback_days: int = 60) -> Dict:
        """
        Calculate portfolio Beta with respect to Nifty 50
        
        Args:
            lookback_days: Number of days to look back for correlation
            
        Returns:
            Dictionary with beta, correlation, and other metrics
        """
        try:
            holdings = self.get_current_holdings()
            if holdings.empty:
                return {
                    'beta': 0.0,
                    'correlation': 0.0,
                    'portfolio_value': 0.0,
                    'error': 'No holdings found'
                }
            
            # Get historical price data for portfolio and Nifty
            portfolio_returns = []
            nifty_returns = []
            
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get date range
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=lookback_days)
            
            # Calculate portfolio daily returns
            dates = []
            for date in pd.date_range(start_date, end_date):
                date_str = date.strftime('%Y-%m-%d')
                
                # Get portfolio value for this date
                # Note: Only equity holdings have historical OHLC data for beta calculation
                # MF holdings are included in total portfolio value but not in historical returns
                portfolio_value = 0
                
                # Calculate equity holdings value from historical prices
                equity_holdings = holdings[holdings['holding_type'] == 'EQUITY']
                for _, holding in equity_holdings.iterrows():
                    try:
                        # Get the most recent price for this symbol up to the current date
                        symbol = str(holding['trading_symbol'])  # Ensure it's a string
                        logging.debug(f"Fetching price for {symbol} on {date_str}")
                        cursor.execute("""
                            SELECT price_close
                            FROM my_schema.rt_intraday_price
                            WHERE scrip_id = %s
                            AND price_date <= %s
                            ORDER BY price_date DESC
                            LIMIT 1
                        """, (symbol, date_str))
                        result = cursor.fetchone()
                        if result and result[0]:
                            portfolio_value += holding['quantity'] * float(result[0])
                    except Exception as query_error:
                        logging.warning(f"Error fetching price for {holding.get('trading_symbol', 'UNKNOWN')} on {date_str}: {query_error}")
                        # Continue with next holding
                        continue
                
                # Add MF holdings current value (they don't have daily historical data, use current value)
                # For beta calculation, we'll only use equity holdings for returns calculation
                # but add MF value to total portfolio value for today's calculation
                if date_str == datetime.now().strftime('%Y-%m-%d'):
                    mf_holdings = holdings[holdings['holding_type'] == 'MF']
                    if not mf_holdings.empty:
                        portfolio_value += mf_holdings['current_value'].sum()
                
                if portfolio_value > 0:
                    dates.append(date_str)
                    portfolio_returns.append(portfolio_value)
            
            # Get Nifty returns
            cursor.execute("""
                SELECT price_date, price_close
                FROM my_schema.rt_intraday_price
                WHERE scrip_id = 'NIFTY 50'
                AND price_date >= %s
                AND price_date <= %s
                ORDER BY price_date
            """, (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
            
            nifty_data = cursor.fetchall()
            conn.close()
            
            if len(portfolio_returns) < 10 or len(nifty_data) < 10:
                # Return total portfolio value (equity + MF)
                total_portfolio_value = float(holdings['current_value'].sum())
                return {
                    'beta': 0.0,
                    'correlation': 0.0,
                    'portfolio_value': total_portfolio_value,
                    'error': 'Insufficient data for beta calculation (need at least 10 days of historical data for equity holdings and Nifty 50)',
                    'equity_value': float(holdings[holdings['holding_type'] == 'EQUITY']['current_value'].sum()) if 'holding_type' in holdings.columns else 0,
                    'mf_value': float(holdings[holdings['holding_type'] == 'MF']['current_value'].sum()) if 'holding_type' in holdings.columns and 'MF' in holdings['holding_type'].values else 0,
                    'equity_holdings_count': len(holdings[holdings['holding_type'] == 'EQUITY']) if 'holding_type' in holdings.columns else len(holdings),
                    'mf_holdings_count': len(holdings[holdings['holding_type'] == 'MF']) if 'holding_type' in holdings.columns and 'MF' in holdings['holding_type'].values else 0
                }
            
            # Convert to returns (percentage change)
            portfolio_df = pd.DataFrame({
                'date': dates[:len(portfolio_returns)],
                'value': portfolio_returns
            })
            portfolio_df['returns'] = portfolio_df['value'].pct_change()
            
            nifty_df = pd.DataFrame(nifty_data, columns=['date', 'price'])
            nifty_df['returns'] = nifty_df['price'].pct_change()
            
            # Align dates
            merged = pd.merge(
                portfolio_df[['date', 'returns']],
                nifty_df[['date', 'returns']],
                on='date',
                suffixes=('_portfolio', '_nifty')
            )
            
            merged = merged.dropna()
            
            if len(merged) < 10:
                return {
                    'beta': 0.0,
                    'correlation': 0.0,
                    'portfolio_value': float(holdings['current_value'].sum()),
                    'error': 'Insufficient aligned data'
                }
            
            # Calculate correlation
            correlation = merged['returns_portfolio'].corr(merged['returns_nifty'])
            
            # Calculate beta: Cov(Portfolio, Nifty) / Var(Nifty)
            covariance = merged['returns_portfolio'].cov(merged['returns_nifty'])
            variance_nifty = merged['returns_nifty'].var()
            
            beta = covariance / variance_nifty if variance_nifty > 0 else 0.0
            
            # Calculate total portfolio value (equity + MF)
            total_portfolio_value = float(holdings['current_value'].sum())
            equity_value = float(holdings[holdings['holding_type'] == 'EQUITY']['current_value'].sum()) if 'holding_type' in holdings.columns else total_portfolio_value
            mf_value = float(holdings[holdings['holding_type'] == 'MF']['current_value'].sum()) if 'holding_type' in holdings.columns and 'MF' in holdings['holding_type'].values else 0
            
            return {
                'beta': float(beta),
                'correlation': float(correlation),
                'portfolio_value': total_portfolio_value,  # Total value including MF
                'equity_value': equity_value,
                'mf_value': mf_value,
                'days_analyzed': len(merged),
                'portfolio_volatility': float(merged['returns_portfolio'].std()),
                'nifty_volatility': float(merged['returns_nifty'].std()),
                'note': 'Beta calculated using equity holdings only (MF holdings included in portfolio value)'
            }
            
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Error calculating portfolio beta: {error_msg}")
            import traceback
            logging.error(traceback.format_exc())
            # Ensure error message is safe (no % formatting issues)
            if '%' in error_msg:
                error_msg = error_msg.replace('%', '%%')
            return {
                'beta': 0.0,
                'correlation': 0.0,
                'portfolio_value': 0.0,
                'error': error_msg
            }
    
    def _get_nifty_price(self) -> Optional[float]:
        """Get current Nifty price from database"""
        try:
            price = self.options_fetcher.get_nifty_current_price()
            if price:
                return price
            
            # Fallback to OHLC data
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT price_close
                FROM my_schema.rt_intraday_price
                WHERE scrip_id = 'NIFTY 50'
                AND price_date = (SELECT MAX(price_date) FROM my_schema.rt_intraday_price WHERE scrip_id = 'NIFTY 50')
                ORDER BY price_date DESC
                LIMIT 1
            """)
            result = cursor.fetchone()
            conn.close()
            
            if result and result[0]:
                return float(result[0])
            return None
        except Exception as e:
            logging.error(f"Error getting Nifty price: {e}")
            return None
    
    def _suggest_protective_puts(self, portfolio_value: float, beta: float, target_hedge_ratio: float, 
                                   nifty_price: float) -> List[Dict]:
        """Suggest protective puts strategy"""
        suggestions = []
        
        try:
            # Get available expiry dates
            expiries = self.options_fetcher.get_expiry_dates()
            if not expiries:
                return suggestions
            
            # Select near month and far month expiries
            near_expiry = expiries[0] if len(expiries) > 0 else None
            far_expiry = expiries[1] if len(expiries) > 1 else near_expiry
            
            # Calculate hedge quantity (in shares)
            hedge_value = portfolio_value * beta * target_hedge_ratio
            lot_size = 50  # Nifty lot size
            
            # Get ATM and slightly OTM put strikes
            atm_strike, _ = self.options_fetcher.get_atm_strikes(nifty_price)
            otm_strikes = [atm_strike - 100, atm_strike - 200, atm_strike - 300]  # OTM puts
            
            strikes_to_consider = [atm_strike] + [s for s in otm_strikes if s > 0]
            
            for expiry in [near_expiry, far_expiry]:
                if not expiry:
                    continue
                    
                for strike in strikes_to_consider:
                    # Get PUT options for this strike and expiry
                    options_chain = self.options_fetcher.get_options_chain(
                        expiry=expiry,
                        strike_range=(strike - 25, strike + 25),
                        option_type='PE',
                        min_volume=100,  # Minimum liquidity
                        min_oi=10000
                    )
                    
                    if options_chain.empty:
                        continue
                    
                    # Get the option closest to desired strike
                    option = options_chain.iloc[0]
                    
                    premium_per_share = float(option['last_price']) if option['last_price'] else 0
                    if premium_per_share <= 0:
                        continue
                    
                    # Calculate quantity in lots
                    num_lots = max(1, int(hedge_value / (strike * lot_size)))
                    total_premium = premium_per_share * lot_size * num_lots
                    
                    # Calculate margin (premium only for buying puts)
                    margin_result = self.margin_calculator.calculate_options_margin(
                        instrument_token=int(option['instrument_token']),
                        quantity=num_lots,
                        strike_price=float(option['strike_price']),
                        premium=premium_per_share,
                        option_type='PE',
                        is_long=True
                    )
                    
                    suggestions.append({
                        'strategy_type': 'PROTECTIVE_PUT',
                        'strategy_name': 'Protective Put',
                        'instrument': option['tradingsymbol'],
                        'instrument_token': int(option['instrument_token']),
                        'direction': 'LONG',
                        'option_type': 'PE',
                        'strike_price': float(option['strike_price']),
                        'expiry': expiry.strftime('%Y-%m-%d') if isinstance(expiry, date) else str(expiry),
                        'quantity': num_lots,
                        'lot_size': lot_size,
                        'entry_price': premium_per_share,
                        'total_premium': float(total_premium),
                        'margin_required': float(margin_result.get('total_required', total_premium)),
                        'coverage_value': float(hedge_value),
                        'coverage_percentage': float((hedge_value / portfolio_value) * 100) if portfolio_value > 0 else 0,
                        'rationale': f'Protective put at {strike} strike to hedge {target_hedge_ratio*100:.0f}% portfolio downside',
                        'max_profit': 'Unlimited (from portfolio)',
                        'max_loss': f'Premium paid: ₹{total_premium:,.2f}',
                        'break_even': f'Strike - Premium = {strike - premium_per_share:.2f}'
                    })
                    
                    # Limit to top 3 suggestions per expiry
                    if len(suggestions) >= 6:
                        break
                    
        except Exception as e:
            logging.error(f"Error suggesting protective puts: {e}")
            import traceback
            logging.error(traceback.format_exc())
        
        return suggestions
    
    def _suggest_covered_calls(self, portfolio_value: float, beta: float, nifty_price: float) -> List[Dict]:
        """Suggest covered calls strategy"""
        suggestions = []
        
        try:
            # Get available expiry dates
            expiries = self.options_fetcher.get_expiry_dates()
            if not expiries:
                return suggestions
            
            near_expiry = expiries[0] if len(expiries) > 0 else None
            
            # Get OTM call strikes (above current price)
            atm_strike, _ = self.options_fetcher.get_atm_strikes(nifty_price)
            otm_strikes = [atm_strike + 50, atm_strike + 100, atm_strike + 200, atm_strike + 300]
            
            lot_size = 50
            
            for strike in otm_strikes:
                if not near_expiry:
                    continue
                    
                # Get CALL options for this strike and expiry
                options_chain = self.options_fetcher.get_options_chain(
                    expiry=near_expiry,
                    strike_range=(strike - 25, strike + 25),
                    option_type='CE',
                    min_volume=100,
                    min_oi=10000
                )
                
                if options_chain.empty:
                    continue
                
                option = options_chain.iloc[0]
                
                premium_per_share = float(option['last_price']) if option['last_price'] else 0
                if premium_per_share <= 0:
                    continue
                
                # Calculate quantity based on portfolio coverage (typically sell calls on 50-100% of portfolio)
                portfolio_lots = max(1, int(portfolio_value / (nifty_price * lot_size)))
                num_lots = max(1, int(portfolio_lots * 0.5))  # Cover 50% of portfolio
                
                total_premium = premium_per_share * lot_size * num_lots
                
                # Calculate margin for selling calls
                margin_result = self.margin_calculator.calculate_options_margin(
                    instrument_token=int(option['instrument_token']),
                    quantity=num_lots,
                    strike_price=float(option['strike_price']),
                    premium=premium_per_share,
                    option_type='CE',
                    is_long=False
                )
                
                suggestions.append({
                    'strategy_type': 'COVERED_CALL',
                    'strategy_name': 'Covered Call',
                    'instrument': option['tradingsymbol'],
                    'instrument_token': int(option['instrument_token']),
                    'direction': 'SHORT',
                    'option_type': 'CE',
                    'strike_price': float(option['strike_price']),
                    'expiry': near_expiry.strftime('%Y-%m-%d') if isinstance(near_expiry, date) else str(near_expiry),
                    'quantity': num_lots,
                    'lot_size': lot_size,
                    'entry_price': premium_per_share,
                    'total_premium_income': float(total_premium),
                    'margin_required': float(margin_result.get('total_required', total_premium)),
                    'coverage_value': float(portfolio_value * 0.5),
                    'coverage_percentage': 50.0,
                    'rationale': f'Sell OTM calls at {strike} strike to generate income, cap upside at {strike}',
                    'max_profit': f'Premium received + (Strike - Portfolio entry): ₹{total_premium:,.2f}',
                    'max_loss': 'Unlimited downside (from portfolio)',
                    'break_even': f'Portfolio entry - Premium = {nifty_price - premium_per_share:.2f}'
                })
                
                if len(suggestions) >= 4:
                    break
                    
        except Exception as e:
            logging.error(f"Error suggesting covered calls: {e}")
            import traceback
            logging.error(traceback.format_exc())
        
        return suggestions
    
    def _suggest_collar(self, portfolio_value: float, beta: float, target_hedge_ratio: float, 
                         nifty_price: float) -> List[Dict]:
        """Suggest collar strategy (protective put + covered call)"""
        suggestions = []
        
        try:
            expiries = self.options_fetcher.get_expiry_dates()
            if not expiries:
                return suggestions
            
            near_expiry = expiries[0]
            lot_size = 50
            
            # Get protective put (ATM or slightly OTM)
            atm_strike, _ = self.options_fetcher.get_atm_strikes(nifty_price)
            put_strike = atm_strike - 100  # Slightly OTM put
            
            # Get covered call (OTM)
            call_strike = atm_strike + 200  # OTM call
            
            # Get PUT option
            puts = self.options_fetcher.get_options_chain(
                expiry=near_expiry,
                strike_range=(put_strike - 25, put_strike + 25),
                option_type='PE',
                min_volume=100,
                min_oi=10000
            )
            
            # Get CALL option
            calls = self.options_fetcher.get_options_chain(
                expiry=near_expiry,
                strike_range=(call_strike - 25, call_strike + 25),
                option_type='CE',
                min_volume=100,
                min_oi=10000
            )
            
            if puts.empty or calls.empty:
                return suggestions
            
            put_option = puts.iloc[0]
            call_option = calls.iloc[0]
            
            put_premium = float(put_option['last_price']) if put_option['last_price'] else 0
            call_premium = float(call_option['last_price']) if call_option['last_price'] else 0
            
            if put_premium <= 0 or call_premium <= 0:
                return suggestions
            
            # Calculate net cost (premium paid for put - premium received from call)
            net_cost_per_share = put_premium - call_premium
            hedge_value = portfolio_value * beta * target_hedge_ratio
            num_lots = max(1, int(hedge_value / (put_strike * lot_size)))
            
            put_total_cost = put_premium * lot_size * num_lots
            call_total_income = call_premium * lot_size * num_lots
            net_cost = put_total_cost - call_total_income
            
            # Margins
            put_margin = self.margin_calculator.calculate_options_margin(
                instrument_token=int(put_option['instrument_token']),
                quantity=num_lots,
                strike_price=float(put_option['strike_price']),
                premium=put_premium,
                option_type='PE',
                is_long=True
            )
            
            call_margin = self.margin_calculator.calculate_options_margin(
                instrument_token=int(call_option['instrument_token']),
                quantity=num_lots,
                strike_price=float(call_option['strike_price']),
                premium=call_premium,
                option_type='CE',
                is_long=False
            )
            
            total_margin = (put_margin.get('total_required', put_total_cost) + 
                          call_margin.get('total_required', 0))
            
            suggestions.append({
                'strategy_type': 'COLLAR',
                'strategy_name': 'Collar (Protective Put + Covered Call)',
                'instruments': [
                    {
                        'instrument': put_option['tradingsymbol'],
                        'instrument_token': int(put_option['instrument_token']),
                        'option_type': 'PE',
                        'direction': 'LONG',
                        'strike_price': float(put_option['strike_price']),
                        'premium': put_premium,
                        'total_cost': put_total_cost
                    },
                    {
                        'instrument': call_option['tradingsymbol'],
                        'instrument_token': int(call_option['instrument_token']),
                        'option_type': 'CE',
                        'direction': 'SHORT',
                        'strike_price': float(call_option['strike_price']),
                        'premium': call_premium,
                        'total_income': call_total_income
                    }
                ],
                'quantity': num_lots,
                'lot_size': lot_size,
                'expiry': near_expiry.strftime('%Y-%m-%d') if isinstance(near_expiry, date) else str(near_expiry),
                'net_cost': float(net_cost),
                'margin_required': float(total_margin),
                'coverage_value': float(hedge_value),
                'coverage_percentage': float((hedge_value / portfolio_value) * 100) if portfolio_value > 0 else 0,
                'rationale': f'Collar: Buy {put_strike} PE (cost: ₹{put_premium:.2f}) + Sell {call_strike} CE (income: ₹{call_premium:.2f}), Net: ₹{net_cost:,.2f}',
                'max_profit': f'Call strike - Portfolio entry - Net cost',
                'max_loss': f'Put strike - Portfolio entry - Net cost',
                'break_even': f'Portfolio entry + Net cost'
            })
            
        except Exception as e:
            logging.error(f"Error suggesting collar: {e}")
            import traceback
            logging.error(traceback.format_exc())
        
        return suggestions
    
    def _suggest_futures_hedge(self, portfolio_value: float, beta: float, target_hedge_ratio: float,
                                 nifty_price: float) -> List[Dict]:
        """Suggest futures hedge (original strategy)"""
        suggestions = []
        
        try:
            lot_size = 50
            hedge_value = portfolio_value * beta * target_hedge_ratio
            hedge_quantity = int((hedge_value / nifty_price) / lot_size) * lot_size
            
            if hedge_quantity <= 0:
                return suggestions
            
            contract_value = hedge_quantity * nifty_price
            estimated_margin = contract_value * 0.12
            
            suggestions.append({
                'strategy_type': 'FUTURES_SHORT',
                'strategy_name': 'Short Futures Hedge',
                'instrument': 'NIFTY FUT',
                'instrument_token': self.nifty_token,
                'direction': 'SHORT',
                'quantity': hedge_quantity,
                'lot_size': lot_size,
                'entry_price': nifty_price,
                'contract_value': float(contract_value),
                'estimated_margin': float(estimated_margin),
                'margin_required': float(estimated_margin),
                'hedge_value': float(hedge_value),
                'coverage_percentage': float((hedge_value / portfolio_value) * 100) if portfolio_value > 0 else 0,
                'rationale': f'Short Nifty futures to hedge {target_hedge_ratio*100:.0f}% of portfolio exposure (Beta: {beta:.2f})',
                'estimated_premium': 0.0
            })
        except Exception as e:
            logging.error(f"Error suggesting futures hedge: {e}")
        
        return suggestions
    
    def suggest_hedge(self, target_hedge_ratio: float = 0.5, strategy_type: str = 'all') -> Dict:
        """
        Suggest comprehensive hedging strategies for portfolio
        
        Args:
            target_hedge_ratio: Target hedge ratio (0.0 to 1.0), e.g., 0.5 for 50% hedge
            strategy_type: 'all', 'puts', 'calls', 'collars', 'futures', or comma-separated list
            
        Returns:
            Dictionary with hedge suggestions for all strategies
        """
        try:
            holdings = self.get_current_holdings()
            logging.info(f"suggest_hedge: Received holdings DataFrame with {len(holdings)} rows, empty={holdings.empty}")
            if len(holdings) > 0:
                logging.info(f"suggest_hedge: Holdings sample: {holdings[['trading_symbol', 'quantity', 'current_value', 'holding_type']].head(5).to_dict('records')}")
            
            if holdings.empty:
                logging.warning("suggest_hedge: Holdings DataFrame is empty!")
                return {
                    'strategies': [],
                    'total_hedge_value': 0.0,
                    'error': 'No holdings found',
                    'diagnostics': {
                        'holdings_count': 0,
                        'equity_count': 0,
                        'mf_count': 0,
                        'portfolio_value': 0.0,
                        'equity_value': 0.0,
                        'mf_value': 0.0,
                        'beta': 0.0,
                        'correlation': 0.0,
                        'nifty_price': 0.0,
                        'expiries_available': len(self.options_fetcher.get_expiry_dates()),
                        'futures_strategies': 0,
                        'puts_strategies': 0,
                        'calls_strategies': 0,
                        'collars_strategies': 0
                    }
                }
            
            # Calculate portfolio metrics
            beta_result = self.calculate_portfolio_beta()
            
            # Get current Nifty price
            nifty_price = self._get_nifty_price()
            
            # Prepare diagnostics early (even if there are errors)
            equity_count = len(holdings[holdings['holding_type'] == 'EQUITY']) if 'holding_type' in holdings.columns else len(holdings)
            mf_count = len(holdings[holdings['holding_type'] == 'MF']) if 'holding_type' in holdings.columns and 'MF' in holdings['holding_type'].values else 0
            equity_value = float(holdings[holdings['holding_type'] == 'EQUITY']['current_value'].sum()) if 'holding_type' in holdings.columns and 'EQUITY' in holdings['holding_type'].values else float(holdings['current_value'].sum())
            mf_value = float(holdings[holdings['holding_type'] == 'MF']['current_value'].sum()) if 'holding_type' in holdings.columns and 'MF' in holdings['holding_type'].values else 0
            portfolio_value = float(beta_result.get('portfolio_value', equity_value + mf_value))
            
            diagnostics = {
                'holdings_count': len(holdings),
                'equity_count': equity_count,
                'mf_count': mf_count,
                'portfolio_value': float(portfolio_value),
                'equity_value': float(equity_value),
                'mf_value': float(mf_value),
                'beta': float(beta_result.get('beta', 0.0)),
                'correlation': float(beta_result.get('correlation', 0.0)),
                'nifty_price': float(nifty_price or 0),
                'expiries_available': len(self.options_fetcher.get_expiry_dates()),
                'futures_strategies': 0,
                'puts_strategies': 0,
                'calls_strategies': 0,
                'collars_strategies': 0
            }
            
            logging.info(f"suggest_hedge: Early diagnostics calculated: {diagnostics}")
            
            if 'error' in beta_result:
                logging.info(f"suggest_hedge: Returning early with error. Diagnostics: {diagnostics}")
                result = {
                    'strategies': [],
                    'total_strategies': 0,
                    'total_hedge_value': 0.0,
                    'error': beta_result['error'],
                    'diagnostics': diagnostics
                }
                logging.info(f"suggest_hedge: Early return result: {result}")
                return result
            
            portfolio_value = beta_result['portfolio_value']
            beta = beta_result['beta']
            correlation = beta_result['correlation']
            
            if not nifty_price:
                # Update diagnostics with error info
                diagnostics['nifty_price'] = 0.0
                return {
                    'strategies': [],
                    'total_hedge_value': 0.0,
                    'error': 'Cannot determine Nifty price',
                    'diagnostics': diagnostics
                }
            
            # Parse strategy_type filter
            strategy_types = [s.strip().lower() for s in strategy_type.split(',')] if ',' in strategy_type else [strategy_type.lower()]
            all_strategies = strategy_type.lower() == 'all' or 'all' in strategy_types
            
            all_suggestions = []
            
            # Generate all strategies
            if all_strategies or 'futures' in strategy_types or 'future' in strategy_types:
                all_suggestions.extend(self._suggest_futures_hedge(portfolio_value, beta, target_hedge_ratio, nifty_price))
            
            if all_strategies or 'puts' in strategy_types or 'put' in strategy_types or 'protective' in strategy_types:
                all_suggestions.extend(self._suggest_protective_puts(portfolio_value, beta, target_hedge_ratio, nifty_price))
            
            if all_strategies or 'calls' in strategy_types or 'call' in strategy_types or 'covered' in strategy_types:
                all_suggestions.extend(self._suggest_covered_calls(portfolio_value, beta, nifty_price))
            
            if all_strategies or 'collar' in strategy_types or 'collars' in strategy_types:
                all_suggestions.extend(self._suggest_collar(portfolio_value, beta, target_hedge_ratio, nifty_price))
            
            # Calculate total metrics
            total_hedge_value = portfolio_value * beta * target_hedge_ratio
            total_premium_cost = sum(s.get('total_premium', s.get('net_cost', 0)) for s in all_suggestions if s.get('strategy_type') != 'COVERED_CALL')
            total_premium_income = sum(s.get('total_premium_income', 0) for s in all_suggestions if s.get('strategy_type') == 'COVERED_CALL')
            total_margin = sum(s.get('margin_required', s.get('estimated_margin', 0)) for s in all_suggestions)
            
            # Diagnostic information - update the diagnostics we calculated earlier with strategy counts
            logging.info(f"suggest_hedge: Processing diagnostics. Holdings columns: {holdings.columns.tolist()}")
            logging.info(f"suggest_hedge: Holdings types: {holdings['holding_type'].unique() if 'holding_type' in holdings.columns else 'N/A'}")
            
            equity_count = len(holdings[holdings['holding_type'] == 'EQUITY']) if 'holding_type' in holdings.columns else len(holdings)
            mf_count = len(holdings[holdings['holding_type'] == 'MF']) if 'holding_type' in holdings.columns and 'MF' in holdings['holding_type'].values else 0
            equity_value = float(holdings[holdings['holding_type'] == 'EQUITY']['current_value'].sum()) if 'holding_type' in holdings.columns and 'EQUITY' in holdings['holding_type'].values else float(holdings['current_value'].sum())
            mf_value = float(holdings[holdings['holding_type'] == 'MF']['current_value'].sum()) if 'holding_type' in holdings.columns and 'MF' in holdings['holding_type'].values else 0
            
            logging.info(f"suggest_hedge: Calculated counts - Equity: {equity_count}, MF: {mf_count}, Total: {len(holdings)}")
            logging.info(f"suggest_hedge: Calculated values - Equity: ₹{equity_value:,.2f}, MF: ₹{mf_value:,.2f}, Total: ₹{equity_value + mf_value:,.2f}")
            
            # Update diagnostics with latest values and strategy counts
            diagnostics.update({
                'holdings_count': len(holdings),
                'equity_count': equity_count,
                'mf_count': mf_count,
                'portfolio_value': float(portfolio_value),
                'equity_value': equity_value,
                'mf_value': mf_value,
                'beta': float(beta),
                'correlation': float(correlation),
                'nifty_price': float(nifty_price),
                'futures_strategies': len([s for s in all_suggestions if s.get('strategy_type') == 'FUTURES_SHORT']),
                'puts_strategies': len([s for s in all_suggestions if s.get('strategy_type') == 'PROTECTIVE_PUT']),
                'calls_strategies': len([s for s in all_suggestions if s.get('strategy_type') == 'COVERED_CALL']),
                'collars_strategies': len([s for s in all_suggestions if s.get('strategy_type') == 'COLLAR'])
            })
            
            # Generate diagnostic message if no strategies found
            diagnostic_message = None
            if len(all_suggestions) == 0:
                issues = []
                if len(holdings) == 0:
                    issues.append("No holdings found in portfolio")
                if beta_result.get('error'):
                    issues.append(f"Beta calculation error: {beta_result['error']}")
                if not nifty_price:
                    issues.append("Cannot determine Nifty price")
                if diagnostics['expiries_available'] == 0:
                    issues.append("No options expiry dates available (options data may still be loading)")
                
                if all_strategies or any(t in strategy_types for t in ['puts', 'calls', 'collars', 'put', 'call', 'collar', 'protective', 'covered']):
                    if diagnostics['expiries_available'] == 0:
                        issues.append("Options data required but not available - wait for KiteFetchOptions to complete")
                    else:
                        issues.append("No options found matching liquidity criteria (min volume: 100, min OI: 10000)")
                
                diagnostic_message = "; ".join(issues) if issues else "Unknown issue"
            
            result = {
                'strategies': all_suggestions,
                'total_strategies': len(all_suggestions),
                'portfolio_value': float(portfolio_value),
                'portfolio_beta': float(beta),
                'portfolio_correlation': float(correlation),
                'target_hedge_ratio': target_hedge_ratio,
                'total_hedge_value': float(total_hedge_value),
                'total_premium_cost': float(total_premium_cost),
                'total_premium_income': float(total_premium_income),
                'net_hedging_cost': float(total_premium_cost - total_premium_income),
                'total_margin_required': float(total_margin),
                'hedge_effectiveness_score': float(correlation * 100) if correlation else 0.0,
                'nifty_current_price': float(nifty_price),
                'diagnostics': diagnostics
            }
            
            if diagnostic_message:
                result['diagnostic_message'] = diagnostic_message
            
            # Persist suggestions history
            try:
                self._save_suggestions_history(all_suggestions, diagnostics, analysis_date=datetime.now().strftime('%Y-%m-%d'))
            except Exception as e:
                logging.error(f"Failed to persist suggestions history: {e}")

            return result
            
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Error suggesting hedge: {error_msg}")
            import traceback
            logging.error(traceback.format_exc())
            # Ensure error message is safe (no % formatting issues)
            if '%' in error_msg:
                error_msg = error_msg.replace('%', '%%')
            
            # Try to get diagnostics even in error case
            try:
                holdings = self.get_current_holdings()
                equity_count = len(holdings[holdings['holding_type'] == 'EQUITY']) if not holdings.empty and 'holding_type' in holdings.columns else 0
                mf_count = len(holdings[holdings['holding_type'] == 'MF']) if not holdings.empty and 'holding_type' in holdings.columns and 'MF' in holdings['holding_type'].values else 0
                equity_value = float(holdings[holdings['holding_type'] == 'EQUITY']['current_value'].sum()) if not holdings.empty and 'holding_type' in holdings.columns and 'EQUITY' in holdings['holding_type'].values else 0.0
                mf_value = float(holdings[holdings['holding_type'] == 'MF']['current_value'].sum()) if not holdings.empty and 'holding_type' in holdings.columns and 'MF' in holdings['holding_type'].values else 0.0
            except:
                equity_count = mf_count = 0
                equity_value = mf_value = 0.0
            
            return {
                'strategies': [],
                'total_strategies': 0,
                'total_hedge_value': 0.0,
                'error': error_msg,
                'diagnostics': {
                    'holdings_count': equity_count + mf_count,
                    'equity_count': equity_count,
                    'mf_count': mf_count,
                    'portfolio_value': float(equity_value + mf_value),
                    'equity_value': float(equity_value),
                    'mf_value': float(mf_value),
                    'beta': 0.0,
                    'correlation': 0.0,
                    'nifty_price': 0.0,
                    'expiries_available': len(self.options_fetcher.get_expiry_dates()) if hasattr(self, 'options_fetcher') else 0,
                    'futures_strategies': 0,
                    'puts_strategies': 0,
                    'calls_strategies': 0,
                    'collars_strategies': 0
                }
            }
    
    def calculate_var(self, confidence_level: float = 0.95, time_horizon: int = 1) -> Dict:
        """
        Calculate Value at Risk (VaR) for portfolio
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            time_horizon: Time horizon in days
            
        Returns:
            Dictionary with VaR metrics
        """
        try:
            holdings = self.get_current_holdings()
            if holdings.empty:
                return {
                    'var_95': 0.0,
                    'var_99': 0.0,
                    'error': 'No holdings found'
                }
            
            portfolio_value = float(holdings['current_value'].sum())
            
            # Calculate portfolio volatility (simplified approach)
            beta_result = self.calculate_portfolio_beta()
            portfolio_vol = beta_result.get('portfolio_volatility', 0.02)  # Default 2% daily
            
            if portfolio_vol == 0:
                portfolio_vol = 0.02  # Default assumption
            
            # VaR calculation (simplified)
            # VaR = Portfolio Value * Volatility * Z-score * sqrt(time_horizon)
            z_score_95 = 1.645  # 95% confidence
            z_score_99 = 2.326  # 99% confidence
            
            var_95 = portfolio_value * portfolio_vol * z_score_95 * np.sqrt(time_horizon)
            var_99 = portfolio_value * portfolio_vol * z_score_99 * np.sqrt(time_horizon)
            
            return {
                'var_95': float(var_95),
                'var_99': float(var_99),
                'portfolio_value': float(portfolio_value),
                'portfolio_volatility': float(portfolio_vol),
                'confidence_level': confidence_level,
                'time_horizon': time_horizon
            }
            
        except Exception as e:
            logging.error(f"Error calculating VaR: {e}")
            return {
                'var_95': 0.0,
                'var_99': 0.0,
                'error': str(e)
            }

    def _save_suggestions_history(self, suggestions: List[Dict], diagnostics: Dict, analysis_date: Optional[str] = None) -> None:
        """Persist generated derivative suggestions to database for history/audit."""
        try:
            if not suggestions:
                return
            conn = get_db_connection()
            cursor = conn.cursor()
            rows = []
            for s in suggestions:
                expiry_val = s.get('expiry')
                if isinstance(expiry_val, str):
                    try:
                        expiry_dt = datetime.strptime(expiry_val, '%Y-%m-%d').date()
                    except Exception:
                        expiry_dt = None
                elif isinstance(expiry_val, date):
                    expiry_dt = expiry_val
                else:
                    expiry_dt = None

                rows.append((
                    (datetime.now().date() if analysis_date is None else (datetime.strptime(analysis_date, '%Y-%m-%d').date() if isinstance(analysis_date, str) else analysis_date)),
                    'TPO',
                    s.get('strategy_type'),
                    s.get('strategy_name') or s.get('strategy_type'),
                    s.get('instrument'),
                    s.get('instrument_token'),
                    s.get('direction'),
                    s.get('quantity'),
                    s.get('lot_size'),
                    s.get('entry_price'),
                    s.get('strike_price'),
                    expiry_dt,
                    s.get('total_premium'),
                    s.get('total_premium_income'),
                    s.get('margin_required') or s.get('estimated_margin'),
                    s.get('hedge_value'),
                    s.get('coverage_percentage'),
                    float(diagnostics.get('portfolio_value') or 0.0),
                    float(diagnostics.get('beta') or 0.0),
                    s.get('rationale'),
                    json.dumps(s.get('tpo_levels_used') or {}),
                    json.dumps(diagnostics or {})
                ))

            if rows:
                cursor.executemany(
                    """
                    INSERT INTO my_schema.derivative_suggestions (
                        analysis_date, source, strategy_type, strategy_name, instrument, instrument_token,
                        direction, quantity, lot_size, entry_price, strike_price, expiry, total_premium,
                        total_premium_income, margin_required, hedge_value, coverage_percentage,
                        portfolio_value, beta, rationale, tpo_context, diagnostics
                    ) VALUES (
                        %s,%s,%s,%s,%s,%s,
                        %s,%s,%s,%s,%s,%s,%s,
                        %s,%s,%s,%s,
                        %s,%s,%s,%s,%s
                    )
                    """,
                    rows
                )
                conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            logging.error(f"Error saving derivative suggestions history: {e}")

