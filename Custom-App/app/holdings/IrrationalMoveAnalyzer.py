"""
Irrational Move Analyzer for Holdings
Identifies stocks with irrational gains or losses that may need exit consideration
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import json
from common.Boilerplate import get_db_connection
from common.TechnicalIndicators import calculate_rsi, calculate_macd
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IrrationalMoveAnalyzer:
    """
    Analyzes holdings for irrational price movements
    """
    
    def __init__(self):
        """Initialize the analyzer"""
        self.conn = None
        self.cursor = None
        self.nifty_symbol = 'NIFTY50'
        
    def _get_connection(self):
        """Get database connection"""
        if self.conn is None or self.conn.closed:
            self.conn = get_db_connection()
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        return self.conn, self.cursor
    
    def analyze_all_holdings(self, analysis_date: date = None) -> Dict:
        """
        Main entry point to analyze all holdings
        
        Args:
            analysis_date: Date for analysis (default: today)
            
        Returns:
            Dictionary with analysis results
        """
        if analysis_date is None:
            analysis_date = date.today()
        
        try:
            conn, cursor = self._get_connection()
            
            # Get current holdings
            holdings = self._get_current_holdings(cursor)
            if not holdings:
                logger.warning("No holdings found for analysis")
                return {"success": False, "message": "No holdings found"}
            
            # Calculate portfolio statistics
            portfolio_stats = self.calculate_portfolio_stats(holdings)
            
            # Get market data (Nifty50)
            market_data = self._get_market_data(cursor, analysis_date)
            
            results = []
            for holding in holdings:
                try:
                    analysis = self._analyze_holding(holding, portfolio_stats, market_data, cursor)
                    if analysis:
                        results.append(analysis)
                        # Save to database
                        self.save_analysis(analysis, cursor)
                except Exception as e:
                    logger.error(f"Error analyzing {holding.get('trading_symbol')}: {e}")
                    continue
            
            conn.commit()
            
            return {
                "success": True,
                "analysis_date": str(analysis_date),
                "total_holdings": len(holdings),
                "analyzed": len(results),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error in analyze_all_holdings: {e}")
            import traceback
            logger.error(traceback.format_exc())
            if self.conn:
                self.conn.rollback()
            return {"success": False, "error": str(e)}
    
    def _get_current_holdings(self, cursor) -> List[Dict]:
        """Get current holdings from database"""
        cursor.execute("""
            SELECT 
                h.trading_symbol,
                h.instrument_token,
                h.quantity,
                h.average_price,
                h.last_price,
                h.pnl,
                CASE 
                    WHEN (h.quantity * h.average_price) != 0 
                    THEN (h.pnl / (h.quantity * h.average_price)) * 100
                    ELSE 0
                END as pnl_pct_change
            FROM my_schema.holdings h
            WHERE h.run_date = (SELECT MAX(run_date) FROM my_schema.holdings)
        """)
        return [dict(row) for row in cursor.fetchall()]
    
    def _get_market_data(self, cursor, analysis_date: date) -> Dict:
        """Get Nifty50 market data for comparison"""
        try:
            # Get today's Nifty50 price
            cursor.execute("""
                SELECT price_close, price_date
                FROM my_schema.rt_intraday_price
                WHERE scrip_id = 'NIFTY50'
                AND price_date::date = %s
                ORDER BY price_date DESC
                LIMIT 1
            """, (analysis_date,))
            today_data = cursor.fetchone()
            
            # Get previous day's Nifty50 price
            cursor.execute("""
                SELECT price_close, price_date
                FROM my_schema.rt_intraday_price
                WHERE scrip_id = 'NIFTY50'
                AND price_date::date < %s
                ORDER BY price_date DESC
                LIMIT 1
            """, (analysis_date,))
            prev_data = cursor.fetchone()
            
            if today_data and prev_data:
                today_price = float(today_data['price_close'])
                prev_price = float(prev_data['price_close'])
                market_change_pct = ((today_price - prev_price) / prev_price) * 100
                
                return {
                    "today_price": today_price,
                    "prev_price": prev_price,
                    "change_pct": market_change_pct
                }
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
        
        return {"today_price": None, "prev_price": None, "change_pct": 0.0}
    
    def calculate_portfolio_stats(self, holdings: List[Dict]) -> Dict:
        """
        Calculate portfolio average and standard deviation
        
        Args:
            holdings: List of holding dictionaries
            
        Returns:
            Dictionary with portfolio statistics
        """
        if not holdings:
            return {"avg_pnl": 0.0, "std_dev": 0.0, "pnl_list": []}
        
        pnl_list = []
        for holding in holdings:
            pnl_pct = holding.get('pnl_pct_change', 0.0)
            if pnl_pct is not None:
                pnl_list.append(float(pnl_pct))
        
        if not pnl_list:
            return {"avg_pnl": 0.0, "std_dev": 0.0, "pnl_list": []}
        
        avg_pnl = np.mean(pnl_list)
        std_dev = np.std(pnl_list) if len(pnl_list) > 1 else 0.0
        
        return {
            "avg_pnl": float(avg_pnl),
            "std_dev": float(std_dev),
            "pnl_list": pnl_list
        }
    
    def check_statistical_outlier(self, holding_pnl_pct: float, portfolio_stats: Dict) -> Tuple[bool, float]:
        """
        Check if holding is a statistical outlier using z-score
        
        Args:
            holding_pnl_pct: Holding's P&L percentage
            portfolio_stats: Portfolio statistics dictionary
            
        Returns:
            Tuple of (is_outlier, z_score)
        """
        portfolio_mean = portfolio_stats.get('avg_pnl', 0.0)
        portfolio_std = portfolio_stats.get('std_dev', 0.0)
        
        if portfolio_std == 0:
            return False, 0.0
        
        z_score = (holding_pnl_pct - portfolio_mean) / portfolio_std
        is_outlier = abs(z_score) > 2.0
        
        return is_outlier, float(z_score)
    
    def check_market_correlation(self, stock_change_pct: float, market_data: Dict, 
                                  trading_symbol: str, cursor) -> Tuple[bool, str, float]:
        """
        Check if stock movement contradicts market movement
        
        Args:
            stock_change_pct: Stock's percentage change
            market_data: Market data dictionary
            trading_symbol: Stock symbol
            cursor: Database cursor
            
        Returns:
            Tuple of (is_mismatch, reason, correlation_score)
        """
        market_change_pct = market_data.get('change_pct', 0.0)
        
        # Check for obvious mismatch
        if stock_change_pct > 5.0 and market_change_pct < -1.0:
            return True, "Stock up while market down", 0.0
        elif stock_change_pct < -5.0 and market_change_pct > 1.0:
            return True, "Stock down while market up", 0.0
        
        # Calculate correlation over last 20 days
        try:
            correlation = self._calculate_correlation(trading_symbol, cursor)
            if correlation < -0.3 and abs(stock_change_pct) > 3.0:
                return True, f"Negative correlation ({correlation:.2f}) with large move", correlation
        except Exception as e:
            logger.debug(f"Could not calculate correlation for {trading_symbol}: {e}")
        
        return False, None, 0.0
    
    def _calculate_correlation(self, trading_symbol: str, cursor) -> float:
        """Calculate correlation between stock and Nifty50 over last 20 days"""
        try:
            cursor.execute("""
                WITH stock_prices AS (
                    SELECT price_date, price_close
                    FROM my_schema.rt_intraday_price
                    WHERE scrip_id = %s
                    AND price_date >= CURRENT_DATE - INTERVAL '20 days'
                    ORDER BY price_date
                ),
                nifty_prices AS (
                    SELECT price_date, price_close
                    FROM my_schema.rt_intraday_price
                    WHERE scrip_id = 'NIFTY50'
                    AND price_date >= CURRENT_DATE - INTERVAL '20 days'
                    ORDER BY price_date
                )
                SELECT 
                    s.price_close as stock_price,
                    n.price_close as nifty_price
                FROM stock_prices s
                JOIN nifty_prices n ON s.price_date = n.price_date
                ORDER BY s.price_date
            """, (trading_symbol,))
            
            data = cursor.fetchall()
            if len(data) < 10:
                return 0.0
            
            stock_prices = [float(row['stock_price']) for row in data]
            nifty_prices = [float(row['nifty_price']) for row in data]
            
            # Calculate daily returns
            stock_returns = np.diff(stock_prices) / stock_prices[:-1]
            nifty_returns = np.diff(nifty_prices) / nifty_prices[:-1]
            
            if len(stock_returns) < 2 or len(nifty_returns) < 2:
                return 0.0
            
            correlation = np.corrcoef(stock_returns, nifty_returns)[0, 1]
            return float(correlation) if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            logger.debug(f"Error calculating correlation: {e}")
            return 0.0
    
    def check_technical_indicators(self, trading_symbol: str, pnl_pct: float, 
                                   cursor) -> Tuple[bool, Dict]:
        """
        Check technical indicators for mismatches
        
        Args:
            trading_symbol: Stock symbol
            pnl_pct: P&L percentage
            cursor: Database cursor
            
        Returns:
            Tuple of (is_mismatch, indicators_dict)
        """
        indicators = {
            "rsi": None,
            "macd_divergence": False,
            "volume_anomaly": False,
            "volume_ratio": 1.0
        }
        
        try:
            # Get price data for last 30 days
            cursor.execute("""
                SELECT price_close, volume, price_date
                FROM my_schema.rt_intraday_price
                WHERE scrip_id = %s
                AND price_date >= CURRENT_DATE - INTERVAL '30 days'
                ORDER BY price_date
            """, (trading_symbol,))
            
            price_data = cursor.fetchall()
            if len(price_data) < 14:
                return False, indicators
            
            # Convert to DataFrame
            df = pd.DataFrame([{
                'close': float(row['price_close']),
                'volume': float(row['volume']) if row['volume'] else 0.0,
                'date': row['price_date']
            } for row in price_data])
            
            # Calculate RSI
            if len(df) >= 15:
                rsi_series = calculate_rsi(df['close'], period=14)
                indicators['rsi'] = float(rsi_series.iloc[-1]) if not rsi_series.isna().iloc[-1] else None
            
            # Check volume anomaly
            if len(df) >= 20:
                recent_volume = df['volume'].tail(5).mean()
                avg_volume = df['volume'].mean()
                if avg_volume > 0:
                    volume_ratio = recent_volume / avg_volume
                    indicators['volume_ratio'] = float(volume_ratio)
                    indicators['volume_anomaly'] = volume_ratio > 2.0 and abs(pnl_pct) > 5.0
            
            # Check MACD divergence (simplified)
            if len(df) >= 26:
                macd_data = calculate_macd(df['close'], 12, 26, 9)
                macd_line = macd_data['macd']
                signal_line = macd_data['signal']
                
                if not macd_line.isna().iloc[-1] and not signal_line.isna().iloc[-1]:
                    current_macd = float(macd_line.iloc[-1])
                    current_signal = float(signal_line.iloc[-1])
                    prev_macd = float(macd_line.iloc[-2]) if len(macd_line) > 1 else current_macd
                    
                    # Check for divergence: price up but MACD down, or vice versa
                    price_trend = df['close'].iloc[-1] > df['close'].iloc[-5] if len(df) >= 5 else False
                    macd_trend = current_macd > prev_macd
                    
                    if price_trend != macd_trend and abs(pnl_pct) > 5.0:
                        indicators['macd_divergence'] = True
            
            # Check for technical mismatch
            is_mismatch = False
            if indicators['rsi']:
                if indicators['rsi'] > 70 and pnl_pct > 10.0:
                    is_mismatch = True
                elif indicators['rsi'] < 30 and pnl_pct < -10.0:
                    is_mismatch = True
            
            if indicators['volume_anomaly']:
                is_mismatch = True
            
            if indicators['macd_divergence']:
                is_mismatch = True
            
            return is_mismatch, indicators
            
        except Exception as e:
            logger.error(f"Error checking technical indicators for {trading_symbol}: {e}")
            return False, indicators
    
    def check_time_anomalies(self, trading_symbol: str, cursor) -> Tuple[bool, Dict]:
        """
        Check for time-based anomalies (sudden moves, velocity)
        
        Args:
            trading_symbol: Stock symbol
            cursor: Database cursor
            
        Returns:
            Tuple of (is_anomaly, time_data_dict)
        """
        time_data = {
            "days_since_large_move": 0,
            "move_velocity": 0.0
        }
        
        try:
            # Get last 5 days of price data
            cursor.execute("""
                SELECT price_close, price_date
                FROM my_schema.rt_intraday_price
                WHERE scrip_id = %s
                AND price_date >= CURRENT_DATE - INTERVAL '5 days'
                ORDER BY price_date
            """, (trading_symbol,))
            
            price_data = cursor.fetchall()
            if len(price_data) < 2:
                return False, time_data
            
            prices = [float(row['price_close']) for row in price_data]
            dates = [row['price_date'] for row in price_data]
            
            # Calculate daily changes
            daily_changes = []
            for i in range(1, len(prices)):
                if prices[i-1] > 0:
                    change_pct = ((prices[i] - prices[i-1]) / prices[i-1]) * 100
                    daily_changes.append(abs(change_pct))
            
            # Check for sudden large move (>5% in single day)
            is_anomaly = False
            days_since_large = 0
            for i, change in enumerate(daily_changes):
                if change > 5.0:
                    is_anomaly = True
                    days_since_large = len(daily_changes) - i
                    break
            
            # Calculate velocity (average % change per day)
            if len(daily_changes) > 0:
                move_velocity = np.mean(daily_changes)
                time_data['move_velocity'] = float(move_velocity)
                
                # High velocity with large cumulative move
                if move_velocity > 3.0 and len(prices) > 1:
                    total_move = abs(((prices[-1] - prices[0]) / prices[0]) * 100)
                    if total_move > 10.0:
                        is_anomaly = True
            
            time_data['days_since_large_move'] = days_since_large
            
            return is_anomaly, time_data
            
        except Exception as e:
            logger.error(f"Error checking time anomalies for {trading_symbol}: {e}")
            return False, time_data
    
    def check_fundamental_mismatch(self, trading_symbol: str, pnl_pct: float, 
                                    cursor) -> Tuple[bool, Dict]:
        """
        Check for fundamental mismatches
        
        Args:
            trading_symbol: Stock symbol
            pnl_pct: P&L percentage
            cursor: Database cursor
            
        Returns:
            Tuple of (is_mismatch, fundamental_data_dict)
        """
        fundamental_data = {
            "pe_ratio": None,
            "prophet_prediction_diff": None,
            "news_sentiment_diff": None
        }
        
        try:
            # Get PE ratio from fundamental data
            cursor.execute("""
                SELECT pe_ratio
                FROM my_schema.fundamental_data fd
                WHERE fd.scrip_id = %s
                ORDER BY fd.fetch_date DESC
                LIMIT 1
            """, (trading_symbol,))
            pe_result = cursor.fetchone()
            if pe_result and pe_result['pe_ratio']:
                fundamental_data['pe_ratio'] = float(pe_result['pe_ratio'])
            
            # Get Prophet prediction
            cursor.execute("""
                SELECT predicted_price_change_pct
                FROM my_schema.prophet_predictions
                WHERE scrip_id = %s
                AND status = 'ACTIVE'
                ORDER BY run_date DESC
                LIMIT 1
            """, (trading_symbol,))
            prophet_result = cursor.fetchone()
            if prophet_result and prophet_result['predicted_price_change_pct']:
                predicted = float(prophet_result['predicted_price_change_pct'])
                fundamental_data['prophet_prediction_diff'] = abs(pnl_pct - predicted)
            
            # Get news sentiment
            cursor.execute("""
                SELECT combined_sentiment_score
                FROM my_schema.combined_sentiment
                WHERE scrip_id = %s
                ORDER BY calculation_date DESC
                LIMIT 1
            """, (trading_symbol,))
            sentiment_result = cursor.fetchone()
            if sentiment_result and sentiment_result['combined_sentiment_score']:
                sentiment = float(sentiment_result['combined_sentiment_score'])
                # If sentiment is negative but price is up, or vice versa
                if (sentiment < -0.1 and pnl_pct > 5.0) or (sentiment > 0.1 and pnl_pct < -5.0):
                    fundamental_data['news_sentiment_diff'] = abs(sentiment - (pnl_pct / 100))
            
            # Check for mismatch
            is_mismatch = False
            
            # Extreme PE with large move
            if fundamental_data['pe_ratio']:
                if (fundamental_data['pe_ratio'] > 50 or fundamental_data['pe_ratio'] < 5) and abs(pnl_pct) > 5.0:
                    is_mismatch = True
            
            # Prophet prediction mismatch
            if fundamental_data['prophet_prediction_diff'] and fundamental_data['prophet_prediction_diff'] > 10.0:
                is_mismatch = True
            
            # News sentiment mismatch
            if fundamental_data['news_sentiment_diff'] and fundamental_data['news_sentiment_diff'] > 0.15:
                is_mismatch = True
            
            return is_mismatch, fundamental_data
            
        except Exception as e:
            logger.debug(f"Error checking fundamental mismatch for {trading_symbol}: {e}")
            return False, fundamental_data
    
    def calculate_irrational_score(self, flags: Dict) -> float:
        """
        Calculate overall irrational score (0-100)
        
        Args:
            flags: Dictionary with all analysis flags
            
        Returns:
            Irrational score (0-100)
        """
        score = 0.0
        
        # Statistical outlier: 25 points
        if flags.get('is_statistical_outlier', False):
            z_score = abs(flags.get('z_score', 0.0))
            score += 25.0 * min(z_score / 3.0, 1.0)
        
        # Market mismatch: 20 points
        if flags.get('is_market_mismatch', False):
            score += 20.0
        
        # Technical mismatch: 20 points
        if flags.get('is_technical_mismatch', False):
            score += 20.0
        
        # Time anomaly: 15 points
        if flags.get('is_time_anomaly', False):
            score += 15.0
        
        # Fundamental mismatch: 20 points
        if flags.get('is_fundamental_mismatch', False):
            score += 20.0
        
        return min(score, 100.0)
    
    def generate_exit_recommendation(self, irrational_score: float, pnl_pct: float, 
                                     flags: Dict) -> Tuple[str, str]:
        """
        Generate exit recommendation based on score
        
        Args:
            irrational_score: Calculated irrational score
            pnl_pct: P&L percentage
            flags: Analysis flags
            
        Returns:
            Tuple of (recommendation, reason)
        """
        if irrational_score >= 70:
            reasons = []
            if flags.get('is_statistical_outlier'):
                reasons.append("Statistical outlier")
            if flags.get('is_market_mismatch'):
                reasons.append("Market correlation mismatch")
            if flags.get('is_technical_mismatch'):
                reasons.append("Technical indicator mismatch")
            if flags.get('is_time_anomaly'):
                reasons.append("Time-based anomaly")
            if flags.get('is_fundamental_mismatch'):
                reasons.append("Fundamental mismatch")
            
            reason = "Multiple indicators suggest irrational move: " + ", ".join(reasons)
            return "STRONG", reason
            
        elif irrational_score >= 50:
            return "MODERATE", "Some indicators suggest review needed"
        elif irrational_score >= 30:
            return "WEAK", "Minor anomalies detected"
        else:
            return "NONE", "No significant irrational patterns"
    
    def _analyze_holding(self, holding: Dict, portfolio_stats: Dict, 
                         market_data: Dict, cursor) -> Optional[Dict]:
        """
        Analyze a single holding
        
        Args:
            holding: Holding dictionary
            portfolio_stats: Portfolio statistics
            market_data: Market data
            cursor: Database cursor
            
        Returns:
            Analysis dictionary or None
        """
        trading_symbol = holding.get('trading_symbol')
        pnl_pct = holding.get('pnl_pct_change', 0.0) or 0.0
        
        if pnl_pct == 0.0:
            return None
        
        # Run all checks
        is_outlier, z_score = self.check_statistical_outlier(pnl_pct, portfolio_stats)
        is_market_mismatch, market_reason, correlation = self.check_market_correlation(
            pnl_pct, market_data, trading_symbol, cursor
        )
        is_technical_mismatch, technical_data = self.check_technical_indicators(
            trading_symbol, pnl_pct, cursor
        )
        is_time_anomaly, time_data = self.check_time_anomalies(trading_symbol, cursor)
        is_fundamental_mismatch, fundamental_data = self.check_fundamental_mismatch(
            trading_symbol, pnl_pct, cursor
        )
        
        # Compile flags
        flags = {
            'is_statistical_outlier': is_outlier,
            'z_score': z_score,
            'is_market_mismatch': is_market_mismatch,
            'is_technical_mismatch': is_technical_mismatch,
            'is_time_anomaly': is_time_anomaly,
            'is_fundamental_mismatch': is_fundamental_mismatch
        }
        
        # Calculate score
        irrational_score = self.calculate_irrational_score(flags)
        
        # Generate recommendation
        exit_recommendation, exit_reason = self.generate_exit_recommendation(
            irrational_score, pnl_pct, flags
        )
        
        # Determine type
        irrational_type = 'gain' if pnl_pct > 0 else 'loss'
        
        # Compile analysis details
        analysis_details = {
            'flags': flags,
            'technical_data': technical_data,
            'time_data': time_data,
            'fundamental_data': fundamental_data,
            'market_reason': market_reason,
            'correlation': correlation
        }
        
        return {
            'trading_symbol': trading_symbol,
            'instrument_token': holding.get('instrument_token'),
            'analysis_date': date.today(),
            'pnl_pct_change': pnl_pct,
            'today_pnl_pct': pnl_pct,  # Can be enhanced with today's specific P&L
            'current_price': holding.get('last_price'),
            'average_price': holding.get('average_price'),
            'is_statistical_outlier': is_outlier,
            'z_score': z_score,
            'portfolio_avg_pnl': portfolio_stats.get('avg_pnl'),
            'portfolio_std_dev': portfolio_stats.get('std_dev'),
            'is_market_mismatch': is_market_mismatch,
            'market_change_pct': market_data.get('change_pct'),
            'correlation_score': correlation,
            'is_technical_mismatch': is_technical_mismatch,
            'rsi': technical_data.get('rsi'),
            'macd_divergence': technical_data.get('macd_divergence'),
            'volume_anomaly': technical_data.get('volume_anomaly'),
            'volume_ratio': technical_data.get('volume_ratio'),
            'is_time_anomaly': is_time_anomaly,
            'days_since_large_move': time_data.get('days_since_large_move'),
            'move_velocity': time_data.get('move_velocity'),
            'is_fundamental_mismatch': is_fundamental_mismatch,
            'pe_ratio': fundamental_data.get('pe_ratio'),
            'prophet_prediction_diff': fundamental_data.get('prophet_prediction_diff'),
            'news_sentiment_diff': fundamental_data.get('news_sentiment_diff'),
            'irrational_score': irrational_score,
            'irrational_type': irrational_type,
            'exit_recommendation': exit_recommendation,
            'exit_reason': exit_reason,
            'analysis_details': analysis_details
        }
    
    def save_analysis(self, analysis: Dict, cursor):
        """
        Save analysis results to database
        
        Args:
            analysis: Analysis dictionary
            cursor: Database cursor
        """
        try:
            cursor.execute("""
                INSERT INTO my_schema.holdings_irrational_analysis (
                    trading_symbol, instrument_token, analysis_date, run_date,
                    pnl_pct_change, today_pnl_pct, current_price, average_price,
                    is_statistical_outlier, z_score, portfolio_avg_pnl, portfolio_std_dev,
                    is_market_mismatch, market_change_pct, correlation_score,
                    is_technical_mismatch, rsi, macd_divergence, volume_anomaly, volume_ratio,
                    is_time_anomaly, days_since_large_move, move_velocity,
                    is_fundamental_mismatch, pe_ratio, prophet_prediction_diff, news_sentiment_diff,
                    irrational_score, irrational_type, exit_recommendation, exit_reason,
                    analysis_details
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (trading_symbol, analysis_date) 
                DO UPDATE SET
                    pnl_pct_change = EXCLUDED.pnl_pct_change,
                    today_pnl_pct = EXCLUDED.today_pnl_pct,
                    current_price = EXCLUDED.current_price,
                    average_price = EXCLUDED.average_price,
                    is_statistical_outlier = EXCLUDED.is_statistical_outlier,
                    z_score = EXCLUDED.z_score,
                    portfolio_avg_pnl = EXCLUDED.portfolio_avg_pnl,
                    portfolio_std_dev = EXCLUDED.portfolio_std_dev,
                    is_market_mismatch = EXCLUDED.is_market_mismatch,
                    market_change_pct = EXCLUDED.market_change_pct,
                    correlation_score = EXCLUDED.correlation_score,
                    is_technical_mismatch = EXCLUDED.is_technical_mismatch,
                    rsi = EXCLUDED.rsi,
                    macd_divergence = EXCLUDED.macd_divergence,
                    volume_anomaly = EXCLUDED.volume_anomaly,
                    volume_ratio = EXCLUDED.volume_ratio,
                    is_time_anomaly = EXCLUDED.is_time_anomaly,
                    days_since_large_move = EXCLUDED.days_since_large_move,
                    move_velocity = EXCLUDED.move_velocity,
                    is_fundamental_mismatch = EXCLUDED.is_fundamental_mismatch,
                    pe_ratio = EXCLUDED.pe_ratio,
                    prophet_prediction_diff = EXCLUDED.prophet_prediction_diff,
                    news_sentiment_diff = EXCLUDED.news_sentiment_diff,
                    irrational_score = EXCLUDED.irrational_score,
                    irrational_type = EXCLUDED.irrational_type,
                    exit_recommendation = EXCLUDED.exit_recommendation,
                    exit_reason = EXCLUDED.exit_reason,
                    analysis_details = EXCLUDED.analysis_details
            """, (
                analysis['trading_symbol'],
                analysis['instrument_token'],
                analysis['analysis_date'],
                date.today(),
                analysis['pnl_pct_change'],
                analysis['today_pnl_pct'],
                analysis['current_price'],
                analysis['average_price'],
                analysis['is_statistical_outlier'],
                analysis['z_score'],
                analysis['portfolio_avg_pnl'],
                analysis['portfolio_std_dev'],
                analysis['is_market_mismatch'],
                analysis['market_change_pct'],
                analysis['correlation_score'],
                analysis['is_technical_mismatch'],
                analysis['rsi'],
                analysis['macd_divergence'],
                analysis['volume_anomaly'],
                analysis['volume_ratio'],
                analysis['is_time_anomaly'],
                analysis['days_since_large_move'],
                analysis['move_velocity'],
                analysis['is_fundamental_mismatch'],
                analysis['pe_ratio'],
                analysis['prophet_prediction_diff'],
                analysis['news_sentiment_diff'],
                analysis['irrational_score'],
                analysis['irrational_type'],
                analysis['exit_recommendation'],
                analysis['exit_reason'],
                json.dumps(analysis['analysis_details'])
            ))
        except Exception as e:
            logger.error(f"Error saving analysis for {analysis.get('trading_symbol')}: {e}")
            raise
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

