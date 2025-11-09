"""
Fundamental Sentiment Analyzer
Analyzes fundamental metrics to generate sentiment scores
"""

import logging
from typing import Dict, List, Optional
from datetime import date
import psycopg2
from psycopg2.extras import RealDictCursor
from common.Boilerplate import get_db_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FundamentalSentimentAnalyzer:
    """
    Analyzes fundamental metrics to generate sentiment scores
    """
    
    def __init__(self):
        pass
    
    def calculate_fundamental_sentiment(self, scrip_id: str, fundamental_data: Optional[Dict] = None) -> float:
        """
        Calculate fundamental sentiment score based on financial metrics
        
        Args:
            scrip_id: Stock symbol
            fundamental_data: Dictionary with fundamental metrics (if None, fetches from DB)
            
        Returns:
            Sentiment score (-1 to 1)
        """
        if fundamental_data is None:
            fundamental_data = self._get_latest_fundamental_data(scrip_id)
        
        if not fundamental_data:
            logger.warning(f"No fundamental data available for {scrip_id}")
            return 0.0
        
        # Calculate financial health score
        health_score = self.score_financial_health(fundamental_data)
        
        # Compare with sector averages
        sector_comparison = self.compare_with_sector_averages(scrip_id, fundamental_data)
        
        # Combine scores
        sentiment_score = self._combine_scores(health_score, sector_comparison)
        
        # Normalize to -1 to 1 range
        return self.normalize_sentiment_score(sentiment_score)
    
    def compare_with_sector_averages(self, scrip_id: str, metrics: Dict) -> Dict:
        """
        Compare fundamental metrics with sector averages
        
        Args:
            scrip_id: Stock symbol
            metrics: Dictionary with fundamental metrics
            
        Returns:
            Dictionary with comparison scores
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get sector code
            cursor.execute("""
                SELECT sector_code 
                FROM my_schema.master_scrips 
                WHERE scrip_id = %s
            """, (scrip_id,))
            
            result = cursor.fetchone()
            if not result or not result[0]:
                cursor.close()
                conn.close()
                return {'score': 0.0, 'comparisons': {}}
            
            sector_code = result[0]
            
            # Calculate sector averages
            cursor.execute("""
                SELECT 
                    AVG(pe_ratio) as avg_pe,
                    AVG(pb_ratio) as avg_pb,
                    AVG(debt_to_equity) as avg_debt_equity,
                    AVG(roe) as avg_roe,
                    AVG(roce) as avg_roce,
                    AVG(current_ratio) as avg_current_ratio,
                    AVG(revenue_growth) as avg_revenue_growth,
                    AVG(profit_growth) as avg_profit_growth
                FROM my_schema.fundamental_data fd
                JOIN my_schema.master_scrips ms ON fd.scrip_id = ms.scrip_id
                WHERE ms.sector_code = %s
                AND fd.fetch_date >= CURRENT_DATE - INTERVAL '30 days'
            """, (sector_code,))
            
            sector_avg = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if not sector_avg:
                return {'score': 0.0, 'comparisons': {}}
            
            comparisons = {}
            score_sum = 0.0
            count = 0
            
            # Compare P/E ratio (lower is better, but not too low)
            if metrics.get('pe_ratio') and sector_avg[0]:
                pe_ratio = metrics['pe_ratio']
                avg_pe = sector_avg[0]
                if avg_pe > 0:
                    pe_ratio_norm = pe_ratio / avg_pe
                    if 0.8 <= pe_ratio_norm <= 1.2:  # Close to sector average
                        pe_score = 0.5
                    elif pe_ratio_norm < 0.8:  # Lower than average (good)
                        pe_score = 0.7
                    else:  # Higher than average (potentially overvalued)
                        pe_score = 0.3
                    comparisons['pe_ratio'] = pe_score
                    score_sum += pe_score
                    count += 1
            
            # Compare P/B ratio (lower is generally better)
            if metrics.get('pb_ratio') and sector_avg[1]:
                pb_ratio = metrics['pb_ratio']
                avg_pb = sector_avg[1]
                if avg_pb > 0:
                    pb_ratio_norm = pb_ratio / avg_pb
                    if pb_ratio_norm < 0.9:  # Lower than average (good)
                        pb_score = 0.7
                    elif pb_ratio_norm <= 1.1:  # Close to average
                        pb_score = 0.5
                    else:  # Higher than average
                        pb_score = 0.3
                    comparisons['pb_ratio'] = pb_score
                    score_sum += pb_score
                    count += 1
            
            # Compare ROE (higher is better)
            if metrics.get('roe') and sector_avg[3]:
                roe = metrics['roe']
                avg_roe = sector_avg[3]
                if avg_roe > 0:
                    roe_ratio = roe / avg_roe
                    if roe_ratio > 1.2:  # Higher than average (good)
                        roe_score = 0.8
                    elif roe_ratio >= 0.9:  # Close to or above average
                        roe_score = 0.6
                    else:  # Lower than average
                        roe_score = 0.3
                    comparisons['roe'] = roe_score
                    score_sum += roe_score
                    count += 1
            
            # Compare ROCE (higher is better)
            if metrics.get('roce') and sector_avg[4]:
                roce = metrics['roce']
                avg_roce = sector_avg[4]
                if avg_roce > 0:
                    roce_ratio = roce / avg_roce
                    if roce_ratio > 1.2:  # Higher than average (good)
                        roce_score = 0.8
                    elif roce_ratio >= 0.9:  # Close to or above average
                        roce_score = 0.6
                    else:  # Lower than average
                        roce_score = 0.3
                    comparisons['roce'] = roce_score
                    score_sum += roce_score
                    count += 1
            
            # Compare revenue growth (higher is better)
            if metrics.get('revenue_growth') and sector_avg[6]:
                rev_growth = metrics['revenue_growth']
                avg_rev_growth = sector_avg[6]
                if rev_growth > avg_rev_growth * 1.1:  # Significantly higher
                    rev_score = 0.8
                elif rev_growth >= avg_rev_growth * 0.9:  # Close to or above average
                    rev_score = 0.6
                else:  # Lower than average
                    rev_score = 0.3
                comparisons['revenue_growth'] = rev_score
                score_sum += rev_score
                count += 1
            
            # Compare profit growth (higher is better)
            if metrics.get('profit_growth') and sector_avg[7]:
                profit_growth = metrics['profit_growth']
                avg_profit_growth = sector_avg[7]
                if profit_growth > avg_profit_growth * 1.1:  # Significantly higher
                    profit_score = 0.8
                elif profit_growth >= avg_profit_growth * 0.9:  # Close to or above average
                    profit_score = 0.6
                else:  # Lower than average
                    profit_score = 0.3
                comparisons['profit_growth'] = profit_score
                score_sum += profit_score
                count += 1
            
            avg_score = score_sum / count if count > 0 else 0.0
            
            return {
                'score': avg_score,
                'comparisons': comparisons,
                'sector_code': sector_code
            }
            
        except Exception as e:
            logger.error(f"Error comparing with sector averages for {scrip_id}: {e}")
            return {'score': 0.0, 'comparisons': {}}
    
    def score_financial_health(self, metrics: Dict) -> float:
        """
        Score financial health based on fundamental ratios
        
        Args:
            metrics: Dictionary with fundamental metrics
            
        Returns:
            Health score (0 to 1)
        """
        score_sum = 0.0
        count = 0
        
        # P/E Ratio (lower is better, but not too low - indicates undervaluation or poor prospects)
        pe_ratio = metrics.get('pe_ratio')
        if pe_ratio:
            if 10 <= pe_ratio <= 25:  # Reasonable range
                pe_score = 0.7
            elif 5 <= pe_ratio < 10 or 25 < pe_ratio <= 35:  # Acceptable range
                pe_score = 0.5
            elif pe_ratio < 5:  # Very low (could be undervalued or poor)
                pe_score = 0.3
            else:  # Very high (overvalued)
                pe_score = 0.2
            score_sum += pe_score
            count += 1
        
        # P/B Ratio (lower is generally better)
        pb_ratio = metrics.get('pb_ratio')
        if pb_ratio:
            if pb_ratio < 1:  # Undervalued
                pb_score = 0.8
            elif 1 <= pb_ratio <= 3:  # Reasonable
                pb_score = 0.6
            elif 3 < pb_ratio <= 5:  # High
                pb_score = 0.4
            else:  # Very high
                pb_score = 0.2
            score_sum += pb_score
            count += 1
        
        # Debt to Equity (lower is better)
        debt_equity = metrics.get('debt_to_equity')
        if debt_equity is not None:
            if debt_equity < 0.5:  # Low debt (good)
                debt_score = 0.8
            elif 0.5 <= debt_equity <= 1.0:  # Moderate debt
                debt_score = 0.6
            elif 1.0 < debt_equity <= 2.0:  # High debt
                debt_score = 0.4
            else:  # Very high debt
                debt_score = 0.2
            score_sum += debt_score
            count += 1
        
        # ROE (higher is better)
        roe = metrics.get('roe')
        if roe is not None:
            if roe > 20:  # Excellent
                roe_score = 0.9
            elif 15 <= roe <= 20:  # Good
                roe_score = 0.7
            elif 10 <= roe < 15:  # Average
                roe_score = 0.5
            elif 5 <= roe < 10:  # Below average
                roe_score = 0.3
            else:  # Poor
                roe_score = 0.1
            score_sum += roe_score
            count += 1
        
        # ROCE (higher is better)
        roce = metrics.get('roce')
        if roce is not None:
            if roce > 20:  # Excellent
                roce_score = 0.9
            elif 15 <= roce <= 20:  # Good
                roce_score = 0.7
            elif 10 <= roce < 15:  # Average
                roce_score = 0.5
            elif 5 <= roce < 10:  # Below average
                roce_score = 0.3
            else:  # Poor
                roce_score = 0.1
            score_sum += roce_score
            count += 1
        
        # Current Ratio (liquidity - higher is better, but not too high)
        current_ratio = metrics.get('current_ratio')
        if current_ratio is not None:
            if 1.5 <= current_ratio <= 3.0:  # Healthy range
                current_score = 0.8
            elif 1.0 <= current_ratio < 1.5:  # Acceptable
                current_score = 0.6
            elif current_ratio >= 3.0:  # Too high (inefficient)
                current_score = 0.5
            else:  # Low (liquidity concerns)
                current_score = 0.3
            score_sum += current_score
            count += 1
        
        # Revenue Growth (higher is better)
        revenue_growth = metrics.get('revenue_growth')
        if revenue_growth is not None:
            if revenue_growth > 20:  # Excellent growth
                rev_score = 0.9
            elif 10 <= revenue_growth <= 20:  # Good growth
                rev_score = 0.7
            elif 5 <= revenue_growth < 10:  # Moderate growth
                rev_score = 0.5
            elif 0 <= revenue_growth < 5:  # Slow growth
                rev_score = 0.3
            else:  # Negative growth
                rev_score = 0.1
            score_sum += rev_score
            count += 1
        
        # Profit Growth (higher is better)
        profit_growth = metrics.get('profit_growth')
        if profit_growth is not None:
            if profit_growth > 20:  # Excellent growth
                profit_score = 0.9
            elif 10 <= profit_growth <= 20:  # Good growth
                profit_score = 0.7
            elif 5 <= profit_growth < 10:  # Moderate growth
                profit_score = 0.5
            elif 0 <= profit_growth < 5:  # Slow growth
                profit_score = 0.3
            else:  # Negative growth
                profit_score = 0.1
            score_sum += profit_score
            count += 1
        
        avg_score = score_sum / count if count > 0 else 0.0
        
        return avg_score
    
    def normalize_sentiment_score(self, raw_score: float) -> float:
        """
        Normalize raw score to -1 to 1 range
        
        Args:
            raw_score: Raw score (0 to 1)
            
        Returns:
            Normalized score (-1 to 1)
        """
        # Convert 0-1 range to -1 to 1 range
        # 0.5 (neutral) -> 0.0
        # 1.0 (excellent) -> 1.0
        # 0.0 (poor) -> -1.0
        normalized = (raw_score - 0.5) * 2.0
        
        # Clamp to -1 to 1 range
        return max(-1.0, min(1.0, normalized))
    
    def _combine_scores(self, health_score: float, sector_comparison: Dict) -> float:
        """
        Combine health score and sector comparison
        
        Args:
            health_score: Financial health score (0 to 1)
            sector_comparison: Sector comparison dictionary
            
        Returns:
            Combined score (0 to 1)
        """
        sector_score = sector_comparison.get('score', 0.5)
        
        # Weight: 60% health score, 40% sector comparison
        combined = (health_score * 0.6) + (sector_score * 0.4)
        
        return combined
    
    def _get_latest_fundamental_data(self, scrip_id: str) -> Optional[Dict]:
        """
        Get latest fundamental data from database
        
        Args:
            scrip_id: Stock symbol
            
        Returns:
            Dictionary with fundamental metrics or None
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT 
                    pe_ratio, pb_ratio, debt_to_equity, roe, roce,
                    current_ratio, quick_ratio, eps, revenue_growth,
                    profit_growth, dividend_yield, market_cap
                FROM my_schema.fundamental_data
                WHERE scrip_id = %s
                ORDER BY fetch_date DESC
                LIMIT 1
            """, (scrip_id,))
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result:
                return dict(result)
            return None
            
        except Exception as e:
            logger.error(f"Error getting fundamental data for {scrip_id}: {e}")
            return None

