"""
Combined Sentiment Calculator
Combines news-based and fundamental-based sentiment scores
"""

import logging
from typing import Dict, Optional
from datetime import date
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from common.Boilerplate import get_db_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CombinedSentimentCalculator:
    """
    Combines news-based and fundamental-based sentiment scores
    """
    
    def __init__(self, news_weight: float = 0.5, fundamental_weight: float = 0.5):
        """
        Initialize with weights for news and fundamental sentiment
        
        Args:
            news_weight: Weight for news sentiment (default 0.5)
            fundamental_weight: Weight for fundamental sentiment (default 0.5)
        """
        # Normalize weights to sum to 1.0
        total_weight = news_weight + fundamental_weight
        if total_weight > 0:
            self.news_weight = news_weight / total_weight
            self.fundamental_weight = fundamental_weight / total_weight
        else:
            self.news_weight = 0.5
            self.fundamental_weight = 0.5
    
    def calculate_combined_sentiment(
        self, 
        scrip_id: str, 
        calculation_date: Optional[date] = None,
        news_sentiment_score: Optional[float] = None,
        fundamental_sentiment_score: Optional[float] = None
    ) -> Dict:
        """
        Calculate combined sentiment score from news and fundamental sentiment
        
        Args:
            scrip_id: Stock symbol
            calculation_date: Date for calculation (defaults to today)
            news_sentiment_score: News sentiment score (if None, fetches from DB)
            fundamental_sentiment_score: Fundamental sentiment score (if None, fetches from DB)
            
        Returns:
            Dictionary with combined sentiment scores and metadata
        """
        if calculation_date is None:
            calculation_date = date.today()
        
        # Get news sentiment if not provided
        if news_sentiment_score is None:
            news_sentiment_score = self._get_latest_news_sentiment(scrip_id)
        
        # Get fundamental sentiment if not provided
        if fundamental_sentiment_score is None:
            fundamental_sentiment_score = self._get_latest_fundamental_sentiment(scrip_id)
        
        # Calculate weighted average
        combined_score = (
            (news_sentiment_score or 0.0) * self.news_weight +
            (fundamental_sentiment_score or 0.0) * self.fundamental_weight
        )
        
        # Clamp to -1 to 1 range
        combined_score = max(-1.0, min(1.0, combined_score))
        
        # Prepare metadata
        metadata = {
            'news_weight': self.news_weight,
            'fundamental_weight': self.fundamental_weight,
            'news_sentiment_score': news_sentiment_score,
            'fundamental_sentiment_score': fundamental_sentiment_score,
            'calculation_date': calculation_date.isoformat()
        }
        
        result = {
            'scrip_id': scrip_id,
            'calculation_date': calculation_date,
            'news_sentiment_score': news_sentiment_score,
            'fundamental_sentiment_score': fundamental_sentiment_score,
            'combined_sentiment_score': round(combined_score, 4),
            'news_weight': self.news_weight,
            'fundamental_weight': self.fundamental_weight,
            'metadata': metadata
        }
        
        # Save to database
        self.save_combined_sentiment(result)
        
        return result
    
    def save_combined_sentiment(self, sentiment_data: Dict) -> bool:
        """
        Save combined sentiment to database
        
        Args:
            sentiment_data: Dictionary with sentiment scores
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            metadata_json = json.dumps(sentiment_data.get('metadata', {}))
            
            cursor.execute("""
                INSERT INTO my_schema.combined_sentiment (
                    scrip_id, calculation_date, news_sentiment_score,
                    fundamental_sentiment_score, combined_sentiment_score,
                    news_weight, fundamental_weight, metadata
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s::jsonb
                )
                ON CONFLICT (scrip_id, calculation_date) 
                DO UPDATE SET
                    news_sentiment_score = EXCLUDED.news_sentiment_score,
                    fundamental_sentiment_score = EXCLUDED.fundamental_sentiment_score,
                    combined_sentiment_score = EXCLUDED.combined_sentiment_score,
                    news_weight = EXCLUDED.news_weight,
                    fundamental_weight = EXCLUDED.fundamental_weight,
                    metadata = EXCLUDED.metadata
            """, (
                sentiment_data['scrip_id'],
                sentiment_data['calculation_date'],
                sentiment_data.get('news_sentiment_score'),
                sentiment_data.get('fundamental_sentiment_score'),
                sentiment_data.get('combined_sentiment_score'),
                sentiment_data.get('news_weight'),
                sentiment_data.get('fundamental_weight'),
                metadata_json
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"Saved combined sentiment for {sentiment_data['scrip_id']}: {sentiment_data.get('combined_sentiment_score', 0):.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving combined sentiment for {sentiment_data.get('scrip_id', 'unknown')}: {e}")
            if conn:
                conn.rollback()
                conn.close()
            return False
    
    def _get_latest_news_sentiment(self, scrip_id: str) -> Optional[float]:
        """
        Get latest aggregate news sentiment score from database
        
        Args:
            scrip_id: Stock symbol
            
        Returns:
            Aggregate sentiment score (-1 to 1) or None
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Calculate weighted average of recent news sentiment
            cursor.execute("""
                SELECT 
                    AVG(sentiment_score) as avg_sentiment,
                    COUNT(*) as article_count
                FROM my_schema.news_sentiment
                WHERE scrip_id = %s
                AND article_date >= CURRENT_DATE - INTERVAL '7 days'
            """, (scrip_id,))
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result and result[0] is not None and result[1] > 0:
                return float(result[0])
            
            return None
            
        except Exception as e:
            logger.warning(f"Error getting news sentiment for {scrip_id}: {e}")
            return None
    
    def _get_latest_fundamental_sentiment(self, scrip_id: str) -> Optional[float]:
        """
        Get latest fundamental sentiment score
        
        Note: This should be calculated by FundamentalSentimentAnalyzer
        For now, we'll check if there's a stored value or calculate it
        
        Args:
            scrip_id: Stock symbol
            
        Returns:
            Fundamental sentiment score (-1 to 1) or None
        """
        # This would typically be calculated by FundamentalSentimentAnalyzer
        # For now, we return None and expect it to be provided or calculated separately
        return None
    
    def get_latest_combined_sentiment(self, scrip_id: str) -> Optional[Dict]:
        """
        Get latest combined sentiment from database
        
        Args:
            scrip_id: Stock symbol
            
        Returns:
            Dictionary with sentiment scores or None
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT 
                    scrip_id, calculation_date, news_sentiment_score,
                    fundamental_sentiment_score, combined_sentiment_score,
                    news_weight, fundamental_weight, metadata
                FROM my_schema.combined_sentiment
                WHERE scrip_id = %s
                ORDER BY calculation_date DESC
                LIMIT 1
            """, (scrip_id,))
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result:
                return dict(result)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting combined sentiment for {scrip_id}: {e}")
            return None

