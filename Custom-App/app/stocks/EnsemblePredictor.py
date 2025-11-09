"""
Ensemble Predictor
Combines Prophet predictions with sentiment scores using weighted ensemble
"""

import logging
from typing import Dict, Optional
from datetime import date
import psycopg2
from psycopg2.extras import RealDictCursor
from common.Boilerplate import get_db_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Combines Prophet predictions with sentiment scores using weighted ensemble
    """
    
    def __init__(self, prophet_weight: float = 0.7, sentiment_weight: float = 0.3):
        """
        Initialize with weights for Prophet and sentiment
        
        Args:
            prophet_weight: Weight for Prophet prediction (default 0.7)
            sentiment_weight: Weight for sentiment score (default 0.3)
        """
        # Normalize weights to sum to 1.0
        total_weight = prophet_weight + sentiment_weight
        if total_weight > 0:
            self.prophet_weight = prophet_weight / total_weight
            self.sentiment_weight = sentiment_weight / total_weight
        else:
            self.prophet_weight = 0.7
            self.sentiment_weight = 0.3
    
    def enhance_prediction(
        self,
        scrip_id: str,
        prophet_predicted_change_pct: float,
        prophet_confidence: float,
        combined_sentiment_score: Optional[float] = None,
        run_date: Optional[date] = None
    ) -> Dict:
        """
        Enhance Prophet prediction with sentiment score
        
        Args:
            scrip_id: Stock symbol
            prophet_predicted_change_pct: Prophet predicted price change percentage
            prophet_confidence: Prophet prediction confidence (0-100)
            combined_sentiment_score: Combined sentiment score (-1 to 1, if None fetches from DB)
            run_date: Run date for the prediction
            
        Returns:
            Dictionary with enhanced prediction and metadata
        """
        if run_date is None:
            run_date = date.today()
        
        # Get sentiment score if not provided
        if combined_sentiment_score is None:
            combined_sentiment_score = self._get_latest_combined_sentiment(scrip_id, run_date)
        
        # Convert sentiment score (-1 to 1) to price change adjustment
        # Positive sentiment -> upward adjustment, negative -> downward adjustment
        sentiment_adjustment = combined_sentiment_score * 5.0  # Scale: -5% to +5% adjustment
        
        # Calculate enhanced prediction
        # Base prediction from Prophet
        base_prediction = prophet_predicted_change_pct
        
        # Sentiment adjustment weighted by sentiment weight
        sentiment_contribution = sentiment_adjustment * self.sentiment_weight
        
        # Enhanced prediction combines Prophet and sentiment
        enhanced_prediction = base_prediction + sentiment_contribution
        
        # Adjust confidence based on sentiment alignment
        # If sentiment aligns with prediction direction, increase confidence
        # If sentiment contradicts prediction, decrease confidence
        sentiment_alignment = 1.0
        if prophet_predicted_change_pct > 0 and combined_sentiment_score > 0:
            # Both positive - increase confidence
            sentiment_alignment = 1.1
        elif prophet_predicted_change_pct < 0 and combined_sentiment_score < 0:
            # Both negative - increase confidence
            sentiment_alignment = 1.1
        elif prophet_predicted_change_pct > 0 and combined_sentiment_score < -0.3:
            # Contradiction - decrease confidence
            sentiment_alignment = 0.9
        elif prophet_predicted_change_pct < 0 and combined_sentiment_score > 0.3:
            # Contradiction - decrease confidence
            sentiment_alignment = 0.9
        
        # Enhanced confidence
        enhanced_confidence = prophet_confidence * (
            self.prophet_weight + (self.sentiment_weight * sentiment_alignment)
        )
        
        # Clamp confidence to 0-100 range
        enhanced_confidence = max(0.0, min(100.0, enhanced_confidence))
        
        # Prepare metadata
        metadata = {
            'prophet_weight': self.prophet_weight,
            'sentiment_weight': self.sentiment_weight,
            'prophet_predicted_change_pct': prophet_predicted_change_pct,
            'prophet_confidence': prophet_confidence,
            'combined_sentiment_score': combined_sentiment_score,
            'sentiment_adjustment': sentiment_adjustment,
            'sentiment_contribution': sentiment_contribution,
            'sentiment_alignment': sentiment_alignment,
            'run_date': run_date.isoformat()
        }
        
        result = {
            'scrip_id': scrip_id,
            'run_date': run_date,
            'prophet_predicted_change_pct': prophet_predicted_change_pct,
            'prophet_confidence': prophet_confidence,
            'combined_sentiment_score': combined_sentiment_score,
            'enhanced_predicted_price_change_pct': round(enhanced_prediction, 4),
            'enhanced_prediction_confidence': round(enhanced_confidence, 2),
            'metadata': metadata
        }
        
        return result
    
    def _get_latest_combined_sentiment(self, scrip_id: str, run_date: date) -> float:
        """
        Get latest combined sentiment score from database
        
        Args:
            scrip_id: Stock symbol
            run_date: Run date for the prediction
            
        Returns:
            Combined sentiment score (-1 to 1), defaults to 0.0 if not found
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get latest combined sentiment (within last 7 days)
            cursor.execute("""
                SELECT combined_sentiment_score
                FROM my_schema.combined_sentiment
                WHERE scrip_id = %s
                AND calculation_date >= %s - INTERVAL '7 days'
                ORDER BY calculation_date DESC
                LIMIT 1
            """, (scrip_id, run_date))
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result and result[0] is not None:
                return float(result[0])
            
            # Default to neutral if no sentiment found
            return 0.0
            
        except Exception as e:
            logger.warning(f"Error getting combined sentiment for {scrip_id}: {e}")
            return 0.0
    
    def update_prophet_prediction_with_sentiment(
        self,
        scrip_id: str,
        run_date: date,
        prediction_days: int,
        enhanced_prediction: Dict
    ) -> bool:
        """
        Update prophet_predictions table with enhanced prediction
        
        Args:
            scrip_id: Stock symbol
            run_date: Run date
            prediction_days: Prediction days
            enhanced_prediction: Dictionary with enhanced prediction data
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get sentiment metadata
            metadata = enhanced_prediction.get('metadata', {})
            sentiment_metadata = {
                'prophet_weight': metadata.get('prophet_weight'),
                'sentiment_weight': metadata.get('sentiment_weight'),
                'sentiment_adjustment': metadata.get('sentiment_adjustment'),
                'sentiment_contribution': metadata.get('sentiment_contribution'),
                'sentiment_alignment': metadata.get('sentiment_alignment')
            }
            
            # Get news and fundamental sentiment scores
            news_sentiment = self._get_news_sentiment_score(scrip_id, run_date)
            fundamental_sentiment = self._get_fundamental_sentiment_score(scrip_id, run_date)
            
            import json
            sentiment_metadata_json = json.dumps(sentiment_metadata)
            
            cursor.execute("""
                UPDATE my_schema.prophet_predictions
                SET 
                    news_sentiment_score = %s,
                    fundamental_sentiment_score = %s,
                    combined_sentiment_score = %s,
                    enhanced_predicted_price_change_pct = %s,
                    enhanced_prediction_confidence = %s,
                    sentiment_metadata = %s::jsonb
                WHERE scrip_id = %s
                AND run_date = %s
                AND prediction_days = %s
            """, (
                news_sentiment,
                fundamental_sentiment,
                enhanced_prediction.get('combined_sentiment_score'),
                enhanced_prediction.get('enhanced_predicted_price_change_pct'),
                enhanced_prediction.get('enhanced_prediction_confidence'),
                sentiment_metadata_json,
                scrip_id,
                run_date,
                prediction_days
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"Updated enhanced prediction for {scrip_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating enhanced prediction for {scrip_id}: {e}")
            if conn:
                conn.rollback()
                conn.close()
            return False
    
    def _get_news_sentiment_score(self, scrip_id: str, run_date: date) -> Optional[float]:
        """Get news sentiment score"""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT AVG(sentiment_score) as avg_sentiment
                FROM my_schema.news_sentiment
                WHERE scrip_id = %s
                AND article_date >= %s - INTERVAL '7 days'
            """, (scrip_id, run_date))
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result and result[0] is not None:
                return float(result[0])
            
            return None
            
        except Exception as e:
            logger.warning(f"Error getting news sentiment for {scrip_id}: {e}")
            return None
    
    def _get_fundamental_sentiment_score(self, scrip_id: str, run_date: date) -> Optional[float]:
        """Get fundamental sentiment score from combined_sentiment table"""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT fundamental_sentiment_score
                FROM my_schema.combined_sentiment
                WHERE scrip_id = %s
                AND calculation_date >= %s - INTERVAL '7 days'
                ORDER BY calculation_date DESC
                LIMIT 1
            """, (scrip_id, run_date))
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result and result[0] is not None:
                return float(result[0])
            
            return None
            
        except Exception as e:
            logger.warning(f"Error getting fundamental sentiment for {scrip_id}: {e}")
            return None

