#!/usr/bin/env python3
"""
Refresh Swing Trade Recommendations
Generates and saves swing trade recommendations to the database
Runs every 30 minutes during market hours (same frequency as InsertOHLC.py)
"""

import sys
import os
import logging
from datetime import date, datetime

# Add app directory to path for imports
app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, app_dir)

# Setup logging - logs directory should be at app root
log_dir = os.path.join(app_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'swing_trades_refresh.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

def refresh_swing_trades():
    """
    Generate and save swing trade recommendations to database
    Uses default filtering criteria: min_gain=10%, max_gain=20%, min_confidence=70%
    """
    try:
        from stocks.SwingTradeScanner import SwingTradeScanner
        from common.Boilerplate import get_db_connection
        import psycopg2.extras
        
        logging.info("=" * 60)
        logging.info("Starting Swing Trade Recommendations Refresh")
        logging.info("=" * 60)
        
        # Initialize database connection
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        today = date.today()
        
        # Check existing recommendations for today (optional - for logging)
        cursor.execute("""
            SELECT COUNT(*) as count, MAX(generated_at) as last_generated
            FROM my_schema.swing_trade_suggestions
            WHERE run_date = %s AND status = 'ACTIVE'
        """, (today,))
        
        existing = cursor.fetchone()
        existing_count = existing['count'] if existing else 0
        last_generated = existing['last_generated'] if existing and existing.get('last_generated') else None
        
        logging.info(f"Existing active recommendations for today: {existing_count}")
        if last_generated:
            logging.info(f"Last generated at: {last_generated}")
        logging.info("Generating new recommendations (will replace existing for today)...")
        
        # Initialize scanner with default criteria
        scanner = SwingTradeScanner(
            min_gain=10.0,
            max_gain=20.0,
            min_confidence=70.0
        )
        
        logging.info("Scanning all stocks for swing trade opportunities...")
        
        # Scan all stocks (use default limit or all stocks)
        # Using 100 as limit to balance between coverage and performance
        recommendations = scanner.scan_all_stocks(limit=100)
        
        logging.info(f"Found {len(recommendations)} recommendations from stock scan")
        
        # Also scan Nifty50 and add to recommendations
        logging.info("Scanning Nifty50 stocks...")
        nifty_recommendations = scanner.scan_nifty()
        if nifty_recommendations:
            logging.info(f"Found {len(nifty_recommendations)} Nifty50 recommendations")
            recommendations.extend(nifty_recommendations)
        else:
            logging.info("No Nifty50 recommendations found")
        
        # Get Prophet predictions and enrich recommendations
        logging.info("Enriching recommendations with Prophet predictions...")
        try:
            cursor.execute("""
                SELECT scrip_id, predicted_price_change_pct, prediction_confidence, prediction_days, run_date
                FROM my_schema.prophet_predictions
                WHERE run_date = (SELECT MAX(run_date) FROM my_schema.prophet_predictions WHERE status = 'ACTIVE' AND prediction_days = 30)
                AND prediction_days = 30
                AND status = 'ACTIVE'
            """)
            
            predictions_rows = cursor.fetchall()
            
            # If no 30-day predictions found, get latest predictions
            if not predictions_rows:
                cursor.execute("""
                    SELECT scrip_id, predicted_price_change_pct, prediction_confidence, prediction_days, run_date
                    FROM my_schema.prophet_predictions pp1
                    WHERE status = 'ACTIVE'
                    AND run_date = (SELECT MAX(run_date) FROM my_schema.prophet_predictions WHERE status = 'ACTIVE')
                """)
                predictions_rows = cursor.fetchall()
            
            # Build predictions map
            predictions_map = {}
            if predictions_rows:
                for row in predictions_rows:
                    scrip_id = row['scrip_id'].upper()
                    if scrip_id not in predictions_map:
                        predictions_map[scrip_id] = dict(row)
                    else:
                        # Prefer 30-day predictions
                        existing = predictions_map[scrip_id]
                        existing_days = existing.get('prediction_days', 0)
                        row_days = row.get('prediction_days', 0)
                        if row_days == 30 or (existing_days != 30 and row_days > existing_days):
                            predictions_map[scrip_id] = dict(row)
            
            logging.info(f"Found {len(predictions_map)} Prophet predictions")
            
            # Add predictions to recommendations
            matched_count = 0
            for rec in recommendations:
                scrip_id = rec.get('scrip_id')
                if scrip_id:
                    scrip_id_upper = scrip_id.upper()
                    if scrip_id_upper in predictions_map:
                        pred = predictions_map[scrip_id_upper]
                        
                        # Convert and add prediction data
                        try:
                            pred_pct = pred.get('predicted_price_change_pct')
                            if pred_pct is not None and not (isinstance(pred_pct, float) and (pred_pct != pred_pct)):
                                rec['prophet_prediction_pct'] = float(pred_pct)
                            else:
                                rec['prophet_prediction_pct'] = None
                        except (ValueError, TypeError):
                            rec['prophet_prediction_pct'] = None
                        
                        try:
                            pred_conf = pred.get('prediction_confidence')
                            if pred_conf is not None and not (isinstance(pred_conf, float) and (pred_conf != pred_conf)):
                                rec['prophet_confidence'] = float(pred_conf)
                            else:
                                rec['prophet_confidence'] = None
                        except (ValueError, TypeError):
                            rec['prophet_confidence'] = None
                        
                        pred_days = pred.get('prediction_days')
                        rec['prediction_days'] = int(pred_days) if pred_days is not None else None
                        
                        if rec['prophet_prediction_pct'] is not None:
                            matched_count += 1
                    else:
                        rec['prophet_prediction_pct'] = None
                        rec['prophet_confidence'] = None
                        rec['prediction_days'] = None
                else:
                    rec['prophet_prediction_pct'] = None
                    rec['prophet_confidence'] = None
                    rec['prediction_days'] = None
            
            logging.info(f"Matched {matched_count} recommendations with Prophet predictions")
            
        except Exception as e:
            logging.warning(f"Error enriching with Prophet predictions: {e}")
            # Continue without predictions
            for rec in recommendations:
                rec['prophet_prediction_pct'] = None
                rec['prophet_confidence'] = None
                rec['prediction_days'] = None
        
        # Sort by confidence score (highest first)
        recommendations.sort(key=lambda x: x.get('confidence_score', 0), reverse=True)
        
        # Close cursor and connection before saving (save_recommendations opens its own connection)
        cursor.close()
        conn.close()
        
        # Save to database
        logging.info(f"Saving {len(recommendations)} recommendations to database...")
        success = scanner.save_recommendations(recommendations, today)
        
        if success:
            logging.info(f"✓ Successfully saved {len(recommendations)} swing trade recommendations")
            
            # Log summary statistics
            if recommendations:
                avg_confidence = sum(r.get('confidence_score', 0) for r in recommendations) / len(recommendations)
                avg_gain = sum(r.get('potential_gain_pct', 0) for r in recommendations) / len(recommendations)
                pattern_types = {}
                for r in recommendations:
                    ptype = r.get('pattern_type', 'Unknown')
                    pattern_types[ptype] = pattern_types.get(ptype, 0) + 1
                
                logging.info(f"Summary Statistics:")
                logging.info(f"  - Average Confidence: {avg_confidence:.2f}%")
                logging.info(f"  - Average Potential Gain: {avg_gain:.2f}%")
                logging.info(f"  - Pattern Types: {dict(pattern_types)}")
        else:
            logging.error("Failed to save recommendations to database")
            return False
        
        logging.info("=" * 60)
        logging.info("Swing Trade Recommendations Refresh Completed Successfully")
        logging.info("=" * 60)
        
        return True
        
    except Exception as e:
        logging.error(f"Error refreshing swing trade recommendations: {e}")
        import traceback
        logging.error(traceback.format_exc())
        try:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()
        except:
            pass
        return False

if __name__ == "__main__":
    success = refresh_swing_trades()
    sys.exit(0 if success else 1)

