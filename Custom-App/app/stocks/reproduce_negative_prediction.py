
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import logging

# Add app directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock modules that might be missing
sys.modules['common.Boilerplate'] = MagicMock()
sys.modules['sentiment.NewsSentimentAnalyzer'] = MagicMock()
sys.modules['sentiment.FundamentalSentimentAnalyzer'] = MagicMock()
sys.modules['sentiment.CombinedSentimentCalculator'] = MagicMock()
sys.modules['stocks.EnsemblePredictor'] = MagicMock()

# Now import the class under test
from stocks.ProphetPricePredictor import ProphetPricePredictor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dropping_price_data(days=100):
    """Create price data with a sharp drop at the end"""
    dates = [datetime.now() - timedelta(days=x) for x in range(days)]
    dates.reverse()
    
    # Stable price then drop
    prices = [100.0] * (days - 10)
    # Sharp drop
    for i in range(10):
        prices.append(100.0 - (i * 10)) # Drops to 10
        
    df = pd.DataFrame({
        'ds': dates,
        'y': prices
    })
    return df

def create_negative_trend_data(days=100):
    """Create price data with a strong negative trend"""
    dates = [datetime.now() - timedelta(days=x) for x in range(days)]
    dates.reverse()
    
    # Linear drop
    prices = [max(10, 200 - (x * 2)) for x in range(days)]
    
    df = pd.DataFrame({
        'ds': dates,
        'y': prices
    })
    return df

def test_negative_prediction():
    print("Testing negative prediction scenario...")
    
    predictor = ProphetPricePredictor(prediction_days=30, enable_sentiment=False)
    
    # Mock get_price_data
    # Scenario 1: Sharp drop
    df_drop = create_dropping_price_data()
    
    with patch.object(predictor, 'get_price_data', return_value=df_drop):
        print("\n--- Scenario 1: Sharp Drop ---")
        result = predictor.predict_price('TEST_DROP')
        if result:
            print(f"Current Price: {result['current_price']}")
            print(f"Predicted Price: {result['predicted_price']}")
            print(f"Daily Predictions (first 5):")
            for p in result['daily_predictions'][:5]:
                print(f"  {p['date']}: {p['predicted_price']}")
            print(f"Daily Predictions (last 5):")
            for p in result['daily_predictions'][-5:]:
                print(f"  {p['date']}: {p['predicted_price']}")
        else:
            print("Prediction failed (returned None)")

    # Scenario 2: Strong negative trend
    df_trend = create_negative_trend_data()
    
    with patch.object(predictor, 'get_price_data', return_value=df_trend):
        print("\n--- Scenario 2: Strong Negative Trend ---")
        result = predictor.predict_price('TEST_TREND')
        if result:
            print(f"Current Price: {result['current_price']}")
            print(f"Predicted Price: {result['predicted_price']}")
            print(f"Daily Predictions (last 5):")
            for p in result['daily_predictions'][-5:]:
                print(f"  {p['date']}: {p['predicted_price']}")
        else:
            print("Prediction failed (returned None)")

if __name__ == "__main__":
    test_negative_prediction()
