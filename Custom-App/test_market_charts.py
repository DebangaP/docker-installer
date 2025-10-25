#!/usr/bin/env python3
"""
Test script to debug MarketCharts issues
"""

import sys
import traceback
from MarketCharts import MarketChartsGenerator

def test_market_charts():
    try:
        print("Creating MarketChartsGenerator...")
        DB_CONFIG = {
            'host': 'postgres',
            'database': 'mydb',
            'user': 'postgres',
            'password': 'postgres',
            'port': 5432
        }
        
        gen = MarketChartsGenerator(DB_CONFIG)
        print("✓ MarketChartsGenerator created successfully")
        
        # Test pre-market data
        print("\nTesting pre-market data...")
        pre_market_df = gen.get_pre_market_ticks("2025-10-20")
        print(f"✓ Pre-market data shape: {pre_market_df.shape}")
        
        # Test market data
        print("\nTesting market data...")
        market_df = gen.get_market_ticks("2025-10-20")
        print(f"✓ Market data shape: {market_df.shape}")
        
        # Test trades data
        print("\nTesting trades data...")
        trades_df = gen.get_latest_trades()
        print(f"✓ Trades data shape: {trades_df.shape}")
        
        # Test margin data
        print("\nTesting margin data...")
        margin_df = gen.get_margin_data()
        print(f"✓ Margin data shape: {margin_df.shape}")
        
        # Test holdings data
        print("\nTesting holdings data...")
        holdings_df = gen.get_holdings_data()
        print(f"✓ Holdings data shape: {holdings_df.shape}")
        
        # Test positions data
        print("\nTesting positions data...")
        positions_df = gen.get_positions_data()
        print(f"✓ Positions data shape: {positions_df.shape}")
        
        # Test gainers/losers data
        print("\nTesting gainers/losers data...")
        gainers_df = gen.get_gainers_losers()
        print(f"✓ Gainers data shape: {gainers_df.shape}")
        if not gainers_df.empty:
            print("Sample gainers data:")
            print(gainers_df.head())
        
        # Test dashboard generation
        print("\nTesting dashboard generation...")
        dashboard_image = gen.generate_market_dashboard("2025-10-20")
        print(f"✓ Dashboard generated successfully, image length: {len(dashboard_image)}")
        
        # Test gainers/losers chart generation
        print("\nTesting gainers/losers chart generation...")
        gainers_image = gen.generate_gainers_losers_chart()
        print(f"✓ Gainers chart generated successfully, image length: {len(gainers_image)}")
        
        print("\n🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_market_charts()
    sys.exit(0 if success else 1)
