#!/usr/bin/env python3
"""
Example usage of Market Bias Analyzer
Demonstrates how to use the MarketBiasAnalyzer class for generating market bias and key levels
"""

import sys
import os
# Add app directory to path for imports
app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, app_dir)

from market.MarketBiasAnalyzer import MarketBiasAnalyzer, PostgresDataFetcher
from market.CalculateTPO import TPOProfile
import json
from datetime import datetime

def main():
    """
    Main function demonstrating market bias analysis usage
    """
    print("=" * 80)
    print("MARKET BIAS ANALYZER - EXAMPLE USAGE")
    print("=" * 80)
    
    # Database configuration
    DB_CONFIG = {
        'host': 'postgres',
        'database': 'mydb',
        'user': 'postgres',
        'password': 'postgres',
        'port': 5432
    }
    
    try:
        # Initialize components
        print("Initializing database connection...")
        db_fetcher = PostgresDataFetcher(**DB_CONFIG)
        
        print("Initializing Market Bias Analyzer...")
        bias_analyzer = MarketBiasAnalyzer(
            db_fetcher=db_fetcher,
            instrument_token=256265,  # Nifty 50
            tick_size=5.0
        )
        
        # Example 1: Analyze current date (live mode)
        print("\n" + "=" * 60)
        print("EXAMPLE 1: LIVE MARKET ANALYSIS")
        print("=" * 60)
        
        current_date = datetime.now().strftime('%Y-%m-%d')
        print(f"Analyzing market for: {current_date}")
        
        live_analysis = bias_analyzer.generate_comprehensive_analysis(current_date)
        
        print("\n--- LIVE MARKET BIAS ANALYSIS ---")
        print(f"Analysis Timestamp: {live_analysis['analysis_timestamp']}")
        print(f"Analysis Date: {live_analysis['analysis_date']}")
        
        # Display comprehensive bias
        comprehensive_bias = live_analysis['comprehensive_bias']
        print(f"\nComprehensive Bias Score: {comprehensive_bias['bias_score']:.2f}")
        print(f"Bias Direction: {comprehensive_bias['bias_direction']}")
        print(f"Bias Strength: {comprehensive_bias['bias_strength']}")
        
        # Display bias factors
        print("\nBias Factors:")
        for factor in comprehensive_bias['bias_factors']:
            print(f"  - {factor}")
        
        # Display key levels
        print("\n--- KEY LEVELS ---")
        key_levels = live_analysis['key_levels']
        
        if key_levels['pivot_points']:
            print("Pivot Points:")
            for level_type, level_data in key_levels['pivot_points'].items():
                print(f"  - {level_type.upper()}: {level_data['price']:.2f}")
        
        if key_levels['value_area_levels']:
            print("Value Area Levels:")
            for level_type, level_data in key_levels['value_area_levels'].items():
                print(f"  - {level_type.upper()}: {level_data['price']:.2f}")
        
        if key_levels['support_levels']:
            print("Support Levels:")
            for level in key_levels['support_levels']:
                print(f"  - {level['price']:.2f} (Strength: {level['strength']})")
        
        if key_levels['resistance_levels']:
            print("Resistance Levels:")
            for level in key_levels['resistance_levels']:
                print(f"  - {level['price']:.2f} (Strength: {level['strength']})")
        
        # Display market structure
        print("\n--- MARKET STRUCTURE ---")
        market_structure = live_analysis['market_structure']
        print(f"Day Type: {market_structure['day_type']}")
        print(f"Distribution Type: {market_structure['distribution_type']}")
        print(f"Session Range: {market_structure['session_range']:.2f}")
        print(f"Value Area Range: {market_structure['value_area_range']:.2f}")
        print(f"POC Strength: {market_structure['poc_strength']:.1f}%")
        
        # Display trading recommendations
        print("\n--- TRADING RECOMMENDATIONS ---")
        recommendations = live_analysis['trading_recommendations']
        print(f"Primary Bias: {recommendations['primary_bias']}")
        print(f"Risk Level: {recommendations['risk_level']}")
        print(f"Position Sizing: {recommendations['position_sizing']}")
        
        if recommendations['key_levels_to_watch']:
            print("Key Levels to Watch:")
            for level in recommendations['key_levels_to_watch']:
                print(f"  - {level['level']:.2f} ({level['type']}) - {level['importance']}")
        
        if recommendations['entry_signals']:
            print("Entry Signals:")
            for signal in recommendations['entry_signals']:
                print(f"  - {signal['signal']}: {signal['condition']} (Confidence: {signal['confidence']})")
        
        if recommendations['exit_signals']:
            print("Exit Signals:")
            for signal in recommendations['exit_signals']:
                print(f"  - {signal['signal']}: {signal['condition']} (Confidence: {signal['confidence']})")
        
        # Example 2: Analyze specific historical date
        print("\n" + "=" * 60)
        print("EXAMPLE 2: HISTORICAL ANALYSIS")
        print("=" * 60)
        
        historical_date = "2025-01-20"  # Example date
        print(f"Analyzing historical market for: {historical_date}")
        
        historical_analysis = bias_analyzer.generate_comprehensive_analysis(historical_date)
        
        print(f"\nHistorical Bias Score: {historical_analysis['comprehensive_bias']['bias_score']:.2f}")
        print(f"Historical Bias Direction: {historical_analysis['comprehensive_bias']['bias_direction']}")
        print(f"Historical Bias Strength: {historical_analysis['comprehensive_bias']['bias_strength']}")
        
        # Example 3: Individual analysis components
        print("\n" + "=" * 60)
        print("EXAMPLE 3: INDIVIDUAL ANALYSIS COMPONENTS")
        print("=" * 60)
        
        # Pre-market analysis only
        print("Pre-market Analysis:")
        pre_market_analysis = bias_analyzer.analyze_pre_market_bias(current_date)
        print(f"  Bias Score: {pre_market_analysis['bias_score']:.2f}")
        print(f"  Bias Direction: {pre_market_analysis['bias_direction']}")
        print(f"  Bias Strength: {pre_market_analysis['bias_strength']}")
        
        # Real-time analysis only
        print("\nReal-time Analysis:")
        real_time_analysis = bias_analyzer.analyze_real_time_bias(current_date)
        print(f"  Bias Score: {real_time_analysis['bias_score']:.2f}")
        print(f"  Bias Direction: {real_time_analysis['bias_direction']}")
        print(f"  Bias Strength: {real_time_analysis['bias_strength']}")
        
        # Example 4: Generate visualization
        print("\n" + "=" * 60)
        print("EXAMPLE 4: GENERATING VISUALIZATION")
        print("=" * 60)
        
        print("Generating market bias visualization...")
        plot_image = bias_analyzer.plot_bias_analysis(live_analysis)
        
        print(f"Plot generated successfully!")
        print(f"Base64 image length: {len(plot_image)} characters")
        print("Image can be displayed in web browsers or saved to file.")
        
        # Example 5: Save analysis to JSON
        print("\n" + "=" * 60)
        print("EXAMPLE 5: SAVING ANALYSIS TO JSON")
        print("=" * 60)
        
        # Save comprehensive analysis to JSON file
        output_file = f"market_bias_analysis_{current_date}.json"
        with open(output_file, 'w') as f:
            json.dump(live_analysis, f, indent=2, default=str)
        
        print(f"Analysis saved to: {output_file}")
        
        # Example 6: API-like usage
        print("\n" + "=" * 60)
        print("EXAMPLE 6: API-LIKE USAGE")
        print("=" * 60)
        
        # Simulate API response format
        api_response = {
            "analysis_date": current_date,
            "is_historical": False,
            "analysis": live_analysis
        }
        
        print("API Response Format:")
        print(f"  Analysis Date: {api_response['analysis_date']}")
        print(f"  Is Historical: {api_response['is_historical']}")
        print(f"  Bias Score: {api_response['analysis']['comprehensive_bias']['bias_score']:.2f}")
        print(f"  Bias Direction: {api_response['analysis']['comprehensive_bias']['bias_direction']}")
        
        print("\n" + "=" * 80)
        print("MARKET BIAS ANALYZER EXAMPLE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Clean up database connection
        if 'db_fetcher' in locals():
            db_fetcher.close()
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
