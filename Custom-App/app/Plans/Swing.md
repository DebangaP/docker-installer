Swing Trading Recommendations System - Implementation Plan
PROJECT OVERVIEW
Building an automated swing trading recommendation system that scans stocks for opportunities targeting 10-20% gains using technical analysis, pattern recognition, and risk management.

OBJECTIVES
Scan all stocks in master_scrips table for swing trade opportunities
Identify patterns that could lead to 10-20% gains
Calculate entry, target, and stop-loss levels
Score opportunities by confidence level
Store all generated recommendations in database with current date
Display recommendations in UI with detailed analysis
DATABASE SCHEMA
New Table: swing_trade_suggestions
Following the pattern of derivative_suggestions table, create:

id: SERIAL PRIMARY KEY
generated_at: TIMESTAMP DEFAULT CURRENT_TIMESTAMP
analysis_date: DATE (current date)
run_date: DATE DEFAULT CURRENT_DATE
scrip_id: VARCHAR(10) - Stock symbol
instrument_token: BIGINT - Instrument token
pattern_type: VARCHAR(50) - Pattern identified (e.g., 'Pullback to Support', 'Breakout', 'Golden Cross')
direction: VARCHAR(10) - 'BUY' or 'SELL'
entry_price: DOUBLE PRECISION - Recommended entry price
target_price: DOUBLE PRECISION - Target price for exit
stop_loss: DOUBLE PRECISION - Stop loss price
potential_gain_pct: DOUBLE PRECISION - Expected gain percentage
risk_reward_ratio: DOUBLE PRECISION - Risk to reward ratio
confidence_score: DOUBLE PRECISION - Confidence score (0-100)
holding_period_days: INT - Expected holding period (3-10 days)
current_price: DOUBLE PRECISION - Current stock price
sma_20: DOUBLE PRECISION - 20-day moving average
sma_50: DOUBLE PRECISION - 50-day moving average
sma_200: DOUBLE PRECISION - 200-day moving average
rsi_14: DOUBLE PRECISION - 14-period RSI
macd: DOUBLE PRECISION - MACD value
macd_signal: DOUBLE PRECISION - MACD signal line
atr_14: DOUBLE PRECISION - 14-period Average True Range
volume_trend: VARCHAR(20) - Volume trend ('increasing', 'decreasing', 'stable')
support_level: DOUBLE PRECISION - Nearest support level
resistance_level: DOUBLE PRECISION - Nearest resistance level
rationale: TEXT - Explanation of the recommendation
technical_context: JSONB - Additional technical indicators
diagnostics: JSONB - Pattern detection diagnostics
status: VARCHAR(20) DEFAULT 'ACTIVE' - Status of recommendation
Indexes
idx_swing_trade_analysis_date ON swing_trade_suggestions(analysis_date)
idx_swing_trade_scrip_id ON swing_trade_suggestions(scrip_id)
idx_swing_trade_pattern ON swing_trade_suggestions(pattern_type)
idx_swing_trade_confidence ON swing_trade_suggestions(confidence_score DESC)
idx_swing_trade_run_date ON swing_trade_suggestions(run_date)
BACKEND MODULE: SwingTradeScanner.py
Location
Custom-App/app/SwingTradeScanner.py

Structure
Following the pattern of OptionsScanner.py:

SwingTradeScanner Class
init: Initialize with database connection
scan_stocks: Main method to scan all stocks
_fetch_price_data: Get last 90 days OHLC data for a stock
_calculate_indicators: Calculate all technical indicators
_detect_patterns: Detect swing trade patterns
_calculate_entry_targets: Calculate entry, target, stop-loss
_calculate_confidence: Score the opportunity
_save_recommendations: Save to database
Technical Indicators to Implement
SMA (20, 50, 200) - Already available via candlestick API
RSI (14-period) - New implementation
MACD (12, 26, 9) - New implementation
ATR (14-period) - New implementation for stop-loss
Volume moving average - New implementation
Support/Resistance detection - Leverage existing MicroLevelDetector
Patterns to Detect
Pullback to Support: Price near support with bullish reversal signals
Breakout from Consolidation: Price breaking above resistance with volume
Golden Cross: SMA 20 crossing above SMA 50
Bullish Divergence: Price lower lows, RSI higher lows
Volume Surge: Unusual volume spike with price movement
Oversold Bounce: RSI oversold (<30) with price at support
Entry/Exit Calculation
Entry: Current price or next pullback level
Target: Next resistance level or Fibonacci extension (10-20% range)
Stop-loss: Below support or ATR-based (typically 5-7%)
Risk-reward: Minimum 2:1 ratio required
Confidence Scoring
Multiple indicators aligning: +20 points each
Volume confirmation: +15 points
Support/resistance strength: +15 points
Trend alignment: +10 points
Pattern clarity: +10 points
Maximum: 100 points
API ENDPOINT
Location
Custom-App/app/KiteAccessToken.py

Endpoint: /api/swing_trades
GET /api/swing_trades
Parameters:

min_gain: float (default: 10.0) - Minimum target gain %
max_gain: float (default: 20.0) - Maximum target gain %
min_confidence: float (default: 70.0) - Minimum confidence score
pattern_type: str (optional) - Filter by pattern type
limit: int (default: 20) - Number of results
analysis_date: str (optional) - Get recommendations for specific date
Response:

success: bool
recommendations: List[Dict] - List of swing trade opportunities
total_found: int - Total opportunities found
analysis_date: str - Date of analysis
Endpoint: /api/swing_trades_history
GET /api/swing_trades_history
Parameters:

start_date: str (optional) - Start date filter
end_date: str (optional) - End date filter
scrip_id: str (optional) - Filter by stock symbol
pattern_type: str (optional) - Filter by pattern
limit: int (default: 100) - Number of results
Response:

success: bool
recommendations: List[Dict] - Historical recommendations
total_found: int
DATABASE INITIALIZATION
Location
Add DDL to Schema.sql (Postgres/Schema.sql)
Add table creation to DBInit.py (via execute_safe_ddl) OR create via DBInit.py method
Process
Following the requirement: "For any new tables required, the DDL should be added to DBInit.py and executed using database cursor. Also, the DDL should be added to Schema.sql"

UI IMPLEMENTATION
Location
Custom-App/app/templates/dashboard.html

Section: Swing Trade Recommendations
Add new section in Live Market tab (similar to Options Chain Scanner section)

Display fields:

Stock Symbol (clickable - opens candlestick chart)
Pattern Type
Entry Price
Target Price
Stop Loss
Potential Gain %
Risk/Reward Ratio
Confidence Score
Expected Holding Period
Rationale
Sparkline chart (last 30 days)
Features:

Sort by confidence score (default)
Filter by pattern type
Filter by gain range (10-20%)
Refresh button to generate new recommendations
Show historical recommendations button
Export to Excel/CSV
DOCUMENTATION
Create SwingPlan.txt
Location: docker-installer/SwingPlan.txt

Document the complete implementation plan
Include all technical details
Include database schema
Include API endpoints
Include UI specifications
Include pattern detection logic details
IMPLEMENTATION STEPS
Create SwingPlan.txt documentation file with complete plan details
Create SwingTradeScanner.py module
Add database table DDL to Schema.sql
Add database table creation to DBInit.py (execute_safe_ddl)
Implement technical indicators (RSI, MACD, ATR)
Implement pattern detection logic for individual stocks
Implement pattern detection logic for Nifty
Implement confidence scoring
Add /api/swing_trades endpoint (includes individual stocks)
Add /api/swing_trades_nifty endpoint for Nifty-specific recommendations
Add /api/swing_trades_history endpoint
Add UI section in dashboard.html for stock recommendations
Add UI section for Nifty recommendations (can be combined or separate)
Add JavaScript functions for loading/displaying recommendations
Test with sample stocks and Nifty
Add error handling and logging
DATA PERSISTENCE
All generated recommendations will be saved to swing_trade_suggestions table with:

generated_at: Current timestamp when recommendation was generated
analysis_date: Current date
run_date: Current date (from DEFAULT CURRENT_DATE)
This allows:

Tracking recommendations by date
Historical analysis of recommendation performance
Backtesting of recommendations
Performance metrics over time
TECHNICAL INDICATORS LIBRARY
Create helper module: TechnicalIndicators.py (or add to existing file)

calculate_rsi(price_data, period=14)
calculate_macd(price_data, fast=12, slow=26, signal=9)
calculate_atr(high, low, close, period=14)
detect_support_resistance(price_data)
calculate_fibonacci_levels(high, low, retracement_levels)
RISK MANAGEMENT
Minimum risk-reward ratio: 2:1
Stop-loss: Dynamic based on ATR (typically 1.5-2x ATR)
Maximum holding period: 10 days
Position sizing suggestion based on ATR
Risk per trade: Maximum 2% of portfolio per recommendation