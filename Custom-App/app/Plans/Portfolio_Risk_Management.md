# Portfolio Risk Management System Implementation Plan

## Overview
Build a comprehensive portfolio risk management system that provides real-time alerts during market hours, user-configurable risk thresholds, mutual fund underlying instruments analysis, and separate/combined risk calculations for equity and mutual fund portfolios.

## Components

### 1. Risk Configuration Module
**File**: `docker-installer/Custom-App/app/risk/RiskConfigManager.py`
- Store user-configurable risk thresholds in database
- Default thresholds:
  - Single stock concentration: 20% of portfolio
  - Single sector concentration: 40% of portfolio
  - Portfolio Beta: ±1.5 (warning), ±2.0 (critical)
  - Portfolio VaR (95%): 5% of portfolio value (warning), 10% (critical)
  - Maximum drawdown: 15% (warning), 25% (critical)
  - Liquidity risk: <10% in liquid stocks (warning)
- CRUD operations for threshold management
- Validation of threshold values

**Database Schema**: Add `my_schema.risk_config` table
- Columns: id, config_key, config_value, threshold_type, warning_level, created_at, updated_at
- Indexes on config_key

### 2. Equity Portfolio Risk Calculator
**File**: `docker-installer/Custom-App/app/risk/EquityRiskCalculator.py`
- Calculate portfolio-level metrics:
  - Portfolio volatility (standard deviation of returns)
  - Portfolio Beta (vs Nifty50) - leverage existing PortfolioHedgeAnalyzer
  - Value at Risk (VaR) at 95% and 99% confidence
  - Conditional VaR (CVaR/Expected Shortfall)
  - Maximum Drawdown (MDD)
  - Sharpe Ratio
  - Sortino Ratio
  - Portfolio correlation matrix
- Calculate concentration risks:
  - Single stock concentration (% of portfolio)
  - Sector concentration (% by sector)
  - Top 5 holdings concentration
- Calculate liquidity risks:
  - % of portfolio in liquid stocks (volume-based)
  - Average daily volume vs portfolio size
- Use historical price data from `rt_intraday_price` table
- Calculate over 60-day, 90-day, and 1-year periods

### 3. Mutual Fund Underlying Instruments Fetcher
**File**: `docker-installer/Custom-App/app/risk/MFHoldingsFetcher.py`
- Fetch MF underlying holdings from:
  - AMFI website (monthly portfolio disclosure)
  - SEBI APIs (if available)
  - Alternative: Yahoo Finance or other public sources
- Parse and store MF holdings in database
- Handle different MF schemes and their holdings
- Update frequency: Monthly (AMFI publishes monthly)

**Database Schema**: Add `my_schema.mf_underlying_holdings` table
- Columns: id, mf_symbol, mf_scheme_name, stock_symbol, stock_name, weight_pct, quantity, value, as_on_date, created_at
- Indexes on mf_symbol, as_on_date, stock_symbol

### 4. Mutual Fund Risk Calculator
**File**: `docker-installer/Custom-App/app/risk/MFRiskCalculator.py`
- Calculate MF portfolio risks based on underlying holdings:
  - Effective portfolio composition (aggregate underlying holdings)
  - Sector concentration of MF portfolio
  - Stock concentration (if same stock in multiple MFs)
  - Combined equity + MF sector exposure
  - Combined equity + MF stock concentration
- Calculate MF-specific risks:
  - Tracking error (if index fund)
  - Active share (if active fund)
  - Expense ratio impact
  - Fund manager risk (if available)
- Use underlying holdings from `mf_underlying_holdings` table
- Aggregate holdings across all user's MF holdings

### 5. Combined Portfolio Risk Analyzer
**File**: `docker-installer/Custom-App/app/risk/CombinedRiskAnalyzer.py`
- Combine equity and MF risks:
  - Total portfolio value (equity + MF)
  - Combined sector exposure
  - Combined stock concentration (direct equity + MF underlying)
  - Combined portfolio Beta (weighted average)
  - Combined VaR
  - Combined maximum drawdown
- Identify overlapping exposures:
  - Same stock in equity and MF holdings
  - Sector over-concentration across equity and MF
- Calculate diversification score

### 6. Real-Time Risk Monitor
**File**: `docker-installer/Custom-App/app/risk/RealTimeRiskMonitor.py`
- Monitor risk metrics during market hours (9 AM - 3:30 PM IST)
- Check thresholds every 5 minutes (configurable)
- Compare current risk metrics against user thresholds
- Generate alerts when thresholds breached:
  - Warning level: Log and display in dashboard
  - Critical level: Immediate alert + notification
- Track alert history

**Database Schema**: Add `my_schema.risk_alerts` table
- Columns: id, alert_type, alert_level, metric_name, current_value, threshold_value, message, recommendation, created_at, acknowledged, acknowledged_at
- Indexes on alert_level, created_at, acknowledged

### 7. Risk Recommendations Engine
**File**: `docker-installer/Custom-App/app/risk/RiskRecommendationsEngine.py`
- Generate corrective action recommendations:
  - Reduce position size (for concentration risk)
  - Diversify sector exposure
  - Add hedging positions (for high beta)
  - Increase liquid holdings (for liquidity risk)
  - Rebalance portfolio
  - Exit recommendations (for extreme risks)
- Prioritize recommendations by risk severity
- Provide specific actionable steps:
  - Which stocks to reduce
  - Which sectors to diversify into
  - Suggested hedge positions (using existing PortfolioHedgeAnalyzer)
  - Target allocation percentages

### 8. API Endpoints
**File**: `docker-installer/Custom-App/app/KiteAccessToken.py`
- `/api/risk/config` - GET/POST/PUT for risk threshold configuration
- `/api/risk/equity` - GET equity portfolio risk metrics
- `/api/risk/mf` - GET mutual fund portfolio risk metrics
- `/api/risk/combined` - GET combined portfolio risk metrics
- `/api/risk/alerts` - GET current risk alerts
- `/api/risk/recommendations` - GET risk mitigation recommendations
- `/api/risk/mf-holdings/fetch` - POST trigger MF holdings fetch
- `/api/risk/mf-holdings/{mf_symbol}` - GET underlying holdings for a MF

### 9. Frontend Dashboard
**File**: `docker-installer/Custom-App/app/templates/dashboard.html`
- Add "Risk Management" tab/section
- Display real-time risk metrics:
  - Portfolio volatility, Beta, VaR, MDD
  - Concentration metrics (stock, sector)
  - Liquidity metrics
- Display active alerts with color coding:
  - Green: All metrics within limits
  - Yellow: Warning level breaches
  - Red: Critical level breaches
- Display recommendations section:
  - List of recommended actions
  - Priority indicators
  - Action buttons (manual execution)
- Risk configuration panel:
  - Editable threshold values
  - Save/Reset functionality
- MF underlying holdings viewer:
  - Table showing MF holdings breakdown
  - Sector-wise aggregation
  - Stock overlap analysis

### 10. Scheduled Jobs
**File**: `docker-installer/Custom-App/app/scheduled_jobs/RiskMonitoringScheduler.py`
- Daily risk calculation job (after market close)
- Monthly MF holdings fetch job (1st of month, after AMFI publishes)
- Real-time monitoring job (runs during market hours, every 5 minutes)

**File**: `docker-installer/Custom-App/app/crontab.txt`
- Add cron job for daily risk calculation: `0 16 * * 1-5` (4 PM IST, after market close)
- Add cron job for monthly MF holdings fetch: `0 20 1 * *` (8 PM IST, 1st of month)
- Add cron job for real-time monitoring: `*/5 9-15 * * 1-5` (every 5 minutes during market hours)

### 11. WebSocket Integration (Optional Enhancement)
- Push real-time risk alerts via WebSocket
- Update risk metrics in real-time on dashboard
- Integrate with existing KiteWS infrastructure

## Implementation Order

1. Database schema updates (risk_config, risk_alerts, mf_underlying_holdings tables)
2. Risk configuration module (RiskConfigManager.py)
3. Equity risk calculator (EquityRiskCalculator.py)
4. MF holdings fetcher (MFHoldingsFetcher.py)
5. MF risk calculator (MFRiskCalculator.py)
6. Combined risk analyzer (CombinedRiskAnalyzer.py)
7. Risk recommendations engine (RiskRecommendationsEngine.py)
8. Real-time risk monitor (RealTimeRiskMonitor.py)
9. API endpoints integration
10. Frontend dashboard updates
11. Scheduled jobs setup
12. Testing and validation

## Dependencies

- Existing: PortfolioHedgeAnalyzer.py (for Beta calculation)
- Existing: TechnicalIndicators.py (for volatility calculations)
- New: AMFI/SEBI data fetching libraries (requests, BeautifulSoup, or APIs)
- Existing: Database connection (Boilerplate.py)
- Existing: WebSocket infrastructure (for real-time updates)

## Testing Considerations

- Test with various portfolio compositions
- Test threshold breach scenarios
- Test MF holdings fetching with different MF schemes
- Test real-time monitoring during market hours
- Validate risk calculations against known benchmarks
- Test recommendation generation logic

## Risk Metrics Calculation Details

### Equity Portfolio Risk Calculation

**Portfolio Volatility:**
- Calculate daily returns for each holding
- Weight returns by portfolio allocation
- Calculate portfolio return time series
- Compute standard deviation of portfolio returns
- Annualize: σ_annual = σ_daily × √252

**Value at Risk (VaR):**
- Historical VaR: Use historical returns distribution
- Parametric VaR: VaR = Portfolio Value × σ × Z-score × √time_horizon
- Monte Carlo VaR: Simulate portfolio returns

**Maximum Drawdown:**
- Track peak portfolio value
- Calculate drawdown = (Peak - Current) / Peak
- Track maximum drawdown over time period

**Sharpe Ratio:**
- Sharpe = (Portfolio Return - Risk Free Rate) / Portfolio Volatility
- Risk-free rate: 6% annual (configurable)

**Concentration Risk:**
- Single stock: Max % in any single stock
- Sector: Max % in any single sector (from master_scrips.sector_code)
- Top 5: Combined % of top 5 holdings

**Liquidity Risk:**
- Calculate average daily volume for each holding
- Compare portfolio size to daily volume
- Flag if portfolio > 10% of 30-day average volume

### Mutual Fund Risk Calculation

**Effective Portfolio Composition:**
- Aggregate all underlying holdings across user's MF holdings
- Weight by MF holding value and underlying weight
- Calculate effective exposure to each stock/sector

**Sector Concentration:**
- Map underlying holdings to sectors
- Calculate sector weights in MF portfolio
- Compare against thresholds

**Stock Overlap:**
- Identify stocks held in multiple MFs
- Calculate combined exposure (direct + indirect)
- Flag if combined exposure exceeds thresholds

### Combined Risk Calculation

**Total Portfolio Metrics:**
- Combine equity and MF portfolio values
- Calculate combined sector exposure
- Calculate combined stock concentration
- Weighted average Beta: (Equity Beta × Equity Value + MF Beta × MF Value) / Total Value

**Overlap Analysis:**
- Find stocks in both equity holdings and MF underlying holdings
- Calculate total exposure (direct + indirect)
- Flag if exceeds concentration limits

## Alert Types

1. **Concentration Alerts:**
   - Single stock > threshold
   - Single sector > threshold
   - Top 5 holdings > threshold

2. **Volatility Alerts:**
   - Portfolio volatility > threshold
   - VaR > threshold
   - Maximum drawdown > threshold

3. **Beta Alerts:**
   - Portfolio Beta > threshold (high market correlation)
   - Portfolio Beta < -threshold (inverse correlation)

4. **Liquidity Alerts:**
   - Low liquidity holdings > threshold
   - Portfolio size vs daily volume ratio > threshold

5. **MF-Specific Alerts:**
   - High expense ratio
   - Poor tracking error (for index funds)
   - Low active share (for active funds)

## Recommendation Types

1. **Reduce Position:**
   - Stock name, current %, target %, reduction amount

2. **Diversify Sector:**
   - Over-concentrated sector, suggested sectors, target allocation

3. **Add Hedge:**
   - Suggested hedge instrument, quantity, expected impact

4. **Rebalance:**
   - Current allocation, target allocation, rebalancing steps

5. **Exit Position:**
   - Stock/MF name, reason, suggested exit strategy

