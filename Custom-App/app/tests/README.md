# Risk Management Test Suite

# Run unit tests only (no database needed)
cd Custom-App/app
python -m unittest discover tests -p "test_risk_management.py" -v

# Run all tests including integration
python tests/run_tests.py --integration

# Run specific test class
python -m unittest tests.test_risk_management.TestRiskMetricsCalculator -v


This directory contains comprehensive test cases for the Risk Management module.

## Test Files

- `test_risk_management.py` - Main test suite covering:
  - RiskMetricsCalculator
  - RiskConcentrationAnalyzer
  - AdvancedHedgingStrategies
  - RiskService
  - API endpoints
  - Edge cases and error handling

## Running Tests

### Run all tests:
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

### Run specific test class:
```bash
python -m unittest tests.test_risk_management.TestRiskMetricsCalculator -v
```

### Run using the test runner:
```bash
python tests/run_tests.py
```

### Run from project root:
```bash
cd Custom-App/app
python -m unittest discover tests -v
```

## Test Coverage

### RiskMetricsCalculator Tests
- Returns calculation
- Sharpe ratio calculation
- Sortino ratio calculation
- Volatility calculation
- VaR (Historical, Parametric, Monte Carlo)
- CVaR calculation
- Max drawdown calculation
- Beta calculation
- Holdings data retrieval
- Historical prices retrieval

### RiskConcentrationAnalyzer Tests
- Holdings data retrieval
- Stock concentration analysis
- Sector concentration analysis
- Comprehensive concentration analysis

### AdvancedHedgingStrategies Tests
- Delta hedging (positive, negative, neutral delta)
- Tail risk hedging
- Correlation hedging (with and without existing holdings)
- Volatility hedging
- All strategies aggregation

### RiskService Tests
- Portfolio risk metrics retrieval
- Concentration analysis
- Error handling when services unavailable

### API Endpoint Tests
- `/api/risk/metrics` endpoint
- `/api/risk/concentration` endpoint
- `/api/risk/hedging/advanced` endpoint

### Edge Cases
- Empty holdings
- Insufficient data
- Zero portfolio value
- Negative delta
- Database connection failures

## Mocking

Tests use `unittest.mock` to mock:
- Database connections
- External API calls
- File I/O operations

This allows tests to run without requiring:
- Active database connections
- Real market data
- External API access

## Test Data

Tests use synthetic data for:
- Price series
- Returns series
- Holdings data
- Market data

## Notes

- Tests are designed to be isolated and independent
- Each test cleans up after itself
- Mock objects are used to avoid external dependencies
- Tests verify both success and error cases

