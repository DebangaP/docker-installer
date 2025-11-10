# KiteAccessToken.py Refactoring Plan

## Overview
Refactor `KiteAccessToken.py` (~9,500 lines, 113+ endpoints) to keep only token management functionality in this file, while moving all other endpoints and business logic to appropriate router modules organized by domain.

## Current State Analysis
- **File**: `docker-installer/Custom-App/app/KiteAccessToken.py`
- **Size**: ~9,500 lines
- **Endpoints**: 113+ API endpoints
- **Structure**:
  - Token management functions (lines ~184-1373)
  - FastAPI app initialization (line 219)
  - Multiple domain endpoints mixed together
  - Helper functions scattered throughout

## What Stays in KiteAccessToken.py

### Token Management Functions:
- `is_access_token_valid(access_token)` - Validate access token
- `generate_new_access_token(request_token)` - Generate new access token
- `get_access_token()` - Main token retrieval logic
- `is_token_fetched_today()` - Check if token was fetched today
- `is_token_currently_valid()` - Check current token validity

### Token Management Endpoints:
- `GET /` - Home/login page with token handling
- `GET /redirect` - OAuth redirect handler
- `GET /logout` - Logout endpoint
- `GET /dashboard` - Dashboard page (requires token)

### App Initialization:
- FastAPI app instance
- Template configuration
- Static files mounting
- Middleware setup
- Global configuration variables

### Core Utilities (may stay or move to common):
- `get_cache_headers()` - Cache header helper
- `cached_json_response()` - Cached response helper
- `cache_get_json()` / `cache_set_json()` - Redis cache helpers

## What Moves Out

### 1. Holdings Endpoints → `api/routers/holdings.py`
- `GET /api/holdings` - List holdings
- `POST /api/refresh_holdings` - Refresh holdings
- `GET /api/mf_holdings` - Mutual fund holdings
- `POST /api/refresh_mf_nav` - Refresh MF NAV
- `GET /api/today_pnl_summary` - Today's P&L summary
- `GET /api/portfolio_history` - Portfolio history
- `GET /api/holdings/patterns` - Holdings patterns
- `GET /api/holdings/irrational-analysis` - Irrational analysis
- `GET /api/portfolio_hedge_analysis` - Portfolio hedge analysis
- `enrich_holdings_with_today_pnl()` - Helper function

### 2. Market Data Endpoints → `api/routers/market.py`
- `GET /api/market_dashboard` - Market dashboard
- `GET /api/market_bias` - Market bias analysis
- `GET /api/market_bias_chart` - Market bias chart
- `GET /api/premarket_analysis` - Premarket analysis
- `GET /api/footprint_analysis` - Footprint analysis
- `GET /api/orderflow_analysis` - Order flow analysis
- `GET /api/micro_levels` - Micro levels
- `GET /api/tpo_charts` - TPO charts
- `GET /api/tpo_5day_chart` - 5-day TPO chart
- `POST /api/calculate_tpo` - Calculate TPO
- `GET /api/candlestick_chart` - Candlestick chart
- `GET /api/candlestick/{trading_symbol}` - Candlestick data
- `GET /api/trades_table` - Trades table
- `GET /api/gainers_losers` - Gainers/losers
- `GET /api/gainers` - Top gainers
- `GET /api/losers` - Top losers

### 3. Options Endpoints → `api/routers/options.py`
- `GET /api/options_latest` - Latest options data
- `GET /api/options_scanner` - Options scanner
- `GET /api/options_chain` - Options chain
- `GET /api/options_data` - Options data
- `GET /api/options_oi_analysis` - OI analysis
- `GET /api/options_oi_chart` - OI chart
- `GET /api/options_historical_oi` - Historical OI
- `POST /api/fetch_options` - Fetch options data
- `GET /api/derivatives_suggestions` - Derivatives suggestions
- `GET /api/derivatives_history` - Derivatives history

### 4. Fundamentals Endpoints → `api/routers/fundamentals.py`
- `GET /api/fundamentals/list` - List fundamentals
- `GET /api/fundamentals/fetch` - Fetch fundamentals
- `GET /api/fundamentals/fetch-yahoo` - Fetch Yahoo fundamentals
- `POST /api/fundamentals/fetch-yahoo` - Fetch Yahoo fundamentals (POST)
- `POST /api/fundamentals/cancel` - Cancel fetch
- `POST /api/fundamentals/clear-cancel` - Clear cancel flag
- `POST /api/fundamentals/cancel-yahoo` - Cancel Yahoo fetch
- `POST /api/fundamentals/clear-cancel-yahoo` - Clear Yahoo cancel flag
- `GET /api/fundamentals/update-mcap` - Update market cap
- `POST /api/fundamentals/update-mcap` - Update market cap (POST)
- `POST /api/fundamentals/cancel-mcap` - Cancel market cap update
- `POST /api/fundamentals/clear-cancel-mcap` - Clear market cap cancel flag

### 5. Stocks/Predictions Endpoints → `api/routers/stocks.py`
- `GET /api/swing_trades` - Swing trades
- `GET /api/swing_trades_nifty` - Swing trades Nifty
- `GET /api/swing_trades_history` - Swing trades history
- `POST /api/refresh_swing_trades` - Refresh swing trades
- `GET /api/prophet_predictions/generate` - Generate predictions
- `GET /api/prophet_predictions/top_gainers` - Top gainers
- `GET /api/prophet_predictions/index_projections` - Index projections
- `GET /api/prophet_predictions/sectoral_averages` - Sectoral averages
- `GET /api/prophet_predictions/symbol_search` - Symbol search
- `GET /api/sparkline/{trading_symbol}` - Sparkline chart
- `GET /api/sparklines` - Multiple sparklines
- `POST /api/refresh_stock_prices` - Refresh stock prices
- `POST /api/add_new_stock` - Add new stock

### 6. Sentiment Endpoints → `api/routers/sentiment.py`
- `GET /api/sentiment/list_news` - List news sentiment
- `GET /api/sentiment/fetch_news` - Fetch news
- `GET /api/sentiment/calculate_fundamental` - Calculate fundamental sentiment

### 7. System/Admin Endpoints → `api/routers/system.py`
- `GET /api/system_status` - System status
- `GET /api/analysis_date` - Get analysis date
- `POST /api/analysis_date` - Set analysis date
- `GET /api/available_dates` - Available dates
- `POST /api/refresh_futures` - Refresh futures
- `POST /api/start_kitews` - Start KiteWS
- `POST /api/insert_ohlc` - Insert OHLC data
- `GET /api/margin_data` - Margin data
- `GET /api/margin/available` - Available margin
- `POST /api/margin/calculate` - Calculate margin

### 8. Data Import/Export Endpoints → `api/routers/data.py`
- `POST /api/export_data` - Export data
- `POST /api/import_data` - Import data

### 9. Helper Functions → `api/utils/`
- `calculate_supertrend()` → `api/utils/technical_indicators.py`
- `get_holdings_data()` → `api/services/holdings_service.py`
- `get_system_status()` → `api/services/system_service.py`
- Other helper functions to appropriate service files

## Proposed Structure

### Directory Structure:
```
docker-installer/Custom-App/app/
├── KiteAccessToken.py          # Token management only
├── api/
│   ├── __init__.py
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── holdings.py
│   │   ├── market.py
│   │   ├── options.py
│   │   ├── fundamentals.py
│   │   ├── stocks.py
│   │   ├── sentiment.py
│   │   ├── system.py
│   │   └── data.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── holdings_service.py
│   │   ├── market_service.py
│   │   ├── system_service.py
│   │   └── ...
│   └── utils/
│       ├── __init__.py
│       ├── cache.py
│       ├── technical_indicators.py
│       └── responses.py
└── main.py                      # App initialization (optional)
```

## Implementation Steps

### Phase 1: Setup Router Infrastructure
1. Create `api/` directory structure
2. Create `api/routers/` directory
3. Create `api/services/` directory
4. Create `api/utils/` directory
5. Create base router files with FastAPI `APIRouter`
6. Update `KiteAccessToken.py` to include routers

### Phase 2: Extract Holdings Module
1. Create `api/routers/holdings.py`
2. Move holdings endpoints to router
3. Create `api/services/holdings_service.py`
4. Move holdings helper functions to service
5. Update imports in `KiteAccessToken.py`
6. Test holdings endpoints

### Phase 3: Extract Market Module
1. Create `api/routers/market.py`
2. Move market data endpoints to router
3. Create `api/services/market_service.py`
4. Move market helper functions to service
5. Update imports
6. Test market endpoints

### Phase 4: Extract Options Module
1. Create `api/routers/options.py`
2. Move options endpoints to router
3. Create `api/services/options_service.py`
4. Move options helper functions to service
5. Update imports
6. Test options endpoints

### Phase 5: Extract Fundamentals Module
1. Create `api/routers/fundamentals.py`
2. Move fundamentals endpoints to router
3. Create `api/services/fundamentals_service.py`
4. Move fundamentals helper functions to service
5. Update imports
6. Test fundamentals endpoints

### Phase 6: Extract Stocks Module
1. Create `api/routers/stocks.py`
2. Move stocks/predictions endpoints to router
3. Create `api/services/stocks_service.py`
4. Move stocks helper functions to service
5. Update imports
6. Test stocks endpoints

### Phase 7: Extract Sentiment Module
1. Create `api/routers/sentiment.py`
2. Move sentiment endpoints to router
3. Create `api/services/sentiment_service.py`
4. Move sentiment helper functions to service
5. Update imports
6. Test sentiment endpoints

### Phase 8: Extract System Module
1. Create `api/routers/system.py`
2. Move system/admin endpoints to router
3. Create `api/services/system_service.py`
4. Move system helper functions to service
5. Update imports
6. Test system endpoints

### Phase 9: Extract Data Module
1. Create `api/routers/data.py`
2. Move import/export endpoints to router
3. Create `api/services/data_service.py`
4. Move data helper functions to service
5. Update imports
6. Test data endpoints

### Phase 10: Extract Utilities
1. Create `api/utils/cache.py` - Cache helpers
2. Create `api/utils/technical_indicators.py` - Technical indicators
3. Create `api/utils/responses.py` - Response helpers
4. Update all imports across modules
5. Test all functionality

### Phase 11: Cleanup and Documentation
1. Remove unused imports from `KiteAccessToken.py`
2. Add docstrings to all router modules
3. Create API documentation
4. Update README with new structure
5. Final testing

## Router Implementation Pattern

### Example: `api/routers/holdings.py`
```python
from fastapi import APIRouter, Query, Request
from api.services.holdings_service import HoldingsService
from api.utils.responses import cached_json_response

router = APIRouter(prefix="/api", tags=["holdings"])

@router.get("/holdings")
async def get_holdings(
    request: Request,
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=100),
    sort_by: str = Query(None),
    sort_dir: str = Query("asc"),
    search: str = Query(None)
):
    service = HoldingsService()
    data = service.get_holdings(page, per_page, sort_by, sort_dir, search)
    return cached_json_response(data, "/api/holdings")
```

### Example: `api/services/holdings_service.py`
```python
from common.Boilerplate import get_db_connection
import psycopg2.extras

class HoldingsService:
    def get_holdings(self, page, per_page, sort_by, sort_dir, search):
        # Business logic here
        conn = get_db_connection()
        # ... implementation
        return holdings_data
```

## Key Considerations

### Dependencies:
- All routers need access to `get_db_connection()` from `common.Boilerplate`
- All routers need access to `redis_client` from `common.Boilerplate`
- All routers need access to `get_access_token()` from `KiteAccessToken.py` or `common.Boilerplate`
- Cache helpers should be in `api/utils/cache.py`

### Backward Compatibility:
- All API endpoints must maintain same paths
- All response formats must remain unchanged
- All query parameters must work the same way
- No breaking changes to frontend

### Testing Strategy:
- Test each module independently after extraction
- Test all endpoints after each phase
- Verify cache behavior
- Verify authentication/authorization
- Performance testing

## Migration Checklist

- [ ] Phase 1: Setup router infrastructure
- [ ] Phase 2: Extract holdings module
- [ ] Phase 3: Extract market module
- [ ] Phase 4: Extract options module
- [ ] Phase 5: Extract fundamentals module
- [ ] Phase 6: Extract stocks module
- [ ] Phase 7: Extract sentiment module
- [ ] Phase 8: Extract system module
- [ ] Phase 9: Extract data module
- [ ] Phase 10: Extract utilities
- [ ] Phase 11: Cleanup and documentation
- [ ] Final testing of all endpoints
- [ ] Performance verification
- [ ] Documentation update

## Benefits

1. **Maintainability**: Each domain is in its own file
2. **Testability**: Modules can be tested independently
3. **Scalability**: Easy to add new endpoints to appropriate modules
4. **Clarity**: Clear separation of concerns
5. **Reusability**: Services can be reused across routers
6. **Organization**: Logical grouping of related functionality

## Risks and Mitigation

### Risk 1: Breaking Changes
- **Mitigation**: Maintain exact API paths and response formats
- **Mitigation**: Test each phase thoroughly before moving to next

### Risk 2: Circular Dependencies
- **Mitigation**: Careful import structure, use dependency injection
- **Mitigation**: Keep common utilities in separate modules

### Risk 3: Performance Degradation
- **Mitigation**: No changes to business logic, only organization
- **Mitigation**: Performance testing after each phase

### Risk 4: Missing Dependencies
- **Mitigation**: Comprehensive testing of all endpoints
- **Mitigation**: Careful tracking of imports and dependencies

