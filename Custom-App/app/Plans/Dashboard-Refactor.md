# Dashboard Refactoring Plan

## Overview
Break down the large dashboard.html file (10,000+ lines) into smaller, maintainable components by separating HTML structure, JavaScript functionality, and CSS styles into organized files.

## Current State Analysis
- **File**: `docker-installer/Custom-App/app/templates/dashboard.html`
- **Size**: ~10,500 lines
- **Structure**: 
  - Lines 8-547: CSS styles (inline `<style>` tag)
  - Lines 584-2254: HTML structure with multiple tabs
  - Lines 2257-10465: JavaScript (inline `<script>` tag)

## Proposed Structure

### 1. CSS Organization
**Location**: `docker-installer/Custom-App/app/static/css/`

- `dashboard-base.css` - Base styles (body, layout, sidebar, header)
- `dashboard-components.css` - Reusable component styles (cards, buttons, tables)
- `dashboard-tabs.css` - Tab-specific styles (holdings, stocks, utilities, etc.)
- `dashboard-modals.css` - Modal styles (candlestick, OI charts)
- `dashboard-responsive.css` - Media queries and responsive styles

### 2. JavaScript Organization
**Location**: `docker-installer/Custom-App/app/static/js/`

- `dashboard-core.js` - Core utilities (TabLoadTracker, date functions, refresh system)
- `dashboard-api.js` - API communication functions
- `dashboard-holdings.js` - Holdings tab functionality
- `dashboard-stocks.js` - Stocks tab functionality (gainers, predictions, index projections)
- `dashboard-utilities.js` - Utilities tab functionality
- `dashboard-charts.js` - Chart rendering (candlestick, OI, gauge charts)
- `dashboard-modals.js` - Modal management
- `dashboard-tables.js` - Table sorting, pagination, filtering

### 3. HTML Component Organization
**Location**: `docker-installer/Custom-App/app/templates/components/`

- `sidebar.html` - Sidebar navigation component
- `header.html` - Header component
- `holdings-tab.html` - Holdings tab content
- `stocks-tab.html` - Stocks tab content
- `utilities-tab.html` - Utilities tab content
- `candlestick-modal.html` - Candlestick chart modal
- `oi-chart-modal.html` - OI chart modal

### 4. Main Dashboard Template
**Location**: `docker-installer/Custom-App/app/templates/dashboard.html`

- Simplified main template that includes:
  - Component includes (Jinja2 `{% include %}`)
  - CSS file references
  - JavaScript file references
  - Minimal inline code (only critical initialization)

## Implementation Steps

### Phase 1: Extract CSS
1. Create `static/css/` directory structure
2. Extract CSS from `<style>` tag into separate files
3. Update `dashboard.html` to link CSS files
4. Test visual appearance matches original

### Phase 2: Extract JavaScript
1. Create `static/js/` directory structure
2. Extract JavaScript functions into logical modules
3. Ensure proper function scoping (avoid global conflicts)
4. Update `dashboard.html` to include JS files in correct order
5. Test all functionality works

### Phase 3: Extract HTML Components
1. Create `templates/components/` directory
2. Extract tab content into separate component files
3. Extract modals into separate component files
4. Use Jinja2 `{% include %}` to compose main template
5. Test all tabs and modals work correctly

### Phase 4: Optimization
1. Minify CSS/JS files (optional, for production)
2. Add source maps for debugging
3. Document component dependencies
4. Create component usage guide

## File Dependencies

### CSS Load Order:
1. `dashboard-base.css`
2. `dashboard-components.css`
3. `dashboard-tabs.css`
4. `dashboard-modals.css`
5. `dashboard-responsive.css`

### JavaScript Load Order:
1. `dashboard-core.js` (utilities, base functions)
2. `dashboard-api.js` (API functions)
3. `dashboard-tables.js` (table utilities)
4. `dashboard-charts.js` (chart utilities)
5. `dashboard-modals.js` (modal management)
6. Tab-specific files (`dashboard-holdings.js`, `dashboard-stocks.js`, `dashboard-utilities.js`)

## Considerations

- **Jinja2 Template Variables**: Ensure all template variables (e.g., `{{ show_holdings }}`) are properly passed to components
- **Global Variables**: Carefully manage shared state between modules
- **Event Handlers**: Ensure event listeners are properly attached after DOM loads
- **API Endpoints**: All API endpoints remain unchanged
- **Backward Compatibility**: Maintain full functionality during migration

## Testing Checklist

- [ ] All tabs load and display correctly
- [ ] All modals open and function properly
- [ ] All charts render correctly
- [ ] All API calls work
- [ ] Table sorting/filtering works
- [ ] Responsive design maintained
- [ ] No console errors
- [ ] Performance not degraded

## Implementation Todos

1. Extract CSS from dashboard.html into separate files in static/css/ directory
2. Extract core JavaScript utilities into dashboard-core.js and dashboard-api.js
3. Extract tab-specific JavaScript into separate files (holdings, stocks, utilities)
4. Extract chart-related JavaScript into dashboard-charts.js and dashboard-modals.js
5. Extract HTML components into templates/components/ directory
6. Update dashboard.html to include extracted CSS, JS, and HTML components
7. Test all functionality to ensure nothing is broken after refactoring

