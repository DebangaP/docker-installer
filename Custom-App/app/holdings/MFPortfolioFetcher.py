"""
Mutual Fund Portfolio Holdings Fetcher
Fetches constituent stock holdings for mutual funds from various sources
"""
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional
import logging
import time
import re
from common.Boilerplate import get_db_connection
import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)


class MFPortfolioFetcher:
    """Fetches portfolio holdings (constituent stocks) for Mutual Funds"""
    
    def __init__(self):
        """Initialize MF Portfolio Fetcher"""
        self.mftool = None
        self._init_mftool()
    
    def _init_mftool(self):
        """Initialize mftool library for AMFI data"""
        try:
            from mftool import Mftool
            self.mftool = Mftool()
            logging.info("mftool library initialized successfully")
        except ImportError:
            logging.warning("mftool library not available. Will use alternative methods.")
            self.mftool = None
        except Exception as e:
            logging.warning(f"Error initializing mftool: {e}. Will use alternative methods.")
            self.mftool = None
    
    def get_scheme_code_for_mf(self, mf_symbol: str) -> Optional[str]:
        """
        Get AMFI scheme code for a mutual fund symbol
        
        Args:
            mf_symbol: Mutual Fund trading symbol
            
        Returns:
            AMFI scheme code if found, None otherwise
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT scheme_code 
                FROM my_schema.mf_nav_history 
                WHERE mf_symbol = %s AND scheme_code IS NOT NULL 
                LIMIT 1
            """, (mf_symbol,))
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result:
                # Handle both dict (RealDictCursor) and tuple results
                if isinstance(result, dict):
                    return result.get('scheme_code')
                else:
                    return result[0] if len(result) > 0 else None
            
            return None
            
        except Exception as e:
            logging.error(f"Error fetching scheme code for {mf_symbol}: {e}")
            return None
    
    def get_yahoo_symbol_for_mf(self, mf_symbol: str) -> Optional[str]:
        """
        Get Yahoo Finance symbol for a mutual fund (Zerodha symbol)
        
        Args:
            mf_symbol: Mutual Fund trading symbol from Zerodha
            
        Returns:
            Yahoo Finance symbol if found, None otherwise
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Check if we have stored Yahoo symbol
            cursor.execute("""
                SELECT yahoo_symbol 
                FROM my_schema.mf_nav_history 
                WHERE mf_symbol = %s AND yahoo_symbol IS NOT NULL 
                LIMIT 1
            """, (mf_symbol,))
            
            result = cursor.fetchone()
            if result:
                # Handle both dict (RealDictCursor) and tuple results
                if isinstance(result, dict):
                    yahoo_sym = result.get('yahoo_symbol')
                else:
                    yahoo_sym = result[0] if len(result) > 0 else None
                
                cursor.close()
                conn.close()
                
                if yahoo_sym:
                    return yahoo_sym
            
            cursor.close()
            conn.close()
            
            # If not found, try to discover it
            return self._discover_yahoo_symbol(mf_symbol)
            
        except Exception as e:
            logging.error(f"Error fetching Yahoo symbol for {mf_symbol}: {e}")
            return None
    
    def _discover_yahoo_symbol(self, mf_symbol: str) -> Optional[str]:
        """
        Discover Yahoo Finance symbol by trying different variations using web scraping
        
        Args:
            mf_symbol: Mutual Fund trading symbol from Zerodha
            
        Returns:
            Yahoo Finance symbol if found, None otherwise
        """
        try:
            import requests
            
            # Common variations to try
            variations = [
                f"{mf_symbol}.NS",  # With .NS suffix
                mf_symbol,  # As-is
                f"{mf_symbol}-BO",  # Some MFs have -BO suffix
                f"{mf_symbol}.BO",  # Some MFs have .BO suffix
            ]
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            
            for yahoo_symbol in variations:
                try:
                    # Try to access the Yahoo Finance quote page
                    url = f"https://finance.yahoo.com/quote/{yahoo_symbol}"
                    response = requests.get(url, headers=headers, timeout=5, allow_redirects=False)
                    
                    # Check if page is valid (200 status and doesn't contain error messages)
                    if response.status_code == 200:
                        # Check if page contains error messages or redirects to lookup
                        page_text = response.text.lower()
                        if 'symbol lookup' not in page_text and 'not found' not in page_text and 'invalid symbol' not in page_text:
                            # Try to verify it's a mutual fund by checking holdings page
                            holdings_url = f"https://finance.yahoo.com/quote/{yahoo_symbol}/holdings"
                            holdings_response = requests.get(holdings_url, headers=headers, timeout=5)
                            
                            if holdings_response.status_code == 200:
                                # Valid symbol found, store it
                                self._save_yahoo_symbol(mf_symbol, yahoo_symbol)
                                logging.info(f"Discovered Yahoo symbol for {mf_symbol}: {yahoo_symbol}")
                                return yahoo_symbol
                except Exception as e:
                    logging.debug(f"Tried {yahoo_symbol} for {mf_symbol}: {e}")
                    continue
            
            logging.warning(f"Could not discover Yahoo symbol for {mf_symbol}")
            return None
            
        except Exception as e:
            logging.error(f"Error discovering Yahoo symbol for {mf_symbol}: {e}")
            return None
    
    def _save_yahoo_symbol(self, mf_symbol: str, yahoo_symbol: str):
        """
        Save Yahoo Finance symbol mapping to database
        
        Args:
            mf_symbol: Mutual Fund trading symbol from Zerodha
            yahoo_symbol: Yahoo Finance symbol
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Update all records for this MF symbol
            cursor.execute("""
                UPDATE my_schema.mf_nav_history 
                SET yahoo_symbol = %s
                WHERE mf_symbol = %s
            """, (yahoo_symbol, mf_symbol))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logging.info(f"Saved Yahoo symbol mapping: {mf_symbol} -> {yahoo_symbol}")
            
        except Exception as e:
            logging.error(f"Error saving Yahoo symbol for {mf_symbol}: {e}")
    
    def fetch_portfolio_from_mftool(self, scheme_code: str) -> List[Dict]:
        """
        Fetch portfolio holdings from mftool library
        
        Args:
            scheme_code: AMFI scheme code
            
        Returns:
            List of dictionaries with portfolio holdings
        """
        if not self.mftool:
            logging.warning("mftool not available, cannot fetch portfolio")
            return []
        
        try:
            logging.info(f"Attempting to fetch portfolio from mftool for scheme_code: {scheme_code}")
            
            # Try to get scheme info which may contain portfolio data
            # Note: mftool API may vary, checking common methods
            scheme_info = None
            
            # Try get_scheme_info if available
            if hasattr(self.mftool, 'get_scheme_info'):
                try:
                    logging.debug(f"Trying get_scheme_info for scheme {scheme_code}")
                    scheme_info = self.mftool.get_scheme_info(scheme_code)
                    logging.debug(f"get_scheme_info returned: {type(scheme_info)}")
                except Exception as e:
                    logging.debug(f"get_scheme_info not available or failed: {e}")
            
            # Try get_scheme_details if available
            if not scheme_info and hasattr(self.mftool, 'get_scheme_details'):
                try:
                    logging.debug(f"Trying get_scheme_details for scheme {scheme_code}")
                    scheme_info = self.mftool.get_scheme_details(scheme_code)
                    logging.debug(f"get_scheme_details returned: {type(scheme_info)}")
                except Exception as e:
                    logging.debug(f"get_scheme_details not available or failed: {e}")
            
            # Try get_portfolio_holdings if available (some versions of mftool have this)
            if not scheme_info and hasattr(self.mftool, 'get_portfolio_holdings'):
                try:
                    logging.info(f"Trying get_portfolio_holdings for scheme {scheme_code}")
                    portfolio_data = self.mftool.get_portfolio_holdings(scheme_code)
                    if portfolio_data:
                        logging.info(f"get_portfolio_holdings returned data: {type(portfolio_data)}, length: {len(portfolio_data) if hasattr(portfolio_data, '__len__') else 'N/A'}")
                        return self._parse_portfolio_data(portfolio_data)
                    else:
                        logging.warning(f"get_portfolio_holdings returned empty data for scheme {scheme_code}")
                except Exception as e:
                    logging.warning(f"get_portfolio_holdings not available or failed for scheme {scheme_code}: {e}")
            
            # Try get_scheme_holdings if available (alternative method name)
            if not scheme_info and hasattr(self.mftool, 'get_scheme_holdings'):
                try:
                    logging.info(f"Trying get_scheme_holdings for scheme {scheme_code}")
                    portfolio_data = self.mftool.get_scheme_holdings(scheme_code)
                    if portfolio_data:
                        logging.info(f"get_scheme_holdings returned data: {type(portfolio_data)}, length: {len(portfolio_data) if hasattr(portfolio_data, '__len__') else 'N/A'}")
                        return self._parse_portfolio_data(portfolio_data)
                except Exception as e:
                    logging.warning(f"get_scheme_holdings not available or failed for scheme {scheme_code}: {e}")
            
            # If scheme_info contains portfolio data, extract it
            if scheme_info and isinstance(scheme_info, dict):
                logging.debug(f"Scheme info keys: {list(scheme_info.keys())}")
                # Check for common portfolio data keys
                portfolio_data = scheme_info.get('portfolio', None) or \
                                scheme_info.get('holdings', None) or \
                                scheme_info.get('top_holdings', None) or \
                                scheme_info.get('top_10_holdings', None)
                
                if portfolio_data:
                    logging.info(f"Found portfolio data in scheme_info for {scheme_code}")
                    return self._parse_portfolio_data(portfolio_data)
                else:
                    logging.warning(f"Scheme info found but no portfolio data keys present for {scheme_code}")
            
            logging.warning(f"No portfolio data found in mftool response for scheme {scheme_code}")
            return []
            
        except Exception as e:
            logging.error(f"Error fetching portfolio from mftool for scheme {scheme_code}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return []
    
    def _parse_portfolio_data(self, portfolio_data: List) -> List[Dict]:
        """
        Parse portfolio data from various formats
        
        Args:
            portfolio_data: Raw portfolio data (list of dicts or similar)
            
        Returns:
            List of parsed portfolio holdings
        """
        holdings = []
        
        try:
            for item in portfolio_data:
                if isinstance(item, dict):
                    holding = {
                        'stock_symbol': item.get('symbol') or item.get('scrip_id') or item.get('stock_symbol', ''),
                        'stock_name': item.get('name') or item.get('stock_name') or item.get('company_name', ''),
                        'weight_pct': float(item.get('weight', 0) or item.get('weight_pct', 0) or item.get('percentage', 0)),
                        'quantity': float(item.get('quantity', 0) or item.get('shares', 0) or 0),
                        'value': float(item.get('value', 0) or item.get('amount', 0) or 0),
                        'sector': item.get('sector', '') or item.get('industry', '')
                    }
                    
                    # Only add if we have at least a stock symbol
                    if holding['stock_symbol']:
                        holdings.append(holding)
            
            logging.info(f"Parsed {len(holdings)} portfolio holdings")
            return holdings
            
        except Exception as e:
            logging.error(f"Error parsing portfolio data: {e}")
            return []
    
    def fetch_portfolio_from_yahoo(self, mf_symbol: str) -> List[Dict]:
        """
        Fetch portfolio holdings from Yahoo Finance using web scraping only
        
        Args:
            mf_symbol: Mutual Fund trading symbol from Zerodha
            
        Returns:
            List of dictionaries with portfolio holdings
        """
        try:
            # Get Yahoo Finance symbol (discover if not stored)
            yahoo_symbol = self.get_yahoo_symbol_for_mf(mf_symbol)
            
            if not yahoo_symbol:
                # Fallback: try common variations
                variations = [
                    f"{mf_symbol}.NS",
                    mf_symbol,
                    f"{mf_symbol}-BO",
                    f"{mf_symbol}.BO"
                ]
            else:
                variations = [yahoo_symbol]
            
            # Try web scraping for each symbol variation
            logging.info(f"Attempting to scrape Yahoo Finance holdings for {mf_symbol}")
            for yahoo_code in variations:
                try:
                    scraped_holdings = self._scrape_yahoo_portfolio(yahoo_code)
                    if scraped_holdings:
                        # Save the symbol if we discovered it
                        if yahoo_code not in [f"{mf_symbol}.NS", mf_symbol] and not yahoo_symbol:
                            self._save_yahoo_symbol(mf_symbol, yahoo_code)
                        logging.info(f"Successfully scraped {len(scraped_holdings)} holdings from Yahoo Finance for {yahoo_code}")
                        return scraped_holdings
                except Exception as e:
                    logging.debug(f"Web scraping failed for {yahoo_code}: {e}")
                    import traceback
                    logging.debug(traceback.format_exc())
                    continue
            
            logging.warning(f"No portfolio data found on Yahoo Finance for {mf_symbol}")
            return []
            
        except Exception as e:
            logging.error(f"Error fetching portfolio from Yahoo Finance for {mf_symbol}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return []
    
    def _scrape_yahoo_portfolio(self, yahoo_symbol: str) -> List[Dict]:
        """
        Scrape portfolio holdings from Yahoo Finance web page
        Uses multiple strategies to extract holdings data from Yahoo Finance's dynamic content
        
        Args:
            yahoo_symbol: Yahoo Finance symbol (e.g., "INF0R8F01018.NS")
            
        Returns:
            List of portfolio holdings
        """
        try:
            import requests
            from bs4 import BeautifulSoup
            import json
            
            # Strategy 1: Try the holdings page
            url = f"https://finance.yahoo.com/quote/{yahoo_symbol}/holdings"
            
            logging.info(f"Scraping Yahoo Finance holdings page: {url}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            session = requests.Session()
            response = session.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            html_content = response.text
            soup = BeautifulSoup(html_content, 'html.parser')
            holdings = []
            
            # Strategy 2: Extract JSON data from script tags (Yahoo Finance embeds data here)
            # Look for root.App.main or similar structures that contain the actual data
            scripts = soup.find_all('script')
            
            for script in scripts:
                if not script.string:
                    continue
                    
                script_text = script.string
                
                # Look for the main data structure - Yahoo Finance uses root.App.main
                # This contains all the page data including holdings
                if 'root.App.main' in script_text or 'QuoteSummaryStore' in script_text:
                    try:
                        # Extract the JSON object from root.App.main
                        # Pattern: root.App.main = {...};
                        main_match = re.search(r'root\.App\.main\s*=\s*(\{.*?\});', script_text, re.DOTALL)
                        if main_match:
                            json_str = main_match.group(1)
                            data = json.loads(json_str)
                            
                            # Navigate through the structure to find holdings
                            # Structure: context.dispatcher.stores.QuoteSummaryStore.quoteSummary.holdings
                            holdings_data = self._extract_holdings_from_yahoo_json(data)
                            if holdings_data:
                                logging.info(f"Extracted {len(holdings_data)} holdings from root.App.main JSON")
                                return holdings_data
                    except json.JSONDecodeError as e:
                        logging.debug(f"JSON decode error: {e}")
                        continue
                    except Exception as e:
                        logging.debug(f"Error extracting from root.App.main: {e}")
                        continue
                
                # Also look for direct holdings references
                if 'holdings' in script_text.lower() and ('[' in script_text or '{' in script_text):
                    try:
                        # Try to find JSON arrays or objects with holdings
                        # Pattern: "holdings": [...]
                        holdings_patterns = [
                            r'"holdings"\s*:\s*(\[.*?\])',
                            r'holdings:\s*(\[.*?\])',
                            r'"portfolioHoldings"\s*:\s*(\[.*?\])',
                            r'portfolioHoldings:\s*(\[.*?\])',
                        ]
                        
                        for pattern in holdings_patterns:
                            matches = re.finditer(pattern, script_text, re.DOTALL)
                            for match in matches:
                                try:
                                    json_str = match.group(1)
                                    holdings_list = json.loads(json_str)
                                    if isinstance(holdings_list, list) and len(holdings_list) > 0:
                                        parsed = self._parse_portfolio_data(holdings_list)
                                        if parsed:
                                            logging.info(f"Extracted {len(parsed)} holdings from direct JSON pattern")
                                            return parsed
                                except (json.JSONDecodeError, ValueError):
                                    continue
                    except Exception as e:
                        logging.debug(f"Error searching for holdings in script: {e}")
                        continue
            
            # Strategy 3: Try Yahoo Finance API endpoint directly
            # Yahoo Finance has internal API endpoints
            try:
                api_holdings = self._try_yahoo_api_endpoint(yahoo_symbol)
                if api_holdings:
                    logging.info(f"Extracted {len(api_holdings)} holdings from Yahoo API endpoint")
                    return api_holdings
            except Exception as e:
                logging.debug(f"Yahoo API endpoint failed: {e}")
            
            # Strategy 4: Parse HTML tables as fallback
            tables = soup.find_all('table')
            for table in tables:
                table_holdings = self._parse_html_table(table)
                if table_holdings:
                    holdings.extend(table_holdings)
            
            if holdings:
                logging.info(f"Scraped {len(holdings)} holdings from HTML tables")
                return holdings
            
            logging.warning(f"Could not find holdings data on Yahoo Finance page for {yahoo_symbol}")
            logging.debug(f"Page title: {soup.title.string if soup.title else 'N/A'}")
            logging.debug(f"Response length: {len(html_content)} characters")
            return []
            
        except ImportError:
            logging.warning("requests or BeautifulSoup not available for web scraping. Install with: pip install requests beautifulsoup4")
            return []
        except Exception as e:
            logging.error(f"Error scraping Yahoo Finance portfolio for {yahoo_symbol}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return []
    
    def _extract_holdings_from_yahoo_json(self, data: dict) -> List[Dict]:
        """
        Extract holdings data from Yahoo Finance's JSON structure
        
        Args:
            data: Parsed JSON from root.App.main
            
        Returns:
            List of holdings or empty list
        """
        try:
            # Navigate through Yahoo Finance's nested structure
            # Typical path: context.dispatcher.stores.QuoteSummaryStore.quoteSummary.holdings
            paths_to_try = [
                ['context', 'dispatcher', 'stores', 'QuoteSummaryStore', 'quoteSummary', 'holdings'],
                ['context', 'dispatcher', 'stores', 'QuoteSummaryStore', 'quoteSummary', 'topHoldings', 'holdings'],
                ['quoteSummary', 'holdings'],
                ['holdings'],
            ]
            
            for path in paths_to_try:
                current = data
                try:
                    for key in path:
                        if isinstance(current, dict) and key in current:
                            current = current[key]
                        else:
                            raise KeyError(f"Path {path} not found")
                    
                    if isinstance(current, list):
                        return self._parse_portfolio_data(current)
                    elif isinstance(current, dict) and 'holdings' in current:
                        holdings_list = current['holdings']
                        if isinstance(holdings_list, list):
                            return self._parse_portfolio_data(holdings_list)
                except (KeyError, TypeError):
                    continue
            
            return []
        except Exception as e:
            logging.debug(f"Error extracting holdings from JSON structure: {e}")
            return []
    
    def _try_yahoo_api_endpoint(self, yahoo_symbol: str) -> List[Dict]:
        """
        Try to fetch holdings from Yahoo Finance's internal API
        
        Args:
            yahoo_symbol: Yahoo Finance symbol
            
        Returns:
            List of holdings or empty list
        """
        try:
            import requests
            
            # Yahoo Finance uses query2 API endpoint
            # Format: https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}?modules=topHoldings
            url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{yahoo_symbol}"
            params = {
                'modules': 'topHoldings,holdings'
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                # Extract holdings from API response
                result = data.get('quoteSummary', {}).get('result', [])
                if result and len(result) > 0:
                    quote_data = result[0]
                    
                    # Try topHoldings first
                    if 'topHoldings' in quote_data:
                        holdings_data = quote_data['topHoldings'].get('holdings', [])
                        if holdings_data:
                            return self._parse_portfolio_data(holdings_data)
                    
                    # Try holdings
                    if 'holdings' in quote_data:
                        holdings_data = quote_data['holdings']
                        if isinstance(holdings_data, list):
                            return self._parse_portfolio_data(holdings_data)
            
            return []
        except Exception as e:
            logging.debug(f"Yahoo API endpoint error: {e}")
            return []
    
    def _parse_html_table(self, table) -> List[Dict]:
        """
        Parse HTML table to extract holdings data
        
        Args:
            table: BeautifulSoup table element
            
        Returns:
            List of holdings
        """
        holdings = []
        try:
            rows = table.find_all('tr')
            if len(rows) < 2:
                return []
            
            # Get headers
            header_row = rows[0]
            headers = [th.get_text(strip=True).lower() for th in header_row.find_all(['th', 'td'])]
            
            # Check if this looks like a holdings table
            if not any(keyword in ' '.join(headers) for keyword in ['symbol', 'name', 'weight', 'shares', 'value', 'percentage', 'holding']):
                return []
            
            logging.info(f"Found holdings table with headers: {headers}")
            
            # Parse data rows
            for row in rows[1:]:
                cells = row.find_all(['td', 'th'])
                if len(cells) < 2:
                    continue
                
                cell_texts = [cell.get_text(strip=True) for cell in cells]
                holding = {}
                
                for i, header in enumerate(headers):
                    if i >= len(cell_texts):
                        break
                    
                    value = cell_texts[i]
                    
                    if 'symbol' in header or 'ticker' in header:
                        holding['stock_symbol'] = value.upper().strip()
                    elif 'name' in header or 'company' in header:
                        holding['stock_name'] = value.strip()
                    elif 'weight' in header or 'percentage' in header or '%' in header:
                        weight_str = re.sub(r'[^\d.]', '', value)
                        try:
                            holding['weight_pct'] = float(weight_str)
                        except:
                            holding['weight_pct'] = 0.0
                    elif 'shares' in header or 'quantity' in header:
                        shares_str = re.sub(r'[^\d.]', '', value.replace(',', ''))
                        try:
                            holding['quantity'] = float(shares_str)
                        except:
                            holding['quantity'] = 0.0
                    elif 'value' in header or 'amount' in header:
                        value_str = re.sub(r'[^\d.]', '', value.replace(',', '').replace('$', ''))
                        try:
                            holding['value'] = float(value_str)
                        except:
                            holding['value'] = 0.0
                    elif 'sector' in header or 'industry' in header:
                        holding['sector'] = value.strip()
                
                if holding.get('stock_symbol'):
                    holdings.append(holding)
            
            return holdings
        except Exception as e:
            logging.debug(f"Error parsing HTML table: {e}")
            return []
    
    def _parse_yahoo_holdings(self, holdings_data, mf_symbol: str) -> List[Dict]:
        """
        Parse Yahoo Finance holdings data
        
        Args:
            holdings_data: Holdings data from yfinance
            mf_symbol: MF symbol for reference
            
        Returns:
            List of parsed holdings
        """
        holdings = []
        
        try:
            # Yahoo Finance holdings format may vary
            if isinstance(holdings_data, pd.DataFrame):
                for idx, row in holdings_data.iterrows():
                    holding = {
                        'stock_symbol': str(row.get('Symbol', '') or row.get('symbol', '') or idx),
                        'stock_name': str(row.get('Name', '') or row.get('name', '') or ''),
                        'weight_pct': float(row.get('Weight', 0) or row.get('weight', 0) or row.get('%', 0) or 0),
                        'quantity': float(row.get('Shares', 0) or row.get('quantity', 0) or 0),
                        'value': float(row.get('Value', 0) or row.get('value', 0) or 0),
                        'sector': str(row.get('Sector', '') or row.get('sector', '') or '')
                    }
                    
                    if holding['stock_symbol']:
                        holdings.append(holding)
            
            elif isinstance(holdings_data, dict):
                for symbol, data in holdings_data.items():
                    holding = {
                        'stock_symbol': str(symbol),
                        'stock_name': str(data.get('name', '') if isinstance(data, dict) else ''),
                        'weight_pct': float(data.get('weight', 0) if isinstance(data, dict) else 0),
                        'quantity': float(data.get('shares', 0) if isinstance(data, dict) else 0),
                        'value': float(data.get('value', 0) if isinstance(data, dict) else 0),
                        'sector': str(data.get('sector', '') if isinstance(data, dict) else '')
                    }
                    
                    if holding['stock_symbol']:
                        holdings.append(holding)
            
            logging.info(f"Parsed {len(holdings)} holdings from Yahoo Finance for {mf_symbol}")
            return holdings
            
        except Exception as e:
            logging.error(f"Error parsing Yahoo Finance holdings: {e}")
            return []
    
    def map_stock_symbols(self, holdings: List[Dict]) -> List[Dict]:
        """
        Map stock symbols to master_scrips table for standardization
        
        Args:
            holdings: List of holdings with stock symbols
            
        Returns:
            List of holdings with mapped symbols
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            mapped_holdings = []
            
            for holding in holdings:
                stock_symbol = holding.get('stock_symbol', '').strip().upper()
                if not stock_symbol:
                    continue
                
                # Try to find in master_scrips
                cursor.execute("""
                    SELECT scrip_id, sector_code
                    FROM my_schema.master_scrips
                    WHERE UPPER(scrip_id) = %s 
                    OR UPPER(yahoo_code) = %s
                    LIMIT 1
                """, (stock_symbol, stock_symbol))
                
                result = cursor.fetchone()
                
                if result:
                    # Handle both dict (RealDictCursor) and tuple results
                    if isinstance(result, dict):
                        mapped_scrip_id = result.get('scrip_id')
                        mapped_sector = result.get('sector_code')
                    else:
                        mapped_scrip_id = result[0] if len(result) > 0 else None
                        mapped_sector = result[1] if len(result) > 1 else None
                    
                    if mapped_scrip_id:
                        # Use mapped symbol
                        holding['stock_symbol'] = mapped_scrip_id
                        if not holding.get('sector') and mapped_sector:
                            holding['sector'] = mapped_sector or ''
                else:
                    # Keep original symbol, try to clean it
                    holding['stock_symbol'] = stock_symbol
                
                mapped_holdings.append(holding)
            
            cursor.close()
            conn.close()
            
            logging.info(f"Mapped {len(mapped_holdings)} stock symbols")
            return mapped_holdings
            
        except Exception as e:
            logging.error(f"Error mapping stock symbols: {e}")
            return holdings
    
    def save_portfolio_to_database(self, mf_symbol: str, scheme_code: Optional[str], 
                                   holdings: List[Dict], portfolio_date: date = None) -> int:
        """
        Save portfolio holdings to database
        
        Args:
            mf_symbol: Mutual Fund trading symbol
            scheme_code: AMFI scheme code (optional)
            holdings: List of portfolio holdings
            portfolio_date: Date of portfolio snapshot (defaults to today)
            
        Returns:
            Number of records inserted/updated
        """
        if not holdings:
            return 0
        
        if portfolio_date is None:
            portfolio_date = date.today()
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            records_inserted = 0
            
            for holding in holdings:
                try:
                    stock_symbol = holding.get('stock_symbol', '').strip()
                    if not stock_symbol:
                        continue
                    
                    cursor.execute("""
                        INSERT INTO my_schema.mf_portfolio_holdings 
                        (mf_symbol, scheme_code, stock_symbol, stock_name, weight_pct, 
                         quantity, value, sector, portfolio_date, fetch_date)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                        ON CONFLICT (mf_symbol, stock_symbol, portfolio_date) 
                        DO UPDATE SET 
                            stock_name = EXCLUDED.stock_name,
                            weight_pct = EXCLUDED.weight_pct,
                            quantity = EXCLUDED.quantity,
                            value = EXCLUDED.value,
                            sector = EXCLUDED.sector,
                            fetch_date = NOW()
                    """, (
                        mf_symbol,
                        scheme_code,
                        stock_symbol,
                        holding.get('stock_name', ''),
                        holding.get('weight_pct', 0.0),
                        holding.get('quantity', 0.0),
                        holding.get('value', 0.0),
                        holding.get('sector', ''),
                        portfolio_date
                    ))
                    
                    records_inserted += 1
                    
                except Exception as e:
                    logging.warning(f"Error inserting portfolio holding for {mf_symbol}/{holding.get('stock_symbol')}: {e}")
                    continue
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logging.info(f"Saved {records_inserted} portfolio holdings for {mf_symbol}")
            return records_inserted
            
        except Exception as e:
            logging.error(f"Error saving portfolio to database for {mf_symbol}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return 0
    
    def get_latest_portfolio(self, mf_symbol: str) -> List[Dict]:
        """
        Get latest portfolio holdings from database
        
        Args:
            mf_symbol: Mutual Fund trading symbol (or ISIN - will try to match)
            
        Returns:
            List of portfolio holdings
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # First, try to find the actual tradingsymbol if mf_symbol is an ISIN
            actual_mf_symbol = mf_symbol
            cursor.execute("""
                SELECT DISTINCT tradingsymbol 
                FROM my_schema.mf_holdings
                WHERE (tradingsymbol = %s OR isin = %s)
                AND run_date = (SELECT MAX(run_date) FROM my_schema.mf_holdings)
                LIMIT 1
            """, (mf_symbol, mf_symbol))
            
            result = cursor.fetchone()
            if result:
                # Handle both dict (RealDictCursor) and tuple results
                if isinstance(result, dict):
                    actual_mf_symbol = result.get('tradingsymbol') or mf_symbol
                else:
                    actual_mf_symbol = result[0] if result[0] else mf_symbol
                
                if actual_mf_symbol != mf_symbol:
                    logging.debug(f"Mapped {mf_symbol} to tradingsymbol {actual_mf_symbol}")
            
            # Now query portfolio holdings with the actual symbol
            cursor.execute("""
                SELECT 
                    stock_symbol,
                    stock_name,
                    weight_pct,
                    quantity,
                    value,
                    sector,
                    portfolio_date
                FROM my_schema.mf_portfolio_holdings
                WHERE mf_symbol = %s
                AND portfolio_date = (
                    SELECT MAX(portfolio_date) 
                    FROM my_schema.mf_portfolio_holdings 
                    WHERE mf_symbol = %s
                )
                ORDER BY weight_pct DESC NULLS LAST
            """, (actual_mf_symbol, actual_mf_symbol))
            
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            
            holdings = [dict(row) for row in rows]
            
            if len(holdings) == 0:
                logging.warning(f"No portfolio holdings found for {mf_symbol} (tried as {actual_mf_symbol}). Portfolio may need to be fetched first.")
            else:
                logging.info(f"Retrieved {len(holdings)} portfolio holdings for {mf_symbol} (tradingsymbol: {actual_mf_symbol})")
            
            return holdings
            
        except Exception as e:
            logging.error(f"Error getting latest portfolio for {mf_symbol}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return []
    
    def fetch_portfolio_for_mf(self, mf_symbol: str, scheme_code: Optional[str] = None, 
                               portfolio_date: date = None) -> Dict:
        """
        Fetch portfolio holdings for a mutual fund (main method)
        
        Args:
            mf_symbol: Mutual Fund trading symbol
            scheme_code: AMFI scheme code (optional, will be fetched if not provided)
            portfolio_date: Date of portfolio snapshot (defaults to today)
            
        Returns:
            Dictionary with success status and holdings data
        """
        try:
            if portfolio_date is None:
                portfolio_date = date.today()
            
            # Get scheme code if not provided
            if not scheme_code:
                scheme_code = self.get_scheme_code_for_mf(mf_symbol)
            
            holdings = []
            source = None
            
            # Try mftool first if scheme code is available
            if scheme_code and self.mftool:
                logging.info(f"Attempting to fetch portfolio from mftool for {mf_symbol} (scheme_code: {scheme_code})")
                holdings = self.fetch_portfolio_from_mftool(scheme_code)
                if holdings:
                    source = 'mftool'
                    logging.info(f"Successfully fetched {len(holdings)} holdings from mftool for {mf_symbol}")
                else:
                    logging.warning(f"mftool returned no holdings for {mf_symbol} (scheme_code: {scheme_code})")
            else:
                if not scheme_code:
                    logging.info(f"No scheme code found for {mf_symbol}, skipping mftool")
                if not self.mftool:
                    logging.info(f"mftool not available, skipping mftool")
            
            # Fallback to Yahoo Finance
            if not holdings:
                logging.info(f"Attempting to fetch portfolio from Yahoo Finance for {mf_symbol}")
                holdings = self.fetch_portfolio_from_yahoo(mf_symbol)
                if holdings:
                    source = 'yahoo'
                    logging.info(f"Successfully fetched {len(holdings)} holdings from Yahoo Finance for {mf_symbol}")
                else:
                    logging.warning(f"Yahoo Finance returned no holdings for {mf_symbol}")
            
            if not holdings:
                error_details = []
                if not scheme_code:
                    error_details.append("no AMFI scheme code")
                if not self.mftool:
                    error_details.append("mftool not available")
                error_details.append("Yahoo Finance returned no data")
                
                return {
                    'success': False,
                    'error': f'No portfolio data found for {mf_symbol}. Reasons: {", ".join(error_details)}',
                    'mf_symbol': mf_symbol,
                    'scheme_code': scheme_code,
                    'holdings': [],
                    'source': None
                }
            
            # Map stock symbols to master_scrips
            holdings = self.map_stock_symbols(holdings)
            
            # Save to database
            records_saved = self.save_portfolio_to_database(mf_symbol, scheme_code, holdings, portfolio_date)
            
            return {
                'success': True,
                'mf_symbol': mf_symbol,
                'scheme_code': scheme_code,
                'holdings': holdings,
                'holdings_count': len(holdings),
                'records_saved': records_saved,
                'portfolio_date': portfolio_date,
                'source': source
            }
            
        except Exception as e:
            logging.error(f"Error fetching portfolio for {mf_symbol}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'mf_symbol': mf_symbol,
                'holdings': [],
                'source': None
            }
    
    def fetch_portfolios_for_all_held_mfs(self) -> Dict:
        """
        Fetch portfolios for all mutual funds currently held
        
        Returns:
            Dictionary with results for each MF
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get all unique MF symbols from mf_holdings
            cursor.execute("""
                SELECT DISTINCT tradingsymbol, fund, isin
                FROM my_schema.mf_holdings
                WHERE run_date = (SELECT MAX(run_date) FROM my_schema.mf_holdings)
                AND tradingsymbol IS NOT NULL
                AND tradingsymbol != ''
            """)
            
            mf_list = cursor.fetchall()
            cursor.close()
            conn.close()
            
            results = {
                'total_mfs': len(mf_list),
                'successful': 0,
                'failed': 0,
                'results': []
            }
            
            for mf_symbol, fund_name, isin in mf_list:
                if not mf_symbol:
                    continue
                
                logging.info(f"Fetching portfolio for {mf_symbol} ({fund_name}) [ISIN: {isin}]")
                result = self.fetch_portfolio_for_mf(mf_symbol)
                
                if result.get('success'):
                    results['successful'] += 1
                    logging.info(f"✓ Successfully fetched portfolio for {mf_symbol}: {result.get('holdings_count', 0)} holdings from {result.get('source', 'unknown')}")
                else:
                    results['failed'] += 1
                    error_msg = result.get('error', 'Unknown error')
                    logging.warning(f"✗ Failed to fetch portfolio for {mf_symbol}: {error_msg}")
                
                results['results'].append(result)
                
                # Add delay to avoid rate limiting
                time.sleep(1)
            
            logging.info(f"Portfolio fetch completed: {results['successful']} successful, {results['failed']} failed")
            return results
            
        except Exception as e:
            logging.error(f"Error fetching portfolios for all MFs: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return {
                'total_mfs': 0,
                'successful': 0,
                'failed': 0,
                'error': str(e),
                'results': []
            }

