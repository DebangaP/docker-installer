"""
Screener.in Data Fetcher
Fetches fundamental data from screener.in for stocks
"""

import requests
from bs4 import BeautifulSoup
import logging
from typing import Dict, List, Optional
from datetime import date, datetime
import json
import time
import re
import psycopg2
from psycopg2.extras import RealDictCursor
from common.Boilerplate import get_db_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import yfinance for market cap fallback
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not available. Market cap fallback will not work.")


class ScreenerDataFetcher:
    """
    Fetches fundamental data from screener.in for stocks
    """
    
    def __init__(self):
        self.base_url = "https://www.screener.in"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self._cancellation_flag_key = "fundamental_fetch_cancelled"
        # Initialize Redis client once for reuse
        self._redis_client = None
        self._init_redis_client()
        # Ensure database table exists
        self._ensure_system_flags_table()
    
    def _init_redis_client(self):
        """Initialize Redis client for cancellation flag"""
        try:
            import redis
            self._redis_client = redis.Redis(host='redis', port=6379, decode_responses=True, socket_connect_timeout=2)
            # Test connection
            self._redis_client.ping()
            logger.debug("Redis client initialized successfully")
        except Exception as e:
            logger.debug(f"Redis not available, will use database: {e}")
            self._redis_client = None
    
    def _ensure_system_flags_table(self):
        """Ensure system_flags table exists in database"""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS my_schema.system_flags (
                    flag_key VARCHAR(100) PRIMARY KEY,
                    value VARCHAR(10),
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            logger.warning(f"Error ensuring system_flags table exists: {e}")
    
    def check_cancellation(self) -> bool:
        """
        Check if batch processing has been cancelled by user
        
        Returns:
            True if cancelled, False otherwise
        """
        try:
            # Try Redis first (faster)
            if self._redis_client:
                try:
                    cancelled = self._redis_client.get(self._cancellation_flag_key)
                    if cancelled and (cancelled == '1' or cancelled == 'true'):
                        return True
                except Exception as e:
                    logger.debug(f"Redis check failed, trying database: {e}")
                    # Reinitialize Redis client
                    self._init_redis_client()
            
            # Fallback to database
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT value FROM my_schema.system_flags 
                WHERE flag_key = %s
            """, (self._cancellation_flag_key,))
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            if result and result[0]:
                cancelled_value = result[0]
                # Only return True if value is '1' or 'true', don't log empty values
                if cancelled_value == '1' or cancelled_value == 'true':
                    return True
            return False
        except Exception as e:
            logger.debug(f"Error checking cancellation flag: {e}")
            return False
    
    def set_cancellation_flag(self, cancelled: bool = True):
        """
        Set cancellation flag for batch processing
        
        Args:
            cancelled: True to cancel, False to clear cancellation
        """
        try:
            # Try Redis first
            if self._redis_client:
                try:
                    if cancelled:
                        self._redis_client.set(self._cancellation_flag_key, '1', ex=3600)  # Expire after 1 hour
                        logger.info(f"Cancellation flag set in Redis with key: {self._cancellation_flag_key}")
                    else:
                        # When clearing, delete from Redis
                        deleted = self._redis_client.delete(self._cancellation_flag_key)
                        if deleted:
                            logger.info(f"Cancellation flag cleared from Redis (key: {self._cancellation_flag_key})")
                        else:
                            logger.debug(f"Cancellation flag not found in Redis (key: {self._cancellation_flag_key})")
                    # Also set in database as backup
                except Exception as e:
                    logger.warning(f"Redis operation failed, using database only: {e}")
                    self._init_redis_client()
            
            # Always set in database as well (for reliability)
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO my_schema.system_flags (flag_key, value, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (flag_key) 
                DO UPDATE SET value = %s, updated_at = NOW()
            """, (self._cancellation_flag_key, '1' if cancelled else '0', '1' if cancelled else '0'))
            conn.commit()
            cursor.close()
            conn.close()
            if cancelled:
                logger.info(f"Cancellation flag set in database (key: {self._cancellation_flag_key})")
            else:
                logger.info(f"Cancellation flag cleared in database (key: {self._cancellation_flag_key})")
        except Exception as e:
            logger.error(f"Error setting cancellation flag: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def get_screener_code(self, scrip_id: str) -> Optional[int]:
        """
        Get screener code from master_scrips table
        
        Args:
            scrip_id: Stock symbol (e.g., 'RELIANCE')
            
        Returns:
            Screener code (integer) or None if not found
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT scrip_screener_code 
                FROM my_schema.master_scrips 
                WHERE scrip_id = %s
            """, (scrip_id,))
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result and result[0]:
                return int(result[0])
            return None
            
        except Exception as e:
            logger.error(f"Error getting screener code for {scrip_id}: {e}")
            return None
    
    def fetch_fundamental_data(self, scrip_id: str) -> Optional[Dict]:
        """
        Fetch fundamental data from screener.in
        
        Args:
            scrip_id: Stock symbol (e.g., 'RELIANCE')
            
        Returns:
            Dictionary containing fundamental metrics or None if fetch fails
        """
        # Screener.in uses company names/slugs in URLs, not numeric codes
        # Try multiple URL formats
        url_attempts = [
            f"{self.base_url}/company/{scrip_id}/",  # Try scrip_id directly
            f"{self.base_url}/company/{scrip_id.upper()}/",  # Try uppercase
        ]
        
        # If we have a screener code, try that too (though it's less likely to work)
        screener_code = self.get_screener_code(scrip_id)
        if screener_code:
            url_attempts.append(f"{self.base_url}/company/{screener_code}/")
        
        # Try searching for the company if direct URLs fail
        for url in url_attempts:
            try:
                logger.info(f"Trying to fetch fundamental data for {scrip_id} from {url}")
                
                response = self.session.get(url, timeout=30, allow_redirects=True)
                
                # Check if we got a valid page (not 404)
                if response.status_code == 200:
                    # Check if the page contains company data (not a search page or error)
                    if 'company' in response.url.lower() and response.status_code == 200:
                        # Parse HTML
                        soup = BeautifulSoup(response.content, 'html.parser')
                        
                        # Verify we got a company page (check for common company page elements)
                        # Screener.in company pages typically have specific elements
                        # Check for various indicators that this is a company page
                        is_company_page = (
                            soup.find('div', class_='company-name') or 
                            soup.find('h1') or 
                            soup.find('div', id='top-ratios') or
                            soup.find('div', class_='company-header') or
                            soup.find('table', class_='data-table') or
                            'company' in response.url.lower() and 'search' not in response.url.lower()
                        )
                        
                        if is_company_page:
                            # Extract fundamental data
                            data = self.parse_fundamental_data(soup, scrip_id, screener_code)
                            
                            if data:
                                # Add raw HTML for reference
                                data['raw_html'] = response.text[:10000]  # Store first 10k chars
                            
                            return data
                        else:
                            logger.debug(f"URL {url} did not return a valid company page")
                
            except requests.exceptions.RequestException as e:
                logger.debug(f"Failed to fetch from {url}: {e}")
                continue
            except Exception as e:
                logger.debug(f"Unexpected error for {url}: {e}")
                continue
        
        # If all direct URLs failed, try searching for the company
        logger.info(f"Direct URLs failed for {scrip_id}, trying search...")
        search_url = f"{self.base_url}/search/?q={scrip_id}"
        
        try:
            response = self.session.get(search_url, timeout=30)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find the first company link in search results
                company_links = soup.find_all('a', href=lambda href: href and '/company/' in href)
                if company_links:
                    # Get the first company link
                    company_path = company_links[0].get('href')
                    if company_path:
                        # Construct full URL
                        if company_path.startswith('/'):
                            company_url = f"{self.base_url}{company_path}"
                        else:
                            company_url = f"{self.base_url}/{company_path}"
                        
                        logger.info(f"Found company URL via search: {company_url}")
                        
                        # Fetch the company page
                        response = self.session.get(company_url, timeout=30)
                        if response.status_code == 200:
                            soup = BeautifulSoup(response.content, 'html.parser')
                            data = self.parse_fundamental_data(soup, scrip_id, screener_code)
                            
                            if data:
                                data['raw_html'] = response.text[:10000]
                            
                            return data
        except Exception as e:
            logger.debug(f"Search failed for {scrip_id}: {e}")
        
        logger.warning(f"Could not fetch fundamental data for {scrip_id} from screener.in")
        return None
    
    def get_market_cap_from_master_scrips(self, scrip_id: str) -> Optional[float]:
        """
        Get market cap from master_scrips table
        
        Args:
            scrip_id: Stock symbol
            
        Returns:
            Market cap in crores or None
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get market cap from master_scrips
            cursor.execute("""
                SELECT scrip_mcap 
                FROM my_schema.master_scrips 
                WHERE scrip_id = %s
                AND scrip_mcap IS NOT NULL
                AND scrip_mcap > 0
            """, (scrip_id,))
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result and result[0] is not None:
                try:
                    market_cap = float(result[0])
                    if market_cap > 0:
                        logger.info(f"Fetched market cap from master_scrips for {scrip_id}: {market_cap:.2f} Cr")
                        return market_cap
                    else:
                        logger.debug(f"Market cap in master_scrips for {scrip_id} is 0 or negative: {market_cap}")
                except (ValueError, TypeError) as e:
                    logger.debug(f"Error converting market cap to float for {scrip_id}: {e}, value: {result[0]}")
            else:
                logger.debug(f"No market cap found in master_scrips for {scrip_id}")
            
            return None
            
        except Exception as e:
            logger.warning(f"Error fetching market cap from master_scrips for {scrip_id}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def get_market_cap_from_yahoo(self, scrip_id: str) -> Optional[float]:
        """
        Get market cap from Yahoo Finance as fallback
        
        Args:
            scrip_id: Stock symbol
            
        Returns:
            Market cap in crores or None
        """
        if not YFINANCE_AVAILABLE:
            return None
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get yahoo_code from master_scrips
            cursor.execute("""
                SELECT yahoo_code 
                FROM my_schema.master_scrips 
                WHERE scrip_id = %s
            """, (scrip_id,))
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if not result or not result[0]:
                return None
            
            yahoo_code = result[0]
            
            # Try to fetch market cap from Yahoo Finance
            ticker = yf.Ticker(yahoo_code)
            info = ticker.info
            
            if 'marketCap' in info:
                # Market cap is in USD, convert to INR crores
                # 1 USD ≈ 83 INR (approximate, should use current rate)
                market_cap_usd = info['marketCap']
                market_cap_inr = market_cap_usd * 83  # Approximate conversion
                market_cap_crores = market_cap_inr / 10000000  # Convert to crores
                logger.info(f"Fetched market cap from Yahoo Finance for {scrip_id}: {market_cap_crores:.2f} Cr")
                return market_cap_crores
            
            return None
            
        except Exception as e:
            logger.debug(f"Error fetching market cap from Yahoo Finance for {scrip_id}: {e}")
            return None
    
    def parse_fundamental_data(self, html_content: BeautifulSoup, scrip_id: str, screener_code: int) -> Optional[Dict]:
        """
        Parse HTML content to extract fundamental metrics with improved parsing
        
        Args:
            html_content: BeautifulSoup object of the HTML page
            scrip_id: Stock symbol
            screener_code: Screener code
            
        Returns:
            Dictionary with fundamental metrics
        """
        try:
            data = {
                'scrip_id': scrip_id,
                'screener_code': screener_code,
                'pe_ratio': None,
                'pb_ratio': None,
                'debt_to_equity': None,
                'roe': None,
                'roce': None,
                'current_ratio': None,
                'quick_ratio': None,
                'eps': None,
                'revenue_growth': None,
                'profit_growth': None,
                'dividend_yield': None,
                'market_cap': None,
            }
            
            # Screener.in stores data in script tags as JSON
            # Try to extract from script tags first (most reliable)
            script_tags = html_content.find_all('script', type='application/json')
            for script in script_tags:
                try:
                    json_data = json.loads(script.string)
                    # Navigate through JSON structure to find metrics
                    if isinstance(json_data, dict):
                        # Look for company data, ratios, etc.
                        self._extract_from_json(json_data, data)
                except:
                    continue
            
            # Also try to find data in data attributes
            data_attrs = html_content.find_all(attrs={'data-ratio': True})
            for elem in data_attrs:
                ratio_name = elem.get('data-ratio', '').lower()
                ratio_value = elem.get_text(strip=True)
                self._parse_metric_value(ratio_name, ratio_value, data)
            
            # Find key metrics section
            # Screener.in uses specific CSS classes and structure
            # This is a simplified parser - may need adjustment based on actual HTML structure
            
            # Screener.in uses specific HTML structure
            # Try to find key metrics in various sections
            key_metrics = html_content.find_all('div', class_='key-metrics')
            if not key_metrics:
                # Try alternative selectors
                key_metrics = html_content.find_all('div', {'id': 'top-ratios'})
            
            # Look for ratio cards/boxes - Screener.in often uses these
            ratio_cards = html_content.find_all('div', class_='ratio')
            if not ratio_cards:
                ratio_cards = html_content.find_all('div', class_='metric')
            
            # Look for specific metrics in the page
            # P/E Ratio - try multiple approaches
            pe_elements = html_content.find_all(text=lambda text: text and ('P/E' in text or 'PE' in text or 'Price to Earnings' in text))
            if pe_elements:
                for elem in pe_elements:
                    parent = elem.parent
                    if parent:
                        value_elem = parent.find_next_sibling()
                        if value_elem:
                            try:
                                pe_value = value_elem.get_text(strip=True).replace(',', '')
                                data['pe_ratio'] = float(pe_value) if pe_value and pe_value != '—' else None
                                break
                            except (ValueError, AttributeError):
                                continue
            
            # Market Cap
            mcap_elements = html_content.find_all(text=lambda text: text and 'Market Cap' in text)
            if mcap_elements:
                for elem in mcap_elements:
                    parent = elem.parent
                    if parent:
                        value_elem = parent.find_next_sibling()
                        if value_elem:
                            try:
                                mcap_text = value_elem.get_text(strip=True)
                                # Parse market cap (e.g., "1,23,456 Cr")
                                mcap_value = mcap_text.replace(',', '').replace(' Cr', '').replace(' L', '')
                                if 'Cr' in mcap_text:
                                    data['market_cap'] = float(mcap_value) * 100  # Convert to crores
                                elif 'L' in mcap_text:
                                    data['market_cap'] = float(mcap_value) * 0.01  # Convert lakhs to crores
                                else:
                                    data['market_cap'] = float(mcap_value) if mcap_value else None
                                break
                            except (ValueError, AttributeError):
                                continue
            
            # Try to extract from ratio tables - Screener.in uses data tables
            # Look for all tables with ratio data
            ratio_tables = html_content.find_all('table')
            for table in ratio_tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        metric_name = cells[0].get_text(strip=True).lower()
                        metric_value = cells[1].get_text(strip=True)
                        
                        # Try to extract value from second cell, or third if second is empty
                        if not metric_value or metric_value in ['—', '-', '']:
                            if len(cells) >= 3:
                                metric_value = cells[2].get_text(strip=True)
                        
                        metric_value = metric_value.replace(',', '').replace('%', '').replace('₹', '').strip()
                        self._parse_metric_value(metric_name, metric_value, data)
            
            # Look for key-value pairs in various formats
            # Screener.in often uses <strong> tags for labels
            strong_tags = html_content.find_all('strong')
            for strong in strong_tags:
                text = strong.get_text(strip=True).lower()
                # Check if this looks like a metric label
                if any(keyword in text for keyword in ['pe', 'pb', 'roe', 'roce', 'debt', 'current', 'quick', 'eps', 'dividend', 'market cap']):
                    # Try to find value in next sibling or parent
                    parent = strong.parent
                    if parent:
                        # Look for value in same element or siblings
                        value_elem = parent.find_next_sibling()
                        if not value_elem:
                            # Try to find span or div with value
                            value_elem = parent.find('span') or parent.find('div')
                        if value_elem:
                            value_text = value_elem.get_text(strip=True)
                            self._parse_metric_value(text, value_text, data)
            
            # Also try to find metrics in div structures (Screener.in uses card-based layouts)
            metric_divs = html_content.find_all('div', class_=lambda x: x and ('ratio' in x.lower() or 'metric' in x.lower() or 'value' in x.lower()))
            for div in metric_divs:
                try:
                    label = div.find('div', class_=lambda x: x and ('label' in x.lower() or 'name' in x.lower()))
                    value = div.find('div', class_=lambda x: x and ('value' in x.lower() or 'number' in x.lower()))
                    
                    if label and value:
                        metric_name = label.get_text(strip=True).lower()
                        metric_value = value.get_text(strip=True).replace(',', '').replace('%', '').replace('₹', '').strip()
                        
                        if 'pe' in metric_name or 'p/e' in metric_name:
                            if metric_value and metric_value != '—' and metric_value != '-' and metric_value != '' and data['pe_ratio'] is None:
                                data['pe_ratio'] = float(metric_value)
                        elif 'pb' in metric_name or 'p/b' in metric_name:
                            if metric_value and metric_value != '—' and metric_value != '-' and metric_value != '' and data['pb_ratio'] is None:
                                data['pb_ratio'] = float(metric_value)
                        elif 'debt to equity' in metric_name or 'd/e' in metric_name:
                            if metric_value and metric_value != '—' and metric_value != '-' and metric_value != '' and data['debt_to_equity'] is None:
                                data['debt_to_equity'] = float(metric_value)
                        elif 'roe' in metric_name and 'roce' not in metric_name:
                            if metric_value and metric_value != '—' and metric_value != '-' and metric_value != '' and data['roe'] is None:
                                data['roe'] = float(metric_value)
                        elif 'roce' in metric_name:
                            if metric_value and metric_value != '—' and metric_value != '-' and metric_value != '' and data['roce'] is None:
                                data['roce'] = float(metric_value)
                except (ValueError, AttributeError) as e:
                    logger.debug(f"Error parsing div metric: {e}")
                    continue
            
            # Always try to get market cap from master_scrips first (preferred source)
            # Then fallback to screener.in value if master_scrips doesn't have it
            market_cap_source = 'screener.in'
            master_mcap = self.get_market_cap_from_master_scrips(scrip_id)
            
            if master_mcap and master_mcap > 0:
                # Use market cap from master_scrips (preferred)
                data['market_cap'] = master_mcap
                market_cap_source = 'master_scrips'
                logger.info(f"Using market cap from master_scrips for {scrip_id}: {master_mcap:.2f} Cr")
            elif data['market_cap'] is None or data['market_cap'] == 0 or data['market_cap'] == '':
                # Market cap not found in master_scrips and screener.in, try Yahoo Finance
                logger.info(f"Market cap not found in master_scrips or screener.in for {scrip_id}, trying Yahoo Finance...")
                yahoo_mcap = self.get_market_cap_from_yahoo(scrip_id)
                if yahoo_mcap and yahoo_mcap > 0:
                    data['market_cap'] = yahoo_mcap
                    market_cap_source = 'yahoo_finance'
                    logger.info(f"Using market cap from Yahoo Finance for {scrip_id}: {yahoo_mcap:.2f} Cr")
                else:
                    logger.warning(f"Market cap not found from any source for {scrip_id}")
            else:
                # Market cap was found from screener.in (but master_scrips didn't have it)
                market_cap_source = 'screener.in'
                logger.debug(f"Using market cap from screener.in for {scrip_id}: {data['market_cap']:.2f} Cr (master_scrips not available)")
            
            # Store raw data as JSONB
            data['raw_data'] = {
                'fetch_timestamp': datetime.now().isoformat(),
                'screener_code': screener_code,
                'scrip_id': scrip_id,
                'source': 'screener.in',
                'market_cap_source': market_cap_source
            }
            
            # Log what we successfully extracted
            extracted_metrics = [k for k, v in data.items() if v is not None and k not in ['scrip_id', 'screener_code', 'raw_data']]
            logger.info(f"Extracted {len(extracted_metrics)} metrics for {scrip_id}: {', '.join(extracted_metrics)}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error parsing fundamental data for {scrip_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _extract_from_json(self, json_data: Dict, data: Dict):
        """Extract metrics from JSON structure"""
        # Recursively search JSON for metric values
        if isinstance(json_data, dict):
            for key, value in json_data.items():
                key_lower = str(key).lower()
                if isinstance(value, (int, float)):
                    self._parse_metric_value(key_lower, str(value), data)
                elif isinstance(value, dict):
                    self._extract_from_json(value, data)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            self._extract_from_json(item, data)
    
    def _parse_metric_value(self, metric_name: str, metric_value: str, data: Dict):
        """
        Parse a metric value and update data dictionary
        
        Args:
            metric_name: Name of the metric (lowercase)
            metric_value: Value as string
            data: Data dictionary to update
        """
        if not metric_value or metric_value in ['—', '-', '', 'N/A', 'n/a', 'NA', 'na']:
            return
        
        try:
            # Clean the value
            clean_value = metric_value.replace(',', '').replace('%', '').replace('₹', '').replace('Rs.', '').replace('Rs', '').strip()
            
            # Handle special cases (e.g., "1,23,456 Cr")
            if 'cr' in clean_value.lower() or 'crore' in clean_value.lower():
                clean_value = clean_value.lower().replace('cr', '').replace('crore', '').strip()
                multiplier = 1  # Already in crores
            elif 'l' in clean_value.lower() or 'lakh' in clean_value.lower():
                clean_value = clean_value.lower().replace('l', '').replace('lakh', '').strip()
                multiplier = 0.01  # Convert lakhs to crores
            else:
                multiplier = 1
            
            if not clean_value:
                return
            
            value = float(clean_value) * multiplier
            
            # Map metric names to data keys
            if ('pe' in metric_name or 'p/e' in metric_name or 'price to earnings' in metric_name) and data['pe_ratio'] is None:
                data['pe_ratio'] = value
            elif ('pb' in metric_name or 'p/b' in metric_name or 'price to book' in metric_name) and data['pb_ratio'] is None:
                data['pb_ratio'] = value
            elif ('debt to equity' in metric_name or 'd/e' in metric_name or 'debt/equity' in metric_name or 'debt-to-equity' in metric_name) and data['debt_to_equity'] is None:
                data['debt_to_equity'] = value
            elif ('roe' in metric_name and 'roce' not in metric_name and 'return on equity' in metric_name) and data['roe'] is None:
                data['roe'] = value
            elif ('roce' in metric_name or 'return on capital employed' in metric_name) and data['roce'] is None:
                data['roce'] = value
            elif 'current ratio' in metric_name and data['current_ratio'] is None:
                data['current_ratio'] = value
            elif 'quick ratio' in metric_name and data['quick_ratio'] is None:
                data['quick_ratio'] = value
            elif ('eps' in metric_name or 'earnings per share' in metric_name) and data['eps'] is None:
                data['eps'] = value
            elif 'dividend yield' in metric_name and data['dividend_yield'] is None:
                data['dividend_yield'] = value
            elif ('market cap' in metric_name or 'mcap' in metric_name or 'market capitalization' in metric_name) and data['market_cap'] is None:
                data['market_cap'] = value
            elif ('revenue growth' in metric_name or 'sales growth' in metric_name) and data['revenue_growth'] is None:
                data['revenue_growth'] = value
            elif ('profit growth' in metric_name or 'net profit growth' in metric_name) and data['profit_growth'] is None:
                data['profit_growth'] = value
        except (ValueError, AttributeError) as e:
            logger.debug(f"Error parsing metric {metric_name} with value {metric_value}: {e}")
    
    def save_fundamental_data(self, scrip_id: str, data: Dict, fetch_date: Optional[date] = None) -> bool:
        """
        Save fundamental data to database
        
        Args:
            scrip_id: Stock symbol
            data: Dictionary containing fundamental metrics
            fetch_date: Date of fetch (defaults to today)
            
        Returns:
            True if saved successfully, False otherwise
        """
        if not data:
            return False
        
        if fetch_date is None:
            fetch_date = date.today()
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Extract raw_data if present
            raw_data = data.get('raw_data', {})
            if isinstance(raw_data, dict):
                raw_data_json = json.dumps(raw_data)
            else:
                raw_data_json = json.dumps({'data': str(raw_data)})
            
            cursor.execute("""
                INSERT INTO my_schema.fundamental_data (
                    scrip_id, fetch_date, pe_ratio, pb_ratio, debt_to_equity,
                    roe, roce, current_ratio, quick_ratio, eps,
                    revenue_growth, profit_growth, dividend_yield, market_cap, raw_data
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb
                )
                ON CONFLICT (scrip_id, fetch_date) 
                DO UPDATE SET
                    pe_ratio = EXCLUDED.pe_ratio,
                    pb_ratio = EXCLUDED.pb_ratio,
                    debt_to_equity = EXCLUDED.debt_to_equity,
                    roe = EXCLUDED.roe,
                    roce = EXCLUDED.roce,
                    current_ratio = EXCLUDED.current_ratio,
                    quick_ratio = EXCLUDED.quick_ratio,
                    eps = EXCLUDED.eps,
                    revenue_growth = EXCLUDED.revenue_growth,
                    profit_growth = EXCLUDED.profit_growth,
                    dividend_yield = EXCLUDED.dividend_yield,
                    market_cap = EXCLUDED.market_cap,
                    raw_data = EXCLUDED.raw_data
            """, (
                scrip_id, fetch_date,
                data.get('pe_ratio'),
                data.get('pb_ratio'),
                data.get('debt_to_equity'),
                data.get('roe'),
                data.get('roce'),
                data.get('current_ratio'),
                data.get('quick_ratio'),
                data.get('eps'),
                data.get('revenue_growth'),
                data.get('profit_growth'),
                data.get('dividend_yield'),
                data.get('market_cap'),
                raw_data_json
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"Saved fundamental data for {scrip_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving fundamental data for {scrip_id}: {e}")
            if conn:
                conn.rollback()
                conn.close()
            return False
    
    def has_fundamental_data(self, scrip_id: str) -> bool:
        """
        Check if fundamental data exists in the database for a stock
        
        Args:
            scrip_id: Stock symbol
            
        Returns:
            True if data exists, False otherwise
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Check if we have ANY fundamental data (regardless of age)
            cursor.execute("""
                SELECT COUNT(*) as data_count
                FROM my_schema.fundamental_data
                WHERE scrip_id = %s
            """, (scrip_id,))
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result and result[0] > 0:
                logger.debug(f"Fundamental data exists in database for {scrip_id}")
                return True
            
            # No data exists
            logger.info(f"No fundamental data found in database for {scrip_id}")
            return False
            
        except Exception as e:
            logger.warning(f"Error checking fundamental data existence for {scrip_id}: {e}")
            # If we can't check, assume no data exists
            return False
    
    def needs_fundamental_fetch(self, scrip_id: str, days_threshold: int = 30, force_refresh: bool = False) -> bool:
        """
        Check if fundamental data needs to be fetched from internet
        
        Args:
            scrip_id: Stock symbol
            days_threshold: Number of days threshold (default: 30 for monthly) - only used if force_refresh is False
            force_refresh: If True, always return True (force fetch)
            
        Returns:
            True if fetch is needed, False if data exists in database
        """
        if force_refresh:
            logger.info(f"Force refresh requested for {scrip_id}")
            return True
        
        # First check if data exists in database
        if self.has_fundamental_data(scrip_id):
            # Data exists, check if it's older than threshold
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT MAX(fetch_date) as last_fetch_date
                    FROM my_schema.fundamental_data
                    WHERE scrip_id = %s
                """, (scrip_id,))
                
                result = cursor.fetchone()
                cursor.close()
                conn.close()
                
                if result and result[0]:
                    last_fetch_date = result[0]
                    # Handle both date and datetime objects
                    if isinstance(last_fetch_date, datetime):
                        last_fetch_date = last_fetch_date.date()
                    days_since_fetch = (date.today() - last_fetch_date).days
                    
                    if days_since_fetch >= days_threshold:
                        logger.info(f"Fundamental data for {scrip_id} is {days_since_fetch} days old (>= {days_threshold}), needs refresh")
                        return True
                    else:
                        logger.info(f"Fundamental data for {scrip_id} is {days_since_fetch} days old (< {days_threshold}), skipping fetch - using existing data")
                        return False
            except Exception as e:
                logger.warning(f"Error checking fundamental data age for {scrip_id}: {e}")
                # If we can't check age, assume data is recent enough
                return False
        
        # No data exists in database, need to fetch
        logger.info(f"No fundamental data found in database for {scrip_id}, needs fetch from internet")
        return True
    
    def fetch_all_holdings_fundamentals(self, force_refresh: bool = False, days_threshold: int = 30, batch_size: int = 5, delay_between_batches: float = 0.0) -> List[Dict]:
        """
        Fetch fundamental data for all holdings in batches (only if data is older than threshold)
        
        Args:
            force_refresh: If True, fetch regardless of last fetch date
            days_threshold: Number of days threshold (default: 30 for monthly)
            batch_size: Number of stocks to process in each batch (default: 5)
            delay_between_batches: Delay in seconds between batches (default: 0.0)
        
        Returns:
            List of dictionaries with fetch results
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get distinct scrip_ids from holdings
            cursor.execute("""
                SELECT DISTINCT ms.scrip_id
                FROM my_schema.holdings h
                JOIN my_schema.master_scrips ms ON h.trading_symbol = ms.scrip_id
                WHERE ms.scrip_screener_code IS NOT NULL
            """)
            
            holdings = cursor.fetchall()
            cursor.close()
            conn.close()
            
            results = []
            fetched_count = 0
            skipped_count = 0
            total_stocks = len(holdings)
            
            # Clear cancellation flag at start
            self.set_cancellation_flag(False)
            
            # Process in batches
            cancelled = False
            for batch_idx in range(0, total_stocks, batch_size):
                # Check for cancellation before each batch
                if self.check_cancellation():
                    logger.info(f"Batch processing cancelled by user at batch {batch_idx // batch_size + 1}")
                    results.append({'scrip_id': 'BATCH_CANCELLED', 'status': 'cancelled', 'reason': 'User cancelled batch processing'})
                    cancelled = True
                    break
                
                batch = holdings[batch_idx:batch_idx + batch_size]
                logger.info(f"Processing batch {batch_idx // batch_size + 1} of {(total_stocks + batch_size - 1) // batch_size} ({len(batch)} stocks)")
                
                for (scrip_id,) in batch:
                    # Check for cancellation before each stock
                    if self.check_cancellation():
                        logger.info(f"Batch processing cancelled by user while processing {scrip_id}")
                        results.append({'scrip_id': 'BATCH_CANCELLED', 'status': 'cancelled', 'reason': 'User cancelled batch processing'})
                        cancelled = True
                        break
                    
                    if scrip_id:
                        try:
                            # First check if data exists in database
                            if self.has_fundamental_data(scrip_id):
                                # Data exists, check if we need to refresh (based on age or force_refresh)
                                if not self.needs_fundamental_fetch(scrip_id, days_threshold, force_refresh):
                                    logger.info(f"Skipping {scrip_id} - fundamental data exists in database and is recent (less than {days_threshold} days old)")
                                    results.append({'scrip_id': scrip_id, 'status': 'skipped', 'reason': f'Data exists in database and is recent (less than {days_threshold} days old)'})
                                    skipped_count += 1
                                    continue
                                else:
                                    logger.info(f"Data exists for {scrip_id} but is old (>= {days_threshold} days) or force_refresh is True, fetching from internet...")
                            else:
                                logger.info(f"No data in database for {scrip_id}, fetching from internet...")
                            
                            logger.info(f"Fetching fundamentals for {scrip_id} from internet ({fetched_count + skipped_count + 1}/{total_stocks})")
                            
                            # Check cancellation before starting fetch
                            if self.check_cancellation():
                                logger.info(f"Batch processing cancelled before fetching {scrip_id}")
                                results.append({'scrip_id': 'BATCH_CANCELLED', 'status': 'cancelled', 'reason': 'User cancelled batch processing'})
                                cancelled = True
                                break
                            
                            data = self.fetch_fundamental_data(scrip_id)
                            
                            # Check cancellation after fetch completes
                            if self.check_cancellation():
                                logger.info(f"Batch processing cancelled after fetching {scrip_id}")
                                results.append({'scrip_id': 'BATCH_CANCELLED', 'status': 'cancelled', 'reason': 'User cancelled batch processing'})
                                cancelled = True
                                break
                            
                            if data:
                                # Use today's date as fetch_date for trend analysis
                                fetch_date = date.today()
                                self.save_fundamental_data(scrip_id, data, fetch_date=fetch_date)
                                results.append({'scrip_id': scrip_id, 'status': 'success', 'data': data, 'fetch_date': str(fetch_date)})
                                fetched_count += 1
                            else:
                                results.append({'scrip_id': scrip_id, 'status': 'failed', 'error': 'No data returned from internet'})
                        except Exception as e:
                            logger.error(f"Error processing {scrip_id}: {e}")
                            results.append({'scrip_id': scrip_id, 'status': 'failed', 'error': str(e)})
                        
                        # Rate limiting - be respectful to screener.in
                        # Check cancellation during delay (check every second)
                        for _ in range(3):
                            if self.check_cancellation():
                                logger.info(f"Batch processing cancelled during delay after {scrip_id}")
                                results.append({'scrip_id': 'BATCH_CANCELLED', 'status': 'cancelled', 'reason': 'User cancelled batch processing'})
                                cancelled = True
                                break
                            time.sleep(1)
                        # If cancelled, break out of stock loop and outer loop
                        if cancelled:
                            break
                
                # Check for cancellation before delay between batches
                if cancelled:
                    break
                
                # Delay between batches to avoid overloading (only if delay > 0)
                if not cancelled and batch_idx + batch_size < total_stocks and delay_between_batches > 0:
                    logger.info(f"Waiting {delay_between_batches} seconds before next batch...")
                    # Check cancellation during delay (check every second)
                    for _ in range(int(delay_between_batches)):
                        if self.check_cancellation():
                            logger.info("Batch processing cancelled during delay")
                            cancelled = True
                            break
                        time.sleep(1)
                    if cancelled:
                        break
            
            logger.info(f"Fundamental fetch summary: {fetched_count} fetched, {skipped_count} skipped (recent data exists)")
            return results
            
        except Exception as e:
            logger.error(f"Error fetching holdings fundamentals: {e}")
            return []
    
    def fetch_nifty50_fundamentals(self, force_refresh: bool = False, days_threshold: int = 30, batch_size: int = 5, delay_between_batches: float = 0.0) -> List[Dict]:
        """
        Fetch fundamental data for all Nifty50 stocks in batches (only if data is older than threshold)
        
        Args:
            force_refresh: If True, fetch regardless of last fetch date
            days_threshold: Number of days threshold (default: 30 for monthly)
            batch_size: Number of stocks to process in each batch (default: 5)
            delay_between_batches: Delay in seconds between batches (default: 0.0)
        
        Returns:
            List of dictionaries with fetch results
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get Nifty50 stocks - typically identified by scrip_group or a specific list
            cursor.execute("""
                SELECT DISTINCT scrip_id
                FROM my_schema.master_scrips
                WHERE scrip_screener_code IS NOT NULL
                AND (scrip_group LIKE '%NIFTY50%' OR scrip_group LIKE '%NIFTY_50%' OR scrip_group = 'NIFTY50')
                AND scrip_id != 'NIFTY50'
                AND scrip_id != 'Nifty_50'
            """)
            
            nifty50_stocks = cursor.fetchall()
            
            # If no results, try to get from a predefined list or holdings
            if not nifty50_stocks:
                # Common Nifty50 stocks - you may want to maintain this list separately
                # For now, we'll try to get stocks that are likely Nifty50 based on market cap
                cursor.execute("""
                    SELECT DISTINCT scrip_id
                    FROM my_schema.master_scrips
                    WHERE scrip_screener_code IS NOT NULL
                    AND scrip_mcap > 50000  -- Large cap stocks likely to be in Nifty50
                    AND scrip_id NOT IN ('NIFTY50', 'Nifty_50', 'Nifty5')
                    ORDER BY scrip_mcap DESC
                    LIMIT 50
                """)
                nifty50_stocks = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            results = []
            fetched_count = 0
            skipped_count = 0
            total_stocks = len(nifty50_stocks)
            
            # Clear cancellation flag at start
            self.set_cancellation_flag(False)
            
            # Process in batches
            cancelled = False
            for batch_idx in range(0, total_stocks, batch_size):
                # Check for cancellation before each batch
                if self.check_cancellation():
                    logger.info(f"Nifty50 batch processing cancelled by user at batch {batch_idx // batch_size + 1}")
                    results.append({'scrip_id': 'BATCH_CANCELLED', 'status': 'cancelled', 'reason': 'User cancelled batch processing'})
                    cancelled = True
                    break
                
                batch = nifty50_stocks[batch_idx:batch_idx + batch_size]
                logger.info(f"Processing Nifty50 batch {batch_idx // batch_size + 1} of {(total_stocks + batch_size - 1) // batch_size} ({len(batch)} stocks)")
                
                for (scrip_id,) in batch:
                    # Check for cancellation before each stock
                    if self.check_cancellation():
                        logger.info(f"Nifty50 batch processing cancelled by user while processing {scrip_id}")
                        results.append({'scrip_id': 'BATCH_CANCELLED', 'status': 'cancelled', 'reason': 'User cancelled batch processing'})
                        break
                    
                    if scrip_id:
                        try:
                            # First check if data exists in database
                            if self.has_fundamental_data(scrip_id):
                                # Data exists, check if we need to refresh (based on age or force_refresh)
                                if not self.needs_fundamental_fetch(scrip_id, days_threshold, force_refresh):
                                    logger.info(f"Skipping {scrip_id} - fundamental data exists in database and is recent (less than {days_threshold} days old)")
                                    results.append({'scrip_id': scrip_id, 'status': 'skipped', 'reason': f'Data exists in database and is recent (less than {days_threshold} days old)'})
                                    skipped_count += 1
                                    continue
                                else:
                                    logger.info(f"Data exists for {scrip_id} but is old (>= {days_threshold} days) or force_refresh is True, fetching from internet...")
                            else:
                                logger.info(f"No data in database for {scrip_id}, fetching from internet...")
                            
                            logger.info(f"Fetching fundamentals for Nifty50 stock: {scrip_id} from internet ({fetched_count + skipped_count + 1}/{total_stocks})")
                            
                            # Check cancellation before starting fetch
                            if self.check_cancellation():
                                logger.info(f"Nifty50 batch processing cancelled before fetching {scrip_id}")
                                results.append({'scrip_id': 'BATCH_CANCELLED', 'status': 'cancelled', 'reason': 'User cancelled batch processing'})
                                cancelled = True
                                break
                            
                            data = self.fetch_fundamental_data(scrip_id)
                            
                            # Check cancellation after fetch completes
                            if self.check_cancellation():
                                logger.info(f"Nifty50 batch processing cancelled after fetching {scrip_id}")
                                results.append({'scrip_id': 'BATCH_CANCELLED', 'status': 'cancelled', 'reason': 'User cancelled batch processing'})
                                cancelled = True
                                break
                            
                            if data:
                                # Use today's date as fetch_date for trend analysis
                                fetch_date = date.today()
                                self.save_fundamental_data(scrip_id, data, fetch_date=fetch_date)
                                results.append({'scrip_id': scrip_id, 'status': 'success', 'data': data, 'fetch_date': str(fetch_date)})
                                fetched_count += 1
                            else:
                                results.append({'scrip_id': scrip_id, 'status': 'failed', 'error': 'No data returned from internet'})
                        except Exception as e:
                            logger.error(f"Error processing {scrip_id}: {e}")
                            results.append({'scrip_id': scrip_id, 'status': 'failed', 'error': str(e)})
                        
                        # Rate limiting - be respectful to screener.in
                        # Check cancellation during delay (check every second)
                        for _ in range(3):
                            if self.check_cancellation():
                                logger.info(f"Nifty50 batch processing cancelled during delay after {scrip_id}")
                                results.append({'scrip_id': 'BATCH_CANCELLED', 'status': 'cancelled', 'reason': 'User cancelled batch processing'})
                                cancelled = True
                                break
                            time.sleep(1)
                        # If cancelled, break out of stock loop and outer loop
                        if cancelled:
                            break
                
                # Check for cancellation before delay between batches
                if cancelled:
                    break
                
                # Delay between batches to avoid overloading (only if delay > 0)
                if not cancelled and batch_idx + batch_size < total_stocks and delay_between_batches > 0:
                    logger.info(f"Waiting {delay_between_batches} seconds before next batch...")
                    # Check cancellation during delay (check every second)
                    for _ in range(int(delay_between_batches)):
                        if self.check_cancellation():
                            logger.info("Nifty50 batch processing cancelled during delay")
                            cancelled = True
                            break
                        time.sleep(1)
                    if cancelled:
                        break
            
            logger.info(f"Nifty50 fundamental fetch summary: {fetched_count} fetched, {skipped_count} skipped (recent data exists)")
            return results
            
        except Exception as e:
            logger.error(f"Error fetching Nifty50 fundamentals: {e}")
            return []

