"""
News Sentiment Analyzer
Fetches news articles and performs sentiment analysis
"""

import requests
from bs4 import BeautifulSoup
import logging
from typing import Dict, List, Optional
from datetime import date, datetime, timedelta
import json
import time
import urllib.parse
import psycopg2
from psycopg2.extras import RealDictCursor
from common.Boilerplate import get_db_connection

# Try to import sentiment analysis libraries
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logging.warning("vaderSentiment not available. Install with: pip install vaderSentiment")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logging.warning("textblob not available. Install with: pip install textblob")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsSentimentAnalyzer:
    """
    Fetches news articles and performs sentiment analysis
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Initialize sentiment analyzers
        if VADER_AVAILABLE:
            self.vader_analyzer = SentimentIntensityAnalyzer()
        else:
            self.vader_analyzer = None
        
        # News sources
        self.news_sources = {
            'moneycontrol': 'https://www.moneycontrol.com/news/business/stocks/',
            'economic_times': 'https://economictimes.indiatimes.com/markets/stocks/news',
            'business_standard': 'https://www.business-standard.com/markets/news',
        }
    
    def fetch_news_articles(self, scrip_id: str, days: int = 7) -> List[Dict]:
        """
        Fetch news articles for a stock
        
        Args:
            scrip_id: Stock symbol (e.g., 'RELIANCE')
            days: Number of days to look back for news
            
        Returns:
            List of dictionaries containing news articles
        """
        articles = []
        
        # Search for news on multiple sources
        search_queries = [
            f"{scrip_id} stock",
            f"{scrip_id} share price",
            f"{scrip_id} company news",
        ]
        
        for query in search_queries:
            # Try Moneycontrol
            moneycontrol_articles = self._fetch_from_moneycontrol(scrip_id, query, days)
            articles.extend(moneycontrol_articles)
            
            # Try Economic Times
            et_articles = self._fetch_from_economic_times(scrip_id, query, days)
            articles.extend(et_articles)
            
            # Rate limiting
            time.sleep(1)
        
        # Remove duplicates based on title
        seen_titles = set()
        unique_articles = []
        for article in articles:
            title = article.get('title', '').lower().strip()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_articles.append(article)
        
        return unique_articles[:20]  # Limit to 20 most recent articles
    
    def _fetch_from_moneycontrol(self, scrip_id: str, query: str, days: int) -> List[Dict]:
        """Fetch news from Moneycontrol"""
        articles = []
        
        # Try multiple URL formats for Moneycontrol
        url_attempts = [
            f"https://www.moneycontrol.com/news/tags/{scrip_id}.html",  # Tag-based URL
            f"https://www.moneycontrol.com/news/business/stocks/{scrip_id.lower()}-news.html",  # Stock-specific page
            f"https://www.moneycontrol.com/news/search-result.html?q={query}",  # Search URL (may not work)
        ]
        
        for search_url in url_attempts:
            try:
                logger.debug(f"Trying Moneycontrol URL: {search_url}")
                response = self.session.get(search_url, timeout=30, allow_redirects=True)
                
                # Check if we got a valid page
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Try multiple selectors for news articles
                    news_items = (
                        soup.find_all('div', class_='news_list') or
                        soup.find_all('li', class_='clearfix') or
                        soup.find_all('div', class_='news-item') or
                        soup.find_all('article') or
                        soup.find_all('div', class_='news-card') or
                        soup.find_all('div', class_='story')
                    )
                    
                    if news_items:
                        logger.debug(f"Found {len(news_items)} news items from Moneycontrol")
                        
                        for item in news_items[:10]:  # Limit to 10 per source
                            try:
                                # Try multiple selectors for title
                                title_elem = (
                                    item.find('h2') or
                                    item.find('h3') or
                                    item.find('a', class_='title') or
                                    item.find('a', class_='news-title') or
                                    item.find('a')
                                )
                                
                                if not title_elem:
                                    continue
                                
                                title = title_elem.get_text(strip=True)
                                if not title:
                                    continue
                                
                                # Get link
                                link = title_elem.get('href', '') if title_elem.name == 'a' else ''
                                if not link:
                                    link_elem = item.find('a')
                                    if link_elem:
                                        link = link_elem.get('href', '')
                                
                                if not link:
                                    continue
                                
                                if not link.startswith('http'):
                                    link = f"https://www.moneycontrol.com{link}" if link.startswith('/') else f"https://www.moneycontrol.com/{link}"
                                
                                # Get article date
                                date_elem = (
                                    item.find('span', class_='date') or
                                    item.find('time') or
                                    item.find('div', class_='date') or
                                    item.find('span', class_='time')
                                )
                                article_date = date.today()
                                if date_elem:
                                    date_text = date_elem.get_text(strip=True)
                                    # Try to parse date in various formats
                                    date_formats = ['%b %d, %Y', '%d %b %Y', '%Y-%m-%d', '%d-%m-%Y', '%B %d, %Y']
                                    for fmt in date_formats:
                                        try:
                                            article_date = datetime.strptime(date_text, fmt).date()
                                            break
                                        except:
                                            continue
                                
                                # Get content snippet
                                content_elem = (
                                    item.find('p') or
                                    item.find('div', class_='content') or
                                    item.find('div', class_='summary') or
                                    item.find('div', class_='description')
                                )
                                content = content_elem.get_text(strip=True) if content_elem else ''
                                
                                # Filter articles by date (only include articles within the specified days)
                                days_ago = (date.today() - article_date).days
                                if days_ago > days:
                                    continue
                                
                                articles.append({
                                    'title': title,
                                    'content': content,
                                    'source': 'moneycontrol',
                                    'url': link,
                                    'article_date': article_date,
                                    'scrip_id': scrip_id
                                })
                            except Exception as e:
                                logger.debug(f"Error parsing Moneycontrol article: {e}")
                                continue
                        
                        # If we found articles, break out of URL attempts
                        if articles:
                            break
                else:
                    logger.debug(f"Moneycontrol URL returned status {response.status_code}: {search_url}")
                    
            except requests.exceptions.RequestException as e:
                logger.debug(f"Failed to fetch from Moneycontrol URL {search_url}: {e}")
                continue
            except Exception as e:
                logger.debug(f"Unexpected error with Moneycontrol URL {search_url}: {e}")
                continue
        
        if not articles:
            logger.debug(f"No articles found from Moneycontrol for {scrip_id}")
        
        return articles
    
    def _fetch_from_economic_times(self, scrip_id: str, query: str, days: int) -> List[Dict]:
        """Fetch news from Economic Times"""
        articles = []
        
        # Try multiple URL formats for Economic Times
        url_attempts = [
            f"https://economictimes.indiatimes.com/topic/{scrip_id}",
            f"https://economictimes.indiatimes.com/topic/{scrip_id.lower()}",
            f"https://economictimes.indiatimes.com/markets/stocks/news",
        ]
        
        for search_url in url_attempts:
            try:
                logger.debug(f"Trying Economic Times URL: {search_url}")
                response = self.session.get(search_url, timeout=30, allow_redirects=True)
                
                # Check if we got a valid page
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Try multiple selectors for news articles
                    news_items = (
                        soup.find_all('div', class_='eachStory') or
                        soup.find_all('li', class_='article') or
                        soup.find_all('article') or
                        soup.find_all('div', class_='story') or
                        soup.find_all('div', class_='news-item')
                    )
                    
                    if news_items:
                        logger.debug(f"Found {len(news_items)} news items from Economic Times")
                        
                        for item in news_items[:10]:  # Limit to 10 per source
                            try:
                                # Try multiple selectors for title
                                title_elem = (
                                    item.find('h3') or
                                    item.find('h2') or
                                    item.find('a', class_='title') or
                                    item.find('a', class_='news-title') or
                                    item.find('a')
                                )
                                
                                if not title_elem:
                                    continue
                                
                                title = title_elem.get_text(strip=True)
                                if not title:
                                    continue
                                
                                # Check if title contains the scrip_id (for topic pages)
                                if '/topic/' in search_url and scrip_id.upper() not in title.upper():
                                    continue
                                
                                # Get link
                                link = title_elem.get('href', '') if title_elem.name == 'a' else ''
                                if not link:
                                    link_elem = item.find('a')
                                    if link_elem:
                                        link = link_elem.get('href', '')
                                
                                if not link:
                                    continue
                                
                                if not link.startswith('http'):
                                    link = f"https://economictimes.indiatimes.com{link}" if link.startswith('/') else f"https://economictimes.indiatimes.com/{link}"
                                
                                # Get article date
                                date_elem = (
                                    item.find('time') or
                                    item.find('span', class_='date') or
                                    item.find('div', class_='date') or
                                    item.find('span', class_='time')
                                )
                                article_date = date.today()
                                if date_elem:
                                    date_text = date_elem.get_text(strip=True)
                                    # Try to parse date in various formats
                                    date_formats = ['%b %d, %Y', '%d %b %Y', '%Y-%m-%d', '%d-%m-%Y', '%B %d, %Y']
                                    for fmt in date_formats:
                                        try:
                                            article_date = datetime.strptime(date_text, fmt).date()
                                            break
                                        except:
                                            continue
                                
                                # Get content snippet
                                content_elem = (
                                    item.find('p') or
                                    item.find('div', class_='summary') or
                                    item.find('div', class_='content') or
                                    item.find('div', class_='description')
                                )
                                content = content_elem.get_text(strip=True) if content_elem else ''
                                
                                # Filter articles by date (only include articles within the specified days)
                                days_ago = (date.today() - article_date).days
                                if days_ago > days:
                                    continue
                                
                                articles.append({
                                    'title': title,
                                    'content': content,
                                    'source': 'economic_times',
                                    'url': link,
                                    'article_date': article_date,
                                    'scrip_id': scrip_id
                                })
                            except Exception as e:
                                logger.debug(f"Error parsing Economic Times article: {e}")
                                continue
                        
                        # If we found articles, break out of URL attempts
                        if articles:
                            break
                else:
                    logger.debug(f"Economic Times URL returned status {response.status_code}: {search_url}")
                    
            except requests.exceptions.RequestException as e:
                logger.debug(f"Failed to fetch from Economic Times URL {search_url}: {e}")
                continue
            except Exception as e:
                logger.debug(f"Unexpected error with Economic Times URL {search_url}: {e}")
                continue
        
        if not articles:
            logger.debug(f"No articles found from Economic Times for {scrip_id}")
        
        return articles
    
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment of text using VADER or TextBlob
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment_score (-1 to 1) and sentiment_label
        """
        if not text:
            return {
                'sentiment_score': 0.0,
                'sentiment_label': 'neutral'
            }
        
        sentiment_score = 0.0
        sentiment_label = 'neutral'
        
        # Use VADER if available (better for financial news)
        if self.vader_analyzer:
            try:
                scores = self.vader_analyzer.polarity_scores(text)
                sentiment_score = scores['compound']  # Range: -1 to 1
                
                if sentiment_score >= 0.05:
                    sentiment_label = 'positive'
                elif sentiment_score <= -0.05:
                    sentiment_label = 'negative'
                else:
                    sentiment_label = 'neutral'
            except Exception as e:
                logger.warning(f"VADER sentiment analysis error: {e}")
        
        # Fallback to TextBlob if VADER not available or failed
        if sentiment_score == 0.0 and TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity  # Range: -1 to 1
                sentiment_score = polarity
                
                if polarity >= 0.1:
                    sentiment_label = 'positive'
                elif polarity <= -0.1:
                    sentiment_label = 'negative'
                else:
                    sentiment_label = 'neutral'
            except Exception as e:
                logger.warning(f"TextBlob sentiment analysis error: {e}")
        
        return {
            'sentiment_score': round(sentiment_score, 4),
            'sentiment_label': sentiment_label
        }
    
    def calculate_aggregate_sentiment(self, articles: List[Dict]) -> float:
        """
        Calculate weighted average sentiment from articles
        
        Args:
            articles: List of article dictionaries with sentiment_score
            
        Returns:
            Aggregate sentiment score (-1 to 1)
        """
        if not articles:
            return 0.0
        
        # Weight articles by recency (more recent = higher weight)
        total_weight = 0.0
        weighted_sum = 0.0
        
        today = date.today()
        
        for article in articles:
            sentiment_score = article.get('sentiment_score', 0.0)
            article_date = article.get('article_date', today)
            
            # Calculate weight based on recency (exponential decay)
            days_ago = (today - article_date).days
            weight = max(0.1, 1.0 / (1.0 + days_ago * 0.2))  # Decay factor
            
            weighted_sum += sentiment_score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        aggregate_score = weighted_sum / total_weight
        
        # Normalize to -1 to 1 range
        return max(-1.0, min(1.0, aggregate_score))
    
    def save_news_sentiment(self, scrip_id: str, articles: List[Dict], aggregate_score: float) -> bool:
        """
        Save news articles and sentiment scores to database
        
        Args:
            scrip_id: Stock symbol
            articles: List of article dictionaries
            aggregate_score: Aggregate sentiment score
            
        Returns:
            True if saved successfully, False otherwise
        """
        if not articles:
            return False
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            saved_count = 0
            for article in articles:
                try:
                    sentiment_score = article.get('sentiment_score', 0.0)
                    sentiment_label = article.get('sentiment_label', 'neutral')
                    article_date = article.get('article_date', date.today())
                    source = article.get('source', 'unknown')
                    title = article.get('title', '')
                    content = article.get('content', '')
                    
                    raw_data = {
                        'url': article.get('url', ''),
                        'fetch_timestamp': datetime.now().isoformat()
                    }
                    
                    cursor.execute("""
                        INSERT INTO my_schema.news_sentiment (
                            scrip_id, article_date, source, title, content,
                            sentiment_score, sentiment_label, raw_data
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s::jsonb
                        )
                        ON CONFLICT (scrip_id, article_date, source, title) 
                        DO UPDATE SET
                            sentiment_score = EXCLUDED.sentiment_score,
                            sentiment_label = EXCLUDED.sentiment_label,
                            content = EXCLUDED.content,
                            raw_data = EXCLUDED.raw_data
                    """, (
                        scrip_id, article_date, source, title, content,
                        sentiment_score, sentiment_label, json.dumps(raw_data)
                    ))
                    
                    saved_count += 1
                except Exception as e:
                    logger.warning(f"Error saving article for {scrip_id}: {e}")
                    continue
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"Saved {saved_count} news articles for {scrip_id} with aggregate score: {aggregate_score:.4f}")
            return saved_count > 0
            
        except Exception as e:
            logger.error(f"Error saving news sentiment for {scrip_id}: {e}")
            if conn:
                conn.rollback()
                conn.close()
            return False
    
    def fetch_and_analyze(self, scrip_id: str, days: int = 7) -> Optional[Dict]:
        """
        Fetch news articles, analyze sentiment, and save results
        
        Args:
            scrip_id: Stock symbol
            days: Number of days to look back
            
        Returns:
            Dictionary with aggregate sentiment score and article count
        """
        try:
            # Fetch articles
            articles = self.fetch_news_articles(scrip_id, days)
            
            if not articles:
                logger.warning(f"No news articles found for {scrip_id}")
                return None
            
            # Analyze sentiment for each article
            for article in articles:
                # Combine title and content for sentiment analysis
                text = f"{article.get('title', '')} {article.get('content', '')}"
                sentiment = self.analyze_sentiment(text)
                article['sentiment_score'] = sentiment['sentiment_score']
                article['sentiment_label'] = sentiment['sentiment_label']
            
            # Calculate aggregate sentiment
            aggregate_score = self.calculate_aggregate_sentiment(articles)
            
            # Save to database
            self.save_news_sentiment(scrip_id, articles, aggregate_score)
            
            return {
                'scrip_id': scrip_id,
                'aggregate_sentiment_score': aggregate_score,
                'article_count': len(articles),
                'articles': articles
            }
            
        except Exception as e:
            logger.error(f"Error in fetch_and_analyze for {scrip_id}: {e}")
            return None

