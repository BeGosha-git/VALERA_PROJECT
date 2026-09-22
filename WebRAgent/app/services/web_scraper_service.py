import os
import logging
import requests
from bs4 import BeautifulSoup
import html2text
import time
from urllib.parse import urlparse
from fake_useragent import UserAgent

# Configure logging
logger = logging.getLogger(__name__)

class WebScraperService:
    """Service for scraping web content from URLs"""
    
    def __init__(self):
        """Initialize the web scraper with configuration from environment variables"""
        # Get user agent configuration - either use a specific one or rotate with fake_useragent
        self.user_agent_mode = os.environ.get('SCRAPER_USER_AGENT_MODE', 'fixed')  # Default to fixed to be safe
        self.user_agent = os.environ.get('SCRAPER_USER_AGENT', 'RAGSystem/1.0 (+https://github.com/yourusername/ragfromscratch)')
        
        # Rate limiting config
        self.rate_limit = float(os.environ.get('SCRAPER_RATE_LIMIT', 1.0))  # Seconds between requests
        self.timeout = int(os.environ.get('SCRAPER_TIMEOUT', 10))  # Seconds to wait for response
        
        # Content limits
        self.max_content_length = int(os.environ.get('SCRAPER_MAX_CONTENT_LENGTH', 100000))  # Chars
        
        # Initialize fake_useragent if we're using rotating agents
        if self.user_agent_mode == 'rotate':
            try:
                self.ua = UserAgent()
                logger.info("Successfully initialized UserAgent rotation")
            except Exception as e:
                logger.warning(f"Failed to initialize UserAgent rotation: {e}. Falling back to fixed user agent.")
                self.user_agent_mode = 'fixed'
        
        logger.info(f"Initialized WebScraperService with user agent mode: {self.user_agent_mode}")
    
    def get_user_agent(self):
        """Get a user agent string based on the configured mode"""
        if self.user_agent_mode == 'rotate':
            try:
                return self.ua.random
            except:
                return self.user_agent
        return self.user_agent
    
    def scrape_url(self, url):
        """
        Scrape content from a URL
        
        Args:
            url (str): The URL to scrape
            
        Returns:
            dict: Scraped content with metadata or None if scraping failed
        """
        try:
            # Check if URL is valid
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                logger.warning(f"Invalid URL format: {url}")
                return None
            
            logger.info(f"Scraping content from URL: {url}")
            
            # Set up headers with user agent
            headers = {
                'User-Agent': self.get_user_agent(),
                'Accept': 'text/html,application/xhtml+xml,application/xml',
                'Accept-Language': 'en-US,en;q=0.9',
            }
            
            # Make the request
            response = requests.get(
                url,
                headers=headers,
                timeout=self.timeout,
                stream=True  # Stream to handle large responses
            )
            
            # Check response status
            response.raise_for_status()
            
            # Determine content type
            content_type = response.headers.get('Content-Type', '').lower()
            
            # If not HTML, return limited info
            if 'text/html' not in content_type:
                logger.info(f"URL is not HTML (Content-Type: {content_type}): {url}")
                return {
                    'url': url,
                    'title': url.split('/')[-1] or url,
                    'content': f"[Non-HTML content: {content_type}]",
                    'content_type': content_type,
                    'success': False
                }
            
            # Parse the HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = ''
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
            else:
                # Try to find an h1 as fallback
                h1 = soup.find('h1')
                if h1 and h1.get_text():
                    title = h1.get_text().strip()
                else:
                    title = url.split('/')[-1] or url
            
            # Remove unwanted elements that often contain noise
            for element in soup.select('nav, footer, header, aside, script, style, noscript, iframe, svg, button, form'):
                element.decompose()
            
            # Get main content - try article or main tags first, then body
            main_content = soup.select_one('article, main, #content, .content, .article, .post')
            if not main_content:
                main_content = soup.body
            
            # Convert to plain text
            if main_content:
                h = html2text.HTML2Text()
                h.ignore_links = False
                h.ignore_images = True
                h.ignore_tables = False
                h.ignore_emphasis = True
                
                text_content = h.handle(str(main_content))
                
                # Limit content length
                if len(text_content) > self.max_content_length:
                    text_content = text_content[:self.max_content_length] + "... [Content truncated]"
            else:
                text_content = "[No content extracted]"
            
            # Rate limiting
            time.sleep(self.rate_limit)
            
            # Return the scraped content
            return {
                'url': url,
                'title': title,
                'content': text_content,
                'content_type': content_type,
                'success': True
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error when scraping {url}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error scraping URL {url}: {str(e)}")
            return None
    
    def scrape_urls(self, urls, max_urls=5):
        """
        Scrape content from multiple URLs
        
        Args:
            urls (list): List of URLs to scrape
            max_urls (int): Maximum number of URLs to scrape
            
        Returns:
            list: List of scraped content dictionaries
        """
        results = []
        
        # Limit the number of URLs to scrape
        urls_to_scrape = urls[:max_urls]
        
        for url in urls_to_scrape:
            result = self.scrape_url(url)
            if result and result.get('success', False):
                results.append(result)
        
        return results