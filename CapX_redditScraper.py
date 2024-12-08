import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import re
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RedditStockScraper:
    def __init__(self, subreddits = ['investing', 'stocks', 'wallstreetbets', 'stockmarket', 'finance',   'NIFTY', 'SENSEX','cryptocurrency', 'algo_trading', 'valueinvesting', 'dividends', 'daytrading',   'pennystocks', 'options', 'FIRE', 'Stock_Picks', 'IndiaInvestments', 'Trading', 'TheDividendGrowth',    'ETFs', 'InvestingIndia', 'Economics', 'LeveragedTrading', 'growthinvesting', 'retirement',    'StockMarketIndia', 'InvestmentClub', 'StockAnalysis', 'microcapstocks', 'spacs',    'CryptoCurrencyTrading', 'AlternativeInvestments', 'mutualfunds', 'HFT'],

                 chrome_driver_path='/usr/local/bin/chromedriver'):
        """
        Initialize the Reddit Stock Market Scraper
        
        :param subreddits: List of subreddits to scrape
        :param chrome_driver_path: Path to ChromeDriver executable
        """
        chrome_options = Options()
        chrome_options.add_argument("--headless") 
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36")
        
        try:
            self.driver = webdriver.Chrome(
                service=Service(chrome_driver_path), 
                options=chrome_options
            )
        except Exception as e:
            logger.error(f"Failed to initialize WebDriver: {e}")
            raise
        
        self.subreddits = subreddits
        self.scraped_data = []

    def _clean_text(self, text):
        """
        Clean and normalize text data
        
        :param text: Raw text to clean
        :return: Cleaned text
        """
        if not text:
            return ""
        cleaned = re.sub(r'\s+', ' ', str(text))
        return cleaned.strip()

    def scrape_subreddit(self, subreddit, max_posts=50):
        """
        Scrape posts from a specific subreddit
        
        :param subreddit: Subreddit name to scrape
        :param max_posts: Maximum number of posts to scrape
        """
        url = f"https://old.reddit.com/r/{subreddit}/top/?sort=top&t=week"
        logger.info(f"Scraping subreddit: {subreddit}")
        
        try:
            self.driver.get(url)
            
            time.sleep(5)
            
            for _ in range(3):
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
            
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            posts = soup.find_all('div', class_=re.compile(r'thing'))[:max_posts]
            
            logger.info(f"Found {len(posts)} posts in {subreddit}")
            
            for post in posts:
                try:
                    title_elem = post.find('a', class_='title')
                    title = self._clean_text(title_elem.text) if title_elem else "No Title"
                    
                    upvotes_elem = post.find('div', class_=re.compile(r'score'))
                    upvotes = self._clean_text(upvotes_elem.text) if upvotes_elem else "0"
                    
                    comments_elem = post.find('a', class_='comments')
                    comments_count = self._clean_text(comments_elem.text) if comments_elem else "0"
                    
                    stock_tickers = re.findall(r'\$[A-Z]{1,5}', title)
                    sentiment = self._determine_sentiment(title)
                    
                    post_data = {
                        'subreddit': subreddit,
                        'title': title,
                        'upvotes': upvotes,
                        'comments': comments_count,
                        'stock_tickers': ', '.join(stock_tickers) if stock_tickers else 'N/A',
                        'sentiment': sentiment,
                        'timestamp': datetime.now()
                    }
                    
                    self.scraped_data.append(post_data)
                
                except Exception as post_error:
                    logger.error(f"Error processing individual post: {post_error}")
        
        except Exception as e:
            logger.error(f"Error scraping {subreddit}: {e}")
    
    def _determine_sentiment(self, text):
        """
        Basic sentiment analysis for stock-related posts
        
        :param text: Text to analyze
        :return: Sentiment score
        """
        if not text:
            return "Neutral"
        
        positive_words = ['buy', 'bullish', 'moon', 'gain', 'profit', 'opportunity']
        negative_words = ['sell', 'bearish', 'crash', 'loss', 'risk', 'dump']
        
        text = text.lower()
        
        positive_count = sum(word in text for word in positive_words)
        negative_count = sum(word in text for word in negative_words)
        
        if positive_count > negative_count:
            return 'Positive'
        elif negative_count > positive_count:
            return 'Negative'
        else:
            return 'Neutral'
    
    def run_scraper(self):
        """
        Run scraper for all specified subreddits
        """
        for subreddit in self.subreddits:
            self.scrape_subreddit(subreddit)
    
    def export_to_csv(self, filename='reddit_stock_data.csv'):
        """
        Export scraped data to CSV
        
        :param filename: Output filename
        """
        if not self.scraped_data:
            logger.warning("No data to export. Scraping might have failed.")
            return
        
        try:
            df = pd.DataFrame(self.scraped_data)
            df.to_csv(filename, index=False)
            logger.info(f"Data exported to {filename}")
            logger.info(f"Total rows exported: {len(df)}")
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
    
    def close(self):
        """
        Close the WebDriver
        """
        if hasattr(self, 'driver'):
            self.driver.quit()

def main():

    scraper = RedditStockScraper(
        subreddits = ['investing', 'stocks', 'wallstreetbets', 'stockmarket', 'finance',   'NIFTY', 'SENSEX','cryptocurrency', 'algo_trading', 'valueinvesting', 'dividends', 'daytrading',   'pennystocks', 'options', 'FIRE', 'Stock_Picks', 'IndiaInvestments', 'Trading', 'TheDividendGrowth',    'ETFs', 'InvestingIndia', 'Economics', 'LeveragedTrading', 'growthinvesting', 'retirement',    'StockMarketIndia', 'InvestmentClub', 'StockAnalysis', 'microcapstocks', 'spacs',    'CryptoCurrencyTrading', 'AlternativeInvestments', 'mutualfunds', 'HFT'],
        chrome_driver_path='/usr/local/bin/chromedriver' 
    )
    
    try:

        scraper.run_scraper()
        
        scraper.export_to_csv()
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    
    finally:
        scraper.close()

if __name__ == "__main__":
    main()