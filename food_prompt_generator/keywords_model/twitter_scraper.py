"""
Real-time Twitter Scraper
=========================

Uses Selenium and undetected-chromedriver to scrape real tweets from Twitter profiles.
Handles authentication, scrolling, and tweet extraction in real-time.
"""

import time
import logging
import re
from typing import List, Dict, Optional, Tuple, Union, Any
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, 
    NoSuchElementException, 
    StaleElementReferenceException,
    WebDriverException
)
import undetected_chromedriver as uc
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TwitterScraper:
    """Real-time Twitter scraper using Selenium"""
    
    def __init__(self):
        self.driver: Optional[Union[uc.Chrome, webdriver.Chrome]] = None
        self.wait: Optional[WebDriverWait] = None
        self.is_logged_in = False
        
    def setup_driver(self) -> bool:
        """Setup undetected Chrome driver"""
        try:
            logger.info("🚀 Setting up Chrome driver...")
            
            # Try multiple approaches for Chrome options
            try:
                # Approach 1: Minimal options
                options = uc.ChromeOptions()
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                options.add_argument('--disable-blink-features=AutomationControlled')
                
                # Create driver with minimal options
                self.driver = uc.Chrome(options=options)
                
            except Exception as e1:
                logger.warning(f"First approach failed: {e1}")
                try:
                    # Approach 2: No options
                    self.driver = uc.Chrome()
                except Exception as e2:
                    logger.warning(f"Second approach failed: {e2}")
                    try:
                        # Approach 3: Use regular selenium as fallback
                        from selenium.webdriver.chrome.service import Service
                        from selenium.webdriver.chrome.options import Options
                        
                        chrome_options = Options()
                        chrome_options.add_argument('--no-sandbox')
                        chrome_options.add_argument('--disable-dev-shm-usage')
                        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
                        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
                        chrome_options.add_experimental_option('useAutomationExtension', False)
                        
                        # Type ignore because we're intentionally using regular selenium as fallback
                        self.driver = webdriver.Chrome(options=chrome_options)  # type: ignore
                        logger.info("✅ Using regular Selenium Chrome driver")
                    except Exception as e3:
                        logger.error(f"All Chrome driver approaches failed: {e3}")
                        return False
            
            # Setup wait
            if self.driver is not None:
                self.wait = WebDriverWait(self.driver, 10)
            else:
                logger.error("❌ Driver is None after setup")
                return False
            
            logger.info("✅ Chrome driver setup successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup Chrome driver: {e}")
            return False
    
    def login_to_twitter(self, username: str, password: str) -> bool:
        """Login to Twitter using provided credentials"""
        try:
            logger.info(f"🔐 Logging in to Twitter as {username}")
            
            if not self.driver or not self.wait:
                logger.error("❌ Driver not initialized")
                return False
            
            # Navigate to Twitter login
            self.driver.get("https://twitter.com/login")
            time.sleep(5)  # Increased wait time
            
            # Try multiple selectors for username input
            username_input = None
            selectors = [
                'input[autocomplete="username"]',
                'input[name="text"]',
                'input[type="text"]',
                'input[data-testid="ocfEnterTextTextInput"]'
            ]
            
            for selector in selectors:
                try:
                    username_input = self.wait.until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                    logger.info(f"✅ Found username input with selector: {selector}")
                    break
                except TimeoutException:
                    continue
            
            if not username_input:
                logger.error("❌ Could not find username input field")
                return False
            
            username_input.clear()
            username_input.send_keys(username)
            time.sleep(1)
            
            # Try multiple ways to click Next
            next_clicked = False
            next_selectors = [
                "//span[text()='Next']",
                "//span[text()='Next']/..",
                "//div[@role='button' and contains(text(), 'Next')]",
                "//button[contains(text(), 'Next')]"
            ]
            
            for selector in next_selectors:
                try:
                    next_button = self.wait.until(
                        EC.element_to_be_clickable((By.XPATH, selector))
                    )
                    next_button.click()
                    next_clicked = True
                    logger.info(f"✅ Clicked Next with selector: {selector}")
                    break
                except (TimeoutException, Exception):
                    continue
            
            if not next_clicked:
                logger.error("❌ Could not click Next button")
                return False
            
            time.sleep(3)  # Increased wait time
            
            # Try multiple selectors for password input
            password_input = None
            password_selectors = [
                'input[name="password"]',
                'input[type="password"]',
                'input[data-testid="password"]'
            ]
            
            for selector in password_selectors:
                try:
                    password_input = self.wait.until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                    logger.info(f"✅ Found password input with selector: {selector}")
                    break
                except TimeoutException:
                    continue
            
            if not password_input:
                logger.error("❌ Could not find password input field")
                return False
            
            password_input.clear()
            password_input.send_keys(password)
            time.sleep(1)
            
            # Try multiple ways to click Login
            login_clicked = False
            login_selectors = [
                "//span[text()='Log in']",
                "//span[text()='Log in']/..",
                "//div[@role='button' and contains(text(), 'Log in')]",
                "//button[contains(text(), 'Log in')]"
            ]
            
            for selector in login_selectors:
                try:
                    login_button = self.wait.until(
                        EC.element_to_be_clickable((By.XPATH, selector))
                    )
                    login_button.click()
                    login_clicked = True
                    logger.info(f"✅ Clicked Login with selector: {selector}")
                    break
                except (TimeoutException, Exception):
                    continue
            
            if not login_clicked:
                logger.error("❌ Could not click Login button")
                return False
            
            time.sleep(8)  # Increased wait time for login to complete
            
            # Check if login successful
            current_url = self.driver.current_url
            logger.info(f"Current URL after login: {current_url}")
            
            if ("home" in current_url or "twitter.com" in current_url or "x.com" in current_url):
                self.is_logged_in = True
                logger.info("✅ Successfully logged in to Twitter")
                return True
            else:
                logger.error(f"❌ Login failed - current URL: {current_url}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Login failed: {e}")
            return False
    
    def scrape_profile_tweets(self, target_handle: str, max_tweets: int = 100) -> List[str]:
        """Scrape tweets from a specific Twitter profile"""
        try:
            if not self.is_logged_in or not self.driver:
                logger.error("❌ Not logged in to Twitter or driver not initialized")
                return []
            
            logger.info(f"📊 Scraping tweets from @{target_handle}")
            
            # Navigate to profile
            profile_url = f"https://twitter.com/{target_handle}"
            self.driver.get(profile_url)
            time.sleep(3)
            
            tweets = []
            last_height = self.driver.execute_script("return document.body.scrollHeight")
            
            while len(tweets) < max_tweets:
                # Find tweet elements
                tweet_elements = self.driver.find_elements(
                    By.CSS_SELECTOR, 
                    'article[data-testid="tweet"]'
                )
                
                for tweet_element in tweet_elements:
                    if len(tweets) >= max_tweets:
                        break
                    
                    try:
                        # Extract tweet text
                        tweet_text_element = tweet_element.find_element(
                            By.CSS_SELECTOR, 
                            'div[data-testid="tweetText"]'
                        )
                        tweet_text = tweet_text_element.text.strip()
                        
                        if tweet_text and tweet_text not in tweets:
                            tweets.append(tweet_text)
                            logger.info(f"📝 Scraped tweet {len(tweets)}: {tweet_text[:50]}...")
                    
                    except NoSuchElementException:
                        continue
                    except StaleElementReferenceException:
                        continue
                
                # Scroll down to load more tweets
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    logger.info("📄 Reached end of profile")
                    break
                last_height = new_height
            
            logger.info(f"✅ Successfully scraped {len(tweets)} tweets from @{target_handle}")
            return tweets
            
        except Exception as e:
            logger.error(f"❌ Failed to scrape tweets: {e}")
            return []
    
    def scrape_trending_tweets(self, max_tweets: int = 50) -> List[str]:
        """Scrape trending tweets from Twitter home"""
        try:
            if not self.is_logged_in or not self.driver:
                logger.error("❌ Not logged in to Twitter or driver not initialized")
                return []
            
            logger.info("📊 Scraping trending tweets from home timeline")
            
            # Navigate to home
            self.driver.get("https://twitter.com/home")
            time.sleep(3)
            
            tweets = []
            last_height = self.driver.execute_script("return document.body.scrollHeight")
            
            while len(tweets) < max_tweets:
                # Find tweet elements
                tweet_elements = self.driver.find_elements(
                    By.CSS_SELECTOR, 
                    'article[data-testid="tweet"]'
                )
                
                for tweet_element in tweet_elements:
                    if len(tweets) >= max_tweets:
                        break
                    
                    try:
                        # Extract tweet text
                        tweet_text_element = tweet_element.find_element(
                            By.CSS_SELECTOR, 
                            'div[data-testid="tweetText"]'
                        )
                        tweet_text = tweet_text_element.text.strip()
                        
                        if tweet_text and tweet_text not in tweets:
                            tweets.append(tweet_text)
                            logger.info(f"📝 Scraped trending tweet {len(tweets)}: {tweet_text[:50]}...")
                    
                    except NoSuchElementException:
                        continue
                    except StaleElementReferenceException:
                        continue
                
                # Scroll down to load more tweets
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    logger.info("📄 Reached end of timeline")
                    break
                last_height = new_height
            
            logger.info(f"✅ Successfully scraped {len(tweets)} trending tweets")
            return tweets
            
        except Exception as e:
            logger.error(f"❌ Failed to scrape trending tweets: {e}")
            return []
    
    def search_tweets(self, query: str, max_tweets: int = 50) -> List[str]:
        """Search for tweets with specific keywords"""
        try:
            if not self.is_logged_in or not self.driver:
                logger.error("❌ Not logged in to Twitter or driver not initialized")
                return []
            
            logger.info(f"🔍 Searching tweets for: {query}")
            
            # Navigate to search
            search_url = f"https://twitter.com/search?q={query}&src=typed_query&f=live"
            self.driver.get(search_url)
            time.sleep(3)
            
            tweets = []
            last_height = self.driver.execute_script("return document.body.scrollHeight")
            
            while len(tweets) < max_tweets:
                # Find tweet elements
                tweet_elements = self.driver.find_elements(
                    By.CSS_SELECTOR, 
                    'article[data-testid="tweet"]'
                )
                
                for tweet_element in tweet_elements:
                    if len(tweets) >= max_tweets:
                        break
                    
                    try:
                        # Extract tweet text
                        tweet_text_element = tweet_element.find_element(
                            By.CSS_SELECTOR, 
                            'div[data-testid="tweetText"]'
                        )
                        tweet_text = tweet_text_element.text.strip()
                        
                        if tweet_text and tweet_text not in tweets:
                            tweets.append(tweet_text)
                            logger.info(f"📝 Scraped search tweet {len(tweets)}: {tweet_text[:50]}...")
                    
                    except NoSuchElementException:
                        continue
                    except StaleElementReferenceException:
                        continue
                
                # Scroll down to load more tweets
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    logger.info("📄 Reached end of search results")
                    break
                last_height = new_height
            
            logger.info(f"✅ Successfully scraped {len(tweets)} search tweets for '{query}'")
            return tweets
            
        except Exception as e:
            logger.error(f"❌ Failed to search tweets: {e}")
            return []
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.driver:
                self.driver.quit()
                logger.info("✅ Chrome driver closed")
        except Exception as e:
            logger.error(f"❌ Error closing driver: {e}")

def main():
    """Test the Twitter scraper"""
    scraper = TwitterScraper()
    
    try:
        # Setup driver
        if not scraper.setup_driver():
            return
        
        # Login (you'll need to provide real credentials)
        username = input("Enter Twitter username (with @): ")
        password = input("Enter Twitter password: ")
        
        if not scraper.login_to_twitter(username, password):
            return
        
        # Test scraping
        target_handle = input("Enter target handle to scrape (without @): ")
        tweets = scraper.scrape_profile_tweets(target_handle, max_tweets=20)
        
        print(f"\n📊 Scraped {len(tweets)} tweets:")
        for i, tweet in enumerate(tweets, 1):
            print(f"{i}. {tweet[:100]}...")
    
    finally:
        scraper.cleanup()

if __name__ == "__main__":
    main() 