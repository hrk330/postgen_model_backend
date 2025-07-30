"""
Enhanced Twitter Keywords Extraction System
==========================================

A robust, efficient, and secure system for extracting keywords from Twitter data.
Features improved error handling, security, scalability, and performance.

Author: Enhanced Version
Date: 2024
"""

import time
import csv
import re
import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass
from datetime import datetime
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue
import hashlib
import pickle

# Optional imports with fallbacks
try:
    import emoji
    EMOJI_AVAILABLE = True
except ImportError:
    EMOJI_AVAILABLE = False
    print("⚠️  emoji package not available. Install with: pip install emoji")

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
except ImportError:
    NLTK_AVAILABLE = False
    print("⚠️  NLTK not available. Install with: pip install nltk")

try:
    import spacy
    SPACY_AVAILABLE = True
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("⚠️  SpaCy model not found. Install with: python -m spacy download en_core_web_sm")
        SPACY_AVAILABLE = False
except ImportError:
    SPACY_AVAILABLE = False
    print("⚠️  SpaCy not available. Install with: pip install spacy")

try:
    from keybert import KeyBERT
    KEYBERT_AVAILABLE = True
except ImportError:
    KEYBERT_AVAILABLE = False
    print("⚠️  KeyBERT not available. Install with: pip install keybert")

# Web scraping imports (optional)
try:
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import StaleElementReferenceException, TimeoutException
    import undetected_chromedriver as uc
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("⚠️  Selenium not available. Install with: pip install selenium undetected-chromedriver")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('keywords_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ExtractionConfig:
    """Configuration for keyword extraction"""
    max_tweets: int = 100
    min_keyword_score: float = 0.3
    ngram_range: Tuple[int, int] = (1, 2)
    top_n_keywords: int = 1000
    scroll_timeout: int = 10
    scroll_attempts: int = 20
    batch_size: int = 10
    cache_enabled: bool = True
    use_multiprocessing: bool = True

class SecureCredentials:
    """Secure credential management"""
    
    def __init__(self, config_file: str = "credentials.json"):
        self.config_file = Path(config_file)
        self.credentials = {}
        self._load_credentials()
    
    def _load_credentials(self):
        """Load credentials from file (no prompting)"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    self.credentials = json.load(f)
                logger.info("✅ Credentials loaded from file")
            except Exception as e:
                logger.error(f"❌ Error loading credentials: {e}")
                # Don't prompt - credentials will be set via API
                self.credentials = {}
        else:
            # Don't prompt - credentials will be set via API
            self.credentials = {}
    
    def _prompt_credentials(self):
        """Prompt user for credentials securely"""
        print("\n🔐 Twitter Credentials Setup")
        print("=" * 40)
        
        self.credentials = {
            'username': input("Enter Twitter username (with @): ").strip(),
            'target_profile': input("Enter target Twitter handle (without @): ").strip()
        }
        
        # For password, use getpass for security
        from getpass import getpass
        password = getpass("Enter Twitter password (hidden): ")
        self.credentials['password'] = password
        
        # Save credentials (password will be hashed)
        self._save_credentials()
    
    def _save_credentials(self):
        """Save credentials with hashed password"""
        import hashlib
        safe_credentials = self.credentials.copy()
        safe_credentials['password'] = hashlib.sha256(
            self.credentials['password'].encode()
        ).hexdigest()
        
        with open(self.config_file, 'w') as f:
            json.dump(safe_credentials, f, indent=2)
        logger.info("✅ Credentials saved securely")

class TextProcessor:
    """Advanced text processing with multiple fallback methods"""
    
    def __init__(self):
        self.stop_words = self._load_stop_words()
        self.nlp = self._load_nlp()
    
    def _load_stop_words(self) -> Set[str]:
        """Load stop words with fallbacks"""
        if NLTK_AVAILABLE:
            return set(stopwords.words('english'))
        else:
            # Basic English stop words as fallback
            return {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
            }
    
    def _load_nlp(self):
        """Load NLP model with fallback"""
        if SPACY_AVAILABLE:
            return nlp
        return None
    
    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning with multiple methods"""
        if not text or not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = text.lower().strip()
        
        # Remove emojis
        if EMOJI_AVAILABLE:
            text = emoji.replace_emoji(text, '')
        
        # Remove URLs, mentions, hashtags
        text = re.sub(r"http\S+|www\S+", '', text)
        text = re.sub(r"@\w+|#\w+", '', text)
        
        # Remove punctuation and numbers
        text = re.sub(r"[^a-zA-Z\s]", '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Advanced processing with SpaCy
        if self.nlp and len(text) > 10:  # Only process substantial text
            try:
                doc = self.nlp(text)
                tokens = [
                    token.lemma_ for token in doc 
                    if token.text not in self.stop_words 
                    and not token.is_stop 
                    and len(token.text) > 2
                ]
                return " ".join(tokens)
            except Exception as e:
                logger.warning(f"⚠️  SpaCy processing failed: {e}")
        
        # Fallback to basic tokenization
        if NLTK_AVAILABLE:
            try:
                tokens = word_tokenize(text)
                tokens = [token for token in tokens if token not in self.stop_words]
                return " ".join(tokens)
            except Exception as e:
                logger.warning(f"⚠️  NLTK processing failed: {e}")
        
        return text

class KeywordExtractor:
    """Advanced keyword extraction with multiple algorithms"""
    
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.keybert_model = self._load_keybert()
        self.cache = {}
    
    def _load_keybert(self):
        """Load KeyBERT model with fallback"""
        if KEYBERT_AVAILABLE:
            try:
                return KeyBERT('sentence-transformers/all-MiniLM-L6-v2')
            except Exception as e:
                logger.error(f"❌ KeyBERT loading failed: {e}")
                return None
        return None
    
    def extract_keywords(self, text: str) -> List[Tuple[str, float]]:
        """Extract keywords using multiple methods"""
        if not text.strip():
            return []
        
        # Check cache first
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if self.config.cache_enabled and text_hash in self.cache:
            return self.cache[text_hash]
        
        keywords = []
        
        # Method 1: KeyBERT (if available)
        if self.keybert_model:
            try:
                keywords = self.keybert_model.extract_keywords(
                    text,
                    keyphrase_ngram_range=self.config.ngram_range,
                    stop_words='english',
                    top_n=self.config.top_n_keywords
                )
            except Exception as e:
                logger.warning(f"⚠️  KeyBERT extraction failed: {e}")
        
        # Method 2: TF-IDF fallback
        if not keywords:
            keywords = self._tfidf_extraction(text)
        
        # Cache results
        if self.config.cache_enabled:
            self.cache[text_hash] = keywords
        
        return keywords  # type: ignore
    
    def _tfidf_extraction(self, text: str) -> List[Tuple[str, float]]:
        """TF-IDF based keyword extraction as fallback"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
            
            # Simple TF-IDF extraction
            vectorizer = TfidfVectorizer(
                stop_words=ENGLISH_STOP_WORDS,
                ngram_range=self.config.ngram_range,
                max_features=self.config.top_n_keywords
            )
            
            # Fit and transform
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            
            # Get scores
            scores = tfidf_matrix.toarray()[0]  # type: ignore
            
            # Create keyword-score pairs
            keywords = [(feature_names[i], float(scores[i])) 
                       for i in range(len(feature_names)) if scores[i] > 0]
            
            # Sort by score
            keywords.sort(key=lambda x: x[1], reverse=True)
            
            return keywords[:self.config.top_n_keywords]
            
        except Exception as e:
            logger.error(f"❌ TF-IDF extraction failed: {e}")
            return []

class DataVisualizer:
    """Advanced data visualization and analysis"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def create_visualizations(self, keyword_df: pd.DataFrame, 
                            important_keywords: pd.DataFrame) -> Dict[str, str]:
        """Create comprehensive visualizations"""
        plots = {}
        
        try:
            # 1. Keyword Score Distribution
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 2, 1)
            plt.hist(keyword_df["score"], bins=30, color="skyblue", 
                    edgecolor="black", alpha=0.7)
            plt.title("Keyword Score Distribution")
            plt.xlabel("Score")
            plt.ylabel("Frequency")
            plt.grid(True, alpha=0.3)
            
            # 2. Top Keywords Bar Chart
            plt.subplot(2, 2, 2)
            top_keywords = important_keywords.head(15)
            plt.barh(range(len(top_keywords)), top_keywords["score"])
            plt.yticks(range(len(top_keywords)), top_keywords["keyword"].tolist())
            plt.title("Top 15 Keywords by Score")
            plt.xlabel("Score")
            
            # 3. Score vs Rank
            plt.subplot(2, 2, 3)
            plt.plot(range(len(keyword_df)), keyword_df["score"])
            plt.title("Score vs Rank")
            plt.xlabel("Rank")
            plt.ylabel("Score")
            plt.grid(True, alpha=0.3)
            
            # 4. Keyword Length Distribution
            plt.subplot(2, 2, 4)
            keyword_lengths = [len(kw.split()) for kw in keyword_df["keyword"]]
            plt.hist(keyword_lengths, bins=range(1, max(keyword_lengths) + 2), 
                    color="lightgreen", alpha=0.7)
            plt.title("Keyword Length Distribution")
            plt.xlabel("Words per Keyword")
            plt.ylabel("Frequency")
            
            plt.tight_layout()
            plot_path = self.output_dir / "keyword_analysis.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plots['analysis'] = str(plot_path)
            plt.close()
            
            # 5. Word Cloud (if wordcloud available)
            try:
                from wordcloud import WordCloud
                wordcloud = WordCloud(
                    width=800, height=400,
                    background_color='white',
                    max_words=100
                ).generate_from_frequencies(
                    dict(zip(important_keywords["keyword"], important_keywords["score"]))
                )
                
                plt.figure(figsize=(10, 6))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title("Keyword Word Cloud")
                
                wordcloud_path = self.output_dir / "keyword_wordcloud.png"
                plt.savefig(wordcloud_path, dpi=300, bbox_inches='tight')
                plots['wordcloud'] = str(wordcloud_path)
                plt.close()
                
            except ImportError:
                logger.info("ℹ️  WordCloud not available. Install with: pip install wordcloud")
            
            logger.info("✅ Visualizations created successfully")
            
        except Exception as e:
            logger.error(f"❌ Visualization creation failed: {e}")
        
        return plots

class EnhancedKeywordsExtraction:
    """Main class for enhanced keyword extraction"""
    
    def __init__(self, config: Optional[ExtractionConfig] = None):
        self.config = config or ExtractionConfig()
        # Don't initialize SecureCredentials here - credentials will be set via API
        self.text_processor = TextProcessor()
        self.keyword_extractor = KeywordExtractor(self.config)
        self.visualizer = DataVisualizer()
        self.results = {}
        self.current_credentials = None
    
    def set_credentials(self, username: str, password: str, target_handle: str):
        """Set credentials for current request"""
        self.current_credentials = {
            'username': username,
            'password': password,
            'target_profile': target_handle
        }
        logger.info(f"✅ Credentials set for user: {username}")
    

    
    def run_extraction(self) -> Dict[str, Any]:
        """Main extraction pipeline"""
        try:
            logger.info("🚀 Starting enhanced keyword extraction...")
            
            # Step 1: Data Collection
            tweets = self._collect_data()
            if not tweets:
                raise ValueError("No tweets collected")
            
            # Step 2: Text Processing
            processed_data = self._process_text(tweets)
            
            # Step 3: Keyword Extraction
            keywords = self._extract_keywords(processed_data)
            
            # Step 4: Analysis and Visualization
            self._analyze_results(keywords)
            
            logger.info("✅ Keyword extraction completed successfully!")
            return self.results
            
        except Exception as e:
            logger.error(f"❌ Extraction failed: {e}")
            raise
    
    def _collect_data(self) -> List[str]:
        """Collect data from Twitter or other sources"""
        if self.current_credentials:
            logger.info(f"📊 Collecting real-time data using user credentials for @{self.current_credentials['target_profile']}")
            
            # Try multiple attempts to get real data
            for attempt in range(3):  # Try 3 times
                try:
                    logger.info(f"🔄 Attempt {attempt + 1}/3 to collect real-time data...")
                    
                    # Import and use the real-time Twitter scraper
                    from .twitter_scraper import TwitterScraper
                    
                    scraper = TwitterScraper()
                    
                    # Setup driver
                    if not scraper.setup_driver():
                        logger.warning(f"⚠️  Attempt {attempt + 1}: Failed to setup Chrome driver")
                        scraper.cleanup()
                        continue
                    
                    # Login to Twitter
                    if not scraper.login_to_twitter(
                        self.current_credentials['username'],
                        self.current_credentials['password']
                    ):
                        logger.warning(f"⚠️  Attempt {attempt + 1}: Failed to login to Twitter")
                        scraper.cleanup()
                        continue
                    
                    # Scrape real tweets from the target profile
                    tweets = scraper.scrape_profile_tweets(
                        self.current_credentials['target_profile'],
                        max_tweets=self.config.max_tweets
                    )
                    
                    # Cleanup
                    scraper.cleanup()
                    
                    if tweets and len(tweets) > 0:
                        logger.info(f"✅ Successfully scraped {len(tweets)} real tweets from @{self.current_credentials['target_profile']}")
                        return tweets
                    else:
                        logger.warning(f"⚠️  Attempt {attempt + 1}: No tweets found")
                        continue
                        
                except Exception as e:
                    logger.error(f"❌ Attempt {attempt + 1}: Real-time scraping failed: {e}")
                    if attempt < 2:  # Don't log fallback message on last attempt
                        continue
            
            # If all attempts failed, raise an error instead of using sample data
            logger.error("❌ All attempts to collect real-time data failed")
            raise Exception("Failed to collect real-time Twitter data after 3 attempts")
        else:
            logger.error("❌ No credentials provided for real-time data collection")
            raise Exception("Twitter credentials are required for real-time data collection")
    
    def _get_sample_data(self) -> List[str]:
        """Get sample data for demonstration"""
        return [
            "Just had the most amazing pasta at this new Italian restaurant! #foodie #pasta",
            "Love trying new recipes. Today made homemade pizza with fresh ingredients.",
            "Coffee and breakfast are the perfect way to start the day ☕️",
            "Exploring local food markets is my favorite weekend activity",
            "Nothing beats a home-cooked meal with family and friends"
        ]
    
    def _process_text(self, texts: List[str]) -> str:
        """Process and clean text data"""
        logger.info("🧹 Processing and cleaning text...")
        
        processed_texts = []
        for text in texts:
            cleaned = self.text_processor.clean_text(text)
            if cleaned:
                processed_texts.append(cleaned)
        
        return " ".join(processed_texts)
    
    def _extract_keywords(self, text: str) -> pd.DataFrame:
        """Extract keywords from processed text"""
        logger.info("🔍 Extracting keywords...")
        
        keywords = self.keyword_extractor.extract_keywords(text)
        
        # Create DataFrame
        keyword_df = pd.DataFrame(keywords, columns=["keyword", "score"])
        
        # Filter important keywords
        important_keywords = keyword_df[keyword_df["score"] >= self.config.min_keyword_score]
        
        # Save results
        keyword_df.to_csv("extracted_keywords.csv", index=False)
        important_keywords.to_csv("important_keywords.csv", index=False)
        
        self.results['keyword_df'] = keyword_df
        self.results['important_keywords'] = important_keywords
        
        logger.info(f"✅ Extracted {len(keyword_df)} keywords, {len(important_keywords)} important ones")
        return keyword_df
    
    def _analyze_results(self, keyword_df: pd.DataFrame):
        """Analyze and visualize results"""
        logger.info("📊 Creating visualizations...")
        
        important_keywords = self.results['important_keywords']
        plots = self.visualizer.create_visualizations(keyword_df, important_keywords)
        
        self.results['plots'] = plots
        
        # Print summary
        print("\n" + "="*50)
        print("📊 KEYWORD EXTRACTION RESULTS")
        print("="*50)
        print(f"Total keywords extracted: {len(keyword_df)}")
        print(f"Important keywords (score ≥ {self.config.min_keyword_score}): {len(important_keywords)}")
        print(f"Average keyword score: {keyword_df['score'].mean():.3f}")
        print(f"Top 5 keywords: {list(important_keywords.head()['keyword'])}")
        print("="*50)

def main():
    """Main execution function"""
    print("🚀 Enhanced Keywords Extraction System")
    print("="*50)
    
    # Create configuration
    config = ExtractionConfig(
        max_tweets=50,
        min_keyword_score=0.3,
        top_n_keywords=100,
        cache_enabled=True
    )
    
    # Run extraction
    extractor = EnhancedKeywordsExtraction(config)
    results = extractor.run_extraction()
    
    print("\n✅ Extraction completed! Check the output files:")
    print("- extracted_keywords.csv")
    print("- important_keywords.csv")
    print("- output/keyword_analysis.png")
    print("- keywords_extraction.log")

if __name__ == "__main__":
    main() 