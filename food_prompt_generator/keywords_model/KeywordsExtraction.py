import time
import csv
import re
import pandas as pd
import matplotlib.pyplot as plt
import emoji
import nltk
import spacy
from nltk.corpus import stopwords
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException
import undetected_chromedriver as uc
from keybert import KeyBERT

# --------------------------
# 🧠 Load NLP Tools
# --------------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")

# --------------------------
# 🔐 Twitter Credentials
# --------------------------
USERNAME = input("Enter your Twitter username (with @): ")
from getpass import getpass
PASSWORD = getpass("Enter your Twitter password (input hidden): ")

TARGET_PROFILE = input("Enter the target Twitter handle (without @): ")


# --------------------------
# ⚙️ Chrome Setup
# --------------------------
options = uc.ChromeOptions()
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("--start-maximized")
options.add_argument("--disable-popup-blocking")

driver = uc.Chrome(options=options)
driver.get("https://twitter.com/login")
time.sleep(5)

# --------------------------
# 🧾 Login to Twitter
# --------------------------
wait = WebDriverWait(driver, 15)
username_input = wait.until(EC.presence_of_element_located((By.NAME, "text")))
username_input.send_keys(USERNAME)
username_input.send_keys(Keys.RETURN)
time.sleep(3)

password_input = wait.until(EC.presence_of_element_located((By.NAME, "password")))
password_input.send_keys(PASSWORD)
password_input.send_keys(Keys.RETURN)
time.sleep(6)

# --------------------------
# 👤 Visit Target Profile
# --------------------------
driver.get(f"https://twitter.com/{TARGET_PROFILE}")
time.sleep(5)

# --------------------------
# 🔁 Scroll to Load More Tweets
# --------------------------
last_height = driver.execute_script("return document.body.scrollHeight")
scroll_attempts = 0
tweets = set()

while len(tweets) < 50 and scroll_attempts < 15:
    driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
    time.sleep(3)

    tweet_elements = driver.find_elements(By.XPATH, '//div[@data-testid="tweetText"]')
    for elem in tweet_elements:
        try:
            text = elem.text.strip()
            if text:
                tweets.add(text)
        except StaleElementReferenceException:
            continue

    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        scroll_attempts += 1
    else:
        scroll_attempts = 0
        last_height = new_height

driver.quit()

# --------------------------
# 💾 Save Raw Tweets
# --------------------------
raw_path = "raw_tweets.csv"
with open(raw_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["text"])
    for tweet in tweets:
        writer.writerow([tweet])

# --------------------------
# ✨ Clean Text Function using NLTK & SpaCy
# --------------------------
def clean_text(text):
    text = text.lower()                          # Lowercase
    text = emoji.replace_emoji(text, '')         # Remove emojis
    text = re.sub(r"http\S+|www\S+", '', text)   # Remove URLs
    text = re.sub(r"@\w+|#\w+", '', text)        # Remove mentions and hashtags
    text = re.sub(r"[^a-zA-Z\s]", '', text)      # Remove punctuation and numbers
    text = re.sub(r'\s+', ' ', text).strip()     # Remove extra spaces

    doc = nlp(text)                              # Apply SpaCy NLP
    tokens = [token.lemma_ for token in doc if token.text not in stop_words and not token.is_stop]
    return " ".join(tokens)

# --------------------------
# 🧼 Apply Cleaning
# --------------------------
df = pd.read_csv(raw_path)
df['cleaned_text'] = df['text'].astype(str).apply(clean_text)
df.to_csv("cleaned_tweets.csv", index=False)

# --------------------------
# 🧠 Extract Keywords
# --------------------------
model = KeyBERT('sentence-transformers/all-MiniLM-L6-v2')
combined_text = " ".join(df['cleaned_text'].dropna().tolist())

keywords = model.extract_keywords(
    combined_text,
    keyphrase_ngram_range=(1, 2),
    stop_words='english',
    top_n=1000
)

keyword_df = pd.DataFrame(keywords, columns=["keyword", "score"])
keyword_df.to_csv("extracted_keywords.csv", index=False)

# --------------------------
# ✅ Save Important Keywords
# --------------------------
important_keywords = keyword_df[keyword_df["score"] >= 0.3]
important_keywords.to_csv("important_keywords.csv", index=False)
print(f"✅ Saved {len(important_keywords)} important keywords to important_keywords.csv")

# --------------------------
# 📊 Plot & Save Histogram
# --------------------------
plt.figure(figsize=(10, 6))
plt.hist(keyword_df["score"], bins=20, color="skyblue", edgecolor="black")
plt.title("Distribution of Keyword Scores")
plt.xlabel("Keyword Score")
plt.ylabel("Frequency")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("keyword_score_distribution.png")
print("📊 Saved keyword score histogram.")
