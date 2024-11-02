import tweepy
import pandas as pd
from datetime import datetime, timedelta
import time
import json
import re
from collections import Counter

class CryptoTwitterMonitor:
    def __init__(self, api_key, api_secret, access_token, access_token_secret):
        """Initialize Twitter API authentication"""
        auth = tweepy.OAuthHandler(api_key, api_secret)
        auth.set_access_token(access_token, access_token_secret)
        self.api = tweepy.API(auth, wait_on_rate_limit=True)
        
        # Keywords to track
        self.search_keywords = [
            "best crypto 2024",
            "crypto gems 2024",
            "altcoin season",
            "next 100x crypto",
            "undervalued crypto",
            "crypto moonshot",
            "altcoin gems",
            "best investment 2024",
            "crypto prediction",
            "upcoming crypto projects"
        ]
        
        # Cryptocurrency symbol patterns
        self.crypto_pattern = r'\$[A-Z]{2,10}\b'  # Matches $BTC, $ETH, etc.
        
        # Track mentioned cryptocurrencies
        self.crypto_mentions = Counter()
        
    def verify_account_quality(self, user):
        """Check if the Twitter account meets quality criteria"""
        min_followers = 5000
        min_account_age_days = 180
        
        account_age = (datetime.now() - user.created_at).days
        
        return (user.followers_count >= min_followers and 
                account_age >= min_account_age_days and 
                user.verified)
    
    def extract_crypto_symbols(self, text):
        """Extract cryptocurrency symbols from text"""
        return re.findall(self.crypto_pattern, text)
    
    def analyze_tweet_sentiment(self, text):
        """Basic sentiment analysis for crypto-related tweets"""
        positive_words = ['bull', 'bullish', 'moon', 'gem', 'growth', 'potential', 'undervalued']
        negative_words = ['bear', 'bearish', 'dump', 'sell', 'drop', 'crash']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        return 'bullish' if positive_count > negative_count else 'bearish'
    
    def search_and_analyze(self, days_back=7):
        """Search and analyze crypto-related tweets"""
        results = []
        since_date = datetime.now() - timedelta(days=days_back)
        
        for keyword in self.search_keywords:
            try:
                tweets = tweepy.Cursor(
                    self.api.search_tweets,
                    q=keyword,
                    lang="en",
                    tweet_mode="extended",
                    since=since_date.strftime('%Y-%m-%d')
                ).items(100)
                
                for tweet in tweets:
                    if self.verify_account_quality(tweet.user):
                        crypto_symbols = self.extract_crypto_symbols(tweet.full_text)
                        if crypto_symbols:
                            sentiment = self.analyze_tweet_sentiment(tweet.full_text)
                            
                            for symbol in crypto_symbols:
                                if sentiment == 'bullish':
                                    self.crypto_mentions[symbol] += 1
                            
                            results.append({
                                'date': tweet.created_at,
                                'user': tweet.user.screen_name,
                                'followers': tweet.user.followers_count,
                                'text': tweet.full_text,
                                'symbols': crypto_symbols,
                                'sentiment': sentiment,
                                'keyword': keyword
                            })
                            
            except tweepy.TweepError as e:
                print(f"Error searching for {keyword}: {str(e)}")
                continue
                
        return pd.DataFrame(results)
    
    def get_top_mentioned_cryptos(self, n=10):
        """Get top n most mentioned cryptocurrencies with bullish sentiment"""
        return pd.DataFrame(
            self.crypto_mentions.most_common(n),
            columns=['Symbol', 'Bullish_Mentions']
        )
    
    def save_results(self, df, filename='crypto_predictions.csv'):
        """Save results to CSV file"""
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

def main():
    # Replace these with your Twitter API credentials
    api_key = "YOUR_API_KEY"
    api_secret = "YOUR_API_SECRET"
    access_token = "YOUR_ACCESS_TOKEN"
    access_token_secret = "YOUR_ACCESS_TOKEN_SECRET"
    
    # Initialize and run the monitor
    monitor = CryptoTwitterMonitor(api_key, api_secret, access_token, access_token_secret)
    
    print("Starting Twitter cryptocurrency analysis...")
    results_df = monitor.search_and_analyze(days_back=7)
    
    # Get and display top mentioned cryptocurrencies
    top_cryptos = monitor.get_top_mentioned_cryptos()
    print("\nTop mentioned cryptocurrencies with bullish sentiment:")
    print(top_cryptos)
    
    # Save results
    monitor.save_results(results_df)
    monitor.save_results(top_cryptos, 'top_cryptos.csv')

if __name__ == "__main__":
    main()