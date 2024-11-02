import tweepy
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import threading
import queue

class EnhancedCryptoTwitterMonitor:
    def __init__(self, api_key, api_secret, access_token, access_token_secret):
        """Initialize Twitter API authentication and data structures"""
        auth = tweepy.OAuthHandler(api_key, api_secret)
        auth.set_access_token(access_token, access_token_secret)
        self.api = tweepy.API(auth, wait_on_rate_limit=True)
        
        # Initialize data structures
        self.crypto_mentions = Counter()
        self.sentiment_scores = {}
        self.influencer_predictions = {}
        self.price_mentions = {}
        self.tweet_queue = queue.Queue()
        
        # Enhanced keywords
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
            "upcoming crypto projects",
            "crypto analysis",
            "crypto technical analysis",
            "crypto fundamental analysis"
        ]
        
    def detailed_sentiment_analysis(self, text):
        """Enhanced sentiment analysis using TextBlob and custom indicators"""
        blob = TextBlob(text)
        
        # Base sentiment from TextBlob
        base_sentiment = blob.sentiment.polarity
        
        # Custom crypto-specific sentiment indicators
        bullish_indicators = ['bull', 'buy', 'moon', 'gem', 'growth', 'accumulate', 'undervalued', 'potential']
        bearish_indicators = ['bear', 'sell', 'dump', 'drop', 'crash', 'avoid', 'overvalued', 'risk']
        
        # Calculate custom sentiment
        text_lower = text.lower()
        bullish_count = sum(1 for word in bullish_indicators if word in text_lower)
        bearish_count = sum(1 for word in bearish_indicators if word in text_lower)
        
        # Combine base and custom sentiment
        custom_sentiment = (bullish_count - bearish_count) / (bullish_count + bearish_count + 1)
        final_sentiment = (base_sentiment + custom_sentiment) / 2
        
        return {
            'sentiment_score': final_sentiment,
            'base_sentiment': base_sentiment,
            'custom_sentiment': custom_sentiment,
            'bullish_indicators': bullish_count,
            'bearish_indicators': bearish_count
        }
    
    def extract_price_predictions(self, text):
        """Extract price predictions from tweets"""
        # Match patterns like "$CRYPTO to $100" or "$CRYPTO target $500"
        price_pattern = r'\$([A-Z]{2,10})\s+(?:to|target|price)\s+\$?(\d+(?:,\d{3})*(?:\.\d{1,2})?)'
        matches = re.finditer(price_pattern, text, re.IGNORECASE)
        
        predictions = {}
        for match in matches:
            crypto = match.group(1)
            price = float(match.group(2).replace(',', ''))
            predictions[crypto] = price
            
        return predictions
    
    def create_visualizations(self, df):
        """Create various visualizations of the data"""
        # 1. Sentiment Distribution
        plt.figure(figsize=(12, 6))
        sns.histplot(data=df, x='sentiment_score', bins=30)
        plt.title('Distribution of Sentiment Scores')
        plt.savefig('sentiment_distribution.png')
        plt.close()
        
        # 2. Top Cryptocurrencies Mentions
        plt.figure(figsize=(12, 6))
        top_cryptos = df['symbols'].value_counts().head(10)
        sns.barplot(x=top_cryptos.values, y=top_cryptos.index)
        plt.title('Top 10 Most Mentioned Cryptocurrencies')
        plt.savefig('top_mentions.png')
        plt.close()
        
        # 3. Interactive Time Series with Plotly
        fig = px.line(df, x='date', y='sentiment_score', 
                     color='symbols', title='Sentiment Over Time by Cryptocurrency')
        fig.write_html('sentiment_timeline.html')
        
        # 4. Influence vs Sentiment
        fig = px.scatter(df, x='followers', y='sentiment_score',
                        hover_data=['user', 'symbols'],
                        title='Account Influence vs Sentiment')
        fig.write_html('influence_sentiment.html')
    
    def real_time_monitor(self, duration_minutes=60):
        """Real-time monitoring of crypto mentions"""
        class MyStreamListener(tweepy.StreamListener):
            def __init__(self, queue):
                super().__init__()
                self.queue = queue
            
            def on_status(self, status):
                self.queue.put(status)
            
            def on_error(self, status_code):
                if status_code == 420:
                    return False
        
        # Start stream
        stream_listener = MyStreamListener(self.tweet_queue)
        stream = tweepy.Stream(auth=self.api.auth, listener=stream_listener)
        
        # Start streaming in a separate thread
        threading.Thread(target=stream.filter, 
                        kwargs={'track': self.search_keywords},
                        daemon=True).start()
        
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        real_time_data = []
        
        while datetime.now() < end_time:
            try:
                tweet = self.tweet_queue.get(timeout=1)
                if self.verify_account_quality(tweet.user):
                    analysis = self.analyze_tweet(tweet)
                    real_time_data.append(analysis)
                    self.update_real_time_dashboard(analysis)
            except queue.Empty:
                continue
            
        return pd.DataFrame(real_time_data)
    
    def analyze_tweet(self, tweet):
        """Comprehensive analysis of a single tweet"""
        text = tweet.full_text if hasattr(tweet, 'full_text') else tweet.text
        sentiment_analysis = self.detailed_sentiment_analysis(text)
        price_predictions = self.extract_price_predictions(text)
        crypto_symbols = self.extract_crypto_symbols(text)
        
        return {
            'date': tweet.created_at,
            'user': tweet.user.screen_name,
            'followers': tweet.user.followers_count,
            'text': text,
            'symbols': crypto_symbols,
            'sentiment_score': sentiment_analysis['sentiment_score'],
            'sentiment_details': sentiment_analysis,
            'price_predictions': price_predictions,
            'retweets': tweet.retweet_count,
            'likes': tweet.favorite_count
        }
    
    def generate_reports(self, df):
        """Generate comprehensive analysis reports"""
        reports = {
            'basic_stats': {
                'total_tweets': len(df),
                'unique_cryptos': df['symbols'].nunique(),
                'unique_users': df['user'].nunique(),
                'average_sentiment': df['sentiment_score'].mean()
            },
            'top_cryptos': df['symbols'].value_counts().head(10).to_dict(),
            'influencer_analysis': df.groupby('user').agg({
                'followers': 'first',
                'sentiment_score': 'mean',
                'symbols': 'count'
            }).sort_values('followers', ascending=False).head(10).to_dict(),
            'price_predictions': pd.DataFrame(df['price_predictions'].tolist()).mean().to_dict(),
            'sentiment_trends': df.groupby(df['date'].dt.date)['sentiment_score'].mean().to_dict()
        }
        
        return reports
    
    def create_dashboard(self):
        """Create an interactive Dash dashboard"""
        app = Dash(__name__)
        
        app.layout = html.Div([
            html.H1('Crypto Twitter Monitor Dashboard'),
            
            dcc.Graph(id='sentiment-timeline'),
            dcc.Graph(id='crypto-mentions'),
            dcc.Graph(id='influencer-impact'),
            
            dcc.Interval(
                id='interval-component',
                interval=60*1000,  # Update every minute
                n_intervals=0
            )
        ])
        
        @app.callback(
            [Output('sentiment-timeline', 'figure'),
             Output('crypto-mentions', 'figure'),
             Output('influencer-impact', 'figure')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            # Update dashboard with latest data
            df = pd.DataFrame(self.get_latest_data())
            
            sentiment_fig = px.line(df, x='date', y='sentiment_score')
            mentions_fig = px.bar(df['symbols'].value_counts().head(10))
            influencer_fig = px.scatter(df, x='followers', y='sentiment_score')
            
            return sentiment_fig, mentions_fig, influencer_fig
        
        return app
    
    def save_results(self, df, reports):
        """Save all results and visualizations"""
        # Save main DataFrame
        df.to_csv('detailed_crypto_analysis.csv', index=False)
        
        # Save reports
        with open('analysis_reports.json', 'w') as f:
            json.dump(reports, f, indent=4, default=str)
        
        # Create visualizations
        self.create_visualizations(df)
        
        print("Analysis complete. Files saved:")
        print("- detailed_crypto_analysis.csv")
        print("- analysis_reports.json")
        print("- sentiment_distribution.png")
        print("- top_mentions.png")
        print("- sentiment_timeline.html")
        print("- influence_sentiment.html")

def main():
    # Replace with your Twitter API credentials
    api_key = "YOUR_API_KEY"
    api_secret = "YOUR_API_SECRET"
    access_token = "YOUR_ACCESS_TOKEN"
    access_token_secret = "YOUR_ACCESS_TOKEN_SECRET"
    
    # Initialize monitor
    monitor = EnhancedCryptoTwitterMonitor(api_key, api_secret, access_token, access_token_secret)
    
    # Start real-time monitoring
    print("Starting real-time monitoring...")
    df = monitor.real_time_monitor(duration_minutes=60)
    
    # Generate reports
    reports = monitor.generate_reports(df)
    
    # Save results and create visualizations
    monitor.save_results(df, reports)
    
    # Start dashboard
    app = monitor.create_dashboard()
    app.run_server(debug=True)

if __name__ == "__main__":
    main()