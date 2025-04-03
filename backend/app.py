from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
import pandas as pd
from datetime import datetime, timedelta
import random
import json

# Download required NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize NLP tools
sia = SentimentIntensityAnalyzer()
nlp = spacy.load('en_core_web_sm')

# Load sample data (in a real app, this would connect to social media APIs)
def load_sample_data():
    # Sample platforms
    platforms = ['Twitter', 'Instagram', 'Facebook']
    
    # Sample brands/topics to track
    topics = ['Apple', 'Samsung', 'Tesla', 'Nike', 'Adidas']
    
    # Sample comments with varying sentiments
    positive_templates = [
        "I love {topic}! Their products are amazing!",
        "{topic} has the best customer service ever!",
        "Just got a new {topic} product and I'm impressed!",
        "Can't believe how good {topic}'s new release is!",
        "{topic} never disappoints, quality is outstanding!"
    ]
    
    negative_templates = [
        "Disappointed with my recent {topic} purchase.",
        "{topic}'s customer service was terrible.",
        "The new {topic} product is overpriced and underwhelming.",
        "Had issues with my {topic} device, won't buy again.",
        "{topic} needs to improve their quality control."
    ]
    
    neutral_templates = [
        "Just ordered something from {topic}, we'll see how it goes.",
        "Has anyone tried the new {topic} product?",
        "Considering switching to {topic}, any thoughts?",
        "{topic} announced a new product today.",
        "Interesting development from {topic}."
    ]
    
    # Generate sample data
    data = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    for _ in range(1000):
        topic = random.choice(topics)
        platform = random.choice(platforms)
        
        # Determine sentiment type
        sentiment_type = random.choices(['positive', 'negative', 'neutral'], weights=[0.4, 0.3, 0.3])[0]
        
        if sentiment_type == 'positive':
            text = random.choice(positive_templates).format(topic=topic)
        elif sentiment_type == 'negative':
            text = random.choice(negative_templates).format(topic=topic)
        else:
            text = random.choice(neutral_templates).format(topic=topic)
        
        # Generate random date within the past month
        random_date = start_date + timedelta(
            seconds=random.randint(0, int((end_date - start_date).total_seconds()))
        )
        
        # Analyze sentiment
        sentiment_scores = sia.polarity_scores(text)
        
        data.append({
            'id': len(data) + 1,
            'text': text,
            'platform': platform,
            'topic': topic,
            'date': random_date.strftime('%Y-%m-%d %H:%M:%S'),
            'sentiment_score': sentiment_scores['compound'],
            'sentiment': 'positive' if sentiment_scores['compound'] >= 0.05 else 
                         'negative' if sentiment_scores['compound'] <= -0.05 else 'neutral',
            'likes': random.randint(0, 1000),
            'shares': random.randint(0, 200),
            'comments': random.randint(0, 50)
        })
    
    return pd.DataFrame(data)

# Load the sample data
df = load_sample_data()

# API Routes
@app.route('/api/sentiment/overview', methods=['GET'])
def sentiment_overview():
    # Overall sentiment distribution
    sentiment_counts = df['sentiment'].value_counts().to_dict()
    
    # Average sentiment score by platform
    platform_sentiment = df.groupby('platform')['sentiment_score'].mean().to_dict()
    
    # Sentiment trend over time (daily)
    df['date_only'] = pd.to_datetime(df['date']).dt.date
    time_series = df.groupby('date_only')['sentiment_score'].mean().reset_index()
    time_series['date_only'] = time_series['date_only'].astype(str)
    
    return jsonify({
        'sentiment_distribution': sentiment_counts,
        'platform_sentiment': platform_sentiment,
        'sentiment_trend': time_series.to_dict(orient='records')
    })

@app.route('/api/sentiment/topics', methods=['GET'])
def topic_sentiment():
    # Sentiment by topic
    topic_sentiment = df.groupby('topic')['sentiment_score'].mean().to_dict()
    
    # Sentiment distribution by topic
    topic_distribution = df.groupby(['topic', 'sentiment']).size().unstack().fillna(0).to_dict()
    
    # Topic engagement (likes, shares, comments)
    topic_engagement = df.groupby('topic')[['likes', 'shares', 'comments']].sum().to_dict()
    
    return jsonify({
        'topic_sentiment': topic_sentiment,
        'topic_distribution': topic_distribution,
        'topic_engagement': topic_engagement
    })

@app.route('/api/sentiment/platforms', methods=['GET'])
def platform_analysis():
    # Sentiment by platform over time
    df['date_only'] = pd.to_datetime(df['date']).dt.date
    platform_time = df.groupby(['platform', 'date_only'])['sentiment_score'].mean().reset_index()
    platform_time['date_only'] = platform_time['date_only'].astype(str)
    
    # Volume by platform
    platform_volume = df['platform'].value_counts().to_dict()
    
    return jsonify({
        'platform_sentiment_trend': platform_time.to_dict(orient='records'),
        'platform_volume': platform_volume
    })

@app.route('/api/posts/recent', methods=['GET'])
def recent_posts():
    # Get the most recent posts with their sentiment
    recent = df.sort_values('date', ascending=False).head(20).to_dict(orient='records')
    return jsonify(recent)

@app.route('/api/sentiment/analyze', methods=['POST'])
def analyze_text():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    sentiment_scores = sia.polarity_scores(text)
    
    # Extract entities using spaCy
    doc = nlp(text)
    entities = [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]
    
    # Determine overall sentiment
    compound = sentiment_scores['compound']
    if compound >= 0.05:
        sentiment = "positive"
    elif compound <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    return jsonify({
        'text': text,
        'sentiment': sentiment,
        'sentiment_scores': sentiment_scores,
        'entities': entities
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)