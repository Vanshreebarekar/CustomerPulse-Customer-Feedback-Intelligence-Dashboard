"""
CustomerPulse — Customer Feedback Intelligence Dashboard
Run with:  python app.py
Then open: http://127.0.0.1:5000
"""
import pandas as pd
from flask import Flask, render_template, jsonify, request

from utils.sentiment import get_sentiment
from utils.topics import tag_topic
from utils.churn import train_churn_model
from utils.chatbot import ReviewRAG

app = Flask(__name__)

# ------------------------------------------------------------------
# Load & process data ONCE when the server starts (not on every request)
# ------------------------------------------------------------------
df = pd.read_csv('data/reviews.csv')

# 1. NLP: sentiment analysis on every review
sentiments = df['review_text'].apply(get_sentiment)
df['sentiment'] = sentiments.apply(lambda x: x[0])
df['sentiment_score'] = sentiments.apply(lambda x: x[1])

# 2. NLP: topic tagging on every review
df['topic'] = df['review_text'].apply(tag_topic)

# 3. ML: churn-risk scoring
df = train_churn_model(df)

# 4. RAG: build the retrieval index once, reuse for every chat question
rag = ReviewRAG(df)


# ------------------------------------------------------------------
# Page route
# ------------------------------------------------------------------
@app.route('/')
def dashboard():
    return render_template('dashboard.html')


# ------------------------------------------------------------------
# API routes — the dashboard's JS calls these to get chart data
# ------------------------------------------------------------------
@app.route('/api/summary')
def api_summary():
    return jsonify({
        "total_reviews": len(df),
        "avg_rating": round(df['rating'].mean(), 2),
        "sentiment_counts": df['sentiment'].value_counts().to_dict(),
        "at_risk_customers": int((df['churn_risk_prob'] > 0.5).sum()),
    })


@app.route('/api/sentiment_trend')
def api_sentiment_trend():
    trend = df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
    trend = trend.reindex(columns=['Positive', 'Neutral', 'Negative'], fill_value=0)
    trend = trend.sort_index()

    return jsonify({
        "dates": trend.index.tolist(),
        "positive": trend['Positive'].tolist(),
        "neutral": trend['Neutral'].tolist(),
        "negative": trend['Negative'].tolist(),
    })


@app.route('/api/topics')
def api_topics():
    return jsonify(df['topic'].value_counts().to_dict())


@app.route('/api/product_ratings')
def api_product_ratings():
    avg_by_product = df.groupby('product')['rating'].mean().round(2)
    return jsonify({
        "products": avg_by_product.index.tolist(),
        "ratings": avg_by_product.values.tolist(),
    })


@app.route('/api/churn_table')
def api_churn_table():
    risky = df[df['churn_risk_prob'] > 0.4].sort_values('churn_risk_prob', ascending=False)
    table = risky[['review_id', 'product', 'rating', 'sentiment', 'churn_risk_prob']].head(15).copy()
    table['churn_risk_prob'] = table['churn_risk_prob'].round(2)
    return jsonify(table.to_dict('records'))


@app.route('/api/chat', methods=['POST'])
def api_chat():
    question = (request.json or {}).get('question', '').strip()
    if not question:
        return jsonify({"answer": "Please type a question.", "sources": []})
    return jsonify(rag.answer(question))


if __name__ == '__main__':
    app.run(debug=True)
