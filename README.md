# 📊 CustomerPulse — AI-Powered Customer Feedback Intelligence Dashboard

Turning scattered customer reviews into actionable business intelligence using NLP, Machine Learning, and RAG.

CustomerPulse analyzes real Amazon customer reviews to automatically detect sentiment, tag complaint topics, predict customer churn risk, and answer natural-language questions grounded in real review data — all displayed on an interactive Flask dashboard.

## 🎯 Problem Statement
Companies collect thousands of reviews but rarely have time to read them all. Critical signals — delivery complaints, quality issues, at-risk customers — get buried in raw text. CustomerPulse surfaces these signals automatically and lets anyone ask questions in plain English and get answers backed by real evidence.

## ✨ Key Features
- 🧠 **Sentiment Analysis** — VADER-based NLP scores every review Positive/Neutral/Negative
- 🏷️ **Topic Tagging** — auto-categorizes reviews into Delivery, Pricing, Quality, Support, App
- ⚠️ **Churn Risk Prediction** — Logistic Regression flags customers likely to churn
- 💬 **RAG-Powered Chatbot** — ask "Why are customers unhappy with delivery?" and get answers retrieved from real reviews, no hallucination
- 📈 **Interactive Dashboard** — KPIs, sentiment trends, topic distribution, product comparison, churn leaderboard

## 🏗️ Architecture

Raw Reviews (CSV)
│
▼
┌─────────────────────┐
│ Data Preprocessing │ Cleaning, column mapping, sampling
└─────────────────────┘
│
▼
┌─────────────────────────────────────────┐
│ Analysis Pipeline │
│ • Sentiment Analysis (VADER) │
│ • Topic Tagging (keyword-based NLP) │
│ • Churn Risk Scoring (Logistic Regression) │
│ • TF-IDF Vector Index (for RAG retrieval) │
└─────────────────────────────────────────┘
│
▼
┌─────────────────┐ ┌──────────────────┐
│ Flask REST API │────▶│ Dashboard (UI) │
└─────────────────┘ └──────────────────┘

## 🛠️ Tech Stack
Python • Flask • Pandas • scikit-learn • VADER Sentiment • TF-IDF/Cosine Similarity (RAG) • Chart.js • Bootstrap
**Dataset:** [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) (568K+ real reviews, Kaggle/SNAP)


<img width="1877" height="903" alt="Screenshot 2026-09-26 113832" src="https://github.com/user-attachments/assets/aa423e52-2796-4814-82a8-0673320ca4c3" />

