# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 06:58:47 2026

@author: HP
"""

# ==========================================================
# CUSTOMER REVIEW ANALYSIS USING NLP (TF-IDF + SIMILARITY)
# ==========================================================

# FLOW:
# Business Understanding
# → Data Understanding
# → Exploratory Data Analysis (EDA)
# → Data Preprocessing
# → Feature Engineering (TF-IDF)
# → Similarity Analysis

# ==========================================================
# BUSINESS UNDERSTANDING
# ==========================================================

# 1. Business Problem Statement:
# E-commerce platforms receive thousands of customer reviews.
# Manually analyzing them is time-consuming and inefficient.

# 2. Simplified Context:
# Reviews are unstructured text data containing opinions,
# complaints, and praise about products and services.

# 3. Problem Identification:
# - Large volume of unstructured text
# - Duplicate or similar complaints
# - Difficulty in identifying key issues quickly

# 4. Business Objective:
# Maximize: Understanding of customer sentiment & issues
# Minimize: Manual effort in review analysis
# Enable: Automated keyword extraction & similarity detection


# ==========================================================
# STEP 1: Import Required Libraries
# ==========================================================

import pandas as pd          # Data handling
import numpy as np           # Numerical operations
import re                    # Text cleaning
import string                # Punctuation handling

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================================
# STEP 2: Load the Dataset
# ==========================================================

df = pd.read_csv("C:/7-NLP/flipkart_reviews.csv")

# Preview dataset
df.head()


# ==========================================================
# STEP 3: BASIC EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================================

print("Dataset Shape:", df.shape)
print("Dataset Size:", df.size)

df.info()

# Inference:
# - Dataset contains customer reviews in textual format.
# - Presence of null values or incorrect column names
#   must be handled before NLP processing.


# ==========================================================
# STEP 4: IDENTIFY & STANDARDIZE TEXT COLUMN
# ==========================================================

"""
IMPORTANT:
Rename text column to a standard name for processing.
"""

df = df.rename({'Review': 'review_text'}, axis=1)

# Drop missing reviews
df = df.dropna(subset=['review_text'])

print("Records after removing missing reviews:", df.shape)


# ==========================================================
# STEP 5: TEXT PREPROCESSING (DATA PREPARATION)
# ==========================================================

# Objective:
# Remove noise so NLP models capture meaningful words

def clean_text(text):
    """
    Cleans raw text by:
    1. Lowercasing
    2. Removing punctuation
    3. Removing digits
    4. Removing extra spaces
    """
    text = str(text).lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Apply text cleaning
df['clean_text'] = df['review_text'].apply(clean_text)

# Compare original vs cleaned
df[['review_text', 'clean_text']].head()


# ==========================================================
# STEP 6: EXPLORATORY TEXT ANALYSIS
# ==========================================================

# Length of reviews
df['review_length'] = df['clean_text'].apply(len)

plt.figure(figsize=(8,4))
sns.histplot(df['review_length'], bins=30, kde=True)
plt.title("Distribution of Review Lengths")
plt.show()

# Inference:
# Helps understand verbosity of customer feedback.


# ==========================================================
# STEP 7: FEATURE ENGINEERING - TF-IDF VECTORIZATION
# ==========================================================

"""
TF-IDF converts text into numerical features.
TF  → Word importance in a review
IDF → Word uniqueness across all reviews
"""

tfidf = TfidfVectorizer(
    stop_words='english',   # Removes common English words
    ngram_range=(1, 2),     # Considers unigrams & bigrams
    max_df=0.85,            # Ignores words appearing in >85% docs
    min_df=2                # Keeps words appearing in at least 2 docs
)

X_tfidf = tfidf.fit_transform(df['clean_text'])

print("TF-IDF Matrix Shape:", X_tfidf.shape)

# ==========================================================
# STEP 8: SIMILARITY ANALYSIS USING COSINE SIMILARITY
# ==========================================================

# Cosine similarity measures similarity between review vectors
cosine_sim = cosine_similarity(X_tfidf)

print("Cosine Similarity Matrix Shape:", cosine_sim.shape)


# Example: Find similarity of first review with others
similar_scores = list(enumerate(cosine_sim[0]))

# Sort reviews based on similarity score (descending)
sorted_similar_reviews = sorted(similar_scores, key=lambda x: x[1], reverse=True)

print("\nTop 5 Reviews Similar to Review 0:\n")

for index, score in sorted_similar_reviews[1:6]:
    print("Review Index:", index)
    print("Similarity Score:", round(score, 3))
    print("Review Text:", df['review_text'].iloc[index])
    print("-" * 60)


# ==========================================================
# STEP 9: TOP KEYWORDS USING TF-IDF SCORES
# ==========================================================

# Sum TF-IDF scores for each word
tfidf_sum = np.array(X_tfidf.sum(axis=0)).flatten()

# Get feature names (words)
feature_names = tfidf.get_feature_names_out()

# Create DataFrame of words with scores
tfidf_df = pd.DataFrame({
    'word': feature_names,
    'score': tfidf_sum
})

# Sort words by importance
top_keywords = tfidf_df.sort_values(by='score', ascending=False).head(20)

print("\nTop 20 Important Keywords Across All Reviews:\n")
print(top_keywords)


# ==========================================================
# STEP 10: VISUALIZE TOP TF-IDF KEYWORDS
# ==========================================================

plt.figure(figsize=(10,6))
sns.barplot(x='score', y='word', data=top_keywords)
plt.title("Top TF-IDF Keywords Across All Reviews")
plt.xlabel("TF-IDF Score")
plt.ylabel("Keywords")
plt.show()


# ==========================================================
# STEP 11: SIMILARITY HEATMAP (OPTIONAL VISUALIZATION)
# ==========================================================

plt.figure(figsize=(8,6))
sns.heatmap(cosine_sim[:20, :20], cmap='viridis')
plt.title("Cosine Similarity Heatmap (First 20 Reviews)")
plt.show()


# ==========================================================
# FINAL BUSINESS INTERPRETATION
# ==========================================================

"""
BUSINESS INSIGHTS:

1. TF-IDF successfully extracted important keywords from reviews.
   This helps identify common issues, product features, and sentiments.

2. Cosine similarity detects similar reviews.
   This helps:
   - Identify duplicate complaints
   - Group similar feedback
   - Detect repeated product issues

3. Automated analysis reduces manual effort significantly.

4. Businesses can:
   - Prioritize frequently mentioned problems
   - Improve products based on common complaints
   - Detect trending issues early
   - Automate customer feedback monitoring

CONCLUSION:
Using NLP (TF-IDF + Cosine Similarity), unstructured customer
reviews are converted into structured numerical insights.
This enables data-driven decision making and improves
customer satisfaction.
"""