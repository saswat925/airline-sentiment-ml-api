
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.stem import WordNetLemmatizer

st.set_page_config(
    page_title="Airline Sentiment Analytics Dashboard",
    page_icon="✈️",
    layout="wide"
)

# Setup NLP components
LEMMATIZER = WordNetLemmatizer()
NEGATION_WORDS = {'not', 'no', 'never', 'neither', 'nor', 'none', 'cannot', "n't", 'hardly', 'scarcely', 'barely'}
try:
    from nltk.corpus import stopwords
    STOPWORDS_SET = set(stopwords.words('english')) - NEGATION_WORDS
except Exception:
    STOPWORDS_SET = set()

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r"[^a-zA-Z\s']", '', text)
    tokens = text.split()
    cleaned_tokens = [LEMMATIZER.lemmatize(word) for word in tokens if word not in STOPWORDS_SET]
    return " ".join(cleaned_tokens)

# Load Model Pipeline
@st.cache_resource
def load_model():
    return joblib.load('best_sentiment_pipeline.joblib')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

st.title("✈️ Airline Sentiment Analytics Dashboard")
st.markdown("""
Welcome to the production-grade Airline Sentiment Analysis System. 
This dashboard lets you analyze real-time text input and explore historical patterns extracted from customer tweets.
""")

tab1, tab2, tab3 = st.tabs(["🔮 Real-time Predictor", "📊 Insights & Historical Patterns", "📈 Model Architecture & Metrics"])

with tab1:
    st.header("Real-Time Tweet Sentiment Analysis")
    user_input = st.text_area(
        "Enter passenger tweet or feedback:",
        "This was the worst customer service! Our luggage was lost and the flight was delayed by over 4 hours."
    )
    
    if st.button("Analyze Sentiment", type="primary"):
        if user_input.strip() == "":
            st.warning("Please enter some text to analyze.")
        elif model is None:
            st.error("Classification model is not loaded.")
        else:
            cleaned = preprocess_text(user_input)
            prediction = model.predict([cleaned])[0]
            
            if prediction == 'positive':
                st.success(f"Predicted Sentiment: **{prediction.upper()}** 😊")
            elif prediction == 'neutral':
                st.info(f"Predicted Sentiment: **{prediction.upper()}** 😐")
            else:
                st.error(f"Predicted Sentiment: **{prediction.upper()}** 😡")
                
            st.markdown(f"**Preprocessed Text:** `{cleaned}`")

with tab2:
    st.header("Historical Customer Patterns")
    # Try to load dataset
    @st.cache_data
    def load_data():
        return pd.read_csv('Tweets.csv')
        
    try:
        df_tweets = load_data()
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sentiment Distribution")
            fig, ax = plt.subplots(figsize=(6, 4.5))
            sns.countplot(data=df_tweets, x='airline_sentiment', ax=ax, palette='viridis')
            st.pyplot(fig)
            
        with col2:
            st.subheader("Negative Reason Breakdown")
            fig, ax = plt.subplots(figsize=(6, 4.5))
            neg_reasons = df_tweets['negativereason'].value_counts().head(5)
            sns.barplot(x=neg_reasons.values, y=neg_reasons.index, ax=ax, palette='rocket')
            st.pyplot(fig)
            
    except Exception as e:
        st.warning(f"Could not load historical dataset visualizers: {e}")

with tab3:
    st.header("Model Performance Overviews")
    st.markdown("""
    - **Selected Architecture:** `LinearSVC` with TF-IDF vectorization
    - **Validation Macro-F1:** `0.7053`
    - **Unseen Test Set Macro-F1:** `0.7137`
    - **Unseen Test Accuracy:** `78.37%`
    """)
    st.subheader("Validation Performance Benchmark Summary")
    performance_df = pd.DataFrame({
        'Model': ['Dummy (Baseline)', 'Naive Bayes', 'Logistic Regression', 'Linear SVC (Best)'],
        'Accuracy (%)': [63.0, 76.96, 77.37, 77.60],
        'Macro-F1': [0.2569, 0.6778, 0.7021, 0.7053]
    })
    st.table(performance_df)
