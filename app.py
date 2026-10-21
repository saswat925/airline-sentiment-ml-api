import joblib
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Instantiate FastAPI
app = FastAPI(
    title="Airline Sentiment Analysis API",
    description="A production-ready FastAPI for predicting airline customer sentiment from tweets.",
    version="1.0.0"
)

# Setup NLP components
NEGATION_WORDS = {'not', 'no', 'never', 'neither', 'nor', 'none', 'cannot', "n't", 'hardly', 'scarcely', 'barely'}
try:
    from nltk.corpus import stopwords
    STOPWORDS_SET = set(stopwords.words('english')) - NEGATION_WORDS
except Exception:
    STOPWORDS_SET = set()

LEMMATIZER = WordNetLemmatizer()

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

# Load the serialized model pipeline
try:
    model = joblib.load('best_sentiment_pipeline.joblib')
except Exception as e:
    model = None
    print(f"Warning: Could not load model pipeline: {e}")

# Define Pydantic models for validation
class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The tweet or feedback text to analyze.", example="My flight got cancelled and nobody is helping!")

class SentimentResponse(BaseModel):
    text: str
    cleaned_text: str
    prediction: str

@app.get("/health")
def health_check():
    if model is None:
        return {"status": "unhealthy", "error": "Model pipeline not loaded."}
    return {"status": "healthy"}

@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(request: SentimentRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model pipeline is not loaded or unavailable.")

    cleaned = preprocess_text(request.text)
    # Predict sentiment using pre-trained pipeline
    prediction = model.predict([cleaned])[0]

    return SentimentResponse(
        text=request.text,
        cleaned_text=cleaned,
        prediction=prediction
    )
