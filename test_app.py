
import pytest
from fastapi.testclient import TestClient
from app import app, model

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_predict_sentiment_positive():
    response = client.post(
        "/predict",
        json={"text": "The service on my flight today was absolutely amazing, thank you!"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] == "positive"

def test_predict_sentiment_negative():
    response = client.post(
        "/predict",
        json={"text": "This is the worst customer service, flight delayed and lost my bags."}
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] == "negative"

def test_predict_validation_error():
    # Empty body or missing field
    response = client.post("/predict", json={})
    assert response.status_code == 422
