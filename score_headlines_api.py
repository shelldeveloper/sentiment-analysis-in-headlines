"""This script deploys a model to describe the sentiment of news headlines via a FastAPI web service."""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
import joblib

# Load model
app             = FastAPI()
# model           = SentenceTransformer("/opt/huggingface_models/all-MiniLM-L6-v2")
model           = SentenceTransformer("all-MiniLM-L6-v2")
clf             = joblib.load('svm.joblib')

# GET /status: Health check
@app.get('/status')
def status():
    """Health check endpoint that returns service status."""
    return {'status':'OK'}

# POST /score_headlines: Accept a list of headlines and return labels
class ListOfHeadlines(BaseModel):
    headlines: List[str]

@app.post('/score_headlines')
def score_headlines(headline_data: ListOfHeadlines):
    """Accepts a list of headlines and returns predicted sentiment labels."""
    embeddings  = model.encode(headline_data.headlines)
    predictions = clf.predict(embeddings)
    return {'labels': predictions.tolist()}

# fastapi dev score_headlines_api.py --port 8021
