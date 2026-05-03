import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from src.predict import predict
from src.explain import get_explanation

app = FastAPI()

class UserInput(BaseModel):
    followers: float
    following: float
    tweets: float
    likes: float
    listed: float
    account_age_days: float

@app.post("/predict")
def detect_bot(data: UserInput):
    # Map raw input to the 11 features the model expects
    f, fo, t, l, li, age = data.followers, data.following, data.tweets, data.likes, data.listed, data.account_age_days
    
    feats = {
    "followers": f, 
    "following": fo, 
    "tweets": t, 
    "likes": l, 
    "listed": li,
    "ratio": f / (fo + 1),  # +1 prevents crash
    "tweets_per_following": t / (fo + 1),
    "account_age_days": age,
    "tweets_per_day": t / (age + 1),
    "engagement": l / (t + 1),
    "followers_growth": f / (age + 1)
}

    # EXACT ORDER AS TRAIN.PY
    order = ["followers", "following", "tweets", "likes", "listed", "ratio", "tweets_per_following", "account_age_days", "tweets_per_day", "engagement", "followers_growth"]
    vector = [feats[name] for name in order]
    
    score = predict(vector)
    reasons = get_explanation(vector)
    return {"bot_score": float(score), "label": "bot" if score >= 0.5 else "human", "top_reasons": reasons} 