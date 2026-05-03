import sys
import os
import numpy as np
from fastapi import FastAPI, HTTPException
from ntscraper import Nitter

# Add parent directory to sys.path so 'src' is discoverable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import predict_bot
from src.explain import get_explanation

app = FastAPI(title="Advanced Bot Detection API")

NITTER_INSTANCES = [
    'https://nitter.poast.org', 
    'https://nitter.privacydev.net', 
    'https://nitter.cz'
]

@app.get("/")
def home():
    return {"message": "Bot Intelligence API is running! Use /analyze/{username} to test."}

@app.get("/analyze/{username}")
async def analyze(username: str):
    try:
        scraper = Nitter(instances=NITTER_INSTANCES)
        
        # 1. Attempt to scrape
        try:
            profile = scraper.get_profile_info(username)
        except Exception:
            profile = None

        # 2. FALLBACK LOGIC (Demo Mode)
        if not profile or 'stats' not in profile:
            user_lower = username.lower().replace('@', '')
            
            if user_lower == "elonmusk":
                stats = {'followers': 185000000, 'tweets': 45000, 'following': 600, 'likes': 25000}
            elif user_lower == "nasa":
                stats = {'followers': 82000000, 'tweets': 75000, 'following': 200, 'likes': 18000}
            elif user_lower == "hourly_shiba":
                stats = {'followers': 500000, 'tweets': 150000, 'following': 10, 'likes': 50}
            elif user_lower == "bot_tester":
                stats = {'followers': 2, 'tweets': 8000, 'following': 4500, 'likes': 1}
            else:
                # If it's not a demo account and scraper fails, return "Unknown"
                return {
                    "username": username,
                    "bot_probability": "0.00%",
                    "verdict": "Unknown",
                    "top_reasons": ["Scraper blocked by X. Try 'elonmusk' or 'hourly_shiba' for demo."]
                }
        else:
            stats = profile['stats']
        
        # 3. Process data for Model
        raw_data = {
            'followers_count': stats.get('followers', 0),
            'statuses_count': stats.get('tweets', 0),
            'friends_count': stats.get('following', 0),
            'favourites_count': stats.get('likes', 0),
            'listed_count': 0, 
            'age_days': 365 
        }

        score = predict_bot(raw_data)
        
        # 4. Feature list for SHAP
        activity_rate = raw_data['statuses_count'] / max(raw_data['age_days'], 1)
        follower_ratio = raw_data['followers_count'] / (raw_data['followers_count'] + raw_data['friends_count'] + 1)

        feature_list = [
            np.log1p(raw_data['followers_count']), 
            np.log1p(raw_data['statuses_count']),
            np.log1p(raw_data['friends_count']), 
            np.log1p(raw_data['favourites_count']),
            np.log1p(raw_data['listed_count']), 
            float(raw_data['age_days']),
            float(activity_rate),
            float(follower_ratio)
        ]
        
        reasons = get_explanation(feature_list)

        return {
            "username": username,
            "bot_probability": f"{score*100:.2f}%",
            "verdict": "Bot" if score > 0.5 else "Human",
            "top_reasons": reasons
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))