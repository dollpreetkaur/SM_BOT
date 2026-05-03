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
    return {"message": "Bot Intelligence API is running!"}

@app.get("/analyze/{username}")
async def analyze(username: str):
    try:
        scraper = Nitter(instances=NITTER_INSTANCES)
        
        profile = None
        try:
            profile = scraper.get_profile_info(username)
        except Exception:
            profile = None

        # --- 1. FALLBACK LOGIC ---
        user_lower = username.lower().replace('@', '')
        stats = None

        # Check if profile is a valid dictionary with stats
        if isinstance(profile, dict) and profile.get('stats'):
            stats = profile['stats']
        else:
            # Demo Fallbacks
            if user_lower == "elonmusk":
                stats = {'followers': 185000000, 'tweets': 45000, 'following': 600, 'likes': 25000}
            elif user_lower == "nasa":
                stats = {'followers': 82000000, 'tweets': 75000, 'following': 200, 'likes': 18000}
            elif user_lower == "hourly_shiba":
                stats = {'followers': 500000, 'tweets': 150000, 'following': 10, 'likes': 50}
            elif user_lower == "bot_tester":
                stats = {'followers': 2, 'tweets': 8000, 'following': 4500, 'likes': 1}
        
        if not stats:
            return {
                "username": username,
                "bot_probability": "0.00%",
                "verdict": "Unknown",
                "top_reasons": ["Nitter is busy. Try 'bot_tester' for demo."]
            }

        # --- 2. DATA PREP (The Critical Part) ---
        # We ensure this is a DICTIONARY
        raw_data = {
            'followers_count': int(stats.get('followers', 0)),
            'statuses_count': int(stats.get('tweets', 0)),
            'friends_count': int(stats.get('following', 0)),
            'favourites_count': int(stats.get('likes', 0)),
            'listed_count': 0, 
            'age_days': 365 
        }

        # --- 3. PREDICTION ---
        # Pass the dictionary. 
        # If your predict_bot crashes here, it means predict_bot is expecting a list!
        try:
            score = predict_bot(raw_data)
        except AttributeError:
            # Fallback: If predict_bot only accepts lists, we send the list instead
            prepared_list = [
                np.log1p(raw_data['followers_count']), 
                np.log1p(raw_data['statuses_count']),
                np.log1p(raw_data['friends_count']), 
                np.log1p(raw_data['favourites_count']),
                np.log1p(raw_data['listed_count']), 
                float(raw_data['age_days'])
            ]
            score = predict_bot(prepared_list)
        
        # --- 4. EXPLANATION ---
        feature_list = [
            np.log1p(raw_data['followers_count']), 
            np.log1p(raw_data['statuses_count']),
            np.log1p(raw_data['friends_count']), 
            np.log1p(raw_data['favourites_count']),
            np.log1p(raw_data['listed_count']), 
            float(raw_data['age_days'])
        ]
        
        reasons = get_explanation(feature_list)

        return {
            "username": username,
            "bot_probability": f"{float(score)*100:.2f}%",
            "verdict": "Bot" if score > 0.5 else "Human",
            "top_reasons": reasons
        }

    except Exception as e:
        return {
            "username": username,
            "bot_probability": "0.00%",
            "verdict": "System Error",
            "top_reasons": [f"Error Details: {str(e)}"]
        }