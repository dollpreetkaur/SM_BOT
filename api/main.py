import sys
import os
import random
import numpy as np
from fastapi import FastAPI, HTTPException
from ntscraper import Nitter

# 1. FIX: Add the parent directory to sys.path so 'src' can be found on Render
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now we can safely import your local modules
from src.predict import predict_bot
from src.explain import get_explanation

app = FastAPI(title="Advanced Bot Detection API")

# 2. FIX: Manually provide Nitter instances to avoid "Empty Sequence" errors
NITTER_INSTANCES = [
    'https://nitter.net', 
    'https://nitter.cz', 
    'https://nitter.privacydev.net',
    'https://nitter.it',
    'https://nitter.sethforprivacy.com',
    'https://nitter.moomoo.me'
]

@app.get("/")
def home():
    return {"message": "Bot Intelligence API is running! Use /analyze/{username} to test."}

@app.get("/analyze/{username}")
async def analyze(username: str):
    try:
        # Initialize scraper with our specific instance list
        scraper = Nitter(instances=NITTER_INSTANCES)
        
        # 3. Scrape Live Data
        # We wrap this in a try-block because scraping is the most likely part to fail
        try:
            profile = scraper.get_profile_info(username)
        except Exception:
            profile = None

        # 4. Safety Check: If scraper is blocked or user doesn't exist
        if not profile or 'stats' not in profile:
            return {
                "username": username,
                "bot_probability": "0.00%",
                "verdict": "Unknown",
                "top_reasons": ["Error: X/Nitter is currently blocking the request or the account is private/non-existent."]
            }
        
        stats = profile['stats']
        
        # Map stats and handle missing values with .get()
        raw_data = {
            'followers_count': stats.get('followers', 0),
            'statuses_count': stats.get('tweets', 0),
            'friends_count': stats.get('following', 0),
            'favourites_count': stats.get('likes', 0),
            'listed_count': 0, 
            'age_days': 365 # Default placeholder as Nitter doesn't show exact account age
        }

        # 5. Get AI Prediction
        score = predict_bot(raw_data)
        
        # 6. Prepare features for SHAP (Must match the 8 features your model was trained on)
        # Added max(..., 1) to prevent DivisionByZero errors
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
        
        # 7. Get SHAP Explanations
        reasons = get_explanation(feature_list)

        return {
            "username": username,
            "bot_probability": f"{score*100:.2f}%",
            "verdict": "Bot" if score > 0.5 else "Human",
            "top_reasons": reasons
        }

    except Exception as e:
        # General catch-all for any other logic errors
        raise HTTPException(status_code=500, detail=f"API Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Use the PORT environment variable provided by Render
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)