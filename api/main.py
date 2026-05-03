from fastapi import FastAPI, HTTPException
from ntscraper import Nitter
from src.predict import predict_bot
from src.explain import get_explanation

app = FastAPI(title="Advanced Bot Detection API")
scraper = Nitter()

@app.get("/analyze/{username}")
async def analyze(username: str):
    try:
        # 1. Scrape Live Data
        # Using Nitter (Twitter Mirror) to get real-time stats
        profile = scraper.get_profile_info(username)
        if not profile:
            raise HTTPException(status_code=404, detail="User not found")
        
        stats = profile['stats']
        # Map Nitter stats to our model's format
        raw_data = {
            'followers_count': stats['followers'],
            'statuses_count': stats['tweets'],
            'friends_count': stats['following'],
            'favourites_count': stats['likes'],
            'listed_count': 0, # Nitter doesn't always provide listed
            'age_days': 365     # Placeholder
        }

        # 2. Get AI Prediction
        score = predict_bot(raw_data)
        
        # 3. Get SHAP Explanations
        # Convert raw_data to the 8-feature list for SHAP
        feature_list = [
            np.log1p(raw_data['followers_count']), 
            np.log1p(raw_data['statuses_count']),
            np.log1p(raw_data['friends_count']), 
            np.log1p(raw_data['favourites_count']),
            np.log1p(raw_data['listed_count']), 
            raw_data['age_days'],
            raw_data['statuses_count'] / raw_data['age_days'],
            raw_data['followers_count'] / (raw_data['followers_count'] + raw_data['friends_count'] + 1)
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