import numpy as np
import joblib
import os
import json
from fastapi import FastAPI
from pathlib import Path

# 1. INITIALIZE THE APP (This fixes your NameError)
app = FastAPI()

# 2. SETUP PATHS
project_root = Path(__file__).resolve().parent.parent
MODEL_PATH = project_root / "models" / "model.pkl"
VAULT_DIR = project_root / "data" / "forensic_vault"

# 3. LOAD THE BRAIN
if MODEL_PATH.exists():
    model = joblib.load(str(MODEL_PATH))
    print("✅ Advanced XGBoost Model loaded successfully.")
else:
    print(f"❌ ERROR: Model not found at {MODEL_PATH}")
    model = None

# 4. THE ANALYZE ROUTE
@app.get("/analyze/{username}")
async def analyze(username: str):
    username = username.lower().strip().replace('@', '')
    vault_path = VAULT_DIR / f"{username}.json"

    # --- DATA INGESTION ---
    if vault_path.exists():
        with open(vault_path, 'r') as f:
            raw_data = json.load(f)
    else:
        return {"error": "Account not in vault and Live Bridge is rate-limited."}

    # --- FEATURE ENGINEERING (Matches your 8-feature train.py) ---
    stats = raw_data.get('stats', {})
    
    # We apply the manual Log1p scaling here
    features = np.array([[
        np.log1p(float(stats.get('followers', 0))), 
        np.log1p(float(stats.get('tweets', 0))),    
        np.log1p(float(stats.get('following', 0))), 
        np.log1p(float(stats.get('likes', 0))),     
        0.0,                                        # log_listed_count
        365.0,                                      # account_age_days
        float(stats.get('tweets', 0)) / 365.0,      # tweet_velocity
        float(stats.get('followers', 0)) / (float(stats.get('following', 0)) + 1) # reputation
    ]])

    # --- ML INFERENCE ---
    probability = model.predict_proba(features)[0][1]
    
    return {
        "username": username,
        "bot_probability": f"{probability * 100:.2f}%",
        "verdict": "Bot" if probability > 0.5 else "Human"
    }