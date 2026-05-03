import numpy as np
import joblib
import os

# Load model once to save memory
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'model.pkl')
model = joblib.load(MODEL_PATH)

def predict_bot(stats):
    """
    Input: Dictionary of raw stats from the scraper (or demo fallback)
    Output: Probability (0.0 to 1.0)
    """
    
    # 1. Feature Engineering (MATCHING YOUR 6-COLUMN MODEL)
    # We remove 'velocity' and 'reputation' because the model doesn't recognize them.
    
    age = float(stats.get('age_days', 365)) 
    
    features = [
        np.log1p(float(stats.get('followers_count', 0))),
        np.log1p(float(stats.get('statuses_count', 0))),
        np.log1p(float(stats.get('friends_count', 0))),
        np.log1p(float(stats.get('favourites_count', 0))),
        np.log1p(float(stats.get('listed_count', 0))),
        age
    ]

    # 2. Inference
    # Now passing exactly 6 columns in a 2D array shape
    try:
        prob = model.predict_proba([features])[0][1]
        return float(prob)
    except Exception as e:
        print(f"Inference Error: {e}")
        # If it still fails, check if the model is a regressor or classifier
        return 0.0