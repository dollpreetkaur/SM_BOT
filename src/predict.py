import numpy as np
import joblib
import os

# Load model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'model.pkl')
model = joblib.load(MODEL_PATH)

def predict_bot(stats):
    """
    Bulletproof Input: Handles Dict, List, or even Numpy Arrays
    """
    try:
        # CASE 1: stats is a Dictionary (e.g., {'followers_count': 100...})
        if isinstance(stats, dict):
            features = [
                np.log1p(float(stats.get('followers_count', 0))),
                np.log1p(float(stats.get('statuses_count', 0))),
                np.log1p(float(stats.get('friends_count', 0))),
                np.log1p(float(stats.get('favourites_count', 0))),
                np.log1p(float(stats.get('listed_count', 0))),
                float(stats.get('age_days', 365))
            ]
        
        # CASE 2: stats is already a List or Array (e.g., [4.5, 2.1...])
        elif isinstance(stats, (list, np.ndarray)):
            # If it's a list, it's already processed, so just take the first 6
            features = [float(x) for x in stats[:6]]
            
        else:
            return 0.0

        # Inference
        # We wrap it in another list because predict_proba expects a 2D array [[f1, f2...]]
        prob = model.predict_proba([features])[0][1]
        return float(prob)

    except Exception as e:
        print(f"Prediction Error: {e}")
        return 0.0
    