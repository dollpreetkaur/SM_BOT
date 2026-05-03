import numpy as np
import joblib
import os

# Load model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'model.pkl')
model = joblib.load(MODEL_PATH)

def predict_bot(stats):
    """
    Hybrid Input: Works with Dict or List
    """
    # If the API sent a dictionary
    if isinstance(stats, dict):
        features = [
            np.log1p(float(stats.get('followers_count', 0))),
            np.log1p(float(stats.get('statuses_count', 0))),
            np.log1p(float(stats.get('friends_count', 0))),
            np.log1p(float(stats.get('favourites_count', 0))),
            np.log1p(float(stats.get('listed_count', 0))),
            float(stats.get('age_days', 365))
        ]
    # If the API sent a list by mistake
    elif isinstance(stats, list):
        features = stats[:6] # Take only the first 6 items
    else:
        return 0.0

    # Inference
    try:
        prob = model.predict_proba([features])[0][1]
        return float(prob)
    except Exception as e:
        print(f"Model Error: {e}")
        return 0.0