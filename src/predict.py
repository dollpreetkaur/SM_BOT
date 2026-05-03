import numpy as np
import joblib
import os

# Load model once to save memory
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'model.pkl')
model = joblib.load(MODEL_PATH)

def predict_bot(stats):
    """
    Input: Can be a dictionary OR a list
    Output: Probability (0.0 to 1.0)
    """
    # 1. Handle Input Type
    if isinstance(stats, dict):
        # If it's a dictionary (raw data), extract the 6 features
        features = [
            np.log1p(float(stats.get('followers_count', 0))),
            np.log1p(float(stats.get('statuses_count', 0))),
            np.log1p(float(stats.get('friends_count', 0))),
            np.log1p(float(stats.get('favourites_count', 0))),
            np.log1p(float(stats.get('listed_count', 0))),
            float(stats.get('age_days', 365))
        ]
    elif isinstance(stats, list):
        # If it's already a list, just use it (but ensure it's only the first 6)
        features = stats[:6]
    else:
        raise ValueError("Stats must be a list or a dictionary")

    # 2. Inference
    try:
        prob = model.predict_proba([features])[0][1]
        return float(prob)
    except Exception as e:
        print(f"Inference Error: {e}")
        return 0.0