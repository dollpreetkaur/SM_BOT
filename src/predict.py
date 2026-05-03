import numpy as np
import joblib
import os

# Load model once to save memory
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'model.pkl')
model = joblib.load(MODEL_PATH)

def predict_bot(stats):
    """
    Input: Dictionary of raw stats from the scraper
    Output: Probability (0.0 to 1.0)
    """
    # 1. Feature Engineering (Must match the 9 features in train.py)
    age = stats.get('age_days', 365) # Default to 1 year if unknown
    
    features = [
        np.log1p(stats['followers_count']),
        np.log1p(stats['statuses_count']),
        np.log1p(stats['friends_count']),
        np.log1p(stats['favourites_count']),
        np.log1p(stats['listed_count']),
        age,
        stats['statuses_count'] / age, # velocity
        stats['followers_count'] / (stats['followers_count'] + stats['friends_count'] + 1) # reputation
    ]

    # 2. Inference
    prob = model.predict_proba([features])[0][1]
    return float(prob)