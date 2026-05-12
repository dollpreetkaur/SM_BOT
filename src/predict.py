import numpy as np
import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'model.pkl')
model = joblib.load(MODEL_PATH)

def predict_bot(feature_list):
    try:
        # Convert to list if it's a numpy array
        data = list(feature_list)
        
        # FORCE exactly 9 features. No more, no less.
        if len(data) < 9:
            data.extend([0.0] * (9 - len(data)))
        elif len(data) > 9:
            data = data[:9]
            
        # Final check - if this isn't 9, the code will raise an error here
        assert len(data) == 9, f"Feature count mismatch! Expected 9, got {len(data)}"

        # Reshape for XGBoost (1 sample, 9 features)
        input_data = np.array(data).reshape(1, -1)
        
        # Get probability
        prob = model.predict_proba(input_data)[0][1]
        return float(prob)
    except Exception as e:
        raise ValueError(f"XGBoost Input Error: {str(e)}")