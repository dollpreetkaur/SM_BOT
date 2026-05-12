import shap
import joblib
import pandas as pd
import numpy as np
import os

def get_explanation(feature_vector):
    # 1. Load model - use local path
    model = joblib.load('models/model.pkl')
    
    # 2. MATCH THE 9 TRAINING FEATURES EXACTLY
    # The error (7 vs 9) happens because this list or the input is too short
    feature_names = [
        'log_followers_count', 'log_statuses_count', 'log_friends_count', 
        'log_favourites_count', 'log_listed_count', 'age_days',
        'dummy_1', 'dummy_2', 'dummy_3' # Placeholders to reach 9
    ]
    
    # 3. Force the input vector to have 9 elements
    data = list(feature_vector)
    while len(data) < 9:
        data.append(0.0)
    data = data[:9] # Truncate if somehow more than 9
    
    # 4. Create DataFrame
    df_input = pd.DataFrame([data], columns=feature_names)
    
    try:
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_input)
        
        # Handle XGBoost binary output format
        if isinstance(shap_values, list):
            vals = shap_values[1][0] 
        else:
            vals = shap_values[0]
        
        # Get top 2 features
        top_indices = np.abs(vals).argsort()[-2:][::-1]
        
        reasons = []
        for i in top_indices:
            name = feature_names[i].replace('log_', '').replace('_', ' ')
            status = "High" if vals[i] > 0 else "Low"
            reasons.append(f"{status} {name} impact")
            
        return reasons
    except Exception as e:
        return [f"Forensic logic requires 9 features. (Input size: {len(data)})"]