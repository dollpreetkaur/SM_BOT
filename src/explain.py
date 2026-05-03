import shap
import joblib
import pandas as pd

def get_explanation(feature_vector):
    # Load model
    model = joblib.load('models/model.pkl')
    
    # Feature names must match training
    feature_names = ['log_followers_count', 'log_statuses_count', 'log_friends_count', 
                     'age_days', 'tweet_velocity', 'reputation']
    
    # Create DataFrame for SHAP
    df_input = pd.DataFrame([feature_vector], columns=feature_names)
    
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_input)
    
    # Extract top 2 contributing features
    vals = shap_values[0]
    top_indices = vals.argsort()[-2:][::-1]
    
    reasons = []
    for i in top_indices:
        reason = f"High {feature_names[i].replace('_', ' ')}" if vals[i] > 0 else f"Low {feature_names[i].replace('_', ' ')}"
        reasons.append(reason)
        
    return reasons