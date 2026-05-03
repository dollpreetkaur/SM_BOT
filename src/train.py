import pandas as pd
import joblib
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train():
    if not os.path.exists('data/cresciclean.csv'):
        print("❌ Error: Run preprocess.py first!")
        return

    df = pd.read_csv('data/cresciclean.csv')

    # THE ADVANCED FEATURE SET (Must match preprocess.py exactly)
    features = [
        'log_followers_count', 'log_statuses_count', 'log_friends_count',
        'log_favourites_count', 'log_listed_count', # 5 Log counts
        'account_age_days',                        # Age
        'tweet_velocity',                          # Behavioral 1
        'reputation'                               # Behavioral 2
    ]
    
    X = df[features]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = XGBClassifier(
        n_estimators=150, # Increased for advanced patterns
        max_depth=7,      # Deeper for complex ratios
        learning_rate=0.05,
        eval_metric='logloss'
    )

    print(f"🚀 Training Advanced XGBoost on {len(features)} features...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"✅ Model Accuracy: {accuracy_score(y_test, preds):.4f}")
    print("\nAdvanced Classification Report:")
    print(classification_report(y_test, preds))

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/model.pkl')
    print("🧠 Advanced Model saved to models/model.pkl")

if __name__ == '__main__':
    train()