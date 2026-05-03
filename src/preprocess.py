import pandas as pd
import numpy as np
import os
import warnings

# Suppress the datetime warning
warnings.filterwarnings("ignore", category=UserWarning)

def preprocess():
    data_dir = 'data/cresci-2017/'
    output_path = 'data/cresciclean.csv'

    files = [
        ('genuine_accounts.csv', 0),
        ('social_spambots_1.csv', 1), ('social_spambots_2.csv', 1), ('social_spambots_3.csv', 1),
        ('traditional_spambots_1.csv', 1), ('traditional_spambots_2.csv', 1), ('traditional_spambots_3.csv', 1)
    ]

    dfs = []
    for filename, label in files:
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['label'] = label
            dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)

    # --- ADVANCED FEATURE ENGINEERING ---
    full_df['created_at'] = pd.to_datetime(full_df['created_at'], errors='coerce').dt.tz_localize(None)
    full_df['crawled_at'] = pd.to_datetime(full_df['crawled_at'], errors='coerce').dt.tz_localize(None)
    
    # 1. Standardizing Account Age Name
    full_df['account_age_days'] = (full_df['crawled_at'] - full_df['created_at']).dt.days.fillna(0).clip(lower=1)
    
    # 2. Tweet Velocity
    full_df['tweet_velocity'] = full_df['statuses_count'] / full_df['account_age_days']
    
    # 3. Reputation Ratio
    full_df['reputation'] = full_df['followers_count'] / (full_df['followers_count'] + full_df['friends_count'] + 1)
    
    # 4. Log transforms (Ensure names match train.py)
    count_cols = ['followers_count', 'friends_count', 'statuses_count', 'favourites_count', 'listed_count']
    for col in count_cols:
        full_df[f'log_{col}'] = np.log1p(full_df[col].fillna(0))

    full_df.to_csv(output_path, index=False)
    print(f"✅ Advanced Dataset Created with all required columns: {output_path}")

if __name__ == '__main__':
    preprocess()