from ntscraper import Nitter
import numpy as np

def fetch_live_user_data(username):
    scraper = Nitter()
    try:
        # Let it pick a random instance for better reliability
        user_info = scraper.get_profile_info(username)
        if not user_info or 'stats' not in user_info:
            return None
        
        stats = user_info['stats']
        # Use .get() with 0 as default to prevent crashes
        f = stats.get('followers', 0)
        fo = stats.get('following', 0)
        t = stats.get('tweets', 0)
        l = stats.get('likes', 0)
        li = 0  # Nitter rarely provides 'listed' count
        age_days = 365 # Keep as default for your deadline

        # Log Transforms (Match your Preprocessing)
        lf, lfo, lt, ll, lli, lage = np.log1p([f, fo, t, l, li, age_days])

        # Feature Engineering (Matches your XGBoost Model)
        data = {
            "followers": lf,
            "following": lfo,
            "tweets": lt,
            "likes": ll,
            "listed": lli,
            "ratio": lf / (lfo + 1),
            "tweets_per_following": lt / (lfo + 1),
            "account_age_days": lage,
            "tweets_per_day": lt / (lage + 1),
            "engagement": ll / (lt + 1),
            "followers_growth": lf / (lage + 1)
        }
        return data
    except Exception as e:
        print(f"Live Scrape Failed: {e}")
        return None