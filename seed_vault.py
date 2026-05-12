import json
import os

# Create the directory if it doesn't exist
VAULT_DIR = "data/forensic_vault"
os.makedirs(VAULT_DIR, exist_ok=True)

# Sample data for the "Forensic Lab"
# These numbers represent typical profiles for different account types
seed_data = {
    "nasa": {
        "stats": {"followers": 80000000, "following": 300, "tweets": 72000, "likes": 15000},
        "description": "Official NASA account. High reputation, established age."
    },
    "billgates": {
        "stats": {"followers": 65000000, "following": 400, "tweets": 4500, "likes": 200},
        "description": "High reputation human profile."
    },
    "tinycarebot": {
        "stats": {"followers": 120000, "following": 1, "tweets": 50000, "likes": 10},
        "description": "Utility Bot. High tweet velocity, very low following."
    },
    "bot_tester_99": {
        "stats": {"followers": 5, "following": 4500, "tweets": 12000, "likes": 2},
        "description": "Classic Spam Bot. High follow-churn, high activity, low followers."
    },
    "new_user_123": {
        "stats": {"followers": 50, "following": 150, "tweets": 20, "likes": 100},
        "description": "New human user. Low activity, balanced ratios."
    },
    "crypto_spam_99": {
        "stats": {"followers": 12, "following": 3800, "tweets": 15000, "likes": 5},
        "description": "High follow-to-follower ratio. High tweet velocity."
    },

    "airdrop_alert_bot": {
        "stats": {"followers": 150, "following": 0, "tweets": 85000, "likes": 0},
        "description": "Extreme tweet volume, zero engagement with others."
    },
    "politics_echo_01": {
        "stats": {"followers": 80, "following": 2100, "tweets": 450, "likes": 12000},
        "description": "Likely a 'Like Bot' used for artificial engagement."
    }
}

def seed():
    print(f"🚀 Seeding Forensic Vault at {VAULT_DIR}...")
    for username, data in seed_data.items():
        file_path = os.path.join(VAULT_DIR, f"{username}.json")
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"✅ Created vault entry for: @{username}")

if __name__ == "__main__":
    seed()