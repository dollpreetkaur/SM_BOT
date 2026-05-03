import streamlit as st
import requests
import os 

st.set_page_config(page_title="Bot Intelligence", page_icon="🤖")

st.title("🤖 Social Media Bot Intelligence")
st.markdown("Analyze any Twitter/X account for automated behavior using Advanced XGBoost + SHAP.")

# --- DYNAMIC URL LOGIC ---
# On Render, we will set an environment variable called BACKEND_URL.
# Locally, it will default to localhost:8000.
BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

user_input = st.text_input("Enter Username (e.g., elonmusk):")

if st.button("Run Analysis"):
    if not user_input:
        st.warning("Please enter a username first!")
    else:
        with st.spinner("Scraping and Running AI Inference..."):
            try:
                # Use the dynamic BASE_URL
                response = requests.get(f"{BASE_URL}/analyze/{user_input}", timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Big Score Display
                    color = "red" if data['verdict'] == "Bot" else "green"
                    st.subheader(f"Verdict: :{color}[{data['verdict']}]")
                    
                    # Convert "85%" string to 0.85 float for the progress bar
                    prob_value = float(data['bot_probability'].replace('%','')) / 100
                    st.progress(prob_value)
                    st.write(f"Confidence Level: **{data['bot_probability']}**")
                    
                    st.divider()
                    st.subheader("💡 Why was this flagged?")
                    for r in data['top_reasons']:
                        st.info(f"Trigger detected: {r}")
                else:
                    st.error(f"API Error: {response.status_code}. Account might be private or doesn't exist.")
            
            except requests.exceptions.ConnectionError:
                st.error(f"Could not connect to the Backend. If you are local, start your FastAPI server. If on Render, check your BACKEND_URL setting. Current URL: {BASE_URL}")