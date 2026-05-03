import streamlit as st
import requests

st.set_page_config(page_title="Bot Intelligence", page_icon="🤖")

st.title("🤖 Social Media Bot Intelligence")
st.markdown("Analyze any Twitter/X account for automated behavior using Advanced XGBoost + SHAP.")

user_input = st.text_input("Enter Username (e.g., elonmusk):")

if st.button("Run Analysis"):
    with st.spinner("Scraping and Running AI Inference..."):
        # When deploying to Render, change localhost to your backend URL
        response = requests.get(f"http://localhost:8000/analyze/{user_input}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Big Score Display
            color = "red" if data['verdict'] == "Bot" else "green"
            st.subheader(f"Verdict: :{color}[{data['verdict']}]")
            st.progress(float(data['bot_probability'].replace('%','')) / 100)
            st.write(f"Confidence Level: **{data['bot_probability']}**")
            
            st.divider()
            st.subheader("💡 Why was this flagged?")
            for r in data['top_reasons']:
                st.info(f"Trigger detected: {r}")
        else:
            st.error("Account not found or API error.")