import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

# --- CONFIGURATION ---
st.set_page_config(
    page_title="BotIntel | Advanced AI Detector",
    page_icon="🤖",
    layout="wide"
)

# Replace with your actual Render API URL
BACKEND_URL = "https://sm-bot-api.onrender.com/analyze/"

# --- CUSTOM CSS FOR SOPHISTICATION ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3e4259;
    }
    .bot-card {
        padding: 20px;
        border-radius: 15px;
        background: linear-gradient(135deg, #2b313e 0%, #1a1f25 100%);
        color: white;
        border-left: 5px solid #ff4b4b;
    }
    .human-card {
        padding: 20px;
        border-radius: 15px;
        background: linear-gradient(135deg, #1e3a3a 0%, #1a1f25 100%);
        color: white;
        border-left: 5px solid #00d4ff;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛡️ BotIntel AI")
    st.info("Detecting automated behavior using Neural Networks and SHAP explainability.")
    st.divider()
    st.markdown("### **Try Demo Accounts:**")
    st.code("elonmusk\nhourly_shiba\nbot_tester")

# --- MAIN UI ---
st.title("🤖 Social Intelligence Dashboard")
st.markdown("Analyze X (Twitter) accounts for bot-like patterns and automation.")

# Search Layout
col1, col2 = st.columns([3, 1])
with col1:
    username = st.text_input("Enter X Username", placeholder="e.g. elonmusk", help="Type the handle without the @ symbol.")
with col2:
    st.write("##") # Alignment
    search_btn = st.button("Run Analysis", use_container_width=True)

if search_btn and username:
    with st.spinner(" AI is analyzing account patterns..."):
        try:
            response = requests.get(f"{BACKEND_URL}{username}")
            
            if response.status_code == 200:
                data = response.json()
                
                # Handling "Unknown" verdict (Scraper Block)
                if data.get("verdict") == "Unknown":
                    st.warning(f"⚠️ {data['top_reasons'][0]}")
                else:
                    # --- RESULTS LAYOUT ---
                    st.divider()
                    
                    res_col1, res_col2 = st.columns([1, 2])
                    
                    with res_col1:
                        # Circular Probability Gauge
                        prob_val = float(data['bot_probability'].strip('%'))
                        
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = prob_val,
                            title = {'text': "Bot Probability"},
                            gauge = {
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "#ff4b4b" if prob_val > 50 else "#00d4ff"},
                                'steps': [
                                    {'range': [0, 50], 'color': "#1a1f25"},
                                    {'range': [50, 100], 'color': "#2b313e"}
                                ],
                            }
                        ))
                        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
                        st.plotly_chart(fig, use_container_width=True)

                    with res_col2:
                        # Verdict Card
                        v_class = "bot-card" if prob_val > 50 else "human-card"
                        v_icon = "🤖" if prob_val > 50 else "👤"
                        st.markdown(f"""
                            <div class="{v_class}">
                                <h2>{v_icon} Verdict: {data['verdict']}</h2>
                                <p>The AI model is {prob_val}% confident that this account shows {data['verdict'].lower()} characteristics.</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("### 🔍 Key Discovery Reasons")
                        for reason in data['top_reasons']:
                            st.write(f"- {reason}")

                    # --- DETAILED METRICS ---
                    st.divider()
                    st.subheader("📊 Account Fingerprint")
                    m1, m2, m3, m4 = st.columns(4)
                    
                    # Note: These values are parsed from the reasons or could be added to API response
                    # For now, we show the verdict focus
                    m1.metric("Status", data['verdict'])
                    m2.metric("Confidence", data['bot_probability'])
                    m3.metric("Type", "Supervised Learning")
                    m4.metric("Explainability", "SHAP Enabled")

            else:
                st.error("🚨 API connection failed. Please check if the backend is live.")
                
        except Exception as e:
            st.error(f"Connection Error: {e}")

else:
    # Empty State
    st.divider()
    st.image("https://images.unsplash.com/photo-1550751827-4bd374c3f58b?auto=format&fit=crop&q=80&w=1000", caption="BotIntel Artificial Intelligence", use_column_width=True)