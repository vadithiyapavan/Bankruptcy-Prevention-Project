import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model_path = 'logistic_regression_model.pkl'
classifier = joblib.load(model_path)

# Set the page configuration
st.set_page_config(
    page_title="Bankruptcy Prediction",
    page_icon="📉",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 16px;
        }
        .stButton > button:hover {
            background-color: #45a049;
        }
        .stTitle {
            color: #2E86C1;
            font-weight: bold;
            text-align: center;
        }
        .stHeader {
            color: #283747;
            text-align: center;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Title of the application
st.markdown('<h1 class="stTitle">📉 Bankruptcy Prediction System</h1>', unsafe_allow_html=True)

# Input form for user inputs
st.markdown('<h3 class="stHeader">Enter Feature Values:</h3>', unsafe_allow_html=True)

# User-friendly sliders for input
industrial_risk = st.slider("📊 Industrial Risk", 0.0, 1.0, 0.5, help="Rate the industrial risk from 0 (low) to 1 (high)")
management_risk = st.slider("🧑‍💼 Management Risk", 0.0, 1.0, 0.5, help="Rate the management risk from 0 (low) to 1 (high)")
financial_flexibility = st.slider("💰 Financial Flexibility", 0.0, 1.0, 0.5, help="Rate the financial flexibility from 0 (low) to 1 (high)")
credibility = st.slider("🤝 Credibility", 0.0, 1.0, 0.5, help="Rate the credibility from 0 (low) to 1 (high)")
competitiveness = st.slider("🏆 Competitiveness", 0.0, 1.0, 0.5, help="Rate the competitiveness from 0 (low) to 1 (high)")
operating_risk = st.slider("⚙️ Operating Risk", 0.0, 1.0, 0.5, help="Rate the operating risk from 0 (low) to 1 (high)")

# Collect the features into a single array
features = np.array([[industrial_risk, management_risk, financial_flexibility, credibility, competitiveness, operating_risk]])

# Button to make predictions
if st.button("🔍 Predict Bankruptcy"):
    prediction = classifier.predict(features)
    probability = classifier.predict_proba(features)[:, 1]  # Probability of positive class

    # Display prediction with enhanced messages
    st.markdown('<hr>', unsafe_allow_html=True)
    if prediction[0] == 1:
        st.error(f"🚨 **The company is at high risk of bankruptcy!**\n\n**Probability:** {probability[0]:.2f}", icon="⚠️")
        st.markdown(
            """
            <ul>
                <li>📌 Consider reducing operational risks.</li>
                <li>📌 Focus on improving financial flexibility.</li>
                <li>📌 Enhance management strategies.</li>
            </ul>
            """, unsafe_allow_html=True
        )
    else:
        st.success(f"✅ **The company is NOT at risk of bankruptcy.**\n\n**Probability:** {1 - probability[0]:.2f}", icon="💡")
        st.markdown(
            """
            <ul>
                <li>📈 Maintain current strategies to ensure stability.</li>
                <li>📊 Continue monitoring industry risks.</li>
                <li>🌟 Stay competitive in the market.</li>
            </ul>
            """, unsafe_allow_html=True
        )
