import os
import gdown
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

from sklearn.preprocessing import StandardScaler
from detector import EnhancedIsolationForestDetector 


model_path = "fraud_detector_model.pkl"

# Only download if file not already cached
if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?id=1sjPsY2SIeTuzbu-Y5_2yWzatE6CFP3LL"
    gdown.download(url, model_path, quiet=False)


detector = EnhancedIsolationForestDetector()
detector.load_model(model_path)


# -------------------- Load Model --------------------
detector = EnhancedIsolationForestDetector()
detector.load_model('fraud_detector_model.pkl')

st.title("🚨 Transaction Fraud Detection")

# -------------------- Input Method --------------------
option = st.radio("Select input method:", ["Upload CSV", "Single Transaction"])

# -------------------- CSV Upload --------------------
if option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your transaction CSV", type=["csv"])
    
    if uploaded_file:
        transaction_data = pd.read_csv(uploaded_file)
        st.write("Dataset loaded:", transaction_data.shape)

        # Preprocess and scale
        X_features = detector.preprocess_features(transaction_data, fit_encoders=False)
        X_scaled = detector.scaler.transform(X_features)

        # Predict
        results = detector.predict_fraud(X_scaled)
        transaction_data['is_fraud'] = results['is_fraud']
        transaction_data['confidence_score'] = results['confidence_score']

        # Show flagged transactions
        flagged = transaction_data[transaction_data['is_fraud'] == 1]
        st.subheader(f"Flagged Transactions: {len(flagged)}")
        st.dataframe(flagged)

        # -------------------- SHAP Explainability --------------------
        st.subheader("SHAP Feature Importance (sample of flagged transactions)")

        if len(flagged) > 0:
            # Use only a sample for speed
            sample_idx = flagged.index[:200]  # max 200 for fast SHAP
            X_shap = X_scaled[sample_idx]

            explainer = shap.TreeExplainer(detector.iso_forest)
            shap_values = explainer.shap_values(X_shap)

            shap.summary_plot(shap_values, X_shap, feature_names=detector.feature_names, show=False)
            st.pyplot(bbox_inches='tight')  # Display plot in Streamlit
        else:
            st.write("No flagged transactions to explain.")

# -------------------- Single Transaction --------------------
elif option == "Single Transaction":
    st.subheader("Enter Transaction Details")
    
    # Example input fields (extend as needed)
    transaction_id = st.text_input("Transaction ID")
    user_id = st.text_input("User ID")
    amount = st.number_input("Amount", min_value=0.0)
    channel = st.selectbox("Channel", ["web", "mobile", "agent"])
    hour = st.slider("Transaction Hour", 0, 23)
    day_of_week = st.slider("Day of Week", 0, 6)
    user_home_state = st.text_input("User Home State")
    transaction_state = st.text_input("Transaction State")
    distance_from_home_km = st.number_input("Distance from Home (km)", min_value=0.0)
    ip_address = st.text_input("IP Address")
    user_home_ip = st.text_input("User Home IP")
    time_of_day = st.selectbox("Time of Day", ["morning", "afternoon", "evening", "night"])

    if st.button("Predict Fraud"):
        input_df = pd.DataFrame([{
            "transaction_id": transaction_id,
            "user_id": user_id,
            "amount": amount,
            "channel": channel,
            "hour": hour,
            "day_of_week": day_of_week,
            "user_home_state": user_home_state,
            "transaction_state": transaction_state,
            "distance_from_home_km": distance_from_home_km,
            "ip_address": ip_address,
            "user_home_ip": user_home_ip,
            "time_of_day": time_of_day
        }])

        X_features = detector.preprocess_features(input_df, fit_encoders=False)
        X_scaled = detector.scaler.transform(X_features)
        result = detector.predict_fraud(X_scaled)

        st.write("Prediction Result:")
        st.write(result)

        # SHAP for single transaction
        st.subheader("SHAP Explanation for This Transaction")
        explainer = shap.TreeExplainer(detector.iso_forest)
        shap_values = explainer.shap_values(X_scaled)
        shap.force_plot(explainer.expected_value, shap_values[0], X_scaled[0], feature_names=detector.feature_names, matplotlib=True, show=False)
        st.pyplot(bbox_inches='tight')
