import streamlit as st
import requests
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import os

st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("Real-Time Credit Card Fraud Detection")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
scaler = joblib.load(os.path.join(BASE_DIR, '../data/scaler.pkl'))
explainer, x_test = joblib.load(os.path.join(BASE_DIR, '../data/shap_data.pkl'))

# sidebar Transaction Input
st.sidebar.header("Enter Transaction Details")

amount = st.sidebar.number_input("Transaction Amount ", min_value=0.0, value=100.00)

st.sidebar.markdown("**Feature Values (V1-V28)**")
features = []
for i in range(1,29):
    val = st.sidebar.number_input(f"V{i}", min_value=-20.0, max_value=20.0, value=0.0, step=0.000001, format="%.5f")
    features.append(val)

scaled_amount = scaler.transform([[amount]])[0][0]
features.append(scaled_amount)


if st.button("Check Transaction"):
    with st.spinner("Contacting API... please wait (may take 30s on first request)"):
        try:
            # Send request to FastAPI
            response = requests.post(
                "https://fraud-detection-api-7g34.onrender.com/predict",
                json={"features": features},
                timeout= 60    
            )

            if response.status_code == 200:
                result = response.json()
                fraud = result["fraud"]
                prob = result["probability"]

                st.markdown("---------------")
                col1, col2 = st.columns(2)

                with col1:
                    if fraud:
                        st.error(f"ALERT!!!! - FRAUD DETECTED")
                    else:
                        st.success(f"LEGITIMATE TRANSACTION")

                with col2:
                    st.metric("Fraud Probabitlity: ", f"{prob*100:.2f}%")

                # Gauge Bar
                st.markdown("**Risk Level**")
                st.progress(prob)

                # SHAP Explanation
                st.markdown("-----------")
                st.subheader("Why did the model decide this?")

                input_df = pd.DataFrame([features], columns=x_test.columns)
                shap_vals = explainer.shap_values(input_df)

                fig, ax = plt.subplots(figsize=(10,5))
                shap.summary_plot(shap_vals, input_df, plot_type = "bar", show = False)
                st.pyplot(fig)

            # else:
            #     st.error("API not reachable. Make sure FastAPI is running.")

        except requests.exceptions.Timeout:
            st.error("API is waking up, please click the button again in 30 seconds. ")
        except requests.exceptions.ConnectionError:
            st.error("API is not reachable. Make sure FastAPI is running.")
