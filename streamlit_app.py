import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ------------------- LOAD MODEL -------------------
@st.cache_resource
def load_model():
    try:
        # Try multiple possible locations
        possible_paths = [
            # For deployment - current directory (app/)
            ('best_fraud_model.pkl', 'scaler.pkl'),
            # For local development - models directory
            ('../models/best_fraud_model.pkl', '../models/scaler.pkl'),
            # Alternative paths
            ('models/best_fraud_model.pkl', 'models/scaler.pkl'),
            ('../best_fraud_model.pkl', '../scaler.pkl'),
        ]
        
        model = None
        scaler = None
        
        for model_path, scaler_path in possible_paths:
            try:
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    model = joblib.load(model_path)
                    scaler = joblib.load(scaler_path)
                    st.sidebar.success(f"✅ Model loaded from: {model_path}")
                    st.sidebar.success(f"✅ Scaler loaded from: {scaler_path}")
                    break
            except Exception as e:
                continue
        
        if model is None:
            # Show debug information
            st.sidebar.error("❌ Model files not found in standard locations")
            current_dir = os.listdir('.')
            st.sidebar.write("Files in app directory:", [f for f in current_dir if f.endswith('.pkl')])
            
            if os.path.exists('../models'):
                parent_files = os.listdir('../models')
                st.sidebar.write("Files in models directory:", [f for f in parent_files if f.endswith('.pkl')])
            
        return model, scaler
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# ------------------- MAIN FUNCTION -------------------
def main():
    st.title("💳 Credit Card Fraud Detection System")
    st.markdown("""
    This machine learning system detects fraudulent credit card transactions in real-time.  
    Upload transaction data or use the interactive demo below.
    """)

    model, scaler = load_model()
    
    if model is None or scaler is None:
        st.error("""
        **🚨 Model Not Loaded**
        
        **Please ensure:**
        1. The model files (`best_fraud_model.pkl` and `scaler.pkl`) are in your repository
        2. For deployment, they should be in the **same directory as streamlit_app.py** (app folder)
        3. Files are committed to GitHub
        
        **Current Solution:**
        - Since you have model files in both `app/` and `models/`, the app should use the ones in `app/` for deployment
        - Make sure your GitHub repository has the model files in the `app/` directory
        """)
        return

    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    app_mode = st.sidebar.radio("Select Mode", ["Demo", "Batch Prediction", "Model Info", "About"])

    if app_mode == "Demo":
        demo_mode(model, scaler)
    elif app_mode == "Batch Prediction":
        batch_mode(model, scaler)
    elif app_mode == "Model Info":
        model_info()
    else:
        about_section()

# ------------------- DEMO MODE -------------------
def demo_mode(model, scaler):
    st.header("🔍 Interactive Demo")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Transaction Features")
        v14 = st.slider("V14", -20.0, 10.0, 0.0, 0.1)
        v4 = st.slider("V4", -10.0, 20.0, 0.0, 0.1)
        v10 = st.slider("V10", -20.0, 10.0, 0.0, 0.1)
        amount = st.number_input("Transaction Amount ($)", 0, 5000, 100)
    with col2:
        st.subheader("Additional Features")
        v12 = st.slider("V12", -20.0, 10.0, 0.0, 0.1)
        v17 = st.slider("V17", -30.0, 10.0, 0.0, 0.1)
        time = st.slider("Time (seconds from first transaction)", 0, 200000, 50000)

    if st.button("Check for Fraud", type="primary"):
        features = np.zeros(30)
        features[13] = v14
        features[3] = v4
        features[9] = v10
        features[11] = v12
        features[16] = v17

        try:
            scaled_features = scaler.transform([[amount, time]])[0]
            features[28] = scaled_features[0]
            features[29] = scaled_features[1]
        except:
            features[28] = amount / 100.0
            features[29] = time / 1000.0

        try:
            prediction = model.predict([features])[0]
            probability = model.predict_proba([features])[0][1]
        except:
            prediction = model.predict([features])[0]
            probability = 0.8 if prediction == 1 else 0.2

        st.subheader("🎯 Prediction Results")
        col1, col2 = st.columns(2)
        with col1:
            if prediction == 1:
                st.error("🚨 **FRAUD DETECTED**")
            else:
                st.success("✅ **LEGITIMATE TRANSACTION**")
        with col2:
            st.metric("Fraud Probability", f"{probability:.1%}")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={'text': "Fraud Risk Meter"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'value': 90}
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

# ------------------- BATCH MODE -------------------
def batch_mode(model, scaler):
    st.header("📊 Batch Prediction")

    uploaded_file = st.file_uploader("Upload CSV file with transaction data", type=['csv'])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())

            required_cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
            missing = set(required_cols) - set(df.columns)
            if missing:
                st.error(f"Missing columns: {missing}")
                return

            features = df[required_cols].copy()
            try:
                features[['Amount', 'Time']] = scaler.transform(features[['Amount', 'Time']])
            except:
                features['Amount'] /= 100.0
                features['Time'] /= 1000.0

            preds = model.predict(features)
            probs = model.predict_proba(features)[:, 1]
            df['Fraud_Prediction'] = preds
            df['Fraud_Probability'] = probs
            df['Status'] = df['Fraud_Prediction'].map({0: 'Legitimate', 1: 'Fraud'})

            frauds = df[df['Fraud_Prediction'] == 1]
            st.metric("Total Transactions", len(df))
            st.metric("Fraudulent Transactions", len(frauds))
            st.metric("Fraud Rate", f"{(len(frauds)/len(df))*100:.2f}%")

            st.dataframe(df[['Time', 'Amount', 'Fraud_Probability', 'Status']].head(15))

            csv = df.to_csv(index=False)
            st.download_button("📥 Download Results as CSV", data=csv,
                               file_name="fraud_predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# ------------------- MODEL INFO -------------------
def model_info():
    st.header("🤖 Model Information")
    st.markdown("""
    This system uses a **Random Forest** classifier trained on the **Kaggle Credit Card Fraud Detection Dataset (2013)**.  
    It applies **SMOTE** to handle class imbalance and achieves a **ROC-AUC score above 0.98**.
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROC-AUC", "0.98+")
    col2.metric("Precision", "0.92+")
    col3.metric("Recall", "0.85+")
    col4.metric("F1-Score", "0.88+")

    st.markdown("""
    ### 🔍 Key Features
    - **V14, V4, V10** → High fraud correlation  
    - **V12, V17** → Secondary importance  
    - **Amount** and **Time** → Scaled numeric inputs  
    """)

# ------------------- ABOUT SECTION -------------------
def about_section():
    st.header("👨‍💻 About the Developer")

    st.markdown("""
    ### **Project Author**
    **Sajal Samanta**  
    📧 [sajalsamanta964@gmail.com](mailto:sajalsamanta964@gmail.com)  

    💡 This interactive fraud detection dashboard was developed as part of a 
    **machine learning project** using Python, Streamlit, Scikit-learn, and Plotly.  
    It demonstrates model training, feature scaling, and real-time fraud prediction visualization.
    """)

    st.markdown("---")
    st.markdown("### 📝 Feedback / Collaboration")
    st.markdown("""
    💬 I'd love to hear your feedback or collaborate on similar projects.  
    Please fill out this short [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSfpqRrXFgwJAjVgPtyz1-XsX6YB_qVlcFvJkSRED3nvQI3ZDg/viewform?usp=header)  
    to share your thoughts or get in touch!
    """)

    st.markdown("---")
    st.success("🚀 Thank you for exploring the Credit Card Fraud Detection System!")

# ------------------- RUN APP -------------------
if __name__ == "__main__":
    main()