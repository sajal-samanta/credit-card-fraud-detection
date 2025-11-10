
# 💳 Credit Card Fraud Detection System

An interactive **machine learning web app** built with **Streamlit** that detects fraudulent credit card transactions in real-time.  
This project demonstrates the use of **classification algorithms**, **data balancing (SMOTE)**, and **feature scaling** to identify suspicious activities in financial data.

---

## 🚀 Live Demo
🔗 **[Try the App Here](https://credit-card-fraud-detection-djpk2qfpzyfov9yfmavpsr.streamlit.app/)**  


---

## 🧠 Project Overview

Financial fraud detection is one of the most critical challenges in fintech.  
This project leverages machine learning to predict whether a credit card transaction is **fraudulent or legitimate** using anonymized transaction data.

The app provides:
- **Real-time fraud detection demo** using user-controlled feature sliders  
- **Batch prediction mode** for uploaded CSV transaction data  
- **Model performance insights and metrics visualization**  
- **Contact section** for collaboration and feedback  

---

## 🧩 Tech Stack

| Category | Tools Used |
|-----------|-------------|
| **Frontend / UI** | Streamlit, Plotly |
| **Machine Learning** | Scikit-learn, RandomForestClassifier, SMOTE |
| **Data Handling** | Pandas, NumPy |
| **Model Deployment** | Streamlit Cloud / Localhost |
| **Model Persistence** | Joblib |

---



# 4️⃣ Run the app
 cd fraud_project
 
  py -m streamlit run app/streamlit_app.py



  
📁 Folder Structure
swift
Copy code
creditcard_fraud_detection/
│
├── app.py                      ← Streamlit main app file
├── models/
│   ├── best_fraud_model.pkl     ← Trained model
│   └── scaler.pkl               ← Feature scaler
├── requirements.txt
├── README.md
└── (optional) data/             ← Training dataset (not required for deployment)






graph TD
    A[Transaction Data] --> B[Data Preprocessing]
    B --> C[Feature Scaling]
    C --> D[SMOTE Balancing]
    D --> E[Model Training]
    E --> F[Random Forest]
    E --> G[XGBoost]
    E --> H[SVM]
    E --> I[Logistic Regression]
    F --> J[Model Selection]
    J --> K[Streamlit Web App]
    K --> L[Real-time Prediction]
    K --> M[Batch Analysis]
    K --> N[Visual Analytics]

    
📊 Model Information
Algorithm: Random Forest Classifier

Data Preprocessing: StandardScaler, SMOTE (to handle class imbalance)

Evaluation Metrics:

ROC-AUC Score: 0.98+

Precision: 0.92+

Recall: 0.85+

F1-Score: 0.88+

🔍 Key Predictive Features
Feature	Description
V14, V4, V10	Strongly correlated with fraudulent behavior
V12, V17	Secondary indicators
Amount, Time	Scaled numeric inputs

🧪 Features in the Web App
🎛️ Demo Mode: Adjust parameters interactively to simulate transactions

📂 Batch Mode: Upload CSV files for bulk prediction

📈 Model Info: View performance metrics and feature importance

👨‍💻 About Developer: Contact and collaboration details

🧑‍💻 Developer
Sajal Samanta
📧 sajalsamanta964@gmail.com

💡 Passionate about data science, machine learning, and AI-powered financial solutions.
Feel free to reach out for collaborations or feedback!

📝 Feedback / Contact
I’d love to hear your thoughts on this project!
Please fill out this short Google Form to share feedback or suggestions.

🌐 License
This project is licensed under the MIT License — you’re free to use, modify, and distribute it with attribution.

⭐ If you find this project helpful, please give it a star on GitHub!

Would you like me to tailor the **live demo link** and GitHub repo URL (with your actual username/repo name) in the README for you?  
If you send me your **GitHub repo link**, I’ll fill it in perfectly and format it for you.
