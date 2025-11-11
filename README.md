
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



Key Features
Real-Time Prediction: Users enter transaction attributes (e.g., amount, time, location-based features) via a simple form, and the app outputs a fraud probability score along with a binary classification (fraudulent or not).
Model Insights: Displays feature importance and model performance metrics (e.g., accuracy, precision, recall) to explain predictions.
Data Visualization: Includes charts and graphs for exploratory data analysis, such as transaction distributions and fraud patterns over time.
User-Friendly Interface: Clean, intuitive design with sliders, dropdowns, and text inputs for easy interaction, no coding required.



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





## 🚀 Working video 
🔗 **[Watch the video here  Here](https://drive.google.com/file/d/1DFoq8OFrZuCiIomSBjEDNRy17rNT1Tub/view?usp=drivesdk)**

  
fraud_project/
├── app/
│   ├── streamlit_app.py          # Main Streamlit application
│   ├── best_fraud_model.pkl      # Trained model
│   └── scaler.pkl               # Feature scaler
├── models/
│   ├── best_fraud_model.pkl     # Backup model files
│   └── scaler.pkl              # Backup scaler files
├── notebooks/
│   └── fraud_detection.ipynb    # Jupyter notebook with EDA & training
├── requirements.txt             # Python dependencies
└── README.md                   # Project documentation






    
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


🌐 License
This project is licensed under the MIT License — you’re free to use, modify, and distribute it with attribution.

⭐ If you find this project helpful, please give it a star on GitHub!

