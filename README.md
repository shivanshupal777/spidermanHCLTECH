# TEAM SPIDERMAN – HCLTECH  
## Customer Churn Prediction

Customer Churn refers to identifying customers who are likely to stop using a company’s service in the near future.

Dataset used: **Customer Churn Dataset(Kaggle)**  
Link: [(https://www.kaggle.com/datasets/blastchar/telco-customer-churn)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

The project includes dataset exploration, preprocessing, feature engineering, model development, handling class imbalance, evaluation, and deployment design.
---

## 📝 Problem Statement 
The goal is to build a Machine Learning Prediction model to predict the Customer Churn, technique to handle imbalance Dataset, Evaluation Metrices to be  utilized & Confusion Matrix/ ROC Curve Explaination.

---
---

## 📌 Introduction  
Customer churn directly affects business revenue, customer retention, and long-term growth.  
This project builds a **supervised machine learning classification model** to predict whether a customer is likely to churn based on account history and service usage patterns.

---

## 📂 Dataset Description  

The dataset contains:

- Customer demographic information  
- Account details  
- Service usage behavior  
- Churn outcome (Yes/No)

**File Included:**  
- `Telco-Customer-Churn.csv`

### **2.1 Dataset Features**

| Column | Description |
|--------|-------------|
| CustomerID | Unique customer identifier |
| Gender | Male/Female |
| Age | Customer age |
| SeniorCitizen | Indicates senior citizen status |
| Partner | Yes/No |
| Dependents | Yes/No |
| Tenure | Number of months customer has stayed |
| PhoneService | Yes/No |
| MultipleLines | Type of phone line service |
| InternetService | DSL/Fiber Optic/No |
| OnlineSecurity | Subscribed or not |
| OnlineBackup | Yes/No |
| DeviceProtection | Yes/No |
| TechSupport | Yes/No |
| StreamingTV | Yes/No |
| StreamingMovies | Yes/No |
| Contract | Month-to-month / One year / Two year |
| PaperlessBilling | Yes/No |
| PaymentMethod | Type of payment method |
| MonthlyCharges | Monthly bill |
| TotalCharges | Lifetime charges |
| Churn | Target variable (Yes/No) |

### **2.2 Dataset Characteristics**

- Mixed features: numerical, categorical, binary  
- Contains missing/blank values (especially in `TotalCharges`)  
- Highly imbalanced (majority “No churn”)  
- Strong correlation with:
  - Contract type  
  - Tenure  
  - Monthly charges  

---

## 🚀 Project/Design Workflow  

### **Step 1: Data Loading**
- Import dataset  
- Perform initial checks:
  - Dataset shape  
  - Column types  
  - Duplicates  
  - Missing values  

---

### **Step 2: Data Cleaning**
- Convert blank numerical values (e.g., `TotalCharges`) to numeric  
- Handle missing values  
- Convert Yes/No columns → binary (1/0)  
- Strip whitespace in string fields  

---

### **Step 3: Exploratory Data Analysis (EDA)**
Analyze:

- Churn distribution  
- Tenure vs Churn  
- MonthlyCharges vs Churn  
- Contract type influence  
- Correlation heatmaps  
- Customer behavior patterns  

**Key observations:**
- Month-to-month customers churn the most  
- Higher monthly charges = higher churn  
- Long-tenure customers are more stable  
- Lack of technical support/security increases churn probability  

---

### **Step 4: Encoding and Preprocessing**
- One-hot encode multi-class categorical features  
- Label encode binary features  
- Standardize / Normalize numerical variables  
- Split data into train & test sets  

---

### **Step 5: Handling Imbalance**
Techniques applied:

- **ADASYN "Adaptive Synthetic Sampling"**  
- **Random Oversampling**  
- **Class Weight adjustments**

Evaluation focuses on:

- **Recall**
- **Precision**

instead of only accuracy, due to imbalance.

---

### **Step 6: Feature Engineering**
Examples:

- Tenure grouping (0–12, 12–24, etc.)  
- Charges ratio = `TotalCharges / Tenure`  
- Binary mapping for service features  
- Interaction features (Contract × Payment method)  

---

### **Step 7: Model Development**
Models trained:

- XGBOOST 
- Random Forest Classifier  

💡 **Tree-based models (Random Forest, XGBoost) usually perform best.**

---

### **Step 8: Model Evaluation**
Metrics:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC-AUC  
- Confusion Matrix  

📌 **High recall for churn class is critical**, because missing a churner (False Negative) hurts revenue.

---

### **Step 9: Model Saving**
Final model exported as `model.pkl` using:

- `pickle`  
- `joblib`  

## 📊 Confusion Matrix Interpretation

- **TP (True Positive):** Correctly predicted churn  
- **TN (True Negative):** Correctly predicted non-churn  
- **FP (False Positive):** Incorrectly predicted churn  
- **FN (False Negative):** Churn predicted as non-churn  

👉 **Reducing FN is most important** to prevent customer loss.

---

## 📈 ROC-AUC Analysis  
- ROC curve measures the classifier’s ability to separate churn vs non-churn.  
- AUC > 0.80 is considered strong.  

---
## Result (On telco-customer-churn Dataset) Model:
================= CLASSIFICATION REPORT =================
                precision   recall   f1-score   support

           0       0.85      0.86      0.85      1035
           1       0.60      0.57      0.58       374

    accuracy                           0.78      1409
   macro avg       0.72      0.72      0.72      1409
weighted avg       0.78      0.78      0.78      1409


================= ROC-AUC SCORE =================
0.8274044795783926

================= CONFUSION MATRIX =================
[[893 142]
[161 213]]

Model and Scaler Saved Successfully!

## 🚀 Deployment (Streamlit Web Application)

This project includes a simple and interactive Streamlit-based web application that allows real-time customer churn prediction using the trained XGBoost model. The deployment makes the model accessible to non-technical users through an easy-to-use interface.

📌 **Overview**

The goal of this deployment is to provide a user-friendly web interface where users can input customer details and instantly receive:

Churn Prediction: Yes/No
Churn Probability Score
Clear classification with colored messages
The deployment uses:
xgboost_churn_model.pkl → Saved trained model
scaler.pkl → Saved MinMaxScaler used during training
Both are necessary to ensure consistent and accurate predictions.

🧠 **How the Deployment Works**
 1. Load Saved Model and Scaler
The app loads the trained XGBoost model and the scaler:

model = joblib.load("xgboost_churn_model.pkl")
scaler = joblib.load("scaler.pkl")

The model predicts churn.
The scaler ensures the new input data matches the scale of the training data.

 2. Build the User Interface (Streamlit)

Streamlit automatically generates a clean UI.
Users input customer details such as:

Tenure
Monthly Charges
Total Charges
Contract type
Internet service

These inputs are collected using Streamlit widgets like number_input and selectbox.

 3. Convert Inputs into Model Format

User inputs are converted to the exact feature format expected by the ML model, including one-hot encoded fields:

df = pd.DataFrame([data])

 4. Scale Numerical Inputs

Before feeding to the model, inputs are scaled using the previously saved scaler:

df_scaled = scaler.transform(df)

 5. Predict Churn

The model predicts:

Class (0 = No Churn, 1 = Churn)
Probability (0.0 – 1.0)
pred = model.predict(df_scaled)[0]
prob = model.predict_proba(df_scaled)[0][1]

 6. Display Results

The app clearly displays the final output:

st.error("Customer is likely to CHURN")
st.success("Customer will NOT churn")


