# TEAM SPIDERMAN – HCLTECH  
## Customer Churn Prediction

Customer Churn refers to identifying customers who are likely to stop using a company’s service in the near future.

Dataset used: **Customer Churn Dataset by Muhammad Shahid Azeem (Kaggle)**  
Link: https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset

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
- `Customer-Churn-Records.csv`

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

## 🚀 Project Workflow  

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

- **SMOTE (Synthetic Minority Oversampling Technique)**  
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

---

### **Step 10: Deployment Design**

**Backend APIs**
- Flask  
- FastAPI  

**Dashboard / UI**
- Streamlit  

**Containerization**
- Docker  

**Cloud Deployment**
- AWS EC2  
- Azure App Service  
- Google Cloud Platform (GCP)  

---

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

## 🔮 Future Enhancements  

- Automated hyperparameter tuning (GridSearchCV, Optuna)  
- Model explainability using SHAP / LIME  
- Customer segmentation via clustering  
- CRM integration  
- Real-time monitoring of churn probabilities  

---
