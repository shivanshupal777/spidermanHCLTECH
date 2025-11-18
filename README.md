# TEAM SPIDERMAN HCLTECH

<h1>Customer Churn Prediction</h1>
Customer Churn is the process of identifying which customers are likely to stop using a company’s product or service in the near future.

Dataset for Customer Churn Prediction taken from Kaggle dataset:
Customer Churn Dataset by Muhammad Shahid Azeem
Link: https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset

The project includes dataset exploration, preprocessing, feature engineering, model development, class imbalance handling, evaluation, and deployment design.

1. Introduction

Customer churn significantly impacts business revenue, customer retention strategy, and long-term sustainability. Predicting churn allows organizations to take proactive actions to reduce customer loss.

This project builds a supervised machine learning classification model to determine whether a customer is likely to churn using historical account information and service usage patterns.

2. Dataset Description

The dataset contains customer demographic information, account details, service usage patterns, and churn outcome.


Files Included: Customer-Churn-Records.csv - Kaggle

2.1 Features (as per dataset)

Typical columns in the dataset include:

| Column           | Description                              |
| ---------------- | ---------------------------------------- |
| CustomerID       | Unique identifier                        |
| Gender           | Male/Female                              |
| Age              | Customer age                             |
| SeniorCitizen    | Indicates senior citizen status          |
| Partner          | Yes/No                                   |
| Dependents       | Yes/No                                   |
| Tenure           | Number of months the customer has stayed |
| PhoneService     | Yes/No                                   |
| MultipleLines    | Type of phone line service               |
| InternetService  | DSL/Fiber Optic/No                       |
| OnlineSecurity   | Customer subscribed or not               |
| OnlineBackup     | Yes/No                                   |
| DeviceProtection | Yes/No                                   |
| TechSupport      | Yes/No                                   |
| StreamingTV      | Yes/No                                   |
| StreamingMovies  | Yes/No                                   |
| Contract         | Month-to-month, One year, Two year       |
| PaperlessBilling | Yes/No                                   |
| PaymentMethod    | Type of payment method                   |
| MonthlyCharges   | Monthly bill                             |
| TotalCharges     | Lifetime charges                         |
| Churn            | Target variable (Yes/No)                 |

2.2 Dataset Characteristics

Mixed feature types (numerical, categorical, binary).
Contains missing or blank values (particularly in TotalCharges).
Highly imbalanced target variable (more "No" churn cases).
Strong correlation between contract type, tenure, monthly charges, and churn.

3. Project Workflow

The project follows a complete machine learning pipeline.

  Step 1: Data Loading
   ‣ Import the dataset and perform initial checks such as shape, column types, duplicates, and missing values.

  Step 2: Data Cleaning
  ‣ Convert blank entries in numerical columns (e.g., TotalCharges) to appropriate numeric values.
  ‣ Handle missing values.
  ‣ Convert Yes/No categorical variables to binary form.
  ‣ Strip unnecessary whitespace.

  Step 3: Exploratory Data Analysis (EDA)
  ‣ Churn distribution
  ‣ Tenure vs churn
  ‣ MonthlyCharges vs churn
  ‣ Contract type impact on churn
  ‣ Correlation heatmaps
  ‣ Customer behavior patterns
  ‣ Key observations typically include:
  ‣ Month-to-month contract customers churn the most
  ‣ Customers with high monthly charges have higher churn
  ‣ Long-tenure customers are more stable
  ‣ Lack of technical support and security services increases churn probability

  Step 4: Encoding and Preprocessing
  ‣ One-hot encode multi-class categorical features
  ‣ Label encode binary features
  ‣ Standardize or normalize numerical columns
  ‣ Split dataset into training and testing sets
 
Step 5: Handling Imbalance

Since churn data is imbalanced:

Techniques used:

SMOTE (Synthetic Minority Oversampling Technique)

Random oversampling

Class weight adjustments

Evaluation based on recall and precision instead of accuracy

Step 6: Feature Engineering

Examples:

Tenure grouping (0–12, 12–24, etc.)

Charges ratio: TotalCharges / Tenure

Binary mapping for service features

Interaction variables for contract + payment method

Step 7: Model Development

Multiple models can be trained and compared, including:

Logistic Regression
Random Forest Classifier
SVM

Given the nature of churn data, tree-based models (Random Forest, XGBoost) generally perform better.

Step 8: Model Evaluation

Metrics used:

Accuracy

Precision

Recall

F1-score

ROC-AUC score

Confusion Matrix

Reason: In churn prediction, high recall for the churn class is essential because missing a churn customer (false negative) is more costly for the business.

Step 9: Model Saving

The final model is exported as model.pkl using pickle or joblib for deployment.

Step 10: Deployment Design

Possible deployment approaches:

Flask or FastAPI backend for real-time predictions

Streamlit application for interactive dashboards

Containerization using Docker

Cloud deployment (AWS EC2, Azure App Service, or GCP)

4. Confusion Matrix Interpretation

A confusion matrix is used to measure classification performance.

True Positive (TP): Correctly predicted churn

True Negative (TN): Correctly predicted non-churn

False Positive (FP): Incorrectly predicted churn

False Negative (FN): Churn misclassified as non-churn

Reducing FN is the highest priority because missing a churner directly impacts revenue.

5. ROC-AUC Analysis

The ROC curve evaluates the classifier's ability to distinguish churn vs non-churn across thresholds.
A high AUC score indicates strong discriminatory power.
Customer churn models typically aim for AUC > 0.80.


6. Future Enhancements

Automated hyperparameter tuning (GridSearchCV, Optuna)

Model explainability using SHAP and LIME

Customer segmentation using clustering

Integration with CRM systems

Real-time monitoring of churn probabilities

