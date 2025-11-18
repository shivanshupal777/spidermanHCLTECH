# ---------------------------------------------------------------
# IMPORT LIBRARIES
# ---------------------------------------------------------------

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import ADASYN
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib

# ---------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------

data = pd.read_csv("customer_churn_encoded.csv")

# ---------------------------------------------------------------
# CLEAN + PREPROCESS
# ---------------------------------------------------------------

# 1. Remove customerID (not useful for prediction)
data = data.drop("customerID", axis=1)

# 2. Convert TotalCharges to numeric (some rows contain spaces or text)
# pd.to_numeric(errors='coerce') converts invalid values to NaN
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors='coerce')

# 3. Fill missing TotalCharges with the median
data["TotalCharges"] = data["TotalCharges"].fillna(data["TotalCharges"].median())

# 4. Encode Churn into numeric (Yes = 1, No = 0)
data["Churn_Yes"] = data["Churn"].map({"No": 0, "Yes": 1})

# Remove original Churn
data = data.drop("Churn", axis=1)

# 5. Convert all categorical columns to numeric using One-Hot Encoding
data = pd.get_dummies(data, drop_first=True)

# ---------------------------------------------------------------
# SPLIT FEATURES & TARGET
# ---------------------------------------------------------------

X = data.drop("Churn_Yes", axis=1)
y = data["Churn_Yes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

# ---------------------------------------------------------------
# FEATURE SCALING
# ---------------------------------------------------------------

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)  # Fit on train
X_test = scaler.transform(X_test)        # Only transform test

# ---------------------------------------------------------------
# FIX IMBALANCE WITH ADASYN
# ---------------------------------------------------------------

adasyn = ADASYN(sampling_strategy=1.0, random_state=0)
X_train, y_train = adasyn.fit_resample(X_train, y_train)

# ---------------------------------------------------------------
# XGBOOST MODEL
# ---------------------------------------------------------------

model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    learning_rate=0.1,
    max_depth=6,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=0
)

model.fit(X_train, y_train)

# ---------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_prob)

print("\n================= CLASSIFICATION REPORT =================")
print(classification_report(y_test, y_pred))

print("\n================= ROC-AUC SCORE =================")
print(roc_auc)

print("\n================= CONFUSION MATRIX =================")
print(confusion_matrix(y_test, y_pred))

# ---------------------------------------------------------------
# SAVE MODEL + SCALER
# ---------------------------------------------------------------

joblib.dump(model, "xgboost_churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel and Scaler Saved Successfully!")
