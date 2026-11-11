import os
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import shap


# ============================================================
# MEDTRUST AI
# Step 10: SHAP Explanations
# ============================================================

print("\n==========================================")
print("MEDTRUST AI - SHAP EXPLANATIONS")
print("==========================================\n")


# ============================================================
# 1. LOAD DATA
# ============================================================

PROJECT_PATH = os.path.dirname(__file__)

train_df = pd.read_csv(
    os.path.join(PROJECT_PATH, "train.csv")
)

test_df = pd.read_csv(
    os.path.join(PROJECT_PATH, "test.csv")
)


print("Training shape:", train_df.shape)
print("Test shape:", test_df.shape)


# ============================================================
# 2. DEFINE TARGET AND IDS
# ============================================================

TARGET = "icu_mortality"

ID_COLUMNS = [
    "subject_id",
    "hadm_id",
    "stay_id"
]


X_train = train_df.drop(
    columns=[TARGET] + ID_COLUMNS
)

y_train = train_df[TARGET]


X_test = test_df.drop(
    columns=[TARGET] + ID_COLUMNS
)

y_test = test_df[TARGET]


# ============================================================
# 3. HANDLE MISSING VALUES
# ============================================================

print("\n==========================================")
print("PREPARING FEATURES")
print("==========================================")


imputer = SimpleImputer(
    strategy="median"
)


X_train_imputed = imputer.fit_transform(
    X_train
)

X_test_imputed = imputer.transform(
    X_test
)


X_train_imputed = pd.DataFrame(
    X_train_imputed,
    columns=X_train.columns,
    index=X_train.index
)


X_test_imputed = pd.DataFrame(
    X_test_imputed,
    columns=X_test.columns,
    index=X_test.index
)


# ============================================================
# 4. TRAIN XGBOOST
# ============================================================

print("\n==========================================")
print("TRAINING XGBOOST")
print("==========================================")


xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)


xgb_model.fit(
    X_train_imputed,
    y_train
)


print(
    "XGBoost training completed."
)


# ============================================================
# 5. CREATE SHAP EXPLAINER
# ============================================================

print("\n==========================================")
print("CREATING SHAP EXPLAINER")
print("==========================================")


explainer = shap.TreeExplainer(
    xgb_model
)


# ============================================================
# 6. CALCULATE SHAP VALUES
# ============================================================

print("\nCalculating SHAP values...")

shap_values = explainer.shap_values(
    X_test_imputed
)


# ============================================================
# 7. HANDLE SHAP OUTPUT
# ============================================================

# Some SHAP versions return an Explanation object.
# Convert it into a NumPy array when necessary.

if hasattr(
    shap_values,
    "values"
):

    shap_values = shap_values.values


shap_values = np.asarray(
    shap_values
)


print(
    "SHAP values shape:",
    shap_values.shape
)


# ============================================================
# 8. GLOBAL FEATURE IMPORTANCE
# ============================================================

print("\n==========================================")
print("GLOBAL SHAP FEATURE IMPORTANCE")
print("==========================================")


mean_abs_shap = np.mean(
    np.abs(shap_values),
    axis=0
)


importance_df = pd.DataFrame(
    {
        "feature": X_test.columns,
        "mean_absolute_shap": mean_abs_shap
    }
)


importance_df = importance_df.sort_values(
    by="mean_absolute_shap",
    ascending=False
)


print(
    importance_df.head(15).to_string(
        index=False
    )
)


# ============================================================
# 9. SAVE GLOBAL IMPORTANCE
# ============================================================

importance_file = os.path.join(
    PROJECT_PATH,
    "shap_global_importance.csv"
)


importance_df.to_csv(
    importance_file,
    index=False
)


# ============================================================
# 10. CREATE PATIENT-LEVEL SHAP DATA
# ============================================================

print("\n==========================================")
print("CREATING PATIENT-LEVEL EXPLANATIONS")
print("==========================================")


shap_df = pd.DataFrame(
    shap_values,
    columns=[
        "SHAP_" + str(column)
        for column in X_test.columns
    ]
)


# Add identifiers and actual outcome

shap_df.insert(
    0,
    "subject_id",
    test_df["subject_id"].values
)


shap_df.insert(
    1,
    "hadm_id",
    test_df["hadm_id"].values
)


shap_df.insert(
    2,
    "stay_id",
    test_df["stay_id"].values
)


shap_df.insert(
    3,
    "icu_mortality",
    test_df["icu_mortality"].values
)


# ============================================================
# 11. ADD MODEL PROBABILITY
# ============================================================

test_probabilities = xgb_model.predict_proba(
    X_test_imputed
)[:, 1]


shap_df.insert(
    4,
    "xgboost_probability",
    test_probabilities
)


# ============================================================
# 12. SAVE PATIENT-LEVEL SHAP VALUES
# ============================================================

patient_shap_file = os.path.join(
    PROJECT_PATH,
    "patient_shap_values.csv"
)


shap_df.to_csv(
    patient_shap_file,
    index=False
)


# ============================================================
# 13. SHOW ONE EXAMPLE
# ============================================================

print("\n==========================================")
print("EXAMPLE PATIENT EXPLANATION")
print("==========================================")


patient_index = 0


patient_values = shap_values[
    patient_index
]


patient_explanation = pd.DataFrame(
    {
        "feature": X_test.columns,
        "shap_value": patient_values
    }
)


patient_explanation[
    "absolute_shap"
] = np.abs(
    patient_explanation["shap_value"]
)


patient_explanation = patient_explanation.sort_values(
    by="absolute_shap",
    ascending=False
)


print(
    patient_explanation.head(10).to_string(
        index=False
    )
)


# ============================================================
# 14. COMPLETE
# ============================================================

print("\n==========================================")
print("STEP 10 COMPLETE")
print("==========================================")

print(
    "\nGlobal SHAP importance saved to:"
)

print(
    importance_file
)

print(
    "\nPatient-level SHAP values saved to:"
)

print(
    patient_shap_file
)

print(
    "\nNext step:"
)

print(
    "Build explanation reliability measurements."
)