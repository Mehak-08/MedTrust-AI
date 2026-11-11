import os
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# MEDTRUST AI
# Step 8: Compare Baseline Models
# ============================================================

print("\n==========================================")
print("MEDTRUST AI - MODEL COMPARISON")
print("==========================================\n")


PROJECT_PATH = os.path.dirname(__file__)


# ============================================================
# 1. LOAD TEST PREDICTIONS
# ============================================================

logistic_file = os.path.join(
    PROJECT_PATH,
    "logistic_regression_test_predictions.csv"
)

rf_file = os.path.join(
    PROJECT_PATH,
    "random_forest_test_predictions.csv"
)

xgb_file = os.path.join(
    PROJECT_PATH,
    "xgboost_test_predictions.csv"
)


logistic = pd.read_csv(logistic_file)
random_forest = pd.read_csv(rf_file)
xgboost = pd.read_csv(xgb_file)


# ============================================================
# 2. FUNCTION TO CALCULATE METRICS
# ============================================================

def calculate_metrics(data):

    y_true = data["icu_mortality"]

    y_pred = data["predicted_class"]

    y_prob = data["predicted_probability"]

    return {
        "Accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "F1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "AUROC": roc_auc_score(
            y_true,
            y_prob
        )
    }


# ============================================================
# 3. CALCULATE RESULTS
# ============================================================

logistic_metrics = calculate_metrics(
    logistic
)

rf_metrics = calculate_metrics(
    random_forest
)

xgb_metrics = calculate_metrics(
    xgboost
)


# ============================================================
# 4. CREATE COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame(
    [
        logistic_metrics,
        rf_metrics,
        xgb_metrics
    ],
    index=[
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ]
)


# ============================================================
# 5. DISPLAY RESULTS
# ============================================================

print("==========================================")
print("TEST SET MODEL COMPARISON")
print("==========================================\n")


print(
    comparison.round(4).to_string()
)


# ============================================================
# 6. SAVE RESULTS
# ============================================================

comparison_file = os.path.join(
    PROJECT_PATH,
    "model_comparison.csv"
)


comparison.to_csv(
    comparison_file
)


print("\n==========================================")
print("COMPARISON SAVED")
print("==========================================")

print(
    comparison_file
)


# ============================================================
# 7. IMPORTANT NOTE
# ============================================================

print("\n==========================================")
print("IMPORTANT")
print("==========================================")

print(
    "The current demo dataset contains only"
)

print(
    "12 ICU mortality cases in total."
)

print(
    "\nTherefore, these metrics are being used"
)

print(
    "to validate the pipeline, not as final"
)

print(
    "clinical performance claims."
)


print("\n==========================================")
print("STEP 8 COMPLETE")
print("==========================================")

print(
    "\nNext step: Calibration of XGBoost probabilities."
)