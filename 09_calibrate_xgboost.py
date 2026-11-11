import os
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss
)

from xgboost import XGBClassifier


# ============================================================
# MEDTRUST AI
# Step 9: XGBoost Probability Calibration
# ============================================================

print("\n==========================================")
print("MEDTRUST AI - XGBOOST CALIBRATION")
print("==========================================\n")


# ============================================================
# 1. LOAD DATA
# ============================================================

PROJECT_PATH = os.path.dirname(__file__)

train_df = pd.read_csv(
    os.path.join(PROJECT_PATH, "train.csv")
)

val_df = pd.read_csv(
    os.path.join(PROJECT_PATH, "validation.csv")
)

test_df = pd.read_csv(
    os.path.join(PROJECT_PATH, "test.csv")
)


print("Training shape:", train_df.shape)
print("Validation shape:", val_df.shape)
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


# ============================================================
# 3. CREATE FEATURES
# ============================================================

X_train = train_df.drop(
    columns=[TARGET] + ID_COLUMNS
)

y_train = train_df[TARGET]


X_val = val_df.drop(
    columns=[TARGET] + ID_COLUMNS
)

y_val = val_df[TARGET]


X_test = test_df.drop(
    columns=[TARGET] + ID_COLUMNS
)

y_test = test_df[TARGET]


# ============================================================
# 4. CREATE BASE XGBOOST MODEL
# ============================================================

base_xgb = XGBClassifier(
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


# ============================================================
# 5. CREATE CALIBRATION PIPELINE
# ============================================================
#
# sigmoid = Platt scaling
#
# The calibration model learns how to convert raw model
# probabilities into better-calibrated probabilities.
#
# We use cross-validation inside the training data so the
# calibration process does not directly learn from the test set.
#


calibrated_model = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "calibrator",
            CalibratedClassifierCV(
                estimator=base_xgb,
                method="sigmoid",
                cv=5
            )
        )
    ]
)


# ============================================================
# 6. TRAIN CALIBRATED MODEL
# ============================================================

print("\n==========================================")
print("TRAINING CALIBRATED XGBOOST")
print("==========================================")

calibrated_model.fit(
    X_train,
    y_train
)

print(
    "Calibration completed."
)


# ============================================================
# 7. GET PROBABILITIES
# ============================================================

val_probabilities = calibrated_model.predict_proba(
    X_val
)[:, 1]


test_probabilities = calibrated_model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 8. CALCULATE BRIER SCORE
# ============================================================
#
# Lower Brier score is better.
#
# It measures how close predicted probabilities are
# to the actual outcomes.
#


val_brier = brier_score_loss(
    y_val,
    val_probabilities
)

test_brier = brier_score_loss(
    y_test,
    test_probabilities
)


# ============================================================
# 9. CALCULATE AUROC
# ============================================================

val_auc = roc_auc_score(
    y_val,
    val_probabilities
)

test_auc = roc_auc_score(
    y_test,
    test_probabilities
)


# ============================================================
# 10. DISPLAY RESULTS
# ============================================================

print("\n==========================================")
print("CALIBRATED MODEL RESULTS")
print("==========================================")


print("\nValidation:")

print(
    "AUROC:",
    round(val_auc, 4)
)

print(
    "Brier Score:",
    round(val_brier, 4)
)


print("\nTest:")

print(
    "AUROC:",
    round(test_auc, 4)
)

print(
    "Brier Score:",
    round(test_brier, 4)
)


# ============================================================
# 11. SHOW PROBABILITIES
# ============================================================

results = test_df[
    [
        "subject_id",
        "hadm_id",
        "stay_id",
        TARGET
    ]
].copy()


results[
    "calibrated_probability"
] = test_probabilities


results[
    "calibrated_confidence"
] = np.maximum(
    test_probabilities,
    1 - test_probabilities
)


print("\n==========================================")
print("CALIBRATED TEST PROBABILITIES")
print("==========================================")

print(
    results.to_string(
        index=False
    )
)


# ============================================================
# 12. SAVE RESULTS
# ============================================================

results_file = os.path.join(
    PROJECT_PATH,
    "calibrated_xgboost_predictions.csv"
)


results.to_csv(
    results_file,
    index=False
)


# ============================================================
# 13. SAVE CALIBRATION METRICS
# ============================================================

metrics = pd.DataFrame(
    {
        "dataset": [
            "validation",
            "test"
        ],

        "AUROC": [
            val_auc,
            test_auc
        ],

        "Brier_score": [
            val_brier,
            test_brier
        ]
    }
)


metrics_file = os.path.join(
    PROJECT_PATH,
    "calibration_metrics.csv"
)


metrics.to_csv(
    metrics_file,
    index=False
)


# ============================================================
# 14. COMPLETE
# ============================================================

print("\n==========================================")
print("STEP 9 COMPLETE")
print("==========================================")

print(
    "\nPredictions saved to:"
)

print(
    results_file
)

print(
    "\nCalibration metrics saved to:"
)

print(
    metrics_file
)

print(
    "\nNext step: SHAP explanations."
)