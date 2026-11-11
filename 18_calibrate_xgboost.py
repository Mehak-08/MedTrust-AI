# ============================================================
# MEDTRUST AI
# STEP 18 - LEAKAGE-SAFE XGBOOST CALIBRATION
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    log_loss
)

from xgboost import XGBClassifier


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

TRAIN_FILE = os.path.join(
    BASE_DIR,
    "train_patient_level.csv"
)

VALIDATION_FILE = os.path.join(
    BASE_DIR,
    "validation_patient_level.csv"
)

TEST_FILE = os.path.join(
    BASE_DIR,
    "test_patient_level.csv"
)

TARGET = "icu_mortality"


print("\n==========================================")
print("MEDTRUST AI - XGBOOST CALIBRATION")
print("==========================================\n")


# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

train = pd.read_csv(TRAIN_FILE)
validation = pd.read_csv(VALIDATION_FILE)
test = pd.read_csv(TEST_FILE)

print("Train:", train.shape)
print("Validation:", validation.shape)
print("Test:", test.shape)


# ------------------------------------------------------------
# 3. PREPARE FEATURES
# ------------------------------------------------------------

DROP_COLUMNS = [
    "subject_id",
    "hadm_id",
    "stay_id"
]

feature_columns = [
    col for col in train.columns
    if col not in DROP_COLUMNS + [TARGET]
]

X_train = train[feature_columns].copy()
X_validation = validation[feature_columns].copy()
X_test = test[feature_columns].copy()

y_train = train[TARGET].astype(int)
y_validation = validation[TARGET].astype(int)
y_test = test[TARGET].astype(int)


# ------------------------------------------------------------
# 4. CHECK TARGET CLASSES
# ------------------------------------------------------------

if y_train.nunique() < 2:
    raise ValueError(
        "Training data must contain both mortality classes."
    )

if y_validation.nunique() < 2:
    raise ValueError(
        "Validation data must contain both classes "
        "to calculate AUROC."
    )

if y_test.nunique() < 2:
    print(
        "WARNING: Test set contains only one class. "
        "Test AUROC may not be available."
    )


# ------------------------------------------------------------
# 5. BUILD XGBOOST MODEL
# ------------------------------------------------------------

base_model = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "xgb",
        XGBClassifier(
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
    )
])


# ------------------------------------------------------------
# 6. FIT MODEL ON TRAINING DATA
# ------------------------------------------------------------

print("\nTraining XGBoost...")

base_model.fit(
    X_train,
    y_train
)


# ------------------------------------------------------------
# 7. GET RAW PROBABILITIES
# ------------------------------------------------------------

raw_validation_prob = (
    base_model.predict_proba(X_validation)[:, 1]
)

raw_test_prob = (
    base_model.predict_proba(X_test)[:, 1]
)


# ------------------------------------------------------------
# 8. CALIBRATE USING TRAINING DATA ONLY
# ------------------------------------------------------------

print(
    "\nFitting sigmoid calibration "
    "using training data..."
)

# Calibration uses cross-validation within TRAIN only.
# Validation and test labels are not used to fit calibration.

calibrated_model = CalibratedClassifierCV(
    estimator=base_model,
    method="sigmoid",
    cv=3
)

calibrated_model.fit(
    X_train,
    y_train
)


# ------------------------------------------------------------
# 9. CALIBRATED PROBABILITIES
# ------------------------------------------------------------

calibrated_validation_prob = (
    calibrated_model.predict_proba(
        X_validation
    )[:, 1]
)

calibrated_test_prob = (
    calibrated_model.predict_proba(
        X_test
    )[:, 1]
)


# ------------------------------------------------------------
# 10. METRICS
# ------------------------------------------------------------

def calculate_metrics(
    y_true,
    probabilities
):

    result = {}

    result["Brier"] = brier_score_loss(
        y_true,
        probabilities
    )

    result["Log_Loss"] = log_loss(
        y_true,
        probabilities,
        labels=[0, 1]
    )

    if y_true.nunique() == 2:
        result["AUROC"] = roc_auc_score(
            y_true,
            probabilities
        )
    else:
        result["AUROC"] = np.nan

    return result


results = []

for split_name, y, raw_prob, calibrated_prob in [
    (
        "Validation",
        y_validation,
        raw_validation_prob,
        calibrated_validation_prob
    ),
    (
        "Test",
        y_test,
        raw_test_prob,
        calibrated_test_prob
    )
]:

    raw_metrics = calculate_metrics(
        y,
        raw_prob
    )

    calibrated_metrics = calculate_metrics(
        y,
        calibrated_prob
    )

    results.append({
        "Dataset": split_name,
        "Probability_Type": "Raw",
        **raw_metrics
    })

    results.append({
        "Dataset": split_name,
        "Probability_Type": "Calibrated",
        **calibrated_metrics
    })


results_df = pd.DataFrame(results)


print("\n==========================================")
print("CALIBRATION RESULTS")
print("==========================================")

print(
    results_df.to_string(index=False)
)


# ------------------------------------------------------------
# 11. SAVE METRICS
# ------------------------------------------------------------

metrics_path = os.path.join(
    BASE_DIR,
    "step18_calibration_metrics.csv"
)

results_df.to_csv(
    metrics_path,
    index=False
)


# ------------------------------------------------------------
# 12. SAVE VALIDATION PREDICTIONS
# ------------------------------------------------------------

validation_output = validation[
    [
        "subject_id",
        "hadm_id",
        "stay_id",
        TARGET
    ]
].copy()

validation_output["raw_probability"] = (
    raw_validation_prob
)

validation_output["calibrated_probability"] = (
    calibrated_validation_prob
)

validation_output["calibrated_confidence"] = (
    np.maximum(
        calibrated_validation_prob,
        1 - calibrated_validation_prob
    )
)

validation_output.to_csv(
    os.path.join(
        BASE_DIR,
        "step18_validation_predictions.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# 13. SAVE TEST PREDICTIONS
# ------------------------------------------------------------

test_output = test[
    [
        "subject_id",
        "hadm_id",
        "stay_id",
        TARGET
    ]
].copy()

test_output["raw_probability"] = raw_test_prob

test_output["calibrated_probability"] = (
    calibrated_test_prob
)

test_output["calibrated_confidence"] = (
    np.maximum(
        calibrated_test_prob,
        1 - calibrated_test_prob
    )
)

test_output.to_csv(
    os.path.join(
        BASE_DIR,
        "step18_test_predictions.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# 14. FINISH
# ------------------------------------------------------------

print("\n==========================================")
print("STEP 18 COMPLETE")
print("==========================================")

print("\nCreated files:")

print("step18_calibration_metrics.csv")
print("step18_validation_predictions.csv")
print("step18_test_predictions.csv")

print("\nIMPORTANT:")
print(
    "Calibration was fitted using training data only."
)

print(
    "Use validation data to choose routing thresholds."
)

print(
    "Keep test data untouched until final evaluation."
)

print(
    "These are still demo-dataset results, "
    "not clinical performance claims."
)

print("\n==========================================")