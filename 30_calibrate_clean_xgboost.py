# ============================================================
# MEDTRUST AI
# STEP 30 - LEAKAGE-SAFE XGBOOST CALIBRATION
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)

from xgboost import XGBClassifier


warnings.filterwarnings("ignore")


print("\n==========================================")
print("MEDTRUST AI - XGBOOST CALIBRATION")
print("==========================================\n")


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_DIR = r"C:\Users\MEHAK\OneDrive\Desktop\MIMIC-Project"

TRAIN_FILE = os.path.join(
    PROJECT_DIR,
    "step28_train_raw.csv"
)

VALIDATION_FILE = os.path.join(
    PROJECT_DIR,
    "step28_validation_raw.csv"
)

TEST_FILE = os.path.join(
    PROJECT_DIR,
    "step28_test_raw.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "step30_calibration"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. LOAD DATA
# ============================================================

train = pd.read_csv(TRAIN_FILE)
validation = pd.read_csv(VALIDATION_FILE)
test = pd.read_csv(TEST_FILE)


TARGET = "icu_mortality"

ID_COLUMNS = [
    "subject_id",
    "hadm_id",
    "stay_id"
]


feature_columns = [
    c
    for c in train.columns
    if c not in ID_COLUMNS
    and c != TARGET
]


X_train = train[
    feature_columns
].copy()

y_train = train[
    TARGET
].astype(int)


X_validation = validation[
    feature_columns
].copy()

y_validation = validation[
    TARGET
].astype(int)


X_test = test[
    feature_columns
].copy()

y_test = test[
    TARGET
].astype(int)


# ============================================================
# 3. CONVERT FEATURES TO NUMERIC
# ============================================================

for column in feature_columns:

    X_train[column] = pd.to_numeric(
        X_train[column],
        errors="coerce"
    )

    X_validation[column] = pd.to_numeric(
        X_validation[column],
        errors="coerce"
    )

    X_test[column] = pd.to_numeric(
        X_test[column],
        errors="coerce"
    )


# ============================================================
# 4. BASE XGBOOST
# ============================================================

base_xgb = XGBClassifier(

    n_estimators=300,

    max_depth=3,

    learning_rate=0.03,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="binary:logistic",

    eval_metric="logloss",

    random_state=42,

    n_jobs=-1
)


# ============================================================
# 5. LEAKAGE-SAFE PIPELINE
# ============================================================

base_pipeline = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "model",
            base_xgb
        )
    ]
)


# ============================================================
# 6. CALIBRATE
# ============================================================

print(
    "Training calibrated XGBoost..."
)

calibrated_model = CalibratedClassifierCV(
    estimator=base_pipeline,
    method="sigmoid",
    cv=3
)


calibrated_model.fit(
    X_train,
    y_train
)


# ============================================================
# 7. PREDICTIONS
# ============================================================

validation_probability = (
    calibrated_model
    .predict_proba(
        X_validation
    )[:, 1]
)


test_probability = (
    calibrated_model
    .predict_proba(
        X_test
    )[:, 1]
)


# ============================================================
# 8. METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probability
):

    try:

        auroc = roc_auc_score(
            y_true,
            probability
        )

    except ValueError:

        auroc = np.nan


    try:

        auprc = average_precision_score(
            y_true,
            probability
        )

    except ValueError:

        auprc = np.nan


    brier = brier_score_loss(
        y_true,
        probability
    )


    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "Brier": brier
    }


validation_metrics = calculate_metrics(
    y_validation,
    validation_probability
)


test_metrics = calculate_metrics(
    y_test,
    test_probability
)


# ============================================================
# 9. SAVE PREDICTIONS
# ============================================================

validation_predictions = validation[
    [
        "subject_id",
        "hadm_id",
        "stay_id",
        TARGET
    ]
].copy()


validation_predictions[
    "calibrated_probability"
] = validation_probability


validation_predictions[
    "predictive_confidence"
] = np.maximum(
    validation_probability,
    1 - validation_probability
)


test_predictions = test[
    [
        "subject_id",
        "hadm_id",
        "stay_id",
        TARGET
    ]
].copy()


test_predictions[
    "calibrated_probability"
] = test_probability


test_predictions[
    "predictive_confidence"
] = np.maximum(
    test_probability,
    1 - test_probability
)


validation_predictions.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "step30_validation_predictions.csv"
    ),
    index=False
)


test_predictions.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "step30_test_predictions.csv"
    ),
    index=False
)


# ============================================================
# 10. SAVE METRICS
# ============================================================

metrics = pd.DataFrame([

    {
        "Split": "Validation",
        **validation_metrics
    },

    {
        "Split": "Test",
        **test_metrics
    }

])


metrics.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "step30_calibration_metrics.csv"
    ),
    index=False
)


# ============================================================
# 11. PRINT RESULTS
# ============================================================

print("\n==========================================")
print("CALIBRATION RESULTS")
print("==========================================")

print("\nValidation:")

print(
    "AUROC:",
    validation_metrics["AUROC"]
)

print(
    "AUPRC:",
    validation_metrics["AUPRC"]
)

print(
    "Brier:",
    validation_metrics["Brier"]
)


print("\nTest:")

print(
    "AUROC:",
    test_metrics["AUROC"]
)

print(
    "AUPRC:",
    test_metrics["AUPRC"]
)

print(
    "Brier:",
    test_metrics["Brier"]
)


print("\nProbability summary:")

print(
    "\nValidation probability range:",
    validation_probability.min(),
    "to",
    validation_probability.max()
)

print(
    "Test probability range:",
    test_probability.min(),
    "to",
    test_probability.max()
)


print("\n==========================================")
print("STEP 30 COMPLETE")
print("==========================================")

print(
    "\nSaved:"
)

print(
    os.path.join(
        OUTPUT_DIR,
        "step30_validation_predictions.csv"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "step30_test_predictions.csv"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "step30_calibration_metrics.csv"
    )
)

print("\n==========================================")