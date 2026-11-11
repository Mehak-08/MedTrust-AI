# ============================================================
# MEDTRUST AI
# STEP 29 - LEAKAGE-SAFE MODEL TRAINING
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from xgboost import XGBClassifier


warnings.filterwarnings("ignore")


print("\n==========================================")
print("MEDTRUST AI - CLEAN MODEL TRAINING")
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
    "step29_clean_models"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("Loading datasets...")

train = pd.read_csv(TRAIN_FILE)
validation = pd.read_csv(VALIDATION_FILE)
test = pd.read_csv(TEST_FILE)


print(
    "Train:",
    train.shape
)

print(
    "Validation:",
    validation.shape
)

print(
    "Test:",
    test.shape
)


# ============================================================
# 3. TARGET + ID COLUMNS
# ============================================================

TARGET = "icu_mortality"

ID_COLUMNS = [
    "subject_id",
    "hadm_id",
    "stay_id"
]


# Only actual model features

feature_columns = [
    c
    for c in train.columns
    if c not in ID_COLUMNS
    and c != TARGET
]


print(
    "\nNumber of features:",
    len(feature_columns)
)


# ============================================================
# 4. CREATE X / y
# ============================================================

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
# 5. CHECK TARGET DISTRIBUTION
# ============================================================

print("\nTarget distribution:")

print(
    "Train:",
    y_train.value_counts().to_dict()
)

print(
    "Validation:",
    y_validation.value_counts().to_dict()
)

print(
    "Test:",
    y_test.value_counts().to_dict()
)


# ============================================================
# 6. HANDLE BOOLEAN COLUMNS
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
# 7. MODELS
# ============================================================

models = {}


# ------------------------------------------------------------
# Logistic Regression
# ------------------------------------------------------------

models["Logistic Regression"] = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=42
            )
        )
    ]
)


# ------------------------------------------------------------
# Random Forest
# ------------------------------------------------------------

models["Random Forest"] = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# ------------------------------------------------------------
# XGBoost
# ------------------------------------------------------------

models["XGBoost"] = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "model",
            XGBClassifier(
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
        )
    ]
)


# ============================================================
# 8. METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    probabilities
):

    predictions = (
        probabilities >= 0.5
    ).astype(int)


    accuracy = accuracy_score(
        y_true,
        predictions
    )


    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )


    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )


    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )


    try:

        auroc = roc_auc_score(
            y_true,
            probabilities
        )

    except ValueError:

        auroc = np.nan


    try:

        auprc = average_precision_score(
            y_true,
            probabilities
        )

    except ValueError:

        auprc = np.nan


    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUROC": auroc,
        "AUPRC": auprc
    }


# ============================================================
# 9. TRAIN MODELS
# ============================================================

all_metrics = []

prediction_tables = {}


for model_name, model in models.items():

    print("\n------------------------------------------")
    print(
        "Training:",
        model_name
    )
    print("------------------------------------------")


    # IMPORTANT:
    # Imputer is fitted ONLY here on X_train.

    model.fit(
        X_train,
        y_train
    )


    # Validation probabilities

    validation_probability = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )


    # Test probabilities

    test_probability = (
        model.predict_proba(
            X_test
        )[:, 1]
    )


    # Metrics

    validation_metrics = calculate_metrics(
        y_validation,
        validation_probability
    )

    test_metrics = calculate_metrics(
        y_test,
        test_probability
    )


    # Add model name

    validation_row = {
        "Model": model_name,
        "Split": "Validation"
    }

    validation_row.update(
        validation_metrics
    )


    test_row = {
        "Model": model_name,
        "Split": "Test"
    }

    test_row.update(
        test_metrics
    )


    all_metrics.append(
        validation_row
    )

    all_metrics.append(
        test_row
    )


    # Save predictions

    validation_predictions = validation[
        [
            "subject_id",
            "hadm_id",
            "stay_id",
            TARGET
        ]
    ].copy()


    validation_predictions[
        "predicted_probability"
    ] = validation_probability


    validation_predictions[
        "predicted_class"
    ] = (
        validation_probability >= 0.5
    ).astype(int)


    test_predictions = test[
        [
            "subject_id",
            "hadm_id",
            "stay_id",
            TARGET
        ]
    ].copy()


    test_predictions[
        "predicted_probability"
    ] = test_probability


    test_predictions[
        "predicted_class"
    ] = (
        test_probability >= 0.5
    ).astype(int)


    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )


    validation_predictions.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{safe_name}_validation_predictions.csv"
        ),
        index=False
    )


    test_predictions.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{safe_name}_test_predictions.csv"
        ),
        index=False
    )


    prediction_tables[
        model_name
    ] = {
        "validation": validation_predictions,
        "test": test_predictions
    }


    print(
        "\nValidation AUROC:",
        validation_metrics["AUROC"]
    )

    print(
        "Validation AUPRC:",
        validation_metrics["AUPRC"]
    )

    print(
        "Test AUROC:",
        test_metrics["AUROC"]
    )

    print(
        "Test AUPRC:",
        test_metrics["AUPRC"]
    )


# ============================================================
# 10. SAVE ALL METRICS
# ============================================================

metrics_df = pd.DataFrame(
    all_metrics
)


metrics_file = os.path.join(
    OUTPUT_DIR,
    "step29_model_metrics.csv"
)


metrics_df.to_csv(
    metrics_file,
    index=False
)


# ============================================================
# 11. SAVE XGBOOST TEST PREDICTIONS
# ============================================================

xgb_test = prediction_tables[
    "XGBoost"
]["test"].copy()


xgb_test.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "step29_xgboost_test_predictions.csv"
    ),
    index=False
)


# ============================================================
# 12. PRINT FINAL TABLE
# ============================================================

print("\n==========================================")
print("MODEL COMPARISON")
print("==========================================\n")


print(
    metrics_df.to_string(
        index=False
    )
)


# ============================================================
# 13. IMPORTANT LEAKAGE CHECK
# ============================================================

print("\n==========================================")
print("LEAKAGE CHECK")
print("==========================================")

print(
    "\nImputation is inside each model Pipeline."
)

print(
    "The imputer was fitted using TRAIN data only."
)

print(
    "Validation and test data were transformed "
    "using the train-fitted imputer."
)

print(
    "\nNo global imputation was performed."
)


# ============================================================
# 14. COMPLETE
# ============================================================

print("\n==========================================")
print("STEP 29 COMPLETE")
print("==========================================")

print(
    "\nMetrics saved to:"
)

print(
    metrics_file
)

print(
    "\nXGBoost test predictions saved to:"
)

print(
    os.path.join(
        OUTPUT_DIR,
        "step29_xgboost_test_predictions.csv"
    )
)

print("\n==========================================")