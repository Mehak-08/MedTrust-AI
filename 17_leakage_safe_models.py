# ============================================================
# MEDTRUST AI
# STEP 17 - LEAKAGE-SAFE PREPROCESSING + MODEL TRAINING
# ============================================================

import os
import pandas as pd
import numpy as np

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
    brier_score_loss,
    confusion_matrix
)

from xgboost import XGBClassifier


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


print("\n==========================================")
print("MEDTRUST AI - LEAKAGE SAFE MODELS")
print("==========================================\n")


# ------------------------------------------------------------
# 1. LOAD PATIENT-LEVEL SPLITS
# ------------------------------------------------------------

train = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "train_patient_level.csv"
    )
)

validation = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "validation_patient_level.csv"
    )
)

test = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "test_patient_level.csv"
    )
)


print("Train:", train.shape)
print("Validation:", validation.shape)
print("Test:", test.shape)


# ------------------------------------------------------------
# 2. TARGET
# ------------------------------------------------------------

TARGET = "icu_mortality"


# ------------------------------------------------------------
# 3. REMOVE IDENTIFIERS
# ------------------------------------------------------------

DROP_COLUMNS = [
    "subject_id",
    "hadm_id",
    "stay_id"
]


feature_columns = [
    column
    for column in train.columns
    if column not in DROP_COLUMNS + [TARGET]
]


X_train = train[
    feature_columns
].copy()

X_validation = validation[
    feature_columns
].copy()

X_test = test[
    feature_columns
].copy()


y_train = train[
    TARGET
].copy()

y_validation = validation[
    TARGET
].copy()

y_test = test[
    TARGET
].copy()


print(
    "\nNumber of model features:",
    len(feature_columns)
)


# ------------------------------------------------------------
# 4. LEAKAGE CHECK
# ------------------------------------------------------------

for forbidden in [
    "outtime",
    "window_end",
    "los",
    "deathtime",
    "hospital_expire_flag"
]:

    if forbidden in feature_columns:

        raise ValueError(
            f"Potential leakage column found: {forbidden}"
        )


print(
    "Leakage column check: PASSED"
)


# ------------------------------------------------------------
# 5. CREATE MODELS
# ------------------------------------------------------------

models = {

    "Logistic Regression": Pipeline([
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
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]),


    "Random Forest": Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    ]),


    "XGBoost": Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "model",
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
}


# ------------------------------------------------------------
# 6. EVALUATION FUNCTION
# ------------------------------------------------------------

def evaluate_model(
    model,
    X,
    y,
    name
):

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]


    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )


    try:

        auroc = roc_auc_score(
            y,
            probabilities
        )

    except ValueError:

        auroc = np.nan


    brier = brier_score_loss(
        y,
        probabilities
    )


    cm = confusion_matrix(
        y,
        predictions
    )


    if cm.shape == (2, 2):

        tn, fp, fn, tp = (
            cm.ravel()
        )

        sensitivity = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0
        )

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else 0
        )

    else:

        sensitivity = np.nan
        specificity = np.nan


    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUROC": auroc,
        "Brier": brier,
        "Sensitivity": sensitivity,
        "Specificity": specificity
    }


# ------------------------------------------------------------
# 7. TRAIN + EVALUATE
# ------------------------------------------------------------

all_results = []


for name, model in models.items():

    print("\n==========================================")
    print(name)
    print("==========================================")


    print("Training...")


    # IMPORTANT:
    # The imputer is fitted ONLY on training data.

    model.fit(
        X_train,
        y_train
    )


    print(
        "Evaluating validation set..."
    )


    validation_result = evaluate_model(
        model,
        X_validation,
        y_validation,
        name
    )


    print(
        "Validation AUROC:",
        validation_result["AUROC"]
    )

    print(
        "Validation Brier:",
        validation_result["Brier"]
    )


    print(
        "Evaluating test set..."
    )


    test_result = evaluate_model(
        model,
        X_test,
        y_test,
        name
    )


    print(
        "Test AUROC:",
        test_result["AUROC"]
    )

    print(
        "Test Brier:",
        test_result["Brier"]
    )


    validation_result[
        "Dataset"
    ] = "Validation"


    test_result[
        "Dataset"
    ] = "Test"


    all_results.append(
        validation_result
    )

    all_results.append(
        test_result
    )


    # --------------------------------------------------------
    # SAVE TEST PREDICTIONS
    # --------------------------------------------------------

    test_probabilities = (
        model.predict_proba(X_test)[:, 1]
    )


    prediction_output = test[
        [
            "subject_id",
            "hadm_id",
            "stay_id",
            TARGET
        ]
    ].copy()


    prediction_output[
        "predicted_probability"
    ] = test_probabilities


    prediction_output[
        "predicted_class"
    ] = (
        test_probabilities >= 0.5
    ).astype(int)


    safe_name = (
        name.lower()
        .replace(" ", "_")
    )


    prediction_output.to_csv(
        os.path.join(
            BASE_DIR,
            f"step17_{safe_name}_predictions.csv"
        ),
        index=False
    )


# ------------------------------------------------------------
# 8. RESULTS TABLE
# ------------------------------------------------------------

results_df = pd.DataFrame(
    all_results
)


results_df = results_df[
    [
        "Model",
        "Dataset",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "AUROC",
        "Brier",
        "Sensitivity",
        "Specificity"
    ]
]


results_path = os.path.join(
    BASE_DIR,
    "step17_leakage_safe_model_results.csv"
)


results_df.to_csv(
    results_path,
    index=False
)


# ------------------------------------------------------------
# 9. PRINT RESULTS
# ------------------------------------------------------------

print("\n==========================================")
print("FINAL LEAKAGE-SAFE RESULTS")
print("==========================================")

print(
    results_df.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 10. SAVE FEATURE LIST
# ------------------------------------------------------------

pd.DataFrame({
    "feature": feature_columns
}).to_csv(
    os.path.join(
        BASE_DIR,
        "step17_feature_list.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# 11. SUMMARY
# ------------------------------------------------------------

print("\n==========================================")
print("STEP 17 COMPLETE")
print("==========================================")

print(
    "\nImputation is now performed "
    "inside the model pipeline."
)

print(
    "The imputer is fitted only on TRAIN data."
)

print(
    "Validation and test data do not "
    "influence preprocessing."
)

print(
    "\nCreated:"
)

print(
    "step17_leakage_safe_model_results.csv"
)

print(
    "step17_logistic_regression_predictions.csv"
)

print(
    "step17_random_forest_predictions.csv"
)

print(
    "step17_xgboost_predictions.csv"
)

print(
    "step17_feature_list.csv"
)

print("\n==========================================")