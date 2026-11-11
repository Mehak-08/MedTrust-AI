import os
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# MEDTRUST AI
# Step 5: Logistic Regression Baseline
# ============================================================

print("\n==========================================")
print("MEDTRUST AI - LOGISTIC REGRESSION")
print("==========================================\n")


# ============================================================
# 1. LOAD DATA
# ============================================================

PROJECT_PATH = os.path.dirname(__file__)

train_file = os.path.join(
    PROJECT_PATH,
    "train.csv"
)

validation_file = os.path.join(
    PROJECT_PATH,
    "validation.csv"
)

test_file = os.path.join(
    PROJECT_PATH,
    "test.csv"
)


print("Loading datasets...")

train_df = pd.read_csv(train_file)
val_df = pd.read_csv(validation_file)
test_df = pd.read_csv(test_file)


print("Training shape:", train_df.shape)
print("Validation shape:", val_df.shape)
print("Test shape:", test_df.shape)


# ============================================================
# 2. DEFINE TARGET
# ============================================================

TARGET = "icu_mortality"


# ============================================================
# 3. REMOVE IDENTIFIER COLUMNS
# ============================================================

ID_COLUMNS = [
    "subject_id",
    "hadm_id",
    "stay_id"
]


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
# 4. CREATE LOGISTIC REGRESSION PIPELINE
# ============================================================
#
# Simple explanation:
#
# Step 1:
# Fill missing values using training data.
#
# Step 2:
# Standardize numerical features.
#
# Step 3:
# Train Logistic Regression.
#
# The pipeline prevents information from the validation/test
# sets being used while fitting preprocessing.
#


model = Pipeline(
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
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)


# ============================================================
# 5. TRAIN MODEL
# ============================================================

print("\n==========================================")
print("TRAINING MODEL")
print("==========================================")

model.fit(
    X_train,
    y_train
)

print("Logistic Regression training completed.")


# ============================================================
# 6. MAKE VALIDATION PREDICTIONS
# ============================================================

val_predictions = model.predict(
    X_val
)

val_probabilities = model.predict_proba(
    X_val
)[:, 1]


# ============================================================
# 7. VALIDATION RESULTS
# ============================================================

print("\n==========================================")
print("VALIDATION RESULTS")
print("==========================================")


val_accuracy = accuracy_score(
    y_val,
    val_predictions
)

val_precision = precision_score(
    y_val,
    val_predictions,
    zero_division=0
)

val_recall = recall_score(
    y_val,
    val_predictions,
    zero_division=0
)

val_f1 = f1_score(
    y_val,
    val_predictions,
    zero_division=0
)


print(
    "Accuracy :",
    round(val_accuracy, 4)
)

print(
    "Precision:",
    round(val_precision, 4)
)

print(
    "Recall   :",
    round(val_recall, 4)
)

print(
    "F1 Score :",
    round(val_f1, 4)
)


# ============================================================
# 8. VALIDATION AUROC
# ============================================================

try:

    val_auc = roc_auc_score(
        y_val,
        val_probabilities
    )

    print(
        "AUROC    :",
        round(val_auc, 4)
    )

except ValueError:

    print(
        "AUROC    : Cannot be calculated"
    )


# ============================================================
# 9. VALIDATION CONFUSION MATRIX
# ============================================================

print("\nValidation Confusion Matrix:")

print(
    confusion_matrix(
        y_val,
        val_predictions
    )
)


# ============================================================
# 10. TEST PREDICTIONS
# ============================================================

test_predictions = model.predict(
    X_test
)

test_probabilities = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 11. TEST RESULTS
# ============================================================

print("\n==========================================")
print("TEST RESULTS")
print("==========================================")


test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

test_precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_auc = roc_auc_score(
    y_test,
    test_probabilities
)


print(
    "Accuracy :",
    round(test_accuracy, 4)
)

print(
    "Precision:",
    round(test_precision, 4)
)

print(
    "Recall   :",
    round(test_recall, 4)
)

print(
    "F1 Score :",
    round(test_f1, 4)
)

print(
    "AUROC    :",
    round(test_auc, 4)
)


# ============================================================
# 12. TEST CONFUSION MATRIX
# ============================================================

print("\nTest Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        test_predictions
    )
)


# ============================================================
# 13. SHOW PREDICTED PROBABILITIES
# ============================================================

results = test_df[
    [
        "subject_id",
        "hadm_id",
        "stay_id",
        TARGET
    ]
].copy()


results["predicted_probability"] = (
    test_probabilities
)

results["predicted_class"] = (
    test_predictions
)


print("\n==========================================")
print("TEST PREDICTIONS")
print("==========================================")

print(
    results.to_string(
        index=False
    )
)


# ============================================================
# 14. SAVE RESULTS
# ============================================================

results_file = os.path.join(
    PROJECT_PATH,
    "logistic_regression_test_predictions.csv"
)


results.to_csv(
    results_file,
    index=False
)


print("\n==========================================")
print("STEP 5 COMPLETE")
print("==========================================")

print(
    "Results saved to:"
)

print(
    results_file
)

print(
    "\nNext step: Random Forest baseline."
)