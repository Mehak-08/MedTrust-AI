import os
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
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
# Step 6: Random Forest Baseline
# ============================================================

print("\n==========================================")
print("MEDTRUST AI - RANDOM FOREST")
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
# 4. CREATE RANDOM FOREST PIPELINE
# ============================================================

model = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
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

print(
    "Random Forest training completed."
)


# ============================================================
# 6. VALIDATION PREDICTIONS
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

val_auc = roc_auc_score(
    y_val,
    val_probabilities
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

print(
    "AUROC    :",
    round(val_auc, 4)
)


print("\nValidation Confusion Matrix:")

print(
    confusion_matrix(
        y_val,
        val_predictions
    )
)


# ============================================================
# 8. TEST PREDICTIONS
# ============================================================

test_predictions = model.predict(
    X_test
)

test_probabilities = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 9. TEST RESULTS
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
# 10. TEST CONFUSION MATRIX
# ============================================================

print("\nTest Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        test_predictions
    )
)


# ============================================================
# 11. SHOW TEST PREDICTIONS
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
# 12. SAVE RESULTS
# ============================================================

results_file = os.path.join(
    PROJECT_PATH,
    "random_forest_test_predictions.csv"
)


results.to_csv(
    results_file,
    index=False
)


# ============================================================
# 13. FEATURE IMPORTANCE
# ============================================================

print("\n==========================================")
print("TOP FEATURE IMPORTANCE")
print("==========================================")


# Get the trained Random Forest from the pipeline
rf_model = model.named_steps[
    "classifier"
]


feature_names = X_train.columns


importance_df = pd.DataFrame(
    {
        "feature": feature_names,
        "importance": rf_model.feature_importances_
    }
)


importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)


print(
    importance_df.head(15).to_string(
        index=False
    )
)


# ============================================================
# 14. SAVE FEATURE IMPORTANCE
# ============================================================

importance_file = os.path.join(
    PROJECT_PATH,
    "random_forest_feature_importance.csv"
)


importance_df.to_csv(
    importance_file,
    index=False
)


# ============================================================
# 15. COMPLETE
# ============================================================

print("\n==========================================")
print("STEP 6 COMPLETE")
print("==========================================")

print(
    "Predictions saved to:"
)

print(
    results_file
)

print(
    "\nFeature importance saved to:"
)

print(
    importance_file
)

print(
    "\nNext step: XGBoost primary model."
)