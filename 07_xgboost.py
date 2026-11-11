import os
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from xgboost import XGBClassifier


# ============================================================
# MEDTRUST AI
# Step 7: XGBoost Primary Model
# ============================================================

print("\n==========================================")
print("MEDTRUST AI - XGBOOST")
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
# 2. TARGET AND ID COLUMNS
# ============================================================

TARGET = "icu_mortality"

ID_COLUMNS = [
    "subject_id",
    "hadm_id",
    "stay_id"
]


# ============================================================
# 3. CREATE FEATURES AND TARGET
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
# 4. CREATE XGBOOST MODEL
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
    ]
)


# ============================================================
# 5. TRAIN
# ============================================================

print("\n==========================================")
print("TRAINING XGBOOST")
print("==========================================")

model.fit(
    X_train,
    y_train
)

print(
    "XGBoost training completed."
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
# 7. VALIDATION METRICS
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


print("Accuracy :", round(val_accuracy, 4))
print("Precision:", round(val_precision, 4))
print("Recall   :", round(val_recall, 4))
print("F1 Score :", round(val_f1, 4))
print("AUROC    :", round(val_auc, 4))


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
# 9. TEST METRICS
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


print("Accuracy :", round(test_accuracy, 4))
print("Precision:", round(test_precision, 4))
print("Recall   :", round(test_recall, 4))
print("F1 Score :", round(test_f1, 4))
print("AUROC    :", round(test_auc, 4))


print("\nTest Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        test_predictions
    )
)


# ============================================================
# 10. SAVE TEST PREDICTIONS
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


results_file = os.path.join(
    PROJECT_PATH,
    "xgboost_test_predictions.csv"
)


results.to_csv(
    results_file,
    index=False
)


# ============================================================
# 11. FEATURE IMPORTANCE
# ============================================================

print("\n==========================================")
print("TOP XGBOOST FEATURES")
print("==========================================")


xgb_model = model.named_steps[
    "classifier"
]


importance_df = pd.DataFrame(
    {
        "feature": X_train.columns,
        "importance": xgb_model.feature_importances_
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
# 12. SAVE FEATURE IMPORTANCE
# ============================================================

importance_file = os.path.join(
    PROJECT_PATH,
    "xgboost_feature_importance.csv"
)


importance_df.to_csv(
    importance_file,
    index=False
)


# ============================================================
# 13. COMPLETE
# ============================================================

print("\n==========================================")
print("STEP 7 COMPLETE")
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
    "\nNext step: Compare the three models."
)