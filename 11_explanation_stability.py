import os
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import shap


print("\n==========================================")
print("MEDTRUST AI - EXPLANATION STABILITY")
print("==========================================\n")


# -------------------------------------------------
# PROJECT PATH
# -------------------------------------------------

PROJECT_PATH = os.path.dirname(__file__)


# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

train_df = pd.read_csv(
    os.path.join(
        PROJECT_PATH,
        "train.csv"
    )
)

test_df = pd.read_csv(
    os.path.join(
        PROJECT_PATH,
        "test.csv"
    )
)


print(
    "Training shape:",
    train_df.shape
)

print(
    "Test shape:",
    test_df.shape
)


# -------------------------------------------------
# TARGET AND ID COLUMNS
# -------------------------------------------------

TARGET = "icu_mortality"

ID_COLUMNS = [
    "subject_id",
    "hadm_id",
    "stay_id"
]


# -------------------------------------------------
# PREPARE FEATURES
# -------------------------------------------------

X_train = train_df.drop(
    columns=[TARGET] + ID_COLUMNS
)

y_train = train_df[TARGET]

X_test = test_df.drop(
    columns=[TARGET] + ID_COLUMNS
)


print("\n==========================================")
print("PREPARING FEATURES")
print("==========================================\n")


# -------------------------------------------------
# IMPUTATION
# -------------------------------------------------

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


# -------------------------------------------------
# TRAIN XGBOOST
# -------------------------------------------------

print("\n==========================================")
print("TRAINING XGBOOST")
print("==========================================\n")


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


# -------------------------------------------------
# SHAP EXPLAINER
# -------------------------------------------------

print("\n==========================================")
print("CREATING SHAP EXPLAINER")
print("==========================================\n")


explainer = shap.TreeExplainer(
    xgb_model
)


# -------------------------------------------------
# ORIGINAL SHAP VALUES
# -------------------------------------------------

print(
    "Calculating original SHAP values..."
)


original_shap = explainer.shap_values(
    X_test_imputed
)


if hasattr(
    original_shap,
    "values"
):
    original_shap = original_shap.values


original_shap = np.asarray(
    original_shap
)


print(
    "Original SHAP shape:",
    original_shap.shape
)


# -------------------------------------------------
# CREATE PERTURBED DATA
# -------------------------------------------------

print("\n==========================================")
print("CREATING PERTURBED DATA")
print("==========================================\n")


X_test_perturbed = X_test_imputed.copy()


# Random generator for reproducibility
rng = np.random.default_rng(
    42
)


# -------------------------------------------------
# PERTURB EACH FEATURE
# -------------------------------------------------

for column in X_test_perturbed.columns:

    values = X_test_perturbed[column].values

    # Estimate feature scale
    feature_std = np.std(
        X_train_imputed[column].values
    )

    # Avoid zero standard deviation
    if feature_std == 0:
        feature_std = 1.0

    # Add small Gaussian noise
    noise = rng.normal(
        loc=0.0,
        scale=0.05 * feature_std,
        size=len(values)
    )

    X_test_perturbed[column] = (
        values + noise
    )


print(
    "Small perturbations applied."
)


# -------------------------------------------------
# CALCULATE PERTURBED SHAP
# -------------------------------------------------

print(
    "\nCalculating perturbed SHAP values..."
)


perturbed_shap = explainer.shap_values(
    X_test_perturbed
)


if hasattr(
    perturbed_shap,
    "values"
):
    perturbed_shap = perturbed_shap.values


perturbed_shap = np.asarray(
    perturbed_shap
)


print(
    "Perturbed SHAP shape:",
    perturbed_shap.shape
)


# -------------------------------------------------
# CALCULATE SHAP STABILITY
# -------------------------------------------------

print("\n==========================================")
print("CALCULATING SHAP STABILITY")
print("==========================================\n")


stability_scores = []


for i in range(
    len(X_test_imputed)
):

    original = original_shap[i]

    perturbed = perturbed_shap[i]


    # Difference between explanations
    difference = np.mean(
        np.abs(
            original - perturbed
        )
    )


    # Scale by original explanation magnitude
    magnitude = np.mean(
        np.abs(original)
    )


    # Avoid division by zero
    if magnitude == 0:

        stability = 1.0

    else:

        relative_change = (
            difference /
            (magnitude + 1e-8)
        )

        stability = 1.0 / (
            1.0 +
            relative_change
        )


    # Keep score between 0 and 1
    stability = max(
        0.0,
        min(
            1.0,
            stability
        )
    )


    stability_scores.append(
        stability
    )


# -------------------------------------------------
# CREATE RESULTS TABLE
# -------------------------------------------------

results_df = pd.DataFrame(
    {
        "subject_id":
            test_df["subject_id"].values,

        "hadm_id":
            test_df["hadm_id"].values,

        "stay_id":
            test_df["stay_id"].values,

        "icu_mortality":
            test_df["icu_mortality"].values,

        "shap_stability":
            stability_scores
    }
)


# -------------------------------------------------
# ADD MODEL PROBABILITY
# -------------------------------------------------

probabilities = xgb_model.predict_proba(
    X_test_imputed
)[:, 1]


results_df[
    "xgboost_probability"
] = probabilities


# -------------------------------------------------
# SAVE RESULTS
# -------------------------------------------------

output_file = os.path.join(
    PROJECT_PATH,
    "explanation_stability.csv"
)


results_df.to_csv(
    output_file,
    index=False
)


# -------------------------------------------------
# DISPLAY RESULTS
# -------------------------------------------------

print(
    results_df.to_string(
        index=False
    )
)


print("\n==========================================")
print("SUMMARY")
print("==========================================\n")


print(
    "Average SHAP stability:",
    round(
        results_df[
            "shap_stability"
        ].mean(),
        4
    )
)


print(
    "Minimum SHAP stability:",
    round(
        results_df[
            "shap_stability"
        ].min(),
        4
    )
)


print(
    "Maximum SHAP stability:",
    round(
        results_df[
            "shap_stability"
        ].max(),
        4
    )
)


print("\n==========================================")
print("STEP 11 COMPLETE")
print("==========================================\n")


print(
    "Explanation stability saved to:"
)

print(
    output_file
)


print(
    "\nNext step:"
)

print(
    "Combine explanation stability with calibrated confidence."
)