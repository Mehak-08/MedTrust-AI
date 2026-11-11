# ================================================================
# MedTrust AI - ONE FILE FINAL RESEARCH PIPELINE
# Reliability-Aware Clinical Decision Routing
#
# Run:
# python FINAL_MEDTRUST_AI.py
#
# Input:
#   mimic_raw_ml_dataset.csv
#
# Output:
#   final_medtrust_report/
#       MedTrust_AI_Final_Research_Report.docx
#       MedTrust_AI_Final_Research_Report.txt
#       model_performance.csv
#       routing_results.csv
#       shap_feature_importance.csv
#       patient_results.csv
#       model_performance.png
#       shap_importance.png
#       routing_distribution.png
#       confidence_vs_ers.png
#       coverage_vs_error.png
#
# NOTE:
# This uses the MIMIC-IV DEMO dataset currently available locally.
# Results are methodological/demo results, NOT clinical claims.
# ================================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
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
    average_precision_score,
    brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV

try:
    from xgboost import XGBClassifier
except ImportError:
    print("\nERROR: xgboost is not installed.")
    print("Run:")
    print("pip install xgboost")
    raise

try:
    from lightgbm import LGBMClassifier
except ImportError:
    print("\nERROR: lightgbm is not installed.")
    print("Run:")
    print("pip install lightgbm")
    raise

try:
    import shap
except ImportError:
    print("\nERROR: shap is not installed.")
    print("Run:")
    print("pip install shap")
    raise

try:
    from docx import Document
    from docx.shared import Inches, Pt
except ImportError:
    print("\nERROR: python-docx is not installed.")
    print("Run:")
    print("pip install python-docx")
    raise


# ================================================================
# SETTINGS
# ================================================================

BASE_DIR = r"C:\Users\MEHAK\OneDrive\Desktop\MIMIC-Project"

DATA_FILE = os.path.join(
    BASE_DIR,
    "mimic_raw_ml_dataset.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "final_medtrust_report"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42

# Set to True to display interactive matplotlib windows on your screen
SHOW_GRAPHS_INTERACTIVELY = False


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def safe_auc(y_true, probabilities):
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return roc_auc_score(y_true, probabilities)
    except Exception:
        return np.nan


def safe_auprc(y_true, probabilities):
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return average_precision_score(y_true, probabilities)
    except Exception:
        return np.nan


def calculate_metrics(name, y_true, probabilities):
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "AUROC": safe_auc(y_true, probabilities),
        "AUPRC": safe_auprc(y_true, probabilities),
        "Brier": brier_score_loss(
            y_true, probabilities
        )
    }


def make_pipeline(model, scale=False):

    steps = [
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]

    if scale:
        steps.append(
            (
                "scaler",
                StandardScaler()
            )
        )

    steps.append(("model", model))

    return Pipeline(steps)


# ================================================================
# START
# ================================================================

print("\n")
print("=" * 70)
print("MEDTRUST AI")
print("RELIABILITY-AWARE CLINICAL DECISION ROUTING")
print("=" * 70)


# ================================================================
# 1. LOAD DATA
# ================================================================

print("\n[1/10] Loading dataset...")

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        "\nDataset not found:\n"
        + DATA_FILE
        + "\n\nMake sure mimic_raw_ml_dataset.csv exists "
          "inside the MIMIC-Project folder."
    )

df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)


# ================================================================
# 2. IDENTIFY TARGET + FEATURES
# ================================================================

print("\n[2/10] Preparing features...")

TARGET = "icu_mortality"

if TARGET not in df.columns:
    raise ValueError(
        "Target column 'icu_mortality' was not found."
    )

# Keep identifiers for final patient-level results
identifier_columns = [
    c for c in [
        "subject_id",
        "hadm_id",
        "stay_id"
    ]
    if c in df.columns
]

y = pd.to_numeric(
    df[TARGET],
    errors="coerce"
).fillna(0).astype(int)

# Never use identifiers or outcome as predictors
excluded = identifier_columns + [TARGET]

feature_columns = [
    c for c in df.columns
    if c not in excluded
]

X = df[feature_columns].copy()

# Convert everything to numeric where possible
for column in X.columns:
    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )

print("Patients/stays:", len(df))
print("Features:", len(feature_columns))
print("Positive outcomes:", int(y.sum()))
print("Negative outcomes:", int((y == 0).sum()))


# ================================================================
# 3. PATIENT-LEVEL SPLIT
# ================================================================

print("\n[3/10] Creating patient-level train/validation/test split...")

if "subject_id" in df.columns:

    subjects = df["subject_id"].unique()

    train_subjects, temp_subjects = train_test_split(
        subjects,
        test_size=0.30,
        random_state=RANDOM_STATE
    )

    validation_subjects, test_subjects = train_test_split(
        temp_subjects,
        test_size=0.50,
        random_state=RANDOM_STATE
    )

    train_mask = df["subject_id"].isin(train_subjects)
    validation_mask = df["subject_id"].isin(validation_subjects)
    test_mask = df["subject_id"].isin(test_subjects)

else:

    # Fallback only if subject_id is unavailable
    indices = np.arange(len(df))

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y if y.nunique() > 1 else None
    )

    validation_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=RANDOM_STATE
    )

    train_mask = np.zeros(len(df), dtype=bool)
    validation_mask = np.zeros(len(df), dtype=bool)
    test_mask = np.zeros(len(df), dtype=bool)

    train_mask[train_idx] = True
    validation_mask[validation_idx] = True
    test_mask[test_idx] = True


X_train = X.loc[train_mask].copy()
X_validation = X.loc[validation_mask].copy()
X_test = X.loc[test_mask].copy()

y_train = y.loc[train_mask].copy()
y_validation = y.loc[validation_mask].copy()
y_test = y.loc[test_mask].copy()

print("Train:", len(X_train))
print("Validation:", len(X_validation))
print("Test:", len(X_test))


# ================================================================
# 4. TRAIN FOUR MODELS (LR, RF, XGBOOST, LIGHTGBM)
# ================================================================

print("\n[4/10] Training LR, Random Forest, XGBoost and LightGBM...")

lr_model = make_pipeline(
    LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE
    ),
    scale=True
)

rf_model = make_pipeline(
    RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
)

xgb_base = XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

xgb_model = make_pipeline(
    xgb_base
)

lgbm_base = LGBMClassifier(
    n_estimators=300,
    max_depth=3,
    num_leaves=7,
    learning_rate=0.03,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    objective="binary",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    verbose=-1,
    n_jobs=-1
)

lgbm_model = make_pipeline(
    lgbm_base
)

models = {
    "Logistic Regression": lr_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model,
    "LightGBM": lgbm_model
}

model_results = []

validation_probabilities = {}
test_probabilities = {}

for name, model in models.items():

    print("Training:", name)

    model.fit(
        X_train,
        y_train
    )

    val_prob = model.predict_proba(
        X_validation
    )[:, 1]

    test_prob = model.predict_proba(
        X_test
    )[:, 1]

    validation_probabilities[name] = val_prob
    test_probabilities[name] = test_prob

    result = calculate_metrics(
        name,
        y_test,
        test_prob
    )

    model_results.append(result)


model_metrics = pd.DataFrame(
    model_results
)

print("\nModel performance:")
print(
    model_metrics.round(4).to_string(
        index=False
    )
)


# ================================================================
# 5. CALIBRATED MODELS (XGBOOST + LIGHTGBM)
# ================================================================

print("\n[5/10] Calibrating XGBoost and LightGBM...")

calibrated_xgb = CalibratedClassifierCV(
    estimator=make_pipeline(
        XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    ),
    method="sigmoid",
    cv=3
)

calibrated_xgb.fit(
    X_train,
    y_train
)

calibrated_val_prob = calibrated_xgb.predict_proba(
    X_validation
)[:, 1]

calibrated_test_prob = calibrated_xgb.predict_proba(
    X_test
)[:, 1]

calibrated_xgb_metrics = calculate_metrics(
    "Calibrated XGBoost",
    y_test,
    calibrated_test_prob
)

calibrated_lgbm = CalibratedClassifierCV(
    estimator=make_pipeline(
        LGBMClassifier(
            n_estimators=300,
            max_depth=3,
            num_leaves=7,
            learning_rate=0.03,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            objective="binary",
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=-1
        )
    ),
    method="sigmoid",
    cv=3
)

calibrated_lgbm.fit(
    X_train,
    y_train
)

calibrated_lgbm_val_prob = calibrated_lgbm.predict_proba(
    X_validation
)[:, 1]

calibrated_lgbm_test_prob = calibrated_lgbm.predict_proba(
    X_test
)[:, 1]

calibrated_lgbm_metrics = calculate_metrics(
    "Calibrated LightGBM",
    y_test,
    calibrated_lgbm_test_prob
)

print("\nCalibrated XGBoost:")
for key, value in calibrated_xgb_metrics.items():
    if key != "Model":
        print(
            f"{key}: "
            f"{value:.4f}"
            if pd.notna(value)
            else f"{key}: N/A"
        )

print("\nCalibrated LightGBM:")
for key, value in calibrated_lgbm_metrics.items():
    if key != "Model":
        print(
            f"{key}: "
            f"{value:.4f}"
            if pd.notna(value)
            else f"{key}: N/A"
        )


# ================================================================
# 6. SHAP EXPLANATIONS
# ================================================================

print("\n[6/10] Generating SHAP explanations...")

# Fit a fresh underlying XGBoost model.
# SHAP is applied to this underlying tree model rather than
# directly to the calibrated wrapper.

clean_imputer = SimpleImputer(
    strategy="median"
)

X_train_imp = clean_imputer.fit_transform(
    X_train
)

X_test_imp = clean_imputer.transform(
    X_test
)

final_xgb_tree = XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

final_xgb_tree.fit(
    X_train_imp,
    y_train
)

explainer = shap.TreeExplainer(
    final_xgb_tree
)

shap_output = explainer(
    X_test_imp
)

shap_values = shap_output.values

# Handle SHAP API variations
if isinstance(shap_values, list):
    shap_values = shap_values[-1]

shap_values = np.asarray(
    shap_values
)

if shap_values.ndim == 3:
    shap_values = shap_values[:, :, -1]

if shap_values.shape[1] != len(feature_columns):
    raise ValueError(
        "SHAP feature count does not match model feature count."
    )

mean_abs_shap = np.mean(
    np.abs(shap_values),
    axis=0
)

shap_importance = pd.DataFrame({
    "Feature": feature_columns,
    "Mean_Absolute_SHAP": mean_abs_shap
}).sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
)

print("\nTop SHAP features:")
print(
    shap_importance.head(10).round(4).to_string(
        index=False
    )
)


# ================================================================
# 7. CALCULATE EXPLANATION RELIABILITY
# ================================================================

print("\n[7/10] Calculating Explanation Reliability Score...")

# We use repeated controlled perturbations.
#
# 1. SHAP stability:
#    How stable are SHAP explanations when input values are
#    slightly perturbed?
#
# 2. Ranking consistency:
#    How consistently do important features remain important?
#
# 3. Sign consistency:
#    Whether feature contribution directions remain stable.
#
# 4. Faithfulness:
#    Whether important features have measurable impact on
#    model probability.
#
# This is a research prototype for the current demo dataset.

rng = np.random.default_rng(
    RANDOM_STATE
)

train_std = np.nanstd(
    X_train_imp,
    axis=0
)

train_std = np.where(
    np.isfinite(train_std) & (train_std > 0),
    train_std,
    1.0
)

ERS_values = []

for i in range(len(X_test_imp)):

    original_row = X_test_imp[i].copy()

    original_shap = shap_values[i]

    # ------------------------------------------------------------
    # A. SHAP STABILITY
    # ------------------------------------------------------------

    perturbed_shaps = []

    for repeat in range(5):

        perturbation = (
            rng.normal(
                0,
                0.03,
                size=len(feature_columns)
            )
            * train_std
        )

        perturbed_row = (
            original_row + perturbation
        )

        perturbed_shap = explainer(
            perturbed_row.reshape(1, -1)
        ).values

        if isinstance(
            perturbed_shap,
            list
        ):
            perturbed_shap = perturbed_shap[-1]

        perturbed_shap = np.asarray(
            perturbed_shap
        )

        if perturbed_shap.ndim == 3:
            perturbed_shap = perturbed_shap[:, :, -1]

        perturbed_shaps.append(
            perturbed_shap.flatten()
        )

    perturbed_shaps = np.asarray(
        perturbed_shaps
    )

    # Correlation between original SHAP and
    # perturbed SHAP vectors
    correlations = []

    for row in perturbed_shaps:

        if (
            np.std(original_shap) > 0
            and np.std(row) > 0
        ):
            corr = np.corrcoef(
                original_shap,
                row
            )[0, 1]

            if np.isfinite(corr):
                correlations.append(
                    max(0, corr)
                )

    stability = (
        np.mean(correlations)
        if correlations
        else 0.0
    )

    # ------------------------------------------------------------
    # B. RANKING CONSISTENCY
    # ------------------------------------------------------------

    original_rank = np.argsort(
        -np.abs(original_shap)
    )

    top_k = min(
        10,
        len(feature_columns)
    )

    original_top = set(
        original_rank[:top_k]
    )

    rank_scores = []

    for row in perturbed_shaps:

        perturbed_rank = np.argsort(
            -np.abs(row)
        )

        perturbed_top = set(
            perturbed_rank[:top_k]
        )

        intersection = len(
            original_top.intersection(
                perturbed_top
            )
        )

        union = len(
            original_top.union(
                perturbed_top
            )
        )

        if union > 0:
            rank_scores.append(
                intersection / union
            )

    ranking_consistency = (
        np.mean(rank_scores)
        if rank_scores
        else 0.0
    )

    # ------------------------------------------------------------
    # C. SIGN CONSISTENCY
    # ------------------------------------------------------------

    top_indices = original_rank[:top_k]

    sign_scores = []

    for feature_index in top_indices:

        original_sign = np.sign(
            original_shap[feature_index]
        )

        if original_sign == 0:
            continue

        same_sign = np.mean(
            np.sign(
                perturbed_shaps[
                    :,
                    feature_index
                ]
            )
            == original_sign
        )

        sign_scores.append(
            same_sign
        )

    sign_consistency = (
        np.mean(sign_scores)
        if sign_scores
        else 0.0
    )

    # ------------------------------------------------------------
    # D. FAITHFULNESS
    # ------------------------------------------------------------

    # Remove top important features by replacing them
    # with their training medians.
    baseline_probability = final_xgb_tree.predict_proba(
        original_row.reshape(1, -1)
    )[0, 1]

    feature_effects = []

    train_medians = np.nanmedian(
        X_train_imp,
        axis=0
    )

    for feature_index in top_indices:

        modified_row = original_row.copy()

        modified_row[feature_index] = (
            train_medians[feature_index]
        )

        modified_probability = (
            final_xgb_tree.predict_proba(
                modified_row.reshape(1, -1)
            )[0, 1]
        )

        effect = abs(
            baseline_probability
            - modified_probability
        )

        feature_effects.append(
            effect
        )

    if feature_effects:

        max_effect = max(
            feature_effects
        )

        # Normalize to [0,1]
        faithfulness = min(
            1.0,
            max_effect * 5.0
        )

    else:
        faithfulness = 0.0

    # ------------------------------------------------------------
    # FINAL ERS
    # ------------------------------------------------------------

    ers = (
        0.25 * stability
        + 0.25 * ranking_consistency
        + 0.25 * sign_consistency
        + 0.25 * faithfulness
    )

    ers = float(
        np.clip(
            ers,
            0,
            1
        )
    )

    ERS_values.append(ers)


ERS_values = np.asarray(
    ERS_values
)

print(
    "\nAverage Explanation Reliability Score:",
    round(float(np.mean(ERS_values)), 4)
)

print(
    "Minimum ERS:",
    round(float(np.min(ERS_values)), 4)
)

print(
    "Maximum ERS:",
    round(float(np.max(ERS_values)), 4)
)


# ================================================================
# 8. MEDTRUST ROUTING
# ================================================================

print("\n[8/10] Performing reliability-aware routing...")

test_confidence = np.maximum(
    calibrated_test_prob,
    1 - calibrated_test_prob
)

# Joint reliability:
#
# 50% calibrated predictive confidence
# 50% explanation reliability
#
# These weights are clearly treated as a prototype
# for this demo rather than a final clinical threshold.

joint_reliability = (
    0.50 * test_confidence
    + 0.50 * ERS_values
)

# Thresholds are intentionally conservative prototype
# thresholds for demonstration.
#
# For the full research study these must be frozen from
# validation data before final testing.

ACCEPT_THRESHOLD = 0.70
REVIEW_THRESHOLD = 0.50


def medtrust_route(score):

    if score >= ACCEPT_THRESHOLD:
        return "ACCEPT"

    elif score >= REVIEW_THRESHOLD:
        return "REVIEW"

    else:
        return "DEFER"


def confidence_route(score):

    if score >= ACCEPT_THRESHOLD:
        return "ACCEPT"

    elif score >= REVIEW_THRESHOLD:
        return "REVIEW"

    else:
        return "DEFER"


medtrust_routes = [
    medtrust_route(
        score
    )
    for score in joint_reliability
]

confidence_only_routes = [
    confidence_route(
        score
    )
    for score in test_confidence
]


# ================================================================
# 9. FINAL PATIENT-LEVEL RESULTS
# ================================================================

patient_results = df.loc[
    test_mask,
    identifier_columns
].reset_index(drop=True)

patient_results["actual_outcome"] = (
    y_test.reset_index(drop=True)
)

patient_results["calibrated_probability"] = (
    calibrated_test_prob
)

patient_results["calibrated_probability_xgb"] = (
    calibrated_test_prob
)

patient_results["calibrated_probability_lgbm"] = (
    calibrated_lgbm_test_prob
)

patient_results["model_disagreement"] = np.abs(
    calibrated_test_prob - calibrated_lgbm_test_prob
)

patient_results["predictive_confidence"] = (
    test_confidence
)

patient_results["explanation_reliability"] = (
    ERS_values
)

patient_results["joint_reliability"] = (
    joint_reliability
)

patient_results["MedTrust_Route"] = (
    medtrust_routes
)

patient_results["Confidence_Only_Route"] = (
    confidence_only_routes
)

patient_results["Routing_Changed"] = (
    patient_results["MedTrust_Route"]
    !=
    patient_results["Confidence_Only_Route"]
)


# ================================================================
# ROUTING METRICS
# ================================================================

def routing_metrics(
    results,
    route_column
):

    accepted = (
        results[
            route_column
        ] == "ACCEPT"
    )

    reviewed = (
        results[
            route_column
        ] == "REVIEW"
    )

    deferred = (
        results[
            route_column
        ] == "DEFER"
    )

    total = len(results)

    accepted_count = int(
        accepted.sum()
    )

    review_count = int(
        reviewed.sum()
    )

    defer_count = int(
        deferred.sum()
    )

    coverage = (
        accepted_count / total
        if total > 0
        else 0
    )

    if accepted_count > 0:

        accepted_error = (
            results.loc[
                accepted,
                "actual_outcome"
            ].mean()
        )

    else:

        accepted_error = np.nan

    return {
        "Strategy": route_column,
        "ACCEPT": accepted_count,
        "REVIEW": review_count,
        "DEFER": defer_count,
        "Coverage": coverage,
        "Accepted_Error": accepted_error
    }


medtrust_metrics = routing_metrics(
    patient_results,
    "MedTrust_Route"
)

confidence_metrics = routing_metrics(
    patient_results,
    "Confidence_Only_Route"
)

routing_metrics_df = pd.DataFrame([
    medtrust_metrics,
    confidence_metrics
])


# ================================================================
# HIGH-CONFIDENCE / LOW-ERS CASES
# ================================================================

high_conf_low_ers = (
    (patient_results["predictive_confidence"] >= 0.80)
    &
    (patient_results["explanation_reliability"] < 0.50)
)

high_conf_low_ers_count = int(
    high_conf_low_ers.sum()
)

routing_changes = int(
    patient_results["Routing_Changed"].sum()
)


# ================================================================
# 10. SAVE DATA + CREATE GRAPHS + REPORT
# ================================================================

print("\n[9/10] Saving results and creating graphs...")


# ------------------------------------------------
# SAVE CSVs
# ------------------------------------------------

all_model_metrics = pd.concat(
    [
        model_metrics,
        pd.DataFrame(
            [
                calibrated_xgb_metrics,
                calibrated_lgbm_metrics
            ]
        )
    ],
    ignore_index=True
)

all_model_metrics.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_performance.csv"
    ),
    index=False
)

routing_metrics_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "routing_results.csv"
    ),
    index=False
)

patient_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "patient_results.csv"
    ),
    index=False
)

shap_importance.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "shap_feature_importance.csv"
    ),
    index=False
)

print("  [+] Saved CSV files:")
print("      - model_performance.csv")
print("      - routing_results.csv")
print("      - patient_results.csv")
print("      - shap_feature_importance.csv")

print("\n  [+] Generating and saving figures to disk:")


# ------------------------------------------------
# GRAPH 1: MODEL PERFORMANCE
# ------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    all_model_metrics["Model"],
    all_model_metrics["AUROC"]
)

plt.ylabel("AUROC")
plt.xlabel("Model")
plt.title(
    "Model Performance Comparison"
)

plt.ylim(0, 1)

plt.xticks(
    rotation=20,
    ha="right"
)

plt.tight_layout()

graph1_path = os.path.join(
    OUTPUT_DIR,
    "model_performance.png"
)

plt.savefig(
    graph1_path,
    dpi=300
)

print(f"      1/5 -> Saved: {graph1_path}")

if SHOW_GRAPHS_INTERACTIVELY:
    plt.show()

plt.close()


# ------------------------------------------------
# GRAPH 2: SHAP IMPORTANCE
# ------------------------------------------------

top_features = (
    shap_importance
    .head(10)
    .sort_values(
        "Mean_Absolute_SHAP"
    )
)

plt.figure(
    figsize=(9, 6)
)

plt.barh(
    top_features["Feature"],
    top_features["Mean_Absolute_SHAP"]
)

plt.xlabel(
    "Mean Absolute SHAP Value"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Top 10 Features by SHAP Importance"
)

plt.tight_layout()

graph2_path = os.path.join(
    OUTPUT_DIR,
    "shap_importance.png"
)

plt.savefig(
    graph2_path,
    dpi=300
)

print(f"      2/5 -> Saved: {graph2_path}")

if SHOW_GRAPHS_INTERACTIVELY:
    plt.show()

plt.close()


# ------------------------------------------------
# GRAPH 3: ROUTING DISTRIBUTION
# ------------------------------------------------

routing_plot = pd.DataFrame({
    "Strategy": [
        "Confidence Only",
        "MedTrust AI"
    ],
    "ACCEPT": [
        confidence_metrics["ACCEPT"],
        medtrust_metrics["ACCEPT"]
    ],
    "REVIEW": [
        confidence_metrics["REVIEW"],
        medtrust_metrics["REVIEW"]
    ],
    "DEFER": [
        confidence_metrics["DEFER"],
        medtrust_metrics["DEFER"]
    ]
})

plt.figure(
    figsize=(9, 6)
)

x = np.arange(
    len(routing_plot)
)

width = 0.25

plt.bar(
    x - width,
    routing_plot["ACCEPT"],
    width,
    label="ACCEPT"
)

plt.bar(
    x,
    routing_plot["REVIEW"],
    width,
    label="REVIEW"
)

plt.bar(
    x + width,
    routing_plot["DEFER"],
    width,
    label="DEFER"
)

plt.xticks(
    x,
    routing_plot["Strategy"]
)

plt.ylabel(
    "Number of Test Cases"
)

plt.title(
    "Confidence-Only vs MedTrust Routing"
)

plt.legend()

plt.tight_layout()

graph3_path = os.path.join(
    OUTPUT_DIR,
    "routing_distribution.png"
)

plt.savefig(
    graph3_path,
    dpi=300
)

print(f"      3/5 -> Saved: {graph3_path}")

if SHOW_GRAPHS_INTERACTIVELY:
    plt.show()

plt.close()


# ------------------------------------------------
# GRAPH 4: CONFIDENCE VS ERS
# ------------------------------------------------

plt.figure(
    figsize=(9, 6)
)

plt.scatter(
    patient_results[
        "predictive_confidence"
    ],
    patient_results[
        "explanation_reliability"
    ]
)

plt.axhline(
    0.50,
    linestyle="--"
)

plt.axvline(
    0.80,
    linestyle="--"
)

plt.xlabel(
    "Predictive Confidence"
)

plt.ylabel(
    "Explanation Reliability Score"
)

plt.title(
    "Predictive Confidence vs Explanation Reliability"
)

plt.tight_layout()

graph4_path = os.path.join(
    OUTPUT_DIR,
    "confidence_vs_ers.png"
)

plt.savefig(
    graph4_path,
    dpi=300
)

print(f"      4/5 -> Saved: {graph4_path}")

if SHOW_GRAPHS_INTERACTIVELY:
    plt.show()

plt.close()


# ------------------------------------------------
# GRAPH 5: COVERAGE VS ACCEPTED ERROR
# ------------------------------------------------

strategies = [
    "Confidence Only",
    "MedTrust AI"
]

coverage_values = [
    confidence_metrics["Coverage"],
    medtrust_metrics["Coverage"]
]

error_values = [
    confidence_metrics["Accepted_Error"],
    medtrust_metrics["Accepted_Error"]
]

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    coverage_values,
    error_values,
    s=100
)

for i, strategy in enumerate(
    strategies
):

    plt.annotate(
        strategy,
        (
            coverage_values[i],
            error_values[i]
        ),
        xytext=(8, 8),
        textcoords="offset points"
    )

plt.xlabel(
    "Accepted Coverage"
)

plt.ylabel(
    "Accepted Error"
)

plt.title(
    "Selective Routing: Coverage vs Accepted Error"
)

plt.xlim(0, 1)

plt.tight_layout()

graph5_path = os.path.join(
    OUTPUT_DIR,
    "coverage_vs_error.png"
)

plt.savefig(
    graph5_path,
    dpi=300
)

print(f"      5/5 -> Saved: {graph5_path}")

if SHOW_GRAPHS_INTERACTIVELY:
    plt.show()

plt.close()

print(f"  All 5 graphs and 4 CSVs ready in: {OUTPUT_DIR}")


# ================================================================
# FINAL TEXT REPORT
# ================================================================

print("\n[10/10] Creating final research report...")


def fmt(value):

    if pd.isna(value):
        return "N/A"

    return f"{value:.4f}"


report_lines = []

report_lines.append(
    "MEDTRUST AI\n"
    "RELIABILITY-AWARE CLINICAL DECISION ROUTING"
)

report_lines.append(
    "\n============================================================"
)

report_lines.append(
    "\n1. RESEARCH OVERVIEW\n"
)

report_lines.append(
    "MedTrust AI investigates whether clinical AI reliability "
    "should be assessed using predictive confidence alone or "
    "by combining predictive confidence with explanation "
    "reliability."
)

report_lines.append(
    "\nThe proposed framework combines a calibrated prediction "
    "confidence score with an Explanation Reliability Score "
    "(ERS) and uses the combined signal to route predictions "
    "to ACCEPT, REVIEW, or DEFER."
)

report_lines.append(
    "\n============================================================"
)

report_lines.append(
    "\n2. RESEARCH PROBLEM\n"
)

report_lines.append(
    "A model can produce a high-confidence prediction while "
    "its explanation may be unstable or unreliable. Therefore, "
    "confidence alone may not completely describe whether a "
    "prediction should be routinely accepted."
)

report_lines.append(
    "\nMedTrust AI studies explanation reliability as a second "
    "reliability signal."
)

report_lines.append(
    "\n============================================================"
)

report_lines.append(
    "\n3. DATASET\n"
)

report_lines.append(
    "Dataset: MIMIC-IV Clinical Database Demo v2.2"
)

report_lines.append(
    f"\nTotal records/stays used: {len(df)}"
)

report_lines.append(
    f"\nPredictor variables: {len(feature_columns)}"
)

report_lines.append(
    f"\nPositive ICU mortality outcomes: {int(y.sum())}"
)

report_lines.append(
    f"\nNegative ICU mortality outcomes: {int((y == 0).sum())}"
)

report_lines.append(
    "\nThe current experiment uses the MIMIC-IV DEMO dataset "
    "available locally. Therefore, these results should be "
    "treated as methodological validation rather than "
    "clinical evidence."
)

report_lines.append(
    "\n============================================================"
)

report_lines.append(
    "\n4. MACHINE LEARNING MODELS\n"
)

report_lines.append(
    "Four predictive models were evaluated:"
)

report_lines.append(
    "\n- Logistic Regression"
)

report_lines.append(
    "\n- Random Forest"
)

report_lines.append(
    "\n- XGBoost"
)

report_lines.append(
    "\n- LightGBM"
)

report_lines.append(
    "\nBoth XGBoost and LightGBM were additionally calibrated using sigmoid "
    "probability calibration."
)

report_lines.append(
    "\n============================================================"
)

report_lines.append(
    "\n5. MODEL PERFORMANCE\n"
)

for _, row in all_model_metrics.iterrows():

    report_lines.append(
        f"\n{row['Model']}:"
    )

    report_lines.append(
        f"\n  Accuracy: {fmt(row['Accuracy'])}"
    )

    report_lines.append(
        f"\n  Precision: {fmt(row['Precision'])}"
    )

    report_lines.append(
        f"\n  Recall: {fmt(row['Recall'])}"
    )

    report_lines.append(
        f"\n  F1: {fmt(row['F1'])}"
    )

    report_lines.append(
        f"\n  AUROC: {fmt(row['AUROC'])}"
    )

    report_lines.append(
        f"\n  AUPRC: {fmt(row['AUPRC'])}"
    )

    report_lines.append(
        f"\n  Brier Score: {fmt(row['Brier'])}"
    )


report_lines.append(
    "\n============================================================"
)

report_lines.append(
    "\n6. EXPLAINABILITY\n"
)

report_lines.append(
    "\nSHAP TreeExplainer was used to investigate the contribution "
    "of individual features to the XGBoost predictions."
)

report_lines.append(
    "\nTop features by mean absolute SHAP value:"
)

for _, row in shap_importance.head(10).iterrows():

    report_lines.append(
        f"\n- {row['Feature']}: "
        f"{row['Mean_Absolute_SHAP']:.4f}"
    )


report_lines.append(
    "\n============================================================"
)

report_lines.append(
    "\n7. EXPLANATION RELIABILITY SCORE\n"
)

report_lines.append(
    f"\nAverage ERS: {np.mean(ERS_values):.4f}"
)

report_lines.append(
    f"\nMinimum ERS: {np.min(ERS_values):.4f}"
)

report_lines.append(
    f"\nMaximum ERS: {np.max(ERS_values):.4f}"
)

report_lines.append(
    "\nThe prototype ERS combines explanation stability, "
    "feature-ranking consistency, sign consistency and "
    "faithfulness under controlled perturbations."
)

report_lines.append(
    "\n============================================================"
)

report_lines.append(
    "\n8. MEDTRUST ROUTING\n"
)

report_lines.append(
    f"\nPrototype ACCEPT threshold: "
    f"{ACCEPT_THRESHOLD:.2f}"
)

report_lines.append(
    f"\nPrototype REVIEW threshold: "
    f"{REVIEW_THRESHOLD:.2f}"
)

report_lines.append(
    "\nMedTrust AI combines:"
)

report_lines.append(
    "\n  50% predictive confidence"
)

report_lines.append(
    "\n  50% explanation reliability"
)

report_lines.append(
    "\nRouting categories:"
)

report_lines.append(
    "\n  ACCEPT = prediction has sufficient joint reliability"
)

report_lines.append(
    "\n  REVIEW = prediction requires human review"
)

report_lines.append(
    "\n  DEFER = prediction has insufficient reliability"
)


report_lines.append(
    "\n============================================================"
)

report_lines.append(
    "\n9. ROUTING RESULTS\n"
)

report_lines.append(
    "\nConfidence-only strategy:"
)

report_lines.append(
    f"\n  ACCEPT: "
    f"{confidence_metrics['ACCEPT']}"
)

report_lines.append(
    f"\n  REVIEW: "
    f"{confidence_metrics['REVIEW']}"
)

report_lines.append(
    f"\n  DEFER: "
    f"{confidence_metrics['DEFER']}"
)

report_lines.append(
    f"\n  Coverage: "
    f"{confidence_metrics['Coverage']:.4f}"
)

report_lines.append(
    f"\n  Accepted error: "
    f"{confidence_metrics['Accepted_Error']:.4f}"
    if pd.notna(
        confidence_metrics["Accepted_Error"]
    )
    else "\n  Accepted error: N/A"
)


report_lines.append(
    "\n\nMedTrust AI strategy:"
)

report_lines.append(
    f"\n  ACCEPT: "
    f"{medtrust_metrics['ACCEPT']}"
)

report_lines.append(
    f"\n  REVIEW: "
    f"{medtrust_metrics['REVIEW']}"
)

report_lines.append(
    f"\n  DEFER: "
    f"{medtrust_metrics['DEFER']}"
)

report_lines.append(
    f"\n  Coverage: "
    f"{medtrust_metrics['Coverage']:.4f}"
)

report_lines.append(
    f"\n  Accepted error: "
    f"{medtrust_metrics['Accepted_Error']:.4f}"
    if pd.notna(
        medtrust_metrics["Accepted_Error"]
    )
    else "\n  Accepted error: N/A"
)

report_lines.append(
    f"\n\nRouting changed for "
    f"{routing_changes} test cases."
)

report_lines.append(
    f"\nHigh-confidence / low-ERS cases: "
    f"{high_conf_low_ers_count}"
)


report_lines.append(
    "\n============================================================"
)

report_lines.append(
    "\n10. RESEARCH INTERPRETATION\n"
)

report_lines.append(
    "\nThe experiment demonstrates a prototype reliability-aware "
    "routing mechanism in which predictive confidence is not "
    "treated as the only reliability signal."
)

report_lines.append(
    "\nThe additional explanation-reliability signal can identify "
    "cases where predictive confidence and explanation reliability "
    "do not agree."
)

report_lines.append(
    "\nSuch cases can be routed for additional human review instead "
    "of being automatically accepted."
)


report_lines.append(
    "\n============================================================"
)

report_lines.append(
    "\n11. LIMITATIONS\n"
)

report_lines.append(
    "\n- The current experiment uses the MIMIC-IV DEMO dataset."
)

report_lines.append(
    "\n- The number of mortality-positive cases is very small."
)

report_lines.append(
    "\n- Therefore performance metrics can be unstable."
)

report_lines.append(
    "\n- ERS components are a research prototype and require "
    "further validation."
)

report_lines.append(
    "\n- Routing thresholds and the 50/50 joint-reliability "
    "weight are prototype choices for this demonstration."
)

report_lines.append(
    "\n- The results should not be interpreted as evidence "
    "for clinical deployment."
)

report_lines.append(
    "\n============================================================"
)

report_lines.append(
    "\n12. FUTURE WORK\n"
)

report_lines.append(
    "\n- Validate on the full MIMIC-IV dataset."
)

report_lines.append(
    "\n- Perform temporal validation."
)

report_lines.append(
    "\n- Perform external validation using eICU-CRD."
)

report_lines.append(
    "\n- Improve and formally validate the ERS definition."
)

report_lines.append(
    "\n- Select routing thresholds using validation data."
)

report_lines.append(
    "\n- Evaluate statistical uncertainty using confidence intervals."
)

report_lines.append(
    "\n- Investigate out-of-distribution and domain-shift cases."
)


report_lines.append(
    "\n============================================================"
)

report_lines.append(
    "\n13. CONCLUSION\n"
)

report_lines.append(
    "\nMedTrust AI presents a reliability-aware clinical decision "
    "routing framework that combines calibrated predictive "
    "confidence with explanation reliability. The prototype "
    "demonstrates how predictions can be categorized into "
    "ACCEPT, REVIEW and DEFER rather than relying exclusively "
    "on model confidence."
)

report_lines.append(
    "\nThe current results are intended as a methodological "
    "demonstration on the available MIMIC-IV demo dataset. "
    "Full-dataset and external validation are required before "
    "drawing clinical conclusions."
)


final_report_text = "\n".join(
    report_lines
)


# ================================================================
# SAVE TXT REPORT
# ================================================================

txt_path = os.path.join(
    OUTPUT_DIR,
    "MedTrust_AI_Final_Research_Report.txt"
)

with open(
    txt_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        final_report_text
    )


# ================================================================
# CREATE DOCX REPORT
# ================================================================

doc = Document()

title = doc.add_heading(
    "MedTrust AI",
    0
)

subtitle = doc.add_paragraph(
    "Reliability-Aware Clinical Decision Routing"
)

subtitle.runs[0].bold = True

doc.add_paragraph(
    "Final Research Demonstration Report"
)

doc.add_paragraph(
    "MIMIC-IV Clinical Database Demo v2.2"
)

doc.add_heading(
    "Executive Summary",
    level=1
)

doc.add_paragraph(
    "MedTrust AI investigates whether clinical AI reliability "
    "should be assessed using predictive confidence alone or "
    "using both predictive confidence and explanation reliability. "
    "The framework produces an additional Explanation Reliability "
    "Score (ERS) and combines it with calibrated predictive "
    "confidence to route predictions into ACCEPT, REVIEW or DEFER."
)

doc.add_heading(
    "Research Problem",
    level=1
)

doc.add_paragraph(
    "A model may produce a high-confidence prediction while its "
    "explanation is unstable or unreliable. MedTrust AI therefore "
    "treats explanation reliability as a complementary reliability "
    "signal rather than defining reliability only through model "
    "confidence."
)

doc.add_heading(
    "Dataset",
    level=1
)

doc.add_paragraph(
    f"MIMIC-IV Clinical Database Demo v2.2 was used. "
    f"The current dataset contains {len(df)} records/stays and "
    f"{int(y.sum())} positive ICU mortality outcomes."
)

doc.add_paragraph(
    "Because this experiment uses the MIMIC-IV DEMO dataset, "
    "the results are methodological and should not be interpreted "
    "as clinical performance evidence."
)

doc.add_heading(
    "Models",
    level=1
)

doc.add_paragraph(
    "Logistic Regression, Random Forest, XGBoost and LightGBM were evaluated. "
    "Both XGBoost and LightGBM were additionally calibrated using sigmoid calibration."
)

doc.add_heading(
    "Model Performance",
    level=1
)

table = doc.add_table(
    rows=1,
    cols=7
)

table.style = "Table Grid"

headers = [
    "Model",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "AUROC",
    "AUPRC"
]

for i, header in enumerate(headers):

    table.rows[0].cells[i].text = header

for _, row in all_model_metrics.iterrows():

    cells = table.add_row().cells

    cells[0].text = str(
        row["Model"]
    )

    cells[1].text = fmt(
        row["Accuracy"]
    )

    cells[2].text = fmt(
        row["Precision"]
    )

    cells[3].text = fmt(
        row["Recall"]
    )

    cells[4].text = fmt(
        row["F1"]
    )

    cells[5].text = fmt(
        row["AUROC"]
    )

    cells[6].text = fmt(
        row["AUPRC"]
    )


doc.add_heading(
    "Explainability",
    level=1
)

doc.add_paragraph(
    "SHAP TreeExplainer was used to estimate feature contributions "
    "for the XGBoost model."
)

for _, row in shap_importance.head(10).iterrows():

    doc.add_paragraph(
        f"{row['Feature']} — "
        f"mean absolute SHAP = "
        f"{row['Mean_Absolute_SHAP']:.4f}",
        style="List Bullet"
    )


doc.add_heading(
    "Explanation Reliability",
    level=1
)

doc.add_paragraph(
    f"Average Explanation Reliability Score (ERS): "
    f"{np.mean(ERS_values):.4f}"
)

doc.add_paragraph(
    f"Minimum ERS: {np.min(ERS_values):.4f}"
)

doc.add_paragraph(
    f"Maximum ERS: {np.max(ERS_values):.4f}"
)

doc.add_paragraph(
    "The prototype ERS combines explanation stability, "
    "feature-ranking consistency, sign consistency and "
    "faithfulness under controlled perturbations."
)


doc.add_heading(
    "Reliability-Aware Routing",
    level=1
)

doc.add_paragraph(
    "MedTrust AI combines 50% calibrated predictive confidence "
    "and 50% explanation reliability for the prototype joint "
    "reliability score."
)

doc.add_paragraph(
    f"ACCEPT threshold: {ACCEPT_THRESHOLD:.2f}"
)

doc.add_paragraph(
    f"REVIEW threshold: {REVIEW_THRESHOLD:.2f}"
)


doc.add_heading(
    "Routing Comparison",
    level=1
)

table2 = doc.add_table(
    rows=1,
    cols=6
)

table2.style = "Table Grid"

headers2 = [
    "Strategy",
    "ACCEPT",
    "REVIEW",
    "DEFER",
    "Coverage",
    "Accepted Error"
]

for i, header in enumerate(headers2):

    table2.rows[0].cells[i].text = header

for _, row in routing_metrics_df.iterrows():

    cells = table2.add_row().cells

    cells[0].text = str(
        row["Strategy"]
    )

    cells[1].text = str(
        row["ACCEPT"]
    )

    cells[2].text = str(
        row["REVIEW"]
    )

    cells[3].text = str(
        row["DEFER"]
    )

    cells[4].text = fmt(
        row["Coverage"]
    )

    cells[5].text = fmt(
        row["Accepted_Error"]
    )


doc.add_paragraph(
    f"Routing changed for {routing_changes} test cases."
)

doc.add_paragraph(
    f"High-confidence / low-ERS cases identified: "
    f"{high_conf_low_ers_count}"
)


doc.add_heading(
    "Graphs",
    level=1
)

graphs = [
    (
        "Model Performance",
        "model_performance.png"
    ),
    (
        "SHAP Feature Importance",
        "shap_importance.png"
    ),
    (
        "Routing Distribution",
        "routing_distribution.png"
    ),
    (
        "Predictive Confidence vs Explanation Reliability",
        "confidence_vs_ers.png"
    ),
    (
        "Coverage vs Accepted Error",
        "coverage_vs_error.png"
    )
]

for heading, filename in graphs:

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    if os.path.exists(path):

        doc.add_heading(
            heading,
            level=2
        )

        doc.add_picture(
            path,
            width=Inches(6.2)
        )


doc.add_heading(
    "Limitations",
    level=1
)

limitations = [
    "The current experiment uses the MIMIC-IV DEMO dataset.",
    "The number of mortality-positive cases is very small.",
    "Performance estimates may therefore be unstable.",
    "The current ERS is a research prototype requiring further validation.",
    "Prototype routing thresholds and weights should be selected using validation data in the final study.",
    "The results should not be interpreted as evidence for clinical deployment."
]

for item in limitations:

    doc.add_paragraph(
        item,
        style="List Bullet"
    )


doc.add_heading(
    "Future Work",
    level=1
)

future_work = [
    "Validate the framework on the full MIMIC-IV dataset.",
    "Perform temporal validation.",
    "Perform external validation using eICU-CRD.",
    "Further formalize and validate the Explanation Reliability Score.",
    "Freeze routing thresholds using validation data before final testing.",
    "Add confidence intervals and statistical comparisons.",
    "Evaluate out-of-distribution and domain-shift cases."
]

for item in future_work:

    doc.add_paragraph(
        item,
        style="List Bullet"
    )


doc.add_heading(
    "Conclusion",
    level=1
)

doc.add_paragraph(
    "MedTrust AI demonstrates a reliability-aware clinical "
    "decision-routing concept in which calibrated predictive "
    "confidence is complemented by explanation reliability. "
    "The framework provides ACCEPT, REVIEW and DEFER routes "
    "instead of relying exclusively on model confidence."
)

doc.add_paragraph(
    "The current results are a methodological demonstration "
    "using the available MIMIC-IV demo dataset. Full MIMIC-IV "
    "and external eICU validation are required before clinical "
    "conclusions can be made."
)


doc_path = os.path.join(
    OUTPUT_DIR,
    "MedTrust_AI_Final_Research_Report.docx"
)

doc.save(
    doc_path
)


# ================================================================
# CREATE PDF REPORT
# ================================================================

try:
    from generate_pdf import build_pdf
    pdf_path = build_pdf(OUTPUT_DIR)
except Exception as pdf_err:
    pdf_path = None
    print(f"\n[Notice] Could not generate PDF report automatically: {pdf_err}")


# ================================================================
# FINISHED
# ================================================================

print("\n")
print("=" * 70)
print("MEDTRUST AI FINAL PIPELINE COMPLETED")
print("=" * 70)

print("\nOutput folder:")
print(OUTPUT_DIR)

print("\nGenerated files:")

for filename in sorted(
    os.listdir(OUTPUT_DIR)
):

    print(
        " -",
        filename
    )

print("\nFINAL SUMMARY")
print("-" * 70)

print(
    "Test cases:",
    len(patient_results)
)

print(
    "Average predictive confidence:",
    round(
        float(
            np.mean(
                test_confidence
            )
        ),
        4
    )
)

print(
    "Average explanation reliability:",
    round(
        float(
            np.mean(
                ERS_values
            )
        ),
        4
    )
)

print(
    "Routing changes:",
    routing_changes
)

print(
    "High-confidence / low-ERS cases:",
    high_conf_low_ers_count
)

print("\nResearch reports:")
print(" - Word (.docx):", doc_path)
if pdf_path and os.path.exists(pdf_path):
    print(" - PDF (.pdf):  ", pdf_path)

print("\nDONE.")