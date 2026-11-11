# ============================================================
# MEDTRUST AI
# STEP 19 - EXPLANATION RELIABILITY SCORE (ERS)
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


print("\n==========================================")
print("MEDTRUST AI - EXPLANATION RELIABILITY")
print("==========================================\n")


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

train = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "train_patient_level.csv"
    )
)

test = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "test_patient_level.csv"
    )
)


TARGET = "icu_mortality"

DROP_COLUMNS = [
    "subject_id",
    "hadm_id",
    "stay_id",
    TARGET
]


features = [
    col
    for col in train.columns
    if col not in DROP_COLUMNS
]


X_train = train[features].copy()
X_test = test[features].copy()

y_train = train[TARGET].astype(int)
y_test = test[TARGET].astype(int)


# ------------------------------------------------------------
# 3. IMPUTE USING TRAINING MEDIANS ONLY
# ------------------------------------------------------------

train_medians = X_train.median(
    numeric_only=True
)


X_train = X_train.fillna(
    train_medians
)

X_test = X_test.fillna(
    train_medians
)


X_train = X_train.fillna(0)
X_test = X_test.fillna(0)


# ------------------------------------------------------------
# 4. TRAIN XGBOOST
# ------------------------------------------------------------

print("Training XGBoost...")


model = XGBClassifier(
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


model.fit(
    X_train,
    y_train
)


print("Model trained.")


# ------------------------------------------------------------
# 5. ORIGINAL PREDICTIONS
# ------------------------------------------------------------

original_probability = (
    model.predict_proba(X_test)[:, 1]
)


# ------------------------------------------------------------
# 6. TREE FEATURE IMPORTANCE
# ------------------------------------------------------------

feature_importance = (
    model.feature_importances_
)


importance_df = pd.DataFrame({

    "feature": features,

    "importance": feature_importance
})


importance_df = (
    importance_df
    .sort_values(
        "importance",
        ascending=False
    )
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# 7. NORMALIZED FEATURE IMPORTANCE
# ------------------------------------------------------------

total_importance = (
    importance_df["importance"]
    .sum()
)


if total_importance > 0:

    importance_df[
        "normalized_importance"
    ] = (
        importance_df["importance"]
        /
        total_importance
    )

else:

    importance_df[
        "normalized_importance"
    ] = 0


# ------------------------------------------------------------
# 8. TOP FEATURES
# ------------------------------------------------------------

TOP_K = min(
    10,
    len(importance_df)
)


top_features = (
    importance_df
    .head(TOP_K)
    ["feature"]
    .tolist()
)


print("\nTop explanation features:")

for feature in top_features:

    print(
        "-",
        feature
    )


# ------------------------------------------------------------
# 9. EXPLANATION STABILITY
# ------------------------------------------------------------

print(
    "\nCalculating explanation stability..."
)


stability_scores = []


for index in range(len(X_test)):

    row = X_test.iloc[
        index
    ].copy()


    perturbation = (
        np.random.default_rng(
            42 + index
        )
        .normal(
            0,
            0.05,
            len(features)
        )
    )


    perturbed = row.copy()


    for i, feature in enumerate(features):

        scale = (
            train[feature]
            .std()
        )

        if pd.isna(scale) or scale == 0:

            scale = 1.0


        perturbed[
            feature
        ] = (
            row[feature]
            +
            perturbation[i] * scale
        )


    original_values = (
        row.values.astype(float)
    )

    perturbed_values = (
        perturbed.values.astype(float)
    )


    relative_change = (
        np.mean(
            np.abs(
                perturbed_values
                -
                original_values
            )
            /
            (
                np.abs(
                    original_values
                )
                + 1e-6
            )
        )
    )


    stability = (
        1.0
        /
        (
            1.0
            +
            relative_change
        )
    )


    stability = float(
        np.clip(
            stability,
            0,
            1
        )
    )


    stability_scores.append(
        stability
    )


# ------------------------------------------------------------
# 10. FAITHFULNESS
# ------------------------------------------------------------

print(
    "Calculating explanation faithfulness..."
)


faithfulness_scores = []


for index in range(len(X_test)):

    row = X_test.iloc[
        index
    ].copy()


    original_prediction = (
        original_probability[index]
    )


    effects = []


    for feature in top_features:

        modified = row.copy()

        modified[
            feature
        ] = train_medians.get(
            feature,
            0
        )


        modified_probability = (
            model.predict_proba(
                pd.DataFrame(
                    [modified],
                    columns=features
                )
            )[0, 1]
        )


        effect = abs(
            original_prediction
            -
            modified_probability
        )


        effects.append(
            effect
        )


    if len(effects) == 0:

        faithfulness = 0.0

    else:

        mean_effect = np.mean(
            effects
        )

        max_effect = np.max(
            effects
        )

        if max_effect > 0:

            faithfulness = (
                mean_effect
                /
                max_effect
            )

        else:

            faithfulness = 0.0


    faithfulness_scores.append(
        float(
            np.clip(
                faithfulness,
                0,
                1
            )
        )
    )


# ------------------------------------------------------------
# 11. RANKING CONSISTENCY
# ------------------------------------------------------------

print(
    "Calculating ranking consistency..."
)


ranking_scores = []


for index in range(len(X_test)):

    row = X_test.iloc[
        index
    ].copy()


    effects = []


    for feature in top_features:

        modified = row.copy()

        modified[
            feature
        ] = train_medians.get(
            feature,
            0
        )


        modified_probability = (
            model.predict_proba(
                pd.DataFrame(
                    [modified],
                    columns=features
                )
            )[0, 1]
        )


        effects.append(
            abs(
                original_probability[index]
                -
                modified_probability
            )
        )


    if len(effects) <= 1:

        ranking = 1.0

    else:

        original_rank = np.argsort(
            np.argsort(
                -np.array(effects)
            )
        )


        # Small controlled perturbation
        perturbed_effects = (
            np.array(effects)
            +
            np.random.default_rng(
                100 + index
            ).normal(
                0,
                0.01,
                len(effects)
            )
        )


        perturbed_rank = np.argsort(
            np.argsort(
                -perturbed_effects
            )
        )


        ranking = 1.0 - (
            np.mean(
                np.abs(
                    original_rank
                    -
                    perturbed_rank
                )
            )
            /
            max(
                1,
                len(effects) - 1
            )
        )


    ranking_scores.append(
        float(
            np.clip(
                ranking,
                0,
                1
            )
        )
    )


# ------------------------------------------------------------
# 12. SIGN CONSISTENCY
# ------------------------------------------------------------

print(
    "Calculating sign consistency..."
)


sign_scores = []


for index in range(len(X_test)):

    row = X_test.iloc[
        index
    ].copy()


    signs = []


    for feature in top_features:

        modified = row.copy()

        modified[
            feature
        ] = train_medians.get(
            feature,
            0
        )


        modified_probability = (
            model.predict_proba(
                pd.DataFrame(
                    [modified],
                    columns=features
                )
            )[0, 1]
        )


        difference = (
            original_probability[index]
            -
            modified_probability
        )


        if difference > 0:

            signs.append(1)

        elif difference < 0:

            signs.append(-1)

        else:

            signs.append(0)


    nonzero = [
        value
        for value in signs
        if value != 0
    ]


    if len(nonzero) == 0:

        consistency = 1.0

    else:

        majority = (
            1
            if sum(nonzero) >= 0
            else -1
        )


        consistency = (
            sum(
                value == majority
                for value in nonzero
            )
            /
            len(nonzero)
        )


    sign_scores.append(
        float(
            np.clip(
                consistency,
                0,
                1
            )
        )
    )


# ------------------------------------------------------------
# 13. FINAL ERS
# ------------------------------------------------------------

stability_scores = np.array(
    stability_scores
)

faithfulness_scores = np.array(
    faithfulness_scores
)

ranking_scores = np.array(
    ranking_scores
)

sign_scores = np.array(
    sign_scores
)


ERS = (
    0.25 * stability_scores
    +
    0.25 * faithfulness_scores
    +
    0.25 * ranking_scores
    +
    0.25 * sign_scores
)


ERS = np.clip(
    ERS,
    0,
    1
)


# ------------------------------------------------------------
# 14. SAVE PATIENT-LEVEL ERS
# ------------------------------------------------------------

output = test[
    [
        "subject_id",
        "hadm_id",
        "stay_id",
        TARGET
    ]
].copy()


output[
    "prediction_probability"
] = original_probability


output[
    "stability"
] = stability_scores


output[
    "faithfulness"
] = faithfulness_scores


output[
    "ranking_consistency"
] = ranking_scores


output[
    "sign_consistency"
] = sign_scores


output[
    "ERS"
] = ERS


output_path = os.path.join(
    BASE_DIR,
    "step19_explanation_reliability.csv"
)


output.to_csv(
    output_path,
    index=False
)


# ------------------------------------------------------------
# 15. SAVE FEATURE IMPORTANCE
# ------------------------------------------------------------

importance_path = os.path.join(
    BASE_DIR,
    "step19_feature_importance.csv"
)


importance_df.to_csv(
    importance_path,
    index=False
)


# ------------------------------------------------------------
# 16. PRINT RESULTS
# ------------------------------------------------------------

print("\n==========================================")
print("EXPLANATION RELIABILITY RESULTS")
print("==========================================")

print(
    "\nAverage stability:",
    f"{stability_scores.mean():.4f}"
)

print(
    "Average faithfulness:",
    f"{faithfulness_scores.mean():.4f}"
)

print(
    "Average ranking consistency:",
    f"{ranking_scores.mean():.4f}"
)

print(
    "Average sign consistency:",
    f"{sign_scores.mean():.4f}"
)

print(
    "Average ERS:",
    f"{ERS.mean():.4f}"
)

print(
    "Minimum ERS:",
    f"{ERS.min():.4f}"
)

print(
    "Maximum ERS:",
    f"{ERS.max():.4f}"
)


# ------------------------------------------------------------
# 17. FINAL FILES
# ------------------------------------------------------------

print("\n==========================================")
print("STEP 19 COMPLETE")
print("==========================================")

print(
    "\nCreated:"
)

print(
    "step19_explanation_reliability.csv"
)

print(
    "step19_feature_importance.csv"
)

print(
    "\nERS now combines:"
)

print(
    "1. Stability"
)

print(
    "2. Faithfulness"
)

print(
    "3. Ranking consistency"
)

print(
    "4. Sign consistency"
)

print(
    "\nIMPORTANT:"
)

print(
    "This is still a prototype ERS implementation "
    "on the MIMIC-IV DEMO dataset."
)

print(
    "Final research thresholds and weighting "
    "must be selected using validation data."
)

print("\n==========================================")