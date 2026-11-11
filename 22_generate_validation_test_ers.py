# ============================================================
# MEDTRUST AI
# STEP 22 - GENERATE VALIDATION + TEST ERS
# ============================================================
#
# Purpose:
#   Generate Explanation Reliability Score (ERS) separately
#   for validation and test data.
#
# ERS components:
#   1. Stability
#   2. Faithfulness
#   3. Ranking consistency
#   4. Sign consistency
#
# IMPORTANT:
#   Validation and test ERS are calculated independently.
#   Test data is NOT used to select thresholds.
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd

from xgboost import XGBClassifier


warnings.filterwarnings("ignore")


print("\n==========================================")
print("MEDTRUST AI - VALIDATION + TEST ERS")
print("==========================================\n")


# ============================================================
# 1. PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# 2. LOAD DATA
# ============================================================

train_path = os.path.join(
    BASE_DIR,
    "train_patient_level.csv"
)

validation_path = os.path.join(
    BASE_DIR,
    "validation_patient_level.csv"
)

test_path = os.path.join(
    BASE_DIR,
    "test_patient_level.csv"
)


train_df = pd.read_csv(train_path)
validation_df = pd.read_csv(validation_path)
test_df = pd.read_csv(test_path)


print("Train shape:", train_df.shape)
print("Validation shape:", validation_df.shape)
print("Test shape:", test_df.shape)


# ============================================================
# 3. TARGET
# ============================================================

TARGET = "icu_mortality"


# ============================================================
# 4. REMOVE NON-FEATURE COLUMNS
# ============================================================

DROP_COLUMNS = [
    "subject_id",
    "hadm_id",
    "stay_id",
    TARGET
]


feature_columns = [
    col
    for col in train_df.columns
    if col not in DROP_COLUMNS
]


print(
    "\nNumber of model features:",
    len(feature_columns)
)


# ============================================================
# 5. PREPARE X AND y
# ============================================================

X_train = train_df[
    feature_columns
].copy()

y_train = train_df[
    TARGET
].astype(int)


X_validation = validation_df[
    feature_columns
].copy()

y_validation = validation_df[
    TARGET
].astype(int)


X_test = test_df[
    feature_columns
].copy()

y_test = test_df[
    TARGET
].astype(int)


# ============================================================
# 6. NUMERIC CONVERSION
# ============================================================

for col in feature_columns:

    X_train[col] = pd.to_numeric(
        X_train[col],
        errors="coerce"
    )

    X_validation[col] = pd.to_numeric(
        X_validation[col],
        errors="coerce"
    )

    X_test[col] = pd.to_numeric(
        X_test[col],
        errors="coerce"
    )


# ============================================================
# 7. TRAIN-ONLY MEDIAN IMPUTATION
# ============================================================

train_medians = X_train.median(
    numeric_only=True
)


X_train = X_train.fillna(
    train_medians
)

X_validation = X_validation.fillna(
    train_medians
)

X_test = X_test.fillna(
    train_medians
)


# Any remaining missing values become 0.
# This is only a safety fallback.

X_train = X_train.fillna(0)
X_validation = X_validation.fillna(0)
X_test = X_test.fillna(0)


# ============================================================
# 8. TRAIN XGBOOST
# ============================================================

print("\nTraining XGBoost...")


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


# ============================================================
# 9. PREDICTIONS
# ============================================================

validation_prob = model.predict_proba(
    X_validation
)[:, 1]


test_prob = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================

importance = pd.Series(
    model.feature_importances_,
    index=feature_columns
).sort_values(
    ascending=False
)


TOP_N = min(
    10,
    len(importance)
)


top_features = (
    importance
    .head(TOP_N)
    .index
    .tolist()
)


print("\nTop explanation features:")

for feature in top_features:

    print(
        f"- {feature}"
    )


# ============================================================
# 11. TRAINING STANDARD DEVIATION
# ============================================================

train_std = X_train.std(
    numeric_only=True
)

train_std = train_std.replace(
    0,
    1e-8
)


# ============================================================
# 12. TRAINING MEDIANS FOR FAITHFULNESS
# ============================================================

train_median = X_train.median(
    numeric_only=True
)


# ============================================================
# 13. STABILITY FUNCTION
# ============================================================

def calculate_stability(
    X_data,
    original_probabilities
):

    scores = []

    rng = np.random.RandomState(42)

    for index in range(
        len(X_data)
    ):

        row = X_data.iloc[
            index
        ].copy()

        original_probability = (
            original_probabilities[
                index
            ]
        )

        perturbation = (
            rng.normal(
                0,
                0.05,
                len(feature_columns)
            )
            *
            train_std[
                feature_columns
            ].values
        )

        perturbed_row = (
            row.values
            +
            perturbation
        )

        perturbed_df = pd.DataFrame(
            [perturbed_row],
            columns=feature_columns
        )

        perturbed_probability = (
            model.predict_proba(
                perturbed_df
            )[0, 1]
        )

        probability_change = abs(
            perturbed_probability
            -
            original_probability
        )

        relative_change = (
            probability_change
            /
            (
                abs(
                    original_probability
                )
                +
                1e-8
            )
        )

        stability = (
            1
            /
            (
                1
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

        scores.append(
            stability
        )

    return np.array(
        scores
    )


# ============================================================
# 14. FAITHFULNESS FUNCTION
# ============================================================

def calculate_faithfulness(
    X_data,
    original_probabilities
):

    scores = []

    for index in range(
        len(X_data)
    ):

        row = X_data.iloc[
            index
        ].copy()

        original_probability = (
            original_probabilities[
                index
            ]
        )

        effects = []

        for feature in top_features:

            modified_row = row.copy()

            modified_row[
                feature
            ] = train_median[
                feature
            ]

            modified_df = pd.DataFrame(
                [modified_row],
                columns=feature_columns
            )

            modified_probability = (
                model.predict_proba(
                    modified_df
                )[0, 1]
            )

            effect = abs(
                original_probability
                -
                modified_probability
            )

            effects.append(
                effect
            )

        effects = np.array(
            effects
        )

        if (
            len(effects) == 0
            or
            effects.max() == 0
        ):

            faithfulness = 0.0

        else:

            faithfulness = (
                effects.mean()
                /
                effects.max()
            )

        faithfulness = float(
            np.clip(
                faithfulness,
                0,
                1
            )
        )

        scores.append(
            faithfulness
        )

    return np.array(
        scores
    )


# ============================================================
# 15. RANKING CONSISTENCY
# ============================================================

def calculate_ranking_consistency(
    X_data
):

    scores = []

    rng = np.random.RandomState(42)

    for index in range(
        len(X_data)
    ):

        row = X_data.iloc[
            index
        ].copy()

        effects = []

        for feature in top_features:

            modified_row = row.copy()

            modified_row[
                feature
            ] = train_median[
                feature
            ]

            modified_df = pd.DataFrame(
                [modified_row],
                columns=feature_columns
            )

            original_probability = (
                model.predict_proba(
                    pd.DataFrame(
                        [row],
                        columns=feature_columns
                    )
                )[0, 1]
            )

            modified_probability = (
                model.predict_proba(
                    modified_df
                )[0, 1]
            )

            effects.append(
                abs(
                    original_probability
                    -
                    modified_probability
                )
            )

        effects = np.array(
            effects
        )

        original_rank = np.argsort(
            -effects
        )

        noisy_effects = (
            effects
            +
            rng.normal(
                0,
                0.001,
                len(effects)
            )
        )

        noisy_rank = np.argsort(
            -noisy_effects
        )

        if len(
            original_rank
        ) <= 1:

            consistency = 1.0

        else:

            position_difference = (
                np.abs(
                    original_rank
                    -
                    noisy_rank
                )
            )

            max_difference = (
                len(original_rank)
                *
                len(original_rank)
            )

            consistency = (
                1
                -
                position_difference.mean()
                /
                max_difference
            )

        consistency = float(
            np.clip(
                consistency,
                0,
                1
            )
        )

        scores.append(
            consistency
        )

    return np.array(
        scores
    )


# ============================================================
# 16. SIGN CONSISTENCY
# ============================================================

def calculate_sign_consistency(
    X_data
):

    scores = []

    for index in range(
        len(X_data)
    ):

        row = X_data.iloc[
            index
        ].copy()

        signs = []

        original_probability = (
            model.predict_proba(
                pd.DataFrame(
                    [row],
                    columns=feature_columns
                )
            )[0, 1]
        )

        for feature in top_features:

            modified_row = row.copy()

            modified_row[
                feature
            ] = train_median[
                feature
            ]

            modified_df = pd.DataFrame(
                [modified_row],
                columns=feature_columns
            )

            modified_probability = (
                model.predict_proba(
                    modified_df
                )[0, 1]
            )

            difference = (
                original_probability
                -
                modified_probability
            )

            if difference > 0:

                signs.append(1)

            elif difference < 0:

                signs.append(-1)

            else:

                signs.append(0)

        signs = np.array(
            signs
        )

        non_zero = signs[
            signs != 0
        ]

        if len(non_zero) == 0:

            consistency = 1.0

        else:

            positive = (
                non_zero > 0
            ).sum()

            negative = (
                non_zero < 0
            ).sum()

            majority = max(
                positive,
                negative
            )

            consistency = (
                majority
                /
                len(non_zero)
            )

        consistency = float(
            np.clip(
                consistency,
                0,
                1
            )
        )

        scores.append(
            consistency
        )

    return np.array(
        scores
    )


# ============================================================
# 17. GENERATE ERS FOR ONE DATASET
# ============================================================

def generate_ers(
    X_data,
    probabilities,
    dataset_name
):

    print(
        f"\nCalculating ERS for {dataset_name}..."
    )

    print(
        "  - Stability..."
    )

    stability = calculate_stability(
        X_data,
        probabilities
    )

    print(
        "  - Faithfulness..."
    )

    faithfulness = calculate_faithfulness(
        X_data,
        probabilities
    )

    print(
        "  - Ranking consistency..."
    )

    ranking = calculate_ranking_consistency(
        X_data
    )

    print(
        "  - Sign consistency..."
    )

    sign = calculate_sign_consistency(
        X_data
    )

    ers = (
        0.25 * stability
        +
        0.25 * faithfulness
        +
        0.25 * ranking
        +
        0.25 * sign
    )

    return pd.DataFrame({

        "predictive_probability":
            probabilities,

        "stability":
            stability,

        "faithfulness":
            faithfulness,

        "ranking_consistency":
            ranking,

        "sign_consistency":
            sign,

        "ERS":
            ers
    })


# ============================================================
# 18. GENERATE VALIDATION ERS
# ============================================================

validation_ers = generate_ers(
    X_validation,
    validation_prob,
    "VALIDATION"
)


# ============================================================
# 19. GENERATE TEST ERS
# ============================================================

test_ers = generate_ers(
    X_test,
    test_prob,
    "TEST"
)


# ============================================================
# 20. ADD IDENTIFIERS
# ============================================================

validation_ers.insert(
    0,
    "subject_id",
    validation_df[
        "subject_id"
    ].values
)

validation_ers.insert(
    1,
    "hadm_id",
    validation_df[
        "hadm_id"
    ].values
)

validation_ers.insert(
    2,
    "stay_id",
    validation_df[
        "stay_id"
    ].values
)

validation_ers.insert(
    3,
    TARGET,
    y_validation.values
)


test_ers.insert(
    0,
    "subject_id",
    test_df[
        "subject_id"
    ].values
)

test_ers.insert(
    1,
    "hadm_id",
    test_df[
        "hadm_id"
    ].values
)

test_ers.insert(
    2,
    "stay_id",
    test_df[
        "stay_id"
    ].values
)

test_ers.insert(
    3,
    TARGET,
    y_test.values
)


# ============================================================
# 21. ADD PREDICTIVE CONFIDENCE
# ============================================================

validation_ers[
    "predictive_confidence"
] = np.maximum(
    validation_ers[
        "predictive_probability"
    ],
    1 -
    validation_ers[
        "predictive_probability"
    ]
)


test_ers[
    "predictive_confidence"
] = np.maximum(
    test_ers[
        "predictive_probability"
    ],
    1 -
    test_ers[
        "predictive_probability"
    ]
)


# ============================================================
# 22. JOINT RELIABILITY
# ============================================================

validation_ers[
    "joint_reliability"
] = (
    0.50
    *
    validation_ers[
        "predictive_confidence"
    ]
    +
    0.50
    *
    validation_ers[
        "ERS"
    ]
)


test_ers[
    "joint_reliability"
] = (
    0.50
    *
    test_ers[
        "predictive_confidence"
    ]
    +
    0.50
    *
    test_ers[
        "ERS"
    ]
)


# ============================================================
# 23. SAVE VALIDATION ERS
# ============================================================

validation_ers.to_csv(
    os.path.join(
        BASE_DIR,
        "step22_validation_ers.csv"
    ),
    index=False
)


# ============================================================
# 24. SAVE TEST ERS
# ============================================================

test_ers.to_csv(
    os.path.join(
        BASE_DIR,
        "step22_test_ers.csv"
    ),
    index=False
)


# ============================================================
# 25. SAVE SUMMARY
# ============================================================

summary = pd.DataFrame({

    "Dataset": [
        "Validation",
        "Test"
    ],

    "Cases": [
        len(validation_ers),
        len(test_ers)
    ],

    "Average_Stability": [
        validation_ers[
            "stability"
        ].mean(),

        test_ers[
            "stability"
        ].mean()
    ],

    "Average_Faithfulness": [
        validation_ers[
            "faithfulness"
        ].mean(),

        test_ers[
            "faithfulness"
        ].mean()
    ],

    "Average_Ranking": [
        validation_ers[
            "ranking_consistency"
        ].mean(),

        test_ers[
            "ranking_consistency"
        ].mean()
    ],

    "Average_Sign": [
        validation_ers[
            "sign_consistency"
        ].mean(),

        test_ers[
            "sign_consistency"
        ].mean()
    ],

    "Average_ERS": [
        validation_ers[
            "ERS"
        ].mean(),

        test_ers[
            "ERS"
        ].mean()
    ],

    "Average_Confidence": [
        validation_ers[
            "predictive_confidence"
        ].mean(),

        test_ers[
            "predictive_confidence"
        ].mean()
    ],

    "Average_Joint_Reliability": [
        validation_ers[
            "joint_reliability"
        ].mean(),

        test_ers[
            "joint_reliability"
        ].mean()
    ]
})


summary.to_csv(
    os.path.join(
        BASE_DIR,
        "step22_ers_summary.csv"
    ),
    index=False
)


# ============================================================
# 26. DISPLAY RESULTS
# ============================================================

print("\n==========================================")
print("VALIDATION ERS RESULTS")
print("==========================================")

print(
    "Average stability:",
    f"{validation_ers['stability'].mean():.4f}"
)

print(
    "Average faithfulness:",
    f"{validation_ers['faithfulness'].mean():.4f}"
)

print(
    "Average ranking consistency:",
    f"{validation_ers['ranking_consistency'].mean():.4f}"
)

print(
    "Average sign consistency:",
    f"{validation_ers['sign_consistency'].mean():.4f}"
)

print(
    "Average ERS:",
    f"{validation_ers['ERS'].mean():.4f}"
)

print(
    "Average confidence:",
    f"{validation_ers['predictive_confidence'].mean():.4f}"
)

print(
    "Average joint reliability:",
    f"{validation_ers['joint_reliability'].mean():.4f}"
)


print("\n==========================================")
print("TEST ERS RESULTS")
print("==========================================")

print(
    "Average stability:",
    f"{test_ers['stability'].mean():.4f}"
)

print(
    "Average faithfulness:",
    f"{test_ers['faithfulness'].mean():.4f}"
)

print(
    "Average ranking consistency:",
    f"{test_ers['ranking_consistency'].mean():.4f}"
)

print(
    "Average sign consistency:",
    f"{test_ers['sign_consistency'].mean():.4f}"
)

print(
    "Average ERS:",
    f"{test_ers['ERS'].mean():.4f}"
)

print(
    "Average confidence:",
    f"{test_ers['predictive_confidence'].mean():.4f}"
)

print(
    "Average joint reliability:",
    f"{test_ers['joint_reliability'].mean():.4f}"
)

# ============================================================
# 27. COMPLETE
# ============================================================

print("\n==========================================")
print("STEP 22 COMPLETE")
print("==========================================")

print("\nCreated:")

print(
    "step22_validation_ers.csv"
)

print(
    "step22_test_ers.csv"
)

print(
    "step22_ers_summary.csv"
)

print("\nIMPORTANT:")
print(
    "Validation ERS and test ERS are now "
    "calculated separately."
)

print(
    "Test ERS is NOT used to select "
    "routing thresholds."
)

print(
    "\nNext:"
)

print(
    "Step 23 will select thresholds using "
    "validation ERS and then freeze them "
    "before evaluating the test set."
)

print("\n==========================================")