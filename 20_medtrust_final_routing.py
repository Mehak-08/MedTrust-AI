# ============================================================
# MEDTRUST AI
# STEP 20 - FINAL RELIABILITY-AWARE ROUTING
# ============================================================

import os
import pandas as pd
import numpy as np


print("\n==========================================")
print("MEDTRUST AI - FINAL ROUTING")
print("==========================================\n")


# ------------------------------------------------------------
# 1. PATH
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ------------------------------------------------------------
# 2. LOAD CALIBRATED PREDICTIONS
# ------------------------------------------------------------

test_predictions = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "step18_test_predictions.csv"
    )
)


# ------------------------------------------------------------
# 3. LOAD EXPLANATION RELIABILITY
# ------------------------------------------------------------

ers_data = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "step19_explanation_reliability.csv"
    )
)


print(
    "Prediction records:",
    len(test_predictions)
)

print(
    "ERS records:",
    len(ers_data)
)


# ------------------------------------------------------------
# 4. MERGE USING STAY ID
# ------------------------------------------------------------

if "stay_id" not in test_predictions.columns:

    raise ValueError(
        "stay_id missing from prediction file."
    )


if "stay_id" not in ers_data.columns:

    raise ValueError(
        "stay_id missing from ERS file."
    )


df = pd.merge(
    test_predictions,
    ers_data[
        [
            "stay_id",
            "ERS",
            "stability",
            "faithfulness",
            "ranking_consistency",
            "sign_consistency"
        ]
    ],
    on="stay_id",
    how="inner"
)


print(
    "Merged records:",
    len(df)
)


if len(df) == 0:

    raise ValueError(
        "No records matched between predictions and ERS."
    )


# ------------------------------------------------------------
# 5. CHECK DUPLICATES
# ------------------------------------------------------------

if df["stay_id"].duplicated().any():

    raise ValueError(
        "Duplicate stay_id found after merging."
    )


# ------------------------------------------------------------
# 6. CONVERT VALUES
# ------------------------------------------------------------

df[
    "calibrated_probability"
] = pd.to_numeric(
    df[
        "calibrated_probability"
    ],
    errors="coerce"
)


df["ERS"] = pd.to_numeric(
    df["ERS"],
    errors="coerce"
)


df = df.dropna(
    subset=[
        "calibrated_probability",
        "ERS"
    ]
).copy()


# ------------------------------------------------------------
# 7. PREDICTIVE CONFIDENCE
# ------------------------------------------------------------

df[
    "predictive_confidence"
] = np.maximum(
    df[
        "calibrated_probability"
    ],
    1 -
    df[
        "calibrated_probability"
    ]
)


# ------------------------------------------------------------
# 8. JOINT RELIABILITY
# ------------------------------------------------------------

# Prototype equal weighting.
#
# Final research version:
# weights should be selected using validation data.

df[
    "joint_reliability"
] = (
    0.50
    *
    df[
        "predictive_confidence"
    ]
    +
    0.50
    *
    df["ERS"]
)


# ------------------------------------------------------------
# 9. PROTOTYPE ROUTING THRESHOLDS
# ------------------------------------------------------------

ACCEPT_THRESHOLD = 0.70
REVIEW_THRESHOLD = 0.50


print("\n==========================================")
print("ROUTING THRESHOLDS")
print("==========================================")

print(
    "ACCEPT:",
    ACCEPT_THRESHOLD
)

print(
    "REVIEW:",
    REVIEW_THRESHOLD
)

print(
    "DEFER: <",
    REVIEW_THRESHOLD
)


# ------------------------------------------------------------
# 10. ROUTING FUNCTION
# ------------------------------------------------------------

def route_prediction(
    reliability
):

    if reliability >= ACCEPT_THRESHOLD:

        return "ACCEPT"

    elif reliability >= REVIEW_THRESHOLD:

        return "REVIEW"

    else:

        return "DEFER"


df[
    "medtrust_decision"
] = df[
    "joint_reliability"
].apply(
    route_prediction
)


# ------------------------------------------------------------
# 11. CONFIDENCE-ONLY ROUTING
# ------------------------------------------------------------

df[
    "confidence_only_decision"
] = df[
    "predictive_confidence"
].apply(
    route_prediction
)


# ------------------------------------------------------------
# 12. ROUTING DISTRIBUTION
# ------------------------------------------------------------

routes = [
    "ACCEPT",
    "REVIEW",
    "DEFER"
]


medtrust_counts = (
    df[
        "medtrust_decision"
    ]
    .value_counts()
    .reindex(
        routes,
        fill_value=0
    )
)


confidence_counts = (
    df[
        "confidence_only_decision"
    ]
    .value_counts()
    .reindex(
        routes,
        fill_value=0
    )
)


print("\n==========================================")
print("MEDTRUST ROUTING")
print("==========================================")


for route in routes:

    count = medtrust_counts[
        route
    ]

    percentage = (
        count
        /
        len(df)
        *
        100
    )

    print(
        f"{route}: "
        f"{count} "
        f"({percentage:.2f}%)"
    )


print("\n==========================================")
print("CONFIDENCE-ONLY ROUTING")
print("==========================================")


for route in routes:

    count = confidence_counts[
        route
    ]

    percentage = (
        count
        /
        len(df)
        *
        100
    )

    print(
        f"{route}: "
        f"{count} "
        f"({percentage:.2f}%)"
    )


# ------------------------------------------------------------
# 13. ROUTING CHANGES
# ------------------------------------------------------------

df[
    "routing_changed"
] = (
    df[
        "confidence_only_decision"
    ]
    !=
    df[
        "medtrust_decision"
    ]
)


changed = int(
    df[
        "routing_changed"
    ].sum()
)


changed_percentage = (
    changed
    /
    len(df)
    *
    100
)


print("\n==========================================")
print("ROUTING CHANGES")
print("==========================================")


print(
    "Changed:",
    changed
)

print(
    "Percentage:",
    f"{changed_percentage:.2f}%"
)


# ------------------------------------------------------------
# 14. ACCEPTED ERROR RATE
# ------------------------------------------------------------

TARGET = "icu_mortality"


def accepted_error_rate(
    data,
    decision_column
):

    accepted = data[
        data[
            decision_column
        ] == "ACCEPT"
    ]

    if len(accepted) == 0:

        return np.nan

    errors = (
        accepted[TARGET] == 1
    ).sum()

    return (
        errors
        /
        len(accepted)
    )


medtrust_error = (
    accepted_error_rate(
        df,
        "medtrust_decision"
    )
)


confidence_error = (
    accepted_error_rate(
        df,
        "confidence_only_decision"
    )
)


# ------------------------------------------------------------
# 15. COVERAGE
# ------------------------------------------------------------

medtrust_coverage = (
    medtrust_counts["ACCEPT"]
    /
    len(df)
)


confidence_coverage = (
    confidence_counts["ACCEPT"]
    /
    len(df)
)


print("\n==========================================")
print("SELECTIVE PERFORMANCE")
print("==========================================")


print(
    "MedTrust accepted coverage:",
    f"{medtrust_coverage:.4f}"
)

print(
    "Confidence-only coverage:",
    f"{confidence_coverage:.4f}"
)


print(
    "MedTrust accepted error:",
    medtrust_error
)

print(
    "Confidence-only accepted error:",
    confidence_error
)


# ------------------------------------------------------------
# 16. HIGH CONFIDENCE / LOW ERS MISMATCH
# ------------------------------------------------------------

high_confidence = (
    df[
        "predictive_confidence"
    ]
    >= 0.85
)


low_ers = (
    df["ERS"]
    < 0.50
)


mismatch = (
    high_confidence
    &
    low_ers
)


mismatch_count = int(
    mismatch.sum()
)


print("\n==========================================")
print("CONFIDENCE / ERS MISMATCH")
print("==========================================")


print(
    "High-confidence / low-ERS cases:",
    mismatch_count
)


# ------------------------------------------------------------
# 17. SAVE PATIENT-LEVEL RESULTS
# ------------------------------------------------------------

output_columns = [
    "subject_id",
    "hadm_id",
    "stay_id",
    TARGET,
    "calibrated_probability",
    "predictive_confidence",
    "stability",
    "faithfulness",
    "ranking_consistency",
    "sign_consistency",
    "ERS",
    "joint_reliability",
    "confidence_only_decision",
    "medtrust_decision",
    "routing_changed"
]


output_columns = [
    col
    for col in output_columns
    if col in df.columns
]


df[
    output_columns
].to_csv(
    os.path.join(
        BASE_DIR,
        "step20_medtrust_routing.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# 18. SAVE ROUTING SUMMARY
# ------------------------------------------------------------

summary = pd.DataFrame({

    "Decision": routes,

    "Confidence_Only_Count": [
        confidence_counts[r]
        for r in routes
    ],

    "MedTrust_Count": [
        medtrust_counts[r]
        for r in routes
    ],

    "Confidence_Only_Percentage": [
        confidence_counts[r]
        /
        len(df)
        *
        100
        for r in routes
    ],

    "MedTrust_Percentage": [
        medtrust_counts[r]
        /
        len(df)
        *
        100
        for r in routes
    ]
})


summary.to_csv(
    os.path.join(
        BASE_DIR,
        "step20_routing_summary.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# 19. SAVE METRICS
# ------------------------------------------------------------

metrics = pd.DataFrame({

    "Metric": [

        "Total test cases",

        "MedTrust ACCEPT coverage",

        "Confidence-only ACCEPT coverage",

        "MedTrust accepted error",

        "Confidence-only accepted error",

        "Routing changes",

        "Routing changes percentage",

        "High-confidence low-ERS mismatch"
    ],

    "Value": [

        len(df),

        medtrust_coverage,

        confidence_coverage,

        medtrust_error,

        confidence_error,

        changed,

        changed_percentage,

        mismatch_count
    ]
})


metrics.to_csv(
    os.path.join(
        BASE_DIR,
        "step20_routing_metrics.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# 20. SAVE REVIEW CASES
# ------------------------------------------------------------

review_cases = df[
    df[
        "medtrust_decision"
    ] != "ACCEPT"
].copy()


review_cases.to_csv(
    os.path.join(
        BASE_DIR,
        "step20_review_defer_cases.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# 21. FINAL SUMMARY
# ------------------------------------------------------------

print("\n==========================================")
print("STEP 20 COMPLETE")
print("==========================================")

print(
    "\nAverage predictive confidence:",
    f"{df['predictive_confidence'].mean():.4f}"
)

print(
    "Average ERS:",
    f"{df['ERS'].mean():.4f}"
)

print(
    "Average joint reliability:",
    f"{df['joint_reliability'].mean():.4f}"
)


print("\nCreated:")

print(
    "step20_medtrust_routing.csv"
)

print(
    "step20_routing_summary.csv"
)

print(
    "step20_routing_metrics.csv"
)

print(
    "step20_review_defer_cases.csv"
)


print("\nIMPORTANT:")
print(
    "Thresholds are currently prototype thresholds."
)

print(
    "They must be selected using validation data "
    "before final test evaluation."
)

print(
    "Current results are from the MIMIC-IV DEMO dataset."
)

print("\n==========================================")