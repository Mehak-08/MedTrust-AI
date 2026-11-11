import os
import pandas as pd
import numpy as np

print("\n==========================================")
print("MEDTRUST AI - RELIABILITY ROUTING")
print("==========================================\n")

PROJECT_PATH = os.path.dirname(__file__)

# -------------------------------------------------
# LOAD STEP 11 RESULTS
# -------------------------------------------------

stability_file = os.path.join(
    PROJECT_PATH,
    "explanation_stability.csv"
)

df = pd.read_csv(stability_file)

print("Loaded patients:", len(df))


# -------------------------------------------------
# CALIBRATED CONFIDENCE
# -------------------------------------------------
# We use the calibrated XGBoost predictions
# created in Step 9.

calibrated_file = os.path.join(
    PROJECT_PATH,
    "calibrated_xgboost_predictions.csv"
)

if os.path.exists(calibrated_file):

    calibrated_df = pd.read_csv(
        calibrated_file
    )

    print(
        "Loaded calibrated predictions:",
        len(calibrated_df)
    )

    # Find probability column automatically
    probability_columns = [
        c for c in calibrated_df.columns
        if "prob" in c.lower()
    ]

    if len(probability_columns) > 0:

        calibrated_probability = (
            calibrated_df[
                probability_columns[0]
            ].values
        )

        if len(calibrated_probability) == len(df):

            df["calibrated_probability"] = (
                calibrated_probability
            )

        else:

            print(
                "\nWARNING: Calibrated prediction "
                "length does not match stability data."
            )

            df["calibrated_probability"] = (
                df["xgboost_probability"]
            )

    else:

        print(
            "\nProbability column not found."
        )

        df["calibrated_probability"] = (
            df["xgboost_probability"]
        )

else:

    print(
        "\nCalibrated prediction file not found."
    )

    print(
        "Using XGBoost probability temporarily."
    )

    df["calibrated_probability"] = (
        df["xgboost_probability"]
    )


# -------------------------------------------------
# CALIBRATED CONFIDENCE
# -------------------------------------------------

df["calibrated_confidence"] = np.maximum(
    df["calibrated_probability"],
    1 - df["calibrated_probability"]
)


# -------------------------------------------------
# EXPLANATION RELIABILITY
# -------------------------------------------------

df["explanation_reliability"] = (
    df["shap_stability"]
)


# -------------------------------------------------
# EXPLANATION RELIABILITY SCORE (ERS)
# -------------------------------------------------
#
# Current research prototype:
#
# ERS = explanation stability
#
# Later, the PRD can extend this with:
# stability + faithfulness + ranking +
# sign consistency.
#
# For this prototype we keep the score
# transparent and reproducible.

df["ERS"] = (
    df["explanation_reliability"]
)


# -------------------------------------------------
# JOINT RELIABILITY SCORE
# -------------------------------------------------
#
# This combines:
#
# 1. Predictive confidence
# 2. Explanation reliability
#
# Equal weighting is used as a transparent
# prototype rather than claiming that these
# weights are optimal.

df["joint_reliability"] = (
    0.5 * df["calibrated_confidence"]
    +
    0.5 * df["ERS"]
)


# -------------------------------------------------
# ROUTING THRESHOLDS
# -------------------------------------------------
#
# Prototype thresholds:
#
# >= 0.85  -> ACCEPT
# >= 0.70  -> REVIEW
# <  0.70  -> DEFER
#
# These are NOT final research thresholds.
# Final thresholds should be selected using
# validation data and frozen before test evaluation.

ACCEPT_THRESHOLD = 0.85
REVIEW_THRESHOLD = 0.70


def route_prediction(score):

    if score >= ACCEPT_THRESHOLD:

        return "ACCEPT"

    elif score >= REVIEW_THRESHOLD:

        return "REVIEW"

    else:

        return "DEFER"


df["routing_decision"] = (
    df["joint_reliability"]
    .apply(route_prediction)
)


# -------------------------------------------------
# PREDICTED CLASS
# -------------------------------------------------

df["predicted_class"] = (
    df["calibrated_probability"] >= 0.5
).astype(int)


# -------------------------------------------------
# CORRECT / INCORRECT
# -------------------------------------------------

df["prediction_correct"] = (
    df["predicted_class"]
    ==
    df["icu_mortality"]
)


# -------------------------------------------------
# ROUTING SUMMARY
# -------------------------------------------------

print("\n==========================================")
print("RELIABILITY SUMMARY")
print("==========================================\n")

print(
    "Average calibrated confidence:",
    round(
        df["calibrated_confidence"].mean(),
        4
    )
)

print(
    "Average explanation reliability:",
    round(
        df["explanation_reliability"].mean(),
        4
    )
)

print(
    "Average ERS:",
    round(
        df["ERS"].mean(),
        4
    )
)

print(
    "Average joint reliability:",
    round(
        df["joint_reliability"].mean(),
        4
    )
)


# -------------------------------------------------
# ROUTING COUNTS
# -------------------------------------------------

print("\n==========================================")
print("ROUTING DECISIONS")
print("==========================================\n")

routing_counts = (
    df["routing_decision"]
    .value_counts()
)

print(
    routing_counts.to_string()
)


# -------------------------------------------------
# ROUTING PERCENTAGES
# -------------------------------------------------

routing_percentages = (
    df["routing_decision"]
    .value_counts(
        normalize=True
    )
    * 100
)

print("\nRouting percentages:")

for decision, percentage in (
    routing_percentages.items()
):

    print(
        f"{decision}: "
        f"{percentage:.2f}%"
    )


# -------------------------------------------------
# ACCEPTED SET
# -------------------------------------------------

accepted = df[
    df["routing_decision"]
    ==
    "ACCEPT"
]


reviewed = df[
    df["routing_decision"]
    ==
    "REVIEW"
]


deferred = df[
    df["routing_decision"]
    ==
    "DEFER"
]


# -------------------------------------------------
# SELECTIVE RISK
# -------------------------------------------------

if len(accepted) > 0:

    accepted_error_rate = (
        1 -
        accepted["prediction_correct"].mean()
    )

else:

    accepted_error_rate = np.nan


print("\n==========================================")
print("SELECTIVE ROUTING METRICS")
print("==========================================\n")

print(
    "Total patients:",
    len(df)
)

print(
    "Accepted:",
    len(accepted)
)

print(
    "Reviewed:",
    len(reviewed)
)

print(
    "Deferred:",
    len(deferred)
)

print(
    "Acceptance coverage:",
    round(
        len(accepted) / len(df),
        4
    )
)

print(
    "Accepted error rate:",
    accepted_error_rate
)


# -------------------------------------------------
# HIGH-CONFIDENCE / LOW-ERS MISMATCH
# -------------------------------------------------

HIGH_CONFIDENCE = 0.85
LOW_ERS = 0.80

mismatch = df[
    (
        df["calibrated_confidence"]
        >= HIGH_CONFIDENCE
    )
    &
    (
        df["ERS"]
        < LOW_ERS
    )
]


print("\n==========================================")
print("HIGH-CONFIDENCE / LOW-ERS MISMATCH")
print("==========================================\n")

print(
    "Mismatch count:",
    len(mismatch)
)


# -------------------------------------------------
# SAVE COMPLETE RESULTS
# -------------------------------------------------

output_file = os.path.join(
    PROJECT_PATH,
    "medtrust_routing_results.csv"
)

df.to_csv(
    output_file,
    index=False
)


# -------------------------------------------------
# SAVE ROUTING SUMMARY
# -------------------------------------------------

summary = pd.DataFrame(
    {
        "metric": [
            "total_patients",
            "accepted",
            "reviewed",
            "deferred",
            "acceptance_coverage",
            "accepted_error_rate",
            "average_confidence",
            "average_ERS",
            "average_joint_reliability",
            "high_confidence_low_ERS_mismatch"
        ],

        "value": [
            len(df),
            len(accepted),
            len(reviewed),
            len(deferred),
            len(accepted) / len(df),
            accepted_error_rate,
            df["calibrated_confidence"].mean(),
            df["ERS"].mean(),
            df["joint_reliability"].mean(),
            len(mismatch)
        ]
    }
)


summary_file = os.path.join(
    PROJECT_PATH,
    "medtrust_routing_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)


# -------------------------------------------------
# DISPLAY SAMPLE
# -------------------------------------------------

print("\n==========================================")
print("SAMPLE ROUTING RESULTS")
print("==========================================\n")

print(
    df[
        [
            "subject_id",
            "icu_mortality",
            "calibrated_probability",
            "calibrated_confidence",
            "ERS",
            "joint_reliability",
            "routing_decision"
        ]
    ].to_string(
        index=False
    )
)


print("\n==========================================")
print("MEDTRUST ROUTING COMPLETE")
print("==========================================\n")

print(
    "Full results saved to:"
)

print(
    output_file
)

print(
    "\nSummary saved to:"
)

print(
    summary_file
)

print(
    "\nIMPORTANT:"
)

print(
    "These thresholds are prototype thresholds."
)

print(
    "Final research thresholds must be selected"
)

print(
    "on validation data and frozen before test evaluation."
)

print(
    "\nNext: final evaluation and research results."
)