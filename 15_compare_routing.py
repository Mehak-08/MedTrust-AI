# ============================================================
# MEDTRUST AI
# STEP 15 - CONFIDENCE-ONLY vs MEDTRUST ROUTING
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# PROJECT PATH
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "final_results")

os.makedirs(RESULTS_DIR, exist_ok=True)


print("\n==========================================")
print("MEDTRUST AI - ROUTING COMPARISON")
print("==========================================\n")


# ------------------------------------------------------------
# 1. FIND PATIENT-LEVEL ROUTING DATA
# ------------------------------------------------------------

# IMPORTANT:
# Do NOT use final_results/routing_summary.csv.
# That file contains only 3 summary rows.
#
# We need the 21 patient-level routing records.

possible_files = [
    os.path.join(BASE_DIR, "medtrust_routing.csv"),
    os.path.join(BASE_DIR, "routing_results.csv"),
    os.path.join(BASE_DIR, "routing_output.csv"),
    os.path.join(BASE_DIR, "medtrust_routing_results.csv"),
    os.path.join(BASE_DIR, "calibrated_xgboost_predictions.csv"),
]


routing_file = None


for file in possible_files:

    if os.path.exists(file):

        routing_file = file
        break


if routing_file is None:

    print("ERROR: Patient-level routing data was not found.")

    print("\nChecked:")

    for file in possible_files:
        print(file)

    print(
        "\nPlease check the CSV created by Step 12."
    )

    raise SystemExit


print("Using patient-level file:")
print(routing_file)


# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(routing_file)


print("\nLoaded records:", len(df))


print("\nColumns found:")
print(df.columns.tolist())


# ------------------------------------------------------------
# 3. NORMALIZE COLUMN NAMES
# ------------------------------------------------------------

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)


# ------------------------------------------------------------
# 4. DISPLAY NORMALIZED COLUMNS
# ------------------------------------------------------------

print("\nNormalized columns:")
print(df.columns.tolist())


# ------------------------------------------------------------
# 5. FIND COLUMNS
# ------------------------------------------------------------

def find_column(names):

    for name in names:

        if name in df.columns:
            return name

    return None


confidence_col = find_column([
    "confidence",
    "calibrated_confidence",
    "model_confidence"
])


ers_col = find_column([
    "ers",
    "explanation_reliability",
    "explanation_reliability_score",
    "explanation_reliability_score_(ers)"
])


target_col = find_column([
    "icu_mortality",
    "mortality",
    "target",
    "label"
])


print("\nDetected columns:")

print("Confidence:", confidence_col)
print("ERS:", ers_col)
print("Target:", target_col)


# ------------------------------------------------------------
# 6. SPECIAL CASE:
# IF STEP 12 FILE IS NOT FOUND, USE EXISTING
# CALIBRATED PREDICTIONS + ERS
# ------------------------------------------------------------

if confidence_col is None or ers_col is None:

    prediction_file = os.path.join(
        BASE_DIR,
        "calibrated_xgboost_predictions.csv"
    )

    ers_file = os.path.join(
        BASE_DIR,
        "explanation_stability.csv"
    )

    if (
        os.path.exists(prediction_file)
        and
        os.path.exists(ers_file)
    ):

        print(
            "\nPatient-level routing CSV was not found."
        )

        print(
            "Rebuilding patient-level routing data "
            "from existing Step 9 + Step 11 files..."
        )

        predictions = pd.read_csv(
            prediction_file
        )

        ers_data = pd.read_csv(
            ers_file
        )


        print(
            "\nPrediction columns:"
        )

        print(
            predictions.columns.tolist()
        )


        print(
            "\nERS columns:"
        )

        print(
            ers_data.columns.tolist()
        )


        # Normalize

        predictions.columns = (
            predictions.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

        ers_data.columns = (
            ers_data.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )


        # Find prediction probability

        probability_col = find_column([
            "probability",
            "predicted_probability",
            "calibrated_probability",
            "y_probability",
            "prob_1",
            "prediction"
        ])


        # Because find_column uses df,
        # manually search predictions

        probability_col = None

        for col in [
            "probability",
            "predicted_probability",
            "calibrated_probability",
            "y_probability",
            "prob_1",
            "prediction"
        ]:

            if col in predictions.columns:

                probability_col = col
                break


        # Find target

        prediction_target_col = None

        for col in [
            "icu_mortality",
            "mortality",
            "target",
            "actual",
            "y_true"
        ]:

            if col in predictions.columns:

                prediction_target_col = col
                break


        # Find ERS

        ers_value_col = None

        for col in [
            "ers",
            "stability",
            "explanation_reliability",
            "explanation_reliability_score"
        ]:

            if col in ers_data.columns:

                ers_value_col = col
                break


        # Find ID columns

        prediction_id = None

        for col in [
            "subject_id",
            "stay_id",
            "hadm_id"
        ]:

            if col in predictions.columns:

                prediction_id = col
                break


        ers_id = None

        for col in [
            "subject_id",
            "stay_id",
            "hadm_id"
        ]:

            if col in ers_data.columns:

                ers_id = col
                break


        if (
            probability_col is None
            or
            prediction_target_col is None
            or
            ers_value_col is None
        ):

            print(
                "\nCould not automatically reconstruct "
                "the routing data."
            )

            print(
                "Prediction columns:",
                predictions.columns.tolist()
            )

            print(
                "ERS columns:",
                ers_data.columns.tolist()
            )

            raise SystemExit


        # ----------------------------------------------------
        # MERGE
        # ----------------------------------------------------

        if (
            prediction_id is not None
            and
            ers_id is not None
            and
            prediction_id == ers_id
        ):

            df = pd.merge(
                predictions,
                ers_data[
                    [ers_id, ers_value_col]
                ],
                on=prediction_id,
                how="inner"
            )

        else:

            # If no common ID exists,
            # align by row because both files were
            # generated from the same test set.

            min_rows = min(
                len(predictions),
                len(ers_data)
            )

            df = predictions.iloc[
                :min_rows
            ].copy()

            df["ers"] = (
                ers_data[
                    ers_value_col
                ]
                .iloc[:min_rows]
                .values
            )

            ers_value_col = "ers"


        # Rename important columns

        df["mortality"] = pd.to_numeric(
            df[prediction_target_col],
            errors="coerce"
        )

        df["probability"] = pd.to_numeric(
            df[probability_col],
            errors="coerce"
        )

        df["ers"] = pd.to_numeric(
            df[ers_value_col],
            errors="coerce"
        )


        # Calibrated confidence

        df["confidence"] = np.maximum(
            df["probability"],
            1 - df["probability"]
        )


        confidence_col = "confidence"
        ers_col = "ers"
        target_col = "mortality"


        print(
            "\nSuccessfully reconstructed "
            "patient-level routing data."
        )

        print(
            "Records:",
            len(df)
        )


# ------------------------------------------------------------
# 7. FINAL COLUMN CHECK
# ------------------------------------------------------------

if confidence_col is None:

    raise ValueError(
        "Confidence column could not be found."
    )


if ers_col is None:

    raise ValueError(
        "ERS column could not be found."
    )


if target_col is None:

    raise ValueError(
        "Mortality/target column could not be found."
    )


# ------------------------------------------------------------
# 8. NUMERIC CONVERSION
# ------------------------------------------------------------

df[confidence_col] = pd.to_numeric(
    df[confidence_col],
    errors="coerce"
)


df[ers_col] = pd.to_numeric(
    df[ers_col],
    errors="coerce"
)


df[target_col] = pd.to_numeric(
    df[target_col],
    errors="coerce"
)


df = df.dropna(
    subset=[
        confidence_col,
        ers_col,
        target_col
    ]
).copy()


print(
    "\nFinal patient records:",
    len(df)
)


# ------------------------------------------------------------
# 9. JOINT RELIABILITY
# ------------------------------------------------------------

df["joint_reliability"] = (
    0.5 * df[confidence_col]
    +
    0.5 * df[ers_col]
)


# ------------------------------------------------------------
# 10. PROTOTYPE THRESHOLDS
# ------------------------------------------------------------

ACCEPT_THRESHOLD = 0.85
REVIEW_THRESHOLD = 0.70


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
    "DEFER: below",
    REVIEW_THRESHOLD
)


# ------------------------------------------------------------
# 11. CONFIDENCE-ONLY ROUTING
# ------------------------------------------------------------

def confidence_route(value):

    if value >= ACCEPT_THRESHOLD:

        return "ACCEPT"

    elif value >= REVIEW_THRESHOLD:

        return "REVIEW"

    else:

        return "DEFER"


df["confidence_only_route"] = (
    df[confidence_col]
    .apply(confidence_route)
)


# ------------------------------------------------------------
# 12. MEDTRUST ROUTING
# ------------------------------------------------------------

def medtrust_route(value):

    if value >= ACCEPT_THRESHOLD:

        return "ACCEPT"

    elif value >= REVIEW_THRESHOLD:

        return "REVIEW"

    else:

        return "DEFER"


df["medtrust_route"] = (
    df["joint_reliability"]
    .apply(medtrust_route)
)


# ------------------------------------------------------------
# 13. ROUTING COUNTS
# ------------------------------------------------------------

routes = [
    "ACCEPT",
    "REVIEW",
    "DEFER"
]


confidence_counts = (
    df["confidence_only_route"]
    .value_counts()
    .reindex(
        routes,
        fill_value=0
    )
)


medtrust_counts = (
    df["medtrust_route"]
    .value_counts()
    .reindex(
        routes,
        fill_value=0
    )
)


print("\n==========================================")
print("CONFIDENCE-ONLY")
print("==========================================")

print(confidence_counts)


print("\n==========================================")
print("MEDTRUST")
print("==========================================")

print(medtrust_counts)


# ------------------------------------------------------------
# 14. ROUTING CHANGE
# ------------------------------------------------------------

df["routing_changed"] = (
    df["confidence_only_route"]
    !=
    df["medtrust_route"]
)


changed_count = int(
    df["routing_changed"].sum()
)


changed_percent = (
    changed_count / len(df) * 100
)


print("\n==========================================")
print("ROUTING CHANGES")
print("==========================================")

print(
    "Changed:",
    changed_count
)

print(
    "Percentage:",
    f"{changed_percent:.2f}%"
)


# ------------------------------------------------------------
# 15. ACCEPTED ERROR RATE
# ------------------------------------------------------------

def accepted_error(
    data,
    route_column
):

    accepted = data[
        data[route_column] == "ACCEPT"
    ]

    if len(accepted) == 0:

        return np.nan

    errors = (
        accepted[target_col] == 1
    ).sum()

    return (
        errors / len(accepted)
    )


confidence_error = accepted_error(
    df,
    "confidence_only_route"
)


medtrust_error = accepted_error(
    df,
    "medtrust_route"
)


# ------------------------------------------------------------
# 16. COVERAGE
# ------------------------------------------------------------

confidence_coverage = (
    confidence_counts["ACCEPT"]
    /
    len(df)
)


medtrust_coverage = (
    medtrust_counts["ACCEPT"]
    /
    len(df)
)


print("\n==========================================")
print("SELECTIVE COVERAGE")
print("==========================================")

print(
    "Confidence-only:",
    f"{confidence_coverage:.4f}"
)

print(
    "MedTrust:",
    f"{medtrust_coverage:.4f}"
)


print("\nAccepted error:")

print(
    "Confidence-only:",
    confidence_error
)

print(
    "MedTrust:",
    medtrust_error
)


# ------------------------------------------------------------
# 17. COMPARISON TABLE
# ------------------------------------------------------------

comparison = pd.DataFrame({

    "Metric": [

        "Total Patients",

        "ACCEPT Count",

        "REVIEW Count",

        "DEFER Count",

        "ACCEPT Coverage",

        "Accepted Error Rate",

        "Routing Changed Count",

        "Routing Changed Percentage"
    ],

    "Confidence_Only": [

        len(df),

        confidence_counts["ACCEPT"],

        confidence_counts["REVIEW"],

        confidence_counts["DEFER"],

        confidence_coverage,

        confidence_error,

        changed_count,

        changed_percent
    ],

    "MedTrust_Confidence_ERS": [

        len(df),

        medtrust_counts["ACCEPT"],

        medtrust_counts["REVIEW"],

        medtrust_counts["DEFER"],

        medtrust_coverage,

        medtrust_error,

        changed_count,

        changed_percent
    ]
})


comparison_path = os.path.join(
    RESULTS_DIR,
    "confidence_vs_medtrust_comparison.csv"
)


comparison.to_csv(
    comparison_path,
    index=False
)


# ------------------------------------------------------------
# 18. ROUTING DISTRIBUTION TABLE
# ------------------------------------------------------------

routing_table = pd.DataFrame({

    "Route": routes,

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
        / len(df)
        * 100
        for r in routes
    ],

    "MedTrust_Percentage": [
        medtrust_counts[r]
        / len(df)
        * 100
        for r in routes
    ]
})


routing_table.to_csv(
    os.path.join(
        RESULTS_DIR,
        "routing_distribution_comparison.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# 19. PATIENT-LEVEL COMPARISON
# ------------------------------------------------------------

patient_columns = [
    confidence_col,
    ers_col,
    "joint_reliability",
    "confidence_only_route",
    "medtrust_route",
    "routing_changed",
    target_col
]


patient_comparison = df[
    patient_columns
].copy()


patient_comparison.to_csv(
    os.path.join(
        RESULTS_DIR,
        "patient_routing_comparison.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# 20. ROUTING BAR CHART
# ------------------------------------------------------------

print(
    "\nCreating routing comparison chart..."
)


x = np.arange(
    len(routes)
)


width = 0.35


plt.figure(
    figsize=(9, 6)
)


plt.bar(
    x - width / 2,
    [
        confidence_counts[r]
        for r in routes
    ],
    width,
    label="Confidence-only"
)


plt.bar(
    x + width / 2,
    [
        medtrust_counts[r]
        for r in routes
    ],
    width,
    label="MedTrust"
)


plt.xticks(
    x,
    routes
)


plt.ylabel(
    "Number of Patients"
)


plt.xlabel(
    "Routing Decision"
)


plt.title(
    "Confidence-Only vs MedTrust Routing"
)


plt.legend()


plt.tight_layout()


plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "06_confidence_vs_medtrust_routing.png"
    ),
    dpi=300
)


plt.close()


# ------------------------------------------------------------
# 21. CONFIDENCE VS ERS PLOT
# ------------------------------------------------------------

print(
    "Creating confidence vs ERS plot..."
)


plt.figure(
    figsize=(8, 6)
)


plt.scatter(
    df[confidence_col],
    df[ers_col],
    alpha=0.75
)


plt.axvline(
    ACCEPT_THRESHOLD,
    linestyle="--",
    label="Accept threshold"
)


plt.axhline(
    ACCEPT_THRESHOLD,
    linestyle="--"
)


plt.axvline(
    REVIEW_THRESHOLD,
    linestyle=":",
    label="Review threshold"
)


plt.axhline(
    REVIEW_THRESHOLD,
    linestyle=":"
)


plt.xlabel(
    "Calibrated Confidence"
)


plt.ylabel(
    "Explanation Reliability Score (ERS)"
)


plt.title(
    "Predictive Confidence vs Explanation Reliability"
)


plt.legend()


plt.tight_layout()


plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "07_confidence_vs_ers_routing.png"
    ),
    dpi=300
)


plt.close()


# ------------------------------------------------------------
# 22. PRINT CHANGED RECORDS
# ------------------------------------------------------------

changed = df[
    df["routing_changed"]
].copy()


if len(changed) > 0:

    print(
        "\n=========================================="
    )

    print(
        "PATIENTS WITH CHANGED ROUTING"
    )

    print(
        "=========================================="
    )


    print(
        changed[
            [
                confidence_col,
                ers_col,
                "joint_reliability",
                "confidence_only_route",
                "medtrust_route",
                target_col
            ]
        ].to_string(
            index=False
        )
    )

else:

    print(
        "\nNo routing changes under "
        "the current prototype thresholds."
    )


# ------------------------------------------------------------
# 23. FINAL OUTPUT
# ------------------------------------------------------------

print("\n==========================================")
print("STEP 15 COMPLETE")
print("==========================================")

print("\nResults saved to:")

print(
    RESULTS_DIR
)


print("\nCreated:")

print(
    "confidence_vs_medtrust_comparison.csv"
)

print(
    "routing_distribution_comparison.csv"
)

print(
    "patient_routing_comparison.csv"
)

print(
    "06_confidence_vs_medtrust_routing.png"
)

print(
    "07_confidence_vs_ers_routing.png"
)


print("\n==========================================")
print("IMPORTANT")
print("==========================================")

print(
    "These are DEMO dataset results."
)

print(
    "Routing thresholds are prototype thresholds."
)

print(
    "Do not use these numbers as final clinical claims."
)