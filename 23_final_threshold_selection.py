# ============================================================
# MEDTRUST AI
# STEP 23 - FINAL VALIDATION-BASED THRESHOLD SELECTION
# ============================================================

import os
import numpy as np
import pandas as pd


print("\n==========================================")
print("MEDTRUST AI - FINAL THRESHOLD SELECTION")
print("==========================================\n")


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# 1. LOAD VALIDATION ERS
# ============================================================

validation = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "step22_validation_ers.csv"
    )
)


print(
    "Validation cases:",
    len(validation)
)


# ============================================================
# 2. ROUTING FUNCTION
# ============================================================

def route(
    score,
    accept_threshold,
    review_threshold
):

    if score >= accept_threshold:
        return "ACCEPT"

    elif score >= review_threshold:
        return "REVIEW"

    else:
        return "DEFER"


# ============================================================
# 3. EVALUATION FUNCTION
# ============================================================

def evaluate(
    data,
    score_column,
    accept_threshold,
    review_threshold
):

    decisions = data[
        score_column
    ].apply(
        lambda x: route(
            x,
            accept_threshold,
            review_threshold
        )
    )

    accepted = data[
        decisions == "ACCEPT"
    ]

    if len(accepted) == 0:

        accepted_error = np.nan

    else:

        accepted_error = (
            accepted["icu_mortality"].sum()
            /
            len(accepted)
        )

    coverage = (
        len(accepted)
        /
        len(data)
    )

    review_rate = (
        (decisions == "REVIEW").sum()
        /
        len(data)
    )

    defer_rate = (
        (decisions == "DEFER").sum()
        /
        len(data)
    )

    return {
        "accept_threshold":
            accept_threshold,

        "review_threshold":
            review_threshold,

        "coverage":
            coverage,

        "accepted_error":
            accepted_error,

        "review_rate":
            review_rate,

        "defer_rate":
            defer_rate
    }


# ============================================================
# 4. THRESHOLD GRID
# ============================================================

thresholds = np.round(
    np.arange(
        0.40,
        0.96,
        0.05
    ),
    2
)


results = []


for accept_threshold in thresholds:

    for review_threshold in thresholds:

        if review_threshold > accept_threshold:
            continue

        result = evaluate(
            validation,
            "joint_reliability",
            accept_threshold,
            review_threshold
        )

        results.append(result)


results_df = pd.DataFrame(
    results
)


# ============================================================
# 5. TARGET ERROR
# ============================================================

TARGET_ERROR = 0.10


# ============================================================
# 6. FIND FEASIBLE OPTIONS
# ============================================================

feasible = results_df[
    results_df["accepted_error"].notna()
    &
    (
        results_df["accepted_error"]
        <= TARGET_ERROR
    )
].copy()


print("\n==========================================")
print("THRESHOLD SEARCH")
print("==========================================")

print(
    "Target accepted error:",
    TARGET_ERROR
)


# ============================================================
# 7. SELECT THRESHOLDS
# ============================================================

if len(feasible) > 0:

    # First maximize coverage.
    # Then prefer lower accepted error.

    feasible = feasible.sort_values(
        by=[
            "coverage",
            "accepted_error"
        ],
        ascending=[
            False,
            True
        ]
    )

    selected = feasible.iloc[0]

    reason = (
        "Maximum validation coverage "
        "subject to accepted error <= 10%."
    )

else:

    available = results_df[
        results_df["accepted_error"].notna()
    ].copy()

    available = available.sort_values(
        by=[
            "accepted_error",
            "coverage"
        ],
        ascending=[
            True,
            False
        ]
    )

    selected = available.iloc[0]

    reason = (
        "No threshold met the 10% target; "
        "minimum validation accepted error "
        "was selected."
    )


accept_threshold = float(
    selected["accept_threshold"]
)

review_threshold = float(
    selected["review_threshold"]
)


# ============================================================
# 8. DISPLAY
# ============================================================

print("\n==========================================")
print("SELECTED THRESHOLDS")
print("==========================================")

print(
    "ACCEPT threshold:",
    accept_threshold
)

print(
    "REVIEW threshold:",
    review_threshold
)

print(
    "Validation coverage:",
    f"{selected['coverage']:.4f}"
)

print(
    "Validation accepted error:",
    f"{selected['accepted_error']:.4f}"
)

print(
    "Validation review rate:",
    f"{selected['review_rate']:.4f}"
)

print(
    "Validation defer rate:",
    f"{selected['defer_rate']:.4f}"
)

print(
    "\nSelection reason:",
    reason
)


# ============================================================
# 9. APPLY FROZEN THRESHOLDS TO VALIDATION
# ============================================================

validation[
    "final_validation_decision"
] = validation[
    "joint_reliability"
].apply(
    lambda x: route(
        x,
        accept_threshold,
        review_threshold
    )
)


# ============================================================
# 10. SAVE THRESHOLD SEARCH
# ============================================================

results_df.to_csv(
    os.path.join(
        BASE_DIR,
        "step23_threshold_search.csv"
    ),
    index=False
)


# ============================================================
# 11. SAVE FROZEN THRESHOLDS
# ============================================================

threshold_file = pd.DataFrame({

    "Parameter": [
        "accept_threshold",
        "review_threshold",
        "target_accepted_error"
    ],

    "Value": [
        accept_threshold,
        review_threshold,
        TARGET_ERROR
    ]
})


threshold_file.to_csv(
    os.path.join(
        BASE_DIR,
        "step23_frozen_thresholds.csv"
    ),
    index=False
)


# ============================================================
# 12. SAVE VALIDATION ROUTING
# ============================================================

validation.to_csv(
    os.path.join(
        BASE_DIR,
        "step23_validation_final_routing.csv"
    ),
    index=False
)


# ============================================================
# 13. COMPLETE
# ============================================================

print("\n==========================================")
print("STEP 23 COMPLETE")
print("==========================================")

print("\nCreated:")

print(
    "step23_threshold_search.csv"
)

print(
    "step23_frozen_thresholds.csv"
)

print(
    "step23_validation_final_routing.csv"
)

print("\nIMPORTANT:")
print(
    "These thresholds were selected ONLY "
    "using validation data."
)

print(
    "They are now FROZEN and must not be "
    "changed using test results."
)

print("\nNext:")
print(
    "Step 24 = apply the frozen thresholds "
    "to the untouched test set."
)

print("\n==========================================")