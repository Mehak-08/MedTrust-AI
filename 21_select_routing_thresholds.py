# ============================================================
# MEDTRUST AI
# STEP 21 - VALIDATION-BASED ROUTING THRESHOLD SELECTION
# ============================================================

import os
import pandas as pd
import numpy as np


print("\n==========================================")
print("MEDTRUST AI - VALIDATION THRESHOLD SEARCH")
print("==========================================\n")


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ------------------------------------------------------------
# 1. LOAD VALIDATION PREDICTIONS
# ------------------------------------------------------------

val_predictions = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "step18_validation_predictions.csv"
    )
)


# ------------------------------------------------------------
# 2. LOAD ERS
# ------------------------------------------------------------

ers = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "step19_explanation_reliability.csv"
    )
)


print(
    "Validation prediction records:",
    len(val_predictions)
)

print(
    "ERS records:",
    len(ers)
)


# ------------------------------------------------------------
# IMPORTANT
# ------------------------------------------------------------
# Step 19 currently contains ERS for the TEST set.
#
# Therefore we cannot use that ERS directly for validation.
#
# This script creates a validation ERS using the same
# prototype ERS methodology.
#
# To keep the workflow simple, we use the validation
# prediction probability together with the ERS distribution
# learned from the test prototype only as a temporary demo.
#
# For the final research version, ERS must be generated
# separately on validation and test using the corresponding
# model/explanations.


# ------------------------------------------------------------
# 3. CHECK REQUIRED COLUMNS
# ------------------------------------------------------------

required_probability = "calibrated_probability"

if required_probability not in val_predictions.columns:

    raise ValueError(
        f"Missing column: {required_probability}"
    )


if "icu_mortality" not in val_predictions.columns:

    raise ValueError(
        "icu_mortality missing from validation predictions."
    )


# ------------------------------------------------------------
# 4. CREATE VALIDATION PROTOTYPE ERS
# ------------------------------------------------------------

# Since validation ERS has not yet been generated,
# use the average prototype ERS from Step 19.
#
# This is ONLY a temporary pipeline check.
# It must NOT be used as the final research result.

average_ers = ers["ERS"].mean()


val_predictions["ERS"] = average_ers


# ------------------------------------------------------------
# 5. PREDICTIVE CONFIDENCE
# ------------------------------------------------------------

val_predictions[
    "predictive_confidence"
] = np.maximum(
    val_predictions[
        "calibrated_probability"
    ],
    1 -
    val_predictions[
        "calibrated_probability"
    ]
)


# ------------------------------------------------------------
# 6. JOINT RELIABILITY
# ------------------------------------------------------------

val_predictions[
    "joint_reliability"
] = (
    0.5
    *
    val_predictions[
        "predictive_confidence"
    ]
    +
    0.5
    *
    val_predictions[
        "ERS"
    ]
)


# ------------------------------------------------------------
# 7. ROUTING FUNCTION
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 8. EVALUATE ONE THRESHOLD PAIR
# ------------------------------------------------------------

def evaluate_thresholds(
    data,
    score_column,
    accept_threshold,
    review_threshold
):

    decisions = data[
        score_column
    ].apply(
        lambda x:
        route(
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
            accepted[
                "icu_mortality"
            ].sum()
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


# ------------------------------------------------------------
# 9. SEARCH THRESHOLDS
# ------------------------------------------------------------

thresholds = np.round(
    np.arange(
        0.50,
        0.96,
        0.05
    ),
    2
)


results = []


for accept_threshold in thresholds:

    for review_threshold in thresholds:

        # Review threshold cannot be higher
        # than the ACCEPT threshold.

        if review_threshold > accept_threshold:

            continue

        result = evaluate_thresholds(
            val_predictions,
            "joint_reliability",
            accept_threshold,
            review_threshold
        )

        results.append(
            result
        )


results_df = pd.DataFrame(
    results
)


# ------------------------------------------------------------
# 10. TARGET ACCEPTED ERROR
# ------------------------------------------------------------

TARGET_ERROR = 0.10


print("\n==========================================")
print("THRESHOLD SEARCH")
print("==========================================")

print(
    "Target accepted error:",
    TARGET_ERROR
)


# ------------------------------------------------------------
# 11. FIND FEASIBLE THRESHOLDS
# ------------------------------------------------------------

feasible = results_df[
    (
        results_df[
            "accepted_error"
        ].notna()
    )
    &
    (
        results_df[
            "accepted_error"
        ]
        <=
        TARGET_ERROR
    )
].copy()


if len(feasible) > 0:

    # Primary objective:
    # maximize accepted coverage.
    #
    # Tie-break:
    # lower accepted error.

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

    selection_reason = (
        "Highest validation coverage "
        "while keeping accepted error "
        "at or below the target."
    )


else:

    # If no threshold satisfies the target,
    # choose the threshold with minimum
    # validation accepted error.

    fallback = results_df[
        results_df[
            "accepted_error"
        ].notna()
    ].copy()

    fallback = fallback.sort_values(
        by=[
            "accepted_error",
            "coverage"
        ],
        ascending=[
            True,
            False
        ]
    )

    selected = fallback.iloc[0]

    selection_reason = (
        "No threshold satisfied the target; "
        "selected minimum validation accepted error."
    )


# ------------------------------------------------------------
# 12. DISPLAY SELECTED THRESHOLDS
# ------------------------------------------------------------

selected_accept = float(
    selected[
        "accept_threshold"
    ]
)


selected_review = float(
    selected[
        "review_threshold"
    ]
)


print("\n==========================================")
print("SELECTED THRESHOLDS")
print("==========================================")


print(
    "ACCEPT threshold:",
    selected_accept
)

print(
    "REVIEW threshold:",
    selected_review
)

print(
    "Validation coverage:",
    f"{selected['coverage']:.4f}"
)

print(
    "Validation accepted error:",
    selected[
        "accepted_error"
    ]
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
    "\nReason:",
    selection_reason
)


# ------------------------------------------------------------
# 13. SAVE ALL THRESHOLD SEARCH RESULTS
# ------------------------------------------------------------

results_df.to_csv(
    os.path.join(
        BASE_DIR,
        "step21_threshold_search.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# 14. SAVE SELECTED THRESHOLDS
# ------------------------------------------------------------

selected_df = pd.DataFrame({

    "Parameter": [

        "accept_threshold",

        "review_threshold",

        "target_accepted_error",

        "validation_coverage",

        "validation_accepted_error",

        "validation_review_rate",

        "validation_defer_rate"
    ],

    "Value": [

        selected_accept,

        selected_review,

        TARGET_ERROR,

        selected[
            "coverage"
        ],

        selected[
            "accepted_error"
        ],

        selected[
            "review_rate"
        ],

        selected[
            "defer_rate"
        ]
    ]
})


selected_df.to_csv(
    os.path.join(
        BASE_DIR,
        "step21_selected_thresholds.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# 15. CREATE VALIDATION ROUTING
# ------------------------------------------------------------

val_predictions[
    "validation_decision"
] = val_predictions[
    "joint_reliability"
].apply(
    lambda x:
    route(
        x,
        selected_accept,
        selected_review
    )
)


val_predictions.to_csv(
    os.path.join(
        BASE_DIR,
        "step21_validation_routing.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# 16. COMPLETE
# ------------------------------------------------------------

print("\n==========================================")
print("STEP 21 COMPLETE")
print("==========================================")

print(
    "\nCreated:"
)

print(
    "step21_threshold_search.csv"
)

print(
    "step21_selected_thresholds.csv"
)

print(
    "step21_validation_routing.csv"
)

print("\nIMPORTANT:")
print(
    "These thresholds are selected using "
    "validation data only."
)

print(
    "They must remain frozen when evaluating "
    "the final test set."
)

print(
    "\nNOTE:"
)

print(
    "The validation ERS used here is only a "
    "temporary prototype because Step 19 "
    "currently generated ERS for the test set."
)

print(
    "Before the final paper experiment, we "
    "must generate a separate validation ERS."
)

print("\n==========================================")