# ============================================================
# MEDTRUST AI
# STEP 24 - FINAL TEST EVALUATION
# ============================================================

import os
import numpy as np
import pandas as pd


print("\n==========================================")
print("MEDTRUST AI - FINAL TEST EVALUATION")
print("==========================================\n")


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# 1. LOAD TEST ERS
# ============================================================

test = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "step22_test_ers.csv"
    )
)


# ============================================================
# 2. LOAD FROZEN THRESHOLDS
# ============================================================

thresholds = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "step23_frozen_thresholds.csv"
    )
)


accept_threshold = float(
    thresholds.loc[
        thresholds["Parameter"]
        == "accept_threshold",
        "Value"
    ].iloc[0]
)


review_threshold = float(
    thresholds.loc[
        thresholds["Parameter"]
        == "review_threshold",
        "Value"
    ].iloc[0]
)


print("Frozen ACCEPT threshold:", accept_threshold)
print("Frozen REVIEW threshold:", review_threshold)


# ============================================================
# 3. ROUTING FUNCTION
# ============================================================

def route(score):

    if score >= accept_threshold:
        return "ACCEPT"

    elif score >= review_threshold:
        return "REVIEW"

    else:
        return "DEFER"


# ============================================================
# 4. MEDTRUST ROUTING
# ============================================================

test["medtrust_decision"] = (
    test["joint_reliability"]
    .apply(route)
)


# ============================================================
# 5. CONFIDENCE-ONLY ROUTING
# ============================================================

test["confidence_only_decision"] = (
    test["predictive_confidence"]
    .apply(route)
)


# ============================================================
# 6. ROUTING COUNTS
# ============================================================

routes = [
    "ACCEPT",
    "REVIEW",
    "DEFER"
]


print("\n==========================================")
print("MEDTRUST TEST ROUTING")
print("==========================================")


for r in routes:

    count = (
        test["medtrust_decision"] == r
    ).sum()

    percentage = (
        count
        /
        len(test)
        *
        100
    )

    print(
        f"{r}: {count} "
        f"({percentage:.2f}%)"
    )


print("\n==========================================")
print("CONFIDENCE-ONLY TEST ROUTING")
print("==========================================")


for r in routes:

    count = (
        test["confidence_only_decision"] == r
    ).sum()

    percentage = (
        count
        /
        len(test)
        *
        100
    )

    print(
        f"{r}: {count} "
        f"({percentage:.2f}%)"
    )


# ============================================================
# 7. ACCEPTED ERROR
# ============================================================

def accepted_error(
    data,
    decision_column
):

    accepted = data[
        data[decision_column] == "ACCEPT"
    ]

    if len(accepted) == 0:
        return np.nan

    errors = (
        accepted["icu_mortality"] == 1
    ).sum()

    return errors / len(accepted)


medtrust_error = accepted_error(
    test,
    "medtrust_decision"
)


confidence_error = accepted_error(
    test,
    "confidence_only_decision"
)


# ============================================================
# 8. COVERAGE
# ============================================================

medtrust_coverage = (
    test["medtrust_decision"]
    .eq("ACCEPT")
    .mean()
)


confidence_coverage = (
    test["confidence_only_decision"]
    .eq("ACCEPT")
    .mean()
)


# ============================================================
# 9. ROUTING CHANGES
# ============================================================

test["routing_changed"] = (
    test["medtrust_decision"]
    !=
    test["confidence_only_decision"]
)


routing_changes = int(
    test["routing_changed"].sum()
)


# ============================================================
# 10. HIGH-CONFIDENCE / LOW-ERS
# ============================================================

mismatch = (
    (test["predictive_confidence"] >= 0.85)
    &
    (test["ERS"] < 0.50)
)


mismatch_count = int(
    mismatch.sum()
)


# ============================================================
# 11. PRINT FINAL METRICS
# ============================================================

print("\n==========================================")
print("FINAL TEST RESULTS")
print("==========================================")


print(
    "\nTest cases:",
    len(test)
)


print(
    "\nMedTrust ACCEPT coverage:",
    f"{medtrust_coverage:.4f}"
)


print(
    "Confidence-only ACCEPT coverage:",
    f"{confidence_coverage:.4f}"
)


print(
    "\nMedTrust accepted error:",
    medtrust_error
)


print(
    "Confidence-only accepted error:",
    confidence_error
)


print(
    "\nRouting changes:",
    routing_changes
)


print(
    "High-confidence / low-ERS cases:",
    mismatch_count
)


print(
    "\nAverage confidence:",
    f"{test['predictive_confidence'].mean():.4f}"
)


print(
    "Average ERS:",
    f"{test['ERS'].mean():.4f}"
)


print(
    "Average joint reliability:",
    f"{test['joint_reliability'].mean():.4f}"
)


# ============================================================
# 12. SAVE PATIENT-LEVEL TEST RESULTS
# ============================================================

test.to_csv(
    os.path.join(
        BASE_DIR,
        "step24_final_test_routing.csv"
    ),
    index=False
)


# ============================================================
# 13. SAVE SUMMARY
# ============================================================

summary = pd.DataFrame({

    "Metric": [

        "Test cases",

        "Frozen ACCEPT threshold",

        "Frozen REVIEW threshold",

        "MedTrust ACCEPT coverage",

        "Confidence-only ACCEPT coverage",

        "MedTrust accepted error",

        "Confidence-only accepted error",

        "Routing changes",

        "High-confidence low-ERS cases",

        "Average predictive confidence",

        "Average ERS",

        "Average joint reliability"
    ],

    "Value": [

        len(test),

        accept_threshold,

        review_threshold,

        medtrust_coverage,

        confidence_coverage,

        medtrust_error,

        confidence_error,

        routing_changes,

        mismatch_count,

        test["predictive_confidence"].mean(),

        test["ERS"].mean(),

        test["joint_reliability"].mean()
    ]
})


summary.to_csv(
    os.path.join(
        BASE_DIR,
        "step24_final_test_metrics.csv"
    ),
    index=False
)


# ============================================================
# 14. SAVE ROUTING DISTRIBUTION
# ============================================================

distribution = pd.DataFrame({

    "Decision": routes,

    "Confidence_Only_Count": [
        (
            test["confidence_only_decision"] == r
        ).sum()
        for r in routes
    ],

    "MedTrust_Count": [
        (
            test["medtrust_decision"] == r
        ).sum()
        for r in routes
    ]
})


distribution.to_csv(
    os.path.join(
        BASE_DIR,
        "step24_routing_distribution.csv"
    ),
    index=False
)


# ============================================================
# 15. COMPLETE
# ============================================================

print("\n==========================================")
print("STEP 24 COMPLETE")
print("==========================================")

print("\nCreated:")

print(
    "step24_final_test_routing.csv"
)

print(
    "step24_final_test_metrics.csv"
)

print(
    "step24_routing_distribution.csv"
)

print("\nIMPORTANT:")
print(
    "The test set was evaluated using "
    "frozen validation-selected thresholds."
)

print(
    "Do NOT change these thresholds based "
    "on the test results."
)

print("\nNext:")
print(
    "Step 25 = final confidence-only vs "
    "MedTrust comparison and graphs."
)

print("\n==========================================")