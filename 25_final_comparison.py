# ============================================================
# MEDTRUST AI
# STEP 25 - FINAL COMPARISON + GRAPHS
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


print("\n==========================================")
print("MEDTRUST AI - FINAL COMPARISON")
print("==========================================\n")


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "final_medtrust_results"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 1. LOAD FINAL TEST RESULTS
# ============================================================

df = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "step24_final_test_routing.csv"
    )
)


metrics = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "step24_final_test_metrics.csv"
    )
)


print(
    "Final test cases:",
    len(df)
)


# ============================================================
# 2. ROUTING COUNTS
# ============================================================

routes = [
    "ACCEPT",
    "REVIEW",
    "DEFER"
]


confidence_counts = [
    (
        df["confidence_only_decision"] == r
    ).sum()
    for r in routes
]


medtrust_counts = [
    (
        df["medtrust_decision"] == r
    ).sum()
    for r in routes
]


# ============================================================
# 3. ROUTING COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame({

    "Decision": routes,

    "Confidence_Only_Count":
        confidence_counts,

    "MedTrust_Count":
        medtrust_counts
})


comparison[
    "Confidence_Only_Percentage"
] = (
    comparison[
        "Confidence_Only_Count"
    ]
    /
    len(df)
    *
    100
)


comparison[
    "MedTrust_Percentage"
] = (
    comparison[
        "MedTrust_Count"
    ]
    /
    len(df)
    *
    100
)


comparison.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "25_routing_comparison.csv"
    ),
    index=False
)


# ============================================================
# 4. SELECTIVE PERFORMANCE
# ============================================================

def calculate_metrics(
    decision_column
):

    accepted = df[
        df[decision_column] == "ACCEPT"
    ]

    coverage = (
        len(accepted)
        /
        len(df)
    )

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

    review_rate = (
        (
            df[decision_column]
            == "REVIEW"
        ).sum()
        /
        len(df)
    )

    defer_rate = (
        (
            df[decision_column]
            == "DEFER"
        ).sum()
        /
        len(df)
    )

    return [
        coverage,
        accepted_error,
        review_rate,
        defer_rate
    ]


confidence_metrics = calculate_metrics(
    "confidence_only_decision"
)

medtrust_metrics = calculate_metrics(
    "medtrust_decision"
)


performance = pd.DataFrame({

    "Metric": [
        "ACCEPT Coverage",
        "Accepted Error",
        "REVIEW Rate",
        "DEFER Rate"
    ],

    "Confidence_Only": confidence_metrics,

    "MedTrust": medtrust_metrics
})


performance.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "25_selective_performance.csv"
    ),
    index=False
)


# ============================================================
# 5. ROUTING CHANGE ANALYSIS
# ============================================================

routing_changes = (
    df[
        "confidence_only_decision"
    ]
    !=
    df[
        "medtrust_decision"
    ]
)


change_count = int(
    routing_changes.sum()
)


change_percentage = (
    change_count
    /
    len(df)
    *
    100
)


change_summary = pd.DataFrame({

    "Metric": [
        "Routing changes",
        "Routing change percentage"
    ],

    "Value": [
        change_count,
        change_percentage
    ]
})


change_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "25_routing_changes.csv"
    ),
    index=False
)


# ============================================================
# 6. HIGH-CONFIDENCE / LOW-ERS CASES
# ============================================================

mismatch = df[
    (
        df[
            "predictive_confidence"
        ] >= 0.85
    )
    &
    (
        df["ERS"] < 0.50
    )
].copy()


mismatch.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "25_high_confidence_low_ers.csv"
    ),
    index=False
)


# ============================================================
# 7. GRAPH 1 - ROUTING DISTRIBUTION
# ============================================================

x = np.arange(
    len(routes)
)

width = 0.35


plt.figure(
    figsize=(9, 6)
)

plt.bar(
    x - width / 2,
    confidence_counts,
    width,
    label="Confidence-only"
)

plt.bar(
    x + width / 2,
    medtrust_counts,
    width,
    label="MedTrust"
)

plt.xticks(
    x,
    routes
)

plt.ylabel(
    "Number of patients"
)

plt.xlabel(
    "Routing decision"
)

plt.title(
    "Confidence-only vs MedTrust Routing"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "25_routing_distribution.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 8. GRAPH 2 - ACCEPT COVERAGE
# ============================================================

coverage_values = [
    confidence_metrics[0] * 100,
    medtrust_metrics[0] * 100
]


plt.figure(
    figsize=(8, 6)
)

plt.bar(
    [
        "Confidence-only",
        "MedTrust"
    ],
    coverage_values
)

plt.ylabel(
    "ACCEPT coverage (%)"
)

plt.title(
    "Accepted Prediction Coverage"
)

plt.ylim(
    0,
    100
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "25_accept_coverage.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 9. GRAPH 3 - ACCEPTED ERROR
# ============================================================

error_values = [
    confidence_metrics[1],
    medtrust_metrics[1]
]


plt.figure(
    figsize=(8, 6)
)

plt.bar(
    [
        "Confidence-only",
        "MedTrust"
    ],
    error_values
)

plt.ylabel(
    "Accepted error rate"
)

plt.title(
    "Error Rate Among Accepted Predictions"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "25_accepted_error.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 10. GRAPH 4 - CONFIDENCE VS ERS
# ============================================================

plt.figure(
    figsize=(9, 7)
)

plt.scatter(
    df[
        "predictive_confidence"
    ],
    df["ERS"],
    alpha=0.8
)

plt.axvline(
    0.85,
    linestyle="--",
    label="Confidence = 0.85"
)

plt.axhline(
    0.50,
    linestyle="--",
    label="ERS = 0.50"
)

plt.xlabel(
    "Predictive confidence"
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
        OUTPUT_DIR,
        "25_confidence_vs_ers.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 11. GRAPH 5 - JOINT RELIABILITY
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.hist(
    df[
        "joint_reliability"
    ],
    bins=10
)

plt.xlabel(
    "Joint reliability"
)

plt.ylabel(
    "Number of patients"
)

plt.title(
    "Distribution of Joint Reliability"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "25_joint_reliability_distribution.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 12. FINAL RESEARCH TABLE
# ============================================================

final_table = pd.DataFrame({

    "Metric": [

        "Test cases",

        "Confidence-only ACCEPT coverage",

        "MedTrust ACCEPT coverage",

        "Confidence-only accepted error",

        "MedTrust accepted error",

        "Confidence-only REVIEW rate",

        "MedTrust REVIEW rate",

        "Confidence-only DEFER rate",

        "MedTrust DEFER rate",

        "Routing changes",

        "Routing change percentage",

        "High-confidence / low-ERS cases",

        "Average predictive confidence",

        "Average ERS",

        "Average joint reliability"
    ],

    "Value": [

        len(df),

        confidence_metrics[0],

        medtrust_metrics[0],

        confidence_metrics[1],

        medtrust_metrics[1],

        confidence_metrics[2],

        medtrust_metrics[2],

        confidence_metrics[3],

        medtrust_metrics[3],

        change_count,

        change_percentage,

        len(mismatch),

        df[
            "predictive_confidence"
        ].mean(),

        df[
            "ERS"
        ].mean(),

        df[
            "joint_reliability"
        ].mean()
    ]
})


final_table.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "25_final_research_results.csv"
    ),
    index=False
)


# ============================================================
# 13. PRINT RESULTS
# ============================================================

print("\n==========================================")
print("FINAL COMPARISON")
print("==========================================")

print("\nConfidence-only:")

print(
    "  ACCEPT:",
    confidence_counts[0]
)

print(
    "  REVIEW:",
    confidence_counts[1]
)

print(
    "  DEFER:",
    confidence_counts[2]
)

print(
    "  Coverage:",
    f"{confidence_metrics[0]:.4f}"
)

print(
    "  Accepted error:",
    confidence_metrics[1]
)


print("\nMedTrust:")

print(
    "  ACCEPT:",
    medtrust_counts[0]
)

print(
    "  REVIEW:",
    medtrust_counts[1]
)

print(
    "  DEFER:",
    medtrust_counts[2]
)

print(
    "  Coverage:",
    f"{medtrust_metrics[0]:.4f}"
)

print(
    "  Accepted error:",
    medtrust_metrics[1]
)


print("\nRouting changes:")
print(
    f"{change_count} "
    f"({change_percentage:.2f}%)"
)


print(
    "\nHigh-confidence / low-ERS:",
    len(mismatch)
)


# ============================================================
# 14. COMPLETE
# ============================================================

print("\n==========================================")
print("STEP 25 COMPLETE")
print("==========================================")

print("\nCreated folder:")

print(
    "final_medtrust_results"
)

print("\nCreated files:")

print(
    "25_routing_comparison.csv"
)

print(
    "25_selective_performance.csv"
)

print(
    "25_routing_changes.csv"
)

print(
    "25_high_confidence_low_ers.csv"
)

print(
    "25_final_research_results.csv"
)

print(
    "25_routing_distribution.png"
)

print(
    "25_accept_coverage.png"
)

print(
    "25_accepted_error.png"
)

print(
    "25_confidence_vs_ers.png"
)

print(
    "25_joint_reliability_distribution.png"
)

print("\nIMPORTANT:")
print(
    "These results are from the MIMIC-IV DEMO dataset "
    "and should not be presented as final clinical performance."
)

print("\n==========================================")