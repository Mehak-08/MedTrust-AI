# ============================================================
# MEDTRUST AI
# STEP 26 - FINAL REPORT
# ============================================================

import os
import pandas as pd
from datetime import datetime


print("\n==========================================")
print("MEDTRUST AI - FINAL REPORT")
print("==========================================\n")


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "final_medtrust_results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# 1. LOAD FINAL RESULTS
# ============================================================

results_file = os.path.join(
    RESULTS_DIR,
    "25_final_research_results.csv"
)

comparison_file = os.path.join(
    RESULTS_DIR,
    "25_routing_comparison.csv"
)

performance_file = os.path.join(
    RESULTS_DIR,
    "25_selective_performance.csv"
)

changes_file = os.path.join(
    RESULTS_DIR,
    "25_routing_changes.csv"
)


results = pd.read_csv(results_file)
comparison = pd.read_csv(comparison_file)
performance = pd.read_csv(performance_file)
changes = pd.read_csv(changes_file)


# ============================================================
# 2. HELPER FUNCTION
# ============================================================

def get_value(metric):

    row = results[
        results["Metric"] == metric
    ]

    if len(row) == 0:
        return None

    return row.iloc[0]["Value"]


# ============================================================
# 3. EXTRACT RESULTS
# ============================================================

test_cases = get_value(
    "Test cases"
)

confidence_coverage = get_value(
    "Confidence-only ACCEPT coverage"
)

medtrust_coverage = get_value(
    "MedTrust ACCEPT coverage"
)

confidence_error = get_value(
    "Confidence-only accepted error"
)

medtrust_error = get_value(
    "MedTrust accepted error"
)

confidence_review = get_value(
    "Confidence-only REVIEW rate"
)

medtrust_review = get_value(
    "MedTrust REVIEW rate"
)

confidence_defer = get_value(
    "Confidence-only DEFER rate"
)

medtrust_defer = get_value(
    "MedTrust DEFER rate"
)

routing_changes = get_value(
    "Routing changes"
)

routing_change_percentage = get_value(
    "Routing change percentage"
)

high_confidence_low_ers = get_value(
    "High-confidence / low-ERS cases"
)

avg_confidence = get_value(
    "Average predictive confidence"
)

avg_ers = get_value(
    "Average ERS"
)

avg_joint = get_value(
    "Average joint reliability"
)


# ============================================================
# 4. ROUTING COUNTS
# ============================================================

def get_count(
    column,
    decision
):

    row = comparison[
        comparison["Decision"] == decision
    ]

    if len(row) == 0:
        return 0

    return int(
        row.iloc[0][column]
    )


confidence_accept = get_count(
    "Confidence_Only_Count",
    "ACCEPT"
)

confidence_review_count = get_count(
    "Confidence_Only_Count",
    "REVIEW"
)

confidence_defer_count = get_count(
    "Confidence_Only_Count",
    "DEFER"
)


medtrust_accept = get_count(
    "MedTrust_Count",
    "ACCEPT"
)

medtrust_review_count = get_count(
    "MedTrust_Count",
    "REVIEW"
)

medtrust_defer_count = get_count(
    "MedTrust_Count",
    "DEFER"
)


# ============================================================
# 5. CREATE TEXT REPORT
# ============================================================

report_path = os.path.join(
    RESULTS_DIR,
    "MedTrust_AI_Final_Report.txt"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "============================================================\n"
    )

    f.write(
        "MEDTRUST AI\n"
    )

    f.write(
        "Reliability-Aware Clinical Decision Routing\n"
    )

    f.write(
        "FINAL DEMO EXPERIMENT REPORT\n"
    )

    f.write(
        "============================================================\n\n"
    )


    f.write(
        "Generated: "
        + datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        + "\n\n"
    )


    # --------------------------------------------------------
    # PROJECT
    # --------------------------------------------------------

    f.write(
        "1. PROJECT OVERVIEW\n"
    )

    f.write(
        "------------------------------------------------------------\n"
    )

    f.write(
        "MedTrust AI investigates whether predictive confidence "
        "alone is sufficient for deciding whether a clinical "
        "prediction should be accepted.\n\n"
    )

    f.write(
        "The framework combines predictive confidence with an "
        "Explanation Reliability Score (ERS) to produce an "
        "ACCEPT, REVIEW, or DEFER routing decision.\n\n"
    )


    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    f.write(
        "2. DATASET AND TASK\n"
    )

    f.write(
        "------------------------------------------------------------\n"
    )

    f.write(
        "Dataset: MIMIC-IV Clinical Database DEMO v2.2\n"
    )

    f.write(
        "Task: ICU mortality prediction\n"
    )

    f.write(
        "Outcome: Death during the same ICU stay\n"
    )

    f.write(
        "Index time: ICU admission\n"
    )

    f.write(
        "Feature window: First 6 hours after ICU admission\n"
    )

    f.write(
        "Split: Patient-level train / validation / test split\n"
    )

    f.write(
        f"Test cases: {int(test_cases)}\n\n"
    )


    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------

    f.write(
        "3. MODEL PIPELINE\n"
    )

    f.write(
        "------------------------------------------------------------\n"
    )

    f.write(
        "Primary prediction model: XGBoost\n"
    )

    f.write(
        "Baseline models: Logistic Regression and Random Forest\n"
    )

    f.write(
        "Calibration: Sigmoid / Platt-style calibration\n"
    )

    f.write(
        "Explanation method: SHAP\n"
    )

    f.write(
        "Reliability signals: stability, faithfulness, ranking "
        "consistency, and sign consistency\n\n"
    )


    # --------------------------------------------------------
    # ROUTING
    # --------------------------------------------------------

    f.write(
        "4. ROUTING RESULTS\n"
    )

    f.write(
        "------------------------------------------------------------\n\n"
    )


    f.write(
        "CONFIDENCE-ONLY ROUTING\n"
    )

    f.write(
        f"ACCEPT : {confidence_accept}\n"
    )

    f.write(
        f"REVIEW : {confidence_review_count}\n"
    )

    f.write(
        f"DEFER  : {confidence_defer_count}\n"
    )

    f.write(
        f"ACCEPT coverage : "
        f"{confidence_coverage:.4f}\n"
    )

    f.write(
        f"Accepted error : "
        f"{confidence_error:.4f}\n\n"
    )


    f.write(
        "MEDTRUST ROUTING\n"
    )

    f.write(
        f"ACCEPT : {medtrust_accept}\n"
    )

    f.write(
        f"REVIEW : {medtrust_review_count}\n"
    )

    f.write(
        f"DEFER  : {medtrust_defer_count}\n"
    )

    f.write(
        f"ACCEPT coverage : "
        f"{medtrust_coverage:.4f}\n"
    )

    f.write(
        f"Accepted error : "
        f"{medtrust_error:.4f}\n\n"
    )


    # --------------------------------------------------------
    # ROUTING CHANGES
    # --------------------------------------------------------

    f.write(
        "5. ROUTING CHANGES\n"
    )

    f.write(
        "------------------------------------------------------------\n"
    )

    f.write(
        f"Changed decisions: "
        f"{int(routing_changes)}\n"
    )

    f.write(
        f"Percentage changed: "
        f"{routing_change_percentage:.2f}%\n"
    )

    f.write(
        f"High-confidence / low-ERS cases: "
        f"{int(high_confidence_low_ers)}\n\n"
    )


    # --------------------------------------------------------
    # RELIABILITY
    # --------------------------------------------------------

    f.write(
        "6. RELIABILITY SIGNALS\n"
    )

    f.write(
        "------------------------------------------------------------\n"
    )

    f.write(
        f"Average predictive confidence: "
        f"{avg_confidence:.4f}\n"
    )

    f.write(
        f"Average ERS: "
        f"{avg_ers:.4f}\n"
    )

    f.write(
        f"Average joint reliability: "
        f"{avg_joint:.4f}\n\n"
    )


    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    f.write(
        "7. INTERPRETATION\n"
    )

    f.write(
        "------------------------------------------------------------\n"
    )

    f.write(
        "The demo experiment demonstrates the routing mechanism "
        "by combining predictive confidence and explanation "
        "reliability.\n\n"
    )

    f.write(
        "The comparison shows how adding ERS can change routing "
        "decisions for cases where predictive confidence is high "
        "but explanation reliability is comparatively low.\n\n"
    )

    f.write(
        "These results are intended as a methodological "
        "demonstration on the MIMIC-IV DEMO dataset, not as "
        "clinical performance evidence.\n\n"
    )


    # --------------------------------------------------------
    # LIMITATIONS
    # --------------------------------------------------------

    f.write(
        "8. IMPORTANT LIMITATIONS\n"
    )

    f.write(
        "------------------------------------------------------------\n"
    )

    f.write(
        "1. The MIMIC-IV DEMO dataset is very small.\n"
    )

    f.write(
        "2. The number of ICU mortality events is very small.\n"
    )

    f.write(
        "3. ERS components are prototype reliability measures "
        "and require stronger validation.\n"
    )

    f.write(
        "4. Calibration estimates are unstable on a tiny dataset.\n"
    )

    f.write(
        "5. The current experiment does not establish clinical "
        "utility or safety.\n"
    )

    f.write(
        "6. Final research experiments should use the full "
        "MIMIC-IV dataset.\n"
    )

    f.write(
        "7. External validation using eICU-CRD remains required.\n"
    )

    f.write(
        "8. Temporal and out-of-distribution evaluation remains "
        "required.\n\n"
    )


    # --------------------------------------------------------
    # NEXT RESEARCH PHASE
    # --------------------------------------------------------

    f.write(
        "9. NEXT RESEARCH PHASE\n"
    )

    f.write(
        "------------------------------------------------------------\n"
    )

    f.write(
        "The next phase should rebuild the feature table from "
        "raw MIMIC-IV data without global pre-split imputation.\n\n"
    )

    f.write(
        "Imputation must be fitted only on the training data "
        "and then applied to validation and test data.\n\n"
    )

    f.write(
        "The full MIMIC-IV cohort should then be used to perform "
        "the final prediction, calibration, explanation reliability, "
        "threshold selection, and routing experiments.\n\n"
    )

    f.write(
        "Afterward, the frozen pipeline should be evaluated on "
        "eICU-CRD for external validation.\n\n"
    )


    f.write(
        "============================================================\n"
    )

    f.write(
        "END OF DEMO REPORT\n"
    )

    f.write(
        "============================================================\n"
    )


# ============================================================
# 6. PRINT SUMMARY
# ============================================================

print(
    "Report created successfully!"
)

print(
    "\nLocation:"
)

print(
    report_path
)

print(
    "\nTest cases:",
    int(test_cases)
)

print(
    "\nConfidence-only:"
)

print(
    "  ACCEPT:",
    confidence_accept
)

print(
    "  REVIEW:",
    confidence_review_count
)

print(
    "  DEFER:",
    confidence_defer_count
)

print(
    "\nMedTrust:"
)

print(
    "  ACCEPT:",
    medtrust_accept
)

print(
    "  REVIEW:",
    medtrust_review_count
)

print(
    "  DEFER:",
    medtrust_defer_count
)

print(
    "\nRouting changes:",
    int(routing_changes),
    f"({routing_change_percentage:.2f}%)"
)

print(
    "\n=========================================="
)

print(
    "STEP 26 COMPLETE"
)

print(
    "=========================================="
)