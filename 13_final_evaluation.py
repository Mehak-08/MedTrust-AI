import os
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    brier_score_loss
)

print("\n==========================================")
print("MEDTRUST AI - FINAL EVALUATION")
print("==========================================\n")

PROJECT_PATH = os.path.dirname(__file__)

# -------------------------------------------------
# LOAD RESULTS
# -------------------------------------------------

results_file = os.path.join(
    PROJECT_PATH,
    "medtrust_routing_results.csv"
)

df = pd.read_csv(results_file)

print("Patients evaluated:", len(df))


# -------------------------------------------------
# BASIC PREDICTION METRICS
# -------------------------------------------------

y_true = df["icu_mortality"]

y_pred = df["predicted_class"]

probabilities = df["calibrated_probability"]


accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

brier = brier_score_loss(
    y_true,
    probabilities
)


# -------------------------------------------------
# AUROC
# -------------------------------------------------

try:

    auroc = roc_auc_score(
        y_true,
        probabilities
    )

except ValueError:

    auroc = np.nan


# -------------------------------------------------
# CONFUSION MATRIX
# -------------------------------------------------

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()


# -------------------------------------------------
# SENSITIVITY / SPECIFICITY
# -------------------------------------------------

if (tp + fn) > 0:

    sensitivity = (
        tp /
        (tp + fn)
    )

else:

    sensitivity = np.nan


if (tn + fp) > 0:

    specificity = (
        tn /
        (tn + fp)
    )

else:

    specificity = np.nan


# -------------------------------------------------
# ROUTING METRICS
# -------------------------------------------------

accepted = df[
    df["routing_decision"] == "ACCEPT"
]

reviewed = df[
    df["routing_decision"] == "REVIEW"
]

deferred = df[
    df["routing_decision"] == "DEFER"
]


coverage = (
    len(accepted) /
    len(df)
)


if len(accepted) > 0:

    accepted_error = (
        1 -
        accepted["prediction_correct"].mean()
    )

else:

    accepted_error = np.nan


# -------------------------------------------------
# AVERAGE RELIABILITY
# -------------------------------------------------

average_confidence = (
    df["calibrated_confidence"].mean()
)

average_ers = (
    df["ERS"].mean()
)

average_joint = (
    df["joint_reliability"].mean()
)


# -------------------------------------------------
# HIGH-CONFIDENCE LOW-ERS
# -------------------------------------------------

mismatch = df[
    (
        df["calibrated_confidence"] >= 0.85
    )
    &
    (
        df["ERS"] < 0.80
    )
]


# -------------------------------------------------
# PRINT RESULTS
# -------------------------------------------------

print("\n==========================================")
print("PREDICTION PERFORMANCE")
print("==========================================\n")

print(
    f"Accuracy:      {accuracy:.4f}"
)

print(
    f"Precision:     {precision:.4f}"
)

print(
    f"Recall:        {recall:.4f}"
)

print(
    f"F1 Score:      {f1:.4f}"
)

print(
    f"AUROC:         {auroc:.4f}"
)

print(
    f"Sensitivity:   {sensitivity:.4f}"
)

print(
    f"Specificity:   {specificity:.4f}"
)

print(
    f"Brier Score:   {brier:.4f}"
)


print("\n==========================================")
print("CONFUSION MATRIX")
print("==========================================\n")

print(
    "TN:",
    tn
)

print(
    "FP:",
    fp
)

print(
    "FN:",
    fn
)

print(
    "TP:",
    tp
)


print("\n==========================================")
print("RELIABILITY RESULTS")
print("==========================================\n")

print(
    f"Average confidence: "
    f"{average_confidence:.4f}"
)

print(
    f"Average ERS: "
    f"{average_ers:.4f}"
)

print(
    f"Average joint reliability: "
    f"{average_joint:.4f}"
)


print("\n==========================================")
print("ROUTING RESULTS")
print("==========================================\n")

print(
    "ACCEPT:",
    len(accepted)
)

print(
    "REVIEW:",
    len(reviewed)
)

print(
    "DEFER:",
    len(deferred)
)

print(
    f"Acceptance coverage: "
    f"{coverage:.4f}"
)

print(
    f"Accepted error rate: "
    f"{accepted_error:.4f}"
)

print(
    "High-confidence / low-ERS cases:",
    len(mismatch)
)


# -------------------------------------------------
# SAVE FINAL METRICS
# -------------------------------------------------

metrics = pd.DataFrame(
    {
        "metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "AUROC",
            "Sensitivity",
            "Specificity",
            "Brier Score",
            "Average Confidence",
            "Average ERS",
            "Average Joint Reliability",
            "ACCEPT Count",
            "REVIEW Count",
            "DEFER Count",
            "Acceptance Coverage",
            "Accepted Error Rate",
            "High Confidence Low ERS"
        ],

        "value": [
            accuracy,
            precision,
            recall,
            f1,
            auroc,
            sensitivity,
            specificity,
            brier,
            average_confidence,
            average_ers,
            average_joint,
            len(accepted),
            len(reviewed),
            len(deferred),
            coverage,
            accepted_error,
            len(mismatch)
        ]
    }
)


metrics_file = os.path.join(
    PROJECT_PATH,
    "final_medtrust_metrics.csv"
)

metrics.to_csv(
    metrics_file,
    index=False
)


# -------------------------------------------------
# SAVE CONFUSION MATRIX
# -------------------------------------------------

confusion_df = pd.DataFrame(
    cm,
    index=[
        "Actual_0",
        "Actual_1"
    ],
    columns=[
        "Predicted_0",
        "Predicted_1"
    ]
)

confusion_file = os.path.join(
    PROJECT_PATH,
    "confusion_matrix.csv"
)

confusion_df.to_csv(
    confusion_file
)


# -------------------------------------------------
# FINAL MESSAGE
# -------------------------------------------------

print("\n==========================================")
print("STEP 13 COMPLETE")
print("==========================================\n")

print(
    "Final metrics saved to:"
)

print(
    metrics_file
)

print(
    "\nConfusion matrix saved to:"
)

print(
    confusion_file
)

print(
    "\nNEXT:"
)

print(
    "Generate final charts and research report."
)