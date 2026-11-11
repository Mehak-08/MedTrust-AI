import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from docx import Document
from docx.shared import Inches

print("\n==========================================")
print("MEDTRUST AI - FINAL REPORT GENERATOR")
print("==========================================\n")


# =================================================
# PROJECT PATH
# =================================================

PROJECT_PATH = os.path.dirname(__file__)


# =================================================
# FILE PATHS
# =================================================

routing_file = os.path.join(
    PROJECT_PATH,
    "medtrust_routing_results.csv"
)

metrics_file = os.path.join(
    PROJECT_PATH,
    "final_medtrust_metrics.csv"
)

model_file = os.path.join(
    PROJECT_PATH,
    "model_comparison.csv"
)


# =================================================
# LOAD DATA
# =================================================

print("Loading project results...")


routing_df = pd.read_csv(
    routing_file
)

metrics_df = pd.read_csv(
    metrics_file
)


if os.path.exists(model_file):

    model_df = pd.read_csv(
        model_file
    )

else:

    model_df = pd.DataFrame()


print(
    "Routing records:",
    len(routing_df)
)


# =================================================
# OUTPUT FOLDER
# =================================================

output_folder = os.path.join(
    PROJECT_PATH,
    "final_results"
)

os.makedirs(
    output_folder,
    exist_ok=True
)


# =================================================
# EXTRACT METRICS
# =================================================

def get_metric(name):

    row = metrics_df[
        metrics_df["metric"] == name
    ]

    if len(row) == 0:
        return np.nan

    return row.iloc[0]["value"]


accuracy = get_metric(
    "Accuracy"
)

precision = get_metric(
    "Precision"
)

recall = get_metric(
    "Recall"
)

f1 = get_metric(
    "F1"
)

auroc = get_metric(
    "AUROC"
)

sensitivity = get_metric(
    "Sensitivity"
)

specificity = get_metric(
    "Specificity"
)

brier = get_metric(
    "Brier Score"
)

avg_confidence = get_metric(
    "Average Confidence"
)

avg_ers = get_metric(
    "Average ERS"
)

avg_joint = get_metric(
    "Average Joint Reliability"
)

accept_count = get_metric(
    "ACCEPT Count"
)

review_count = get_metric(
    "REVIEW Count"
)

defer_count = get_metric(
    "DEFER Count"
)

coverage = get_metric(
    "Acceptance Coverage"
)

accepted_error = get_metric(
    "Accepted Error Rate"
)


# =================================================
# CHART 1
# ROUTING DISTRIBUTION
# =================================================

print("\nCreating routing chart...")


routing_counts = (
    routing_df["routing_decision"]
    .value_counts()
)


plt.figure(
    figsize=(8, 5)
)

routing_counts.plot(
    kind="bar"
)

plt.title(
    "MedTrust AI Routing Distribution"
)

plt.xlabel(
    "Routing Decision"
)

plt.ylabel(
    "Number of Patients"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()


routing_chart = os.path.join(
    output_folder,
    "01_routing_distribution.png"
)

plt.savefig(
    routing_chart,
    dpi=300
)

plt.close()


# =================================================
# CHART 2
# CONFIDENCE VS ERS
# =================================================

print(
    "Creating confidence vs ERS chart..."
)


plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    routing_df[
        "calibrated_confidence"
    ],
    routing_df[
        "ERS"
    ]
)

plt.axvline(
    0.85,
    linestyle="--"
)

plt.axhline(
    0.80,
    linestyle="--"
)

plt.xlabel(
    "Calibrated Predictive Confidence"
)

plt.ylabel(
    "Explanation Reliability Score (ERS)"
)

plt.title(
    "Predictive Confidence vs Explanation Reliability"
)

plt.tight_layout()


confidence_ers_chart = os.path.join(
    output_folder,
    "02_confidence_vs_ers.png"
)

plt.savefig(
    confidence_ers_chart,
    dpi=300
)

plt.close()


# =================================================
# CHART 3
# JOINT RELIABILITY
# =================================================

print(
    "Creating joint reliability chart..."
)


plt.figure(
    figsize=(9, 5)
)

plt.bar(
    range(
        len(routing_df)
    ),
    routing_df[
        "joint_reliability"
    ]
)

plt.axhline(
    0.85,
    linestyle="--"
)

plt.axhline(
    0.70,
    linestyle="--"
)

plt.xlabel(
    "Test Patient"
)

plt.ylabel(
    "Joint Reliability"
)

plt.title(
    "MedTrust AI Joint Reliability Scores"
)

plt.tight_layout()


joint_chart = os.path.join(
    output_folder,
    "03_joint_reliability.png"
)

plt.savefig(
    joint_chart,
    dpi=300
)

plt.close()


# =================================================
# CHART 4
# ERS DISTRIBUTION
# =================================================

print(
    "Creating ERS distribution..."
)


plt.figure(
    figsize=(8, 5)
)

plt.hist(
    routing_df["ERS"],
    bins=8
)

plt.xlabel(
    "Explanation Reliability Score"
)

plt.ylabel(
    "Number of Patients"
)

plt.title(
    "Distribution of Explanation Reliability"
)

plt.tight_layout()


ers_chart = os.path.join(
    output_folder,
    "04_ers_distribution.png"
)

plt.savefig(
    ers_chart,
    dpi=300
)

plt.close()


# =================================================
# CHART 5
# MODEL COMPARISON
# =================================================

if len(model_df) > 0:

    print(
        "Creating model comparison chart..."
    )

    if "AUROC" in model_df.columns:

        plt.figure(
            figsize=(9, 5)
        )

        plt.bar(
            model_df.iloc[:, 0],
            model_df["AUROC"]
        )

        plt.ylabel(
            "AUROC"
        )

        plt.xlabel(
            "Model"
        )

        plt.title(
            "Model Comparison"
        )

        plt.xticks(
            rotation=20
        )

        plt.tight_layout()

        model_chart = os.path.join(
            output_folder,
            "05_model_comparison.png"
        )

        plt.savefig(
            model_chart,
            dpi=300
        )

        plt.close()

    else:

        model_chart = None

else:

    model_chart = None


# =================================================
# CREATE FINAL RESULTS TABLE
# =================================================

print(
    "\nCreating final results table..."
)


final_results = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "AUROC",
            "Sensitivity",
            "Specificity",
            "Brier Score",
            "Average Predictive Confidence",
            "Average Explanation Reliability",
            "Average Joint Reliability",
            "ACCEPT Count",
            "REVIEW Count",
            "DEFER Count",
            "Acceptance Coverage",
            "Accepted Error Rate"
        ],

        "Value": [
            accuracy,
            precision,
            recall,
            f1,
            auroc,
            sensitivity,
            specificity,
            brier,
            avg_confidence,
            avg_ers,
            avg_joint,
            accept_count,
            review_count,
            defer_count,
            coverage,
            accepted_error
        ]
    }
)


final_results_file = os.path.join(
    output_folder,
    "final_results_table.csv"
)

final_results.to_csv(
    final_results_file,
    index=False
)


# =================================================
# ROUTING TABLE
# =================================================

routing_summary = pd.DataFrame(
    {
        "Decision": [
            "ACCEPT",
            "REVIEW",
            "DEFER"
        ],

        "Count": [
            accept_count,
            review_count,
            defer_count
        ],

        "Percentage": [
            accept_count / len(routing_df) * 100,
            review_count / len(routing_df) * 100,
            defer_count / len(routing_df) * 100
        ]
    }
)


routing_summary_file = os.path.join(
    output_folder,
    "routing_summary.csv"
)

routing_summary.to_csv(
    routing_summary_file,
    index=False
)


# =================================================
# CREATE RESEARCH SUMMARY TXT
# =================================================

print(
    "Creating research summary..."
)


summary_text = f"""
MEDTRUST AI
Reliability-Aware Clinical Decision Routing
============================================

PROJECT OVERVIEW
----------------

MedTrust AI is a research framework designed to investigate
whether predictive confidence alone is sufficient to determine
whether a clinical AI prediction should be accepted.

The framework combines:

1. Predictive confidence
2. Explanation reliability

The combined signals are used to route predictions into:

ACCEPT
REVIEW
DEFER


DATASET AND TASK
----------------

Dataset:
MIMIC-IV Clinical Database Demo v2.2

Task:
ICU mortality prediction

Feature window:
First 6 hours after ICU admission

Prediction target:
Death during the same ICU stay


MODEL PIPELINE
--------------

Baseline models:

- Logistic Regression
- Random Forest

Primary model:

- XGBoost

Explanation method:

- SHAP TreeExplainer

Explanation reliability prototype:

- SHAP stability under small controlled perturbations


FINAL DEMO RESULTS
------------------

Number of test patients:
{len(routing_df)}

Accuracy:
{accuracy:.4f}

Precision:
{precision:.4f}

Recall:
{recall:.4f}

F1:
{f1:.4f}

AUROC:
{auroc:.4f}

Sensitivity:
{sensitivity:.4f}

Specificity:
{specificity:.4f}

Brier Score:
{brier:.4f}


RELIABILITY RESULTS
-------------------

Average predictive confidence:
{avg_confidence:.4f}

Average explanation reliability:
{avg_ers:.4f}

Average joint reliability:
{avg_joint:.4f}


ROUTING RESULTS
---------------

ACCEPT:
{accept_count}

REVIEW:
{review_count}

DEFER:
{defer_count}

Acceptance coverage:
{coverage:.4f}

Accepted error rate:
{accepted_error:.4f}


INTERPRETATION
--------------

The prototype demonstrates how predictive confidence and
explanation stability can be combined into a reliability-aware
routing mechanism.

The current demonstration produced mostly ACCEPT decisions,
with a smaller number of REVIEW decisions.

No DEFER decisions were produced under the current prototype
thresholds.


IMPORTANT LIMITATIONS
---------------------

1. The demonstration uses the MIMIC-IV clinical database demo,
   which is very small compared with the full MIMIC-IV dataset.

2. Only a small number of ICU deaths are present in the demo.

3. Therefore the reported performance values are pipeline
   validation results and should NOT be interpreted as
   clinical performance claims.

4. The current explanation reliability score is a prototype
   based on SHAP stability.

5. A complete ERS should additionally investigate
   faithfulness, ranking consistency, and sign consistency.

6. The current ACCEPT/REVIEW/DEFER thresholds are prototype
   thresholds.

7. Final research thresholds should be selected using
   validation data and frozen before test evaluation.

8. Final research should use patient-level splitting,
   leakage-safe preprocessing, temporal validation, and
   external validation on eICU-CRD.


RESEARCH CONTRIBUTION
---------------------

The central research contribution being investigated is the
use of explanation reliability as an additional signal beyond
predictive confidence for clinical decision routing.

The intended routing policy is:

High confidence + reliable explanation
        -> ACCEPT

Uncertain or conflicting reliability signals
        -> REVIEW

Low overall reliability
        -> DEFER


FUTURE WORK
-----------

1. Run the pipeline on full MIMIC-IV.

2. Freeze the exact cohort and outcome definition.

3. Use patient-level and temporal splits.

4. Fit preprocessing only on training data.

5. Improve the Explanation Reliability Score.

6. Add explanation faithfulness.

7. Add feature ranking consistency.

8. Add sign consistency.

9. Select routing thresholds using validation data.

10. Evaluate selective risk.

11. Test temporal robustness.

12. Perform external validation on eICU-CRD.

13. Compare confidence-only routing against
    confidence + explanation reliability routing.


CONCLUSION
----------

The current prototype successfully implements the main
MedTrust AI pipeline from prediction to explanation and
reliability-aware routing.

The demo establishes the technical workflow but does not
establish clinical effectiveness. Larger-scale validation is
required before making clinical or general performance claims.
"""


summary_file = os.path.join(
    output_folder,
    "MedTrust_AI_Research_Summary.txt"
)

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        summary_text
    )


# =================================================
# CREATE DOCX REPORT
# =================================================

print(
    "Creating Word research report..."
)


document = Document()


document.add_heading(
    "MedTrust AI",
    level=0
)

document.add_paragraph(
    "Reliability-Aware Clinical Decision Routing"
)


document.add_heading(
    "1. Project Overview",
    level=1
)

document.add_paragraph(
    "MedTrust AI investigates whether predictive confidence "
    "alone is sufficient for determining whether a clinical "
    "AI prediction should be accepted. The framework combines "
    "predictive confidence with explanation reliability to "
    "support ACCEPT, REVIEW, and DEFER routing."
)


document.add_heading(
    "2. Dataset and Task",
    level=1
)

document.add_paragraph(
    "The current prototype uses the MIMIC-IV Clinical "
    "Database Demo v2.2 and performs ICU mortality prediction "
    "using information from the first six hours after ICU admission."
)


document.add_heading(
    "3. Model Pipeline",
    level=1
)

document.add_paragraph(
    "Logistic Regression and Random Forest were used as "
    "baseline models. XGBoost was used as the primary "
    "tree-based model. SHAP TreeExplainer was used to "
    "generate feature-level explanations."
)


document.add_heading(
    "4. Explanation Reliability",
    level=1
)

document.add_paragraph(
    f"The prototype measured explanation stability under "
    f"small controlled perturbations. The average explanation "
    f"reliability score in the current test set was "
    f"{avg_ers:.4f}."
)


document.add_heading(
    "5. Reliability-Aware Routing",
    level=1
)

document.add_paragraph(
    f"The average predictive confidence was "
    f"{avg_confidence:.4f}, while the average joint reliability "
    f"was {avg_joint:.4f}."
)


document.add_paragraph(
    f"Routing produced {int(accept_count)} ACCEPT decisions, "
    f"{int(review_count)} REVIEW decisions, and "
    f"{int(defer_count)} DEFER decisions."
)


document.add_heading(
    "6. Prediction Results",
    level=1
)

table = document.add_table(
    rows=1,
    cols=2
)

table.style = "Table Grid"

table.rows[0].cells[0].text = "Metric"
table.rows[0].cells[1].text = "Value"


for metric, value in zip(
    final_results["Metric"],
    final_results["Value"]
):

    row = table.add_row()

    row.cells[0].text = str(
        metric
    )

    if isinstance(
        value,
        (float, np.floating)
    ):

        row.cells[1].text = f"{value:.4f}"

    else:

        row.cells[1].text = str(
            value
        )


document.add_heading(
    "7. Research Limitations",
    level=1
)

document.add_paragraph(
    "The current results are based on the MIMIC-IV demo "
    "dataset and a very small test set. They should therefore "
    "be interpreted as pipeline-development results rather "
    "than clinical effectiveness evidence."
)


document.add_paragraph(
    "Before final research evaluation, preprocessing should "
    "be leakage-safe, patient-level splitting should be used, "
    "and temporal and external validation should be performed."
)


document.add_heading(
    "8. Future Research",
    level=1
)

future_points = [
    "Run the pipeline on full MIMIC-IV.",
    "Use patient-level and temporal validation.",
    "Improve the Explanation Reliability Score.",
    "Add explanation faithfulness.",
    "Measure feature-ranking consistency.",
    "Measure sign consistency.",
    "Select routing thresholds using validation data.",
    "Evaluate selective risk.",
    "Perform external validation on eICU-CRD.",
    "Compare confidence-only routing with joint confidence and explanation reliability routing."
]


for point in future_points:

    document.add_paragraph(
        point,
        style="List Bullet"
    )


# -------------------------------------------------
# ADD CHARTS
# -------------------------------------------------

document.add_heading(
    "9. Visual Results",
    level=1
)


if os.path.exists(
    routing_chart
):

    document.add_picture(
        routing_chart,
        width=Inches(6)
    )


if os.path.exists(
    confidence_ers_chart
):

    document.add_picture(
        confidence_ers_chart,
        width=Inches(6)
    )


if os.path.exists(
    ers_chart
):

    document.add_picture(
        ers_chart,
        width=Inches(6)
    )


docx_file = os.path.join(
    output_folder,
    "MedTrust_AI_Final_Report.docx"
)

document.save(
    docx_file
)


# =================================================
# CREATE PDF
# =================================================

print(
    "Creating PDF report..."
)


try:

    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        Image
    )

    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet


    pdf_file = os.path.join(
        output_folder,
        "MedTrust_AI_Final_Report.pdf"
    )


    styles = getSampleStyleSheet()


    pdf = SimpleDocTemplate(
        pdf_file,
        pagesize=A4
    )


    story = []


    story.append(
        Paragraph(
            "MedTrust AI",
            styles["Title"]
        )
    )

    story.append(
        Paragraph(
            "Reliability-Aware Clinical Decision Routing",
            styles["Heading2"]
        )
    )

    story.append(
        Spacer(
            1,
            12
        )
    )


    story.append(
        Paragraph(
            summary_text.replace(
                "\n",
                "<br/>"
            ),
            styles["BodyText"]
        )
    )


    story.append(
        Spacer(
            1,
            15
        )
    )


    story.append(
        Paragraph(
            "Final Metrics",
            styles["Heading2"]
        )
    )


    pdf_table_data = [
        [
            "Metric",
            "Value"
        ]
    ]


    for metric, value in zip(
        final_results["Metric"],
        final_results["Value"]
    ):

        if isinstance(
            value,
            (float, np.floating)
        ):

            value_text = f"{value:.4f}"

        else:

            value_text = str(
                value
            )


        pdf_table_data.append(
            [
                str(metric),
                value_text
            ]
        )


    pdf_table = Table(
        pdf_table_data
    )


    pdf_table.setStyle(
        TableStyle(
            [
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey
                )
            ]
        )
    )


    story.append(
        pdf_table
    )


    story.append(
        Spacer(
            1,
            15
        )
    )


    if os.path.exists(
        routing_chart
    ):

        story.append(
            Image(
                routing_chart,
                width=450,
                height=280
            )
        )


    story.append(
        Spacer(
            1,
            15
        )
    )


    if os.path.exists(
        confidence_ers_chart
    ):

        story.append(
            Image(
                confidence_ers_chart,
                width=450,
                height=330
            )
        )


    pdf.build(
        story
    )


    pdf_created = True


except Exception as error:

    pdf_created = False

    print(
        "\nPDF creation warning:"
    )

    print(
        error
    )


# =================================================
# FINAL OUTPUT
# =================================================

print("\n==========================================")
print("FINAL MEDTRUST PACKAGE COMPLETE")
print("==========================================\n")


print(
    "Results folder:"
)

print(
    output_folder
)


print(
    "\nCreated files:"
)


print(
    "01_routing_distribution.png"
)

print(
    "02_confidence_vs_ers.png"
)

print(
    "03_joint_reliability.png"
)

print(
    "04_ers_distribution.png"
)

if model_chart is not None:

    print(
        "05_model_comparison.png"
    )


print(
    "final_results_table.csv"
)

print(
    "routing_summary.csv"
)

print(
    "MedTrust_AI_Research_Summary.txt"
)

print(
    "MedTrust_AI_Final_Report.docx"
)


if pdf_created:

    print(
        "MedTrust_AI_Final_Report.pdf"
    )


print("\n==========================================")
print("MEDTRUST AI PIPELINE FINISHED")
print("==========================================")

print(
    "\nIMPORTANT:"
)

print(
    "Current results are from the MIMIC-IV DEMO dataset."
)

print(
    "They are for pipeline validation, not clinical claims."
)

print(
    "\nFor the final research version:"
)

print(
    "Full MIMIC-IV -> patient-level split -> "
    "leakage-safe preprocessing -> validation thresholds -> "
    "test evaluation -> eICU external validation."
)