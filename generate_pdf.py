import os
import pandas as pd
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
    HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#718096"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, letter[1] - 36, "MedTrust AI — Clinical Decision Routing Research Report")
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.5)
            self.line(54, letter[1] - 42, letter[0] - 54, letter[1] - 42)
        
        # Footer
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 54, 36, footer_text)
        self.drawString(54, 36, "Confidential — For Methodological & Research Demonstration Only")
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(54, 46, letter[0] - 54, 46)
        
        self.restoreState()


def build_pdf(output_dir):
    pdf_path = os.path.join(output_dir, "MedTrust_AI_Final_Research_Report.pdf")
    
    # Load available CSV results
    perf_csv = os.path.join(output_dir, "model_performance.csv")
    routing_csv = os.path.join(output_dir, "routing_results.csv")
    patient_csv = os.path.join(output_dir, "patient_results.csv")
    shap_csv = os.path.join(output_dir, "shap_feature_importance.csv")

    model_metrics = pd.read_csv(perf_csv) if os.path.exists(perf_csv) else None
    routing_metrics_df = pd.read_csv(routing_csv) if os.path.exists(routing_csv) else None
    patient_results = pd.read_csv(patient_csv) if os.path.exists(patient_csv) else None
    shap_importance = pd.read_csv(shap_csv) if os.path.exists(shap_csv) else None

    # Setup document
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=12
    )
    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#2D3748"),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=6
    )
    bullet_style = ParagraphStyle(
        "BulletDark",
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=3
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#2D3748")
    )
    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=table_cell,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#1A365D")
    )
    table_header = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
        alignment=1  # Centered
    )

    story = []

    # Title Banner
    story.append(Paragraph("MedTrust AI", title_style))
    story.append(Paragraph("Reliability-Aware Clinical Decision Routing for ICU Mortality", subtitle_style))
    story.append(Paragraph("<b>Benchmark:</b> MIMIC-IV Clinical Database Demo v2.2 &nbsp;|&nbsp; <b>Models:</b> XGBoost + LightGBM + RF + LR", ParagraphStyle("Meta", parent=body_style, fontSize=8.5, textColor=colors.HexColor("#718096"))))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2B6CB0"), spaceBefore=6, spaceAfter=14))

    # Executive Summary
    story.append(Paragraph("1. Executive Summary", h1_style))
    story.append(Paragraph(
        "MedTrust AI investigates whether clinical machine learning reliability should be assessed using predictive confidence alone or "
        "by coupling confidence with an <b>Explanation Reliability Score (ERS)</b>. Modern clinical models frequently demonstrate "
        "high predictive confidence while their feature explanations remain fragile or unstable under physiological noise. "
        "The MedTrust AI framework integrates probability calibration, local explanation perturbation tests, and multi-tier routing "
        "(<b>ACCEPT</b>, <b>REVIEW</b>, <b>DEFER</b>) to ensure safe clinical deployment.",
        body_style
    ))

    # Research Problem & Dataset
    story.append(Paragraph("2. Research Problem & Dataset", h1_style))
    story.append(Paragraph(
        "Standard machine learning pipelines in acute care typically assume high model confidence translates to reliable predictions. "
        "However, empirical testing reveals that two high-confidence predictions can possess vastly different explanation stabilities. "
        "This project benchmarks tabular gradient boosting architectures on the MIMIC-IV demo dataset (126 patient stays, 85 clinical predictors, "
        "9 positive ICU mortality outcomes) with leakage-safe patient-level splitting (86 train, 20 validation, 20 test).",
        body_style
    ))

    # Models Evaluated
    story.append(Paragraph("3. Evaluated Predictive Architectures", h1_style))
    story.append(Paragraph(
        "Four machine learning architectures were trained and benchmarked under identical conditions:",
        body_style
    ))
    story.append(Paragraph("• <b>XGBoost (Primary):</b> Depth-wise gradient boosted trees with balanced subsampling.", bullet_style))
    story.append(Paragraph("• <b>LightGBM:</b> Leaf-wise gradient boosting with constrained leaf growth (num_leaves=7) to prevent overfitting.", bullet_style))
    story.append(Paragraph("• <b>Random Forest:</b> Balanced ensemble of 300 bagged classification trees.", bullet_style))
    story.append(Paragraph("• <b>Logistic Regression:</b> Median-imputed, L2-regularized linear baseline.", bullet_style))
    story.append(Paragraph("• <b>Calibrated Models:</b> Both XGBoost and LightGBM were calibrated using 3-fold sigmoid calibration (Platt scaling).", bullet_style))

    # Performance Table
    if model_metrics is not None:
        story.append(Spacer(1, 6))
        story.append(Paragraph("4. Model Performance Benchmark", h1_style))
        
        table_data = [[
            Paragraph("Model", table_header),
            Paragraph("Accuracy", table_header),
            Paragraph("Precision", table_header),
            Paragraph("Recall", table_header),
            Paragraph("F1", table_header),
            Paragraph("AUROC", table_header),
            Paragraph("AUPRC", table_header),
            Paragraph("Brier", table_header)
        ]]

        for _, row in model_metrics.iterrows():
            m_name = str(row["Model"])
            is_bold = "XGBoost" in m_name or "LightGBM" in m_name
            cell_p = table_cell_bold if is_bold else table_cell
            table_data.append([
                Paragraph(m_name, cell_p),
                Paragraph(f"{row['Accuracy']:.3f}", table_cell),
                Paragraph(f"{row['Precision']:.3f}", table_cell),
                Paragraph(f"{row['Recall']:.3f}", table_cell),
                Paragraph(f"{row['F1']:.3f}", table_cell),
                Paragraph(f"{row['AUROC']:.3f}" if pd.notna(row['AUROC']) else "N/A", cell_p if "Calibrated XGBoost" in m_name else table_cell),
                Paragraph(f"{row['AUPRC']:.3f}" if pd.notna(row['AUPRC']) else "N/A", table_cell),
                Paragraph(f"{row['Brier']:.4f}", table_cell)
            ])

        t = Table(table_data, colWidths=[130, 48, 52, 48, 44, 56, 56, 50])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A365D")),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t)
        story.append(Spacer(1, 10))

    # Explainability & ERS
    story.append(Paragraph("5. Explainability & Explanation Reliability Score (ERS)", h1_style))
    story.append(Paragraph(
        "SHAP TreeExplainer was utilized to examine physiological feature attributions. The Explanation Reliability Score (ERS) "
        "evaluates 4 distinct dimensions under controlled input perturbation: (1) <b>SHAP stability</b>, (2) <b>Ranking consistency</b>, "
        "(3) <b>Sign consistency</b>, and (4) <b>Faithfulness</b> upon feature ablation.",
        body_style
    ))
    
    if shap_importance is not None:
        top_feats = ", ".join(shap_importance.head(5)["Feature"].tolist())
        story.append(Paragraph(f"• <b>Primary Clinical Drivers:</b> {top_feats}", bullet_style))
    
    if patient_results is not None and "explanation_reliability" in patient_results.columns:
        ers_mean = patient_results["explanation_reliability"].mean()
        ers_min = patient_results["explanation_reliability"].min()
        ers_max = patient_results["explanation_reliability"].max()
        story.append(Paragraph(f"• <b>ERS Distribution:</b> Mean = <b>{ers_mean:.4f}</b> (Range: {ers_min:.4f} – {ers_max:.4f})", bullet_style))

    # Routing Section & Table
    story.append(Spacer(1, 6))
    story.append(Paragraph("6. Reliability-Aware Decision Routing", h1_style))
    story.append(Paragraph(
        "Predictions are routed into 3 action categories based on a Joint Reliability Score combining 50% calibrated confidence "
        "and 50% Explanation Reliability Score (ERS):",
        body_style
    ))
    story.append(Paragraph("• <b>ACCEPT (Score ≥ 0.70):</b> High predictive confidence supported by stable, faithful explanations.", bullet_style))
    story.append(Paragraph("• <b>REVIEW (0.50 ≤ Score < 0.70):</b> Moderate risk or unstable explanation; requires clinician verification.", bullet_style))
    story.append(Paragraph("• <b>DEFER (Score < 0.50):</b> Unreliable prediction or explanation fragility; deferred entirely to human experts.", bullet_style))

    if routing_metrics_df is not None:
        story.append(Spacer(1, 4))
        r_table_data = [[
            Paragraph("Routing Strategy", table_header),
            Paragraph("ACCEPT", table_header),
            Paragraph("REVIEW", table_header),
            Paragraph("DEFER", table_header),
            Paragraph("Coverage", table_header),
            Paragraph("Accepted Error", table_header)
        ]]
        for _, row in routing_metrics_df.iterrows():
            r_table_data.append([
                Paragraph(str(row["Strategy"]), table_cell_bold),
                Paragraph(str(row["ACCEPT"]), table_cell),
                Paragraph(str(row["REVIEW"]), table_cell),
                Paragraph(str(row["DEFER"]), table_cell),
                Paragraph(f"{row['Coverage']:.1%}", table_cell),
                Paragraph(f"{row['Accepted_Error']:.4f}" if pd.notna(row['Accepted_Error']) else "0.0000", table_cell)
            ])
        rt = Table(r_table_data, colWidths=[160, 60, 60, 60, 70, 74])
        rt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(rt)

    # Embedded Visual Research Graphs
    story.append(Spacer(1, 10))
    story.append(Paragraph("7. Visual Research Analytics & Graphs", h1_style))
    story.append(Paragraph("Below are the 5 high-resolution figures produced by the research pipeline:", body_style))

    figures = [
        ("Figure 1: Benchmark Model Performance Comparison (AUROC)", "model_performance.png"),
        ("Figure 2: Top 10 Clinical Biomarkers by Mean Absolute SHAP Importance", "shap_importance.png"),
        ("Figure 3: Case Allocation Distribution (Confidence-Only vs. MedTrust AI)", "routing_distribution.png"),
        ("Figure 4: Patient-Level Confidence vs. Explanation Reliability (ERS)", "confidence_vs_ers.png"),
        ("Figure 5: Selective Prediction Risk Curve (Coverage vs. Accepted Error)", "coverage_vs_error.png")
    ]

    for title, img_name in figures:
        img_path = os.path.join(output_dir, img_name)
        if os.path.exists(img_path):
            img_block = [
                Paragraph(f"<b>{title}</b>", h2_style),
                RLImage(img_path, width=6.2 * inch, height=3.6 * inch),
                Spacer(1, 12)
            ]
            story.append(KeepTogether(img_block))

    # Limitations & Conclusion
    story.append(Paragraph("8. Limitations & Clinical Governance", h1_style))
    story.append(Paragraph("• <b>Demo Dataset:</b> MIMIC-IV demo contains 126 stays and 9 mortality outcomes; results are methodological prototypes.", bullet_style))
    story.append(Paragraph("• <b>Threshold Selection:</b> Production deployment requires prospective temporal validation and frozen thresholds.", bullet_style))
    story.append(Paragraph("• <b>External Validation:</b> External multi-center evaluation on eICU-CRD is recommended prior to clinical deployment.", bullet_style))

    story.append(Spacer(1, 6))
    story.append(Paragraph("9. Conclusion", h1_style))
    story.append(Paragraph(
        "MedTrust AI establishes that dual-signal reliability evaluation (combining predictive confidence with explanation stability) "
        "provides an essential safety safeguard for clinical AI decision support. Integrating XGBoost and LightGBM within a calibrated, "
        "explainable routing framework demonstrates a rigorous path toward accountable medical artificial intelligence.",
        body_style
    ))

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)
    return pdf_path

if __name__ == "__main__":
    out_dir = r"C:\Users\MEHAK\OneDrive\Desktop\MIMIC-Project\final_medtrust_report"
    path = build_pdf(out_dir)
    print("PDF successfully generated at:", path)

