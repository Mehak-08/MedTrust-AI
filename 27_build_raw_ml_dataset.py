# ============================================================
# MEDTRUST AI
# STEP 27 - BUILD RAW 6-HOUR ML DATASET
# ============================================================
#
# IMPORTANT:
# - NO imputation is performed here.
# - Missing values remain NaN.
# - Imputation will be fitted ONLY on training data later.
# - This fixes the earlier global-imputation problem.
# ============================================================

import os
import pandas as pd
import numpy as np


print("\n==========================================")
print("MEDTRUST AI - RAW 6-HOUR DATASET")
print("==========================================\n")


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = r"C:\mimic dataset\mimic-iv-clinical-database-demo-2.2"

PROJECT_DIR = r"C:\Users\MEHAK\OneDrive\Desktop\MIMIC-Project"

ICU_FILE = os.path.join(
    BASE_DIR,
    "icu",
    "icustays.csv.gz"
)

PATIENT_FILE = os.path.join(
    BASE_DIR,
    "hosp",
    "patients.csv.gz"
)

ADMISSIONS_FILE = os.path.join(
    BASE_DIR,
    "hosp",
    "admissions.csv.gz"
)

CHARTEVENTS_FILE = os.path.join(
    BASE_DIR,
    "icu",
    "chartevents.csv.gz"
)

LABEVENTS_FILE = os.path.join(
    BASE_DIR,
    "hosp",
    "labevents.csv.gz"
)

D_ITEMS_FILE = os.path.join(
    BASE_DIR,
    "icu",
    "d_items.csv.gz"
)

D_LABITEMS_FILE = os.path.join(
    BASE_DIR,
    "hosp",
    "d_labitems.csv.gz"
)

OUTPUT_FILE = os.path.join(
    PROJECT_DIR,
    "mimic_raw_ml_dataset.csv"
)


# ============================================================
# 2. CHECK FILES
# ============================================================

required_files = [
    ICU_FILE,
    PATIENT_FILE,
    ADMISSIONS_FILE,
    CHARTEVENTS_FILE,
    LABEVENTS_FILE,
    D_ITEMS_FILE,
    D_LABITEMS_FILE
]


for file_path in required_files:

    if not os.path.exists(file_path):

        print("ERROR: File not found:")
        print(file_path)

        raise FileNotFoundError(file_path)


print("All required MIMIC files found.\n")


# ============================================================
# 3. LOAD BASIC TABLES
# ============================================================

print("Loading ICU stays...")

icu = pd.read_csv(
    ICU_FILE,
    compression="gzip"
)

print(
    "ICU stays:",
    len(icu)
)


print("\nLoading patients...")

patients = pd.read_csv(
    PATIENT_FILE,
    compression="gzip"
)

print(
    "Patients:",
    len(patients)
)


print("\nLoading admissions...")

admissions = pd.read_csv(
    ADMISSIONS_FILE,
    compression="gzip"
)

print(
    "Admissions:",
    len(admissions)
)


print("\nLoading item dictionaries...")

d_items = pd.read_csv(
    D_ITEMS_FILE,
    compression="gzip"
)

d_labitems = pd.read_csv(
    D_LABITEMS_FILE,
    compression="gzip"
)


# ============================================================
# 4. PREPARE ICU COHORT
# ============================================================

print("\nPreparing ICU cohort...")


# Convert times

icu["intime"] = pd.to_datetime(
    icu["intime"]
)

icu["outtime"] = pd.to_datetime(
    icu["outtime"]
)


admissions["deathtime"] = pd.to_datetime(
    admissions["deathtime"]
)


# Keep first ICU stay for each hospital admission

icu = (
    icu
    .sort_values(
        [
            "subject_id",
            "hadm_id",
            "intime"
        ]
    )
    .drop_duplicates(
        subset=["subject_id", "hadm_id"],
        keep="first"
    )
)


# ============================================================
# 5. CREATE ICU MORTALITY LABEL
# ============================================================

print("Creating ICU mortality label...")


cohort = icu[
    [
        "subject_id",
        "hadm_id",
        "stay_id",
        "first_careunit",
        "intime",
        "outtime"
    ]
].copy()


cohort = cohort.merge(
    admissions[
        [
            "subject_id",
            "hadm_id",
            "deathtime"
        ]
    ],
    on=[
        "subject_id",
        "hadm_id"
    ],
    how="left"
)


cohort["icu_mortality"] = (
    cohort["deathtime"].notna()
    &
    (
        cohort["deathtime"]
        >=
        cohort["intime"]
    )
    &
    (
        cohort["deathtime"]
        <=
        cohort["outtime"]
    )
).astype(int)


print(
    "\nICU mortality:"
)

print(
    cohort["icu_mortality"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 6. KEEP STAYS WITH AT LEAST 6 HOURS
# ============================================================

cohort["stay_hours"] = (
    (
        cohort["outtime"]
        -
        cohort["intime"]
    )
    .dt.total_seconds()
    /
    3600
)


cohort = cohort[
    cohort["stay_hours"] >= 6
].copy()


print(
    "\nStays >= 6 hours:",
    len(cohort)
)


# ============================================================
# 7. LOAD CHART EVENTS
# ============================================================

print("\nLoading chart events...")

charts = pd.read_csv(
    CHARTEVENTS_FILE,
    compression="gzip",
    usecols=[
        "subject_id",
        "stay_id",
        "charttime",
        "itemid",
        "valuenum"
    ]
)


charts["charttime"] = pd.to_datetime(
    charts["charttime"]
)


# ============================================================
# 8. LOAD LAB EVENTS
# ============================================================

print("Loading laboratory events...")

labs = pd.read_csv(
    LABEVENTS_FILE,
    compression="gzip",
    usecols=[
        "subject_id",
        "hadm_id",
        "charttime",
        "itemid",
        "valuenum"
    ]
)


labs["charttime"] = pd.to_datetime(
    labs["charttime"]
)


# ============================================================
# 9. DEFINE VITALS
# ============================================================

vital_items = {

    220045: "heart_rate",

    220210: "resp_rate",

    220277: "spo2",

    220179: "sbp",

    220180: "dbp",

    220181: "mbp",

    223762: "temperature"
}


# ============================================================
# 10. FILTER VITALS
# ============================================================

charts = charts[
    charts["itemid"].isin(
        vital_items.keys()
    )
].copy()


charts["feature"] = (
    charts["itemid"]
    .map(vital_items)
)


# ============================================================
# 11. MERGE ICU TIME
# ============================================================

charts = charts.merge(
    cohort[
        [
            "subject_id",
            "stay_id",
            "intime"
        ]
    ],
    on=[
        "subject_id",
        "stay_id"
    ],
    how="inner"
)


# ============================================================
# 12. FIRST 6 HOURS ONLY
# ============================================================

charts["hours_from_intime"] = (
    (
        charts["charttime"]
        -
        charts["intime"]
    )
    .dt.total_seconds()
    /
    3600
)


charts = charts[
    (
        charts["hours_from_intime"] >= 0
    )
    &
    (
        charts["hours_from_intime"] <= 6
    )
].copy()


charts = charts[
    charts["valuenum"].notna()
].copy()


print(
    "Vital measurements in first 6h:",
    len(charts)
)


# ============================================================
# 13. AGGREGATE VITALS
# ============================================================

vital_features = (
    charts
    .groupby(
        [
            "subject_id",
            "stay_id",
            "feature"
        ]
    )["valuenum"]
    .agg(
        [
            "mean",
            "min",
            "max"
        ]
    )
    .reset_index()
)


vital_features["feature_mean"] = (
    vital_features["feature"]
    +
    "_mean"
)


vital_features["feature_min"] = (
    vital_features["feature"]
    +
    "_min"
)


vital_features["feature_max"] = (
    vital_features["feature"]
    +
    "_max"
)


vital_mean = vital_features.pivot(
    index=[
        "subject_id",
        "stay_id"
    ],
    columns="feature_mean",
    values="mean"
)


vital_min = vital_features.pivot(
    index=[
        "subject_id",
        "stay_id"
    ],
    columns="feature_min",
    values="min"
)


vital_max = vital_features.pivot(
    index=[
        "subject_id",
        "stay_id"
    ],
    columns="feature_max",
    values="max"
)


vital_mean.columns.name = None
vital_min.columns.name = None
vital_max.columns.name = None


# ============================================================
# 14. DEFINE LABS
# ============================================================

lab_items = {

    50971: "potassium",

    50983: "sodium",

    50912: "creatinine",

    50902: "chloride",

    51006: "bun",

    51221: "hematocrit",

    50882: "bicarbonate",

    50868: "anion_gap",

    51265: "platelets",

    51222: "hemoglobin",

    51301: "wbc",

    50931: "glucose",

    50960: "magnesium",

    50893: "calcium",

    50970: "phosphate",

    50878: "ast",

    50885: "bilirubin",

    50861: "alt",

    50863: "alk_phos",

    50820: "ph"
}


# ============================================================
# 15. FILTER LABS
# ============================================================

labs = labs[
    labs["itemid"].isin(
        lab_items.keys()
    )
].copy()


labs["feature"] = (
    labs["itemid"]
    .map(lab_items)
)


# ============================================================
# 16. MERGE ICU TIME
# ============================================================

labs = labs.merge(
    cohort[
        [
            "subject_id",
            "hadm_id",
            "intime"
        ]
    ],
    on=[
        "subject_id",
        "hadm_id"
    ],
    how="inner"
)


# ============================================================
# 17. FIRST 6 HOURS ONLY
# ============================================================

labs["hours_from_intime"] = (
    (
        labs["charttime"]
        -
        labs["intime"]
    )
    .dt.total_seconds()
    /
    3600
)


labs = labs[
    (
        labs["hours_from_intime"] >= 0
    )
    &
    (
        labs["hours_from_intime"] <= 6
    )
].copy()


labs = labs[
    labs["valuenum"].notna()
].copy()


print(
    "Lab measurements in first 6h:",
    len(labs)
)


# ============================================================
# 18. AGGREGATE LABS
# ============================================================

lab_features = (
    labs
    .groupby(
        [
            "subject_id",
            "hadm_id",
            "feature"
        ]
    )["valuenum"]
    .agg(
        [
            "mean",
            "min",
            "max"
        ]
    )
    .reset_index()
)


lab_features["feature_mean"] = (
    lab_features["feature"]
    +
    "_mean"
)


lab_features["feature_min"] = (
    lab_features["feature"]
    +
    "_min"
)


lab_features["feature_max"] = (
    lab_features["feature"]
    +
    "_max"
)


lab_mean = lab_features.pivot(
    index=[
        "subject_id",
        "hadm_id"
    ],
    columns="feature_mean",
    values="mean"
)


lab_min = lab_features.pivot(
    index=[
        "subject_id",
        "hadm_id"
    ],
    columns="feature_min",
    values="min"
)


lab_max = lab_features.pivot(
    index=[
        "subject_id",
        "hadm_id"
    ],
    columns="feature_max",
    values="max"
)


lab_mean.columns.name = None
lab_min.columns.name = None
lab_max.columns.name = None


# ============================================================
# 19. MERGE ALL FEATURES
# ============================================================

print("\nCombining features...")


dataset = cohort[
    [
        "subject_id",
        "hadm_id",
        "stay_id",
        "intime",
        "icu_mortality"
    ]
].copy()


dataset = dataset.merge(
    vital_mean.reset_index(),
    on=[
        "subject_id",
        "stay_id"
    ],
    how="left"
)


dataset = dataset.merge(
    vital_min.reset_index(),
    on=[
        "subject_id",
        "stay_id"
    ],
    how="left"
)


dataset = dataset.merge(
    vital_max.reset_index(),
    on=[
        "subject_id",
        "stay_id"
    ],
    how="left"
)


dataset = dataset.merge(
    lab_mean.reset_index(),
    on=[
        "subject_id",
        "hadm_id"
    ],
    how="left"
)


dataset = dataset.merge(
    lab_min.reset_index(),
    on=[
        "subject_id",
        "hadm_id"
    ],
    how="left"
)


dataset = dataset.merge(
    lab_max.reset_index(),
    on=[
        "subject_id",
        "hadm_id"
    ],
    how="left"
)


# ============================================================
# 20. ADD DEMOGRAPHICS
# ============================================================

demo = patients[
    [
        "subject_id",
        "gender",
        "anchor_age"
    ]
].copy()


dataset = dataset.merge(
    demo,
    on="subject_id",
    how="left"
)


# ============================================================
# 21. ONE-HOT ENCODE GENDER
# ============================================================

dataset = pd.get_dummies(
    dataset,
    columns=["gender"],
    prefix="gender",
    dummy_na=True
)


# ============================================================
# 22. REMOVE NON-PREDICTIVE / LEAKAGE COLUMNS
# ============================================================

dataset = dataset.drop(
    columns=[
        "intime"
    ],
    errors="ignore"
)


# ============================================================
# 23. ENSURE NUMERIC FEATURES
# ============================================================

for column in dataset.columns:

    if column in [
        "subject_id",
        "hadm_id",
        "stay_id",
        "icu_mortality"
    ]:
        continue

    dataset[column] = pd.to_numeric(
        dataset[column],
        errors="coerce"
    )


# ============================================================
# 24. CHECK TARGET
# ============================================================

dataset = dataset[
    dataset["icu_mortality"].notna()
].copy()


dataset["icu_mortality"] = (
    dataset["icu_mortality"]
    .astype(int)
)


# ============================================================
# 25. REMOVE DUPLICATES
# ============================================================

dataset = dataset.drop_duplicates(
    subset=["stay_id"]
)


# ============================================================
# 26. SORT
# ============================================================

dataset = dataset.sort_values(
    "stay_id"
).reset_index(
    drop=True
)


# ============================================================
# 27. IMPORTANT:
#     DO NOT IMPUTE
# ============================================================

print(
    "\nIMPORTANT:"
)

print(
    "No missing-value imputation was performed."
)

print(
    "Missing values remain NaN."
)


# ============================================================
# 28. MISSINGNESS SUMMARY
# ============================================================

feature_columns = [
    c
    for c in dataset.columns
    if c not in [
        "subject_id",
        "hadm_id",
        "stay_id",
        "icu_mortality"
    ]
]


missing_summary = pd.DataFrame({

    "feature": feature_columns,

    "missing_count": [
        dataset[c].isna().sum()
        for c in feature_columns
    ],

    "missing_percentage": [
        dataset[c].isna().mean() * 100
        for c in feature_columns
    ]
})


missing_summary = (
    missing_summary
    .sort_values(
        "missing_percentage",
        ascending=False
    )
)


missing_summary.to_csv(
    os.path.join(
        PROJECT_DIR,
        "step27_missingness_summary.csv"
    ),
    index=False
)


# ============================================================
# 29. SAVE RAW DATASET
# ============================================================

dataset.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 30. PRINT SUMMARY
# ============================================================

print("\n==========================================")
print("RAW DATASET CREATED")
print("==========================================")

print(
    "\nDataset shape:",
    dataset.shape
)

print(
    "\nPatients:",
    dataset["subject_id"].nunique()
)

print(
    "ICU stays:",
    dataset["stay_id"].nunique()
)

print(
    "\nOutcome distribution:"
)

print(
    dataset[
        "icu_mortality"
    ].value_counts()
    .sort_index()
)


print(
    "\nTotal missing values:",
    int(
        dataset[
            feature_columns
        ].isna().sum().sum()
    )
)


print(
    "\nOutput:"
)

print(
    OUTPUT_FILE
)

print(
    "\nMissingness summary:"
)

print(
    os.path.join(
        PROJECT_DIR,
        "step27_missingness_summary.csv"
    )
)


print("\n==========================================")
print("STEP 27 COMPLETE")
print("==========================================")