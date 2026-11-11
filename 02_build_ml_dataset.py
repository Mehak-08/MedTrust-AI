import os
import pandas as pd


# ============================================================
# MEDTRUST AI
# Step 2: Build ML Dataset
# ============================================================

print("\n==========================================")
print("MEDTRUST AI - BUILDING ML DATASET")
print("==========================================\n")


# ============================================================
# 1. PATHS
# ============================================================

BASE_PATH = r"C:\mimic dataset\mimic-iv-clinical-database-demo-2.2"

HOSPITAL_PATH = os.path.join(BASE_PATH, "hosp")
ICU_PATH = os.path.join(BASE_PATH, "icu")


# ============================================================
# 2. LOAD BASIC DATA
# ============================================================

print("Loading patients...")

patients = pd.read_csv(
    os.path.join(HOSPITAL_PATH, "patients.csv.gz")
)

print("Patients loaded:", patients.shape)


print("\nLoading admissions...")

admissions = pd.read_csv(
    os.path.join(HOSPITAL_PATH, "admissions.csv.gz")
)

print("Admissions loaded:", admissions.shape)


print("\nLoading ICU stays...")

icustays = pd.read_csv(
    os.path.join(ICU_PATH, "icustays.csv.gz")
)

print("ICU stays loaded:", icustays.shape)


# ============================================================
# 3. LOAD CHART EVENTS
# ============================================================

print("\nLoading chart events...")

chartevents = pd.read_csv(
    os.path.join(ICU_PATH, "chartevents.csv.gz")
)

print("Chart events loaded:", chartevents.shape)


# ============================================================
# 4. LOAD LAB EVENTS
# ============================================================

print("\nLoading lab events...")

labevents = pd.read_csv(
    os.path.join(HOSPITAL_PATH, "labevents.csv.gz")
)

print("Lab events loaded:", labevents.shape)


# ============================================================
# 5. CONVERT TIME COLUMNS
# ============================================================

print("\nConverting time columns...")

icustays["intime"] = pd.to_datetime(icustays["intime"])
icustays["outtime"] = pd.to_datetime(icustays["outtime"])

admissions["admittime"] = pd.to_datetime(admissions["admittime"])
admissions["dischtime"] = pd.to_datetime(admissions["dischtime"])
admissions["deathtime"] = pd.to_datetime(
    admissions["deathtime"],
    errors="coerce"
)

chartevents["charttime"] = pd.to_datetime(
    chartevents["charttime"]
)

labevents["charttime"] = pd.to_datetime(
    labevents["charttime"]
)


# ============================================================
# 6. CREATE ICU MORTALITY LABEL
# ============================================================

print("\nCreating ICU mortality label...")

icu_data = icustays.merge(
    admissions[
        [
            "subject_id",
            "hadm_id",
            "deathtime",
            "hospital_expire_flag"
        ]
    ],
    on=["subject_id", "hadm_id"],
    how="left"
)

icu_data["icu_mortality"] = (
    icu_data["deathtime"].notna()
    &
    (icu_data["deathtime"] >= icu_data["intime"])
    &
    (icu_data["deathtime"] <= icu_data["outtime"])
).astype(int)


print("\nICU mortality distribution:")

print(
    icu_data["icu_mortality"].value_counts()
)


# ============================================================
# 7. CREATE 6-HOUR FEATURE WINDOW
# ============================================================

print("\nCreating first 6-hour window...")

icu_data["window_end"] = (
    icu_data["intime"]
    + pd.Timedelta(hours=6)
)


# ============================================================
# 8. KEEP ONLY ICU STAYS THAT HAVE A FULL 6-HOUR WINDOW
# ============================================================

# For the first version, we require the ICU stay
# to last at least 6 hours.

icu_data = icu_data[
    icu_data["outtime"] >= icu_data["window_end"]
].copy()


print(
    "ICU stays after requiring 6-hour window:",
    len(icu_data)
)


# ============================================================
# 9. VITAL SIGN ITEM IDs
# ============================================================

vital_items = {

    "heart_rate": 220045,

    "sbp": 220179,

    "dbp": 220180,

    "mbp": 220181,

    "resp_rate": 220210,

    "spo2": 220277,

    "temperature": 223762
}


# ============================================================
# 10. FILTER CHART EVENTS
# ============================================================

print("\nFiltering vital-sign data...")

selected_vital_ids = list(vital_items.values())

vitals = chartevents[
    chartevents["itemid"].isin(selected_vital_ids)
].copy()


# ============================================================
# 11. MERGE VITALS WITH ICU WINDOWS
# ============================================================

print("Matching vital signs to ICU stays...")

vitals = vitals.merge(
    icu_data[
        [
            "stay_id",
            "intime",
            "window_end"
        ]
    ],
    on="stay_id",
    how="inner"
)


# Keep only measurements from first 6 hours.

vitals = vitals[
    (vitals["charttime"] >= vitals["intime"])
    &
    (vitals["charttime"] <= vitals["window_end"])
].copy()


# ============================================================
# 12. CONVERT VITAL VALUES TO NUMERIC
# ============================================================

vitals["valuenum"] = pd.to_numeric(
    vitals["valuenum"],
    errors="coerce"
)


# Remove rows without numeric values.

vitals = vitals[
    vitals["valuenum"].notna()
].copy()


print(
    "Vital measurements in 6-hour window:",
    len(vitals)
)


# ============================================================
# 13. MAP ITEM IDs TO FEATURE NAMES
# ============================================================

reverse_vital_items = {
    value: key
    for key, value in vital_items.items()
}


vitals["feature"] = vitals["itemid"].map(
    reverse_vital_items
)


# ============================================================
# 14. CREATE VITAL FEATURES
# ============================================================

# For each patient/ICU stay and vital:
# use mean, minimum and maximum.

vital_features = (
    vitals
    .groupby(["stay_id", "feature"])["valuenum"]
    .agg(["mean", "min", "max"])
    .reset_index()
)


# Convert long format to wide format.

vital_features = vital_features.pivot(
    index="stay_id",
    columns="feature"
)


# Flatten column names.

vital_features.columns = [
    f"{feature}_{stat}"
    for stat, feature in vital_features.columns
]


vital_features = vital_features.reset_index()


print("\nVital feature table:")
print(vital_features.head())


# ============================================================
# 15. IMPORTANT LAB ITEM IDs
# ============================================================

# We use the frequently occurring blood chemistry /
# hematology items found during dataset exploration.

lab_items = {

    "potassium": 50971,

    "sodium": 50983,

    "creatinine": 50912,

    "chloride": 50902,

    "bun": 51006,

    "hematocrit": 51221,

    "bicarbonate": 50882,

    "anion_gap": 50868,

    "platelets": 51265,

    "hemoglobin": 51222,

    "wbc": 51301,

    "glucose": 50931,

    "magnesium": 50960,

    "calcium": 50893,

    "phosphate": 50970,

    "ast": 50878,

    "bilirubin_total": 50885,

    "alt": 50861,

    "alkaline_phosphatase": 50863,

    "ph": 50820
}


# ============================================================
# 16. FILTER LAB EVENTS
# ============================================================

print("\nFiltering laboratory data...")

selected_lab_ids = list(lab_items.values())


labs = labevents[
    labevents["itemid"].isin(selected_lab_ids)
].copy()


print(
    "Selected lab measurements:",
    len(labs)
)


# ============================================================
# 17. MERGE LABS WITH ICU WINDOWS
# ============================================================

print("Matching laboratory values to ICU stays...")

labs = labs.merge(
    icu_data[
        [
            "subject_id",
            "hadm_id",
            "stay_id",
            "intime",
            "window_end"
        ]
    ],
    on=["subject_id", "hadm_id"],
    how="inner"
)


# ============================================================
# 18. KEEP FIRST 6 HOURS
# ============================================================

labs = labs[
    (labs["charttime"] >= labs["intime"])
    &
    (labs["charttime"] <= labs["window_end"])
].copy()


# ============================================================
# 19. CONVERT LAB VALUES TO NUMERIC
# ============================================================

labs["valuenum"] = pd.to_numeric(
    labs["valuenum"],
    errors="coerce"
)


labs = labs[
    labs["valuenum"].notna()
].copy()


print(
    "Lab measurements in 6-hour window:",
    len(labs)
)


# ============================================================
# 20. MAP LAB IDs TO FEATURE NAMES
# ============================================================

reverse_lab_items = {
    value: key
    for key, value in lab_items.items()
}


labs["feature"] = labs["itemid"].map(
    reverse_lab_items
)


# ============================================================
# 21. CREATE LAB FEATURES
# ============================================================

lab_features = (
    labs
    .groupby(["stay_id", "feature"])["valuenum"]
    .agg(["mean", "min", "max"])
    .reset_index()
)


lab_features = lab_features.pivot(
    index="stay_id",
    columns="feature"
)


lab_features.columns = [
    f"{feature}_{stat}"
    for stat, feature in lab_features.columns
]


lab_features = lab_features.reset_index()


print("\nLab feature table:")
print(lab_features.head())


# ============================================================
# 22. ADD PATIENT DEMOGRAPHICS
# ============================================================

print("\nAdding patient demographics...")

demographics = patients[
    [
        "subject_id",
        "gender",
        "anchor_age"
    ]
].copy()


# ============================================================
# 23. MERGE DEMOGRAPHICS
# ============================================================

icu_data = icu_data.merge(
    demographics,
    on="subject_id",
    how="left"
)


# ============================================================
# 24. MERGE VITAL FEATURES
# ============================================================

ml_data = icu_data.merge(
    vital_features,
    on="stay_id",
    how="left"
)


# ============================================================
# 25. MERGE LAB FEATURES
# ============================================================

ml_data = ml_data.merge(
    lab_features,
    on="stay_id",
    how="left"
)


# ============================================================
# 26. SELECT SAFE ML COLUMNS
# ============================================================

# IMPORTANT:
#
# We deliberately DO NOT include:
#
# outtime
# window_end
# los
# deathtime
# hospital_expire_flag
#
# because these contain outcome/future information.

base_columns = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "gender",
    "anchor_age",
    "icu_mortality"
]


feature_columns = [
    column
    for column in ml_data.columns
    if (
        column.endswith("_mean")
        or column.endswith("_min")
        or column.endswith("_max")
    )
]


final_columns = base_columns + feature_columns


ml_data = ml_data[
    final_columns
].copy()


# ============================================================
# 27. CHECK DATASET BEFORE MISSING VALUE HANDLING
# ============================================================

print("\n==========================================")
print("ML DATASET BEFORE MISSING VALUE HANDLING")
print("==========================================")

print(
    "Shape:",
    ml_data.shape
)

print("\nColumns:")

for column in ml_data.columns:
    print(column)


print("\nTarget distribution:")

print(
    ml_data["icu_mortality"].value_counts()
)


# ============================================================
# 28. MISSING VALUE REPORT
# ============================================================

print("\n==========================================")
print("MISSING VALUE REPORT")
print("==========================================")

missing_report = (
    ml_data.isna()
    .sum()
    .sort_values(ascending=False)
)


missing_report = missing_report[
    missing_report > 0
]


print(missing_report)


# ============================================================
# 29. HANDLE DEMOGRAPHIC MISSING VALUES
# ============================================================

# Age should normally be present.
# Gender should normally be present.
#
# We use simple safe handling for categorical gender.

ml_data["gender"] = ml_data["gender"].fillna("Unknown")


# ============================================================
# 30. ONE-HOT ENCODE GENDER
# ============================================================

ml_data = pd.get_dummies(
    ml_data,
    columns=["gender"],
    drop_first=False
)


# ============================================================
# 31. HANDLE NUMERIC MISSING VALUES
# ============================================================

# At this stage, many lab/vital values can be missing because
# a test was not performed during the first 6 hours.
#
# For this first reproducible pipeline:
# use median imputation calculated from THIS dataset.

numeric_columns = ml_data.select_dtypes(
    include=["number"]
).columns


numeric_feature_columns = [
    column
    for column in numeric_columns
    if column not in [
        "subject_id",
        "hadm_id",
        "stay_id",
        "icu_mortality"
    ]
]


for column in numeric_feature_columns:

    median_value = ml_data[column].median()

    ml_data[column] = ml_data[column].fillna(
        median_value
    )


# ============================================================
# 32. FINAL DATASET CHECK
# ============================================================

print("\n==========================================")
print("FINAL ML DATASET")
print("==========================================")

print(
    "Final shape:",
    ml_data.shape
)


print("\nRemaining missing values:")

remaining_missing = (
    ml_data.isna()
    .sum()
)

remaining_missing = remaining_missing[
    remaining_missing > 0
]

print(remaining_missing)


print("\nFinal target distribution:")

print(
    ml_data["icu_mortality"].value_counts()
)


# ============================================================
# 33. SAVE DATASET
# ============================================================

OUTPUT_FILE = os.path.join(
    os.path.dirname(__file__),
    "mimic_ml_dataset.csv"
)


ml_data.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n==========================================")
print("DATASET SAVED SUCCESSFULLY")
print("==========================================")

print(
    "Saved to:"
)

print(
    OUTPUT_FILE
)


print("\nFirst 5 rows:")

print(
    ml_data.head()
)


print("\n==========================================")
print("STEP 2 COMPLETE")
print("==========================================")

print(
    "\nNext step:"
)

print(
    "We will inspect the generated ML dataset"
)

print(
    "before training any machine-learning model."
)