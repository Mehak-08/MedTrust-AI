import os
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

mimic_path = r"C:\mimic dataset\mimic-iv-clinical-database-demo-2.2"

hosp_path = os.path.join(mimic_path, "hosp")
icu_path = os.path.join(mimic_path, "icu")


# ============================================================
# 2. HELPER FUNCTION
# ============================================================

def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# 3. LOAD PATIENTS
# ============================================================

print_section("PATIENTS")

patients = pd.read_csv(
    os.path.join(hosp_path, "patients.csv.gz")
)

print("Columns:")
print(patients.columns.tolist())

print("\nShape:", patients.shape)

print("\nFirst 5 rows:")
print(patients.head())

print("\nMissing values:")
print(patients.isna().sum())


# ============================================================
# 4. LOAD ICU STAYS
# ============================================================

print_section("ICU STAYS")

icustays = pd.read_csv(
    os.path.join(icu_path, "icustays.csv.gz")
)

print("Columns:")
print(icustays.columns.tolist())

print("\nShape:", icustays.shape)

print("\nFirst 5 rows:")
print(icustays.head())

print("\nMissing values:")
print(icustays.isna().sum())


# ============================================================
# 5. LOAD ADMISSIONS
# ============================================================

print_section("ADMISSIONS")

admissions = pd.read_csv(
    os.path.join(hosp_path, "admissions.csv.gz")
)

print("Columns:")
print(admissions.columns.tolist())

print("\nShape:", admissions.shape)

print("\nFirst 5 rows:")
print(admissions.head())

print("\nMissing values:")
print(admissions.isna().sum())


# ============================================================
# 6. LOAD CHART EVENTS
# ============================================================

print_section("CHART EVENTS")

chartevents = pd.read_csv(
    os.path.join(icu_path, "chartevents.csv.gz")
)

print("Columns:")
print(chartevents.columns.tolist())

print("\nShape:", chartevents.shape)

print("\nMissing values:")
print(chartevents.isna().sum())


# ============================================================
# 7. LOAD LAB EVENTS
# ============================================================

print_section("LAB EVENTS")

labevents = pd.read_csv(
    os.path.join(hosp_path, "labevents.csv.gz")
)

print("Columns:")
print(labevents.columns.tolist())

print("\nShape:", labevents.shape)

print("\nMissing values:")
print(labevents.isna().sum())


# ============================================================
# 8. LOAD ITEM DICTIONARY
# ============================================================

print_section("D_ITEMS")

d_items = pd.read_csv(
    os.path.join(icu_path, "d_items.csv.gz")
)

print("Shape:", d_items.shape)

print("\nFirst 10 rows:")
print(d_items.head(10))


# ============================================================
# 9. LOAD LAB ITEM DICTIONARY
# ============================================================

print_section("D_LABITEMS")

d_labitems = pd.read_csv(
    os.path.join(hosp_path, "d_labitems.csv.gz")
)

print("Shape:", d_labitems.shape)

print("\nFirst 20 rows:")
print(d_labitems.head(20))


# ============================================================
# 10. VITAL SIGN ITEM IDS
# ============================================================

print_section("VITAL SIGN ITEM IDS")

vital_itemids = [
    220045,  # Heart Rate
    220050,  # Arterial BP systolic
    220051,  # Arterial BP diastolic
    220052,  # Arterial BP mean
    220179,  # NIBP systolic
    220180,  # NIBP diastolic
    220181,  # NIBP mean
    220210,  # Respiratory Rate
    220277,  # O2 saturation
    223762   # Temperature Celsius
]

vital_info = d_items[
    d_items["itemid"].isin(vital_itemids)
][
    ["itemid", "label", "unitname"]
].sort_values("itemid")

print(vital_info.to_string(index=False))


# ============================================================
# 11. ACTUAL VITAL DATA COUNTS
# ============================================================

print_section("ACTUAL VITAL DATA COUNTS")

vital_counts = (
    chartevents[
        chartevents["itemid"].isin(vital_itemids)
    ]
    .groupby("itemid")
    .size()
    .reset_index(name="count")
)

vital_counts = vital_counts.merge(
    vital_info,
    on="itemid",
    how="left"
)

vital_counts = vital_counts[
    ["itemid", "label", "unitname", "count"]
].sort_values(
    "count",
    ascending=False
)

print(vital_counts.to_string(index=False))


# ============================================================
# 12. IMPORTANT LAB ITEM SEARCH
# ============================================================

print_section("IMPORTANT LAB ITEM IDS")

lab_keywords = {
    "Lactate": "lactate",
    "Creatinine": "creatinine",
    "Glucose": "glucose",
    "Sodium": "sodium",
    "Potassium": "potassium",
    "Chloride": "chloride",
    "Hemoglobin": "hemoglobin",
    "Hematocrit": "hematocrit",
    "Platelet": "platelet",
    "White Blood": "white blood",
    "WBC": "wbc",
    "Bilirubin": "bilirubin",
    "Albumin": "albumin",
    "Calcium": "calcium"
}

for name, keyword in lab_keywords.items():

    print("\n---", name, "---")

    result = d_labitems[
        d_labitems["label"]
        .str.contains(keyword, case=False, na=False)
    ]

    if len(result) > 0:
        print(
            result[
                ["itemid", "label", "fluid", "category"]
            ].to_string(index=False)
        )
    else:
        print("No matching items found.")


# ============================================================
# 13. MOST FREQUENT LABS
# ============================================================

print_section("LAB ITEM FREQUENCIES")

lab_frequency = (
    labevents
    .groupby("itemid")
    .size()
    .reset_index(name="count")
)

lab_frequency = lab_frequency.merge(
    d_labitems[
        ["itemid", "label", "fluid", "category"]
    ],
    on="itemid",
    how="left"
)

lab_frequency = lab_frequency.sort_values(
    "count",
    ascending=False
)

print(
    lab_frequency.head(40).to_string(index=False)
)


# ============================================================
# 14. PREPARE DATES
# ============================================================

print_section("DATE PREPARATION")

icustays["intime"] = pd.to_datetime(
    icustays["intime"],
    errors="coerce"
)

icustays["outtime"] = pd.to_datetime(
    icustays["outtime"],
    errors="coerce"
)

admissions["admittime"] = pd.to_datetime(
    admissions["admittime"],
    errors="coerce"
)

admissions["dischtime"] = pd.to_datetime(
    admissions["dischtime"],
    errors="coerce"
)

admissions["deathtime"] = pd.to_datetime(
    admissions["deathtime"],
    errors="coerce"
)


# ============================================================
# 15. MERGE ICU + ADMISSIONS
# ============================================================

print_section("ICU + ADMISSIONS MERGE")

icu_admissions = icustays.merge(
    admissions[
        [
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "deathtime",
            "hospital_expire_flag"
        ]
    ],
    on=["subject_id", "hadm_id"],
    how="left"
)

print("Merged shape:", icu_admissions.shape)

print("\nFirst 5 rows:")
print(
    icu_admissions[
        [
            "subject_id",
            "hadm_id",
            "stay_id",
            "intime",
            "outtime",
            "deathtime",
            "hospital_expire_flag"
        ]
    ].head().to_string(index=False)
)


# ============================================================
# 16. DEFINE ICU MORTALITY
# ============================================================

print_section("MORTALITY OUTCOME CHECK")

# ICU mortality = patient died between ICU admission and ICU discharge
icu_admissions["icu_mortality"] = (
    icu_admissions["deathtime"].notna()
    &
    (icu_admissions["deathtime"] >= icu_admissions["intime"])
    &
    (icu_admissions["deathtime"] <= icu_admissions["outtime"])
).astype(int)


print("Total ICU stays:", len(icu_admissions))

print("\nICU mortality:")
print(
    icu_admissions["icu_mortality"]
    .value_counts()
    .sort_index()
)

print("\nHospital mortality:")
print(
    icu_admissions["hospital_expire_flag"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 17. COMPARE ICU VS HOSPITAL DEATH
# ============================================================

print_section("ICU VS HOSPITAL DEATH COMPARISON")

comparison = pd.crosstab(
    icu_admissions["icu_mortality"],
    icu_admissions["hospital_expire_flag"]
)

print(comparison)


# ============================================================
# 18. DEATH COUNTS
# ============================================================

icu_deaths = int(
    icu_admissions["icu_mortality"].sum()
)

hospital_deaths = int(
    icu_admissions["hospital_expire_flag"].sum()
)

print("\nNumber of deaths during ICU stay:", icu_deaths)
print("Number of hospital deaths:", hospital_deaths)


# ============================================================
# 19. SAMPLE DEATH RECORDS
# ============================================================

print_section("SAMPLE ICU DEATH RECORDS")

death_records = icu_admissions[
    icu_admissions["icu_mortality"] == 1
][
    [
        "subject_id",
        "hadm_id",
        "stay_id",
        "intime",
        "outtime",
        "deathtime",
        "hospital_expire_flag",
        "icu_mortality"
    ]
]

print(
    death_records.head(20).to_string(index=False)
)


# ============================================================
# 20. CHECK DEATH TIMING
# ============================================================

print_section("DEATH TIMING CHECK")

death_check = icu_admissions[
    icu_admissions["deathtime"].notna()
].copy()

death_check["death_after_icu_admission"] = (
    death_check["deathtime"]
    >= death_check["intime"]
)

death_check["death_before_icu_discharge"] = (
    death_check["deathtime"]
    <= death_check["outtime"]
)

print(
    death_check[
        [
            "subject_id",
            "stay_id",
            "intime",
            "outtime",
            "deathtime",
            "death_after_icu_admission",
            "death_before_icu_discharge"
        ]
    ].head(20).to_string(index=False)
)


# ============================================================
# 21. UNIQUE PATIENT / STAY COUNTS
# ============================================================

print_section("DATASET SIZE SUMMARY")

print("Unique patients:", patients["subject_id"].nunique())
print("Hospital admissions:", admissions["hadm_id"].nunique())
print("ICU stays:", icustays["stay_id"].nunique())

print(
    "ICU stays with ICU death:",
    icu_admissions.loc[
        icu_admissions["icu_mortality"] == 1,
        "stay_id"
    ].nunique()
)


# ============================================================
# 22. FINAL RESEARCH TARGET SUMMARY
# ============================================================

print_section("FINAL RESEARCH TARGET SUMMARY")

print("""
Primary prediction target:
    ICU Mortality

Target definition:
    1 = patient dies during the ICU stay
    0 = patient does not die during the ICU stay

Why we are checking this:
    We need a clearly defined outcome before building
    the machine-learning dataset.

IMPORTANT:
    We will NOT use information that happens after the
    prediction/index time as model features.

Current dataset:
    MIMIC-IV Clinical Database Demo

This demo dataset is being used to validate the pipeline.
The final research experiments should use the full
MIMIC-IV dataset rather than relying on this tiny demo.
""")


# ============================================================
# 23. FINISHED
# ============================================================

print_section("EXPLORATION COMPLETE")

print("Everything loaded and checked successfully.")
print("Next step: build the leakage-controlled ML dataset.")
print("The ML dataset will contain:")
print("  - ICU stay ID")
print("  - demographic features")
print("  - early ICU vital-sign features")
print("  - early laboratory features")
print("  - ICU mortality target")