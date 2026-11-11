# ============================================================
# MEDTRUST AI
# STEP 16 - PATIENT-LEVEL + LEAKAGE-SAFE DATA SPLIT
# ============================================================

import os
import pandas as pd
from sklearn.model_selection import train_test_split


print("\n==========================================")
print("MEDTRUST AI - PATIENT LEVEL SPLIT")
print("==========================================\n")


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "mimic_ml_dataset.csv"
)


# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(DATA_FILE)

print(
    "Dataset shape:",
    df.shape
)


# ------------------------------------------------------------
# 3. BASIC CHECK
# ------------------------------------------------------------

required_columns = [
    "subject_id",
    "icu_mortality"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Required column missing: {column}"
        )


print(
    "\nUnique patients:",
    df["subject_id"].nunique()
)

print(
    "Total ICU stays:",
    len(df)
)


# ------------------------------------------------------------
# 4. CHECK PATIENTS WITH MULTIPLE ICU STAYS
# ------------------------------------------------------------

patient_stays = (
    df.groupby("subject_id")
    .size()
    .sort_values(
        ascending=False
    )
)


multi_stay_patients = (
    patient_stays[
        patient_stays > 1
    ]
)


print(
    "\nPatients with multiple ICU stays:",
    len(multi_stay_patients)
)


if len(multi_stay_patients) > 0:

    print(
        "\nTop patients by number of ICU stays:"
    )

    print(
        multi_stay_patients.head(10)
    )


# ------------------------------------------------------------
# 5. PATIENT-LEVEL SPLIT
# ------------------------------------------------------------

patients = (
    df["subject_id"]
    .drop_duplicates()
    .values
)


print(
    "\nTotal unique patients:",
    len(patients)
)


# First split:
# 70% train
# 30% temporary

train_patients, temp_patients = (
    train_test_split(
        patients,
        test_size=0.30,
        random_state=42
    )
)


# Second split:
# 15% validation
# 15% test

validation_patients, test_patients = (
    train_test_split(
        temp_patients,
        test_size=0.50,
        random_state=42
    )
)


# ------------------------------------------------------------
# 6. CREATE DATASETS
# ------------------------------------------------------------

train_df = df[
    df["subject_id"].isin(
        train_patients
    )
].copy()


validation_df = df[
    df["subject_id"].isin(
        validation_patients
    )
].copy()


test_df = df[
    df["subject_id"].isin(
        test_patients
    )
].copy()


# ------------------------------------------------------------
# 7. VERIFY NO PATIENT OVERLAP
# ------------------------------------------------------------

train_ids = set(
    train_df["subject_id"]
)

validation_ids = set(
    validation_df["subject_id"]
)

test_ids = set(
    test_df["subject_id"]
)


train_val_overlap = (
    train_ids &
    validation_ids
)

train_test_overlap = (
    train_ids &
    test_ids
)

val_test_overlap = (
    validation_ids &
    test_ids
)


print("\n==========================================")
print("PATIENT OVERLAP CHECK")
print("==========================================")

print(
    "Train ∩ Validation:",
    len(train_val_overlap)
)

print(
    "Train ∩ Test:",
    len(train_test_overlap)
)

print(
    "Validation ∩ Test:",
    len(val_test_overlap)
)


if (
    len(train_val_overlap) > 0
    or
    len(train_test_overlap) > 0
    or
    len(val_test_overlap) > 0
):

    raise ValueError(
        "Patient leakage detected!"
    )


print(
    "\nSUCCESS: No patient appears "
    "in more than one split."
)


# ------------------------------------------------------------
# 8. TARGET DISTRIBUTION
# ------------------------------------------------------------

def show_distribution(
    name,
    data
):

    print(
        f"\n{name}"
    )

    print(
        "Rows:",
        len(data)
    )

    print(
        "Unique patients:",
        data["subject_id"].nunique()
    )

    print(
        "Mortality counts:"
    )

    print(
        data[
            "icu_mortality"
        ].value_counts()
        .sort_index()
    )

    print(
        "Mortality percentage:"
    )

    print(
        (
            data[
                "icu_mortality"
            ]
            .value_counts(
                normalize=True
            )
            .sort_index()
            * 100
        ).round(2)
    )


show_distribution(
    "TRAIN",
    train_df
)

show_distribution(
    "VALIDATION",
    validation_df
)

show_distribution(
    "TEST",
    test_df
)


# ------------------------------------------------------------
# 9. CHECK WHETHER EACH SPLIT HAS BOTH CLASSES
# ------------------------------------------------------------

for name, data in [
    ("TRAIN", train_df),
    ("VALIDATION", validation_df),
    ("TEST", test_df)
]:

    classes = (
        data[
            "icu_mortality"
        ]
        .nunique()
    )

    if classes < 2:

        print(
            f"\nWARNING: {name} contains "
            "only one target class."
        )


# ------------------------------------------------------------
# 10. SAVE SPLITS
# ------------------------------------------------------------

train_path = os.path.join(
    BASE_DIR,
    "train_patient_level.csv"
)

validation_path = os.path.join(
    BASE_DIR,
    "validation_patient_level.csv"
)

test_path = os.path.join(
    BASE_DIR,
    "test_patient_level.csv"
)


train_df.to_csv(
    train_path,
    index=False
)

validation_df.to_csv(
    validation_path,
    index=False
)

test_df.to_csv(
    test_path,
    index=False
)


# ------------------------------------------------------------
# 11. SAVE PATIENT IDS
# ------------------------------------------------------------

pd.DataFrame({
    "subject_id": sorted(
        train_ids
    )
}).to_csv(
    os.path.join(
        BASE_DIR,
        "train_patient_ids.csv"
    ),
    index=False
)


pd.DataFrame({
    "subject_id": sorted(
        validation_ids
    )
}).to_csv(
    os.path.join(
        BASE_DIR,
        "validation_patient_ids.csv"
    ),
    index=False
)


pd.DataFrame({
    "subject_id": sorted(
        test_ids
    )
}).to_csv(
    os.path.join(
        BASE_DIR,
        "test_patient_ids.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# 12. FINAL SUMMARY
# ------------------------------------------------------------

print("\n==========================================")
print("STEP 16 COMPLETE")
print("==========================================")

print(
    "\nTrain:",
    train_df.shape
)

print(
    "Validation:",
    validation_df.shape
)

print(
    "Test:",
    test_df.shape
)

print(
    "\nPatient-level separation verified."
)

print(
    "\nCreated files:"
)

print(
    "train_patient_level.csv"
)

print(
    "validation_patient_level.csv"
)

print(
    "test_patient_level.csv"
)

print(
    "train_patient_ids.csv"
)

print(
    "validation_patient_ids.csv"
)

print(
    "test_patient_ids.csv"
)

print("\n==========================================")
print("IMPORTANT")
print("==========================================")

print(
    "This is a DEMO dataset."
)

print(
    "The final research version should use "
    "the same patient-level principle on full MIMIC-IV."
)