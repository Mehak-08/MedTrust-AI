# ============================================================
# MEDTRUST AI
# STEP 28 - PATIENT-LEVEL SPLIT FROM RAW DATA
# ============================================================

import os
import pandas as pd
from sklearn.model_selection import train_test_split


print("\n==========================================")
print("MEDTRUST AI - PATIENT LEVEL SPLIT")
print("==========================================\n")


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_DIR = r"C:\Users\MEHAK\OneDrive\Desktop\MIMIC-Project"

INPUT_FILE = os.path.join(
    PROJECT_DIR,
    "mimic_raw_ml_dataset.csv"
)


# ============================================================
# 2. LOAD RAW DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print(
    "Loaded dataset:",
    df.shape
)

print(
    "Unique patients:",
    df["subject_id"].nunique()
)


# ============================================================
# 3. GET UNIQUE PATIENTS
# ============================================================

patients = (
    df["subject_id"]
    .drop_duplicates()
    .values
)


# ============================================================
# 4. FIRST SPLIT
#    70% TRAIN
#    30% TEMP
# ============================================================

train_patients, temp_patients = train_test_split(
    patients,
    test_size=0.30,
    random_state=42
)


# ============================================================
# 5. SECOND SPLIT
#    TEMP -> 15% VALIDATION
#           15% TEST
# ============================================================

validation_patients, test_patients = train_test_split(
    temp_patients,
    test_size=0.50,
    random_state=42
)


# ============================================================
# 6. CREATE DATASETS
# ============================================================

train = df[
    df["subject_id"].isin(
        train_patients
    )
].copy()


validation = df[
    df["subject_id"].isin(
        validation_patients
    )
].copy()


test = df[
    df["subject_id"].isin(
        test_patients
    )
].copy()


# ============================================================
# 7. VERIFY NO PATIENT OVERLAP
# ============================================================

train_ids = set(
    train["subject_id"]
)

validation_ids = set(
    validation["subject_id"]
)

test_ids = set(
    test["subject_id"]
)


print("\nPatient overlap checks:")

print(
    "Train ∩ Validation:",
    len(train_ids & validation_ids)
)

print(
    "Train ∩ Test:",
    len(train_ids & test_ids)
)

print(
    "Validation ∩ Test:",
    len(validation_ids & test_ids)
)


assert len(
    train_ids & validation_ids
) == 0

assert len(
    train_ids & test_ids
) == 0

assert len(
    validation_ids & test_ids
) == 0


# ============================================================
# 8. SAVE
# ============================================================

train_file = os.path.join(
    PROJECT_DIR,
    "step28_train_raw.csv"
)

validation_file = os.path.join(
    PROJECT_DIR,
    "step28_validation_raw.csv"
)

test_file = os.path.join(
    PROJECT_DIR,
    "step28_test_raw.csv"
)


train.to_csv(
    train_file,
    index=False
)

validation.to_csv(
    validation_file,
    index=False
)

test.to_csv(
    test_file,
    index=False
)


# ============================================================
# 9. PRINT RESULTS
# ============================================================

print("\n==========================================")
print("SPLIT COMPLETE")
print("==========================================")

print(
    "\nTRAIN"
)

print(
    "Patients:",
    train["subject_id"].nunique()
)

print(
    "Rows:",
    len(train)
)

print(
    "Deaths:",
    int(
        train["icu_mortality"].sum()
    )
)


print(
    "\nVALIDATION"
)

print(
    "Patients:",
    validation["subject_id"].nunique()
)

print(
    "Rows:",
    len(validation)
)

print(
    "Deaths:",
    int(
        validation["icu_mortality"].sum()
    )
)


print(
    "\nTEST"
)

print(
    "Patients:",
    test["subject_id"].nunique()
)

print(
    "Rows:",
    len(test)
)

print(
    "Deaths:",
    int(
        test["icu_mortality"].sum()
    )
)


print(
    "\nSaved:"
)

print(
    train_file
)

print(
    validation_file
)

print(
    test_file
)


print("\n==========================================")
print("STEP 28 COMPLETE")
print("==========================================")