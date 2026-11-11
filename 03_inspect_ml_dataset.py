import os
import pandas as pd


# ============================================================
# MEDTRUST AI
# Step 3: Inspect ML Dataset
# ============================================================

print("\n==========================================")
print("MEDTRUST AI - ML DATASET INSPECTION")
print("==========================================\n")


# ============================================================
# 1. PATH
# ============================================================

DATA_FILE = os.path.join(
    os.path.dirname(__file__),
    "mimic_ml_dataset.csv"
)


# ============================================================
# 2. CHECK FILE
# ============================================================

if not os.path.exists(DATA_FILE):

    print("ERROR:")
    print("mimic_ml_dataset.csv was not found.")

    print("\nMake sure you have successfully run:")
    print("02_build_ml_dataset.py")

    raise SystemExit


# ============================================================
# 3. LOAD DATA
# ============================================================

print("Loading ML dataset...")

df = pd.read_csv(DATA_FILE)

print("Dataset loaded successfully.")


# ============================================================
# 4. BASIC INFORMATION
# ============================================================

print("\n==========================================")
print("BASIC DATASET INFORMATION")
print("==========================================")

print("Shape:")
print(df.shape)


print("\nNumber of rows:", len(df))
print("Number of columns:", len(df.columns))


# ============================================================
# 5. COLUMN LIST
# ============================================================

print("\n==========================================")
print("COLUMNS")
print("==========================================")

for i, column in enumerate(df.columns, start=1):

    print(
        f"{i}. {column}"
    )


# ============================================================
# 6. FIRST 5 ROWS
# ============================================================

print("\n==========================================")
print("FIRST 5 ROWS")
print("==========================================")

print(
    df.head()
)


# ============================================================
# 7. TARGET DISTRIBUTION
# ============================================================

print("\n==========================================")
print("TARGET DISTRIBUTION")
print("==========================================")

print(
    df["icu_mortality"].value_counts()
)


print("\nTarget percentages:")

print(
    df["icu_mortality"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# 8. MISSING VALUES
# ============================================================

print("\n==========================================")
print("MISSING VALUES")
print("==========================================")

missing = df.isna().sum()

missing = missing[
    missing > 0
]

if len(missing) == 0:

    print("No missing values found.")

else:

    print(missing)


# ============================================================
# 9. DUPLICATE ROWS
# ============================================================

print("\n==========================================")
print("DUPLICATE CHECK")
print("==========================================")

duplicates = df.duplicated().sum()

print(
    "Duplicate rows:",
    duplicates
)


# ============================================================
# 10. DUPLICATE ICU STAYS
# ============================================================

print("\n==========================================")
print("ICU STAY ID CHECK")
print("==========================================")

duplicate_stays = df["stay_id"].duplicated().sum()

print(
    "Duplicate stay_id values:",
    duplicate_stays
)


# ============================================================
# 11. DATA TYPES
# ============================================================

print("\n==========================================")
print("DATA TYPES")
print("==========================================")

print(
    df.dtypes
)


# ============================================================
# 12. NUMERIC SUMMARY
# ============================================================

print("\n==========================================")
print("NUMERIC FEATURE SUMMARY")
print("==========================================")

numeric_df = df.select_dtypes(
    include=["number"]
)

print(
    numeric_df.describe().T
)


# ============================================================
# 13. CHECK INF VALUES
# ============================================================

print("\n==========================================")
print("INFINITE VALUE CHECK")
print("==========================================")

numeric_columns = df.select_dtypes(
    include=["number"]
).columns

inf_count = 0

for column in numeric_columns:

    count = (
        df[column]
        .isin([float("inf"), float("-inf")])
        .sum()
    )

    if count > 0:

        print(
            column,
            ":",
            count
        )

        inf_count += count


if inf_count == 0:

    print("No infinite values found.")


# ============================================================
# 14. CHECK TARGET VALUES
# ============================================================

print("\n==========================================")
print("TARGET VALIDATION")
print("==========================================")

valid_targets = df["icu_mortality"].isin(
    [0, 1]
).all()

print(
    "Target contains only 0 and 1:",
    valid_targets
)


# ============================================================
# 15. CHECK ID COLUMNS
# ============================================================

print("\n==========================================")
print("ID INFORMATION")
print("==========================================")

print(
    "Unique patients:",
    df["subject_id"].nunique()
)

print(
    "Unique admissions:",
    df["hadm_id"].nunique()
)

print(
    "Unique ICU stays:",
    df["stay_id"].nunique()
)


# ============================================================
# 16. FEATURE COUNT
# ============================================================

identifier_columns = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "icu_mortality"
]

feature_columns = [
    column
    for column in df.columns
    if column not in identifier_columns
]


print("\n==========================================")
print("FEATURE COUNT")
print("==========================================")

print(
    "Number of ML features:",
    len(feature_columns)
)


# ============================================================
# 17. CHECK FOR OBVIOUS LEAKAGE COLUMNS
# ============================================================

print("\n==========================================")
print("LEAKAGE COLUMN CHECK")
print("==========================================")

dangerous_columns = [
    "outtime",
    "deathtime",
    "los",
    "hospital_expire_flag",
    "window_end"
]

found_leakage = []

for column in dangerous_columns:

    if column in df.columns:

        found_leakage.append(column)


if len(found_leakage) == 0:

    print(
        "No obvious leakage columns found."
    )

else:

    print(
        "WARNING - possible leakage columns:"
    )

    for column in found_leakage:

        print(
            "-",
            column
        )


# ============================================================
# 18. FEATURE NAMES
# ============================================================

print("\n==========================================")
print("ML FEATURES")
print("==========================================")

for feature in feature_columns:

    print(
        feature
    )


# ============================================================
# 19. FINAL RESULT
# ============================================================

print("\n==========================================")
print("INSPECTION COMPLETE")
print("==========================================")

print(
    "\nIf the checks above look correct,"
)

print(
    "the next step is to split the dataset"
)

print(
    "into training, validation and test sets."
)