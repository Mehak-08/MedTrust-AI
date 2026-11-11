import os
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# MEDTRUST AI
# Step 4: Train / Validation / Test Split
# ============================================================

print("\n==========================================")
print("MEDTRUST AI - DATASET SPLITTING")
print("==========================================\n")


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATA_FILE = os.path.join(
    os.path.dirname(__file__),
    "mimic_ml_dataset.csv"
)


if not os.path.exists(DATA_FILE):

    print("ERROR:")
    print("mimic_ml_dataset.csv was not found.")

    print("\nRun 02_build_ml_dataset.py first.")

    raise SystemExit


print("Loading dataset...")

df = pd.read_csv(DATA_FILE)

print(
    "Dataset shape:",
    df.shape
)


# ============================================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================================

TARGET = "icu_mortality"

ID_COLUMNS = [
    "subject_id",
    "hadm_id",
    "stay_id"
]


X = df.drop(
    columns=[TARGET]
)

y = df[TARGET]


# ============================================================
# 3. REMOVE IDENTIFIERS FROM MODEL FEATURES
# ============================================================

X = X.drop(
    columns=ID_COLUMNS
)


print("\n==========================================")
print("FEATURE / TARGET INFORMATION")
print("==========================================")

print(
    "Number of features:",
    X.shape[1]
)

print(
    "Number of samples:",
    X.shape[0]
)

print(
    "Target values:"
)

print(
    y.value_counts()
)


# ============================================================
# 4. FIRST SPLIT
# ============================================================
#
# 70% -> Training
# 30% -> Temporary set
#
# The temporary set will later become:
# 15% validation
# 15% test
#
# Stratification keeps the target distribution
# approximately similar between groups.
#
# IMPORTANT:
# The demo dataset has only 12 positive cases,
# so this is mainly for pipeline development.


print("\n==========================================")
print("CREATING TRAINING / TEMPORARY SPLIT")
print("==========================================")


X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# ============================================================
# 5. SECOND SPLIT
# ============================================================
#
# Split the 30% temporary set equally:
#
# 15% -> Validation
# 15% -> Test
#


X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


# ============================================================
# 6. DISPLAY SIZES
# ============================================================

print("\n==========================================")
print("SPLIT SIZES")
print("==========================================")

print(
    "Training:",
    X_train.shape
)

print(
    "Validation:",
    X_val.shape
)

print(
    "Test:",
    X_test.shape
)


# ============================================================
# 7. DISPLAY TARGET DISTRIBUTION
# ============================================================

print("\n==========================================")
print("TARGET DISTRIBUTION")
print("==========================================")


print("\nTraining:")

print(
    y_train.value_counts()
)


print("\nValidation:")

print(
    y_val.value_counts()
)


print("\nTest:")

print(
    y_test.value_counts()
)


# ============================================================
# 8. TARGET PERCENTAGES
# ============================================================

print("\n==========================================")
print("TARGET PERCENTAGES")
print("==========================================")

print("\nTraining:")

print(
    y_train.value_counts(
        normalize=True
    ).mul(100).round(2)
)


print("\nValidation:")

print(
    y_val.value_counts(
        normalize=True
    ).mul(100).round(2)
)


print("\nTest:")

print(
    y_test.value_counts(
        normalize=True
    ).mul(100).round(2)
)


# ============================================================
# 9. REBUILD COMPLETE SPLIT DATAFRAMES
# ============================================================
#
# We save IDs together with the features.
#
# This is useful later when we need to trace a prediction
# back to the original ICU stay.


train_df = df.loc[
    X_train.index
].copy()


val_df = df.loc[
    X_val.index
].copy()


test_df = df.loc[
    X_test.index
].copy()


# ============================================================
# 10. SAVE SPLITS
# ============================================================

PROJECT_PATH = os.path.dirname(__file__)


train_file = os.path.join(
    PROJECT_PATH,
    "train.csv"
)


val_file = os.path.join(
    PROJECT_PATH,
    "validation.csv"
)


test_file = os.path.join(
    PROJECT_PATH,
    "test.csv"
)


train_df.to_csv(
    train_file,
    index=False
)


val_df.to_csv(
    val_file,
    index=False
)


test_df.to_csv(
    test_file,
    index=False
)


# ============================================================
# 11. VERIFY SAVED FILES
# ============================================================

print("\n==========================================")
print("FILES SAVED")
print("==========================================")

print(
    "Training:",
    train_file
)

print(
    "Validation:",
    val_file
)

print(
    "Test:",
    test_file
)


# ============================================================
# 12. FINAL CHECK
# ============================================================

print("\n==========================================")
print("FINAL CHECK")
print("==========================================")

print(
    "Training rows:",
    len(train_df)
)

print(
    "Validation rows:",
    len(val_df)
)

print(
    "Test rows:",
    len(test_df)
)

print(
    "Total rows:",
    len(train_df)
    + len(val_df)
    + len(test_df)
)


print(
    "\nOriginal rows:",
    len(df)
)


# ============================================================
# 13. COMPLETE
# ============================================================

print("\n==========================================")
print("STEP 4 COMPLETE")
print("==========================================")

print(
    "\nWe now have:"
)

print(
    "1. train.csv"
)

print(
    "2. validation.csv"
)

print(
    "3. test.csv"
)

print(
    "\nNext step:"
)

print(
    "Train the Logistic Regression baseline."
)