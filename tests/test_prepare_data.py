import sys
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.data_io import load_data, validate_data
from core.preprocessing import prepare_data
from config import (PROTOCOL_PATH, DATA_PATH, RESULTS_PATH)

# ============================================================
# CREATE RESULTS FOLDER
# ============================================================

RESULTS_PATH.mkdir(
    parents=True,
    exist_ok=True
)

print(f"\nResults folder: {RESULTS_PATH}")
print(f"Exists: {RESULTS_PATH.exists()}")

PREPROCESSED_DATA_PATH = os.path.join(RESULTS_PATH, "data_processed.xlsx")

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("TEST SETUP")
print("=" * 60)

protocol = load_data.load_protocol(PROTOCOL_PATH)
df_main = load_data.load_data_file(DATA_PATH, "Trials")
df_subjects = load_data.load_data_file(DATA_PATH, "Subjects")

print("\nProtocol loaded:")
print(protocol)

print("\nMain data loaded:")
print(df_main.head())

print("\nSubjects data loaded:")
print(df_subjects.head())

# ============================================================
# VALIDATE DATA
# ============================================================
print("\n" + "=" * 60)
print("TEST VALIDATION")
print("=" * 60)

protocol_valid = validate_data.validate_protocol(protocol)

data_valid = validate_data.validate_main_data(
    df_main,
    protocol
)

print("\nValidation results:")
print(f"Protocol: {'OK' if protocol_valid else 'FAILED'}")
print(f"Subject data: {'OK' if data_valid else 'FAILED'}")

# ------------------------------------------------------------
# TEST PREPARE DATA
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TEST PREPARE DATA")
print("=" * 60)

df_prepared = prepare_data.prepare_data(
    df_main,
    df_subjects,
    protocol,
    output_path=PREPROCESSED_DATA_PATH
)

print("\nPrepared data:")
print(df_prepared.head())

print("\nColumns:")
print(df_prepared.columns.tolist())

print("\nShape:")
print(df_prepared.shape)

print("\nExpected illusion:")
print(
    df_prepared[
        ["pattern_pair", "expected_kinesthetic_illusion"]
    ].drop_duplicates()
)