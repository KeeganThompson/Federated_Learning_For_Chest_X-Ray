import pandas as pd
import numpy as np
from pathlib import Path

# === Config ===
INPUT_CSV = "train.csv"
OUTPUT_DIR = Path("hospitals_split")
N_HOSPITALS = 20
RANDOM_SEED = 42

# === Load Data ===
df = pd.read_csv(INPUT_CSV)

# --- Patient ID extraction from file ---
def extract_patient_id(path):
    parts = path.split('/')
    # Find the first segment that starts with 'patient'
    for part in parts:
        if part.startswith('patient'):
            return part
    raise ValueError(f"Could not find patient ID in path: {path}")

df["PatientID"] = df["Path"].apply(extract_patient_id)

# --- Split patients into hospitals ---
patients = df["PatientID"].unique()
np.random.seed(RANDOM_SEED)
np.random.shuffle(patients)
splits = np.array_split(patients, N_HOSPITALS)

OUTPUT_DIR.mkdir(exist_ok=True)

for i, hospital_patients in enumerate(splits, 1):
    subset = df[df["PatientID"].isin(hospital_patients)].copy()
    subset["HospitalID"] = f"hospital_{i:02d}"
    out_path = OUTPUT_DIR / f"hospital_{i:02d}.csv"
    subset.to_csv(out_path, index=False)
    print(f"Saved {len(subset)} rows to {out_path} ({len(hospital_patients)} patients)")
