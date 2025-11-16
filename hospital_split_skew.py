import pandas as pd
import numpy as np
from pathlib import Path

# === Config ===
INPUT_CSV = "train.csv"
OUTPUT_DIR = Path("hospitals_split")
N_HOSPITALS = 20
RANDOM_SEED = 42

# --- Skewing Configuration for CheXpert ---
SKEW_RATIO = 0.8 
SKEW_CONFIG = {
    1: {'column': 'Atelectasis', 'label': 1},
    2: {'column': 'Cardiomegaly', 'label': 1},
    3: {'column': 'Edema', 'label': 1},
    4: {'column': 'Pleural Effusion', 'label': 1}
}
# ------------------------------------------

# === Load Data ===
df = pd.read_csv(INPUT_CSV)

# --- Patient ID extraction from file ---
def extract_patient_id(path):
    parts = path.split('/')
    for part in parts:
        if part.startswith('patient'):
            return part
    raise ValueError(f"Could not find patient ID in path: {path}")

df["PatientID"] = df["Path"].apply(extract_patient_id)

# --- Skewed Split Logic ---
patients = df["PatientID"].unique()
np.random.seed(RANDOM_SEED)

target_patients_per_hospital = len(patients) // N_HOSPITALS

hospital_splits = {}
assigned_patients = set()

# 1. Skew the first 4 hospitals
for i in range(1, 5):
    config = SKEW_CONFIG[i]
    skew_column = config['column']
    skew_label = config['label']
    
    skew_patients_pool = df[
        (df[skew_column] == skew_label) & 
        (~df["PatientID"].isin(assigned_patients))
    ]["PatientID"].unique()
    
    other_patients_pool = df[
        (df[skew_column] != skew_label) & 
        (~df["PatientID"].isin(assigned_patients))
    ]["PatientID"].unique()
    
    skew_count = int(target_patients_per_hospital * SKEW_RATIO)
    other_count = target_patients_per_hospital - skew_count
    
    selected_skew = np.random.choice(
        skew_patients_pool, 
        min(skew_count, len(skew_patients_pool)), 
        replace=False
    )
    
    selected_other = np.random.choice(
        other_patients_pool, 
        min(other_count, len(other_patients_pool)), 
        replace=False
    )
    
    hospital_patients = np.concatenate([selected_skew, selected_other])
    hospital_splits[i] = hospital_patients
    assigned_patients.update(hospital_patients)
    
    print(f"Hospital {i}: Skewed towards '{skew_column}' Positive. Assigned {len(selected_skew)} skew patients and {len(selected_other)} others.")

# 2. Assign the remaining patients to the rest of the hospitals (5 to N_HOSPITALS)
remaining_patients = [p for p in patients if p not in assigned_patients]

num_remaining_hospitals = N_HOSPITALS - 4

if num_remaining_hospitals > 0:
    remaining_splits = np.array_split(remaining_patients, num_remaining_hospitals)
    
    for i, remaining_hospital_patients in enumerate(remaining_splits, 5):
        if i <= N_HOSPITALS:
            hospital_splits[i] = remaining_hospital_patients

# 3. Save the datasets
OUTPUT_DIR.mkdir(exist_ok=True)

for i in range(1, N_HOSPITALS + 1):
    if i in hospital_splits:
        hospital_patients = hospital_splits[i]
        subset = df[df["PatientID"].isin(hospital_patients)].copy()
        subset["HospitalID"] = f"hospital_{i:02d}"
        out_path = OUTPUT_DIR / f"hospital_{i:02d}.csv"
        subset.to_csv(out_path, index=False)
        print(f"Saved {len(subset)} rows to {out_path} ({len(hospital_patients)} patients)")
    else:
         print(f"Hospital {i} split was not created (issue with calculation).")