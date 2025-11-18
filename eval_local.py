# eval_local.py — evaluate a single hospital's local model on its data split

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

# ===== CONFIG =====
HOSPITAL_ID = "hospital_04" # Target hospital ID
ROUND_ID = "4"              # Round number for output filename (e.g., round1)
# HOSPITAL_DIR = "hospitals_split_skewed"
HOSPITAL_DIR = "hospitals_split"
CSV_PATH = f"{HOSPITAL_DIR}/{HOSPITAL_ID}.csv" # Data split for the target hospital
IMG_ROOT = Path("CheXpert-v1.0-small")
# Checkpoint for the locally trained model
# LOCAL_CKPT = "hospital04_skewed_densenet_torchvision.pt" 
LOCAL_CKPT = "hospital04_densenet_torchvision.pt" 
BATCH_SIZE = 32

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"[INFO] Using device: {DEVICE}")

# ===== Dataset =====
class CheXpertDataset(Dataset):
    def __init__(self, csv_path, img_root, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_root = Path(img_root)
        self.transform = transform
        self.label_cols = [
            'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
            'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
            'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
            'Pleural Other', 'Fracture', 'Support Devices'
        ]
        # NaN, -1 (uncertain) -> 0
        self.df[self.label_cols] = self.df[self.label_cols].fillna(0).replace(-1, 0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        raw_path = row["Path"]                
        img_path = Path(raw_path)

        parts = img_path.parts
        if "train" in parts:
            train_idx = parts.index("train")
            img_path = Path(*parts[train_idx:])
        full_path = self.img_root / img_path

        image = Image.open(full_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        labels = torch.tensor(row[self.label_cols].values.astype("float32"))
        return image, labels

# ===== Transforms =====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ===== DataLoader =====
dataset = CheXpertDataset(CSV_PATH, IMG_ROOT, transform=transform)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,   
)

print(f"[INFO] Eval dataset size: {len(dataset)} samples")

# ===== load Local Model (DenseNet121) =====
print(f"\n[INFO] Loading local checkpoint from: {LOCAL_CKPT}")
model = models.densenet121(weights=None)
# Model head replaced for 14-label multi-label classification
model.classifier = nn.Sequential(
    nn.Linear(model.classifier.in_features, 14),
    nn.Sigmoid()
)

try:
    # Load the local checkpoint
    state_dict = torch.load(LOCAL_CKPT, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print("[WARN] missing keys:", missing[:5], "| unexpected:", unexpected[:5])
    else:
        print("[INFO] State dict loaded without key issues.")
except FileNotFoundError:
    print(f"[ERROR] Local checkpoint not found at {LOCAL_CKPT}. Exiting.")
    exit()

model = model.to(DEVICE)
model.eval()

criterion = nn.BCELoss()

# ===== Evaluation Loop & Data Collection =====
total_loss = 0.0
n = 0
all_labels = []
all_preds = []

with torch.no_grad():
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
        
        # Collect data for metric calculation
        all_labels.append(labels.cpu().numpy())
        all_preds.append(outputs.cpu().numpy())


avg_loss = total_loss / max(n, 1)

# Concatenate all predictions and labels
all_labels = np.concatenate(all_labels, axis=0)
all_preds = np.concatenate(all_preds, axis=0)

# ===== Metric Calculation (AUROC & AUPRC) and Data Structuring =====
label_cols = dataset.label_cols
auroc_scores = []
auprc_scores = []
per_label_results = []

print("\n--- Per-Label Metrics ---")
for i, label in enumerate(label_cols):
    y_true = all_labels[:, i]
    y_pred = all_preds[:, i]
    
    auroc = np.nan
    auprc = np.nan
    metric_calculated = False

    # Only calculate if there is more than one class present (essential for AUROC)
    if len(np.unique(y_true)) > 1:
        try:
            auroc = roc_auc_score(y_true, y_pred)
            auprc = average_precision_score(y_true, y_pred)
            auroc_scores.append(auroc)
            auprc_scores.append(auprc)
            metric_calculated = True
            print(f"| {label:<25} | AUROC: {auroc:.4f} | AUPRC: {auprc:.4f} |")
        except ValueError as e:
            print(f"[ERROR] Metric calculation failed for label '{label}': {e}")
    
    if not metric_calculated:
        # If only one class, skip calculation and print N/A
        print(f"| {label:<25} | AUROC: N/A | AUPRC: N/A | (Skipped - single class)")

    per_label_results.append({
        'Metric_Type': 'Per-Label',
        'Label': label,
        'AUROC': auroc,
        'AUPRC': auprc,
        'BCE_Loss': np.nan, # Per-label loss is usually not calculated
        'Samples_Evaluated': n
    })


# ===== Final Summary & CSV Export =====

df_results = pd.DataFrame(per_label_results)

# Add final summary row
if auroc_scores:
    mean_auroc = np.mean(auroc_scores)
    mean_auprc = np.mean(auprc_scores)
    
    summary_row = pd.Series({
        'Metric_Type': 'Summary',
        'Label': 'Mean_Valid_Labels',
        'AUROC': mean_auroc,
        'AUPRC': mean_auprc,
        'BCE_Loss': avg_loss,
        'Samples_Evaluated': n
    })
    
    # Append the summary row to the DataFrame
    df_results.loc[len(df_results)] = summary_row

    # Print summary to console
    print("\n--- Final Summary ---")
    print(f"✅ BCE loss on {CSV_PATH}: {avg_loss:.4f}")
    print(f" Mean AUROC (mAUC) across {len(auroc_scores)} labels: {mean_auroc:.4f}")
    print(f" Mean AUPRC (mAP) across {len(auprc_scores)} labels: {mean_auprc:.4f}")
else:
    print(" [WARN] Could not calculate Mean AUROC/AUPRC (insufficient labels with >1 class).")
    
print(f" (samples evaluated: {n})")

# Define output file path and save
OUTPUT_FILENAME = f"{HOSPITAL_ID}_evaluation_round{ROUND_ID}.csv"
try:
    df_results.to_csv(OUTPUT_FILENAME, index=False)
    print(f"\n[INFO] Results successfully saved to {OUTPUT_FILENAME}")
except Exception as e:
    print(f"[ERROR] Failed to save results to CSV: {e}")