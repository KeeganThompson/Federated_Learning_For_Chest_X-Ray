# eval_global.py — evaluate global FedAvg model across a range of hospital splits

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
IMG_ROOT = Path("CheXpert-v1.0-small")
# GLOBAL_CKPT = "global/round1_global_skewed.pt"
GLOBAL_CKPT = "global/round1_global.pt"
BATCH_SIZE = 32
HOSPITAL_IDS = [f"hospital_{i:02d}" for i in range(5, 11)]
# HOSPITAL_DIR = "hospitals_split_skewed"
HOSPITAL_DIR = "hospitals_split"

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"[INFO] Using device: {DEVICE}")

# ===== Dataset Class =====
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
        # Uncertainty handling: NaN, -1 (uncertain) -> 0
        self.df[self.label_cols] = self.df[self.label_cols].fillna(0).replace(-1, 0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        raw_path = row["Path"]
        img_path = Path(raw_path)

        # Fix path structure based on typical CheXpert data layout
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

# ===== Model Definition and Loading =====

def load_global_model():
    """Defines and loads the global DenseNet121 model."""
    print(f"\n[INFO] Loading global checkpoint from: {GLOBAL_CKPT}")
    model = models.densenet121(weights=None)
    # Model head replaced for 14-label multi-label classification
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, 14),
        nn.Sigmoid() # Use Sigmoid for multi-label output
    )

    try:
        state_dict = torch.load(GLOBAL_CKPT, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[WARN] missing keys: {missing[:3]}... | unexpected: {unexpected[:3]}...")
        else:
            print("[INFO] State dict loaded without key issues.")
    except FileNotFoundError:
        print(f"[ERROR] Global checkpoint not found at {GLOBAL_CKPT}. Please ensure the model file exists.")
        return None

    model = model.to(DEVICE)
    model.eval()
    return model

# Load the model once
global_model = load_global_model()
if global_model is None:
    exit()

criterion = nn.BCELoss()
label_cols = CheXpertDataset(Path(HOSPITAL_DIR) / f"{HOSPITAL_IDS[0]}.csv", IMG_ROOT).label_cols

# ===== Evaluation Function =====

def evaluate_hospital(hospital_id, model):
    """
    Evaluates the model on a single hospital's test split.
    Returns: (mean_auroc, mean_auprc) or (None, None) if metrics cannot be calculated.
    """
    csv_filename = f"{hospital_id}.csv"
    csv_path = Path(HOSPITAL_DIR) / csv_filename
    
    print(f"\n=============== Evaluating on {csv_filename} ===============")

    # DataLoader Setup
    try:
        dataset = CheXpertDataset(csv_path, IMG_ROOT, transform=transform)
        if len(dataset) == 0:
            print(f"[WARN] Dataset for {csv_filename} is empty. Skipping.")
            return None, None
    except FileNotFoundError:
        print(f"[ERROR] CSV file not found for {csv_filename}. Skipping.")
        return None, None
    except pd.errors.EmptyDataError:
        print(f"[ERROR] CSV file is empty for {csv_filename}. Skipping.")
        return None, None

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    print(f"[INFO] Eval dataset size: {len(dataset)} samples")

    # Evaluation Loop & Data Collection
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

            all_labels.append(labels.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())

    avg_loss = total_loss / max(n, 1)

    # Concatenate and Calculate Metrics
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    auroc_scores = []
    auprc_scores = []
    
    for i, label in enumerate(label_cols):
        y_true = all_labels[:, i]
        y_pred = all_preds[:, i]

        if len(np.unique(y_true)) > 1:
            try:
                auroc = roc_auc_score(y_true, y_pred)
                auprc = average_precision_score(y_true, y_pred)
                auroc_scores.append(auroc)
                auprc_scores.append(auprc)
            except ValueError as e:
                print(f"[ERROR] Metric calculation failed for label '{label}': {e}")


    # Final Summary for current hospital
    print("\n--- Summary ---")
    print(f"BCE Loss: {avg_loss:.4f}")

    if auroc_scores:
        mean_auroc = np.mean(auroc_scores)
        mean_auprc = np.mean(auprc_scores)
        
        print(f" Mean AUROC (mAUC) across {len(auroc_scores)} valid labels: {mean_auroc:.4f}")
        print(f" Mean AUPRC (mAP) across {len(auprc_scores)} valid labels: {mean_auprc:.4f}")
        return mean_auroc, mean_auprc
    else:
        print(" [WARN] Could not calculate Mean AUROC/AUPRC (insufficient labels with >1 class).")
        return None, None

    print(f" (Samples evaluated: {n})")


# ===== Main Execution =====
print("\n=======================================================")
print(f"Starting Federated Evaluation on {len(HOSPITAL_IDS)} Hospitals")
print("=======================================================")

# Lists to store the mean AUROC and AUPRC from each hospital
all_hospital_aurocs = []
all_hospital_auprcs = []

for hospital_id in HOSPITAL_IDS:
    m_auroc, m_auprc = evaluate_hospital(hospital_id, global_model)
    
    if m_auroc is not None:
        all_hospital_aurocs.append(m_auroc)
    if m_auprc is not None:
        all_hospital_auprcs.append(m_auprc)

# Final calculation of the average metrics across all hospitals
print("\n=======================================================")
print("FINAL GLOBAL AVERAGE METRICS (Across All Hospitals)")
print("=======================================================")

if all_hospital_aurocs:
    final_mean_auroc = np.mean(all_hospital_aurocs)
    print(f"Grand Mean AUROC (mAUC) across {len(all_hospital_aurocs)} hospitals: {final_mean_auroc:.4f}")
else:
    print("Could not calculate Grand Mean AUROC (No valid hospital metrics collected).")

if all_hospital_auprcs:
    final_mean_auprc = np.mean(all_hospital_auprcs)
    print(f"Grand Mean AUPRC (mAP) across {len(all_hospital_auprcs)} hospitals: {final_mean_auprc:.4f}")
else:
    print("Could not calculate Grand Mean AUPRC (No valid hospital metrics collected).")

print("\nFederated Evaluation Complete.")
print("=======================================================")