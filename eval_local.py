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

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

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

def eval_local(hospital_id, round_id, hospital_dir="hospitals_split", img_root=None, batch_size=32):
    """
    Evaluate a single hospital's local model on its data split.
    
    Args:
        hospital_id: Hospital ID (integer, e.g., 1, 2, 3)
        round_id: Round number (integer)
        hospital_dir: Directory containing hospital CSV files (default: "hospitals_split")
        img_root: Root directory for images (default: Path.cwd())
        batch_size: Batch size for evaluation (default: 32)
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    if img_root is None:
        img_root = Path.cwd() / "CheXpert-v1.0-small"
    else:
        img_root = Path(img_root)
    
    csv_path = f"{hospital_dir}/hospital_" + f"{hospital_id:02d}" + ".csv"
    local_ckpt = f"hospital{hospital_id}_densenet_torchvision{round_id}.pt"
    
    print(f"[INFO] Using device: {DEVICE}")
    print(f"[INFO] Evaluating local model for hospital {hospital_id}, round {round_id}")
    
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
    dataset = CheXpertDataset(csv_path, img_root, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,   
    )
    
    print(f"[INFO] Eval dataset size: {len(dataset)} samples")
    
    # ===== load Local Model (DenseNet121) =====
    print(f"\n[INFO] Loading local checkpoint from: {local_ckpt}")
    model = models.densenet121(weights=None)
    # Model head replaced for 14-label multi-label classification
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, 14),
        nn.Sigmoid()
    )
    
    try:
        # Load the local checkpoint
        state_dict = torch.load(local_ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print("[WARN] missing keys:", missing[:5], "| unexpected:", unexpected[:5])
        else:
            print("[INFO] State dict loaded without key issues.")
    except FileNotFoundError:
        print(f"[ERROR] Local checkpoint not found at {local_ckpt}. Skipping evaluation.")
        return None
    
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
        print(f"✅ BCE loss on {csv_path}: {avg_loss:.4f}")
        print(f" Mean AUROC (mAUC) across {len(auroc_scores)} labels: {mean_auroc:.4f}")
        print(f" Mean AUPRC (mAP) across {len(auroc_scores)} labels: {mean_auprc:.4f}")
    else:
        print(" [WARN] Could not calculate Mean AUROC/AUPRC (insufficient labels with >1 class).")
        
    print(f" (samples evaluated: {n})")
    
    # Define output file path and save
    output_filename = f"eval_results/hospital{hospital_id}_local_evaluation_round{round_id}.csv"
    Path("eval_results").mkdir(exist_ok=True)
    try:
        df_results.to_csv(output_filename, index=False)
        print(f"\n[INFO] Results successfully saved to {output_filename}")
    except Exception as e:
        print(f"[ERROR] Failed to save results to CSV: {e}")
    
    # Return summary metrics
    result = {
        'hospital_id': hospital_id,
        'round_id': round_id,
        'bce_loss': avg_loss,
        'mean_auroc': np.mean(auroc_scores) if auroc_scores else np.nan,
        'mean_auprc': np.mean(auprc_scores) if auprc_scores else np.nan,
        'samples_evaluated': n
    }
    return result


if __name__ == "__main__":
    # For backward compatibility, allow running as script
    import sys
    if len(sys.argv) >= 3:
        hospital_id = int(sys.argv[1])
        round_id = int(sys.argv[2])
        eval_local(hospital_id, round_id)
    else:
        print("Usage: python eval_local.py <hospital_id> <round_id>")
