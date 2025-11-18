# eval_global_new.py — evaluate global FedAvg model across a range of hospital splits

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

def load_global_model(global_ckpt, device=None):
    """Defines and loads the global DenseNet121 model."""
    if device is None:
        device = DEVICE
    
    print(f"\n[INFO] Loading global checkpoint from: {global_ckpt}")
    model = models.densenet121(weights=None)
    # Model head replaced for 14-label multi-label classification
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, 14),
        nn.Sigmoid() # Use Sigmoid for multi-label output
    )

    try:
        state_dict = torch.load(global_ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[WARN] missing keys: {missing[:3]}... | unexpected: {unexpected[:3]}...")
        else:
            print("[INFO] State dict loaded without key issues.")
    except FileNotFoundError:
        print(f"[ERROR] Global checkpoint not found at {global_ckpt}. Please ensure the model file exists.")
        return None

    model = model.to(device)
    model.eval()
    return model

# ===== Evaluation Function =====

def evaluate_hospital(hospital_id, model, hospital_dir, img_root, batch_size=32, device=None):
    """
    Evaluates the model on a single hospital's test split.
    Returns: (mean_auroc, mean_auprc, avg_loss) or (None, None, None) if metrics cannot be calculated.
    """
    if device is None:
        device = DEVICE
    
    csv_filename = f"{hospital_id}.csv"
    csv_path = Path(hospital_dir) / csv_filename
    
    print(f"\n=============== Evaluating on {csv_filename} ===============")

    # DataLoader Setup
    try:
        dataset = CheXpertDataset(csv_path, img_root, transform=transform)
        if len(dataset) == 0:
            print(f"[WARN] Dataset for {csv_filename} is empty. Skipping.")
            return None, None, None
    except FileNotFoundError:
        print(f"[ERROR] CSV file not found for {csv_filename}. Skipping.")
        return None, None, None
    except pd.errors.EmptyDataError:
        print(f"[ERROR] CSV file is empty for {csv_filename}. Skipping.")
        return None, None, None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    print(f"[INFO] Eval dataset size: {len(dataset)} samples")

    criterion = nn.BCELoss()
    label_cols = dataset.label_cols

    # Evaluation Loop & Data Collection
    total_loss = 0.0
    n = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

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
        return mean_auroc, mean_auprc, avg_loss
    else:
        print(" [WARN] Could not calculate Mean AUROC/AUPRC (insufficient labels with >1 class).")
        return None, None, avg_loss

    print(f" (Samples evaluated: {n})")


def eval_global_all(round_id, hospital_ids, hospital_dir="hospitals_split", img_root=None, batch_size=32):
    """
    Evaluate the global model across multiple hospitals.
    
    Args:
        round_id: Round number (integer) - used to determine global checkpoint path
        hospital_ids: List of hospital IDs (integers, e.g., [1, 2, 3])
        hospital_dir: Directory containing hospital CSV files (default: "hospitals_split")
        img_root: Root directory for images (default: Path.cwd())
        batch_size: Batch size for evaluation (default: 32)
    
    Returns:
        dict: Dictionary containing evaluation metrics across all hospitals
    """
    if img_root is None:
        img_root = Path.cwd()
    else:
        img_root = Path(img_root)
    
    global_ckpt = f"global/round{round_id}_global.pt"
    
    print(f"[INFO] Using device: {DEVICE}")
    print("\n=======================================================")
    print(f"Starting Federated Evaluation on {len(hospital_ids)} Hospitals")
    print("=======================================================")
    
    # Load the global model
    global_model = load_global_model(global_ckpt, DEVICE)
    if global_model is None:
        return None
    
    # Lists to store the mean AUROC and AUPRC from each hospital
    all_hospital_aurocs = []
    all_hospital_auprcs = []
    all_hospital_losses = []
    hospital_results = []
    
    for hospital_id in hospital_ids:
        m_auroc, m_auprc, avg_loss = evaluate_hospital(
            hospital_id, global_model, hospital_dir, img_root, batch_size, DEVICE
        )
        
        hospital_results.append({
            'hospital_id': hospital_id,
            'mean_auroc': m_auroc,
            'mean_auprc': m_auprc,
            'bce_loss': avg_loss
        })
        
        if m_auroc is not None:
            all_hospital_aurocs.append(m_auroc)
        if m_auprc is not None:
            all_hospital_auprcs.append(m_auprc)
        if avg_loss is not None:
            all_hospital_losses.append(avg_loss)
    
    # Final calculation of the average metrics across all hospitals
    print("\n=======================================================")
    print("FINAL GLOBAL AVERAGE METRICS (Across All Hospitals)")
    print("=======================================================")
    
    result = {
        'round_id': round_id,
        'hospital_results': hospital_results
    }
    
    if all_hospital_aurocs:
        final_mean_auroc = np.mean(all_hospital_aurocs)
        result['grand_mean_auroc'] = final_mean_auroc
        print(f"Grand Mean AUROC (mAUC) across {len(all_hospital_aurocs)} hospitals: {final_mean_auroc:.4f}")
    else:
        result['grand_mean_auroc'] = np.nan
        print("Could not calculate Grand Mean AUROC (No valid hospital metrics collected).")
    
    if all_hospital_auprcs:
        final_mean_auprc = np.mean(all_hospital_auprcs)
        result['grand_mean_auprc'] = final_mean_auprc
        print(f"Grand Mean AUPRC (mAP) across {len(all_hospital_auprcs)} hospitals: {final_mean_auprc:.4f}")
    else:
        result['grand_mean_auprc'] = np.nan
        print("Could not calculate Grand Mean AUPRC (No valid hospital metrics collected).")
    
    if all_hospital_losses:
        final_mean_loss = np.mean(all_hospital_losses)
        result['grand_mean_loss'] = final_mean_loss
        print(f"Grand Mean BCE Loss across {len(all_hospital_losses)} hospitals: {final_mean_loss:.4f}")
    else:
        result['grand_mean_loss'] = np.nan
    
    print("\nFederated Evaluation Complete.")
    print("=======================================================")
    
    # Save results to CSV
    Path("eval_results").mkdir(exist_ok=True)
    results_df = pd.DataFrame(hospital_results)
    output_filename = f"eval_results/global_evaluation_round{round_id}.csv"
    try:
        results_df.to_csv(output_filename, index=False)
        print(f"\n[INFO] Results successfully saved to {output_filename}")
    except Exception as e:
        print(f"[ERROR] Failed to save results to CSV: {e}")
    
    return result


if __name__ == "__main__":
    # For backward compatibility, allow running as script
    import sys
    if len(sys.argv) >= 2:
        round_id = int(sys.argv[1])
        # Default hospital IDs if not provided
        hospital_ids = [1, 2, 3] if len(sys.argv) == 2 else [int(x) for x in sys.argv[2:]]
        eval_global_all(round_id, hospital_ids)
    else:
        print("Usage: python eval_global_new.py <round_id> [hospital_id1] [hospital_id2] ...")
