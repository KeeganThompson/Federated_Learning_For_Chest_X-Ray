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
    Returns: Dictionary with metrics or None if evaluation cannot be performed.
    """
    if device is None:
        device = DEVICE
    
    csv_filename = f"hospital_" + f"{hospital_id:02d}" + ".csv"
    csv_path = Path(hospital_dir) / csv_filename
    
    print(f"\n=============== Evaluating on {csv_filename} ===============")

    # DataLoader Setup
    try:
        dataset = CheXpertDataset(csv_path, img_root, transform=transform)
        if len(dataset) == 0:
            print(f"[WARN] Dataset for {csv_filename} is empty. Skipping.")
            return None
    except FileNotFoundError:
        print(f"[ERROR] CSV file not found for {csv_filename}. Skipping.")
        return None
    except pd.errors.EmptyDataError:
        print(f"[ERROR] CSV file is empty for {csv_filename}. Skipping.")
        return None

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
    print(f"Samples evaluated: {n}")

    if auroc_scores:
        mean_auroc = np.mean(auroc_scores)
        mean_auprc = np.mean(auprc_scores)
        
        print(f" Mean AUROC (mAUC) across {len(auroc_scores)} valid labels: {mean_auroc:.4f}")
        print(f" Mean AUPRC (mAP) across {len(auprc_scores)} valid labels: {mean_auprc:.4f}")
        
        # Return dictionary with all metrics
        return {
            'Hospital_ID': hospital_id,
            'Samples_Evaluated': n,
            'BCE_Loss': avg_loss,
            'Mean_AUROC': mean_auroc,
            'Mean_AUPRC': mean_auprc
        }
    else:
        print(" [WARN] Could not calculate Mean AUROC/AUPRC (insufficient labels with >1 class).")
        return {
            'Hospital_ID': hospital_id,
            'Samples_Evaluated': n,
            'BCE_Loss': avg_loss,
            'Mean_AUROC': np.nan,
            'Mean_AUPRC': np.nan
        }


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
        img_root = Path.cwd() / "CheXpert-v1.0-small"
    else:
        img_root = Path(img_root) 
    
    global_ckpt = f"global/round{round_id}_global.pt"
    
    print(f"[INFO] Using device: {DEVICE}")
    
    # Load the global model
    global_model = load_global_model(global_ckpt, DEVICE)
    if global_model is None:
        return None
    
    # ===== Main Execution =====
    print("\n=======================================================")
    print(f"Starting Federated Evaluation on {len(hospital_ids)} Hospitals")
    print("=======================================================")
    
    # List to store the summary dictionary from each hospital
    global_results_list = []
    
    for hospital_id in hospital_ids:
        result = evaluate_hospital(hospital_id, global_model, hospital_dir, img_root, batch_size, DEVICE)
    
        if result is not None:
            global_results_list.append(result)
    
    # Convert list of results to DataFrame
    if not global_results_list:
        print("[ERROR] No valid evaluation results collected.")
        return None
    
    df_results = pd.DataFrame(global_results_list)
    
    # Final calculation of the average metrics across all hospitals
    print("\n=======================================================")
    print("FINAL GLOBAL AVERAGE METRICS (Across All Hospitals)")
    print("=======================================================")
    
    # Initialize variables
    final_mean_auroc = np.nan
    final_mean_auprc = np.nan
    final_weighted_bce = np.nan
    
    if not df_results.empty:
        # Calculate the simple mean for AUROC and AUPRC (Average of means)
        final_mean_auroc = df_results['Mean_AUROC'].mean()
        final_mean_auprc = df_results['Mean_AUPRC'].mean()
    
        # --- FIX: Calculate the Weighted Mean for BCE Loss ---
        losses = df_results['BCE_Loss'].to_numpy()
        samples = df_results['Samples_Evaluated'].to_numpy()
    
        weighted_total_loss = np.sum(losses * samples)
        total_samples = np.sum(samples)
        final_weighted_bce = weighted_total_loss / total_samples
    
        # Print final grand average to console
        print(f"Weighted Mean BCE Loss: {final_weighted_bce:.4f}")
        print(f"Grand Mean AUROC (mAUC) across {len(df_results)} hospitals: {final_mean_auroc:.4f}")
        print(f"Grand Mean AUPRC (mAP) across {len(df_results)} hospitals: {final_mean_auprc:.4f}")
    
        # Create the grand mean summary row for the CSV
        grand_mean_row = pd.Series({
            'Hospital_ID': 'Global_Mean (Weighted BCE Loss)',
            'Samples_Evaluated': f"N={len(df_results)} (Total Samples: {total_samples})",
            'BCE_Loss': final_weighted_bce,
            'Mean_AUROC': final_mean_auroc,
            'Mean_AUPRC': final_mean_auprc
        })
    
        # Append the grand mean row to the DataFrame
        df_results.loc[len(df_results)] = grand_mean_row
    
    else:
        print("Could not calculate Grand Mean metrics (No valid hospital metrics collected).")
    
    print("\nFederated Evaluation Complete.")
    print("=======================================================")
    
    # Save results to CSV
    Path("eval_results").mkdir(exist_ok=True)
    output_filename = f"eval_results/global_evaluation_round{round_id}.csv"
    try:
        df_results.to_csv(output_filename, index=False)
        print(f"\n[INFO] Results successfully saved to {output_filename}")
    except Exception as e:
        print(f"[ERROR] Failed to save results to CSV: {e}")
        import traceback
        traceback.print_exc()
    
    # Return summary dictionary
    result = {
        'round_id': round_id,
        'hospital_results': global_results_list,
        'grand_mean_auroc': final_mean_auroc,
        'grand_mean_auprc': final_mean_auprc,
        'grand_mean_loss': final_weighted_bce
    }
    
    return result


if __name__ == "__main__":
    # For backward compatibility, allow running as script
    import sys
    if len(sys.argv) >= 2:
        round_id = int(sys.argv[1])
        # Default hospital IDs if not provided
        hospital_ids = range(5,11)
        eval_global_all(round_id, hospital_ids)
    else:
        print("Usage: python eval_global_new.py <round_id> [hospital_id1] [hospital_id2] ...")
