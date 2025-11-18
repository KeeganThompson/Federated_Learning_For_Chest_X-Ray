# eval_global.py — evaluate global FedAvg model on one hospital split

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from pathlib import Path

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
            img_path = Path(*parts[train_idx:])  # "train/patient00040/..."
        full_path = self.img_root / img_path    # archive/train/patient00040/...

        image = Image.open(full_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        labels = torch.tensor(row[self.label_cols].values.astype("float32"))
        return image, labels

def eval_global(hospital_id, round_id, hospital_dir="hospitals_split", img_root=None, batch_size=32):
    """
    Evaluate global FedAvg model on one hospital split.
    
    Args:
        hospital_id: Hospital ID (integer, e.g., 1, 2, 3)
        round_id: Round number (integer) - used to determine global checkpoint path
        hospital_dir: Directory containing hospital CSV files (default: "hospitals_split")
        img_root: Root directory for images (default: Path.cwd())
        batch_size: Batch size for evaluation (default: 32)
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    if img_root is None:
        img_root = Path.cwd()
    else:
        img_root = Path(img_root)
    
    csv_path = f"{hospital_dir}/{hospital_id}.csv"
    global_ckpt = f"global/round{round_id}_global.pt"
    
    print(f"[INFO] Using device: {DEVICE}")
    print(f"[INFO] Evaluating global model on hospital {hospital_id}, round {round_id}")
    
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
    
    # ===== load Global Model =====
    model = models.densenet121(weights=None)
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, 14),
        nn.Sigmoid()
    )
    
    try:
        state_dict = torch.load(global_ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print("[WARN] missing keys:", missing[:5], "| unexpected:", unexpected[:5])
        else:
            print("[INFO] State dict loaded without key issues.")
    except FileNotFoundError:
        print(f"[ERROR] Global checkpoint not found at {global_ckpt}. Skipping evaluation.")
        return None
    
    model = model.to(DEVICE)
    model.eval()
    
    criterion = nn.BCELoss()
    
    # ===== Evaluation Loop =====
    total_loss = 0.0
    n = 0
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
    
            outputs = model(imgs)
            loss = criterion(outputs, labels)
    
            total_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)
    
    avg_loss = total_loss / max(n, 1)
    
    print(f"\n✅ Global model BCE loss on {csv_path}: {avg_loss:.4f}")
    print(f"   (samples evaluated: {n})")
    
    result = {
        'hospital_id': hospital_id,
        'round_id': round_id,
        'bce_loss': avg_loss,
        'samples_evaluated': n
    }
    return result


if __name__ == "__main__":
    # For backward compatibility, allow running as script
    import sys
    if len(sys.argv) >= 3:
        hospital_id = int(sys.argv[1])
        round_id = int(sys.argv[2])
        eval_global(hospital_id, round_id)
    else:
        print("Usage: python eval_global.py <hospital_id> <round_id>")
