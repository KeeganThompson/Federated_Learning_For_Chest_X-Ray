# eval_global.py — evaluate global FedAvg model on one hospital split

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from pathlib import Path

# ===== CONFIG =====
CSV_PATH = "hospitals_split/hospital_01.csv"   
IMG_ROOT = Path("archive")                   
GLOBAL_CKPT = "global/round1_global.pt"       
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
            img_path = Path(*parts[train_idx:])  # "train/patient00040/..."
        full_path = self.img_root / img_path    # archive/train/patient00040/...

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

# ===== load Global Model =====
model = models.densenet121(weights=None)
model.classifier = nn.Sequential(
    nn.Linear(model.classifier.in_features, 14),
    nn.Sigmoid()
)

state_dict = torch.load(GLOBAL_CKPT, map_location="cpu")
missing, unexpected = model.load_state_dict(state_dict, strict=False)
if missing or unexpected:
    print("[WARN] missing keys:", missing[:5], "| unexpected:", unexpected[:5])
else:
    print("[INFO] State dict loaded without key issues.")

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

print(f"\n✅ Global model BCE loss on {CSV_PATH}: {avg_loss:.4f}")
print(f"   (samples evaluated: {n})")
