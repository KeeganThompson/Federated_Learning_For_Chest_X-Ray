import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# === Custom Dataset ===
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
        # Handle NaN and -1 (CheXpert uncertain labels → 0 baseline)
        self.df[self.label_cols] = self.df[self.label_cols].fillna(0).replace(-1, 0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(row["Path"])
        # Strip leading "train/" if present
        if img_path.parts[0] == "train":
            img_path = Path(*img_path.parts[1:])
        full_path = self.img_root / img_path
        image = Image.open(full_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        labels = torch.tensor(row[self.label_cols].values.astype("float32"))
        return image, labels

def run_local(hospital_id, round_id):
    # === CONFIG ===
    GLOBAL_WEIGHTS_PATH = f"global/round{round_id}_global.pt"
    # CSV_PATH = "hospitals_split_skewed/hospital_01.csv"
    CSV_PATH = f"hospitals_split/{hospital_id}.csv"
    IMG_ROOT = Path.cwd()
    BATCH_SIZE = 16
    EPOCHS = 3
    LR = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # SAVE_PATH = "hospital01_skewed_densenet_torchvision.pt"
    # LOSS_SAVE_PATH = f"hospital_01_skewed_loss.csv"
    SAVE_PATH = f"hospital{hospital_id}_densenet_torchvision{round_id}.pt"
    LOSS_SAVE_PATH = f"loss/hospital{hospital_id}_loss{round_id}.csv"

    # === Transforms ===
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # === DataLoader ===
    dataset = CheXpertDataset(CSV_PATH, IMG_ROOT, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # === Torchvision DenseNet121 ===
    model = models.densenet121(weights="IMAGENET1K_V1")
    # Replace classifier for 14-label multi-label task
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, 14),
        nn.Sigmoid()
    )
    model = model.to(DEVICE)

    try:
        # Load the global state dictionary from the server
        global_weights = torch.load(GLOBAL_WEIGHTS_PATH, map_location=DEVICE)
        
        # Load the state dictionary into your local model
        model.load_state_dict(global_weights)
        print(f"✅ Successfully loaded global weights from {GLOBAL_WEIGHTS_PATH}.")

    except FileNotFoundError:
        print(f"⚠️ Global weights file not found at {GLOBAL_WEIGHTS_PATH}. Starting training from scratch with ImageNet pretraining.")
    except Exception as e:
        print(f"🛑 Error loading global weights: {e}. Starting training from scratch.")

    # === Loss & Optimizer ===
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    epoch_losses = []

    # === Training Loop ===
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for imgs, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        epoch_loss = running_loss / len(dataset)
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}")

    loss_df = pd.DataFrame({
        'epoch': range(1, EPOCHS + 1),
        'loss': epoch_losses
    })
    loss_df.to_csv(LOSS_SAVE_PATH, index=False)
    print(f"\n✅ Epoch losses saved to {LOSS_SAVE_PATH}")


    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\n✅ Model saved to {SAVE_PATH}")