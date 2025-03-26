import torch
import torch.nn as nn
from models.trackformer_model import TrackFormerModel
from utils.coco_utils import load_data
from loss import HungarianMatcher, SetCriterion
import os
os.makedirs("checkpoints", exist_ok=True)

# Hyperparameters
NUM_CLASSES = 1
HIDDEN_DIM = 256
NUM_QUERIES = 100
LR = 1e-4
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# יצירת המודל
model = TrackFormerModel(num_classes=NUM_CLASSES, hidden_dim=HIDDEN_DIM, num_queries=NUM_QUERIES).to(DEVICE)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# Loss
matcher = HungarianMatcher(cost_class=1.0, cost_bbox=1.0)
criterion = SetCriterion(num_classes=NUM_CLASSES, matcher=matcher)

# DataLoader
dataloader = load_data(batch_size=2)

# Loop בסיסי
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for images, targets in dataloader:
        images = list(img.to(DEVICE) for img in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        images_tensor = torch.stack(images)
        outputs = model(images_tensor)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {total_loss:.4f}")

    # שמירה של המודל
    save_dir = r"C:\Users\User\PycharmProjects\trackformer-sperm\checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"trackformer_epoch{epoch + 1}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved to {save_path}")

