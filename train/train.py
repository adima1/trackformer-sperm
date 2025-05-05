import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from models.trackformer_model import TrackFormerModel
from utils.coco_utils import load_data
from loss import HungarianMatcher, SetCriterion
import os

# יצירת תקיית שמירה
os.makedirs("checkpoints", exist_ok=True)

# Hyperparameters
NUM_CLASSES = 2
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

# Paths to new simple data
image_dir = r"C:\tracformer_modle\all_image_of_sperm\Images_whit_B_BOX"
annotation_file = r"C:\tracformer_modle\all_image_of_sperm\annotations_simple.json"

# DataLoader מותאם לדאטה החדש
dataloader = load_data(image_dir=image_dir, annotation_file=annotation_file, batch_size=2)

# מעקב אחרי איבוד
epoch_losses = []

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for images, targets in dataloader:
        images = list(img.to(DEVICE) for img in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        images_tensor = torch.stack(images)
        for t in targets:
            print("LABELS IN BATCH:", t['labels'].tolist())

        outputs = model(images_tensor)

        # חישוב הפסד – בלי track_id
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    epoch_losses.append(total_loss)
    print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {total_loss:.4f}")

    # שמירה של המודל
    save_dir = r"C:\tracformer_modle\trackformer-sperm\checkpoints_all"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"trackformer_epoch{epoch + 1}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved to {save_path}")

# ציור גרף הפסד
plt.figure(figsize=(8,5))
plt.plot(range(1, EPOCHS+1), epoch_losses, marker='o')
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

