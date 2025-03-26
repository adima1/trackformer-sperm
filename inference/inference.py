import torch
import cv2
import os
import numpy as np
from torchvision import transforms
from PIL import Image
from models.trackformer_model import TrackFormerModel

# הגדרות
IMAGE_PATH = r"C:\Users\User\Desktop\new_project\1000_images\Image\frame_0000.png"
MODEL_WEIGHTS = r"C:\Users\User\PycharmProjects\trackformer-sperm\checkpoints\trackformer_epoch5.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# טענת המודל
model = TrackFormerModel()
model.to(DEVICE)
model.eval()
if MODEL_WEIGHTS:
    model.load_state_dict(torch.load(MODEL_WEIGHTS))

# טרנספורם
transform = transforms.Compose([
    transforms.ToTensor(),
])

# טעינת תמונה
image_pil = Image.open(IMAGE_PATH).convert("RGB")
image_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)  # [1, 3, H, W]

# הרצת המודל
with torch.no_grad():
    outputs = model(image_tensor)

# חילוץ תחזיות
pred_logits = outputs['pred_logits'][0]     # [num_queries, num_classes+1]
pred_boxes = outputs['pred_boxes'][0]       # [num_queries, 4] - normalized

# ניקוי תחזיות: ניקח רק אלו שהמודל בטוח בהן (score גבוה)
scores = pred_logits.softmax(-1)[..., :-1].max(-1).values  # confidence
keep = scores > 0.7
boxes = pred_boxes[keep].cpu().numpy()
scores = scores[keep].cpu().numpy()

# המרת תיבות מ-normalized ל-absolute
h, w = image_pil.size[1], image_pil.size[0]
boxes_abs = []
for box in boxes:
    cx, cy, bw, bh = box
    x1 = int((cx - bw/2) * w)
    y1 = int((cy - bh/2) * h)
    x2 = int((cx + bw/2) * w)
    y2 = int((cy + bh/2) * h)
    boxes_abs.append((x1, y1, x2, y2))

# ציור על התמונה
image_np = np.array(image_pil)
for (x1, y1, x2, y2), score in zip(boxes_abs, scores):
    cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image_np, f"{score:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# הצגה
cv2.imshow("Prediction", image_np)
cv2.waitKey(0)
cv2.destroyAllWindows()
