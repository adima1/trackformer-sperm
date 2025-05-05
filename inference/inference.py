import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
from PIL import Image
from models.trackformer_model import TrackFormerModel

# ======= הגדרות =======
CHECKPOINT_PATH = r"C:\tracformer_modle\trackformer-sperm\checkpoints_all\trackformer_epoch5.pth"
IMAGE_PATH = r"C:\tracformer_modle\all_image_of_sperm\image_before_tagging\frame_0126.png"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======= טען מודל =======
model = TrackFormerModel(num_classes=2, hidden_dim=256, num_queries=100)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ======= פונקציה להצגה =======
def visualize_single_image(image_path):
    transform = transforms.Compose([transforms.ToTensor()])
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)

    pred_logits = outputs['pred_logits'][0]  # [num_queries, num_classes]
    pred_boxes = outputs['pred_boxes'][0]    # [num_queries, 4]
    scores = pred_logits.softmax(-1)[..., :-1].max(-1).values  # confidence ללא no-object

    keep = scores > 0.5
    boxes = pred_boxes[keep].cpu()
    scores = scores[keep].cpu()

    # === תצוגה ===
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for box, score in zip(boxes, scores):
        x, y, w, h = box
        rect = patches.Rectangle((x * image.width, y * image.height),
                                 w * image.width, h * image.height,
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x * image.width, y * image.height, f"{score:.2f}", color='red', fontsize=10)

    ax.set_title(f"Predictions for {os.path.basename(image_path)}")
    ax.axis("off")
    plt.show()

# ======= הרצה =======
visualize_single_image(IMAGE_PATH)
