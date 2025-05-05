import os
import json
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class SimpleDetectionDataset(Dataset):
    """
    Dataset שמתאים לפורמט פשוט:
    {
        "image1.png": [[x, y, w, h], ...],
        ...
    }
    """
    def __init__(self, image_dir, annotation_file, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms

        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)

        self.image_files = list(self.annotations.keys())

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        img_path = os.path.join(self.image_dir, image_file)
        image = Image.open(img_path).convert("RGB")

        if self.transforms:
            image = self.transforms(image)

        bboxes = self.annotations[image_file]

        boxes = torch.tensor(bboxes, dtype=torch.float32)
        labels = torch.zeros((len(bboxes),), dtype=torch.int64)  # כל תיבה = sperm = label 0
        labels = torch.clamp(labels, min=0, max=0)  # לוודא שאין שגיאות

        # DEBUG: הדפסת תוויות
        if any(labels > 0):
            print(f"[WARNING] Labels > 0 detected in {image_file}: {labels.tolist()}")

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }

        return image, target


def load_data(image_dir, annotation_file, batch_size=2, shuffle=True):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = SimpleDetectionDataset(image_dir, annotation_file, transforms=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda x: tuple(zip(*x)))

    return dataloader
