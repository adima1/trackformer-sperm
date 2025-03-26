import os
import json
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class COCODetectionDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms

        with open(annotation_file, 'r') as f:
            coco = json.load(f)

        self.images = coco['images']
        self.annotations = coco['annotations']
        self.categories = coco['categories']

        # קישור בין תמונה לרשימת תוויות
        self.img_id_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_id_to_anns:
                self.img_id_to_anns[img_id] = []
            self.img_id_to_anns[img_id].append(ann)

        # מיפוי ID לתמונה
        self.id_to_filename = {img['id']: img['file_name'] for img in self.images}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info['id']
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")

        if self.transforms:
            image = self.transforms(image)

        # קבלת התוויות לתמונה זו
        anns = self.img_id_to_anns.get(img_id, [])
        boxes = []
        labels = []
        track_ids = []
        for ann in anns:
            boxes.append(ann['bbox'])  # [x, y, w, h]
            labels.append(ann['category_id'])
            track_ids.append(ann.get('track_id', -1))

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        track_ids = torch.tensor(track_ids, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'track_ids': track_ids,
            'image_id': torch.tensor([img_id])
        }

        return image, target


def load_data(batch_size=2):
    image_dir = r"C:\Users\User\PycharmProjects\trackformer-sperm\data\sperm_video_02\frames"
    annotation_file = r"C:\Users\User\PycharmProjects\trackformer-sperm\data\annotations.json"

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = COCODetectionDataset(image_dir, annotation_file, transforms=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    return dataloader
