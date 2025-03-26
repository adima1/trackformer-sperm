import cv2
import os
import numpy as np
import json
from scipy.optimize import linear_sum_assignment

# נתיבים
images_folder = r"C:\Users\User\Desktop\new_project\20_images\Image"
output_folder = r"C:\Users\User\Desktop\new_project\20_images\TrackedImages"
output_json_path = r"C:\Users\User\Desktop\new_project\20_images\annotations_coco_trackformer.json"

os.makedirs(output_folder, exist_ok=True)

coco_output = {
    "images": [],
    "annotations": [],
    "categories": [{"id": 0, "name": "sperm", "supercategory": "object"}]
}
annotation_id = 1
image_id = 1
next_track_id = 1


# פונקציית זיהוי

def detect_white_objects_simple(frame, min_area=50):
    frame = cv2.convertScaleAbs(frame, alpha=2.0, beta=50)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 230), (255, 30, 255))
    blurred_mask = cv2.GaussianBlur(mask, (3, 3), 0)
    contours, _ = cv2.findContours(blurred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, w, h))
    return boxes


# חישוב מרחק בין תיבות (מרכזים)
def box_center(box):
    x, y, w, h = box
    return np.array([x + w / 2, y + h / 2])


def match_boxes(prev_boxes, curr_boxes, threshold=50):
    cost_matrix = np.zeros((len(prev_boxes), len(curr_boxes)))
    for i, b1 in enumerate(prev_boxes):
        for j, b2 in enumerate(curr_boxes):
            cost_matrix[i, j] = np.linalg.norm(box_center(b1) - box_center(b2))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = {}
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] < threshold:
            matches[j] = i  # curr box j matched to prev box i
    return matches


# תיוג ראשי
prev_boxes = []
prev_ids = []
frame_files = sorted([f for f in os.listdir(images_folder) if f.endswith(".png") or f.endswith(".jpg")])

for frame_index, image_file in enumerate(frame_files):
    image_path = os.path.join(images_folder, image_file)
    output_path = os.path.join(output_folder, image_file)
    frame = cv2.imread(image_path)
    if frame is None:
        continue

    height, width = frame.shape[:2]
    boxes = detect_white_objects_simple(frame, min_area=20)

    # התאמה לפריים קודם
    track_ids = []
    matches = match_boxes(prev_boxes, boxes) if prev_boxes else {}
    for i, box in enumerate(boxes):
        if i in matches:
            matched_prev_idx = matches[i]
            track_ids.append(prev_ids[matched_prev_idx])
        else:
            track_ids.append(next_track_id)
            next_track_id += 1

    # ציור
    for (x, y, w, h), tid in zip(boxes, track_ids):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {tid}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imwrite(output_path, frame)

    # COCO image entry
    coco_output["images"].append({
        "id": image_id,
        "file_name": image_file,
        "width": width,
        "height": height,
        "frame_id": frame_index
    })

    for (x, y, w, h), tid in zip(boxes, track_ids):
        coco_output["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 0,
            "bbox": [x, y, w, h],
            "area": w * h,
            "iscrowd": 0,
            "track_id": tid
        })
        annotation_id += 1

    image_id += 1
    prev_boxes = boxes
    prev_ids = track_ids

# שמירת JSON
with open(output_json_path, "w") as f:
    json.dump(coco_output, f, indent=2)

print(f"\n✅ COCO-Track JSON with consistent track_ids saved to: {output_json_path}")
