import cv2
import os
import numpy as np
import json

# נתיבים
images_folder = r"C:\Users\User\Desktop\new_project\20_images\Image"
output_folder = r"C:\Users\User\Desktop\new_project\20_images\TrackedImages"
output_json_path = r"C:\Users\User\Desktop\new_project\20_images\annotations_coco_trackformer.json"

os.makedirs(output_folder, exist_ok=True)

# מבנה COCO + TrackFormer
coco_output = {
    "images": [],
    "annotations": [],
    "categories": [{"id": 0, "name": "sperm", "supercategory": "object"}]
}
annotation_id = 1
image_id = 1
track_id_counter = 1  # לכל תיבה, מזהה ייחודי (לא עקבי עדיין בין פריימים)

def detect_white_objects_simple(frame, min_area=50):
    frame = cv2.convertScaleAbs(frame, alpha=2.0, beta=50)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 230])
    upper_white = np.array([255, 30, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    blurred_mask = cv2.GaussianBlur(mask, (3, 3), 0)
    contours, _ = cv2.findContours(blurred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, w, h))
    return boxes

def draw_tracked_boxes(frame, boxes):
    for idx, (x, y, w, h) in enumerate(boxes, start=1):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {idx}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame

def annotate_and_save_coco(images_folder, output_folder, min_area=50):
    global annotation_id, image_id, track_id_counter

    frame_files = sorted([f for f in os.listdir(images_folder) if f.endswith(".png") or f.endswith(".jpg")])

    for frame_index, image_file in enumerate(frame_files):
        image_path = os.path.join(images_folder, image_file)
        output_path = os.path.join(output_folder, image_file)

        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Failed to load image: {image_path}")
            continue

        height, width = frame.shape[:2]
        boxes = detect_white_objects_simple(frame, min_area=min_area)

        if len(boxes) == 0:
            print(f"No objects detected in: {image_file}")
        else:
            frame_with_tracking = draw_tracked_boxes(frame, boxes)
            cv2.imwrite(output_path, frame_with_tracking)
            print(f"Saved annotated frame: {output_path}")

            # שמירת מידע על התמונה + frame_id
            coco_output["images"].append({
                "id": image_id,
                "file_name": image_file,
                "width": width,
                "height": height,
                "frame_id": frame_index  # הוספת frame_id לפי הסדר
            })

            for (x, y, w, h) in boxes:
                coco_output["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 0,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "track_id": track_id_counter
                })
                annotation_id += 1
                track_id_counter += 1

            image_id += 1

# הרצת הקוד
annotate_and_save_coco(images_folder, output_folder, min_area=20)

# שמירת JSON
with open(output_json_path, "w") as json_file:
    json.dump(coco_output, json_file, indent=2)

print(f"\n✅ קובץ JSON מוכן לשימוש עם TrackFormer: {output_json_path}")
