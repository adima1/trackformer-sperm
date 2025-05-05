import cv2
import os
import json
import numpy as np

# נתיבים
images_folder = r"C:\tracformer_modle\40_image_of_sperm\40_image_befor_tag"
output_folder = r"C:\tracformer_modle\40_image_of_sperm\Images_whit_B_BOX"
output_json_path = r"C:\tracformer_modle\40_image_of_sperm\annotations_simple.json"

os.makedirs(output_folder, exist_ok=True)

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
            boxes.append([x, y, w, h])
    return boxes

# יצירת מבנה JSON פשוט
annotations_simple = {}

frame_files = sorted([f for f in os.listdir(images_folder) if f.endswith(".png") or f.endswith(".jpg")])

for image_file in frame_files:
    image_path = os.path.join(images_folder, image_file)
    output_path = os.path.join(output_folder, image_file)
    frame = cv2.imread(image_path)
    if frame is None:
        continue

    boxes = detect_white_objects_simple(frame, min_area=20)

    # ציור תיבות בלבד
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imwrite(output_path, frame)

    # שמירת הקואורדינטות עבור התמונה
    annotations_simple[image_file] = boxes

# שמירת JSON הפשוט
with open(output_json_path, "w") as f:
    json.dump(annotations_simple, f, indent=2)

print(f"\n✅ JSON פשוט עם רק תיבות לכל תמונה נשמר אל: {output_json_path}")
