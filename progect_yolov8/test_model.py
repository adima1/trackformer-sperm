import cv2
import os
from ultralytics import YOLO

# טען את המודל שלך
model = YOLO("runs/detect/train3/weights/best.pt")

# נתיב לתיקיית התמונות
image_dir = r"C:\tracformer_modle\all_image_of_sperm\image_before_tagging"
output_video = "sperm_detection_video_2.avi"

# הגדר שם לתיקיית תוצאות
output_dir = "frames_with_boxes"
os.makedirs(output_dir, exist_ok=True)

# קרא את קבצי התמונות לפי סדר
images = sorted([img for img in os.listdir(image_dir) if img.endswith(".png") or img.endswith(".jpg")])

# הגדר משתנה לכתיבת וידאו
first_image = cv2.imread(os.path.join(image_dir, images[0]))
height, width, _ = first_image.shape
fourcc = cv2.VideoWriter_fourcc(*"XVID")
video = cv2.VideoWriter(output_video, fourcc, 4, (width, height))  # 10fps

# עבור על כל תמונה
for image_name in images:
    image_path = os.path.join(image_dir, image_name)

    # הרץ את המודל
    results = model(image_path)[0]

    # קרא את התמונה
    img = cv2.imread(image_path)

    # עבור כל תיבת זיהוי
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # שמור את התמונה
    output_path = os.path.join(output_dir, image_name)
    cv2.imwrite(output_path, img)

    # הוסף לפריים לווידאו
    video.write(img)

video.release()
print("סרטון נשמר בשם:", output_video)
