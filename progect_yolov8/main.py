from ultralytics import YOLO

model = YOLO('yolov8n.yaml')  # או s/m/l בהתאם למשאבים שלך
model.train(data='dataset/data.yaml', epochs=100, imgsz=256)


