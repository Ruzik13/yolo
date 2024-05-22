from ultralytics import YOLO
model = YOLO('yolov8n.pt')
for i in range (1,21): results = model(f'photo{i}.jpg', save = True)


