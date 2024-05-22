import random
from ultralytics import YOLO
import cv2

video = cv2.VideoCapture(0)
out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (1920, 1080))

model = YOLO("yolov8n.pt")
colors = [[random.randint(0,255) for i in range(3)] for name in model.names]
confidence_treshold = 0.7

while True:
    ret,frame = video.read()
    if not ret : break
    frame = cv2.flip(frame,1)
    detections = model(frame)[0]
    for data in detections.boxes.data.tolist(): #[xmin, ymin, xmas, ymax,confidence, class_id]
        class_name = detections.names[int(data[5])]
        class_id = int(data[5])
        xmin = int(data[0])
        ymin = int(data[1])
        xmax = int(data[2])
        ymax = int(data[3])
        confidence = data[4]
        if confidence < confidence_treshold : continue
        r,g,b = colors[class_id]        
        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0,255,0),2)
        cv2.putText(frame, class_name, (xmin, ymin-20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(r, g, b), thickness=2)
    out.write(frame)
    cv2.imshow('winnner',frame)
    if cv2.waitKey(2) == ord('q') : break
video.release()
out.release()
cv2.destroyAllWindows()

