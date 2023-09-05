from ultralytics import YOLO
import cv2
import torch
import numpy as np

model = YOLO('yolov8n-seg.pt')
results = model('https://ultralytics.com/images/bus.jpg', imgsz=640)
img = cv2.imread('bus.jpg')
img = cv2.resize(img, (480, 640))

for index, mask in enumerate(results[0].masks):
    box = results[0].boxes[index]
    print("object class is", results[0].names[int(box.cls[0])])
    m = torch.squeeze(mask.data)
    composite = torch.stack((m, m, m), 2)
    tmp = img * composite.cpu().numpy().astype(np.uint8)
    cv2.imshow("result", tmp)
    cv2.waitKey(0)