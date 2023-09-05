import cv2
from ultralytics import YOLO
import pyrealsense2 as rs

from ultralytics import YOLO
import cv2
import torch
import numpy as np

model = YOLO('yolov8n-seg.pt')
results = model('https://ultralytics.com/images/bus.jpg', imgsz=640)
img = cv2.imread('bus.jpg')
img = cv2.resize(img, (480, 640))

for result in results:
    print("1")
    for mask in result.masks:
        m = torch.squeeze(mask.data)
        composite = torch.stack((m, m, m), 2)
        tmp = img * composite.cpu().numpy().astype(np.uint8)
        cv2.imshow("result", tmp)
        cv2.waitKey(0)

# class DetectedObject:
#     def __init__(self, mask, box, label, score, class_name):
#         self.mask = mask
#         self.box = box
#         self.label = label
#         self.score = score
#         self.class_name = class_name

# # Load the image
# img = cv2.imread("/home/pedrobolsa/.darwin/datasets/pedro-team/tomates/images/20230624_151803.jpg")
# H, W, _ =   img.shape

# # Create the YOLOv8 model
# model = YOLO("./runs/segment/tomate/weights/best.pt")

# # Perform the prediction
# results = model.predict(img, show=True)

# detected_objects = []


# for result in results:
#     print(result)
#     for j, mask in enumerate(result.masks.data):
#         mask = mask.cpu().numpy() * 255
#         mask = cv2.resize(mask, (W,H))
#         # cv2.imwrite(f"./output{j}.png", mask)





# while True:
#     pass





# # Check if masks are available in the result
# if results[0].masks is not None:
#     # Convert mask to numpy array
#     print(results[0].masks)
#     masks = results[0].masks.numpy()

#     # Get the first mask
#     mask = masks[0]

#     # Apply the mask to the image
#     segmented_img = cv2.bitwise_and(img, img, mask=mask)

#     # Display the segmented image
#     cv2.imshow("Segmented Image", segmented_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()