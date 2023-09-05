import cv2
from ultralytics import YOLO

class DetectedObject:
    def __init__(self, mask, box, label, score, class_name):
        self.mask = mask
        self.box = box
        self.label = label
        self.score = score
        self.class_name = class_name

# Load the image
img = cv2.imread("./datasets/treeDataset/test/images/IMG_20230515_110505_2_jpg.rf.3ed88a390de5777d8011da13d39705c3.jpg")
H, W, _ =   img.shape

model = YOLO('./runs/segment/arvore/weights/best.pt')
model.predict(source=img, show=True)

# Perform the prediction
results = model.predict(img)

detected_objects = []


for result in results:
    print(result)
    for j, mask in enumerate(result.masks.data):
        box = results[0].boxes[j]
        label = results[0].names[int(box.cls[0])]
        mask = mask.cpu().numpy() * 255
        mask = cv2.resize(mask, (W,H))
        # cv2.imwrite(f"./output{j}.png", mask)

cv2.waitKey(0)





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