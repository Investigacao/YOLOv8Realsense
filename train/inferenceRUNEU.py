import cv2
from ultralytics import YOLO
import pyrealsense2 as rs
import os

# Folder path containing the images
folder_path = "/home/pedrobolsa/Downloads/archive/train/malignant/"

# Output video file path
output_video_path = "./video2.avi"

# Load the first image to get the dimensions
first_image_path = os.path.join(folder_path, os.listdir(folder_path)[0])
first_image = cv2.imread(first_image_path)
H, W, _ = first_image.shape

# Define the video codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 3  # Adjust the desired frames per second
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H))

# Create the YOLOv8 model
model = YOLO("./runs/segment/train4/weights/best.pt")

# Iterate over the images in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg"):
        # Load the image
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)

        # Perform the prediction
        results = model.predict(img, show=True)

        annotated_frame = results[0].plot()

        # Write the image to the output video
        output_video.write(annotated_frame)

        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            break

# Release the video writer and close any open windows
output_video.release()
cv2.destroyAllWindows()
