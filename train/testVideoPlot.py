import cv2
from ultralytics import YOLO
import pyrealsense2 as rs
import numpy as np


# Path to the bag file
bag_file_path = '/home/pedrobolsa/Downloads/20230428_115454.bag'

# Create a pipeline and configure playback
pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file(bag_file_path)
pipeline.start(config)

# Get the color sensor profile
color_profile = rs.video_stream_profile(pipeline.get_active_profile().get_stream(rs.stream.color))

# Get the width and height of the color frames
color_width, color_height = color_profile.get_intrinsics().width, color_profile.get_intrinsics().height

# Load the YOLOv8 model
model = YOLO('./runs/segment/tomate/weights/best.pt')

# Open the video file
# video_path = "./output.mp4"
# cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while True:

    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    if not color_frame:
        break

    # Convert the color frame to a numpy array
    color_image = np.asanyarray(color_frame.get_data())[:, :, ::-1]

    # Run YOLOv8 inference on the frame
    results = model(color_image)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # # Add additional information to the annotated frame
    # additional_info = "Additional Information"
    # cv2.putText(annotated_frame, additional_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the annotated frame
    cv2.imshow("YOLOv8 Inference", annotated_frame)

    # Check if the user pressed 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cv2.destroyAllWindows()




# import cv2
# from ultralytics import YOLO

# # Load the YOLOv8 model
# model = YOLO('./runs/segment/tomate/weights/best.pt')

# # Open the video file
# video_path = "./output.mp4"
# cap = cv2.VideoCapture(video_path)

# # Loop through the video frames
# while cap.isOpened():
#     # Read a frame from the video
#     success, frame = cap.read()

#     if success:
#         # Run YOLOv8 inference on the frame
#         results = model(frame)

#         # Visualize the results on the frame
#         annotated_frame = results[0].plot()

#         # # Add additional information to the annotated frame
#         # additional_info = "Additional Information"
#         # cv2.putText(annotated_frame, additional_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#         # Display the annotated frame
#         cv2.imshow("YOLOv8 Inference", annotated_frame)

#         # Check if the user pressed 'q' to exit
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# # Release the video capture and close the window
# cap.release()
# cv2.destroyAllWindows()
