import cv2
from ultralytics import YOLO
import pyrealsense2 as rs
import numpy as np
import time
from math import *
import argparse
import json
from threading import Thread, Event
import websocket
from flask import Flask, jsonify
import multiprocessing
# from sorting import custom_sort

# OFFSET robot Front
OFFSET_FRONT = 0.035
#OFFSET robot Height
OFFSET_HEIGHT = 0.05

# Color_Camera
HFOV = 69
VFOV = 42

RESOLUTION_X = 1280
RESOLUTION_Y = 720

CENTER_POINT_X = RESOLUTION_X / 2
CENTER_POINT_Y = RESOLUTION_Y / 2

manager = multiprocessing.Manager()
messageDigitalTwin = manager.list()
messageRobot = manager.list()
sendMessage = False
tomatoDetected = Event()


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', help='name.bag to stream using file instead of webcam')
parser.add_argument('-ws', '--wsURL', help='url to connect to websocket server')
parser.add_argument('-fl', '--flask', type=int, help='flask server port')
args = parser.parse_args()

pipeline = rs.pipeline()
config = rs.config()

if args.flask:
    pass
    # # Create and start a separate thread for the Flask server
    # flaskThread = Thread(target=run_flask_server)
    # flaskThread.daemon = True  # Set the thread as a daemon so it will exit when the main program exits
    # flaskThread.start()

if args.wsURL:
    pass
    # ws = websocket.WebSocket()
    # ws.connect(args.wsURL)
    # sendMessageTopics(ws, ["robot", "digital_twin"])

    # wsThread = Thread(target=websocket_thread, args=(ws,))
    # wsThread.daemon = True
    # wsThread.start()

if args.file:
    try:
        config.enable_device_from_file(args.file, repeat_playback=False)
        
    except:
        print("Cannot enable device from: '{}'".format(args.file))
else:
    config.enable_stream(rs.stream.depth, RESOLUTION_X, RESOLUTION_Y, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, RESOLUTION_X, RESOLUTION_Y, rs.format.bgr8, 30)

# Create a pipeline and configure playback
# config.enable_device_from_file(bag_file_path)
profile = pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

# Get the color sensor profile
color_profile = rs.video_stream_profile(pipeline.get_active_profile().get_stream(rs.stream.color))

# Get the width and height of the color frames
color_width, color_height = color_profile.get_intrinsics().width, color_profile.get_intrinsics().height

depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
print(f"Depth Scale: {depth_scale:.4f}m")

# Load the YOLOv8 model
model = YOLO('./runs/segment/tomate/weights/best.pt')

# Open the video file
# video_path = "./output.mp4"
# cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while True:
    time_start = time.time()
    
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    # get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    # colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

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

    time_end = time.time()
    total_time = time_end - time_start
    print("FPS: {:.2f}\n".format(1/total_time))

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
