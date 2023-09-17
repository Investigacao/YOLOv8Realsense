import pyrealsense2 as rs
import numpy as np
import cv2

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

# Create OpenCV window for color stream
cv2.namedWindow('Color Stream', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Color Stream', color_width, color_height)

# Create an OpenCV VideoWriter object to save the color frames as .mp4
output_file = 'output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(output_file, fourcc, 30, (color_width, color_height))

# Main loop to process frames from the bag file
while True:
    # Wait for the next available frame
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    if not color_frame:
        break

    # Convert the color frame to a numpy array
    color_image = np.asanyarray(color_frame.get_data())[:, :, ::-1]

    # Display the color stream
    cv2.imshow('Color Stream', color_image)

    # Write the color frame to the video file
    output_video.write(color_image)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
pipeline.stop()
output_video.release()
