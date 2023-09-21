import pyrealsense2 as rs

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start the pipeline
pipeline.start(config)

try:
    # Wait for the next set of frames
    frames = pipeline.wait_for_frames()

    # Get the intrinsics of the depth camera
    depth_sensor = (
        frames.get_depth_frame().profile.as_video_stream_profile().get_intrinsics()
    )
    print("Depth Camera Intrinsics:")
    print(f"Width: {depth_sensor.width}, Height: {depth_sensor.height}")
    print(f"Focal Length: ({depth_sensor.fx}, {depth_sensor.fy})")
    print(f"Principal Point: ({depth_sensor.ppx}, {depth_sensor.ppy})")
    print(f"Distortion Coefficients: {depth_sensor.coeffs}")

    # Get the intrinsics of the color camera
    color_sensor = (
        frames.get_color_frame().profile.as_video_stream_profile().get_intrinsics()
    )
    print("\nColor Camera Intrinsics:")
    print(f"Width: {color_sensor.width}, Height: {color_sensor.height}")
    print(f"Focal Length: ({color_sensor.fx}, {color_sensor.fy})")
    print(f"Principal Point: ({color_sensor.ppx}, {color_sensor.ppy})")
    print(f"Distortion Coefficients: {color_sensor.coeffs}")

finally:
    # Stop the RealSense pipeline
    pipeline.stop()
