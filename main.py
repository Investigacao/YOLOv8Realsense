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
from myapp.sorting import custom_sort
import subprocess
import base64
import math
import sys

# OFFSET robot Front
OFFSET_FRONT = 0.15
# OFFSET robot Height
OFFSET_HEIGHT = 0.18

# Color_Camera
HFOV = 69
VFOV = 42

RESOLUTION_X = 1280
RESOLUTION_Y = 720

CENTER_POINT_X = RESOLUTION_X / 2
CENTER_POINT_Y = RESOLUTION_Y / 2

app = Flask(__name__)

manager = multiprocessing.Manager()
messageDigitalTwin = manager.list()
messageRobot = manager.list()
sendMessage = False
tomatoDetected = Event()


def handleThreadResponse(ws, msg):
    global messageDigitalTwin
    global messageRobot
    global sendMessage

    sendMessage = True
    tomatoDetected.wait()
    if msg["command"] == "D1":
        messageRobot = [messageRobot[0]]
        print(f"messageRobot: {messageRobot}")
    # Create a set of the target objects
    # sendMessageTopic(ws, "digital_twin", list(messageDigitalTwin))
    sendMessageTopic(ws, "robot", list(messageRobot))

    tomatoDetected.clear()
    sendMessage = False


def websocket_thread(ws):
    while True:
        msg = json.loads(ws.recv())
        print(f"Received: {msg}")

        # if(msg.upper() != "XYZ"):
        #     continue
        handleThreadResponse(ws, msg)


# def sendMessageTopics(ws, topics):
#     for topic in topics:
#         message = {"topic": topic, "data": f"Realsense connected to {topic}"}
#         ws.send(json.dumps(message))


def sendMessageTopic(ws, topic, message):
    ws.send(json.dumps({"topic": topic, "data": {"command": "M", "data": message}}))


@app.route("/get_message_digital_twin", methods=["GET"])
def get_message_digital_twin():
    global messageDigitalTwin
    print(f"messageDigitalTwin FLASK: {list(messageDigitalTwin)}")
    return jsonify(list(messageDigitalTwin))


# Define a function to run the Flask server in a separate thread
def run_flask_server():
    app.run(host="0.0.0.0", port=5000)


class IMU:
    def __init__(self):
        self.yaw = 0
        self.roll = 0
        self.pitch = 0
        self.exit_event = Event()

    def startIMU(self):
        self.imu_config = rs.config()

        self.imu_config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
        self.imu_config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

    def update_imu(self):
        self.imu_pipe = rs.pipeline()
        self.imu_pipe.start(self.imu_config)
        try:
            first = True
            alpha = 0.98
            totalgyroangleY = 0
            while not self.exit_event.is_set():
                # Wait for a coherent pair of frames: depth and color
                mot_frames = self.imu_pipe.wait_for_frames()
                # ? gather IMU data
                accel = mot_frames[0].as_motion_frame().get_motion_data()
                gyro = mot_frames[1].as_motion_frame().get_motion_data()

                ts = mot_frames.get_timestamp()

                # calculation for the first frame
                if first:
                    first = False
                    last_ts_gyro = ts

                    # accelerometer calculation
                    accel_angle_z = math.degrees(math.atan2(accel.y, accel.z))
                    accel_angle_x = math.degrees(
                        math.atan2(
                            accel.x, math.sqrt(accel.y * accel.y + accel.z * accel.z)
                        )
                    )
                    accel_angle_y = math.degrees(math.pi)

                    continue
                # calculation for the second frame onwards

                # gyrometer calculations
                dt_gyro = (ts - last_ts_gyro) / 1000
                last_ts_gyro = ts

                gyro_angle_x = gyro.x * dt_gyro
                gyro_angle_y = gyro.y * dt_gyro
                gyro_angle_z = gyro.z * dt_gyro

                dangleX = gyro_angle_x * 57.2958
                dangleY = gyro_angle_y * 57.2958
                dangleZ = gyro_angle_z * 57.2958

                totalgyroangleX = accel_angle_x + dangleX
                # totalgyroangleY = accel_angle_y + dangleY
                totalgyroangleY = accel_angle_y + dangleY + totalgyroangleY
                totalgyroangleZ = accel_angle_z + dangleZ

                # accelerometer calculation
                accel_angle_z = math.degrees(math.atan2(accel.y, accel.z))
                accel_angle_x = math.degrees(
                    math.atan2(
                        accel.x, math.sqrt(accel.y * accel.y + accel.z * accel.z)
                    )
                )
                # accel_angle_y = math.degrees(math.pi)
                accel_angle_y = 0

                # combining gyrometer and accelerometer angles
                self.roll = totalgyroangleX * alpha + accel_angle_x * (
                    1 - alpha
                )  # Roll
                self.pitch = totalgyroangleZ * alpha + accel_angle_z * (
                    1 - alpha
                )  # Pitch
                self.yaw = totalgyroangleY  # Yaw

                self.yawtest = self.yaw
                self.rolltest = self.roll
                self.pitchtest = self.pitch

                # print("Angle -  X: " + str(round(combinedangleX,2)) + "   Y: " + str(round(combinedangleY,2)) + "   Z: " + str(round(combinedangleZ,2)))

        except Exception as e:
            print(f"Error in IMU update: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        action="store_true",
        help="name.bag to stream using file instead of webcam",
    )
    parser.add_argument(
        "-ws", "--wsURL", action="store_true", help="url to connect to websocket server"
    )
    parser.add_argument("-fl", "--flask", action="store_true", help="flask server port")
    parser.add_argument("-r", "--rtmp", action="store_true", help="rtmp key")
    parser.add_argument(
        "-wsF",
        "--wsFrame",
        help="url to connect to websocket server to send annotated frames",
    )
    parser.add_argument("-i", "--imu", action="store_true", help="Enable IMU reading")
    args = parser.parse_args()

    if args.flask:
        # Create and start a separate thread for the Flask server
        flaskThread = Thread(target=run_flask_server)
        flaskThread.daemon = True  # Set the thread as a daemon so it will exit when the main program exits
        flaskThread.start()

    wsThread = None
    ws = websocket.WebSocket()
    if args.wsURL:
        cookie_value = "topics=ai;device=realsense"

        ws.connect("ws://10.79.179.10:31080/", cookie=cookie_value)
        # ws.connect("ws://localhost:3000/", cookie=cookie_value)
        # sendMessageTopics(ws, ["robot", "digital_twin"])
        wsThread = Thread(target=websocket_thread, args=(ws,))
        wsThread.daemon = True
        wsThread.start()

    if args.rtmp:
        command = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "{}x{}".format(1280, 720),
            "-r",
            "25",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "flv",
            "-flvflags",
            "no_duration_filesize",
            "rtmp://10.79.179.10:31935/live/test",
        ]
        # 192.168.1.203:30439
        p = subprocess.Popen(command, stdin=subprocess.PIPE)

    if args.wsFrame:
        wsFrame = websocket.WebSocket()
        wsFrame.connect(args.wsFrame)

    config = rs.config()
    if args.file:
        try:
            config.enable_device_from_file(
                "./myapp/bagFiles/20230428_115454.bag", repeat_playback=False
            )

        except:
            print("Cannot enable device from: '{}'".format(args.file))
    else:
        config.enable_stream(
            rs.stream.depth, RESOLUTION_X, RESOLUTION_Y, rs.format.z16, 30
        )
        config.enable_stream(
            rs.stream.color, RESOLUTION_X, RESOLUTION_Y, rs.format.bgr8, 30
        )

    # Load the image
    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    imu_pipe = None
    imu_thread = None
    if args.imu:
        imu = IMU()
        imu.startIMU()

        imu_thread = Thread(target=imu.update_imu)
        imu_thread.daemon = True
        imu_thread.start()

    align_to = rs.stream.color
    align = rs.align(align_to)

    # Start streaming

    colorizer = rs.colorizer()

    # Get the color sensor profile
    color_profile = rs.video_stream_profile(
        pipeline.get_active_profile().get_stream(rs.stream.color)
    )

    # Get the width and height of the color frames
    color_width, color_height = (
        color_profile.get_intrinsics().width,
        color_profile.get_intrinsics().height,
    )

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    print(f"Depth Scale: {depth_scale:.4f}m")

    #! NEW TEST
    depth_min = 0.11  # meter
    depth_max = 2.5  # meter
    depth_intrin = (
        profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    )
    color_intrin = (
        profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    )

    depth_to_color_extrin = (
        profile.get_stream(rs.stream.depth)
        .as_video_stream_profile()
        .get_extrinsics_to(profile.get_stream(rs.stream.color))
    )
    color_to_depth_extrin = (
        profile.get_stream(rs.stream.color)
        .as_video_stream_profile()
        .get_extrinsics_to(profile.get_stream(rs.stream.depth))
    )

    #! NEW TEST OVER

    # Create a window for displaying the results
    # cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Detections", 800, 600)

    # model = YOLO("./myapp/models/yolov8m-tomato.pt")
    model = YOLO("./myapp/models/tomato-308-yoloM.pt")

    # when stream is finished, RuntimeError is raised, hence this
    # exception block to capture this
    tempDTList = []
    tempRobotList = []
    try:
        while True:
            print("=====================================")

            # messageDigitalTwin[:] = []
            # messageRobot[:] = []
            tempDTList.clear()
            tempRobotList.clear()
            time_start = time.time()
            try:
                # frames = pipeline.wait_for_frames()
                frames = pipeline.wait_for_frames()
                if frames.size() < 2:
                    print("Not enough frames")
                    break
                    # Inputs are not ready yet
            except RuntimeError:
                pipeline.stop()
                if args.imu:
                    imu.imu_pipe.stop()

            # align the deph to color frame
            aligned_frames = align.process(frames)

            # get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            # colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

            # validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                print("No frames received")

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            if args.file:
                color_image = np.asanyarray(color_frame.get_data())[:, :, ::-1]
            else:
                color_image = np.asanyarray(color_frame.get_data())

            # Perform the prediction
            results = model.predict(
                source=color_image, line_width=5, boxes=False, device=0
            )

            annotated_frame = results[0].plot()

            for result in results:
                if result.masks:
                    for j, mask_xy in enumerate(result.masks.xy):
                        box = result.boxes[
                            j
                        ]  # Access the corresponding box for the current mask
                        xywh = box.xywh[0].tolist()
                        cX, cY, bWidth, bHeight = xywh
                        cX = int(cX)
                        cY = int(cY)

                        #! Tentativa 1
                        # # Create a binary mask based on the pixel coordinates
                        # mask_binary = np.zeros((color_height, color_width), dtype=np.uint8)
                        # mask_x, mask_y = mask_xy[:, 0].astype(int), mask_xy[:, 1].astype(int)  # Separate x and y coordinates
                        # mask_binary[mask_y, mask_x] = 255

                        # # Apply the binary mask to the depth image
                        # masked_depth = depth_image * mask_binary

                        # # Calculate the average depth within the masked region
                        # average_depth = np.mean(masked_depth[mask_binary != 0])
                        # average_depth = round(average_depth * depth_scale, 3)

                        #! Tentativa 2
                        # mask_coordinates = np.array(mask_xy)
                        # segmented_depth = depth_image[mask_coordinates[:, 1].astype(int), mask_coordinates[:, 0].astype(int)]
                        # segmented_depth = segmented_depth * depth_scale
                        # average_depth = np.mean(segmented_depth)

                        # masks_test = results[0].masks.cpu().numpy()
                        # mask_test = masks_test[0]
                        # segmented_img = cv2.bitwise_and(colorized_depth, colorized_depth, mask=mask_test)
                        # cv2.imshow("Segmented Image", segmented_img)

                        average_depth = aligned_depth_frame.get_distance(
                            int(cX), int(cY)
                        )
                        # print(f"Label: {result.names[int(box.cls[0])]}\nAverage depth: {average_depth}\n aligned_depth_frame: {aligned_depth_frame.get_distance(int(cX), int(cY))}\n")

                        # ? Angulos da relacao ao centro da camera com o centro da mascara
                        H_Angle = ((cX - CENTER_POINT_X) / CENTER_POINT_X) * (HFOV / 2)
                        V_Angle = ((cY - CENTER_POINT_Y) / CENTER_POINT_Y) * (VFOV / 2)

                        # ? Straight line distance from camera to object
                        distanceToFruit = (
                            (average_depth / cos(radians(H_Angle))) ** 2
                            + (average_depth * tan(radians(V_Angle))) ** 2
                        ) ** 0.5

                        # ? Straight line distance from object to claw
                        depthFromObjectToClaw = round(average_depth - OFFSET_FRONT, 3)
                        # new_Distance_to_Claw = (((average_depth - 3.5)/cos(radians(H_Angle)))**2 + (((average_depth-3.5)*tan(radians(V_Angle)))+5)**2)**0.5

                        # ? Relative Coordinates calculation
                        # * Y calculation (how far the arm has to move left or right)
                        # * if the object is to the left of the center of the camera, the value is negative

                        # * Z Calculation (how much the arm has to move up or down)
                        # * (after multiplying by -1) -> if the object is above the center of the camera, the value is positive
                        zCoordinate = (tan(radians(V_Angle)) * average_depth * -1) + (
                            OFFSET_HEIGHT / 2
                        )

                        if zCoordinate < -0.02:
                            zCoordinate += 0.025
                        elif zCoordinate < 0:
                            zCoordinate += 0.032
                        elif zCoordinate < 0.025:
                            zCoordinate += 0.035
                        elif zCoordinate < 0.05:
                            zCoordinate += 0.043
                        elif zCoordinate < 0.075:
                            zCoordinate += 0.05
                        elif zCoordinate < 0.1:
                            zCoordinate += 0.057
                        elif zCoordinate < 0.125:
                            zCoordinate += 0.064

                        zCoordinate = round(zCoordinate, 3)

                        # ? Calculus of the fruit width and height considering the depth to the object
                        fruit_width_pixels = int((cX + bWidth / 2 - (cX - bWidth / 2)))
                        fruit_height_pixels = int(
                            (cY + bHeight / 2 - (cY - bHeight / 2))
                        )

                        fruit_width = round(
                            ((fruit_width_pixels * distanceToFruit) / RESOLUTION_X), 3
                        )
                        fruit_height = round(
                            ((fruit_height_pixels * distanceToFruit) / RESOLUTION_Y), 3
                        )

                        focal_length = 641.5252685546875
                        cx = 643.2230224609375
                        cy = 353.0408935546875

                        yCoordinate = ((cX - cx) * average_depth) / focal_length

                        ## Correcao para y=0
                        yCoordinate = yCoordinate - 0.045

                        yCoordinate = (
                            0.000204135 * (yCoordinate**2)
                            + 0.712439 * yCoordinate
                            - 0.00610406
                        )

                        yCoordinate = round(yCoordinate, 3)

                        vertical_distance_meters = (
                            (cy - cY) * average_depth
                        ) / focal_length

                        vertical_distance_meters = round(
                            vertical_distance_meters + OFFSET_HEIGHT, 3
                        )

                        vertical_distance_meters = (
                            0.096866 * (vertical_distance_meters**2)
                            + 0.68 * vertical_distance_meters
                            - 0.00160242
                        )

                        vertical_distance_meters = round(vertical_distance_meters, 3)

                        # TODO -> CHECK IF THIS MAKES SENSE
                        claw_origin = (OFFSET_FRONT, 0, -OFFSET_HEIGHT)
                        fruit_location = (
                            depthFromObjectToClaw,
                            yCoordinate,
                            zCoordinate,
                        )

                        distanceClawFruit = (
                            (fruit_location[0] - claw_origin[0]) ** 2
                            + (fruit_location[1] - claw_origin[1]) ** 2
                            + (fruit_location[2] - claw_origin[2]) ** 2
                        ) ** 0.5
                        distanceClawFruit = round(distanceClawFruit, 6)

                        objectName = result.names[int(box.cls[0])]
                        if (
                            objectName == "Red-Tomato"
                            or objectName == "Early-Red-Tomato"
                        ) and depthFromObjectToClaw > 0.1:
                            # messageDigitalTwin.append({"X": depthFromObjectToClaw, "Y": yCoordinate, "Z": zCoordinate, "W": fruit_width, "H": fruit_height}) # , "Class": "Tomato"
                            # messageRobot.append({"X": depthFromObjectToClaw, "Y": yCoordinate, "Z": zCoordinate, "dClawToFruit": distanceClawFruit }) # , "Class": "Tomato"
                            tempDTList.append(
                                {
                                    "X": depthFromObjectToClaw,
                                    "Y": yCoordinate,
                                    "Z": zCoordinate,
                                    "W": fruit_width,
                                    "H": fruit_height,
                                }
                            )  # , "Class": "Tomato"
                            # if args.imu:
                            #     # print(
                            #     #     f"Roll: {imu.roll} Pitch: {imu.pitch} Yaw: {imu.yaw}"
                            #     # )
                            #     tempDTList.append(
                            #         {
                            #             "X": depthFromObjectToClaw,
                            #             "Y": yCoordinate,
                            #             "Z": zCoordinate,
                            #             "W": fruit_width,
                            #             "H": fruit_height,
                            #             "R": imu.roll,
                            #             "P": imu.pitch,
                            #             "Y": imu.yaw,
                            #         }
                            #     )  # , "Class": "Tomato"
                            # else:
                            #     tempDTList.append(
                            #         {
                            #             "X": depthFromObjectToClaw,
                            #             "Y": yCoordinate,
                            #             "Z": zCoordinate,
                            #             "W": fruit_width,
                            #             "H": fruit_height,
                            #         }
                            #     )  # , "Class": "Tomato"

                            tempRobotList.append(
                                {
                                    "x": depthFromObjectToClaw,
                                    "y": yCoordinate,
                                    "z": vertical_distance_meters,
                                    "w": fruit_width,
                                    "dClawToFruit": distanceClawFruit,
                                    "cX": cX,
                                    "cY": cY,
                                    "bWidth": bWidth,
                                    "bHeight": bHeight,
                                }
                            )  # , "Class": "Tomato"

                        # draw a circle on the center of the box in annotated frame in blue color
                        # cv2.circle(annotated_frame, (int(cX), int(cY)), 5, (255, 0, 0), -1)
            tempRobotList.sort(key=lambda x: x["dClawToFruit"])
            for k, obj in enumerate(tempRobotList):
                k += 1
                cX = obj["cX"]
                cY = obj["cY"]
                depthFromObjectToClaw = obj["x"]
                yCoordinate = obj["y"]
                vertical_distance_meters = obj["z"]
                fruit_width = obj["w"]
                bWidth = obj["bWidth"]
                bHeight = obj["bHeight"]

                x_tl = int(cX - (bWidth))
                y_tl = int(cY)

                font = cv2.FONT_HERSHEY_SIMPLEX

                cv2.putText(
                    annotated_frame,
                    f"{k:02d}",
                    (x_tl, y_tl),
                    font,
                    1,
                    (255, 255, 0),
                    2,
                )

                font_scale = 0.5
                font_thickness = 2
                font_color = (75, 0, 130)
                line_height = (
                    int(cv2.getTextSize("A", font, font_scale, font_thickness)[0][1])
                    + 2
                )
                coords = (int(cX - bWidth / 2), int(cY) + 20)
                text = f"X: {depthFromObjectToClaw}\nY: {yCoordinate}\nZ: {vertical_distance_meters}\nW: {fruit_width}"
                lines = text.split("\n")
                for i, line in enumerate(lines):
                    y = coords[1] + i * line_height
                    cv2.putText(
                        annotated_frame,
                        line,
                        (coords[0], y),
                        font,
                        font_scale,
                        font_color,
                        font_thickness,
                    )

            messageDigitalTwin[:] = tempDTList

            for robot in tempRobotList:
                del robot["cX"]
                del robot["cY"]
                del robot["bWidth"]
                del robot["bHeight"]
                del robot["dClawToFruit"]

            messageRobot[:] = tempRobotList

            if sendMessage and messageDigitalTwin:
                tomatoDetected.set()
                # cv2.imwrite("frame.png", annotated_frame)

            if args.rtmp:
                # ? RTMP
                rtmp_frames = annotated_frame[:, :, ::-1]
                p.stdin.write(rtmp_frames.tobytes())

            if args.wsFrame:
                # Convert the frame to JPEG format
                _, buffer = cv2.imencode(".jpg", annotated_frame)
                frame_data = base64.b64encode(buffer).decode("utf-8")

                # Send the frame data via WebSocket
                wsFrame.send(frame_data)
                print("Frame sent")

            cv2.imshow("YOLOv8 Inference", annotated_frame)
            # cv2.imshow("Depth", colorized_depth)

            time_end = time.time()
            total_time = time_end - time_start
            print("FPS: {:.2f}\n".format(1 / total_time))

            key = cv2.waitKey(1)
            if key == ord("q") or key == 27:
                break
        cv2.destroyAllWindows()
        print("Finished")

    finally:
        if args.rtmp:
            p.stdin.close()
            p.wait()
        pipeline.stop()
        if args.imu:
            imu.exit_event.set()
            imu.imu_pipe.stop()
