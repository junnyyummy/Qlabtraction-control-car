# -----------------------------------------------------------------------------#
# ------------------Skills Progression 1 - Task Automation---------------------#
# -----------------------------------------------------------------------------#
# ----------------------------Lab 3 - Line Following---------------------------#
# -----------------------------------------------------------------------------#

# --------------------------
# Imports
# --------------------------
from pal.products.qbot_platform import QBotPlatformDriver, Keyboard, \
    QBotPlatformCSICamera, QBotPlatformRealSense, QBotPlatformLidar
from qbot_platform_functions import QBPVision
from quanser.hardware import HILError
from pal.products.qbot_platform import QBotPlatformLidar
from pal.utilities.probe import Probe
from pal.utilities.gamepad import LogitechF710
import time
import cv2
from qlabs_setup import setup
import signal
import time
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# --------------------------
# Section A - Initial Setup
# --------------------------
# Initialize QBot platform position and orientation in simulation
setup(locationQBotP=[-1.35, 0.3, 0.05], rotationQBotP=[0, 0, 0], verbose=True)
time.sleep(2)  # Allow time for simulation to initialize

# Create a directory to store captured images
image_dir = "images-qlab"
os.makedirs(image_dir, exist_ok=True)  # Ensure the directory exists

# Create a CSV file to store categorized data
automatic_start = True
record_data = True
if record_data:
    csv_filename = "record-qlab.csv"
    with open(csv_filename, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time", "Num","Category", "Col", "Row", "Angle", "ForSpd", "TurnSpd", "ImageFile"])

# Network configuration
ipHost, ipDriver = 'localhost', 'localhost'

# Control variables initialization
commands, arm, noKill = np.zeros((2), dtype=np.float64), 0, True  # Command array, arm state, kill switch
frameRate, sampleRate = 60.0, 1 / 60.0  # Camera and control loop rates
counter, counterDown = 0, 0  # Frame counters
endFlag, offset, forSpd, turnSpd = False, 0, 0, 0  # Control flags and speeds

time_list = []
col_list = []
angle_list = []

# Timing control
startTime = time.time()  # Program start timestamp


def compute_accuracy(col, row, angle, target_number):
    """
    Calculate line following accuracy
    col: Detected line position (column coordinate)
    row: Detected line position (row coordinate)
    angle: Current heading angle of the robot
    target_number: Number of detected targets
    """
    # Set ideal center positions
    ideal_col = 160  # Assuming image center is at column 160
    ideal_angle = 0  # Ideal heading angle is 0 degrees

    # Calculate deviations
    col_error = abs(col - ideal_col) if target_number > 0 else 999  # 999 indicates lost track
    angle_error = abs(angle - ideal_angle)

    return col_error, angle_error


def update_plot():
    """
    Real-time plotting of robot's line following deviations
    """
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(time_list, col_list, label="Column Deviation (px)")
    plt.axhline(y=0, color="r", linestyle="--", label="Ideal Line")
    plt.xlabel("Time (s)")
    plt.ylabel("Column Deviation")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(time_list, angle_list, label="Angle Deviation (deg)", color="g")
    plt.axhline(y=0, color="r", linestyle="--", label="Ideal Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle Deviation")
    plt.legend()
    plt.grid()

    plt.pause(0.1)  # Brief pause to update plot


def signal_handler(sig, frame):
    """
    Handle Ctrl+C termination signal and save plot upon exit
    """
    global endFlag
    print("\nCtrl+C detected. Saving the plot and exiting...")

    # Save final plot
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time_list, col_list, label="Column Deviation (px)")
    plt.axhline(y=0, color="r", linestyle="--", label="Ideal Line")
    plt.xlabel("Time (s)")
    plt.ylabel("Column Deviation")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(time_list, angle_list, label="Angle Deviation (deg)", color="g")
    plt.axhline(y=0, color="r", linestyle="--", label="Ideal Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle Deviation")
    plt.legend()
    plt.grid()

    filename = f"line_following_accuracy.png"
    plt.savefig(filename)
    print(f"Plot saved as {filename}")

    endFlag = True


# Bind Ctrl+C signal handler
signal.signal(signal.SIGINT, signal_handler)


def elapsed_time():
    """Returns time elapsed since program start in seconds"""
    return time.time() - startTime


timeHIL, prevTimeHIL = elapsed_time(), elapsed_time() - 0.017  # Hardware timing variables

try:
    # --------------------------
    # Section B - Hardware Initialization
    # --------------------------
    myQBot = QBotPlatformDriver(mode=1, ip=ipDriver)  # QBot platform driver
    downCam = QBotPlatformCSICamera(frameRate=frameRate, exposure=39.0, gain=17.0)  # Downward camera
    keyboard = Keyboard()  # Keyboard input handler
    vision = QBPVision()  # Computer vision utilities
    lidar = QBotPlatformLidar()  # Initialize LiDAR sensor
    probe = Probe(ip=ipHost)  # Debugging probe for data visualization

    # Configure debug displays
    probe.add_display(imageSize=[200, 320, 1], scaling=True, scalingFactor=2, name='Raw Image')
    probe.add_display(imageSize=[200, 320, 1], scaling=False, scalingFactor=2, name='Binary Image')

    # Timing and state variables
    startTime = time.time()
    time.sleep(0.5)  # Initialization pause
    last_angle = 0.0  # Previous detected line angle
    cross_flag_C = 0  # Crossroad detection flag
    cross_flag_T = 0 
    cross_count = 0
    rorl = 10 

    # --------------------------
    # Main Control Loop
    # --------------------------
    while noKill and not endFlag:
        t = elapsed_time()

        # Maintain connection to debug probe
        if not probe.connected:
            probe.check_connection()

        newLidar = lidar.read()
        if newLidar:
            lidar_ranges = lidar.distances
            lidar_angles = lidar.angles

        if probe.connected:
            # --------------------------
            # Keyboard Input Handling
            # --------------------------
            newkeyboard = keyboard.read()
            if newkeyboard:
                arm = keyboard.k_space  # Arm/disarm control
                lineFollow = keyboard.k_7  # Toggle line following mode
                keyboardComand = keyboard.bodyCmd  # Manual control commands
                if keyboard.k_u:
                    noKill = False  # Emergency stop

            if automatic_start:
                lineFollow = 1
                arm = 1

            # --------------------------
            # Section C - Control Mode Selection
            # --------------------------
            if not lineFollow:
                # Manual control mode
                commands = np.array([keyboardComand[0], keyboardComand[1]], dtype=np.float64)
            else:
                # Autonomous line following mode
                commands = np.array([forSpd, turnSpd], dtype=np.float64)

            # --------------------------
            # Hardware Communication
            # --------------------------
            newHIL = myQBot.read_write_std(timestamp=time.time() - startTime,
                                           arm=arm,
                                           commands=commands)
            if newHIL:
                timeHIL = time.time()
                newDownCam = downCam.read()

                if newDownCam:
                    counterDown += 1

                    # --------------------------
                    # Section D - Image Processing Pipeline
                    # --------------------------
                    # D.1 - Image Preprocessing
                    undistorted = vision.df_camera_undistort(downCam.imageData)  # Correct lens distortion
                    gray_sm = cv2.resize(undistorted, (320, 200))  # Resize for processing

                    # D.2 - Thresholding and ROI Selection
                    binary = vision.subselect_and_threshold(gray_sm, 0, 200, 175, 255)

                    # D.3 - Blob Detection and Analysis
                    col, row, area, angle, target_number, debug_image = vision.image_find_objects(
                        binary, 8, 1000, 12000
                    )
                    print(
                        f"Detected blob: col={col}, row={row}, area={area}, angle={angle}, target_number={target_number}")

                    if record_data:
                        # ------------------------
                        # Classify the robot's current state
                        # ------------------------
                        if target_number == 0:
                            category = "Off Track"  # Robot has lost track
                        elif target_number == 1:
                            category = "On Track"  # Correctly following
                        elif target_number in [2, 3] and row > 150:
                            category = "Corner - No Forward Path"  # At corner with blocked path
                        elif target_number in [2, 3]:
                            category = "Corner"  # Normal corner
                        elif target_number == 4:
                            category = "Crossroad"  # Crossroad
                        else:
                            category = "Unknown"  # Other scenarios

                        # ------------------------
                        # Save processed images
                        # ------------------------
                        image_filename = f"{image_dir}/{t:.2f}_{category}.png"
                        cv2.imwrite(image_filename, gray_sm)
                        image_filename = f"{image_dir}/{t:.2f}_{category}_ans.png"
                        cv2.imwrite(image_filename, debug_image)

                        # ------------------------
                        # Store categorized data
                        # ------------------------
                        with open(csv_filename, "a", newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([t, target_number,category, col, row, angle, forSpd, turnSpd, image_filename])

                    col_error, angle_error = compute_accuracy(col, row, angle,
                                                              target_number)

                    # Record data
                    time_list.append(counterDown)
                    col_list.append(col_error)
                    angle_list.append(angle_error)

                    # --------------------------
                    # Section E - Control Logic
                    # --------------------------
                    angle_speed = angle - last_angle  # Calculate angular velocity

                    # Decision tree for different scenarios
                    if cross_count > 0:  # Handling crossroads
                        cross_count -= 1
                        if rorl>0:
                            forSpd, turnSpd = 0, 0.5  # Sharp turn
                        else:
                            forSpd, turnSpd = 0, -0.5
                    else:

                        if target_number == 0:  # No line detected
                                forSpd, turnSpd = -0.25, 0.0  # Reverse slowly
                        elif target_number == 3 and row > 150 and cross_flag_T ==0:  # Crossroad detection
                            forSpd, turnSpd = -0.5, 0.0  # Prepare for turn
                            cross_count = 100  # Set crossroad handling duration
                            cross_flag_T =1
                        elif target_number == 4 and cross_flag_C==0:
                            forSpd, turnSpd = 0, 0
                            cross_count = 100
                            cross_flag_C=1
                        else:  # Normal line following
                            # PID-like control using line position and angle
                            turnSpd = angle * 0.03 + angle_speed * 0.4 - (col - 160) / 160 * 1.0
                            forSpd = 0.35  # Base forward speed
                            cross_flag_C =0
                            cross_flag_T =0

                    last_angle = angle  # Store current angle

                    # --------------------------
                    # Debug Visualization
                    # --------------------------
                    if counterDown % 4 == 0:  # Throttle display updates
                        probe.send(name='Raw Image', imageData=gray_sm)
                        probe.send(name='Binary Image', imageData=binary)
                        cv2.imshow("Debug Image", debug_image)
                        cv2.waitKey(1)  # Refresh display window

                prevTimeHIL = timeHIL

except KeyboardInterrupt:
    print('User interrupted.')
    signal_handler(signal.SIGINT, [])
except HILError as h:
    print(h.get_error_message())
finally:
    # --------------------------
    # Cleanup Operations
    # --------------------------
    downCam.terminate()
    myQBot.terminate()
    probe.terminate()
    keyboard.terminate()
    cv2.destroyAllWindows()