import numpy as np
import cv2

from pal.products.qbot_platform import QBotPlatformDriver
from pal.utilities.math import Calculus
from scipy.ndimage import median_filter
from pal.utilities.math import Calculus
from pal.utilities.stream import BasicStream
from quanser.common import Timeout


class QBPMovement():
    """ This class contains the functions for the QBot Platform such as
    Forward/Inverse Differential Drive Kinematics etc. """

    def __init__(self):
        self.WHEEL_RADIUS = QBotPlatformDriver.WHEEL_RADIUS  # radius of the wheel (meters)
        self.WHEEL_BASE = QBotPlatformDriver.WHEEL_BASE  # distance between wheel contact points on the ground (meters)
        self.WHEEL_WIDTH = QBotPlatformDriver.WHEEL_WIDTH  # thickness of the wheel (meters)
        self.ENCODER_COUNTS = QBotPlatformDriver.ENCODER_COUNTS  # encoder counts per channel
        self.ENCODER_MODE = QBotPlatformDriver.ENCODER_MODE  # multiplier for a quadrature encoder

    def diff_drive_inverse_velocity_kinematics(self, forSpd, turnSpd):
        """This function is for the differential drive inverse velocity
        kinematics for the QBot Platform. It converts provided body speeds
        (forward speed in m/s and turn speed in rad/s) into corresponding
        wheel speeds (rad/s)."""

        # ------------Replace the following lines with your code---------------#
        wL = 0
        wR = 0
        # ---------------------------------------------------------------------#
        return wL, wR

    def diff_drive_forward_velocity_kinematics(self, wL, wR):
        """This function is for the differential drive forward velocity
        kinematics for the QBot Platform. It converts provided wheel speeds
        (rad/s) into corresponding body speeds (forward speed in m/s and
        turn speed in rad/s)."""
        # ------------Replace the following lines with your code---------------#
        forSpd = 0
        turnSpd = 0
        # ---------------------------------------------------------------------#
        return forSpd, turnSpd


class QBPVision():
    def __init__(self):
        self.imageCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def undistort_img(self, distImgs, cameraMatrix, distCoefficients):
        """
        This function undistorts a general camera, given the camera matrix and
        coefficients.
        """

        undist = cv2.undistort(distImgs,
                               cameraMatrix,
                               distCoefficients,
                               None,
                               cameraMatrix)
        return undist

    def df_camera_undistort(self, image):
        """
        This function undistorts the downward camera using the camera
        intrinsics and coefficients."""
        CSICamIntrinsics = np.array([[419.36179672, 0, 292.01381114],
                                     [0, 420.30767196, 201.61650657],
                                     [0, 0, 1]])
        CSIDistParam = np.array([-7.42983302e-01,
                                 9.24162996e-01,
                                 -2.39593372e-04,
                                 1.66230745e-02,
                                 -5.27787439e-01])
        undistortedImage = self.undistort_img(
            image,
            CSICamIntrinsics,
            CSIDistParam
        )
        return undistortedImage

    def subselect_and_threshold(self, image, row_start, row_end, min_threshold, max_threshold):
        """
        Extracts a region of interest (ROI) from the input image, applies preprocessing steps,
        and performs binary thresholding.

        Parameters:
        image (numpy.ndarray): Input image (grayscale or color)
        row_start (int): Start row index of ROI
        row_end (int): End row index of ROI
        min_threshold (int): Minimum threshold value
        max_threshold (int): Maximum threshold value

        Returns:
        numpy.ndarray: Binary thresholded image
        """
        # Select region of interest (ROI)
        gray_roi = image[row_start:row_end, :]

        # Convert to grayscale if the image has multiple channels
        if len(gray_roi.shape) == 3:  # Check if the image is color (RGB/BGR)
            gray_roi = cv2.cvtColor(gray_roi, cv2.COLOR_BGR2GRAY)

        # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray_roi)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray_eq, (5, 5), 0)

        # Define a 3x3 kernel for morphological operations
        kernel = np.ones((3, 3), np.uint8)

        # Apply erosion to reduce small noise and refine edges
        binary_eroded = cv2.erode(blurred, kernel, iterations=1)

        # Apply binary thresholding
        _, binary = cv2.threshold(binary_eroded, min_threshold, max_threshold, cv2.THRESH_BINARY)

        return binary

    def image_find_objects(self, image, connectivity=8, minArea=10, maxArea=10000):
        """
        Detect connected components in binary image and select optimal line segment.
        Selection criteria:
        1. Filter components by area range
        2. Select two components closest to image center
        3. Choose the upper-most component from candidates
        4. Return parameters of selected component and visualization

        Parameters:
            image (numpy.ndarray): Input binary image
            connectivity (int): 4 or 8 for pixel connectivity
            minArea (int): Minimum component area threshold
            maxArea (int): Maximum component area threshold

        Returns:
            tuple: (col, row, area, angle, num_detected, debug_image)
                - col, row: Centroid coordinates of selected component
                - area: Area of selected component
                - angle: Orientation angle in degrees (-90 adjustment for visualization)
                - num_detected: Number of valid components detected
                - debug_image: Visualization of detection results
        """

        # Convert to BGR color space for annotation
        output_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)

        # Perform connected component analysis
        numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            image, connectivity, cv2.CV_32S
        )

        all_lines = []
        # Process each component (skip background at index 0)
        for i in range(1, numLabels):
            area = stats[i, cv2.CC_STAT_AREA]

            # Area filtering
            if not (minArea < area < maxArea):
                continue

            # Get centroid coordinates
            col, row = int(centroids[i][0]), int(centroids[i][1])

            # Calculate orientation using image moments
            component_mask = (labels == i).astype(np.uint8)
            moments = cv2.moments(component_mask)

            # Handle division stability
            denominator = (moments["mu20"] - moments["mu02"])
            if denominator == 0:
                angle = 0.0
            else:
                angle = -0.5 * np.arctan2(2 * moments["mu11"], denominator)

            # Normalize angle to [0, pi) range
            if angle < 0:
                angle += np.pi
            angle = np.unwrap([angle])[0]  # Handle angular continuity

            all_lines.append((col, row, area, angle))

        num_detected = len(all_lines)
        best_line = (0, 0, 0, 0)  # Default empty values

        # Selection logic
        if num_detected == 1:
            best_line = all_lines[0]
        elif num_detected >= 2:
            # Primary sort: angular proximity to vertical (π/2)
            all_lines.sort(key=lambda x: abs(x[3] - np.pi / 2))

            # Get top 2 candidates
            candidates = all_lines[:2]

            # Secondary selection criteria
            angular_threshold = np.pi / 4
            if abs(candidates[1][3] - np.pi / 2) > angular_threshold:
                best_line = candidates[0]  # Fallback to first candidate
            else:
                # Choose upper-most component (minimum row coordinate)
                best_line = min(candidates, key=lambda x: x[1])

        # Visualization parameters
        ARROW_LENGTH = 60
        COLOR_DETECTED = (0, 0, 255)  # Red: detected components
        COLOR_SELECTED = (0, 255, 0)  # Green: selected component
        COLOR_ARROW = (255, 0, 0)  # Blue: direction arrows
        COLOR_FINAL_ARROW = (128, 128, 128)  # Gray: final direction

        # Draw all detected components
        for col, row, area, angle in all_lines:
            # Calculate arrow endpoints
            dx = int(ARROW_LENGTH * np.cos(angle))
            dy = int(ARROW_LENGTH * np.sin(angle))

            # Draw component center
            cv2.circle(output_image, (col, row), 5, COLOR_DETECTED, -1)
            # Draw direction arrow
            cv2.arrowedLine(output_image, (col, row),
                            (col + dx, row - dy), COLOR_ARROW, 2)

        # Draw selected component
        final_col, final_row, final_area, final_angle = best_line
        if final_area > 0:
            dx = int(ARROW_LENGTH * np.cos(final_angle))
            dy = int(ARROW_LENGTH * np.sin(final_angle))

            # Highlight selected component
            cv2.circle(output_image, (final_col, final_row), 8, COLOR_SELECTED, -1)
            cv2.arrowedLine(output_image, (final_col, final_row),
                            (final_col + dx, final_row - dy),
                            COLOR_FINAL_ARROW, 3)

        # Angle adjustment for visualization convention
        adjusted_angle = np.degrees(final_angle) - 90

        return (final_col, final_row, final_area,
                adjusted_angle, num_detected, output_image)

    def line_to_speed_map(self, sampleRate, saturation):
        print('Stupid pid structure, please do not use it.')
        # integrator = Calculus().integrator(dt=sampleRate, saturation=saturation)
        # derivative = Calculus().differentiator(dt=sampleRate)
        # next(integrator)
        # next(derivative)
        # forSpd, turnSpd = 0, 0
        # offset = 0
        #
        # while True:
        # col, kP, kD = yield forSpd, turnSpd
        #
        # if col is not None:
        #     #-----------Complete the following lines of code--------------#
        #     error = col - 160
        #     angle = -np.arctan2(error, 320)
        #     turnSpd = kP * angle + kD * derivative.send(angle)
        #     forSpd = 0.3 - max(min((kP * angle + kD * derivative.send(angle)) * 3, 0.3), 0.0)
        #     #-------------------------------------------------------------#
        #     offset = integrator.send(25 * turnSpd)


class QBPRanging():
    def __init__(self):
        pass

    def adjust_and_subsample(self, ranges, angles, end=-1, step=4):

        # correct angles data
        angles_corrected = -1 * angles + np.pi / 2
        # return every 4th sample
        return ranges[0:end:step], angles_corrected[0:end:step]

    def correct_lidar(self, lidarPosition, ranges, angles):

        # Convert lidar data from polar into cartesian, and add lidar position
        # Then Convert back into polar coordinates

        # -------Replace the following line with your code---------#
        # Determine the start of the focus region 
        ranges_c = None
        angles_c = None
        # ---------------------------------------------------------#

        return ranges_c, angles_c

    def detect_obstacle(self, ranges, angles, forSpd, forSpeedGain, turnSpd, turnSpeedGain, minThreshold,
                        obstacleNumPoints):

        halfNumPoints = 205
        quarterNumPoints = round(halfNumPoints / 2)

        # Grab the first half of ranges and angles representing 180 degrees
        frontRanges = ranges[0:halfNumPoints]
        frontAngles = angles[0:halfNumPoints]

        # Starting index in top half          1     West
        # Mid point in west quadrant         51     North-west
        # Center index in top half          102     North
        # Mid point in east quadrant     51+102     North-east
        # Ending index in top half          205     East

        ### Section 1 - Dynamic Focus Region ###

        # -------Replace the following line with your code---------#
        # Determine the start of the focus region 
        startingIndex = 0
        # ---------------------------------------------------------#

        # Setting the upper and lower bound such that the starting index 
        # is always in the first quarant
        if startingIndex < 0:
            startingIndex = 0
        elif startingIndex > 102:
            startingIndex = 102

        # Pick quarterNumPoints in ranges and angles from the front half
        # this will be the region you monitor for obstacles
        monitorRanges = frontRanges[startingIndex:startingIndex + quarterNumPoints]
        monitorAngles = frontAngles[startingIndex:startingIndex + quarterNumPoints]

        ### Section 2 - Dynamic Stop Distance ###

        # -------Replace the following line with your code---------#
        # Determine safetyThreshold based on Forward Speed 
        safetyThreshold = 1

        # ---------------------------------------------------------#

        # At angles corresponding to monitorAngles, pick uniform ranges based on
        # a safety threshold
        safetyAngles = monitorAngles
        safetyRanges = safetyThreshold * monitorRanges / monitorRanges

        ### Section 3 - Obstacle Detection ###

        # -------Replace the following line with your code---------#
        # Total number of obstacles detected between 
        # minThreshold & safetyThreshold
        # Then determine obstacleFlag based on obstacleNumPoints

        obstacleFlag = 0

        # ---------------------------------------------------------#

        # Lidar Ranges and Angles for plotting (both scan & safety zone)
        plottingRanges = np.append(monitorRanges, safetyRanges)
        plottingAngles = np.append(monitorAngles, safetyAngles)

        return plottingRanges, plottingAngles, obstacleFlag
