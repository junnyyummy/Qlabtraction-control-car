# cnn_deploy_dual_model.py - 使用两个模型预测 Category 和 Position，并记录结果至 CSV 和图片

import torch
import torch.nn as nn
import numpy as np
import time
import cv2
import os
import signal
import math
import csv
from torchvision import transforms
from qlabs_setup import setup
from pal.products.qbot_platform import QBotPlatformDriver, QBotPlatformCSICamera, QBotPlatformLidar
from qbot_platform_functions import QBPVision
from pal.utilities.probe import Probe
from pal.products.qbot_platform import Keyboard
from quanser.hardware import HILError

# ------------------------
# 保存路径设置
# ------------------------
RESULT_DIR = "result_images"
CSV_FILE = "result.csv"
os.makedirs(RESULT_DIR, exist_ok=True)
with open(CSV_FILE, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Time", "Category", "Col", "Row", "Angle", "ForSpd", "TurnSpd", "ImageFile"])

# ------------------------
# 模型定义
# ------------------------
class CategoryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        dummy_input = torch.zeros(1, 1, 200, 320)
        out = self.features(dummy_input)
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(out.view(1, -1).shape[1], 256), nn.ReLU(),
            nn.Linear(256, 5)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return self.classifier(x)

class PositionRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        dummy_input = torch.zeros(1, 1, 200, 320)
        out = self.features(dummy_input)
        self.flatten = nn.Flatten()
        self.regressor = nn.Sequential(
            nn.Linear(out.view(1, -1).shape[1], 256), nn.ReLU(),
            nn.Linear(256, 3)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return self.regressor(x)

# ------------------------
# 初始化
# ------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cat_model = CategoryClassifier().to(DEVICE)
pos_model = PositionRegressor().to(DEVICE)
cat_model.load_state_dict(torch.load("Category.pth", map_location=DEVICE))
pos_model.load_state_dict(torch.load("Position.pth", map_location=DEVICE))
cat_model.eval()
pos_model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

setup(locationQBotP=[-1.35, 0.3, 0.05], rotationQBotP=[0, 0, 0], verbose=True)
time.sleep(2)

ipHost, ipDriver = 'localhost', 'localhost'
endFlag = False
startTime = time.time()
last_angle = 0.0
cross_flag = 0
rorl = 1
cross_flag_C = 0
cross_flag_T = 0
cross_count = 0

signal.signal(signal.SIGINT, lambda sig, frame: exit())

CATEGORY_MAP_REV = {
    0: "On Track",
    1: "Off Track",
    2: "Corner",
    3: "Corner - No Forward Path",
    4: "Crossroad"
}

try:
    myQBot = QBotPlatformDriver(mode=1, ip=ipDriver)
    downCam = QBotPlatformCSICamera(frameRate=60.0, exposure=39.0, gain=17.0)
    lidar = QBotPlatformLidar()
    vision = QBPVision()
    probe = Probe(ip=ipHost)
    keyboard = Keyboard()

    probe.add_display(imageSize=[200, 320, 1], scaling=True, scalingFactor=2, name='Raw Image')

    frame_count = 0
    while not endFlag:
        newLidar = lidar.read()
        if newLidar:
            lidar_ranges = lidar.distances
            lidar_angles = lidar.angles

        newDownCam = downCam.read()
        if not newDownCam:
            continue

        img = vision.df_camera_undistort(downCam.imageData)
        gray_sm = cv2.resize(img, (320, 200))
        input_tensor = transform(gray_sm).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            category_logits = cat_model(input_tensor)
            category_index = int(torch.argmax(category_logits, dim=1).item())
            category_label = CATEGORY_MAP_REV.get(category_index, "Unknown")

            pos_output = pos_model(input_tensor)[0]
            col = pos_output[0].item() * 160 + 160
            row = pos_output[1].item() * 100 + 100
            angle = pos_output[2].item() * math.pi

        angle_speed = angle - last_angle

        if cross_count > 0:
            cross_count -= 1
            forSpd, turnSpd = (0, 0.5) if rorl > 0 else (0, -0.5)
        else:
            if category_label == "Corner - No Forward Path" and cross_flag_C == 0:
                cross_count = 100
                cross_flag_C = 1
            elif category_label == "On Track":
                forSpd = 0.35
                turnSpd = angle * 0.03 + angle_speed * 0.4 - (col - 160) / 160 * 1.0
                cross_flag_C = 0
            elif category_label == "Off Track":
                forSpd, turnSpd = -0.25, 0.0
            else:
                turnSpd = angle * 0.03 + angle_speed * 0.4 - (col - 160) / 160 * 1.0
                forSpd = 0.35

        timestamp = round(time.time() - startTime, 2)
        img_filename = f"frame_{frame_count:05d}.png"
        img_path = os.path.join(RESULT_DIR, img_filename)
        cv2.imwrite(img_path, gray_sm)

        with open(CSV_FILE, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, category_label, round(col, 2), round(row, 2), round(angle, 4), round(forSpd, 3), round(turnSpd, 3), img_path])

        last_angle = angle
        frame_count += 1

        myQBot.read_write_std(timestamp=timestamp, arm=1, commands=np.array([forSpd, turnSpd]))
        probe.send(name='Raw Image', imageData=gray_sm)
        cv2.imshow("Live", gray_sm)
        cv2.waitKey(1)

except KeyboardInterrupt:
    print("KeyboardInterrupt")
finally:
    downCam.terminate()
    myQBot.terminate()
    lidar.terminate()
    probe.terminate()
    keyboard.terminate()
    cv2.destroyAllWindows()
