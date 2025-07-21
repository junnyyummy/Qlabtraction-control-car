# category_and_position_model.py - 分类模型和位置回归模型合并训练

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import cv2
import math

CSV_FILE = "record_qlab_new.csv"
IMG_SIZE = (320, 200)
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.0003
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {DEVICE}")

CATEGORY_MAP = {
    'On Track': 0,
    'Off Track': 1,
    'Corner': 2,
    'Corner - No Forward Path': 3,
    'Crossroad': 4
}

# ------------------------
# 分类模型数据加载
# ------------------------
def load_classification_data(csv_path, transform):
    print("📥 Loading classification dataset...")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["ImageFile", "Category"])
    images, labels = [], []
    for _, row in df.iterrows():
        img = cv2.imread(row["ImageFile"], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, IMG_SIZE)
        if transform:
            img = transform(img)
        category_index = CATEGORY_MAP.get(str(row["Category"]).strip(), 0)
        label = torch.tensor(category_index, dtype=torch.long)
        images.append(img)
        labels.append(label)
    print(f"✅ Loaded {len(images)} classification samples.")
    return torch.stack(images).to(DEVICE), torch.tensor(labels).to(DEVICE)

# ------------------------
# 回归模型数据加载（Col, Row, Angle）
# ------------------------
def load_regression_data(csv_path, transform):
    print("📥 Loading regression dataset...")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["ImageFile", "Col", "Row", "Angle"])
    images, labels = [], []
    for _, row in df.iterrows():
        img = cv2.imread(row["ImageFile"], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, IMG_SIZE)
        if transform:
            img = transform(img)
        col_norm = (float(row["Col"]) - 160.0) / 160.0
        row_norm = (float(row["Row"]) - 100.0) / 100.0
        angle_norm = float(row["Angle"]) / math.pi
        label = torch.tensor([col_norm, row_norm, angle_norm], dtype=torch.float32)
        images.append(img)
        labels.append(label)
    print(f"✅ Loaded {len(images)} regression samples.")
    return torch.stack(images).to(DEVICE), torch.stack(labels).to(DEVICE)

# ------------------------
# 卷积分类模型结构
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

# ------------------------
# 卷积回归模型结构（Col, Row, Angle）
# ------------------------
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
# 主训练流程
# ------------------------
if __name__ == "__main__":
    print("🚀 Starting training...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 分类任务
    x_class, y_class = load_classification_data(CSV_FILE, transform)
    split_class = int(len(x_class) * 0.7)
    train_xc, test_xc = x_class[:split_class], x_class[split_class:]
    train_yc, test_yc = y_class[:split_class], y_class[split_class:]

    cat_model = CategoryClassifier().to(DEVICE)
    optimizer_c = optim.Adam(cat_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        cat_model.train()
        total_loss = 0
        for i in range(0, len(train_xc), BATCH_SIZE):
            xb = train_xc[i:i+BATCH_SIZE]
            yb = train_yc[i:i+BATCH_SIZE]
            optimizer_c.zero_grad()
            pred = cat_model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer_c.step()
            total_loss += loss.item()
        print(f"[Category] Epoch {epoch+1}: Loss = {total_loss:.4f}")

    torch.save(cat_model.state_dict(), "Category.pth")

    # 回归任务
    x_reg, y_reg = load_regression_data(CSV_FILE, transform)
    split_reg = int(len(x_reg) * 0.7)
    train_xr, test_xr = x_reg[:split_reg], x_reg[split_reg:]
    train_yr, test_yr = y_reg[:split_reg], y_reg[split_reg:]

    pos_model = PositionRegressor().to(DEVICE)
    optimizer_r = optim.Adam(pos_model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_r, mode="min", factor=0.5, patience=5)

    for epoch in range(EPOCHS):
        pos_model.train()
        total_loss = 0
        for i in range(0, len(train_xr), BATCH_SIZE):
            xb = train_xr[i:i+BATCH_SIZE]
            yb = train_yr[i:i+BATCH_SIZE]
            optimizer_r.zero_grad()
            pred = pos_model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            loss.backward()
            optimizer_r.step()
            total_loss += loss.item()
        scheduler.step(total_loss)
        print(f"[Position] Epoch {epoch+1}: Loss = {total_loss:.4f}")

    torch.save(pos_model.state_dict(), "Position.pth")