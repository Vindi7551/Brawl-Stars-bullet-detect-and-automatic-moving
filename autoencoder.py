import cv2
import numpy as np
import pygetwindow as gw
import pyautogui
from ultralytics import YOLO
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim

# 加载YOLO模型
def load_model(model_path):
    model = YOLO(model_path)
    return model

# 数据预处理
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转为灰度图像
    frame = cv2.resize(frame, (128, 128))  # 调整大小
    frame = frame.astype('float32') / 255.0  # 归一化
    frame = np.expand_dims(frame, axis=0)
    frame = np.expand_dims(frame, axis=0)
    return frame

import cv2
import numpy as np
import pygetwindow as gw
import pyautogui
from ultralytics import YOLO
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim

# 加载YOLO模型
def load_model(model_path):
    model = YOLO(model_path)
    return model

# 数据预处理
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转为灰度图像
    frame = cv2.resize(frame, (128, 128))  # 调整大小
    frame = frame.astype('float32') / 255.0  # 归一化
    frame = np.expand_dims(frame, axis=0)
    frame = np.expand_dims(frame, axis=0)
    return frame

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # (16, 64, 64)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # (32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)  # (64, 26, 26)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),  # (32, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # (16, 64, 64)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # (1, 128, 128)
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 初始化自动编码器
autoencoder = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

def train_autoencoder(frame, autoencoder, optimizer, criterion):
    frame_tensor = torch.tensor(frame, dtype=torch.float32)
    optimizer.zero_grad()
    output = autoencoder(frame_tensor)
    loss = criterion(output, frame_tensor)
    loss.backward()
    optimizer.step()
    return loss.item()

def make_decision(autoencoder_output, charactor_coords, bullet_enemy_coords, hit_effect_coords):
    # 你的决策逻辑
    pass

# 发送移动命令
def send_move_command(start_x, start_y, end_x, end_y):
    subprocess.Popen(["adb", "shell", "input", "swipe", str(start_x), str(start_y), str(end_x), str(end_y), "500"])

# 处理实时视频输入
model_path = 'best.pt'
model = load_model(model_path)

scrcpy_process = subprocess.Popen(["D:/scrcpy-win32-v2.4/scrcpy-win32-v2.4/scrcpy.exe"], stdout=subprocess.PIPE)

while True:
    scrcpy_windows = gw.getWindowsWithTitle('23113RKC6C')
    if scrcpy_windows:
        scrcpy_window = scrcpy_windows[0]
        screenshot = pyautogui.screenshot(region=(
            scrcpy_window.left,
            scrcpy_window.top,
            scrcpy_window.width,
            scrcpy_window.height
        ))
        frame = np.array(screenshot)
        preprocessed_frame = preprocess_frame(frame)
        charactor_coords, bullet_enemy_coords, hit_effect_coords = detect_objects(frame, model)

        loss = train_autoencoder(preprocessed_frame, autoencoder, optimizer, criterion)
        print(f"Training loss: {loss}")

        autoencoder_output = autoencoder(torch.tensor(preprocessed_frame, dtype=torch.float32)).detach().numpy()
        make_decision(autoencoder_output, charactor_coords, bullet_enemy_coords, hit_effect_coords)

        # 显示图像
        cv2.imshow('Scrcpy Window Capture', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    else:
        print("未找到 scrcpy 窗口")

cv2.destroyAllWindows()
scrcpy_process.terminate()
