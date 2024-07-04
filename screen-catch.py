import cv2
import numpy as np
import pygetwindow as gw
import pyautogui
from ultralytics import YOLO
import subprocess
import math
from torchvision.ops import nms
import torch

# 加载模型
def load_model(model_path):
    model = YOLO(model_path)
    return model

# 计算角度和速度
def calculate_angle_and_speed(prev_coords, curr_coords):
    if prev_coords is None or len(prev_coords) == 0 or curr_coords is None or len(curr_coords) == 0:
        return None 

# 发送移动命令
def send_move_command(angle, speed,bullet_enemy_coords,prev_bullet_enemy_coords, charactor_coords):
    if bullet_enemy_coords is None or len(prev_bullet_enemy_coords) == 0 or charactor_coords is None :
        return None 
    # 根据角度和速度计算移动命令
    distance = speed * 10  # 调整比例以适应实际速度
    if angle is not None and speed is not None:
        if prev_bullet_enemy_coords[0][0]+(prev_bullet_enemy_coords[0][1]-charactor_coords[0][1])(prev_bullet_enemy_coords[0][0]-bullet_enemy_coords[0][0])/(prev_bullet_enemy_coords[0][1]-bullet_enemy_coords[0][1])<charactor_coords[0][0]:     
       # 计算滑动终点
            start_x = 320  # 假设屏幕中心为起点，适应1080x2400分辨率
            start_y = 800
            end_x = start_x + int(distance * math.cos(math.radians(angle)))
            end_y = start_y + int(distance * math.sin(math.radians(angle)))
        else:
            angle+=180
            start_x = 320  # 假设屏幕中心为起点，适应1080x2400分辨率
            start_y = 800
            end_x = start_x + int(distance * math.cos(math.radians(angle)))
            end_y = start_y + int(distance * math.sin(math.radians(angle)))
    
        # 打印移动方向和距离
        print(f"Moving from ({start_x}, {start_y}) to ({end_x}, {end_y}) with speed {speed:.2f}")

        # 使用 adb 命令模拟滑动
        subprocess.Popen(["adb", "shell", "input", "swipe", str(start_x), str(start_y), str(end_x), str(end_y), "500"])

# 处理单帧图像
def process_frame(frame, model):
    # 运行模型进行推理
    results = model(frame)
    # 初始化坐标
    charactor_coords = []
    bullet_enemy_coords = []
    # 初始化最高置信度
    max_conf_charactor = 0
    max_conf_bullet_enemy = 0
   
    # 检查是否有检测结果并且至少有一个边界框
    if len(results) > 0 and len(results[0].boxes) > 0:
        result = results[0] 
        cursor = result.boxes
        xyxy_list = cursor.xyxy.cpu().numpy()  # 获取所有边界框的坐标
        conf_list = cursor.conf.cpu().numpy()  # 获取所有置信度
        cls_id_list = cursor.cls.cpu().numpy()  # 获取所有类别索引       

           # 应用置信度阈值过滤
        keep = conf_list > 0.8
        xyxy_list = xyxy_list[keep]
        conf_list = conf_list[keep]
        cls_id_list = cls_id_list[keep]
        
        for xyxy, conf, cls_id in zip(xyxy_list, conf_list, cls_id_list):
            if conf > 0.8:
                x1, y1, x2, y2 = xyxy[:4]
                if x1 > 500 and x2 < 1900:
                    # 绘制边界框
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    # 获取类别名称
                    class_names = ["bullet", "bullet-enemy", "charactor"]  # 类别名称
                    class_name = class_names[int(cls_id)]  # 获取类别名称
                    # 在图像上添加置信度和类别名称
                    label = f'{class_name} {conf:.2f}'
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    # 如果类别是 "charactor"，记录最高置信度的坐标
                    if class_name == "charactor" and conf > max_conf_charactor:
                        max_conf_charactor = conf
                        charactor_coords.append(((int(x1) + int(x2)) / 2, (int(y1) + int(y2)) / 2))
                        
                    # 如果类别是 "bullet-enemy"，记录最高置信度的坐标
                    if class_name == "bullet-enemy" and conf > max_conf_bullet_enemy:
                        max_conf_bullet_enemy = conf
                        bullet_enemy_coords.append(((int(x1) + int(x2)) / 2, (int(y1) + int(y2)) / 2))

    return frame, charactor_coords, bullet_enemy_coords

# 启动 scrcpy 进程
scrcpy_process = subprocess.Popen(["D:/scrcpy-win32-v2.4/scrcpy-win32-v2.4/scrcpy.exe"], stdout=subprocess.PIPE)

# 加载模型
model_path = 'best.pt'
model = load_model(model_path)

# 定义屏幕分辨率
width, height = 1080, 2400
prev_bullet_enemy_coords = []
prev_angle = None

while True:
    # 获取所有标题为 'scrcpy' 的窗口
    scrcpy_windows = gw.getWindowsWithTitle('23113RKC6C')
    if scrcpy_windows:
        scrcpy_window = scrcpy_windows[0]  # 假设 scrcpy 只有一个窗口

        # 捕获窗口的屏幕截图
        screenshot = pyautogui.screenshot(region=(
            scrcpy_window.left,  # 窗口左上角的横坐标
            scrcpy_window.top,   # 窗口左上角的纵坐标
            scrcpy_window.width, # 窗口的宽度
            scrcpy_window.height # 窗口的高度
        ))
        # 将截图转换为 OpenCV 图像
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        max_conf_charactor = 0
        max_conf_bullet_enemy = 0
      
        # 处理图像并获取检测结果
        frame, charactor_coords, bullet_enemy_coords = process_frame(frame, model)

        if bullet_enemy_coords and prev_bullet_enemy_coords:
            angle = calculate_angle_and_speed(prev_bullet_enemy_coords, bullet_enemy_coords)
            if angle is not None and angle != prev_angle:
                send_move_command(angle, 10,bullet_enemy_coords, prev_bullet_enemy_coords, charactor_coords)
                prev_angle = angle
        
        # 更新上一帧的 `bullet-enemy` 坐标
        if bullet_enemy_coords:
            prev_bullet_enemy_coords = bullet_enemy_coords

        # 显示图像
        cv2.imshow('Scrcpy Window Capture', frame)
        key = cv2.waitKey(1)
        if key == 27:  # 按 ESC 键退出
            break
        
    else: 
        print("未找到 scrcpy 窗口")

# 关闭窗口和进程
cv2.destroyAllWindows()
scrcpy_process.terminate()
 