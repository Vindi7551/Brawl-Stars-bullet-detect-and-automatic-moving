import cv2
import subprocess
import numpy as np
from ultralytics import YOLO
import math

def load_model(model_path):
    model = YOLO(model_path)
    return model

model_path = 'best.pt'
model = load_model(model_path)
def process_frame(frame, model):
    # 运行模型进行推理
    results = model(frame)

    # 检查是否有检测结果并且至少有一个边界框
    if len(results) > 0 and len(results[0].boxes) > 0:
        for result in results:
            cursor = result.boxes
            xyxy = cursor.xyxy.cpu().numpy()[0]  # 获取坐标
            conf = cursor.conf.cpu().numpy()[0]  # 获取置信度
            cls_id = cursor.cls.cpu().numpy()[0]  # 假设这样获取类别索引
            
            if conf > 0.5:
                x1, y1, x2, y2 = xyxy[:4]
                # 绘制边界框
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # 获取类别名称，假设有一个列表或字典转换类别索引到名称
                class_names = ["bullet", "bullet-enemy", "charactor"]  # 示例类别名称
                class_name = class_names[int(cls_id)]  # 获取类别名称
                
                # 在图像上添加置信度和类别名称
                label = f'{class_name} {conf:.2f}'
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
    return frame

def send_move_command(angle, speed):
    # 根据角度和速度计算移动命令
    if angle is not None and speed is not None:
        # 将角度转化为直角方向（顺时针90度）
        angle += 90
        if angle > 360:
            angle -= 360

        # 确定滑动的距离
        distance = speed * 10  # 调整比例以适应实际速度

        # 计算滑动终点
        start_x = 320  # 假设屏幕中心为起点，适应1080x2400分辨率
        start_y = 800
        end_x = start_x + int(distance * math.cos(math.radians(angle)))
        end_y = start_y + int(distance * math.sin(math.radians(angle)))

        # 打印移动方向和距离
        print(f"Moving from ({start_x}, {start_y}) to ({end_x}, {end_y}) with speed {speed:.2f}")

        # 使用 adb 命令模拟滑动
        subprocess.run(["adb", "shell", "input", "swipe", str(start_x), str(start_y), str(end_x), str(end_y), "300"])


while True:
    send_move_command(90,10)