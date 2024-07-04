import cv2
import numpy as np
import socket
import sys
import subprocess

# 定义屏幕分辨率
width, height = 1080, 2400

# 创建一个TCP/IP套接字
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 连接服务器
server_address = ('localhost', 5038)
print(f"Connecting to {server_address}")
client_socket.connect(server_address)

# 开启屏幕捕获
# capture = cv2.VideoCapture(0)
capture = subprocess.Popen(["D:/scrcpy-win32-v2.4/scrcpy-win32-v2.4/scrcpy.exe --raw-video=- | ffmpeg -v verbose -i - -f rawvideo -pix_fmt bgr24 - 2> ffmpeg_error.log"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

while True:
    try:
        # 读取屏幕图像
        frame_bytes = capture.stdout.read(width * height * 3)
        if not frame_bytes:
            print("Failed to capture frame")
            break

        # 将图像转换为字节流
        # frame_bytes = frame.tobytes()

        # 发送图像数据到服务器
        client_socket.sendall(frame_bytes)

    except Exception as e:
        print(f"Exception: {e}")
        break

# 关闭连接
client_socket.close()
# 修改这部分代码
if capture:
    capture.terminate()  # 终止进程
    capture.wait()       # 等待进程结束
