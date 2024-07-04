import cv2
import numpy as np
import socket
import sys

# 定义屏幕分辨率
width, height = 1080, 2400

# 创建一个TCP/IP套接字
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 绑定套接字到端口
server_address = ('localhost', 5038)
print(f"Starting server on {server_address}")
server_socket.bind(server_address)

# 监听传入连接
server_socket.listen(1)

# 等待连接
print("Waiting for a connection...")
connection, client_address = server_socket.accept()
print(f"Connection from {client_address}")

while True:
    try:
        # 从连接中接收数据
        data = connection.recv(width * height * 3)
        if not data:
            print("No data received")
            break

        # 将接收到的数据转换为NumPy数组，并重塑为图像
        frame_np = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))

        # 显示图像
        cv2.imshow('Frame', frame_np)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Exception: {e}")
        break

# 关闭连接
connection.close()
server_socket.close()
