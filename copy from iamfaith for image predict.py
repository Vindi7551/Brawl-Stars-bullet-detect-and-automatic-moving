import aicmder as cmder  # 导入aicmder库作为cmder
from aicmder.module.module import serving, moduleinfo  # 从aicmder.module.module导入serving和moduleinfo装饰器
import io  # 导入io库用于输入输出操作
from PIL import Image  # 从PIL库导入Image用于图像处理
import json  # 导入json库用于JSON操作
import base64  # 导入base64库用于编码转换
import cv2  # 导入cv2库用于图像处理
import numpy as np  # 导入numpy库用于数值计算

import torch  # 导入torch库用于深度学习
from utils.torch_utils import select_device, load_classifier, time_sync  # 从utils.torch_utils导入辅助函数
from models.experimental import attempt_load  # 从models.experimental导入模型加载函数
from utils.augmentations import letterbox  # 从utils.augmentations导入图像预处理函数
from utils.general import non_max_suppression, scale_coords  # 从utils.general导入非极大抑制和坐标缩放函数
from utils.plots import colors, plot_one_box  # 从utils.plots导入颜色选择和绘图函数

def readb64(base64_string, save=False):  # 定义函数用于读取base64编码的图像
    img_array = io.BytesIO(base64.b64decode(base64_string))  # 将base64字符串解码并转换为字节流
    pimg = Image.open(img_array)  # 打开图像文件
    if save:  # 如果需要保存图像
        pimg.save('image.png', 'PNG')  # 保存图像为PNG格式
    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)  # 将图像从RGB转换为BGR格式

@moduleinfo(name='image')  # 使用moduleinfo装饰器定义模块信息
class ImagePredictor(cmder.Module):  # 定义一个类继承自cmder.Module
    
    def __init__(self, file_path, **kwargs) -> None:  # 类的初始化方法
        print('init', file_path)  # 打印初始化信息
        self.device = select_device('')  # 选择设备
        weights = ["/home/faith/android_viewer/thirdparty/yolov5/runs/train/exp2/weights/best.pt"]  # 指定模型权重文件
        self.model = attempt_load(weights, map_location=self.device)  # 加载模型
        self.stride = int(self.model.stride.max())  # 获取模型的步长
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # 获取类别名称

        self.imgsz = 768  # 设置图像大小
        if self.device.type != 'cpu':  # 如果设备不是CPU
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # 运行一次模型以初始化

    @serving  # 使用serving装饰器标记为服务函数
    def predict(self, img_base64):  # 定义预测函数
        print('receive')  # 打印接收信息
        try:
            img0 = readb64(img_base64)  # 读取并解码图像
            img = letterbox(img0, self.imgsz, stride=self.stride)[0]  # 对图像进行缩放和填充
            img = img.transpose((2, 0, 1))[::-1]  # 将图像从HWC格式转换为CHW格式，并从BGR转为RGB
            img = np.ascontiguousarray(img)  # 确保数组内存是连续的

            img = torch.from_numpy(img).to(self.device)  # 将图像转换为torch张量并传输到设备
            img = img.float()  # 将图像数据类型转换为浮点数
            
            img /= 255.0  # 将图像像素值归一化到0-1
            if len(img.shape) == 3:  # 如果图像是单张图像而非批量
                img = img[None]  # 增加一个批次维度
            
            t1 = time_sync()  # 获取当前时间
            visualize, augment = False, False  # 设置是否进行可视化和增强
            pred = self.model(img, augment=augment, visualize=visualize)[0]  # 进行模型推理

            conf_thres, iou_thres, classes, agnostic_nms, max_det= 0.25, 0.45, None, False, 1000  # 设置NMS的参数
            line_thickness = 3  # 设置绘制框的线条粗细
            hide_labels = False  # 设置是否隐藏标签
            hide_conf = False  # 设置是否隐藏置信度
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)  # 应用非极大抑制
            t2 = time_sync()  # 获取当前时间
            print(f'Done. ({t2 - t1:.3f}s)')  # 打印推理时间
            result = []  # 初始化结果列表
            for i, det in enumerate(pred):  # 遍历每张图像的检测结果
                if len(det):  # 如果有检测到的目标
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()  # 调整坐标尺寸

                    for *xyxy, conf, cls in reversed(det):  # 遍历每个检测到的目标
                        c = int(cls)  # 获取类别编号
                        label = None if hide_labels else (self.names[c] if hide_conf else f'{self.names[c]} {conf:.2f}')  # 设置标签
                        pos = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]  # 获取位置
                        result.append({'label': label, 'pos': pos, 'conf': f'{conf:.2f}'})  # 添加结果
            result_json = json.dumps(result)  # 将结果转换为JSON格式
            print(result_json)  # 打印结果
            return result_json  # 返回结果
        except Exception as e:  # 捕获异常
            print(e)  # 打印异常信息
            return "OK"  # 返回OK
