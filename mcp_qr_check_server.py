from io import BytesIO

from fastapi import Response
from fastapi.responses import JSONResponse

import ssl
import requests
import cv2
import numpy as np
import time

import logging
from ultralytics import YOLO
import torch

from mcp.server.fastmcp import FastMCP

# 创建日志记录器
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 日志格式
log_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ssl._create_default_https_context = ssl._create_unverified_context

location = "./"
detector_prototxt_path = location + "detect.prototxt"
detector_caffe_model_path = location + "detect.caffemodel"
super_resolution_prototxt_path = location + "sr.prototxt"
super_resolution_caffe_model_path = location + "sr.caffemodel"

detector = cv2.wechat_qrcode.WeChatQRCode(
    detector_prototxt_path,
    detector_caffe_model_path,
    super_resolution_prototxt_path,
    super_resolution_caffe_model_path
)

# 加载 YOLOv8 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO(location + 'best-yolov8.pt')
model.to(device)

mcp = FastMCP("qr_check")


@mcp.tool(name="图片二维码检测工具", description="当用户输入一个图片url期望检测图片中的二维码信息时，调用本工具检测")
async def qr_check(
        url: str,
):
    logger.info(f"qr check urls: {url}")

    start_time = time.time()
    logger.info(f"qr check download {url} at {start_time}")
    image = url2pil2(url)
    end_time = time.time()
    logger.info(f"qr check download cost {end_time - start_time}秒")

    results = []
    predictions = model(image)

    # 初始化 WeChatQRCode 检测器

    # 遍历 YOLOv8 的预测结果
    for det in predictions:
        if det.boxes is not None and len(det.boxes) > 0:
            for box in det.boxes:
                # 获取边界框坐标 (xyxy 格式)
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())

                # 裁剪二维码区域
                cropped_qr_code = image[y_min:y_max, x_min:x_max]

                # 使用 WeChatQRCode 检测和解码
                decoded_info, points = detector.detectAndDecode(cropped_qr_code)

                if decoded_info:
                    for i, content in enumerate(decoded_info):
                        if content:
                            results.append({
                                'content': content,
                                'bbox': [x_min, y_min, x_max, y_max]
                            })

    return results


def url2pil2(img_url):
    # 设置用户代理
    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36'
    headers = {'user-agent': user_agent}
    # 发送请求
    response = requests.get(img_url, headers=headers, timeout=1)
    img_data = BytesIO(response.content)
    img = cv2.imdecode(np.frombuffer(img_data.read(), np.uint8), cv2.IMREAD_COLOR)

    return img


if __name__ == '__main__':
    mcp.run(transport="sse")
