import cv2
from ultralytics import YOLO
import torch


def recognize_qrcode_with_yolov8(image_path):
    """
    使用YOLOv8检测二维码位置，裁剪后使用cv2.wechat_qrcode识别内容。

    Args:
        image_path (str): 图片路径。

    Returns:
        list: 包含识别到的二维码内容和位置信息的列表。
              每个元素是一个字典，包含 'content' (二维码内容) 和 'bbox' (边界框坐标 [x_min, y_min, x_max, y_max])。
              如果未检测到或识别到二维码，则返回空列表。
    """
    results = []

    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法加载图片: {image_path}")
        return results

    # 使用 YOLOv8 进行预测
    predictions = model(img)

    # 初始化 WeChatQRCode 检测器

    # 遍历 YOLOv8 的预测结果
    for det in predictions:
        if det.boxes is not None and len(det.boxes) > 0:
            for box in det.boxes:
                # 获取边界框坐标 (xyxy 格式)
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())

                # 裁剪二维码区域
                cropped_qr_code = img[y_min:y_max, x_min:x_max]

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


if __name__ == '__main__':
    # 替换为你的图片路径和 YOLOv8 模型路径
    image_file = 't.png'

    location = "./"
    detector_prototxt_path = location + "detect.prototxt"
    detector_caffe_model_path = location + "detect.caffemodel"
    super_resolution_prototxt_path = location + "sr.prototxt"
    super_resolution_caffe_model_path = location + "sr.caffemodel"

    # 加载 YOLOv8 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(location + 'best-yolov8.pt')
    model.to(device)

    # 加载 WeChatQRCode 模型
    detector = cv2.wechat_qrcode.WeChatQRCode(
        detector_prototxt_path,
        detector_caffe_model_path,
        super_resolution_prototxt_path,
        super_resolution_caffe_model_path
    )

    # 执行二维码识别
    recognized_codes = recognize_qrcode_with_yolov8(image_file)

    if recognized_codes:
        print("检测到并识别到的二维码:")
        for code_info in recognized_codes:
            print(f"  内容: {code_info['content']}")
            print(f"  位置 (xmin, ymin, xmax, ymax): {code_info['bbox']}")
    else:
        print("未检测到或识别到二维码。")
