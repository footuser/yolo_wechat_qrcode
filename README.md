[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/footuser-yolo-wechat-qrcode-badge.png)](https://mseep.ai/app/footuser-yolo-wechat-qrcode)

# yolo_wechat_qrcode
提高微信二维码识别精确率的小工具

### 原理

二阶段检测:

首先使用训练好的二维码目标识别yolov8模型，识别大图片中小二维码的位置，并裁剪出来


然后使用微信开源的wechat_qrcode模型，进行二维码检测，提取二维码内容



提高了直接使用二维码检测部分大图小二维码场景下无法识别的问题


### 试用

pip install -r requirements.txt

python test.py

### 运行结果
0: 384x640 1 qr_code, 268.3ms
Speed: 3.0ms preprocess, 268.3ms inference, 2.0ms postprocess per image at shape (1, 3, 384, 640)
检测到并识别到的二维码:
  内容: http://weixin.qq.com/r/mp/YEk0LAPEkF7vrUdP9xxy
  位置 (xmin, ymin, xmax, ymax): [564, 286, 610, 332]

### api接口
```shell
python app.py

测试 http://127.0.0.1:8080/check?url=
```

### MCP SERVER
```shell
python mcp_qr_check_server.py

sse地址：http://0.0.0.0:8000
```

### MCP CLIENT
```shell
python mcp_client.py
```