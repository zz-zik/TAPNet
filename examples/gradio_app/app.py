# -*- coding: utf-8 -*-
"""
@Project : TFPNet4-DAF-Adaptive
@FileName: app.py.py
@Time    : 2025/3/10 下午4:34
@Author  : ZhouFei
@Email   : zhoufei.net@outlook.com
@Desc    : 
@Usage   :
"""
import gradio as gr
from inference import load_model, preprocess_image, predict, rgb_transform, tir_transform
from PIL import Image
import numpy as np
import cv2
import os
import socket

# 动态获取主机的 IP 地址
def get_host_ip():
    try:
        host_ip = socket.gethostbyname(socket.gethostname())
    except:
        host_ip = "127.0.0.1"  # 默认回退到本地地址
    return host_ip


# 设置 Gradio 的服务器参数
os.environ["GRADIO_SERVER_NAME"] = get_host_ip()
os.environ["GRADIO_SERVER_PORT"] = "7860"
os.environ["ORIGINS"] = "*"


# 加载模型
model, device = load_model(
    config_path="../../configs/DroneRGBT/resnet.json",
    weight_path="../../weights/best_mae.pth",
    gpu_id="0"
)


def gradio_interface(rgb_image, tir_image):
    """
    Gradio 接口函数
    """
    # 预处理图像
    rgb_img, tir_img = preprocess_image(rgb_image, tir_image, rgb_transform, tir_transform)

    # 运行模型并获取预测结果
    points, predict_cnt = predict(model, device, rgb_img, tir_img)

    # 可视化预测结果（在 RGB 图像上绘制预测点）
    img_to_draw = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)
    for p in points:
        img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)
    result_image = Image.fromarray(cv2.cvtColor(img_to_draw, cv2.COLOR_BGR2RGB))

    return result_image, f"预测人数: {predict_cnt}"


# 创建 Gradio 接口
demo = gr.Interface(
    fn=gradio_interface,
    inputs=[gr.Image(type="pil", label="RGB 图像"), gr.Image(type="pil", label="TIR 图像")],
    outputs=[gr.Image(type="pil", label="预测结果"), gr.Textbox(label="人数统计")],
    title="双光注意力融合人群头部点计数模型",
    description="上传 RGB 图像和 TIR 图像，模型将预测图像中的人数并可视化预测点。"
)

# 启动 Gradio 应用
demo.launch(
    server_name=os.environ["GRADIO_SERVER_NAME"],
    server_port=int(os.environ["GRADIO_SERVER_PORT"]),
    show_error=True,
)

# 自定义打印信息
custom_message = "* Port Forwarding Post URL: "
port_forwarding_url = "http://172.16.15.10:30106   "  # 你的自定义 URL
print(custom_message + port_forwarding_url)