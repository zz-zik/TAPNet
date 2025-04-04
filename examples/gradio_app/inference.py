# -*- coding: utf-8 -*-
"""
@Project : TFPNet4-DAF-Adaptive
@FileName: inference.py
@Time    : 2025/3/10 下午4:37
@Author  : ZhouFei
@Email   : zhoufei.net@outlook.com
@Desc    : 
@Usage   :
"""
import os
import sys
import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
base_dir= os.path.abspath(os.path.join(current_dir, '../..'))
# 将项目根目录添加到 sys.path
if base_dir not in sys.path:
    sys.path.append(base_dir)

from models import build_model
from configs import get_args_json

def load_model(config_path, weight_path, gpu_id="0"):
    """
    加载模型并返回
    """
    cfg = get_args_json(config_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(cfg, training=False)
    model.to(device)

    if weight_path is not None:
        checkpoint = torch.load(weight_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
    model.eval()

    return model, device


def preprocess_image(rgb_image, tir_image, rgb_transform, tir_transform):
    """
    预处理图像
    """
    # 调整大小（假设两张图像大小一致）
    width, height = rgb_image.size
    new_width = width // 128 * 128
    new_height = height // 128 * 128
    rgb_image = rgb_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    tir_image = tir_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    rgb_img = rgb_transform(rgb_image)
    tir_img = tir_transform(tir_image)

    return rgb_img, tir_img


def predict(model, device, rgb_img, tir_img, threshold=0.8):
    """
    运行模型并返回预测结果
    """
    rgb_img = rgb_img.unsqueeze(0).to(device)
    tir_img = tir_img.unsqueeze(0).to(device)

    outputs = model(rgb_img, tir_img)
    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
    outputs_points = outputs['pred_points'][0]

    points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
    predict_cnt = int((outputs_scores > threshold).sum())

    return points, predict_cnt


# 定义数据预处理
rgb_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

tir_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.492, 0.168, 0.430], std=[0.317, 0.174, 0.191]),
])