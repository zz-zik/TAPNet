import os
import cv2
import torch
import argparse
import warnings
import numpy as np
from PIL import Image
from tqdm import tqdm
from models import build_model
from configs import get_args_json, get_args_parser
import torchvision.transforms as standard_transforms
from util.logger import setup_logger, get_environment_info
from crowd_datasets.Drone.Drone import Drone

warnings.filterwarnings('ignore')


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="./configs/DroneRGBT/resnet.json", help='The path of config file')
    parser.add_argument('--input_dir', type=str, default="/sxs/zhoufei/P2PNet/DroneRGBT/test", help='The path of input image')
    parser.add_argument('--weight_path', type=str, default="./work_dirs/DroneRGBT/2024_12_11_17_52-mse_7.3667/ckpt/best_mae.pth", help='The path of weight file')
    parser.add_argument('--gpu_id', type=str, default='0', help='the gpu used for training')
    parser.add_argument('--threshold', type=float, default=0.8, help='The threshold of prediction')
    parser.add_argument('--outputs', type=str, default="./outputs/result", help='The path of output image')
    args = parser.parse_args()
    return args


def main():
    args = args_parser()
    cfg = get_args_json(args.config) if args.config != "" else get_args_parser()
    logger = setup_logger('TAPNet', os.path.join(args.outputs, 'log'), 0)
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(cfg.gpu_id)
    device = torch.device('cuda')
    logger.info(get_environment_info())

    # 获得模型
    model = build_model(cfg, training=False)
    # move to GPU
    model.to(device)
    # 加载训练好的模型
    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    # 转换为评估模式
    model.eval()

    # 定义数据预处理
    rgb_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
    ])

    tir_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(
            mean=[0.492, 0.168, 0.430],
            std=[0.317, 0.174, 0.191]),
    ])

    # 使用 Drone 类读取 RGB 和 TIR 图像路径
    test_set = Drone(data_root=args.input_dir, modal=cfg.modal, rgb_transform=rgb_transform, tir_transform=tir_transform, infer=True)
    rgb_images = test_set.img_list
    tir_images = [test_set.img_map[rgb_path][0] for rgb_path in rgb_images]

    count_list = []

    for i in tqdm(range(len(rgb_images))):
        rgb_path = rgb_images[i]
        tir_path = tir_images[i]
        rgb_raw = Image.open(rgb_path)
        tir_raw = Image.open(tir_path)
        # 四舍五入
        width, height = rgb_raw.size
        new_width = width // 128 * 128
        new_height = height // 128 * 128
        # 新版本的Pillow
        rgb_raw = rgb_raw.resize((new_width, new_height), Image.Resampling.LANCZOS)
        tir_raw = tir_raw.resize((new_width, new_height), Image.Resampling.LANCZOS)
        # 预处理
        rgb_img = rgb_transform(rgb_raw)
        tir_img = tir_transform(tir_raw)

        samples_rgb = torch.Tensor(rgb_img).unsqueeze(0)
        samples_tir = torch.Tensor(tir_img).unsqueeze(0)
        samples_rgb = samples_rgb.to(device)
        samples_tir = samples_tir.to(device)

        # 运行模型
        outputs = model(samples_rgb, samples_tir)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

        outputs_points = outputs['pred_points'][0]

        threshold = args.threshold
        # 过滤低置信度的预测点
        points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
        predict_cnt = int((outputs_scores > threshold).sum())  # 预测点数量

        # NOTE: 重定向到文件
        count_list.append(predict_cnt)
        # logger.info(f"{i+1},{predict_cnt}")
        # 预测
        size = 2  # 预测点的大小
        # 确保张量在CPU上
        img_to_draw = cv2.cvtColor(np.array(rgb_raw), cv2.COLOR_RGB2BGR)  # 转换为 cv2 格式
        for p in points:
            img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
        # 保存可视化图像
        img_path = os.path.join(args.outputs, 'images')
        os.makedirs(img_path, exist_ok=True) if not os.path.exists(img_path) else None
        cv2.imwrite(os.path.join(os.path.join(args.outputs, 'images'), f"{i + 1}-{predict_cnt}.jpg"), img_to_draw)


if __name__ == '__main__':
    main()