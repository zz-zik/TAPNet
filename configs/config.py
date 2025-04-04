# -*- coding: utf-8 -*-
"""
@Project : TFPNet
@Time    : 2024/08/09 9:56
@FileName: config.py
@author  : ZhouFei
@Email   : zhoufei.net@outlook.com
@Desc    : 选择使用 argparse 中的 get_args_parser 或者读取指定目录下面的 json 文件来加载 args
"""
import json
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('Model Parameter', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)  # 学习率
    parser.add_argument('--lr_backbone', default=1e-5, type=float)  # 骨干网络的学习率
    parser.add_argument('--batch_size', default=2, type=int)  # 训练的batch size
    parser.add_argument('--weight_decay', default=1e-4, type=float)  # 权重衰减
    parser.add_argument('--epochs', default=200, type=int)  # 训练的epoch数
    parser.add_argument('--lr_drop', default=200, type=int)  # 学习率衰减的epoch数
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')  # 梯度裁剪的最大值

    # 数据集参数
    parser.add_argument('--dataset', default='GAIIC2024', choices=['GAIIC2024', 'SHHA', 'DroneRGBT'])
    parser.add_argument('--data_root', default='/sxs/zhoufei/P2PNet/GAIIC2024',
                        help='path where the dataset is')
    parser.add_argument("--aug_dict", default={'flip': True, 'resizing': True, 'patch': True, 'offset': True},
                        help='data augmentation parameters')

    # 模型参数
    parser.add_argument('--modal', default='RT', choices=['RT', 'R', 'T'],
                        help='Number of modes: 0 for RGB, 1 for TIR, 2 for RGB+TIR')
    parser.add_argument('--fusion_type', default='pixel', choices=['pixel', 'feature'], help='Fusion module type')
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")  # 冻结权重

    # * pixel
    parser.add_argument('--dim', default=32, type=int, help='ahead_pixel_fusion embedding dim')
    parser.add_argument('--num_blocks', default=[1, 1], type=int, nargs='+', help='ahead_pixel_fusion transformer num_blocks')
    parser.add_argument('--num_heads', default=4, type=int, help='Number of heads in the attention layers')
    parser.add_argument('--num_layers', default=1, type=int, help='ahead_pixel_fusion Detail Feature Extractor num_layers')

    # * backbone
    parser.add_argument('--backbone', default='resnet50',  # TODO: ResNet101
                        choices=['vgg16', 'vgg16_bn', 'ResNet18', 'resnet34',
                                 'resnet50', 'resnet101', 'resnet152', 'ConvNeXt_Tiny', 'ConvNeXt_small',
                                 'ConvNeXt_base', 'ConvNeXt_large', 'ConvNeXt_xlarge', 'SwinTransformer'], type=str,
                        help="Name of the convolutional backbone to use")  # 骨干网络
    parser.add_argument('--dilation', default=False, type=bool,
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")

    # * neck
    parser.add_argument('--num_channels', default=256, type=int, help='Number of channels in the decoder')
    parser.add_argument('--dilations', default=[2, 4, 8], type=int, help='Dilations of last three stages')
    parser.add_argument('--no_aspp', default=False, type=bool, help='If True, no ASPP module will be used')

    # * ifi head
    parser.add_argument("--ifi_dict", default={"feat_layers": [3, 4], "line": 2, "row": 2, "sync_bn": False,
                                               "require_grad": False, "head_layers": [512, 256, 256], "pos_dim": 32,
                                               "ultra_pe": False, "learn_pe": False, "unfold": False, "local": False,
                                               "stride": 1}, help='ifi parameters')
    parser.add_argument("--aux_en", default=False, help='whether to use auxiliary loss')
    parser.add_argument("--aux_number", default=[1, 1], type=int, nargs='+', help='number of auxiliary loss')
    parser.add_argument("--aux_range", default=[1, 4], type=int, nargs='+', help='range of auxiliary loss')
    parser.add_argument("--aux_kwargs", default={'pos_coef': 1., 'neg_coef': 1., 'pos_loc': 0., 'neg_loc': 0.},
                        type=dict, help='kwargs for auxiliary loss')

    # * loss parameters
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")  # 分类损失系数
    parser.add_argument('--set_cost_point', default=0.05, type=float,
                        help="L1 point coefficient in the matching cost")  # 点损失系数
    # * Loss coefficients
    parser.add_argument('--point_loss_coef', default=0.0002, type=float)
    parser.add_argument("--loss_aux", default=0., type=float, help='loss coefficient for auxiliary loss')
    # * Matcher
    parser.add_argument('--eos_coef', default=0.2, type=float,
                        help="Relative classification weight of the no-object class")  # 无目标类别的权重

    # * misc
    parser.add_argument("--output_dir", default="./work_dirs", help="Path to output file")
    parser.add_argument("--vis_dir", default=False, type=bool, help="用于确定是否保存训练中的图像")

    # * Evaluation
    parser.add_argument('--threshold', default=0.2, type=float, help='threshold for evaluation')
    parser.add_argument('--seed', default=42, type=int)  # 随机数种子

    # TODO: 恢复训练的模型权重路径
    parser.add_argument('--resume', default='', help='resume from checkpoint')  # 恢复训练

    parser.add_argument('--start_epoch', default=1, type=int, metavar='N', help='start epoch')  # 开始的epoch数
    parser.add_argument('--eval', action='store_true')  # 是否进行评估
    parser.add_argument('--num_workers', default=4, type=int)  # 加载数据时的线程数
    parser.add_argument('--start_eval', default=5, type=int, help='start evaluating in every n epoch')
    parser.add_argument('--eval_freq', default=1, type=int,
                        help='frequency of evaluation, default setting is evaluating in every 5 epoch')  # 评估的频率
    parser.add_argument('--f1_score', default=True, type=bool, help='whether to use f1 score')
    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for training')  # 训练使用的gpu编号

    return parser


def get_args_json(json_path='config.json'):
    # 读取指定目录下面的 json 文件并返回 args
    with open(json_path, 'r', encoding='utf-8') as file:
        cfg = json.load(file)

    # 创建一个 argparse.Namespace 对象
    args = argparse.Namespace()

    # 遍历 cfg 字典，将键值对设置为 Namespace 对象的属性
    for key, value in cfg.items():
        setattr(args, key, value)

    return args


def get_args_config():
    parser = argparse.ArgumentParser('TAPNet')
    parser.add_argument('-c', '--config_file', type=str, default="", help='The path of config file')
    args = parser.parse_args()
    config_file = args.config_file
    if config_file != "":
        args = get_args_json(config_file)
        print(f"Load parameters from JSON file: {config_file}")
    else:
        parsers = get_args_parser()
        args = parsers.parse_args()
        print("Load parameters from command parser")
    return args


# 使用示例
if __name__ == '__main__':
    # 命令行参数优先，如果需要从 JSON 文件读取参数，可以调用 get_args_config(config_file='config.json')
    # args = get_args_config("/sxs/zhoufei/P2PNet/TFPNet-P2PNet-APGCC-DETR-rgb/configs/GAIIC2/resnet.json")
    args = get_args_config()
    print(args)
