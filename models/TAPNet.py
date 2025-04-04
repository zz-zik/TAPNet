# -*- coding: utf-8 -*-
"""
@Project : TFPNet4-DAF
@FileName: adaptive_pixel.py
@Time    : 2024/8/19 下午12:33
@Author  : ZhouFei
@Email   : zhoufei.net@outlook.com
@Desc    : https://github.com/Angknpng/SACNet/blob/main/Code/lib/model.py
"""

import torch
from torch import nn

from models.ahead_pixel_fusion import AdaptiveAhead
from models.necks import build_neck
from models.backbone import build_backbone
from models.dense_head import build_ifi_head
from models.fusion import TFAM
from util.misc import NestedTensor
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


# P2PNet 模式的定义
class TAPNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        cfg.ifi_dict['num_classes'] = 2  # 背景 + 2 个目标类别
        cfg.ifi_dict['num_anchor_points'] = cfg.ifi_dict['row'] * cfg.ifi_dict['line']  # 所有锚点的数量
        input_channels = build_backbone(cfg).get_outplanes()
        cfg.num_channels = build_backbone(cfg).num_channels  # TODO: default: 256 512
        self.feat_layers = cfg.ifi_dict['feat_layers']
        self.modal = cfg.modal  # 几个模态的数据
        self.fusion_type = cfg.fusion_type

        # fusion
        if self.modal == 'RT':
            if self.fusion_type == 'pixel':
                self.ahead = AdaptiveAhead(inp_channels=3, out_channels=3, dim=cfg.dim, num_blocks=cfg.num_blocks,
                                           num_heads=cfg.num_heads, num_layers=cfg.num_layers, )
            elif self.fusion_type == 'feature':
                # fusion
                if 1 in self.feat_layers:
                    self.fusion1 = TFAM(in_channel=160)
                if 2 in self.feat_layers:
                    self.fusion2 = TFAM(in_channel=80)
                if 3 in self.feat_layers:
                    self.fusion3 = TFAM(in_channel=40)
                self.fusion4 = TFAM(in_channel=20)

        # backbone
        self.backbone = build_backbone(cfg)
        self.neck = build_neck(cfg, input_channels)
        self.ifi_head = build_ifi_head(cfg)

    def forward(self, samples_rgb: NestedTensor, samples_tir: NestedTensor = None):
        """
        The forward expects a NestedTensor, which consists of:
            - samples_rgb: batched images , of shape [batch_size, num_channels, h, w]
            - bimodal: samples_tir is not None
        """
        if self.modal == 'RT' and samples_tir is not None:
            if self.fusion_type == 'pixel':
                return self._pixel_multi_spectral(samples_rgb, samples_tir)
            elif self.fusion_type == 'feature':
                return self._feature_multi_spectral(samples_rgb, samples_tir)
        else:
            # extract image features
            features = self.backbone(samples_rgb)
            # decoder
            fusions = self.neck(features)
            # ifi head
            outputs = self.ifi_head(samples_rgb, fusions)
            return outputs

    def _pixel_multi_spectral(self, samples_rgb, samples_tir):
        # ahead_pixel_fusion
        outputs_ahead = self.ahead(samples_rgb, samples_tir)
        fusions = outputs_ahead['fused_img']
        # extract image features
        features = self.backbone(fusions)
        # decoder
        fusions = self.neck(features)
        # ifi head
        outputs = self.ifi_head(outputs_ahead['fused_img'], fusions)
        return outputs, outputs_ahead

    def _feature_multi_spectral(self, samples_rgb, samples_tir):
        features: dict = {}
        features_rgb = self.backbone(samples_rgb)
        features_tir = self.backbone(samples_tir)
        if 1 in self.feat_layers:
            features['4x'] = self.fusion1(features_rgb['4x'], features_tir['4x'])
        if 2 in self.feat_layers:
            features['8x'] = self.fusion2(features_rgb['8x'], features_tir['8x'])
        if 3 in self.feat_layers:
            features['16x'] = self.fusion3(features_rgb['16x'], features_tir['16x'])
        features['32x'] = self.fusion4(features_rgb['32x'], features_tir['32x'])
        # decoder
        fusions = self.neck(features)
        # ifi head
        outputs = self.ifi_head(samples_rgb, fusions)
        return outputs


# 测试build函数
if __name__ == '__main__':
    from configs.config import get_args_config
    from models import build_model

    args = get_args_config()
    training = True
    device = torch.device(f'cuda:{args.gpu_id}')  # 定义训练使用的gpu
    model, _ = build_model(args, training)  # 调用build函数
    model = model.to(device)  # 加载到gpu上
    # print(model)
    img_rgb = torch.randn(1, 3, 512, 640).to(device)
    img_tir = torch.randn(1, 3, 512, 640).to(device)
    out_rgb, outputs_ahead = model(img_rgb, img_tir)  # 运行模型
    print('Number of parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    # print(out)
    # print(model)
    from thop import profile

    flops, params = profile(model, inputs=(img_rgb, img_tir))
    print(f"FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M")
