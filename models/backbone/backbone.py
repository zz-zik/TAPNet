# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -*- coding: utf-8 -*-
"""
@Project : TFPNet
@Time    : 2024/08/09 13:40
@FileName: backbone.py
@author  : ZhouFei
@Desc    : Backbone modules.
"""
import torch
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import List
from util.misc import is_main_process


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "4x", "layer2": "8x", "layer3": "16x", "layer4": "32x"}
        else:
            return_layers = {'layer4': "4x"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def get_outplanes(self) -> List[int]:
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 512, 640)
            dummy_output = self.forward(dummy_input)  # 使用普通 Tensor
            outplanes = [x.shape[1] for x in dummy_output.values()]
        return outplanes

    def forward(self, x):
        xs = self.body(x)
        out = {name: x for name, x in xs.items()}
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str, train_backbone: bool, num_channels: int, return_interm_layers: bool, dilation: bool):
        # Dynamically obtain the pre-trained weight class corresponding to the model
        weights_class = getattr(torchvision.models, f"{name}_Weights".replace("resnet", "ResNet"), None)
        if weights_class is None:
            raise ValueError(f"Model {name} does not have a default weights class.")
        # get the default pre training weights
        default_weights = weights_class.DEFAULT
        # Use the weights parameter and select the weights based on the is_main_process results
        backbone = getattr(torchvision.models, name)(replace_stride_with_dilation=[False, False, dilation],
                                                     weights=is_main_process() and default_weights,
                                                     norm_layer=FrozenBatchNorm2d)
        # num_channels = 256 if name in ('resnet18', 'resnet34', 'resnet50') else 512
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


def build_backbone(args):
    train_backbone = args.lr_backbone > 0
    model = Backbone(args.backbone, train_backbone, args.num_channels, True, args.dilation)
    return model


if __name__ == '__main__':
    import time

    start_time = time.time()
    backbone = Backbone('resnet50', True, 256, True, False)
    print('Number of parameters: {}'.format(sum(p.numel() for p in backbone.parameters() if p.requires_grad)))
    x = torch.randn(1, 3, 512, 640)
    outs = backbone(x)
    for name, out in outs.items():
        print(f"name:{name}, img shape:{out.shape}")

    from thop import profile

    flops, params = profile(backbone, inputs=(x,))
    print(f"flops:{flops / 1e9}, params:{params / 1e6}")

    from configs.config import get_args_config
    args = get_args_config()
    models = build_backbone(args)
    print(models.num_channels)
    print(models.get_outplanes())
