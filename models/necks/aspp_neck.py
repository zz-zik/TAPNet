# -*- coding: utf-8 -*-
"""
@Project : TFPNet
@Time    : 2024/08/21 10:08
@FileName: aspp_neck.py
@author  : ZhouFei
@Desc    : 参考 https://github.com/AaronCIH/APGCC/blob/main/apgcc/models/Decoder.py
"""
import torch
from torch import nn
import torch.nn.functional as F


def conv_norm_relu(in_channels, out_channels, kernel_size, norm_layer):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size),
        norm_layer(out_channels),
        nn.ReLU(inplace=True),
    )


class Neck(nn.Module):
    def __init__(self, in_planes, feat_layers=[3, 4], num_channels=256, sync_bn=False, dilations=[2, 4, 8],
                 no_aspp=False, ):
        super(Neck, self).__init__()
        norm_layer = nn.SyncBatchNorm if sync_bn else nn.BatchNorm2d
        self.in_planes = in_planes  # feat1,2,3,4's out_dims;
        self.feat_num = len(feat_layers)  # change the encoder feature num.
        self.feat_layers = feat_layers  # control the number of decoder features.
        self.channels = num_channels  # default 256
        self.no_aspp = no_aspp

        # Embedding Neck.
        if 1 in self.feat_layers:
            self.enc1 = conv_norm_relu(self.in_planes[0], num_channels, kernel_size=1, norm_layer=norm_layer)
        if 2 in self.feat_layers:
            self.enc2 = conv_norm_relu(self.in_planes[1], num_channels, kernel_size=1, norm_layer=norm_layer)
        if 3 in self.feat_layers:
            if self.no_aspp:
                self.head3 = conv_norm_relu(self.in_planes[2], num_channels, kernel_size=1, norm_layer=norm_layer)
            else:
                self.conv3 = nn.Conv2d(self.in_planes[2], num_channels, kernel_size=1)
                self.aspp = ASPP(num_channels, inner_planes=num_channels, sync_bn=sync_bn, dilations=dilations)
                self.head3 = nn.Sequential(
                    nn.Conv2d(self.aspp.get_outplanes(), num_channels, kernel_size=3, padding=1, dilation=1,
                              bias=False),
                    norm_layer(num_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(0.1))
        if 4 in self.feat_layers:
            if self.no_aspp:
                self.head4 = conv_norm_relu(self.in_planes[-1], num_channels, kernel_size=1, norm_layer=norm_layer)
            else:
                self.conv4 = nn.Conv2d(self.in_planes[-1], num_channels, kernel_size=1)
                self.aspp = ASPP(num_channels, inner_planes=num_channels, sync_bn=sync_bn, dilations=dilations)
                self.head4 = nn.Sequential(
                    nn.Conv2d(self.aspp.get_outplanes(), num_channels, kernel_size=3, padding=1, dilation=1,
                              bias=False),
                    norm_layer(num_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(0.1))


    def forward(self, features: dict):
        # Embedding encoding.
        out = []
        if 1 in self.feat_layers:
            out.append(self.enc1(features['4x']))
        if 2 in self.feat_layers:
            out.append(self.enc2(features['8x']))
        if 3 in self.feat_layers:
            if self.no_aspp:
                aspp3_out = self.head3(features['16x'])
            else:
                aspp3_out = self.aspp(self.conv3(features['16x']))
                aspp3_out = self.head3(aspp3_out)
            out.append(aspp3_out)
        if 4 in self.feat_layers:
            if self.no_aspp:
                aspp_out = self.head4(features['32x'])
            else:
                aspp_out = self.aspp(self.conv4(features['32x']))
                aspp_out = self.head4(aspp_out)
            out.append(aspp_out)

        return out


class ASPP(nn.Module):
    """
    Reference:
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """
    def __init__(self, in_planes, inner_planes=256, sync_bn=False, bn=False, dilations=[12, 24, 36]):
        super(ASPP, self).__init__()

        norm_layer = nn.SyncBatchNorm if sync_bn else nn.BatchNorm2d
        if not bn:
            self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                       nn.Conv2d(in_planes, inner_planes, kernel_size=1,
                                                 padding=0, dilation=1, bias=False),
                                       nn.ReLU(inplace=True))
            self.conv2 = nn.Sequential(nn.Conv2d(in_planes, inner_planes, kernel_size=1,
                                                 padding=0, dilation=1, bias=False),
                                       nn.ReLU(inplace=True))
            self.conv3 = nn.Sequential(nn.Conv2d(in_planes, inner_planes, kernel_size=3,
                                                 padding=dilations[0], dilation=dilations[0], bias=False),
                                       nn.ReLU(inplace=True))
            self.conv4 = nn.Sequential(nn.Conv2d(in_planes, inner_planes, kernel_size=3,
                                                 padding=dilations[1], dilation=dilations[1], bias=False),
                                       nn.ReLU(inplace=True))
            self.conv5 = nn.Sequential(nn.Conv2d(in_planes, inner_planes, kernel_size=3,
                                                 padding=dilations[2], dilation=dilations[2], bias=False),
                                       nn.ReLU(inplace=True))
        else:
            self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                       nn.Conv2d(in_planes, inner_planes, kernel_size=1, padding=0, dilation=1, bias=False),
                                       norm_layer(inner_planes),
                                       nn.ReLU(inplace=True))
            self.conv2 = nn.Sequential(
                                       nn.Conv2d(in_planes, inner_planes, kernel_size=1, padding=0, dilation=1, bias=False),
                                       norm_layer(inner_planes),
                                       nn.ReLU(inplace=True))
            self.conv3 = nn.Sequential(nn.Conv2d(in_planes, inner_planes, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
                                       norm_layer(inner_planes),
                                       nn.ReLU(inplace=True))
            self.conv4 = nn.Sequential(nn.Conv2d(in_planes, inner_planes, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
                                       norm_layer(inner_planes),
                                       nn.ReLU(inplace=True))
            self.conv5 = nn.Sequential(nn.Conv2d(in_planes, inner_planes, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
                                       norm_layer(inner_planes),
                                       nn.ReLU(inplace=True))
        self.out_planes = (len(dilations) + 2) * inner_planes

    def get_outplanes(self):
        return self.out_planes

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.interpolate(self.conv1(x), size=(h, w), mode='bilinear', align_corners=True)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        aspp_out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        return aspp_out


def build_neck(args, input_channels):
    return Neck(
        in_planes=input_channels,
        feat_layers=args.ifi_dict['feat_layers'],
        num_channels=args.num_channels,
        sync_bn=args.ifi_dict['sync_bn'],
        dilations=args.dilations,
        no_aspp=args.no_aspp)


if __name__ == '__main__':
    from thop import profile

    x = {'4x': torch.randn(1, 256, 128, 160), '8x': torch.randn(2, 512, 64, 80), '16x': torch.randn(2, 1024, 32, 40),
         '32x': torch.randn(2, 2048, 16, 20)}
    in_planes = [256, 512, 1024, 2048]
    model = Neck(in_planes, feat_layers=[3, 4], num_channels=256, )
    output = model(x)
    flops, params = profile(model, inputs=(x,))
    # 输出模型的FLOPs和参数量
    print('Number of parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print(f"FLOPs: {flops}, Params: {params}")
    print(f"FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M")
