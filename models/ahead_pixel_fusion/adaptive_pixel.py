# -*- coding: utf-8 -*-
"""
@Project : TFPNet4-DAF
@FileName: adaptive_pixel.py
@Time    : 2024/10/19 下午12:33
@Author  : ZhouFei
@Email   : zhoufei.net@outlook.com
@Desc    : 
"""
from torch import nn
from models.ahead_pixel_fusion import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtractor, DetailFeatureExtractor


class AdaptiveAhead(nn.Module):
    def __init__(self, inp_channels=3, out_channels=3, dim=64, num_blocks=[4, 4], num_heads=8, num_layers=1):
        super(AdaptiveAhead, self).__init__()
        self.lE = Restormer_Encoder(inp_channels=inp_channels, dim=dim, num_blocks=num_blocks)
        self.lD = Restormer_Decoder(out_channels=out_channels, dim=dim, num_blocks=num_blocks)
        self.BaseFuseLayer = BaseFeatureExtractor(dim=dim, num_heads=num_heads)
        self.DetailFuseLayer = DetailFeatureExtractor(num_layers=num_layers, dim=dim)

    def to_grayscale(self, img):
        # Convert RGB to grayscale using the luminance method
        weights = torch.tensor([0.299, 0.587, 0.114], device=img.device).view(1, 3, 1, 1)  # Ensure weights are on the same device as img
        gray = (img * weights).sum(dim=1,
                                   keepdim=True)  # Perform element-wise multiplication and sum across the color channels
        return gray

    def to_rgb(self, gray_img):
        # Duplicate grayscale values across three channels to form an RGB image
        rgb = gray_img.repeat(1, 3, 1, 1)
        return rgb

    def forward(self, img_VI, img_IR):
        # Convert input images to grayscale
        # img_VI = self.to_grayscale(img_VI)
        # img_VI = self.to_grayscale(img_IR)

        # start 1
        feature_V_B, feature_V_D, _ = self.lE(img_VI)
        feature_I_B, feature_I_D, _ = self.lE(img_IR)
        # data_VI_hat, _ = self.lD(img_VI, feature_V_B, feature_V_D)
        # data_IR_hat, _ = self.lD(img_IR, feature_I_B, feature_I_D)

        # start 2
        feature_F_B = self.BaseFuseLayer(feature_V_B + feature_I_B)
        feature_F_D = self.DetailFuseLayer(feature_V_D + feature_I_D)
        fused_img, _ = self.lD(img_VI, feature_F_B, feature_F_D)

        # Convert fused grayscale image back to RGB
        # fused_img = self.to_rgb(fused_img)

        # # 字典包含上述需要计算loss
        # outputs = {'fused_img': fused_img, 'feature_V_B': feature_V_B, 'feature_V_D': feature_V_D,
        #            'feature_I_B': feature_I_B, 'feature_I_D': feature_I_D, 'data_VI_hat': data_VI_hat,
        #            'data_IR_hat': data_IR_hat, 'feature_F_B': feature_F_B, 'feature_F_D': feature_F_D}
        outputs = {'fused_img': fused_img, 'feature_V_B': feature_V_B, 'feature_V_D': feature_V_D,
                   'feature_I_B': feature_I_B, 'feature_I_D': feature_I_D, 'feature_F_B': feature_F_B,
                   'feature_F_D': feature_F_D}
        return outputs


if __name__ == "__main__":
    import torch
    import time
    start_time = time.time()
    # 创建模型实例
    model = AdaptiveAhead(inp_channels=3, out_channels=3, dim=64, num_heads=8, num_layers=1)
    # 将模型转移到 CUDA 设备上
    model = model.cuda()
    # 创建模拟的输入数据并转移到 CUDA 设备上
    input_vi = torch.randn(1, 3, 128, 128).cuda()
    input_ir = torch.randn(1, 3, 128, 128).cuda()  # 确保输入的红外图像尺寸与可见光图像相同

    # 获取模型输出
    outputs = model(input_vi, input_ir)
    print(outputs['fused_img'].shape, time.time()-start_time)

    from thop import profile
    flops, params = profile(model, inputs=(input_vi, input_ir))
    print(f"FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M")