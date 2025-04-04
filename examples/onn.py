"""
@Project : TFPNet
@Time    : 2024/08/03 16:02
@FileName: onn.py
@author  : ZhouFei
@Desc    : 将权重/sxs/zhoufei/P2PNet/TFPNet-APGCC/outputs/ckpt/best_mae.pth转换为onnx
"""
import torch
from configs.config import get_args_config
from models import build_model

args = get_args_config()

device = torch.device(f'cuda:{args.gpu_id}')
# 实例化网络
model = build_model(args, training=False)
model = model.to(device)
# TODO: 导入权重
model.load_state_dict(torch.load('/sxs/zhoufei/P2PNet/TFPNet-DAF/outputs_GAR_9.5730-f1_0.4620/ckpt/best_mae.pth'), strict=False)

# 将模型转换为推理模式
model.eval()

# 设置输入
batch_size = 1  # just a random number
x_rgb = torch.randn(batch_size, 3, 512, 640).to(device)  # 输入RGB数据
x_tir = torch.randn(batch_size, 3, 512, 640).to(device)  # 输入TIR数据
# 导出ONNX模型

# 使用torch.no_grad()确保在导出期间不跟踪梯度

torch.onnx.export(model,
                  (x_rgb, x_tir),
                  "/sxs/zhoufei/P2PNet/TFPNet-DAF/outputs_GAR_9.5730-f1_0.4620/ckpt/model.onnx",
                  export_params=True,
                  opset_version=16,
                  do_constant_folding=True,
                  input_names=['input_rgb', 'input_tir'],
                  output_names=['output'],
                  dynamic_axes={'input_rgb': {0: 'batch_size'},  # 根据需要添加动态轴
                                'input_tir': {0: 'batch_size'}})
print('Successfully exported ONNX model')
# 计算原始Pytorch模型的输出，用于验证导出的ONNX 模型是否能计算出相同的值。
output = model(x_rgb, x_tir)  # 计算原始Pytorch模型的输出
# print("PyTorch output shape: ", output.shape)
