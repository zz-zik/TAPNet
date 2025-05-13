<!--# [TAPNet: Transformer-based Auxiliary Point Detection Network for Crowd Counting Tasks](https://arxiv.org/abs/2505.06937v1) -->

English | [简体中文](README_cn.md) | [English](README.md) | [CSDN Blog](https://blog.csdn.net/weixin_62828995?spm=1000.2115.3001.5343)

<h2 align="center">
  TAPNet: Transformer-based Dual-Optical Attention Fusion for Crowd Counting
</h2>

<p align="center">
    <a href="https://huggingface.co/spaces/yourusername/TAPNet">
        <img alt="hf" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
    </a>
    <a href="https://github.com/zz-zik/TAPNet/blob/master/LICENSE">
        <img alt="license" src="https://img.shields.io/badge/LICENSE-Apache%202.0-blue">
    </a>
    <a href="https://github.com/zz-zik/TAPNet/pulls">
        <img alt="prs" src="https://img.shields.io/github/issues-pr/zz-zik/TAPNet">
    </a>
    <a href="https://github.com/zz-zik/TAPNet/issues">
        <img alt="issues" src="https://img.shields.io/github/issues/zz-zik/TAPNet?color=olive">
    </a>
    <a href="https://arxiv.org/abs/2505.06937v1">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2505.06937v1-red">
    </a>
    <a href="https://results.pre-commit.ci/latest/github/zz-zik/TAPNet/master">
        <img alt="pre-commit.ci status" src="https://results.pre-commit.ci/badge/github/zz-zik/TAPNet/master.svg">
    </a>
    <a href="https://github.com/zz-zik/TAPNet">
        <img alt="stars" src="https://img.shields.io/github/stars/zz-zik/TAPNet">
    </a>
</p>

<p align="center">
    📄 This is the official implementation of the paper:
    <br>
    <a href="https://arxiv.org/abs/2505.06937v1">Transformer-Based Dual-Optical Attention Fusion Crowd Head Point Counting and Localization Network</a>
</p>

<p align="center">
Fei Zhou, Yi Li, Mingqing Zhu
</p>

<p align="center">
Neusoft Institute Guangdong, China & Airace Technology Co.,Ltd., China
</p>

<p align="center">
    <a href="https://paperswithcode.com/sota/crowd-counting">
        <img alt="sota" src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transformer-based-dual-optical-attention-fusion/crowd-counting">
    </a>
</p>

<p align="center">
<strong>如果您喜欢 TAPNet，请为我们点赞⭐！您的支持是我们不断进步的动力！TAPNet 是一种最先进的人群计数模型，它利用双光注意融合和自适应特征分解来提高夜间条件和密集人群等复杂场景下的准确性和鲁棒性。</strong>
</p>

TAPNet 是一种最先进的人群计数模型，它利用双光学注意力融合和自适应特征分解，提高了在夜间条件和密集人群等复杂场景下的准确性和鲁棒性。

## 🚀 更新

- [x] **[2024.05.07]** 启动项目。
- [x] **[2024.06.13]** 参加 2024 年全球人工智能计数创新大赛第 2 赛道。
- [x] **[2024.07.23]** 推出双光学注意力融合模块（DAFP）。
- [x] **[2024.08.05]** 新增自适应双光学特征分解融合模块（AFDF）。
- [x] **[2024.08.12]** 优化了训练策略，以提高模型在错位图像条件下的鲁棒性。
- [x] **[2025.03.12]** 添加了用户界面演示。
- [x] **[2025.04.03]** 上传到 GitHub。
- [x] **[2025.05.13]** 上传到 [arXiv](https://arxiv.org/abs/2505.06937v1)。

## 模型评估结果

| Method   | Backbone  | Approach | Modality | DroneRGBT MAE↓ | DroneRGBT MSE↓ | DroneRGBT F1↑ | GAII C2 MAE↓ | GAII C2 MSE↓ | GAII C2 F1↑ |
|----------|-----------|----------|----------|----------------|----------------|---------------|--------------|--------------|-------------|
| P2PNet   | Vgg16     | Point    | RGB      | 10.83          | 17.09          | 0.596         | 10.95        | 21.01        | 0.455       |
| CLTR     | ResNet50  | Point    | RGB      | 12.06          | 20.86          | 0.587         | 11.37        | 21.88        | 0.423       |
| PET      | Vgg16     | Point    | RGB      | 10.92          | 16.85          | **0.611**     | 10.10        | 17.36        | 0.412       |
| APGCC    | Vgg16     | Point    | RGB      | 11.50          | 16.61          | 0.603         | 10.35        | 18.92        | 0.409       |
| DroneNet | YOLOv5    | Detection| TIR      | 18.6           | 25.2           | -             | 15.86        | 25.62        | 0.379       |
|          |           |          | R-T      | 10.1           | 18.8           | -             | 9.93         | 17.39        | 0.491       |
| MMCount  | CNN       | Map      | TIR      | 16.0           | 23.3           | -             | 15.25        | 22.82        | 0.334       |
|          |           |          | R-T      | **9.2**        | 18.0           | -             | 9.78         | 19.33        | 0.489       |
|          |           |          | RGB      | 10.32          | **16.14**      | 0.610         | **8.54**     | **13.63**    | **0.506**   |
| TAPNet (ours) | ResNet50 | Point    | TIR      | 13.15          | 19.86          | 0.586         | 13.91        | 20.06        | 0.465       |
|          |           |          | R-T      | **7.32**       | **11.54**      | **0.657**     | **7.87**     | **13.25**    | **0.526**   |

## 入门指南

### 安装

#### 克隆镜像

```shell
git clone git@github.com:zz-zik/TAPNet.git
```
 
#### Requirements 安装

安装脚本可执行下述命令进行安装:
```shell
cd TFPNet
pip install -r requirements.txt
```

#### setup 安装
安装脚本可执行下述命令进行安装:
```shell
cd TFPNet
bash setup.sh
```

### Data Preparation
Download the DroneRGBT dataset and organize it as follows:
```text
dataset/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   └── ...
│   ├── val/
│   │   ├── img1.jpg
│   │   └── ...
└── annotations/
    ├── train.json
    └── val.json
```

### 训练

#### 训练DroneRGBT数据集

```shell
python main.py -c configs/DroneRGBT/resnet.json
```

#### 训练GAIIC2数据集

```shell
python main.py -c configs/GAIIC2/resnet.json
```

#### 测试

```shell
python test.py -c configs/DroneRGBT/resnet.json --input_dir /path/to/your/test/images --weight_path /path/to/your/weight/path 
```

### app demo

```shell
cd examples/gradio_app
bash run.sh
```

### 框架说明
该框架包含以下模块：
<details>
  <summary>👉 点击查看</summary>

```text
TFPNet
    ├─configs               # 配置文件
    │  ├─GAIIC2             # GAIIC2数据集配置文件
    │  ├─DroneRGBT          # DroneRGBT数据集配置文件
    │  └─SHHA               # SHHA数据集配置文件
    ├─crowd_datasets        # 数据集加载
    │  ├─SHHA               # SHHA数据集加载
    │  ├─GAIIC              # GAIIC2数据集加载
    │  └─Drone              # Drone数据集加载
    ├─models                # 模型文件
    │  ├─backbone           # Backbone文件
    │  ├─neck               # Neck文件
    │  ├─dense_head         # 模型头文件
    │  ├─ahead_pixel_fusion # 像素级自适应注意力融合模块
    │  ├─losses             # 损失函数文件
    │  ├─matcher.py         # 匹配器文件
    │  └─TFPNet.py          # 主函数文件
    ├─examples              # 示例文件
    │  ├─gradio_app         # Gradio界面文件
    │  └─arg2format         # 参数格式化文件
    ├─scripts               # 脚本文件
    ├─util                  # 工具文件
    ├─work_dirs             # 模型权重文件
    ├─weights               # 预训练权重文件
    └─output                # 输出文件
```
</details>

### Citation
If you use RT-FINE in your research, please cite:

<details open>
<summary> bibtex </summary>

```bibtex
@article{zhou2025transformer,
  title={Transformer-Based Dual-Optical Attention Fusion Crowd Head Point Counting and Localization Network},
  author={Zhou, Fei and Li, Yi and Zhu, Mingqing},
  journal={arXiv preprint arXiv:2505.06937v1},
  year={2025},
  eprint={2505.06937v1},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
</details>

