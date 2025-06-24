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
<strong>If you like TAPNet, please give us a ⭐! Your support motivates us to keep improving!</strong>
</p>

TAPNet is a state-of-the-art crowd counting model that leverages dual-optical attention fusion and adaptive feature decomposition to enhance accuracy and robustness in complex scenarios such as night conditions and dense crowds.


## 🚀 Updates
- ✅ **[2024.05.07]** Initiated the project.
- ✅ **[2024.06.13]** Participated in the 2024 Global Artificial Intelligence Counting Innovation Competition Track 2.
- ✅ **[2024.07.23]** Introduced the Dual-Optical Attention Fusion Module (DAFP).
- ✅ **[2024.08.05]** Added the Adaptive Dual-Optical Feature Decomposition Fusion Module (AFDF).
- ✅ **[2024.08.12]** Optimized training strategies to improve model robustness in misaligned image conditions.
- ✅ **[2025.03.12]** Added a UI interface demo demonstration.
- ✅ **[2025.04.03]** Uploaded to GitHub.
- ✅ **[2025.05.13]** Uploaded to [arXiv](https://arxiv.org/abs/2505.06937v1).

## Model Zoo

![SOTA](https://i-blog.csdnimg.cn/direct/d60a5459d6c94b6aa6aecc7a04954f7f.png)

## Quick Start

### Installation

#### Clone the repository

```shell
git clone git@github.com:zz-zik/TAPNet.git
```

#### Requirements Install

The installation script can be installed by executing the following commands:
```shell
cd TFPNet
pip install -r requirements.txt
```

#### setup Install
The installation script can be installed by executing the following commands:
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

### Training

#### Train the DroneRGBT dataset

```shell
python main.py -c configs/DroneRGBT/resnet.json
```

#### Train the GAIIC2 dataset

```shell
python main.py -c configs/GAIIC2/resnet.json
```

#### Testing

```shell
python test.py -c configs/DroneRGBT/resnet.json --input_dir /path/to/your/test/images --weight_path /path/to/your/weight/path 
```

### app demo

```shell
cd examples/gradio_app
bash run.sh
```

### Frame
The framework consists of the following modules：
<details>
  <summary>👉 Click to view</summary>

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
If you use TAPNet in your research, please cite:

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
