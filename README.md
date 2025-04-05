# 基于Transformer的辅助点引导网络的双光人群计数和定位算法

## 简介

近年来，无人机视角下的人群计数任务引起了广泛的关注，这项工作致力于准确识别图像中人群的数量，并确定目标的具体位置。
然而，该领域普遍存在着人群密集遮挡、弱光等复杂场景下精准计数难的问题。为此，本文提出了双光注意力融合人群头部点计数模型（TAFP）。通过引入红外图像的互补信息，提出了双光注意力融合模块（DAFP），从而弥补单一传感器在夜间等不良条件下的成像限制，提升全天时人群计数的准确性和鲁棒性。为了充分利用不同的模态信息，解决图像对之间系统性错位所带来的定位不准确的问题，本文还提出了一种自适应双光特征分解融合模块（AFDF），该模块能够同时学习两种模态间的内在关系，有效地对齐模态间的潜在空间特征。此外，我们还优化了训练策略以实现增强模型的鲁棒性，包括空间随机偏移数据增强，目的是在不损失偏移图像计数精度的同时，进一步提高模型在图像错位条件下两种模态中物体定位的准确性。
我们在两个具有挑战性的公共数据集DroneRGBT和GAIIC2上进行了广泛的实验，证明所提出的方法在性能上优于现有的技术，尤其在具有挑战性的密集型弱光场景下，并且我们所提出的训练策略能够优化模型性能和训练成本。

## 框架说明

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

## 变更日志

- 2024年05月07日: 开始此项目
- 2024年06月13日: 在2024全球人工智能计数创新大赛赛道二中参赛
- 2024年07月23日: 增加了双光注意力融合模块（DAFP）
- 2024年08月05日: 增加了自适应双光特征分解融合模块（AFDF）
- 2024年08月12日: 优化了训练策略，提升模型在图像错位条件下两种模态中物体定位的准确性
- 2024年09月16日: 完善此工程的全部训练测试脚本，README.md文件，以及示例文件
- 2024年09月23日: 完成所有实验
- 2025年03月12日: 添加了UI界面demo演示
- 2025年04月03日: 上传到github

## 安装

### 克隆镜像

```shell
git clone git@github.com:zz-zik/TAPNet.git
```

### pip 安装依赖包

安装脚本可执行下述命令进行安装:
```shell
cd TFPNet
pip install -r requirements.txt
```

### setup.sh 安装依赖包
安装脚本可执行下述命令进行安装:
```shell
cd TFPNet
bash setup.sh
```

## 训练

### 训练DroneRGBT数据集

```shell
python main.py -c configs/DroneRGBT/resnet.json
```

### 训练GAIIC2数据集

```shell
python main.py -c configs/GAIIC2/resnet.json
```

### 测试

```shell
python test.py -c configs/DroneRGBT/resnet.json --input_dir /path/to/your/test/images --weight_path /path/to/your/weight/path 
```

## app demo

```shell
cd examples/gradio_app
bash run.sh
```


