# This file is used to crop the images and labels of Drone dataset.
import os
import random
import re
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class Drone(Dataset):
    def __init__(self, data_root, modal, rgb_transform=None, tir_transform=None, train=False, resizing=False,
                 patch=False, flip=False, offset=False, infer=False):
        self.data_path = data_root  # 'GAIIC2024-1'
        self.train = train  # 是否为训练集
        self.modal = modal  # RGB or TIR
        # 数据地址
        self.data_dir = os.path.join(self.data_path, 'train' if self.train else 'test') if not infer else self.data_path # 数据集路径
        self.rgb_dir = os.path.join(self.data_dir, 'rgb')  # 可见光图像路径
        self.tir_dir = os.path.join(self.data_dir, 'tir')  # 红外光图像路径
        # TODO: 测试集路径
        self.labels_dir = os.path.join(self.data_dir, 'labels')  # 可见光标注文件路径
        self.save_crop = False  # 是否保存裁剪后的图像
        self.img_map = {}  # 存储可见光图像和红外光图像的对应关系
        self.img_list = []  # 存储可见光图像的路径列表

        img_paths = [filename for filename in os.listdir(self.rgb_dir) if filename.endswith('.jpg')]
        for filename in img_paths:
            rgb_img_path = os.path.join(self.rgb_dir, filename)
            tir_img_path = os.path.join(self.tir_dir, filename.replace('.jpg', 'R.jpg'))
            # 读取标注xml文件，1R.xml表示标注文件
            labels_path = os.path.join(self.labels_dir, f"{filename.split('.')[0]}R.xml")
            # 确保可见光与红外光图像都存在。
            if os.path.isfile(rgb_img_path) and os.path.isfile(tir_img_path) and os.path.isfile(labels_path):
                self.img_map[rgb_img_path] = (tir_img_path, labels_path)  # 存储可见光图像和红外光图像的对应关系
                # 为self.img_map['path/1.jpg'] = ('path/1R.jpg', 'path/1R.xml')
                self.img_list.append(rgb_img_path)  # 存储可见光图像的路径列表
        # 对self.img_list进行排序，排序规则为文件名数字排序
        self.img_list = sort_filenames_numerically(list(self.img_map.keys()))

        self.nSamples = len(self.img_list)  # 数据长度
        # 数据预处理
        self.rgb_transform = rgb_transform  # 数据增强
        self.tir_transform = tir_transform  # 数据增强
        self.resizing = resizing  # 是否进行图像缩放
        self.patch = patch  # 是否为patch模式,patch模式下，将图像分割成多个小块，每个块作为单独的样本
        self.flip = flip  # 是否翻转图像
        self.offset = offset

        self.count = 0  # NOTE

    def __len__(self):
        return self.nSamples

    def remove_n_points(self, points, n):
        if len(points) <= n:  # 如果列表为空或n不大于0，直接返回原列表
            raise Exception('points被删没了')
        # 随机打乱嵌套列表
        random.shuffle(points)
        # 移除前n个子列表
        points = points[n:]
        return points

    def __getitem__(self, index):
        rgb_img, tir_img, rgb_target, tir_target = self.__getsingleitem__(index)

        if self.modal == 'RT':
            # 封装最终的输出，包括可见光图像、红外光图像和标注
            return rgb_img, tir_img, rgb_target, tir_target
        if self.modal == 'R':
            return rgb_img, rgb_target
        if self.modal == 'T':
            return tir_img, tir_target

    def __getsingleitem__(self, index):
        assert index < self.nSamples, 'index range error'

        # Get paths for both RGB and TIR images and their annotations
        rgb_img_path = self.img_list[index]
        tir_img_path, labels_path = self.img_map[rgb_img_path]

        # Load both RGB and TIR images and their respective annotations
        rgb_img, rgb_points = load_data(rgb_img_path, labels_path)
        tir_img, tir_points = load_data(tir_img_path, labels_path)

        # If transformations are defined, apply them to both RGB and TIR images
        if self.rgb_transform is not None:
            rgb_img = self.rgb_transform(rgb_img)
        if self.tir_transform is not None:
            tir_img = self.tir_transform(tir_img)

        # Apply resizing (data augmentation) if needed
        if self.resizing and self.train:
            scale_range = [0.7, 1.3]  # Random scaling range
            min_size = min(rgb_img.shape[1:])  # Minimum size of the image
            scale = random.uniform(*scale_range)

            if scale * min_size > 128:
                # Apply the same scaling to both images and their corresponding points
                rgb_img = torch.nn.functional.upsample_bilinear(rgb_img.unsqueeze(0), scale_factor=scale).squeeze(0)
                tir_img = torch.nn.functional.upsample_bilinear(tir_img.unsqueeze(0), scale_factor=scale).squeeze(0)
                rgb_points *= scale
                tir_points *= scale

        # Apply random cropping if patch mode is enabled
        if self.patch and self.train:
            # 同步裁剪RGB和TIR图像
            rgb_img, tir_img, rgb_points, tir_points = random_crop(rgb_img, tir_img, rgb_points, tir_points, 4)
            # rgb_img, tir_img, rgb_points, tir_points = random_crop_with_drift(rgb_img, tir_img, rgb_points, tir_points, 4, offset=self.offset, delta_x=10, delta_y=10)
            self.count += 11
            for i, _ in enumerate(rgb_points):
                rgb_points[i] = torch.Tensor(rgb_points[i])
            for i, _ in enumerate(tir_points):
                tir_points[i] = torch.Tensor(tir_points[i])

        # Apply random flipping to both RGB and TIR images
        if random.random() > 0.5 and self.flip:
            # 同步翻转RGB和TIR
            rgb_img, tir_img, rgb_points, tir_points = synchronized_flip(rgb_img, tir_img, rgb_points, tir_points)

        if not self.patch:
            rgb_points, tir_points = [rgb_points], [tir_points]
        # Convert the images to tensors
        rgb_img = torch.Tensor(rgb_img)
        tir_img = torch.Tensor(tir_img)

        # Package the targets (annotations) for both RGB and TIR
        rgb_target = self._prepare_target(rgb_img_path, rgb_points)
        tir_target = self._prepare_target(rgb_img_path, tir_points)

        # Return both images and their respective annotations
        return rgb_img, tir_img, rgb_target, tir_target

    # Helper function to prepare the target (annotations)
    def _prepare_target(self, img_path, points):
        # pack up related infos
        target = [{} for i in range(len(points))]
        for i, _ in enumerate(points):
            target[i]['point'] = torch.Tensor(points[i])
            # NOTE
            # 将图片的绝对路径，按照'/'分割（Windows和Linux文件分隔符不同），取列表最后一个元素，为文件名
            # 将文件名按照'.'分割，取第一个元素，即去除了文件扩展名
            # 将纯文件名按照'_'分割，取最后一个元素，为文件id
            image_id = int(img_path.split(os.sep)[-1].split('.')[0].split('_')[-1])
            image_id = torch.Tensor([image_id]).long()
            target[i]['image_id'] = image_id
            target[i]['labels'] = torch.ones([points[i].shape[0]]).long()
        return target


# 定义一个辅助函数，用于从文件名中提取数字
def sort_filenames_numerically(filenames):
    # 定义一个辅助函数，用于从文件名中提取数字
    def numeric_key(filename):
        # 使用正则表达式找到文件名中的所有数字序列，并将其转换为整数列表
        numbers = list(map(int, re.findall(r'\d+', filename)))
        # 如果没有数字，返回一个默认值（例如0），以确保文件名也能被排序
        return (tuple(numbers), filename) if numbers else ((), filename)

    # 使用sorted函数和自定义的排序键
    return sorted(filenames, key=numeric_key)


# 读取xml文件，返回标注点坐标
def read_annotation(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = root.findall('object')
    points = []
    for obj in objects:
        if obj is not None:
            point = obj.find('point')
            if point is not None:
                x = float(point.find('x').text)
                y = float(point.find('y').text)
                points.append([x, y])
            else:
                print('No point in object')
    return points


def load_data(img_dir, xml_dir):
    # 读取图像并转换为NumPy数组
    img = cv2.imread(img_dir)
    points = read_annotation(xml_dir)
    # 转换为RGB图像
    images = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return images, np.array(points)


# 定义同步裁剪函数
def random_crop(img_rgb, img_tir, den_rgb, den_tir, num_patch=4):
    half_h = 128
    half_w = 128
    result_rgb_img = np.zeros([num_patch, img_rgb.shape[0], half_h, half_w])
    result_tir_img = np.zeros([num_patch, img_tir.shape[0], half_h, half_w])
    result_rgb_den = []
    result_tir_den = []

    for i in range(num_patch):
        start_h = random.randint(0, img_rgb.shape[1] - half_h)
        start_w = random.randint(0, img_rgb.shape[2] - half_w)
        end_h = start_h + half_h
        end_w = start_w + half_w

        result_rgb_img[i] = img_rgb[:, start_h:end_h, start_w:end_w]
        result_tir_img[i] = img_tir[:, start_h:end_h, start_w:end_w]

        idx_rgb = (den_rgb[:, 0] >= start_w) & (den_rgb[:, 0] <= end_w) & (den_rgb[:, 1] >= start_h) & (
                den_rgb[:, 1] <= end_h)
        idx_tir = (den_tir[:, 0] >= start_w) & (den_tir[:, 0] <= end_w) & (den_tir[:, 1] >= start_h) & (
                den_tir[:, 1] <= end_h)

        record_rgb_den = den_rgb[idx_rgb]
        record_rgb_den[:, 0] -= start_w
        record_rgb_den[:, 1] -= start_h
        result_rgb_den.append(record_rgb_den)

        record_tir_den = den_tir[idx_tir]
        record_tir_den[:, 0] -= start_w
        record_tir_den[:, 1] -= start_h
        result_tir_den.append(record_tir_den)

    return result_rgb_img, result_tir_img, result_rgb_den, result_tir_den


def random_crop_with_drift(img_rgb, img_tir, den_rgb, den_tir, num_patch=4, offset=True, delta_x=10, delta_y=10):
    half_h = 128
    half_w = 128
    result_rgb_img = np.zeros([num_patch, img_rgb.shape[0], half_h, half_w])
    result_tir_img = np.zeros([num_patch, img_tir.shape[0], half_h, half_w])
    result_rgb_den = []
    result_tir_den = []

    for i in range(num_patch):
        start_h_rgb = random.randint(0, img_rgb.shape[1] - half_h)  # 随机选择裁剪的起始位置y坐标
        start_w_rgb = random.randint(0, img_rgb.shape[2] - half_w)  # 随机选择裁剪的起始位置x坐标
        end_h_rgb = start_h_rgb + half_h  # 随机裁剪图像的右下顶点y
        end_w_rgb = start_w_rgb + half_w  # 随机裁剪图像的右下顶点x

        if offset:
            # 对RGB图像应用随机偏移
            drift_x = random.randint(-delta_x, delta_x)  # 生成随机的x偏移量
            drift_y = random.randint(-delta_y, delta_y)  # 生成随机的y偏移量

            # 确保偏移后的裁剪区域在图像范围内
            start_h_tir = max(0, min(start_h_rgb + drift_y, img_tir.shape[1] - half_h))  # 应用偏移后的起点y坐标
            start_w_tir = max(0, min(start_w_rgb + drift_x, img_tir.shape[2] - half_w))  # 应用偏移后的起点x坐标
            end_h_tir = start_h_tir + half_h  # 应用偏移后的终点y坐标
            end_w_tir = start_w_tir + half_w  # 应用偏移后的终点x坐标
        else:
            start_h_tir = start_h_rgb
            start_w_tir = start_w_rgb
            end_h_tir = end_h_rgb
            end_w_tir = end_w_rgb

        result_rgb_img[i] = img_rgb[:, start_h_rgb:end_h_rgb, start_w_rgb:end_w_rgb]
        result_tir_img[i] = img_tir[:, start_h_tir:end_h_tir, start_w_tir:end_w_tir]

        # 调整RGB图像的标注点坐标
        idx_rgb = (den_rgb[:, 0] >= start_w_rgb) & (den_rgb[:, 0] <= end_w_rgb) & (den_rgb[:, 1] >= start_h_rgb) & (
                den_rgb[:, 1] <= end_h_rgb)
        record_rgb_den = den_rgb[idx_rgb]
        record_rgb_den[:, 0] -= start_w_rgb
        record_rgb_den[:, 1] -= start_h_rgb
        result_rgb_den.append(record_rgb_den)

        # 调整TIR图像的标注点坐标
        idx_tir = (den_tir[:, 0] >= start_w_tir) & (den_tir[:, 0] <= end_w_tir) & (den_tir[:, 1] >= start_h_tir) & (
                den_tir[:, 1] <= end_h_tir)
        record_tir_den = den_tir[idx_tir]
        record_tir_den[:, 0] -= start_w_tir
        record_tir_den[:, 1] -= start_h_tir
        result_tir_den.append(record_tir_den)

    return result_rgb_img, result_tir_img, result_rgb_den, result_tir_den


# 定义同步翻转函数
def synchronized_flip(rgb_img, tir_img, rgb_target, tir_target):
    rgb_img = torch.Tensor(rgb_img[:, :, :, ::-1].copy())
    tir_img = torch.Tensor(tir_img[:, :, :, ::-1].copy())
    for i, _ in enumerate(rgb_target):
        rgb_target[i][:, 0] = 128 - rgb_target[i][:, 0]  # 翻转后的坐标
    for i, _ in enumerate(tir_target):
        tir_target[i][:, 0] = 128 - tir_target[i][:, 0]  # 同步翻转TIR的坐标
    return rgb_img, tir_img, rgb_target, tir_target


# 显示图像和标注点
def show_image(img_rgb, img_tir, points_rgb, points_tir):
    # 转换为 numpy 数组并调整维度
    if img_rgb.shape[0] == 3:
        img_rgb = img_rgb.numpy().astype(np.uint8).transpose(1, 2, 0)  # 转换为 (H, W, C) 格式
        img_tir = img_tir.numpy().astype(np.uint8).transpose(1, 2, 0)  # 转换为 (H, W, C) 格式

    def plot_image_with_points(image, points, title):
        fig, ax = plt.subplots(1)  # 创建一个子图
        ax.imshow(image)
        for point in points:
            rect = plt.Rectangle((point[0] - 1, point[1] - 1), 2, 2, fill=False, edgecolor='r', linewidth=2)
            ax.add_patch(rect)
        ax.set_title(title)
        # 显示图像坐标轴
        ax.axis('off')  # 可选择是否显示坐标轴
        plt.show()

    # 显示可见光图像和标注点
    plot_image_with_points(img_rgb, points_rgb, 'RGB Image with Points')
    # 显示红外光图像和标注点
    plot_image_with_points(img_tir, points_tir, 'TIR Image with Points')


if __name__ == '__main__':
    import torchvision.transforms as standard_transforms

    data_path = '/sxs/zhoufei/P2PNet/DroneRGBT'  # 输入文件路径
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
    train_dataset = Drone(data_path, modal='RT', rgb_transform=rgb_transform,
                          tir_transform=tir_transform, train=True, resizing=False, patch=False, flip=False,
                          offset=False)
    val_dataset = Drone(data_path, modal='RT', rgb_transform=rgb_transform,
                        tir_transform=tir_transform, train=False, resizing=False, patch=False, flip=False, offset=False)
    print('训练集样本数：', len(train_dataset))
    print('测试集样本数：', len(val_dataset))
    print('训练集第1个样本可见光图像形状：', train_dataset[0][0].shape, '红外光图像形状：', train_dataset[0][1].shape,
          '标注点target：', len(train_dataset[0][2]))
    print('测试集第1个样本可见光图像形状：', val_dataset[0][0].shape, '红外光图像形状：', val_dataset[0][1].shape,
          '标注点target：', len(val_dataset[0][2]))
    # for i in range(len(val_dataset)):
    #    img_rgb, img_tir, target = val_dataset[i]  # 返回分别为可见光图像，红外光图像，标注点坐标
    # 显示训练集第1个样本可见光图像和标注点
    img_rgb, img_tir, target_rgb, target_tir = val_dataset[938]
    show_image(img_rgb, img_tir, target_rgb[0]['point'], target_tir[0]['point'])
