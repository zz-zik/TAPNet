import os
import random
import re
import scipy.io as io
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2


class SHHA(Dataset):
    def __init__(self, data_root, transform=None, train=False, resizing=False, patch=False, flip=False):
        self.data_path = data_root  # 'ShanghaiTech_A、ShanghaiTech——B'
        self.train = train  # 是否为训练集
        # 数据地址
        self.data_dir = os.path.join(self.data_path, 'train_data' if self.train else 'test_data')  # 数据集路径
        self.img_dir = os.path.join(self.data_dir, 'images')  # 图像路径
        self.mat_dir = os.path.join(self.data_dir, 'ground-truth')  # 标注文件mat路径
        self.img_map = {}  # 图像路径和gt路径映射
        self.img_list = []  # 存储可见光图像的路径列表
        img_paths = [filename for filename in os.listdir(self.img_dir) if filename.endswith('.jpg')]
        for filename in img_paths:
            img_path = os.path.join(self.img_dir, filename)
            # 读取标注mat文件，1R.mat表示标注文件
            mat_path = os.path.join(self.mat_dir, f"GT_{filename.split('.')[0]}.mat")
            if os.path.isfile(img_path) and os.path.isfile(mat_path):
                self.img_map[img_path] = mat_path  # 储存图像路径映射
                self.img_list.append(img_path)  # 存储图像的路径列表
        # 对self.img_list进行排序，排序规则为文件名数字排序
        self.img_list = sort_filenames_numerically(list(self.img_map.keys()))

        self.nSamples = len(self.img_list)  # 数据长度
        # 数据预处理
        self.transform = transform  # 数据增强
        self.resizing = resizing  # 是否进行图像缩放
        self.patch = patch  # 是否为patch模式,patch模式下，将图像分割成多个小块，每个块作为单独的样本
        self.flip = flip  # 是否翻转图像
        self.count = 0  # NOTE

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_path = self.img_list[index]
        mat_path = self.img_map[img_path]
        # 加载图像和地面实况
        img, point = load_data(img_path, mat_path)
        # 应用数据增强
        if self.transform is not None:
            img = self.transform(img)

        # 封装输出
        if self.resizing:
            # data augmentation -> random scale
            scale_range = [0.7, 1.3]  # NOTE: 随机缩放范围
            min_size = min(img.shape[1:])  # 获取图像的最小边长
            scale = random.uniform(*scale_range)  # 随机生成缩放比例
            # scale the image and points
            if scale * min_size > 128:
                img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor=scale).squeeze(0)
                point *= scale
        # random crop augumentaiton
        if self.patch:
            # print(f"==== processing {img_path.split(os.sep)[-1]} ====")
            img, point = random_crop(img, point)
            self.count += 1  # NOTE
            # print(f"processed: {self.count}")
            for i, _ in enumerate(point):
                point[i] = torch.Tensor(point[i])
        # random flipping
        if random.random() > 0.5 and self.flip:
            # random flip
            img = torch.Tensor(img[:, :, :, ::-1].copy())
            for i, _ in enumerate(point):
                point[i][:, 0] = 128 - point[i][:, 0]

        if not self.patch:
            point = [point]

        img = torch.Tensor(img)
        # pack up related infos
        target = [{} for i in range(len(point))]
        for i, _ in enumerate(point):
            target[i]['point'] = torch.Tensor(point[i])
            # NOTE
            # 将图片的绝对路径，按照'/'分割（Windows和Linux文件分隔符不同），取列表最后一个元素，为文件名
            # 将文件名按照'.'分割，取第一个元素，即去除了文件扩展名
            # 将纯文件名按照'_'分割，取最后一个元素，为文件id
            image_id = int(img_path.split(os.sep)[-1].split('.')[0].split('_')[-1])
            image_id = torch.Tensor([image_id]).long()
            target[i]['image_id'] = image_id
            target[i]['labels'] = torch.ones([point[i].shape[0]]).long()

        return img, target


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


# 读取mat文件，返回标注点坐标
def read_annotation(mat_path):
    # 读取.mat文件
    mat = io.loadmat(mat_path)
    keypoints = mat["image_info"][0, 0][0, 0][0]  # 获取标注点坐标
    points = []
    # 遍历points数组，将其转换为所需的格式
    for keypoint in keypoints:
        x, y = float(keypoint[0]), float(keypoint[1])
        points.append([x, y])
    return points


def load_data(img_path, gt_path):
    # 加载图像
    img = cv2.imread(img_path)
    # 加载标注实测点
    points = read_annotation(gt_path)
    # 转换为RGB图像
    images = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return images, np.array(points)


# 随机裁剪图像和点
def random_crop(img, den, num_patch=4):
    half_h = 128
    half_w = 128
    result_img = np.zeros([num_patch, img.shape[0], half_h, half_w])
    result_den = []
    # crop num_patch for each image
    for i in range(num_patch):
        start_h = random.randint(0, img.size(1) - half_h)
        start_w = random.randint(0, img.size(2) - half_w)
        end_h = start_h + half_h
        end_w = start_w + half_w
        # copy the cropped rect
        result_img[i] = img[:, start_h:end_h, start_w:end_w]
        # copy the cropped points
        idx = (den[:, 0] >= start_w) & (den[:, 0] <= end_w) & (den[:, 1] >= start_h) & (den[:, 1] <= end_h)
        # shift the corrdinates
        record_den = den[idx]
        record_den[:, 0] -= start_w
        record_den[:, 1] -= start_h

        result_den.append(record_den)

    return result_img, result_den


if __name__ == '__main__':
    data_path = "/sxs/zhoufei/P2PNet/ShanghaiTech/part_A"
    train_dataset = SHHA(data_path, train=True, resizing=False, patch=False, flip=False)
    val_dataset = SHHA(data_path, train=False, resizing=False, patch=False, flip=False)
    print('训练集样本数：', len(train_dataset))
    print('测试集样本数：', len(val_dataset))
    print('训练集第1个样本可见光图像形状：', train_dataset[0][0].shape, '标注点target：', len(train_dataset[0][1]))
    print('测试集第1个样本可见光图像形状：', val_dataset[0][0].shape, '标注点target：', len(val_dataset[0][1]))
