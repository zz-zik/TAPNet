import torchvision.transforms as standard_transforms
# from .SHHA import SHHA
from crowd_datasets.SHHA.SHHA import SHHA


# 去正则化用于获取原始图像


class DeNormalize(object):
    """
    在给定平均值和 std 的情况下，对图像进行去正态化处理。
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def loading_data(data_root, modal, aug_dict):
    # 预处理转换
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
    ])
    # 创建训练数据集
    train_set = SHHA(data_root, transform=transform, train=True, resizing=aug_dict['resizing'], patch=aug_dict['patch'], flip=aug_dict['flip'])
    # 创建验证数据集
    val_set = SHHA(data_root, transform=transform, train=False, resizing=False, patch=False, flip=False)

    return train_set, val_set


if __name__ == '__main__':
    data_root = '/sxs/zhoufei/P2PNet/ShanghaiTech/part_A'
    aug_dict = {'resizing': True, 'patch': True, 'flip': True}
    data_loader_train, data_loader_val = loading_data(data_root, aug_dict)
    batch_train = next(iter(data_loader_train))
    train_img, train_target = batch_train
    batch_val = next(iter(data_loader_val))
    val_img, val_target = batch_val
    print(train_img[0].shape, train_img[1].shape, train_img[2].shape)
    print("train img shape:", train_img.shape, "target len:", len(train_target))
    print("val img shape:", val_img.shape, "target len:", len(val_target))