import torchvision.transforms as standard_transforms
from crowd_datasets.Drone.Drone import Drone, show_image


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def loading_data(data_root, modal, aug_dict):
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
    train_set = Drone(data_root, modal, rgb_transform=rgb_transform,
                      tir_transform=tir_transform, train=True, resizing=aug_dict['resizing'], patch=aug_dict['patch'], flip=aug_dict['flip'], offset=aug_dict['offset'])
    val_set = Drone(data_root, modal, rgb_transform=rgb_transform,
                    tir_transform=tir_transform, train=False, resizing=False, patch=False, flip=False, offset=False)
    return train_set, val_set


# 测试GAIIC_Dataset类
if __name__ == '__main__':
    data_root = '/sxs/zhoufei/P2PNet/DroneRGBT'
    aug_dict = {'flip': False, 'resizing': False, 'patch': False, 'offset': False}
    data_loader_train, data_loader_val = loading_data(data_root, 'RT', aug_dict)
    print(f"# Train {data_loader_train.nSamples}, Val {data_loader_val.nSamples}")
    batch_train = next(iter(data_loader_train))
    img_rgb, img_tir, target_rgb, target_tir = batch_train
    print(f"train图像RGB shape: {img_rgb.shape}, 图像TIR shape: {img_tir.shape}, target 长度: ", len(target_rgb))
    batch_val = next(iter(data_loader_val))
    img_rgb, img_tir, target_rgb, target_tir = batch_val
    print(f"val图像RGB shape: {img_rgb.shape}, 图像TIR shape: {img_tir.shape}, target 长度: ", len(target_rgb))
    # 显示训练集第1个样本可见光图像和标注点
    img_rgb, img_tir, target_rgb, target_tir = batch_train[0][0], batch_train[1][0], batch_train[2][0], batch_train[3][0]
    show_image(img_rgb, img_tir, target_rgb['point'], target_tir['point'])

'''
    for j in range(len(target)):
        print('points:{} rgb_id:{} tir_id:{} labels:{}'.format(target[j]['point'], target[j]['rgb_id'],
                                                               target[j]['tir_id'], target[j]['labels']), '\n')
        # 打印target字典中每个键值对的 shape
        print('point shape:', target[j]['point'].shape, 'rgb_id shape:', target[j]['rgb_id'].shape,
              'tir_id shape:', target[j]['tir_id'].shape, 'labels shape:', target[j]['labels'].shape)
        break
'''
