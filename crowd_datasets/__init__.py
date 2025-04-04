# build dataset according to given 'dataset_file'
import torch
from util.misc import collate_fn_crowd, collate_fn_crowds
from torch.utils.data import DataLoader


def build_dataset(cfg):
    train_set, val_set = None, None
    if cfg.dataset == 'SHHA':  # SHHA数据集
        if cfg.modal == 'R':
            from crowd_datasets.SHHA.loading_data import loading_data
            train_set, val_set = loading_data(cfg.data_root, cfg.modal, cfg.aug_dict)
        else:
            print(f'{cfg.dataset} not {cfg.modal}')
    elif cfg.dataset == 'GAIIC2024':  # GAIIC2024数据集
        from crowd_datasets.GAIIC.loading_data import loading_data
        train_set, val_set = loading_data(cfg.data_root, cfg.modal, cfg.aug_dict)
    elif cfg.dataset == 'DroneRGBT':  # Drone数据集
        from crowd_datasets.Drone.loading_data import loading_data
        train_set, val_set = loading_data(cfg.data_root, cfg.modal, cfg.aug_dict)
    else:
        print(f'{cfg.dataset} not found')
    sampler_train = torch.utils.data.RandomSampler(train_set)  # 随机采样
    sampler_val = torch.utils.data.SequentialSampler(val_set)  # 顺序采样
    # print(f"cfg.batch_size = {cfg.batch_size}")
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, cfg.batch_size, drop_last=True)
    # 用于训练的采样器
    if cfg.modal == 'RT':
        collate_fn = collate_fn_crowds
    else:
        collate_fn = collate_fn_crowd
    data_loader_train = DataLoader(train_set, batch_sampler=batch_sampler_train,
                                   collate_fn=collate_fn, num_workers=cfg.num_workers)
    data_loader_val = DataLoader(val_set, 1, sampler=sampler_val,
                                 collate_fn=collate_fn, num_workers=cfg.num_workers)
    print("------------------------ preprocess dataset ------------------------")
    print("# modal:", cfg.modal)
    print("# Dataset:", cfg.dataset)
    print("# Data_path:", cfg.data_root)
    print("# Data offset:", cfg.aug_dict['offset'])
    print(f"# Train {train_set.nSamples}, Val {val_set.nSamples}")
    return data_loader_train, data_loader_val


if __name__ == '__main__':
    import argparse
    from configs.config import get_args_config
    parser = argparse.ArgumentParser(description='GAIIC2024')  # 定义参数解析器
    parser.add_argument('--dataset', default="GAIIC2024")
    parser.add_argument('--data_root', default="/sxs/zhoufei/P2PNet/GAIIC2024", type=str)
    parser.add_argument('--RGB_T', default=2)
    parser.add_argument("--aug_dict", default={'flip': True, 'resizing': True, 'patch': True, 'offset': True}, type=dict,
                        help='data augmentation parameters')
    parser.add_argument('--batch_size', default=4, type=int)  # 训练的batch size
    parser.add_argument('--num_workers', default=16, type=int)  # 加载数据时的线程数
    parser.add_argument('--modal', default='RT')
    # args = get_args_config()
    cfg = parser.parse_args()
    dataset = build_dataset(cfg)
    # 打印数据集信息
    print(dataset)
