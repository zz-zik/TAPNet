import argparse
import os
import numpy as np
import torch
import datetime
import logging
import pprint
import random
import time
import warnings
from tensorboardX import SummaryWriter
from configs import get_args_json, get_args_parser
from crowd_datasets import build_dataset
from engines import train_one, train_two, evaluate_one_crowd_counting, evaluate_two_crowd_counting
from models import build_model
from util.logger import setup_logger, get_environment_info, get_output_dir
from util.misc import get_rank

warnings.filterwarnings('ignore')


def get_args_config():
    parser = argparse.ArgumentParser('TAPNet')
    parser.add_argument('-c', '--config', type=str, default="", help='The path of config file')
    args = parser.parse_args()
    config = args.config
    if config != "":
        cfg = get_args_json(config)
        print(f"Load parameters from JSON file: {config}")
    else:
        parsers = get_args_parser()
        cfg = parsers.parse_args()
        print("Load parameters from command parser")
    return cfg


def main():
    cfg = get_args_config()
    out_dir, log_dir, ckpt_dir, tb_dir, cfg.count_dir, cfg.vis_dir = get_output_dir(cfg)
    logger = setup_logger('TAPNet', log_dir, 0)
    logger.info('Eval Log %s' % time.strftime("%c"))
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(cfg.gpu_id)
    env_info = get_environment_info()
    logger.info(env_info)
    if cfg.frozen_weights is not None:
        assert cfg.masks, "Frozen training is meant for segmentation only"
    # backup the arguments
    logger.info('Running with config:')
    logger.info(pprint.pformat(vars(cfg)))
    device = torch.device('cuda')
    # fix the seed for reproducibility
    seed = cfg.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    logger.info('------------------------ model params ------------------------')
    model, criterion = build_model(cfg, training=True)
    # move to GPU
    model.to(device)
    if isinstance(criterion, tuple) and len(criterion) == 2:
        for loss in criterion:
            loss.to(device)
    else:
        criterion.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params: %d', n_parameters)
    # 对模型的不同部分使用不同的优化参数
    param_dicts = [{
        "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad],
        "lr": cfg.lr
    },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": cfg.lr_backbone,
        },
    ]
    # TODO:Adam is used by default
    optimizer = torch.optim.Adam(param_dicts, lr=cfg.lr)
    # optimizer = torch.optim.AdamW(param_dicts, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.aug_dict['patch']:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.lr_drop)
    else:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                                                                  verbose=True)
    # 用于训练的采样器
    data_loader_train, data_loader_val = build_dataset(cfg=cfg)

    if cfg.frozen_weights is not None:
        checkpoint = torch.load(cfg.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])
    # 如果存在，则恢复权重和训练状态
    if cfg.resume:
        # 加载之前训过的模型的参数文件
        logger.info('------------------------ Continue training ------------------------')
        logging.warning(f"loading from {cfg.resume}")
        checkpoint = torch.load(cfg.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not cfg.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            cfg.start_epoch = checkpoint['epoch'] + 1

    logger.info("------------------------ Start training ------------------------")
    start_time = time.time()
    # 保存训练期间的f1_8、mae和mse
    mae = []
    mse = []
    f1_8 = []
    # 创建tensorboard
    writer = SummaryWriter(tb_dir)

    step = 0
    # 开始训练
    for epoch in range(cfg.start_epoch, cfg.epochs):
        t1 = time.time()
        if cfg.modal == 'RT':
            stat, metric_logger = train_two(cfg, model, criterion, data_loader_train, optimizer, device, epoch)
        else:
            stat, metric_logger = train_one(cfg, model, criterion, data_loader_train, optimizer, device, epoch)
        # stat, metric_logger = train_one_epoch(cfg, model, criterion, data_loader_train, optimizer, device, epoch)
        time.sleep(1)  # 避免tensorboard卡顿
        t2 = time.time()
        # 记录训练损失
        # print("[ep %d][lr %.7f][%.2fs]" % (epoch, optimizer.param_groups[0]['lr'], t2 - t1), "Averaged stats:", metric_logger)
        if writer is not None:
            logger.info("[ep %d][lr %.7f][%.2fs] Averaged stats: %s", epoch, optimizer.param_groups[0]['lr'], t2 - t1,
                        metric_logger)
            writer.add_scalar('loss/loss', stat['loss'], epoch)
            writer.add_scalar('loss/loss_ce', stat['loss_ce'], epoch)

        # 根据调度改变 lr
        lr_scheduler.step(stat['loss'])
        # 每隔一纪元保存最新权重
        checkpoint_latest_path = os.path.join(ckpt_dir, 'latest.pth')
        torch.save({
            'epoch': epoch,
            'step': step,
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'loss': stat['loss'],
            'loss_ce': stat['loss_ce'],
        }, checkpoint_latest_path)
        # 开始评估
        if epoch % cfg.eval_freq == 0 and epoch >= cfg.start_eval:
            t1 = time.time()
            if cfg.modal == 'RT':
                result = evaluate_two_crowd_counting(cfg, model, data_loader_val, device, epoch)
            else:
                result = evaluate_one_crowd_counting(cfg, model, data_loader_val, device, epoch)
            # result = evaluate_crowd_counting(cfg, model, data_loader_val, device)
            t2 = time.time()

            mae.append(result[0])
            mse.append(result[1])
            f1_8.append(result[2]['f1_8'])
            fps = len(data_loader_val.dataset) / (t2 - t1)
            # document the results of the assessment
            if not cfg.f1_score:
                # print("[ep %d][%.3fs][%.5ffps] mae: %.4f, mse: %.4f, best mae: %.4f" % (epoch, t2 - t1, fps, result[0], result[1], np.min(mae)) ,'\n')
                logger.info("[ep %d][%.3fs][%.5ffps] mae: %.4f, mse: %.4f, ---- @best mae: %.4f, @best mse: %.4f" % \
                            (epoch, t2 - t1, fps, result[0], result[1], np.min(mae), np.min(mse)))
            else:
                logger.info(
                    "[ep %d][%.3fs][%.5ffps] mae: %.4f, mse: %.4f, ap_4: %.4f, ar_4: %.4f, f1_4: %.4f, ap_8: %.4f, ar_8: %.4f, f1_8: %.4f, ---- @best mae: %.4f, @best mse: %.4f" % \
                    (epoch, t2 - t1, fps, result[0], result[1], result[2]['ap_4'], result[2]['ar_4'], result[2]['f1_4'],
                     result[2]['ap_8'], result[2]['ar_4'], result[2]['f1_8'], np.min(mae), np.min(mse)))
            print('')
            # recored the evaluation results
            if writer is not None:
                # logger.info("metric/mae@[ep %d]: %s ,metric/mse: %s", epoch, result[0], result[1])
                writer.add_scalar('metric/mae', result[0], step)
                writer.add_scalar('metric/mse', result[1], step)
                step += 1

            # 从一开始就保存最好的模型
            if abs(np.min(mae) - result[0]) < 0.01:
                checkpoint_best_path = os.path.join(ckpt_dir, 'best_mae.pth')
                torch.save({
                    'epoch': epoch,
                    'step': step,
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'loss': stat['loss'],
                    'loss_ce': stat['loss_ce'],
                }, checkpoint_best_path)
    # 培训总时间
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Summary of results')
    logger.info(env_info)
    logger.info('Training time {}'.format(total_time_str))
    # 打印最好的f1_score和mae与mse
    logger.info("Best f1_8: %.4f, mae: %.4f, Best mse: %.4f" % (np.max(f1_8), np.min(mae), np.min(mse)))
    logger.info('Results saved to {}'.format(cfg.output_dir))


if __name__ == '__main__':
    main()
