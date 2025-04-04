# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
Mostly copy-paste from DETR (https://github.com/facebookresearch/detr).
"""
import argparse

import math
import os
import sys
from typing import Iterable
import cv2
import numpy as np
import torch
import torchvision.transforms as standard_transforms
from matplotlib import pyplot as plt
from tqdm import tqdm
import util.misc as utils


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def vis(samples, targets, pred, img_type, vis_dir, des=None, save_epochs=None, epoch=None, save_img_indices=None):
    """
    samples -> tensor: [batch, 3, H, W]
    targets -> list of dict: [{'points':[], 'image_id': str}]
    pred -> list: [num_preds, 2]
    des -> str: description of the current epoch
    save_epochs -> list of epochs to save images
    epoch -> int: current epoch
    save_img_indices -> list of indices to save images for each epoch
    """
    # Only proceed if we're in a specified epoch
    if save_epochs and epoch not in save_epochs:
        return

    gts = [t['point'].tolist() for t in targets]
    pil_to_tensor = standard_transforms.ToTensor()
    if img_type == 'R':
        transform = standard_transforms.Compose([
            DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            standard_transforms.ToPILImage()
        ])
    if img_type == 'T':
        transform = standard_transforms.Compose([
            DeNormalize(mean=[0.492, 0.168, 0.430], std=[0.317, 0.174, 0.191]),
            standard_transforms.ToPILImage()
        ])

    # draw one by one
    for idx in range(samples.shape[0]):
        # Check if we should save this particular image
        if save_img_indices and idx not in save_img_indices:
            continue

        sample = transform(samples[idx])
        sample = pil_to_tensor(sample.convert('RGB')).numpy() * 255
        sample_gt = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()
        sample_pred = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()

        size = 2
        # Draw ground truths and predictions
        for t in gts[idx]:
            sample_gt = cv2.circle(sample_gt, (int(t[0]), int(t[1])), size, (0, 255, 0), -1)
        for p in pred[idx]:
            sample_pred = cv2.circle(sample_pred, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)

        name_rgb = targets[idx]['image_id']
        vis_subdir = 'rgb' if img_type == 'rgb' else 'tir'
        vis_subdir = os.path.join(vis_dir, vis_subdir)
        os.makedirs(vis_subdir, exist_ok=True)

        if des is not None:
            cv2.imwrite(
                os.path.join(vis_subdir, f"{int(name_rgb)}_{des}_gt_{len(gts[idx])}_pred_{len(pred[idx])}_gt.jpg"),
                sample_gt)
            cv2.imwrite(
                os.path.join(vis_subdir, f"{int(name_rgb)}_{des}_gt_{len(gts[idx])}_pred_{len(pred[idx])}_pred.jpg"),
                sample_pred)
        else:
            cv2.imwrite(os.path.join(vis_subdir, f"{int(name_rgb)}_gt_{len(gts[idx])}_pred_{len(pred[idx])}_gt.jpg"),
                        sample_gt)
            cv2.imwrite(os.path.join(vis_subdir, f"{int(name_rgb)}_gt_{len(gts[idx])}_pred_{len(pred[idx])}_pred.jpg"),
                        sample_pred)


# 单光训练
def train_one(cfg: argparse.Namespace, model: torch.nn.Module, criterion: torch.nn.Module,
              data_loader: Iterable, optimizer: torch.optim.Optimizer,
              device: torch.device, epoch: int):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # 遍历所有训练样本
    for samples, targets in tqdm(data_loader):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # forward
        outputs = model(samples)
        # 计算损失
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce all losses
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        # backward
        optimizer.zero_grad()
        losses.backward()
        if cfg.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_max_norm)
        optimizer.step()
        # update logger
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    metric_logger.synchronize_between_processes()  # gather the stats from all processes
    # print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, metric_logger


# 双光训练
def train_two(cfg: argparse.Namespace, model: torch.nn.Module, criterion: torch.nn.Module,
              data_loader: Iterable, optimizer: torch.optim.Optimizer,
              device: torch.device, epoch: int):
    model.train()
    if isinstance(criterion, tuple) and len(criterion) == 2:
        criterion_ahead, criterion_crowd = criterion
        criterion_ahead.train()
    else:
        criterion_ahead = None
        criterion_crowd = criterion
    criterion_crowd.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # 遍历所有训练样本
    for (samples_rgb, samples_tir, targets, _) in tqdm(data_loader):
        samples_rgb = samples_rgb.to(device)
        samples_tir = samples_tir.to(device)
        # print(f"samples_rgb: {samples_rgb.shape}, samples_tir: {samples_tir.shape}")
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # forward
        if cfg.fusion_type == 'pixel':
            outputs, outputs_ahead = model(samples_rgb, samples_tir)
            # 计算ahead损失
            loss_ahead = criterion_ahead(outputs_ahead, samples_rgb, samples_tir)
            # 计算回归分类损失
            loss_dict = criterion_crowd(outputs, targets)
            weight_dict = criterion_crowd.weight_dict
            losses_points = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            # 计算总损失
            losses = losses_points + loss_ahead
        elif cfg.fusion_type == 'feature':
            outputs = model(samples_rgb, samples_tir)
            # 计算损失
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce all losses
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        # backward
        optimizer.zero_grad()
        losses.backward()
        if cfg.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_max_norm)
        optimizer.step()
        # update logger
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    metric_logger.synchronize_between_processes()  # gather the stats from all processes
    # print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, metric_logger


@torch.no_grad()
def evaluate_one_crowd_counting(cfg: argparse.Namespace, model: torch.nn.Module, data_loader: Iterable,
                                device: torch.device, epoch: int):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # 对所有图像进行推理，计算 MAE
    maes = []
    mses = []
    nMAE = 0
    intervals = {}
    tp_sum_4 = 0
    gt_sum = 0
    et_sum = 0
    tp_sum_8 = 0
    counts_pred, counts_true = [], []  # 存储预测人数和真实人数
    img_id = []  # 创建储存图片id的列表
    for samples, targets in tqdm(data_loader):
        samples = samples.to(device)

        outputs = model(samples)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
        outputs_points = outputs['pred_points'][0]
        gt_cnt = targets[0]['point'].shape[0]

        threshold = cfg.threshold

        points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
        predict_cnt = int((outputs_scores > threshold).sum())
        # if specified, save the visualized images
        # TODO: 可以保存可视化的图片
        if cfg.vis_dir is not None:
            vis(samples, targets, [points], cfg.modal, cfg.vis_dir, des=epoch, save_epochs=[100, 200], epoch=epoch, save_img_indices=[500, 900])
        # accumulate MAE, MSE
        mae = abs(predict_cnt - gt_cnt)
        mse = (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)
        maes.append(float(mae))
        mses.append(float(mse))

        # 存储预测人数和真实人数
        counts_pred.append(predict_cnt)
        counts_true.append(gt_cnt)
        img_id.append(int(targets[0]['image_id']))

        # nMAE += mae/gt_cnt
        interval = int(gt_cnt / 250)
        if interval not in intervals:
            intervals[interval] = [mae / gt_cnt]
        else:
            intervals[interval].append(mae / gt_cnt)

        tp_4 = utils.compute_tp(points, targets[0]['point'], 4)
        tp_8 = utils.compute_tp(points, targets[0]['point'], 8)
        tp_sum_4 += tp_4
        gt_sum += gt_cnt
        et_sum += predict_cnt
        tp_sum_8 += tp_8
    # calc MAE, MSE
    mae = np.mean(maes)
    mse = np.sqrt(np.mean(mses))

    # nMAE /= len(data_loader)
    ap_4 = tp_sum_4 / float(et_sum + 1e-10)
    ar_4 = tp_sum_4 / float(gt_sum + 1e-10)
    f1_4 = 2 * ap_4 * ar_4 / (ap_4 + ar_4 + 1e-10)
    ap_8 = tp_sum_8 / float(et_sum + 1e-10)
    ar_8 = tp_sum_8 / float(gt_sum + 1e-10)
    f1_8 = 2 * ap_8 * ar_8 / (ap_8 + ar_8 + 1e-10)
    local_result = {'ap_4': ap_4, 'ar_4': ar_4, 'f1_4': f1_4, 'ap_8': ap_8, 'ar_8': ar_8, 'f1_8': f1_8}
    # 保存计数结果
    save_counts_to_file_sorted(f"{cfg.count_dir}/counting_{mae:.2f}.txt", img_id, counts_pred, counts_true)
    return (mae, mse) if not cfg.f1_score else (mae, mse, local_result)


@torch.no_grad()
def evaluate_two_crowd_counting(cfg: argparse.Namespace, model: torch.nn.Module, data_loader: Iterable,
                                device: torch.device, epoch: int):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # 对所有图像进行推理，计算 MAE
    maes = []
    mses = []
    nMAE = 0
    intervals = {}
    tp_sum_4 = 0
    gt_sum = 0
    et_sum = 0
    tp_sum_8 = 0
    counts_pred, counts_true = [], []  # 存储预测人数和真实人数
    img_id = []  # 创建储存图片id的列表
    for (samples_rgb, samples_tir, targets_rgb, targets_tir) in tqdm(data_loader):
        samples_rgb = samples_rgb.to(device)
        samples_tir = samples_tir.to(device)

        outputs = model(samples_rgb, samples_tir)[0] if cfg.fusion_type == 'pixel' else model(samples_rgb, samples_tir)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

        outputs_points = outputs['pred_points'][0]
        gt_cnt = (targets_rgb[0]['point'].shape[0] + targets_tir[0]['point'].shape[0]) / 2

        threshold = cfg.threshold

        points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
        predict_cnt = int((outputs_scores > threshold).sum())
        # if specified, save the visualized images
        # TODO: 可以保存可视化的图片
        if cfg.vis_dir is not None:
            vis(samples_rgb, targets_rgb, [points], 'R', cfg.vis_dir, des=epoch, save_epochs=[100, 200], epoch=epoch, save_img_indices=[500, 900])
            vis(samples_tir, targets_tir, [points], 'T', cfg.vis_dir, des=epoch, save_epochs=[100, 200], epoch=epoch, save_img_indices=[500, 900])
        # accumulate MAE, MSE
        mae = abs(predict_cnt - gt_cnt)
        mse = (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)
        maes.append(float(mae))
        mses.append(float(mse))

        # 存储预测人数和真实人数
        counts_pred.append(predict_cnt)
        counts_true.append(gt_cnt)
        img_id.append(int(targets_rgb[0]['image_id']))

        # nMAE += mae/gt_cnt
        interval = int(gt_cnt / 250)
        if interval not in intervals:
            intervals[interval] = [mae / gt_cnt]
        else:
            intervals[interval].append(mae / gt_cnt)

        # tp_4 = (utils.compute_tp(points, targets_rgb[0]['point'], 4) + utils.compute_tp(points, targets_tir[0]['point'], 4)) / 2
        # tp_8 = (utils.compute_tp(points, targets_rgb[0]['point'], 8) + utils.compute_tp(points, targets_tir[0]['point'], 8)) / 2
        tp_4 = utils.compute_tp(points, targets_rgb[0]['point'], 4)
        tp_8 = utils.compute_tp(points, targets_rgb[0]['point'], 8)
        tp_sum_4 += tp_4
        gt_sum += gt_cnt
        et_sum += predict_cnt
        tp_sum_8 += tp_8
    # calc MAE, MSE
    mae = np.mean(maes)
    mse = np.sqrt(np.mean(mses))

    # nMAE /= len(data_loader)
    ap_4 = tp_sum_4 / float(et_sum + 1e-10)
    ar_4 = tp_sum_4 / float(gt_sum + 1e-10)
    f1_4 = 2 * ap_4 * ar_4 / (ap_4 + ar_4 + 1e-10)
    ap_8 = tp_sum_8 / float(et_sum + 1e-10)
    ar_8 = tp_sum_8 / float(gt_sum + 1e-10)
    f1_8 = 2 * ap_8 * ar_8 / (ap_8 + ar_8 + 1e-10)
    local_result = {'ap_4': ap_4, 'ar_4': ar_4, 'f1_4': f1_4, 'ap_8': ap_8, 'ar_8': ar_8, 'f1_8': f1_8}
    # 保存计数结果
    save_counts_to_file_sorted(f"{cfg.count_dir}/counting_{mae:.2f}.txt", img_id, counts_pred, counts_true)
    return (mae, mse) if not cfg.f1_score else (mae, mse, local_result)


def save_counts_to_file_sorted(filename, img_ids, counts, counts_true=None):
    """
    保存人群计数到文件，并根据图片ID排序。
    :param filename: 要写入的文件名。
    :param img_ids: 图片ID列表。
    :param counts: 预测计数或计数列表。
    :param counts_true: 真实计数列表。
    """
    # 确保输入的计数列表长度相同
    assert len(img_ids) == len(counts), "img_ids and counts must have the same length"
    if counts_true is not None:
        assert len(img_ids) == len(counts_true), "img_ids and counts_true must have the same length"

    # 创建一个字典来关联图片ID和对应的计数
    count_dict = {img_id: count for img_id, count in zip(img_ids, counts)}
    if counts_true is not None:
        true_count_dict = {img_id: true_count for img_id, true_count in zip(img_ids, counts_true)}

    # 根据图片ID对字典进行排序
    sorted_img_ids = sorted(count_dict.keys())
    if counts_true is not None:
        sorted_true_counts = {k: true_count_dict[k] for k in sorted_img_ids}

    with open(filename, 'w') as f:
        for img_id in sorted_img_ids:
            img_name = f"{img_id}.jpg"  # 假设图片名是ID加上".jpg"
            formatted_count = f"{count_dict[img_id]:.2f}"
            if counts_true is not None:
                formatted_true_count = f"{sorted_true_counts[img_id]:.2f}"
                f.write(f"{os.path.splitext(img_name)[0]}, {formatted_count}, {formatted_true_count}\n")
            else:
                f.write(f"{os.path.splitext(img_name)[0]}, {formatted_count}\n")


def draw_results(local_result, count_dir):
    """"""
    metrics = ['ap_4', 'ar_4', 'f1_4', 'ap_8', 'ar_8', 'f1_8']
    values = [local_result[metric] for metric in metrics]

    plt.figure(figsize=(10, 6))
    plt.bar(metrics, values, color='blue')
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Crowd Counting and Localization Metrics')
    plt.ylim(0, 1.2)
    plt.savefig(f"{count_dir}/local_results.png")
    plt.close()
