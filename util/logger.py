# -*- coding: utf-8 -*-
"""
@Project : TFPNet4-DAF
@FileName: logger.py
@Time    : 2024/10/27 上午11:21
@Author  : ZhouFei
@Email   : zhoufei.net@outlook.com
@Desc    : 
"""

import logging
import os
import sys
import pytz
import torch
from datetime import datetime


def get_beijing_timestamp():
    """Get Beijing time timestamp in the format of YYYY_MM_DD_HH_MM."""
    beijing_tz = pytz.timezone('Asia/Shanghai')
    now = datetime.now(beijing_tz)
    return now.strftime("%Y_%m_%d_%H_%M")


def setup_logger(log_name, log_dir, distributed_rank, train=True):
    """logging setup
    Args:
        log_name: Log file header name
        log_dir: Log file save directory
        distributed_rank: The rank of the process in distributed training.
    Returns:
        logger: A logger object.
    """
    logger = logging.getLogger()
    if logger.hasHandlers():  # 检查是否已经存在处理程序
        logger.handlers.clear()  # 清空已有的处理程序
    logger.setLevel(logging.DEBUG)
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)  # CRITICAL 是默认的日志级别
    formatter = logging.Formatter("%(message)s")  # 控制台打印
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)  # Create a directory (if it does not exist)
        # 使用当前时间戳命名日志文件
        timestamp = get_beijing_timestamp()
        fh = logging.FileHandler(os.path.join(log_dir, f"train_{timestamp}.log" if train else f"eval_{timestamp}.log"),
                                 mode='w')
        fh.setLevel(logging.DEBUG)
        # 设置日志格式，包括北京时间的时间戳
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


# 用于创建各种输出路径
def get_output_dir(cfg):
    """Create output directory if it does not exist.
    Args:
        cfg: Config object.
    Returns:
        output_dir: Output directory path.
    """
    # 创建当前时间戳文件名
    timestamp = get_beijing_timestamp()
    out_dir = os.path.join(str(cfg.output_dir), str(cfg.dataset), str(timestamp))
    log_dir = os.path.join(out_dir, 'log')
    ckpt_dir = os.path.join(out_dir, 'ckpt')
    tb_dir = os.path.join(out_dir, 'tb')
    count_dir = os.path.join(out_dir, 'count')
    vis_dir = os.path.join(out_dir, 'vis') if cfg.vis_dir else None

    # 确保文件不为空
    for dirs in [d for d in [out_dir, log_dir, ckpt_dir, tb_dir, count_dir, vis_dir] if d is not None]:
        if dirs:
            os.makedirs(dirs, exist_ok=True)

    return out_dir, log_dir, ckpt_dir, tb_dir, count_dir, vis_dir


def get_environment_info():
    info = (
        f"Environment info 🚀: "
        f"Python-{sys.version.split()[0]} torch-{torch.__version__} "
        f"CUDA:{torch.cuda.current_device()} "
        f"({torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)}MiB)"
    )
    return info
