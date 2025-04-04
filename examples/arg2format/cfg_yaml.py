# -*- coding: utf-8 -*-
"""
@Project : TFPNet4-DAF
@FileName: cfg_yaml.py
@Time    : 2024/10/27 上午10:27
@Author  : ZhouFei
@Email   : zhoufei.net@outlook.com
@Desc    : 将参数文件解析并保存到.yaml文件中
@Usage   :
    $ python cfg_yaml.py \
                --cfg_dir /path/to/config.py \
                --out /path/to/
"""


import argparse
import yaml
import importlib.util
import os


def get_args_from_config(cfg_path):
    spec = importlib.util.spec_from_file_location("config", cfg_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # Initialize parser and parse defaults
    parser = config.get_args_parser()
    args = parser.parse_args([])
    args_dict = vars(args)

    return args_dict


def save_to_yaml(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as file:
        yaml.dump(data, file, default_flow_style=False, allow_unicode=True)


def main(cfg_dir, out_dir):
    args_data = get_args_from_config(cfg_dir)
    out_file = os.path.join(out_dir, 'config.yaml')
    save_to_yaml(args_data, out_file)
    print(f"Configuration saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse config file and save to .yaml")
    parser.add_argument("--cfg_dir", required=True, help="Path to the config.py file")
    parser.add_argument("--out", required=True, help="Output directory for the YAML file")
    # parser.add_argument("--cfg_dir", default='/sxs/zhoufei/P2PNet/TFPNet-DAF/configs/config.py', help="Path to the config.py file")
    # parser.add_argument("--out", default='/sxs/zhoufei/P2PNet/TFPNet-DAF/configs', help="Output directory for the YAML file")
    args = parser.parse_args()
    main(args.cfg_dir, args.out)

