# -*- coding: utf-8 -*-
"""
@Project : TFPNet4-DAF
@FileName: cfg_json.py
@Time    : 2024/10/27 上午10:46
@Author  : ZhouFei
@Email   : zhoufei.net@outlook.com
@Desc    : 将参数文件解析并保存到.json文件中
@Usage   :
    $ python cfg_json.py \
                --cfg_dir /path/to/config.py \
                --out /path/to/
"""

import argparse
import json
import os
import importlib.util


def get_args_from_config(cfg_path):
    spec = importlib.util.spec_from_file_location("config", cfg_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # Initialize parser and parse defaults
    parser = config.get_args_parser()
    args = parser.parse_args([])
    args_dict = vars(args)

    return args_dict


def save_to_json(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def main(cfg_dir, out_dir):
    config_data = get_args_from_config(cfg_dir)
    out_file = os.path.join(out_dir, 'config.json')
    save_to_json(config_data, out_file)
    print(f"Configuration saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and save config parameters to .json")
    parser.add_argument("--cfg_dir", required=True, help="Path to the config.py file")
    parser.add_argument("--out", required=True, help="Output directory for the JSON file")
    # parser.add_argument("--cfg_dir", default='/sxs/zhoufei/P2PNet/TFPNet-DAF/configs/config.py', help="Path to the config.py file")
    # parser.add_argument("--out", default='/sxs/zhoufei/P2PNet/TFPNet-DAF/configs', help="Output directory for the YAML file")
    args = parser.parse_args()
    main(args.cfg_dir, args.out)
