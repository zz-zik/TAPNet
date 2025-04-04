#!/bin/bash
# 禁用特定的 shellcheck 检查项
# shellcheck disable=SC2290

# 获取脚本所在的目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# 切换到项目根目录
cd "$SCRIPT_DIR/.." || exit

# 设置 PYTHON 的路径
export PATCH=/opt/conda/envs/pytorch/bin/python3.10

# 执行 Python 脚本并传入配置文件路径
"$PATCH" main.py -c configs/GAIIC2/resnet_T.json
