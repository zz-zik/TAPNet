#!/bin/bash

# 设置日志文件路径
LOG_FILE="./install.log"

export PATH="/opt/conda/bin:$PATH"

# 指定虚拟环境名称
export CONDA_ENV_NAME="tafp"

# 切换到工作目录
echo "切换到工作目录" | tee -a "$LOG_FILE"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.." || { echo "切换工作目录失败。" | tee -a "$LOG_FILE"; exit 1; }

# 检查日志文件是否存在，不存在则创建
if [ -f "$LOG_FILE" ]; then
    > "$LOG_FILE"  # 清空文件内容
else
    touch "$LOG_FILE"  # 创建文件
fi

# 创建 Conda 虚拟环境
if conda env list | grep -qwE "^$CONDA_ENV_NAME\s+"; then
    echo "虚拟环境 '$CONDA_ENV_NAME' 已存在。" | tee -a "$LOG_FILE"
else
    echo "虚拟环境 '$CONDA_ENV_NAME' 不存在，开始创建..." | tee -a "$LOG_FILE"
    conda create -n "$CONDA_ENV_NAME" python=3.10 -y >> "$LOG_FILE" 2>&1
    if [ $? -ne 0 ]; then
        echo "虚拟环境 '$CONDA_ENV_NAME' 创建失败。" | tee -a "$LOG_FILE"
        exit 1
    else
        echo "虚拟环境 '$CONDA_ENV_NAME' 创建成功。" | tee -a "$LOG_FILE"
    fi
fi

# 激活 Conda 环境
source /opt/conda/bin/activate "$CONDA_ENV_NAME" >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then
    echo "激活 conda 环境失败: $CONDA_ENV_NAME" | tee -a "$LOG_FILE"
    exit 1
else
    echo "激活成功 conda 环境: $CONDA_ENV_NAME" | tee -a "$LOG_FILE"
fi

# 安装依赖项
if [ -f "requirements.txt" ]; then
    echo "开始安装依赖项..." | tee -a "$LOG_FILE"
    pip install -r requirements.txt >> "$LOG_FILE" 2>&1
    if [ $? -ne 0 ]; then
        echo "安装依赖项失败。" | tee -a "$LOG_FILE"
        exit 1
    else
        echo "依赖项安装完成。" | tee -a "$LOG_FILE"
    fi
else
    echo "未找到 requirements.txt 文件，跳过依赖项安装。" | tee -a "$LOG_FILE"
fi

# 安装完成
echo "安装完成！" | tee -a "$LOG_FILE"
