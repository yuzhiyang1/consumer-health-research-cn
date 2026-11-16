#!/bin/bash

conda_setup() {
    echo "Setting up pha_framework environment using conda..."
    
    # 删除已有的 pha_framework 环境(如果存在)
    conda env remove -n pha_framework -y || echo "No existing pha_framework environment found"
    
    # 创建和激活初始 pha_framework 环境，并使用 Python 3.12
    conda create -n pha_framework python=3.12 pip -y || exit 1
    
    # 源 conda,以确保在脚本中‘conda激活'的作品
    source "$(conda info --base)/etc/profile.d/conda.sh" || exit 1
    conda activate pha_framework || exit 1
    
    # 校验 pip 和用于调试的蟒蛇信息
    echo "Using pip from: $(which pip)"
    echo "Python version: $(python --version)"
    
    # 从 requirements.txt 安装标准包
    echo "Installing dependencies from requirements.txt..."
    python -m pip install -r requirements.txt || exit 1
    
    # 从 Google DeepMind 官方仓库安装 OneTwo。
    # 使用 --no-deps 保留已安装的依赖版本，例如 numpy。
    echo "Installing OneTwo..."
    python -m pip install --no-deps git+https://github.com/google-deepmind/onetwo || exit 1
    
    # 用 Jupyter 注册内核
    echo "Registering Jupyter kernel..."
    python -m ipykernel install --user --name pha_framework --display-name "pha_framework (Python 3.12)" || exit 1
    
    echo "----------------------------------------------------------------"
    echo "Setup complete! Environment 'pha_framework' is ready."
    echo "To start using it, run: conda activate pha_framework"
    echo "----------------------------------------------------------------"
}

conda_setup
