#!/bin/bash

# CheXpert Backend 启动脚本

echo "======================================"
echo "  CheXpert Backend API 启动脚本"
echo "======================================"

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python版本: $python_version"

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "未找到虚拟环境，正在创建..."
    python3 -m venv venv
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source venv/bin/activate

# 安装依赖
echo "检查并安装依赖..."
pip install -r requirements.txt

# 创建必要的目录
echo "创建必要的目录..."
mkdir -p uploads logs app/models/base_models app/models/finetuned

# 检查.env文件
if [ ! -f ".env" ]; then
    echo "警告: .env文件不存在，正在从.env.example复制..."
    cp .env.example .env
    echo "请编辑.env文件填写实际配置！"
fi

# 启动服务
echo "======================================"
echo "启动FastAPI服务..."
echo "API文档: http://localhost:8000/docs"
echo "======================================"

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
