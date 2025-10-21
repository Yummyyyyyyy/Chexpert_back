@echo off
REM CheXpert Backend 启动脚本 (Windows)

echo ======================================
echo   CheXpert Backend API 启动脚本
echo ======================================

REM 检查虚拟环境
if not exist "venv" (
    echo 未找到虚拟环境，正在创建...
    python -m venv venv
)

REM 激活虚拟环境
echo 激活虚拟环境...
call venv\Scripts\activate.bat

REM 安装依赖
echo 检查并安装依赖...
pip install -r requirements.txt

REM 创建必要的目录
echo 创建必要的目录...
if not exist "uploads" mkdir uploads
if not exist "logs" mkdir logs
if not exist "app\models\base_models" mkdir app\models\base_models
if not exist "app\models\finetuned" mkdir app\models\finetuned

REM 检查.env文件
if not exist ".env" (
    echo 警告: .env文件不存在，正在从.env.example复制...
    copy .env.example .env
    echo 请编辑.env文件填写实际配置！
)

REM 启动服务
echo ======================================
echo 启动FastAPI服务...
echo API文档: http://localhost:8000/docs
echo ======================================

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

pause
