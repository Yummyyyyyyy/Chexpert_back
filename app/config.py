"""
配置文件 - Configuration Settings
所有配置项集中管理，方便团队成员修改
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """
    应用配置类
    使用环境变量或.env文件进行配置
    """
    
    # ============ 基础配置 ============
    APP_NAME: str = "CheXpert Backend API"
    APP_VERSION: str = "2.0.0"  # 版本升级到2.0.0
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = True  # 生产环境改为False
    
    # ============ 服务器配置 ============
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True  # 开发模式自动重载，生产环境改为False
    
    # ============ CORS配置（前后端分离必需）============
    # 【重要】团队成员需要根据前端地址修改
    CORS_ORIGINS: list = [
        "http://localhost:3000",  # 前端开发地址
        "http://localhost:5173",  # Vite默认端口
        "http://127.0.0.1:3000",
        # 【TODO】添加前端生产环境域名
        # "https://your-frontend-domain.com"
    ]
    
    # ============ 文件上传配置 ============
    UPLOAD_DIR: str = "uploads"  # 上传文件临时存储目录
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB，可根据医学图像大小调整
    ALLOWED_EXTENSIONS: set = {".jpg", ".jpeg", ".png", ".dcm"}  # 支持的图像格式
    
    # ============ 模型路径配置 ============
    # 【新增】热力图模型配置
    MODEL_PATH: str = "final_global_model.pth"  # 热力图生成模型路径
    
    # 【新增】病症标签模型配置
    PATHOLOGY_MODEL_PATH: str = "pathology_model.pt"  # 病症标签分类模型路径（DenseNet121-14类）
    
    # 【说明】
    # - MODEL_PATH: 用于生成热力图的CAM模型
    # - PATHOLOGY_MODEL_PATH: 用于14类病症标签分类的模型
    # 两个模型可以独立使用，也可以同时使用
    
    # ============ 病症标签配置 ============
    # 【新增】病症分析默认配置
    PATHOLOGY_DEFAULT_TOP_N: int = 5  # 默认返回Top5病症
    PATHOLOGY_DEFAULT_THRESHOLD: float = 0.3  # 默认概率阈值30%
    PATHOLOGY_MAX_TOP_N: int = 14  # 最多返回14类（全部）
    
    # ============ Colab API 配置 ============
    # LLaVA 模型通过 Colab 远程调用
    # COLAB_API_URL: Optional[str] = None  # Colab API 地址 (例如: https://xxxx.ngrok.io/predict)
    # 【重要】部署 Colab 后,请设置此 URL
    # 步骤:
    # 1. 在 Colab 中运行 LLaVA 推理服务
    # 2. 使用 ngrok 或 Cloudflare Tunnel 映射到公网
    # 3. 将映射后的 URL 填写到这里
    # 例如: COLAB_API_URL = "https://abc123.ngrok.io/predict"
    COLAB_API_URL: str = "https://electrophoretic-garnet-bouncily.ngrok-free.dev/predict"
    
    # ============ LLAVA-7B API 配置 ============
    # LLAVA-7B 模型通过 ngrok 远程调用
    # 参考: model_llava/deploy/test_ngork.py
    # 【重要】请设置 LLAVA-7B 模型的 ngrok URL
    # 格式: https://xxxx.ngrok-free.dev/inference
    LLAVA_7B_API_URL: str = "https://outwardly-electromotive-lady.ngrok-free.dev/inference"  # LLAVA-7B API 地址
    # 示例: LLAVA_7B_API_URL = "https://outwardly-electromotive-lady.ngrok-free.dev/inference"
    
    # ============ 第三方API配置 ============
    # 【TODO】团队成员需要添加实际的API密钥
    THIRD_PARTY_API_URL: Optional[str] = None  # 知识图谱API地址
    THIRD_PARTY_API_KEY: Optional[str] = None  # API密钥
    API_TIMEOUT: int = 120  # API调用超时时间（秒） - Colab 推理可能需要较长时间
    
    # ============ 日志配置 ============
    LOG_LEVEL: str = "INFO"  # DEBUG/INFO/WARNING/ERROR
    LOG_FILE: str = "logs/app.log"
    
    # ============ 数据库配置（可选）============
    # 如果需要存储用户数据或历史记录
    # DATABASE_URL: Optional[str] = None
    
    # ============ 静态资源配置 ============
    # 【新增】静态文件路径配置
    STATIC_DIR: str = "static"  # 静态资源根目录
    HEATMAP_DIR: str = "static/heatmaps"  # 热力图保存目录
    UPLOAD_STATIC_DIR: str = "static/uploads"  # 上传图片保存目录
    
    class Config:
        env_file = ".env"  # 从.env文件读取配置
        case_sensitive = True


# 全局配置实例
settings = Settings()


# 启动时创建必要的目录
def init_directories():
    """初始化必要的目录结构"""
    directories = [
        settings.UPLOAD_DIR,
        settings.STATIC_DIR,
        settings.HEATMAP_DIR,
        settings.UPLOAD_STATIC_DIR,
        "logs",
        "data"  # 【新增】数据目录（存储历史记录）
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ 目录创建/检查完成: {directory}")
    
    print("✅ 所有必要目录初始化完成")


def check_models():
    """
    检查模型文件是否存在
    【新增】启动时验证模型文件
    """
    models_status = {
        "热力图模型 (CAM)": {
            "path": settings.MODEL_PATH,
            "exists": os.path.exists(settings.MODEL_PATH)
        },
        "病症标签模型 (DenseNet121)": {
            "path": settings.PATHOLOGY_MODEL_PATH,
            "exists": os.path.exists(settings.PATHOLOGY_MODEL_PATH)
        }
    }
    
    print("\n" + "="*60)
    print("模型文件检查:")
    print("="*60)
    
    all_exists = True
    for model_name, info in models_status.items():
        status = "✅ 存在" if info["exists"] else "❌ 缺失"
        print(f"{model_name}:")
        print(f"  路径: {info['path']}")
        print(f"  状态: {status}")
        if not info["exists"]:
            all_exists = False
            print(f"  ⚠️  警告: 该模型功能将不可用")
        print()
    
    print("="*60)
    
    if not all_exists:
        print("⚠️  部分模型文件缺失，请按照以下步骤操作:")
        print("1. 将热力图模型文件重命名为: final_global_model.pth")
        print("2. 将病症标签模型文件重命名为: pathology_model.pt")
        print("3. 将两个模型文件放在项目根目录")
        print("4. 重启服务")
        print("="*60)
    else:
        print("✅ 所有模型文件检查通过")
        print("="*60)
    
    return all_exists


def print_startup_info():
    """
    【新增】打印启动信息
    """
    print("\n" + "="*60)
    print(f"🚀 {settings.APP_NAME} v{settings.APP_VERSION}")
    print("="*60)
    print(f"服务地址: http://{settings.HOST}:{settings.PORT}")
    print(f"API文档: http://{settings.HOST}:{settings.PORT}/docs")
    print(f"调试模式: {'开启' if settings.DEBUG else '关闭'}")
    print(f"自动重载: {'开启' if settings.RELOAD else '关闭'}")
    print("\n可用接口:")
    print("  - GET  /health                        # 健康检查")
    print("  - GET  /api/v1/history                # 历史记录")
    print("  - POST /api/v1/image/analyze          # 热力图分析")
    print("  - POST /api/v1/pathology/analyze      # 病症标签分析 【新增】")
    print("  - POST /api/v1/image/analyze-full     # 综合分析 【新增】")
    print("  - GET  /api/v1/pathology/labels       # 获取标签列表 【新增】")
    print("  - POST /api/v1/report/generate        # 生成报告")
    print("="*60 + "\n")


# 【新增】启动时的完整初始化流程
def init_all():
    """
    完整的初始化流程
    在应用启动时调用
    """
    print("\n🔧 开始初始化...")
    
    # 1. 创建目录
    init_directories()
    
    # 2. 检查模型
    models_ok = check_models()
    
    # 3. 打印启动信息
    print_startup_info()
    
    if not models_ok:
        print("⚠️  警告: 部分功能可能不可用，请检查模型文件")
    
    return models_ok


if __name__ == "__main__":
    # 测试配置
    init_all()
    print("\n配置测试:")
    print(f"APP_NAME: {settings.APP_NAME}")
    print(f"APP_VERSION: {settings.APP_VERSION}")
    print(f"MODEL_PATH: {settings.MODEL_PATH}")
    print(f"PATHOLOGY_MODEL_PATH: {settings.PATHOLOGY_MODEL_PATH}")
    print(f"CORS_ORIGINS: {settings.CORS_ORIGINS}")
