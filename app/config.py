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
    APP_VERSION: str = "2.0.0"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = True

    # ============ 服务器配置 ============
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True

    # ============ CORS配置 ============
    CORS_ORIGINS: list = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000"
    ]

    # ============ 文件上传配置 ============
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024
    ALLOWED_EXTENSIONS: set = {".jpg", ".jpeg", ".png", ".dcm"}

    # ============ 模型路径配置 ============
    MODEL_PATH: str = "final_global_model.pth"
    PATHOLOGY_MODEL_PATH: str = "pathology_model.pt"

    PATHOLOGY_DEFAULT_TOP_N: int = 5
    PATHOLOGY_DEFAULT_THRESHOLD: float = 0.3
    PATHOLOGY_MAX_TOP_N: int = 14

    # ============ Colab API 配置 ============
    COLAB_API_URL: str = "https://electrophoretic-garnet-bouncily.ngrok-free.dev/predict"

    # ============ LLAVA-7B API 配置 ============
    LLAVA_7B_API_URL: str = "https://outwardly-electromotive-lady.ngrok-free.dev/inference"

    # ============ 第三方API配置 ============
    THIRD_PARTY_API_URL: Optional[str] = None
    THIRD_PARTY_API_KEY: Optional[str] = None
    API_TIMEOUT: int = 120

    # ============ 日志配置 ============
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"

    # ============ 静态资源配置 ============
    STATIC_DIR: str = "static"
    HEATMAP_DIR: str = "static/heatmaps"
    UPLOAD_STATIC_DIR: str = "static/uploads"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()


def init_directories():
    """初始化必要的目录结构"""
    directories = [
        settings.UPLOAD_DIR,
        settings.STATIC_DIR,
        settings.HEATMAP_DIR,
        settings.UPLOAD_STATIC_DIR,
        "logs",
        "data"
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ 目录创建/检查完成: {directory}")
    print("✅ 所有必要目录初始化完成")


def check_models():
    """启动时验证模型文件"""
    models_status = {
        "热力图模型 (CAM)": {"path": settings.MODEL_PATH, "exists": os.path.exists(settings.MODEL_PATH)},
        "病症标签模型 (DenseNet121)": {"path": settings.PATHOLOGY_MODEL_PATH, "exists": os.path.exists(settings.PATHOLOGY_MODEL_PATH)}
    }

    all_exists = True
    for model_name, info in models_status.items():
        if not info["exists"]:
            all_exists = False
            print(f"⚠️ {model_name} 缺失: {info['path']}")
    return all_exists


def print_startup_info():
    print(f"🚀 {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"服务地址: http://{settings.HOST}:{settings.PORT}")


def init_all():
    print("\n🔧 开始初始化...")
    init_directories()
    models_ok = check_models()
    print_startup_info()
    if not models_ok:
        print("⚠️ 部分模型文件缺失，请检查模型路径")


if __name__ == "__main__":
    init_all()
