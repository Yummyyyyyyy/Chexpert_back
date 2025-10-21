"""
日志工具函数
统一日志格式和处理
"""
from loguru import logger
import sys
from app.config import settings


def setup_logger():
    """
    配置日志系统

    【TODO】后端团队成员可选优化:
    1. 添加日志分级存储（ERROR单独存储）
    2. 添加日志轮转策略
    3. 集成第三方日志服务（如Sentry）
    """
    # 移除默认logger
    logger.remove()

    # 控制台输出
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.LOG_LEVEL
    )

    # 文件输出
    logger.add(
        settings.LOG_FILE,
        rotation="500 MB",  # 文件大小轮转
        retention="10 days",  # 保留10天
        compression="zip",  # 压缩旧日志
        level=settings.LOG_LEVEL,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )

    return logger
