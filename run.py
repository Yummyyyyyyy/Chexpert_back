"""
启动脚本 - 解决模块导入问题
运行: python run.py
"""
import sys
import os

# 将项目根目录添加到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入并运行应用
if __name__ == "__main__":
    import uvicorn
    from app.config import settings

    print("=" * 60)
    print(f"  启动 {settings.APP_NAME}")
    print("=" * 60)
    print(f"📝 API文档: http://{settings.HOST}:{settings.PORT}/docs")
    print(f"🔍 健康检查: http://{settings.HOST}:{settings.PORT}/health")
    print("=" * 60)

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL.lower()
    )
