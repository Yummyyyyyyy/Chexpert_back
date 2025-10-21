"""
API路由汇总
所有业务接口都在这里注册
"""
from fastapi import APIRouter

from app.api.v1.endpoints import upload, llava_report, llava7b_report, third_party


# 创建v1版本的总路由
api_router = APIRouter()

# 注册各个功能模块的路由
api_router.include_router(
    upload.router,
    prefix="/image",
    tags=["功能1: 图像分析"]
)

api_router.include_router(
    llava_report.router,
    prefix="/report",
    tags=["功能2: 报告生成"]
)

api_router.include_router(
    llava7b_report.router,
    prefix="/report",
    tags=["功能2扩展: LLAVA-7B报告生成"]
)

api_router.include_router(
    third_party.router,
    prefix="/knowledge",
    tags=["功能3: 知识图谱"]
)
