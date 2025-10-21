"""
功能2: 基于自研LLaVA模型生成医学报告
"""
from fastapi import APIRouter, HTTPException, status
from loguru import logger

from app.models.schemas import ReportRequest, ReportResponse
from app.services.llava_service import get_llava_service


router = APIRouter()


# ============ API端点 ============
@router.post("/generate", response_model=ReportResponse)
async def generate_report(request: ReportRequest):
    """
    生成医学报告接口

    **功能**: 基于上传的X光片，调用自研LLaVA模型生成详细医学报告

    **前端调用示例**:
    ```javascript
    const response = await fetch('http://localhost:8000/api/v1/report/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            image_path: '/uploads/original_xxx.jpg',
            prompt: '请详细分析这张胸部X光片'
        })
    });
    const result = await response.json();
    console.log(result.report);  // 显示生成的报告
    ```

    **【TODO - 后端团队成员需要修改的部分】**:
    1. 在 `app/services/llava_service.py` 中实现真实的LLaVA模型推理
    2. 替换下方的模拟报告为实际模型输出
    3. 优化提示词工程，提高报告质量
    4. 添加报告后处理逻辑（格式化、去重等）
    """
    try:
        logger.info(f"📝 收到报告生成请求: {request.image_path}")

        # 【核心逻辑】调用LLaVA服务生成报告
        # 使用单例模式,避免重复加载模型
        llava_service = get_llava_service()
        report, processing_time = await llava_service.generate_report(
            image_path=request.image_path,
            prompt=request.prompt,
            pathology_labels=request.pathology_labels
        )

        logger.success(f"✅ 报告生成完成，耗时: {processing_time:.2f}秒")

        return ReportResponse(
            success=True,
            message="报告生成成功",
            report=report,
            processing_time=processing_time
        )

    except FileNotFoundError as e:
        logger.error(f"❌ 图片文件未找到: {request.image_path}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="图片文件不存在"
        )
    except Exception as e:
        logger.error(f"❌ 报告生成失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"报告生成失败: {str(e)}"
        )


@router.get("/test")
async def test_report_endpoint():
    """测试接口，验证路由是否正常"""
    return {
        "endpoint": "llava_report",
        "status": "working",
        "message": "报告生成接口正常运行"
    }
