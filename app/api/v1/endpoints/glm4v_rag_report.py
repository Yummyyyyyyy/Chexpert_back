"""
功能2扩展: 基于 GLM-4V + RAG 生成医学报告（模板）
"""
from fastapi import APIRouter, HTTPException, status
from loguru import logger

from app.models.schemas import Glm4vRagReportRequest, ReportResponse
from app.services.glm4v_rag_service import get_glm4v_rag_service


router = APIRouter()


@router.post("/generate-glm4v-rag", response_model=ReportResponse)
async def generate_glm4v_rag_report(request: Glm4vRagReportRequest):
    try:
        logger.info(f"📝 收到 GLM-4V + RAG 报告生成请求: {request.image_path}")

        service = get_glm4v_rag_service()
        report, processing_time = await service.generate_report(
            image_path=request.image_path,
            prompt=request.prompt,
            pathology_labels=request.pathology_labels,
            rag_query=request.rag_query,
            classifier_probs=request.classifier_probs,
            top_k=request.top_k or 5,
        )

        return ReportResponse(
            success=True,
            message="GLM-4V + RAG 报告生成成功",
            report=report,
            processing_time=processing_time,
        )

    except FileNotFoundError:
        logger.error(f"图片文件未找到: {request.image_path}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="图片文件不存在",
        )
    except Exception as e:
        logger.error(f"GLM-4V + RAG 报告生成失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"GLM-4V + RAG 报告生成失败: {str(e)}",
        )


@router.get("/test-glm4v-rag")
async def test_glm4v_rag_endpoint():
    return {
        "endpoint": "glm4v_rag_report",
        "status": "working",
        "message": "GLM-4V + RAG 报告生成接口正常运行",
    }
