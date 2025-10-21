"""
功能2扩展: 基于LLAVA-7B模型生成医学报告
新增的第二个LLAVA模型调用接口
"""
from fastapi import APIRouter, HTTPException, status
from loguru import logger

from app.models.schemas import Llava7bReportRequest, ReportResponse
from app.services.llava7b_service import get_llava7b_service


router = APIRouter()


# ============ API端点 ============
@router.post("/generate-v2", response_model=ReportResponse)
async def generate_report_v2(request: Llava7bReportRequest):
    """
    生成医学报告接口 (LLAVA-7B版本)

    **功能**: 基于上传的X光片,调用LLAVA-7B模型生成详细医学报告

    **与 /generate 接口的区别**:
    - 使用LLAVA-7B模型 (参考 model_llava/deploy/test_ngork.py)
    - 支持support_info参数 (预留给未来的/api/v1/image/analyze接口)
    - 自动去除重复句子

    **前端调用示例**:
    ```javascript
    const response = await fetch('http://localhost:8000/api/v1/report/generate-v2', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            image_path: '/uploads/original_xxx.jpg',
            prompt: '请详细分析这张胸部X光片',
            support_info: 'Cardiomegaly detected with 87% confidence'  // 可选,当前可忽略
        })
    });
    const result = await response.json();
    console.log(result.report);  // 显示生成的报告
    ```

    **参数说明**:
    - image_path: 图片路径 (必填)
    - prompt: 自定义提示词 (可选,为空则使用默认prompt)
    - support_info: 分类结果信息 (可选,预留字段)
        - 当前 /api/v1/image/analyze 接口未完全实现
        - 此参数预留给未来使用,可以传入但不影响当前功能
        - 未来可传入如: "Detected: Cardiomegaly (87%), Edema (65%)"

    **实现细节**:
    1. 参考 model_llava/deploy/test_ngork.py 的调用方式
    2. 使用 multipart/form-data 发送图片和prompt
    3. Prompt模板参考 model_llava/deploy/prompt.txt
    4. 自动清理重复句子
    """
    try:
        logger.info(f"📝 收到 LLAVA-7B 报告生成请求: {request.image_path}")

        # 【核心逻辑】调用LLAVA-7B服务生成报告
        llava7b_service = get_llava7b_service()
        report, processing_time = await llava7b_service.generate_report(
            image_path=request.image_path,
            prompt=request.prompt,
            support_info=request.support_info,
            pathology_labels=request.pathology_labels
        )

        logger.success(f"✅ LLAVA-7B 报告生成完成,耗时: {processing_time:.2f}秒")

        return ReportResponse(
            success=True,
            message="LLAVA-7B报告生成成功",
            report=report,
            processing_time=processing_time
        )

    except FileNotFoundError as e:
        logger.error(f"❌ 图片文件未找到: {request.image_path}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="图片文件不存在"
        )
    except ValueError as e:
        logger.error(f"❌ 配置错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"❌ LLAVA-7B 报告生成失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LLAVA-7B 报告生成失败: {str(e)}"
        )


@router.get("/test-v2")
async def test_llava7b_endpoint():
    """测试接口,验证LLAVA-7B路由是否正常"""
    return {
        "endpoint": "llava7b_report",
        "status": "working",
        "message": "LLAVA-7B报告生成接口正常运行",
        "model": "LLAVA-7B",
        "reference": "model_llava/deploy/test_ngork.py"
    }
