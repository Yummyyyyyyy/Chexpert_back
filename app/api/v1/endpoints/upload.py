"""
功能1: 图像上传与分析
接收X光片，返回热力图和分类结果
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, status
import os
from loguru import logger

from app.config import settings
from app.models.schemas import AnalysisResponse
from app.models.heatmap_generator import HeatmapGenerator
from app.utils.image_utils import validate_image, save_upload_file


router = APIRouter()


# ============ API端点 ============
@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(
    file: UploadFile = File(..., description="上传的X光片图像")
):
    """
    图像分析接口

    **功能**: 上传X光片，返回疾病分类结果和热力图

    **前端调用示例**:
    ```javascript
    const formData = new FormData();
    formData.append('file', imageFile);

    const response = await fetch('http://localhost:8000/api/v1/image/analyze', {
        method: 'POST',
        body: formData
    });
    const result = await response.json();
    console.log(result.heatmap_image_url);  // 显示热力图
    console.log(result.classifications);     // 显示分类结果
    ```

    **返回格式**:
    - success: 是否成功
    - message: 提示信息
    - original_image_url: 原始图片路径
    - heatmap_image_url: 热力图路径
    - classifications: 疾病分类结果数组

    **【TODO - 后端团队成员需要修改的部分】**:
    1. 在 `app/models/heatmap_generator.py` 中实现真实的模型推理逻辑
    2. 替换下方的模拟数据为实际模型输出
    3. 根据实际模型调整分类类别和置信度计算
    """
    try:
        # 1. 验证文件格式
        if not validate_image(file.filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"不支持的文件格式，仅支持: {settings.ALLOWED_EXTENSIONS}"
            )

        logger.info(f"📥 收到图像上传请求: {file.filename}")

        # 2. 保存上传的文件
        original_path = await save_upload_file(file, prefix="original")
        logger.info(f"💾 图片已保存: {original_path}")

        # 3. 【核心逻辑】调用模型生成热力图和分类结果
        # 【TODO】这里需要后端团队成员实现真实的模型推理
        heatmap_generator = HeatmapGenerator()
        heatmap_path, classifications = await heatmap_generator.generate(original_path)

        logger.success(f"✅ 分析完成，检测到 {len(classifications)} 个疾病类别")

        # 4. 返回结果
        # original_path 和 heatmap_path 都是相对路径或文件名
        # 需要添加 /uploads/ 前缀以便前端访问
        return AnalysisResponse(
            success=True,
            message="图像分析完成",
            original_image_url=f"/uploads/{os.path.basename(original_path)}",
            heatmap_image_url=f"/uploads/{os.path.basename(heatmap_path)}",
            classifications=classifications
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 图像分析失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"图像分析失败: {str(e)}"
        )


@router.get("/test")
async def test_upload_endpoint():
    """测试接口，验证路由是否正常"""
    return {
        "endpoint": "upload",
        "status": "working",
        "message": "图像分析接口正常运行"
    }
