"""
功能3: 调用第三方API生成疾病知识图谱
"""
from fastapi import APIRouter, HTTPException, status
from loguru import logger

from app.models.schemas import KnowledgeGraphRequest, KnowledgeGraphResponse
from app.services.third_party_service import ThirdPartyService


router = APIRouter()


# ============ API端点 ============
@router.post("/query", response_model=KnowledgeGraphResponse)
async def query_knowledge_graph(request: KnowledgeGraphRequest):
    """
    查询疾病知识图谱接口

    **功能**: 调用第三方API，获取疾病的知识图谱信息

    **前端调用示例**:
    ```javascript
    const response = await fetch('http://localhost:8000/api/v1/knowledge/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            disease_name: 'Cardiomegaly',
            language: 'zh'
        })
    });
    const result = await response.json();
    console.log(result.knowledge_graph);  // 渲染知识图谱
    ```

    **【TODO - 后端团队成员需要修改的部分】**:
    1. 在 `app/services/third_party_service.py` 中实现真实的第三方API调用
    2. 在 `app/config.py` 中配置实际的API_URL和API_KEY
    3. 根据第三方API返回格式调整响应结构
    4. 添加错误处理和重试机制
    5. 考虑添加缓存机制，避免重复调用
    """
    try:
        logger.info(f"🔍 收到知识图谱查询请求: {request.disease_name}")

        # 【核心逻辑】调用第三方服务
        # 【TODO】这里需要后端团队成员实现真实的API调用
        third_party_service = ThirdPartyService()
        knowledge_graph = await third_party_service.query_disease(
            disease_name=request.disease_name,
            language=request.language
        )

        logger.success(f"✅ 知识图谱查询成功: {request.disease_name}")

        return KnowledgeGraphResponse(
            success=True,
            message="知识图谱查询成功",
            disease_name=request.disease_name,
            knowledge_graph=knowledge_graph
        )

    except Exception as e:
        logger.error(f"❌ 知识图谱查询失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"知识图谱查询失败: {str(e)}"
        )


@router.get("/test")
async def test_knowledge_endpoint():
    """测试接口，验证路由是否正常"""
    return {
        "endpoint": "third_party",
        "status": "working",
        "message": "知识图谱接口正常运行"
    }
