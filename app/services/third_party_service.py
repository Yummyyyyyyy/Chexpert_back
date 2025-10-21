"""
第三方API服务
调用外部API获取疾病知识图谱
"""
from loguru import logger
from typing import Dict, Any
import httpx
import asyncio

from app.config import settings


class ThirdPartyService:
    """
    第三方API调用服务

    【TODO - 后端团队成员需要实现】:
    1. 在 config.py 中配置实际的API地址和密钥
    2. 实现真实的API调用逻辑
    3. 添加错误处理和重试机制
    4. 考虑添加结果缓存（避免重复查询同一疾病）
    5. 处理API限流问题
    """

    def __init__(self):
        self.api_url = settings.THIRD_PARTY_API_URL
        self.api_key = settings.THIRD_PARTY_API_KEY
        self.timeout = settings.API_TIMEOUT

    async def query_disease(
        self,
        disease_name: str,
        language: str = "zh"
    ) -> Dict[str, Any]:
        """
        查询疾病知识图谱

        参数:
            disease_name: 疾病名称
            language: 语言（zh/en）

        返回:
            知识图谱数据（JSON格式）

        【TODO】后端团队成员实现:
        ```python
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.api_url}/knowledge_graph",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "disease": disease_name,
                    "language": language
                },
                timeout=self.timeout
            )

            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"API调用失败: {response.status_code}")
        ```
        """
        logger.warning("⚠️  第三方API调用逻辑待实现 (third_party_service.py:65)")

        # ============ 以下是模拟代码，供前端调试使用 ============
        # 【后端团队成员需要替换为真实实现】

        # 检查API配置
        if not self.api_url or not self.api_key:
            logger.warning("⚠️  未配置第三方API地址或密钥，使用模拟数据")

        # 模拟API调用延迟
        await asyncio.sleep(0.8)

        # 模拟知识图谱数据（【TODO】替换为真实API返回）
        mock_knowledge_graph = {
            "disease_name": disease_name,
            "language": language,
            "description": self._get_mock_description(disease_name, language),
            "symptoms": [
                "呼吸困难",
                "胸闷",
                "心悸",
                "乏力"
            ],
            "causes": [
                "高血压",
                "冠心病",
                "心脏瓣膜病",
                "先天性心脏病"
            ],
            "treatments": [
                "药物治疗：利尿剂、血管扩张剂",
                "手术治疗：心脏瓣膜置换、搭桥手术",
                "生活方式调整：低盐饮食、适度运动"
            ],
            "prevention": [
                "控制血压",
                "定期体检",
                "健康饮食",
                "戒烟限酒"
            ],
            "related_diseases": [
                "心力衰竭",
                "肺动脉高压",
                "心律失常"
            ],
            "graph_visualization": {
                "nodes": [
                    {"id": "1", "label": disease_name, "type": "disease"},
                    {"id": "2", "label": "高血压", "type": "cause"},
                    {"id": "3", "label": "呼吸困难", "type": "symptom"},
                    {"id": "4", "label": "药物治疗", "type": "treatment"}
                ],
                "edges": [
                    {"from": "2", "to": "1", "label": "引起"},
                    {"from": "1", "to": "3", "label": "导致"},
                    {"from": "4", "to": "1", "label": "治疗"}
                ]
            }
        }

        logger.info(f"🔍 模拟返回知识图谱: {disease_name}")
        return mock_knowledge_graph

    def _get_mock_description(self, disease_name: str, language: str) -> str:
        """获取疾病描述（模拟）"""
        descriptions = {
            "zh": {
                "Cardiomegaly": "心脏肥大是指心脏体积异常增大，通常由高血压、心脏瓣膜病或其他心脏疾病引起。X光片上表现为心胸比>0.5。",
                "Edema": "肺水肿是指肺部积液过多，导致气体交换障碍。常见于心力衰竭、肾功能不全等疾病。",
                "Pneumonia": "肺炎是肺部的炎症性疾病,通常由细菌、病毒或其他病原体感染引起。"
            },
            "en": {
                "Cardiomegaly": "Cardiomegaly refers to an abnormally enlarged heart, usually caused by hypertension, heart valve disease, or other cardiac conditions.",
                "Edema": "Pulmonary edema is excessive fluid accumulation in the lungs, leading to impaired gas exchange.",
                "Pneumonia": "Pneumonia is an inflammatory condition of the lung usually caused by infection with bacteria, viruses, or other pathogens."
            }
        }

        return descriptions.get(language, {}).get(
            disease_name,
            f"关于 {disease_name} 的详细信息，请查阅医学文献。"
        )
