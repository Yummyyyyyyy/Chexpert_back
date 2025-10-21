"""
数据模型定义
定义API请求和响应的数据结构
"""
from pydantic import BaseModel
from typing import List, Optional


# ============ 图像分析相关模型 ============
class ClassificationResult(BaseModel):
    """分类结果"""
    label: str  # 疾病类别，如 "Cardiomegaly", "Edema"等
    confidence: float  # 置信度 0-1
    description: Optional[str] = None  # 疾病描述（可选）


class AnalysisResponse(BaseModel):
    """图像分析响应"""
    success: bool
    message: str
    original_image_url: str  # 原始图片URL
    heatmap_image_url: str  # 热力图URL
    classifications: List[ClassificationResult]  # 分类结果列表


# ============ 报告生成相关模型 ============
class ReportRequest(BaseModel):
    """报告生成请求"""
    image_path: str  # 图片路径（可以是前面上传接口返回的路径）
    prompt: Optional[str] = "请分析这张X光片并生成详细的医学报告"  # 自定义提示词


class ReportResponse(BaseModel):
    """报告生成响应"""
    success: bool
    message: str
    report: str  # 生成的医学报告文本
    processing_time: Optional[float] = None  # 处理时间（秒）


# ============ 知识图谱相关模型 ============
class KnowledgeGraphRequest(BaseModel):
    """知识图谱请求"""
    disease_name: str  # 疾病名称，如 "Cardiomegaly"
    language: Optional[str] = "zh"  # 语言，zh中文/en英文


class KnowledgeGraphResponse(BaseModel):
    """知识图谱响应"""
    success: bool
    message: str
    disease_name: str
    knowledge_graph: dict  # 知识图谱数据（JSON格式）


# ============ LLAVA-7B报告生成相关模型 ============
class Llava7bReportRequest(BaseModel):
    """LLAVA-7B报告生成请求"""
    image_path: str  # 图片路径（可以是前面上传接口返回的路径）
    prompt: Optional[str] = None  # 自定义提示词（可选）
    support_info: Optional[str] = None  # 分类结果支持信息（可选，预留给未来的/api/v1/image/analyze接口）
