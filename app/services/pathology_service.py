"""
病症标签分析服务
"""

from PIL import Image
import logging
from typing import Dict, List
from datetime import datetime
from pathlib import Path

from app.models.pathology_model import get_pathology_model
from app.config import Config

logger = logging.getLogger(__name__)


class PathologyService:
    """病症标签分析服务类"""
    
    def __init__(self):
        """初始化服务"""
        # 从配置获取模型路径
        model_path = Config.PATHOLOGY_MODEL_PATH
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Pathology model not found at {model_path}")
        
        # 获取模型实例
        self.model = get_pathology_model(model_path)
        logger.info("PathologyService initialized")
    
    def analyze_image(
        self, 
        image_path: str,
        top_n: int = None,
        threshold: float = None
    ) -> Dict:
        """
        分析图像的病症标签
        
        Args:
            image_path: 图像文件路径
            top_n: 返回Top N结果（None表示返回全部）
            threshold: 概率阈值（None表示不过滤）
            
        Returns:
            分析结果字典
        """
        try:
            logger.info(f"Analyzing pathology for image: {image_path}")
            
            # 加载图像
            image = Image.open(image_path)
            
            # 预测病症
            pathologies = self.model.predict(image)
            
            # 应用过滤
            if threshold is not None:
                pathologies = self.model.filter_by_threshold(pathologies, threshold)
            
            # 应用Top N限制
            if top_n is not None:
                pathologies = self.model.get_top_pathologies(pathologies, top_n)
            
            # 构建响应
            result = {
                'success': True,
                'pathologies': pathologies,
                'total_count': len(pathologies),
                'timestamp': datetime.now().isoformat(),
                'model_info': {
                    'name': 'DenseNet121',
                    'version': 'v1.0',
                    'num_classes': 14
                }
            }
            
            # 添加统计信息
            result['statistics'] = self._calculate_statistics(pathologies)
            
            logger.info(f"Analysis completed: {len(pathologies)} pathologies detected")
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _calculate_statistics(self, pathologies: List[Dict]) -> Dict:
        """
        计算病症统计信息
        
        Args:
            pathologies: 病症列表
            
        Returns:
            统计信息字典
        """
        if not pathologies:
            return {
                'high_risk_count': 0,
                'medium_risk_count': 0,
                'low_risk_count': 0,
                'avg_probability': 0.0
            }
        
        severity_counts = {'high': 0, 'medium': 0, 'low': 0, 'minimal': 0}
        total_prob = 0.0
        
        for p in pathologies:
            severity_counts[p['severity']] += 1
            total_prob += p['probability']
        
        return {
            'high_risk_count': severity_counts['high'],
            'medium_risk_count': severity_counts['medium'],
            'low_risk_count': severity_counts['low'],
            'minimal_risk_count': severity_counts['minimal'],
            'avg_probability': round(total_prob / len(pathologies), 4)
        }
    
    def batch_analyze(self, image_paths: List[str]) -> List[Dict]:
        """
        批量分析图像
        
        Args:
            image_paths: 图像路径列表
            
        Returns:
            分析结果列表
        """
        results = []
        for path in image_paths:
            result = self.analyze_image(path)
            results.append(result)
        return results