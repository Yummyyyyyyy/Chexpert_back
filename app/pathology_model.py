# app/pathology_model.py
# -*- coding: utf-8 -*-
"""
病症标签分类模型
基于DenseNet121的14类CheXpert病症分类
"""

import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class PathologyClassifier:
    """病症标签分类器"""
    
    # CheXpert 14类病症标签
    PATHOLOGY_LABELS = [
        'No Finding',
        'Enlarged Cardiomediastinum',
        'Cardiomegaly',
        'Lung Opacity',
        'Lung Lesion',
        'Edema',
        'Consolidation',
        'Pneumonia',
        'Atelectasis',
        'Pneumothorax',
        'Pleural Effusion',
        'Pleural Other',
        'Fracture',
        'Support Devices'
    ]
    
    # 中文翻译
    LABEL_TRANSLATIONS = {
        'No Finding': '未发现异常',
        'Enlarged Cardiomediastinum': '心纵膈增大',
        'Cardiomegaly': '心脏肥大',
        'Lung Opacity': '肺部不透明',
        'Lung Lesion': '肺部病变',
        'Edema': '水肿',
        'Consolidation': '实变',
        'Pneumonia': '肺炎',
        'Atelectasis': '肺不张',
        'Pneumothorax': '气胸',
        'Pleural Effusion': '胸腔积液',
        'Pleural Other': '其他胸膜异常',
        'Fracture': '骨折',
        'Support Devices': '支持设备'
    }
    
    # 病症严重程度阈值
    SEVERITY_THRESHOLDS = {
        'high': 0.8,
        'medium': 0.6,
        'low': 0.4,
    }
    
    def __init__(self, model_path: str):
        """
        初始化模型
        
        Args:
            model_path: .pt模型文件路径
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # 构建DenseNet121模型
        self.model = models.densenet121(pretrained=False)
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, 14)
        
        # 加载权重
        self._load_weights(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        logger.info("PathologyClassifier initialized successfully")
    
    def _load_weights(self, model_path: str):
        """
        加载模型权重（智能处理各种格式）
        
        Args:
            model_path: 模型文件路径
        """
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            
            # 处理不同的保存格式
            if isinstance(state_dict, dict):
                if 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                elif 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
            
            # 【关键】处理键名前缀
            new_state_dict = {}
            for key, value in state_dict.items():
                # 移除可能的前缀：'densenet121.', 'model.', 'module.'
                new_key = key
                for prefix in ['densenet121.', 'model.', 'module.']:
                    if new_key.startswith(prefix):
                        new_key = new_key.replace(prefix, '', 1)
                        break
                
                new_state_dict[new_key] = value
            
            # 加载权重
            missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
            
            # 记录警告
            if missing_keys:
                logger.warning(f"Missing keys when loading model: {len(missing_keys)} keys")
                logger.debug(f"Missing keys: {missing_keys[:5]}...")  # 只显示前5个
            
            if unexpected_keys:
                logger.warning(f"Unexpected keys when loading model: {len(unexpected_keys)} keys")
                logger.debug(f"Unexpected keys: {unexpected_keys[:5]}...")
            
            # 检查分类器层是否正确加载
            classifier_weight = self.model.classifier.weight
            if classifier_weight.shape[0] == 14:
                logger.info(f"✅ Model loaded successfully: 14 classes output")
                logger.info(f"   Classifier shape: {classifier_weight.shape}")
            else:
                logger.error(f"❌ Classifier output dimension mismatch: {classifier_weight.shape}")
            
            logger.info(f"Model weights loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")
            raise
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        预处理图像
        
        Args:
            image: PIL Image对象
            
        Returns:
            预处理后的张量
        """
        # 转换为RGB模式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 应用变换
        tensor = self.transform(image)
        
        # 添加batch维度
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def predict(self, image: Image.Image) -> List[Dict]:
        """
        预测病症标签
        
        Args:
            image: PIL Image对象
            
        Returns:
            病症预测结果列表，按概率降序排列
        """
        try:
            # 预处理图像
            input_tensor = self.preprocess_image(image)
            
            # 推理
            with torch.no_grad():
                outputs = self.model(input_tensor)
                # 使用Sigmoid获取每个类别的概率
                probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
            
            # 构建结果
            results = []
            for idx, prob in enumerate(probabilities):
                label = self.PATHOLOGY_LABELS[idx]
                results.append({
                    'label': label,
                    'label_cn': self.LABEL_TRANSLATIONS.get(label, label),
                    'probability': float(prob),
                    'severity': self._get_severity(float(prob)),
                    'rank': 0
                })
            
            # 按概率降序排序
            results.sort(key=lambda x: x['probability'], reverse=True)
            
            # 设置排名
            for idx, result in enumerate(results):
                result['rank'] = idx + 1
            
            logger.info(f"Prediction completed: {len(results)} pathologies")
            logger.debug(f"Top 3: {[(r['label'], format(r['probability'], '.3f')) for r in results[:3]]}")

            
            return results
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def _get_severity(self, probability: float) -> str:
        """根据概率判断严重程度"""
        if probability >= self.SEVERITY_THRESHOLDS['high']:
            return 'high'
        elif probability >= self.SEVERITY_THRESHOLDS['medium']:
            return 'medium'
        elif probability >= self.SEVERITY_THRESHOLDS['low']:
            return 'low'
        else:
            return 'minimal'
    
    def get_top_pathologies(self, results: List[Dict], top_n: int = 5) -> List[Dict]:
        """获取Top N的病症"""
        return results[:top_n]
    
    def filter_by_threshold(self, results: List[Dict], threshold: float = 0.3) -> List[Dict]:
        """根据概率阈值过滤结果"""
        return [r for r in results if r['probability'] >= threshold]