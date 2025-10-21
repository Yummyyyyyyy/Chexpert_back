# app/pathology_model.py
# -*- coding: utf-8 -*-
"""
病症标签分类模型
基于DenseNet121的14类CheXpert病症分类
"""

from __future__ import annotations
import os
import logging
from pathlib import Path
from typing import List, Dict, Optional

import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------- 默认权重路径（支持环境变量覆盖） ----------------
# 以当前文件为基准定位 Chexpert_back/weights 目录
_BASE_DIR = Path(__file__).resolve().parent          # app/
_WEIGHT_DIR = _BASE_DIR.parent / "weights"           # Chexpert_back/weights

DEFAULT_PATHOLOGY_WEIGHTS = os.getenv(
    "PATHOLOGY_WEIGHTS",
    str(_WEIGHT_DIR / "pathology_model.pt")          # ← 按你的要求改为 .pt
)


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

    def __init__(self, weights_path: Optional[str] = None, device: str = "cpu"):
        """
        初始化模型

        Args:
            weights_path: 模型权重路径；不传则使用 DEFAULT_PATHOLOGY_WEIGHTS（pathology_model.pt）
            device: 'cpu' 或 'cuda'
        """
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.weights_path = weights_path or DEFAULT_PATHOLOGY_WEIGHTS

        logger.info(f"[Pathology] Using device: {self.device}")
        logger.info(f"[Pathology] Weights: {self.weights_path}")

        # 优先尝试按 TorchScript 整模型加载（常见 .pt）
        self.model: Optional[torch.nn.Module] = None
        if self._try_load_torchscript(self.weights_path):
            logger.info("[Pathology] Loaded TorchScript model.")
        else:
            # 回退到 state_dict → DenseNet121
            logger.info("[Pathology] Fallback to DenseNet121 + state_dict.")
            self._build_and_load_state_dict(self.weights_path)

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
        logger.info("[Pathology] Classifier initialized successfully.")

    # ---------------- 加载相关 ----------------

    def _try_load_torchscript(self, path: str) -> bool:
        """
        若 `pathology_model.pt` 是脚本化/跟踪的整模型，则用 torch.jit.load 直接加载。
        成功返回 True；失败返回 False（交由 state_dict 逻辑处理）。
        """
        try:
            # TorchScript 一定要用 jit.load（torch.load 得到的对象不一定能直接 eval）
            model = torch.jit.load(path, map_location=self.device)
            # 简单 sanity check：必须是可调用的 Module
            if not isinstance(model, torch.jit.RecursiveScriptModule) and not isinstance(model, torch.nn.Module):
                return False
            self.model = model
            return True
        except Exception as e:
            # 不是 TorchScript 或加载失败，回退
            logger.debug(f"[Pathology] TorchScript load failed or not a scripted model: {e}")
            return False

    def _build_and_load_state_dict(self, path: str) -> None:
        """
        构建 DenseNet121 并把 state_dict 加载进去（兼容不同 key 前缀）
        """
        # 构建 DenseNet121
        base = models.densenet121(pretrained=False)
        num_features = base.classifier.in_features
        base.classifier = nn.Linear(num_features, 14)  # 14 类输出
        self.model = base

        # 读取 state_dict
        state = torch.load(path, map_location=self.device)
        if isinstance(state, dict):
            if 'model_state_dict' in state:
                state = state['model_state_dict']
            elif 'state_dict' in state:
                state = state['state_dict']

        # 处理键名前缀
        new_state = {}
        if isinstance(state, dict):
            for k, v in state.items():
                nk = k
                for prefix in ['densenet121.', 'model.', 'module.']:
                    if nk.startswith(prefix):
                        nk = nk[len(prefix):]
                        break
                new_state[nk] = v
        else:
            raise RuntimeError("Unsupported weight format for state_dict loading.")

        missing, unexpected = self.model.load_state_dict(new_state, strict=False)
        if missing:
            logger.warning(f"[Pathology] Missing keys: {len(missing)} (show first 5): {missing[:5]}")
        if unexpected:
            logger.warning(f"[Pathology] Unexpected keys: {len(unexpected)} (show first 5): {unexpected[:5]}")

        # 维度确认
        out_dim = self.model.classifier.weight.shape[0]
        if out_dim != 14:
            logger.error(f"[Pathology] Classifier output dimension mismatch: {out_dim} (expect 14)")

    # ---------------- 预处理 & 推理 ----------------

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        预处理图像
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        tensor = self.transform(image).unsqueeze(0)
        return tensor.to(self.device)

    def predict(self, image: Image.Image) -> List[Dict]:
        """
        预测病症标签

        Returns:
            病症预测结果列表（按概率降序）
        """
        try:
            x = self.preprocess_image(image)
            with torch.no_grad():
                outputs = self.model(x)
                # TorchScript/自定义模型有时直接给 logits，做一次 sigmoid
                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[0]
                probs = torch.sigmoid(outputs).detach().cpu().numpy()[0]

            results: List[Dict] = []
            for idx, p in enumerate(probs):
                label = self.PATHOLOGY_LABELS[idx]
                results.append({
                    'label': label,
                    'label_cn': self.LABEL_TRANSLATIONS.get(label, label),
                    'probability': float(p),
                    'severity': self._get_severity(float(p)),
                    'rank': 0
                })

            # 排序与排名
            results.sort(key=lambda r: r['probability'], reverse=True)
            for i, r in enumerate(results):
                r['rank'] = i + 1

            logger.info(f"[Pathology] Prediction OK: {len(results)} labels")
            return results

        except Exception as e:
            logger.error(f"[Pathology] Prediction failed: {e}")
            raise

    # ---------------- 业务辅助 ----------------

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
