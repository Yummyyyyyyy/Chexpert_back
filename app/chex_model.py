# app/chex_model.py
# -*- coding: utf-8 -*-
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models

_MODEL: Optional[nn.Module] = None
_CLASS_NAMES: Optional[List[str]] = None
_DEVICE: Optional[torch.device] = None
_TRANSFORM: Optional[T.Compose] = None
_LAST_FEATS: Optional[torch.Tensor] = None  # 用于 CAM

def _get_device(explicit: Optional[str] = None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _build_densenet121(num_classes: int) -> nn.Module:
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, num_classes),
        nn.Sigmoid(),  # 多标签
    )
    return model

def init_model(
    model_path: str,
    class_names: Optional[List[str]] = None,
    device: Optional[str] = None,
    use_imagenet_norm: bool = False,
) -> Tuple[nn.Module, List[str]]:
    global _MODEL, _CLASS_NAMES, _DEVICE, _TRANSFORM, _LAST_FEATS

    _DEVICE = _get_device(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型权重未找到: {model_path}")

    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state_dict = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state_dict = state["model_state_dict"]
    else:
        state_dict = state

    # 更宽松的类别数推断
    num_classes = None
    for k, v in state_dict.items():
        if ("classifier" in k) and k.endswith(".weight") and isinstance(v, torch.Tensor) and v.ndim == 2:
            num_classes = v.shape[0]
            break
    if num_classes is None:
        if not class_names:
            raise ValueError("无法从权重推断类别数，请传入 class_names")
        num_classes = len(class_names)

    if class_names is None:
        if isinstance(state, dict) and "class_names" in state and isinstance(state["class_names"], (list, tuple)):
            class_names = list(state["class_names"])
        else:
            class_names = [f"C{i}" for i in range(num_classes)]
    else:
        assert len(class_names) == num_classes, "class_names 长度与权重类别数不一致"

    # 统一权重键前缀，兼容 "densenet121." / "module." 等保存格式
    normalized_state = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("densenet121.", "model.", "module."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        normalized_state[new_key] = value

    model = _build_densenet121(num_classes)
    missing, unexpected = model.load_state_dict(normalized_state, strict=False)
    if missing or unexpected:
        print(f"[chex_model] load_state_dict 提示 missing={missing} unexpected={unexpected}")

    model.eval().to(_DEVICE)

    def _hook(module, inp, out):
        global _LAST_FEATS
        _LAST_FEATS = out.detach()  # [B,C,H,W]
    model.features.register_forward_hook(_hook)

    tfms = [T.Resize((224, 224)), T.ToTensor()]
    if use_imagenet_norm:
        tfms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    _TRANSFORM = T.Compose(tfms)

    _MODEL = model
    _CLASS_NAMES = class_names

    print(f"[chex_model] 模型已加载: classes={len(class_names)} device={_DEVICE}")
    return _MODEL, _CLASS_NAMES

def _to_pil(img: Union[str, Image.Image, np.ndarray]) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, str):
        if not os.path.exists(img):
            raise FileNotFoundError(f"图片不存在: {img}")
        return Image.open(img).convert("RGB")
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)
    raise TypeError(f"不支持的输入类型: {type(img)}")

def _make_cam(feats: torch.Tensor, classifier: nn.Module, probs: np.ndarray,
              pos_indices: List[int], target_size) -> Optional[np.ndarray]:
    if isinstance(classifier, nn.Sequential):
        linear = None
        for m in classifier.modules():
            if isinstance(m, nn.Linear):
                linear = m
                break
    elif isinstance(classifier, nn.Linear):
        linear = classifier
    else:
        linear = None
    if linear is None:
        return None

    with torch.no_grad():
        w = linear.weight.detach().cpu().numpy()  # [K, C]
        f = feats.detach().cpu().numpy()[0]       # [C, H, W]

    cams = []
    for idx in pos_indices:
        if idx < 0 or idx >= w.shape[0]:
            continue
        cam = (w[idx][:, None, None] * f).sum(axis=0)
        cam = np.maximum(cam, 0)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        cams.append(cam)
    if not cams:
        top_idx = int(np.argmax(probs))
        cam = (w[top_idx][:, None, None] * f).sum(axis=0)
        cam = np.maximum(cam, 0)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        cams = [cam]

    cam_sum = np.clip(np.sum(cams, axis=0), 0, 1)
    cam_resized = cv2.resize(cam_sum, (target_size[0], target_size[1]))
    heatmap = (cam_resized * 255).astype(np.uint8)
    return heatmap

def infer(
    image: Union[str, Image.Image, np.ndarray],
    generate_heatmap: bool = True,
    threshold: float = 0.5,
    alpha: float = 0.45,
    return_top_k: Optional[int] = None
) -> Dict[str, Any]:
    global _MODEL, _CLASS_NAMES, _DEVICE, _TRANSFORM, _LAST_FEATS
    if _MODEL is None or _CLASS_NAMES is None or _TRANSFORM is None:
        raise RuntimeError("模型未初始化。请先调用 init_model(model_path, ...)")

    pil = _to_pil(image)
    orig_w, orig_h = pil.size

    tensor = _TRANSFORM(pil).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        _LAST_FEATS = None
        outputs = _MODEL(tensor)  # [1,K], Sigmoid
        probs = outputs.squeeze(0).detach().cpu().numpy()

    preds = (probs > float(threshold)).astype(np.int32)
    probs_dict = {lbl: float(p) for lbl, p in zip(_CLASS_NAMES, probs)}
    preds_dict = {lbl: int(v) for lbl, v in zip(_CLASS_NAMES, preds)}

    idx_sorted = np.argsort(-probs)
    findings = []
    for idx in idx_sorted:
        if preds[idx] == 1:
            findings.append({"label": _CLASS_NAMES[idx], "confidence": float(probs[idx])})
    if return_top_k is not None:
        findings = findings[: int(return_top_k)]

    heat_img = None
    if generate_heatmap and _LAST_FEATS is not None:
        pos_indices = [i for i, v in enumerate(preds) if v == 1]
        cam_u8 = _make_cam(_LAST_FEATS, _MODEL.classifier, probs, pos_indices, (orig_w, orig_h))
        if cam_u8 is not None:
            base = np.array(pil.convert("RGB"))
            color = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            overlay = (alpha * color + (1 - alpha) * base).astype(np.uint8)
            heat_img = Image.fromarray(overlay)

    return {
        "probs": probs_dict,
        "preds": preds_dict,
        "positive_findings": findings,
        "heatmap": heat_img,
    }
