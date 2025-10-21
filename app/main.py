# app/main.py
# -*- coding: utf-8 -*-
import os, io, uuid, datetime, time, json
from typing import Optional, List

import numpy as np
from PIL import Image

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch

# =========================
# 可选：读取你的既有配置
# =========================
try:
    from app.config import settings
except Exception:
    class _S:
        APP_NAME = "CheXpert Inference API"
        CORS_ORIGINS = ["*"]
        MODEL_PATH = "final_global_model.pth"
        PATHOLOGY_MODEL_PATH = "pathology_model.pt"  # 新增
    settings = _S()

# =========================
# 路径与静态目录（自动创建）
# =========================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STATIC_ROOT = os.path.join(ROOT, "static")
UPLOAD_DIR = os.path.join(STATIC_ROOT, "uploads")
HEATMAP_DIR = os.path.join(STATIC_ROOT, "heatmaps")
for d in (STATIC_ROOT, UPLOAD_DIR, HEATMAP_DIR):
    os.makedirs(d, exist_ok=True)

# 历史记录数据文件
DATA_DIR = os.path.join(ROOT, "data")
HISTORY_FILE = os.path.join(DATA_DIR, "analysis_history.jsonl")
os.makedirs(DATA_DIR, exist_ok=True)

# =========================
# 创建应用 & CORS & 静态资源
# =========================
app = FastAPI(title=getattr(settings, "APP_NAME", "CheXpert Inference API v2.0"))
app.add_middleware(
    CORSMiddleware,
    allow_origins=getattr(settings, "CORS_ORIGINS", ["*"]),
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")

# =========================
# 导入模型（热力图模型）
# =========================
from .chex_model import init_model, infer as model_infer  # noqa

MODEL_PATH = getattr(settings, "MODEL_PATH", "final_global_model.pth")

# CheXpert-14 类名
CHEX_CLASSES: List[str] = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Lesion", "Lung Opacity", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices"
]

# 热力图模型
_MODEL = None
_CLASS_NAMES = None
try:
    _MODEL, _CLASS_NAMES = init_model(
        MODEL_PATH,
        class_names=CHEX_CLASSES,
        use_imagenet_norm=False
    )
    print("[app] 热力图模型加载 OK，classes:", len(_CLASS_NAMES))
except Exception as e:
    print("[app] 热力图模型加载失败：", e)

# =========================
# 新增：导入病症标签模型
# =========================
_PATHOLOGY_MODEL = None
try:
    from .pathology_model import PathologyClassifier
    PATHOLOGY_MODEL_PATH = getattr(settings, "PATHOLOGY_MODEL_PATH", "pathology_model.pt")
    if os.path.exists(PATHOLOGY_MODEL_PATH):
        _PATHOLOGY_MODEL = PathologyClassifier(PATHOLOGY_MODEL_PATH)
        print("[app] 病症标签模型加载 OK")
    else:
        print(f"[app] 病症标签模型文件不存在: {PATHOLOGY_MODEL_PATH}")
except Exception as e:
    print("[app] 病症标签模型加载失败：", e)

# =========================
# 工具函数
# =========================
def _to_pil_from_upload(f: UploadFile) -> Image.Image:
    name = (f.filename or "").lower()
    data = f.file.read()
    if name.endswith(".dcm"):
        import pydicom
        ds = pydicom.dcmread(io.BytesIO(data))
        arr = ds.pixel_array.astype(np.float32)
        arr = (255 * (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)).astype(np.uint8)
        if arr.ndim == 2:
            return Image.fromarray(arr).convert("RGB")
        else:
            return Image.fromarray(arr[..., :3])
    return Image.open(io.BytesIO(data)).convert("RGB")

def _append_history(item: dict) -> None:
    """
    扩展历史记录格式，支持病症标签
    item: {
        "date", "file_name", "diagnosis", "confidence", "status",
        "pathologies": [...],  # 新增：病症标签列表
        "heatmap_url": "...",
        "original_url": "..."
    }
    """
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

def _read_history() -> List[dict]:
    if not os.path.exists(HISTORY_FILE):
        return []
    out = []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return list(reversed(out))  # 最新在前

# =========================
# 健康检查
# =========================
@app.get("/health")
def health():
    heatmap_ok = _MODEL is not None and _CLASS_NAMES is not None
    pathology_ok = _PATHOLOGY_MODEL is not None
    
    return JSONResponse({
        "status": "ok" if (heatmap_ok or pathology_ok) else "not_ready",
        "services": {
            "heatmap": {
                "loaded": heatmap_ok,
                "model": "DenseNet121 (CAM)"
            },
            "pathology": {
                "loaded": pathology_ok,
                "model": "DenseNet121 (14-class)"
            }
        },
        "device": "cuda:0" if torch.cuda.is_available() else "cpu"
    }, status_code=200 if (heatmap_ok or pathology_ok) else 503)

# =========================
# 历史记录查询
# =========================
@app.get("/api/v1/history")
def get_history(page: int = 1, page_size: int = 10):
    data = _read_history()
    total = len(data)
    start = max(0, (page - 1) * page_size)
    end = start + page_size
    return {
        "success": True,
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": data[start:end],
    }

# =========================
# 原有推理接口（热力图）- 保持不变
# =========================
@app.post("/api/v1/image/analyze")
def analyze(
    request: Request,
    file: UploadFile = File(...),
    generate_heatmap: bool = Form(True),
    threshold: float = Form(0.5),
    alpha: float = Form(0.45),
    return_top_k: Optional[int] = Form(None),
):
    if _MODEL is None:
        raise HTTPException(status_code=424, detail={"error_code": "MODEL_NOT_READY", "message": "Heatmap model not loaded"})

    # 1) 解析文件
    try:
        pil = _to_pil_from_upload(file)
    except Exception:
        raise HTTPException(status_code=400, detail={"error_code": "BAD_FILE_TYPE", "message": "Only jpg/png/dcm supported."})

    # 2) 推理
    try:
        t0 = time.time()
        out = model_infer(
            pil,
            generate_heatmap=generate_heatmap,
            threshold=float(threshold),
            alpha=float(alpha),
            return_top_k=return_top_k
        )
        infer_ms = int((time.time() - t0) * 1000)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error_code": "INFERENCE_ERROR", "message": str(e)})

    # 3) 保存图片
    day = datetime.date.today().isoformat()
    up_dir = os.path.join(UPLOAD_DIR, day)
    hm_dir = os.path.join(HEATMAP_DIR, day)
    os.makedirs(up_dir, exist_ok=True)
    os.makedirs(hm_dir, exist_ok=True)

    uid = uuid.uuid4().hex[:8]
    orig_path = os.path.join(up_dir, f"{uid}.png")
    pil.save(orig_path)

    heatmap_url = None
    if generate_heatmap and out.get("heatmap") is not None:
        hm_path = os.path.join(hm_dir, f"{uid}.png")
        out["heatmap"].save(hm_path)
        heatmap_url = f"{request.base_url}static/heatmaps/{day}/{uid}.png".rstrip("/")
    original_url = f"{request.base_url}static/uploads/{day}/{uid}.png".rstrip("/")

    # 4) 组装返回
    classifications = [
        {"label": it["label"], "confidence": it["confidence"], "description": ""}
        for it in out.get("positive_findings", [])
    ]
    best_label, best_prob = None, None
    if out.get("probs"):
        best_label, best_prob = max(out["probs"].items(), key=lambda kv: kv[1])
    if not classifications and best_label is not None:
        classifications = [{"label": best_label, "confidence": float(best_prob), "description": ""}]

    # 5) 写历史
    diag_label = classifications[0]["label"] if classifications else (best_label or "-")
    diag_conf = classifications[0]["confidence"] if classifications else float(best_prob or 0.0)
    _append_history({
        "date": day,
        "file_name": file.filename or f"{uid}.png",
        "diagnosis": diag_label,
        "confidence": float(diag_conf),
        "status": "completed",
        "heatmap_url": heatmap_url,
        "original_url": original_url
    })

    return {
        "success": True,
        "classifications": classifications,
        "heatmap_image_url": heatmap_url,
        "original_image_url": original_url,
        "meta": {
            "model_name": "DenseNet121",
            "inference_time_ms": infer_ms,
            "threshold": float(threshold),
            "device": "cuda:0" if torch.cuda.is_available() else "cpu"
        }
    }

# =========================
# 新增：病症标签分析接口
# =========================
@app.post("/api/v1/pathology/analyze")
def analyze_pathology(
    request: Request,
    file: UploadFile = File(...),
    top_n: Optional[int] = Form(None),
    threshold: Optional[float] = Form(None),
):
    """
    病症标签分析接口
    
    参数:
    - file: 上传的X光片图像
    - top_n: 返回前N个病症（可选）
    - threshold: 概率阈值，只返回超过此阈值的病症（可选）
    
    返回:
    - pathologies: 病症列表，按概率降序排列
    - statistics: 统计信息
    """
    if _PATHOLOGY_MODEL is None:
        raise HTTPException(
            status_code=424,
            detail={"error_code": "MODEL_NOT_READY", "message": "Pathology model not loaded"}
        )

    # 1) 解析文件
    try:
        pil = _to_pil_from_upload(file)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail={"error_code": "BAD_FILE_TYPE", "message": "Only jpg/png/dcm supported."}
        )

    # 2) 推理
    try:
        t0 = time.time()
        pathologies = _PATHOLOGY_MODEL.predict(pil)
        infer_ms = int((time.time() - t0) * 1000)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error_code": "INFERENCE_ERROR", "message": str(e)}
        )

    # 3) 应用过滤
    if threshold is not None:
        pathologies = [p for p in pathologies if p['probability'] >= threshold]
    
    if top_n is not None:
        pathologies = pathologies[:top_n]

    # 4) 保存原始图片
    day = datetime.date.today().isoformat()
    up_dir = os.path.join(UPLOAD_DIR, day)
    os.makedirs(up_dir, exist_ok=True)
    
    uid = uuid.uuid4().hex[:8]
    orig_path = os.path.join(up_dir, f"{uid}.png")
    pil.save(orig_path)
    original_url = f"{request.base_url}static/uploads/{day}/{uid}.png".rstrip("/")

    # 5) 计算统计信息
    statistics = _calculate_pathology_statistics(pathologies)

    # 6) 返回结果
    return {
        "success": True,
        "pathologies": pathologies,
        "total_count": len(pathologies),
        "statistics": statistics,
        "original_image_url": original_url,
        "meta": {
            "model_name": "DenseNet121",
            "model_type": "pathology_classifier",
            "inference_time_ms": infer_ms,
            "device": "cuda:0" if torch.cuda.is_available() else "cpu",
            "timestamp": datetime.datetime.now().isoformat()
        }
    }

def _calculate_pathology_statistics(pathologies: List[dict]) -> dict:
    """计算病症统计信息"""
    if not pathologies:
        return {
            "high_risk_count": 0,
            "medium_risk_count": 0,
            "low_risk_count": 0,
            "minimal_risk_count": 0,
            "avg_probability": 0.0
        }
    
    severity_counts = {"high": 0, "medium": 0, "low": 0, "minimal": 0}
    total_prob = 0.0
    
    for p in pathologies:
        severity_counts[p.get('severity', 'minimal')] += 1
        total_prob += p['probability']
    
    return {
        "high_risk_count": severity_counts['high'],
        "medium_risk_count": severity_counts['medium'],
        "low_risk_count": severity_counts['low'],
        "minimal_risk_count": severity_counts['minimal'],
        "avg_probability": round(total_prob / len(pathologies), 4)
    }

# =========================
# 新增：综合分析接口（热力图 + 病症标签）
# =========================
@app.post("/api/v1/image/analyze-full")
def analyze_full(
    request: Request,
    file: UploadFile = File(...),
    generate_heatmap: bool = Form(True),
    heatmap_threshold: float = Form(0.5),
    heatmap_alpha: float = Form(0.45),
    pathology_top_n: Optional[int] = Form(5),
    pathology_threshold: Optional[float] = Form(0.3),
):
    """
    综合分析接口：同时返回热力图和病症标签
    
    这是推荐使用的接口，一次调用获取所有分析结果
    """
    # 检查模型状态
    heatmap_available = _MODEL is not None
    pathology_available = _PATHOLOGY_MODEL is not None
    
    if not (heatmap_available or pathology_available):
        raise HTTPException(
            status_code=424,
            detail={"error_code": "NO_MODEL_READY", "message": "No model available"}
        )

    # 1) 解析文件
    try:
        pil = _to_pil_from_upload(file)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail={"error_code": "BAD_FILE_TYPE", "message": "Only jpg/png/dcm supported."}
        )

    result = {
        "success": True,
        "heatmap_data": None,
        "pathology_data": None,
    }

    # 2) 执行热力图分析
    if heatmap_available and generate_heatmap:
        try:
            t0 = time.time()
            heatmap_out = model_infer(
                pil,
                generate_heatmap=True,
                threshold=float(heatmap_threshold),
                alpha=float(heatmap_alpha),
                return_top_k=None
            )
            heatmap_ms = int((time.time() - t0) * 1000)
            
            # 保存热力图
            day = datetime.date.today().isoformat()
            hm_dir = os.path.join(HEATMAP_DIR, day)
            os.makedirs(hm_dir, exist_ok=True)
            
            uid = uuid.uuid4().hex[:8]
            if heatmap_out.get("heatmap") is not None:
                hm_path = os.path.join(hm_dir, f"{uid}.png")
                heatmap_out["heatmap"].save(hm_path)
                heatmap_url = f"{request.base_url}static/heatmaps/{day}/{uid}.png".rstrip("/")
            else:
                heatmap_url = None
            
            result["heatmap_data"] = {
                "heatmap_image_url": heatmap_url,
                "positive_findings": heatmap_out.get("positive_findings", []),
                "inference_time_ms": heatmap_ms
            }
        except Exception as e:
            print(f"[analyze_full] Heatmap generation failed: {e}")
            result["heatmap_data"] = {"error": str(e)}

    # 3) 执行病症标签分析
    if pathology_available:
        try:
            t0 = time.time()
            pathologies = _PATHOLOGY_MODEL.predict(pil)
            pathology_ms = int((time.time() - t0) * 1000)
            
            # 应用过滤
            if pathology_threshold is not None:
                pathologies = [p for p in pathologies if p['probability'] >= pathology_threshold]
            if pathology_top_n is not None:
                pathologies = pathologies[:pathology_top_n]
            
            result["pathology_data"] = {
                "pathologies": pathologies,
                "total_count": len(pathologies),
                "statistics": _calculate_pathology_statistics(pathologies),
                "inference_time_ms": pathology_ms
            }
        except Exception as e:
            print(f"[analyze_full] Pathology analysis failed: {e}")
            result["pathology_data"] = {"error": str(e)}

    # 4) 保存原始图片
    day = datetime.date.today().isoformat()
    up_dir = os.path.join(UPLOAD_DIR, day)
    os.makedirs(up_dir, exist_ok=True)
    
    uid = uuid.uuid4().hex[:8]
    orig_path = os.path.join(up_dir, f"{uid}.png")
    pil.save(orig_path)
    original_url = f"{request.base_url}static/uploads/{day}/{uid}.png".rstrip("/")
    
    result["original_image_url"] = original_url

    # 5) 写入历史记录（扩展格式）
    pathology_list = result.get("pathology_data", {}).get("pathologies", [])
    top_pathology = pathology_list[0] if pathology_list else None
    
    _append_history({
        "date": day,
        "file_name": file.filename or f"{uid}.png",
        "diagnosis": top_pathology["label"] if top_pathology else "Unknown",
        "confidence": top_pathology["probability"] if top_pathology else 0.0,
        "status": "completed",
        "heatmap_url": result.get("heatmap_data", {}).get("heatmap_image_url"),
        "original_url": original_url,
        "pathologies": pathology_list[:5] if pathology_list else []  # 保存前5个
    })

    # 6) 返回结果
    result["meta"] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "services_used": {
            "heatmap": result["heatmap_data"] is not None,
            "pathology": result["pathology_data"] is not None
        }
    }

    return result

# =========================
# 新增：获取病症标签列表
# =========================
@app.get("/api/v1/pathology/labels")
def get_pathology_labels():
    """获取所有可能的病症标签"""
    if _PATHOLOGY_MODEL is None:
        raise HTTPException(
            status_code=424,
            detail={"error_code": "MODEL_NOT_READY", "message": "Pathology model not loaded"}
        )
    
    labels = []
    for idx, (label, label_cn) in enumerate(zip(
        _PATHOLOGY_MODEL.PATHOLOGY_LABELS,
        [_PATHOLOGY_MODEL.LABEL_TRANSLATIONS.get(l, l) for l in _PATHOLOGY_MODEL.PATHOLOGY_LABELS]
    )):
        labels.append({
            "index": idx,
            "label": label,
            "label_cn": label_cn
        })
    
    return {
        "success": True,
        "labels": labels,
        "total_count": len(labels)
    }

# =========================
# 报告生成（扩展支持病症标签）
# =========================
from pydantic import BaseModel
from fastapi import Body

class ReportIn(BaseModel):
    patient_id: str | None = None
    study_id: str | None = None
    classifications: list[dict] = []
    pathologies: list[dict] = []  # 新增：病症标签数据
    heatmap_image_url: str | None = None
    original_image_url: str | None = None
    notes: str | None = None

@app.post("/api/v1/report/generate")
def generate_report(payload: ReportIn = Body(...)):
    """
    生成医疗报告（支持热力图分类和病症标签）
    """
    # 优先使用病症标签数据
    if payload.pathologies:
        items = sorted(payload.pathologies, key=lambda x: x.get("probability", 0), reverse=True)[:5]
        
        if not items:
            text = (
                "Impression:\n"
                "- No specific abnormality detected. Correlate clinically.\n\n"
                "Findings:\n"
                "- The AI model did not identify abnormalities above the set threshold.\n"
            )
        else:
            # 分类病症严重程度
            high_risk = [it for it in items if it.get('severity') == 'high']
            medium_risk = [it for it in items if it.get('severity') == 'medium']
            
            primary = items[0]
            
            text = f"Impression:\n"
            if high_risk:
                text += f"- **{primary.get('label', 'Unknown')}** detected with high confidence ({primary.get('probability', 0):.2%}).\n"
                text += f"- {len(high_risk)} high-risk finding(s) identified.\n"
            else:
                text += f"- Findings are most consistent with **{primary.get('label', 'Unknown')}** ({primary.get('probability', 0):.2%}).\n"
            
            text += "\nFindings:\n"
            for idx, it in enumerate(items, 1):
                label = it.get('label', 'Unknown')
                label_cn = it.get('label_cn', '')
                prob = it.get('probability', 0)
                severity = it.get('severity', 'minimal')
                
                severity_text = {
                    'high': '⚠️ High Risk',
                    'medium': '⚡ Medium Risk',
                    'low': '✓ Low Risk',
                    'minimal': '○ Minimal'
                }.get(severity, '')
                
                text += f"{idx}. {label} ({label_cn}) - {prob:.1%} {severity_text}\n"
    
    # 使用原有的热力图分类数据
    elif payload.classifications:
        items = sorted(payload.classifications, key=lambda x: x.get("confidence", 0), reverse=True)[:3]
        if not items:
            text = (
                "Impression:\n"
                "- No specific abnormality detected. Correlate clinically.\n\n"
                "Findings:\n"
                "- The AI model did not identify abnormalities above the set threshold.\n"
            )
        else:
            lines = [f"- {it.get('label','?')} (confidence {it.get('confidence',0):.2f})" for it in items]
            primary = items[0].get("label", "N/A")
            text = (
                f"Impression:\n- Findings are most consistent with **{primary}**.\n\n"
                "Findings:\n" + "\n".join(lines) + "\n"
            )
    else:
        text = "Impression:\n- Insufficient data for report generation.\n"
    
    # 追加图像信息
    if payload.heatmap_image_url:
        text += f"\nHeatmap: {payload.heatmap_image_url}\n"
    if payload.original_image_url:
        text += f"Original: {payload.original_image_url}\n"
    
    return {
        "success": True,
        "report_text": text.strip(),
        "generated_at": datetime.datetime.now().isoformat()
    }