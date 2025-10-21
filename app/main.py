# app/main.py
# -*- coding: utf-8 -*-
import os, io, uuid, datetime, time, json
from typing import Optional, List

import numpy as np
from PIL import Image

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Body
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch
from pydantic import BaseModel

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
        PATHOLOGY_MODEL_PATH = "pathology_model.pt"
    settings = _S()

# =========================
# 路径与静态目录
# =========================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STATIC_ROOT = os.path.join(ROOT, "static")
UPLOAD_DIR = os.path.join(STATIC_ROOT, "uploads")
HEATMAP_DIR = os.path.join(STATIC_ROOT, "heatmaps")
DATA_DIR = os.path.join(ROOT, "data")
HISTORY_FILE = os.path.join(DATA_DIR, "analysis_history.jsonl")

for d in (STATIC_ROOT, UPLOAD_DIR, HEATMAP_DIR, DATA_DIR):
    os.makedirs(d, exist_ok=True)

# =========================
# 创建应用 & CORS
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
from .chex_model import init_model, infer as model_infer

MODEL_PATH = getattr(settings, "MODEL_PATH", "final_global_model.pth")

CHEX_CLASSES: List[str] = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Lesion", "Lung Opacity", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices"
]

_MODEL, _CLASS_NAMES = None, None
try:
    _MODEL, _CLASS_NAMES = init_model(MODEL_PATH, class_names=CHEX_CLASSES, use_imagenet_norm=False)
    print("[app] ✅ 热力图模型加载成功")
except Exception as e:
    print("[app] ⚠️ 热力图模型加载失败：", e)

# =========================
# 病症标签模型
# =========================
_PATHOLOGY_MODEL = None
try:
    from .pathology_model import PathologyClassifier
    PATHOLOGY_MODEL_PATH = getattr(settings, "PATHOLOGY_MODEL_PATH", "pathology_model.pt")
    if os.path.exists(PATHOLOGY_MODEL_PATH):
        _PATHOLOGY_MODEL = PathologyClassifier(PATHOLOGY_MODEL_PATH)
        print("[app] ✅ 病症标签模型加载成功")
    else:
        print(f"[app] ⚠️ 病症标签模型文件不存在: {PATHOLOGY_MODEL_PATH}")
except Exception as e:
    print("[app] ⚠️ 病症标签模型加载失败：", e)

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
        return Image.fromarray(arr).convert("RGB")
    return Image.open(io.BytesIO(data)).convert("RGB")

def _append_history(item: dict):
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

def _read_history() -> List[dict]:
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    return list(reversed(lines))

# =========================
# 健康检查
# =========================
@app.get("/health")
def health():
    return JSONResponse({
        "status": "ok" if (_MODEL or _PATHOLOGY_MODEL) else "not_ready",
        "services": {
            "heatmap": bool(_MODEL),
            "pathology": bool(_PATHOLOGY_MODEL),
        },
        "device": "cuda:0" if torch.cuda.is_available() else "cpu"
    })

# =========================
# 历史记录
# =========================
@app.get("/api/v1/history")
def get_history(page: int = 1, page_size: int = 10):
    data = _read_history()
    total = len(data)
    return {
        "success": True,
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": data[(page - 1) * page_size: page * page_size],
    }

# =========================
# 热力图推理接口
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
        raise HTTPException(424, "Heatmap model not loaded")
    pil = _to_pil_from_upload(file)
    t0 = time.time()
    out = model_infer(pil, generate_heatmap, threshold, alpha, return_top_k)
    infer_ms = int((time.time() - t0) * 1000)

    day = datetime.date.today().isoformat()
    up_dir = os.path.join(UPLOAD_DIR, day)
    hm_dir = os.path.join(HEATMAP_DIR, day)
    os.makedirs(up_dir, exist_ok=True)
    os.makedirs(hm_dir, exist_ok=True)
    uid = uuid.uuid4().hex[:8]

    orig_path = os.path.join(up_dir, f"{uid}.png")
    pil.save(orig_path)
    original_url = f"{request.base_url}static/uploads/{day}/{uid}.png".rstrip("/")

    heatmap_url = None
    if generate_heatmap and out.get("heatmap"):
        hm_path = os.path.join(hm_dir, f"{uid}.png")
        out["heatmap"].save(hm_path)
        heatmap_url = f"{request.base_url}static/heatmaps/{day}/{uid}.png".rstrip("/")

    _append_history({
        "date": day,
        "file_name": file.filename or f"{uid}.png",
        "diagnosis": out["positive_findings"][0]["label"] if out.get("positive_findings") else "-",
        "confidence": out["positive_findings"][0]["confidence"] if out.get("positive_findings") else 0.0,
        "status": "completed",
        "heatmap_url": heatmap_url,
        "original_url": original_url,
    })

    return {
        "success": True,
        "classifications": out.get("positive_findings", []),
        "heatmap_image_url": heatmap_url,
        "original_image_url": original_url,
        "meta": {"inference_time_ms": infer_ms}
    }

# =========================
# 病症标签分析接口
# =========================
@app.post("/api/v1/pathology/analyze")
def analyze_pathology(request: Request, file: UploadFile = File(...)):
    if _PATHOLOGY_MODEL is None:
        raise HTTPException(424, "Pathology model not loaded")
    pil = _to_pil_from_upload(file)
    t0 = time.time()
    pathologies = _PATHOLOGY_MODEL.predict(pil)
    infer_ms = int((time.time() - t0) * 1000)
    return {
        "success": True,
        "pathologies": pathologies,
        "meta": {"inference_time_ms": infer_ms}
    }

# =========================
# 获取病症标签列表
# =========================
@app.get("/api/v1/pathology/labels")
def get_pathology_labels():
    if _PATHOLOGY_MODEL is None:
        raise HTTPException(424, "Pathology model not loaded")
    return {"labels": _PATHOLOGY_MODEL.PATHOLOGY_LABELS}

# =========================
# 报告生成接口
# =========================
class ReportIn(BaseModel):
    patient_id: str | None = None
    classifications: list[dict] = []
    pathologies: list[dict] = []
    heatmap_image_url: str | None = None
    original_image_url: str | None = None

@app.post("/api/v1/report/generate")
def generate_report(payload: ReportIn = Body(...)):
    if payload.pathologies:
        main_label = payload.pathologies[0]["label"]
        main_prob = payload.pathologies[0]["probability"]
        text = f"Impression:\n- Findings most consistent with {main_label} ({main_prob:.2%})"
    elif payload.classifications:
        main_label = payload.classifications[0]["label"]
        main_prob = payload.classifications[0]["confidence"]
        text = f"Impression:\n- Findings most consistent with {main_label} ({main_prob:.2%})"
    else:
        text = "Impression:\n- No abnormal findings."

    return {"success": True, "report_text": text}
