# app/main.py —— 使用 chex_model 的真实签名（init_model(model_path, class_names, device)）
import os
import traceback
from io import BytesIO
from datetime import datetime
from typing import Optional, List, Dict

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from fastapi import Body
from pathlib import Path
import json
from datetime import datetime
import os
# 病理分类器
from app.pathology_model import PathologyClassifier

# CheXpert 模型（模块级全局：init_model 后，infer 直接用）
from .chex_model import init_model as chex_init_model, infer as chex_infer

app = FastAPI(title="Chexpert Backend")

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---------------- 静态目录 ----------------
os.makedirs("static/originals", exist_ok=True)
os.makedirs("static/heatmaps", exist_ok=True)
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception:
    pass

# ---------------- 通用读图 ----------------
def _read_to_pil(data: bytes) -> Image.Image:
    try:
        ds = pydicom.dcmread(BytesIO(data))
        arr = apply_voi_lut(ds.pixel_array, ds)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = 255 - arr
        return Image.fromarray(arr).convert("RGB")
    except Exception:
        return Image.open(BytesIO(data)).convert("RGB")

# ---------------- 权重绝对路径（按你提供的） ----------------
DEFAULT_PATHOLOGY_WEIGHTS = os.getenv(
    "PATHOLOGY_WEIGHTS",
    r"C:\Users\32068\Desktop\chexpert-new\chexpert\Chexpert_back\weights\pathology_model.pt"
)
DEFAULT_CHEXPERT_WEIGHTS = os.getenv(
    "CHEXPERT_WEIGHTS",
    r"C:\Users\32068\Desktop\chexpert-new\chexpert\Chexpert_back\weights\final_global_model.pth"
)

# ======================================================================
#                          Pathology
# ======================================================================
_pathology_model: Optional[PathologyClassifier] = None

@app.on_event("startup")
def _load_pathology():
    global _pathology_model
    try:
        _pathology_model = PathologyClassifier(DEFAULT_PATHOLOGY_WEIGHTS)
        print("[startup] Pathology model loaded:", DEFAULT_PATHOLOGY_WEIGHTS)
    except Exception as e:
        print(f"[startup] Pathology model load failed: {e}")
        traceback.print_exc()
        _pathology_model = None

@app.post("/api/v1/pathology/analyze")
async def pathology_analyze(
    file: UploadFile = File(...),
    top_k: Optional[int] = Form(None),
    min_prob: Optional[float] = Form(None),
):
    if _pathology_model is None:
        return {"success": False, "message": "Pathology model not loaded"}

    raw = await file.read()
    pil = _read_to_pil(raw)

    results = _pathology_model.predict(pil)

    if isinstance(min_prob, (int, float)):
        results = _pathology_model.filter_by_threshold(results, float(min_prob))
    if isinstance(top_k, (int, float)):
        results = results[: int(top_k)]

    classifications = [
        {
            "label": r.get("label"),
            "confidence": float(r.get("probability", r.get("confidence", 0.0))),
            "description": r.get("label_cn", ""),
        }
        for r in results
    ]

    return {
        "success": True,
        "classifications": classifications,
        "meta": {"model": "PathologyClassifier", "top_k": top_k, "min_prob": min_prob},
    }

@app.get("/api/v1/pathology/labels")
def pathology_labels():
    return {"labels": PathologyClassifier.PATHOLOGY_LABELS}

# ======================================================================
#                          CheXpert（使用 chex_model）
# ======================================================================
CHEXPERT14 = [
    "No Finding","Enlarged Cardiomediastinum","Cardiomegaly","Lung Lesion","Lung Opacity",
    "Edema","Consolidation","Pneumonia","Atelectasis","Pneumothorax",
    "Pleural Effusion","Pleural Other","Fracture","Support Devices"
]

# 标记是否加载成功（chex_model 内部有全局模型）
_chexpert_loaded = False

@app.on_event("startup")
def _load_chexpert():
    global _chexpert_loaded
    try:
        # chex_model.init_model 的真实签名：init_model(model_path, class_names=None, device=None, use_imagenet_norm=False)
        chex_init_model(
            model_path=DEFAULT_CHEXPERT_WEIGHTS,
            class_names=CHEXPERT14,
            device="cpu",
            use_imagenet_norm=False
        )
        _chexpert_loaded = True
        print("[startup] CheXpert model loaded:", DEFAULT_CHEXPERT_WEIGHTS)
    except Exception as e:
        print(f"[startup] CheXpert load failed: {e}")
        traceback.print_exc()
        _chexpert_loaded = False

def _save_image(img: Image.Image, subdir: str, suffix: str = ".jpg") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    rel = f"/static/{subdir}/{ts}{suffix}"
    os.makedirs(os.path.dirname("." + rel), exist_ok=True)
    img.save("." + rel)
    return rel

def _normalize_chexpert_output(out, top_k: Optional[int]) -> List[Dict]:
    # chex_infer 返回形如：
    # {
    #   "probs": {label: prob, ...},
    #   "preds": {label: 0/1, ...},
    #   "positive_findings": [{"label":..., "confidence":...}, ...],
    #   "heatmap": PIL.Image or None,
    # }
    cls: List[Dict] = []
    if isinstance(out, dict) and "probs" in out and isinstance(out["probs"], dict):
        items = sorted(out["probs"].items(), key=lambda x: x[1], reverse=True)
        if top_k: items = items[: int(top_k)]
        cls = [{"label": k, "confidence": float(v)} for k, v in items]
    if isinstance(out, dict) and "positive_findings" in out and isinstance(out["positive_findings"], list):
        # 优先展示阳性 Top-K（如果有）
        pos = out["positive_findings"]
        if top_k: pos = pos[: int(top_k)]
        if pos:
            cls = [{"label": i.get("label", "Unknown"), "confidence": float(i.get("confidence", 0.0))} for i in pos]
    return cls

@app.post("/api/v1/image/analyze")
@app.post("/api/v1/analyze")   # 可用别名
async def chexpert_analyze(
    file: UploadFile = File(...),
    generate_heatmap: bool = Form(True),
    threshold: float = Form(0.5),
    alpha: float = Form(0.45),
    return_top_k: Optional[int] = Form(10),
):
    if not _chexpert_loaded:
        return {"success": False, "message": "CheXpert model not loaded"}

    raw = await file.read()
    pil = _read_to_pil(raw)

    # 保存原图
    original_rel = _save_image(pil, "originals", ".jpg")

    # ⚠️ chex_infer 不接受 model 参数；直接传 image 等即可
    out = chex_infer(
        image=pil,
        generate_heatmap=generate_heatmap,
        threshold=threshold,
        alpha=alpha,
        return_top_k=return_top_k
    )

    classifications = _normalize_chexpert_output(out, return_top_k)

    # 处理热力图
    heatmap_rel = None
    if isinstance(out, dict) and out.get("heatmap") is not None:
        heat_img = out["heatmap"]
        try:
            heatmap_rel = _save_image(heat_img, "heatmaps", ".jpg")
        except Exception:
            heatmap_rel = None

    return {
        "success": True,
        "classifications": classifications,
        "heatmap_image_url": heatmap_rel,
        "original_image_url": original_rel,
        "meta": {
            "model": "CheXpert/chex_model",
            "threshold": threshold,
            "alpha": alpha,
            "top_k": return_top_k,
        },
    }

# ======================================================================
#                          健康检查
# ======================================================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "pathology_loaded": _pathology_model is not None,
        "chexpert_loaded": _chexpert_loaded,
        "weights": {
            "pathology": DEFAULT_PATHOLOGY_WEIGHTS,
            "chexpert": DEFAULT_CHEXPERT_WEIGHTS,
        },
    }


# 顶部缺少的话补充这些 import
import json
import uuid
from pathlib import Path
from fastapi import Body

# =========================
#        历史记录（文件存储）
# =========================
ROOT_DIR = Path(__file__).resolve().parent.parent  # Chexpert_back/
HISTORY_DIR = ROOT_DIR / "data"
HISTORY_FILE = HISTORY_DIR / "analysis_history.json"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)
if not HISTORY_FILE.exists():
    HISTORY_FILE.write_text("[]", encoding="utf-8")

def _load_history():
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []

def _save_history(items):
    HISTORY_FILE.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

@app.post("/api/v1/history/add")
def history_add(payload: dict = Body(...)):
    """
    保存一条历史记录。
    期望字段（前端可多可少，缺省均可）：
    - file_name: str
    - top1: str
    - confidence: float
    - diagnosis: list[ {label, confidence} ]   # Top-K
    - heatmap_url: str
    - original_url: str
    - status: str = "completed"
    - source: str = "chexpert"
    """
    items = _load_history()
    item = {
        "id": str(uuid.uuid4()),
        "ts": datetime.now().isoformat(timespec="seconds"),
        "file_name": payload.get("file_name") or "",
        "top1": payload.get("top1") or "",
        "confidence": float(payload.get("confidence") or 0),
        "diagnosis": payload.get("diagnosis") or [],
        "heatmap_url": payload.get("heatmap_url"),
        "original_url": payload.get("original_url"),
        "status": payload.get("status") or "completed",
        "source": payload.get("source") or "chexpert",
    }
    items.append(item)
    # 只保留最近 500 条（可按需修改）
    if len(items) > 500:
        items = items[-500:]
    _save_history(items)
    return {"ok": True, "item": item}

@app.get("/api/v1/history/list")
def history_list():
    items = _load_history()
    # 时间倒序返回
    items = sorted(items, key=lambda x: x.get("ts", ""), reverse=True)
    return {"items": items}

# ======= 静态报告目录 =======
REPORT_DIR = Path("static/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

@app.post("/api/v1/report/generate")
def generate_report(payload: dict = Body(...)):
    """
    生成一份简易 Markdown 报告（前端 LLaVA 报告页调用）
    允许的输入（可缺省）：
      - chexpert: {classifications: [{label, confidence}], heatmap_image_url, original_image_url, meta?...}
      - pathology: {classifications: [{label, confidence}], meta?...}
      - patient_info: {id, name, age, sex}  # 可选
      - notes: string                        # 可选：医生补充说明
    返回：
      { success, report_text, report_url }
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    title = f"AI Diagnostic Report {ts}"

    chex = payload.get("chexpert") or {}
    patho = payload.get("pathology") or {}
    pi    = payload.get("patient_info") or {}
    notes = (payload.get("notes") or "").strip()

    def top_k_lines(items, k=5):
        lines = []
        if isinstance(items, list):
            items = sorted(items, key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
            for it in items[:k]:
                label = it.get("label", "Unknown")
                conf  = float(it.get("confidence", 0.0))
                lines.append(f"- {label}: {conf:.1%}")
        return "\n".join(lines) if lines else "- (None)"

    chex_lines  = top_k_lines(chex.get("classifications") or [], 5)
    patho_lines = top_k_lines(patho.get("classifications") or [], 5)

    heatmap_url  = chex.get("heatmap_image_url") or chex.get("heatmapUrl") or ""
    original_url = chex.get("original_image_url") or chex.get("originalImageUrl") or ""

    # --- 生成 Markdown ---
    md = []
    md.append(f"# {title}\n")
    md.append(f"**Generated at:** {datetime.now().isoformat(timespec='seconds')}\n")
    if pi:
        md.append("## Patient Information")
        md.append(f"- ID: {pi.get('id','-')}")
        md.append(f"- Name: {pi.get('name','-')}")
        md.append(f"- Age/Sex: {pi.get('age','-')} / {pi.get('sex','-')}\n")

    md.append("## CheXpert Top Findings")
    md.append(chex_lines + "\n")

    md.append("## Pathology Top Findings")
    md.append(patho_lines + "\n")

    if notes:
        md.append("## Physician Notes")
        md.append(notes + "\n")

    if original_url:
        md.append("## Original Image")
        md.append(f"![original]({original_url})\n")
    if heatmap_url:
        md.append("## Heatmap")
        md.append(f"![heatmap]({heatmap_url})\n")

    md_text = "\n".join(md)

    # 保存到 static/reports
    out_md = REPORT_DIR / f"report_{ts}.md"
    out_md.write_text(md_text, encoding="utf-8")

    return {
        "success": True,
        "report_text": md_text,
        "report_url": f"/static/reports/{out_md.name}",
    }