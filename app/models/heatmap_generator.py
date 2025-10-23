"""
Heatmap generator that wraps the shared CheXpert DenseNet model.
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
from typing import Dict, Iterable, List, Tuple

from io import BytesIO

from loguru import logger
from PIL import Image
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

from app.chex_model import init_model as chex_init_model, infer as chex_infer
from app.config import settings
from app.models.schemas import ClassificationResult
from app.pathology_model import PathologyClassifier

# Keep class names aligned with the CheXpert training labels.
CHEXPERT_CLASS_NAMES: List[str] = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Lesion",
    "Lung Opacity",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

# Reuse the optional human readable descriptions maintained by the pathology model.
LABEL_TRANSLATIONS: Dict[str, str] = getattr(
    PathologyClassifier, "LABEL_TRANSLATIONS", {}
)

# Default inference parameters consistent with chex_model.infer defaults.
DEFAULT_THRESHOLD = 0.5
DEFAULT_ALPHA = 0.45


def _bytes_to_pil(data: bytes) -> Image.Image:
    """
    Decode raw bytes into a PIL image with the same DICOM handling as app.main._read_to_pil.
    """
    try:
        ds = pydicom.dcmread(BytesIO(data))
        arr = apply_voi_lut(ds.pixel_array, ds)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = 255 - arr
        return Image.fromarray(arr).convert("RGB")
    except Exception:
        return Image.open(BytesIO(data)).convert("RGB")


class HeatmapGenerator:
    """
    Provide classification scores and Grad-CAM heatmaps via the shared chex_model module.
    The DenseNet weights are large, so we initialise them only once per process.
    """

    _init_lock = threading.Lock()
    _model_ready = False

    def __init__(self) -> None:
        self._ensure_model_loaded()

    def _ensure_model_loaded(self) -> None:
        """Initialise chex_model once and cache the global module state."""
        if HeatmapGenerator._model_ready:
            return

        with HeatmapGenerator._init_lock:
            if HeatmapGenerator._model_ready:
                return

            weights_path = settings.MODEL_PATH
            if not os.path.exists(weights_path):
                raise FileNotFoundError(
                    f"CheXpert weight file not found: {weights_path}"
                )

            chex_init_model(
                model_path=weights_path,
                class_names=CHEXPERT_CLASS_NAMES,
                device="cpu",
                use_imagenet_norm=False,
            )

            HeatmapGenerator._model_ready = True
            logger.info("CheXpert model initialised successfully.")

    async def generate(self, image_path: str) -> Tuple[str, List[ClassificationResult]]:
        """
        Run inference asynchronously and return the heatmap path plus model predictions.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image does not exist: {image_path}")

        heatmap_path, results = await asyncio.to_thread(
            self._run_inference, image_path
        )
        return heatmap_path, results

    def _run_inference(self, image_path: str) -> Tuple[str, List[ClassificationResult]]:
        """
        Actual inference body executed inside a worker thread.
        """
        pil_image = self._load_image_with_dicom_support(image_path)

        infer_output = chex_infer(
            image=pil_image,
            generate_heatmap=True,
            threshold=DEFAULT_THRESHOLD,
            alpha=DEFAULT_ALPHA,
            return_top_k=None,
        )

        heatmap_path = self._save_heatmap_if_needed(
            infer_output.get("heatmap"), image_path
        )
        classifications = self._build_classifications(infer_output)
        logger.info(f"Top-3 probs: {sorted(infer_output['probs'].items(), key=lambda kv: kv[1], reverse=True)[:3]}")
        return heatmap_path, classifications

    def _load_image_with_dicom_support(self, image_path: str) -> Image.Image:
        """
        Load image bytes and convert to PIL.Image with DICOM specific handling.
        """
        with open(image_path, "rb") as fh:
            data = fh.read()
        return _bytes_to_pil(data)

    def _save_heatmap_if_needed(
        self, heatmap_image: Image.Image | None, fallback_path: str
    ) -> str:
        """
        Persist the generated heatmap into the uploads directory.
        When the model fails to produce a heatmap we fall back to the original file.
        """
        if heatmap_image is None:
            logger.warning(
                "CheXpert inference returned no heatmap; falling back to original image."
            )
            return fallback_path

        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        filename = f"heatmap_{int(time.time() * 1000)}.jpg"
        full_path = os.path.join(settings.UPLOAD_DIR, filename)
        try:
            heatmap_image.save(full_path, format="JPEG")
            logger.info("Heatmap saved to {}", full_path)
            return full_path
        except Exception as exc:  # pragma: no cover - best effort logging
            logger.error(
                "Failed to save heatmap ({}); returning original image path.", exc
            )
            return fallback_path

    def _build_classifications(
        self, infer_output: Dict
    ) -> List[ClassificationResult]:
        """
        Construct the API-friendly classification response list.
        We prioritise positive findings; fall back to the highest probabilities otherwise.
        """
        positives = infer_output.get("positive_findings") or []
        items: Iterable[Dict]
        if positives:
            items = positives
        else:
            probs = infer_output.get("probs") or {}
            items = (
                {"label": label, "confidence": float(prob)}
                for label, prob in sorted(
                    probs.items(), key=lambda kv: kv[1], reverse=True
                )
            )

        results: List[ClassificationResult] = []
        for item in items:
            label = item.get("label")
            confidence = float(item.get("confidence", 0.0))
            if not label:
                continue
            description = LABEL_TRANSLATIONS.get(label)
            results.append(
                ClassificationResult(
                    label=label,
                    confidence=confidence,
                    description=description,
                )
            )

        if not results:
            logger.warning("CheXpert inference returned no usable classification results.")

        return results
