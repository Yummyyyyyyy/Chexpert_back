"""
GLM-4V + RAG 模型服务 - 模板实现
提供基础结构与占位实现，便于后续接入真实模型与检索。
"""
from typing import Tuple, Optional, List, Dict
from loguru import logger
from datetime import datetime
from pathlib import Path
import os
import time

from app.config import settings

# Prefer the user's real rag.py if provided; fall back to local stub
try:
    # If you place your real rag.py under app/services, this will resolve
    from app.services.rag import (
        LABEL2PHRASE,
        probs_to_query_text,
        text_search,
        fetch_evidences_by_rids,
        search_wiki,
        format_rag_context_en,
        format_external_refs_en,
    )
except Exception:  # pragma: no cover
    LABEL2PHRASE = {}
    def probs_to_query_text(probs: Dict[str, float], view: Optional[str] = None) -> str:
        return ", ".join([f"{k}:{v:.2f}" for k, v in probs.items()])
    def text_search(query: str, kb_dir: str, topk: int, encoder: str):
        return []
    def fetch_evidences_by_rids(rids, facts_path: str):
        return []
    def search_wiki(query: str, wiki_idx_dir: str, wiki_topk: int, encoder: str):
        return []
    def format_rag_context_en(meta: Dict, clf: Dict[str, float], evidences: List[Dict]):
        top = ", ".join([f"{k}:{v:.2f}" for k, v in list(clf.items())[:6]]) if clf else "N/A"
        return f"TASK: IMPRESSION.\nMETA: {meta}\nCLASSIFIER: {top}\nEVIDENCES: {len(evidences)} items"
    def format_external_refs_en(wiki_refs: List[Dict]):
        return "\n".join([f"[W{i+1}] {w.get('title','')}: {w.get('snippet','')}" for i, w in enumerate(wiki_refs)])


class GLM4VRAGService:
    """GLM-4V + RAG 服务"""

    def __init__(self):
        self.api_url = getattr(settings, 'GLM4V_API_URL', None)
        self.api_timeout = getattr(settings, 'API_TIMEOUT', 120)
        self.rag_index_path = getattr(settings, 'RAG_INDEX_PATH', None)
        self.rag_embedding_model = getattr(settings, 'RAG_EMBEDDING_MODEL', None)

    async def generate_report(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        pathology_labels: Optional[List[str]] = None,
        rag_query: Optional[str] = None,
        classifier_probs: Optional[Dict[str, float]] = None,
        top_k: int = 5,
    ) -> Tuple[str, float]:
        """
        只使用 KB + Wiki 的组合 RAG 方式构建提示，调用 GLM-4V 生成报告。
        如果缺少检索资源或 API Key，将自动回退到本地占位输出。
        """
        image_path = (image_path or "").lstrip('/')

        start_time = time.time()
        logger.info("🧠 启动 GLM-4V + RAG (KB + Wiki) 报告生成流程")

        # ---- Build query ----
        # Prefer explicit rag_query; otherwise derive from pathology labels
        logger.info("Incoming rag_query: {}", rag_query)
        if rag_query and rag_query.strip():
            query = rag_query.strip()
        elif classifier_probs:
            try:
                query = probs_to_query_text(classifier_probs, view=None)
            except Exception:
                query = ", ".join(f"{k}:{v:.2f}" for k, v in classifier_probs.items())
        elif pathology_labels:
            # Create a pseudo probs dict with constant weights to reuse probs_to_query_text interface
            pseudo_probs = {lbl: 1.0 for lbl in pathology_labels[: max(1, min(len(pathology_labels), 6))]}
            query = probs_to_query_text(pseudo_probs, view=None)
        else:
            query = ""
        logger.info("Resolved RAG query: {}", query)

        # ---- Retrieve KB and Wiki ----
        kb_evidences = []
        wiki_refs = []

        try:
            if query and settings.RAG_KB_DIR and settings.RAG_FACTS_PATH:
                kb_rids = text_search(query, settings.RAG_KB_DIR, top_k, settings.RAG_ENCODER)
                if kb_rids:
                    kb_evidences = fetch_evidences_by_rids(kb_rids, settings.RAG_FACTS_PATH)
            if query and settings.RAG_WIKI_IDX:
                wiki_refs = search_wiki(query, settings.RAG_WIKI_IDX, getattr(settings, 'RAG_WIKI_TOPK', 4), settings.RAG_ENCODER)
        except Exception as e:  # keep robust
            logger.warning(f"RAG 检索失败，使用空上下文: {e}")
            kb_evidences, wiki_refs = [], []

        # ---- Build KB + Wiki system text ----
        # Build a light-weight meta and classifier dict from labels
        meta = {"source": "CheXpert", "image_path": image_path}
        if classifier_probs:
            clf = {
                label: max(0.0, min(1.0, float(prob)))
                for label, prob in classifier_probs.items()
                if label
            }
        else:
            clf = {lbl: 1.0 for lbl in (pathology_labels or [])}

        system_kb_plus_wiki = format_rag_context_en(meta, clf, kb_evidences)
        if wiki_refs:
            system_kb_plus_wiki = (
                system_kb_plus_wiki
                + "\n\nEXTERNAL REFERENCES (Wikipedia; do not override patient-specific evidence; "
                  "use only to support wording or general priors):\n\n"
                + format_external_refs_en(wiki_refs)
            )

        # user_instruction = (
        #     "Please produce the IMPRESSION (English, 1–4 bullets) with inline citations [W#] when Wikipedia supports wording."
        # )

        user_instruction = (
            "Please refer to the provided evidence and Wiki reference, "
            "and generate the chest X-ray IMPRESSION according to the instructions above. "
            "Produce two versions:\n"
            "1. **Professional Version** – concise, clinically precise, and suitable for a radiology report.\n"
            "2. **Patient-Friendly Version** – clear, explanatory, and easy for non-medical readers to understand, "
            "including simple explanations of causes, symptoms, and common treatments where relevant.\n"
            "Ensure both versions are grounded in the evidence and consistent with each other."
        )


        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_kb_plus_wiki}]},
            {"role": "user", "content": [{"type": "text", "text": user_instruction}]},
        ]

        if settings.DEBUG:
            logger.info(f"RAG results — evidences: {len(kb_evidences)}, wiki: {len(wiki_refs)}")
            pretty = self._format_messages_for_debug(messages)
            logger.info("\n================ GLM-4V PROMPT (KB+Wiki) ================\n" + pretty + "\n" + "="*64)

        # ---- Call GLM-4V ----
        report_text: Optional[str] = None
        try:
            if settings.ZHIPU_API_KEY:
                report_text = self._call_glm(messages, model=settings.GLM4V_MODEL, temperature=settings.GLM4V_TEMPERATURE)
        except Exception as e:
            logger.warning(f"GLM-4V 远程调用失败，将回退到占位输出: {e}")

        # Fallback placeholder if no API key or call failed
        if not report_text:
            report_text = self._compose_placeholder_report(user_instruction, [ev.get("snippet") or ev.get("text") or "" for ev in kb_evidences] + [r.get("snippet") or r.get("text") or "" for r in wiki_refs])

        processing_time = time.time() - start_time
        self._save_report(report_text, image_path, pathology_labels)
        logger.success(f"✅ GLM-4V + RAG (KB+Wiki) 报告生成完成，用时 {processing_time:.2f}s")
        return report_text, processing_time

    def _call_glm(self, messages: List[Dict], model: str, temperature: float) -> str:
        """调用 ZhipuAI Chat Completions（延迟导入以避免无依赖时报错）。"""
        try:
            from zhipuai import ZhipuAI
        except Exception as e:
            raise RuntimeError("缺少 zhipuai 依赖，请在后端环境安装：pip install zhipuai") from e

        client = ZhipuAI(api_key=settings.ZHIPU_API_KEY)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        content = resp.choices[0].message.content
        # zhipuai SDK returns list[dict] or str depending on version; normalize
        if isinstance(content, list):
            # concatenate text parts
            text_chunks = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_chunks.append(block.get("text") or "")
            return "\n".join([t for t in text_chunks if t]) or str(content)
        return content or ""

    def _mock_retrieval(self, query: Optional[str], top_k: int) -> List[str]:
        if not query:
            return []
        return [f"Evidence {i+1}: Standard chest radiograph patterns and differentials." for i in range(max(0, top_k))]

    def _format_messages_for_debug(self, messages, max_chars=5000):
        import json as _json
        lines = []
        for i, msg in enumerate(messages, 1):
            role = msg.get("role", "?")
            lines.append(f"--- Message #{i} | role: {role} ---")
            for j, block in enumerate(msg.get("content", []), 1):
                if block.get("type") == "text":
                    txt = block.get("text", "") or ""
                    if len(txt) > max_chars:
                        txt = txt[:max_chars] + "…"
                        lines.append(f"[{j}] {txt}")
                    elif block.get("type") == "image_url":
                        url = (block.get("image_url") or {}).get("url")
                        lines.append(f"[{j}] [image_url] {url}")
                    else:
                        lines.append(f"[{j}] {_json.dumps(block, ensure_ascii=False)}")
        return "\n".join(lines)
    
    
    def _compose_placeholder_report(self, prompt: str, contexts: List[str]) -> str:
        """拼装占位报告文本"""
        ctx_block = "\n".join(f"- {c}" for c in contexts) if contexts else "(no retrieved contexts)"
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return (
            f"Generated: {now}\n"
            f"Model: GLM-4V (RAG-enabled placeholder)\n"
            f"Prompt: {prompt}\n\n"
            f"Retrieved Contexts:\n{ctx_block}\n\n"
            "FINDINGS:\n"
            "- Cardiomediastinal contours within normal limits.\n"
            "- Lungs are clear without focal consolidation.\n"
            "- No pleural effusion or pneumothorax identified.\n\n"
            "IMPRESSION:\n"
            "- No acute cardiopulmonary process.\n\n"
            "SUMMARY:\n"
            "- Unremarkable chest radiograph; correlate clinically if symptoms persist.\n"
        )

    def _save_report(self, report_text: str, image_path: str, pathology_labels: Optional[List[str]] = None) -> Optional[str]:
        """保存报告到本地目录"""
        try:
            reports_dir = Path("reports/glm4v_rag")
            reports_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = os.path.basename(image_path).split('.')[0] if image_path else "image"
            report_filename = f"{timestamp}_{image_filename}.txt"
            report_path = reports_dir / report_filename

            lines = []
            lines.append("=" * 80)
            lines.append("GLM-4V + RAG Medical Report (Placeholder)")
            lines.append("=" * 80)
            lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"Image: {image_path}")
            if pathology_labels:
                lines.append(f"Detected Pathologies: {', '.join(pathology_labels)}")
            lines.append("" )
            lines.append(report_text)
            lines.append("" )
            lines.append("=" * 80)

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(lines))

            logger.info(f"📁 报告已保存: {report_path}")
            return str(report_path)
        except Exception as e:
            logger.error(f"保存报告失败: {str(e)}")
            return None


# 全局单例
_glm4v_rag_service_instance: Optional[GLM4VRAGService] = None


def get_glm4v_rag_service() -> GLM4VRAGService:
    global _glm4v_rag_service_instance
    if _glm4v_rag_service_instance is None:
        _glm4v_rag_service_instance = GLM4VRAGService()
    return _glm4v_rag_service_instance
