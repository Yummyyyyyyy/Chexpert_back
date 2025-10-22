# -*- coding: utf-8 -*-
"""
Utility functions for CheXpert-Plus retrieval and prompt construction.

This file exposes:
- LABEL2PHRASE
- probs_to_query_text(...)
- text_search(...)
- fetch_evidences_by_rids(...)
- search_wiki(...)
- format_rag_context_en(...)
- format_external_refs_en(...)

You can import these in main.py to run four generation modes:
No RAG, KB-only, Wiki-only, KB+Wiki.
"""

import os
import json
import pickle

# ---------- Label → short phrase (used to compose semantic queries) ----------
LABEL2PHRASE = {
    "No Finding": "no acute cardiopulmonary abnormality",
    "Enlarged Cardiomediastinum": "enlarged cardiomediastinum",
    "Cardiomegaly": "cardiomegaly",
    "Lung Opacity": "air-space opacity",
    "Lung Lesion": "lung lesion",
    "Edema": "pulmonary edema",
    "Consolidation": "consolidation",
    "Pneumonia": "pneumonia",
    "Atelectasis": "atelectasis",
    "Pneumothorax": "pneumothorax",
    "Pleural Effusion": "pleural effusion",
    "Pleural Other": "pleural abnormality",
    "Fracture": "fracture",
    "Support Devices": "lines and tubes"
}

# ---------- Build semantic query from classifier probabilities ----------
def probs_to_query_text(probs: dict, view: str = None, t_pos=0.60, t_neg=0.10, max_pos=3) -> str:
    """
    Turn classifier probabilities into a concise semantic query.
    - Positive (>= t_pos): add "<phrase> present"
    - Key negatives (<= t_neg): add "no <phrase>" (for pneumothorax/pleural effusion)
    - Uncertain (between t_neg and t_pos for a few key labels): add "possible <phrase>"
    """
    include_key_negs = ("Pneumothorax", "Pleural Effusion")
    pos, neg, unc = [], [], []

    for lab, p in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        phrase = LABEL2PHRASE.get(lab, lab.lower())
        if p >= t_pos:
            if lab == "Support Devices":
                pos.append("support devices present; line position stable")
            else:
                pos.append(f"{phrase} present")
        elif p <= t_neg and lab in include_key_negs:
            neg.append(f"no {phrase}")
        else:
            if lab in ("Pleural Effusion", "Pneumothorax", "Consolidation", "Atelectasis") and t_neg < p < t_pos:
                unc.append(f"possible {phrase}")

    pos = pos[:max_pos]
    parts = ["; ".join(x) for x in (pos, neg, unc) if x]
    if view:
        parts.append(f"{view} view")
    return "; ".join(parts)

# ---------- Dataset KB dense retrieval ----------
def text_search(query, kb_dir, topk=8, encoder="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Search your dataset KB (reports) by dense retrieval.
    Requires:
      kb_dir/faiss.index
      kb_dir/id_map.pkl   (each row dict contains {"offset": int, "report_id": str})
    Returns a list of report_ids in ranked order.
    """
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np

    if not os.path.exists(os.path.join(kb_dir, "faiss.index")):
        return []

    model = SentenceTransformer(encoder)
    q = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0].astype("float32")

    index = faiss.read_index(os.path.join(kb_dir, "faiss.index"))
    with open(os.path.join(kb_dir, "id_map.pkl"), "rb") as f:
        id_map = pickle.load(f)
    id2rid = {row["offset"]: row["report_id"] for row in id_map}

    D, I = index.search(q[None, :], topk)
    return [id2rid[i] for i in I[0] if i in id2rid]

def fetch_evidences_by_rids(rids, facts_path):
    """
    Collect evidence objects (key sentences + facts) for the given report IDs.
    facts_path should point to reports_facts.jsonl.
    """
    rid_set = set(rids)
    evidences = []
    order = {rid: i for i, rid in enumerate(rids)}

    if not os.path.exists(facts_path):
        return evidences

    with open(facts_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rid = d.get("report_id")
            if rid in rid_set:
                evidences.append({
                    "report_id": rid,
                    "section": d.get("section", ""),
                    "key_sentences": d.get("impression_key_sentences", []),
                    "facts": d.get("facts", [])
                })
    evidences.sort(key=lambda x: order.get(x["report_id"], 1e9))
    return evidences

# ---------- Wikipedia retrieval ----------
def search_wiki(query, wiki_dir, topk=4, encoder="sentence-transformers/all-MiniLM-L6-v2", max_chars=700):
    """
    Search Wikipedia index (built from your wiki_texts/*.txt).
    wiki_dir must contain:
      - faiss.index
      - id_map.pkl (with {"offset": int, "doc_id": str})
      - chunks.jsonl (records: {"doc_id","title","source_url","text",...})
    Returns: list of refs [{ref_id,title,url,snippet,license}]
    """
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np

    if not wiki_dir or not os.path.exists(os.path.join(wiki_dir, "faiss.index")):
        return []

    model = SentenceTransformer(encoder)
    q = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0].astype("float32")

    index = faiss.read_index(os.path.join(wiki_dir, "faiss.index"))
    with open(os.path.join(wiki_dir, "id_map.pkl"), "rb") as f:
        id_map = pickle.load(f)
    id2doc = {row["offset"]: row["doc_id"] for row in id_map}

    chunks = {}
    with open(os.path.join(wiki_dir, "chunks.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            chunks[d["doc_id"]] = d

    D, I = index.search(q[None, :], topk)
    refs = []
    for rank, off in enumerate(I[0], 1):
        did = id2doc.get(off)
        d = chunks.get(did)
        if d:
            snip = d["text"]
            if len(snip) > max_chars:
                snip = snip[:max_chars] + "…"
            refs.append({
                "ref_id": f"W{rank}",
                "title": d.get("title", ""),
                "url": d.get("source_url", ""),
                "snippet": snip,
                "license": d.get("license", "CC BY-SA 4.0 (Wikipedia)")
            })
    return refs

# ---------- Formatting helpers for prompts ----------
def format_rag_context_en(meta, clf, evidences):
    """
    Create a strict system prompt that forces grounding to retrieved evidences (dataset KB).
    """
    lines = []
    # lines.append(
    #     "TASK: Generate a concise chest X-ray IMPRESSION. "
    #     "Ground every claim in the evidence below. If evidence is conflicting or insufficient, "
    #     "use hedging (e.g., 'may represent', 'recommend clinical correlation' or 'comparison with prior'). "
    #     "Output 1–4 bullet points; be precise and avoid speculation."
    # )
    # lines.append(
    #     "TASK: Generate two versions of the chest X-ray IMPRESSION based on the evidence below. "
    #     "1. **Professional Version**: Write a concise, structured, and clinically precise impression suitable for a radiology report. "
    #     "Ground every statement in the evidence. Use formal radiologic phrasing (e.g., 'consistent with', 'may represent', 'no evidence of'). "
    #     "Output 2–5 bullet points, each summarizing a key finding with its severity, anatomic context, and interval change if applicable. "
    #     "Avoid speculation and remain objective. "
    #     "2. **Patient-Friendly Version**: Rewrite the impression in clear, plain language that a non-medical reader can understand. "
    #     "Explain what the findings mean in everyday terms and what their significance might be, avoiding medical jargon. "
    #     "Use 3–6 bullet points or short paragraphs, each 1–2 sentences long, giving enough context for the patient to understand the situation. "
    #     "If findings are uncertain, express this gently (e.g., 'the image suggests that...', 'this could mean...', 'your doctor may want to compare with previous scans'). "
    #     "Both versions must remain consistent with the same underlying evidence."
    # )
    lines.append(
        "TASK: Generate two versions of the chest X-ray IMPRESSION based on the evidence below. "
        "1. **Professional Version**: Write a concise, structured, and clinically precise impression suitable for a radiology report. "
        "Ground every statement in the evidence. Use formal radiologic phrasing (e.g., 'consistent with', 'may represent', 'no evidence of'). "
        "Output 2–5 bullet points, each summarizing a key finding with its severity, if applicable. "
        "Avoid speculation and remain objective. "
        "2. **Patient-Friendly Version**: Rewrite the impression in clear, plain language that a non-medical reader can understand. "
        "For each finding, explain: (a) what it means in simple terms, (b) possible common causes or contributing factors, "
        "(c) typical symptoms a person might experience, and (d) common management or treatment approaches (if relevant). "
        "Use neutral and reassuring tone—avoid alarmist wording and emphasize that this summary is for understanding, not diagnosis. "
        "If findings are uncertain, express this gently (e.g., 'the image suggests that...', 'this could mean...', 'your doctor may want to compare with previous scans'). "
        "Each explanation should be 2–3 sentences long, informative yet easy to read. "
        "Both versions must remain consistent with the same underlying evidence."
    )


    if meta:
        lines.append(f"PATIENT META: {json.dumps(meta, ensure_ascii=False)}")
    if clf:
        top = sorted(clf.items(), key=lambda x: x[1], reverse=True)[:6]
        top_str = ", ".join([f"{k}: {v:.2f}" for k, v in top])
        lines.append(f"CLASSIFIER (Top): {top_str}")

    for i, e in enumerate(evidences, 1):
        lines.append(f"EVIDENCE #{i} | report_id={e.get('report_id')} | section={e.get('section','')}")
        for s in e.get("key_sentences") or []:
            lines.append(f"  - key sentence: {s}")
        for f in e.get("facts") or []:
            obs = f.get("obs", "")
            pol = f.get("polarity", "")
            anat = f.get("anat") or ""
            mods = ", ".join(f.get("modifiers") or [])
            seg = f"  - fact: {obs} [{pol}]"
            if anat:
                seg += f" @ {anat}"
            if mods:
                seg += f" ({mods})"
            lines.append(seg)

    lines.append(
        "OUTPUT REQUIREMENTS: "
        "• Use bullet points. "
        "• Explicitly indicate present/absent/uncertain findings. "
        "• Do not introduce information not supported by the evidence."
    )
    return "\n".join(lines)

def format_external_refs_en(refs, max_chars=700):
    """
    Convert wiki refs into a citation block for system prompt.
    """
    out = []
    for r in refs:
        snip = r.get("snippet", "")
        if len(snip) > max_chars:
            snip = snip[:max_chars] + "…"
        out.append(
            f"[{r['ref_id']}] {r['title']} — {r['url']} (Wikipedia, CC BY-SA 4.0)\n"
            f"Snippet: {snip}"
        )
    return "\n\n".join(out)
