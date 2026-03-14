from __future__ import annotations

"""
generator.py — File 2 pipeline (converted from file2_v4.ipynb)
Handles: JD embedding, ChromaDB retrieval (READ-ONLY), gap analysis,
         LLM generation, critic loop, hallucination guard, ATS scoring.

ChromaDB is READ-ONLY here. Run ingest_to_chroma.py separately to populate it.
ChromaDB path: Resume_Builder/chroma_db  (3 levels above this file)
"""

# pathlib.Path needed immediately in CONFIG
from pathlib import Path

# ────────────────────────────────────────────────────────────────
# SYSTEM CONFIGURATION
# ────────────────────────────────────────────────────────────────

CONFIG = {
    # LLM inference (Ollama local models)
    "ollama_model":    "mistral:7b-instruct-q4_K_M",   # primary
    "fallback_model":  "phi3:mini",                     # VRAM fallback

    # Embedding model (must match File 1 and ingest_to_chroma.py)
    "embedding_model": "BAAI/bge-base-en-v1.5",

    # ChromaDB retrieval  ← must match ingest_to_chroma.py COLLECTION_NAME
    "chroma_collection": "market_reference",
    "chroma_db_path":    str(Path(__file__).resolve().parent.parent.parent / "chroma_db"),
    "top_k_chunks":      5,

    # Source data for ingestion (read-only, never modified)
    "chunks_file":       "./preprocessed_chunks.json",

    # Chunking (must match File 1 for consistency)
    "chunk_size":    300,
    "chunk_overlap": 50,

    # Generation / critic loop
    "max_retries":              3,     # max critic-loop iterations
    "critic_pass_threshold":    72,    # min alignment score (0–100) to accept draft
    "ollama_timeout":           240,   # base timeout per Ollama call (seconds)
    "ollama_timeout_attempts":  3,     # progressive retry attempts on timeout
    "ollama_timeout_max":       900,   # upper cap for progressive timeout retries

    # Hallucination guard
    "hallucination_threshold":  0.30,  # max fraction of flagged sentences before forced regen
    "sentence_sim_min":         0.40,  # per-sentence cosine sim floor vs original resume

    # ATS scoring weights (must sum to 1.0)
    "ats_weights": {
        "keyword_coverage":   0.40,
        "embedding_sim":      0.35,
        "section_completeness": 0.25,
    },

    # File 1 artifact paths (relative to this notebook)
    "artifacts": {
        "resume_embeddings":  "./resume_embeddings.npy",
        "structured_resume":  "./structured_resume.json",
        "pii_vault":          "./pii_vault.json",
    },

    # Output paths
    "output": {
        "gap_report":         "gap_report.json",
        "generated_draft":    "generated_resume_draft.txt",
        "final_resume":       "final_resume.txt",
        "ats_report":         "ats_report.json",
    },

    "log_level": "DEBUG",
}

print("✓ CONFIG loaded.")
print(f"  Primary model      : {CONFIG['ollama_model']}")
print(f"  Fallback model     : {CONFIG['fallback_model']}")
print(f"  Embedding model    : {CONFIG['embedding_model']}")
print(f"  ChromaDB path      : {CONFIG['chroma_db_path']}")
print(f"  ChromaDB collection: {CONFIG['chroma_collection']}")
print(f"  Chunks file        : {CONFIG['chunks_file']}")
print(f"  Top-K chunks       : {CONFIG['top_k_chunks']}")
print(f"  Max retries        : {CONFIG['max_retries']}")
print(f"  Ollama timeout     : {CONFIG['ollama_timeout']}s")
print(f"  Timeout attempts   : {CONFIG['ollama_timeout_attempts']}")
print(f"  Hallucination thr. : {CONFIG['hallucination_threshold']}")


# ────────────────────────────────────────────────────────────────
# IMPORTS & LOGGING
# ────────────────────────────────────────────────────────────────

import json
import logging
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import requests

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

try:
    import chromadb
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False

# ── PII-safe log formatter ────────────────────────────────────────

_PII_LOG_PATTERNS = [
    re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'),
    re.compile(r'https?://\S+', re.IGNORECASE),
    re.compile(r'\b\+?[\d\s\-().]{7,20}\b'),
]

class _PIISafeFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        for pat in _PII_LOG_PATTERNS:
            msg = pat.sub("[REDACTED]", msg)
        return msg

def _setup_logger(name: str = "file2_pipeline", level: str = "DEBUG") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.DEBUG))
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(_PIISafeFormatter(
        fmt="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(h)
    logger.propagate = False
    return logger

log = _setup_logger(level=CONFIG.get("log_level", "DEBUG"))

log.info("Imports complete. chromadb=%s  sentence-transformers=%s  nltk=%s",
         _CHROMA_AVAILABLE, _ST_AVAILABLE, _NLTK_AVAILABLE)

# ────────────────────────────────────────────────────────────────
# LOAD FILE 1 ARTIFACTS
# ────────────────────────────────────────────────────────────────

def _resolve(key: str, config: dict) -> Path:
    raw = config["artifacts"][key]
    p = Path(raw)
    if not p.is_absolute():
        # Resolve relative to this notebook's directory
        p = (Path(__file__).parent / raw).resolve() if "__file__" in dir() else (Path(raw)).resolve()
    return p


def load_artifacts(config: dict | None = None) -> dict:
    """
    Load all File 1 outputs from disk.

    Returns
    -------
    dict with keys:
        resume_embedding  : np.ndarray  shape (768,)
        structured_resume : dict
        vault             : dict[str, str]  token → original
    """
    cfg = config or CONFIG
    arts = cfg["artifacts"]

    results: dict[str, Any] = {}

    # ── resume_embeddings.npy ─────────────────────────────────────
    emb_path = Path(arts["resume_embeddings"])
    if not emb_path.exists():
        raise FileNotFoundError(
            f"resume_embeddings.npy not found at '{emb_path}'. "
            "Run File 1 first to generate embeddings."
        )
    resume_emb = np.load(str(emb_path))
    if resume_emb.ndim == 2:
        # Full-doc embedding stored as (1, 768) — squeeze
        resume_emb = resume_emb.squeeze(0)
    results["resume_embedding"] = resume_emb.astype(np.float32)
    log.info("Resume embedding loaded. shape=%s", resume_emb.shape)

    # ── structured_resume.json ────────────────────────────────────
    schema_path = Path(arts["structured_resume"])
    if not schema_path.exists():
        raise FileNotFoundError(
            f"structured_resume.json not found at '{schema_path}'. "
            "Run File 1 first."
        )
    with open(schema_path, "r", encoding="utf-8") as f:
        results["structured_resume"] = json.load(f)
    log.info("Structured resume loaded. top-level keys=%s",
             list(results["structured_resume"].keys()))

    # ── pii_vault.json ────────────────────────────────────────────
    vault_path = Path(arts["pii_vault"])
    if not vault_path.exists():
        raise FileNotFoundError(
            f"pii_vault.json not found at '{vault_path}'. "
            "Run File 1 first."
        )
    with open(vault_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    results["vault"] = {e["token"]: e["original"] for e in entries}
    log.info("PII vault loaded. %d token mappings.", len(results["vault"]))

    print(f"✓ Artifacts loaded:")
    print(f"  resume_embedding  shape={results['resume_embedding'].shape}")
    print(f"  structured_resume sections={list(results['structured_resume'].keys())}")
    print(f"  pii_vault         entries={len(results['vault'])}")
    return results


# Artifacts loaded lazily inside run_pipeline() after CONFIG paths are overridden by main.py

# ────────────────────────────────────────────────────────────────
# JD EMBEDDING
# ────────────────────────────────────────────────────────────────

_EMBED_MODEL_CACHE: dict[str, SentenceTransformer] = {}

def _get_embed_model(model_name: str) -> "SentenceTransformer":
    """Load and cache the sentence-transformer model."""
    if not _ST_AVAILABLE:
        raise ImportError(
            "sentence-transformers is not installed. "
            "Run the dependency installation cell first."
        )
    if model_name not in _EMBED_MODEL_CACHE:
        log.info("Loading embedding model: %s  (first call — may take ~30s) …", model_name)
        _EMBED_MODEL_CACHE[model_name] = SentenceTransformer(model_name)
        log.info("Model loaded and cached.")
    return _EMBED_MODEL_CACHE[model_name]


def embed_job_description(
    text: str,
    model_name: str | None = None,
    use_query_prefix: bool = True,
) -> np.ndarray:
    """
    Embed the job description into a dense vector.

    Parameters
    ----------
    text             : str   — Raw job description text.
    model_name       : str   — Sentence-transformer model identifier.
    use_query_prefix : bool  — If True, prepend BGE query instruction prefix.

    Returns
    -------
    np.ndarray  shape (dim,), dtype float32, L2-normalised.
    """
    mn = model_name or CONFIG["embedding_model"]
    model = _get_embed_model(mn)

    if use_query_prefix:
        query = "Represent this sentence for searching relevant passages: " + text.strip()
    else:
        query = text.strip()

    log.info("Embedding JD. model=%s  chars=%d  prefix=%s", mn, len(text), use_query_prefix)
    vec = model.encode(query, normalize_embeddings=True, show_progress_bar=False)
    vec = np.array(vec, dtype=np.float32)

    log.info("JD embedding done. shape=%s  norm=%.4f", vec.shape, float(np.linalg.norm(vec)))
    return vec


print("✓ embed_job_description() defined.")

# ────────────────────────────────────────────────────────────────
# CHROMADB RETRIEVAL
# ────────────────────────────────────────────────────────────────

_chroma_client_cache: dict[str, Any] = {}

def _get_chroma_collection(db_path: str, collection_name: str):
    """Return (and cache) a ChromaDB persistent collection handle."""
    if not _CHROMA_AVAILABLE:
        raise ImportError(
            "chromadb is not installed. "
            "Run the dependency installation cell first."
        )
    cache_key = f"{db_path}::{collection_name}"
    if cache_key not in _chroma_client_cache:
        log.info("Connecting to ChromaDB at '%s' …", db_path)
        client = chromadb.PersistentClient(path=str(Path(db_path).resolve()))
        try:
            col = client.get_collection(name=collection_name)
            log.info("Collection '%s' loaded. count=%d", collection_name, col.count())
        except Exception as exc:
            raise RuntimeError(
                f"ChromaDB collection '{collection_name}' not found at '{db_path}'. "
                f"Run the ingestion notebook first.\n  Detail: {exc}"
            ) from exc
        _chroma_client_cache[cache_key] = col
    return _chroma_client_cache[cache_key]


def query_chromadb(
    jd_embedding: np.ndarray,
    config: dict | None = None,
) -> list[dict]:
    """
    Query ChromaDB with the JD embedding vector and return Top-K chunks.

    Parameters
    ----------
    jd_embedding : np.ndarray  — L2-normalised JD embedding (dim,).
    config       : dict        — Pipeline config.

    Returns
    -------
    list[dict]  Each dict has keys:
        id, document, metadata, distance, similarity
    Sorted by similarity descending.
    """
    cfg = config or CONFIG
    db_path    = cfg["chroma_db_path"]
    col_name   = cfg["chroma_collection"]
    top_k      = cfg["top_k_chunks"]

    col = _get_chroma_collection(db_path, col_name)

    query_vec = jd_embedding.tolist()
    log.info("Querying ChromaDB. collection=%s  top_k=%d", col_name, top_k)

    results = col.query(
        query_embeddings=[query_vec],
        n_results=min(top_k, col.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[dict] = []
    ids        = results["ids"][0]
    documents  = results["documents"][0]
    metadatas  = results["metadatas"][0]
    distances  = results["distances"][0]   # L2 or cosine distance (ChromaDB default)

    for i, (cid, doc, meta, dist) in enumerate(zip(ids, documents, metadatas, distances)):
        # Convert L2 distance to approximate cosine similarity
        # For normalised vectors: cosine_sim = 1 - (dist² / 2)
        sim = max(0.0, float(1.0 - (dist ** 2) / 2.0))
        chunks.append({
            "rank":       i + 1,
            "id":         cid,
            "document":   doc,
            "metadata":   meta or {},
            "distance":   round(float(dist), 6),
            "similarity": round(sim, 4),
        })

    log.info(
        "ChromaDB retrieval complete. %d chunks returned. "
        "Top similarity=%.4f  Bottom=%.4f",
        len(chunks),
        chunks[0]["similarity"] if chunks else 0,
        chunks[-1]["similarity"] if chunks else 0,
    )

    for c in chunks:
        log.debug("  [rank=%d] sim=%.4f  id=%s", c["rank"], c["similarity"], c["id"])

    return chunks


print("✓ query_chromadb() defined.")

# ────────────────────────────────────────────────────────────────
# GAP ANALYSIS
# ────────────────────────────────────────────────────────────────

# ── Keyword extraction helpers ────────────────────────────────────

_STOP_WORDS = frozenset({
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "is","are","was","were","be","been","being","have","has","had","will",
    "would","could","should","may","might","must","shall","can","do","does",
    "did","not","no","we","you","they","it","this","that","these","those",
    "our","your","their","us","him","her","them","who","which","what","when",
    "where","how","why","there","here","from","by","as","if","so","about",
    "into","through","during","before","after","above","below","between",
    "each","all","any","both","few","more","most","other","some","such",
    "than","then","too","very","just","also","again","further","any","same",
    "experience","work","team","role","position","candidate","required",
    "responsibilities","skills","ability","strong","excellent","good",
    "years","year","using","use","including","make","ensure","work",
})

def _extract_keywords(text: str, min_len: int = 3) -> set[str]:
    """Extract meaningful single- and bigram keywords from text."""
    # Normalize
    text = text.lower()
    # Tokenize
    tokens = re.findall(r'\b[a-z][a-z0-9#+.\-]*\b', text)
    # Filter stops and short tokens
    filtered = [t for t in tokens if t not in _STOP_WORDS and len(t) >= min_len]
    kws: set[str] = set(filtered)
    # Bigrams from consecutive filtered tokens — skip nonsensical/generic combos
    _GENERIC_WORDS = {"new", "key", "across", "offer", "join", "seeking",
                       "pvt", "ltd", "responsible", "motivated", "salary",
                       "package", "mentorship", "opportunity", "environment"}
    for i in range(len(filtered) - 1):
        if filtered[i] not in _GENERIC_WORDS and filtered[i+1] not in _GENERIC_WORDS:
            kws.add(f"{filtered[i]} {filtered[i+1]}")
    return kws


def _extract_resume_text_flat(structured_resume: dict) -> str:
    """Flatten the structured resume schema into a single text blob."""
    parts: list[str] = []

    def _walk(obj: Any) -> None:
        if isinstance(obj, str):
            parts.append(obj)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)
        elif isinstance(obj, dict):
            for v in obj.values():
                _walk(v)

    _walk(structured_resume)
    return " ".join(parts)


def _section_keyword_coverage(section_text: str, jd_keywords: set[str]) -> float:
    """Return fraction of JD keywords present in a section."""
    if not jd_keywords:
        return 1.0
    sec_kws = _extract_keywords(section_text)
    hit = len(jd_keywords & sec_kws)
    return round(hit / len(jd_keywords), 4)


def compute_gap_analysis(
    jd_embedding:      np.ndarray,
    resume_embedding:  np.ndarray,
    jd_text:           str,
    structured_resume: dict,
) -> dict:
    """
    Perform embedding-space and keyword-level gap analysis.

    Parameters
    ----------
    jd_embedding      : np.ndarray  — L2-normalised JD embedding.
    resume_embedding  : np.ndarray  — L2-normalised resume embedding from File 1.
    jd_text           : str         — Raw job description text.
    structured_resume : dict        — Sanitized structured schema from File 1.

    Returns
    -------
    dict  — gap_report with similarity metrics, missing keywords, and recommendations.
    """
    log.info("Computing gap analysis …")

    # ── 1. Cosine similarity (global) ─────────────────────────────
    jd_vec  = jd_embedding.reshape(1, -1)
    res_vec = resume_embedding.reshape(1, -1)
    overall_sim = float(cosine_similarity(jd_vec, res_vec)[0][0])
    gap_score   = round(1.0 - overall_sim, 4)
    overall_sim = round(overall_sim, 4)
    log.info("Overall cosine similarity: %.4f  gap: %.4f", overall_sim, gap_score)

    # ── 2. Keyword analysis ───────────────────────────────────────
    jd_keywords     = _extract_keywords(jd_text)
    resume_flat     = _extract_resume_text_flat(structured_resume)
    resume_keywords = _extract_keywords(resume_flat)

    present_keywords = sorted(jd_keywords & resume_keywords)
    missing_keywords = sorted(jd_keywords - resume_keywords)
    keyword_coverage = round(len(present_keywords) / max(len(jd_keywords), 1), 4)
    log.info(
        "Keywords — JD total=%d  present=%d  missing=%d  coverage=%.2f%%",
        len(jd_keywords), len(present_keywords), len(missing_keywords),
        keyword_coverage * 100,
    )

    # ── 3. Per-section keyword coverage ──────────────────────────
    section_coverage: dict[str, float] = {}
    _score_sections = ["skills", "experience", "projects", "summary", "education"]
    for sec in _score_sections:
        blob = structured_resume.get(sec, "")
        if isinstance(blob, (list, dict)):
            blob = json.dumps(blob)
        section_coverage[sec] = _section_keyword_coverage(blob, jd_keywords)

    weak_sections = [s for s, sc in section_coverage.items() if sc < 0.3]
    log.info("Section coverage: %s", section_coverage)
    log.info("Weak sections: %s", weak_sections)

    # ── 4. Natural-language recommendations ──────────────────────
    recommendations: list[str] = []
    if missing_keywords:
        top_missing = missing_keywords[:15]
        recommendations.append(
            f"Add these high-priority missing keywords: {', '.join(top_missing)}."
        )
    if gap_score > 0.4:
        recommendations.append(
            "Significant semantic gap detected. Emphasise domain-specific terminology "
            "matching the role's technical requirements."
        )
    if "skills" in weak_sections:
        recommendations.append(
            "Skills section has low JD keyword coverage. "
            "Expand technical skills to include tools/frameworks mentioned in the JD."
        )
    if "projects" in weak_sections:
        recommendations.append(
            "Projects section is weakly aligned to the JD. "
            "Highlight projects that demonstrate required technologies."
        )
    if keyword_coverage < 0.5:
        recommendations.append(
            "Overall keyword coverage is below 50%. "
            "Resume needs significant tailoring to match this role."
        )

    gap_report = {
        "overall_similarity":  overall_sim,
        "gap_score":           gap_score,
        "keyword_coverage":    keyword_coverage,
        "jd_keyword_count":    len(jd_keywords),
        "present_keywords":    present_keywords[:50],   # cap for readability
        "missing_keywords":    missing_keywords[:80],
        "section_coverage":    section_coverage,
        "weak_sections":       weak_sections,
        "recommendations":     recommendations,
    }

    print(f"  Overall similarity : {overall_sim:.2%}")
    print(f"  Gap score          : {gap_score:.2%}")
    print(f"  Keyword coverage   : {keyword_coverage:.2%}  ({len(present_keywords)}/{len(jd_keywords)})")
    print(f"  Missing keywords   : {len(missing_keywords)}  top={missing_keywords[:8]}")
    print(f"  Weak sections      : {weak_sections}")
    log.info("Gap analysis complete.")
    return gap_report


print("✓ compute_gap_analysis() defined.")

# ────────────────────────────────────────────────────────────────
# BOTTLENECK LAYER
# ────────────────────────────────────────────────────────────────

_QUANTIFIER_RE = re.compile(
    r'\b(\d+[\.,]?\d*\s*(%|percent|x|×|k\b|m\b|billion|million|thousand|'
    r'users|requests|ms\b|seconds|hours|days|weeks|'
    r'lines|commits|deployments|incidents|customers|revenue|usd|\$|£|€|₹))',
    re.IGNORECASE,
)


def _score_bullet(bullet: str, jd_keywords: set[str]) -> float:
    """
    Score an experience bullet by impact signals and JD relevance.
    Returns a float in [0, 1].
    """
    b = bullet.lower()
    quant_hits = len(_QUANTIFIER_RE.findall(b))
    kw_hits    = sum(1 for kw in jd_keywords if kw in b)
    # Heuristic: quantified achievement = +0.4; each JD keyword = +0.1 (capped)
    score = min(1.0, quant_hits * 0.4 + kw_hits * 0.1)
    return round(score, 3)


def _flatten_skills(skills_obj: Any) -> list[str]:
    """Flatten skills dict or list into a flat list of strings."""
    if isinstance(skills_obj, list):
        return [str(s) for s in skills_obj if s]
    if isinstance(skills_obj, dict):
        out: list[str] = []
        for v in skills_obj.values():
            out.extend(_flatten_skills(v))
        return out
    if isinstance(skills_obj, str):
        return [s.strip() for s in re.split(r'[,;|•]', skills_obj) if s.strip()]
    return []


def bottleneck_layer(
    structured_resume: dict,
    gap_report:        dict,
    max_skills:        int = 20,
    max_bullets:       int = 10,
    max_projects:      int = 3,
) -> dict:
    """
    Extract high-signal resume content prioritised for the target role.

    Parameters
    ----------
    structured_resume : dict  — Sanitized schema from File 1.
    gap_report        : dict  — Output of compute_gap_analysis().
    max_skills        : int   — Max skill items to include.
    max_bullets       : int   — Max experience bullets to include.
    max_projects      : int   — Max projects to include.

    Returns
    -------
    dict  — optimized_content_plan
    """
    log.info("Running bottleneck layer …")
    jd_keywords = set(gap_report.get("present_keywords", []))
    missing_kws = set(gap_report.get("missing_keywords", []))

    # ── 1. Skills ─────────────────────────────────────────────────
    all_skills = _flatten_skills(structured_resume.get("skills", []))
    # Prioritise skills that appear in JD keywords
    priority_skills = [s for s in all_skills if s.lower() in jd_keywords]
    other_skills    = [s for s in all_skills if s.lower() not in jd_keywords]
    selected_skills = list(dict.fromkeys(priority_skills + other_skills))[:max_skills]
    log.info("Skills: total=%d  priority=%d  selected=%d",
             len(all_skills), len(priority_skills), len(selected_skills))

    # ── 2. Experience bullets ──────────────────────────────────────
    experience = structured_resume.get("experience", [])
    scored_bullets: list[tuple[float, str, str]] = []   # (score, company+title, bullet)
    for job in experience:
        if not isinstance(job, dict):
            continue
        label = f"{job.get('title','?')} @ {job.get('company', job.get('raw_header','?'))}"
        for bullet in job.get("bullets", []):
            if not isinstance(bullet, str):
                continue
            sc = _score_bullet(bullet, jd_keywords)
            scored_bullets.append((sc, label, bullet))

    scored_bullets.sort(key=lambda x: x[0], reverse=True)
    top_bullets = [
        {"score": sc, "context": ctx, "bullet": b}
        for sc, ctx, b in scored_bullets[:max_bullets]
    ]
    log.info("Experience bullets: total=%d  selected=%d",
             len(scored_bullets), len(top_bullets))

    # ── 3. Projects ───────────────────────────────────────────────
    projects = structured_resume.get("projects", [])
    scored_projects: list[tuple[float, dict]] = []
    for proj in projects:
        if not isinstance(proj, dict):
            continue
        proj_text = " ".join([
            proj.get("name", ""),
            proj.get("description", ""),
            " ".join(proj.get("technologies", [])),
            " ".join(proj.get("bullets", [])),
        ]).lower()
        hits  = sum(1 for kw in jd_keywords if kw in proj_text)
        score = min(1.0, hits * 0.12)
        scored_projects.append((score, proj))

    scored_projects.sort(key=lambda x: x[0], reverse=True)
    top_projects = [p for _, p in scored_projects[:max_projects]]
    log.info("Projects: total=%d  selected=%d", len(projects), len(top_projects))

    # ── 4. Summary / objective ────────────────────────────────────
    summary = structured_resume.get("summary", "")
    if isinstance(summary, list):
        summary = " ".join(summary)

    # ── 5. Education ──────────────────────────────────────────────
    education = structured_resume.get("education", [])

    # ── 6. Personal info (tokens — PII not restored yet) ──────────
    personal_info = structured_resume.get("personal_info", {})

    # ── 7. Interests, goals, certifications ───────────────────────
    interests      = structured_resume.get("interests", [])
    goals          = structured_resume.get("goals", [])
    achievements   = structured_resume.get("achievements", [])
    certifications = structured_resume.get("certifications", [])

    # ── 8. Assemble content plan ───────────────────────────────────
    plan = {
        "personal_info":          personal_info,
        "summary":                summary,
        "priority_skills":        selected_skills,
        "_skills_categorized":    structured_resume.get("skills", {}),
        "missing_skills_to_add":  sorted(missing_kws)[:15],
        "top_experience_bullets": top_bullets,
        "top_projects":           top_projects,
        "education":              education,
        "certifications":         certifications,
        "gap_recommendations":    gap_report.get("recommendations", []),
        "keyword_coverage":       gap_report.get("keyword_coverage", 0),
        "overall_similarity":     gap_report.get("overall_similarity", 0),
        "interests":              interests,
        "goals":                  goals,
        "achievements":           achievements,
    }

    print(f"  Priority skills     : {len(selected_skills)}")
    print(f"  Experience bullets  : {len(top_bullets)}")
    print(f"  Projects            : {len(top_projects)}")
    print(f"  Missing skills hint : {sorted(missing_kws)[:8]}")
    log.info("Bottleneck layer complete.")
    return plan


print("✓ bottleneck_layer() defined.")

# ────────────────────────────────────────────────────────────────
# LLM GENERATION (OLLAMA)
# ────────────────────────────────────────────────────────────────

_OLLAMA_BASE_URL = "http://localhost:11434"
_VRAM_ERROR_SIGNALS = [
    "out of memory", "oom", "cuda error", "memory allocation",
    "context length exceeded", "model too large", "not enough memory",
]

def _is_vram_error(err_text: str) -> bool:
    t = err_text.lower()
    return any(sig in t for sig in _VRAM_ERROR_SIGNALS)


def _progressive_timeouts(base_timeout: int, attempts: int, max_timeout: int) -> list[int]:
    """Return a progressively increasing timeout schedule in seconds."""
    base = max(30, int(base_timeout))
    tries = max(1, int(attempts))
    cap = max(base, int(max_timeout))

    vals: list[int] = []
    current = base
    for _ in range(tries):
        vals.append(min(current, cap))
        # Add 50% each retry; enough headroom for slower local inference
        current = int(current * 1.5)
        if current <= vals[-1]:
            current = vals[-1] + 30
    return vals


def _ollama_generate(
    prompt:    str,
    model:     str,
    timeout:   int = 120,
    stream:    bool = False,
    timeout_attempts: int = 1,
    timeout_max: int = 900,
) -> str:
    """
    Call Ollama /api/generate endpoint.

    Returns the generated text string.
    Raises RuntimeError on HTTP errors, TimeoutError on repeated timeouts,
    or MemoryError on VRAM issues.
    """
    payload = {
        "model":  model,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
            "num_predict": 2048,
        },
    }

    timeout_schedule = _progressive_timeouts(timeout, timeout_attempts, timeout_max)

    for attempt_idx, cur_timeout in enumerate(timeout_schedule, start=1):
        log.info(
            "Calling Ollama. model=%s  prompt_len=%d  timeout=%ss  attempt=%d/%d",
            model, len(prompt), cur_timeout, attempt_idx, len(timeout_schedule)
        )
        t0 = time.time()

        try:
            resp = requests.post(
                f"{_OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=cur_timeout,
            )
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"Cannot connect to Ollama at {_OLLAMA_BASE_URL}. "
                "Is Ollama running? Start it with: ollama serve"
            ) from exc
        except requests.exceptions.ReadTimeout as exc:
            if attempt_idx < len(timeout_schedule):
                log.warning(
                    "Ollama read timeout on model '%s' (attempt %d/%d, timeout=%ss). Retrying …",
                    model, attempt_idx, len(timeout_schedule), cur_timeout,
                )
                continue
            raise TimeoutError(
                f"Ollama timed out on model '{model}' after {len(timeout_schedule)} attempt(s). "
                f"Last timeout={cur_timeout}s."
            ) from exc
        except requests.exceptions.Timeout as exc:
            if attempt_idx < len(timeout_schedule):
                log.warning(
                    "Ollama timeout on model '%s' (attempt %d/%d, timeout=%ss). Retrying …",
                    model, attempt_idx, len(timeout_schedule), cur_timeout,
                )
                continue
            raise TimeoutError(
                f"Ollama request timed out on model '{model}' after {len(timeout_schedule)} attempt(s). "
                f"Last timeout={cur_timeout}s."
            ) from exc

        if resp.status_code != 200:
            err = resp.text[:400]
            if _is_vram_error(err):
                raise MemoryError(f"VRAM error on model '{model}': {err}")
            raise RuntimeError(f"Ollama HTTP {resp.status_code}: {err}")

        try:
            data = resp.json()
            text = data.get("response", "").strip()
        except Exception as exc:
            raise RuntimeError(f"Failed to parse Ollama response: {exc}") from exc

        elapsed = round(time.time() - t0, 1)
        log.info("Ollama response received. chars=%d  elapsed=%.1fs", len(text), elapsed)
        return text

    # Defensive fallback; loop should always return or raise.
    raise TimeoutError(f"Ollama timed out for model '{model}'.")


def _build_generation_prompt(
    content_plan:  dict,
    chroma_chunks: list[dict],
    jd_text:       str,
) -> str:
    """Construct the full generation prompt from content plan + RAG context."""

    # ── RAG context block ─────────────────────────────────────────
    rag_lines: list[str] = []
    if chroma_chunks:
        rag_lines.append("=== FORMAL JOB POSTING EXAMPLES (for tone & format guidance) ===")
        for c in chroma_chunks[:3]:   # use top-3 for brevity
            snippet = c["document"][:600].strip().replace("\n", " ")
            rag_lines.append(f"• {snippet}")
        rag_lines.append("=== END EXAMPLES ===\n")

    rag_context = "\n".join(rag_lines)

    # ── Resume facts block ────────────────────────────────────────
    pi = content_plan.get("personal_info", {})

    skills_str   = ", ".join(content_plan.get("priority_skills", []))
    # Build skills string preserving categories from structured data
    skills_obj = content_plan.get("_skills_categorized", {})
    if skills_obj and isinstance(skills_obj, dict):
        skills_parts = []
        for cat, items in skills_obj.items():
            if items:
                cat_label = cat.replace("_", " ").title()
                skills_parts.append(f"{cat_label}: {', '.join(items)}")
        skills_str = "\n".join(skills_parts) if skills_parts else skills_str
    missing_str  = ", ".join(content_plan.get("missing_skills_to_add", []))

    bullets_str = ""
    for b in content_plan.get("top_experience_bullets", []):
        ctx    = b.get("context", "")
        bullet = b.get("bullet", "")
        bullets_str += f"  [{ctx}] {bullet}\n"

    projects_str = ""
    for p in content_plan.get("top_projects", []):
        name  = p.get("name", "Project")
        desc  = p.get("description", "")
        tech  = ", ".join(p.get("technologies", []))
        bullets = p.get("bullets", [])
        projects_str += f"  • {name}: {desc} (Tech: {tech})\n"
        for b in bullets:
            projects_str += f"    - {b}\n"

    edu_str = ""
    for e in content_plan.get("education", []):
        if isinstance(e, dict):
            gpa = e.get('gpa', '')
            gpa_part = f" | CGPA: {gpa}" if gpa else ""
            edu_str += f"  • {e.get('degree','?')} — {e.get('institution','?')} {e.get('dates','')}{gpa_part}\n"

    recs_str     = "\n".join(f"  - {r}" for r in content_plan.get("gap_recommendations", []))
    summary      = content_plan.get("summary", "")
    interests    = content_plan.get("interests", [])
    goals        = content_plan.get("goals", [])
    achievements = content_plan.get("achievements", [])
    interests_str    = ", ".join(interests[:8])    if interests    else ""
    goals_str        = ", ".join(goals[:5])        if goals        else ""
    achievements_str = "\n".join(f"  • {a}" for a in achievements[:5]) if achievements else ""

    # ── Certifications block ──────────────────────────────────────
    certifications     = content_plan.get("certifications", [])
    certifications_str = ""
    for c in certifications:
        if isinstance(c, dict):
            name   = c.get("name", "")
            issuer = c.get("issuer", "")
            date   = c.get("date", "")
            certifications_str += f"  • {name}"
            if issuer: certifications_str += f" — {issuer}"
            if date:   certifications_str += f" ({date})"
            certifications_str += "\n"

    prompt = f"""You are an expert resume writer. Write a complete, professional resume tailored to the job description below.

CRITICAL RULES — READ CAREFULLY:
1. ONLY use facts from the RESUME FACTS section. DO NOT invent any job titles, companies, internships, or work experience that are not listed.
2. If "Key Experience Bullets" is empty or says NONE, the candidate has NO work experience — OMIT the EXPERIENCE section entirely. Never fabricate experience.
3. Output ONLY the resume text — no commentary, no explanations, no markdown formatting.
4. Use EXACTLY these section headers in ALL CAPS on their own line, in this order: PROFESSIONAL SUMMARY, EDUCATION, TECHNICAL SKILLS, EXPERIENCE (only if real experience exists), PROJECTS, CERTIFICATIONS
5. Start the resume with the candidate's name on the first line, then contact details.
6. For TECHNICAL SKILLS: list as comma-separated values grouped by category, NOT as bullet sentences.
7. For CERTIFICATIONS: list ALL certifications provided — do not skip or write "None provided".
8. Do NOT use "---" separators, markdown, or asterisk bullets. Use • for bullet points.

---
TARGET JOB DESCRIPTION:
{jd_text[:1500].strip()}

---
{rag_context}
---
RESUME FACTS (use ONLY these — do not add anything else):

Personal Info: {content_plan.get("personal_info", {})}
Summary hint: {summary}
Skills (verified, list all): {skills_str}
Missing keywords to weave in naturally: {missing_str}

Key Experience Bullets (ONLY real experience — if empty, candidate has NO work experience):
{bullets_str if bullets_str.strip() else "  [NONE — do not fabricate any experience]"}

Projects:
{projects_str}
Education:
{edu_str}
Certifications (include ALL of these):
{certifications_str if certifications_str.strip() else "  [None provided]"}
Interests: {interests_str}
Career Goals: {goals_str}

---
OUTPUT FORMAT (follow exactly):

[Full Name]
[Email] | [Phone] | [LinkedIn] | [GitHub]

PROFESSIONAL SUMMARY
[3-4 sentences tailored to the job]

EDUCATION
[Degree] — [Institution] ([Year])
CGPA: [if available]

TECHNICAL SKILLS
Programming Languages: [comma separated]
Web Development: [comma separated]
Libraries & Frameworks: [comma separated]
Databases: [comma separated]
Tools: [comma separated]

EXPERIENCE
[Only if real experience exists from RESUME FACTS above. If none, OMIT this section entirely.]

PROJECTS
[Project Name] | [Technologies]
• [bullet]
• [bullet]

CERTIFICATIONS
• [cert name] — [issuer] ([year])

BEGIN RESUME:
"""
    return prompt


def generate_resume(
    content_plan:  dict,
    chroma_chunks: list[dict],
    jd_text:       str,
    config:        dict | None = None,
) -> tuple[str, str]:
    """
    Generate a tailored resume draft using Ollama.

    Parameters
    ----------
    content_plan  : dict        — Output of bottleneck_layer().
    chroma_chunks : list[dict]  — Top-K ChromaDB chunks for RAG context.
    jd_text       : str         — Raw job description text.
    config        : dict        — Pipeline config.

    Returns
    -------
    (draft_text, model_used) : tuple[str, str]
    """
    cfg      = config or CONFIG
    primary  = cfg["ollama_model"]
    fallback = cfg["fallback_model"]
    timeout  = cfg.get("ollama_timeout", 240)
    timeout_attempts = cfg.get("ollama_timeout_attempts", 3)
    timeout_max = cfg.get("ollama_timeout_max", 900)

    prompt = _build_generation_prompt(content_plan, chroma_chunks, jd_text)
    log.info("Generation prompt built. total_chars=%d", len(prompt))

    # ── Try primary model ─────────────────────────────────────────
    try:
        log.info("Attempting generation with primary model: %s", primary)
        draft = _ollama_generate(
            prompt,
            primary,
            timeout=timeout,
            timeout_attempts=timeout_attempts,
            timeout_max=timeout_max,
        )
        log.info("Primary model succeeded. draft_chars=%d", len(draft))
        # Strip prompt leakage artifacts
        draft = re.sub(r'(?i)\s*BEGIN\s+(IMPROVED\s+)?RESUME[.:!]?\s*$', '', draft).rstrip()
        return draft, primary

    except (MemoryError, TimeoutError) as model_err:
        log.warning(
            "Primary model '%s' failed (%s). Falling back to '%s' …",
            primary, type(model_err).__name__, fallback,
        )
        print(
            f"  ⚠ Primary model '{primary}' failed ({type(model_err).__name__}) "
            f"— falling back to '{fallback}' …"
        )

    except RuntimeError as rte:
        log.error("Generation failed on '%s': %s", primary, str(rte)[:200])
        raise

    # ── Fallback model ────────────────────────────────────────────
    try:
        log.info("Attempting generation with fallback model: %s", fallback)
        draft = _ollama_generate(
            prompt,
            fallback,
            timeout=timeout,
            timeout_attempts=timeout_attempts,
            timeout_max=timeout_max,
        )
        log.info("Fallback model succeeded. draft_chars=%d", len(draft))
        # Strip prompt leakage artifacts
        draft = re.sub(r'(?i)\s*BEGIN\s+(IMPROVED\s+)?RESUME[.:!]?\s*$', '', draft).rstrip()
        print(f"  ✓ Generated with fallback model: {fallback}")
        return draft, fallback

    except Exception as exc:
        raise RuntimeError(
            f"Both primary ('{primary}') and fallback ('{fallback}') models failed.\n"
            f"Primary error: timeout/VRAM/runtime\nFallback error: {exc}\n"
            "Ensure Ollama is running and models are pulled:\n"
            f"  ollama pull {primary}\n  ollama pull {fallback}\n"
            "If needed, increase CONFIG['ollama_timeout'] and CONFIG['ollama_timeout_attempts']."
        ) from exc


print("✓ generate_resume() defined.")

# ────────────────────────────────────────────────────────────────
# CRITIC / DISCRIMINATOR LOOP
# ────────────────────────────────────────────────────────────────

def _build_critic_prompt(draft: str, jd_text: str) -> str:
    return f"""You are a senior technical recruiter and resume expert. Your job is to critically evaluate the following resume against the target job description.

Evaluate and respond in EXACTLY this JSON format (no markdown, no extra text):
{{
  "score": <integer 0-100 representing overall JD alignment>,
  "weak_sections": ["<section name>", ...],
  "missing_keywords": ["<keyword>", ...],
  "specific_issues": ["<issue description>", ...],
  "improvement_suggestions": ["<actionable suggestion>", ...]
}}

JOB DESCRIPTION:
{jd_text[:1200].strip()}

RESUME DRAFT:
{draft[:3000].strip()}

YOUR EVALUATION (JSON only):"""


def _build_refined_prompt(
    draft:       str,
    critic_eval: dict,
    content_plan: dict,
    jd_text:     str,
    chroma_chunks: list[dict],
) -> str:
    issues   = "\n".join(f"  - {i}" for i in critic_eval.get("specific_issues", []))
    suggs    = "\n".join(f"  - {s}" for s in critic_eval.get("improvement_suggestions", []))
    miss_kws = ", ".join(critic_eval.get("missing_keywords", [])[:15])
    base_prompt = _build_generation_prompt(content_plan, chroma_chunks, jd_text)
    return f"""{base_prompt}

PREVIOUS DRAFT SCORE: {critic_eval.get('score', 0)}/100

CRITIC FEEDBACK — ISSUES TO FIX:
{issues}

CRITIC IMPROVEMENT SUGGESTIONS:
{suggs}

MISSING KEYWORDS TO INCORPORATE (only where factually justified):
{miss_kws}

Rewrite the complete resume addressing ALL the above feedback. Output ONLY the resume text.

BEGIN IMPROVED RESUME:"""


def _parse_critic_json(raw: str) -> dict:
    """Extract and parse the JSON block from a critic response."""
    # Try direct parse
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass
    # Try extracting JSON object from surrounding text
    m = re.search(r'\{[\s\S]+\}', raw)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    # Fallback: extract score with regex
    score_m = re.search(r'"score"\s*:\s*(\d+)', raw)
    score = int(score_m.group(1)) if score_m else 50
    return {
        "score": score,
        "weak_sections": [],
        "missing_keywords": [],
        "specific_issues": [f"Could not parse critic JSON. Raw: {raw[:200]}"],
        "improvement_suggestions": [],
    }


def critic_loop(
    initial_draft:  str,
    model_used:     str,
    chroma_chunks:  list[dict],
    jd_text:        str,
    content_plan:   dict,
    config:         dict | None = None,
) -> dict:
    """
    Iteratively refine the resume draft using a critic LLM.

    Parameters
    ----------
    initial_draft  : str         — First generation from generate_resume().
    model_used     : str         — Model that produced the initial draft.
    chroma_chunks  : list[dict]  — ChromaDB context.
    jd_text        : str         — Job description text.
    content_plan   : dict        — Optimised content plan.
    config         : dict        — Pipeline config.

    Returns
    -------
    dict with keys:
        final_draft       : str
        final_score       : int
        iterations        : int
        model_used        : str
        evaluation_history: list[dict]
    """
    cfg             = config or CONFIG
    max_retries     = cfg.get("max_retries", 3)
    threshold       = cfg.get("critic_pass_threshold", 72)
    timeout         = cfg.get("ollama_timeout", 240)
    timeout_attempts= cfg.get("ollama_timeout_attempts", 3)
    timeout_max     = cfg.get("ollama_timeout_max", 900)
    fallback_model  = cfg["fallback_model"]

    current_draft    = initial_draft
    eval_history:    list[dict] = []
    best_draft       = initial_draft
    best_score       = 0

    print(f"\n  Critic loop: max_retries={max_retries}  threshold={threshold}")

    for iteration in range(max_retries + 1):   # +1: first pass is evaluation only
        log.info("Critic iteration %d/%d …", iteration, max_retries)
        print(f"  [Iteration {iteration}] Evaluating draft …", end=" ", flush=True)

        # ── Evaluate current draft ───────────────────────────────
        critic_prompt = _build_critic_prompt(current_draft, jd_text)
        try:
            raw_eval = _ollama_generate(
                critic_prompt,
                model_used,
                timeout=timeout,
                timeout_attempts=timeout_attempts,
                timeout_max=timeout_max,
            )
        except (MemoryError, TimeoutError):
            log.warning("Critic call failed on '%s' — switching to fallback '%s'.", model_used, fallback_model)
            model_used = fallback_model
            raw_eval = _ollama_generate(
                critic_prompt,
                model_used,
                timeout=timeout,
                timeout_attempts=timeout_attempts,
                timeout_max=timeout_max,
            )

        eval_result = _parse_critic_json(raw_eval)
        score = int(eval_result.get("score", 0))
        eval_result["iteration"] = iteration
        eval_result["draft_length"] = len(current_draft)
        eval_history.append(eval_result)

        print(f"score={score}/100", end="")

        if score > best_score:
            best_score = score
            best_draft = current_draft
            log.info("New best draft at iteration %d  score=%d", iteration, score)

        if score >= threshold:
            print(f"  ✓ Passed threshold ({threshold}).")
            log.info("Critic threshold met at iteration %d. score=%d", iteration, score)
            break

        if iteration >= max_retries:
            print(f"\n  ⚠ Max retries reached. Best score={best_score}/100.")
            log.warning("Max retries reached. Returning best draft (score=%d).", best_score)
            break

        print(f"  < threshold ({threshold}) — regenerating …")
        log.info("Score %d < threshold %d — regenerating (iteration %d) …",
                 score, threshold, iteration + 1)

        # ── Regenerate with critic feedback ──────────────────────
        refined_prompt = _build_refined_prompt(
            current_draft, eval_result, content_plan, jd_text, chroma_chunks
        )
        try:
            current_draft = _ollama_generate(
                refined_prompt,
                model_used,
                timeout=timeout,
                timeout_attempts=timeout_attempts,
                timeout_max=timeout_max,
            )
            # Strip prompt leakage from regenerated draft
            current_draft = re.sub(r'(?i)\s*BEGIN\s+(IMPROVED\s+)?RESUME[.:!]?\s*$', '', current_draft).rstrip()
        except (MemoryError, TimeoutError):
            model_used = fallback_model
            current_draft = _ollama_generate(
                refined_prompt,
                fallback_model,
                timeout=timeout,
                timeout_attempts=timeout_attempts,
                timeout_max=timeout_max,
            )
            # Strip prompt leakage from regenerated draft
            current_draft = re.sub(r'(?i)\s*BEGIN\s+(IMPROVED\s+)?RESUME[.:!]?\s*$', '', current_draft).rstrip()
        except RuntimeError as exc:
            log.error("Regeneration failed at iteration %d: %s", iteration + 1, exc)
            break

    print(f"\n  Critic loop complete. best_score={best_score}/100  iterations={len(eval_history)}")
    return {
        "final_draft":        best_draft,
        "final_score":        best_score,
        "iterations":         len(eval_history),
        "model_used":         model_used,
        "evaluation_history": eval_history,
    }


print("✓ critic_loop() defined.")

# ────────────────────────────────────────────────────────────────
# HALLUCINATION GUARD
# ────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using NLTK if available, else regex fallback."""
    if _NLTK_AVAILABLE:
        try:
            return [s.strip() for s in sent_tokenize(text) if s.strip()]
        except Exception:
            pass
    # Regex fallback
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def hallucination_guard(
    draft:            str,
    resume_embedding: np.ndarray,
    config:           dict | None = None,
    model_name:       str | None  = None,
    strip_flagged:    bool = True,
) -> dict:
    """
    Detect and optionally remove hallucinated sentences from the generated draft.

    Parameters
    ----------
    draft             : str         — Generated resume text.
    resume_embedding  : np.ndarray  — Original resume embedding from File 1.
    config            : dict        — Pipeline config.
    model_name        : str         — Embedding model to use.
    strip_flagged     : bool        — If True, remove flagged sentences from output.

    Returns
    -------
    dict with keys:
        clean_draft         : str
        hallucination_ratio : float
        flagged_sentences   : list[dict]
        sentence_scores     : list[dict]
        pass_guard          : bool   — True if ratio ≤ threshold
    """
    cfg       = config or CONFIG
    mn        = model_name or cfg["embedding_model"]
    sim_floor = cfg.get("sentence_sim_min", 0.40)
    h_thresh  = cfg.get("hallucination_threshold", 0.30)

    log.info("Running hallucination guard. sim_floor=%.2f  h_thresh=%.2f", sim_floor, h_thresh)

    if not _ST_AVAILABLE:
        log.warning("sentence-transformers unavailable — skipping hallucination guard.")
        return {
            "clean_draft":         draft,
            "hallucination_ratio": 0.0,
            "flagged_sentences":   [],
            "sentence_scores":     [],
            "pass_guard":          True,
            "note":                "Guard skipped: sentence-transformers not installed.",
        }

    sentences = _split_sentences(draft)
    if not sentences:
        return {
            "clean_draft":         draft,
            "hallucination_ratio": 0.0,
            "flagged_sentences":   [],
            "sentence_scores":     [],
            "pass_guard":          True,
        }

    log.info("Embedding %d sentences …", len(sentences))
    model = _get_embed_model(mn)

    # Batch-encode all sentences
    sent_vecs = model.encode(
        sentences,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64,
    )   # shape: (n_sentences, dim)

    # Compare each sentence to the original resume embedding
    res_vec = resume_embedding.reshape(1, -1)   # (1, dim)
    sims    = cosine_similarity(sent_vecs, res_vec).flatten()   # (n_sentences,)

    sentence_scores: list[dict] = []
    flagged:         list[dict] = []
    clean_sentences: list[str]  = []

    for i, (sent, sim) in enumerate(zip(sentences, sims)):
        sim_f    = round(float(sim), 4)
        is_flagged = sim_f < sim_floor

        # Skip very short lines (headers, section markers — not substantive claims)
        is_header = len(sent.split()) <= 4
        if is_header:
            is_flagged = False

        entry = {
            "index":      i,
            "sentence":   sent[:120] + ("…" if len(sent) > 120 else ""),
            "similarity": sim_f,
            "flagged":    is_flagged,
        }
        sentence_scores.append(entry)

        if is_flagged:
            flagged.append(entry)
            log.debug("  [FLAGGED] sim=%.4f  '%s …'", sim_f, sent[:60])
        else:
            clean_sentences.append(sent)

    total_non_header = max(1, sum(1 for e in sentence_scores if not (len(e["sentence"].split()) <= 4)))
    h_ratio = round(len(flagged) / total_non_header, 4)
    pass_guard = h_ratio <= h_thresh

    clean_draft = " ".join(clean_sentences) if strip_flagged and not pass_guard else draft

    log.info(
        "Hallucination guard complete. sentences=%d  flagged=%d  ratio=%.2f%%  pass=%s",
        len(sentences), len(flagged), h_ratio * 100, pass_guard,
    )

    print(f"  Sentences analysed : {len(sentences)}")
    print(f"  Flagged            : {len(flagged)}")
    print(f"  Hallucination ratio: {h_ratio:.1%}")
    print(f"  Guard pass         : {'✓' if pass_guard else '✗ (flagged sentences stripped)'}")

    if flagged:
        print(f"  Top flagged        : {flagged[0]['sentence']}")

    return {
        "clean_draft":         clean_draft,
        "hallucination_ratio": h_ratio,
        "flagged_sentences":   flagged,
        "sentence_scores":     sentence_scores,
        "pass_guard":          pass_guard,
    }


print("✓ hallucination_guard() defined.")

# ────────────────────────────────────────────────────────────────
# ATS SCORE CALCULATION
# ────────────────────────────────────────────────────────────────

_REQUIRED_SECTIONS = [
    "summary",
    "skills",
    "experience",
    "education",
    "projects",
]

_SECTION_HEADER_PATTERNS = {
    "summary":    re.compile(r'\b(summary|objective|profile|about)\b', re.IGNORECASE),
    "skills":     re.compile(r'\b(skills|competencies|technologies|expertise)\b', re.IGNORECASE),
    "experience": re.compile(r'\b(experience|employment|work history|career)\b', re.IGNORECASE),
    "education":  re.compile(r'\b(education|academic|qualifications)\b', re.IGNORECASE),
    "projects":   re.compile(r'\b(projects|portfolio|notable work)\b', re.IGNORECASE),
}


def _compute_section_completeness(text: str) -> tuple[float, dict[str, bool]]:
    """Check which required sections are present AND have content in the resume text."""
    section_present: dict[str, bool] = {}
    lines = text.split('\n')
    for section, pat in _SECTION_HEADER_PATTERNS.items():
        found = False
        for i, line in enumerate(lines):
            if pat.search(line):
                # Check if there's actual content in the next few lines
                content_lines = [l.strip() for l in lines[i+1:i+5] if l.strip()]
                if content_lines:
                    found = True
                break
        section_present[section] = found
    score = sum(1 for v in section_present.values() if v) / len(_REQUIRED_SECTIONS)
    return round(score, 4), section_present


def compute_ats_score(
    final_draft:      str,
    jd_text:          str,
    jd_embedding:     np.ndarray,
    config:           dict | None = None,
    model_name:       str | None  = None,
) -> dict:
    """
    Compute a multi-dimensional ATS compatibility score.

    Parameters
    ----------
    final_draft      : str         — The final (guard-cleaned) resume text.
    jd_text          : str         — Raw job description text.
    jd_embedding     : np.ndarray  — L2-normalised JD embedding.
    config           : dict        — Pipeline config.
    model_name       : str         — Embedding model.

    Returns
    -------
    dict  — ATS report with overall score and component breakdown.
    """
    cfg        = config or CONFIG
    mn         = model_name or cfg["embedding_model"]
    weights    = cfg.get("ats_weights", {
        "keyword_coverage":     0.40,
        "embedding_sim":        0.35,
        "section_completeness": 0.25,
    })

    log.info("Computing ATS score …")

    # ── 1. Keyword coverage ───────────────────────────────────────
    jd_kws     = _extract_keywords(jd_text)
    draft_kws  = _extract_keywords(final_draft)
    covered    = jd_kws & draft_kws
    kw_coverage = round(len(covered) / max(len(jd_kws), 1), 4)
    log.info("Keyword coverage: %.2f%%  (%d/%d)",
             kw_coverage * 100, len(covered), len(jd_kws))

    # ── 2. Embedding similarity ───────────────────────────────────
    emb_sim = 0.0
    if _ST_AVAILABLE:
        model        = _get_embed_model(mn)
        draft_vec    = model.encode(final_draft, normalize_embeddings=True,
                                    show_progress_bar=False)
        draft_vec    = np.array(draft_vec, dtype=np.float32).reshape(1, -1)
        jd_vec       = jd_embedding.reshape(1, -1)
        emb_sim      = float(cosine_similarity(draft_vec, jd_vec)[0][0])
        emb_sim      = round(max(0.0, emb_sim), 4)
        log.info("Embedding similarity (draft ↔ JD): %.4f", emb_sim)
    else:
        log.warning("sentence-transformers unavailable — embedding similarity set to 0.")

    # ── 3. Section completeness ───────────────────────────────────
    sec_score, section_present = _compute_section_completeness(final_draft)
    log.info("Section completeness: %.2f%%  sections=%s", sec_score * 100, section_present)

    # ── 4. Weighted composite score ───────────────────────────────
    w_kw  = weights.get("keyword_coverage", 0.40)
    w_emb = weights.get("embedding_sim", 0.35)
    w_sec = weights.get("section_completeness", 0.25)

    composite = (kw_coverage * w_kw) + (emb_sim * w_emb) + (sec_score * w_sec)
    ats_score = round(composite * 100, 1)

    # ── 5. Letter grade ───────────────────────────────────────────
    grade = "A" if ats_score >= 85 else \
            "B" if ats_score >= 70 else \
            "C" if ats_score >= 55 else \
            "D" if ats_score >= 40 else "F"

    missing_kws = sorted(jd_kws - draft_kws)

    report = {
        "ats_score":         ats_score,
        "grade":             grade,
        "breakdown": {
            "keyword_coverage":     round(kw_coverage * 100, 1),
            "embedding_similarity": round(emb_sim * 100, 1),
            "section_completeness": round(sec_score * 100, 1),
        },
        "weights":           weights,
        "section_presence":  section_present,
        "jd_keywords_total": len(jd_kws),
        "keywords_covered":  len(covered),
        "keywords_missing":  missing_kws[:30],
    }

    print(f"\n  {'='*45}")
    print(f"  ATS SCORE : {ats_score}/100  (Grade: {grade})")
    print(f"  {'='*45}")
    print(f"  Keyword coverage     : {kw_coverage:.1%}  ({len(covered)}/{len(jd_kws)})")
    print(f"  Embedding similarity : {emb_sim:.1%}")
    print(f"  Section completeness : {sec_score:.1%}  {section_present}")
    print(f"  {'='*45}")
    log.info("ATS score: %.1f/100  grade=%s", ats_score, grade)

    return report


print("✓ compute_ats_score() defined.")

# ────────────────────────────────────────────────────────────────
# PII DE-SANITIZATION
# ────────────────────────────────────────────────────────────────

# Matches both typed tokens (<EMAIL_1_hash>) and legacy generic tokens (<TOKEN_N_hash>)
_TOKEN_RE = re.compile(r'<[A-Z_]+_\d+_[0-9a-fA-F]+>')


def restore_pii(sanitized_text: str, vault: dict, *, authorized: bool = False) -> str:
    """
    Replace all ``<TOKEN_N_hash>`` placeholders in *sanitized_text* with
    their original PII values stored in *vault*.

    Parameters
    ----------
    sanitized_text : str   — Text containing ``<TOKEN_N_hash>`` placeholders.
    vault          : dict  — Mapping ``{token: original_value}`` loaded from
                             ``pii_vault.json``.
    authorized     : bool  — Must be explicitly set to ``True`` by the caller.
                             Acts as a human-in-the-loop gate to prevent
                             accidental PII exposure.

    Returns
    -------
    str  — Resume text with all tokens replaced by real PII.

    Raises
    ------
    RuntimeError  — If ``authorized=False`` or *vault* is empty.
    """
    if not authorized:
        raise RuntimeError(
            "restore_pii() requires authorized=True.  "
            "Explicitly pass authorized=True to confirm PII exposure is intentional."
        )

    if not vault:
        raise RuntimeError("PII vault is empty — cannot de-sanitize text.")

    log.info("Beginning PII restore.  Vault size: %d token(s).", len(vault))

    restored_count = 0
    unknown_tokens: list[str] = []

    def _replace(match: re.Match) -> str:
        nonlocal restored_count
        token = match.group(0)
        if token in vault:
            restored_count += 1
            return vault[token]
        unknown_tokens.append(token)
        return token          # keep verbatim — do not silently delete

    result = _TOKEN_RE.sub(_replace, sanitized_text)

    if unknown_tokens:
        log.warning(
            "%d unknown token(s) were left unchanged: %s",
            len(unknown_tokens),
            unknown_tokens[:10],
        )

    remaining = _TOKEN_RE.findall(result)
    if remaining:
        log.warning("Unreplaced tokens still present: %s", remaining[:10])

    log.info("PII restore complete.  Replaced %d token(s).", restored_count)
    return result


# ── Convenience: load vault from disk ────────────────────────────
def load_vault(vault_path: str | None = None) -> dict:
    """Load the PII vault JSON file and return a {token: original} dict."""
    path = vault_path or CONFIG["artifacts"]["pii_vault"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"PII vault not found at: {path}")
    with open(path, encoding="utf-8") as f:
        entries = json.load(f)
    # entries is a list of dicts; convert to token→original mapping
    vault = {e["token"]: e["original"] for e in entries}
    log.info("Loaded PII vault: %d token(s) from '%s'.", len(vault), path)
    return vault


print("✓ restore_pii() and load_vault() defined.")

# ────────────────────────────────────────────────────────────────
# SAVE FILE 2 OUTPUTS
# ────────────────────────────────────────────────────────────────

def save_file2_outputs(
    gap_report:      dict,
    ats_report:      dict,
    generated_draft: str,
    final_resume:    str,
    config:          dict | None = None,
) -> dict[str, str]:
    """
    Persist all File 2 pipeline artifacts to disk.

    Parameters
    ----------
    gap_report      : dict  — Output of compute_gap_analysis().
    ats_report      : dict  — Output of compute_ats_score().
    generated_draft : str   — Raw LLM-generated resume text.
    final_resume    : str   — PII-restored, guard-cleaned final resume.
    config          : dict  — Pipeline config for output paths.

    Returns
    -------
    dict  — Mapping of artifact name to absolute file path.
    """
    cfg  = config or CONFIG
    out  = cfg.get("output", {})

    saved: dict[str, str] = {}

    def _write_json(obj: dict, key: str, default: str) -> None:
        path = out.get(key, default)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, ensure_ascii=False)
        saved[key] = os.path.abspath(path)
        log.info("Saved %s → %s  (%d bytes)", key, path, os.path.getsize(path))

    def _write_text(text: str, key: str, default: str) -> None:
        path = out.get(key, default)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        saved[key] = os.path.abspath(path)
        log.info("Saved %s → %s  (%d bytes)", key, path, os.path.getsize(path))

    _write_json(gap_report,  "gap_report",     "gap_report.json")
    _write_json(ats_report,  "ats_report",     "ats_report.json")
    _write_text(generated_draft, "generated_draft", "generated_resume_draft.txt")
    _write_text(final_resume,    "final_resume",    "final_resume.txt")

    print("\n  Saved artifacts:")
    for name, path in saved.items():
        print(f"    • {name:<25} → {path}")

    return saved


print("✓ save_file2_outputs() defined.")

# ────────────────────────────────────────────────────────────────
# FULL PIPELINE ORCHESTRATOR
# ────────────────────────────────────────────────────────────────

import time as _time


def run_pipeline(
    jd_text:        str,
    config:         dict | None = None,
    artifacts_path: dict | None = None,
    *,
    authorized_pii: bool = False,
) -> dict:
    """
    Execute the complete File 2 pipeline end-to-end.

    Parameters
    ----------
    jd_text        : str   — Raw job description text.
    config         : dict  — Pipeline configuration; defaults to module-level CONFIG.
    artifacts_path : dict  — Override artifact file paths (embeddings, resume, vault).
    authorized_pii : bool  — Must be True to allow PII de-sanitization in final output.
                             If False the final resume retains placeholder tokens.

    Returns
    -------
    dict  — Comprehensive result with keys:
              gap_report, ats_report, generated_draft, final_resume,
              jd_chunks, content_plan, critic_score, saved_paths,
              runtime_seconds, config_snapshot
    """
    cfg    = config or CONFIG
    t0     = _time.perf_counter()
    result: dict = {}

    _banner = lambda s: print(f"\n{'═'*55}\n  {s}\n{'═'*55}")

    # ── Stage 0: Load artifacts ───────────────────────────────────
    _banner("Stage 0 — Loading artifacts")
    artifacts = load_artifacts(cfg)                        # sets ARTIFACTS global

    # ── Stage 1: Embed JD ─────────────────────────────────────────
    _banner("Stage 1 — Embedding job description")
    jd_emb = embed_job_description(
        text=jd_text,
        model_name=cfg["embedding_model"],
    )

    # ── Stage 2: Retrieve JD chunks from ChromaDB ─────────────────
    _banner("Stage 2 — Querying ChromaDB")
    jd_chunks = query_chromadb(jd_emb, config=cfg)
    result["jd_chunks"] = jd_chunks
    print(f"  Retrieved {len(jd_chunks)} relevant JD chunk(s).")

    # ── Stage 3: Gap analysis ─────────────────────────────────────
    _banner("Stage 3 — Gap analysis")
    gap_report = compute_gap_analysis(
        jd_embedding      = jd_emb,
        resume_embedding  = artifacts["resume_embedding"],
        jd_text           = jd_text,
        structured_resume = artifacts["structured_resume"],
    )
    result["gap_report"] = gap_report
    print(f"  Overall similarity : {gap_report.get('overall_similarity', 0):.1%}")
    print(f"  Missing keywords   : {len(gap_report.get('missing_keywords', []))}")

    # ── Stage 4: Bottleneck optimisation ─────────────────────────
    _banner("Stage 4 — Bottleneck layer")
    content_plan = bottleneck_layer(
        structured_resume = artifacts["structured_resume"],
        gap_report        = gap_report,
    )
    result["content_plan"] = content_plan

    # ── Stage 5: Generate resume draft ───────────────────────────
    _banner("Stage 5 — LLM generation (Ollama)")
    raw_draft, _model_used = generate_resume(
        content_plan  = content_plan,
        chroma_chunks = jd_chunks,
        jd_text       = jd_text,
        config        = cfg,
    )
    result["generated_draft"] = raw_draft

    # ── Stage 6: Critic loop ──────────────────────────────────────
    _banner("Stage 6 — Critic loop (iterative refinement)")
    critic_result = critic_loop(
        initial_draft = raw_draft,
        model_used    = _model_used,
        chroma_chunks = jd_chunks,
        jd_text       = jd_text,
        content_plan  = content_plan,
        config        = cfg,
    )
    refined_draft = critic_result["final_draft"]
    critic_score  = critic_result["final_score"]
    result["critic_score"] = critic_score
    print(f"  Final critic score : {critic_score}/100")

    # ── Stage 7: Hallucination guard ─────────────────────────────
    _banner("Stage 7 — Hallucination guard")
    guard_result = hallucination_guard(
        draft             = refined_draft,
        resume_embedding  = artifacts["resume_embedding"],
        config            = cfg,
    )
    guarded_draft = guard_result["clean_draft"]
    print(f"  Hallucination ratio : {guard_result['hallucination_ratio']:.1%}  "
          f"pass={guard_result['pass_guard']}")

    # ── Stage 8: PII de-sanitization ─────────────────────────────
    _banner("Stage 8 — PII de-sanitization")
    vault = artifacts.get("vault")
    if vault is None:
        try:
            vault = load_vault(cfg["artifacts"]["pii_vault"])
        except FileNotFoundError:
            log.warning("PII vault not found — skipping de-sanitization.")
            vault = {}

    if vault and authorized_pii:
        final_resume = restore_pii(guarded_draft, vault, authorized=True)
        print("  PII tokens restored successfully.")
    else:
        final_resume = guarded_draft
        if vault and not authorized_pii:
            print("  NOTE: PII not restored — pass authorized_pii=True to enable.")
    result["final_resume"] = final_resume

    # ── Stage 9: ATS score ────────────────────────────────────────
    _banner("Stage 9 — ATS scoring")
    ats_report = compute_ats_score(
        final_draft  = final_resume,
        jd_text      = jd_text,
        jd_embedding = jd_emb,
        config       = cfg,
    )
    result["ats_report"] = ats_report

    # ── Stage 10: Save outputs ────────────────────────────────────
    _banner("Stage 10 — Saving outputs")
    saved_paths = save_file2_outputs(
        gap_report      = gap_report,
        ats_report      = ats_report,
        generated_draft = raw_draft,
        final_resume    = final_resume,
        config          = cfg,
    )
    result["saved_paths"] = saved_paths

    # ── Summary ───────────────────────────────────────────────────
    elapsed = round(_time.perf_counter() - t0, 1)
    result["runtime_seconds"]  = elapsed
    result["config_snapshot"]  = cfg.copy()

    _banner("Pipeline Complete")
    print(f"  ATS Score      : {ats_report['ats_score']}/100  (Grade {ats_report['grade']})")
    print(f"  Critic Score   : {critic_score}/100")
    print(f"  Total runtime  : {elapsed}s")
    print(f"  Final resume   : {saved_paths.get('final_resume', 'N/A')}")
    print(f"{'═'*55}\n")

    return result


print("✓ run_pipeline() defined.")