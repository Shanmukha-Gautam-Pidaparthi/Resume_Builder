from __future__ import annotations

"""
processor.py — File 1 pipeline (converted from file1_v4.ipynb)
Handles: text extraction, PII detection/tokenization, embedding, structured schema.
ChromaDB is NOT used in this file.
"""

# ────────────────────────────────────────────────────────────────
# SYSTEM CONFIGURATION
# ────────────────────────────────────────────────────────────────

CONFIG = {
    # HuggingFace sentence-transformer model for dense embeddings
    "embedding_model": "BAAI/bge-base-en-v1.5",

    # Text chunking parameters
    "chunk_size": 100,   # reduced from 300 — short resumes need smaller chunks
    "chunk_overlap": 20,

    # PII detection categories — all active by default
    "pii_fields": [
        "name", "phone", "email", "address", "government_id",
        "demographics", "financial", "urls", "profiles", "social_handles",
    ],

    # Output file paths
    "output": {
        "structured_resume":  "structured_resume.json",
        "pii_vault":          "pii_vault.json",
        "embeddings":         "resume_embeddings.npy",
        "chunks_embeddings":  "resume_chunks_embeddings.npy",
        "preprocessed_chunks": "preprocessed_chunks.json",
    },

    # Logging
    "log_level": "DEBUG",

    # Optional chunk-level embeddings
    "embed_chunks": True,

    # ── File role mapping (Phase 1 — no-resume mode) ────────────
    # Files are routed to dedicated parsers based on their filename stem.
    # Matching is case-insensitive and ignores extension.
    # Any filename not listed here is treated as "general" (ignored by role-router,
    # but still merged into the raw text for embedding).
    #
    # Required:  about        → personal info, education, work experience
    #            skills       → technical skills
    #            projects     → projects
    # Optional:  certifications / certs / certificates → certifications
    "file_roles": {
        "about":            "about",
        "skills":           "skills",
        "projects":         "projects",
        "certifications":   "certifications",
        "certs":            "certifications",
        "certificates":     "certifications",
    },

    # Required file roles — pipeline warns (but continues) if missing
    "required_roles": ["about", "skills", "projects"],

    # Supported file extensions
    "supported_extensions": [".pdf", ".docx", ".doc", ".txt", ".md"],
}

print("✓ CONFIG loaded.")
print(f"  Embedding model : {CONFIG['embedding_model']}")
print(f"  Chunk size      : {CONFIG['chunk_size']} tokens (overlap {CONFIG['chunk_overlap']})")
print(f"  PII fields      : {len(CONFIG['pii_fields'])} categories active")
print(f"  File roles      : {list(CONFIG['file_roles'].values())}")
print(f"  Required roles  : {CONFIG['required_roles']}")


# ────────────────────────────────────────────────────────────────
# IMPORTS & PII-SAFE LOGGING SETUP
# ────────────────────────────────────────────────────────────────

import hashlib
import json
import logging
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import getpass

import numpy as np

# ── Document extraction ──────────────────────────────────────────
try:
    import fitz  # PyMuPDF
    _PYMUPDF_AVAILABLE = True
except ImportError:
    _PYMUPDF_AVAILABLE = False
    logging.warning("PyMuPDF not available — PDF extraction will fall back to pdfplumber.")

try:
    import pdfplumber
    _PDFPLUMBER_AVAILABLE = True
except ImportError:
    _PDFPLUMBER_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    _DOCX_AVAILABLE = True
except ImportError:
    _DOCX_AVAILABLE = False

# ── NLP / Embeddings ─────────────────────────────────────────────
try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    _SPACY_AVAILABLE = True
except Exception:
    _SPACY_AVAILABLE = False
    _nlp = None

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False


# ── PII-Safe Log Formatter ───────────────────────────────────────

_PII_LOG_PATTERNS = [
    re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'),   # email
    re.compile(r'\b\+?[\d\s\-().]{7,20}\b'),                                  # phone-like
    re.compile(r'https?://\S+', re.IGNORECASE),                               # URLs
    re.compile(r'www\.\S+', re.IGNORECASE),                                   # www URLs
]

class PIISafeFormatter(logging.Formatter):
    """Scrubs common PII patterns from log messages before emission."""

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        for pat in _PII_LOG_PATTERNS:
            msg = pat.sub("[REDACTED]", msg)
        return msg


def _setup_logger(name: str = "resume_pipeline", level: str = "DEBUG") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger   # already configured
    logger.setLevel(getattr(logging, level.upper(), logging.DEBUG))
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        PIISafeFormatter(
            fmt="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


log = _setup_logger(level=CONFIG.get("log_level", "DEBUG"))

log.info("Imports complete.")
log.info(f"PyMuPDF={_PYMUPDF_AVAILABLE}  pdfplumber={_PDFPLUMBER_AVAILABLE}  "
         f"python-docx={_DOCX_AVAILABLE}  spaCy={_SPACY_AVAILABLE}  "
         f"sentence-transformers={_ST_AVAILABLE}")

# ────────────────────────────────────────────────────────────────
# SALT PROVISIONING
# Priority:
#   1. Environment variable PII_SALT  (CI/secrets manager)
#   2. Existing salt file (.pii_salt) (persisted from a previous run)
#   3. Auto-generate a new cryptographically random salt → save to file
# ────────────────────────────────────────────────────────────────

import secrets

_SALT_FILE = Path(__file__).resolve().parent / ".pii_salt"   # always next to processor.py

def _get_salt() -> str:
    """
    Retrieve or generate the hashing salt.

    Returns the salt as a plain string. Never logs the value itself.
    """
    # ── 1. Environment variable ───────────────────────────────────
    salt = os.environ.get("PII_SALT", "").strip()
    if salt:
        log.info("Salt loaded from environment variable PII_SALT. length=%d", len(salt))
        return salt

    # ── 2. Persisted salt file ────────────────────────────────────
    if _SALT_FILE.exists():
        salt = _SALT_FILE.read_text(encoding="utf-8").strip()
        if salt:
            log.info("Salt loaded from persisted file: %s  length=%d", _SALT_FILE, len(salt))
            return salt
        log.warning("Salt file %s exists but is empty — generating new salt.", _SALT_FILE)

    # ── 3. Generate a new random salt and persist it ──────────────
    salt = secrets.token_hex(32)   # 256-bit random salt → 64-char hex string
    _SALT_FILE.write_text(salt, encoding="utf-8")
    try:
        import stat
        _SALT_FILE.chmod(stat.S_IRUSR | stat.S_IWUSR)   # 0o600 — owner read/write only
    except (AttributeError, NotImplementedError):
        pass   # Windows: chmod not fully supported; use icacls manually if needed

    log.info(
        "New random salt generated and saved → %s  length=%d  [value hidden]",
        _SALT_FILE, len(salt),
    )
    print(f"  ✓ New random salt generated → {_SALT_FILE}")
    print(f"    ⚠  Keep this file secret — add '{_SALT_FILE}' to .gitignore")
    print(f"    ⚠  Losing it means tokens cannot be re-mapped to originals.")
    return salt


# Obtain the salt once; passed explicitly to all hashing functions.
SALT: str = _get_salt()
log.info("Salt provisioned. Length=%d chars. [value hidden]", len(SALT))

# ────────────────────────────────────────────────────────────────
# MULTI-FILE INPUT HANDLING + TEXT EXTRACTION
# ────────────────────────────────────────────────────────────────

import glob as _glob

# ── Original single-file helpers (unchanged) ─────────────────────

def _clean_text(raw: str) -> str:
    """Normalise and clean extracted text."""
    text = unicodedata.normalize("NFKC", raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).rstrip() for line in text.split("\n")]
    result, blank_count = [], 0
    for line in lines:
        if line == "":
            blank_count += 1
            if blank_count <= 2:
                result.append(line)
        else:
            blank_count = 0
            result.append(line)
    return "\n".join(result).strip()


def _extract_pdf_pymupdf(path: Path) -> str:
    if not _PYMUPDF_AVAILABLE:
        raise ImportError("PyMuPDF (fitz) is not installed.")
    doc = fitz.open(str(path))
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return "\n".join(pages)


def _extract_pdf_pdfplumber(path: Path) -> str:
    if not _PDFPLUMBER_AVAILABLE:
        raise ImportError("pdfplumber is not installed.")
    pages = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text(x_tolerance=3, y_tolerance=3) or "")
    return "\n".join(pages)


def _extract_docx(path: Path) -> str:
    if not _DOCX_AVAILABLE:
        raise ImportError("python-docx is not installed.")
    doc = DocxDocument(str(path))
    pieces: list[str] = []
    for block in doc.element.body:
        tag = block.tag.split("}")[-1]
        if tag == "p":
            para_text = ""
            from docx.oxml.ns import qn
            para_text = "".join(
                node.text or "" for node in block.iter() if node.tag == qn("w:t")
            )
            pieces.append(para_text)
        elif tag == "tbl":
            for row in block:
                if row.tag.split("}")[-1] != "tr":
                    continue
                row_texts = []
                for cell in row:
                    if cell.tag.split("}")[-1] != "tc":
                        continue
                    cell_text = "".join(
                        node.text or "" for node in cell.iter() if node.tag.endswith("}t")
                    )
                    row_texts.append(cell_text.strip())
                pieces.append("  |  ".join(row_texts))
    return "\n".join(pieces)


def _extract_txt(path: Path) -> str:
    """Read plain text / markdown files with automatic encoding detection."""
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_bytes().decode("utf-8", errors="replace")


def extract_text(file_path) -> str:
    """
    Extract and clean text from a single file.

    Supported: .pdf, .docx, .doc, .txt, .md
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    log.debug("Extracting text from: %s (format=%s)", path.name, suffix)

    raw_text = ""
    if suffix == ".pdf":
        if _PYMUPDF_AVAILABLE:
            try:
                raw_text = _extract_pdf_pymupdf(path)
            except Exception as exc:
                log.warning("PyMuPDF failed (%s); falling back to pdfplumber.", exc)
        if not raw_text.strip() and _PDFPLUMBER_AVAILABLE:
            raw_text = _extract_pdf_pdfplumber(path)
        if not raw_text.strip():
            raise RuntimeError("PDF extraction failed with all backends.")

    elif suffix in (".docx", ".doc"):
        raw_text = _extract_docx(path)

    elif suffix in (".txt", ".md", ".text"):
        raw_text = _extract_txt(path)

    else:
        raise ValueError(f"Unsupported format '{suffix}'. Supported: .pdf .docx .txt .md")

    cleaned = _clean_text(raw_text)
    log.info("Extracted '%s' → %d chars.", path.name, len(cleaned))
    return cleaned


# ── NEW: Multi-file collector ─────────────────────────────────────

def collect_input_files(paths) -> list[Path]:
    """
    Resolve a flexible input specification to a validated list of Path objects.

    Parameters
    ----------
    paths : str | Path | list[str | Path]
        - A single file path          → [that file]
        - A directory path            → all supported files in that directory
        - A glob pattern              → matched files
        - A list of any of the above → combined and deduplicated

    Returns
    -------
    list[Path]  — Ordered, deduplicated list of readable files.
    """
    supported = set(CONFIG["supported_extensions"])
    if isinstance(paths, (str, Path)):
        paths = [paths]

    resolved: list[Path] = []
    seen: set[Path] = set()

    for entry in paths:
        entry = str(entry)

        # Glob pattern
        if any(c in entry for c in ("*", "?", "[")):
            matched = sorted(_glob.glob(entry, recursive=True))
            for m in matched:
                p = Path(m).resolve()
                if p.suffix.lower() in supported and p not in seen:
                    resolved.append(p)
                    seen.add(p)
            continue

        p = Path(entry).resolve()

        # Directory — collect all supported files inside
        if p.is_dir():
            for child in sorted(p.iterdir()):
                if child.suffix.lower() in supported and child not in seen:
                    resolved.append(child)
                    seen.add(child)
            continue

        # Single file
        if not p.exists():
            log.warning("Input file not found, skipping: %s", p)
            continue
        if p.suffix.lower() not in supported:
            log.warning("Unsupported extension '%s', skipping: %s", p.suffix, p.name)
            continue
        if p not in seen:
            resolved.append(p)
            seen.add(p)

    log.info("collect_input_files: resolved %d file(s): %s",
             len(resolved), [f.name for f in resolved])
    return resolved


# ── NEW: Extract from each file, return dict ──────────────────────

def extract_all_texts(file_paths: list[Path]) -> dict[str, str]:
    """
    Extract text from every file in file_paths.

    Returns
    -------
    dict[str, str]  — filename → cleaned text
    """
    results: dict[str, str] = {}
    for p in file_paths:
        try:
            text = extract_text(p)
            results[p.name] = text
            log.info("  ✓ %s (%d chars)", p.name, len(text))
        except Exception as exc:
            log.warning("  ✗ Failed to extract '%s': %s", p.name, exc)
    return results


# ── NEW: Merge texts from all files ──────────────────────────────

def merge_texts(file_texts: dict[str, str]) -> str:
    """
    Concatenate texts from multiple files with clear source separators.

    Each file's content is preceded by a header like:
        ─── SOURCE: filename.txt ───
    so the schema parser and synthesiser know where each piece came from.
    """
    if not file_texts:
        raise ValueError("No text content to merge.")

    parts: list[str] = []
    for fname, text in file_texts.items():
        header = f"─── SOURCE: {fname} ───"
        parts.append(f"{header}\n{text}")

    merged = "\n\n".join(parts)
    log.info("Merged %d source(s) → %d total chars.", len(file_texts), len(merged))
    return merged


# ── File role router — Phase 1 (no-resume mode) ─────────────────

def route_files_by_role(
    file_texts: dict[str, str],
    config:     dict | None = None,
) -> dict[str, str]:
    """
    Assign each file to a role based on its filename stem.

    Role mapping (case-insensitive, extension stripped):
        about           → "about"        (personal info, education, experience)
        skills          → "skills"       (technical skills)
        projects        → "projects"     (projects)
        certifications  → "certifications"
        certs           → "certifications"
        certificates    → "certifications"
        anything else   → "general"      (included in raw merge but not role-parsed)

    Parameters
    ----------
    file_texts : dict[str, str]   filename → extracted text
    config     : dict             pipeline config (reads config["file_roles"])

    Returns
    -------
    dict[str, str]   role → text   (one text per role; last file wins if duplicates)

    Also logs warnings for any required roles that are missing.
    """
    cfg        = config or CONFIG
    role_map   = {k.lower(): v for k, v in cfg.get("file_roles", {}).items()}
    required   = set(cfg.get("required_roles", ["about", "skills", "projects"]))

    routed: dict[str, str] = {}

    for fname, text in file_texts.items():
        stem = Path(fname).stem.lower()   # e.g. "about" from "about.txt"
        # Exact match first, then substring match (handles "Input_1_Certifications_Aanya.docx")
        role = role_map.get(stem, None)
        if role is None:
            for key, mapped_role in role_map.items():
                if key in stem:
                    role = mapped_role
                    break
        if role is None:
            role = "general"
        if role != "general":
            if role in routed:
                log.warning(
                    "Duplicate file for role '%s': already have one, overwriting with '%s'.",
                    role, fname,
                )
            routed[role] = text
            log.info("  File '%s' → role '%s' (%d chars)", fname, role, len(text))
        else:
            log.info("  File '%s' → role 'general' (will be embedded but not role-parsed)", fname)

    # Warn for missing required roles
    for req in required:
        if req not in routed:
            log.warning(
                "Required file role '%s' not found in inputs. "
                "Expected a file named '%s.txt', '%s.docx', or '%s.pdf'. "
                "Pipeline will continue with empty data for this section.",
                req, req, req, req,
            )

    filled   = [r for r in required if r in routed]
    missing  = [r for r in required if r not in routed]
    log.info(
        "File role routing complete. filled=%s | missing=%s | general=%d",
        filled, missing,
        sum(1 for fn in file_texts if role_map.get(Path(fn).stem.lower(), "general") == "general"),
    )
    return routed


print("✓ collect_input_files() defined.")
print("✓ extract_text() defined.")
print("✓ extract_all_texts() defined.")
print("✓ merge_texts() defined.")
print("✓ route_files_by_role() defined.")


# ────────────────────────────────────────────────────────────────
# ROLE-BASED PROFILE SYNTHESISER
# ────────────────────────────────────────────────────────────────
# Each input file is parsed by a dedicated extractor that knows
# exactly what structure to expect. This prevents cross-file
# contamination (e.g. "TECHNICAL SKILLS" being extracted as a name).
# ────────────────────────────────────────────────────────────────


# ── Shared regex helpers ─────────────────────────────────────────

_EMAIL_RE_S  = re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b')
_PHONE_RE_S  = re.compile(r'(?<!\d)(\+?[\d][\d\s\-().]{7,18}\d)(?!\d)')
_URL_RE_S    = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
# FIX: also match bare linkedin/github URLs without https:// or www.
_BARE_PROFILE_RE = re.compile(
    r'\b(linkedin\.com/in/[\w\-/]+|github\.com/[\w\-/]+|gitlab\.com/[\w\-/]+)',
    re.IGNORECASE,
)
_BULLET_RE_S = re.compile(r'^[\-•*▸►·\u2022]\s*(.+)')
_DATE_RE_S   = re.compile(
    r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|'
    r'January|February|March|April|May|June|July|August|'
    r'September|October|November|December)?'
    r'\s*(20|19)?\d{2}\s*[-–—]\s*'
    r'(Present|Current|Now|Till date|'
    r'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)?\s*(20|19)?\d{0,4}',
    re.IGNORECASE,
)

# Section header words that must NEVER be treated as a person's name
_HEADER_WORDS = {
    "technical", "skills", "education", "experience", "projects", "summary",
    "objective", "profile", "certifications", "contact", "languages", "tools",
    "interests", "hobbies", "awards", "references", "publications", "overview",
    "about", "work", "employment", "career", "academic", "qualifications",
    "competencies", "expertise", "professional",
    # label-style field names that must never be treated as a name
    "location", "address", "phone", "mobile", "email", "linkedin",
    "github", "website", "city", "state", "country",
}


def _is_name_line(line: str) -> bool:
    """
    Return True only if a line looks like a real person's name.
    Rejects ALL-CAPS lines, source markers, section headers, lines with digits.
    """
    line = line.strip()
    if not line:
        return False
    # Reject source markers
    if line.startswith("───") or line.startswith("==="):
        return False
    # Reject ALL-CAPS lines (section headers)
    if line == line.upper() and line.replace(" ", "").isalpha():
        return False
    words = line.split()
    if not (2 <= len(words) <= 4):
        return False
    # Reject if contains digits
    if any(c.isdigit() for c in line):
        return False
    # Reject if any word is a known header keyword
    lower_words = {w.lower().rstrip(":") for w in words}
    if lower_words & _HEADER_WORDS:
        return False
    # Reject if contains email / url / phone
    if _EMAIL_RE_S.search(line) or _URL_RE_S.search(line) or _PHONE_RE_S.search(line):
        return False
    # Accept if all words start with uppercase (title case name)
    if all(w[0].isupper() for w in words if w):
        return True
    return False


# ── about.txt parser ─────────────────────────────────────────────

def _parse_about(text: str) -> dict:
    """
    Parse about.txt (or about.docx) which contains personal info,
    education, and optionally work experience.

    Expected loose structure (user can write in any order):
        Name
        Email / Phone / GitHub / LinkedIn
        Education section (degree, institution, dates, GPA)
        Work Experience section (role, company, dates, bullets)
        Summary / Objective (optional)
    """
    personal: dict[str, str] = {}
    education: list[dict]    = []
    experience: list[dict]   = []
    summary: str             = ""

    lines = [ln for ln in text.split("\n")
             if not ln.strip().startswith("───") and not ln.strip().startswith("===")]

    # ── Extract name (first name-like line in top 20) ────────────
    _LABEL_PREFIX_RE = re.compile(
        r'^\s*(full\s*name|name|candidate|applicant|location|address|'
        r'phone|mobile|email|linkedin|github|website|url)\s*[:\-]\s*',
        re.IGNORECASE,
    )
    for line in lines[:20]:
        stripped_line = _LABEL_PREFIX_RE.sub("", line).strip()
        if "," in stripped_line or ":" in stripped_line:
            continue
        if _is_name_line(stripped_line):
            personal["name"] = stripped_line
            break

    # ── Extract email, phone, URLs ───────────────────────────────
    full_text = "\n".join(lines)
    emails = _EMAIL_RE_S.findall(full_text)
    if emails:
        personal["email"] = emails[0]

    phones = _PHONE_RE_S.findall(full_text)
    if phones:
        personal["phone"] = phones[0].strip()

    # Standard https:// and www. URLs
    for u in _URL_RE_S.findall(full_text):
        ul = u.lower()
        if "github"    in ul: personal.setdefault("github",    u)
        elif "linkedin" in ul: personal.setdefault("linkedin",  u)
        elif "portfolio" in ul or "website" in ul: personal.setdefault("portfolio", u)
    # FIX 1: bare URLs (linkedin.com/... github.com/... without https://)
    for m in _BARE_PROFILE_RE.finditer(full_text):
        u  = m.group(0)
        ul = u.lower()
        if "github"   in ul: personal.setdefault("github",   u)
        elif "linkedin" in ul: personal.setdefault("linkedin", u)
        elif "gitlab"   in ul: personal.setdefault("gitlab",   u)

    # ── Section-split: EDUCATION / EXPERIENCE / SUMMARY ──────────
    _ABOUT_SECTIONS = {
        "education":  re.compile(r'^\s*(education|academic|qualifications?)\s*:?\s*$', re.IGNORECASE),
        "experience": re.compile(r'^\s*(experience|work experience|employment|work history|professional experience)\s*:?\s*$', re.IGNORECASE),
        "summary":    re.compile(r'^\s*(summary|objective|profile|about me|career objective)\s*:?\s*$', re.IGNORECASE),
    }

    section_blobs: dict[str, list[str]] = {k: [] for k in _ABOUT_SECTIONS}
    current_section = None

    # FIX 2: inline-prefix patterns that terminate a section and capture a value
    _INLINE_PREFIX = re.compile(
        r'^\s*(interests?|hobbies|career\s*goals?|goals?|objective|summary)\s*:\s*(.+)',
        re.IGNORECASE,
    )

    inline_interests: str = ""
    inline_goals:     str = ""
    inline_summary:   str = ""

    for line in lines:
        stripped = line.strip()

        # FIX 2: detect inline-prefix lines FIRST — they break any active section
        ip = _INLINE_PREFIX.match(stripped)
        if ip:
            key   = ip.group(1).lower().replace(" ", "")
            value = ip.group(2).strip()
            if "interest" in key or "hobb" in key:
                inline_interests = value
            elif "goal" in key or "objective" in key:
                inline_goals = value
            elif "summary" in key:
                inline_summary = value
            current_section = None   # stop feeding into education/experience
            continue

        matched_section = None
        for sec, pat in _ABOUT_SECTIONS.items():
            if pat.match(stripped):
                matched_section = sec
                break
        if matched_section:
            current_section = matched_section
        elif current_section:
            section_blobs[current_section].append(line)

    # ── Parse education blob ──────────────────────────────────────
    edu_text = "\n".join(section_blobs["education"]).strip()
    # FIX 3: inline-prefix terminator — stop education parsing at these keywords
    _EDU_STOP_RE = re.compile(
        r'^\s*(interests?|hobbies|career\s*goals?|goals?|objective|summary|'
        r'achievements?|awards?|references?)\s*:',
        re.IGNORECASE,
    )
    if edu_text:
        degree_re = re.compile(
            r'\b(B\.?Tech|M\.?Tech|B\.?E\.?|M\.?E\.?|B\.?Sc\.?|M\.?Sc\.?|'
            r'B\.?A\.?|M\.?A\.?|Ph\.?D\.?|MBA|BBA|BCA|MCA|'
            r'Bachelor|Master|Doctor|Associate|Diploma|Certificate)\b',
            re.IGNORECASE,
        )
        gpa_re = re.compile(r'\b(GPA|CGPA|Score|Grade)[:\s]*([\d.]+)', re.IGNORECASE)
        current_edu: dict = {}
        for line in edu_text.split("\n"):
            line = line.strip()
            # FIX 3: stop immediately if we hit an inline-prefix line
            if _EDU_STOP_RE.match(line):
                break
            if not line:
                if current_edu:
                    education.append(current_edu)
                    current_edu = {}
                continue
            dm = degree_re.search(line)
            if dm:
                if current_edu:
                    education.append(current_edu)
                current_edu = {"degree": dm.group(0), "description": line}
            dates = _DATE_RE_S.findall(line)
            if dates:
                current_edu.setdefault("dates", line.strip())
            gm = gpa_re.search(line)
            if gm:
                current_edu["gpa"] = gm.group(2)
            if line and "institution" not in current_edu and not dm:
                current_edu.setdefault("institution", line)
        if current_edu:
            education.append(current_edu)

    # ── Parse experience blob ──────────────────────────────────────
    exp_text = "\n".join(section_blobs["experience"]).strip()
    if exp_text:
        current_job: dict = {}
        for line in exp_text.split("\n"):
            stripped = line.strip()
            if not stripped:
                if current_job:
                    experience.append(current_job)
                    current_job = {}
                continue
            dr = _DATE_RE_S.search(stripped)
            bm = _BULLET_RE_S.match(stripped)
            if dr and not bm:
                if current_job:
                    experience.append(current_job)
                current_job = {"dates": dr.group(0).strip(), "bullets": [], "raw_header": stripped}
            elif bm:
                current_job.setdefault("bullets", []).append(bm.group(1).strip())
            else:
                if current_job and "title" not in current_job:
                    current_job["title"] = stripped
                elif current_job:
                    current_job.setdefault("company", stripped)
        if current_job:
            experience.append(current_job)

    # ── Summary + inline interests/goals ────────────────────────
    # FIX 2: prefer inline-prefix values; fall back to section blob
    summary = inline_summary or " ".join(section_blobs["summary"]).strip()

    log.info(
        "_parse_about: name=%s | email=%s | education=%d | experience=%d",
        personal.get("name", "MISSING"),
        "found" if personal.get("email") else "missing",
        len(education),
        len(experience),
    )

    if not personal.get("name"):
        log.warning(
            "_parse_about: Could not extract a name from about.txt. "
            "personal_info.name will be empty. "
            "Make sure the first line of about.txt is the candidate's full name."
        )

    return {
        "personal_info":     personal,
        "education":         education,
        "experience":        experience,
        "summary":           summary,
        "inline_interests":  inline_interests,
        "inline_goals":      inline_goals,
    }


# ── skills.txt parser ────────────────────────────────────────────

def _parse_skills_file(text: str) -> dict[str, list[str]]:
    """
    Parse skills.txt which has a structured format:

        TECHNICAL SKILLS
        Programming Languages
        Python
        Java (Basic)
        ...
        Web Development
        HTML
        CSS
        ...

    Category headers are identified by known keywords.
    Individual items on their own line (or comma-separated) are extracted.
    """
    # Category header → bucket key mapping
    _CAT_MAP = {
        "programming languages": "languages",
        "languages":             "languages",
        "web development":       "web",
        "web":                   "web",
        "mobile development":    "mobile",
        "mobile":                "mobile",
        "libraries":             "libraries",
        "tools":                 "tools",
        "technologies":          "technologies",
        "frameworks":            "frameworks",
        "databases":             "databases",
        "cloud":                 "cloud",
        "devops":                "devops",
        "soft skills":           "soft",
        "other":                 "other",
    }
    # Top-level section headers to skip (not a category)
    _SKIP_HEADERS = {"technical skills", "skills", "competencies", "expertise"}

    skills: dict[str, list[str]] = {}
    current_cat = "other"

    for line in text.split("\n"):
        # Strip source markers
        stripped = line.strip()
        if not stripped or stripped.startswith("───") or stripped.startswith("==="):
            continue

        lower = stripped.lower().rstrip(":")

        # Skip top-level section title
        if lower in _SKIP_HEADERS:
            continue

        # Detect category header
        matched_cat = None
        for key, bucket in _CAT_MAP.items():
            if lower == key or lower.startswith(key):
                matched_cat = bucket
                break

        if matched_cat:
            current_cat = matched_cat
            # Also capture anything after the colon on the same line
            if ":" in stripped:
                remainder = stripped.split(":", 1)[1].strip()
                if remainder:
                    items = [s.strip() for s in re.split(r'[,;|•]', remainder) if s.strip()]
                    skills.setdefault(current_cat, []).extend(items)
            continue

        # Regular skill line — may be comma-separated or a single item
        bm = _BULLET_RE_S.match(stripped)
        content = bm.group(1) if bm else stripped
        items = [s.strip() for s in re.split(r'[,;|•]', content) if s.strip()]
        skills.setdefault(current_cat, []).extend(items)

    # Deduplicate preserving order
    result = {k: list(dict.fromkeys(v)) for k, v in skills.items() if v}
    total = sum(len(v) for v in result.values())
    log.info("_parse_skills_file: %d categories, %d total skills", len(result), total)
    return result


# ── projects.txt parser ──────────────────────────────────────────

def _parse_projects_file(text: str) -> list[dict]:
    """
    Parse projects.txt which has the structure:

        PROJECTS
        1. Project Name
        Description line.
        Another description line.
        Technologies used: X, Y, Z.

    Each numbered entry becomes a project dict.
    """
    _TECH_RE  = re.compile(r'(?:Technologies used|Tech Stack|Built with|Tools|Stack)[:\s]+(.+)', re.IGNORECASE)
    _URL_RE_L = _URL_RE_S
    _NUM_RE   = re.compile(r'^\s*\d+[\.\)]\s+(.+)')   # "1. Project Name"

    _MD_HDR  = re.compile(r'^#{1,3}\s+(.+)')
    _BOLD_RE = re.compile(r'^\*{1,2}(.+?)\*{1,2}\s*$')
    _PROJ_SKIP = {"projects","my projects","personal projects","key projects",
                  "technologies","tech stack","built with","tools","stack"}

    def _is_proj_heading(s):
        if not s or len(s) > 120: return None
        lower = s.lower().rstrip(":")
        if lower in _PROJ_SKIP: return None
        nm2 = _NUM_RE.match(s)
        if nm2: return nm2.group(1).strip()
        mh = _MD_HDR.match(s)
        if mh: return mh.group(1).strip()
        bm2 = _BOLD_RE.match(s)
        if bm2: return bm2.group(1).strip()
        words = s.split()
        if (2 <= len(words) <= 8 and not s.endswith(".") and "," not in s
                and not any(kw in lower for kw in _PROJ_SKIP)):
            return s
        return None

    projects: list[dict] = []
    current: dict = {}

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("───") or stripped.startswith("==="):
            continue
        lower = stripped.lower()
        if lower.rstrip(":") in _PROJ_SKIP:
            continue

        heading = None
        bm = _BULLET_RE_S.match(stripped)
        tm = _TECH_RE.search(stripped)
        um = _URL_RE_L.search(stripped)

        # Only check for heading if line is NOT a bullet
        if not bm:
            heading = _is_proj_heading(stripped)

        if heading:
            if current:
                projects.append(current)
            current = {"name": heading, "description": "", "technologies": [], "bullets": []}
        elif tm:
            techs = [t.strip() for t in re.split(r'[,;|]', tm.group(1)) if t.strip()]
            current.setdefault("technologies", []).extend(techs)
        elif um:
            current.setdefault("links", []).append(um.group(0))
        elif bm:
            current.setdefault("bullets", []).append(bm.group(1).strip())
        else:
            if current:
                # Append to description
                existing = current.get("description", "")
                current["description"] = (existing + " " + stripped).strip() if existing else stripped

    if current:
        projects.append(current)

    log.info("_parse_projects_file: %d projects extracted", len(projects))
    return projects


# ── certifications parser ────────────────────────────────────────

def _parse_certifications_file(text: str) -> list[dict]:
    """
    Parse certifications.docx / certs.txt.
    Extracts a list of certifications with name, issuer, and date if available.
    """
    _SKIP = {"certifications", "certificates", "licenses", "accreditations", "certs"}
    _DATE_PAT = re.compile(r'\b(20|19)\d{2}\b')
    _ISSUER_PAT = re.compile(r'(?:Issued by|Issuer|Provider|Platform|by)[:\s]+(.+)', re.IGNORECASE)

    certs: list[dict] = []
    current: dict = {}

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("───") or stripped.startswith("==="):
            continue
        if stripped.lower().rstrip(":") in _SKIP:
            continue

        bm    = _BULLET_RE_S.match(stripped)
        dm    = _DATE_PAT.search(stripped)
        im    = _ISSUER_PAT.search(stripped)
        nm_re = re.compile(r'^\s*\d+[\.\)]\s+(.+)')
        num_m = nm_re.match(stripped)

        content = bm.group(1) if bm else (num_m.group(1) if num_m else stripped)

        if im:
            current.setdefault("issuer", im.group(1).strip())
        elif dm:
            current.setdefault("date", dm.group(0))
            if current and "name" not in current:
                current["name"] = content
            elif current:
                certs.append(current)
                current = {}
        else:
            if current:
                certs.append(current)
            current = {"name": content}

    if current and "name" in current:
        certs.append(current)

    log.info("_parse_certifications_file: %d certifications extracted", len(certs))
    return certs


# ── Master synthesiser ───────────────────────────────────────────

def synthesise_profile(
    routed_texts: dict[str, str],
    config:       dict | None = None,
) -> dict:
    """
    Build a structured profile by routing each file to its dedicated parser.

    Parameters
    ----------
    routed_texts : dict[str, str]
        Output of route_files_by_role(). Keys are role names:
        "about", "skills", "projects", "certifications".
        Any key may be absent — missing roles produce empty defaults.

    config : dict  — pipeline config (currently unused but passed for future use).

    Returns
    -------
    dict  — structured schema compatible with File 2 downstream stages.
    """
    log.info("synthesise_profile: starting role-based extraction …")
    log.info("  Roles present : %s", list(routed_texts.keys()))

    # ── about.txt ────────────────────────────────────────────────
    if "about" in routed_texts:
        about_data    = _parse_about(routed_texts["about"])
        personal_info = about_data["personal_info"]
        education     = about_data["education"]
        experience    = about_data["experience"]
        summary       = about_data["summary"]
        # FIX 2: pick up inline interests/goals from about.txt
        _inline_interests = [s.strip() for s in about_data.get("inline_interests", "").split(",") if s.strip()]
        _inline_goals     = [about_data["inline_goals"]] if about_data.get("inline_goals") else []
    else:
        log.warning(
            "synthesise_profile: 'about' role missing. "
            "personal_info will be empty. Add about.txt with your name, "
            "contact details, education, and experience."
        )
        personal_info = {}
        education     = []
        experience    = []
        summary       = ""

    # ── skills.txt ───────────────────────────────────────────────
    if "skills" in routed_texts:
        skills = _parse_skills_file(routed_texts["skills"])
    else:
        log.warning("synthesise_profile: 'skills' role missing. skills will be empty.")
        skills = {}

    # ── projects.txt ─────────────────────────────────────────────
    if "projects" in routed_texts:
        projects = _parse_projects_file(routed_texts["projects"])
    else:
        log.warning("synthesise_profile: 'projects' role missing. projects will be empty.")
        projects = []

    # ── certifications (optional) ─────────────────────────────────
    if "certifications" in routed_texts:
        certifications = _parse_certifications_file(routed_texts["certifications"])
    else:
        certifications = []
        log.info("synthesise_profile: 'certifications' role not provided (optional — skipped).")

    # ── Collect profile URLs from about text ──────────────────────
    profiles: list[dict] = []
    seen_urls: set[str]  = set()
    about_text = routed_texts.get("about", "")

    def _add_profile(u: str) -> None:
        if u in seen_urls:
            return
        seen_urls.add(u)
        ul = u.lower()
        ptype = (
            "github"    if "github"    in ul else
            "linkedin"  if "linkedin"  in ul else
            "gitlab"    if "gitlab"    in ul else
            "portfolio" if ("portfolio" in ul or "website" in ul) else
            "website"
        )
        profiles.append({"type": ptype, "raw_url": u})

    # FIX 1: collect both https:// and bare profile URLs
    for m in _URL_RE_S.finditer(about_text):
        _add_profile(m.group(0))
    for m in _BARE_PROFILE_RE.finditer(about_text):
        _add_profile(m.group(0))

    schema = {
        "personal_info":  personal_info,
        "summary":        summary,
        "education":      education,
        "skills":         skills,
        "experience":     experience,
        "projects":       projects,
        "certifications": certifications,
        "profiles":       profiles,
        "interests":      _inline_interests if "about" in routed_texts else [],
        "goals":          _inline_goals     if "about" in routed_texts else [],
        "achievements":   [],
        "raw_sections":   {role: text for role, text in routed_texts.items()},
        "_metadata": {
            "parser_version": "3.0",
            "input_mode":     "notes_role_based",
            "roles_present":  list(routed_texts.keys()),
        },
    }

    log.info(
        "synthesise_profile complete. "
        "name=%s | education=%d | experience=%d | "
        "skills_cats=%d | projects=%d | certs=%d",
        personal_info.get("name", "MISSING"),
        len(education),
        len(experience),
        len(skills),
        len(projects),
        len(certifications),
    )
    return schema


print("✓ synthesise_profile() defined  (role-based, v3).")
print("✓ _parse_about() defined.")
print("✓ _parse_skills_file() defined.")
print("✓ _parse_projects_file() defined.")
print("✓ _parse_certifications_file() defined.")
print("✓ _is_name_line() defined.")


# ────────────────────────────────────────────────────────────────
# STRUCTURED SCHEMA PARSING
# ────────────────────────────────────────────────────────────────

# ── Section header detection ─────────────────────────────────────

_SECTION_KEYWORDS = {
    "personal_info":  ["contact", "personal information", "personal info", "about me"],
    "education":      ["education", "academic background", "qualifications", "academic"],
    "skills":         ["skills", "technical skills", "competencies", "technologies",
                       "expertise", "proficiencies"],
    "experience":     ["experience", "work experience", "employment", "work history",
                       "professional experience", "career"],
    "projects":       ["projects", "personal projects", "notable projects",
                       "key projects", "portfolio"],
    "profiles":       ["profiles", "links", "online profiles", "social", "portfolio links"],
    "certifications": ["certifications", "certificates", "licenses", "accreditations"],
    "summary":        ["summary", "objective", "profile", "professional summary",
                       "career objective", "about"],
    "awards":         ["awards", "honors", "achievements", "accomplishments"],
    "publications":   ["publications", "papers", "research"],
    "languages":      ["languages", "spoken languages"],
    "interests":      ["interests", "hobbies", "activities"],
    "references":     ["references"],
}

def _detect_section(line: str) -> str | None:
    """Return the canonical section name if a line is a section header, else None."""
    stripped = line.strip()
    # ALL-CAPS line (≥3 chars, no digits-only)
    candidate = stripped.lower().rstrip(":").strip()
    for section, keywords in _SECTION_KEYWORDS.items():
        for kw in keywords:
            if candidate == kw or stripped.upper() == stripped and candidate in kw:
                return section
    # fuzzy: line is ≤5 words, all caps or title case, matches a keyword
    if len(stripped.split()) <= 5:
        for section, keywords in _SECTION_KEYWORDS.items():
            if any(kw in candidate for kw in keywords):
                return section
    return None


def _split_into_sections(text: str) -> dict[str, str]:
    """
    Split resume text into raw section blobs keyed by canonical section name.
    Lines that match no known section are added to 'personal_info' (header region).
    """
    sections: dict[str, list[str]] = {"personal_info": []}
    current = "personal_info"

    for line in text.split("\n"):
        detected = _detect_section(line)
        if detected:
            current = detected
            if current not in sections:
                sections[current] = []
        else:
            sections.setdefault(current, []).append(line)

    return {k: "\n".join(v).strip() for k, v in sections.items() if v}


# ── Field-level parsers ───────────────────────────────────────────

_EMAIL_RE    = re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b')
_PHONE_RE    = re.compile(
    r'(?<!\d)'
    r'(\+?1?\s?)?'
    r'(\(?\d{2,4}\)?[\s.\-]?)'
    r'(\d{3,4}[\s.\-]?)'
    r'(\d{4})'
    r'(\s?(ext|x|ext\.)\s?\d{1,5})?'
    r'(?!\d)',
    re.IGNORECASE,
)
_URL_RE      = re.compile(r'https?://[^\s<>"\']+|www\.[^\s<>"\']+', re.IGNORECASE)
_PROFILE_RES = {
    "github":        re.compile(r'(https?://)?(www\.)?github\.com/[\w\-/.]+', re.IGNORECASE),
    "gitlab":        re.compile(r'(https?://)?(www\.)?gitlab\.com/[\w\-/.]+', re.IGNORECASE),
    "linkedin":      re.compile(r'(https?://)?(www\.)?linkedin\.com/in/[\w\-/.]+', re.IGNORECASE),
    "stackoverflow": re.compile(r'(https?://)?(www\.)?stackoverflow\.com/users/[\w\-/.]+', re.IGNORECASE),
    "twitter":       re.compile(r'(https?://)?(www\.)?twitter\.com/[\w\-]+', re.IGNORECASE),
    "medium":        re.compile(r'(https?://)?(www\.)?medium\.com/@?[\w\-/.]+', re.IGNORECASE),
    "kaggle":        re.compile(r'(https?://)?(www\.)?kaggle\.com/[\w\-/.]+', re.IGNORECASE),
}
_HANDLE_RE   = re.compile(r'@[A-Za-z0-9_]{2,30}\b')

def _extract_personal_info(blob: str) -> dict[str, str]:
    info: dict[str, str] = {}
    emails = _EMAIL_RE.findall(blob)
    if emails:
        info["email"] = emails[0]
    phones = _PHONE_RE.findall(blob)
    if phones:
        info["phone"] = "".join(phones[0]).strip()
    # Name heuristic: first non-empty line that is not an email/phone/URL
    for line in blob.split("\n"):
        line = line.strip()
        if (line
                and not _EMAIL_RE.search(line)
                and not _PHONE_RE.search(line)
                and not _URL_RE.search(line)
                and len(line.split()) <= 6
                and any(c.isalpha() for c in line)):
            info["name"] = line
            break
    # Address: line containing numeric prefix or known address words
    addr_re = re.compile(
        r'\b(\d{1,5}\s[\w\s.,-]{3,60}|'
        r'(street|st\.?|avenue|ave\.?|road|rd\.?|blvd|lane|ln\.?|drive|dr\.?|'
        r'city|state|zip|pincode|postal)\b)',
        re.IGNORECASE,
    )
    for line in blob.split("\n"):
        if addr_re.search(line):
            info["address"] = line.strip()
            break
    return info


def _extract_education(blob: str) -> list[dict]:
    """
    Heuristic education extraction.
    Looks for degree keywords and institution patterns.
    """
    degree_re = re.compile(
        r'\b(B\.?Tech|M\.?Tech|B\.?E\.?|M\.?E\.?|B\.?Sc\.?|M\.?Sc\.?|'
        r'B\.?A\.?|M\.?A\.?|Ph\.?D\.?|MBA|BBA|BCA|MCA|'
        r'Bachelor|Master|Doctor|Associate|Diploma|Certificate)\b',
        re.IGNORECASE,
    )
    date_re  = re.compile(r'\b(19|20)\d{2}\b')
    gpa_re   = re.compile(r'\b(GPA|CGPA|Score|Grade)[:\s]*([\d.]+)', re.IGNORECASE)

    entries, current = [], {}
    for line in blob.split("\n"):
        line = line.strip()
        if not line:
            if current:
                entries.append(current)
                current = {}
            continue
        dm = degree_re.search(line)
        if dm:
            if current:
                entries.append(current)
                current = {}
            current["degree"] = dm.group(0)
            current["description"] = line
        dates = date_re.findall(line)
        if dates:
            current.setdefault("dates", " – ".join(sorted(set(dates))))
        gm = gpa_re.search(line)
        if gm:
            current["gpa"] = gm.group(2)
        if line and "institution" not in current and not dm:
            current.setdefault("institution", line)
    if current:
        entries.append(current)
    return entries or [{"raw": blob}]


def _extract_skills(blob: str) -> dict[str, list[str]]:
    """
    Parse skills section into sub-categories.
    Falls back to returning a flat list under 'general'.
    """
    cat_re = re.compile(
        r'^(technical|programming|languages?|tools?|frameworks?|'
        r'soft\s*skills?|interpersonal|databases?|cloud|devops|'
        r'platforms?|other)[:\s]+',
        re.IGNORECASE,
    )
    skills: dict[str, list[str]] = {"technical": [], "tools": [], "soft": [], "languages": [], "other": []}
    current_cat = "other"

    for line in blob.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = cat_re.match(line)
        if m:
            cat_raw = m.group(1).lower().strip()
            if "language" in cat_raw:
                current_cat = "languages"
            elif "tool" in cat_raw or "framework" in cat_raw or "platform" in cat_raw:
                current_cat = "tools"
            elif "soft" in cat_raw or "interpersonal" in cat_raw:
                current_cat = "soft"
            else:
                current_cat = "technical"
            remainder = line[m.end():]
            items = [s.strip() for s in re.split(r'[,;|•·\u2022]', remainder) if s.strip()]
            skills[current_cat].extend(items)
        else:
            items = [s.strip() for s in re.split(r'[,;|•·\u2022]', line) if s.strip()]
            skills[current_cat].extend(items)

    # Deduplicate
    return {k: list(dict.fromkeys(v)) for k, v in skills.items() if v}


def _extract_experience(blob: str) -> list[dict]:
    """
    Heuristic work-experience extraction.
    Identifies job entries by date range patterns and bullet collections.
    """
    date_range_re = re.compile(
        r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|'
        r'March|April|June|July|August|September|October|November|December)?'
        r'\s*(20|19)?\d{2}\s*[-–—]\s*'
        r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|'
        r'March|April|June|July|August|September|October|November|December|'
        r'Present|Current|Till date|Now)?\s*(20|19)?\d{0,4}',
        re.IGNORECASE,
    )
    bullet_re = re.compile(r'^[-•*▸►·\u2022]\s*(.+)')

    entries, current = [], {}
    for line in blob.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        drm = date_range_re.search(stripped)
        bm  = bullet_re.match(stripped)
        if drm and not bm:
            if current:
                entries.append(current)
            current = {"dates": drm.group(0).strip(), "bullets": [], "raw_header": stripped}
        elif bm:
            current.setdefault("bullets", []).append(bm.group(1).strip())
        else:
            if current and "title" not in current:
                current["title"] = stripped
            elif current:
                current.setdefault("company", stripped)
    if current:
        entries.append(current)
    return entries or [{"raw": blob}]


def _extract_projects(blob: str) -> list[dict]:
    tech_re  = re.compile(
        r'(Technologies|Tech Stack|Built with|Tools|Stack)[:\s]+(.+)',
        re.IGNORECASE,
    )
    url_re   = _URL_RE
    bullet_re = re.compile(r'^[-•*▸►·\u2022]\s*(.+)')

    entries, current = [], {}
    for line in blob.split("\n"):
        stripped = line.strip()
        if not stripped:
            if current:
                entries.append(current)
                current = {}
            continue
        tm = tech_re.search(stripped)
        bm = bullet_re.match(stripped)
        um = url_re.search(stripped)
        if tm:
            techs = [t.strip() for t in re.split(r'[,;|]', tm.group(2)) if t.strip()]
            current.setdefault("technologies", []).extend(techs)
        elif um:
            current.setdefault("links", []).append(um.group(0))
        elif bm:
            current.setdefault("bullets", []).append(bm.group(1).strip())
        else:
            if "name" not in current:
                current["name"] = stripped
            else:
                current.setdefault("description", "")
                current["description"] = (current["description"] + " " + stripped).strip()
    if current:
        entries.append(current)
    return entries or [{"raw": blob}]


def _extract_profiles(blob: str, all_text: str = "") -> list[dict[str, str]]:
    """Extract profile URLs from the profiles blob and optionally the entire text."""
    scan_text = blob + "\n" + all_text
    found: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    for ptype, pat in _PROFILE_RES.items():
        for m in pat.finditer(scan_text):
            url = m.group(0).strip()
            if url not in seen_urls:
                seen_urls.add(url)
                found.append({"type": ptype, "raw_url": url})

    # Generic URLs not already captured
    for m in _URL_RE.finditer(scan_text):
        url = m.group(0).strip()
        if url not in seen_urls:
            seen_urls.add(url)
            found.append({"type": "website", "raw_url": url})

    return found


# ── Main schema parser ────────────────────────────────────────────

def parse_structured_schema(text: str) -> dict:
    """
    Parse free-form resume text into a structured JSON-compatible dictionary.

    Parameters
    ----------
    text : str
        Cleaned resume text (output of extract_text()).

    Returns
    -------
    dict
        Structured schema with keys:
        personal_info, education, skills, experience, projects, profiles, summary, raw_text.
    """
    log.info("Parsing structured schema …")
    sections = _split_into_sections(text)
    log.debug("Detected sections: %s", list(sections.keys()))

    schema: dict[str, Any] = {
        "personal_info": {},
        "summary":       "",
        "education":     [],
        "skills":        {},
        "experience":    [],
        "projects":      [],
        "profiles":      [],
        "raw_sections":  {},
        "_metadata": {
            "parser_version": "1.0",
            "sections_detected": list(sections.keys()),
        },
    }

    # personal_info
    pi_blob = sections.get("personal_info", "")
    header_blob = "\n".join(text.split("\n")[:20])  # first 20 lines often contain contact info
    schema["personal_info"] = _extract_personal_info(pi_blob or header_blob)

    # summary
    schema["summary"] = sections.get("summary", "")

    # education
    if "education" in sections:
        schema["education"] = _extract_education(sections["education"])

    # skills
    if "skills" in sections:
        schema["skills"] = _extract_skills(sections["skills"])

    # experience
    if "experience" in sections:
        schema["experience"] = _extract_experience(sections["experience"])

    # projects
    if "projects" in sections:
        schema["projects"] = _extract_projects(sections["projects"])

    # profiles — scan dedicated section + entire text
    schema["profiles"] = _extract_profiles(
        sections.get("profiles", ""),
        all_text=text,
    )

    # raw sections (for debugging / Fall 2 ingestion)
    schema["raw_sections"] = {k: v for k, v in sections.items()}

    log.info(
        "Schema parsed. personal_info keys=%s | experience=%d | "
        "education=%d | projects=%d | profiles=%d",
        list(schema["personal_info"].keys()),
        len(schema["experience"]),
        len(schema["education"]),
        len(schema["projects"]),
        len(schema["profiles"]),
    )
    return schema


print("✓ parse_structured_schema() defined.")

# ────────────────────────────────────────────────────────────────
# PII DETECTION
# ────────────────────────────────────────────────────────────────

# ── Pattern registry ──────────────────────────────────────────────

_PII_PATTERNS: dict[str, list[re.Pattern]] = {

    "email": [
        # Standard email
        re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'),
        # Obfuscated: john [at] example [dot] com or john(at)example(dot)com
        # NOTE: only matches word "at"/"dot" — standard @ and . are handled by the first regex above
        re.compile(
            r'\b[A-Za-z0-9._%+\-]+'
            r'\s*[\[(]?\s*(?:at)\s*[\])]?\s*'
            r'[A-Za-z0-9.\-]+'
            r'\s*[\[(]?\s*(?:dot)\s*[\])]?\s*'
            r'[A-Za-z]{2,}\b',
            re.IGNORECASE,
        ),
        # john(dot)doe(at)gmail(dot)com
        re.compile(
            r'\b[A-Za-z0-9]+(?:\(dot\)[A-Za-z0-9]+)*'
            r'\(at\)[A-Za-z0-9]+(?:\(dot\)[A-Za-z0-9]+)+\b',
            re.IGNORECASE,
        ),
    ],

    "phone": [
        # International: +1 (555) 555-5555, 555.555.5555, +91-9876543210, etc.
        re.compile(
            r'(?<!\d)'
            r'(\+?(\d{1,3})[\s\-.]?)?'    # country code
            r'(\(?\d{1,4}\)?[\s.\-]?)'    # area code
            r'(\d{2,4}[\s.\-]?){2,3}'     # subscriber
            r'(\s?(ext|x|ext\.)\s?\d{1,5})?'
            r'(?!\d)',
            re.IGNORECASE,
        ),
    ],

    "address": [
        # Heuristic: starts with a number followed by road/street terms
        re.compile(
            r'\b\d{1,5}\s+[\w\s.,-]{3,60}'
            r'(?:street|st\.?|avenue|ave\.?|road|rd\.?|blvd\.?|'
            r'lane|ln\.?|drive|dr\.?|court|ct\.?|place|pl\.?|'
            r'way|circle|terrace|parkway|highway|hwy)\b',
            re.IGNORECASE,
        ),
        # Postal code patterns: US ZIP, UK, CA, IN, AU
        re.compile(
            r'\b('
            r'\d{5}(-\d{4})?'            # US ZIP
            r'|[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}'  # UK postcode
            r'|\d{6}'                     # India PIN
            r'|[A-Z]\d[A-Z]\s?\d[A-Z]\d' # Canada
            r'|(?!(?:19|20)\d{2}\b)\d{4}'  # Australia (exclude years 1900-2099)
            r')\b',
        ),
    ],

    "government_id": [
        # US SSN: 123-45-6789 or 123456789
        re.compile(r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b'),
        # India Aadhaar: XXXX XXXX XXXX
        re.compile(r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b'),
        # India PAN: ABCDE1234F
        re.compile(r'\b[A-Z]{5}\d{4}[A-Z]\b'),
        # UK NINO: AA 99 99 99 A
        re.compile(r'\b[A-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b', re.IGNORECASE),
        # Passport: generic alphanumeric (context-aware min 7 chars)
        re.compile(r'\b(?:passport\s*(?:no|number|#)?[:.\s]*)?[A-Z]{1,2}\d{7,8}\b', re.IGNORECASE),
    ],

    "demographics": [
        # Date of Birth patterns
        re.compile(
            r'\b(DOB|Date of Birth|Born|Birth\s*Date)[:\s]*'
            r'\d{1,2}[\s/\-]\d{1,2}[\s/\-]\d{2,4}\b',
            re.IGNORECASE,
        ),
        # Full date patterns that might be DOB
        re.compile(
            r'\b\d{1,2}\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{4}\b',
            re.IGNORECASE,
        ),
        # Gender
        re.compile(
            r'\b(Gender|Sex)\s*[:\-]?\s*(Male|Female|Non[\s\-]?binary|'
            r'Transgender|Prefer not to say)\b',
            re.IGNORECASE,
        ),
        # Nationality / Religion / Marital status
        re.compile(
            r'\b(Nationality|Citizenship|Religion|Marital\s*Status|'
            r'Caste|Community)\s*[:\-]?\s*[A-Za-z\s]{2,30}\b',
            re.IGNORECASE,
        ),
    ],

    "financial": [
        # Salary / CTC — requires actual digits in the value to avoid matching plain words
        re.compile(
            r'\b(Salary|CTC|Compensation|Gross\s*Pay|Net\s*Pay|'
            r'Expected\s*Salary|Current\s*CTC)[:\s]+'
            r'[\$£€₹]?\s*\d[\d.,LKlk\s]*(?:per\s*(?:annum|month|year|pa))?',
            re.IGNORECASE,
        ),
        # IFSC code (India)
        re.compile(r'\b[A-Z]{4}0[A-Z0-9]{6}\b'),
        # PAN (also in gov IDs but captured here for financial context)
        re.compile(r'\b[A-Z]{5}\d{4}[A-Z]\b'),
    ],

    "urls": [
        re.compile(r'https?://[^\s<>"\'\]]+', re.IGNORECASE),
        re.compile(r'www\.[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}[^\s<>"\'\]]*', re.IGNORECASE),
        # URL shorteners
        re.compile(r'\b(?:bit\.ly|t\.co|tinyurl\.com|goo\.gl|ow\.ly)/[A-Za-z0-9\-_]+\b', re.IGNORECASE),
    ],

    "profiles": [_pat for _pat in _PROFILE_RES.values()],

    "social_handles": [
        re.compile(r'(?<!\w)@[A-Za-z0-9_]{2,30}\b'),
    ],

    "name": [],  # handled via spaCy NER + heuristics below
}


# ── Helper: canonicalize a detected value ────────────────────────

def _canonicalize(value: str, pii_type: str) -> str:
    """Normalize a detected PII value for consistent hashing."""
    v = value.strip()
    if pii_type in ("email", "profiles", "urls", "social_handles"):
        v = v.lower()
    # Remove UTM params and anchors from URLs
    if pii_type in ("urls", "profiles"):
        v = re.sub(r'[?#].+$', '', v)   # strip query string / fragment
        v = v.rstrip("/")
        v = re.sub(r'^https?://(www\.)?', '', v)   # strip scheme + www
        v = v.lower()
    if pii_type in ("phone",):
        v = re.sub(r'[\s\-.()+]', '', v)
    if pii_type == "government_id":
        v = re.sub(r'[\s\-]', '', v).upper()
    return v


# ── Name detection via spaCy ─────────────────────────────────────

def _detect_names_spacy(text: str) -> list[dict]:
    """Use spaCy NER to find PERSON entities."""
    if not _SPACY_AVAILABLE or _nlp is None:
        log.debug("spaCy unavailable — skipping NER name detection.")
        return []
    doc = _nlp(text[:50000])  # cap to 50k chars for performance
    results = []
    for ent in doc.ents:
        if ent.label_ == "PERSON" and len(ent.text.split()) >= 2:
            results.append({
                "original":  ent.text,
                "canonical": ent.text.strip().lower(),
                "type":      "name",
                "start":     ent.start_char,
                "end":       ent.end_char,
            })
    return results


# ── Profile type classifier ───────────────────────────────────────

def _classify_profile_type(url: str) -> str:
    """Return a fine-grained profile type label for a URL."""
    url_lower = url.lower()
    if "github.com" in url_lower:
        return "github"
    if "gitlab.com" in url_lower:
        return "gitlab"
    if "linkedin.com" in url_lower:
        return "linkedin"
    if "stackoverflow.com" in url_lower:
        return "stackoverflow"
    if "twitter.com" in url_lower or "x.com" in url_lower:
        return "twitter"
    if "medium.com" in url_lower:
        return "medium"
    if "kaggle.com" in url_lower:
        return "kaggle"
    if "bitbucket.org" in url_lower:
        return "bitbucket"
    return "website"


# ── Main PII detector ─────────────────────────────────────────────

def detect_pii(
    text: str,
    config: dict | None = None,
    salt: str = "",
) -> list[dict]:
    """
    Detect all PII instances in the given text.

    Parameters
    ----------
    text   : str   — The text to scan.
    config : dict  — Pipeline config (uses CONFIG global if None).
    salt   : str   — Hashing salt (required for determinism; not used in detection itself).

    Returns
    -------
    list[dict]
        Each entry: {original, canonical, type, start, end}
        Sorted by start offset, deduplicated by span overlap.
    """
    cfg = config or CONFIG
    active_fields: set[str] = set(cfg.get("pii_fields", list(_PII_PATTERNS.keys())))
    detections: list[dict] = []

    # ── regex-based detection ────────────────────────────────────
    for field, patterns in _PII_PATTERNS.items():
        if field not in active_fields or field == "name":
            continue
        for pat in patterns:
            for m in pat.finditer(text):
                matched = m.group(0).strip()
                if not matched or len(matched) < 2:
                    continue

                # ── Plausibility filters to reduce false positives ────
                # Email: must contain @ or the word "at" in parentheses
                if field == "email" and "@" not in matched and "(at)" not in matched.lower() and " at " not in matched.lower():
                    continue
                # Financial: must contain at least one digit
                if field == "financial" and not re.search(r'\d', matched):
                    continue

                # Refine type for profile URLs
                det_type = field
                if field == "profiles":
                    det_type = _classify_profile_type(matched)
                elif field == "urls":
                    # Re-classify generic URLs that are actually profiles
                    reclassified = _classify_profile_type(matched)
                    if reclassified != "website":
                        det_type = reclassified

                detections.append({
                    "original":  matched,
                    "canonical": _canonicalize(matched, det_type),
                    "type":      det_type,
                    "start":     m.start(),
                    "end":       m.end(),
                })

    # ── spaCy NER for names ───────────────────────────────────────
    if "name" in active_fields:
        name_hits = _detect_names_spacy(text)
        detections.extend(name_hits)

    # ── Sort by start offset ──────────────────────────────────────
    detections.sort(key=lambda d: d["start"])

    # ── Deduplicate overlapping spans ─────────────────────────────
    # Keep the longest span when two detections overlap; prefer more specific types.
    _TYPE_PRIORITY = {
        "github": 1, "gitlab": 1, "linkedin": 1, "stackoverflow": 1,
        "twitter": 1, "medium": 1, "kaggle": 1, "bitbucket": 1,
        "email": 2, "phone": 2, "name": 2, "government_id": 2,
        "demographics": 3, "financial": 3, "social_handles": 2,
        "urls": 4, "website": 4, "address": 5,
    }
    deduped: list[dict] = []
    for d in detections:
        if not deduped:
            deduped.append(d)
            continue
        prev = deduped[-1]
        # Check overlap
        if d["start"] < prev["end"]:
            # Keep the one with higher priority (lower number) or longer span
            p_pri = _TYPE_PRIORITY.get(prev["type"], 99)
            d_pri = _TYPE_PRIORITY.get(d["type"], 99)
            if d_pri < p_pri:
                deduped[-1] = d   # replace with higher-priority detection
            elif d_pri == p_pri and (d["end"] - d["start"]) > (prev["end"] - prev["start"]):
                deduped[-1] = d   # replace with longer match
            # else keep prev
        else:
            deduped.append(d)

    log.info(
        "PII detection complete. %d raw detections → %d after dedup.",
        len(detections), len(deduped),
    )
    # Mask originals in log output
    for d in deduped:
        masked = d["original"][:2] + "***" + d["original"][-2:]
        log.debug("  [%s] %s", d["type"], masked)

    return deduped


print("✓ detect_pii() defined.")

# ────────────────────────────────────────────────────────────────
# PII TOKENIZATION & VAULT MANAGEMENT
# ────────────────────────────────────────────────────────────────

# ── Type → token prefix map ───────────────────────────────────────

_TOKEN_PREFIX: dict[str, str] = {
    "name":           "NAME",
    "email":          "EMAIL",
    "phone":          "PHONE",
    "address":        "ADDR",
    "government_id":  "GOVID",
    "demographics":   "DEMO",
    "financial":      "FIN",
    "urls":           "URL",
    "website":        "URL",
    "profiles":       "URL",
    "github":         "GITHUB",
    "gitlab":         "GITLAB",
    "linkedin":       "LINKEDIN",
    "stackoverflow":  "SOFLOW",
    "twitter":        "TWITTER",
    "medium":         "MEDIUM",
    "kaggle":         "KAGGLE",
    "bitbucket":      "BITBUCKET",
    "social_handles": "HANDLE",
}


def _make_hash(salt: str, canonical: str) -> str:
    """
    Compute deterministic SHA-256 hash.
    Formula: SHA256(salt + canonicalized_original_value)
    Returns full hex digest (64 chars).
    """
    combined = (salt + canonical).encode("utf-8")
    return hashlib.sha256(combined).hexdigest()


def _make_token(pii_type: str, index: int, hash_value: str) -> str:
    """
    Construct a typed token.
    Format: <PREFIX_INDEX_HASH16>
    e.g. <GITHUB_1_a3f92b01c2d3e4f5>
    """
    prefix = _TOKEN_PREFIX.get(pii_type, "PII")
    short_hash = hash_value[:16]
    return f"<{prefix}_{index}_{short_hash}>"


def tokenize_pii(
    text: str,
    detections: list[dict],
    salt: str,
) -> tuple[str, list[dict]]:
    """
    Replace PII spans in text with deterministic typed tokens.

    Parameters
    ----------
    text       : str         — Original text containing PII.
    detections : list[dict]  — Output of detect_pii().
    salt       : str         — Hashing salt.

    Returns
    -------
    sanitized_text : str
        Text with all detected PII replaced by tokens.
    vault_entries : list[dict]
        List of vault records (token ↔ original mappings).
    """
    if not detections:
        log.info("No PII detections; text unchanged.")
        return text, []

    # ── Build canonical → token mapping (for determinism) ────────
    # Multiple occurrences of the same canonical value → same token
    canonical_to_token: dict[str, str]  = {}
    canonical_to_hash:  dict[str, str]  = {}
    type_counters:      dict[str, int]  = {}   # per-type sequential index
    vault_entries:      list[dict]      = []

    for det in detections:
        ckey = (det["type"], det["canonical"])
        if ckey not in canonical_to_token:
            h = _make_hash(salt, det["canonical"])
            canonical_to_hash[ckey] = h
            type_counters[det["type"]] = type_counters.get(det["type"], 0) + 1
            idx = type_counters[det["type"]]
            token = _make_token(det["type"], idx, h)
            canonical_to_token[ckey] = token
            vault_entries.append({
                "token":     token,
                "original":  det["original"],
                "canonical": det["canonical"],
                "hash":      h,
                "type":      det["type"],
            })
            log.debug(
                "  New token created: type=%s index=%d token=%s",
                det["type"], idx, token,
            )
        else:
            log.debug(
                "  Reusing existing token for duplicate %s value.",
                det["type"],
            )

    # ── Apply replacements right-to-left (preserves offsets) ─────
    # Sort detections by start descending
    ordered = sorted(detections, key=lambda d: d["start"], reverse=True)
    text_chars = list(text)

    for det in ordered:
        token = canonical_to_token[(det["type"], det["canonical"])]
        text_chars[det["start"]:det["end"]] = list(token)

    sanitized_text = "".join(text_chars)

    log.info(
        "Tokenization complete. %d unique PII values replaced. "
        "Vault entries=%d.",
        len(vault_entries), len(vault_entries),
    )
    return sanitized_text, vault_entries


# ── Vault I/O ────────────────────────────────────────────────────

def save_vault(vault_entries: list[dict], out_path: str | Path) -> None:
    """
    Persist the PII vault to disk as JSON.

    Parameters
    ----------
    vault_entries : list[dict]  — Output of tokenize_pii().
    out_path      : str | Path  — Destination file path.

    Security note
    -------------
    Immediately after saving, restrict file permissions:
      Linux/macOS : chmod 600 pii_vault.json
      Windows     : icacls pii_vault.json /inheritance:r /grant:r "%USERNAME%:F"
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    existing: list[dict] = []
    if out.exists():
        try:
            with open(out, "r", encoding="utf-8") as f:
                existing = json.load(f)
            log.debug("Loaded %d existing vault entries from %s.", len(existing), out.name)
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Could not read existing vault (%s); overwriting.", exc)
            existing = []

    # Merge: avoid duplicating tokens
    existing_tokens = {e["token"] for e in existing}
    new_entries = [e for e in vault_entries if e["token"] not in existing_tokens]
    merged = existing + new_entries

    with open(out, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    log.info(
        "Vault saved → %s  (%d new + %d existing = %d total entries).",
        out, len(new_entries), len(existing), len(merged),
    )
    print(f"  ⚠  Vault saved to: {out}")
    print("  ⚠  IMPORTANT: Restrict file permissions immediately:")
    print("       Linux/macOS : chmod 600", out.name)
    print("       Windows     : icacls", out.name, '/inheritance:r /grant:r "%USERNAME%:F"')


def load_vault(vault_path: str | Path) -> dict[str, str]:
    """
    Load vault and return a token → original mapping dict.

    Parameters
    ----------
    vault_path : str | Path  — Path to pii_vault.json.

    Returns
    -------
    dict[str, str]
        Maps token strings to their original PII values.
    """
    path = Path(vault_path)
    if not path.exists():
        raise FileNotFoundError(f"Vault file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    return {e["token"]: e["original"] for e in entries}


def restore_pii(
    sanitized_text: str,
    vault_path: str | Path,
    authorized: bool = False,
) -> str:
    """
    PRIVILEGED OPERATION — Restore original PII values from a sanitized text.

    Parameters
    ----------
    sanitized_text : str         — Text containing PII tokens.
    vault_path     : str | Path  — Path to pii_vault.json.
    authorized     : bool        — Must be explicitly True; prevents accidental calls.

    Returns
    -------
    str  — Text with tokens replaced by original values.

    Raises
    ------
    PermissionError  — If authorized=False.
    FileNotFoundError — If vault not found.
    """
    if not authorized:
        raise PermissionError(
            "restore_pii() requires authorized=True. "
            "This is a privileged operation — ensure caller is authorized before proceeding."
        )
    mapping = load_vault(vault_path)
    restored = sanitized_text
    for token, original in mapping.items():
        restored = restored.replace(token, original)
    log.info("PII restored for %d tokens. [authorized call]", len(mapping))
    return restored


# ── Schema tokenizer: walk schema dict and tokenize string values ──

def _tokenize_value(value: str, tokenize_fn) -> str:
    """Apply the tokenize function to a single string value."""
    sanitized, _ = tokenize_fn(value, detect_pii(value, CONFIG, SALT), SALT)
    return sanitized


def sanitize_schema(schema: dict, salt: str) -> tuple[dict, list[dict]]:
    """
    Walk the structured schema dict recursively and tokenize all PII string values.

    Returns
    -------
    sanitized_schema : dict
        Schema with PII values replaced by tokens.
    all_vault_entries : list[dict]
        Combined vault entries from all fields.
    """
    all_vault: list[dict] = []

    def _walk(obj: Any) -> Any:
        if isinstance(obj, str):
            dets = detect_pii(obj, CONFIG, salt)
            if dets:
                sanitized, v_entries = tokenize_pii(obj, dets, salt)
                all_vault.extend(v_entries)
                return sanitized
            return obj
        if isinstance(obj, list):
            return [_walk(item) for item in obj]
        if isinstance(obj, dict):
            return {k: _walk(v) for k, v in obj.items()}
        return obj

    sanitized = _walk(schema)
    log.info("Schema sanitized. Total vault entries from schema=%d.", len(all_vault))
    return sanitized, all_vault


print("✓ tokenize_pii() defined.")
print("✓ save_vault() defined.")
print("✓ load_vault() defined.")
print("✓ restore_pii() defined.")
print("✓ sanitize_schema() defined.")

# ────────────────────────────────────────────────────────────────
# RESUME EMBEDDING STAGE
# Uses BAAI/bge-base-en-v1.5 via sentence-transformers.
# Primary output: full-document embedding.
# Optional: chunked embeddings (CONFIG["embed_chunks"]).
# ────────────────────────────────────────────────────────────────

from typing import Optional

# ── Text chunker ─────────────────────────────────────────────────

def _chunk_text(
    text: str,
    chunk_size: int = 300,
    overlap: int = 50,
) -> list[str]:
    """
    Split text into overlapping word-level chunks.

    Parameters
    ----------
    text       : str  — Text to split.
    chunk_size : int  — Maximum words per chunk.
    overlap    : int  — Number of words shared between consecutive chunks.

    Returns
    -------
    list[str]  — List of text chunks.
    """
    if not text.strip():
        return []
    words = text.split()
    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i: i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    log.debug("Text chunked into %d chunks (size=%d, overlap=%d).", len(chunks), chunk_size, overlap)
    return chunks


# ── Model loader (cached) ─────────────────────────────────────────

_LOADED_MODELS: dict[str, Any] = {}

def _load_model(model_name: str) -> Any:
    """Load and cache a SentenceTransformer model."""
    if not _ST_AVAILABLE:
        raise ImportError(
            "sentence-transformers is not installed. "
            "Run: pip install sentence-transformers"
        )
    if model_name not in _LOADED_MODELS:
        log.info("Loading embedding model '%s' (first call — may download) …", model_name)
        _LOADED_MODELS[model_name] = SentenceTransformer(model_name)
        log.info("Model '%s' loaded. Embedding dim=%d.",
                 model_name,
                 _LOADED_MODELS[model_name].get_sentence_embedding_dimension())
    return _LOADED_MODELS[model_name]


# ── embed_text ────────────────────────────────────────────────────

def embed_text(
    text: str,
    model_name: str | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    Embed the full sanitized resume text into a single dense vector.

    Parameters
    ----------
    text       : str   — Sanitized resume text.
    model_name : str   — HuggingFace model name (defaults to CONFIG["embedding_model"]).
    normalize  : bool  — L2-normalize the output vector (recommended for cosine similarity).

    Returns
    -------
    np.ndarray  — Shape (embedding_dim,), dtype float32.

    Notes
    -----
    - BGE models benefit from mean-pooling over the full document.
    - For very long documents the model will truncate at its max_seq_length;
      this is acceptable for a full-document representation. Use chunk embeddings
      for fine-grained retrieval (handled by File 2).
    - This function does NOT write to ChromaDB.
    """
    if not text.strip():
        raise ValueError("Cannot embed empty text.")

    name = model_name or CONFIG["embedding_model"]
    model = _load_model(name)

    log.info("Embedding full document (%d chars) …", len(text))
    embedding = model.encode(
        text,
        normalize_embeddings=normalize,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    vec = np.array(embedding, dtype=np.float32)
    log.info("Full-document embedding computed. Shape=%s, dtype=%s.", vec.shape, vec.dtype)
    return vec


# ── embed_chunks ──────────────────────────────────────────────────

def embed_chunks(
    text: str,
    model_name: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    normalize: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """
    Embed the text as overlapping word-level chunks.

    Returns
    -------
    embeddings : np.ndarray   — Shape (n_chunks, embedding_dim), float32.
    chunks     : list[str]    — The text of each corresponding chunk.
    """
    name     = model_name    or CONFIG["embedding_model"]
    c_size   = chunk_size    or CONFIG["chunk_size"]
    c_overlap= chunk_overlap or CONFIG["chunk_overlap"]
    model    = _load_model(name)

    chunks = _chunk_text(text, chunk_size=c_size, overlap=c_overlap)
    if not chunks:
        raise ValueError("Text produced zero chunks.")

    log.info("Embedding %d chunks with model '%s' …", len(chunks), name)
    embeddings = model.encode(
        chunks,
        normalize_embeddings=normalize,
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=32,
    )
    arr = np.array(embeddings, dtype=np.float32)
    log.info("Chunk embeddings computed. Shape=%s.", arr.shape)
    return arr, chunks


print("✓ embed_text() defined.")
print("✓ embed_chunks() defined.")

# ────────────────────────────────────────────────────────────────
# SAVE OUTPUTS
# ────────────────────────────────────────────────────────────────

def save_outputs(
    structured_schema:   dict,
    full_embeddings:     np.ndarray,
    vault_entries:       list[dict],
    chunk_embeddings:    Optional[np.ndarray] = None,
    config:              dict | None = None,
) -> dict[str, str]:
    """
    Persist all File 1 pipeline artifacts to disk.

    Parameters
    ----------
    structured_schema  : dict         — Sanitized structured JSON schema.
    full_embeddings    : np.ndarray   — Full-document embedding vector.
    vault_entries      : list[dict]   — PII token→original vault entries.
    chunk_embeddings   : np.ndarray   — (optional) Per-chunk embeddings.
    config             : dict         — Pipeline config (defaults to CONFIG global).

    Returns
    -------
    dict[str, str]  — Mapping of artifact name → saved file path.
    """
    cfg = config or CONFIG
    out_cfg = cfg.get("output", {})

    saved: dict[str, str] = {}

    # ── structured_resume.json ────────────────────────────────────
    schema_path = Path(out_cfg.get("structured_resume", "structured_resume.json"))
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(structured_schema, f, indent=2, ensure_ascii=False)
    saved["structured_resume"] = str(schema_path)
    log.info("Structured schema saved → %s", schema_path)

    # ── pii_vault.json ────────────────────────────────────────────
    vault_path = Path(out_cfg.get("pii_vault", "pii_vault.json"))
    save_vault(vault_entries, vault_path)
    saved["pii_vault"] = str(vault_path)

    # ── resume_embeddings.npy ─────────────────────────────────────
    emb_path = Path(out_cfg.get("embeddings", "resume_embeddings.npy"))
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(emb_path), full_embeddings)
    saved["embeddings"] = str(emb_path)
    log.info(
        "Full-document embeddings saved → %s  shape=%s", emb_path, full_embeddings.shape
    )

    # ── resume_chunks_embeddings.npy (optional) ───────────────────
    if chunk_embeddings is not None and cfg.get("embed_chunks", False):
        chunks_path = Path(out_cfg.get("chunks_embeddings", "resume_chunks_embeddings.npy"))
        chunks_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(chunks_path), chunk_embeddings)
        saved["chunks_embeddings"] = str(chunks_path)
        log.info(
            "Chunk embeddings saved → %s  shape=%s", chunks_path, chunk_embeddings.shape
        )

    # ── preprocessed_chunks.json (for File 2 ChromaDB ingestion) ──
    # Save chunk texts alongside their embeddings so File 2 can
    # ingest them into ChromaDB without needing to re-chunk.
    if chunk_embeddings is not None and cfg.get("embed_chunks", False):
        chunk_texts = structured_schema.get("_chunk_texts", [])
        if chunk_texts:
            chunks_json_path = Path(
                out_cfg.get("preprocessed_chunks", "preprocessed_chunks.json")
            )
            chunks_json_path.parent.mkdir(parents=True, exist_ok=True)
            payload = [
                {"id": i, "text": t, "source": "resume"}
                for i, t in enumerate(chunk_texts)
            ]
            with open(chunks_json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            saved["preprocessed_chunks"] = str(chunks_json_path)
            log.info(
                "Preprocessed chunks saved → %s  (%d chunks)",
                chunks_json_path, len(payload),
            )

    log.info("All outputs saved. Summary: %s", saved)
    return saved


print("✓ save_outputs() defined.")

# ────────────────────────────────────────────────────────────────
# FULL PIPELINE RUNNER
# ────────────────────────────────────────────────────────────────

def run_pipeline(
    input_files,                  # str | Path | list | glob | directory
    job_description: str = "",
    config:          dict | None = None,
    salt:            str | None = None,
) -> dict:
    """
    Execute the complete File 1 pipeline.

    Phase 1 — No-Resume Mode
    -------------------------
    Files are routed to dedicated parsers by filename stem:
        about.txt        → personal info, education, experience  (required)
        skills.txt       → technical skills                       (required)
        projects.txt     → projects                               (required)
        certifications.* → certifications                         (optional)

    Parameters
    ----------
    input_files : str | Path | list
        File paths, glob pattern, or directory.
    job_description : str
        Job description text (stored, not embedded here).
    config : dict  — defaults to CONFIG global.
    salt   : str   — defaults to SALT global.
    """
    import time
    from typing import Optional

    cfg = config or CONFIG
    s   = salt   or SALT

    log.info("=" * 60)
    log.info("FILE 1 PIPELINE START")
    log.info("=" * 60)
    t0 = time.time()

    # ── Stage 1: Collect files ───────────────────────────────────
    log.info("[Stage 1/9] Collecting input files …")
    file_paths = collect_input_files(input_files)
    if not file_paths:
        raise FileNotFoundError(
            "No valid input files found. "
            "Provide at least one .pdf, .docx, .txt, or .md file."
        )
    log.info("Stage 1 done. Files: %s", [p.name for p in file_paths])

    # ── Stage 2: Extract text from all files ────────────────────
    # (Phase 1: no resume — all files are notes with known roles)
    log.info("[Stage 2/9] Extracting text from all input files …")

    file_texts = extract_all_texts(file_paths)
    if not file_texts:
        raise RuntimeError("Text extraction produced no output from any file.")
    log.info("Stage 2 done. Files with text: %d", len(file_texts))

    # ── Stage 3: Route files by role ─────────────────────────────
    log.info("[Stage 3/9] Routing files to roles …")
    routed_texts = route_files_by_role(file_texts, cfg)
    log.info("Stage 3 done. Roles filled: %s", list(routed_texts.keys()))

    # ── Stage 4: Merge all texts (for embedding + PII) ───────────
    log.info("[Stage 4/9] Merging texts …")
    raw_text = merge_texts(file_texts)
    log.info("Stage 4 done. Merged length=%d chars.", len(raw_text))

    # ── Stage 5: Build structured schema (role-based) ────────────
    log.info("[Stage 5/9] Building structured schema …")
    schema = synthesise_profile(routed_texts, cfg)
    log.info("Stage 5 done.")

    # ── Stage 6: Detect PII ──────────────────────────────────────
    log.info("[Stage 6/9] Detecting PII …")
    detections = detect_pii(raw_text, cfg, s)
    log.info("Stage 6 done. detections=%d", len(detections))

    # ── Stage 7: Tokenize PII ────────────────────────────────────
    log.info("[Stage 7/9] Tokenizing PII …")
    sanitized_text, vault_from_text = tokenize_pii(raw_text, detections, s)
    log.info("Stage 7 done. vault_entries=%d", len(vault_from_text))

    # ── Stage 8: Sanitize schema ──────────────────────────────────
    log.info("[Stage 8/9] Sanitizing structured schema …")
    sanitized_schema, vault_from_schema = sanitize_schema(schema, s)

    seen_tokens = {e["token"] for e in vault_from_text}
    combined_vault = list(vault_from_text)
    for entry in vault_from_schema:
        if entry["token"] not in seen_tokens:
            combined_vault.append(entry)
            seen_tokens.add(entry["token"])

    # Attach job description
    jd_sanitized = ""
    if job_description.strip():
        jd_dets = detect_pii(job_description, cfg, s)
        jd_sanitized, jd_vault = tokenize_pii(job_description, jd_dets, s)
        for entry in jd_vault:
            if entry["token"] not in seen_tokens:
                combined_vault.append(entry)
                seen_tokens.add(entry["token"])

    sanitized_schema["_job_description_sanitized"] = jd_sanitized
    sanitized_schema["_input_files"] = [p.name for p in file_paths]
    sanitized_schema["_resume_file"] = None   # Phase 1: no resume
    sanitized_schema["_input_mode"]  = "notes_role_based"
    log.info("Stage 8 done. total_vault_entries=%d", len(combined_vault))

    # ── Stage 9: Embed ────────────────────────────────────────────
    log.info("[Stage 9/9] Embedding …")
    full_emb = embed_text(sanitized_text, model_name=cfg["embedding_model"])
    log.info("Full-document embedding shape=%s", full_emb.shape)

    chunk_emb: Optional[np.ndarray] = None
    chunk_texts: list[str] = []
    if cfg.get("embed_chunks", False):
        chunk_emb, chunk_texts = embed_chunks(
            sanitized_text,
            model_name    = cfg["embedding_model"],
            chunk_size    = cfg["chunk_size"],
            chunk_overlap = cfg["chunk_overlap"],
        )
        sanitized_schema["_chunk_texts"] = chunk_texts
        log.info("Chunk embeddings shape=%s", chunk_emb.shape)

    log.info("Stage 9 done.")

    # ── Save outputs ──────────────────────────────────────────────
    saved = save_outputs(
        structured_schema = sanitized_schema,
        full_embeddings   = full_emb,
        vault_entries     = combined_vault,
        chunk_embeddings  = chunk_emb,
        config            = cfg,
    )

    elapsed = time.time() - t0
    stats = {
        "input_mode":      "notes_role_based",
        "roles_filled":    list(routed_texts.keys()),
        "files_processed": len(file_texts),
        "raw_text_chars":  len(raw_text),
        "sanitized_chars": len(sanitized_text),
        "pii_detections":  len(detections),
        "vault_entries":   len(combined_vault),
        "embedding_dim":   int(full_emb.shape[0]),
        "chunks_produced": len(chunk_texts),
        "elapsed_seconds": round(elapsed, 2),
    }

    log.info("=" * 60)
    log.info("FILE 1 COMPLETE in %.2fs | roles=%s | files=%d",
             elapsed, list(routed_texts.keys()), len(file_texts))
    log.info("=" * 60)

    return {
        "input_mode":        "notes_role_based",
        "routed_texts":      routed_texts,
        "file_texts":        file_texts,
        "raw_text":          raw_text,
        "sanitized_text":    sanitized_text,
        "structured_schema": sanitized_schema,
        "vault_entries":     combined_vault,
        "full_embedding":    full_emb,
        "chunk_embeddings":  chunk_emb,
        "saved_files":       saved,
        "stats":             stats,
    }


print("✓ run_pipeline() defined.")
print("  Phase 1: role-based notes mode.")
print("  Required files: about.txt, skills.txt, projects.txt")
print("  Optional files: certifications.docx")