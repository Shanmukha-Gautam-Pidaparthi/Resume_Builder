"""
FastAPI backend for Resume Builder.
Endpoints: upload files, generate resume, download PDF.

ChromaDB is READ-ONLY — pre-built by ingest_to_chroma.py.
This file never creates or writes to ChromaDB.
"""
from __future__ import annotations

import copy
import importlib
import sys
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
UPLOAD_DIR  = BACKEND_DIR / "uploads"
OUTPUT_DIR  = BACKEND_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Resume Builder API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_sessions: dict[str, dict] = {}


@app.get("/api/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# 1) Upload files
# ---------------------------------------------------------------------------
@app.post("/api/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    session_id  = uuid.uuid4().hex[:12]
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    for f in files:
        original_name = f.filename or f"file_{len(saved_paths)}"
        ext = Path(original_name).suffix.lower()
        if ext not in {".pdf", ".docx", ".doc", ".txt", ".md"}:
            continue
        dest = session_dir / original_name
        content = await f.read()
        with open(dest, "wb") as out:
            out.write(content)
        saved_paths.append(str(dest))

        # ── Debug: print file contents to console ─────────────────
        print(f"\n{'='*60}")
        print(f"📄 Received file: {original_name}  ({len(content)} bytes)")
        print(f"{'='*60}")
        if ext in {".txt", ".md"}:
            try:
                text = content.decode("utf-8", errors="replace")
                print(text)
            except Exception as e:
                print(f"  [Could not decode: {e}]")
        else:
            print(f"  [Binary file — {ext} — {len(content)} bytes saved]")

    if not saved_paths:
        raise HTTPException(400, "No valid files (.pdf .docx .txt .md)")

    _sessions[session_id] = {"files": saved_paths, "output_dir": None}
    return {"session_id": session_id, "uploaded": [Path(p).name for p in saved_paths]}


# ---------------------------------------------------------------------------
# 2) Generate resume
# ---------------------------------------------------------------------------
@app.post("/api/generate")
async def generate_resume(
    session_id: str = Form(...),
    job_description: str = Form(...),
):
    if session_id not in _sessions:
        raise HTTPException(404, "Session not found. Upload files first.")

    session    = _sessions[session_id]
    file_paths = session["files"]
    out_dir    = OUTPUT_DIR / session_id
    out_dir.mkdir(parents=True, exist_ok=True)
    session["output_dir"] = str(out_dir)

    try:
        # ── processor ────────────────────────────────────────────
        # Import fresh each time — critical so CONFIG mutations don't bleed
        # between concurrent requests. We deep-copy CONFIG for this request.
        import processor as _proc_module
        proc_cfg = copy.deepcopy(_proc_module.CONFIG)
        proc_cfg["output"]["structured_resume"]   = str(out_dir / "structured_resume.json")
        proc_cfg["output"]["pii_vault"]           = str(out_dir / "pii_vault.json")
        proc_cfg["output"]["embeddings"]          = str(out_dir / "resume_embeddings.npy")
        proc_cfg["output"]["chunks_embeddings"]   = str(out_dir / "resume_chunks_embeddings.npy")
        proc_cfg["output"]["preprocessed_chunks"] = str(out_dir / "preprocessed_chunks.json")

        proc_result = _proc_module.run_pipeline(
            input_files=file_paths,
            job_description=job_description,
            config=proc_cfg,
        )

        # ── generator (reads ChromaDB, never writes) ──────────────
        import generator as _gen_module
        gen_cfg = copy.deepcopy(_gen_module.CONFIG)
        gen_cfg["artifacts"]["resume_embeddings"] = str(out_dir / "resume_embeddings.npy")
        gen_cfg["artifacts"]["structured_resume"] = str(out_dir / "structured_resume.json")
        gen_cfg["artifacts"]["pii_vault"]         = str(out_dir / "pii_vault.json")
        gen_cfg["output"]["gap_report"]           = str(out_dir / "gap_report.json")
        gen_cfg["output"]["generated_draft"]      = str(out_dir / "generated_resume_draft.txt")
        gen_cfg["output"]["final_resume"]         = str(out_dir / "final_resume.txt")
        gen_cfg["output"]["ats_report"]           = str(out_dir / "ats_report.json")

        gen_result = _gen_module.run_pipeline(
            jd_text=job_description,
            config=gen_cfg,
            authorized_pii=True,
        )

        final_resume_text = ""
        final_resume_path = out_dir / "final_resume.txt"
        if final_resume_path.exists():
            final_resume_text = final_resume_path.read_text(encoding="utf-8")

        return JSONResponse({
            "status":            "success",
            "session_id":        session_id,
            "final_resume":      final_resume_text,
            "ats_report":        gen_result.get("ats_report", {}),
            "structured_resume": proc_result.get("structured_schema", {}),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Pipeline error: {str(e)}")


# ---------------------------------------------------------------------------
# 3) Download PDF
# ---------------------------------------------------------------------------
@app.get("/api/download-pdf/{session_id}")
def download_pdf(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(404, "Session not found.")

    session = _sessions[session_id]
    out_dir = session.get("output_dir")
    if not out_dir:
        raise HTTPException(400, "Resume not generated yet.")

    out_dir    = Path(out_dir)
    resume_txt = out_dir / "final_resume.txt"
    resume_pdf = out_dir / "final_resume.pdf"

    if not resume_txt.exists():
        raise HTTPException(404, "final_resume.txt not found.")

    try:
        from fpdf import FPDF
        text = resume_txt.read_text(encoding="utf-8")
        pdf  = FPDF()
        pdf.set_auto_page_break(auto=True, margin=20)
        pdf.add_page()
        pdf.set_font("Helvetica", size=11)

        for line in text.split("\n"):
            s = line.strip()
            if s and s == s.upper() and len(s) > 2 and s.replace(" ", "").isalpha():
                pdf.set_font("Helvetica", "B", 13)
                pdf.ln(4)
                pdf.cell(0, 8, s, new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", size=11)
            elif s.startswith(("•", "-", "▸")):
                pdf.cell(8)
                pdf.multi_cell(0, 6, s)
            elif s:
                pdf.multi_cell(0, 6, s)
            else:
                pdf.ln(3)

        pdf.output(str(resume_pdf))

    except ImportError:
        raise HTTPException(500, "fpdf2 not installed. Run: pip install fpdf2")
    except Exception as e:
        raise HTTPException(500, f"PDF generation failed: {e}")

    return FileResponse(path=str(resume_pdf), media_type="application/pdf", filename="resume.pdf")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)