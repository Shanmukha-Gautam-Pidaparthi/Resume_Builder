# Resume Builder

AI-powered resume builder that tailors your resume to job descriptions using Ollama LLMs.

## Project Structure

```
resume_builder/
├── backend/
│   ├── main.py              # FastAPI server (3 endpoints)
│   ├── processor.py         # File 1: text extraction, PII, embeddings
│   ├── generator.py         # File 2: gap analysis, LLM generation, ATS scoring
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── package.json         # React (CRA) with proxy to backend
│   ├── src/
│   │   ├── App.js
│   │   └── Pages/
│   │       ├── DocumentInputPage.jsx
│   │       └── ResumePage.jsx
│   └── public/
└── README.md
```

## Prerequisites

- **Python 3.10+**
- **Node.js 18+** and npm
- **Ollama** running locally with `mistral:7b-instruct-q4_K_M` model pulled

## Setup & Run

### Step 1 — Install & Start Ollama

```bash
# Install Ollama from https://ollama.com
ollama pull mistral:7b-instruct-q4_K_M
ollama pull phi3:mini          # fallback model (optional)
ollama serve                   # keep running in background
```

### Step 2 — Backend

```bash
cd resume_builder/backend
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python main.py
# → Runs at http://localhost:8000
```

### Step 3 — Frontend

```bash
cd resume_builder/frontend
npm install
npm start
# → Runs at http://localhost:3000
```

### Step 4 — Use

1. Open http://localhost:3000
2. Upload your documents (about.txt, skills.txt, projects.txt, etc.)
3. Paste the target job description
4. Click **Generate Resume**
5. Edit the result inline, then **Download PDF**

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Health check |
| `/api/upload` | POST | Upload PDF/DOCX/TXT files → returns `session_id` |
| `/api/generate` | POST | Run pipeline (processor → generator) → returns resume |
| `/api/download-pdf/{session_id}` | GET | Download generated resume as PDF |

## Input Files

For best results, name your input files:

| Filename | Content | Required? |
|---|---|---|
| `about.txt` | Name, contact, education, experience | ✅ Yes |
| `skills.txt` | Technical skills by category | ✅ Yes |
| `projects.txt` | Project descriptions | ✅ Yes |
| `certifications.docx` | Certifications | Optional |

## Notes

- The `.pii_salt` file is auto-generated — add it to `.gitignore`
- Ollama must be running at `http://localhost:11434` before starting the backend
- The frontend proxies API requests to `http://localhost:8000`
