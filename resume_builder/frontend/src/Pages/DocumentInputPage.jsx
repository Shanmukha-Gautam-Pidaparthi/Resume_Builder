import { useState, useRef } from "react";

const API_BASE = "http://localhost:8000";
const STEPS = ["upload", "jobdesc", "resume"];

const GlobalStyle = ({ dark }) => (
  <style>{`
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=DM+Serif+Display:ital@0;1&family=JetBrains+Mono:wght@400;500&display=swap');
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --bg:           ${dark ? "#0d0f14"               : "#eef0f4"};
      --text:         ${dark ? "#e8e6e1"               : "#111827"};
      --subtext:      ${dark ? "#6e7280"               : "#6b7280"};
      --nav-bg:       ${dark ? "rgba(13,15,20,0.9)"    : "rgba(248,249,252,0.97)"};
      --nav-border:   ${dark ? "rgba(255,255,255,0.06)": "rgba(0,0,0,0.1)"};
      --border:       ${dark ? "rgba(255,255,255,0.07)": "rgba(0,0,0,0.1)"};
      --input-bg:     ${dark ? "rgba(255,255,255,0.04)": "rgba(0,0,0,0.04)"};
      --input-border: ${dark ? "rgba(255,255,255,0.09)": "rgba(0,0,0,0.14)"};
      --step-idle:    ${dark ? "#1e2230"               : "#e5e7eb"};
      --step-line:    ${dark ? "#2e3340"               : "#d1d5db"};
      --step-idle-c:  ${dark ? "#666"                  : "#9ca3af"};
      --editor-bg:    ${dark ? "#13161f"               : "#ffffff"};
      --divider:      ${dark ? "rgba(255,255,255,0.07)": "rgba(0,0,0,0.08)"};
      --ghost-border: ${dark ? "#2e3340"               : "#d1d5db"};
      --ghost-color:  ${dark ? "#8a90a0"               : "#6b7280"};
      --add-bg:       ${dark ? "rgba(200,240,100,0.06)": "rgba(80,130,0,0.05)"};
      --add-border:   ${dark ? "rgba(200,240,100,0.25)": "rgba(80,130,0,0.3)"};
      --add-color:    ${dark ? "#8aab3a"               : "#3a6b00"};
      --lime:         ${dark ? "#c8f064"               : "#5a9e00"};
      --logo-dot:     #c8f064;
    }
    body { background: var(--bg); color: var(--text); font-family: 'Sora', sans-serif; transition: background .25s, color .25s; }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--step-line); border-radius: 3px; }
    .app-shell { min-height: 100vh; background: var(--bg); transition: background .25s; }

    /* ── NAV ── */
    .nav { position: fixed; top: 0; left: 0; right: 0; z-index: 200; display: flex; align-items: center; justify-content: space-between; padding: 0 2.5rem; height: 60px; background: var(--nav-bg); backdrop-filter: blur(14px); border-bottom: 1px solid var(--nav-border); }
    .nav-logo { font-family: 'DM Serif Display', serif; font-size: 1.25rem; color: var(--text); letter-spacing: .02em; user-select: none; }
    .nav-logo span { color: var(--logo-dot); }
    .nav-right { display: flex; align-items: center; gap: .7rem; }
    .theme-btn { width: 34px; height: 34px; border-radius: 8px; border: 1px solid var(--ghost-border); background: transparent; cursor: pointer; font-size: 1rem; color: var(--subtext); display: flex; align-items: center; justify-content: center; transition: all .2s; }
    .theme-btn:hover { border-color: var(--lime); color: var(--lime); }
    .nav-login { background: transparent; border: 1px solid ${dark?"rgba(200,240,100,0.35)":"rgba(80,130,0,0.4)"}; color: var(--lime); padding: .38rem 1rem; border-radius: 6px; font-family: 'Sora',sans-serif; font-size: .78rem; font-weight: 500; cursor: pointer; transition: all .2s; }
    .nav-login:hover { background: ${dark?"rgba(200,240,100,0.08)":"rgba(80,130,0,0.07)"}; }

    /* ── STEPPER ── */
    .stepper { display: flex; align-items: center; justify-content: center; padding-top: 96px; padding-bottom: 2rem; }
    .step-item { display: flex; align-items: center; }
    .step-dot { width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: .72rem; font-weight: 600; transition: all .3s; }
    .step-dot.done   { background: ${dark?"#c8f064":"#5a9e00"}; color: #fff; }
    .step-dot.active { background: ${dark?"#c8f064":"#5a9e00"}; color: #fff; box-shadow: 0 0 0 4px ${dark?"rgba(200,240,100,.2)":"rgba(80,130,0,.18)"}; }
    .step-dot.idle   { background: var(--step-idle); color: var(--step-idle-c); border: 1px solid var(--step-line); }
    .step-label { font-size: .7rem; font-weight: 500; letter-spacing: .06em; text-transform: uppercase; margin: 0 .5rem; }
    .step-label.active { color: var(--lime); }
    .step-label.done   { color: ${dark?"#8aab3a":"#3a6b00"}; }
    .step-label.idle   { color: var(--step-idle-c); }
    .step-line { width: 50px; height: 1px; background: var(--step-line); margin: 0 .3rem; }
    .step-line.done { background: ${dark?"#8aab3a":"#5a9e00"}; }

    /* ── PAGES ── */
    .page-center { max-width: 680px; margin: 0 auto; padding: 0 1.5rem 4rem; }

    /* ── UPLOAD ── */
    .upload-heading { font-family: 'DM Serif Display',serif; font-size: 2.5rem; line-height: 1.15; color: var(--text); margin-bottom: .5rem; }
    .upload-heading em { font-style: italic; color: var(--lime); }
    .sub { font-size: .88rem; color: var(--subtext); margin-bottom: 2.5rem; line-height: 1.65; }
    .dropzone { border: 1.5px dashed ${dark?"rgba(200,240,100,0.28)":"rgba(80,130,0,0.3)"}; border-radius: 12px; padding: 3rem 2rem; text-align: center; cursor: pointer; transition: all .22s; background: ${dark?"rgba(200,240,100,0.02)":"rgba(80,130,0,0.02)"}; margin-bottom: 1.5rem; }
    .dropzone:hover, .dropzone.drag { border-color: var(--lime); background: ${dark?"rgba(200,240,100,0.05)":"rgba(80,130,0,0.05)"}; }
    .dropzone-icon { font-size: 2.4rem; margin-bottom: 1rem; }
    .dropzone-text { font-size: .87rem; color: var(--subtext); line-height: 1.7; }
    .dropzone-text strong { color: var(--lime); }
    .dropzone input { display: none; }
    .file-list { display: flex; flex-direction: column; gap: .6rem; margin-bottom: 1.5rem; }
    .file-item { display: flex; align-items: center; justify-content: space-between; background: ${dark?"rgba(200,240,100,0.05)":"rgba(80,130,0,0.05)"}; border: 1px solid ${dark?"rgba(200,240,100,0.12)":"rgba(80,130,0,0.15)"}; border-radius: 8px; padding: .65rem 1rem; }
    .file-item-left { display: flex; align-items: center; gap: .65rem; }
    .file-name { font-size: .81rem; color: var(--text); font-family: 'JetBrains Mono',monospace; }
    .file-size { font-size: .7rem; color: var(--subtext); }
    .file-remove { background: none; border: none; cursor: pointer; color: var(--subtext); font-size: 1rem; transition: color .2s; }
    .file-remove:hover { color: #e06060; }

    /* ── JOB DESC ── */
    .jd-heading { font-family: 'DM Serif Display',serif; font-size: 2.4rem; color: var(--text); margin-bottom: .5rem; }
    .jd-heading em { font-style: italic; color: var(--lime); }
    textarea.jd-input { width: 100%; height: 240px; resize: vertical; background: var(--input-bg); border: 1px solid var(--input-border); border-radius: 10px; padding: 1.1rem 1.3rem; color: var(--text); font-family: 'Sora',sans-serif; font-size: .88rem; line-height: 1.7; outline: none; transition: border .2s; margin-top: 1.2rem; }
    textarea.jd-input:focus { border-color: ${dark?"rgba(200,240,100,0.4)":"rgba(80,130,0,0.5)"}; }
    textarea.jd-input::placeholder { color: var(--subtext); opacity: .55; }
    .tips { margin-top: 1.2rem; background: var(--input-bg); border-left: 2px solid ${dark?"rgba(200,240,100,0.3)":"rgba(80,130,0,0.35)"}; border-radius: 0 8px 8px 0; padding: .9rem 1.1rem; }
    .tips-title { font-size: .71rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--lime); margin-bottom: .45rem; }
    .tips ul { list-style: none; }
    .tips li { font-size: .78rem; color: var(--subtext); margin-bottom: .22rem; }
    .tips li::before { content: '→ '; color: var(--lime); }

    /* ── BUTTONS ── */
    .btn-primary { background: ${dark?"#c8f064":"#3d8b00"}; color: ${dark?"#0d0f14":"#fff"}; border: none; padding: .75rem 2rem; border-radius: 8px; font-family: 'Sora',sans-serif; font-weight: 600; font-size: .9rem; cursor: pointer; transition: all .2s; }
    .btn-primary:hover { background: ${dark?"#d8ff6e":"#2f6b00"}; transform: translateY(-1px); }
    .btn-primary:disabled { opacity: .35; cursor: not-allowed; transform: none; }
    .btn-ghost { background: transparent; border: 1px solid var(--ghost-border); color: var(--ghost-color); padding: .75rem 1.5rem; border-radius: 8px; font-family: 'Sora',sans-serif; font-size: .88rem; cursor: pointer; transition: all .2s; }
    .btn-ghost:hover { border-color: var(--subtext); color: var(--text); }
    .btn-row { display: flex; gap: 1rem; justify-content: flex-end; margin-top: 2rem; }

    /* ── GENERATING ── */
    @keyframes spin { to { transform: rotate(360deg); } }
    @keyframes pg { 0%,100%{opacity:.6}50%{opacity:1} }
    .generating { display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 60vh; gap: 1.5rem; }
    .gen-spinner { width: 58px; height: 58px; border-radius: 50%; border: 2px solid ${dark?"rgba(200,240,100,0.15)":"rgba(80,130,0,0.15)"}; border-top-color: var(--lime); animation: spin .9s linear infinite; }
    .gen-text { font-family: 'DM Serif Display',serif; font-size: 1.45rem; color: var(--lime); animation: pg 1.5s ease infinite; }
    .gen-sub { font-size: .8rem; color: var(--subtext); }
    .gen-error { color: #e06060; font-size: .85rem; text-align: center; max-width: 480px; }
  `}</style>
);

function Stepper({ step }) {
  const steps = [
    { id: "upload", label: "Upload" }, 
    { id: "jobdesc", label: "Job Brief" },
    { id: "resume", label: "Resume" }
  ];
  const idx = STEPS.indexOf(step);
  return (
    <div className="stepper">
      {steps.map((s, i) => {
        const status = i < idx ? "done" : i === idx ? "active" : "idle";
        return (
          <div className="step-item" key={s.id}>
            {i > 0 && <div className={`step-line${i <= idx ? " done" : ""}`} />}
            <div className={`step-dot ${status}`}>{i < idx ? "✓" : i + 1}</div>
            <span className={`step-label ${status}`}>{s.label}</span>
          </div>
        );
      })}
    </div>
  );
}

function UploadPage({ files, setFiles, onNext }) {
  const [drag, setDrag] = useState(false);
  const inputRef = useRef();
  
  const addFiles = fs => {
    const valid = Array.from(fs).filter(f => f.name.match(/\.(pdf|doc|docx|txt)$/i));
    setFiles(p => { const ex = new Set(p.map(f => f.name)); return [...p, ...valid.filter(f => !ex.has(f.name))]; });
  };
  
  const icon = f => f.name.endsWith(".pdf") ? "📄" : f.name.endsWith(".txt") ? "📝" : "📃";
  const sz = b => {
    if (!b || b === 0) return "reading…";
    if (b < 1024) return `${b} B`;
    if (b < 1048576) return `${(b / 1024).toFixed(1)} KB`;
    return `${(b / 1048576).toFixed(1)} MB`;
  };
  
  return (
    <div className="page-center">
      <h1 className="upload-heading">Drop your <em>career docs.</em><br />We'll handle the rest.</h1>
      <p className="sub">
        Name your files exactly as shown below — the AI uses filenames to understand each file's role.
      </p>

      {/* File naming guide */}
      <div style={{
        background: "var(--input-bg)", border: "1px solid var(--input-border)",
        borderRadius: "10px", padding: "1rem 1.3rem", marginBottom: "1.5rem",
        fontSize: ".8rem", lineHeight: "1.75"
      }}>
        <div style={{ fontWeight: 600, fontSize: ".7rem", letterSpacing: ".08em", textTransform: "uppercase", color: "var(--lime)", marginBottom: ".5rem" }}>Required file names</div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: ".25rem .8rem" }}>
          {[
            ["about.txt",            "Your name, contact, education, experience"],
            ["skills.txt",           "Technical skills (languages, tools, frameworks)"],
            ["projects.txt",         "Your projects with descriptions & tech stack"],
            ["certifications.docx",  "Certifications / licenses (optional)"],
          ].map(([name, desc]) => (
            <div key={name} style={{ display: "contents" }}>
              <span style={{ fontFamily: "'JetBrains Mono', monospace", color: "var(--lime)", fontSize: ".75rem" }}>{name}</span>
              <span style={{ color: "var(--subtext)", fontSize: ".75rem" }}>{desc}</span>
            </div>
          ))}
        </div>
      </div>

      <div className={`dropzone${drag?" drag":""}`} onClick={() => inputRef.current?.click()}
        onDragOver={e=>{e.preventDefault();setDrag(true);}} onDragLeave={()=>setDrag(false)}
        onDrop={e=>{e.preventDefault();setDrag(false);addFiles(e.dataTransfer.files);}}>
        <div className="dropzone-icon">⬆</div>
        <div className="dropzone-text"><strong>Click to browse</strong> or drag & drop<br />PDF · DOCX · TXT — multiple files welcome</div>
        <input ref={inputRef} type="file" multiple accept=".pdf,.doc,.docx,.txt" onChange={e=>addFiles(e.target.files)} />
      </div>
      {files.length > 0 && (
        <div className="file-list">
          {files.map((f, i) => (
            <div className="file-item" key={i}>
              <div className="file-item-left">
                <span style={{fontSize:"1.1rem"}}>{icon(f)}</span>
                <div>
                  <div className="file-name">{f.name}</div>
                  <div className="file-size">{sz(f.size)}</div>
                </div>
              </div>
              <button className="file-remove" onClick={()=>setFiles(p=>p.filter((_,j)=>j!==i))}>✕</button>
            </div>
          ))}
        </div>
      )}
      <div className="btn-row">
        <button className="btn-primary" disabled={files.length===0} onClick={onNext}>Continue → Job Description</button>
      </div>
    </div>
  );
}

function JobDescPage({ files, onBack, onNext }) {
  const [jd, setJd] = useState("");
  const [gen, setGen] = useState(false);
  const [error, setError] = useState("");
  
  const go = async () => {
    setGen(true);
    setError("");
    
    try {
      // Step 1: Upload files
      const formData = new FormData();
      files.forEach(f => formData.append("files", f));
      
      const uploadResp = await fetch(`${API_BASE}/api/upload`, {
        method: "POST",
        body: formData,
      });
      
      if (!uploadResp.ok) {
        const err = await uploadResp.json();
        throw new Error(err.detail || "File upload failed");
      }
      
      const uploadData = await uploadResp.json();
      const sessionId = uploadData.session_id;
      
      // Step 2: Generate resume
      const genFormData = new FormData();
      genFormData.append("session_id", sessionId);
      genFormData.append("job_description", jd);
      
      const genResp = await fetch(`${API_BASE}/api/generate`, {
        method: "POST",
        body: genFormData,
      });
      
      if (!genResp.ok) {
        const err = await genResp.json();
        throw new Error(err.detail || "Resume generation failed");
      }
      
      const result = await genResp.json();
      
      // Pass session_id along so ResumePage can download PDF
      onNext({
        sessionId,
        finalResume: result.final_resume,
        atsReport: result.ats_report,
        structuredResume: result.structured_resume,
      });
      
    } catch (err) {
      console.error("Pipeline error:", err);
      setError(err.message || "Something went wrong. Please try again.");
      setGen(false);
    }
  };
  
  if (gen) return (
    <div className="generating">
      <div className="gen-spinner" />
      <div className="gen-text">Crafting your resume…</div>
      <div className="gen-sub">Analysing · Matching keywords · Tailoring</div>
      {error && (
        <>
          <div className="gen-error">⚠ {error}</div>
          <button className="btn-ghost" onClick={() => { setGen(false); setError(""); }}>← Try Again</button>
        </>
      )}
    </div>
  );
  
  return (
    <div className="page-center">
      <h1 className="jd-heading">Paste the <em>job description.</em></h1>
      <p className="sub">Our AI reads the role requirements and tailors your resume to match — keywords, tone, and all.</p>
      <textarea className="jd-input" placeholder={"Paste the full job description here…\n\ne.g. 'We are looking for a Senior Frontend Engineer…'"} value={jd} onChange={e=>setJd(e.target.value)} />
      <div className="tips">
        <div className="tips-title">Pro tips</div>
        <ul>
          <li>Include the full JD — don't trim</li>
          <li>Company name & culture notes help personalise tone</li>
          <li>Salary / seniority clues shape emphasis</li>
        </ul>
      </div>
      <div className="btn-row">
        <button className="btn-ghost" onClick={onBack}>← Back</button>
        <button className="btn-primary" disabled={jd.trim().length<30} onClick={go}>✦ Generate Resume</button>
      </div>
    </div>
  );
}

export default function DocumentInputPage({ onNext, dark, onToggleDark }) {
  const [step, setStep] = useState("upload");
  const [files, setFiles] = useState([]);

  return (
    <>
      <GlobalStyle dark={dark} />
      <div className="app-shell">
        <nav className="nav">
          <div className="nav-logo">resume<span>.</span>ai</div>
          <div className="nav-right">
            <button className="theme-btn" onClick={onToggleDark} title="Toggle theme">
              {dark ? "☀️" : "🌙"}
            </button>
            <button className="nav-login">Log in / Sign up</button>
          </div>
        </nav>
        <Stepper step={step} />
        {step==="upload"  && <UploadPage files={files} setFiles={setFiles} onNext={() => setStep("jobdesc")} />}
        {step==="jobdesc" && <JobDescPage files={files} onBack={() => setStep("upload")} onNext={onNext} />}
      </div>
    </>
  );
}