import { useState, useRef, useCallback, useEffect } from "react";

const defaultResume = {
  name: "Alex Morgan",
  title: "Senior Frontend Engineer",
  email: "alex.morgan@email.com",
  phone: "+1 (555) 234-5678",
  location: "San Francisco, CA",
  linkedin: "linkedin.com/in/alexmorgan",
  summary: "Results-driven frontend engineer with 6+ years of experience building scalable web applications. Passionate about performance, accessibility, and clean code. Led teams across fintech and SaaS domains.",
  experience: [
    { id: 1, company: "Stripe", role: "Senior Frontend Engineer", period: "2021 – Present", bullets: ["Architected a component library used by 12 product teams, reducing UI dev time by 35%.", "Led migration of legacy React class components to hooks, improving test coverage to 92%.", "Collaborated with design & backend to ship Stripe's new Dashboard onboarding flow."] },
    { id: 2, company: "Figma", role: "Frontend Engineer", period: "2018 – 2021", bullets: ["Built real-time collaborative annotation features using WebSockets and CRDT.", "Optimized canvas rendering pipeline, reducing frame drops by 60% on large files.", "Mentored 3 junior engineers and conducted 50+ technical interviews."] },
  ],
  projects: [
    { id: 1, name: "OpenGrid", description: "Open-source CSS grid visualizer (2.1k GitHub stars). Built with Vite + TypeScript." },
    { id: 2, name: "ResumeAI", description: "Personal project: AI-powered resume tailoring tool using Claude API." },
  ],
  education: [{ id: 1, school: "UC Berkeley", degree: "B.S. Computer Science", year: "2018" }],
  skills: ["React", "TypeScript", "Next.js", "GraphQL", "Node.js", "Figma", "Tailwind CSS", "Jest", "Cypress", "AWS"],
};

const ACCENT_COLORS = [
  { label: "Navy",    value: "#2563eb", header: "#1e3a5f", skill_bg: "#dbeafe", skill_color: "#1e40af" },
  { label: "Teal",   value: "#0f766e", header: "#134e4a", skill_bg: "#ccfbf1", skill_color: "#0f766e" },
  { label: "Crimson",value: "#be123c", header: "#4c0519", skill_bg: "#ffe4e6", skill_color: "#be123c" },
  { label: "Violet", value: "#6d28d9", header: "#2e1065", skill_bg: "#ede9fe", skill_color: "#6d28d9" },
  { label: "Amber",  value: "#b45309", header: "#451a03", skill_bg: "#fef3c7", skill_color: "#b45309" },
  { label: "Slate",  value: "#374151", header: "#111827", skill_bg: "#f3f4f6", skill_color: "#374151" },
];
const FONT_OPTIONS = ["Sora", "Georgia", "Palatino Linotype", "Courier New"];

const A4W = 794;
const A4H = 1123;

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
      --page-gap:     20px;
      --tray-bg:      ${dark ? "#070911"               : "#d8dce4"};
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

    /* ── PAGE ── */
    .page-wide   { max-width: 1200px; margin: 0 auto; padding: 2rem 1.5rem 4rem; }

    /* ── BUTTONS ── */
    .btn-primary { background: ${dark?"#c8f064":"#3d8b00"}; color: ${dark?"#0d0f14":"#fff"}; border: none; padding: .75rem 2rem; border-radius: 8px; font-family: 'Sora',sans-serif; font-weight: 600; font-size: .9rem; cursor: pointer; transition: all .2s; }
    .btn-primary:hover { background: ${dark?"#d8ff6e":"#2f6b00"}; transform: translateY(-1px); }
    .btn-primary:disabled { opacity: .35; cursor: not-allowed; transform: none; }
    .btn-ghost { background: transparent; border: 1px solid var(--ghost-border); color: var(--ghost-color); padding: .75rem 1.5rem; border-radius: 8px; font-family: 'Sora',sans-serif; font-size: .88rem; cursor: pointer; transition: all .2s; }
    .btn-ghost:hover { border-color: var(--subtext); color: var(--text); }

    /* ── RESUME LAYOUT ── */
    .resume-layout { display: grid; grid-template-columns: 1fr 305px; gap: 1.5rem; align-items: start; }
    @media (max-width: 900px) { .resume-layout { grid-template-columns: 1fr; } }

    /* ── EDITOR PANEL ── */
    .editor-panel { background: var(--editor-bg); border: 1px solid var(--border); border-radius: 14px; padding: 1.4rem; position: sticky; top: 76px; max-height: calc(100vh - 96px); overflow-y: auto; }
    .editor-panel h3 { font-size: .7rem; font-weight: 600; text-transform: uppercase; letter-spacing: .1em; color: var(--lime); margin-bottom: 1.2rem; }
    .editor-section { margin-bottom: 1.2rem; }
    .editor-section > label { display: block; font-size: .68rem; color: var(--subtext); margin-bottom: .4rem; letter-spacing: .06em; text-transform: uppercase; }
    .color-row { display: flex; gap: .45rem; flex-wrap: wrap; margin-top: .3rem; }
    .color-swatch { width: 22px; height: 22px; border-radius: 50%; cursor: pointer; border: 2.5px solid transparent; transition: all .15s; }
    .color-swatch.active { border-color: var(--lime); transform: scale(1.18); box-shadow: 0 0 0 1px var(--lime); }
    .font-chips { display: flex; flex-wrap: wrap; gap: .38rem; margin-top: .3rem; }
    .font-chip { background: var(--input-bg); border: 1px solid var(--input-border); border-radius: 6px; padding: .35rem .7rem; font-size: .75rem; cursor: pointer; color: var(--subtext); transition: all .15s; }
    .font-chip.active { border-color: var(--lime); color: var(--lime); background: ${dark?"rgba(200,240,100,0.07)":"rgba(80,130,0,0.07)"}; }
    .editor-divider { border: none; border-top: 1px solid var(--divider); margin: 1rem 0; }
    .add-btn { width: 100%; background: var(--add-bg); border: 1px dashed var(--add-border); color: var(--add-color); border-radius: 7px; padding: .48rem .9rem; font-size: .77rem; cursor: pointer; transition: all .2s; font-family: 'Sora',sans-serif; margin-top: .38rem; text-align: left; }
    .add-btn:hover { background: ${dark?"rgba(200,240,100,0.12)":"rgba(80,130,0,0.1)"}; border-color: var(--lime); color: var(--lime); }
    .btn-download { width: 100%; background: ${dark?"#c8f064":"#3d8b00"}; color: ${dark?"#0d0f14":"#fff"}; border: none; padding: .78rem; border-radius: 8px; font-family: 'Sora',sans-serif; font-weight: 700; font-size: .88rem; cursor: pointer; transition: all .2s; display: flex; align-items: center; justify-content: center; gap: .5rem; }
    .btn-download:hover { background: ${dark?"#d8ff6e":"#2f6b00"}; transform: translateY(-1px); }
    .btn-copy { width: 100%; background: var(--input-bg); border: 1px solid var(--input-border); color: var(--subtext); border-radius: 8px; padding: .65rem; font-family: 'Sora',sans-serif; font-size: .82rem; cursor: pointer; transition: all .2s; margin-top: .55rem; }
    .btn-copy:hover { color: var(--text); border-color: var(--subtext); }
    .btn-copy.done { color: var(--lime) !important; border-color: var(--lime) !important; }

    /* ── PAGE COUNT BADGE ── */
    .page-badge { display: inline-flex; align-items: center; gap: .4rem; background: ${dark?"rgba(255,255,255,0.05)":"rgba(0,0,0,0.06)"}; border: 1px solid var(--border); border-radius: 20px; padding: .28rem .75rem; font-size: .74rem; color: var(--subtext); margin-bottom: .85rem; }
    .page-badge strong { color: var(--text); font-weight: 600; }

    /* ── RESUME TRAY ── */
    .resume-tray { background: var(--tray-bg); border-radius: 12px; padding: 24px; display: flex; flex-direction: column; gap: var(--page-gap); overflow: hidden; }
    .tray-page-label { font-size: .65rem; font-weight: 600; letter-spacing: .1em; text-transform: uppercase; color: ${dark?"rgba(255,255,255,0.2)":"rgba(0,0,0,0.25)"}; text-align: center; margin-bottom: calc(var(--page-gap) * -0.5 + 4px); }

    /* ── A4 PAGE CARD ── */
    .page-card { background: #fff; width: 100%; border-radius: 4px; overflow: hidden; box-shadow: 0 4px 24px rgba(0,0,0,0.35); position: relative; min-height: 80px; }
    .page-card-cont { padding: 1.8rem 2.2rem; }

    /* ── RESUME CONTENT ── */
    .r-header { padding: 1.8rem 2.2rem 1.4rem; transition: background .3s; }
    .r-name { font-family: 'DM Serif Display',serif; font-size: 1.9rem; color: #fff; margin-bottom: .2rem; }
    .r-title { font-size: .87rem; color: rgba(255,255,255,.82); font-weight: 400; letter-spacing: .03em; margin-bottom: .9rem; }
    .r-contact { display: flex; flex-wrap: wrap; gap: .3rem 1rem; }
    .r-contact span { font-size: .73rem; color: rgba(255,255,255,.75); display: flex; align-items: center; gap: .3rem; }
    .r-section { margin-bottom: 1.3rem; }
    .r-section-title { font-size: .63rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; margin-bottom: .65rem; padding-bottom: .32rem; border-bottom: 1.5px solid; transition: color .3s, border-color .3s; color: #1a1a2e; }
    .exp-item { margin-bottom: .95rem; position: relative; }
    .exp-item:hover .del-row, .project-item:hover .del-row, .edu-row:hover .del-row, .cert-item:hover .del-row, .lang-item:hover .del-row, .award-item:hover .del-row { opacity: 1; }
    .del-row { opacity: 0; position: absolute; top: 2px; right: -6px; width: 17px; height: 17px; border-radius: 50%; background: #e55; border: none; color: #fff; font-size: .58rem; cursor: pointer; display: flex; align-items: center; justify-content: center; transition: opacity .15s; z-index: 3; }
    .exp-header { display: flex; justify-content: space-between; align-items: baseline; }
    .exp-company { font-weight: 600; font-size: .87rem; color: #1a1a2e; }
    .exp-period { font-size: .71rem; color: #888; font-family: 'JetBrains Mono',monospace; }
    .exp-role { font-size: .76rem; color: #555; margin-bottom: .32rem; }
    .exp-bullets { list-style: none; }
    .exp-bullets li { font-size: .76rem; color: #444; line-height: 1.6; padding-left: .9rem; position: relative; margin-bottom: .17rem; }
    .exp-bullets li::before { content: '▸'; position: absolute; left: 0; font-size: .52rem; top: .19rem; }
    .project-item { margin-bottom: .62rem; position: relative; }
    .project-name { font-weight: 600; font-size: .81rem; color: #1a1a2e; }
    .project-desc { font-size: .75rem; color: #555; line-height: 1.55; }
    .edu-row { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: .48rem; position: relative; }
    .edu-school { font-weight: 600; font-size: .83rem; color: #1a1a2e; }
    .edu-degree { font-size: .76rem; color: #555; }
    .edu-year { font-size: .71rem; color: #888; font-family: 'JetBrains Mono',monospace; }
    .cert-item, .lang-item, .award-item { font-size: .77rem; color: #444; padding: .17rem 0; position: relative; }
    .skills-grid { display: flex; flex-wrap: wrap; gap: .38rem; }
    .skill-tag { font-size: .71rem; padding: .24rem .68rem; border-radius: 4px; font-weight: 500; position: relative; display: inline-flex; align-items: center; gap: .28rem; }
    .del-skill { display: none; width: 13px; height: 13px; border-radius: 50%; background: #e55; border: none; color: #fff; font-size: .5rem; cursor: pointer; align-items: center; justify-content: center; padding: 0; }
    .skill-tag:hover .del-skill { display: flex; }
    .r-summary { font-size: .79rem; color: #444; line-height: 1.72; }

    [contenteditable="true"] { outline: none; border-radius: 3px; transition: background .15s; min-width: 6px; display: inline-block; }
    [contenteditable="true"]:hover { background: rgba(100,180,255,.13); }
    [contenteditable="true"]:focus { background: rgba(100,180,255,.22); }

    .cont-strip { height: 6px; }

    /* ── DOWNLOAD MODAL ── */
    .modal-overlay { position: fixed; inset: 0; z-index: 300; background: rgba(0,0,0,.72); display: flex; flex-direction: column; align-items: center; justify-content: flex-start; overflow-y: auto; padding: 2rem 1rem; backdrop-filter: blur(4px); }
    .modal-box { background: ${dark?"#13161f":"#f8f9fc"}; border-radius: 16px; border: 1px solid var(--border); width: 100%; max-width: 860px; overflow: hidden; box-shadow: 0 30px 80px rgba(0,0,0,.5); }
    .modal-header { display: flex; align-items: center; justify-content: space-between; padding: 1.1rem 1.6rem; border-bottom: 1px solid var(--divider); }
    .modal-header h2 { font-family: 'DM Serif Display',serif; font-size: 1.2rem; color: var(--text); }
    .modal-close { background: none; border: none; cursor: pointer; color: var(--subtext); font-size: 1.3rem; line-height:1; transition: color .2s; }
    .modal-close:hover { color: var(--text); }
    .modal-preview { background: #d0d4dc; padding: 24px; display: flex; flex-direction: column; gap: 20px; align-items: center; max-height: 65vh; overflow-y: auto; }
    .modal-page { background: #fff; width: 100%; max-width: 700px; border-radius: 3px; box-shadow: 0 4px 20px rgba(0,0,0,.28); overflow: hidden; }
    .modal-actions { display: flex; gap: 1rem; padding: 1.1rem 1.6rem; justify-content: flex-end; border-top: 1px solid var(--divider); }
    .modal-btn-dl { background: ${dark?"#c8f064":"#3d8b00"}; color: ${dark?"#0d0f14":"#fff"}; border: none; padding: .7rem 1.8rem; border-radius: 8px; font-family: 'Sora',sans-serif; font-weight: 700; font-size: .88rem; cursor: pointer; transition: all .2s; }
    .modal-btn-dl:hover { background: ${dark?"#d8ff6e":"#2f6b00"}; }
    .modal-btn-cancel { background: transparent; border: 1px solid var(--ghost-border); color: var(--ghost-color); padding: .7rem 1.4rem; border-radius: 8px; font-family: 'Sora',sans-serif; font-size: .88rem; cursor: pointer; transition: all .2s; }
    .modal-btn-cancel:hover { color: var(--text); border-color: var(--subtext); }

    @media print {
      * { -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }
      body { background: #fff !important; }
      .nav, .editor-panel { display: none !important; }
      .resume-layout { display: block !important; }
      .del-row, .del-skill { display: none !important; }
      [contenteditable]:hover, [contenteditable]:focus { background: transparent !important; }
    }
  `}</style>
);

const STEPS = ["upload", "jobdesc", "resume"];

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

function ResumeContent({ accent, exp, projects, education, certifications, languages, awards, skills,
  delExp, delProject, delEdu, delSkill, delCert, delLang, delAward, editable = true }) {

  const CE = editable ? { contentEditable: true, suppressContentEditableWarning: true } : {};

  return <>
    <div className="r-section">
      <div className="r-section-title" style={{ color: accent.value, borderColor: accent.value }}>Summary</div>
      <div className="r-summary" {...CE}>{defaultResume.summary}</div>
    </div>

    <div className="r-section">
      <div className="r-section-title" style={{ color: accent.value, borderColor: accent.value }}>Experience</div>
      {exp.map(e => (
        <div className="exp-item" key={e.id}>
          {editable && <button className="del-row" onClick={() => delExp(e.id)}>✕</button>}
          <div className="exp-header">
            <span className="exp-company" {...CE}>{e.company}</span>
            <span className="exp-period" {...CE}>{e.period}</span>
          </div>
          <div className="exp-role" {...CE}>{e.role}</div>
          <ul className="exp-bullets">
            {e.bullets.map((b, j) => <li key={j}><span {...CE}>{b}</span></li>)}
          </ul>
        </div>
      ))}
    </div>

    {projects.length > 0 && (
      <div className="r-section">
        <div className="r-section-title" style={{ color: accent.value, borderColor: accent.value }}>Projects</div>
        {projects.map(p => (
          <div className="project-item" key={p.id}>
            {editable && <button className="del-row" onClick={() => delProject(p.id)}>✕</button>}
            <div className="project-name" {...CE}>{p.name}</div>
            <div className="project-desc" {...CE}>{p.description}</div>
          </div>
        ))}
      </div>
    )}

    {education.length > 0 && (
      <div className="r-section">
        <div className="r-section-title" style={{ color: accent.value, borderColor: accent.value }}>Education</div>
        {education.map(e => (
          <div className="edu-row" key={e.id}>
            {editable && <button className="del-row" onClick={() => delEdu(e.id)}>✕</button>}
            <div><div className="edu-school" {...CE}>{e.school}</div><div className="edu-degree" {...CE}>{e.degree}</div></div>
            <span className="edu-year" {...CE}>{e.year}</span>
          </div>
        ))}
      </div>
    )}

    {certifications.length > 0 && (
      <div className="r-section">
        <div className="r-section-title" style={{ color: accent.value, borderColor: accent.value }}>Certifications</div>
        {certifications.map(c => (
          <div className="cert-item" key={c.id}>
            {editable && <button className="del-row" onClick={() => delCert(c.id)}>✕</button>}
            <span {...CE}>{c.text}</span>
          </div>
        ))}
      </div>
    )}

    {languages.length > 0 && (
      <div className="r-section">
        <div className="r-section-title" style={{ color: accent.value, borderColor: accent.value }}>Languages</div>
        {languages.map(l => (
          <div className="lang-item" key={l.id}>
            {editable && <button className="del-row" onClick={() => delLang(l.id)}>✕</button>}
            <span {...CE}>{l.text}</span>
          </div>
        ))}
      </div>
    )}

    {awards.length > 0 && (
      <div className="r-section">
        <div className="r-section-title" style={{ color: accent.value, borderColor: accent.value }}>Awards & Honors</div>
        {awards.map(a => (
          <div className="award-item" key={a.id}>
            {editable && <button className="del-row" onClick={() => delAward(a.id)}>✕</button>}
            <span {...CE}>{a.text}</span>
          </div>
        ))}
      </div>
    )}

    <div className="r-section">
      <div className="r-section-title" style={{ color: accent.value, borderColor: accent.value }}>Skills</div>
      <div className="skills-grid">
        {skills.map((s, i) => (
          <span key={i} className="skill-tag" style={{ background: accent.skill_bg, color: accent.skill_color }}>
            <span {...CE}>{s}</span>
            {editable && <button className="del-skill" onClick={() => delSkill(i)}>✕</button>}
          </span>
        ))}
      </div>
    </div>
  </>;
}

function PaginatedResume({ accent, font, fontSize, exp, projects, education, certifications,
  languages, awards, skills, delExp, delProject, delEdu, delSkill, delCert, delLang, delAward, onPageCount }) {

  const containerRef = useRef(null);
  const fullRef = useRef(null);
  const [pages, setPages] = useState([{ id: 1 }]);

  const fzMap = { small: "0.74rem", medium: "0.80rem", large: "0.88rem" };

  useEffect(() => {
    if (!fullRef.current || !containerRef.current) return;
    const containerW = containerRef.current.offsetWidth - 48;
    const pageH = (containerW / A4W) * A4H;
    const totalH = fullRef.current.scrollHeight;
    const numPages = Math.max(1, Math.ceil(totalH / pageH));
    const arr = Array.from({ length: numPages }, (_, i) => ({
      id: i + 1,
      clipTop: i * pageH,
      clipH: pageH,
    }));
    setPages(arr);
    if (onPageCount) onPageCount(numPages);
  // eslint-disable-next-line
  }, [exp, projects, education, certifications, languages, awards, skills, font, fontSize, accent]);

  const containerW = containerRef.current ? containerRef.current.offsetWidth - 48 : 700;
  const pageH = (containerW / A4W) * A4H;

  const sharedContentProps = { accent, exp, projects, education, certifications, languages, awards, skills, delExp, delProject, delEdu, delSkill, delCert, delLang, delAward };

  return (
    <div className="resume-tray" ref={containerRef}>
      <div ref={fullRef} style={{ position:"absolute", visibility:"hidden", pointerEvents:"none", width: containerW || "100%", fontFamily: font, fontSize: fzMap[fontSize], left:0, top:0, zIndex:-1 }}>
        <div className="r-header" style={{ background: accent.header }}>
          <div className="r-name">{defaultResume.name}</div>
          <div className="r-title">{defaultResume.title}</div>
        </div>
        <div className="page-card-cont">
          <ResumeContent {...sharedContentProps} editable={false} />
        </div>
      </div>

      {pages.map((pg, idx) => (
        <div key={pg.id}>
          {idx > 0 && <div className="tray-page-label">— Page {pg.id} —</div>}
          <div className="page-card" style={{ fontFamily: font, fontSize: fzMap[fontSize] }}>
            {idx === 0 ? (
              <>
                <div className="r-header" style={{ background: accent.header }}>
                  <div className="r-name">
                    <span contentEditable suppressContentEditableWarning>{defaultResume.name}</span>
                  </div>
                  <div className="r-title">
                    <span contentEditable suppressContentEditableWarning>{defaultResume.title}</span>
                  </div>
                  <div className="r-contact">
                    {[{icon:"✉",val:defaultResume.email},{icon:"☎",val:defaultResume.phone},{icon:"⌖",val:defaultResume.location},{icon:"in",val:defaultResume.linkedin}].map((c,i)=>(
                      <span key={i}>{c.icon} <span contentEditable suppressContentEditableWarning>{c.val}</span></span>
                    ))}
                  </div>
                </div>
                <div className="page-card-cont" style={{ overflow:"hidden", maxHeight: pages.length > 1 ? `${pageH - 160}px` : undefined }}>
                  <ResumeContent {...sharedContentProps} />
                </div>
              </>
            ) : (
              <>
                <div className="cont-strip" style={{ background: accent.header }} />
                <div className="page-card-cont" style={{ overflow:"hidden", maxHeight: `${pageH - 30}px` }}>
                  <div style={{ marginTop: `calc(-${idx} * (${pageH}px - 160px) + ${idx > 1 ? (idx-1) * 30 : 0}px)` }}>
                    <ResumeContent {...sharedContentProps} />
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

function DownloadModal({ accent, font, fontSize, exp, projects, education, certifications, languages, awards, skills, onClose }) {
  const fzMap = { small:"0.74rem", medium:"0.80rem", large:"0.88rem" };
  const sharedContentProps = { accent, exp, projects, education, certifications, languages, awards, skills,
    delExp:()=>{}, delProject:()=>{}, delEdu:()=>{}, delSkill:()=>{}, delCert:()=>{}, delLang:()=>{}, delAward:()=>{} };

  const handlePrint = () => {
    const content = document.getElementById("print-frame-content").innerHTML;
    const win = window.open("", "_blank", "width=900,height=700");
    win.document.write(`<!DOCTYPE html><html><head>
      <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=DM+Serif+Display:ital@0;1&family=JetBrains+Mono:wght@400;500&display=swap"/>
      <style>
        *{box-sizing:border-box;margin:0;padding:0;} body{font-family:'Sora',sans-serif;background:#fff;color:#1a1a2e;}
        .r-header{padding:1.8rem 2.2rem 1.4rem;background:${accent.header};}
        .r-name{font-family:'DM Serif Display',serif;font-size:1.9rem;color:#fff;margin-bottom:.2rem;}
        .r-title{font-size:.87rem;color:rgba(255,255,255,.82);margin-bottom:.9rem;}
        .r-contact{display:flex;flex-wrap:wrap;gap:.3rem 1rem;}
        .r-contact span{font-size:.73rem;color:rgba(255,255,255,.75);display:flex;align-items:center;gap:.3rem;}
        .page-card-cont{padding:1.8rem 2.2rem;}
        .r-section{margin-bottom:1.3rem;}
        .r-section-title{font-size:.63rem;font-weight:700;letter-spacing:.13em;text-transform:uppercase;margin-bottom:.65rem;padding-bottom:.32rem;border-bottom:1.5px solid ${accent.value};color:${accent.value};}
        .exp-item{margin-bottom:.95rem;} .exp-header{display:flex;justify-content:space-between;}
        .exp-company{font-weight:600;font-size:.87rem;} .exp-period{font-size:.71rem;color:#888;font-family:'JetBrains Mono',monospace;}
        .exp-role{font-size:.76rem;color:#555;margin-bottom:.32rem;}
        .exp-bullets{list-style:none;} .exp-bullets li{font-size:.76rem;color:#444;line-height:1.6;padding-left:.9rem;position:relative;margin-bottom:.17rem;}
        .exp-bullets li::before{content:'▸';position:absolute;left:0;font-size:.52rem;top:.19rem;}
        .project-item{margin-bottom:.62rem;} .project-name{font-weight:600;font-size:.81rem;} .project-desc{font-size:.75rem;color:#555;line-height:1.55;}
        .edu-row{display:flex;justify-content:space-between;margin-bottom:.48rem;} .edu-school{font-weight:600;font-size:.83rem;} .edu-degree{font-size:.76rem;color:#555;} .edu-year{font-size:.71rem;color:#888;font-family:'JetBrains Mono',monospace;}
        .cert-item,.lang-item,.award-item{font-size:.77rem;color:#444;padding:.17rem 0;}
        .skills-grid{display:flex;flex-wrap:wrap;gap:.38rem;}
        .skill-tag{font-size:.71rem;padding:.24rem .68rem;border-radius:4px;font-weight:500;background:${accent.skill_bg};color:${accent.skill_color};}
        .r-summary{font-size:.79rem;color:#444;line-height:1.72;}
        @media print{*{-webkit-print-color-adjust:exact!important;print-color-adjust:exact!important;}}
      </style></head><body>${content}</body></html>`);
    win.document.close();
    setTimeout(() => { win.focus(); win.print(); }, 600);
  };

  return (
    <div className="modal-overlay" onClick={e => { if(e.target===e.currentTarget) onClose(); }}>
      <div className="modal-box">
        <div className="modal-header">
          <h2>Preview & Download</h2>
          <button className="modal-close" onClick={onClose}>✕</button>
        </div>
        <div className="modal-preview">
          <div className="modal-page" id="print-frame-content" style={{ fontFamily: font, fontSize: fzMap[fontSize] }}>
            <div className="r-header" style={{ background: accent.header }}>
              <div className="r-name">{defaultResume.name}</div>
              <div className="r-title">{defaultResume.title}</div>
              <div className="r-contact">
                {[{icon:"✉",val:defaultResume.email},{icon:"☎",val:defaultResume.phone},{icon:"⌖",val:defaultResume.location},{icon:"in",val:defaultResume.linkedin}].map((c,i)=>(
                  <span key={i}>{c.icon} {c.val}</span>
                ))}
              </div>
            </div>
            <div className="page-card-cont">
              <ResumeContent {...sharedContentProps} editable={false} />
            </div>
          </div>
        </div>
        <div className="modal-actions">
          <button className="modal-btn-cancel" onClick={onClose}>Cancel</button>
          <button className="modal-btn-dl" onClick={handlePrint}>⬇ Print / Save as PDF</button>
        </div>
      </div>
    </div>
  );
}

export default function ResumePage({ onBack, dark }) {
  const [accent, setAccent] = useState(ACCENT_COLORS[0]);
  const [font, setFont] = useState("Sora");
  const [fontSize, setFontSize] = useState("medium");
  const [copyDone, setCopyDone] = useState(false);
  const [pageCount, setPageCount] = useState(1);
  const [showDownload, setShowDownload] = useState(false);

  const [exp, setExp] = useState(defaultResume.experience);
  const [projects, setProjects] = useState(defaultResume.projects);
  const [education, setEducation] = useState(defaultResume.education);
  const [skills, setSkills] = useState(defaultResume.skills);
  const [certifications, setCertifications] = useState([]);
  const [languages, setLanguages] = useState([]);
  const [awards, setAwards] = useState([]);
  const nextId = useRef(100);
  const getId = () => ++nextId.current;

  const addExp = () => setExp(p => [...p, { id: getId(), company: "New Company", role: "Your Role", period: "20XX – Present", bullets: ["Key achievement here.", "Another accomplishment."] }]);
  const addProject = () => setProjects(p => [...p, { id: getId(), name: "Project Name", description: "Brief description, tech used, and impact." }]);
  const addEdu = () => setEducation(p => [...p, { id: getId(), school: "University Name", degree: "B.S. Your Major", year: "20XX" }]);
  const addSkill = () => { const s = window.prompt("Skill name:"); if (s?.trim()) setSkills(p => [...p, s.trim()]); };
  const addCert = () => setCertifications(p => [...p, { id: getId(), text: "Certification – Issuer, Year" }]);
  const addLang = () => setLanguages(p => [...p, { id: getId(), text: "Language – Proficiency" }]);
  const addAward = () => setAwards(p => [...p, { id: getId(), text: "Award – Organization, Year" }]);

  const delExp = id => setExp(p => p.filter(e => e.id !== id));
  const delProject = id => setProjects(p => p.filter(e => e.id !== id));
  const delEdu = id => setEducation(p => p.filter(e => e.id !== id));
  const delSkill = i => setSkills(p => p.filter((_, j) => j !== i));
  const delCert = id => setCertifications(p => p.filter(e => e.id !== id));
  const delLang = id => setLanguages(p => p.filter(e => e.id !== id));
  const delAward = id => setAwards(p => p.filter(e => e.id !== id));

  const copyPlain = useCallback(() => {
    const lines = [];
    lines.push(defaultResume.name.toUpperCase());
    lines.push(defaultResume.title);
    lines.push(`${defaultResume.email} | ${defaultResume.phone} | ${defaultResume.location} | ${defaultResume.linkedin}`);
    lines.push("");
    lines.push("SUMMARY");
    lines.push(defaultResume.summary);
    lines.push("");
    lines.push("EXPERIENCE");
    exp.forEach(e => {
      lines.push(`${e.company} — ${e.role} (${e.period})`);
      e.bullets.forEach(b => lines.push(`  • ${b}`));
    });
    if (projects.length) {
      lines.push(""); lines.push("PROJECTS");
      projects.forEach(p => lines.push(`${p.name}: ${p.description}`));
    }
    if (education.length) {
      lines.push(""); lines.push("EDUCATION");
      education.forEach(e => lines.push(`${e.school} — ${e.degree} (${e.year})`));
    }
    if (certifications.length) {
      lines.push(""); lines.push("CERTIFICATIONS");
      certifications.forEach(c => lines.push(`  • ${c.text}`));
    }
    if (languages.length) {
      lines.push(""); lines.push("LANGUAGES");
      languages.forEach(l => lines.push(`  • ${l.text}`));
    }
    if (awards.length) {
      lines.push(""); lines.push("AWARDS & HONORS");
      awards.forEach(a => lines.push(`  • ${a.text}`));
    }
    lines.push(""); lines.push("SKILLS");
    lines.push(skills.join(" · "));

    navigator.clipboard.writeText(lines.join("\n")).then(() => {
      setCopyDone(true); setTimeout(() => setCopyDone(false), 2200);
    });
  }, [exp, projects, education, certifications, languages, awards, skills]);

  const sharedProps = { accent, font, fontSize, exp, projects, education, certifications, languages, awards, skills, delExp, delProject, delEdu, delSkill, delCert, delLang, delAward };
  const headingColor = dark ? "#f0ebe3" : "#111827";

  return (
    <>
      <GlobalStyle dark={dark} />
      <div className="app-shell">
        <nav className="nav">
          <div className="nav-logo">resume<span>.</span>ai</div>
          <div className="nav-right">
            <button className="theme-btn">☀️</button>
          </div>
        </nav>

        <Stepper step="resume" />

        <div className="page-wide">
          {showDownload && <DownloadModal {...sharedProps} onClose={() => setShowDownload(false)} />}

          <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:"1rem" }}>
            <div>
              <h2 style={{ fontFamily:"'DM Serif Display',serif", fontSize:"1.6rem", color: headingColor }}>Your Resume is Ready ✦</h2>
              <p style={{ fontSize:".77rem", color:"#6e7280", marginTop:".25rem" }}>Click any text to edit · Hover items to remove · Add sections from the panel</p>
            </div>
            <button className="btn-ghost" onClick={onBack} style={{ fontSize:".8rem" }}>← Back</button>
          </div>

          <div className="page-badge">
            <span>📄</span>
            <strong>{pageCount}</strong>
            <span>{pageCount===1 ? "page — fits perfectly ✓" : pageCount===2 ? "pages — consider trimming for 1 page" : "pages — recommend condensing"}</span>
          </div>

          <div className="resume-layout">
            <PaginatedResume {...sharedProps} onPageCount={setPageCount} />

            <div className="editor-panel">
              <h3>✦ Customise</h3>

              <div className="editor-section">
                <label>Accent Color</label>
                <div className="color-row">
                  {ACCENT_COLORS.map(c => (
                    <div key={c.label} className={`color-swatch${accent.value===c.value?" active":""}`} style={{ background: c.header }} title={c.label} onClick={() => setAccent(c)} />
                  ))}
                </div>
              </div>

              <div className="editor-section">
                <label>Font</label>
                <div className="font-chips">
                  {FONT_OPTIONS.map(f => (
                    <div key={f} className={`font-chip${font===f?" active":""}`} style={{ fontFamily: f }} onClick={() => setFont(f)}>{f}</div>
                  ))}
                </div>
              </div>

              <div className="editor-section">
                <label>Text Size</label>
                <div className="font-chips">
                  {["small","medium","large"].map(s => (
                    <div key={s} className={`font-chip${fontSize===s?" active":""}`} onClick={() => setFontSize(s)} style={{ textTransform:"capitalize" }}>{s}</div>
                  ))}
                </div>
              </div>

              <hr className="editor-divider" />

              <div className="editor-section">
                <label>Add to Resume</label>
                <button className="add-btn" onClick={addExp}>+ Work Experience</button>
                <button className="add-btn" onClick={addProject}>+ Project</button>
                <button className="add-btn" onClick={addEdu}>+ Education</button>
                <button className="add-btn" onClick={addSkill}>+ Skill</button>
                <button className="add-btn" onClick={addCert}>+ Certification</button>
                <button className="add-btn" onClick={addLang}>+ Language</button>
                <button className="add-btn" onClick={addAward}>+ Award / Honor</button>
              </div>

              <hr className="editor-divider" />

              <button className="btn-download" onClick={() => setShowDownload(true)}>⬇ Download PDF</button>
              <button className={`btn-copy${copyDone?" done":""}`} onClick={copyPlain}>
                {copyDone ? "✓ Copied!" : "Copy as Plain Text"}
              </button>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
