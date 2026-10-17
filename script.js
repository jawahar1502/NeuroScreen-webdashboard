"use strict";

/* NeuroScreen dashboard: plain JavaScript, no libraries.
   Talks to the Python server in app.py through /api/metrics, /api/demo/<kind> and /api/predict. */

const $ = (id) => document.getElementById(id);
const C = {
  ink: "#10242f", muted: "#5b6f7a", line: "#d9e3e8", navy: "#0a2a3d", teal: "#0e7c86",
  ok: "#1d8a5a", okSoft: "#e3f4ec", alert: "#c0392b", alertSoft: "#fbe9e6", amber: "#e6b84a",
  grid: "#0f3140", trace: "#5eead4",
};
const pct = (v, d = 1) => (v * 100).toFixed(d) + "%";
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const WAIT = '<div class="empty">Results appear here after an analysis.</div>';
const PULSE = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 12h4l3-8 4 16 3-8h4"/></svg>';
const EMPTY_RESULT = `<div class="empty">${PULSE}No recording analysed yet. Upload a file or load a sample to see the result.</div>`;

let metrics = null;
let requestId = 0; // ignore answers that arrive after a newer request was started

/* ---------- small helpers ---------- */
function setError(message) { $("err").textContent = message || ""; }

function setBusy(busy) {
  $("demoHealthy").disabled = busy;
  $("demoMdd").disabled = busy;
  if (busy) $("pred").innerHTML = '<div class="empty">Analysing recording...</div>';
}

async function readError(res, fallback) {
  try { return (await res.json()).detail || fallback; } catch { return fallback; }
}

/* ---------- model information (loaded once) ---------- */
async function loadMetrics() {
  try {
    const res = await fetch("/api/metrics");
    if (!res.ok) throw new Error(await readError(res, "Could not load model information."));
    metrics = await res.json();
  } catch (e) {
    setError(e.message === "Failed to fetch" ? "Cannot reach the analysis server. Check that app.py is running." : e.message);
    $("metrics").innerHTML = WAIT.replace("Results appear here after an analysis.", "Model information is not available.");
    $("feats").innerHTML = $("metrics").innerHTML;
    return;
  }
  const demo = Boolean(metrics.demo);
  $("demoBadge").hidden = !demo;
  $("demoBanner").hidden = !demo;
  $("demoRow").hidden = !demo;
  if (metrics.data_note) $("demoNote").textContent = metrics.data_note;
  renderMetrics(metrics);
  renderFeatures(metrics);
}

function renderMetrics(m) {
  const c = m.confusion;
  $("metrics").innerHTML = `
    <div class="metric-grid">
      <div class="metric"><span>Accuracy, average per fold</span><b class="num">${pct(m.cv_accuracy_mean)}</b><br><small>± ${pct(m.cv_accuracy_std)} between folds</small></div>
      <div class="metric"><span>Accuracy, all subjects</span><b class="num">${pct(m.accuracy)}</b><br><small>${m.n_subjects} subjects</small></div>
      <div class="metric"><span>AUC</span><b class="num">${m.auc.toFixed(2)}</b></div>
      <div class="metric"><span>F1 score</span><b class="num">${m.f1.toFixed(2)}</b></div>
    </div>
    <div class="cm" aria-label="Confusion matrix across all subjects">
      <div class="h"></div><div class="h">Predicted healthy</div><div class="h">Predicted MDD</div>
      <div class="h">Actually healthy</div><div class="good num">${c.tn}</div><div class="bad num">${c.fp}</div>
      <div class="h">Actually MDD</div><div class="bad num">${c.fn}</div><div class="good num">${c.tp}</div>
    </div>
    <p class="hint spaced">${m.cv_folds}-fold cross-validation, repeated ${m.cv_repeats} times. Splits are by subject, so each of the
    ${m.n_subjects} people is always scored by a model that never saw them. Each person's score is the average over the repeats.</p>`;
}

function renderFeatures(m) {
  const mx = Math.max(...m.top_features.map((f) => f.importance));
  $("feats").innerHTML = m.top_features.map((f) => `
    <div class="feat"><span>${esc(f.name.replace(/_/g, " "))}</span>
    <div class="bar" style="width:${(f.importance / mx) * 100}%"></div>
    <span class="num">${f.importance.toFixed(3)}</span></div>`).join("");
}

/* ---------- analysis ---------- */
async function analyse(file) {
  const mine = ++requestId;
  setError("");
  setBusy(true);
  try {
    const body = new FormData();
    body.append("file", file);
    let res;
    try {
      res = await fetch("/api/predict", { method: "POST", body });
    } catch {
      throw new Error("Cannot reach the analysis server. Check that app.py is running.");
    }
    if (!res.ok) throw new Error(await readError(res, "The server could not analyse this file."));
    const result = await res.json();
    if (mine !== requestId) return;
    renderAll(result);
    const calm = matchMedia("(prefers-reduced-motion: reduce)").matches;
    $("findings").scrollIntoView({ behavior: calm ? "auto" : "smooth", block: "start" });
  } catch (e) {
    if (mine !== requestId) return;
    setError(e.message);
    $("pred").innerHTML = EMPTY_RESULT;
  } finally {
    if (mine === requestId) setBusy(false);
  }
}

async function loadDemo(kind) {
  setError("");
  try {
    const res = await fetch(`/api/demo/${kind}`);
    if (!res.ok) throw new Error(await readError(res, "Demo recording is not available."));
    analyse(new File([await res.blob()], `demo_${kind}.npy`));
  } catch (e) {
    setError(e.message === "Failed to fetch" ? "Cannot reach the analysis server. Check that app.py is running." : e.message);
  }
}

function renderAll(r) {
  renderResult(r);
  renderTable(r);
  renderBands(r);
  renderEpochs(r);
  renderTrace(r);
  renderHeat(r);
}

/* ---------- result card ---------- */
function renderResult(r) {
  const p = r.probability_mdd, mdd = p >= 0.5;
  const a = Math.PI * (1 - p), nx = 100 + 62 * Math.cos(a), ny = 100 - 62 * Math.sin(a);
  $("pred").innerHTML = `
    <div class="result">
      <span class="status ${mdd ? "mdd" : "hc"}"><i></i>${esc(r.label)}</span>
      <svg class="gauge" viewBox="0 0 200 120" role="img" aria-label="MDD probability ${pct(p)}">
        <defs><linearGradient id="gaugeGrad" x1="0" x2="1" y1="0" y2="0">
          <stop offset="0" stop-color="${C.ok}"/><stop offset=".5" stop-color="${C.amber}"/><stop offset="1" stop-color="${C.alert}"/>
        </linearGradient></defs>
        <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="url(#gaugeGrad)" stroke-width="14" stroke-linecap="round"/>
        <line x1="100" y1="100" x2="${nx}" y2="${ny}" stroke="${C.navy}" stroke-width="3" stroke-linecap="round"/>
        <circle cx="100" cy="100" r="6" fill="${C.navy}" stroke="none"/>
        <text x="20" y="117" text-anchor="middle">0%</text><text x="180" y="117" text-anchor="middle">100%</text>
      </svg>
      <div class="big num">${pct(p)}</div>
      <p class="hint centered">estimated probability of an MDD-like pattern</p>
      <div class="facts">
        <div><span>Confidence</span><b class="num">${pct(r.confidence)}</b></div>
        <div><span>Windows agreeing</span><b class="num">${pct(r.epoch_agreement)}</b></div>
        <div><span>Recording length</span><b class="num">${r.duration_sec} s</b></div>
      </div>
      ${r.demo ? '<p class="demo-note">Demo result from synthetic data. Not a real patient finding.</p>' : ""}
    </div>`;
}

/* ---------- band power table ---------- */
function position(value, healthy, mdd) {
  const dh = Math.abs(value - healthy), dm = Math.abs(value - mdd);
  if (Math.abs(dh - dm) < 0.1 * Math.abs(healthy - mdd)) return ["eq", "Between groups"];
  return dm < dh ? ["mdd", "Closer to MDD"] : ["hc", "Closer to healthy"];
}

function renderTable(r) {
  const ref = metrics && metrics.reference;
  if (!ref) { $("labtable").innerHTML = WAIT; return; }
  const rows = r.bands.map((band, i) => {
    const [cls, text] = position(r.band_mean[i], ref.healthy[i], ref.mdd[i]);
    return `<tr><td><b>${esc(cap(band))}</b></td><td class="r num"><b>${pct(r.band_mean[i])}</b></td>
      <td class="r num">${pct(ref.healthy[i])}</td><td class="r num">${pct(ref.mdd[i])}</td>
      <td><span class="flag ${cls}">${text}</span></td></tr>`;
  }).join("");
  $("labtable").innerHTML = `<div class="table-wrap"><table><thead><tr><th>Band</th><th class="r">This recording</th>
    <th class="r">Healthy group</th><th class="r">MDD group</th><th>Position</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

/* ---------- charts (inline SVG) ---------- */
function renderBands(r) {
  const ref = metrics && metrics.reference;
  if (!ref) { $("bands").innerHTML = WAIT; return; }
  const W = 520, H = 250, L = 40, B = 28, T = 10, n = r.bands.length;
  const series = [["Healthy group", ref.healthy, C.ok], ["MDD group", ref.mdd, C.alert], ["This recording", r.band_mean, C.navy]];
  const mx = Math.max(...series.flatMap((s) => s[1])) * 1.1;
  const gw = (W - L) / n, bw = gw / 4;
  let svg = "";
  for (let i = 0; i <= 4; i++) {
    const y = T + (H - T - B) * (1 - i / 4);
    svg += `<line x1="${L}" x2="${W}" y1="${y}" y2="${y}" stroke="${C.line}"/><text x="${L - 6}" y="${y + 4}" text-anchor="end">${((mx * i) / 4 * 100).toFixed(0)}%</text>`;
  }
  r.bands.forEach((band, i) => {
    series.forEach(([name, vals, color], k) => {
      const h = ((H - T - B) * vals[i]) / mx;
      svg += `<rect x="${L + i * gw + gw * 0.12 + k * bw}" y="${H - B - h}" width="${bw - 2}" height="${h}" rx="2" fill="${color}" stroke="none"><title>${esc(name)}, ${esc(band)}: ${pct(vals[i])}</title></rect>`;
    });
    svg += `<text x="${L + i * gw + gw / 2}" y="${H - 10}" text-anchor="middle">${esc(band)}</text>`;
  });
  $("bands").innerHTML = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Relative band power compared with reference groups">${svg}</svg>
    <div class="legend">${series.map((s) => `<span><b style="background:${s[2]}"></b>${s[0]}</span>`).join("")}</div>`;
}

function renderEpochs(r) {
  const W = 520, H = 250, L = 40, B = 28, T = 10, p = r.epoch_probs, n = p.length;
  const X = (i) => L + ((W - L - 6) * i) / Math.max(n - 1, 1);
  const Y = (v) => T + (H - T - B) * (1 - v);
  let svg = `<rect x="${L}" y="${Y(1)}" width="${W - L}" height="${Y(0.5) - Y(1)}" fill="${C.alertSoft}" stroke="none"/>
    <rect x="${L}" y="${Y(0.5)}" width="${W - L}" height="${Y(0) - Y(0.5)}" fill="${C.okSoft}" stroke="none"/>`;
  [0, 0.5, 1].forEach((v) => {
    svg += `<line x1="${L}" x2="${W}" y1="${Y(v)}" y2="${Y(v)}" stroke="${C.line}"${v === 0.5 ? ' stroke-dasharray="4 3"' : ""}/><text x="${L - 6}" y="${Y(v) + 4}" text-anchor="end">${v.toFixed(1)}</text>`;
  });
  svg += `<polyline points="${p.map((v, i) => `${X(i)},${Y(v)}`).join(" ")}" fill="none" stroke="${C.navy}" stroke-width="2"/>
    <text x="${W / 2}" y="${H - 6}" text-anchor="middle">Window number (4 s each, 2 s step)</text>`;
  $("epochs").innerHTML = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="MDD probability for each window">${svg}</svg>
    <div class="legend"><span><b style="background:${C.alertSoft};border:1px solid ${C.alert}"></b>MDD-like zone</span>
    <span><b style="background:${C.okSoft};border:1px solid ${C.ok}"></b>Healthy-like zone</span></div>`;
}

function renderTrace(r) {
  const W = 640, H = 250, L = 40, d = r.trace.data, n = d[0].length, gap = H / d.length;
  const mx = Math.max(...d.flat().map(Math.abs)) || 1;
  let svg = "";
  for (let i = 0; i <= 25; i++) { const x = L + ((W - L) * i) / 25; svg += `<line x1="${x}" x2="${x}" y1="0" y2="${H}" stroke="${C.grid}" stroke-width=".6"/>`; }
  for (let i = 0; i <= 12; i++) { const y = (H * i) / 12; svg += `<line x1="${L}" x2="${W}" y1="${y}" y2="${y}" stroke="${C.grid}" stroke-width=".6"/>`; }
  d.forEach((row, k) => {
    const mid = gap * k + gap / 2;
    const pts = row.map((v, i) => `${L + ((W - L - 4) * i) / (n - 1)},${mid - (v / mx) * gap * 0.42}`).join(" ");
    svg += `<text x="${L - 8}" y="${mid + 4}" text-anchor="end">${esc(r.trace.channels[k])}</text>
      <polyline points="${pts}" fill="none" stroke="${C.trace}" stroke-width="1.2"/>`;
  });
  $("trace").innerHTML = `<div class="monitor"><svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Filtered EEG trace, four channels">${svg}</svg>
    <div class="legend"><span>Grid lines mark 0.2 s. Amplitude is scaled to fit.</span></div></div>`;
}

function renderHeat(r) {
  const nb = r.bands.length, nc = r.channels.length, cw = 17, ch = 30, L = 50, T = 40;
  let svg = "";
  r.bands.forEach((band, i) => {
    const row = r.band_by_channel.map((c) => c[i]);
    const lo = Math.min(...row), hi = Math.max(...row);
    svg += `<text x="${L - 6}" y="${T + i * ch + ch / 2 + 4}" text-anchor="end">${esc(band)}</text>`;
    row.forEach((v, j) => {
      const t = hi === lo ? 0.5 : (v - lo) / (hi - lo);
      svg += `<rect x="${L + j * cw}" y="${T + i * ch}" width="${cw - 1.5}" height="${ch - 2}" rx="2" fill="${C.teal}" fill-opacity="${(0.1 + 0.9 * t).toFixed(2)}" stroke="none"><title>${esc(r.channels[j])}, ${esc(band)}: ${pct(v)}</title></rect>`;
    });
  });
  r.channels.forEach((c, j) => {
    svg += `<text transform="translate(${L + j * cw + cw / 2 + 3},${T - 6}) rotate(-60)" style="font-size:9px">${esc(c)}</text>`;
  });
  $("heat").innerHTML = `<svg class="chart" viewBox="0 0 ${L + nc * cw + 4} ${T + nb * ch + 4}" role="img" aria-label="Relative power by channel and band">${svg}</svg>
    <div class="legend"><span>Darker means higher power within that band. Hover a cell for its value.</span></div>`;
}

/* ---------- printing: fill the report header with the patient details ---------- */
function fillPrintHeader() {
  const head = $("printHead");
  head.replaceChildren();
  const title = document.createElement("h1");
  title.textContent = "NeuroScreen EEG screening report";
  const line = document.createElement("p");
  const parts = [
    $("pname").value && `Patient: ${$("pname").value}`,
    $("pid").value && `ID: ${$("pid").value}`,
    $("page").value && `Age: ${$("page").value}`,
    $("psex").value && `Sex: ${$("psex").value}`,
    `Study date: ${$("sdate").textContent}`,
  ].filter(Boolean);
  line.textContent = parts.join("   |   ");
  head.append(title, line);
  if (metrics && metrics.demo) {
    const flag = document.createElement("p");
    flag.className = "demo-flag";
    flag.textContent = "DEMO: generated from synthetic data. Not a real patient finding.";
    head.append(flag);
  }
}

/* ---------- wire everything up ---------- */
function init() {
  $("sdate").textContent = new Date().toLocaleDateString(undefined, { day: "2-digit", month: "short", year: "numeric" });
  $("pred").innerHTML = EMPTY_RESULT;
  ["labtable", "bands", "epochs", "trace", "heat"].forEach((id) => { $(id).innerHTML = WAIT; });
  $("metrics").innerHTML = '<div class="empty">Loading...</div>';
  $("feats").innerHTML = '<div class="empty">Loading...</div>';

  $("file").addEventListener("change", (e) => { if (e.target.files[0]) analyse(e.target.files[0]); e.target.value = ""; });
  const drop = $("drop");
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => { if (e.dataTransfer.files[0]) analyse(e.dataTransfer.files[0]); });

  $("demoHealthy").addEventListener("click", () => loadDemo("healthy"));
  $("demoMdd").addEventListener("click", () => loadDemo("mdd"));
  $("printBtn").addEventListener("click", () => window.print());
  window.addEventListener("beforeprint", fillPrintHeader);

  document.querySelectorAll(".sidebar nav a").forEach((a) => a.addEventListener("click", () => {
    document.querySelectorAll(".sidebar nav a").forEach((x) => x.classList.remove("on"));
    a.classList.add("on");
  }));
  loadMetrics();
}

init();
