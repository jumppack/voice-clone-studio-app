"use strict";
// Voice Clone Studio frontend: vanilla JS, no build step, no network beyond 127.0.0.1.

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const api = async (path, opts = {}) => {
  const r = await fetch(path, opts);
  const ct = r.headers.get("content-type") || "";
  const body = ct.includes("json") ? await r.json() : await r.text();
  if (!r.ok) throw new Error((body && body.detail) || r.statusText);
  return body;
};
const form = (obj) => { const f = new FormData(); for (const [k, v] of Object.entries(obj)) if (v !== undefined) f.append(k, v); return f; };
const icon = (id, cls = "ic") => `<svg class="${cls}" aria-hidden="true"><use href="#i-${id}"/></svg>`;
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const toast = (msg) => { const t = $("#toast"); t.textContent = msg; t.classList.add("show"); clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.remove("show"), 2600); };
const fmtTime = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

const S = { screen: "setup", steps: [], step: 1, profile: null, setup: null, presets: [], activePreset: "close", lastOut: null, lastExport: null };

// ---------------- theme ----------------
function applyTheme(t) {
  if (t) document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme;
  const dark = t ? t === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  const b = $("#themeToggle");
  b.innerHTML = icon(dark ? "sun" : "moon");
  b.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
  drawWave();
}
$("#themeToggle").onclick = () => {
  const cur = document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const next = cur === "dark" ? "light" : "dark";
  try { localStorage.setItem("vcs-theme", next); } catch {}
  applyTheme(next);
};

// ---------------- navigation ----------------
function show(screen) {
  S.screen = screen;
  for (const s of $$(".screen")) s.hidden = s.id !== `screen-${screen}`;
  $$(".tab").forEach((t) => { if (t.dataset.screen === screen) t.setAttribute("aria-current", "page"); else t.removeAttribute("aria-current"); });
  if (screen === "record") enterRecord();
  if (screen === "myvoice") enterMyVoice();
  if (screen === "setup" && S.setup) setupGate();
  if (screen === "profile") renderProfile();
  if (screen === "studio") enterStudio();
  if (screen === "sts" && window.STS) STS.enter();
  $("#main").focus({ preventScroll: true });
  window.scrollTo(0, 0);
}
const voiceSource = () => (S.setup && S.setup.settings.voice_source) || null;
function updateTabs() {  // sidebar: Text to speech, Speech to speech, My voice (record / choose takes), Setup
  const st = S.setup && S.setup.settings;
  const setupOk = !!(st && st.consent && st.watermark_ack);
  const up = voiceSource() === "upload";  // My voice from a recording you already have: no Record/Choose steps
  const rec = $('.tab[data-screen="record"]'), prof = $('.tab[data-screen="profile"]');
  if (rec) { rec.hidden = up || !voiceSource(); rec.disabled = !setupOk || voiceSource() !== "record"; }
  if (prof) { prof.hidden = up || !voiceSource(); prof.disabled = !setupOk || !(S.profile && S.profile.takes.length); }
  $('.tab[data-screen="studio"]').disabled = !setupOk;
  $('.tab[data-screen="sts"]').disabled = !setupOk;
  if (window.VOICES) VOICES.renderNav();
}
$$(".tab").forEach((t) => (t.onclick = () => show(t.dataset.screen)));

// ---------------- boot overlay ----------------
// Shown while the speech engine loads into memory (it can't speak or convert until then).
// ?boot=demo keeps it on screen (used to record docs/boot_overlay.mp4).
const DEMO_BOOT = new URLSearchParams(location.search).get("boot") === "demo";
function bootOverlay(on) {
  const el = $("#boot"); if (!el) return;
  if (on && !S.bootT0) S.bootT0 = Date.now();
  if (on && !DEMO_BOOT) {
    const s = Math.round((Date.now() - S.bootT0) / 1000);
    $("#bootSub").textContent = s < 45 ? "Loading the speech models into memory. This takes about 15 to 40 seconds after the app opens."
      : `Still loading (${s} s). A busy Mac or the first start after an update can take a little longer.`;
  }
  if (!on && !el.hidden) toast("Voice engine ready");
  el.hidden = !on;
}

// ---------------- setup ----------------
async function loadSetup() {
  S.setup = await api("/api/setup");
  const items = [...S.setup.items];
  const eng = S.setup.engine;
  items.push({ id: "engine", name: "Speech engine loaded", ok: eng.state === "ready",
    wait: eng.state === "loading", detail: eng.state === "ready" ? `Running on ${eng.device === "mps" ? "the Apple GPU (MPS)" : eng.device}, loaded in ${eng.load_seconds} s`
      : eng.state === "loading" ? "Loading the model into memory (about 15-40 s)…" : eng.state === "error" ? eng.error : "Starts after the weights are present" });
  if (voiceSource() !== "upload") items.push({ id: "mic", name: "Microphone permission", ok: S.micGranted === true, wait: S.micGranted === undefined,
    detail: S.micGranted ? "Allowed" : S.micGranted === false ? "Blocked. Allow it in the browser's site settings." : "Not asked yet", mic: true });
  $("#checklist").innerHTML = items.map((it) => {
    const st = it.ok ? "ok" : it.wait || it.optional ? "wait" : "no";
    const word = it.ok ? "Ready" : it.wait ? (it.id === "engine" ? "Loading" : "Not checked") : it.optional ? "Optional" : "Missing";
    const inst = S.setup.installs[it.id];
    let action = "";
    if (it.mic && !it.ok) action = `<button class="clay-btn" data-mic>Allow microphone</button>`;
    else if (!it.ok && it.command) action = `<button class="clay-btn" data-install="${it.id}">${inst && inst.state === "running" ? "Installing…" : "Install"}</button>`;
    return `<li class="check-item"><span class="status-dot ${st}">${icon(it.ok ? "check" : it.wait ? "warn" : "x")}</span>
      <div><div class="name">${esc(it.name)} <span class="state-word">· ${word}</span></div>
      <div class="detail">${esc(it.detail || "")}${it.size ? ` · ${esc(it.size)}` : ""}</div>
      ${inst && inst.state !== "running" ? `<div class="detail">Install ${esc(inst.state)}</div>` : ""}</div>${action}</li>`;
  }).join("");
  $$("[data-install]").forEach((b) => (b.onclick = () => askInstall(b.dataset.install)));
  $$("[data-mic]").forEach((b) => (b.onclick = async () => { await openMic(); loadSetup(); }));
  $("#consent").checked = S.setup.settings.consent;
  $("#wmAck").checked = S.setup.settings.watermark_ack;
  const src = voiceSource();
  $$('input[name="voiceSource"]').forEach((r) => (r.checked = r.value === src));
  setupGate();
  const weightsOk = S.setup.items.some((i) => i.id === "weights" && i.ok);
  const warming = weightsOk && (eng.state === "loading" || eng.state === "not_loaded");
  bootOverlay(warming || DEMO_BOOT);
  if (warming || Object.values(S.setup.installs).some((i) => i.state === "running")) setTimeout(loadSetup, warming ? 1000 : 2000);
}
function setupGate() {
  const ok = $("#consent").checked && $("#wmAck").checked;
  const need = S.setup.items.filter((i) => !i.ok && i.id !== "ffmpeg" && !i.optional);
  // Setup and permissions: only the checklist and the two boxes
  $("#setupDone").disabled = !ok || need.length > 0;
  $("#setupDoneHint").textContent = need.length ? `Install: ${need.map((n) => n.name).join(", ")}.` : !ok ? "Tick both boxes to continue." : "";
  $("#myvoiceGate").hidden = ok;
  const src = voiceSource();
  const file = $("#voiceFile").files[0];
  const haveUpload = S.profile && S.profile.source === "upload" && S.profile.ready;
  const btn = $("#toRecord");
  $("#uploadArea").hidden = src !== "upload";
  if (src === "upload" && !file && haveUpload) $("#voiceFileName").textContent = `Using ${S.profile.upload_file}. Choose another file to replace it.`;
  btn.textContent = src === "record" ? "Continue to recording" : src === "upload" ? (file ? "Use this recording" : haveUpload ? "Go to text to speech" : "Choose a file first") : "Continue";
  const ffmpegMissing = file && !/\.wav$/i.test(file.name) && S.setup.items.some((i) => i.id === "ffmpeg" && !i.ok);
  btn.disabled = !ok || need.length > 0 || !src || (src === "upload" && !file && !haveUpload) || !!ffmpegMissing || S.uploading;
  $("#setupHint").textContent = need.length ? `Install first (Setup and permissions): ${need.map((n) => n.name).join(", ")}.` : !ok ? "Accept the two boxes in Setup and permissions first."
    : !src ? "Choose how you'd like to give your voice." : ffmpegMissing ? "Install ffmpeg (above) to read MP3 and other compressed files, or choose a WAV." : "";
  updateTabs();
}
async function saveSetting(key, value) {  // one field per request, so two quick clicks can't overwrite each other
  setupGate();
  const s = await api("/api/settings", { method: "POST", body: form({ [key]: value }) });
  S.setup.settings = { ...S.setup.settings, ...s };
  setupGate();
}
function enterMyVoice() {
  setupGate();
  const p = S.profile, box = $("#mvStatus");
  const has = p && p.takes && p.takes.length;
  box.hidden = !has;
  if (has) {
    $("#mvStatusText").textContent = p.source === "upload"
      ? `From ${p.upload_file}: ${p.takes.length} clip${p.takes.length > 1 ? "s" : ""} cut from it${p.ready ? ", ready to use." : "."}`
      : `${p.takes.length} recorded take${p.takes.length > 1 ? "s" : ""}${p.ready ? ", ready to use." : `. Record at least ${p.min_takes} good ones.`}`;
    $("#mvRecord").hidden = p.source === "upload"; $("#mvChoose").hidden = p.source === "upload";
  }
}
$("#mvRecord").onclick = () => show("record");
$("#mvChoose").onclick = () => show("profile");
$("#toSetupLink").onclick = (e) => { e.preventDefault(); show("setup"); };
$("#setupDone").onclick = () => show(window.VOICES && VOICES.anyReady() ? "studio" : "myvoice");
$("#consent").onchange = (e) => saveSetting("consent", e.target.checked);
$("#wmAck").onchange = (e) => saveSetting("watermark_ack", e.target.checked);
$$('input[name="voiceSource"]').forEach((r) => (r.onchange = async () => {
  $("#uploadResult").hidden = true;
  await saveSetting("voice_source", r.value);
  loadSetup();  // the microphone row only applies to recording
}));
const fmtMMSS = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
$("#voiceFile").onchange = () => {
  const f = $("#voiceFile").files[0];
  const mb = f ? f.size / 1048576 : 0;
  $("#voiceFileName").textContent = f ? `${f.name} · ${mb.toFixed(1)} MB${mb > 30 ? " · large file: it works (up to 200 MB), but under 30 MB uploads faster" : ""}` : "No file chosen";
  $("#uploadResult").hidden = true; setupGate();
};
$("#toRecord").onclick = async () => {
  const src = voiceSource();
  if (src === "record") return show("record");
  const f = $("#voiceFile").files[0];
  if (!f) return show("studio");  // keep the recording already in use
  const n = S.profile ? S.profile.takes.length : 0;
  if (n && S.profile.source !== "upload" && !confirm(`This replaces your ${n} recorded take${n > 1 ? "s" : ""} with ${f.name}. Continue?`)) return;
  const btn = $("#toRecord"), out = $("#uploadResult");
  S.uploading = true; setupGate(); btn.textContent = "Reading and checking…"; out.hidden = true;
  try {
    const r = await api("/api/upload-voice", { method: "POST", body: form({ file: f }) });
    S.profile = r; S.setup.settings.voice_source = "upload";
    $("#voiceFile").value = "";
    if (r.ready) { toast(`Voice ready from ${r.file}${r.truncated ? " (the first 10 minutes were used)" : ""}`); S.uploading = false; setupGate(); show("studio"); return; }
    out.className = "verdict redo"; out.hidden = false;
    out.innerHTML = `<h3>${icon("warn")} No usable part in ${esc(r.file)}</h3>
      <ul>${r.takes.map((t) => `<li>${fmtMMSS(t.upload.start)}–${fmtMMSS(t.upload.end)}: ${esc(t.analysis.issues.map((i) => i.text).join(" "))}</li>`).join("")}</ul>
      <p class="small" style="margin:8px 0 0">Try a cleaner recording of one person talking, or record your voice instead.</p>`;
  } catch (e) {
    out.className = "verdict redo"; out.hidden = false;
    out.innerHTML = `<h3>${icon("warn")} Couldn't use that file</h3><p style="margin:6px 0 0">${esc(e.message)}</p>`;
  } finally { S.uploading = false; setupGate(); }
};

async function askInstall(item) {
  const info = await api("/api/setup/install", { method: "POST", body: form({ item, confirm: false }) });
  $("#modalTitle").textContent = "Install " + item;
  $("#modalText").textContent = `This will run the command below: ${info.what}. Nothing runs until you press "Run it".`;
  $("#modalCmd").textContent = info.command;
  const ok = await modal();
  if (!ok) return;
  await api("/api/setup/install", { method: "POST", body: form({ item, confirm: true }) });
  toast("Installing…");
  loadSetup();
}
function modal() {
  return new Promise((res) => {
    const m = $("#modal"); const prev = document.activeElement; m.hidden = false; $("#modalCancel").focus();
    const done = (v) => { m.hidden = true; document.removeEventListener("keydown", key); prev && prev.focus(); res(v); };
    const key = (e) => { if (e.key === "Escape") done(false); if (e.key === "Tab") { const f = [$("#modalCancel"), $("#modalOk")]; const i = f.indexOf(document.activeElement); e.preventDefault(); f[(i + 1) % 2].focus(); } };
    document.addEventListener("keydown", key);
    $("#modalOk").onclick = () => done(true); $("#modalCancel").onclick = () => done(false);
  });
}

// ---------------- microphone (raw PCM via AudioWorklet) ----------------
const MIC = { ctx: null, stream: null, node: null, settings: null, chunks: [], recording: false, level: -90, onLevel: null };
const CONSTRAINTS = (deviceId) => ({ audio: { deviceId: deviceId ? { exact: deviceId } : undefined, echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 } });

async function openMic(deviceId) {
  try {
    if (MIC.stream) MIC.stream.getTracks().forEach((t) => t.stop());
    MIC.stream = await navigator.mediaDevices.getUserMedia(CONSTRAINTS(deviceId));
    S.micGranted = true;
  } catch (e) {
    S.micGranted = false; toast("Microphone blocked: " + e.message); return false;
  }
  const track = MIC.stream.getAudioTracks()[0];
  MIC.settings = track.getSettings();
  if (!MIC.ctx) {
    MIC.ctx = new AudioContext({ latencyHint: "interactive" });
    await MIC.ctx.audioWorklet.addModule("recorder-worklet.js");
  }
  if (MIC.ctx.state === "suspended") await MIC.ctx.resume();
  if (MIC.src) MIC.src.disconnect();
  MIC.src = MIC.ctx.createMediaStreamSource(MIC.stream);
  if (!MIC.node) {
    MIC.node = new AudioWorkletNode(MIC.ctx, "raw-recorder", { numberOfInputs: 1, numberOfOutputs: 1, channelCount: 1, channelCountMode: "explicit" });
    // Route to the destination through a muted gain so the graph is always pulled (some engines
    // deliver silence to sources that don't reach the destination). Nothing is played out loud.
    const mute = MIC.ctx.createGain(); mute.gain.value = 0;
    MIC.node.connect(mute).connect(MIC.ctx.destination);
    MIC.node.port.onmessage = (e) => {
      const d = e.data;
      MIC.level = 20 * Math.log10(Math.max(d.peak, 1e-6));
      if (d.samples && MIC.recording) { MIC.chunks.push(d.samples); MIC.onBlock && MIC.onBlock(d); }
      MIC.onLevel && MIC.onLevel(MIC.level, d);
    };
  }
  MIC.src.connect(MIC.node);
  const s = MIC.settings;
  $("#rawNote").textContent = `Raw capture at ${MIC.ctx.sampleRate / 1000} kHz: echo cancellation ${s.echoCancellation ? "ON" : "off"}, noise suppression ${s.noiseSuppression ? "ON" : "off"}, auto gain ${s.autoGainControl ? "ON" : "off"}.`;
  await listDevices(s.deviceId);
  return true;
}
async function listDevices(current) {
  const devs = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "audioinput");
  $("#deviceSel").innerHTML = devs.map((d, i) => `<option value="${esc(d.deviceId)}" ${d.deviceId === current ? "selected" : ""}>${esc(d.label || "Microphone " + (i + 1))}</option>`).join("");
}
$("#deviceSel").onchange = (e) => openMic(e.target.value);

function startCapture() { MIC.chunks = []; MIC.recording = true; MIC.node.port.postMessage({ recording: true }); }
function stopCapture() {
  MIC.recording = false; MIC.node.port.postMessage({ recording: false });
  const n = MIC.chunks.reduce((a, c) => a + c.length, 0);
  const all = new Float32Array(n); let o = 0; for (const c of MIC.chunks) { all.set(c, o); o += c.length; }
  return all;
}
function encodeWav(samples, sr) {  // 16-bit PCM mono, lossless container
  const buf = new ArrayBuffer(44 + samples.length * 2); const v = new DataView(buf);
  const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  w(0, "RIFF"); v.setUint32(4, 36 + samples.length * 2, true); w(8, "WAVE"); w(12, "fmt "); v.setUint32(16, 16, true);
  v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  w(36, "data"); v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) { const s = Math.max(-1, Math.min(1, samples[i])); v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true); }
  return new Blob([buf], { type: "audio/wav" });
}

function meterLoop() {
  const db = Math.max(-60, Math.min(0, MIC.level));
  const pct = ((db + 60) / 60) * 100;
  $("#meterFill").style.width = `calc(${pct}% - 6px)`;
  $("#meter").setAttribute("aria-valuenow", Math.round(db));
  if (MIC.stream) $("#meterText").textContent = db > -3 ? "Too loud: back off a little" : db > -24 ? "Good level" : db > -40 ? "Quiet: speak up or move closer" : "Listening…";
  requestAnimationFrame(meterLoop);
}

// ---------------- record wizard ----------------
const REC = { state: "idle", t0: 0, speech: 0, timer: null, lastBlob: null, roomFloor: null, audio: new Audio() };
const MAX_S = 15, MIN_SPEECH = 8;

async function enterRecord() {
  if (!S.steps.length) S.steps = (await api("/api/steps")).steps;
  S.profile = await api("/api/profile");
  if (!MIC.stream) await openMic();
  const done = new Set(S.profile.takes.map((t) => t.step));
  if (!REC.init) { REC.init = true; S.step = S.steps.find((s) => !done.has(s.id))?.id || 1; }
  renderStep();
}
function takeFor(step) { return S.profile && S.profile.takes.find((t) => t.step === step); }
function renderBeads() {
  $("#beads").innerHTML = S.steps.map((s) => {
    const t = takeFor(s.id); const cls = t ? (t.analysis.verdict === "redo" ? "redo" : "done") : "";
    const state = t ? (t.analysis.verdict === "redo" ? "needs a redo" : "recorded") : "not recorded";
    return `<li><button class="bead ${cls} ${s.id === S.step ? "current" : ""}" data-step="${s.id}" aria-label="Step ${s.id}, ${esc(s.label)}: ${state}" ${s.id === S.step ? 'aria-current="step"' : ""}>${t ? icon(cls === "redo" ? "redo" : "check") : s.id}</button></li>`;
  }).join("");
  $$(".bead").forEach((b) => (b.onclick = () => { if (REC.state !== "idle") return; S.step = +b.dataset.step; renderStep(); }));
}
function renderStep() {
  const s = S.steps.find((x) => x.id === S.step);
  $("#toneLabel").textContent = `${s.tone} · ${s.label}`;
  $("#stepCount").textContent = `Step ${s.id} of ${S.steps.length}`;
  $("#scriptText").textContent = s.script;
  $("#hintText").textContent = s.hint;
  const t = takeFor(s.id);
  renderVerdict(t);
  $("#playTake").disabled = !t; $("#redoTake").disabled = !t; $("#nextStep").disabled = !t;
  $("#nextStep").textContent = S.step < S.steps.length ? "Next" : "Done";
  const n = S.profile.usable;  // takes marked "redo" don't count toward the minimum
  $("#toProfile").disabled = n < S.profile.min_takes;
  $("#toProfile").textContent = n < S.profile.min_takes ? `Record ${S.profile.min_takes - n} more good take${S.profile.min_takes - n > 1 ? "s" : ""} to continue` : "Continue to voice profile";
  const roomDone = REC.roomFloor !== null || S.profile.takes.length > 0;
  $("#recBtn").disabled = !roomDone;
  $("#roomText").textContent = REC.roomText || (roomDone ? "Room checked." : "Do this once before the first take. Stay quiet for 2 seconds.");
  setRing(0); $("#recTime").textContent = `0.0 s / ${MAX_S} s`;
  $("#recHelp").innerHTML = `Press <kbd>Space</kbd> to start or stop. Stops by itself at ${MAX_S} s; you can stop early once ${MIN_SPEECH} s of speech is in.`;
  renderBeads(); updateTabs();
}
function renderVerdict(t) {
  const v = $("#verdict");
  if (!t) { v.hidden = true; return; }
  const a = t.analysis;
  const title = { good: "Great take", okay: "Usable take", redo: "Better to re-record" }[a.verdict];
  v.className = `verdict ${a.verdict}`; v.hidden = false;
  v.innerHTML = `<h3>${icon(a.verdict === "good" ? "check" : "warn")} ${title}: ${a.score}/100</h3><ul>${a.issues.map((i) => `<li>${esc(i.text)}</li>`).join("")}</ul>
    <p class="small" style="margin:8px 0 0">${a.speech_seconds} s of speech · peak ${a.peak_dbfs} dBFS · noise ${a.noise_floor_dbfs} dBFS</p>`;
}
function setRing(frac) { $("#ringProg").style.strokeDashoffset = String(553 * (1 - Math.min(1, frac))); }

$("#roomBtn").onclick = async () => {
  if (!MIC.stream && !(await openMic())) return;
  $("#roomBtn").disabled = true; $("#roomText").textContent = "Listening… stay quiet.";
  startCapture(); await new Promise((r) => setTimeout(r, 2000)); const x = stopCapture();
  try {
    const r = await api("/api/room-check", { method: "POST", body: form({ file: new File([encodeWav(x, MIC.ctx.sampleRate)], "room.wav") }) });
    REC.roomFloor = r.noise_floor_dbfs; REC.roomText = r.text;
  } catch (e) { REC.roomText = "Room check failed: " + e.message; }
  $("#roomBtn").disabled = false; renderStep();
};

async function startRecording() {
  if (REC.state !== "idle" || $("#recBtn").disabled) return;
  if (!MIC.stream && !(await openMic())) return;
  REC.state = "countdown";
  const cd = $("#countdown"); cd.classList.add("show");
  for (const n of [3, 2, 1]) { cd.textContent = n; await new Promise((r) => setTimeout(r, 800)); if (REC.state !== "countdown") { cd.classList.remove("show"); return; } }
  cd.classList.remove("show");
  REC.state = "recording"; REC.t0 = performance.now(); REC.speech = 0;
  const floor = REC.roomFloor ?? -60;
  MIC.onBlock = (d) => { const db = 20 * Math.log10(Math.max(d.rms, 1e-7)); if (db > Math.max(floor + 12, -48)) REC.speech += d.samples.length / MIC.ctx.sampleRate; };
  startCapture();
  $("#recBtn").classList.add("recording"); $("#recIcon").innerHTML = '<use href="#i-stop"/>'; $("#recLabel").textContent = "Stop";
  REC.timer = setInterval(() => {
    const el = (performance.now() - REC.t0) / 1000;
    setRing(el / MAX_S);
    const canStop = REC.speech >= MIN_SPEECH;
    $("#recTime").textContent = `${el.toFixed(1)} s / ${MAX_S} s`;
    $("#recHelp").textContent = canStop ? "Enough speech captured. Press Space or Stop when you finish the script."
      : `Speech so far: ${REC.speech.toFixed(1)} s. Stop unlocks at ${MIN_SPEECH} s; it stops by itself at ${MAX_S} s.`;
    $("#recBtn").setAttribute("aria-disabled", String(!canStop));
    if (el >= MAX_S) finishRecording();
  }, 100);
}
async function finishRecording() {
  if (REC.state !== "recording") return;
  clearInterval(REC.timer); REC.state = "uploading";
  const x = stopCapture();
  $("#recBtn").classList.remove("recording"); $("#recBtn").removeAttribute("aria-disabled");
  $("#recIcon").innerHTML = '<use href="#i-mic"/>'; $("#recLabel").textContent = "Checking…";
  const blob = encodeWav(x, MIC.ctx.sampleRate);
  try {
    await api(`/api/takes/${S.step}`, { method: "POST", body: form({ file: new File([blob], `step${S.step}.wav`, { type: "audio/wav" }), constraints: JSON.stringify(MIC.settings || {}) }) });
    S.profile = await api("/api/profile");
  } catch (e) { toast("Upload failed: " + e.message); }
  $("#recLabel").textContent = "Record"; REC.state = "idle"; renderStep(); $("#nextStep").focus();
}
function recToggle() {
  if (REC.state === "idle") startRecording();
  else if (REC.state === "countdown") { REC.state = "idle"; $("#countdown").classList.remove("show"); }
  else if (REC.state === "recording") {
    if (REC.speech >= MIN_SPEECH) finishRecording(); else toast(`Keep going: stop unlocks after ${MIN_SPEECH} s of speech.`);
  }
}
$("#recBtn").onclick = recToggle;
document.addEventListener("keydown", (e) => {
  if (e.code !== "Space" || S.screen !== "record" || !$("#modal").hidden) return;
  const tag = (e.target.tagName || "").toLowerCase();
  if (["input", "textarea", "select"].includes(tag)) return;
  if (tag === "button" && e.target.id !== "recBtn") return; // let Space activate the focused button
  e.preventDefault(); recToggle();
});
$("#playTake").onclick = () => { REC.audio.src = `/api/takes/${S.step}/audio?t=${Date.now()}`; REC.audio.play(); };
$("#redoTake").onclick = () => { $("#recBtn").focus(); startRecording(); };
$("#nextStep").onclick = () => {
  const done = new Set(S.profile.takes.map((t) => t.step));
  const next = S.steps.find((s) => s.id > S.step && !done.has(s.id)) || S.steps.find((s) => !done.has(s.id));
  if (next) { S.step = next.id; renderStep(); $("#recBtn").focus(); }
  else if (S.profile.ready) show("profile");
};
$("#toProfile").onclick = () => show("profile");



// ---------------- profile ----------------
async function renderProfile() {
  S.profile = await api("/api/profile");
  const p = S.profile;
  $("#takesTable tbody").innerHTML = p.takes.map((t) => `<tr>
    <td><strong>${esc(t.label)}</strong><div class="small muted">${esc(t.tone)} · ${t.trimmed_seconds} s</div></td>
    <td><span class="score"><span class="badge ${t.analysis.verdict}">${t.analysis.score}</span><span class="small">${{ good: "Good", okay: "Okay", redo: t.source === "upload" ? "Not used" : "Redo" }[t.analysis.verdict]}${t.step === p.best_take ? " · best" : ""}</span></span>${t.analysis.verdict === "redo" ? `<div class="small muted">${esc((t.analysis.issues.find((i) => i.level === "bad") || t.analysis.issues[0] || {}).text || "")}</div>` : ""}</td>
    <td><button class="clay-btn icon-btn round" data-play="${t.step}" aria-label="Play ${esc(t.label)} take">${icon("play")}</button></td>
    <td><label class="radio"><input type="radio" name="voice" value="${t.step}" ${t.step === p.voice_take ? "checked" : ""}><span class="dot"></span><span class="sr-only">Use ${esc(t.label)} for voice</span></label></td>
    <td><label class="radio"><input type="radio" name="style" value="${t.step}" ${t.step === p.style_take ? "checked" : ""}><span class="dot"></span><span class="sr-only">Use ${esc(t.label)} for style</span></label></td>
    <td><button class="clay-btn" data-rerec="${t.step}">${icon("redo")} ${t.source === "upload" ? "Record instead" : "Re-record"}</button></td></tr>`).join("");
  $$("[data-play]").forEach((b) => (b.onclick = () => { REC.audio.src = `/api/takes/${b.dataset.play}/audio?t=${Date.now()}`; REC.audio.play(); }));
  $$("[data-rerec]").forEach((b) => (b.onclick = () => { S.step = +b.dataset.rerec; show("record"); }));
  $$('input[name="voice"]').forEach((r) => (r.onchange = async () => { S.profile = await api("/api/profile", { method: "POST", body: form({ voice_take: r.value }) }); toast("Voice take updated"); }));
  $$('input[name="style"]').forEach((r) => (r.onchange = async () => { S.profile = await api("/api/profile", { method: "POST", body: form({ style_take: r.value }) }); toast("Style take updated"); }));
  $("#profileHint").textContent = p.ready ? "The voice take sets how you sound (first 10 s are used). The style take sets how you speak (first 6 s)." : `You need at least ${p.min_takes} usable takes. Record more, or upload a longer recording on the Record screen.`;
  $("#toStudio").disabled = !p.ready;
  updateTabs();
}
$("#toStudio").onclick = () => show("studio");

// ---------------- studio controls ----------------
const CONTROLS = [
  { key: "similarity", label: "Similarity to the voice", help: "Lower mixes in a neutral stock voice, so it sounds like the chosen voice but not an exact copy.",
    min: 0.5, max: 1.0, step: 0.05, def: 1.0, ends: ["Blended", "Exact copy"], fmt: (v) => `${Math.round(v * 100)}%` },
  { key: "exaggeration", label: "Expressiveness", help: "How much emotion and emphasis goes into the delivery.",
    min: 0.25, max: 1.5, step: 0.05, def: 0.5, ends: ["Calm", "Dramatic"], fmt: (v) => v.toFixed(2) },
  { key: "cfg_weight", label: "Pacing", help: "Lower is slower and more deliberate; higher is brisker.",
    min: 0.2, max: 0.8, step: 0.05, def: 0.5, ends: ["Slow", "Brisk"], fmt: (v) => v.toFixed(2) },
  { key: "depth", label: "Voice depth", help: "Makes the voice deeper or brighter by re-tuning your recording before cloning (the result itself is never pitch-shifted).",
    min: -3, max: 3, step: 0.5, def: 0, ends: ["Deeper", "Brighter"], fmt: (v) => `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v)} semitones` },
  { key: "temperature", label: "Variation", help: "Steady keeps the delivery predictable; lively lets each take come out a bit different.",
    min: 0.4, max: 1.2, step: 0.05, def: 0.8, ends: ["Steady", "Lively"], fmt: (v) => v.toFixed(2) },
];
function renderSliders() {
  $("#sliders").innerHTML = CONTROLS.map((c) => {
    const defPct = ((c.def - c.min) / (c.max - c.min)) * 100;
    return `<div class="ctrl"><div class="ctrl-head"><label for="c-${c.key}">${c.label}</label><span class="ctrl-val" id="v-${c.key}"></span></div>
      <p class="ctrl-help" id="h-${c.key}">${c.help}</p>
      <div class="range-wrap"><input type="range" id="c-${c.key}" min="${c.min}" max="${c.max}" step="${c.step}" value="${c.def}" aria-describedby="h-${c.key}">
        <span class="def-mark${defPct > 90 ? " at-end" : ""}" style="left:calc(15px + (100% - 30px) * ${defPct / 100})" aria-hidden="true"></span></div>
      <div class="ctrl-ends" aria-hidden="true"><span>${c.ends[0]}</span><span>${c.ends[1]}</span></div>
      <p class="ctrl-measure weak-note" id="w-${c.key}" hidden>Small effect in this language (measured). It still works, but expect subtle changes.</p>
      ${c.key === "similarity" ? '<p class="ctrl-measure" id="simMeasured"></p>' : ""}</div>`;
  }).join("");
  for (const c of CONTROLS) {
    const el = $(`#c-${c.key}`);
    const upd = () => { const v = +el.value; $(`#v-${c.key}`).textContent = c.fmt(v) + (Math.abs(v - c.def) < 1e-6 ? " · default" : ""); el.setAttribute("aria-valuetext", c.fmt(v)); };
    el.oninput = () => { upd(); S.activePreset = null; renderPresets(); };
    upd();
  }
}
function params() {
  const p = {}; for (const c of CONTROLS) p[c.key] = +$(`#c-${c.key}`).value;
  p.seed = $("#seedOn").checked ? +$("#seedVal").value : null; return p;
}
function setParams(p) { for (const c of CONTROLS) if (p[c.key] !== undefined) { const el = $(`#c-${c.key}`); el.value = p[c.key]; el.oninput(); } }
async function loadPresets() { S.presets = (await api("/api/presets")).presets; renderPresets(); }
function renderPresets() {
  $("#presets").innerHTML = S.presets.map((p) => `<span class="chip-wrap"><button class="chip" data-preset="${p.id}" aria-pressed="${p.id === S.activePreset}">${esc(p.name)}${p.builtin ? "" : `</button><button class="chip del-chip" data-del="${p.id}" aria-label="Delete preset ${esc(p.name)}">×`}</button></span>`).join("");
  $$("[data-preset]").forEach((b) => (b.onclick = () => { const p = S.presets.find((x) => x.id === b.dataset.preset); setParams(p.params); S.activePreset = p.id; renderPresets(); }));
  $$("[data-del]").forEach((b) => (b.onclick = async () => { S.presets = (await api(`/api/presets/${b.dataset.del}`, { method: "DELETE" })).presets; renderPresets(); }));
}
$("#savePreset").onclick = async () => {
  const name = $("#presetName").value.trim(); if (!name) { $("#presetName").focus(); toast("Give the preset a name"); return; }
  S.presets = (await api("/api/presets", { method: "POST", body: form({ name, params: JSON.stringify(params()) }) })).presets;
  S.activePreset = S.presets[S.presets.length - 1].id; $("#presetName").value = ""; renderPresets(); toast("Preset saved");
};
$("#seedOn").onchange = () => { $("#seedVal").disabled = !$("#seedOn").checked; };

// ---------------- text + length ----------------
// Length comes only from how much text is spoken (the audio is never stretched). The estimate uses
// this voice's measured pace (chars/second), learned from every generation.
S.cps = 15;
S.lang = "en"; S.langs = [];
const lenTarget = () => ($('input[name="lenMode"]:checked').value === "target" ? +$("#lenTarget").value : 0);
const speechSecs = (t) => t.replace(/\s+/g, " ").trim().length / S.cps;
const isCJK = () => (S.langs.find((l) => l.code === S.lang) || {}).cjk;
// Mirrors engine.split_text: . ! ? plus the Hindi danda, Arabic question mark and CJK full-width marks.
function splitSentences(t) { t = t.trim(); return t ? t.split(/(?<=[.!?।॥؟])\s+|(?<=[。！？])\s*/).filter((x) => x.trim()) : []; }
function wordCount(t) { t = t.trim(); if (!t) return 0; return isCJK() ? Math.floor(t.replace(/[\s\p{P}\p{S}]/gu, "").length / 2) : t.split(/\s+/).length; }
function fitToTarget(t, target) {  // mirrors engine.generate: sentences in order while each gets closer
  let tot = 0, n = 0; const ss = splitSentences(t);
  for (const x of ss) { const end = tot + speechSecs(x); if (tot > 0 && (end > target + 1 || Math.abs(end - target) >= Math.abs(tot - target))) break; tot = end; n++; }
  return { seconds: tot, used: n, total: ss.length };
}
function textStats() {
  const t = $("#textIn").value.trim(); const w = wordCount(t);
  const langOk = (S.langs.find((l) => l.code === S.lang) || { available: true }).available;
  const parts = splitSentences(t).length; const secs = speechSecs(t);
  $("#textStats").textContent = `${isCJK() ? `${t.replace(/\s/g, "").length} characters` : `${w} words`} · about ${Math.max(1, Math.round(secs))} s of speech${parts > 1 ? ` · ${parts} sentences, generated one by one` : ""}${w && w < 4 ? " · type at least 4 words" : ""}`;
  const vOk = !window.VOICES || VOICES.ready(ttsVoice());
  $("#genBtn").disabled = w < 4 || !langOk || !vOk;
  $("#previewBtn").disabled = !langOk || !vOk;
  if (!vOk) $("#textStats").textContent = "Pick a voice that's ready, or add one under Your voices (record My voice, or add a clip).";
  const tgt = lenTarget(); const h = $("#lenHelp"); h.classList.remove("warn");
  $("#lenCtrl").hidden = !tgt;
  $("#lenVal").textContent = `${$("#lenTarget").value} s`;
  $("#lenTarget").setAttribute("aria-valuetext", `${$("#lenTarget").value} seconds`);
  $("#previewLbl").textContent = `Preview (${tgt || 5} s)`;
  if (!tgt) { h.textContent = "Generate speaks all of your text. Preview speaks a built-in 5 s passage."; return; }
  if (w < 4) { h.textContent = ""; return; }
  const f = fitToTarget(t, tgt);
  if (secs < tgt - 1.5) {
    const more = Math.ceil(((tgt - secs) * S.cps) / (isCJK() ? 1 : 5.5));  // ~5.5 chars per word incl. space
    h.textContent = `Your text is about ${Math.round(secs)} s. Add about ${more} more ${isCJK() ? "characters" : "words"} to reach ${tgt} s. Speech can't be stretched without changing the voice. Preview always fills ${tgt} s.`;
    h.classList.add("warn");
  } else if (f.used < f.total) {
    h.textContent = `Generate stops after sentence ${f.used} of ${f.total}, at about ${Math.round(f.seconds)} s, the whole sentence nearest ${tgt} s.`;
  } else {
    h.textContent = `Your text fits: about ${Math.round(secs)} s.`;
  }
}
$$('input[name="lenMode"]').forEach((r) => (r.onchange = textStats));
$("#lenTarget").oninput = textStats;
$("#textIn").oninput = textStats;

// ---------------- language ----------------
const DEFAULT_SAMPLE = $("#textIn").value;
async function loadLanguages() {
  let d; try { d = await api("/api/languages"); } catch { return; }
  S.langs = d.languages;
  if (!S.langs.some((l) => l.code === S.lang && l.available)) S.lang = "en";
  $("#langSel").innerHTML = S.langs.map((l) => `<option value="${l.code}" ${l.available ? "" : "disabled"} ${l.code === S.lang ? "selected" : ""}>${esc(l.name)}${l.available ? "" : " · needs download"}</option>`).join("");
  langNote(d.multilingual_ready);
}
function langNote(ready) {
  const n = $("#langNote");
  if (!ready) {
    n.innerHTML = `English only for now. To speak Hindi, Spanish, Chinese and 19 more languages, get the optional multilingual model (about 2.2 GB) on <a href="#" id="toSetupLang">Setup</a>.`;
    $("#toSetupLang").onclick = (e) => { e.preventDefault(); show("setup"); };
    n.hidden = false;
  } else if (S.lang !== "en") {
    n.textContent = "Your voice can come from a recording in any language. A recording in this language usually sounds most natural.";
    n.hidden = false;
  } else n.hidden = true;
}
function applyLanguage(code, save) {
  const prev = S.langs.find((l) => l.code === S.lang), next = S.langs.find((l) => l.code === code);
  if (!next) return;
  const cur = $("#textIn").value.trim();
  if (!cur || cur === DEFAULT_SAMPLE.trim() || (prev && cur === prev.sample)) $("#textIn").value = code === "en" ? DEFAULT_SAMPLE : next.sample;
  S.lang = code; S.cps = next.cps || S.cps;
  $("#textIn").lang = code;
  langNote(S.langs.some((l) => l.code !== "en" && l.available));
  for (const c of CONTROLS) { const w = $(`#w-${c.key}`); if (w) w.hidden = !(next.weak_controls || []).includes(c.key); }
  if (save) saveSetting("language", code);
  textStats();
}
$("#langSel").onchange = (e) => applyLanguage(e.target.value, true);
$("#ttsVoice").onchange = () => { textStats(); const p = S.profile; $("#voiceSrcLine").hidden = !(ttsVoice() === "me" && p && p.source === "upload"); };

const ttsVoice = () => $("#ttsVoice").value || "me";
async function enterStudio() {
  if (!$("#sliders").children.length) { renderSliders(); await loadPresets(); }
  if (window.VOICES) { await VOICES.refresh(); VOICES.fillSelect($("#ttsVoice"), (S.setup && S.setup.settings.tts_voice) || "me"); }
  if (S.setup && S.setup.settings.language) S.lang = S.setup.settings.language;
  await loadLanguages();
  applyLanguage(S.lang, false);
  const line = $("#voiceSrcLine"), p = S.profile;
  if (ttsVoice() === "me" && p && p.source === "upload") {
    const t = p.takes.find((x) => x.step === p.voice_take);
    line.innerHTML = `Voice from <strong>${esc(p.upload_file)}</strong>${t ? ` (${fmtMMSS(t.upload.start)}–${fmtMMSS(t.upload.end)}, the clearest part)` : ""} · <a href="#" id="changeSrc">Change</a>`;
    line.hidden = false; $("#changeSrc").onclick = (e) => { e.preventDefault(); show("myvoice"); };
  } else line.hidden = true;
  textStats();
}

// ---------------- generation ----------------
async function generate(preview) {
  $("#genError").hidden = true;
  $("#genBtn").disabled = $("#previewBtn").disabled = true;
  $("#progWrap").hidden = false; setProg(0, preview ? "Starting preview…" : "Starting…");
  try {
    let job = await api("/api/generate", { method: "POST", body: form({ text: $("#textIn").value, params: JSON.stringify(params()), preview, target: lenTarget() || (preview ? 5 : 0), language: S.lang, voice: ttsVoice() }) });
    const t0 = performance.now();
    while (job.state !== "done" && job.state !== "error") {
      await new Promise((r) => setTimeout(r, 500));
      job = await api(`/api/jobs/${job.id}`);
      const el = (performance.now() - t0) / 1000;
      const frac = Math.min(0.97, Math.max(job.done / job.total, el / job.estimate));
      const left = Math.max(1, Math.round(job.estimate - el));
      setProg(frac, job.state === "loading" ? "Loading the voice engine (first run only)…" : `Part ${Math.min(job.done + 1, job.total)} of ${job.total} · about ${left} s left`);
    }
    if (job.state === "error") throw new Error(job.error);
    if (job.result.cps) { S.cps = job.result.cps; const l = S.langs.find((x) => x.code === (job.result.language || S.lang)); if (l) l.cps = job.result.cps; }
    const un = job.result.unspoken ? ` · ${job.result.unspoken} sentence${job.result.unspoken > 1 ? "s" : ""} past the ${job.result.target} s length not spoken` : "";
    setProg(1, `Done in ${job.elapsed} s for ${job.result.seconds} s of audio${un}.`);
    showResult(job);
  } catch (e) {
    $("#genError").textContent = e.message; $("#genError").hidden = false; setProg(0, "");
  } finally { $("#previewBtn").disabled = false; textStats(); }
}
function setProg(f, txt) { $("#progFill").style.width = `calc(${f * 100}% - 6px)`; $("#prog").setAttribute("aria-valuenow", Math.round(f * 100)); $("#progText").textContent = txt; }
$("#previewBtn").onclick = () => generate(true);
$("#genBtn").onclick = () => generate(false);

function showResult(job) {
  S.lastOut = job; S.lastExport = null;
  const a = $("#outAudio"); a.src = job.result.url; a.load();
  $("#playOut").disabled = false; $("#exportBox").disabled = false; $("#revealBtn").disabled = true; $("#exportMsg").textContent = "";
  const sim = Math.round(job.result.similarity * 100);
  $("#simFill").style.width = `calc(${Math.max(0, Math.min(100, sim))}% - 6px)`;
  $("#simBar").setAttribute("aria-valuenow", sim);
  $("#simText").textContent = `${sim}%`;
  $("#simMeasured").textContent = `Measured on the last result: ${sim}% similar to the voice`;
  $("#outTime").textContent = `0:00 / ${fmtTime(job.result.seconds)}`;
  drawWave();
}
function drawWave() {
  const c = $("#wave"); if (!c || !S.lastOut) return;
  const peaks = S.lastOut.result.peaks; const dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth, h = c.clientHeight; c.width = w * dpr; c.height = h * dpr;
  const g = c.getContext("2d"); g.scale(dpr, dpr); g.clearRect(0, 0, w, h);
  const cs = getComputedStyle(document.documentElement);
  const a = $("#outAudio"); const pos = a.duration ? a.currentTime / a.duration : 0;
  const n = peaks.length, bw = w / n, max = Math.max(...peaks, 0.01);
  for (let i = 0; i < n; i++) {
    const v = (peaks[i] / max) * (h / 2 - 4); g.fillStyle = i / n <= pos ? cs.getPropertyValue("--primary") : cs.getPropertyValue("--muted");
    g.globalAlpha = i / n <= pos ? 1 : 0.55; g.fillRect(i * bw, h / 2 - v, Math.max(1, bw - 0.6), Math.max(1.5, v * 2));
  }
  g.globalAlpha = 1;
}
$("#playOut").onclick = () => { const a = $("#outAudio"); a.paused ? a.play() : a.pause(); };
$("#outAudio").onplay = () => { $("#playOut").innerHTML = icon("pause"); $("#playOut").setAttribute("aria-label", "Pause result"); (function loop() { drawWave(); $("#outTime").textContent = `${fmtTime($("#outAudio").currentTime)} / ${fmtTime(S.lastOut.result.seconds)}`; if (!$("#outAudio").paused) requestAnimationFrame(loop); })(); };
$("#outAudio").onpause = $("#outAudio").onended = () => { $("#playOut").innerHTML = icon("play"); $("#playOut").setAttribute("aria-label", "Play result"); drawWave(); };
$("#wave").onclick = (e) => { const a = $("#outAudio"); if (!a.duration) return; const r = e.target.getBoundingClientRect(); a.currentTime = ((e.clientX - r.left) / r.width) * a.duration; drawWave(); };
window.addEventListener("resize", drawWave);

// ---------------- export ----------------
$$('input[name="fmt"]').forEach((r) => (r.onchange = () => {
  const mp3 = $('input[name="fmt"]:checked').value === "mp3";
  $("#brBox").hidden = !mp3;
  $("#fmtHint").textContent = mp3 ? "Stored at 48 kHz so all bitrates are possible. Smaller files." : "24 kHz, 16-bit, lossless.";
}));
// Save: the macOS app opens a native Save dialog (window.pywebview.api); a browser gets a Save picker
// where it has one (Chrome/Edge) and a normal download otherwise. Nothing goes to a fixed folder.
const desktopApi = () => (window.pywebview && window.pywebview.api && window.pywebview.api.save_export ? window.pywebview.api : null);
function exportName(fmt) { const n = $("#fname").value.trim().replace(/[^A-Za-z0-9 _\-.]/g, "").replace(/^[ .]+|[ .]+$/g, "") || "my-voice"; return n.toLowerCase().endsWith("." + fmt) ? n : `${n}.${fmt}`; }
$("#exportBtn").onclick = async () => {
  if (!S.lastOut) return;
  const fmt = $('input[name="fmt"]:checked').value, bitrate = $("#bitrate").value, name = exportName(fmt);
  const msg = $("#exportMsg"); S.lastExport = null; $("#revealBtn").disabled = true;
  try {
    const d = desktopApi();
    if (d) {
      const r = await d.save_export(S.lastOut.id, fmt, +bitrate, name);
      if (r.cancelled) { msg.textContent = "Not saved."; return; }
      if (r.error) throw new Error(r.error);
      S.lastExport = r; $("#revealBtn").disabled = false;
      msg.textContent = `Saved ${r.name} (${(r.bytes / 1024).toFixed(0)} KB) in ${r.folder}.`;
      return;
    }
    let handle = null;  // requested first, while the click still counts as a user gesture
    if (window.showSaveFilePicker) {
      try {
        handle = await window.showSaveFilePicker({ suggestedName: name, types: [{ description: fmt.toUpperCase() + " audio", accept: { [fmt === "mp3" ? "audio/mpeg" : "audio/wav"]: ["." + fmt] } }] });
      } catch (e) { if (e.name === "AbortError") { msg.textContent = "Not saved."; return; } }
    }
    const res = await fetch("/api/export", { method: "POST", body: form({ output_id: S.lastOut.id, fmt, bitrate, filename: name }) });
    if (!res.ok) { let m = res.statusText; try { m = (await res.json()).detail || m; } catch {} throw new Error(m); }
    const blob = await res.blob();
    if (handle) {
      const w = await handle.createWritable(); await w.write(blob); await w.close();
      msg.textContent = `Saved ${handle.name} (${(blob.size / 1024).toFixed(0)} KB).`;
    } else {
      const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = name;
      document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 4000);
      msg.textContent = `Downloaded ${name} (${(blob.size / 1024).toFixed(0)} KB). Your browser picks the folder; the macOS app asks you where to save.`;
    }
  } catch (e) { msg.textContent = "Export failed: " + e.message; }
};
$("#revealBtn").onclick = () => S.lastExport && desktopApi() && desktopApi().reveal(S.lastExport.path);

// ---------------- boot ----------------
(async function boot() {
  let t = null; try { t = localStorage.getItem("vcs-theme"); } catch {}
  applyTheme(t);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => applyTheme(document.documentElement.dataset.theme));
  try { const p = await navigator.permissions.query({ name: "microphone" }); S.micGranted = p.state === "granted" ? true : p.state === "denied" ? false : undefined; } catch {}
  await loadSetup();
  S.profile = await api("/api/profile");
  if (window.VOICES) await VOICES.refresh();
  setupGate();
  meterLoop();
  const st = S.setup.settings;
  const start = new URLSearchParams(location.search).get("screen");
  if (start && !$(`.tab[data-screen="${start}"]`).disabled) show(start);
  else if (st.consent && st.watermark_ack && (window.VOICES ? VOICES.anyReady() : S.profile.ready)) show("studio");
  else show("setup");
})();

// Native-only controls appear once the macOS app bridge is ready.
const showNative = () => { if (desktopApi()) $("#revealBtn").hidden = false; };
window.addEventListener("pywebviewready", showNative); showNative();
