"use strict";
// Your voices (sidebar library) and the Speech to speech screen. Loaded after app.js and uses its
// helpers ($, $$, api, form, icon, esc, toast, fmtTime, show, S).

const fmtClock = (s) => {
  s = Math.max(0, Math.round(s || 0));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), x = String(s % 60).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${x}` : `${m}:${x}`;
};
const fmtDur = (s) => { s = Math.round(s || 0); if (s < 60) return `${s} s`; const m = Math.round(s / 60); return m < 60 ? `${m} min` : `${Math.floor(m / 60)} h ${m % 60} min`; };

function drawPeaks(c, peaks, pos) {
  if (!c || !peaks || !peaks.length || !c.clientWidth) return;
  const dpr = window.devicePixelRatio || 1, w = c.clientWidth, h = c.clientHeight;
  c.width = w * dpr; c.height = h * dpr;
  const g = c.getContext("2d"); g.scale(dpr, dpr); g.clearRect(0, 0, w, h);
  const cs = getComputedStyle(document.documentElement);
  const n = peaks.length, bw = w / n, max = Math.max(...peaks, 0.01);
  for (let i = 0; i < n; i++) {
    const v = (peaks[i] / max) * (h / 2 - 3), played = i / n <= pos;
    g.fillStyle = played ? cs.getPropertyValue("--primary") : cs.getPropertyValue("--muted");
    g.globalAlpha = played ? 1 : 0.5;
    g.fillRect(i * bw, h / 2 - v, Math.max(1, bw - 0.5), Math.max(1.5, v * 2));
  }
  g.globalAlpha = 1;
}

function uploadFile(url, file, extra, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest(), fd = new FormData(); fd.append("file", file);
    for (const [k, v] of Object.entries(extra || {})) fd.append(k, v);
    xhr.upload.onprogress = (e) => { if (onProgress && e.lengthComputable) onProgress(e.loaded / e.total); };
    xhr.onload = () => {
      let body = null; try { body = JSON.parse(xhr.responseText); } catch {}
      if (xhr.status >= 200 && xhr.status < 300) resolve(body);
      else reject(new Error((body && body.detail) || xhr.statusText || "Upload failed"));
    };
    xhr.onerror = () => reject(new Error("Couldn't read the file."));
    xhr.open("POST", url); xhr.send(fd);
  });
}

// ---------------- Your voices ----------------
const VOICES = (() => {
  let list = [], busy = false;
  const ready = (id) => { const v = list.find((x) => x.id === id); return !!(v && v.ready); };
  const anyReady = () => list.some((v) => v.ready);
  async function refresh() {
    try { list = (await api("/api/voices")).voices; } catch { /* keep the last list */ }
    renderNav();
    return list;
  }
  function label(v) {
    if (v.kind === "profile") return v.ready ? `My voice (${v.source === "upload" ? "from a recording" : `${v.takes} takes`})` : "My voice (not set up yet)";
    return `${v.name} (${Math.round(v.seconds)} s clip)`;
  }
  function fillSelect(sel, want) {
    const cur = sel.value || want;
    sel.innerHTML = list.map((v) => `<option value="${esc(v.id)}"${v.ready ? "" : " disabled"}>${esc(label(v))}</option>`).join("");
    const pick = [cur, want].find((id) => ready(id)) || (list.find((v) => v.ready) || list[0] || {}).id;
    if (pick) sel.value = pick;
  }
  function renderNav() {
    const ul = $("#voiceList"); if (!ul) return;
    const st = S.setup && S.setup.settings, ok = !!(st && st.consent && st.watermark_ack);
    const me = list.find((v) => v.id === "me") || { ready: false, takes: 0 };
    const meMeta = me.ready ? (me.source === "upload" ? "recording" : `${me.takes} takes`) : "set up";
    let html = `<li><div class="voice-row"><button class="nav-item" data-me><svg class="ic"><use href="#i-mic"/></svg>
      <span class="vname">My voice</span><span class="vmeta">${esc(meMeta)}</span></button></div>
      <button class="tab nav-item sub" data-screen="record">Record takes</button>
      <button class="tab nav-item sub" data-screen="profile">Choose takes</button></li>`;
    for (const v of list.filter((x) => x.kind === "clip")) {
      html += `<li><div class="voice-row"><button class="nav-item" data-play="${esc(v.id)}" title="Play ${esc(v.name)}"><svg class="ic"><use href="#i-person"/></svg>
        <span class="vname">${esc(v.name)}</span><span class="vmeta">${Math.round(v.seconds)} s</span></button>
        <button class="vdel" data-del="${esc(v.id)}" aria-label="Remove ${esc(v.name)}" title="Remove">${icon("x")}</button></div></li>`;
    }
    if (busy) html += `<li class="voice-busy"><span class="spinner" aria-hidden="true"></span> Adding the clip…</li>`;
    ul.innerHTML = html;
    // the sub-items behave like the other screen buttons (show + current-page highlight)
    const up = st && st.voice_source === "upload";
    for (const b of $$(".nav-item.sub", ul)) {
      b.hidden = up || !(st && st.voice_source);
      b.disabled = !ok || (b.dataset.screen === "record" ? st.voice_source !== "record" : !(S.profile && S.profile.takes.length));
      if (S.screen === b.dataset.screen) b.setAttribute("aria-current", "page");
      b.onclick = () => show(b.dataset.screen);
    }
    const meBtn = $("[data-me]", ul);
    meBtn.onclick = () => show("myvoice");
    meBtn.dataset.screen = "myvoice";
    if (S.screen === "myvoice") meBtn.setAttribute("aria-current", "page");
    $$("[data-play]", ul).forEach((b) => (b.onclick = () => play(b.dataset.play)));
    $$("[data-del]", ul).forEach((b) => (b.onclick = () => remove(b.dataset.del)));
    $("#addVoiceBtn").hidden = !ok;
  }
  const player = new Audio();
  function play(id) {
    if (!player.paused && player.dataset.id === id) { player.pause(); return; }
    $$("audio").forEach((a) => a.pause());
    player.dataset.id = id; player.src = `/api/voices/${id}/audio?t=${Date.now()}`; player.play().catch(() => {});
  }
  async function remove(id) {
    const v = list.find((x) => x.id === id);
    if (!v || !confirm(`Remove "${v.name}" from Your voices? The clip is deleted from this Mac.`)) return;
    await api(`/api/voices/${id}`, { method: "DELETE" });
    await refresh(); changed();
    toast(`Removed ${v.name}`);
  }
  async function add(file) {
    if (!file) return null;
    const name = (prompt("Name this voice", file.name.replace(/\.[^.]+$/, "").slice(0, 40)) || "").trim();
    if (!name) return null;
    busy = true; renderNav();
    try {
      const v = await uploadFile("/api/voices", file, { name });
      busy = false; await refresh(); changed();
      const warn = (v.notes || []).filter((n) => n.level !== "info").map((n) => n.text).join(" ");
      toast(warn ? `Added ${v.name}. ${warn}` : `Added ${v.name} (${Math.round(v.seconds)} s)`);
      return v;
    } catch (e) { busy = false; renderNav(); toast("Couldn't add the voice: " + e.message); return null; }
  }
  function changed() {  // keep both pickers in sync with the library
    if ($("#ttsVoice")) { VOICES.fillSelect($("#ttsVoice"), $("#ttsVoice").value || "me"); if (typeof textStats === "function") textStats(); }
    if (window.STS) STS.voicesChanged();
  }
  $("#addVoiceFile").onchange = async (e) => { const f = e.target.files[0]; e.target.value = ""; await add(f); };
  return { refresh, fillSelect, renderNav, ready, anyReady, add, list: () => list, label };
})();
window.VOICES = VOICES;

// ---------------- Speech to speech ----------------
const STS = (() => {
  const T = { src: null, result: null, job: null, ab: "new", exported: null, status: null };
  const mode = () => $('input[name="stsMode"]:checked').value;
  const voice = () => $("#stsVoice").value;

  async function enter() {
    await VOICES.refresh();
    const st = S.setup ? S.setup.settings : {};
    VOICES.fillSelect($("#stsVoice"), st.sts_voice || "me");
    const m = $(`input[name="stsMode"][value="${st.sts_mode || "timing"}"]`); if (m && !T.job) m.checked = true;
    await refreshStatus(true);
  }
  async function refreshStatus(first) {
    const s = await api(`/api/sts/status?voice=${encodeURIComponent(voice() || "me")}`); T.status = s;
    if (!s.mp3) { $("#stsMp3").disabled = true; $("#stsMp3").parentElement.title = "MP3 needs ffmpeg with libmp3lame"; }
    if (first) {
      if (s.source && (!T.src || T.src.version !== s.source.version)) showSource(s.source);
      if (!T.result && s.last && s.source && s.last.source_version === s.source.version) showResult(s.last, false);
      if (s.job && !T.job) { T.job = { id: s.job.id, kind: s.job.kind, mode: s.job.mode }; poll(); }
    }
    const styleOpt = $('input[name="stsMode"][value="style"]');
    styleOpt.disabled = !s.style_ready;
    styleOpt.closest(".source-opt").title = s.style_ready ? "" : "Needs the transcription model (Setup)";
    if (!s.style_ready && styleOpt.checked) $('input[name="stsMode"][value="timing"]').checked = true;
    voiceMeta(); buttons();
  }
  function voiceMeta() {
    const v = VOICES.list().find((x) => x.id === voice());
    $("#stsVoiceMeta").textContent = !v ? "Add a voice clip in the sidebar, or record My voice." : v.kind === "profile"
      ? (v.ready ? "Your recorded voice (its voice take)" : "My voice isn't set up yet") : `${Math.round(v.seconds)} s clip${v.verdict && v.verdict !== "good" ? ` · ${v.verdict === "okay" ? "usable" : "poor"} quality` : ""}`;
    $("#stsVoicePlay").disabled = !v || !v.ready;
  }
  function modeNote() {
    const n = $("#stsModeNote");
    n.hidden = mode() !== "style";
    n.textContent = "Speak like the voice transcribes the recording on this Mac, then speaks it again. The words come from the transcript, so check names and unusual words in the result. Measured: it follows a faster voice fully, a slower voice only partly.";
  }
  function buttons() {
    const s = T.status || {};
    const ready = !!(T.src && voice() && VOICES.ready(voice()) && !T.job);
    $("#stsPreviewBtn").disabled = $("#stsConvertBtn").disabled = !ready;
    const est = s.estimates && s.estimates[mode()];
    $("#stsEst").textContent = T.job ? "" : !T.src ? "Add a recording to start." : !VOICES.ready(voice()) ? "Pick a voice that's ready." : est ? `The whole recording takes about ${fmtDur(est)}.` : "";
    $("#stsSrcFile").disabled = !!T.job;
    modeNote();
  }

  // recording
  function showSource(m) {
    T.src = m;
    $("#stsSrcInfo").hidden = !m; $("#stsSrcBtnText").textContent = m ? "Choose another…" : "Choose recording…";
    if (!m) return;
    $("#stsSrcName").textContent = m.name;
    $("#stsSrcFacts").innerHTML = [`${fmtClock(m.seconds)} long`, `${fmtClock(m.speech_seconds)} of speech`].map((t) => `<span class="fact">${esc(t)}</span>`).join("");
    $("#stsSrcNotes").innerHTML = (m.notes || []).map((n) => `<li class="${n.level === "warn" ? "warn" : ""}">${esc(n.text)}</li>`).join("");
    const a = $("#stsASrc"); a.src = `/api/sts/source/audio?v=${m.version}`; a.load();
    $("#stsAOrig").src = a.src; $("#stsAOrig").load();
    const base = m.name.replace(/\.[^.]+$/, "").replace(/[^A-Za-z0-9 _\-.]/g, "").trim() || "recording";
    $("#stsFname").value = `${base} new voice`.slice(0, 80);
    requestAnimationFrame(draw);
  }
  async function pickSource(file) {
    if (!file || T.job) return;
    $("#stsSrcErr").hidden = true; $("#stsSrcBusy").hidden = false;
    const lbl = $("#stsSrcBusy").lastChild, base = " Reading the file…";
    try {
      const m = await uploadFile("/api/sts/source", file, {}, (f) => { if (file.size > 20e6) lbl.textContent = ` Loading ${Math.round(f * 100)}%…`; if (f >= 1) lbl.textContent = base; });
      showSource(m); await refreshStatus(false);
    } catch (e) { $("#stsSrcErr").textContent = e.message; $("#stsSrcErr").hidden = false; }
    $("#stsSrcBusy").hidden = true; lbl.textContent = base;
    buttons();
  }
  $("#stsSrcFile").onchange = (e) => { pickSource(e.target.files[0]); e.target.value = ""; };
  const card = $("#stsSrcCard");
  card.addEventListener("dragover", (e) => { e.preventDefault(); card.classList.add("over"); });
  card.addEventListener("dragleave", (e) => { if (!card.contains(e.relatedTarget)) card.classList.remove("over"); });
  card.addEventListener("drop", (e) => { e.preventDefault(); card.classList.remove("over"); pickSource(e.dataTransfer.files[0]); });

  // voice + mode
  $("#stsVoice").onchange = () => { refreshStatus(false); };
  $$('input[name="stsMode"]').forEach((r) => (r.onchange = buttons));
  const vplay = $("#stsAVoice");
  $("#stsVoicePlay").onclick = () => {
    if (!vplay.paused) { vplay.pause(); return; }
    $$("audio").forEach((a) => a.pause());
    vplay.src = `/api/voices/${voice()}/audio?t=${Date.now()}`; vplay.play().catch(() => {});
  };
  vplay.onplay = () => { $("#stsVoicePlay").innerHTML = icon("pause"); };
  vplay.onpause = vplay.onended = () => { $("#stsVoicePlay").innerHTML = icon("play"); };

  // source mini player
  const sa = $("#stsASrc");
  $("#stsSrcPlay").onclick = () => { $$("audio").forEach((a) => { if (a !== sa) a.pause(); }); sa.paused ? sa.play() : sa.pause(); };
  sa.onplay = () => { $("#stsSrcPlay").innerHTML = icon("pause"); (function loop() { draw(); if (!sa.paused) requestAnimationFrame(loop); })(); };
  sa.onpause = sa.onended = () => { $("#stsSrcPlay").innerHTML = icon("play"); draw(); };
  $("#stsSrcWave").onclick = (e) => { if (!sa.duration) return; const r = e.target.getBoundingClientRect(); sa.currentTime = ((e.clientX - r.left) / r.width) * sa.duration; draw(); };

  // convert
  async function start(preview) {
    $("#stsErr").hidden = true;
    try {
      const j = await api("/api/sts/convert", { method: "POST", body: form({ preview, mode: mode(), voice: voice() }) });
      T.job = { id: j.id, kind: preview ? "preview" : "full", mode: mode() };
      if (S.setup) { S.setup.settings.sts_voice = voice(); S.setup.settings.sts_mode = mode(); }
      poll();
    } catch (e) { $("#stsErr").textContent = e.message; $("#stsErr").hidden = false; }
    buttons();
  }
  $("#stsPreviewBtn").onclick = () => start(true);
  $("#stsConvertBtn").onclick = () => start(false);
  $("#stsCancel").onclick = async () => { if (T.job) { await api(`/api/sts/jobs/${T.job.id}/cancel`, { method: "POST" }); $("#stsProgText").textContent = "Stopping after the current part…"; } };
  function setProg(f, txt) {
    $("#stsProgFill").style.width = `calc(${Math.max(0.01, Math.min(1, f)) * 100}% - 6px)`;
    $("#stsProg").setAttribute("aria-valuenow", Math.round(f * 100));
    $("#stsProgText").textContent = txt;
  }
  async function poll() {
    $("#stsProgBox").hidden = false; buttons();
    const what = T.job.kind === "preview" ? "Preview" : T.job.mode === "style" ? "Re-speaking" : "Converting";
    setProg(0, `${what}: starting…`);
    while (T.job) {
      let j; try { j = await api(`/api/sts/jobs/${T.job.id}`); } catch { break; }
      const p = j.progress || {};
      if (j.state === "running" || j.state === "queued") {
        if (p.stage === "loading") setProg(0.01, `${what}: loading the voice model…`);
        else if (p.stage === "transcribing") setProg(0.02, `${what}: transcribing the recording…`);
        else if (p.total) setProg(p.fraction || 0, `${what}: ${p.stage === "speaking" ? "sentence" : "part"} ${Math.min(p.done + 1, p.total)} of ${p.total} · ${Math.round((p.fraction || 0) * 100)}%` + (p.done ? ` · about ${fmtDur(p.eta)} left` : ""));
        await new Promise((r) => setTimeout(r, 700));
        continue;
      }
      T.job = null;
      if (j.state === "done") { showResult(j.result, true); toast(j.result.kind === "preview" ? "Preview ready" : "Done. The whole recording has the new voice."); }
      else if (j.state === "cancelled") toast("Cancelled");
      else { $("#stsErr").textContent = "Conversion failed: " + (j.error || "unknown error"); $("#stsErr").hidden = false; }
    }
    T.job = null; $("#stsProgBox").hidden = true; buttons();
  }

  // result + A/B
  const resA = () => (T.ab === "new" ? $("#stsANew") : $("#stsAOrig"));
  // timeline mapping: keep-timing results line up 1:1; re-spoken results map by proportion
  const toOrig = (t) => T.result.offset + (T.result.mode === "timing" ? t : t * (T.result.original_seconds / T.result.seconds));
  const fromOrig = (t) => (T.result.mode === "timing" ? t - T.result.offset : (t - T.result.offset) * (T.result.seconds / T.result.original_seconds));
  const resPos = () => { if (!T.result) return 0; const a = resA(); return T.ab === "new" ? a.currentTime : fromOrig(a.currentTime); };
  function showResult(r, scroll) {
    T.result = r; T.exported = null; T.ab = "new";
    $("#stsResCard").hidden = false;
    $("#stsResTitle").textContent = (r.kind === "preview" ? `Preview (${fmtClock(r.offset)} to ${fmtClock(r.offset + r.original_seconds)} of the original)` : "Whole recording")
      + (r.mode === "style" ? " · spoken like the voice" : " · original timing");
    const a = $("#stsANew"); a.src = `/api/sts/outputs/${r.id}`; a.load();
    const sim = r.similarity == null ? null : Math.round(r.similarity * 100);
    $("#stsSimFill").style.width = sim == null ? "0" : `calc(${sim}% - 6px)`;
    $("#stsSimBar").setAttribute("aria-valuenow", sim || 0);
    $("#stsSimText").textContent = sim == null ? "–" : `${sim}%`;
    let meta = `${fmtClock(r.seconds)} of audio (original ${fmtClock(r.original_seconds)}), made in ${fmtDur(r.compute_seconds)}. Voice: ${r.voice_name}.`;
    if (r.mode === "style" && r.pace_reference) meta += ` Speaking pace: original ${r.pace_original}, voice ${r.pace_reference}, result ${r.pace_result} characters per second.`;
    $("#stsResMeta").textContent = meta;
    const low = r.low_pieces || [];
    $("#stsResNotes").innerHTML = low.length ? `<li class="warn">${low.length} part${low.length > 1 ? "s" : ""} matched the voice less well (often music, noise or overlapping voices): `
      + low.slice(0, 8).map((p) => `<button data-t="${p.start}">${fmtClock(p.start + r.offset)}</button>`).join(", ") + "</li>" : "";
    $$("#stsResNotes [data-t]").forEach((b) => (b.onclick = () => seek(+b.dataset.t)));
    $("#stsAbHint").textContent = r.mode === "timing" ? "Switch while it plays to compare the same moment." : "Switch while it plays; the original jumps to about the same place (the lengths differ).";
    $('input[name="stsAb"][value="new"]').checked = true;
    $("#stsExportMsg").textContent = ""; $("#stsRevealBtn").disabled = true;
    updTime(); requestAnimationFrame(draw);
    if (scroll) $("#stsResCard").scrollIntoView({ behavior: "smooth", block: "start" });
  }
  function seek(t) { const a = resA(); a.currentTime = T.ab === "new" ? t : toOrig(t); updTime(); draw(); }
  function updTime() { if (T.result) $("#stsResTime").textContent = `${fmtClock(resPos())} / ${fmtClock(T.result.seconds)}`; }
  $("#stsResPlay").onclick = () => {
    const a = resA(); $$("audio").forEach((x) => { if (x !== a) x.pause(); });
    if (a.paused) { if (resPos() >= T.result.seconds - 0.05) seek(0); a.play(); } else a.pause();
  };
  for (const id of ["#stsANew", "#stsAOrig"]) {
    const a = $(id);
    a.onplay = () => { $("#stsResPlay").innerHTML = icon("pause"); $("#stsResPlay").setAttribute("aria-label", "Pause result");
      (function loop() { if (a !== resA()) return; if (T.ab === "orig" && resPos() >= T.result.seconds) a.pause(); updTime(); draw(); if (!a.paused) requestAnimationFrame(loop); })(); };
    a.onpause = a.onended = () => { if (a === resA()) { $("#stsResPlay").innerHTML = icon("play"); $("#stsResPlay").setAttribute("aria-label", "Play result"); } draw(); };
  }
  $$('input[name="stsAb"]').forEach((r) => (r.onchange = () => {
    const from = resA(), t = resPos(), playing = !from.paused;
    from.pause(); T.ab = r.value; seek(t); if (playing) resA().play();
  }));
  $("#stsResWave").onclick = (e) => { if (!T.result) return; const r = e.target.getBoundingClientRect(); seek(((e.clientX - r.left) / r.width) * T.result.seconds); };

  function draw() {
    if (T.src) drawPeaks($("#stsSrcWave"), T.src.peaks, sa.duration ? sa.currentTime / sa.duration : 0);
    if (T.result) drawPeaks($("#stsResWave"), T.result.peaks, resPos() / T.result.seconds);
  }
  window.addEventListener("resize", draw);

  // export
  $$('input[name="stsFmt"]').forEach((r) => (r.onchange = () => {
    const mp3 = $('input[name="stsFmt"]:checked').value === "mp3";
    $("#stsBrBox").hidden = !mp3;
    $("#stsFmtHint").textContent = mp3 ? "Stored at 48 kHz so all bitrates are possible. Much smaller files." : "24 kHz, 16-bit, lossless.";
  }));
  const native = () => (window.pywebview && window.pywebview.api && window.pywebview.api.save_sts_export ? window.pywebview.api : null);
  const nameFor = (fmt) => { const n = $("#stsFname").value.trim().replace(/[^A-Za-z0-9 _\-.]/g, "").replace(/^[ .]+|[ .]+$/g, "") || "new-voice"; return n.toLowerCase().endsWith("." + fmt) ? n : `${n}.${fmt}`; };
  const size = (b) => (b > 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.round(b / 1024)} KB`);
  $("#stsExportBtn").onclick = async () => {
    if (!T.result) return;
    const fmt = $('input[name="stsFmt"]:checked').value, bitrate = $("#stsBitrate").value, name = nameFor(fmt), msg = $("#stsExportMsg");
    T.exported = null; $("#stsRevealBtn").disabled = true;
    try {
      const d = native();
      if (d) {
        msg.textContent = "Saving…";
        const r = await d.save_sts_export(T.result.id, fmt, +bitrate, name);
        if (r.cancelled) { msg.textContent = "Not saved."; return; }
        if (r.error) throw new Error(r.error);
        T.exported = r; $("#stsRevealBtn").disabled = false;
        msg.textContent = `Saved ${r.name} (${size(r.bytes)}) in ${r.folder}.`;
        return;
      }
      msg.textContent = "Preparing…";
      const res = await fetch("/api/sts/export", { method: "POST", body: form({ output_id: T.result.id, fmt, bitrate, filename: name }) });
      if (!res.ok) { let m = res.statusText; try { m = (await res.json()).detail || m; } catch {} throw new Error(m); }
      const blob = await res.blob();
      const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = name;
      document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 4000);
      msg.textContent = `Downloaded ${name} (${size(blob.size)}).`;
    } catch (e) { msg.textContent = "Export failed: " + e.message; }
  };
  $("#stsRevealBtn").onclick = () => T.exported && native() && native().reveal(T.exported.path);
  const showNative = () => { if (native()) $("#stsRevealBtn").hidden = false; };
  window.addEventListener("pywebviewready", showNative); showNative();

  function voicesChanged() { VOICES.fillSelect($("#stsVoice"), voice() || "me"); voiceMeta(); buttons(); }
  return { enter, voicesChanged, draw };
})();
window.STS = STS;
