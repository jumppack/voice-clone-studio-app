// Headless-Chrome UI test with a FAKE microphone (Chrome's --use-file-for-fake-audio-capture).
// Drives the real app on 127.0.0.1 through the DevTools protocol; saves screenshots.
// Usage: node tests/ui_cdp.mjs <port> <phase>   phase: record | studio
import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const PORT = process.argv[2] || "7862";
const PHASE = process.argv[3] || "record";
const BASE = `http://127.0.0.1:${PORT}`;
const ROOT = resolve(import.meta.dirname, "..");
const SHOTS = `${ROOT}/tests/screenshots`;
mkdirSync(SHOTS, { recursive: true });
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const DBG = 9333 + ({ studio: 1, upload: 2 }[PHASE] || 0);
const chrome = spawn(CHROME, [
  ...(process.env.HEADFUL ? ["--window-position=-3000,-3000"] : ["--headless=new"]), `--remote-debugging-port=${DBG}`, `--user-data-dir=${ROOT}/data_ui/chrome-${PHASE}`,
  "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
  ...(process.env.NOFILE ? [] : [`--use-file-for-fake-audio-capture=${ROOT}/tests/fixtures/fake_mic.wav`]), "--autoplay-policy=no-user-gesture-required",
  "--no-first-run", "--no-default-browser-check", "--disable-extensions", "--disable-background-networking",
  "--disable-component-update", "--disable-sync", ...(process.env.CHROME_EXTRA ? process.env.CHROME_EXTRA.split(" ") : []), "--metrics-recording-only", "--window-size=1100,900", "about:blank",
], { stdio: "ignore" });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let ws, id = 0; const pending = new Map();
async function connect() {
  for (let i = 0; i < 50; i++) {
    try { const l = await (await fetch(`http://127.0.0.1:${DBG}/json/list`)).json(); const p = l.find((t) => t.type === "page"); if (p) { ws = new WebSocket(p.webSocketDebuggerUrl); break; } } catch {}
    await sleep(200);
  }
  await new Promise((r) => (ws.onopen = r));
  ws.onmessage = (m) => { const d = JSON.parse(m.data); if (d.id && pending.has(d.id)) { pending.get(d.id)(d); pending.delete(d.id); } };
}
const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
async function ev(expr) {
  const r = await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true });
  if (r.result && r.result.exceptionDetails) throw new Error(JSON.stringify(r.result.exceptionDetails).slice(0, 400));
  return r.result && r.result.result && r.result.result.value;
}
async function size(w, h = 900) { await send("Emulation.setDeviceMetricsOverride", { width: w, height: h, deviceScaleFactor: 1, mobile: false }); await sleep(300); }
async function theme(dark) { THEME = dark ? "dark" : "light"; await ev(`document.documentElement.dataset.theme=${dark ? "'dark'" : "'light'"}; applyTheme(${dark ? "'dark'" : "'light'"}); true`); await sleep(250); }
async function shot(name, full = true) {
  const m = await send("Page.getLayoutMetrics");
  const h = Math.ceil(m.result.cssContentSize.height);
  const r = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: full, clip: full ? { x: 0, y: 0, width: m.result.cssLayoutViewport.clientWidth, height: Math.min(h, 3000), scale: 1 } : undefined });
  writeFileSync(`${SHOTS}/${name}.png`, Buffer.from(r.result.data, "base64")); console.log("shot", name);
}
async function waitFor(expr, ms = 60000) { const t = Date.now(); while (Date.now() - t < ms) { if (await ev(expr)) return true; await sleep(250); } throw new Error("timeout: " + expr); }
let THEME = "light";
async function nav(url) { await send("Page.navigate", { url }); await sleep(1200); await ev(`applyTheme("${THEME}"); true`).catch(() => {}); }
const out = {};

await connect();
await send("Page.enable"); await send("Runtime.enable");
await send("Browser.grantPermissions", { origin: BASE, permissions: ["audioCapture"] }).catch(() => {});

// Chrome's own fake capture device delivers digital silence on this machine (checked with a plain
// AnalyserNode, headless and headful). So for audio content we use a SHIM: it still calls the real
// getUserMedia with the app's exact constraints (so getSettings() is the browser's real answer), then
// returns a stream that plays tests/fixtures/fake_mic.wav through WebAudio instead of the device audio.
if (!process.env.NO_SHIM) {
  const { readFileSync } = await import("node:fs");
  const b64 = readFileSync(`${ROOT}/tests/fixtures/fake_mic.wav`).toString("base64");
  await send("Page.addScriptToEvaluateOnNewDocument", { source: `(() => {
    const B64 = "${b64}";
    const real = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    navigator.mediaDevices.getUserMedia = async (c) => {
      window.__requested = JSON.parse(JSON.stringify(c));
      const rs = await real(c);
      const realTrack = rs.getAudioTracks()[0];
      window.__realSettings = realTrack.getSettings();
      const ctx = new AudioContext();
      const bytes = Uint8Array.from(atob(B64), (ch) => ch.charCodeAt(0));
      const buf = await ctx.decodeAudioData(bytes.buffer);
      const src = ctx.createBufferSource(); src.buffer = buf; src.loop = true;
      const dst = ctx.createMediaStreamDestination(); src.connect(dst); src.start();
      const t = dst.stream.getAudioTracks()[0];
      t.getSettings = () => realTrack.getSettings();
      window.__shim = true;
      return dst.stream;
    };
  })();` });
}

if (PHASE === "record") {
  await size(1100);
  await nav(BASE + "/");
  await waitFor("!!document.querySelector('.check-item')");
  await theme(false); await shot("01_setup_1100_light");
  await theme(true); await shot("02_setup_1100_dark");
  await theme(false);
  // consent + watermark through real clicks on the labels
  await ev(`document.querySelector('#consent').click(); document.querySelector('#wmAck').click(); true`);
  await waitFor(`S.setup.settings.consent && S.setup.settings.watermark_ack`, 5000);
  out.toRecordEnabled = await ev(`!document.querySelector('#toRecord').disabled`);
  // reload directly on the record screen so the fake mic file starts at its silent lead-in
  await nav(BASE + "/?screen=record");
  await waitFor("typeof MIC!=='undefined' && !!MIC.settings");
  out.trackSettings = await ev(`JSON.stringify(MIC.settings)`);
  out.requested = await ev(`JSON.stringify(CONSTRAINTS().audio)`);
  out.sampleRate = await ev(`MIC.ctx.sampleRate`);
  await ev(`document.querySelector('#roomBtn').click(); true`);
  await waitFor(`REC.roomFloor!==null`, 10000);
  out.room = await ev(`REC.roomText`);
  for (let take = 1; take <= 3; take++) {
    // Space key starts recording (focus is on body)
    await ev(`document.activeElement.blur(); true`);
    await send("Input.dispatchKeyEvent", { type: "keyDown", code: "Space", key: " ", windowsVirtualKeyCode: 32 });
    await send("Input.dispatchKeyEvent", { type: "keyUp", code: "Space", key: " ", windowsVirtualKeyCode: 32 });
    await waitFor(`REC.state==='recording'`, 6000);
    if (take === 1) { await sleep(5000); await shot("03_record_recording_1100_light", false); }
    await waitFor(`REC.state==='idle'`, 30000);
    out[`take${take}`] = await ev(`JSON.stringify((S.profile.takes.find(t=>t.step===${take})||{}).analysis)`);
    if (take === 1) { await shot("04_record_verdict_1100_light"); await size(720); await shot("05_record_verdict_720_light"); await theme(true); await shot("06_record_verdict_720_dark"); await theme(false); await size(1100); }
    await ev(`document.querySelector('#nextStep').click(); true`); await sleep(400);
  }
  out.profileReady = await ev(`S.profile.ready`);
  await ev(`show('profile'); true`); await sleep(800);
  await shot("07_profile_1100_light"); await theme(true); await shot("08_profile_1100_dark"); await theme(false);
  // keyboard focus ring visible
  await send("Input.dispatchKeyEvent", { type: "keyDown", code: "Tab", key: "Tab", windowsVirtualKeyCode: 9 });
  await send("Input.dispatchKeyEvent", { type: "keyUp", code: "Tab", key: "Tab", windowsVirtualKeyCode: 9 });
  await sleep(200);
  out.focused = await ev(`document.activeElement.outerHTML.slice(0,80)`);
}

if (PHASE === "debug") {
  await nav(BASE + "/?screen=record");
  await waitFor("typeof MIC!=='undefined' && !!MIC.settings");
  out.diag = await ev(`(async()=>{const t=MIC.stream.getAudioTracks()[0];const c=new AudioContext();await c.resume();const s=c.createMediaStreamSource(MIC.stream);const an=c.createAnalyser();s.connect(an);an.connect(c.destination);await new Promise(r=>setTimeout(r,1500));const b=new Float32Array(2048);an.getFloatTimeDomainData(b);let m=0;for(const v of b)m=Math.max(m,Math.abs(v));return JSON.stringify({ready:t.readyState,muted:t.muted,enabled:t.enabled,ctx:MIC.ctx.state,ctx2:c.state,max:m,label:t.label})})()`);
  out.levels = await ev(`new Promise(r=>{const a=[];const t=setInterval(()=>a.push(Math.round(MIC.level)),250);setTimeout(()=>{clearInterval(t);r(a.join(','))},6000)})`);
}

if (PHASE === "studio") {
  await size(1100);
  await nav(BASE + "/?screen=studio");
  await waitFor(`S.screen==='studio' && document.querySelectorAll('#sliders input').length===5`);
  await ev(`document.querySelector('[data-preset="narrator"]').click(); true`);
  await ev(`document.querySelector('#seedOn').click(); true`);
  // Length control: whole text by default; "Set length" shows the slider and the fit hint
  out.lenDefault = await ev(`JSON.stringify({sliderHidden: document.querySelector('#lenCtrl').hidden, help: document.querySelector('#lenHelp').textContent, preview: document.querySelector('#previewLbl').textContent})`);
  await ev(`document.querySelector('input[name=lenMode][value=target]').click(); const r=document.querySelector('#lenTarget'); r.value='15'; r.dispatchEvent(new Event('input')); true`);
  out.lenTarget15 = await ev(`JSON.stringify({sliderHidden: document.querySelector('#lenCtrl').hidden, val: document.querySelector('#lenVal').textContent, help: document.querySelector('#lenHelp').textContent, preview: document.querySelector('#previewLbl').textContent})`);
  await ev(`document.querySelector('#previewBtn').click(); true`);
  await sleep(1500); await shot("09_studio_progress_1100_light", false);
  await waitFor(`!!S.lastOut`, 180000);
  out.preview = await ev(`JSON.stringify({target:S.lastOut.result.target, sec:S.lastOut.result.seconds, sim:S.lastOut.result.similarity, el:S.lastOut.elapsed})`);
  // Browser save path: stub the Save picker (headless has no dialog) and check what gets written
  await ev(`window.__saved=null; window.showSaveFilePicker = async (o) => ({ name: o.suggestedName, createWritable: async () => { const parts=[]; return { write: async (b) => parts.push(b), close: async () => { window.__saved = { name: o.suggestedName, bytes: parts.reduce((n,b)=>n+b.size,0), types: JSON.stringify(o.types) }; } }; } }); true`);
  await ev(`document.querySelector('input[name=fmt][value=mp3]').click(); document.querySelector('#bitrate').value='320'; document.querySelector('#fname').value='ui-export-test'; true`);
  await ev(`document.querySelector('#exportBtn').click(); true`);
  await waitFor(`!!window.__saved`, 30000);
  out.export = await ev(`JSON.stringify({saved: window.__saved, msg: document.querySelector('#exportMsg').textContent, revealHidden: document.querySelector('#revealBtn').hidden})`);
  await shot("10_studio_1100_light"); await theme(true); await shot("11_studio_1100_dark");
  await size(720); await shot("12_studio_720_dark"); await theme(false); await shot("13_studio_720_light");
  out.overflowX720 = await ev(`document.documentElement.scrollWidth > innerWidth`);
}
if (PHASE === "upload") {  // Setup → "Use a recording I have" → Studio (needs data_test/uploads from tests/upload_e2e.py)
  const FILE = process.env.UPLOAD_FILE || `${ROOT}/data_test/uploads/daniel_long.mp3`;
  const state = () => ev(`JSON.stringify({btn: document.querySelector('#toRecord').textContent, disabled: document.querySelector('#toRecord').disabled, hint: document.querySelector('#setupHint').textContent, uploadArea: !document.querySelector('#uploadArea').hidden, tabs: [...document.querySelectorAll('.tab')].filter(t => !t.hidden).map(t => t.innerText.replace(/\\s+/g,' ').trim()), micRow: !!document.querySelector('[data-mic]') || [...document.querySelectorAll('.check-item .name')].some(n => /Microphone/.test(n.textContent))})`);
  await size(1100);
  await nav(BASE + "/?screen=setup");
  await waitFor(`!!S.setup && !!S.profile`);
  await ev(`window.confirm = () => true; true`);
  // record path first
  await ev(`document.querySelector('input[name=voiceSource][value=record]').click(); true`);
  await waitFor(`S.setup.settings.voice_source === 'record'`); await sleep(600);
  out.recordChoice = await state();
  await shot("17_setup_record_choice_1100_light");
  await ev(`document.querySelector('#toRecord').click(); true`);
  await waitFor(`S.screen === 'record'`, 10000);
  out.recordLandsOn = await ev(`S.screen`);
  // upload path
  await nav(BASE + "/?screen=setup");
  await waitFor(`!!S.setup && !!S.profile`);
  await ev(`window.confirm = () => true; document.querySelector('input[name=voiceSource][value=upload]').click(); true`);
  await waitFor(`S.setup.settings.voice_source === 'upload'`); await sleep(600);
  out.uploadChoiceNoFile = await state();
  await send("DOM.enable");
  const { root } = await send("DOM.getDocument", { depth: -1 }).then((r) => r.result || r);
  const { nodeId } = await send("DOM.querySelector", { nodeId: root.nodeId, selector: "#voiceFile" }).then((r) => r.result || r);
  await send("DOM.setFileInputFiles", { nodeId, files: [FILE] });
  await ev(`document.querySelector('#voiceFile').dispatchEvent(new Event('change')); true`);
  out.uploadChoiceWithFile = await state();
  await shot("18_setup_upload_choice_1100_light");
  await ev(`document.querySelector('#toRecord').click(); true`);
  await waitFor(`S.screen === 'studio'`, 180000); await sleep(1200);
  out.uploadLandsOn = await ev(`S.screen`);
  out.studioSourceLine = await ev(`document.querySelector('#voiceSrcLine').hidden ? null : document.querySelector('#voiceSrcLine').textContent`);
  out.tabsInStudio = await ev(`[...document.querySelectorAll('.tab')].filter(t => !t.hidden).map(t => t.innerText.replace(/\\s+/g,' ').trim())`);
  await shot("19_studio_from_upload_1100_light", false);
  // reload: should come straight back to the studio
  await nav(BASE + "/");
  await waitFor(`!!S.screen && !!S.profile`); await sleep(800);
  out.afterReload = await ev(`S.screen`);
  await nav(BASE + "/?screen=setup"); await waitFor(`!!S.setup && !!S.profile`); await sleep(600);
  out.setupWithExistingUpload = await state();
  await size(720); await theme(true); await shot("23_setup_upload_720_dark");
  out.overflowX720 = await ev(`document.documentElement.scrollWidth > innerWidth`);
}
console.log(JSON.stringify(out, null, 1));
writeFileSync(`${ROOT}/tests/results/ui_${PHASE}.json`, JSON.stringify(out, null, 1));
ws.close(); chrome.kill();
process.exit(0);
