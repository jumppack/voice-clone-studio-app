// Renders the "warming up" overlay to a video, frame by frame (deterministic, no dropped frames).
//
//   node tests/record_boot_video.mjs <port> [light|dark] [out.mp4]
//
// Needs a running server (any data folder) and Google Chrome. Opens /?boot=demo, pauses every CSS
// animation, then steps them 1/FPS s at a time and screenshots each frame; ffmpeg encodes H.264.
// Two full 8 s loops, then the overlay fades out over the app.
import { spawn, spawnSync } from "node:child_process";
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const ROOT = resolve(import.meta.dirname, "..");
const PORT = process.argv[2] || "7861";
const THEME = process.argv[3] || "light";
const OUT = resolve(process.argv[4] || `${ROOT}/docs/boot_overlay_${THEME}.mp4`);
const FPS = 60, LOOP_S = 8, LOOPS = 2, FADE_S = 0.6, HOLD_S = 0.8;
const W = 1280, H = 720, DPR = 1.5;  // -> 1920x1080
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const DBG = 9333;
const FRAMES = `${ROOT}/tests/results/boot_frames_${THEME}`;
rmSync(FRAMES, { recursive: true, force: true }); mkdirSync(FRAMES, { recursive: true });

const chrome = spawn(CHROME, ["--headless=new", `--remote-debugging-port=${DBG}`, `--user-data-dir=${ROOT}/data_ui/chrome-video`,
  "--hide-scrollbars", "--force-color-profile=srgb", "--no-first-run", "about:blank"], { stdio: "ignore" });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let ws;
for (let i = 0; i < 50 && !ws; i++) {
  try {
    const list = await (await fetch(`http://127.0.0.1:${DBG}/json`)).json();
    const page = list.find((t) => t.type === "page");
    if (page) ws = new WebSocket(page.webSocketDebuggerUrl);
  } catch { await sleep(200); }
}
await new Promise((r) => (ws.onopen = r));
let id = 0; const pending = new Map();
ws.onmessage = (m) => { const d = JSON.parse(m.data); if (d.id && pending.has(d.id)) { pending.get(d.id)(d); pending.delete(d.id); } };
const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const ev = async (expr) => (await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true })).result?.result?.value;

await send("Emulation.setDeviceMetricsOverride", { width: W, height: H, deviceScaleFactor: DPR, mobile: false });
await send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: THEME }, { name: "prefers-reduced-motion", value: "no-preference" }] });
await send("Page.enable");
await send("Page.navigate", { url: `http://127.0.0.1:${PORT}/?boot=demo&screen=studio` });
for (let i = 0; i < 100; i++) { if (await ev(`!!document.querySelector('#boot') && !document.querySelector('#boot').hidden && document.fonts.status === 'loaded'`)) break; await sleep(100); }
await ev(`(async () => { try { localStorage.removeItem('vcs-theme'); } catch {} applyTheme(null); await new Promise(r => setTimeout(r, 900)); return 1; })()`);
// freeze everything; each frame sets every animation to the same time, as if the page had just loaded
await ev(`(() => { window.__t0 = document.timeline.currentTime; for (const a of document.getAnimations()) a.pause(); return document.getAnimations().length; })()`);

let n = 0;
async function frame() {
  const r = await send("Page.captureScreenshot", { format: "png", fromSurface: true, captureBeyondViewport: false });
  writeFileSync(`${FRAMES}/f${String(n++).padStart(5, "0")}.png`, Buffer.from(r.result.data, "base64"));
}
const loopFrames = LOOP_S * LOOPS * FPS;
for (let f = 0; f < loopFrames; f++) {
  const t = (f / FPS) * 1000;
  await ev(`(() => { for (const a of document.getAnimations()) a.currentTime = ${t}; return 1; })()`);
  await frame();
  if (f % 120 === 0) process.stdout.write(`frame ${f}/${loopFrames}\n`);
}
// fade the overlay out over the app (the CSS transition, stepped the same way)
await ev(`(() => { const b = document.querySelector('#boot'); b.hidden = true; return 1; })()`);
await sleep(50);
const fadeAnims = await ev(`(() => { const t = document.getAnimations().filter(a => a instanceof CSSTransition); for (const a of t) { a.pause(); a.currentTime = 0; } window.__fade = t; return t.length; })()`);
const end = Math.round((FADE_S + HOLD_S) * FPS);
for (let f = 0; f < end; f++) {
  const t = (f / FPS) * 1000;
  await ev(`(() => { for (const a of window.__fade) a.currentTime = ${t}; for (const a of document.getAnimations()) if (!(a instanceof CSSTransition)) a.currentTime = ${loopFrames / FPS * 1000 + t}; return 1; })()`);
  await frame();
}
ws.close(); chrome.kill();
console.log(`${n} frames (${fadeAnims} fade transitions); encoding…`);
const enc = spawnSync("ffmpeg", ["-hide_banner", "-loglevel", "error", "-y", "-framerate", String(FPS), "-i", `${FRAMES}/f%05d.png`,
  "-vf", "scale=1920:1080:flags=lanczos,format=yuv420p", "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-profile:v", "high",
  "-movflags", "+faststart", "-tag:v", "avc1", OUT], { stdio: "inherit" });
if (enc.status !== 0) process.exit(1);
rmSync(FRAMES, { recursive: true, force: true });
console.log("wrote", OUT);
