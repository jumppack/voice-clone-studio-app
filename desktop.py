#!/usr/bin/env python3
"""Voice Clone Studio as a native macOS window.

Runs the same FastAPI app in-process on 127.0.0.1 (a free port, never exposed) and shows the
UI in a WKWebView window via pywebview. No browser, no terminal.

    .venv/bin/python desktop.py           # run from the project folder
    python make_app.py                    # build "Voice Clone Studio.app" (see that file)

VCS_SELFTEST=1 opens the window, runs in-page capability checks (WebKit APIs the recorder
needs, microphone constraints), writes data/selftest.json and quits.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
# When launched from the .app bundle the interpreter is not the venv's, so add the venv explicitly.
_site = os.environ.get("VCS_SITE_PACKAGES")
if _site:
    import site

    site.addsitedir(_site)

# Local only, same as run.sh
for k, v in {"HF_HUB_DISABLE_TELEMETRY": "1", "GRADIO_ANALYTICS_ENABLED": "False", "DO_NOT_TRACK": "1",
             "TOKENIZERS_PARALLELISM": "false", "PYTORCH_ENABLE_MPS_FALLBACK": "1"}.items():
    os.environ.setdefault(k, v)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")

APP_NAME = "Voice Clone Studio"


def free_port(preferred: int = 7861) -> int:
    for port in (preferred, 0):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("no free port")


def start_server(port: int):
    import uvicorn

    from app.server import app

    cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="on")
    server = uvicorn.Server(cfg)
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(200):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/steps", timeout=1)
            return server
        except Exception:
            time.sleep(0.05)
    raise RuntimeError("server did not start")


def allow_microphone_for(port: int):
    """WKWebView asks its UI delegate before getUserMedia. pywebview doesn't implement that, so we
    add it: grant only for this app's own origin (http://127.0.0.1:<port>), deny everything else.
    macOS still shows its own one-time microphone prompt (NSMicrophoneUsageDescription)."""
    import objc
    from webview.platforms.cocoa import BrowserView

    class BrowserDelegate(objc.Category(BrowserView.BrowserDelegate)):  # PyObjC: category name == class name
        def webView_requestMediaCapturePermissionForOrigin_initiatedByFrame_type_decisionHandler_(
                self, webview, origin, frame, media_type, handler):
            ok = origin.host() == "127.0.0.1" and int(origin.port()) == port and media_type in (1,)  # 1 = microphone
            print("mic permission request:", origin.host(), origin.port(), media_type, "->", "grant" if ok else "deny", flush=True)
            handler(1 if ok else 2)  # WKPermissionDecisionGrant / Deny


class DesktopApi:
    """Exposed to the page as window.pywebview.api (native-only features)."""

    def __init__(self):
        self._window = None  # private: pywebview exposes public attributes to JS
        self._saved: set[str] = set()  # only files saved this session can be revealed

    def save_export(self, output_id: str, fmt: str, bitrate: int, filename: str):
        """Native Save dialog, then write the export straight to the chosen path."""
        import webview

        from app import server

        fmt = fmt if fmt in server.EXPORT_TYPES else "wav"
        name = server.export_name(filename, fmt)
        settings = server.jload(server.SETTINGS, {})
        start = settings.get("export_dir") or str(Path.home() / "Desktop")
        if not Path(start).is_dir():
            start = str(Path.home())
        res = self._window.create_file_dialog(webview.SAVE_DIALOG, directory=start, save_filename=name)
        if not res:
            return {"cancelled": True}
        dst = Path(res if isinstance(res, str) else res[0]).expanduser()
        if dst.suffix.lower() != "." + fmt:
            dst = dst.with_name(dst.name + "." + fmt)
        try:
            server.render_export(output_id, fmt, int(bitrate), dst)
        except Exception as e:
            return {"error": getattr(e, "detail", None) or str(e)}
        server.jupdate(server.SETTINGS, {}, lambda st: st.update(export_dir=str(dst.parent)))
        self._saved.add(str(dst))
        return {"path": str(dst), "name": dst.name, "folder": str(dst.parent), "bytes": dst.stat().st_size}

    def save_sts_export(self, output_id: str, fmt: str, bitrate: int, filename: str):
        """Same as save_export, for a Speech to speech result."""
        import webview

        from app import server, sts

        fmt = fmt if fmt in server.EXPORT_TYPES else "wav"
        name = server.export_name(filename, fmt)
        settings = server.jload(server.SETTINGS, {})
        start = settings.get("export_dir") or str(Path.home() / "Desktop")
        if not Path(start).is_dir():
            start = str(Path.home())
        res = self._window.create_file_dialog(webview.SAVE_DIALOG, directory=start, save_filename=name)
        if not res:
            return {"cancelled": True}
        dst = Path(res if isinstance(res, str) else res[0]).expanduser()
        if dst.suffix.lower() != "." + fmt:
            dst = dst.with_name(dst.name + "." + fmt)
        try:
            sts.render_export(output_id, fmt, int(bitrate), dst)
        except Exception as e:
            return {"error": getattr(e, "detail", None) or str(e)}
        server.jupdate(server.SETTINGS, {}, lambda st: st.update(export_dir=str(dst.parent)))
        self._saved.add(str(dst))
        return {"path": str(dst), "name": dst.name, "folder": str(dst.parent), "bytes": dst.stat().st_size}

    def reveal(self, path: str):
        if path in self._saved and Path(path).exists():
            import subprocess

            subprocess.Popen(["open", "-R", path])
            return True
        return False


SELFTEST_JS = r"""
(async () => {
  const r = { ua: navigator.userAgent, secure: window.isSecureContext,
    mediaDevices: !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia),
    audioWorklet: typeof AudioWorkletNode === 'function', audioContext: typeof AudioContext === 'function',
    lookbehindRegex: (() => { try { new RegExp('(?<=a)b'); return true } catch { return false } })(),
    colorMix: CSS.supports('color', 'color-mix(in srgb, red 50%, blue)'),
    textWrapPretty: CSS.supports('text-wrap', 'pretty'),
    appBooted: typeof S !== 'undefined' && !!S.setup,
    nativeSave: !!(window.pywebview && window.pywebview.api && window.pywebview.api.save_export) };
  if (__NOMIC__) { r.mic = 'skipped'; window.__selftest = JSON.stringify(r); return; }
  try {
    window.__stage = 'enumerate'; r.devices = (await navigator.mediaDevices.enumerateDevices()).map(d => d.kind).join(',');
    window.__stage = 'getUserMedia called';
    const s = await navigator.mediaDevices.getUserMedia(CONSTRAINTS());
    window.__stage = 'getUserMedia resolved';
    r.mic = 'granted'; r.trackSettings = s.getAudioTracks()[0].getSettings();
    const ctx = new AudioContext(); await ctx.audioWorklet.addModule('recorder-worklet.js'); r.worklet = 'loaded'; r.sampleRate = ctx.sampleRate;
    if (__FULL__) {  // 2 s of real capture through the worklet: proves samples actually arrive
      window.__stage = 'capturing';
      const node = new AudioWorkletNode(ctx, 'raw-recorder'); const mute = ctx.createGain(); mute.gain.value = 0;
      ctx.createMediaStreamSource(s).connect(node); node.connect(mute); mute.connect(ctx.destination);
      let n = 0, peak = 0, sq = 0;
      node.port.onmessage = (e) => { const x = e.data.samples; if (!x) return; for (const v of x) { n++; sq += v * v; if (Math.abs(v) > peak) peak = Math.abs(v); } };
      await ctx.resume(); node.port.postMessage({ recording: true });
      await new Promise(res => setTimeout(res, 2000));
      node.port.postMessage({ recording: false });
      r.capture = { samples: n, seconds: +(n / ctx.sampleRate).toFixed(2), peak_dbfs: +(20 * Math.log10(peak || 1e-9)).toFixed(1),
                    rms_dbfs: +(10 * Math.log10(sq / Math.max(n, 1) || 1e-18)).toFixed(1) };
    }
    s.getTracks().forEach(t => t.stop()); ctx.close();
  } catch (e) { r.mic = 'error: ' + e.name + ': ' + e.message; }
  if (__FULL__) {  // one Preview through the same API the Studio screen uses
    try {
      window.__stage = 'generating'; const t0 = performance.now();
      const fd = new FormData(); fd.append('preview', 'true'); fd.append('params', '{}');
      const j = await (await fetch('/api/generate', { method: 'POST', body: fd })).json();
      if (!j.id) throw new Error(JSON.stringify(j));
      let st; do { await new Promise(res => setTimeout(res, 500)); st = await (await fetch('/api/jobs/' + j.id)).json(); }
      while (st.state !== 'done' && st.state !== 'error');
      r.generate = { state: st.state, error: st.error || null, seconds: st.result && st.result.duration,
                     similarity: st.result && st.result.similarity, wall_s: +((performance.now() - t0) / 1000).toFixed(1) };
    } catch (e) { r.generate = 'error: ' + e.message; }
  }
  window.__selftest = JSON.stringify(r);
})(); true
"""


def snapshot(window, path: Path):
    """Self-test only: WKWebView.takeSnapshot of the visible page -> PNG (no Screen Recording permission needed)."""
    import AppKit
    from PyObjCTools import AppHelper

    done = threading.Event()

    def handler(image, error):
        if image is not None:
            tiff = image.TIFFRepresentation()
            rep = AppKit.NSBitmapImageRep.imageRepWithData_(tiff)
            png = rep.representationUsingType_properties_(AppKit.NSBitmapImageFileTypePNG, {})
            path.parent.mkdir(parents=True, exist_ok=True)
            png.writeToFile_atomically_(str(path), True)
        done.set()

    def go():
        try:
            from webview.platforms.cocoa import BrowserView

            BrowserView.instances[window.uid].webview.takeSnapshotWithConfiguration_completionHandler_(None, handler)
        except Exception as e:  # self-test only: report, don't crash the app
            print("snapshot failed:", repr(e), flush=True)
            done.set()

    AppHelper.callAfter(go)
    if not done.wait(10):
        print("snapshot timed out:", path.name, flush=True)


def main():
    import webview

    port = free_port()
    server = start_server(port)
    allow_microphone_for(port)
    url = f"http://127.0.0.1:{port}/"
    api = DesktopApi()
    window = webview.create_window(APP_NAME, url, width=1180, height=880, min_size=(720, 640),
                                   background_color="#ECE6F3", text_select=True, js_api=api)
    api._window = window

    if os.environ.get("VCS_SELFTEST"):
        def selftest():
            time.sleep(3)
            out = {"port": port}
            try:
                js = (SELFTEST_JS.replace("__NOMIC__", "true" if os.environ.get("VCS_SELFTEST_NOMIC") else "false")
                      .replace("__FULL__", "true" if os.environ.get("VCS_SELFTEST_FULL") else "false"))
                window.evaluate_js(js)  # pywebview doesn't await promises: poll for the result
                res = None
                for _ in range(int(os.environ.get("VCS_SELFTEST_WAIT", "240"))):
                    res = window.evaluate_js("window.__selftest || null")
                    if res:
                        break
                    time.sleep(0.5)
                out.update(json.loads(res) if isinstance(res, str) else {"raw": res, "note": "timed out",
                                                                         "stage": window.evaluate_js("window.__stage || null")})
            except Exception as e:
                out["error"] = repr(e)
            (ROOT / "data").mkdir(exist_ok=True)
            (ROOT / "data" / "selftest.json").write_text(json.dumps(out, indent=1))
            shots = list(filter(None, os.environ.get("VCS_SELFTEST_SHOTS", "").split(",")))
            for _ in range(180 if shots else 0):  # screenshots: wait until the engine-loading overlay is gone
                if window.evaluate_js("!document.querySelector('#boot') || document.querySelector('#boot').hidden"):
                    break
                time.sleep(0.5)
            for spec in shots:
                # spec = name:screen:theme   e.g. studio_dark:studio:dark
                name, screen, theme = spec.split(":")
                window.evaluate_js(f"applyTheme('{theme}'); (() => {{ const t = document.querySelector('.tab[data-screen={screen}]'); if (!t || !t.disabled) show('{screen}'); }})(); true")
                time.sleep(1.5)
                snapshot(window, ROOT / "tests" / "screenshots" / f"{name}.png")
                time.sleep(1.0)
            time.sleep(float(os.environ.get("VCS_SELFTEST_HOLD", "0")))
            window.destroy()

        threading.Thread(target=selftest, daemon=True).start()

    webview.start(private_mode=False, storage_path=str(ROOT / "data" / "webview"))
    server.should_exit = True


if __name__ == "__main__":
    main()
