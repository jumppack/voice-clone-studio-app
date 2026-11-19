"""FastAPI backend. Bound to 127.0.0.1 by run.sh. No telemetry, no cloud calls."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask
from fastapi.staticfiles import StaticFiles

from . import audio
from .engine import (MTL_FILES, SR, TARGET_MAX, TARGET_MIN, Engine, GenParams, mtl_weights_present,
                     preview_sentences, split_text, word_count)
from .languages import CJK, DEFAULT_LANGUAGE, LANGUAGES
from .languages import listing as language_listing
from .scripts import MIN_TAKES, STEPS

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("VCS_DATA", ROOT / "data")).resolve()
TAKES, OUT, CACHE = DATA / "takes", DATA / "outputs", DATA / "cache"
for d in (TAKES, OUT, CACHE):
    d.mkdir(parents=True, exist_ok=True)
SETTINGS, PROFILE, PRESETS = DATA / "settings.json", DATA / "profile.json", DATA / "presets.json"

MODEL_FILES = ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]
BUILTIN_PRESETS = [
    {"id": "close", "name": "Close to me", "builtin": True,
     "params": {"similarity": 1.0, "exaggeration": 0.5, "cfg_weight": 0.5, "depth": 0, "temperature": 0.8}},
    {"id": "narrator", "name": "Narrator", "builtin": True,
     "params": {"similarity": 0.9, "exaggeration": 0.35, "cfg_weight": 0.3, "depth": -1, "temperature": 0.6}},
    {"id": "hype", "name": "Hype", "builtin": True,
     "params": {"similarity": 0.95, "exaggeration": 1.1, "cfg_weight": 0.65, "depth": 1, "temperature": 0.95}},
]

engine = Engine(CACHE)
app = FastAPI(title="Voice Clone Studio", docs_url=None, redoc_url=None, openapi_url=None)
jobs: dict[str, dict] = {}
installs: dict[str, dict] = {}


# ---------------- helpers ----------------
def jload(p: Path, default):
    try:
        return json.loads(p.read_text())
    except Exception:
        return default


STATE_LOCK = threading.RLock()  # FastAPI runs sync routes in a thread pool: serialise read-modify-write


def jsave(p: Path, obj):
    with STATE_LOCK:
        tmp = p.with_name(f".{p.name}.{uuid.uuid4().hex[:6]}.tmp")
        tmp.write_text(json.dumps(obj, indent=2))
        tmp.replace(p)


def jupdate(p: Path, default, fn):
    """Atomic read-modify-write of a JSON file."""
    with STATE_LOCK:
        obj = jload(p, default)
        r = fn(obj)
        obj = obj if r is None else r
        jsave(p, obj)
        return obj


def weights_present() -> dict:
    """Look in the local Hugging Face cache only (no network)."""
    try:
        from huggingface_hub import try_to_load_from_cache

        found = {f: try_to_load_from_cache("ResembleAI/chatterbox", f) for f in MODEL_FILES}
        ok = all(isinstance(v, str) and Path(v).exists() for v in found.values())
        size = sum(Path(v).stat().st_size for v in found.values() if isinstance(v, str) and Path(v).exists())
        return {"ok": ok, "bytes": size}
    except Exception as e:
        return {"ok": False, "bytes": 0, "error": str(e)}


def take_path(step: int) -> Path:
    return TAKES / f"step{step}.wav"


def takes_info() -> list[dict]:
    out = []
    for s in STEPS:
        p = take_path(s["id"])
        meta = jload(p.with_suffix(".json"), None)
        if p.exists() and meta:
            if meta.get("source") == "upload":  # clip cut from an uploaded file
                u = meta["upload"]
                out.append({"step": s["id"], "label": f"Upload part {u['part']}",
                            "tone": f"{u['file']} · {fmt_mmss(u['start'])}–{fmt_mmss(u['end'])}", **meta})
            else:
                out.append({"step": s["id"], "tone": s["tone"], "label": s["label"], **meta})
    return out


def fmt_mmss(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def profile() -> dict:
    takes = takes_info()
    prof = jload(PROFILE, {})
    ids = {t["step"] for t in takes}
    best = max(takes, key=lambda t: t["analysis"]["score"])["step"] if takes else None
    usable = sum(1 for t in takes if t["analysis"]["verdict"] != "redo")
    from_upload = bool(takes) and all(t.get("source") == "upload" for t in takes)
    need = 1 if from_upload else MIN_TAKES  # an uploaded recording needs one usable clip; recording needs 3 takes
    voice = prof.get("voice_take") if prof.get("voice_take") in ids else best
    style = prof.get("style_take") if prof.get("style_take") in ids else voice
    return {"takes": takes, "voice_take": voice, "style_take": style, "best_take": best,
            "ready": usable >= need, "usable": usable, "min_takes": need,
            "source": "upload" if from_upload else ("record" if takes else None),
            "upload_file": takes[0]["upload"]["file"] if from_upload else None}


# ---------------- setup ----------------
INSTALL_CMDS = {
    "packages": [str(ROOT / ".venv/bin/python"), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")],
    "weights": [str(ROOT / ".venv/bin/python"), "-c",
                "from huggingface_hub import hf_hub_download as d;"
                + "".join(f"d('ResembleAI/chatterbox','{f}');" for f in MODEL_FILES)],
    "mtl_weights": [str(ROOT / ".venv/bin/python"), "-c",
                    "from huggingface_hub import hf_hub_download as d;"
                    + "".join(f"d('ResembleAI/chatterbox','{f}');" for f in MTL_FILES)],
    "ffmpeg": ["brew", "install", "ffmpeg"],
    "asr": [str(ROOT / ".venv/bin/python"), "-c",
            "from huggingface_hub import hf_hub_download as d;"
            "d('mlx-community/whisper-large-v3-turbo','config.json');d('mlx-community/whisper-large-v3-turbo','weights.safetensors')"],
}
INSTALL_INFO = {
    "packages": "Python packages from pypi.org (~1.1 GB on disk)",
    "weights": "Chatterbox model weights from huggingface.co (~3.2 GB)",
    "mtl_weights": "Multilingual Chatterbox model from huggingface.co (~2.2 GB), for the 22 languages besides English",
    "ffmpeg": "ffmpeg via Homebrew (~60 MB), needed for MP3 export",
    "asr": "Transcription model mlx-community/whisper-large-v3-turbo from huggingface.co (1.6 GB), for Speak like the voice "
           "(needs the mlx-whisper package: pip install mlx-whisper)",
}


def asr_item() -> dict:
    from .restyle import asr_present

    try:
        import mlx_whisper  # noqa: F401
        pkg = True
    except Exception:
        pkg = False
    ok = pkg and asr_present()
    return {"id": "asr", "name": "Transcription (speech to speech)", "ok": ok, "optional": True,
            "detail": "whisper-large-v3-turbo ready: Speak like the voice works" if ok
            else "optional: needed only for Speak like the voice", "size": "~1.6 GB",
            "command": None if ok else ("download the transcription model from huggingface.co" if pkg
                                        else "pip install mlx-whisper, then download the model")}


@app.get("/api/setup")
def setup_status():
    ff = audio.ffmpeg_path()
    try:
        import importlib.metadata as md

        cb = md.version("chatterbox-tts")
    except Exception:
        cb = None
    w = weights_present()
    s = jload(SETTINGS, {})
    return {
        "items": [
            {"id": "python", "name": "Python environment", "ok": sys.version_info[:2] == (3, 11),
             "detail": f"Python {sys.version.split()[0]} in .venv", "size": "1.0 GB on disk"},
            {"id": "packages", "name": "Chatterbox speech engine", "ok": cb is not None,
             "detail": f"chatterbox-tts {cb}" if cb else "not installed", "size": "~1.1 GB",
             "command": " ".join(INSTALL_CMDS["packages"]) if not cb else None},
            {"id": "weights", "name": "Voice model weights", "ok": w["ok"],
             "detail": f"{w['bytes'] / 1e9:.2f} GB in the local Hugging Face cache" if w["ok"] else "not downloaded",
             "size": "~3.2 GB", "command": None if w["ok"] else "download ResembleAI/chatterbox from huggingface.co"},
            {"id": "mtl_weights", "name": "Other languages", "ok": mtl_weights_present(), "optional": True,
             "detail": "multilingual model ready: Hindi, Spanish, Chinese and 19 more" if mtl_weights_present()
             else "optional: needed only to speak languages other than English", "size": "~2.2 GB",
             "command": None if mtl_weights_present() else "download the multilingual model from huggingface.co"},
            asr_item(),
            {"id": "ffmpeg", "name": "ffmpeg (reads audio files, MP3)", "ok": bool(ff) and audio.has_mp3lame(ff),
             "detail": ff or "not found", "size": "~60 MB", "command": None if ff else "brew install ffmpeg"},
        ],
        "engine": {"state": engine.state, "error": engine.error, "device": str(engine.device) if engine.device else None,
                   "load_seconds": engine.load_seconds},
        "settings": {"consent": bool(s.get("consent")), "watermark_ack": bool(s.get("watermark_ack")),
                     "voice_source": s.get("voice_source"), "language": s.get("language", DEFAULT_LANGUAGE),
                     "tts_voice": s.get("tts_voice", "me"), "sts_voice": s.get("sts_voice", "me"),
                     "sts_mode": s.get("sts_mode", "timing")},
        "installs": installs,
    }


@app.post("/api/setup/install")
def setup_install(item: str = Form(...), confirm: bool = Form(False)):
    if item not in INSTALL_CMDS:
        raise HTTPException(400, "unknown item")
    cmd = INSTALL_CMDS[item]
    if not confirm:  # first call only shows what would run
        return {"item": item, "command": " ".join(cmd), "what": INSTALL_INFO[item], "confirm_required": True}
    if installs.get(item, {}).get("state") == "running":
        return installs[item]
    env = dict(os.environ)
    if item in ("weights", "mtl_weights", "asr"):
        env.pop("HF_HUB_OFFLINE", None)  # the approved model download is the only network use
    rec = installs[item] = {"state": "running", "log": "", "command": " ".join(cmd)}

    def run():
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
            for line in p.stdout:
                rec["log"] = (rec["log"] + line)[-4000:]
            rec["state"] = "done" if p.wait() == 0 else "failed"
        except Exception as e:
            rec["state"], rec["log"] = "failed", str(e)
        if item == "weights" and rec["state"] == "done":
            engine.load_async()

    threading.Thread(target=run, daemon=True).start()
    return rec


@app.post("/api/settings")
def save_settings(consent: bool = Form(None), watermark_ack: bool = Form(None), voice_source: str = Form(None),
                  language: str = Form(None)):
    if voice_source is not None and voice_source not in ("record", "upload"):
        raise HTTPException(400, "voice_source must be record or upload")
    if language is not None and language not in LANGUAGES:
        raise HTTPException(400, "unknown language")

    def upd(s):
        if language is not None:
            s["language"] = language
        if voice_source is not None:
            s["voice_source"] = voice_source
        if consent is not None:
            s["consent"] = consent
        if watermark_ack is not None:
            s["watermark_ack"] = watermark_ack

    return jupdate(SETTINGS, {}, upd)


# ---------------- recording ----------------
@app.get("/api/steps")
def steps():
    return {"steps": STEPS, "min_takes": MIN_TAKES}


async def _read_upload(f: UploadFile):
    data = await f.read()
    try:
        return audio.read_wav_bytes(data)
    except Exception as e:
        raise HTTPException(400, f"Could not read audio: {e}")


@app.post("/api/room-check")
async def room_check(file: UploadFile = File(...)):
    x, sr, _ = await _read_upload(file)
    r = audio.room_noise(x, sr)
    jupdate(SETTINGS, {}, lambda s: s.update(room_floor=r["noise_floor_dbfs"]))
    return r


@app.post("/api/takes/{step}")
async def upload_take(step: int, file: UploadFile = File(...), constraints: str = Form("{}")):
    if step not in {s["id"] for s in STEPS}:
        raise HTTPException(404, "no such step")
    if not jload(SETTINGS, {}).get("consent"):
        raise HTTPException(403, "Please confirm the voice consent in Setup first.")
    x, sr, fmt = await _read_upload(file)
    a = audio.analyze_take(x, sr, room_floor=jload(SETTINGS, {}).get("room_floor"))
    y = audio.trim_silence(x, sr)
    p = take_path(step)
    audio.write_wav(p, y, sr)
    audio.write_wav(p.with_name(f"step{step}_raw.wav"), x, sr)
    meta = {"analysis": a, "trimmed_seconds": round(len(y) / sr, 2), "sr": sr, "upload_format": fmt,
            "capture": jload_str(constraints), "time": time.time()}
    jsave(p.with_suffix(".json"), meta)
    return {"step": step, **meta}


@app.post("/api/upload-voice")
def upload_voice(file: UploadFile = File(...)):  # sync: runs in the threadpool
    """Use an existing recording (WAV, MP3, ...) instead of recording takes. Any length: the file is
    decoded locally (first 10 minutes), cut at pauses into up to 6 clips of 7-15 s (a short file
    becomes one clip), and each clip is checked (plus a same-speaker check across clips).
    The clips REPLACE the current takes. Requires the Setup consent ("my voice or I have permission")."""
    if not jload(SETTINGS, {}).get("consent"):
        raise HTTPException(403, "Please tick \"This is my own voice, or I have the speaker's permission\" in Setup first.")
    data = file.file.read(audio.UPLOAD_MAX_BYTES + 1)
    if len(data) > audio.UPLOAD_MAX_BYTES:
        raise HTTPException(413, "That file is over 200 MB. Please use a shorter or more compressed file (under 30 MB is plenty).")
    try:
        x, sr, fmt = audio.decode_upload(data, file.filename or "")
    except Exception as e:
        raise HTTPException(400, f"Could not read that file: {e}")
    ranges = audio.split_for_takes(x, sr)
    if not ranges:
        raise HTTPException(400, f"No usable speech found. The file needs at least {audio.UPLOAD_MIN_SPEECH_S:g} s of clear talking (10 s or more works best).")
    name = re.sub(r"[^\w .\-]", "", Path(file.filename or "upload").name)[:60] or "upload"
    clips = [x[a:b] for a, b in ranges]
    analyses = [audio.analyze_take(c, sr, room_floor=None, source="upload") for c in clips]

    # same-speaker check: each clip's VoiceEncoder embedding against the median clip
    speaker = None
    if len(clips) >= 2 and engine.state != "error":
        engine.load()
        embs = np.stack([engine.embed(c, sr) for c in clips])
        centre = np.median(embs, axis=0)
        centre /= np.linalg.norm(centre) + 1e-9
        sims = [round(float(v), 3) for v in embs @ centre]
        speaker = {"similarity_to_median": sims, "threshold": SAME_SPEAKER_MIN}
        for a, sim in zip(analyses, sims):
            if sim < SAME_SPEAKER_MIN:
                a["issues"] = [i for i in a["issues"] if i["kind"] != "ok"]
                a["issues"].append({"kind": "speaker", "level": "bad",
                                    "text": "This part sounds like a different speaker. It won't be used."})
                a["verdict"] = "redo"
                a["score"] = min(a["score"], 30)

    for p in TAKES.glob("step*"):  # uploaded clips replace the recorded takes
        p.unlink()
    jupdate(PROFILE, {}, lambda prof: prof.clear())
    for i, (c, (a0, b0), an) in enumerate(zip(clips, ranges, analyses), start=1):
        y = audio.trim_silence(c, sr)
        p = take_path(i)
        audio.write_wav(p, y, sr)
        meta = {"analysis": an, "trimmed_seconds": round(len(y) / sr, 2), "sr": sr, "upload_format": fmt,
                "source": "upload", "upload": {"file": name, "part": i, "start": round(a0 / sr, 2), "end": round(b0 / sr, 2)},
                "time": time.time()}
        jsave(p.with_suffix(".json"), meta)
    jupdate(SETTINGS, {}, lambda st: st.update(voice_source="upload"))
    full = fmt.get("full_seconds")
    return {"file": name, "duration": round(len(x) / sr, 1), "format": fmt, "clips": len(clips),
            "truncated": bool(full and full > len(x) / sr + 1) or len(x) / sr >= audio.UPLOAD_ANALYZE_SECONDS - 1,
            "speaker_check": speaker, **profile()}


SAME_SPEAKER_MIN = 0.75  # VoiceEncoder cosine to the median clip; same speaker measured 0.85+, other speakers ~0.6


def jload_str(s: str):
    try:
        return json.loads(s)
    except Exception:
        return {}


@app.get("/api/takes/{step}/audio")
def take_audio(step: int):
    p = take_path(step)
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="audio/wav")


@app.delete("/api/takes/{step}")
def delete_take(step: int):
    for p in TAKES.glob(f"step{step}*"):
        p.unlink()
    return profile()


@app.get("/api/profile")
def get_profile():
    return profile()


@app.post("/api/profile")
def set_profile(voice_take: int = Form(None), style_take: int = Form(None)):
    def upd(prof):
        if voice_take is not None:
            prof["voice_take"] = voice_take
        if style_take is not None:
            prof["style_take"] = style_take

    jupdate(PROFILE, {}, upd)
    return profile()


# ---------------- presets ----------------
@app.get("/api/presets")
def get_presets():
    return {"presets": BUILTIN_PRESETS + jload(PRESETS, [])}


@app.post("/api/presets")
def add_preset(name: str = Form(...), params: str = Form(...)):
    p = {"id": uuid.uuid4().hex[:8], "name": name.strip()[:40] or "My preset", "builtin": False,
         "params": GenParams.from_dict(json.loads(params)).__dict__}
    jupdate(PRESETS, [], lambda user: user.append(p))
    return get_presets()


@app.delete("/api/presets/{pid}")
def del_preset(pid: str):
    jupdate(PRESETS, [], lambda user: [p for p in user if p["id"] != pid])
    return get_presets()


# ---------------- generation ----------------
@app.post("/api/generate")
def generate(text: str = Form(""), params: str = Form("{}"), preview: bool = Form(False), target: float = Form(0),
             language: str = Form(DEFAULT_LANGUAGE), voice: str = Form("me")):
    """target = length in seconds (5-30) or 0 for 'speak the whole text'. A preview always has a
    target (default 5 s) and is built from the built-in preview passage.
    voice = "me" (the recorded/uploaded profile) or a clip id from Your voices."""
    from . import voices

    vref, sref = voices.ref_paths(voice)
    if target and not TARGET_MIN <= target <= TARGET_MAX:
        raise HTTPException(400, f"Length must be between {TARGET_MIN:g} and {TARGET_MAX:g} seconds.")
    lang = language
    if lang not in LANGUAGES:
        raise HTTPException(400, f"Unsupported language: {lang}")
    if lang != "en" and not mtl_weights_present():
        raise HTTPException(400, "Speaking languages other than English needs the multilingual model. "
                                 "Download it on the Setup screen (Other languages).")
    tgt = float(target) if target else (TARGET_MIN if preview else None)
    pool = preview_sentences(lang)
    text = ("" if lang in CJK else " ").join(pool) if preview else text.strip()
    if word_count(text, lang) < 4:
        raise HTTPException(400, "Please type at least 4 words. Very short lines come out clipped.")
    if len(text) > 5000:
        raise HTTPException(400, "That's a lot of text. Please keep it under 5,000 characters per run.")
    p = GenParams.from_dict(json.loads(params))
    jupdate(SETTINGS, {}, lambda st: st.update(tts_voice=voice))
    jid = uuid.uuid4().hex[:10]
    n = len(split_text(text, lang))
    job = jobs[jid] = {"id": jid, "state": "queued", "done": 0, "total": n, "elapsed": 0.0,
                       "estimate": engine.estimate_seconds(text, tgt, lang) + (40 if engine.state != "ready" else 0)
                       + (10 if lang != "en" and engine.mtl is None else 0),
                       "preview": preview, "text": text, "target": tgt, "params": p.__dict__, "language": lang,
                       "voice": voice}

    def run():
        try:
            job["state"] = "loading" if engine.state != "ready" or (lang != "en" and engine.mtl is None) else "running"
            engine.load()
            if lang != "en":
                engine.load_mtl()
            job["state"] = "running"
            t0 = time.time()

            def prog(i, total, el):
                job.update(done=i, total=total, elapsed=round(time.time() - t0, 1))

            wav, info = engine.generate(text, vref, sref, p, progress=prog, target=tgt,
                                        sentences=pool if preview else None, lang=lang)
            job["text"] = info["text"]
            path = OUT / f"{jid}.wav"
            audio.write_wav(path, wav, SR)
            sim = engine.similarity(wav, vref)
            job.update(state="done", result={**info, "similarity": round(sim, 3), "peaks": audio.peaks(wav),
                                             "url": f"/api/outputs/{jid}.wav"}, elapsed=round(time.time() - t0, 1))
        except Exception as e:
            job.update(state="error", error=f"{e}")

    threading.Thread(target=run, daemon=True).start()
    return job


@app.get("/api/jobs/{jid}")
def job_status(jid: str):
    if jid not in jobs:
        raise HTTPException(404)
    return jobs[jid]


@app.get("/api/outputs/{name}")
def output_file(name: str):
    if not re.fullmatch(r"[0-9a-f]{10}\.wav", name) or not (OUT / name).exists():
        raise HTTPException(404)
    return FileResponse(OUT / name, media_type="audio/wav")


SAFE_NAME = re.compile(r"[^A-Za-z0-9 _\-.]")


EXPORT_TYPES = {"wav": "audio/wav", "mp3": "audio/mpeg"}


def export_name(filename: str, fmt: str) -> str:
    name = SAFE_NAME.sub("", filename).strip(" .")[:80] or "my-voice"
    return name if name.lower().endswith("." + fmt) else f"{name}.{fmt}"


def render_export(output_id: str, fmt: str, bitrate: int, dst: Path) -> Path:
    """Write a generated output as WAV (24 kHz/16-bit) or MP3 (48 kHz, given bitrate) to dst.
    Used by the HTTP export (browser download) and by the macOS app's Save dialog."""
    src = OUT / f"{output_id}.wav"
    if not re.fullmatch(r"[0-9a-f]{10}", output_id) or not src.exists():
        raise HTTPException(404, "Generate something first.")
    if fmt == "wav":
        x, sr = sf.read(str(src), dtype="float32")
        audio.write_wav(dst, x, sr, "PCM_16")
    elif fmt == "mp3":
        if bitrate not in (128, 192, 256, 320):
            raise HTTPException(400, "bitrate must be 128, 192, 256 or 320")
        try:
            audio.export_mp3(src, dst, bitrate)
        except Exception as e:
            raise HTTPException(500, f"MP3 export failed: {e}")
    else:
        raise HTTPException(400, "format must be wav or mp3")
    return dst


@app.post("/api/export")
def export(output_id: str = Form(...), fmt: str = Form("wav"), bitrate: int = Form(192), filename: str = Form("my-voice")):
    """Returns the file itself; where it is saved is up to the user (Save dialog / browser download).
    Nothing is kept on the server side."""
    name = export_name(filename, fmt)
    tmp = CACHE / f"export-{uuid.uuid4().hex[:8]}.{fmt}"
    try:
        render_export(output_id, fmt, bitrate, tmp)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return FileResponse(tmp, media_type=EXPORT_TYPES[fmt], filename=name,
                        background=BackgroundTask(lambda: tmp.unlink(missing_ok=True)))


@app.get("/api/speech-rate")
def speech_rate(language: str = DEFAULT_LANGUAGE):
    lang = language if language in LANGUAGES else DEFAULT_LANGUAGE
    return {"cps": engine.rate(lang), "language": lang, "min": TARGET_MIN, "max": TARGET_MAX}


@app.get("/api/languages")
def languages():
    """Languages, whether the multilingual model is downloaded, and a sample text for each."""
    have = mtl_weights_present()
    out = []
    for item in language_listing():
        pool = preview_sentences(item["code"])
        out.append({**item, "available": item["code"] == "en" or have, "cps": engine.rate(item["code"]),
                    "sample": ("" if item["cjk"] else " ").join(pool[:3])})
    return {"languages": out, "multilingual_ready": have, "default": DEFAULT_LANGUAGE}


@app.on_event("startup")
def _warm():
    if weights_present()["ok"]:
        engine.load_async()  # warm the model in the background so the first Preview is quick


from . import sts, voices  # noqa: E402  (they import this module: engine, DATA, helpers)

@app.middleware("http")
async def _no_stale_ui(request, call_next):
    """The macOS app's WKWebView kept serving a cached old index.html/app.js after an update (it reuses
    responses with Last-Modified without asking). no-cache = always revalidate (ETag), so a rebuilt app
    shows its new UI."""
    resp = await call_next(request)
    if not request.url.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


app.include_router(voices.router)
app.include_router(sts.router)
app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
