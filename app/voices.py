"""Your voices: the shared voice library for text to speech and speech to speech.

Two kinds:
  * "me": the recorded / uploaded voice profile (takes, voice take + style take), as before
  * clips: 3-30 s of one person talking (trimmed, mono 24 kHz), stored in data/voices/<id>.wav.
    A clip is both the voice and the speaking style.
"""
from __future__ import annotations

import re
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from . import server as srv
from .converter import REF_MIN_SPEECH_S, SR_OUT, Converter, decode

VOICES = srv.DATA / "voices"
VOICES.mkdir(parents=True, exist_ok=True)
INDEX = VOICES / "voices.json"
REF_READ_S = 120
MAX_UPLOAD_BYTES = 4 * 1024 ** 3

router = APIRouter(prefix="/api/voices")


def clips() -> list[dict]:
    return [v for v in srv.jload(INDEX, []) if (VOICES / f"{v['id']}.wav").exists()]


def me() -> dict:
    p = srv.profile()
    return {"id": "me", "kind": "profile", "name": "My voice", "ready": bool(p["ready"]), "takes": len(p["takes"]),
            "source": p.get("source")}


def listing() -> list[dict]:
    return [me()] + [{**v, "kind": "clip", "ready": True} for v in clips()]


def ref_paths(voice: str) -> tuple[Path, Path]:
    """(voice reference, style reference) for a voice id."""
    if voice == "me":
        p = srv.profile()
        if not p["ready"]:
            raise HTTPException(400, f"Record at least {srv.MIN_TAKES} takes for My voice first, or pick another voice.")
        return srv.take_path(p["voice_take"]), srv.take_path(p["style_take"])
    if not re.fullmatch(r"[0-9a-f]{8}", voice or "") or not (VOICES / f"{voice}.wav").exists():
        raise HTTPException(404, "That voice isn't in Your voices any more.")
    path = VOICES / f"{voice}.wav"
    return path, path


def voice_name(voice: str) -> str:
    if voice == "me":
        return "My voice"
    return next((v["name"] for v in clips() if v["id"] == voice), "voice")


def save_upload(f: UploadFile, dst_dir: Path) -> Path:
    ext = re.sub(r"[^a-z0-9]", "", Path(f.filename or "").suffix.lower())[:6] or "bin"
    dst = dst_dir / f"upload-{uuid.uuid4().hex[:8]}.{ext}"
    n = 0
    with dst.open("wb") as out:
        while chunk := f.file.read(8 * 1024 * 1024):
            n += len(chunk)
            if n > MAX_UPLOAD_BYTES:
                out.close()
                dst.unlink(missing_ok=True)
                raise HTTPException(413, "That file is over 4 GB.")
            out.write(chunk)
    if n == 0:
        dst.unlink(missing_ok=True)
        raise HTTPException(400, "The file is empty.")
    return dst


@router.get("")
def get_voices():
    return {"voices": listing()}


@router.post("")
def add_voice(file: UploadFile = File(...), name: str = Form("")):
    up = save_upload(file, srv.CACHE)
    try:
        x24 = decode(up, SR_OUT, REF_READ_S)
    except ValueError as e:
        raise HTTPException(400, str(e))
    finally:
        up.unlink(missing_ok=True)
    clip, info = Converter.prepare_reference(x24)
    if info["speech_seconds"] < REF_MIN_SPEECH_S:
        raise HTTPException(400, f"Only {info['speech_seconds']} s of speech in that file. A voice needs at least "
                                 f"{REF_MIN_SPEECH_S:.0f} s, ideally 10 to 30 s of one person talking.")
    vid = uuid.uuid4().hex[:8]
    sf.write(str(VOICES / f"{vid}.wav"), np.clip(clip, -1, 1), SR_OUT, subtype="PCM_16")
    notes = [i for i in info["issues"] if i["kind"] != "ok"]
    if info["given_seconds"] > info["used_seconds"] + 1:
        notes.append({"level": "info", "text": f"Only the first {info['used_seconds']:.0f} s are kept. More than 30 s "
                                               "doesn't help much and makes speech to speech slower."})
    base = name.strip() or Path(file.filename or "voice").stem
    meta = {"id": vid, "name": re.sub(r"[^\w .,'()-]", "", base)[:40] or "Voice", "file": file.filename,
            "seconds": info["used_seconds"], "speech_seconds": info["speech_seconds"], "verdict": info["verdict"],
            "score": info["score"], "notes": notes, "peaks": srv.audio.peaks(clip, 300), "created": time.time()}
    srv.jupdate(INDEX, [], lambda lst: lst.append(meta))
    return {**meta, "kind": "clip", "ready": True}


@router.post("/{vid}")
def rename_voice(vid: str, name: str = Form(...)):
    clean = re.sub(r"[^\w .,'()-]", "", name.strip())[:40]
    if not clean:
        raise HTTPException(400, "Give the voice a name.")

    def upd(lst):
        for v in lst:
            if v["id"] == vid:
                v["name"] = clean
    srv.jupdate(INDEX, [], upd)
    return {"ok": True, "name": clean}


@router.delete("/{vid}")
def delete_voice(vid: str):
    if not re.fullmatch(r"[0-9a-f]{8}", vid):
        raise HTTPException(404, "unknown voice")
    (VOICES / f"{vid}.wav").unlink(missing_ok=True)
    srv.jupdate(INDEX, [], lambda lst: [v for v in lst if v["id"] != vid])
    s = srv.jload(srv.SETTINGS, {})
    if s.get("tts_voice") == vid or s.get("sts_voice") == vid:
        srv.jupdate(srv.SETTINGS, {}, lambda st: st.update({k: "me" for k in ("tts_voice", "sts_voice") if st.get(k) == vid}))
    return {"ok": True}


@router.get("/{vid}/audio")
def voice_audio(vid: str):
    a, _ = ref_paths(vid)
    return srv.FileResponse(a, media_type="audio/wav")
