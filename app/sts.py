"""Speech to speech routes (/api/sts): re-voice a whole recording with a voice from Your voices.

Two modes:
  * "timing" (keep the original timing): voice conversion, converter.py. Same length and timing as the
    original; rhythm and intonation stay the original speaker's, only the voice changes.
  * "style" (speak like the voice): restyle.py. Transcribe, then speak it again with the voice's own
    pace and delivery. Longer or shorter than the original; needs the local transcription model.

One recording at a time, one job at a time (shares the text-to-speech engine and its GPU lock).
Files in data/sts/.
"""
from __future__ import annotations

import json
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from . import server as srv
from . import voices
from .converter import SR_OUT, SR_TOK, Cancelled, Converter, Piece, decode, plan
from .restyle import Cancelled as RespeakCancelled
from .restyle import asr_present, respeak

audio = srv.audio
DATA = srv.DATA / "sts"
OUT, TMP = DATA / "outputs", DATA / "tmp"
for d in (DATA, OUT, TMP):
    d.mkdir(parents=True, exist_ok=True)
SOURCE_WAV, SOURCE_META, LAST = DATA / "source.wav", DATA / "source.json", DATA / "last.json"
SOURCE_MAX_S = 3 * 3600
KEEP_OUTPUTS = 4
MODES = ("timing", "style")
STYLE_RTF = 2.0  # measured: ~1.8-2.1 s of compute per second of re-spoken audio (Apple Silicon)

conv = Converter(srv.engine)
router = APIRouter(prefix="/api/sts")
jobs: dict[str, dict] = {}


def active_job():
    return next((j for j in jobs.values() if j["state"] in ("queued", "running")), None) or \
        next((j for j in srv.jobs.values() if j.get("state") in ("queued", "loading", "running")), None)


def ref_len(voice: str) -> float:
    try:
        a, _ = voices.ref_paths(voice)
        return sf.info(str(a)).duration
    except Exception:
        return 15.0


def estimates(meta: dict | None, voice: str) -> dict | None:
    if not meta:
        return None
    sp = meta["speech_seconds"]
    return {"timing": conv.estimate(sp, min(ref_len(voice), 30.0)), "style": round(sp * STYLE_RTF + 20, 0)}


@router.get("/status")
def status(voice: str = "me"):
    meta = srv.jload(SOURCE_META, None) if SOURCE_WAV.exists() else None
    last = srv.jload(LAST, None)
    j = next((j for j in jobs.values() if j["state"] in ("queued", "running")), None)
    return {"source": meta, "estimates": estimates(meta, voice), "style_ready": asr_present(),
            "job": None if not j else {"id": j["id"], "kind": j["kind"], "mode": j["mode"], "state": j["state"]},
            "last": last if last and (OUT / f"{last['id']}.wav").exists() else None,
            "mp3": audio.has_mp3lame(audio.ffmpeg_path())}


@router.post("/source")
def upload_source(file: UploadFile = File(...)):  # sync: runs in the threadpool
    if any(j["state"] in ("queued", "running") for j in jobs.values()):
        raise HTTPException(409, "Wait for the conversion to finish (or cancel it) first.")
    up = voices.save_upload(file, TMP)
    try:
        x24 = decode(up, SR_OUT, SOURCE_MAX_S + 1)
    except ValueError as e:
        raise HTTPException(400, str(e))
    finally:
        up.unlink(missing_ok=True)
    if len(x24) < SR_OUT:
        raise HTTPException(400, "That file is shorter than a second.")
    if len(x24) > SOURCE_MAX_S * SR_OUT:
        raise HTTPException(400, "That file is over 3 hours. Split it into parts first.")
    import librosa

    x16 = librosa.resample(x24, orig_sr=SR_OUT, target_sr=SR_TOK, res_type="soxr_hq").astype(np.float32)
    sp = [p for p in plan(x16) if p.speech]
    speech_s = sum(p.seconds for p in sp)
    _, floor = audio.speech_mask(x16, SR_TOK)
    if speech_s < 1:
        raise HTTPException(400, "No speech was found in that file.")
    notes = []
    if speech_s / (len(x16) / SR_TOK) < 0.2:
        notes.append({"level": "warn", "text": "Most of this file is quiet or isn't speech. Only the speech parts are changed."})
    if floor > -45:
        notes.append({"level": "warn", "text": f"There's background sound ({floor:.0f} dBFS). With Keep original timing, music or "
                                                "noise under the voice can come out garbled."})
    sf.write(str(SOURCE_WAV), np.clip(x24, -1, 1), SR_OUT, subtype="PCM_16")
    meta = {"name": file.filename or "audio", "seconds": round(len(x24) / SR_OUT, 2), "speech_seconds": round(speech_s, 1),
            "pieces": len(sp), "noise_floor_dbfs": round(floor, 1), "notes": notes, "peaks": audio.peaks(x24, 800),
            "version": uuid.uuid4().hex[:8]}
    srv.jsave(SOURCE_META, meta)
    return meta


@router.get("/source/audio")
def source_audio():
    if not SOURCE_WAV.exists():
        raise HTTPException(404, "No recording yet.")
    return FileResponse(SOURCE_WAV, media_type="audio/wav")


def _prune():
    files = sorted(OUT.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in files[KEEP_OUTPUTS:]:
        p.unlink(missing_ok=True)
        p.with_suffix(".json").unlink(missing_ok=True)


def _run(job: dict):
    try:
        job["state"] = "running"
        job["progress"] = {"stage": "loading", "fraction": 0}
        conv.load()
        vref, _ = voices.ref_paths(job["voice"])
        x24, _ = sf.read(str(SOURCE_WAV), dtype="float32")
        ref24, sr = sf.read(str(vref), dtype="float32", always_2d=True)
        ref24 = ref24.mean(axis=1)
        import librosa

        if sr != SR_OUT:
            ref24 = librosa.resample(ref24, orig_sr=sr, target_sr=SR_OUT)
        ref24, _ = Converter.prepare_reference(ref24)
        x16 = librosa.resample(x24, orig_sr=SR_OUT, target_sr=SR_TOK, res_type="soxr_hq").astype(np.float32)
        pieces, offset, orig_s = plan(x16), 0.0, len(x16) / SR_TOK
        if job["kind"] == "preview":  # the first speech piece (8-30 s, cut in a pause)
            first = next(p for p in pieces if p.speech)
            n16 = int(0.03 * SR_TOK)
            offset = first.a * 0.03
            x16 = x16[first.a * n16: first.b * n16]
            orig_s = len(x16) / SR_TOK
            pieces = [Piece(0, first.b - first.a, True)]

        def prog(p):
            job["progress"] = {**p, "stage": p.get("stage", "converting")}

        if job["mode"] == "timing":
            y, info = conv.convert(x16, ref24, pieces, progress=prog, cancel=job["cancel"])
            extra = {"similarity_min": info["similarity_min"], "low_pieces": info["low_pieces"]}
            sim = info["similarity"]
        else:
            tmp_ref = TMP / f"ref-{job['id']}.wav"
            sf.write(str(tmp_ref), ref24, SR_OUT, subtype="PCM_16")
            try:
                y, info = respeak(srv.engine, x16, tmp_ref, ref24, progress=prog, cancel=job["cancel"])
            finally:
                tmp_ref.unlink(missing_ok=True)
            sims, e_ref = [], conv.embed(ref24, SR_OUT)
            for s in np.array_split(y, max(1, int(len(y) / SR_OUT // 20))):
                if len(s) > SR_OUT * 3:
                    sims.append(float(np.dot(conv.embed(s, SR_OUT), e_ref)))
            sim = round(float(np.median(sims)), 3) if sims else None
            extra = {k: info[k] for k in ("language", "pace_reference", "pace_original", "pace_result", "text")}
        oid = uuid.uuid4().hex[:10]
        sf.write(str(OUT / f"{oid}.wav"), y, SR_OUT, subtype="PCM_16")
        src_meta = srv.jload(SOURCE_META, {})
        result = {"id": oid, "kind": job["kind"], "mode": job["mode"], "offset": offset, "seconds": round(len(y) / SR_OUT, 2),
                  "original_seconds": round(orig_s, 2), "similarity": sim, "compute_seconds": info["compute_seconds"],
                  "source": src_meta.get("name"), "source_version": src_meta.get("version"), "voice": job["voice"],
                  "voice_name": voices.voice_name(job["voice"]), "peaks": audio.peaks(y, 800), "created": time.time(), **extra}
        (OUT / f"{oid}.json").write_text(json.dumps({**result, "detail": info.get("pieces") or info.get("chunks")}))
        srv.jsave(LAST, result)
        _prune()
        job["result"], job["state"] = result, "done"
    except (Cancelled, RespeakCancelled):
        job["state"] = "cancelled"
    except Exception as e:
        job["state"], job["error"] = "error", getattr(e, "detail", None) or f"{e}"


@router.post("/convert")
def convert(preview: bool = Form(False), mode: str = Form("timing"), voice: str = Form("me")):
    s = srv.jload(srv.SETTINGS, {})
    if not (s.get("consent") and s.get("watermark_ack")):
        raise HTTPException(400, "Confirm both boxes in Settings first.")
    if mode not in MODES:
        raise HTTPException(400, "mode must be timing or style")
    if mode == "style" and not asr_present():
        raise HTTPException(400, "Speak like the voice needs the transcription model (see Settings).")
    if not srv.weights_present()["ok"]:
        raise HTTPException(400, "The voice model isn't downloaded yet (see Settings).")
    if not SOURCE_WAV.exists():
        raise HTTPException(400, "Add a recording first.")
    voices.ref_paths(voice)  # validates the voice
    with srv.STATE_LOCK:
        if active_job():
            raise HTTPException(409, "Something is already being generated. Wait for it to finish.")
        jid = uuid.uuid4().hex[:8]
        job = {"id": jid, "kind": "preview" if preview else "full", "mode": mode, "voice": voice, "state": "queued",
               "cancel": threading.Event(), "created": time.time()}
        jobs[jid] = job
    srv.jupdate(srv.SETTINGS, {}, lambda st: st.update(sts_voice=voice, sts_mode=mode))
    threading.Thread(target=_run, args=(job,), daemon=True).start()
    return {"id": jid}


@router.get("/jobs/{jid}")
def job_status(jid: str):
    j = jobs.get(jid)
    if not j:
        raise HTTPException(404, "unknown job")
    return {k: v for k, v in j.items() if k != "cancel"}


@router.post("/jobs/{jid}/cancel")
def job_cancel(jid: str):
    j = jobs.get(jid)
    if not j:
        raise HTTPException(404, "unknown job")
    j["cancel"].set()
    return {"ok": True}


def output_path(oid: str) -> Path:
    p = OUT / f"{oid}.wav"
    if not re.fullmatch(r"[0-9a-f]{10}", oid) or not p.exists():
        raise HTTPException(404, "Convert something first.")
    return p


@router.get("/outputs/{oid}")
def output_audio(oid: str):
    return FileResponse(output_path(oid), media_type="audio/wav")


def render_export(oid: str, fmt: str, bitrate: int, dst: Path) -> Path:
    src = output_path(oid)
    if fmt == "wav":
        shutil.copyfile(src, dst)
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


@router.post("/export")
def export(output_id: str = Form(...), fmt: str = Form("wav"), bitrate: int = Form(192), filename: str = Form("new-voice")):
    name = srv.export_name(filename, fmt)
    tmp = TMP / f"export-{uuid.uuid4().hex[:8]}.{fmt}"
    try:
        render_export(output_id, fmt, bitrate, tmp)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return FileResponse(tmp, media_type=srv.EXPORT_TYPES[fmt], filename=name,
                        background=BackgroundTask(lambda: tmp.unlink(missing_ok=True)))
