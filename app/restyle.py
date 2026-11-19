"""Speech to speech, "speak like the reference" mode.

Voice conversion (converter.py) keeps the original's rhythm, pauses and intonation: the S3 speech tokens
carry them and the reference only sets the timbre. To take on the reference speaker's STYLE as well,
the speech has to be generated again:

  1. transcribe the original locally (mlx-whisper, whisper-large-v3-turbo, Apple GPU, offline)
  2. speak each sentence with the text-to-speech engine, using the reference clip as both the voice
     (s3gen prompt) and the speech-style prompt (T3, first 6 s), so delivery comes from the reference
  3. pace: the reference's speaking rate (characters per second) is measured, and cfg_weight is steered
     per sentence toward it (lower cfg_weight = slower; measured 0.5 -> 0.2: about -12 %)
  4. pauses: between sentences, the reference speaker's typical pause; between paragraphs (original
     pauses >= PARA_GAP_S), the original pause, so the structure of the recording survives

The result no longer lines up with the original's timestamps: it is as long as the reference speaker
would take. Nothing is time-stretched or pitch-shifted (rule: no output post-processing).
"""
from __future__ import annotations

import math
import threading
import time
from pathlib import Path

import numpy as np

from . import audio
from .engine import SR, GenParams, Engine, _trim_chunk, split_text, word_count
from .languages import CJK, LANGUAGES

ASR_REPO = "mlx-community/whisper-large-v3-turbo"
ASR_FILES = ["config.json", "weights.safetensors"]
PARA_GAP_S = 1.5            # an original pause this long starts a new paragraph (kept as-is, capped)
PARA_GAP_MAX_S = 4.0
CFG_MIN, CFG_MAX, CFG_START = 0.2, 0.8, 0.5
CFG_GAIN = 1.2              # cfg change per unit of log pace error


def asr_present() -> bool:
    return asr_path() is not None


def asr_path() -> str | None:
    try:
        from huggingface_hub import try_to_load_from_cache

        paths = [try_to_load_from_cache(ASR_REPO, f) for f in ASR_FILES]
        if all(isinstance(p, str) and Path(p).exists() for p in paths):
            return str(Path(paths[0]).parent)
    except Exception:
        pass
    return None


_asr_lock = threading.Lock()


def transcribe(x16: np.ndarray) -> dict:
    """Local transcription. Returns {"language", "segments": [{"start","end","text"}]}."""
    import mlx_whisper

    p = asr_path()
    if not p:
        raise RuntimeError("The transcription model isn't downloaded yet.")
    with _asr_lock:
        r = mlx_whisper.transcribe(x16.astype(np.float32), path_or_hf_repo=p, condition_on_previous_text=False)
    segs = [{"start": float(s["start"]), "end": float(s["end"]), "text": s["text"].strip()} for s in r["segments"]
            if s["text"].strip()]
    return {"language": r.get("language") or "en", "segments": segs}


def pace_of(text: str, x: np.ndarray, sr: int) -> float:
    """Characters per second over the spoken stretch (edges trimmed, inner pauses included)."""
    y = audio.trim_silence(x, sr, pre=0.0, post=0.0)
    return len(" ".join(text.split())) / max(len(y) / sr, 0.3)


def typical_pause(x: np.ndarray, sr: int) -> float:
    """Median pause between phrases (150 ms to 1.5 s) in the reference, or 0.35 s if none."""
    m, _ = audio.speech_mask(x, sr)
    gaps = [(j - i) * audio.FRAME_S for v, i, j in audio._runs(m) if not v and i > 0 and j < len(m)]
    gaps = [g for g in gaps if 0.15 <= g <= 1.5]
    return float(np.clip(np.median(gaps), 0.15, 1.2)) if gaps else 0.35


def paragraphs(segments: list[dict]) -> list[dict]:
    """Merge transcript segments into paragraphs separated by long original pauses."""
    out = []
    for s in segments:
        if out and s["start"] - out[-1]["end"] < PARA_GAP_S:
            out[-1]["text"] += " " + s["text"]
            out[-1]["end"] = s["end"]
        else:
            gap = s["start"] - (out[-1]["end"] if out else 0.0)
            out.append({"start": s["start"], "end": s["end"], "text": s["text"], "gap_before": max(0.0, gap)})
    return out


class Cancelled(Exception):
    pass


def respeak(eng: Engine, src16: np.ndarray, ref_path: Path, ref24: np.ndarray, progress=None,
            cancel: threading.Event | None = None, seed: int = 7) -> tuple[np.ndarray, dict]:
    """Transcribe src16 and speak it again in the voice AND style of the reference clip."""
    if progress:
        progress({"stage": "transcribing", "fraction": 0.0})
    tr = transcribe(src16)
    lang = tr["language"] if tr["language"] in LANGUAGES else None
    if lang is None:
        raise ValueError(f"The recording seems to be in a language this app can't speak ({tr['language']}).")
    if not tr["segments"]:
        raise ValueError("No words were recognised in the recording.")
    import librosa

    ref_tr = transcribe(librosa.resample(ref24, orig_sr=SR, target_sr=16000))
    ref_text = " ".join(s["text"] for s in ref_tr["segments"])
    target = pace_of(ref_text, ref24, SR) if len(ref_text) > 20 else None
    pause = typical_pause(ref24, SR)
    orig_pace = pace_of(" ".join(s["text"] for s in tr["segments"]), src16, 16000)

    paras = paragraphs(tr["segments"])
    plan = []  # (paragraph index, chunk text)
    for k, p in enumerate(paras):
        for ch in split_text(p["text"], lang) or [p["text"]]:
            plan.append((k, ch))
    eng.load()
    if lang != "en":
        eng.load_mtl()
    total_chars = sum(len(c) for _, c in plan) or 1
    parts, stats, done_chars = [], [], 0
    cfg, said_chars, said_s = CFG_START, 0, 0.0
    t0 = time.time()
    with eng.lock:
        p0 = GenParams()
        conds = eng.conditionals(ref_path, ref_path, p0)
        for i, (k, text) in enumerate(plan):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            p = GenParams(cfg_weight=round(cfg, 3), exaggeration=p0.exaggeration, temperature=p0.temperature)
            w, meta = eng._gen_chunk(conds, text, p, seed + i, lang)
            if meta["hit_cap"]:  # degenerate run: one retry with another seed
                w2, meta2 = eng._gen_chunk(conds, text, p, seed + 1000 + i, lang)
                if not meta2["hit_cap"] or len(w2) < len(w):
                    w, meta = w2, {**meta2, "retried": True}
            w = _trim_chunk(w)
            first_in_para = i == 0 or plan[i - 1][0] != k
            gap = (min(paras[k]["gap_before"], PARA_GAP_MAX_S) if i > 0 else 0.0) if first_in_para else pause
            if parts:
                parts.append(np.zeros(int(gap * SR), np.float32))
            parts.append(w)
            chunk_pace = pace_of(text, w, SR)
            said_chars += len(text)
            said_s += len(audio.trim_silence(w, SR, pre=0.0, post=0.0)) / SR
            if target:  # steer the next sentence's pace toward the reference (integral, damped)
                err = 0.5 * math.log(chunk_pace / target) + 0.5 * math.log((said_chars / max(said_s, 0.3)) / target)
                cfg = float(np.clip(cfg - CFG_GAIN * err, CFG_MIN, CFG_MAX))
            stats.append({"text": text, "cfg_weight": p.cfg_weight, "pace": round(chunk_pace, 2),
                          "seconds": round(len(w) / SR, 2), "hit_cap": meta["hit_cap"]})
            done_chars += len(text)
            if progress:
                spent = time.time() - t0
                frac = done_chars / total_chars
                progress({"stage": "speaking", "done": i + 1, "total": len(plan), "fraction": frac,
                          "elapsed": round(spent, 1), "eta": round(spent / frac * (1 - frac), 1)})
    out = np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, np.float32)
    joiner = "" if lang in CJK else " "
    return out, {"language": lang, "text": joiner.join(c for _, c in plan), "words": word_count(joiner.join(c for _, c in plan), lang),
                 "seconds": round(len(out) / SR, 2), "original_seconds": round(len(src16) / 16000, 2),
                 "pace_reference": None if target is None else round(target, 2),
                 "pace_original": round(orig_pace, 2),
                 "pace_result": round(said_chars / max(said_s, 0.3), 2), "pause_reference": round(pause, 2),
                 "chunks": stats, "compute_seconds": round(time.time() - t0, 1)}
