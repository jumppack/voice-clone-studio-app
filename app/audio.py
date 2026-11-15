"""Audio helpers: WAV I/O, take analysis, silence trim, tape-shift, export.

No model code here. Everything is numpy / soundfile / ffmpeg.
"""
from __future__ import annotations

import io
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

FRAME_S = 0.03  # analysis frame length (30 ms)


def db(x: float) -> float:
    return float(20 * np.log10(max(x, 1e-9)))


def read_wav_bytes(data: bytes) -> tuple[np.ndarray, int, dict]:
    """Decode an uploaded WAV. Rejects anything that is not uncompressed PCM/float."""
    info = sf.info(io.BytesIO(data))
    if info.format != "WAV" or info.subtype not in ("PCM_16", "PCM_24", "PCM_32", "FLOAT"):
        raise ValueError(f"Expected uncompressed WAV, got {info.format}/{info.subtype}")
    x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return x.mean(axis=1), sr, {"subtype": info.subtype, "channels": info.channels, "sr": sr}


def write_wav(path: Path, x: np.ndarray, sr: int, subtype: str = "PCM_16") -> None:
    sf.write(str(path), np.clip(x, -1, 1), sr, subtype=subtype)


def frame_rms_db(x: np.ndarray, sr: int, frame_s: float = FRAME_S) -> np.ndarray:
    n = max(1, int(sr * frame_s))
    k = len(x) // n
    if k == 0:
        return np.array([db(np.sqrt(np.mean(x**2)) if len(x) else 0)])
    f = x[: k * n].reshape(k, n)
    return 20 * np.log10(np.sqrt(np.mean(f**2, axis=1)) + 1e-9)


def speech_mask(x: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """Energy VAD. Returns per-frame bool mask and the estimated noise floor (dBFS)."""
    e = frame_rms_db(x, sr)
    floor = float(np.percentile(e, 10))
    thr = max(floor + 12, -48.0)
    m = e > thr
    # close short gaps (< 300 ms) and drop blips (< 90 ms)
    m = _close(m, int(0.3 / FRAME_S))
    m = _open(m, int(0.09 / FRAME_S))
    return m, floor


def _runs(m: np.ndarray):
    i, n = 0, len(m)
    while i < n:
        j = i
        while j < n and m[j] == m[i]:
            j += 1
        yield m[i], i, j
        i = j


def _close(m, gap):
    m = m.copy()
    for v, i, j in list(_runs(m)):
        if not v and i > 0 and j < len(m) and (j - i) < gap:
            m[i:j] = True
    return m


def _open(m, min_len):
    m = m.copy()
    for v, i, j in list(_runs(m)):
        if v and (j - i) < min_len:
            m[i:j] = False
    return m


def trim_silence(x: np.ndarray, sr: int, pre: float = 0.15, post: float = 0.3) -> np.ndarray:
    m, _ = speech_mask(x, sr)
    if not m.any():
        return x
    idx = np.flatnonzero(m)
    a = max(0, int((idx[0] * FRAME_S - pre) * sr))
    b = min(len(x), int(((idx[-1] + 1) * FRAME_S + post) * sr))
    out = x[a:b].copy()
    f = min(int(0.01 * sr), len(out) // 4)  # 10 ms fades so the cut never clicks
    if f > 0:
        out[:f] *= np.linspace(0, 1, f)
        out[-f:] *= np.linspace(1, 0, f)
    return out


def reverb_tail_db_per_100ms(x: np.ndarray, sr: int) -> float | None:
    """Rough echo detector: median energy decay rate right after speech offsets.

    Dry close-mic speech drops > ~25 dB within 100 ms of a word ending; a reverberant
    room decays much more slowly. Cheap and approximate, so the app words it as a hint.
    """
    e = frame_rms_db(x, sr, 0.01)
    floor = np.percentile(e, 10)
    m = e > max(floor + 15, -45)
    rates = []
    for i in range(1, len(m)):
        if m[i - 1] and not m[i]:
            seg = e[i - 1 : i + 12]  # ~120 ms after the offset
            if len(seg) < 12:
                continue
            peak = seg[0]
            target = max(floor + 3, peak - 30)
            hit = np.flatnonzero(seg <= target)
            t = (hit[0] if len(hit) else len(seg)) * 0.01
            rates.append((peak - max(seg.min(), target)) / max(t, 0.01) * 0.1)
    return float(np.median(rates)) if len(rates) >= 3 else None


def analyze_take(x: np.ndarray, sr: int, room_floor: float | None = None, source: str = "take") -> dict:
    """Checks for a recorded take. Returns metrics, a 0-100 score and plain-language notes.

    room_floor: noise floor measured by the 2 s room check (preferred). The in-take estimate
    (10th percentile of 30 ms frames) is lifted by reverb tails, so it is only a fallback."""
    peak = db(float(np.max(np.abs(x))) if len(x) else 0)
    clipped = int(np.sum(np.abs(x) >= 0.891))  # -1 dBFS
    m, floor = speech_mask(x, sr)
    e = frame_rms_db(x, sr)
    speech_s = float(m.sum() * FRAME_S)
    level = float(np.median(e[: len(m)][m])) if m.any() else -90.0
    take_floor = floor
    if room_floor is not None:
        floor = float(room_floor)
    rev = reverb_tail_db_per_100ms(x, sr) if floor < -45 else None  # noise hides the decay
    snr = level - floor

    issues, score = [], 100.0
    if clipped > 0:
        issues.append({"kind": "clipping", "level": "bad",
                       "text": f"Too loud: the peak hit {peak:.1f} dBFS, so it distorted. Move back a little or lower the input volume."})
        score -= min(40, 10 + clipped / 50)
    quiet_s = float((~m).sum() * FRAME_S)
    floor_known = room_floor is not None or quiet_s >= 1.0  # with no pauses the "floor" is just speech
    if floor > -50 and floor_known:
        issues.append({"kind": "noise", "level": "bad" if floor > -40 else "warn",
                       "text": f"Background noise is high ({floor:.0f} dBFS). Turn off fans or move somewhere quieter."})
        score -= min(35, (floor + 50) * 2.5)
    if level < -35:
        issues.append({"kind": "quiet", "level": "bad" if level < -42 else "warn",
                       "text": "Too quiet. Move closer to the mic or raise the input volume."})
        score -= min(30, (-35 - level) * 3)
    if source == "upload":  # a clip cut from the user's file: short is allowed, just less accurate
        if speech_s < UPLOAD_MIN_SPEECH_S:
            issues.append({"kind": "short", "level": "bad",
                           "text": f"Only {speech_s:.1f} s of speech in this part. Too little to copy a voice from."})
            score -= 60
        elif speech_s < 10:
            issues.append({"kind": "short", "level": "warn",
                           "text": f"Short sample ({speech_s:.1f} s of speech). It works, but 10 s or more sounds closer to the speaker."})
            score -= 10
    elif speech_s < 6:
        issues.append({"kind": "short", "level": "bad",
                       "text": f"Only {speech_s:.1f} s of speech was found. Read the whole script."})
        score -= 40
    elif speech_s < 8:
        issues.append({"kind": "short", "level": "warn",
                       "text": f"A bit short ({speech_s:.1f} s of speech). Aim for about 10 s or more."})
        score -= 15
    if rev is not None and rev < 12:
        issues.append({"kind": "echo", "level": "warn",
                       "text": "The room sounds echoey. Soft furnishings, a closet or a closer mic help."})
        score -= 15
    if not issues:
        issues.append({"kind": "ok", "level": "good", "text": "Sounds clean. Good take."})
    score = float(max(0, min(100, score)))
    verdict = "good" if score >= 75 else ("okay" if score >= 50 else "redo")
    return {
        "peak_dbfs": round(peak, 1), "clipped_samples": clipped, "noise_floor_dbfs": round(floor, 1), "in_take_floor_dbfs": round(take_floor, 1),
        "speech_level_dbfs": round(level, 1), "snr_db": round(snr, 1), "speech_seconds": round(speech_s, 1),
        "duration": round(len(x) / sr, 2), "reverb_decay_db_per_100ms": None if rev is None else round(rev, 1),
        "score": round(score), "verdict": verdict, "issues": issues,
    }


def tape_shift(x: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """'Tape' pitch change: plain resampling, so pitch and speed change together.

    No vocoder / no time-stretch. Only ever applied to the REFERENCE clip.
    """
    if abs(semitones) < 1e-3:
        return x
    import librosa

    factor = 2.0 ** (semitones / 12.0)  # >1 = higher and faster
    return librosa.resample(x, orig_sr=sr * factor, target_sr=sr, res_type="soxr_vhq").astype(np.float32)


def peaks(x: np.ndarray, n: int = 600) -> list[float]:
    """Waveform overview for the UI: max |x| per bucket."""
    if len(x) == 0:
        return []
    k = max(1, len(x) // n)
    f = np.abs(x[: (len(x) // k) * k]).reshape(-1, k).max(axis=1)
    return [round(float(v), 3) for v in f]


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg") or next((p for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg") if Path(p).exists()), None)


def has_mp3lame(ff: str | None) -> bool:
    if not ff:
        return False
    try:
        out = subprocess.run([ff, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=10).stdout
        return "libmp3lame" in out
    except Exception:
        return False


def export_mp3(wav_path: Path, out_path: Path, kbps: int) -> None:
    ff = ffmpeg_path()
    if not ff:
        raise RuntimeError("ffmpeg is not installed")
    subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav_path),
                    "-codec:a", "libmp3lame", "-b:a", f"{kbps}k", "-ar", "48000", "-ac", "1",  # MPEG-2 (24 kHz) tops out at 160 kbps
                    str(out_path)], check=True, timeout=300)


def room_noise(x: np.ndarray, sr: int) -> dict:
    """2 s room check before the first take: median frame level of silence."""
    e = frame_rms_db(x, sr)
    floor = float(np.median(e))
    if floor > -40:
        v, t = "bad", f"Noisy room ({floor:.0f} dBFS). Recordings will carry this noise into the cloned voice."
    elif floor > -50:
        v, t = "warn", f"A bit of background noise ({floor:.0f} dBFS). Usable, but quieter is better."
    else:
        v, t = "good", f"Quiet room ({floor:.0f} dBFS). Good to go."
    return {"noise_floor_dbfs": round(floor, 1), "level": v, "text": t}


# ---------------- uploaded voice files ----------------
UPLOAD_MAX_BYTES = 200 * 1024 * 1024   # hard limit; the UI suggests under 30 MB
UPLOAD_SUGGESTED_BYTES = 30 * 1024 * 1024
UPLOAD_ANALYZE_SECONDS = 10 * 60        # any length is accepted; only the first 10 minutes are used
UPLOAD_MIN_SPEECH_S = 3.0
UPLOAD_SR = 48000


def decode_upload(data: bytes, name: str = "") -> tuple[np.ndarray, int, dict]:
    """Decode an uploaded WAV / MP3 (or anything else ffmpeg reads) to mono float32.
    Uncompressed WAV is read directly; everything else is decoded locally with ffmpeg."""
    try:
        info = sf.info(io.BytesIO(data))
        if info.format == "WAV" and info.subtype in ("PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE", "PCM_U8"):
            x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True, frames=int(UPLOAD_ANALYZE_SECONDS * info.samplerate))
            return x.mean(axis=1), sr, {"format": "WAV", "full_seconds": round(info.duration, 1), "subtype": info.subtype, "channels": info.channels, "sr": sr}
    except Exception:
        pass
    ff = ffmpeg_path()
    if not ff:
        raise ValueError("ffmpeg is needed to read MP3 files (see Setup)")
    r = subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-vn", "-ac", "1",
                        "-ar", str(UPLOAD_SR), "-t", str(UPLOAD_ANALYZE_SECONDS), "-f", "f32le", "pipe:1"],
                       input=data, capture_output=True, timeout=300)
    if r.returncode != 0 or not r.stdout:
        msg = r.stderr.decode(errors="replace").strip().splitlines()
        raise ValueError("not a readable audio file" + (f" ({msg[-1]})" if msg else ""))
    x = np.frombuffer(r.stdout, dtype="<f4").astype(np.float32)
    probe = subprocess.run([ff, "-hide_banner", "-i", "pipe:0"], input=data[:4 * 1024 * 1024], capture_output=True, timeout=30)
    line = next((l.strip() for l in probe.stderr.decode(errors="replace").splitlines() if "Audio:" in l), "")
    return x, UPLOAD_SR, {"format": (name.rsplit(".", 1)[-1].upper() if "." in name else "audio"), "decoded_by": "ffmpeg",
                          "source_stream": line.split("Audio:", 1)[-1].strip()[:120], "sr": UPLOAD_SR}


def split_for_takes(x: np.ndarray, sr: int, min_s: float = 7.0, target_s: float = 13.0, max_s: float = 15.0,
                    max_clips: int = 6) -> list[tuple[int, int]]:
    """Cut a long recording into take-sized clips (min_s..max_s of audio), cutting only in pauses.
    Returns sample ranges, in time order, at most max_clips (the ones with the most speech)."""
    m, _ = speech_mask(x, sr)
    n = max(1, int(sr * FRAME_S))
    if not m.any():
        return []
    # pause centres (frames) that are long enough to cut in without clipping a word
    cuts = [(i + j) // 2 for v, i, j in _runs(m) if not v and (j - i) * FRAME_S >= 0.25]
    speech_runs = [(i, j) for v, i, j in _runs(m) if v]
    start = start_all = max(0, speech_runs[0][0] - int(0.15 / FRAME_S))
    end_all = min(len(m), speech_runs[-1][1] + int(0.3 / FRAME_S))
    clips = []
    while start < end_all:
        lo, hi, tgt = start + int(min_s / FRAME_S), start + int(max_s / FRAME_S), start + int(target_s / FRAME_S)
        options = [c for c in cuts if lo <= c <= hi]
        if end_all - start <= int(max_s / FRAME_S):
            stop = end_all
        elif options:
            stop = min(options, key=lambda c: abs(c - tgt))
        else:  # no pause in range (continuous speech): cut at the quietest frame near the target
            e = frame_rms_db(x, sr)
            window = range(lo, min(hi, len(e)))
            stop = min(window, key=lambda k: e[k]) if len(window) else hi
        if (stop - start) * FRAME_S >= min_s and m[start:stop].sum() * FRAME_S >= min_s * 0.6:
            clips.append((start, stop))
        start = stop
    if not clips and m.sum() * FRAME_S >= UPLOAD_MIN_SPEECH_S:  # short recording: use all of it as one clip
        clips = [(start_all, min(end_all, start_all + int(max_s / FRAME_S)))]
    if len(clips) > max_clips:  # keep the clips with the most speech, then restore time order
        clips = sorted(sorted(clips, key=lambda c: -m[c[0]:c[1]].sum())[:max_clips])
    return [(a * n, min(len(x), b * n)) for a, b in clips]
