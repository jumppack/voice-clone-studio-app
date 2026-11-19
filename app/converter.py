"""Voice Changer engine: re-voice a whole recording with the voice of a short reference clip.

Uses Chatterbox's speech-to-speech path (chatterbox/vc.py, 0.1.7): the S3 tokenizer turns the
original speech into speech tokens (what is said and how, 25 tokens/s), and S3Gen renders those
tokens in the reference speaker's voice (s3gen.embed_ref: x-vector + mel prompt). The voice change
happens entirely at that conditioning stage: nothing is pitch-shifted or vocoded afterwards.
Output timing matches the original exactly (2 mel frames per token), so it stays in sync with video.

Long files: the original is cut into pieces of ~20 s in pauses (one model call per piece);
silences of 2 s or more are kept as silence and not sent to the model. Each piece is placed back
at its original time, so the result has the original length.

Measured (tests/changer_test.py, 48 s source, MPS):
  * reference length: 5 s -> similarity 0.885, 10 s -> 0.896, 20 s -> 0.910, 30 s -> 0.922 (each
    piece takes longer with a longer reference) -> the whole reference is used, up to REF_MAX_S
  * flow-matching steps 4 / 10 / 20 -> 0.893 / 0.896 / 0.890: no measurable gain, library default kept
  * piece length 10 / 20 / 30 / 48 s -> 0.902 / 0.890 / 0.890 / 0.878
"""
from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import audio

SR_OUT = 24000         # S3Gen output rate
SR_TOK = 16000         # S3 tokenizer rate
FRAME = audio.FRAME_S  # 30 ms analysis frames (480 samples @16k, 720 @24k: exact in both)
PIECE_TARGET_S, PIECE_MIN_S, PIECE_MAX_S = 20.0, 8.0, 30.0
LONG_PAUSE_S = 2.0     # pauses this long are kept as silence and skipped
CUT_PAUSE_S = 0.25     # shortest pause a piece may be cut in
REF_MAX_S = 30.0
REF_MIN_SPEECH_S = 3.0
MODEL_FILES = ["s3gen.safetensors", "conds.pt", "ve.safetensors"]
GAIN_LIMIT_DB = 12.0


def weights_present() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return all(isinstance(v := try_to_load_from_cache("ResembleAI/chatterbox", f), str) and Path(v).exists()
                   for f in MODEL_FILES)
    except Exception:
        return False


def decode(path: Path, sr: int, max_s: float | None = None) -> np.ndarray:
    """Any file ffmpeg can read (WAV, MP3, M4A, FLAC, even video) -> mono float32 at sr."""
    ff = audio.ffmpeg_path()
    if not ff:
        raise ValueError("ffmpeg is needed to read audio files. Install it with: brew install ffmpeg")
    cmd = [ff, "-hide_banner", "-loglevel", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr)]
    if max_s:
        cmd += ["-t", str(max_s)]
    r = subprocess.run(cmd + ["-f", "f32le", "pipe:1"], capture_output=True, timeout=1800)
    if r.returncode != 0 or not r.stdout:
        msg = r.stderr.decode(errors="replace").strip().splitlines()
        raise ValueError("That isn't a readable audio file" + (f" ({msg[-1]})" if msg else "."))
    return np.frombuffer(r.stdout, dtype="<f4").astype(np.float32)


@dataclass
class Piece:
    a: int          # start frame (30 ms frames)
    b: int          # end frame (exclusive)
    speech: bool    # False: kept as silence, not converted

    @property
    def seconds(self) -> float:
        return (self.b - self.a) * FRAME


def plan(x16: np.ndarray) -> list[Piece]:
    """Cut the timeline into pieces: speech pieces of PIECE_MIN_S..PIECE_MAX_S cut in pauses, and the
    long pauses (>= LONG_PAUSE_S) between them as silent pieces. Pieces cover every frame, in order."""
    m, _ = audio.speech_mask(x16, SR_TOK)
    n = len(m)
    if n == 0:
        return []
    if not m.any():
        return [Piece(0, n, False)]
    e = audio.frame_rms_db(x16, SR_TOK)
    runs = list(audio._runs(m))
    pre, post = int(0.15 / FRAME), int(0.30 / FRAME)
    # speech islands separated by long pauses (with a little padding kept on both sides of speech)
    islands, cur = [], None
    for v, i, j in runs:
        if not v:
            continue
        a, b = max(0, i - pre), min(n, j + post)
        if cur and a - cur[1] < int(LONG_PAUSE_S / FRAME):
            cur[1] = b
        else:
            if cur:
                islands.append(cur)
            cur = [a, b]
    islands.append(cur)
    cuts = [(i + j) // 2 for v, i, j in runs if not v and (j - i) * FRAME >= CUT_PAUSE_S]
    out, t = [], 0
    for a, b in islands:
        if a > t:
            out.append(Piece(t, a, False))
        start = a
        while start < b:
            lo, hi = start + int(PIECE_MIN_S / FRAME), start + int(PIECE_MAX_S / FRAME)
            tgt = start + int(PIECE_TARGET_S / FRAME)
            if b <= hi:
                stop = b
            else:
                options = [c for c in cuts if lo <= c <= hi]
                if options:
                    stop = min(options, key=lambda c: abs(c - tgt))
                else:  # no pause long enough: cut at the quietest frame in range
                    stop = min(range(lo, hi), key=lambda k: e[k] if k < len(e) else 0)
            out.append(Piece(start, stop, True))
            start = stop
        t = b
    if t < n:
        out.append(Piece(t, n, False))
    # a tail piece shorter than 1 s merges into the previous speech piece (if that stays <= max)
    merged: list[Piece] = []
    for p in out:
        if (merged and p.speech and merged[-1].speech and p.seconds < 1.0
                and merged[-1].seconds + p.seconds <= PIECE_MAX_S):
            merged[-1].b = p.b
        else:
            merged.append(p)
    return merged


def _speech_level_db(x: np.ndarray, sr: int) -> float | None:
    m, _ = audio.speech_mask(x, sr)
    if not m.any():
        return None
    e = audio.frame_rms_db(x, sr)[: len(m)]
    return float(np.percentile(e[m], 75))


def _fade(x: np.ndarray, n: int) -> np.ndarray:
    n = min(n, len(x) // 2)
    if n > 0:
        x[:n] *= np.linspace(0, 1, n, dtype=np.float32)
        x[-n:] *= np.linspace(1, 0, n, dtype=np.float32)
    return x


class Cancelled(Exception):
    pass


class Converter:
    """Uses the text-to-speech Engine's already-loaded S3Gen, VoiceEncoder and watermarker when given
    one (the combined app), so the models are in memory once and share the engine's GPU lock."""

    def __init__(self, engine=None):
        self.engine = engine
        self.s3gen = self.ve = self.wm = None
        self.device = None
        self.state = "not_loaded"
        self.error = None
        self.lock = engine.lock if engine is not None else threading.Lock()  # one GPU job at a time
        self._load_lock = threading.Lock()
        self.speed = 1.0               # measured / modelled compute time, from the last long run

    def load(self):
        if self.s3gen is not None:
            return
        with self._load_lock:
            if self.s3gen is not None:
                return
            self.state = "loading"
            try:
                if self.engine is not None:
                    self.engine.load()
                    m = self.engine.model
                    self.s3gen, self.ve, self.wm, self.device = m.s3gen, m.ve, m.watermarker, self.engine.device
                else:
                    import torch
                    from huggingface_hub import try_to_load_from_cache
                    from safetensors.torch import load_file
                    from chatterbox.models.voice_encoder import VoiceEncoder
                    from chatterbox.vc import ChatterboxVC

                    if not weights_present():
                        raise RuntimeError("The voice model isn't downloaded yet.")
                    dev = "mps" if torch.backends.mps.is_available() else "cpu"
                    ckpt = Path(try_to_load_from_cache("ResembleAI/chatterbox", "s3gen.safetensors")).parent
                    vc = ChatterboxVC.from_local(ckpt, dev)
                    ve = VoiceEncoder()
                    ve.load_state_dict(load_file(try_to_load_from_cache("ResembleAI/chatterbox", "ve.safetensors")))
                    self.s3gen, self.ve, self.wm, self.device = vc.s3gen, ve.eval(), vc.watermarker, dev
                self.state = "ready"
            except Exception as e:
                self.state, self.error = "error", f"{type(e).__name__}: {e}"
                raise

    def load_async(self):
        def go():
            try:
                self.load()
            except Exception:
                pass
        threading.Thread(target=go, daemon=True).start()

    # ---------- voice match ----------
    def embed(self, x: np.ndarray, sr: int) -> np.ndarray:
        import librosa

        self.load()
        x16 = librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=SR_TOK) if sr != SR_TOK else x
        e = self.ve.embeds_from_wavs([x16], sample_rate=SR_TOK).mean(0)
        return e / (np.linalg.norm(e) + 1e-9)

    # ---------- reference ----------
    @staticmethod
    def prepare_reference(x24: np.ndarray) -> tuple[np.ndarray, dict]:
        """Trim silence at both ends and use at most REF_MAX_S. Returns the clip and its checks."""
        clip = audio.trim_silence(x24, SR_OUT, pre=0.05, post=0.2)[: int(REF_MAX_S * SR_OUT)]
        info = audio.analyze_take(clip, SR_OUT, source="upload")
        info["used_seconds"] = round(len(clip) / SR_OUT, 1)
        info["given_seconds"] = round(len(x24) / SR_OUT, 1)
        return clip, info

    # ---------- conversion ----------
    def convert(self, x16: np.ndarray, ref24: np.ndarray, pieces: list[Piece] | None = None,
                progress=None, cancel: threading.Event | None = None) -> tuple[np.ndarray, dict]:
        """Re-voice x16 (mono, 16 kHz) with the voice of ref24 (mono, 24 kHz). Returns 24 kHz audio of
        exactly the original length, plus per-piece stats."""
        import torch

        self.load()
        pieces = plan(x16) if pieces is None else pieces
        n_out = int(round(len(x16) * SR_OUT / SR_TOK))
        out = np.zeros(n_out, np.float32)
        speech = [p for p in pieces if p.speech]
        total_s = sum(p.seconds for p in speech) or 1.0
        stats, done_s, t0 = [], 0.0, time.time()
        ref_emb = self.embed(ref24, SR_OUT)
        with self.lock, torch.inference_mode():
            ref_dict = self.s3gen.embed_ref(ref24, SR_OUT, device=self.device)
            for k, p in enumerate(speech):
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                i16, j16 = p.a * int(FRAME * SR_TOK), min(len(x16), p.b * int(FRAME * SR_TOK))
                i24, j24 = p.a * int(FRAME * SR_OUT), min(n_out, p.b * int(FRAME * SR_OUT))
                src = x16[i16:j16]
                ts = time.time()
                tok, _ = self.s3gen.tokenizer(torch.from_numpy(src).float().to(self.device)[None])
                wav, _ = self.s3gen.inference(speech_tokens=tok, ref_dict=ref_dict)
                y = wav.squeeze(0).detach().cpu().numpy().astype(np.float32)
                y = self.wm.apply_watermark(y, sample_rate=SR_OUT).astype(np.float32)
                need = j24 - i24
                y = y[:need] if len(y) >= need else np.pad(y, (0, need - len(y)))
                # keep the original's loudness per piece (a plain gain, not voice processing)
                ls, ly = _speech_level_db(src, SR_TOK), _speech_level_db(y, SR_OUT)
                gain_db = 0.0 if ls is None or ly is None else float(np.clip(ls - ly, -GAIN_LIMIT_DB, GAIN_LIMIT_DB))
                y = _fade(y * np.float32(10 ** (gain_db / 20)), int(0.008 * SR_OUT))
                out[i24:j24] = y
                sim = float(np.dot(self.embed(y, SR_OUT), ref_emb)) if ly is not None else None
                el = time.time() - ts
                done_s += p.seconds
                stats.append({"start": round(p.a * FRAME, 2), "end": round(p.b * FRAME, 2), "tokens": int(tok.shape[-1]),
                              "seconds_compute": round(el, 2), "similarity": None if sim is None else round(sim, 3),
                              "gain_db": round(gain_db, 1)})
                if progress:
                    spent = time.time() - t0
                    progress({"done": k + 1, "total": len(speech), "fraction": done_s / total_s,
                              "elapsed": round(spent, 1), "eta": round(spent / done_s * (total_s - done_s), 1)})
        peak = float(np.max(np.abs(out))) if len(out) else 0.0
        if peak > 0.98:  # one gain for the whole file if the loudness match pushed it over
            out *= np.float32(0.98 / peak)
        spent = time.time() - t0
        if total_s >= 10:
            self.speed = spent / (total_s * self._rtf(len(ref24) / SR_OUT))
        sims = [s["similarity"] for s in stats if s["similarity"] is not None]
        info = {"pieces": stats, "seconds": round(len(out) / SR_OUT, 2), "speech_seconds": round(total_s, 1),
                "compute_seconds": round(spent, 1),
                "similarity": round(float(np.median(sims)), 3) if sims else None,
                "similarity_min": round(min(sims), 3) if sims else None,
                "low_pieces": [s for s in stats if s["similarity"] is not None and s["similarity"] < 0.75]}
        return out, info

    @staticmethod
    def _rtf(ref_s: float) -> float:
        """Compute seconds per audio second. Measured on Apple Silicon (MPS), 20 s pieces: 0.30 with a
        10 s reference, 0.50 with 20 s, 0.79 with 30 s (the reference is re-read for every piece)."""
        return 0.06 + 0.024 * min(ref_s, REF_MAX_S)

    def estimate(self, speech_s: float, ref_s: float) -> float:
        return round(speech_s * self._rtf(ref_s) * self.speed + 4, 0)
