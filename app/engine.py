"""Chatterbox wrapper: conditioning (tape-shift + speaker blend), chunked generation, similarity.

Rule: the generated speech is never pitch-shifted or vocoded afterwards. All voice changes
happen here, at the conditioning stage, before generation.

Checked against chatterbox-tts 0.1.7 (chatterbox/tts.py):
  * ChatterboxTTS.DEC_COND_LEN = 10 s @24k  -> voice (s3gen.embed_ref)
  * ChatterboxTTS.ENC_COND_LEN = 6 s @16k   -> speech-style prompt tokens (t3)
  * conds.t3.speaker_emb      VoiceEncoder 256-d, unit norm
  * conds.gen["embedding"]    CAMPPlus x-vector 192-d (norm ~14)
  * generate() hard-codes max_new_tokens=1000, so we call t3/s3gen ourselves with a cap.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import audio

SR = 24000
TOKENS_PER_S = 25  # S3 speech tokens per second of audio
PREVIEW_TEXT = "Hi there, this is a quick preview of how my new voice sounds today."
# Preview passage for timed previews (5-30 s): sentences are picked in order, skipping any that would
# move the total further from the target, so the result lands near the chosen length.
PREVIEW_SENTENCES = [
    PREVIEW_TEXT,
    "I recorded a few short samples, and now it can read anything I type.",
    "Some sentences are calm and simple, like this one.",
    "Others carry a little more energy, because that is how people really talk!",
    "Do you think it sounds close enough to the real thing?",
    "Listen for the rhythm, the pauses, and the way each word ends.",
    "The weather has been lovely this week, so I have been taking long walks after dinner.",
    "Yesterday I finally fixed the squeaky door in the kitchen, which felt like a small victory.",
    "If something sounds off, I can record another take and try again.",
    "Good audio starts with a quiet room and a steady distance from the microphone.",
    "Every voice has its own texture, and that is exactly what this tries to keep.",
    "Short lines, long lines, questions and answers all show how well it holds up.",
    "Thanks for listening to this preview.",
    "That is all for now, see you next time.",
]
TARGET_MIN, TARGET_MAX = 5.0, 30.0
OVERSHOOT_S = 1.0  # a sentence may end at most this far past the target (estimated)
DEFAULT_CPS = 15.0  # characters of text per second of speech, until this voice has been measured
MIN_WORDS = 4
MAX_CHUNK_CHARS = 250
CROSSFADE_S = 0.04
PROMPT_SPLICE = os.environ.get("VCS_PROMPT_SPLICE", "1") == "1"
PROMPT_ORDER = os.environ.get("VCS_PROMPT_ORDER", "user_first")
USER_SHRINK = float(os.environ.get("VCS_USER_SHRINK", "1.0"))  # 1.0: user prompt fades out fully at 50%


@dataclass
class GenParams:
    similarity: float = 1.0     # 0.5..1.0  share of the user's speaker embedding
    exaggeration: float = 0.5   # 0.25..1.5
    cfg_weight: float = 0.5     # 0.2..0.8  (lower = slower)
    depth: float = 0.0          # semitones, -3..+3, tape-shift of the reference
    temperature: float = 0.8    # 0.4..1.2
    seed: int | None = None     # fixed seed => repeatable

    @classmethod
    def from_dict(cls, d: dict) -> "GenParams":
        p = cls()
        for k in ("similarity", "exaggeration", "cfg_weight", "depth", "temperature"):
            if d.get(k) is not None:
                setattr(p, k, float(d[k]))
        p.seed = None if d.get("seed") in (None, "", False) else int(d["seed"])
        p.similarity = min(1.0, max(0.5, p.similarity))
        p.exaggeration = min(1.5, max(0.25, p.exaggeration))
        p.cfg_weight = min(0.8, max(0.2, p.cfg_weight))
        p.depth = min(3.0, max(-3.0, p.depth))
        p.temperature = min(1.2, max(0.4, p.temperature))
        return p


def split_text(text: str) -> list[str]:
    """Sentence chunks of <= ~250 chars. Very short sentences are merged with a neighbour
    because Chatterbox clips or garbles lines under ~4 words."""
    text = " ".join(text.split())
    if not text:
        return []
    sents = re.split(r"(?<=[.!?])\s+", text)
    pieces: list[str] = []
    for s in sents:
        while len(s) > MAX_CHUNK_CHARS:  # split long sentences at a comma or space
            cut = max(s.rfind(", ", 0, MAX_CHUNK_CHARS), s.rfind("; ", 0, MAX_CHUNK_CHARS))
            if cut < 60:
                cut = s.rfind(" ", 0, MAX_CHUNK_CHARS)
            cut = cut if cut > 0 else MAX_CHUNK_CHARS
            pieces.append(s[: cut + 1].strip())
            s = s[cut + 1 :].strip()
        if s:
            pieces.append(s)
    chunks: list[str] = []
    for p in pieces:
        if chunks and (len(p.split()) < MIN_WORDS or len(chunks[-1].split()) < MIN_WORDS) \
                and len(chunks[-1]) + len(p) + 1 <= MAX_CHUNK_CHARS:
            chunks[-1] = chunks[-1] + " " + p
        else:
            chunks.append(p)
    if len(chunks) > 1 and len(chunks[-1].split()) < MIN_WORDS:
        last = chunks.pop()
        chunks[-1] += " " + last
    return chunks


def token_cap(chunk: str) -> int:
    """Generous upper bound on speech tokens: slow speech is ~1.6 words/s; x1.6 headroom."""
    words = max(MIN_WORDS, len(chunk.split()))
    est_s = words / 1.6 + 1.0
    return int(min(1000, est_s * 1.6 * TOKENS_PER_S + 25))


def crossfade_join(parts: list[np.ndarray], sr: int = SR, fade_s: float = CROSSFADE_S) -> np.ndarray:
    if not parts:
        return np.zeros(0, np.float32)
    out = parts[0]
    n = int(fade_s * sr)
    for p in parts[1:]:
        k = min(n, len(out), len(p))
        if k == 0:
            out = np.concatenate([out, p])
            continue
        fo = np.cos(np.linspace(0, np.pi / 2, k)) ** 2
        fi = 1 - fo
        out = np.concatenate([out[:-k], out[-k:] * fo + p[:k] * fi, p[k:]])
    return out.astype(np.float32)


def _trim_chunk(x: np.ndarray, sr: int = SR) -> np.ndarray:
    """Trim leading/trailing silence of one chunk to a natural, consistent pause.
    (Cutting silence is not processing the voice.)"""
    return audio.trim_silence(x, sr, pre=0.05, post=0.22)


class Engine:
    def __init__(self, cache_dir: Path):
        self.model = None
        self.device = None
        self.lock = threading.Lock()
        self.state = "not_loaded"  # not_loaded | loading | ready | error
        self.error = None
        self.load_seconds = None
        self.cache_dir = cache_dir
        self._conds_cache: dict[str, object] = {}
        self._ref_embed_cache: dict[str, np.ndarray] = {}
        self.rtf = 1.7  # updated after each generation
        self._rate_file = cache_dir / "speech_rate.json"
        try:
            self.cps = float(json.loads(self._rate_file.read_text())["cps"])
        except (OSError, ValueError, KeyError, TypeError):
            self.cps = DEFAULT_CPS

    # ---------- loading ----------
    def load(self):
        if self.model is not None:
            return
        with self.lock:
            if self.model is not None:
                return
            self.state = "loading"
            t = time.time()
            try:
                import torch
                from chatterbox.tts import ChatterboxTTS

                dev = "mps" if torch.backends.mps.is_available() else "cpu"
                self.model = ChatterboxTTS.from_pretrained(dev)
                self.device = self.model.device
                self.default_conds = self.model.conds  # built-in voice (conds.pt)
                self.state = "ready"
                self.load_seconds = round(time.time() - t, 1)
            except Exception as e:  # surfaced in the setup screen
                self.state, self.error = "error", f"{type(e).__name__}: {e}"
                raise

    def load_async(self):
        threading.Thread(target=lambda: self._safe(self.load), daemon=True).start()

    @staticmethod
    def _safe(fn):
        try:
            fn()
        except Exception:
            pass

    # ---------- conditioning ----------
    def _load_ref(self, path: Path, sr: int) -> np.ndarray:
        import librosa

        x, _ = librosa.load(str(path), sr=sr)
        return x.astype(np.float32)

    def conditionals(self, voice_ref: Path, style_ref: Path, p: GenParams):
        """Build (and cache) Chatterbox conditionals.

        voice_ref -> s3gen voice prompt (first 10 s) + speaker embeddings
        style_ref -> t3 speech-style prompt tokens (first 6 s)
        depth     -> both refs are tape-shifted (resample only) before anything is computed
        similarity-> speaker embeddings lerped toward the built-in default voice
        """
        import torch
        from chatterbox.models.s3tokenizer import S3_SR
        from chatterbox.models.t3.modules.cond_enc import T3Cond
        from chatterbox.tts import Conditionals

        m = self.model
        key = hashlib.sha1(f"{voice_ref}:{voice_ref.stat().st_mtime}:{style_ref}:{style_ref.stat().st_mtime}:"
                           f"{p.depth:.2f}:{p.similarity:.3f}".encode()).hexdigest()
        if key in self._conds_cache:
            c = self._conds_cache[key]
        else:
            v24 = audio.tape_shift(self._load_ref(voice_ref, SR), SR, p.depth)
            s24 = v24 if style_ref == voice_ref else audio.tape_shift(self._load_ref(style_ref, SR), SR, p.depth)
            import librosa

            v16 = librosa.resample(v24, orig_sr=SR, target_sr=S3_SR)
            s16 = librosa.resample(s24, orig_sr=SR, target_sr=S3_SR)

            gen = m.s3gen.embed_ref(v24[: m.DEC_COND_LEN], SR, device=self.device)
            tok, _ = m.s3gen.tokenizer.forward([s16[: m.ENC_COND_LEN]], max_len=m.t3.hp.speech_cond_prompt_len)
            tok = torch.atleast_2d(tok).to(self.device)
            ve = torch.from_numpy(m.ve.embeds_from_wavs([v16], sample_rate=S3_SR)).mean(0, keepdim=True).to(self.device)

            a = p.similarity
            if a < 0.999:
                d = self.default_conds
                # VoiceEncoder embedding: unit-norm lerp
                ve = a * ve + (1 - a) * d.t3.speaker_emb.to(ve)
                ve = ve / ve.norm(dim=-1, keepdim=True)
                # CAMPPlus x-vector (s3gen): lerp direction, keep interpolated length
                xr, xd = gen["embedding"], d.gen["embedding"].to(gen["embedding"])
                mix = a * xr + (1 - a) * xd
                target = a * xr.norm(dim=-1, keepdim=True) + (1 - a) * xd.norm(dim=-1, keepdim=True)
                gen["embedding"] = mix / mix.norm(dim=-1, keepdim=True) * target
                if PROMPT_SPLICE:
                    gen = self._splice_prompt(gen, d.gen, a)

            t3c = T3Cond(speaker_emb=ve, cond_prompt_speech_tokens=tok,
                         emotion_adv=0.5 * torch.ones(1, 1, 1)).to(device=self.device)
            c = Conditionals(t3c, gen)
            if len(self._conds_cache) > 16:
                self._conds_cache.pop(next(iter(self._conds_cache)))
            self._conds_cache[key] = c
        return c

    def _splice_prompt(self, gen: dict, dgen: dict, a: float) -> dict:
        """Mix the s3gen acoustic prompt in time. w = (1-a)/0.5 runs 0 (at 100%) to 1 (at 50%):
        the user's prompt is shortened to (1-w) and followed by w of the built-in voice's prompt.
        Tokens and mels stay aligned (2 mel frames per token). Measured (tests/similarity_experiment*.py):
        embedding lerp alone moved similarity only 0.94 -> 0.91; this gives 0.94 -> 0.81."""
        import torch

        w = min(1.0, max(0.0, (1 - a) / 0.5))
        ut, uf = gen["prompt_token"], gen["prompt_feat"]
        dt, df = dgen["prompt_token"].to(ut), dgen["prompt_feat"].to(uf)
        kd = int(round(w * dt.shape[1]))
        ku = int(round((1 - USER_SHRINK * w) * ut.shape[1]))
        parts_t = [dt[:, :kd], ut[:, :ku]] if PROMPT_ORDER == "default_first" else [ut[:, :ku], dt[:, :kd]]
        parts_f = [df[:, : 2 * kd], uf[:, : 2 * ku]] if PROMPT_ORDER == "default_first" else [uf[:, : 2 * ku], df[:, : 2 * kd]]
        tok = torch.cat(parts_t, dim=1)
        feat = torch.cat(parts_f, dim=1)
        return {**gen, "prompt_token": tok, "prompt_token_len": torch.tensor([tok.shape[1]], device=tok.device),
                "prompt_feat": feat}

    # ---------- generation ----------
    def _gen_chunk(self, conds, text: str, p: GenParams, seed: int | None) -> tuple[np.ndarray, dict]:
        import torch
        import torch.nn.functional as F
        from chatterbox.models.s3tokenizer import drop_invalid_tokens
        from chatterbox.models.t3.modules.cond_enc import T3Cond
        from chatterbox.tts import punc_norm

        m = self.model
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed % (2**32))
        t3c = T3Cond(speaker_emb=conds.t3.speaker_emb, cond_prompt_speech_tokens=conds.t3.cond_prompt_speech_tokens,
                     emotion_adv=p.exaggeration * torch.ones(1, 1, 1)).to(device=self.device)
        tt = m.tokenizer.text_to_tokens(punc_norm(text)).to(self.device)
        tt = torch.cat([tt, tt], dim=0)  # CFG needs two rows (cfg_weight is always > 0 here)
        tt = F.pad(tt, (1, 0), value=m.t3.hp.start_text_token)
        tt = F.pad(tt, (0, 1), value=m.t3.hp.stop_text_token)
        cap = token_cap(text)
        with torch.inference_mode():
            st = m.t3.inference(t3_cond=t3c, text_tokens=tt, max_new_tokens=cap, temperature=p.temperature,
                                cfg_weight=p.cfg_weight, repetition_penalty=1.2, min_p=0.05, top_p=1.0)
            n_raw = int(st.shape[-1])
            st = drop_invalid_tokens(st[0])
            st = st[st < 6561].to(self.device)
            wav, _ = m.s3gen.inference(speech_tokens=st, ref_dict=conds.gen)
            wav = wav.squeeze(0).detach().cpu().numpy()
        wav = m.watermarker.apply_watermark(wav, sample_rate=SR)  # Perth watermark, kept on purpose
        return np.asarray(wav, np.float32), {"tokens": n_raw, "cap": cap, "hit_cap": n_raw >= cap}

    def generate(self, text: str, voice_ref: Path, style_ref: Path, p: GenParams, progress=None,
                 target: float | None = None, sentences: list[str] | None = None) -> tuple[np.ndarray, dict]:
        """Speak `text` (split into sentence chunks), or pick from `sentences` (preview passage).

        target (5-30 s): sentences are taken in order while each one brings the total closer to the
        target; user text is never reordered (the rest is reported as not spoken), the preview
        passage may skip a sentence that would overshoot. Length comes only from how much text is
        spoken; the audio itself is never stretched."""
        self.load()
        pool = sentences is not None
        chunks = list(sentences) if pool else split_text(text)
        if not chunks or (not pool and len(text.split()) < MIN_WORDS):
            raise ValueError(f"Please type at least {MIN_WORDS} words.")
        t0 = time.time()
        with self.lock:
            conds = self.conditionals(voice_ref, style_ref, p)
            parts, info = [], []
            remaining, cum, chars_done = list(chunks), 0.0, 0

            def cps_now():  # this run's measured pace, blended with the voice's history
                return (chars_done / cum * 0.7 + self.cps * 0.3) if cum > 0 else self.cps

            def pick(total, rem, rate):
                if target is None:
                    return 0 if rem else None
                if total == 0:
                    return 0 if rem else None
                best, best_err = None, abs(total - target)
                for j, c in enumerate(rem if pool else rem[:1]):  # user text: next sentence only
                    end = total + len(c) / rate
                    if end <= target + OVERSHOOT_S and abs(end - target) < best_err:
                        best, best_err = j, abs(end - target)
                return best

            def planned():  # chunks still expected, for the progress bar
                n, tot, rem = 0, cum, list(remaining)
                while (j := pick(tot, rem, cps_now())) is not None:
                    tot += len(rem.pop(j)) / cps_now(); n += 1
                return n

            i = 0
            while (j := pick(cum, remaining, cps_now())) is not None:
                ch = remaining.pop(j)
                seed = None if p.seed is None else p.seed + i
                w, meta = self._gen_chunk(conds, ch, p, seed)
                if meta["hit_cap"]:  # degenerate run: one retry with another seed
                    w2, meta2 = self._gen_chunk(conds, ch, p, None if seed is None else seed + 1000)
                    if not meta2["hit_cap"] or len(w2) < len(w):
                        w, meta = w2, {**meta2, "retried": True}
                w = _trim_chunk(w)
                parts.append(w)
                cum += len(w) / SR - (CROSSFADE_S if parts[1:] else 0)
                chars_done += len(ch)
                info.append({"text": ch, **meta, "seconds": round(len(w) / SR, 2)})
                i += 1
                if progress:
                    progress(i, i + planned(), time.time() - t0)
        out = crossfade_join(parts)
        el = time.time() - t0
        dur = len(out) / SR
        if dur > 0:
            self.rtf = 0.5 * self.rtf + 0.5 * (el / dur)
            self.cps = round(0.5 * self.cps + 0.5 * (chars_done / dur), 2) if self.cps != DEFAULT_CPS else round(chars_done / dur, 2)
            try:
                self._rate_file.write_text(json.dumps({"cps": self.cps}))
            except OSError:
                pass
        spoken = " ".join(c["text"] for c in info)
        return out, {"chunks": info, "seconds": round(dur, 2), "elapsed": round(el, 1),
                     "rtf": round(el / max(dur, 1e-3), 2), "target": target, "text": spoken,
                     "unspoken": 0 if pool else len(remaining), "cps": self.cps}

    # ---------- similarity ----------
    def embed(self, x: np.ndarray, sr: int) -> np.ndarray:
        import librosa
        from chatterbox.models.s3tokenizer import S3_SR

        self.load()
        x16 = librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=S3_SR) if sr != S3_SR else x
        e = self.model.ve.embeds_from_wavs([x16], sample_rate=S3_SR).mean(0)
        return e / (np.linalg.norm(e) + 1e-9)

    def ref_embed(self, path: Path) -> np.ndarray:
        key = f"{path}:{path.stat().st_mtime}"
        if key not in self._ref_embed_cache:
            self._ref_embed_cache[key] = self.embed(self._load_ref(path, SR), SR)
        return self._ref_embed_cache[key]

    def similarity(self, x: np.ndarray, ref_path: Path) -> float:
        """Cosine similarity (VoiceEncoder) between generated audio and the user's ORIGINAL take."""
        return float(np.dot(self.embed(x, SR), self.ref_embed(ref_path)))

    def speech_seconds(self, text: str) -> float:
        """Expected length of the spoken audio, from this voice's measured pace."""
        return len(" ".join(text.split())) / self.cps

    def estimate_seconds(self, text: str, target: float | None = None) -> float:
        """Expected wall-clock time for a generation."""
        audio_s = self.speech_seconds(text) if target is None else min(target, self.speech_seconds(text))
        return round(audio_s * self.rtf + 2.0, 1)
