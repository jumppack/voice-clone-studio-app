"""CLI smoke test: analyse fixture takes, generate one sentence, report similarity."""
import os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from pathlib import Path
import numpy as np, soundfile as sf
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import audio
from app.engine import Engine, GenParams

F = Path(__file__).parent / "fixtures"
rng = np.random.default_rng(0)
for name in ["daniel_take1", "daniel_take2", "daniel_take3", "samantha_take1"]:
    x, sr = sf.read(F / f"{name}.wav", dtype="float32")
    x = x + rng.normal(0, 10 ** (-62 / 20), len(x)).astype(np.float32)   # realistic room floor
    a = audio.analyze_take(x, sr)
    print(name, {k: a[k] for k in ("score", "verdict", "noise_floor_dbfs", "speech_seconds", "peak_dbfs", "reverb_decay_db_per_100ms")})
    audio.write_wav(F / f"{name}_room.wav", audio.trim_silence(x, sr), sr)
# bad takes
x, sr = sf.read(F / "daniel_take1.wav", dtype="float32")
print("clipped", audio.analyze_take(np.clip(x * 6, -1, 1), sr)["issues"][0]["text"][:40])
print("noisy", [i["kind"] for i in audio.analyze_take(x + rng.normal(0, 0.02, len(x)).astype(np.float32), sr)["issues"]])
print("quiet", [i["kind"] for i in audio.analyze_take(x * 0.01 + rng.normal(0, 1e-4, len(x)).astype(np.float32), sr)["issues"]])
print("short", [i["kind"] for i in audio.analyze_take(x[: 4 * sr], sr)["issues"]])

e = Engine(Path("data/cache"))
t = time.time(); e.load(); print("load s", round(time.time() - t, 1))
ref = F / "daniel_take1_room.wav"
out, info = e.generate("This is a quick test of the cloned voice, running on the local graphics chip.", ref, ref, GenParams(seed=1))
print(info)
print("sim to daniel", round(e.similarity(out, ref), 3), "sim samantha-ref vs daniel-ref",
      round(float(np.dot(e.ref_embed(F / "samantha_take1_room.wav"), e.ref_embed(ref))), 3))
audio.write_wav(Path("data/smoke.wav"), out, 24000)
