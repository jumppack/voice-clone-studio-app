"""Measure each control at its two extremes on the same sentence and seeds.

Writes tests/results/controls.json and prints a markdown table.
"""
import itertools, json, os, sys, time, resource
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from pathlib import Path
import numpy as np, librosa
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import audio
from app.engine import Engine, GenParams, SR

F = Path(__file__).parent / "fixtures"; R = Path(__file__).parent / "results"; R.mkdir(exist_ok=True)
REF = F / "daniel_take1_room.wav"
TEXT = "The old bridge was closed for repairs, so we took the long road around the lake and arrived just after sunset."
WORDS = len(TEXT.split())
SEEDS = [11, 22, 33]
e = Engine(Path("data/cache")); e.load()
e.generate("Warming up the engine with a short sentence.", REF, REF, GenParams(seed=0))

def metrics(x):
    m, _ = audio.speech_mask(x, SR)
    idx = np.flatnonzero(m)
    span = (idx[-1] - idx[0] + 1) * audio.FRAME_S if len(idx) else len(x) / SR
    f0, vf, _ = librosa.pyin(x, fmin=60, fmax=400, sr=SR, frame_length=2048)
    f0 = f0[vf & ~np.isnan(f0)]
    st = 12 * np.log2(f0 / np.median(f0)) if len(f0) else np.array([0])
    rms = audio.frame_rms_db(x, SR)[: len(m)][m]
    return {"wps": WORDS / span, "f0_med": float(np.median(f0)) if len(f0) else 0, "f0_std_st": float(np.std(st)),
            "rms_std_db": float(np.std(rms)), "sim": e.similarity(x, REF), "dur": len(x) / SR}

def mfcc_dist(a, b):
    A = librosa.feature.mfcc(y=a, sr=SR, n_mfcc=20)[1:]; B = librosa.feature.mfcc(y=b, sr=SR, n_mfcc=20)[1:]
    D, wp = librosa.sequence.dtw(A, B, metric="euclidean")
    return float(D[-1, -1] / len(wp))

def run(p, seeds=SEEDS, keep=False):
    outs, ms = [], []
    for s in seeds:
        x, _ = e.generate(TEXT, REF, REF, GenParams(**{**p, "seed": s})); outs.append(x); ms.append(metrics(x))
    avg = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
    avg["pair_dist"] = float(np.mean([mfcc_dist(a, b) for a, b in itertools.combinations(outs, 2)]))
    avg["dur_std"] = float(np.std([m["dur"] for m in ms]))
    return avg, outs

D = dict(similarity=1.0, exaggeration=0.5, cfg_weight=0.5, depth=0.0, temperature=0.8)
tests = {
    "similarity": [0.5, 1.0], "cfg_weight": [0.2, 0.8], "depth": [-3.0, 3.0],
    "exaggeration": [0.25, 1.5], "temperature": [0.4, 1.2],
}
res = {"default": run(D)[0]}
for k, vals in tests.items():
    for v in vals:
        res[f"{k}={v}"], outs = run({**D, k: v})
        audio.write_wav(R / f"{k}_{v}.wav", outs[0], SR)
        print(k, v, {a: round(b, 3) for a, b in res[f"{k}={v}"].items()}, flush=True)
# fixed seed repeatability
a, _ = e.generate(TEXT, REF, REF, GenParams(seed=5)); b, _ = e.generate(TEXT, REF, REF, GenParams(seed=5))
n = min(len(a), len(b))
res["repeat_same_seed"] = {"len_a": len(a), "len_b": len(b), "max_abs_diff": float(np.max(np.abs(a[:n] - b[:n]))) if n else None,
                           "mfcc_dist": mfcc_dist(a, b)}
print("repeat", res["repeat_same_seed"])
json.dump(res, open(R / "controls.json", "w"), indent=1)
