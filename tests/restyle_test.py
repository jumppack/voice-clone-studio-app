"""Speech to speech, "speak like the reference" mode vs plain voice conversion.

Source: the first 75 s of tests/results/changer_source.wav (run tests/changer_test.py first).
References: the fixture voice read by `say` slowly (115 wpm) and fast (240 wpm).
Reports words per second of each reference, of plain conversion (keeps the original's pace), and of
the re-spoken result, plus VoiceEncoder similarity to the reference.

    HF_HUB_OFFLINE=1 .venv/bin/python tests/restyle_test.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import audio  # noqa: E402
from app.converter import Converter, decode  # noqa: E402
from app.engine import SR, Engine  # noqa: E402
from app.restyle import respeak, transcribe  # noqa: E402

R = ROOT / "tests" / "results"
tmp = Path(tempfile.mkdtemp())
src16 = decode(R / "changer_source.wav", 16000)[: 75 * 16000]
T = ("Every morning I walk to the corner bakery before the sun is fully up. The baker always has the first loaf "
     "waiting, still warm, wrapped in brown paper. We talk about the weather, the neighbors, and whatever happened in "
     "the news. Then I carry the bread home, make a pot of coffee, and sit by the window.")
eng = Engine(tmp)
eng.load()
conv = Converter(eng)
words = len(" ".join(s["text"] for s in transcribe(src16)["segments"]).split())
wps = lambda x, sr, n: round(n / (len(audio.trim_silence(x, sr, 0, 0)) / sr), 2)  # noqa: E731
res = {"original_wps": wps(src16, 16000, words)}
for name, rate in (("slow", 115), ("fast", 240)):
    subprocess.run(["say", "-v", "Daniel", "-r", str(rate), "-o", str(tmp / f"{name}.aiff"), T], check=True)
    ref = decode(tmp / f"{name}.aiff", SR)
    clip, _ = conv.prepare_reference(ref)
    ref_path = tmp / f"{name}.wav"
    audio.write_wav(ref_path, clip, SR)
    e_ref = conv.embed(clip, SR)
    vc, _ = conv.convert(src16, clip)
    y, info = respeak(eng, src16, ref_path, clip)
    audio.write_wav(R / f"restyle_{name}.wav", y, SR)
    res[name] = {"reference_wps": wps(clip, SR, len(T.split())), "convert_wps": wps(vc, SR, words),
                 "respeak_wps": wps(y, SR, words), "convert_similarity": round(float(conv.embed(vc, SR) @ e_ref), 3),
                 "respeak_similarity": round(float(conv.embed(y, SR) @ e_ref), 3),
                 "pace_cps": {k: info[k] for k in ("pace_original", "pace_reference", "pace_result")},
                 "cfg_range": [min(c["cfg_weight"] for c in info["chunks"]), max(c["cfg_weight"] for c in info["chunks"])],
                 "seconds": info["seconds"], "compute_seconds": info["compute_seconds"]}
    print(name, json.dumps(res[name]), flush=True)
(R / "restyle.json").write_text(json.dumps(res, indent=1))
# speaking pace (characters/s while speaking; the whole-file words/s also counts the original's paragraph pauses)
ok = all(abs(res[n]["pace_cps"]["pace_result"] - res[n]["pace_cps"]["pace_reference"])
         < abs(res[n]["pace_cps"]["pace_original"] - res[n]["pace_cps"]["pace_reference"]) for n in ("slow", "fast"))
print(json.dumps(res, indent=1))
print("PASS" if ok else "CHECK")
