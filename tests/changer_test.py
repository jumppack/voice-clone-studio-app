"""Voice Changer check (engine level, no server).

Builds a multi-speaker source with macOS `say` (several voices, with short and long pauses),
re-voices it with the fixture reference (daniel takes 1+2, ~20 s), and reports:
  * length: output must equal the original exactly (timing kept)
  * pause placement: the long silence must still be silent at the same time
  * voice match: VoiceEncoder similarity of each piece to the reference, vs the original speaker's
  * speed: compute seconds per audio second

    HF_HUB_OFFLINE=1 .venv/bin/python tests/changer_test.py            # ~4 min source
    HF_HUB_OFFLINE=1 .venv/bin/python tests/changer_test.py 20         # ~20 min source (slow)
Writes tests/results/changer_{source,converted}.wav and changer.json.
"""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import audio  # noqa: E402
from app.converter import SR_OUT, SR_TOK, Converter, decode, plan  # noqa: E402

F, R = ROOT / "tests" / "fixtures", ROOT / "tests" / "results"
R.mkdir(exist_ok=True)
minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0

TEXT = ("The old lighthouse stood at the edge of the cliff for more than a hundred years. Every night, the keeper "
        "climbed the narrow stairs and lit the great lamp. Ships passing in the dark would see its beam and know they "
        "were close to the rocks. One winter, a terrible storm rolled in from the north. The wind howled and the waves "
        "crashed against the stones below. By morning, the storm had passed, and every ship had made it safely home.")
VOICES = ["Samantha", "Karen", "Moira", "Fred", "Ralph"]

tmp = Path(tempfile.mkdtemp())
parts, sr = [], 24000
i = 0
while sum(len(p) for p in parts) / sr < minutes * 60:
    v = VOICES[i % len(VOICES)]
    aiff = tmp / f"{i}.aiff"
    subprocess.run(["say", "-v", v, "-o", str(aiff), TEXT], check=True)
    parts.append(decode(aiff, sr))
    parts.append(np.zeros(int(sr * (3.0 if i % 2 == 0 else 0.8)), np.float32))  # long / short pause
    i += 1
src24 = np.concatenate(parts)
src24 += np.random.default_rng(0).normal(0, 10 ** (-60 / 20), len(src24)).astype(np.float32)  # faint hiss
src_path = R / "changer_source.wav"
audio.write_wav(src_path, src24, sr)

x16 = decode(src_path, SR_TOK)
ref24 = np.concatenate([decode(F / f"daniel_take{k}_room.wav", SR_OUT) for k in (1, 2)])
conv = Converter()
t = time.time()
conv.load()
print(f"loaded in {time.time() - t:.1f} s")
ref, rinfo = conv.prepare_reference(ref24)
pieces = plan(x16)
speech_pieces = [p for p in pieces if p.speech]
print(f"source {len(x16) / SR_TOK:.1f} s, {len(pieces)} pieces ({len(speech_pieces)} speech), "
      f"lengths {min(p.seconds for p in speech_pieces):.1f}-{max(p.seconds for p in speech_pieces):.1f} s, "
      f"reference used {rinfo['used_seconds']} s, estimate {conv.estimate(sum(p.seconds for p in speech_pieces), rinfo['used_seconds'])} s")

last = [0.0]


def prog(p):
    if time.time() - last[0] > 15:
        last[0] = time.time()
        print(f"  {p['done']}/{p['total']}  {p['fraction'] * 100:.0f}%  eta {p['eta']:.0f} s", flush=True)


y, info = conv.convert(x16, ref, pieces, progress=prog)
audio.write_wav(R / "changer_converted.wav", y, SR_OUT)

# baseline: original speaker vs reference, same pieces
ref_e = conv.embed(ref, SR_OUT)
base = [float(np.dot(conv.embed(src24[int(p.a * 0.03 * sr):int(p.b * 0.03 * sr)], sr), ref_e)) for p in speech_pieces]
# long pauses must stay silent at the same place
sil = [p for p in pieces if not p.speech and p.seconds >= 2]
sil_db = [audio.db(float(np.sqrt(np.mean(y[int(p.a * 0.03 * SR_OUT):int(p.b * 0.03 * SR_OUT)] ** 2)) + 1e-12)) for p in sil]

res = {"source_seconds": round(len(x16) / SR_TOK, 2), "output_seconds": info["seconds"],
       "length_exact": len(y) == round(len(x16) * SR_OUT / SR_TOK),
       "pieces": len(speech_pieces), "silent_pieces": len(sil), "silence_rms_db_max": round(max(sil_db), 1) if sil_db else None,
       "similarity_original_median": round(float(np.median(base)), 3),
       "similarity_converted_median": info["similarity"], "similarity_converted_min": info["similarity_min"],
       "low_pieces": len(info["low_pieces"]), "compute_seconds": info["compute_seconds"],
       "compute_per_audio_second": round(info["compute_seconds"] / info["speech_seconds"], 3),
       "reference": {k: rinfo[k] for k in ("used_seconds", "verdict", "score")}}
(R / "changer.json").write_text(json.dumps({**res, "detail": info["pieces"]}, indent=1))
print(json.dumps(res, indent=1))
ok = res["length_exact"] and res["similarity_converted_median"] > res["similarity_original_median"] + 0.15 \
    and (res["silence_rms_db_max"] is None or res["silence_rms_db_max"] < -60)
print("PASS" if ok else "CHECK")
