"""Multilingual generation check (engine level, no server).

For each language: a 10 s preview from the built-in passage with the fixture voice (seed 7), plus the
token count per chunk. Reports length vs target, speaker similarity to the reference, whether any chunk
hit its token cap (a sign of degenerate output), and the learned pace. Writes WAVs to tests/results/lang_*.wav
so you can listen. It cannot check pronunciation: there is no speech recognizer in the project.

    HF_HUB_OFFLINE=1 .venv/bin/python tests/languages_test.py            # a representative set
    HF_HUB_OFFLINE=1 .venv/bin/python tests/languages_test.py all        # all 23 languages
    HF_HUB_OFFLINE=1 .venv/bin/python tests/languages_test.py hi es      # just these
"""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import audio  # noqa: E402
from app.engine import SR, Engine, GenParams, preview_sentences  # noqa: E402
from app.languages import LANGUAGES  # noqa: E402

F, R = ROOT / "tests" / "fixtures", ROOT / "tests" / "results"
R.mkdir(exist_ok=True)
REF = F / "daniel_take1_room.wav"

args = sys.argv[1:]
langs = list(LANGUAGES) if args == ["all"] else (args or ["en", "hi", "es", "fr", "de", "zh", "ja", "ar", "ru"])
eng = Engine(Path(tempfile.mkdtemp()))  # fresh pace estimates, nothing written to data/
t = time.time()
eng.load()
print(f"English model loaded in {time.time() - t:.1f} s")
rows = []
for lang in langs:
    t = time.time()
    wav, info = eng.generate("", REF, REF, GenParams(seed=7), target=10.0, sentences=preview_sentences(lang), lang=lang)
    el = time.time() - t
    sim = eng.similarity(wav, REF)
    audio.write_wav(R / f"lang_{lang}.wav", wav, SR)
    row = {"lang": lang, "seconds": info["seconds"], "chunks": len(info["chunks"]), "similarity": round(sim, 3),
           "hit_cap": any(c["hit_cap"] for c in info["chunks"]), "retried": any(c.get("retried") for c in info["chunks"]),
           "cps": info["cps"], "elapsed": round(el, 1)}
    rows.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)
(R / "languages.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
bad = [r["lang"] for r in rows if r["hit_cap"] or not 6 <= r["seconds"] <= 11.5]
print("PASS" if not bad else f"CHECK: {bad}")
