"""Compare similarity-blend strategies: embedding lerp only vs. prompt splice (both orders)."""
import tempfile
import os, sys, json
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import engine as E
from app.engine import Engine, GenParams
F = Path(__file__).parent / "fixtures"; REF = F / "daniel_take1_room.wav"
TEXT = "The old bridge was closed for repairs, so we took the long road around the lake and arrived just after sunset."
e = Engine(Path(tempfile.mkdtemp()))  # temp cache: never touches data/; e.load()
out = {}
for mode in ["embed_only", "default_first", "user_first"]:
    E.PROMPT_SPLICE = mode != "embed_only"; E.PROMPT_ORDER = mode
    for a in [0.5, 0.75, 1.0]:
        if a == 1.0 and mode != "embed_only": continue
        e._conds_cache.clear()
        sims = [e.similarity(e.generate(TEXT, REF, REF, GenParams(similarity=a, seed=s))[0], REF) for s in (11, 22, 33)]
        # similarity to the default voice too: generate default-voice reference once
        out[f"{mode}@{a}"] = round(float(np.mean(sims)), 3)
        print(mode, a, out[f"{mode}@{a}"], [round(x, 3) for x in sims], flush=True)
json.dump(out, open(Path(__file__).parent / "results/similarity_experiment.json", "w"), indent=1)
