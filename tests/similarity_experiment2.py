import tempfile
import os, sys, json
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import engine as E
from app.engine import Engine, GenParams
F = Path(__file__).parent / "fixtures"; REF = F / "daniel_take1_room.wav"; SAM = F / "samantha_take1_room.wav"
TEXT = "The old bridge was closed for repairs, so we took the long road around the lake and arrived just after sunset."
e = Engine(Path(tempfile.mkdtemp()))  # temp cache: never touches data/; e.load()
E.PROMPT_SPLICE = True; E.PROMPT_ORDER = "user_first"; E.USER_SHRINK = 1.0
res = {}
for a in [0.5, 0.625, 0.75, 0.875]:
    e._conds_cache.clear()
    sims = [e.similarity(e.generate(TEXT, REF, REF, GenParams(similarity=a, seed=s))[0], REF) for s in (11, 22, 33)]
    res[a] = round(float(np.mean(sims)), 3); print("shrink1", a, res[a], [round(x, 3) for x in sims], flush=True)
# baselines: default voice itself, and a different speaker (Samantha clone) scored against Daniel
m = e.model
import torch
from chatterbox.tts import punc_norm
x = m.generate(TEXT, exaggeration=0.5) if False else None
E.PROMPT_SPLICE = False
sam = np.mean([e.similarity(e.generate(TEXT, SAM, SAM, GenParams(seed=s))[0], REF) for s in (11, 22)])
print("baseline samantha-clone vs daniel", round(float(sam), 3))
