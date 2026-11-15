"""60-word generation: wall time, real-time factor, peak memory."""
import os, sys, time, resource, json
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from app import audio
from app.engine import Engine, GenParams, split_text
F = Path(__file__).parent / "fixtures"; REF = F / "daniel_take1_room.wav"
TEXT = ("Every great trip starts with a small decision. Maybe you pick a road you have never driven, or you say yes to a "
        "friend who wants to see the mountains. Along the way, plans change, the weather turns, and the best moments are "
        "the ones that nobody planned. So pack light, stay curious, and leave a little room for surprises.")
e = Engine(Path("data/cache")); e.load()
e.generate("Warming up the engine with a short sentence.", REF, REF, GenParams(seed=0))
t = time.time()
x, info = e.generate(TEXT, REF, REF, GenParams(seed=3))
el = time.time() - t
r = {"words": len(TEXT.split()), "chunks": len(split_text(TEXT)), "audio_s": info["seconds"], "wall_s": round(el, 1),
     "rtf": round(el / info["seconds"], 2), "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2),
     "mps_driver_gb": round(torch.mps.driver_allocated_memory() / 1e9, 2), "hit_cap": [c["hit_cap"] for c in info["chunks"]],
     "similarity": round(e.similarity(x, REF), 3)}
audio.write_wav(Path(__file__).parent / "results/long_60_words.wav", x, 24000)
print("LONG", r); json.dump(r, open(Path(__file__).parent / "results/long_text.json", "w"), indent=1)
