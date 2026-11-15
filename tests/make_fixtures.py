"""Builds the test fixtures in tests/fixtures/ with the macOS `say` voices (Daniel, Samantha).

They are not in the repository (synthetic system voices aren't ours to redistribute). Run once:

    .venv/bin/python tests/make_fixtures.py

Files: daniel_take{1,2,3}.wav and samantha_take1.wav (clean, 48 kHz mono 16-bit), *_room.wav (the
same speech with a faint room hiss, ~-60 dBFS), and fake_mic.wav (room tone + a take + room tone) for
the headless UI test's microphone shim.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.scripts import STEPS  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "fixtures"
SR = 48000
rng = np.random.default_rng(1234)


def say(voice: str, text: str) -> np.ndarray:
    with tempfile.TemporaryDirectory() as d:
        aiff, wav = Path(d) / "s.aiff", Path(d) / "s.wav"
        subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(aiff), "-ac", "1", "-ar", str(SR),
                        "-c:a", "pcm_s16le", str(wav)], check=True)
        x, _ = sf.read(str(wav), dtype="float32")
    return x * (0.7 / max(1e-6, float(np.abs(x).max())))  # peak about -3 dBFS


def hiss(n: int) -> np.ndarray:
    return (rng.standard_normal(n) * 10 ** (-60 / 20)).astype(np.float32)


def write(name: str, x: np.ndarray) -> None:
    sf.write(str(OUT / f"{name}.wav"), np.clip(x, -1, 1), SR, subtype="PCM_16")
    print("wrote", OUT / f"{name}.wav", f"{len(x) / SR:.2f} s")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    takes = {f"daniel_take{i}": say("Daniel", STEPS[i - 1]["script"]) for i in (1, 2, 3)}
    takes["samantha_take1"] = say("Samantha", STEPS[0]["script"])
    for name, x in takes.items():
        write(name, x)
        write(f"{name}_room", x + hiss(len(x)))
    d = takes["daniel_take1"]
    write("fake_mic", np.concatenate([hiss(3 * SR), d + hiss(len(d)), hiss(int(2.5 * SR))]))


if __name__ == "__main__":
    main()
