"""Upload-a-recording end to end (API). Builds test files from the fixtures, uploads them, checks
splitting / speaker check / refusals, then generates a preview from the uploaded voice.

    VCS_DATA=$PWD/data_test PORT=7862 NO_BROWSER=1 ./run.sh &
    VCS_DATA=$PWD/data_test PORT=7862 .venv/bin/python tests/upload_e2e.py

Uploads REPLACE the takes in that data folder, so never point this at your own data/.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
B = f"http://127.0.0.1:{os.environ.get('PORT', '7861')}"
DATA = Path(os.environ.get("VCS_DATA", ROOT / "data")).resolve()
if DATA == (ROOT / "data").resolve():
    sys.exit("Refusing to run against data/ (it would replace your takes). Set VCS_DATA to a test folder.")
U = DATA / "uploads"
U.mkdir(parents=True, exist_ok=True)
F = ROOT / "tests" / "fixtures"


def load(n):
    x, sr = sf.read(str(F / f"{n}.wav"), dtype="float32", always_2d=True)
    return x.mean(1), sr


def build():
    d = [load(f"daniel_take{i}")[0] for i in (1, 2, 3)]
    sr = load("daniel_take1")[1]
    gap = np.zeros(int(0.6 * sr), np.float32)
    sf.write(str(U / "daniel_long.wav"), np.concatenate([d[0], gap, d[1], gap, d[2]]), sr, subtype="PCM_16")
    sam = load("samantha_take1")[0]
    sf.write(str(U / "mixed_speakers.wav"), np.concatenate([d[0], gap, d[1], gap, sam, gap, d[2]]), sr, subtype="PCM_16")
    sf.write(str(U / "short_6s.wav"), d[0][: int(6 * sr)], sr, subtype="PCM_16")
    sf.write(str(U / "too_short.wav"), d[0][: int(2 * sr)], sr, subtype="PCM_16")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(U / "daniel_long.wav"), "-ac", "2", "-ar", "44100",
                    "-b:a", "192k", str(U / "daniel_long.mp3")], check=True)


def post(path, fields, files=()):
    boundary = uuid.uuid4().hex
    body = b""
    for k, v in fields.items():
        body += f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    for k, p in files:
        body += (f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"; filename="{p.name}"\r\n'
                 f"Content-Type: application/octet-stream\r\n\r\n").encode() + p.read_bytes() + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(B + path, body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def get(path):
    with urllib.request.urlopen(B + path, timeout=60) as r:
        return json.loads(r.read())


def show(name, code, d):
    print(f"== {name}: HTTP {code}")
    if code != 200:
        print("   ", d.get("detail"))
        return
    print(f"    {d['file']} {d['duration']} s {d['format'].get('format')} -> {d['clips']} clips, "
          f"usable {d['usable']}, ready {d['ready']}, voice take {d['voice_take']}")
    print("    speaker similarity to median:", (d["speaker_check"] or {}).get("similarity_to_median"))
    for t in d["takes"]:
        a = t["analysis"]
        print(f"     {t['label']} | {t['tone']} | {t['trimmed_seconds']} s | {a['score']} {a['verdict']} {[i['kind'] for i in a['issues']]}")


def main():
    build()
    post("/api/settings", {"consent": "true"})
    post("/api/settings", {"watermark_ack": "true"})
    results = {}
    for n in ("too_short.wav", "short_6s.wav", "mixed_speakers.wav", "daniel_long.wav", "daniel_long.mp3"):
        code, d = post("/api/upload-voice", {}, [("file", U / n)])
        show(n, code, d)
        results[n] = (code, d)
    post("/api/settings", {"consent": "false"})
    code, d = post("/api/upload-voice", {}, [("file", U / "daniel_long.wav")])
    show("consent box not ticked", code, d)
    consent_refused = code == 403
    post("/api/settings", {"consent": "true"})
    code, d = post("/api/upload-voice", {}, [("file", ROOT / "README.md")])
    show("not audio (README.md)", code, d)

    # generate from the last upload (MP3)
    code, j = post("/api/generate", {"preview": "true", "target": "10", "params": json.dumps({"seed": 7})})
    while j.get("state") not in ("done", "error"):
        time.sleep(1)
        j = get(f"/api/jobs/{j['id']}")
    r = j.get("result", {})
    print(f"== preview from uploaded MP3 voice: {j['state']} {r.get('seconds')} s, similarity {r.get('similarity')} {j.get('error') or ''}")

    ok = (results["too_short.wav"][0] == 400 and consent_refused
          and results["short_6s.wav"][0] == 200 and results["short_6s.wav"][1]["ready"]
          and results["daniel_long.mp3"][1]["source"] == "upload" and results["daniel_long.mp3"][1]["min_takes"] == 1
          and results["daniel_long.wav"][1]["ready"] and results["daniel_long.mp3"][1]["ready"]
          and any(i["kind"] == "speaker" for t in results["mixed_speakers.wav"][1]["takes"] for i in t["analysis"]["issues"])
          and j["state"] == "done")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
