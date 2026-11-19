#!/usr/bin/env python3
"""Build "Voice Clone Studio.app" for macOS (run with the project's venv).

    .venv/bin/python make_app.py                      # builds into ./dist/
    .venv/bin/python make_app.py --dest ~/Applications

What it makes: a small bundle (launcher, icon, Info.plist with the microphone usage text), NOT a
frozen PyTorch bundle (that would be several GB and fragile). The app runs desktop.py from THIS
project folder with THIS folder's .venv, so recordings, presets and exports stay in ./data.
If the folder or venv is moved or deleted, the app shows a dialog; re-run this script.
Same approach as the Speaker Separator app (../Speaker Diarization/make_app.py).
"""
from __future__ import annotations

import argparse
import plistlib
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

SRC = Path(__file__).resolve().parent
LSREGISTER = "/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister"
APP_NAME = "Voice Clone Studio"
BUNDLE_ID = "local.voice-clone-studio"
SHORTCUT_ID = "local.voice-clone-studio.shortcut"
ENTRY = "desktop.py"
MIC_TEXT = "Voice Clone Studio records short samples of your voice to clone it. Audio stays on this Mac."

# A copy installed outside the project (e.g. /Applications) is a small SHORTCUT app: it asks macOS to
# start the full app (./dist) by bundle id. The full app stays inside the project folder because macOS
# silently blocks an unsigned app elsewhere from reading ~/Desktop (no permission prompt is shown).
SHORTCUT_LAUNCHER = """#!/bin/bash
# Starts the full {app_name} app, which lives in its project folder ({target}).
open -b {bundle_id} --args "$@" && exit 0
osascript -e "display dialog \\"{app_name} isn't built in its project folder yet. Run: .venv/bin/python make_app.py{app_flag} --dest /Applications (in {project})\\" buttons {{\\"OK\\"}} default button \\"OK\\" with icon stop with title \\"{app_name}\\"" >/dev/null 2>&1
exit 1
"""

LAUNCHER = """#!/bin/bash
# Launches {app_name} from its project folder with that folder's Python environment.
APP="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$APP/Contents/MacOS/{app_name}"
PROJECT="{project}"
LOG="$PROJECT/data/{log}"
fail() {{
  osascript -e "display dialog \\"$1\\" buttons {{\\"OK\\"}} default button \\"OK\\" with icon stop with title \\"{app_name}\\"" >/dev/null 2>&1
  exit 1
}}
[ -f "$PROJECT/{entry}" ] || fail "{app_name} can't read its project folder ($PROJECT). Re-run make_app.py, or allow access in System Settings > Privacy & Security > Files and Folders."
[ -x "$PY" ] || fail "{app_name} can't find its Python environment. Re-run make_app.py from the project folder."
mkdir -p "$PROJECT/data"
export VCS_SITE_PACKAGES="{site_packages}"
cd "$PROJECT"
exec "$PY" "$PROJECT/{entry}" >>"$LOG" 2>&1
"""


def make_icon_png(size: int = 1024):
    """Clay squircle (lilac) with a coral microphone blob, matching the app's palette."""
    from PIL import Image, ImageChops, ImageDraw, ImageFilter

    k = 2
    S = size * k
    body = int(S * 0.805)
    x0 = (S - body) // 2
    radius = int(body * 0.225)

    def rr(inset=0, dy=0, box=None, r=None):
        m = Image.new("L", (S, S), 0)
        b = box or (x0 + inset, x0 + inset + dy, x0 + body - inset, x0 + body - inset + dy)
        ImageDraw.Draw(m).rounded_rectangle(b, radius=r if r is not None else radius - inset, fill=255)
        return m

    canvas = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    shadow = rr(dy=int(S * 0.014)).filter(ImageFilter.GaussianBlur(S * 0.02))
    canvas.paste(Image.new("RGBA", (S, S), (60, 40, 100, 110)), (0, 0), shadow)
    grad = Image.linear_gradient("L").resize((S, S))
    face = Image.composite(Image.new("RGB", (S, S), (0xd9, 0xcd, 0xee)), Image.new("RGB", (S, S), (0xf7, 0xf3, 0xfb)), grad)
    canvas.paste(face, (0, 0), rr())
    # inset highlight (top) and shade (bottom) = clay edge
    hi = ImageChops.subtract(rr(), rr(dy=int(S * 0.012))).filter(ImageFilter.GaussianBlur(S * 0.006))
    canvas.paste(Image.new("RGBA", (S, S), (255, 255, 255, 200)), (0, 0), hi)
    lo = ImageChops.subtract(rr(), rr(dy=-int(S * 0.014))).filter(ImageFilter.GaussianBlur(S * 0.008))
    canvas.paste(Image.new("RGBA", (S, S), (90, 60, 140, 90)), (0, 0), lo)

    # coral blob with soft shadow
    cx, cy, rb = S // 2, int(S * 0.5), int(S * 0.25)
    blob_box = (cx - rb, cy - rb, cx + rb, cy + rb)
    bm = Image.new("L", (S, S), 0)
    ImageDraw.Draw(bm).ellipse(blob_box, fill=255)
    bsh = bm.filter(ImageFilter.GaussianBlur(S * 0.025))
    canvas.paste(Image.new("RGBA", (S, S), (150, 20, 50, 120)), (int(S * 0.012), int(S * 0.02)), bsh)
    import numpy as np

    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    hx, hy = cx - 0.3 * rb, cy - 0.35 * rb  # highlight top-left
    t = np.clip(np.hypot(xx - hx, yy - hy) / (1.55 * rb), 0, 1)[..., None]
    light, deep = np.array([0xff, 0x7a, 0x8c], np.float32), np.array([0xcf, 0x25, 0x41], np.float32)
    blob = Image.fromarray((light * (1 - t) + deep * t).astype(np.uint8), "RGB")
    canvas.paste(blob, (0, 0), bm)

    d = ImageDraw.Draw(canvas)
    # white mic
    w, h = int(rb * 0.46), int(rb * 0.86)
    d.rounded_rectangle((cx - w // 2, cy - int(h * 0.62), cx + w // 2, cy + int(h * 0.38)), radius=w // 2, fill=(255, 255, 255, 255))
    lw = int(rb * 0.085)
    arc = (cx - int(w * 0.95), cy - int(h * 0.30), cx + int(w * 0.95), cy + int(h * 0.62))
    d.arc(arc, start=0, end=180, fill=(255, 255, 255, 255), width=lw)
    d.line((cx, cy + int(h * 0.62), cx, cy + int(h * 0.82)), fill=(255, 255, 255, 255), width=lw)
    d.rounded_rectangle((cx - int(w * 0.55), cy + int(h * 0.80), cx + int(w * 0.55), cy + int(h * 0.80) + lw), radius=lw // 2, fill=(255, 255, 255, 255))
    return canvas.resize((size, size), Image.Resampling.LANCZOS)


def make_icns(dest: Path) -> None:
    from PIL import Image

    master = make_icon_png(1024)
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "AppIcon.iconset"
        iconset.mkdir()
        for base in (16, 32, 128, 256, 512):
            master.resize((base, base), Image.Resampling.LANCZOS).save(iconset / f"icon_{base}x{base}.png")
            master.resize((base * 2, base * 2), Image.Resampling.LANCZOS).save(iconset / f"icon_{base}x{base}@2x.png")
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(dest)], check=True)
    master.save(dest.with_suffix(".png"))


def build(dest_dir: Path, shortcut_to: Path | None = None) -> Path:
    app = dest_dir / f"{APP_NAME}.app"
    plist_path = app / "Contents" / "Info.plist"
    if app.exists():
        try:
            ours = plistlib.loads(plist_path.read_bytes()).get("CFBundleIdentifier") in (BUNDLE_ID, SHORTCUT_ID)
        except (OSError, ValueError):
            ours = False
        if not ours:
            sys.exit(f"{app} exists and was not built by this script; refusing to replace it.")
        shutil.rmtree(app)

    macos, res = app / "Contents" / "MacOS", app / "Contents" / "Resources"
    macos.mkdir(parents=True)
    res.mkdir(parents=True)
    make_icns(res / "AppIcon.icns")
    if shortcut_to is not None:
        launcher = macos / "launcher"
        launcher.write_text(SHORTCUT_LAUNCHER.format(target=shortcut_to, bundle_id=BUNDLE_ID, project=SRC, app_name=APP_NAME,
                                                     app_flag=""))
        launcher.chmod(0o755)
        plist_path.write_bytes(plistlib.dumps({
            "CFBundleName": APP_NAME, "CFBundleDisplayName": APP_NAME, "CFBundleIdentifier": SHORTCUT_ID,
            "CFBundleExecutable": "launcher", "CFBundleIconFile": "AppIcon", "CFBundlePackageType": "APPL",
            "CFBundleShortVersionString": "1.0", "CFBundleVersion": "1", "LSMinimumSystemVersion": "12.0",
            "LSApplicationCategoryType": "public.app-category.music", "NSHighResolutionCapable": True,
            "LSUIElement": True,  # no Dock icon of its own; the full app appears instead
        }))
        subprocess.run(["codesign", "--force", "--sign", "-", str(app)], check=False, capture_output=True)
        subprocess.run(["touch", str(app)], check=False)
        return app

    # The interpreter is symlinked into the bundle so macOS shows the app's name/icon (and asks for
    # the microphone on behalf of this app) instead of "Python". The venv's site-packages are added
    # by desktop.py via VCS_SITE_PACKAGES, because a symlinked interpreter doesn't find pyvenv.cfg.
    python = Path(sys.executable).resolve()
    (macos / APP_NAME).symlink_to(python)
    launcher = macos / "launcher"
    launcher.write_text(LAUNCHER.format(app_name=APP_NAME, project=SRC, site_packages=sysconfig.get_paths()["purelib"],
                                        entry=ENTRY, log="app.log"))
    launcher.chmod(0o755)

    plist_path.write_bytes(plistlib.dumps({
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleExecutable": "launcher",
        "CFBundleIconFile": "AppIcon",
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": "1.0",
        "CFBundleVersion": "1",
        "LSMinimumSystemVersion": "12.0",
        "LSApplicationCategoryType": "public.app-category.music",
        "NSHighResolutionCapable": True,
        "NSAppSleepDisabled": True,           # generation must not be throttled in the background
        "NSSupportsAutomaticGraphicsSwitching": True,
        **({"NSMicrophoneUsageDescription": MIC_TEXT} if MIC_TEXT else {}),
    }))
    subprocess.run(["touch", str(app)], check=False)
    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, default=SRC / "dist", help="folder to build into (default: ./dist)")
    args = ap.parse_args()
    if sys.platform != "darwin":
        sys.exit("make_app.py builds a macOS app bundle; run it on a Mac.")
    for mod in ("webview", "PIL", "chatterbox", "fastapi", "uvicorn"):
        try:
            __import__(mod)
        except ImportError:
            sys.exit(f"Run this with the project's venv: .venv/bin/python make_app.py (missing module: {mod}).")
    dest = args.dest.expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    full = build(SRC / "dist") if dest != (SRC / "dist").resolve() else None
    if full:
        subprocess.run([LSREGISTER, "-f", str(full)], check=False)  # so `open -b` finds it
        app = build(dest, shortcut_to=full)
        print(f"built:  {full}  (the app)")
        print(f"built:  {app}  (shortcut that opens it)")
    else:
        app = build(dest)
        print(f"built:  {app}")
    print(f"python: {Path(sys.executable).resolve()}")
    print(f"runs:   {SRC / ENTRY} (data in {SRC / 'data'})")


if __name__ == "__main__":
    main()
