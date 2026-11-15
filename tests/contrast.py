"""WCAG contrast of text tokens against the clay surfaces they sit on (light + dark)."""
import re, sys
css = open("web/styles.css").read()
def block(sel):
    i = css.index(sel); j = css.index("{", i); k = css.index("}", j); return css[j:k]
def toks(b): return dict(re.findall(r"--([\w-]+):\s*(#[0-9a-fA-F]{6})", b))
light = toks(block(":root {")); dark = {**light, **toks(block(':root[data-theme="dark"]'))}
def lum(h):
    c = [int(h[i:i+2], 16) / 255 for i in (1, 3, 5)]
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
def cr(a, b):
    la, lb = sorted([lum(a), lum(b)], reverse=True); return (la + 0.05) / (lb + 0.05)
pairs = [("text", "surface"), ("text", "surface-2"), ("text", "well"), ("text", "bg"), ("muted", "surface"), ("muted", "surface-2"),
         ("muted", "well"), ("muted", "bg"), ("on-primary", "primary"), ("on-primary", "primary-2"), ("mint-ink", "mint"),
         ("peach-ink", "peach"), ("lemon-ink", "lemon"), ("lilac-ink", "lilac"), ("bad-ink", "peach"), ("primary", "surface")]
fail = 0
for name, t in [("light", light), ("dark", dark)]:
    for a, b in pairs:
        if name == "dark" and a == "primary": a = "primary-2"
        r = cr(t[a], t[b]); ok = r >= 4.5; fail += not ok
        print(f"{name:5} {a:11} on {b:10} {r:5.2f} {'OK' if ok else 'FAIL'}")
# on-accent (white) on the record blob
for name, t in [("light", light), ("dark", dark)]:
    for bg in ("accent", "accent-2"):
        r = cr("#ffffff", t[bg]); fail += r < 4.5; print(f"{name:5} white       on {bg:10} {r:5.2f} {'OK' if r >= 4.5 else 'FAIL'}")
sys.exit(1 if fail else 0)
