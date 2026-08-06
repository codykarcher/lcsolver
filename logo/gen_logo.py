"""Generate the LCsolver logo.

Style: "LC" in Avenir Next Heavy (black), product word in Helvetica Neue Thin
(colored), with the i/j rectangular tittles replaced by round dots.

Requires the macOS system fonts Avenir Next and Helvetica Neue (reads them
from /System/Library/Fonts) and fontTools. PNG export needs inkscape.

Outputs, written next to this script:
  LCsolver.svg           letterforms as paths - renders anywhere (canonical)
  LCsolver_dark.svg      same but with white "LC", for dark backgrounds
  LCsolver_editable.svg  live text - easy to edit, needs the fonts installed,
                         i/j dots stay rectangular (font-rendered)
  LCsolver.png           transparent-background PNGs (skipped if no inkscape)
  LCsolver_dark.png

To recolor: edit the /* EDIT COLORS HERE */ block in either SVG, or the
COLOR constant here and rerun.
"""
import os
import shutil
import subprocess
from fontTools.ttLib import TTCollection
from fontTools.pens.recordingPen import RecordingPen, replayRecording
from fontTools.pens.svgPathPen import SVGPathPen

NAME = "LCsolver"
COLOR = "#ede13f"

FS = 100        # font size in SVG units
BASE = 100      # baseline y
H = 130         # viewBox height
PAD = 10        # left/right padding
DOT_RADIUS = 34    # font units; round dot replacing i/j rectangular tittles
TITTLE_Y_MIN = 550  # contours entirely above this are tittles (they sit at y 611-714)
PNG_HEIGHT = 1040  # 8x the SVG height

OUT = os.path.dirname(os.path.abspath(__file__))


def load(ttc_path, ps_name):
    for f in TTCollection(ttc_path).fonts:
        if f["name"].getDebugName(6) == ps_name:
            return f
    raise ValueError(f"{ps_name} not found in {ttc_path}")


heavy = load("/System/Library/Fonts/Avenir Next.ttc", "AvenirNext-Heavy")
thin = load("/System/Library/Fonts/HelveticaNeue.ttc", "HelveticaNeue-Thin")


def contours_of(recording):
    out, cur = [], []
    for op, pts in recording:
        cur.append((op, pts))
        if op == "closePath":
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def segment(font, text, x, round_dots=False):
    """Return (svg path elements, advance-measured end x).
    round_dots: replace i/j rectangular tittles with circles."""
    upm = font["head"].unitsPerEm
    s = FS / upm
    cmap = font.getBestCmap()
    gs = font.getGlyphSet()
    paths = []
    for ch in text:
        glyph = gs[cmap[ord(ch)]]
        rec = RecordingPen()
        glyph.draw(rec)
        recording = rec.value
        circle = None
        if round_dots and ch in "ij":
            keep = []
            for c in contours_of(recording):
                ys = [p[1] for op, pts in c for p in pts]
                if min(ys) > TITTLE_Y_MIN:
                    xs = [p[0] for op, pts in c for p in pts]
                    circle = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
                else:
                    keep.extend(c)
            recording = keep
        pen = SVGPathPen(gs)
        replayRecording(recording, pen)
        d = pen.getCommands()
        tf = f'transform="translate({x:.2f},{BASE}) scale({s:.6f},{-s:.6f})"'
        if d:
            paths.append(f'    <path {tf} d="{d}"/>')
        if circle:
            paths.append(
                f'    <circle {tf} cx="{circle[0]:.1f}" cy="{circle[1]:.1f}" r="{DOT_RADIUS}"/>'
            )
        x += glyph.width * s
    return paths, x


def measure(font, text):
    upm = font["head"].unitsPerEm
    cmap = font.getBestCmap()
    gs = font.getGlyphSet()
    return sum(gs[cmap[ord(c)]].width for c in text) * FS / upm


STYLE = """  <style>
    /* EDIT COLORS HERE */
    .lc   {{ fill: {lc_fill}; {lc_extra}}}
    .word {{ fill: {color}; {word_extra}}}
  </style>"""

word = NAME[2:]
width = PAD + measure(heavy, "LC") + measure(thin, word) + PAD

# --- live-text editable version ---
style = STYLE.format(
    color=COLOR,
    lc_fill="#000000",
    lc_extra='font-family: "Avenir Next", "Avenir", sans-serif; font-weight: 900; ',
    word_extra='font-family: "Helvetica Neue", Helvetica, sans-serif; font-weight: 200; ',
)
live = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {H}">
{style}
  <text x="{PAD}" y="{BASE}" font-size="{FS}"><tspan class="lc">LC</tspan><tspan class="word">{word}</tspan></text>
</svg>
"""
with open(os.path.join(OUT, f"{NAME}_editable.svg"), "w") as f:
    f.write(live)

# --- outlined canonical versions (light: black LC; dark: white LC) ---
lc_paths, x_end = segment(heavy, "LC", PAD)
word_paths, _ = segment(thin, word, x_end, round_dots=True)
inkscape = shutil.which("inkscape")
for suffix, lc_fill in (("", "#000000"), ("_dark", "#ffffff")):
    style = STYLE.format(color=COLOR, lc_fill=lc_fill, lc_extra="", word_extra="")
    outlined_path = os.path.join(OUT, f"{NAME}{suffix}.svg")
    outlined = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {H}">
{style}
  <g class="lc">
{chr(10).join(lc_paths)}
  </g>
  <g class="word">
{chr(10).join(word_paths)}
  </g>
</svg>
"""
    with open(outlined_path, "w") as f:
        f.write(outlined)
    if inkscape:
        subprocess.run(
            [inkscape, outlined_path, "-o",
             os.path.join(OUT, f"{NAME}{suffix}.png"), "-h", str(PNG_HEIGHT)],
            check=True, capture_output=True,
        )
if not inkscape:
    print("inkscape not found - PNGs skipped")

print(f"{NAME} {COLOR} width={width:.0f} done")
