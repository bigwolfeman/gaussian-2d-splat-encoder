"""Draw original, deterministic test cards for public encoder examples."""

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "samples" / "targets"
OUT.mkdir(parents=True, exist_ok=True)
SIZE = 768
FONT_PATHS = [
    Path(os.environ.get("DEJAVU_SANS", "")),
    Path("/usr/share/fonts/TTF/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
]
FONT = next((p for p in FONT_PATHS if p.is_file()), None)


def font(size):
    if FONT is None:
        raise FileNotFoundError("Install DejaVu Sans or set DEJAVU_SANS to its .ttf path")
    return ImageFont.truetype(FONT, size)


def poster():
    im = Image.new("RGB", (SIZE, SIZE), "#f6f2e9")
    d = ImageDraw.Draw(im)
    d.rectangle((48, 49, 720, 719), outline="#1d3557", width=5)
    d.text((78, 72), "GAUSSIAN SPLATS", fill="#142d4e", font=font(48))
    d.text((80, 149), "A direct image-to-set encoder", fill="#1d3557", font=font(26))
    for i, s in enumerate(("sharp edges", "thin lettering", "large color fields", "small details")):
        y = 231 + i * 72
        d.rectangle((80, y, 116, y + 36), fill=["#0c5b7c", "#e18428", "#5d4fa2", "#3a7773"][i])
        d.text((139, y + 2), s, fill="#1b2737", font=font(25))
    for i in range(8):
        d.text((82, 560 + i * 18), f"Sample line {i+1:02d}: ABC 123", fill="#1d3557", font=font(12))
    im.save(OUT / "poster.png")


def geometry():
    im = Image.new("RGB", (SIZE, SIZE), "#fafafa")
    d = ImageDraw.Draw(im)
    for x in range(40, 729, 32):
        d.line((x, 40, x, 728), fill="#d8dee8", width=1)
    for y in range(40, 729, 32):
        d.line((40, y, 728, y), fill="#d8dee8", width=1)
    d.ellipse((95, 100, 435, 440), fill="#e6a54b", outline="#274e79", width=6)
    d.polygon([(337, 204), (648, 80), (700, 397)], fill="#365e8d")
    d.rounded_rectangle((120, 476, 660, 663), 35, fill="#5faaa2")
    for i in range(9):
        x = 145 + i * 57
        d.line((x, 490, x + 110, 650), fill="#f5f3e9", width=3)
    d.text((55, 45), "SHAPES / EDGES", fill="#111827", font=font(30))
    im.save(OUT / "geometry.png")


def gradient():
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    c = np.stack([
        35 + 190 * xx / SIZE,
        30 + 160 * yy / SIZE,
        155 + 50 * np.sin(xx / 100) * np.cos(yy / 85),
    ], axis=-1).clip(0, 255).astype(np.uint8)
    im = Image.fromarray(c, "RGB")
    d = ImageDraw.Draw(im)
    d.ellipse((94, 105, 674, 684), outline="#f5f4e9", width=14)
    d.ellipse((188, 204, 574, 590), outline="#102c51", width=8)
    d.text((110, 345), "COLOR FIELD", fill="#fff6dc", font=font(52))
    im.save(OUT / "gradient.png")


if __name__ == "__main__":
    poster()
    geometry()
    gradient()
    print("wrote 3 original targets")
