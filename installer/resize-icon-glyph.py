"""Crop the Diarix split-wave glyph to its content bbox and re-center it at a
larger fill fraction, then regenerate every derived icon size Tauri bundles.

The current icon.png has the glyph occupying only ~63% width / ~47% height of
its 512x512 canvas -- most app icons fill ~85-90%, which is why it reads
small next to other taskbar icons. This rescales from the same source glyph
(no redraw), just with the padding trimmed down to a normal margin.
"""

from pathlib import Path
from PIL import Image

ICONS_DIR = Path(r"Z:\Diarix Studio\diarix-voicebox-upstream-fork-20260713\tauri\src-tauri\icons")
# Vector-traced from app/src/assets/diarix-logo.png and stroke-dilated so the
# naturally thin outline glyph stays legible at small icon sizes (see
# installer/build-icon-master.py) -- the plain upscaled-crop source this
# pointed to previously was crisp at large sizes but read as a near-invisible
# smudge at 32x32.
SOURCE = ICONS_DIR / "icon_master_v2.png"
MASTER_SIZE = 1024
FILL_FRACTION = 0.86  # glyph's larger dimension as a fraction of the canvas

PNG_SIZES = {
    "32x32.png": 32,
    "64x64.png": 64,
    "128x128.png": 128,
    "128x128@2x.png": 256,
    "icon.png": 512,
    "Square30x30Logo.png": 30,
    "Square44x44Logo.png": 44,
    "Square71x71Logo.png": 71,
    "Square89x89Logo.png": 89,
    "Square107x107Logo.png": 107,
    "Square142x142Logo.png": 142,
    "Square150x150Logo.png": 150,
    "Square284x284Logo.png": 284,
    "Square310x310Logo.png": 310,
    "StoreLogo.png": 50,
}

ICO_SIZES = [16, 20, 24, 32, 40, 48, 64, 96, 128, 256]


def build_master(source: Path, size: int, fill: float) -> Image.Image:
    img = Image.open(source).convert("RGBA")
    bbox = img.getbbox()
    if bbox is None:
        raise SystemExit(f"{source} has no non-transparent content")
    glyph = img.crop(bbox)
    gw, gh = glyph.size

    target_max = int(size * fill)
    scale = target_max / max(gw, gh)
    new_w, new_h = max(1, round(gw * scale)), max(1, round(gh * scale))
    glyph = glyph.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    offset = ((size - new_w) // 2, (size - new_h) // 2)
    canvas.paste(glyph, offset, glyph)
    return canvas


def main() -> None:
    master = build_master(SOURCE, MASTER_SIZE, FILL_FRACTION)

    for name, px in PNG_SIZES.items():
        resized = master.resize((px, px), Image.LANCZOS)
        resized.save(ICONS_DIR / name)
        print(f"wrote {name} ({px}x{px})")

    # Pillow's ICO writer resizes DOWN from whichever image .save() is called
    # on -- it does not actually use append_images to embed pre-made frames
    # (that kwarg is silently ignored for ICO). Calling save() on the
    # smallest frame meant every larger size got dropped, so Windows had
    # only a 16x16 bitmap to work with for every context, including the
    # window titlebar icon. Save from the full-res master instead so Pillow
    # downsamples from real detail for every embedded size.
    master.save(
        ICONS_DIR / "icon.ico",
        format="ICO",
        sizes=[(s, s) for s in ICO_SIZES],
    )
    print(f"wrote icon.ico ({ICO_SIZES})")

    verify = Image.open(ICONS_DIR / "icon.png").convert("RGBA")
    bbox = verify.getbbox()
    w, h = verify.size
    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    print(f"new icon.png glyph fill: {bw/w:.3f} x {bh/h:.3f}")


if __name__ == "__main__":
    main()
