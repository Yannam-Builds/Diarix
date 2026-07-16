"""Build the final high-quality Diarix icon master: vector-traced from the
app's own logo asset (clean edges, no upscale blur), then dilated so the
naturally thin outline strokes stay legible at small icon sizes.

Uses a Euclidean-distance-transform dilation (scipy), not PIL's MaxFilter --
MaxFilter's square kernel visibly facets/blockifies curved edges (circles
come out slightly octagonal), which is what made the first pass look
pixelated instead of just bold. EDT dilation with a disk-shaped threshold
keeps every curve round regardless of dilation radius.
"""

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

SOURCE_HIRES = r"C:\Users\prana\AppData\Local\Temp\claude\Z--Diarix-Studio\ee6cca42-8a57-49b9-9ebc-555117dabcf3\scratchpad\logo_hires.png"
OUT_PATH = r"Z:\Diarix Studio\diarix-voicebox-upstream-fork-20260713\tauri\src-tauri\icons\icon_master.png"
DILATE_RADIUS_PX = 14  # radius in SOURCE_HIRES (2048px canvas) pixel units
FILL_FRACTION = 0.86
CANVAS_SIZE = 1024


def main() -> None:
    img = Image.open(SOURCE_HIRES).convert("RGBA")
    alpha = np.array(img.split()[-1], dtype=np.uint8)

    # Distance transform of the background (0-alpha) gives, for every
    # background pixel, its distance to the nearest foreground pixel.
    # Thresholding that distance at DILATE_RADIUS_PX is a true circular
    # (Euclidean) dilation -- unlike a square MaxFilter kernel, this keeps
    # round edges round at any dilation radius.
    is_bg = alpha < 32
    dist_to_fg = distance_transform_edt(is_bg)
    dilated_mask = (dist_to_fg <= DILATE_RADIUS_PX) | (~is_bg)

    dilated_alpha = Image.fromarray((dilated_mask * 255).astype(np.uint8))

    bold = Image.new("RGBA", img.size, (0, 0, 0, 0))
    white = Image.new("RGBA", img.size, (255, 255, 255, 255))
    bold.paste(white, (0, 0), dilated_alpha)

    bbox = bold.getbbox()
    glyph = bold.crop(bbox)
    gw, gh = glyph.size

    target_max = int(CANVAS_SIZE * FILL_FRACTION)
    scale = target_max / max(gw, gh)
    new_w, new_h = max(1, round(gw * scale)), max(1, round(gh * scale))
    glyph = glyph.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))
    offset = ((CANVAS_SIZE - new_w) // 2, (CANVAS_SIZE - new_h) // 2)
    canvas.paste(glyph, offset, glyph)
    canvas.save(OUT_PATH)

    verify_bbox = canvas.getbbox()
    bw, bh = verify_bbox[2] - verify_bbox[0], verify_bbox[3] - verify_bbox[1]
    print(f"saved {OUT_PATH}, glyph fill {bw/CANVAS_SIZE:.3f} x {bh/CANVAS_SIZE:.3f}")


if __name__ == "__main__":
    main()
