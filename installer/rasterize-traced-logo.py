"""Rasterize the vtracer-traced Diarix logo SVG at high resolution using pure
Python (no native cairo dependency, which isn't installable on this machine).

svgelements parses the SVG; each traced <path> element is one compound
shape (an outer contour plus hole subpaths, e.g. a ring or a D-shape with a
pill-shaped cutout) meant to be combined with even-odd fill. PIL has no
built-in even-odd polygon fill, so each subpath is rasterized to its own
mask and XORed together within its group, then groups are OR-ed onto the
master canvas. Supersampled and downsampled with LANCZOS for clean
anti-aliasing at the final size.
"""

import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw
from svgelements import SVG, Path as SvgPath

SVG_PATH = Path(sys.argv[1])
OUT_PATH = Path(sys.argv[2])
TARGET_SIZE = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
SUPERSAMPLE = 4


def main() -> None:
    svg = SVG.parse(str(SVG_PATH))
    paths = [e for e in svg.elements() if isinstance(e, SvgPath)]
    if not paths:
        raise SystemExit("No paths found in traced SVG")

    groups: list[list[list[tuple[float, float]]]] = []
    for p in paths:
        group = []
        for subpath in p.as_subpaths():
            sp = SvgPath(subpath)
            pts = [(sp.point(i / 200).x, sp.point(i / 200).y) for i in range(201)]
            if len(pts) >= 3:
                group.append(pts)
        if group:
            groups.append(group)

    xs = [x for group in groups for poly in group for x, _ in poly]
    ys = [y for group in groups for poly in group for _, y in poly]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    src_w, src_h = max_x - min_x, max_y - min_y

    canvas_size = TARGET_SIZE * SUPERSAMPLE
    scale = canvas_size / max(src_w, src_h)
    off_x = (canvas_size - src_w * scale) / 2
    off_y = (canvas_size - src_h * scale) / 2

    def to_canvas(pt: tuple[float, float]) -> tuple[float, float]:
        x, y = pt
        return ((x - min_x) * scale + off_x, (y - min_y) * scale + off_y)

    canvas = Image.new("1", (canvas_size, canvas_size), 0)
    for group in groups:
        group_mask = Image.new("1", (canvas_size, canvas_size), 0)
        for poly in group:
            scaled = [to_canvas(pt) for pt in poly]
            subpath_mask = Image.new("1", (canvas_size, canvas_size), 0)
            ImageDraw.Draw(subpath_mask).polygon(scaled, fill=1)
            group_mask = ImageChops.logical_xor(group_mask, subpath_mask)
        canvas = ImageChops.logical_or(canvas, group_mask)

    canvas = canvas.convert("L").resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)

    rgba = Image.new("RGBA", (TARGET_SIZE, TARGET_SIZE), (0, 0, 0, 0))
    solid = Image.new("RGBA", (TARGET_SIZE, TARGET_SIZE), (255, 255, 255, 255))
    rgba.paste(solid, (0, 0), canvas)
    rgba.save(OUT_PATH)
    bbox = rgba.getbbox()
    print(f"saved {OUT_PATH} bbox={bbox} canvas={TARGET_SIZE}")


if __name__ == "__main__":
    main()
