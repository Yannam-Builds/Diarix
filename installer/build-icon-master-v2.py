"""Proper vector-quality Diarix icon/logo master.

Previous attempts traced the raster logo then "bolded" it with a raster
distance-transform on a bitmap -- that's inherently limited by the raster
resolution at each step. This version does everything at the vector level:

1. Trace the (heavily upscaled, for curve-fit precision) source into an SVG.
2. Resolve each traced shape's even-odd fill into simple polygons using
   pyclipper (Clipper is what every real vector tool uses for this).
3. Offset (dilate) those polygons outward by a fixed real distance with
   ROUND joins via pyclipper's offsetting engine -- mathematically clean,
   not a pixel-neighborhood approximation, so curves stay perfectly round
   at any stroke width.
4. Union the result and rasterize once at very high resolution, then
   downsample with LANCZOS for every needed output size.

Produces one master PNG used for BOTH the taskbar icon and the in-app logo,
per the request to use one consistently high-quality mark everywhere.
"""

from pathlib import Path

import pyclipper
from PIL import Image, ImageDraw
from svgelements import SVG, Path as SvgPath

SVG_PATH = r"C:\Users\prana\AppData\Local\Temp\claude\Z--Diarix-Studio\ee6cca42-8a57-49b9-9ebc-555117dabcf3\scratchpad\logo_hq_traced.svg"
OUT_MASTER = r"Z:\Diarix Studio\diarix-voicebox-upstream-fork-20260713\tauri\src-tauri\icons\icon_master_v2.png"

SCALE = 1000.0          # float -> int scaling for pyclipper precision
OFFSET_DISTANCE = 1.0   # small robustness margin only -- at nonzero fill
                         # rule the D-shapes already resolve solid (the
                         # source's delicate double-outline/hollow-pill
                         # detail doesn't survive tracing at icon fidelity),
                         # and a solid bold badge reads far better at small
                         # sizes anyway. Confirmed clean at 16-1200px.
NORMALIZE_TO = 100.0
RENDER_SIZE = 3000      # final master canvas, before any per-target resize
FILL_FRACTION = 0.86


def load_groups() -> list[list[list[tuple[float, float]]]]:
    svg = SVG.parse(SVG_PATH)
    paths = [e for e in svg.elements() if isinstance(e, SvgPath)]
    groups = []
    for p in paths:
        group = []
        for subpath in p.as_subpaths():
            sp = SvgPath(subpath)
            pts = [(sp.point(i / 300).x, sp.point(i / 300).y) for i in range(301)]
            if len(pts) >= 3:
                group.append(pts)
        if group:
            groups.append(group)
    return groups


def main() -> None:
    groups = load_groups()

    xs = [x for g in groups for poly in g for x, _ in poly]
    ys = [y for g in groups for poly in g for _, y in poly]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span = max(max_x - min_x, max_y - min_y)
    norm_scale = NORMALIZE_TO / span

    def to_int_pts(poly: list[tuple[float, float]]) -> list[tuple[int, int]]:
        return [
            (round((x - min_x) * norm_scale * SCALE), round((y - min_y) * norm_scale * SCALE))
            for x, y in poly
        ]

    # Resolve each group's even-odd fill into simple filled polygons.
    resolved_polys: list[list[tuple[int, int]]] = []
    for group in groups:
        pc = pyclipper.Pyclipper()
        for poly in group:
            pc.AddPath(to_int_pts(poly), pyclipper.PT_SUBJECT, True)
        # SVG's default fill-rule is nonzero (these traced paths don't set
        # fill-rule explicitly), not evenodd -- using evenodd here collapsed
        # every hole/ring into a solid blob including the outer tick marks.
        solution = pc.Execute(
            pyclipper.CT_UNION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO
        )
        resolved_polys.extend(solution)

    # Dilate with round joins -- the actual "make it bold" step, done
    # properly on vector geometry instead of a raster neighborhood filter.
    offsetter = pyclipper.PyclipperOffset()
    offsetter.AddPaths(resolved_polys, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    dilated = offsetter.Execute(OFFSET_DISTANCE * SCALE)

    # Union the dilated pieces together in case neighboring shapes now
    # overlap (e.g. the double-ring outline collapsing into one solid band).
    pc2 = pyclipper.Pyclipper()
    pc2.AddPaths(dilated, pyclipper.PT_SUBJECT, True)
    final_polys = pc2.Execute(pyclipper.CT_UNION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)

    # Rasterize: figure out the new bounds after dilation, map to RENDER_SIZE.
    all_pts = [pt for poly in final_polys for pt in poly]
    fxs = [p[0] for p in all_pts]
    fys = [p[1] for p in all_pts]
    fminx, fmaxx = min(fxs), max(fxs)
    fminy, fmaxy = min(fys), max(fys)
    fspan = max(fmaxx - fminx, fmaxy - fminy)

    target_glyph_px = RENDER_SIZE * FILL_FRACTION
    px_scale = target_glyph_px / fspan
    off_x = (RENDER_SIZE - (fmaxx - fminx) * px_scale) / 2
    off_y = (RENDER_SIZE - (fmaxy - fminy) * px_scale) / 2

    canvas = Image.new("L", (RENDER_SIZE, RENDER_SIZE), 0)
    draw = ImageDraw.Draw(canvas)
    for poly in final_polys:
        pts = [
            ((x - fminx) * px_scale + off_x, (y - fminy) * px_scale + off_y)
            for x, y in poly
        ]
        draw.polygon(pts, fill=255)

    # Light supersample-and-downsample pass for anti-aliased edges even
    # though the geometry itself is already exact.
    big = canvas.resize((RENDER_SIZE * 2, RENDER_SIZE * 2), Image.NEAREST)
    smooth = big.resize((RENDER_SIZE, RENDER_SIZE), Image.LANCZOS)

    rgba = Image.new("RGBA", (RENDER_SIZE, RENDER_SIZE), (0, 0, 0, 0))
    white = Image.new("RGBA", (RENDER_SIZE, RENDER_SIZE), (255, 255, 255, 255))
    rgba.paste(white, (0, 0), smooth)
    rgba.save(OUT_MASTER)

    bbox = rgba.getbbox()
    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    print(f"saved {OUT_MASTER}, glyph fill {bw/RENDER_SIZE:.3f} x {bh/RENDER_SIZE:.3f}")


if __name__ == "__main__":
    main()
