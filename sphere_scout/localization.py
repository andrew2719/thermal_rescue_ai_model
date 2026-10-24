"""One-person heuristic *thermal blob* locator (not a learned person segmenter).

Finds an approximate hot-region boundary and box in the ORIGINAL 24x32 pixel grid.
Does NOT assert a hotspot is a human, and returns None on ambiguous/no blobs.
"""
from __future__ import annotations
import numpy as np


def _smooth(frame: np.ndarray) -> np.ndarray:
    a = np.pad(frame, 1, mode='edge')
    return (a[:-2, :-2] + 2*a[:-2, 1:-1] + a[:-2, 2:]
            + 2*a[1:-1, :-2] + 4*a[1:-1, 1:-1] + 2*a[1:-1, 2:]
            + a[2:, :-2] + 2*a[2:, 1:-1] + a[2:, 2:]) / 16.0


def locate_one_hot_region(frame: np.ndarray, opts: dict | None = None):
    cfg = dict(min_contrast_c=2.0, min_above_background_c=1.25,
               fraction_of_peak_contrast=0.45, min_area_pixels=4,
               max_area_pixels=170, max_aspect_ratio=5.0,
               max_bbox_width_fraction=0.80)
    cfg.update(opts or {})
    h, w = frame.shape
    sm = _smooth(frame)
    bg = float(np.median(sm))
    delta = float(sm.max() - bg)
    if delta < cfg['min_contrast_c']:
        return None
    threshold = bg + max(cfg['min_above_background_c'], cfg['fraction_of_peak_contrast'] * delta)
    hot = sm >= threshold
    visited = np.zeros((h, w), dtype=bool)
    candidates = []
    for row in range(h):
        for col in range(w):
            if visited[row, col] or not hot[row, col]:
                continue
            queue = [(row, col)]
            visited[row, col] = True
            part = []
            while queue:
                y, x = queue.pop()
                part.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < h and 0 <= xx < w and hot[yy, xx] and not visited[yy, xx]:
                            visited[yy, xx] = True
                            queue.append((yy, xx))
            pts = np.asarray(part, dtype=int)
            ys, xs = pts.T
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1
            area = len(part)
            bw, bh = x2-x1, y2-y1
            if not (cfg['min_area_pixels'] <= area <= cfg['max_area_pixels']):
                continue
            # Reject nearly full-width warm background bands (common in simulation).
            # These are not credible single-person outlines.
            if bw > w * cfg['max_bbox_width_fraction']:
                continue
            if min(bw, bh) < 2 or max(bw / bh, bh / bw) > cfg['max_aspect_ratio']:
                continue
            member = set(part)
            edges = sorted((x, y) for y, x in member
                           if any((y + dy, x + dx) not in member for dy, dx in ((0,1),(0,-1),(1,0),(-1,0))))
            strength = float(sm[ys, xs].mean() - bg)
            at_border = x1 == 0 or y1 == 0 or x2 == w or y2 == h
            score = strength * float(np.sqrt(area)) * (0.65 if at_border else 1.0)
            candidates.append((score, (x1, y1, x2, y2), edges, area))
    if not candidates:
        return None
    _, (x1, y1, x2, y2), edge_points, area = max(candidates, key=lambda t: t[0])
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    cell_r = min(int(cy / (h / 3)), 2)
    cell_c = min(int(cx / (w / 4)), 3)
    return {
        'type': 'thermal_hotspot_estimate',
        'bbox_xyxy_pixels': [x1, y1, x2, y2],  # x2,y2 are exclusive
        'bbox_xyxy_normalized': [x1/w, y1/h, x2/w, y2/h],
        'center_pixel_xy': [cx, cy],
        'center_uv_normalized': [cx/w, cy/h],
        'grid_cell_row_col': [cell_r, cell_c],
        'edge_pixels_xy': [[x, y] for x, y in edge_points],  # thermal boundary, not human silhouette
        'region_area_pixels': area,
    }
