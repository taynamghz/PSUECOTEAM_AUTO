"""
PSU Eco Racing — Perception Stack
detection/road_validator.py  |  Grass/asphalt boundary validator.

How it works
────────────
After Segformer produces a road mask, this module samples a thin strip of
pixels just inside each detected road boundary (left and right) at evenly-
spaced rows.  If that strip is predominantly grass-coloured (green HSV), the
mask is trimmed inward — column by column — until the first non-grass pixel.

The trimmed mask feeds back into _scan_roads → polyfit → centerline, so the
perceived road center shifts *away* from the grass side.  Pure Pursuit then
naturally steers away from the grass without any override.

States per row
──────────────
  inner=asphalt, outer=anything  → boundary confirmed, no change
  inner=grass                    → trim boundary inward to asphalt edge
  both sides grass               → no trim (full off-road — LOST handles this)

Runs in the Segformer worker thread immediately after _road_mask().
Cost: ~1-2 ms (pure NumPy + one cv2.inRange call, no inference).
"""

import numpy as np
import cv2

from perception_stack.config import (
    GRASS_H_MIN, GRASS_H_MAX, GRASS_S_MIN, GRASS_V_MIN,
    GRASS_INNER_PAD, GRASS_FRAC_THRESH, GRASS_MAX_TRIM_FRAC,
    SEG_BOUNDARY_ROWS,
)


def validate_boundaries(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    roi_top: int,
) -> np.ndarray:
    """
    Trim grass pixels from the edges of the Segformer road mask.

    Parameters
    ----------
    frame_bgr : H×W×3 BGR image (CLAHE-normalised, same frame fed to Segformer)
    mask      : H×W bool road mask from Segformer  (True = road)
    roi_top   : first row to scan — rows above this are sky/bonnet, skip them

    Returns
    -------
    corrected : H×W bool mask — same as input but with grass-intruding boundary
                pixels set to False on affected rows.
    """
    H, W = mask.shape

    # Build grass mask once for the whole frame — cheap single HSV threshold
    hsv   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(
        hsv,
        np.array([GRASS_H_MIN, GRASS_S_MIN, GRASS_V_MIN], dtype=np.uint8),
        np.array([GRASS_H_MAX, 255,          255         ], dtype=np.uint8),
    ).astype(bool)   # True where pixel looks like grass

    corrected = mask.copy()

    scan_rows = np.linspace(roi_top, H - 1, SEG_BOUNDARY_ROWS, dtype=int)

    for r in scan_rows:
        road_cols = np.where(mask[r])[0]
        if len(road_cols) < 20:
            continue

        lx      = int(road_cols.min())
        rx      = int(road_cols.max())
        lane_w  = rx - lx
        if lane_w < 30:
            continue

        max_trim = max(1, int(lane_w * GRASS_MAX_TRIM_FRAC))

        # ── Right boundary ────────────────────────────────────────────────
        rs = max(lx + 1, rx - GRASS_INNER_PAD)
        re = rx
        if re > rs and grass[r, rs:re].mean() >= GRASS_FRAC_THRESH:
            new_rx = rx - max_trim          # worst case: full max trim
            for col in range(rx, max(lx, rx - max_trim) - 1, -1):
                if not grass[r, col]:
                    new_rx = col            # first asphalt pixel scanning inward
                    break
            if new_rx < rx:
                corrected[r, new_rx + 1 : rx + 1] = False

        # ── Left boundary ─────────────────────────────────────────────────
        ls = lx
        le = min(rx - 1, lx + GRASS_INNER_PAD)
        if le > ls and grass[r, ls:le].mean() >= GRASS_FRAC_THRESH:
            new_lx = lx + max_trim          # worst case: full max trim
            for col in range(lx, min(rx, lx + max_trim) + 1):
                if not grass[r, col]:
                    new_lx = col            # first asphalt pixel scanning inward
                    break
            if new_lx > lx:
                corrected[r, lx : new_lx] = False

    return corrected
