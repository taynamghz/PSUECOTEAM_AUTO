"""
PSU Eco Racing — Perception Stack
lane/control.py  |  Control-level outputs from polynomial lane fits.

Computes heading angle, curvature, and lookahead point reusing the
existing smoothed polynomial fits — no new expensive steps.

Coordinate conventions (ZED RIGHT_HANDED_Y_UP):
  X  : lateral (positive = right)
  Z  : forward distance = abs(pt[2])  (ZED Z is negative-forward)
  y  : image row — large y = near vehicle, small y = far ahead
"""

import numpy as np
from typing import Optional, Tuple

from perception_stack.config import (
    CTRL_LOOKAHEAD_M,
    CTRL_HEADING_ALPHA,
    CTRL_CURVATURE_ALPHA,
    CTRL_EVAL_Y_FRAC,
    ROI_TOP_FRACTION,
    SEG_FAR_FRAC,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _center_fit(
    lf: Optional[np.ndarray],
    rf: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Average left and right fits into a centerline poly.  Uses whichever side
    is available when the other is None."""
    if lf is not None and rf is not None:
        return (lf + rf) / 2.0
    return lf if lf is not None else rf


# ── Public computation functions ───────────────────────────────────────────────

def compute_heading(
    lf: Optional[np.ndarray],
    rf: Optional[np.ndarray],
    y_eval: float,
) -> float:
    """
    Lane heading angle θ at image row y_eval.

    For the polynomial  x = a·y² + b·y + c  the tangent slope is
        dx/dy = 2a·y + b
    so θ = arctan(2a·y + b).

    Returns radians.  Positive θ means the lane direction tilts toward
    increasing x (rightward in image), which indicates a left road curve
    ahead (standard perspective geometry).
    """
    cf = _center_fit(lf, rf)
    if cf is None:
        return 0.0
    a, b = float(cf[0]), float(cf[1])
    return float(np.arctan(2.0 * a * y_eval + b))


def compute_curvature(
    lf: Optional[np.ndarray],
    rf: Optional[np.ndarray],
    y_eval: float,
    lane_width_px: float,
    lane_width_m: float,
) -> float:
    """
    Signed metric curvature  κ = 1/R  (m⁻¹).

    Derivation:
        κ_px = f''(y) / (1 + f'(y)²)^(3/2)   [image-space, units 1/px]
             = 2a / (1 + (2ay+b)²)^(3/2)

    Metric conversion via lane-width calibration:
        px_per_m ≈ lane_width_px / lane_width_m
        κ_metric = κ_px * px_per_m

    Sign convention: positive κ = curving left (road bends left ahead).
    Returns 0.0 when lane width calibration is unavailable.
    """
    cf = _center_fit(lf, rf)
    if cf is None or lane_width_px < 1.0 or lane_width_m < 0.1:
        return 0.0
    a, b = float(cf[0]), float(cf[1])
    slope = 2.0 * a * y_eval + b
    kappa_px = (2.0 * a) / (1.0 + slope ** 2) ** 1.5
    px_per_m = lane_width_px / lane_width_m
    return float(kappa_px * px_per_m)


def compute_lookahead(
    lf: Optional[np.ndarray],
    rf: Optional[np.ndarray],
    H: int,
    W: int,
    fx: float,
    depth_arr: Optional[np.ndarray] = None,
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[int, int]]]:
    """
    Lookahead point on the lane centreline.

    Evaluates the centreline polynomial at SEG_FAR_FRAC (the lookahead row),
    reads actual ZED Z depth at that pixel via a patch median (handles road
    slopes, camera tilt, terrain variation), then converts lateral offset to
    metres via the pinhole model:

        X_m = (cx_pixel - W/2) * Z_m / fx

    Using real Z from ZED means X_m is accurate regardless of road slope or
    camera mounting variation — no flat-road assumption needed.

    Falls back to CTRL_LOOKAHEAD_M for Z when depth is unavailable or sparse.

    Returns
    -------
    world_pt : (X_m, Z_m) | None
        X_m — lateral offset in metres (positive = road centre right of camera)
        Z_m — actual forward distance to lookahead point in metres
    pixel_pt : (x, y) | None
        Pixel coordinates of the lookahead point.
    """
    cf = _center_fit(lf, rf)
    if cf is None or fx <= 0:
        return None, None

    y_look = int(H * SEG_FAR_FRAC)
    cx = float(np.polyval(cf, y_look))
    if not (0.0 <= cx <= W):
        return None, None

    cx_int = int(np.clip(cx, 0, W - 1))

    # Sample a patch of ZED depth around the centerline pixel.
    # Median over the patch rejects NaN holes and single-pixel outliers.
    # Falls back to assumed depth when the patch has insufficient valid pixels.
    Z_m = CTRL_LOOKAHEAD_M   # default fallback
    if depth_arr is not None:
        pad = 5
        r0, r1 = max(0, y_look - pad), min(H, y_look + pad + 1)
        c0, c1 = max(0, cx_int  - pad), min(W, cx_int  + pad + 1)
        patch = depth_arr[r0:r1, c0:c1]
        # ZED RIGHT_HANDED_Y_UP: forward = −Z  (objects ahead have negative Z)
        valid = patch[np.isfinite(patch) & (patch < -0.1) & (patch > -30.0)]
        if valid.size >= 4:
            Z_m = float(np.median(np.abs(valid)))

    X_m      = (cx - W / 2.0) * Z_m / fx
    pixel_pt = (cx_int, y_look)
    return (X_m, Z_m), pixel_pt


# ── Temporal smoother for scalar control outputs ───────────────────────────────

class ControlSmoother:
    """
    Separate EMA for heading angle and curvature.

    Uses lower alpha values than the polynomial smoother so that these
    derived quantities (which amplify high-frequency noise) are extra stable.
    """

    def __init__(self) -> None:
        self._heading:   Optional[float] = None
        self._curvature: Optional[float] = None

    def update(self, heading: float, curvature: float) -> Tuple[float, float]:
        if self._heading is None:
            self._heading = heading
        else:
            self._heading = (CTRL_HEADING_ALPHA * heading
                             + (1.0 - CTRL_HEADING_ALPHA) * self._heading)

        if self._curvature is None:
            self._curvature = curvature
        else:
            self._curvature = (CTRL_CURVATURE_ALPHA * curvature
                               + (1.0 - CTRL_CURVATURE_ALPHA) * self._curvature)

        return self._heading, self._curvature
