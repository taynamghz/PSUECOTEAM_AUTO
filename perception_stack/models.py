"""
PSU Eco Racing — Perception Stack
models.py  |  Shared data structures passed between modules and to the controller.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class PerceptionResult:
    # ── Lane ──────────────────────────────────────────────────────────────────
    deviation_m:  float = 0.0          # + = vehicle left of centre, - = right
    confidence:   float = 0.0          # average of left/right boundary confidence
    lane_width_m: float = 0.0
    source:       str   = "NONE"       # SEGFORMER | SEG_PARTIAL | LOST | DISABLED
    left_fit:     Optional[np.ndarray] = None   # [a,b,c]  x = poly(y)
    right_fit:    Optional[np.ndarray] = None
    left_conf:    float = 0.0
    right_conf:   float = 0.0

    # ── Stop line ─────────────────────────────────────────────────────────────
    stop_line:      bool          = False
    stop_line_y:    Optional[int] = None
    stop_line_dist: float         = 0.0

    # ── Stop sign ─────────────────────────────────────────────────────────────
    stop_sign:          bool                            = False
    stop_sign_dist_m:   float                           = 0.0
    stop_sign_bbox:     Optional[Tuple[int,int,int,int]] = None   # (x,y,w,h) px

    # ── Control outputs ───────────────────────────────────────────────────────
    heading_angle:   float = 0.0   # radians; θ = arctan(2ay+b) at CTRL_EVAL_Y_FRAC
    curvature:       float = 0.0   # κ = 1/R (m⁻¹); signed — positive = left curve
    lookahead_point: Optional[Tuple[float,float]] = None   # (X_m, Z_m) world space
    lookahead_pixel: Optional[Tuple[int,int]]     = None   # (x, y) image pixels

    # ── Vehicle state ─────────────────────────────────────────────────────────
    speed_kmh: float = 0.0         # current vehicle speed from LLC UART

    # ── Safety ────────────────────────────────────────────────────────────────
    emergency_stop:  bool = False          # True when LOST for > LOST_BRAKE_FRAMES consecutive frames
    avoidance_state: str  = "LANE_FOLLOW"  # LANE_FOLLOW | AVOIDING
