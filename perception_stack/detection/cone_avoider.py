"""
PSU Eco Racing — Perception Stack
detection/cone_avoider.py  |  YOLOv5 cone detector + gap-following avoidance planner.

Architecture:
  ConeDetector       — background thread, YOLOv5 inference, queue(maxsize=1), drop-on-full.
  TrackWidthEstimator— slow EMA of track width, updated ONLY during LANE_FOLLOW.
                       Frozen during AVOIDING/RETURNING so a bad SegFormer frame
                       mid-manoeuvre cannot corrupt the corridor the planner uses.
  ConeAvoider        — stateful gap planner, one call per frame from pipeline.py.

Depth sampling (#1):
  Cone depth sampled from the body of the bbox (40–90 % height, 20 % horizontal
  inset) — avoids the reflective tip and the noisy base/road boundary.
  Consistency check (std/median > 0.25) rejects mixed foreground/background patches.
  No fallback depth — if a cone cannot be reliably localised it is ignored rather
  than accepted with a wrong position.

Gap planning:
  1. Nearest-cone group prioritization (#6): only cones within 1.5 m Z of the
     closest cone are used. Distant cones will be handled when reached.
  2. Each cone occupies [X − GAP_CONE_RADIUS_M, X + GAP_CONE_RADIUS_M].
  3. Overlapping zones merge. Open intervals = passable gaps.
  4. Clearance target: closest point to X=0 inside gap that fits car width.
  5. Score = width − GAP_CENTER_WEIGHT × |clearance_target|. Wider wins.
  6. Corridor comes from TrackWidthEstimator when reliable, else lane EMA.

State machine (#7 — three states):
  LANE_FOLLOW → AVOIDING   when a blocking cone enters AVOIDANCE_TRIGGER_M
  AVOIDING    → RETURNING  when ALL cones exit AVOIDANCE_RELEASE_M
  RETURNING   → LANE_FOLLOW when |dev_m| < RETURN_BAND_M OR after 40-frame timeout
                             (RETURNING actively steers to X=0 via Pure Pursuit)
"""

import logging
import queue
import threading
from typing import List, Optional, Tuple

import cv2
import numpy as np

from perception_stack.config import (
    CONE_MODEL_PATH,
    CONE_CONF_THRESH,
    CONE_IMG_SIZE,
    CONE_SKIP_FRAMES,
    CONE_Z_MIN_M,
    CONE_Z_MAX_M,
    AVOIDANCE_TRIGGER_M,
    AVOIDANCE_RELEASE_M,
    PATH_WIDTH_M,
    RETURN_BAND_M,
    GAP_CAR_WIDTH_M,
    GAP_CONE_RADIUS_M,
    GAP_LOOKAHEAD_M,
    GAP_CENTER_WEIGHT,
    LANE_MARGIN_M,
    SEG_NEAR_FRAC,
    ROI_TOP_FRACTION,
)

# EMA for lane bounds fallback (used when TrackWidthEstimator not yet reliable)
_LANE_BOUNDS_ALPHA = 0.45
# EMA for gap target — snaps on side-switch, smooths same-side jitter
_GAP_TARGET_ALPHA  = 0.35
# Nearest-cone group: include cones within this Z spread of the closest cone
_PRIORITY_Z_SPREAD = 1.5   # metres

log = logging.getLogger(__name__)


# ── Robust cone depth & localisation (#1) ─────────────────────────────────────

def _localise_cone_robust(
    x1: float, y1: float, x2: float, y2: float,
    depth_arr: np.ndarray,
    H: int, W: int, fx: float,
) -> Optional[Tuple[float, float, float]]:
    """
    Localise one YOLO bbox to (X_m, Z_m, confidence).  Returns None if depth
    is unreliable — never falls back to a wrong value.

    Samples the cone body (40–90 % height, 20 % horizontal inset) to avoid:
      • Reflective tip (top of cone) — overexposed, ZED returns NaN / far outliers
      • Base/road boundary (bottom 10 %) — depth discontinuity, blurred stereo edge
    Consistency check: if depth std/median > 0.25 the patch spans foreground and
    background (e.g. cone partially off-screen) — reject entirely.
    """
    bh = y2 - y1
    bw = x2 - x1

    y_top = int(y1 + 0.40 * bh)
    y_bot = int(y1 + 0.90 * bh)
    x_l   = int(max(0,     x1 + 0.20 * bw))
    x_r   = int(min(W - 1, x2 - 0.20 * bw))

    y_top = max(0, min(y_top, H - 1))
    y_bot = max(0, min(y_bot, H - 1))

    if y_bot <= y_top or x_r <= x_l:
        return None

    patch = depth_arr[y_top:y_bot, x_l:x_r]
    valid = patch[np.isfinite(patch) & (patch > 0.2) & (patch < 15.0)]

    if valid.size < 6:
        return None

    Z_median = float(np.median(valid))
    Z_std    = float(np.std(valid))

    if Z_median < 1e-3 or Z_std / Z_median > 0.25:
        return None   # inconsistent patch — mixed foreground/background

    cx_bbox = (x1 + x2) / 2.0
    X_m     = (cx_bbox - W / 2.0) * Z_median / fx
    conf    = min(1.0, valid.size / 20.0) * (1.0 - min(1.0, Z_std / Z_median))

    return X_m, Z_median, conf


def _localise_cones_robust(
    detections: List[Tuple],
    depth_arr:  np.ndarray,
    H: int, W: int, fx: float,
) -> List[Tuple[float, float, float]]:
    """Returns list of (X_m, Z_m, confidence) for each reliably measured cone."""
    out = []
    for x1, y1, x2, y2, _ in detections:
        result = _localise_cone_robust(x1, y1, x2, y2, depth_arr, H, W, fx)
        if result is None:
            continue
        X_m, Z_m, conf = result
        if CONE_Z_MIN_M < Z_m < CONE_Z_MAX_M:
            out.append((X_m, Z_m, conf))
    return out


# ── Nearest-cone group prioritization (#6) ────────────────────────────────────

def _get_priority_cones(
    blocking: List[Tuple[float, float, float]],
) -> List[Tuple[float, float, float]]:
    """
    Return only the nearest Z group of blocking cones.
    Cones further than _PRIORITY_Z_SPREAD metres behind the closest one are
    deferred — the car will replan for them when it gets closer.
    Prevents a distant cone from distorting the immediate gap.
    """
    if not blocking:
        return []
    nearest_Z = min(c[1] for c in blocking)
    return [c for c in blocking if c[1] <= nearest_Z + _PRIORITY_Z_SPREAD]


# ── Lane boundary extraction ───────────────────────────────────────────────────

def _patch_depth_road(
    depth_arr: np.ndarray, y: int, x: int, H: int, W: int, pad: int = 5
) -> float:
    """Median ZED depth for road surface patches (sl.MEASURE.DEPTH = positive)."""
    r0, r1 = max(0, y - pad), min(H, y + pad + 1)
    c0, c1 = max(0, x - pad), min(W, x + pad + 1)
    patch  = depth_arr[r0:r1, c0:c1]
    valid  = patch[np.isfinite(patch) & (patch > 0.1) & (patch < 30.0)]
    return float(np.median(valid)) if valid.size >= 3 else 3.0


def _lane_bounds_m(
    road_mask: Optional[np.ndarray],
    depth_arr: np.ndarray,
    H: int, W: int, fx: float,
) -> Tuple[float, float]:
    """
    Left/right drivable boundary in camera X (metres) at SEG_NEAR_FRAC row.
    Returns (-2.0, 2.0) fallback when mask is unavailable or too sparse.
    """
    if road_mask is None:
        return -2.0, 2.0
    y_near = int(H * SEG_NEAR_FRAC)
    row    = road_mask[y_near].copy()
    row[:int(H * ROI_TOP_FRACTION)] = False
    cols   = np.where(row)[0]
    if len(cols) < 4:
        return -2.0, 2.0
    mid    = int((int(cols[0]) + int(cols[-1])) / 2)
    Z_near = _patch_depth_road(depth_arr, y_near, mid, H, W)
    left_X  = (float(cols[0])  - W / 2.0) * Z_near / fx
    right_X = (float(cols[-1]) - W / 2.0) * Z_near / fx
    return left_X, right_X


# ── Track width estimator (#5) ────────────────────────────────────────────────

class TrackWidthEstimator:
    """
    Slow EMA of track width, updated ONLY during LANE_FOLLOW.
    Frozen during AVOIDING and RETURNING so bad SegFormer frames mid-manoeuvre
    cannot corrupt the corridor the gap planner is using.

    reliable == True after 50 updates (~1.5 s of driving at 30 fps).
    """

    def __init__(self, initial_width_m: float = 3.5):
        self._width    = initial_width_m
        self._centre   = 0.0
        self._alpha    = 0.02      # very slow — track width barely changes
        self._n        = 0

    def update(self, left_X: float, right_X: float) -> None:
        width  = right_X - left_X
        centre = (left_X + right_X) / 2.0
        if 1.5 < width < 8.0:   # sanity gate
            self._width  = self._alpha * width  + (1.0 - self._alpha) * self._width
            self._centre = self._alpha * centre + (1.0 - self._alpha) * self._centre
            self._n     += 1

    @property
    def bounds(self) -> Tuple[float, float]:
        half = self._width / 2.0
        return self._centre - half, self._centre + half

    @property
    def reliable(self) -> bool:
        return self._n >= 50


# ── Exclusion-zone gap planner ────────────────────────────────────────────────

def _clearance_target(gl: float, gr: float) -> float:
    """
    Minimum-deviation X inside [gl, gr] that keeps the car clear of both edges.
    Falls back to midpoint when the gap is narrower than the car.
    """
    half = GAP_CAR_WIDTH_M / 2.0
    lo   = gl + half
    hi   = gr - half
    if lo >= hi:
        return (gl + gr) / 2.0
    return float(np.clip(0.0, lo, hi))


def _find_best_gap(
    blocking_cones: List[Tuple[float, float, float]],   # (X_m, Z_m, conf)
    left_X_m: float,
    right_X_m: float,
) -> float:
    """
    Best gap clearance-target X through the (priority) cone field.
    Never rejects a gap — width used for scoring only.
    """
    safe_left  = left_X_m  + LANE_MARGIN_M
    safe_right = right_X_m - LANE_MARGIN_M

    if not blocking_cones:
        return 0.0
    if safe_right <= safe_left:
        return (left_X_m + right_X_m) / 2.0

    exclusions = sorted(
        (X - GAP_CONE_RADIUS_M, X + GAP_CONE_RADIUS_M)
        for X, _, _ in blocking_cones
    )
    merged: List[List[float]] = []
    for lo, hi in exclusions:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])

    gaps: List[Tuple[float, float, float]] = []
    cursor = safe_left
    for lo, hi in merged:
        if lo > cursor:
            gaps.append((cursor, lo, _clearance_target(cursor, lo)))
        cursor = max(cursor, hi)
    if safe_right > cursor:
        gaps.append((cursor, safe_right, _clearance_target(cursor, safe_right)))

    if not gaps:
        cursor = safe_left
        best_w, best_t = 0.0, (safe_left + safe_right) / 2.0
        for lo, hi in merged:
            if lo > cursor:
                w = lo - cursor
                if w > best_w:
                    best_w = w
                    best_t = _clearance_target(cursor, lo)
            cursor = max(cursor, hi)
        if safe_right > cursor and (safe_right - cursor) > best_w:
            best_t = _clearance_target(cursor, safe_right)
        return float(np.clip(best_t, safe_left, safe_right))

    best_target, best_score = 0.0, float("-inf")
    for gl, gr, ct in gaps:
        score = (gr - gl) - GAP_CENTER_WEIGHT * abs(ct)
        if score > best_score:
            best_score  = score
            best_target = ct

    return float(np.clip(best_target, safe_left, safe_right))


# ── Stateful avoidance planner ─────────────────────────────────────────────────

class ConeAvoider:
    """
    Three-state gap-following planner.

    process() returns (lookahead_world, lookahead_px, state).
      LANE_FOLLOW — returns (None, None, "LANE_FOLLOW")
      AVOIDING    — returns gap waypoint; Commander steers to it
      RETURNING   — returns (0.0, GAP_LOOKAHEAD_M); Commander steers to centre
    """

    def __init__(self):
        self._detector = ConeDetector()
        self._enabled  = False
        self._state    = "LANE_FOLLOW"
        self._gap_target: float = 0.0
        self._return_frames: int = 0

        # Track width estimator — updated only during LANE_FOLLOW
        self._track_width = TrackWidthEstimator(initial_width_m=3.5)

        # Lane bounds EMA — fallback when track width not yet reliable
        self._lane_left_ema:  float = -2.0
        self._lane_right_ema: float =  2.0
        self._lane_ema_ready: bool  = False

        # Debug state
        self._last_cones_cam:   list  = []
        self._last_blocking:    list  = []
        self._last_priority:    list  = []
        self._last_detections:  list  = []
        self._last_gap_X_raw:   float = 0.0
        self._last_gap_X_ema:   float = 0.0
        self._last_lane_ema:    tuple = (-2.0, 2.0)
        self._last_corridor:    tuple = (-2.0, 2.0)

    def init(self) -> bool:
        ok = self._detector.init()
        self._enabled = ok
        return ok

    def process(
        self,
        frame_norm: np.ndarray,
        depth_arr:  np.ndarray,
        road_mask:  Optional[np.ndarray],
        dev_m:      float,
        H: int, W: int, fx: float,
        frame_cnt:  int,
    ) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[int, int]], str]:

        if not self._enabled:
            return None, None, "LANE_FOLLOW"

        if frame_cnt % CONE_SKIP_FRAMES == 0:
            self._detector.submit(frame_norm)
        detections = self._detector.get_result()

        # ── Robust 3D localisation (#1) ───────────────────────────────────────
        cones_cam = _localise_cones_robust(detections, depth_arr, H, W, fx)

        blocking = [
            (X, Z, conf) for X, Z, conf in cones_cam
            if Z < AVOIDANCE_TRIGGER_M and abs(X) < PATH_WIDTH_M
        ]
        closest_z = min((Z for _, Z, _ in blocking), default=float("inf"))

        # Debug log
        if cones_cam:
            cone_strs = ["  X={:+.2f}m Z={:.2f}m conf={:.2f}{}".format(
                X, Z, conf,
                " [BLOCKING]" if (Z < AVOIDANCE_TRIGGER_M and abs(X) < PATH_WIDTH_M)
                else " [side/far]"
            ) for X, Z, conf in cones_cam]
            log.debug("[ConeAvoider] frame=%d  %d cone(s):\n%s",
                      frame_cnt, len(cones_cam), "\n".join(cone_strs))

        self._last_cones_cam  = cones_cam
        self._last_blocking   = blocking
        self._last_detections = detections

        # ── State machine (#7 — three states) ────────────────────────────────
        if self._state == "LANE_FOLLOW":
            if closest_z < AVOIDANCE_TRIGGER_M:
                self._state = "AVOIDING"
                log.info("[ConeAvoider] Cone at %.2fm — engaging avoidance", closest_z)

        elif self._state == "AVOIDING":
            if closest_z > AVOIDANCE_RELEASE_M:
                self._state         = "RETURNING"
                self._return_frames = 0
                self._gap_target    = 0.0
                log.info("[ConeAvoider] Cones clear — returning to centre")

        elif self._state == "RETURNING":
            self._return_frames += 1
            centred   = abs(dev_m) < RETURN_BAND_M
            timed_out = self._return_frames > 40   # 3 s at 30 fps
            if centred or timed_out:
                self._state          = "LANE_FOLLOW"
                self._lane_ema_ready = False
                log.info("[ConeAvoider] Centred (dev=%.3fm, frames=%d) — lane follow",
                         dev_m, self._return_frames)

        # ── Track width update (LANE_FOLLOW only) (#5) ────────────────────────
        raw_left, raw_right = _lane_bounds_m(road_mask, depth_arr, H, W, fx)
        fallback_left, fallback_right = -2.0, 2.0

        if self._state == "LANE_FOLLOW":
            if raw_left != fallback_left and raw_right != fallback_right:
                self._track_width.update(raw_left, raw_right)

        # ── Lane bounds EMA (fallback corridor) ───────────────────────────────
        if not self._lane_ema_ready:
            self._lane_left_ema  = raw_left  if raw_left  != fallback_left  else -0.9
            self._lane_right_ema = raw_right if raw_right != fallback_right else  0.9
            self._lane_ema_ready = True
        else:
            if self._state == "LANE_FOLLOW":
                # Only update EMA during normal driving
                if raw_left != fallback_left:
                    self._lane_left_ema = (_LANE_BOUNDS_ALPHA * raw_left
                                          + (1.0 - _LANE_BOUNDS_ALPHA) * self._lane_left_ema)
                if raw_right != fallback_right:
                    self._lane_right_ema = (_LANE_BOUNDS_ALPHA * raw_right
                                           + (1.0 - _LANE_BOUNDS_ALPHA) * self._lane_right_ema)
            # Frozen during AVOIDING and RETURNING

        # ── Corridor selection (#5) ───────────────────────────────────────────
        if self._track_width.reliable:
            corridor_left, corridor_right = self._track_width.bounds
        else:
            corridor_left  = self._lane_left_ema
            corridor_right = self._lane_right_ema

        self._last_corridor = (corridor_left, corridor_right)
        self._last_lane_ema = (self._lane_left_ema, self._lane_right_ema)

        # ── RETURNING — steer to centre ───────────────────────────────────────
        if self._state == "RETURNING":
            x_px     = int(W / 2.0)
            y_px     = int(H * SEG_NEAR_FRAC)
            world_pt = (0.0, GAP_LOOKAHEAD_M)
            pixel_pt = (x_px, y_px)
            log.debug("[ConeAvoider] RETURNING  dev=%.3fm  frames=%d",
                      dev_m, self._return_frames)
            return world_pt, pixel_pt, "RETURNING"

        if self._state != "AVOIDING":
            return None, None, "LANE_FOLLOW"

        # ── Gap planning ──────────────────────────────────────────────────────
        priority = _get_priority_cones(blocking)   # (#6) nearest group only
        self._last_priority = priority

        gap_X_raw = _find_best_gap(priority, corridor_left, corridor_right)

        # EMA — snap on side-switch, smooth same-side jitter
        prev_gap = self._gap_target
        side_switched = (gap_X_raw * self._gap_target < 0.0
                         and abs(gap_X_raw) > 0.10)
        if side_switched:
            self._gap_target = gap_X_raw
        else:
            self._gap_target = (_GAP_TARGET_ALPHA * gap_X_raw
                                + (1.0 - _GAP_TARGET_ALPHA) * self._gap_target)
        gap_X = self._gap_target

        log.debug("[ConeAvoider] gap raw=%+.3f ema=%+.3f  priority=%d/%d  "
                  "corridor=[%.2f, %.2f]  tw_reliable=%s",
                  gap_X_raw, gap_X, len(priority), len(blocking),
                  corridor_left, corridor_right,
                  self._track_width.reliable)

        Z_gap    = GAP_LOOKAHEAD_M
        x_px     = int(np.clip(W / 2.0 + gap_X * fx / Z_gap, 0, W - 1))
        y_px     = int(H * SEG_NEAR_FRAC)
        world_pt = (gap_X, Z_gap)
        pixel_pt = (x_px, y_px)

        self._last_gap_X_raw = gap_X_raw
        self._last_gap_X_ema = gap_X

        return world_pt, pixel_pt, "AVOIDING"

    @property
    def gap_target(self) -> float:
        return self._gap_target

    def debug_overlay(
        self,
        frame: np.ndarray,
        detections: Optional[list] = None,
        H: int = 0, W: int = 0, fx: float = 700.0,
    ) -> np.ndarray:
        """Full avoidance debug overlay — safe to call every frame."""
        out  = frame.copy()
        dets = detections if detections is not None else self._last_detections
        if H == 0:
            H, W = out.shape[:2]
        y_near = int(H * SEG_NEAR_FRAC)

        # ── YOLO bboxes + labels (index-safe lookup) ──────────────────────────
        _cam_lookup: dict = {}
        for det, cp in zip(self._last_detections, self._last_cones_cam):
            key = (round((det[0] + det[2]) / 2.0, 1),
                   round((det[1] + det[3]) / 2.0, 1))
            _cam_lookup[key] = cp

        priority_set = {(round(X, 3), round(Z, 3)) for X, Z, _ in self._last_priority}

        for x1, y1, x2, y2, conf in dets:
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
            key     = (round((x1 + x2) / 2.0, 1), round((y1 + y2) / 2.0, 1))
            cam_pos = _cam_lookup.get(key)
            is_blocking = cam_pos is not None and any(
                abs(cam_pos[0] - bx) < 0.02 and abs(cam_pos[1] - bz) < 0.02
                for bx, bz, _ in self._last_blocking
            )
            is_priority = cam_pos is not None and (
                round(cam_pos[0], 3), round(cam_pos[1], 3)) in priority_set

            if is_priority:
                colour = (0, 0, 220)      # red — active planning cone
            elif is_blocking:
                colour = (0, 100, 220)    # orange — blocking but deferred
            else:
                colour = (200, 200, 0)    # cyan — side/far

            cv2.rectangle(out, (ix1, iy1), (ix2, iy2), colour, 2)
            if cam_pos:
                label = "X{:+.2f} Z{:.2f}m c{:.0%}".format(
                    cam_pos[0], cam_pos[1], cam_pos[2])
            else:
                label = "conf={:.2f} (no depth)".format(conf)
            cv2.putText(out, label, (ix1, max(iy1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)

        # ── RETURNING state display ───────────────────────────────────────────
        if self._state == "RETURNING":
            cv2.line(out, (W // 2, 0), (W // 2, H), (0, 200, 255), 2)
            cv2.putText(out, "RETURNING  frame {}/40".format(self._return_frames),
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 200, 255), 2, cv2.LINE_AA)
            return out

        if self._state != "AVOIDING":
            cv2.putText(out, "LANE_FOLLOW", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2, cv2.LINE_AA)
            return out

        # ── Corridor lines ────────────────────────────────────────────────────
        corr_left, corr_right = self._last_corridor
        for X_m, col in [(corr_left, (0, 220, 220)), (corr_right, (0, 220, 220))]:
            x_px = int(np.clip(W / 2.0 + X_m * fx / GAP_LOOKAHEAD_M, 0, W - 1))
            for y in range(0, H, 16):
                cv2.line(out, (x_px, y), (x_px, min(y + 8, H - 1)), col, 1)

        # ── Gap bands ─────────────────────────────────────────────────────────
        safe_left  = corr_left  + LANE_MARGIN_M
        safe_right = corr_right - LANE_MARGIN_M
        excl = sorted(
            (X - GAP_CONE_RADIUS_M, X + GAP_CONE_RADIUS_M)
            for X, _, _ in self._last_priority
        )
        merged: List[List[float]] = []
        for lo, hi in excl:
            if merged and lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        cursor  = safe_left
        overlay = out.copy()
        for lo, hi in merged:
            if lo > cursor:
                xl = int(np.clip(W / 2.0 + cursor * fx / GAP_LOOKAHEAD_M, 0, W - 1))
                xr = int(np.clip(W / 2.0 + lo     * fx / GAP_LOOKAHEAD_M, 0, W - 1))
                cv2.rectangle(overlay, (xl, y_near - 20), (xr, y_near + 20),
                              (0, 180, 0), -1)
            xl_e = int(np.clip(W / 2.0 + lo * fx / GAP_LOOKAHEAD_M, 0, W - 1))
            xr_e = int(np.clip(W / 2.0 + hi * fx / GAP_LOOKAHEAD_M, 0, W - 1))
            cv2.rectangle(overlay, (xl_e, y_near - 20), (xr_e, y_near + 20),
                          (0, 0, 180), -1)
            cursor = max(cursor, hi)
        if safe_right > cursor:
            xl = int(np.clip(W / 2.0 + cursor     * fx / GAP_LOOKAHEAD_M, 0, W - 1))
            xr = int(np.clip(W / 2.0 + safe_right * fx / GAP_LOOKAHEAD_M, 0, W - 1))
            cv2.rectangle(overlay, (xl, y_near - 20), (xr, y_near + 20),
                          (0, 180, 0), -1)
        cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)

        # ── Gap target lines ──────────────────────────────────────────────────
        raw_x_px = int(np.clip(W / 2.0 + self._last_gap_X_raw * fx / GAP_LOOKAHEAD_M,
                               0, W - 1))
        ema_x_px = int(np.clip(W / 2.0 + self._last_gap_X_ema * fx / GAP_LOOKAHEAD_M,
                               0, W - 1))
        for y in range(0, H, 16):
            cv2.line(out, (raw_x_px, y), (raw_x_px, min(y + 8, H - 1)),
                     (255, 255, 255), 1)
        cv2.line(out, (ema_x_px, 0), (ema_x_px, H - 1), (0, 255, 80), 2)
        cv2.circle(out, (ema_x_px, y_near), 8, (0, 255, 80), -1)

        # ── HUD ───────────────────────────────────────────────────────────────
        tw_str = "TW:{:.2f}m".format(self._track_width._width) \
                 if self._track_width.reliable else "TW:warming"
        lines = [
            "AVOIDING",
            "priority:{}/{}  z_min:{:.2f}m".format(
                len(self._last_priority), len(self._last_blocking),
                min((Z for _, Z, _ in self._last_blocking), default=0.0)),
            "corridor:[{:.2f},{:.2f}]  {}".format(
                corr_left, corr_right, tw_str),
            "gap raw:{:+.3f}m  ema:{:+.3f}m".format(
                self._last_gap_X_raw, self._last_gap_X_ema),
        ]
        for i, txt in enumerate(lines):
            col = (0, 80, 255) if i == 0 else (230, 230, 230)
            cv2.putText(out, txt, (10, 28 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, col,
                        2 if i == 0 else 1, cv2.LINE_AA)
        return out


# ── Background YOLO cone detector ─────────────────────────────────────────────

class ConeDetector:
    def __init__(self):
        self._model:   object      = None
        self._enabled: bool        = False
        self._in_q:    queue.Queue = queue.Queue(maxsize=1)
        self._result:  List[Tuple] = []
        self._lock                 = threading.Lock()

    def init(self) -> bool:
        try:
            import torch
            self._model      = torch.hub.load(
                "ultralytics/yolov5", "custom",
                path=CONE_MODEL_PATH, force_reload=False, verbose=False,
            )
            self._model.conf = CONE_CONF_THRESH
            self._model.iou  = 0.45
            self._enabled    = True
            threading.Thread(
                target=self._worker, daemon=True, name="ConeDetector"
            ).start()
            log.info("[ConeDetector] Loaded %s", CONE_MODEL_PATH)
            return True
        except Exception as exc:
            log.error("[ConeDetector] Load failed: %s — cone avoidance disabled", exc)
            return False

    def submit(self, frame: np.ndarray) -> None:
        if not self._enabled:
            return
        try:
            self._in_q.put_nowait(frame.copy())
        except queue.Full:
            pass

    def get_result(self) -> List[Tuple]:
        with self._lock:
            return list(self._result)

    def _worker(self) -> None:
        while True:
            frame = self._in_q.get()
            try:
                rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res  = self._model(rgb, size=CONE_IMG_SIZE)
                dets = [
                    (float(x1), float(y1), float(x2), float(y2), float(cf))
                    for *xyxy, cf, _ in res.xyxy[0].cpu().numpy()
                    for x1, y1, x2, y2 in [xyxy]
                ]
                with self._lock:
                    self._result = dets
            except Exception as exc:
                log.warning("[ConeDetector] inference error: %s", exc)
