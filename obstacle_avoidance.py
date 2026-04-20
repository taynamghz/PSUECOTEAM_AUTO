"""
PSU Eco Racing — Shell Eco-Marathon Autonomous Division
obstacle_avoidance.py  |  Standalone cone avoidance — fully self-contained.

FIXES applied over original:
  [SM-1]  State release uses estimated lateral pose error, not _gap_target proxy.
  [SM-2]  Dwell timer: AVOIDING_MIN_DWELL_FRAMES frames before release allowed.
  [SM-3]  CONE_Z_BLOCKING_MIN_M removed — replaced with cone-passed tracking so
          cones are never dropped mid-manoeuvre.
  [GP-1]  Gap planner operates per-cone-Z plane (two planes when cones differ by
          > MULTI_PLANE_Z_THRESH), returns the intersection of passable corridors.
  [GP-2]  Explicit NO_SAFE_GAP state when every gap is < MIN_GAP_M; car stops.
  [GP-3]  Gap score normalised: score = (width/MIN_GAP_M) - GAP_CENTER_WEIGHT*|gc|
  [PP-1]  Avoidance steers with a two-point pursuit: lateral offset fed as an
          arc to a waypoint at AVOID_PURSUIT_Z ahead, not at the cone's Z.
  [PP-2]  AVOID_PURSUIT_Z raised to 3.0 m; trigger raised to 4.0 m so the car
          has time to manoeuvre before reaching the obstacle.
  [CM-1]  Ghost-cone suppression: cones behind the car's current Z are expired
          immediately regardless of age.
  [CM-2]  Depth patch size scales with Z: larger patch at long range.
  [CM-3]  Depth sampling from vertical-centre of bbox, not just base row.
  [SEG-1] lane_bounds_m called once per frame; result cached and shared between
          control and display.
  [SEG-2] Fallback width aligned with PATH_WIDTH_M (±1.5 m, not ±1.2 m).
  [SEG-3] Row sampling increased to 12 samples to reduce sensitivity to gaps.
  [CL-1]  CLAHE applied only for Segformer + display; raw frame sent to YOLO.
"""

import math
import queue
import threading
import time
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pyzed.sl as sl
import torch

from perception_stack.perception.segformer_lane import SegformerLane
from perception_stack.lane.control import compute_lookahead
from perception_stack.control.uart import UARTController
from perception_stack.config import (
    CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE,
    SEG_NEAR_FRAC, SEG_FAR_FRAC, SEG_FIT_TOP_FRAC,
    ROI_TOP_FRACTION,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

# Camera
CAM_RES        = sl.RESOLUTION.HD720
CAM_FPS        = 30
CAM_DEPTH_MODE = sl.DEPTH_MODE.PERFORMANCE

# UART
UART_ENABLED       = True
UART_PORT          = "/dev/ttyTHS1"
UART_BAUD          = 115200
UART_TIMEOUT_S     = 0.01
UART_ACK_TIMEOUT_S = 0.05
UART_HEARTBEAT_S   = 0.080

# Speed
SPEED_STRAIGHT_KMH = 3.0
SPEED_AVOID_KMH    = 2.0
SPEED_STOP_KMH     = 0.0          # [GP-2] used when NO_SAFE_GAP

# Lane-following pure pursuit
WHEELBASE_M           = 1.6
CTRL_LOOKAHEAD_M      = 2.2
CTRL_LANE_DEADBAND_M  = 0.15
STEER_MAX_DEG         = 25.0
STEER_RATE_LIMIT_DEG  = 5.0
STEER_EMA_ALPHA       = 0.20
STEER_TX_DEADBAND_DEG = 1.5

# Cone detector (YOLOv5 custom model)
YOLO_MODEL_PATH  = "best (cones).pt"
YOLO_CONF_THRESH = 0.25
YOLO_IMG_SIZE    = 416
YOLO_SKIP_FRAMES = 2

# Cone 3D localisation
# [CM-2] pad now computed dynamically — see _patch_depth()
CONE_Z_MIN_M   = 0.3
CONE_Z_MAX_M   = 15.0            # track further ahead for earlier warning

# World-frame cone map
CONE_MERGE_RADIUS_M = 0.50
CONE_MAX_AGE_S      = 4.0

# [SM-3] removed CONE_Z_BLOCKING_MIN_M — replaced by passed-cone tracking

# Avoidance trigger
# [PP-2] Raised trigger to 4.0 m and release to 5.0 m for more reaction time.
AVOIDANCE_TRIGGER_M  = 4.0
AVOIDANCE_RELEASE_M  = 5.0
PATH_WIDTH_M         = 1.5       # [SEG-2] aligned with ±1.5 m fallback

# [SM-2] Minimum frames to stay in AVOIDING before release is evaluated
AVOIDING_MIN_DWELL_FRAMES = 20   # at 30 fps ≈ 0.67 s

# [SM-1] Release checks lateral pose error, not gap target
# If the car's estimated lateral position relative to lane centre is within
# this band AND dwell timer is satisfied AND path is clear → release.
LATERAL_RETURN_BAND_M = 0.25

# Gap planner
CAR_WIDTH_M          = 0.90
GAP_SAFETY_MARGIN_M  = 0.20
MIN_GAP_M            = CAR_WIDTH_M + 2 * GAP_SAFETY_MARGIN_M
CONE_RADIUS_M        = 0.15

# [PP-2] Avoidance pure-pursuit fires to a waypoint at this fixed depth.
# Must be > AVOIDANCE_TRIGGER_M so the lookahead is beyond the obstacle.
AVOID_PURSUIT_Z      = 3.0

GAP_CENTER_WEIGHT    = 0.40
GAP_DEADBAND_M       = 0.05
LANE_MARGIN_M        = 0.25

# [GP-1] When two cones differ in Z by more than this, evaluate two planes.
MULTI_PLANE_Z_THRESH = 0.40

# Depth cache
DEPTH_REFRESH_EVERY = 4

# Display
DISPLAY  = True
WIN_NAME = "Cone Avoidance"

CLR_CONE     = (0,  165, 255)
CLR_LANE_L   = (255, 100,   0)
CLR_LANE_R   = (100, 200, 255)
CLR_GAP_CAND = (160, 160, 160)
CLR_GAP_SEL  = (0,   220,  80)
CLR_WARN     = (0,    80, 255)
CLR_OK       = (0,   200,  80)
CLR_PATH     = (0,   200,   0)
CLR_STOP     = (0,     0, 200)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _patch_depth(depth_arr: np.ndarray, y: int, x: int,
                 H: int, W: int, z_hint: float = 2.0) -> float:
    """
    [CM-2] Patch size scales with estimated distance so the sampling window
    covers a roughly constant physical area (~30 cm radius) regardless of Z.
    fx ≈ 700 px for HD720; r_px = 0.30 * fx / Z clamped to [4, 20].
    """
    fx_approx = 700.0
    pad = int(np.clip(0.30 * fx_approx / max(z_hint, 0.5), 4, 20))
    r0, r1 = max(0, y - pad), min(H, y + pad + 1)
    c0, c1 = max(0, x - pad), min(W, x + pad + 1)
    patch  = depth_arr[r0:r1, c0:c1]
    valid  = patch[np.isfinite(patch) & (patch > 0.1) & (patch < 30.0)]
    return float(np.median(valid)) if valid.size >= 3 else CTRL_LOOKAHEAD_M


def _deg_to_byte(deg: float) -> int:
    return int(max(0, min(255, round(127.0 + deg * 127.0 / STEER_MAX_DEG))))


# ─────────────────────────────────────────────────────────────────────────────
# Cone detector — YOLOv5, background thread
# [CL-1] Receives raw (non-CLAHE) frame
# ─────────────────────────────────────────────────────────────────────────────

class ConeDetector:
    def __init__(self):
        self._model:  object       = None
        self._in_q:   queue.Queue  = queue.Queue(maxsize=1)
        self._out_q:  queue.Queue  = queue.Queue(maxsize=1)
        self._result: List[Tuple]  = []

    def init(self) -> bool:
        import os
        if not os.path.isfile(YOLO_MODEL_PATH):
            log.error("[ConeDetector] Model not found: %s", YOLO_MODEL_PATH)
            return False
        try:
            self._model      = torch.hub.load(
                "ultralytics/yolov5", "custom",
                path=YOLO_MODEL_PATH, force_reload=False, verbose=False,
            )
            self._model.conf = YOLO_CONF_THRESH
            self._model.iou  = 0.45
            if torch.cuda.is_available():
                self._model = self._model.cuda().half()
                log.info("[ConeDetector] Loaded %s — CUDA FP16", YOLO_MODEL_PATH)
            else:
                log.info("[ConeDetector] Loaded %s — CPU", YOLO_MODEL_PATH)
            threading.Thread(target=self._worker, daemon=True,
                             name="ConeDetector").start()
            return True
        except Exception as exc:
            log.error("[ConeDetector] Load failed: %s", exc)
            return False

    def submit(self, raw_frame: np.ndarray) -> None:
        """[CL-1] Always submit the RAW frame — do NOT apply CLAHE before this."""
        try:
            self._in_q.put_nowait(raw_frame)
        except queue.Full:
            pass

    def get_result(self) -> List[Tuple]:
        while True:
            try:
                self._result = self._out_q.get_nowait()
            except queue.Empty:
                break
        return self._result

    def _worker(self) -> None:
        while True:
            frame = self._in_q.get()
            try:
                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self._model(rgb, size=YOLO_IMG_SIZE)
                dets = []
                for *xyxy, conf, _ in results.xyxy[0].cpu().float().numpy():
                    dets.append((float(xyxy[0]), float(xyxy[1]),
                                 float(xyxy[2]), float(xyxy[3]), float(conf)))
                if dets:
                    log.info("[ConeDetector] %d cone(s): %s",
                             len(dets), [f"conf={c:.2f}" for *_, c in dets])
                try:
                    self._out_q.put_nowait(dets)
                except queue.Full:
                    try:
                        self._out_q.get_nowait()
                    except queue.Empty:
                        pass
                    self._out_q.put_nowait(dets)
            except Exception as exc:
                log.warning("[ConeDetector] inference error: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# World-frame cone map
# [CM-1] Ghost-cone suppression: cones behind the car are expired immediately.
# ─────────────────────────────────────────────────────────────────────────────

class ConeWorldMap:
    """
    Each entry: [X_w, Z_w, timestamp, passed]
    'passed' is set True when the cone transforms into negative Z in camera
    frame (i.e., the car has driven past it).  Passed cones are excluded from
    all blocking / planning queries so they can never re-trigger avoidance.
    """

    def __init__(self):
        self._cones: List[List] = []   # [X_w, Z_w, timestamp, passed]

    def update(self, cam_cones: List[Tuple[float, float]], pose: sl.Pose) -> None:
        now = time.monotonic()
        tx, tz, R2 = self._pose_xz(pose)
        for X_c, Z_c in cam_cones:
            wpt  = R2 @ np.array([X_c, Z_c]) + np.array([tx, tz])
            X_w, Z_w = float(wpt[0]), float(wpt[1])
            merged = False
            for e in self._cones:
                if math.hypot(e[0] - X_w, e[1] - Z_w) < CONE_MERGE_RADIUS_M:
                    e[0] = 0.7 * e[0] + 0.3 * X_w
                    e[1] = 0.7 * e[1] + 0.3 * Z_w
                    e[2] = now
                    merged = True
                    break
            if not merged:
                self._cones.append([X_w, Z_w, now, False])
        self._cones = [e for e in self._cones if now - e[2] < CONE_MAX_AGE_S]

    def in_camera_frame(self, pose: sl.Pose) -> List[Tuple[float, float]]:
        """
        Returns only cones that are:
          - ahead (Z_c > CONE_Z_MIN_M) and within range (Z_c < CONE_Z_MAX_M)
          - not yet passed
        [CM-1] Marks cones as 'passed' the moment they go behind the car (Z_c ≤ 0).
        """
        tx, tz, R2 = self._pose_xz(pose)
        R2_inv = R2.T
        out = []
        for e in self._cones:
            if e[3]:   # already passed — skip
                continue
            cam  = R2_inv @ (np.array([e[0], e[1]]) - np.array([tx, tz]))
            X_c, Z_c = float(cam[0]), float(cam[1])
            if Z_c <= 0.0:                  # [CM-1] car drove past — mark passed
                e[3] = True
                continue
            if CONE_Z_MIN_M < Z_c < CONE_Z_MAX_M:
                out.append((X_c, Z_c))
        return out

    @staticmethod
    def _pose_xz(pose: sl.Pose):
        tx = pose.get_translation().get()[0]
        tz = pose.get_translation().get()[2]
        R  = np.array(pose.get_rotation_matrix().r, dtype=np.float64).reshape(3, 3)
        R2 = R[[0, 2]][:, [0, 2]]
        return tx, tz, R2


# ─────────────────────────────────────────────────────────────────────────────
# Cone 3D localisation
# [CM-2] Adaptive patch size; [CM-3] sample from vertical bbox centre, not base
# ─────────────────────────────────────────────────────────────────────────────

def localise_cones(
    detections: List[Tuple],
    depth_arr:  np.ndarray,
    H: int, W: int, fx: float,
) -> List[Tuple[float, float]]:
    out = []
    for x1, y1, x2, y2, _ in detections:
        cx = (x1 + x2) / 2.0
        # [CM-3] Use vertical centre of bbox for more reliable depth sampling.
        cy = (y1 + y2) / 2.0
        xi = int(np.clip(cx, 0, W - 1))
        yi = int(np.clip(cy, 0, H - 1))
        # First pass with a generic z_hint; [CM-2] refine patch once we have Z.
        Z_m = _patch_depth(depth_arr, yi, xi, H, W, z_hint=2.0)
        if not (CONE_Z_MIN_M < Z_m < CONE_Z_MAX_M):
            continue
        # Refine with actual Z for better patch sizing
        Z_m = _patch_depth(depth_arr, yi, xi, H, W, z_hint=Z_m)
        if not (CONE_Z_MIN_M < Z_m < CONE_Z_MAX_M):
            continue
        X_m = (cx - W / 2.0) * Z_m / fx
        out.append((X_m, Z_m))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Lane boundary extraction
# [SEG-1] Returns a cached FrameLaneBounds named tuple used by both control and display.
# [SEG-2] Fallback is ±PATH_WIDTH_M (was ±1.2 m, now ±1.5 m to match PATH_WIDTH_M).
# [SEG-3] Row sampling increased to 12 samples.
# ─────────────────────────────────────────────────────────────────────────────

def lane_bounds_m(
    road_mask: Optional[np.ndarray],
    depth_arr: np.ndarray,
    H: int, W: int, fx: float,
) -> Tuple[float, float]:
    """
    Left/right drivable edge in camera X (metres).
    [SEG-2] Fallback returns ±PATH_WIDTH_M (was ±1.2, now ±1.5).
    [SEG-3] Samples up to 12 row intervals instead of 6.
    """
    fallback = (-PATH_WIDTH_M, PATH_WIDTH_M)

    if road_mask is None or not road_mask.any():
        return fallback

    y_near   = int(H * SEG_NEAR_FRAC)
    y_far    = int(H * SEG_FAR_FRAC)
    safe_top = int(H * ROI_TOP_FRACTION)

    left_cols: List[int]  = []
    right_cols: List[int] = []
    # [SEG-3] 12 samples instead of 6
    step = max(1, (y_near - y_far) // 12)
    for y in range(y_far, y_near + 1, step):
        row = road_mask[y].copy()
        row[:safe_top] = False
        cols = np.where(row)[0]
        if len(cols) >= 4:
            left_cols.append(int(cols[0]))
            right_cols.append(int(cols[-1]))

    if len(left_cols) < 3:           # need at least 3 valid rows for median
        return fallback

    left_col  = int(np.median(left_cols))
    right_col = int(np.median(right_cols))
    mid       = (left_col + right_col) // 2
    Z_near    = _patch_depth(depth_arr, y_near, mid, H, W)
    left_X    = (float(left_col)  - W / 2.0) * Z_near / fx
    right_X   = (float(right_col) - W / 2.0) * Z_near / fx
    log.debug("[lane_bounds] rows=%d  L=%.2f m  R=%.2f m",
              len(left_cols), left_X, right_X)
    return left_X, right_X


# ─────────────────────────────────────────────────────────────────────────────
# Path-blocking check
# [SM-3] No longer filters by CONE_Z_BLOCKING_MIN_M.  Passed-cone tracking in
#        ConeWorldMap handles the "car already beside the cone" case correctly.
# ─────────────────────────────────────────────────────────────────────────────

def path_blocking_cones(
    cones_cam: List[Tuple[float, float]],
    left_X_m:  float,
    right_X_m: float,
) -> List[Tuple[float, float]]:
    """
    Cones within the drivable corridor and forward of the car.
    No minimum Z floor — passed-cone tracking handles near cones safely.
    """
    return [
        (X, Z) for (X, Z) in cones_cam
        if CONE_Z_MIN_M < Z < AVOIDANCE_TRIGGER_M and left_X_m <= X <= right_X_m
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Gap planner
# [GP-1] Multi-plane evaluation: intersect corridors from all distinct Z planes.
# [GP-2] Returns NO_SAFE_GAP flag when nothing passable exists.
# [GP-3] Score normalised by MIN_GAP_M.
# ─────────────────────────────────────────────────────────────────────────────

def _gaps_at_plane(
    cones_at_plane: List[Tuple[float, float]],
    safe_left: float,
    safe_right: float,
) -> List[Tuple[float, float, float]]:
    """
    Compute passable gaps (left, right, centre) within [safe_left, safe_right]
    given a set of cones projected onto a single Z plane.
    """
    raw_excl = sorted(
        (X - CONE_RADIUS_M, X + CONE_RADIUS_M)
        for X, _ in cones_at_plane
    )
    merged: List[List[float]] = []
    for lo, hi in raw_excl:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])

    gaps: List[Tuple[float, float, float]] = []
    cursor = safe_left
    for lo, hi in merged:
        if lo - cursor >= MIN_GAP_M:
            centre = (cursor + lo) / 2.0
            gaps.append((cursor, lo, centre))
        cursor = max(cursor, hi)
    if safe_right - cursor >= MIN_GAP_M:
        centre = (cursor + safe_right) / 2.0
        gaps.append((cursor, safe_right, centre))
    return gaps


def find_best_gap(
    blocking_cones: List[Tuple[float, float]],
    left_X_m: float,
    right_X_m: float,
) -> Tuple[float, float, List, List, float, float, bool]:
    """
    Returns:
        gap_centre_X   – X target for steering (m)
        gap_Z          – Z at which waypoint is placed (= AVOID_PURSUIT_Z)
        all_gaps       – list of (left, right, centre) for display
        merged_excl    – merged exclusion intervals for display
        safe_left      – left safe boundary (m)
        safe_right     – right safe boundary (m)
        no_safe_gap    – [GP-2] True when NO passable gap exists

    [GP-1] Multi-plane: groups cones by Z cluster and intersects the set of
    passable X corridors across all planes.  The valid steering corridor is the
    intersection (narrowest common passable region).

    [GP-3] Score = (width / MIN_GAP_M) - GAP_CENTER_WEIGHT * |centre|
    Normalising by MIN_GAP_M keeps the width term dimensionless and comparable
    across scenarios where gaps differ by a fraction of a metre.
    """
    safe_left  = left_X_m  + LANE_MARGIN_M
    safe_right = right_X_m - LANE_MARGIN_M

    if not blocking_cones or safe_right - safe_left < MIN_GAP_M:
        return 0.0, AVOID_PURSUIT_Z, [], [], safe_left, safe_right, False

    # ── [GP-1] Cluster cones into Z planes ───────────────────────────────────
    cones_sorted = sorted(blocking_cones, key=lambda c: c[1])
    planes: List[List[Tuple[float, float]]] = []
    current_plane: List[Tuple[float, float]] = [cones_sorted[0]]
    for cone in cones_sorted[1:]:
        if cone[1] - current_plane[-1][1] <= MULTI_PLANE_Z_THRESH:
            current_plane.append(cone)
        else:
            planes.append(current_plane)
            current_plane = [cone]
    planes.append(current_plane)

    # ── Compute gaps per plane, then intersect ────────────────────────────────
    # Represent the valid X corridor as a union of intervals per plane.
    # The intersection across planes gives the set of X positions passable at
    # ALL planes simultaneously.

    # Start with the full safe corridor as a list of intervals.
    valid_intervals: List[Tuple[float, float]] = [(safe_left, safe_right)]

    all_gaps_for_display: List[Tuple[float, float, float]] = []
    merged_excl_for_display: List[Tuple[float, float]] = []

    for plane in planes:
        plane_gaps = _gaps_at_plane(plane, safe_left, safe_right)
        if not plane_gaps:
            # No passable gap at this plane — total blockage
            return 0.0, AVOID_PURSUIT_Z, [], [], safe_left, safe_right, True

        all_gaps_for_display.extend(plane_gaps)

        # Build exclusions for display from this plane
        raw_excl = sorted((X - CONE_RADIUS_M, X + CONE_RADIUS_M) for X, _ in plane)
        merged: List[List[float]] = []
        for lo, hi in raw_excl:
            if merged and lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        merged_excl_for_display.extend((lo, hi) for lo, hi in merged)

        # Intersect current valid_intervals with plane_gaps
        gap_intervals = [(gl, gr) for gl, gr, _ in plane_gaps]
        new_valid: List[Tuple[float, float]] = []
        for vi_l, vi_r in valid_intervals:
            for gi_l, gi_r in gap_intervals:
                lo = max(vi_l, gi_l)
                hi = min(vi_r, gi_r)
                if hi - lo >= MIN_GAP_M:
                    new_valid.append((lo, hi))
        valid_intervals = new_valid
        if not valid_intervals:
            return 0.0, AVOID_PURSUIT_Z, all_gaps_for_display, merged_excl_for_display, \
                   safe_left, safe_right, True  # [GP-2]

    # ── Score remaining valid intervals ──────────────────────────────────────
    best_centre = 0.0
    best_score  = float("-inf")
    for gl, gr in valid_intervals:
        gc    = (gl + gr) / 2.0
        width = gr - gl
        # [GP-3] Normalised score
        score = (width / MIN_GAP_M) - GAP_CENTER_WEIGHT * abs(gc)
        if score > best_score:
            best_score  = score
            best_centre = gc

    best_centre = float(np.clip(best_centre, safe_left, safe_right))
    return best_centre, AVOID_PURSUIT_Z, all_gaps_for_display, \
           merged_excl_for_display, safe_left, safe_right, False


# ─────────────────────────────────────────────────────────────────────────────
# Pure Pursuit
# [PP-1] Avoidance uses AVOID_PURSUIT_Z as the fixed lookahead depth so the
#        arc geometry is consistent regardless of where the cone happens to be.
# ─────────────────────────────────────────────────────────────────────────────

def pure_pursuit(X_m: float, Z_m: float) -> float:
    """
    Standard pure-pursuit formula.
    For avoidance callers: always pass Z_m = AVOID_PURSUIT_Z so the arc is
    computed to a waypoint at a consistent forward distance, not at the cone.
    """
    ld = math.hypot(X_m, Z_m)
    if ld < 0.1:
        return 0.0
    raw = math.atan2(2.0 * WHEELBASE_M * X_m, ld * ld)
    return max(-STEER_MAX_DEG, min(STEER_MAX_DEG, math.degrees(raw)))


# ─────────────────────────────────────────────────────────────────────────────
# Lateral position estimator
# [SM-1] Used for release condition instead of _gap_target proxy.
# ─────────────────────────────────────────────────────────────────────────────

def estimate_lateral_error(
    lf: Optional[np.ndarray],
    rf: Optional[np.ndarray],
    road_mask: Optional[np.ndarray],
    H: int, W: int, fx: float,
    depth_arr: np.ndarray,
    dev_m_from_seg: float,
) -> float:
    """
    Best estimate of the car's lateral offset from lane centre (metres).
    Positive = car is to the right of centre.

    Priority:
      1. Segformer dev_m if Segformer is confident (road mask has pixels).
      2. Zero (car assumed centred) as safe fallback — avoidance release will
         wait for dwell timer anyway.
    """
    if road_mask is not None and road_mask.any() and dev_m_from_seg is not None:
        return float(dev_m_from_seg)
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Visualiser
# ─────────────────────────────────────────────────────────────────────────────

def draw_overlay(
    frame:        np.ndarray,
    detections:   List[Tuple],
    blocking:     List[Tuple[float, float]],
    road_mask:    Optional[np.ndarray],
    gap_target_X: float,
    left_X_m:     float,
    right_X_m:    float,
    steer_deg:    float,
    state:        str,
    H: int, W: int, fx: float,
    all_gaps:     List[Tuple[float, float, float]] = None,
    exclusions:   List[Tuple[float, float]]        = None,
    safe_l_m:     float = 0.0,
    safe_r_m:     float = 0.0,
    gap_z:        float = AVOID_PURSUIT_Z,
) -> np.ndarray:
    out = frame.copy()
    all_gaps   = all_gaps   or []
    exclusions = exclusions or []

    bot_y = H - 10
    top_y = int(H * 0.38)
    mid_x = W // 2

    def project(X_m: float, Z_m: float):
        Z_m = max(Z_m, 0.1)
        py  = int(bot_y - (min(Z_m, AVOIDANCE_TRIGGER_M) / AVOIDANCE_TRIGGER_M)
                  * (bot_y - top_y))
        px  = int(mid_x + X_m * fx / Z_m)
        return px, py

    def xm_to_px(xm: float, z: float = 1.8) -> int:
        return int(mid_x + xm * fx / z)

    # Road mask overlay
    if road_mask is not None and road_mask.any():
        seg_layer = np.zeros_like(out)
        seg_layer[road_mask] = [0, 220, 60]
        out = cv2.addWeighted(out, 0.65, seg_layer, 0.35, 0)
        mask_u8 = road_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 255, 80), 1)

    # YOLO bboxes
    for x1, y1, x2, y2, conf in detections:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), CLR_CONE, 2)
        cv2.putText(out, f"{conf:.2f}", (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, CLR_CONE, 1)

    # Cone circles at actual depth
    for X_c, Z_c in blocking:
        cx, cy = project(X_c, Z_c)
        r_cone = max(4, int(CONE_RADIUS_M * fx / Z_c))
        r_safe = max(6, int((CONE_RADIUS_M + GAP_SAFETY_MARGIN_M) * fx / Z_c))
        cv2.circle(out, (cx, cy), r_cone, CLR_CONE, -1)
        cv2.circle(out, (cx, cy), r_safe, (80, 80, 255), 1)
        cv2.putText(out, f"Z={Z_c:.1f}", (cx + r_safe + 2, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, CLR_CONE, 1)

    # Lane edges
    lx = xm_to_px(left_X_m)
    rx = xm_to_px(right_X_m)
    cv2.line(out, (lx, bot_y), (lx, top_y), CLR_LANE_L, 3)
    cv2.line(out, (rx, bot_y), (rx, top_y), CLR_LANE_R, 3)
    cv2.putText(out, f"L:{left_X_m:+.2f}m", (lx + 4, top_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, CLR_LANE_L, 1)
    cv2.putText(out, f"R:{right_X_m:+.2f}m", (rx + 4, top_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, CLR_LANE_R, 1)

    # Gap bar
    if state in ("AVOIDING", "NO_SAFE_GAP") and (all_gaps or exclusions):
        bar_t = top_y - 22
        bar_b = top_y - 8
        sl_px = xm_to_px(safe_l_m)
        sr_px = xm_to_px(safe_r_m)
        cv2.rectangle(out, (sl_px, bar_t), (sr_px, bar_b), (60, 60, 60), -1)
        for lo, hi in exclusions:
            cv2.rectangle(out, (xm_to_px(lo), bar_t), (xm_to_px(hi), bar_b),
                          (40, 40, 200), -1)
        for gl, gr, gc in all_gaps:
            is_best = abs(gc - gap_target_X) < 0.05
            col = CLR_GAP_SEL if is_best else CLR_GAP_CAND
            cv2.rectangle(out, (xm_to_px(gl), bar_t), (xm_to_px(gr), bar_b), col, -1)
            if is_best:
                cv2.circle(out, (xm_to_px(gc), (bar_t + bar_b) // 2), 4,
                           (255, 255, 255), -1)
        cv2.rectangle(out, (sl_px, bar_t), (sr_px, bar_b), (180, 180, 180), 1)

    # Planned path (bezier)
    if state == "AVOIDING":
        car_pt = np.array([mid_x, bot_y], dtype=float)
        gap_pt = np.array(project(gap_target_X, gap_z), dtype=float)
        ret_pt = np.array(project(0.0, gap_z * 2.5), dtype=float)
        pts = []
        for t in np.linspace(0.0, 1.0, 30):
            p = ((1 - t)**2 * car_pt
                 + 2 * (1 - t) * t * gap_pt
                 + t**2 * ret_pt)
            pts.append(p.astype(int))
        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(out, [pts], False, CLR_GAP_SEL, 3, cv2.LINE_AA)
        cv2.circle(out, tuple(gap_pt.astype(int)), 8, CLR_GAP_SEL, -1)
        cv2.putText(out, f"gap={gap_target_X:+.2f}m",
                    (int(gap_pt[0]) + 10, int(gap_pt[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_GAP_SEL, 1)

    if state == "LANE_FOLLOW":
        cv2.line(out, (mid_x, bot_y), (mid_x, top_y), CLR_PATH, 1)

    for i, (X_c, Z_c) in enumerate(blocking):
        cv2.putText(out, f"cone X={X_c:+.2f} Z={Z_c:.1f}m",
                    (10, 58 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_CONE, 1)

    # Steering direction
    if steer_deg > 2.0:
        direction, dir_color = "RIGHT >>", (0, 140, 255)
    elif steer_deg < -2.0:
        direction, dir_color = "<< LEFT",  (255, 140, 0)
    else:
        direction, dir_color = "STRAIGHT", (200, 200, 200)

    # Status bar
    if state == "NO_SAFE_GAP":
        color = CLR_STOP
        line1 = "NO SAFE GAP — STOPPING"
    elif state == "AVOIDING":
        color = CLR_WARN
        line1 = (f"AVOIDING  gap={gap_target_X:+.2f}m  "
                 f"steer={steer_deg:+.1f}deg")
    else:
        color = CLR_OK
        line1 = f"LANE FOLLOW  steer={steer_deg:+.1f}deg"

    cv2.rectangle(out, (0, 0), (W, 52), (0, 0, 0), -1)
    cv2.putText(out, line1, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
    cv2.putText(out, direction, (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.70,
                dir_color, 2)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main system
# ─────────────────────────────────────────────────────────────────────────────

class ConeAvoidanceSystem:
    """
    Self-contained avoidance + lane-following system.

    External integration:
        sys = ConeAvoidanceSystem()
        sys.init()
        steer_deg, state = sys.process_frame(frame, depth_arr, pose, fx, H, W)
        # state: "LANE_FOLLOW" | "AVOIDING" | "NO_SAFE_GAP" | "LOST"
    """

    def __init__(self):
        self.cam       = sl.Camera()
        self.image_mat = sl.Mat()
        self.depth_mat = sl.Mat()
        self.pose      = sl.Pose()

        self.seg_lane = SegformerLane()
        self.cone_det = ConeDetector()
        self.cone_map = ConeWorldMap()
        self._clahe   = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT,
                                        tileGridSize=CLAHE_TILE_SIZE)

        self._depth_cache: Optional[np.ndarray] = None
        self._depth_age:   int = DEPTH_REFRESH_EVERY

        self._steer_ema:  float = 0.0
        self._last_raw:   float = 0.0
        self._last_sent:  float = 0.0

        # State machine
        self._state:            str   = "LANE_FOLLOW"
        self._avoiding_frames:  int   = 0    # [SM-2] dwell counter
        self._gap_target:       float = 0.0
        self._last_gap_z:       float = AVOID_PURSUIT_Z
        self._last_gaps:        list  = []
        self._last_excl:        list  = []
        self._last_safe:        tuple = (0.0, 0.0)
        self._no_safe_gap:      bool  = False   # [GP-2]

        # [SEG-1] Single per-frame lane bounds cache (shared between control + display)
        self._lane_bounds: Tuple[float, float] = (-PATH_WIDTH_M, PATH_WIDTH_M)

        self._frame_cnt:  int = 0

        self.uart = UARTController()
        self.cal  = None
        self.H = self.W = None
        self._runtime = sl.RuntimeParameters()

    # ── Init ──────────────────────────────────────────────────────────────────

    def init(self) -> bool:
        log.info("[ConeAvoidance] Initialising camera...")
        sl.Camera.reboot(0)
        time.sleep(3)

        init_p = sl.InitParameters()
        init_p.camera_resolution = CAM_RES
        init_p.camera_fps        = CAM_FPS
        init_p.depth_mode        = CAM_DEPTH_MODE
        init_p.coordinate_units  = sl.UNIT.METER
        init_p.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

        if self.cam.open(init_p) != sl.ERROR_CODE.SUCCESS:
            log.error("[ConeAvoidance] Camera open failed")
            return False

        tp = sl.PositionalTrackingParameters()
        tp.set_floor_as_origin = True
        self.cam.enable_positional_tracking(tp)
        self._runtime.measure3D_reference_frame = sl.REFERENCE_FRAME.WORLD

        info     = self.cam.get_camera_information()
        self.cal = info.camera_configuration.calibration_parameters.left_cam
        self.W   = info.camera_configuration.resolution.width
        self.H   = info.camera_configuration.resolution.height
        log.info("[ConeAvoidance] Camera %dx%d @ %d fps", self.W, self.H, CAM_FPS)

        if not self.seg_lane.init():
            log.error("[ConeAvoidance] SegformerLane init failed")
            return False

        if not self.cone_det.init():
            log.warning("[ConeAvoidance] ConeDetector unavailable — lane-only mode")

        if UART_ENABLED:
            if not self.uart.open():
                log.error("[ConeAvoidance] UART open failed")
                return False
            self.uart.steer(127)
            self.uart.set_speed(0.0)

        return True

    # ── Standalone run loop ───────────────────────────────────────────────────

    def run(self) -> None:
        log.info("[ConeAvoidance] Running — press 'q' to quit.")
        last_hb = time.monotonic()

        try:
            while True:
                if self.cam.grab(self._runtime) != sl.ERROR_CODE.SUCCESS:
                    continue
                self._frame_cnt += 1

                self.cam.retrieve_image(self.image_mat, sl.VIEW.LEFT)
                # [CL-1] Keep raw frame separate — YOLO gets this.
                frame_raw = self.image_mat.get_data()[:, :, :3].copy()

                self._depth_age += 1
                if self._depth_cache is None or self._depth_age >= DEPTH_REFRESH_EVERY:
                    self.cam.retrieve_measure(self.depth_mat, sl.MEASURE.DEPTH)
                    self._depth_cache = self.depth_mat.get_data().squeeze().copy()
                    self._depth_age   = 0

                self.cam.get_position(self.pose, sl.REFERENCE_FRAME.WORLD)

                steer_deg, state = self.process_frame(
                    frame_raw, self._depth_cache, self.pose,
                    self.cal.fx, self.H, self.W,
                )

                if state == "NO_SAFE_GAP":
                    target_kmh = SPEED_STOP_KMH
                elif state == "AVOIDING":
                    target_kmh = SPEED_AVOID_KMH
                else:
                    target_kmh = SPEED_STRAIGHT_KMH

                if UART_ENABLED:
                    self.uart.set_speed(target_kmh)
                    if abs(steer_deg - self._last_sent) >= STEER_TX_DEADBAND_DEG:
                        self.uart.steer(_deg_to_byte(steer_deg))
                        self._last_sent = steer_deg
                    now = time.monotonic()
                    if now - last_hb >= UART_HEARTBEAT_S:
                        self.uart.steer(_deg_to_byte(steer_deg))
                        self.uart.set_speed(target_kmh)
                        last_hb = now

                # [SEG-1] Use cached lane bounds — do NOT call get_result() again.
                left_X, right_X = self._lane_bounds
                all_cones = self.cone_map.in_camera_frame(self.pose)
                blocking  = path_blocking_cones(all_cones, left_X, right_X)
                nearest_z = min((Z for _, Z in blocking), default=float("inf"))
                cone_info = (f"  cones={len(all_cones)}"
                             + (f" blocking={len(blocking)} nearest={nearest_z:.1f}m"
                                if blocking else " blocking=0"))

                if steer_deg > 2.0:
                    direction = "RIGHT"
                elif steer_deg < -2.0:
                    direction = "LEFT"
                else:
                    direction = "STRAIGHT"
                print(f"  [{state:11s}]  steer={steer_deg:+6.2f}deg  {direction}"
                      f"  gap_tgt={self._gap_target:+.2f}m"
                      f"  spd={target_kmh:.1f}km/h{cone_info}")

                if DISPLAY:
                    # [CL-1] Apply CLAHE only for display.
                    frame_display = self._apply_clahe(frame_raw)
                    dets = self.cone_det.get_result()
                    vis  = draw_overlay(
                        frame_display, dets, blocking, self._seg_road_mask,
                        self._gap_target, left_X, right_X,
                        steer_deg, state, self.H, self.W, self.cal.fx,
                        all_gaps=self._last_gaps,
                        exclusions=self._last_excl,
                        safe_l_m=self._last_safe[0],
                        safe_r_m=self._last_safe[1],
                        gap_z=self._last_gap_z,
                    )
                    cv2.imshow(WIN_NAME, vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            self.close()

    # ── Core frame processor ──────────────────────────────────────────────────

    def process_frame(
        self,
        frame:     np.ndarray,   # RAW frame (no CLAHE)
        depth_arr: np.ndarray,
        pose:      sl.Pose,
        fx:        float,
        H:         int,
        W:         int,
    ) -> Tuple[float, str]:
        """
        Returns (steer_deg, state).
        state: "LANE_FOLLOW" | "AVOIDING" | "NO_SAFE_GAP" | "LOST"
        """
        # [CL-1] CLAHE for Segformer only — not for YOLO.
        frame_seg = self._apply_clahe(frame)

        # ── Segformer ─────────────────────────────────────────────────────────
        self.seg_lane.submit(frame_seg, None, H, W, fx)
        lf, rf, road_mask, dev_m, wid_m, lc, rc, source = self.seg_lane.get_result()
        if road_mask is None:
            road_mask = np.zeros((H, W), dtype=bool)

        # Store for display (run loop reads this, not get_result() again) [SEG-1]
        self._seg_road_mask = road_mask

        # [SEG-1] Compute lane bounds ONCE per frame and cache.
        left_X, right_X = lane_bounds_m(road_mask, depth_arr, H, W, fx)
        self._lane_bounds = (left_X, right_X)

        # ── Cone detection + world map ────────────────────────────────────────
        # [CL-1] Submit raw frame to YOLO — no CLAHE.
        if self._frame_cnt % YOLO_SKIP_FRAMES == 0:
            self.cone_det.submit(frame)
        detections      = self.cone_det.get_result()
        cones_cam_fresh = localise_cones(detections, depth_arr, H, W, fx)
        if cones_cam_fresh:
            self.cone_map.update(cones_cam_fresh, pose)
        cones_cam = self.cone_map.in_camera_frame(pose)

        # ── Path-blocking cones — uses cached lane bounds ─────────────────────
        blocking = path_blocking_cones(cones_cam, left_X, right_X)

        if blocking:
            closest_z = min(Z for _, Z in blocking)
        else:
            closest_z = float("inf")

        # ── [SM-1] Lateral error estimate ─────────────────────────────────────
        lat_error = estimate_lateral_error(
            lf, rf, road_mask, H, W, fx, depth_arr, dev_m)

        # ── State machine ─────────────────────────────────────────────────────
        # LANE_FOLLOW → AVOIDING
        if self._state == "LANE_FOLLOW" and closest_z < AVOIDANCE_TRIGGER_M:
            self._state           = "AVOIDING"
            self._avoiding_frames = 0
            log.info("[ConeAvoidance] Cone at %.2f m — engaging avoidance", closest_z)

        # Track time in avoidance [SM-2]
        if self._state == "AVOIDING":
            self._avoiding_frames += 1

        # NO_SAFE_GAP → AVOIDING re-check (planner may find a gap next frame)
        if self._state == "NO_SAFE_GAP" and closest_z >= AVOIDANCE_RELEASE_M:
            self._state = "LANE_FOLLOW"
            log.info("[ConeAvoidance] Path finally clear from NO_SAFE_GAP — resuming")

        # AVOIDING → LANE_FOLLOW
        if self._state == "AVOIDING":
            dwell_ok   = self._avoiding_frames >= AVOIDING_MIN_DWELL_FRAMES  # [SM-2]
            path_clear = closest_z > AVOIDANCE_RELEASE_M
            # [SM-1] Use actual lateral pose error, not _gap_target proxy
            centred    = abs(lat_error) < LATERAL_RETURN_BAND_M
            if dwell_ok and path_clear and centred:
                self._state = "LANE_FOLLOW"
                log.info(
                    "[ConeAvoidance] Dwell=%d frames, path clear, lat_err=%.2f m — "
                    "resuming lane follow", self._avoiding_frames, lat_error)
                self._avoiding_frames = 0

        # ── Steering decision ─────────────────────────────────────────────────
        if self._state in ("AVOIDING", "NO_SAFE_GAP"):
            gap_X, gap_Z, all_gaps, excl, safe_l, safe_r, no_safe = find_best_gap(
                blocking, left_X, right_X)

            self._last_gaps = all_gaps
            self._last_excl = excl
            self._last_safe = (safe_l, safe_r)
            self._last_gap_z = gap_Z

            if no_safe:
                # [GP-2] Total blockage — stop and wait.
                self._state     = "NO_SAFE_GAP"
                self._gap_target = 0.0
                raw_deg = 0.0
                state   = "NO_SAFE_GAP"
                log.warning("[ConeAvoidance] No safe gap — stopping.")
            else:
                self._state      = "AVOIDING"
                self._gap_target = gap_X
                # [PP-1] Always steer to waypoint at AVOID_PURSUIT_Z, not cone Z.
                raw_deg = (0.0 if abs(gap_X) < GAP_DEADBAND_M
                           else pure_pursuit(gap_X, AVOID_PURSUIT_Z))
                state   = "AVOIDING"

        elif source == "LOST":
            raw_deg = 0.0
            state   = "LOST"

        else:
            # Normal lane follow
            lookahead_pt, _ = compute_lookahead(lf, rf, H, W, fx, depth_arr)
            if lookahead_pt is not None:
                X_m, Z_m = lookahead_pt
                raw_deg  = (0.0 if abs(X_m) < CTRL_LANE_DEADBAND_M
                            else pure_pursuit(X_m, Z_m))
            else:
                raw_deg = (0.0 if abs(dev_m) < CTRL_LANE_DEADBAND_M
                           else pure_pursuit(dev_m, CTRL_LOOKAHEAD_M))
            self._gap_target = 0.0
            state = "LANE_FOLLOW"

        # ── Rate limiter + EMA ────────────────────────────────────────────────
        if self._state == "AVOIDING":
            rate_lim = 10.0
            alpha    = 0.50
        else:
            rate_lim = STEER_RATE_LIMIT_DEG
            alpha    = STEER_EMA_ALPHA

        delta = raw_deg - self._last_raw
        if abs(delta) > rate_lim:
            raw_deg = self._last_raw + math.copysign(rate_lim, delta)
        self._last_raw  = raw_deg
        self._steer_ema = alpha * raw_deg + (1.0 - alpha) * self._steer_ema

        return self._steer_ema, state

    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def close(self):
        if UART_ENABLED:
            self.uart.set_speed(0.0)
            self.uart.steer(127)
            self.uart.close()
        self.cam.disable_positional_tracking()
        self.cam.close()
        cv2.destroyAllWindows()
        log.info("[ConeAvoidance] Closed.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    system = ConeAvoidanceSystem()
    if not system.init():
        log.error("Init failed — check camera and model path.")
        raise SystemExit(1)
    system.run()
