"""
PSU Eco Racing — Shell Eco-Marathon Autonomous Division
obstacle_avoidance.py  |  Standalone cone avoidance — fully self-contained test.

Runs everything needed for safe autonomous driving + cone avoidance:

  Normal mode (LANE_FOLLOW):
    Segformer centerline → compute_lookahead → pure pursuit → steer.
    Identical logic to the main pipeline.

  Avoidance mode (AVOIDING):
    Builds a list of passable gaps from the lane structure:
      gaps = all intervals [left_edge, right_edge] in camera X (metres)
             that are (a) not occupied by a cone and (b) wide enough for the car.
    The best gap is the one with the highest score (width + centering preference).
    Pure pursuit steers to the gap centre.

    This naturally handles every arrangement:
      – Single cone dead centre  → picks whichever side has more road space.
      – 2 cones forming a slot   → threads through the slot.
      – Cone far to the side     → ignored (not blocking any gap the car needs).

  Trigger:
    Avoidance only engages when a cone is within the FORWARD PATH CORRIDOR
    (|X_cone| < PATH_WIDTH_M  AND  Z_cone < AVOIDANCE_TRIGGER_M).
    A cone 2 m to the side will never trigger avoidance.

  Transitions:
    LANE_FOLLOW → AVOIDING  when a path-blocking cone enters AVOIDANCE_TRIGGER_M
    AVOIDING    → LANE_FOLLOW  when all path-blocking cones exit AVOIDANCE_RELEASE_M
                               AND the selected gap centre is within RETURN_BAND_M
                               of the lane centre (prevents premature release).

Threading (nothing blocks the camera loop):
  Main thread    — grab, CLAHE, depth, odometry, steering, UART, display
  SegformerLane  — background thread inside imported class
  ConeDetector   — background thread defined here

Run standalone:
    python obstacle_avoidance.py

Integrate into main.py later (swap YOLOv5 → YOLOv8 at that point):
    sys = ConeAvoidanceSystem()
    sys.init()
    steer_deg, state = sys.process_frame(frame, depth_arr, pose, fx, H, W)
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
SPEED_AVOID_KMH    = 2.0          # slow down while manoeuvring

# Lane-following pure pursuit  (matches main pipeline)
WHEELBASE_M           = 1.6       # axle-to-axle — verify physically
CTRL_LOOKAHEAD_M      = 2.2       # fallback assumed depth (m) at lookahead row
CTRL_LANE_DEADBAND_M  = 0.15      # ignore offsets < ±15 cm in LANE_FOLLOW mode
STEER_MAX_DEG         = 25.0
STEER_RATE_LIMIT_DEG  = 5.0       # max change per frame
STEER_EMA_ALPHA       = 0.20
STEER_TX_DEADBAND_DEG = 1.5       # only transmit if angle changed by this much

# Cone detector (YOLOv5 custom model via torch.hub)
YOLO_MODEL_PATH  = "best (cones).pt"
YOLO_CONF_THRESH = 0.25   # lowered — raise if you get false positives
YOLO_IMG_SIZE    = 416
YOLO_SKIP_FRAMES = 2      # run every N frames

# Cone 3D localisation
CONE_DEPTH_PAD = 6                # patch half-size (px) for ZED depth median
CONE_Z_MIN_M          = 0.3
CONE_Z_MAX_M          = 12.0   # must be > AVOIDANCE_TRIGGER_M so cones are tracked before they trigger
CONE_Z_BLOCKING_MIN_M = 0.8    # ignore cones already this close — car is passing them

# World-frame cone map
CONE_MERGE_RADIUS_M = 0.50
CONE_MAX_AGE_S      = 4.0

# Avoidance trigger
# A cone must be BOTH close in Z AND inside the path corridor in X.
# Cones to the side (|X| > PATH_WIDTH_M) are not a threat and are ignored.
AVOIDANCE_TRIGGER_M  = 1.5        # max forward distance to trigger
AVOIDANCE_RELEASE_M  = 2.5        # hysteresis: release when clear beyond this
PATH_WIDTH_M         = 1.5        # fallback half-corridor when road mask unavailable
RETURN_BAND_M        = 0.20       # release only when gap target < this from centre

# Gap planner
CAR_WIDTH_M          = 0.90       # full vehicle width — measure physically
GAP_SAFETY_MARGIN_M  = 0.20       # extra clearance each side (total gap needed = CAR_WIDTH + 2×margin)
MIN_GAP_M            = CAR_WIDTH_M + 2 * GAP_SAFETY_MARGIN_M   # minimum passable gap
CONE_RADIUS_M        = 0.15       # treat each cone as a cylinder of this radius
GAP_LOOKAHEAD_M      = 1.7        # fallback waypoint Z when no cone depth available
GAP_CENTER_WEIGHT    = 0.40       # preference for gaps closer to lane centre (lower = prefer centre)
GAP_DEADBAND_M       = 0.05       # ignore gap offsets < this — matches lane-keeping CTRL_LANE_DEADBAND_M

# Lane edge margin (how far from the edge any gap target must be)
LANE_MARGIN_M = 0.25

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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _patch_depth(depth_arr: np.ndarray, y: int, x: int,
                 H: int, W: int, pad: int = 5) -> float:
    r0, r1 = max(0, y - pad), min(H, y + pad + 1)
    c0, c1 = max(0, x - pad), min(W, x + pad + 1)
    patch  = depth_arr[r0:r1, c0:c1]
    valid  = patch[np.isfinite(patch) & (patch > 0.1) & (patch < 30.0)]
    return float(np.median(valid)) if valid.size >= 3 else CTRL_LOOKAHEAD_M


def _deg_to_byte(deg: float) -> int:
    # 0=full-left(-25°)  127=straight(0°)  255=full-right(+25°)  — matches commander.py
    return int(max(0, min(255, round(127.0 + deg * 127.0 / STEER_MAX_DEG))))


# ─────────────────────────────────────────────────────────────────────────────
# Cone detector — YOLOv5 via torch.hub, background thread
# ─────────────────────────────────────────────────────────────────────────────

class ConeDetector:
    def __init__(self):
        self._model:  object        = None
        self._half:   bool          = False
        self._in_q:   queue.Queue   = queue.Queue(maxsize=1)
        self._out_q:  queue.Queue   = queue.Queue(maxsize=1)
        self._result: List[Tuple]   = []

    def init(self) -> bool:
        import os
        if not os.path.isfile(YOLO_MODEL_PATH):
            log.error("[ConeDetector] Model not found: %s", YOLO_MODEL_PATH)
            return False
        try:
            self._model      = torch.hub.load(
                "ultralytics/yolov5", "custom",
                path=YOLO_MODEL_PATH,
                force_reload=False,
                verbose=False,
            )
            self._model.conf = YOLO_CONF_THRESH
            self._model.iou  = 0.45
            # Move to CUDA + FP16 if available — ~5× faster on Jetson
            if torch.cuda.is_available():
                self._model = self._model.cuda().half()
                self._half  = True
                log.info("[ConeDetector] Loaded %s — CUDA FP16", YOLO_MODEL_PATH)
            else:
                log.info("[ConeDetector] Loaded %s — CPU", YOLO_MODEL_PATH)
            threading.Thread(target=self._worker, daemon=True,
                             name="ConeDetector").start()
            return True
        except Exception as exc:
            log.error("[ConeDetector] Load failed: %s", exc)
            return False

    def submit(self, frame: np.ndarray) -> None:
        try:
            self._in_q.put_nowait(frame)
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
                dets    = []
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
# ─────────────────────────────────────────────────────────────────────────────

class ConeWorldMap:
    def __init__(self):
        self._cones: List[List] = []  # [X_w, Z_w, timestamp]

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
                self._cones.append([X_w, Z_w, now])
        self._cones = [e for e in self._cones if now - e[2] < CONE_MAX_AGE_S]

    def in_camera_frame(self, pose: sl.Pose) -> List[Tuple[float, float]]:
        tx, tz, R2 = self._pose_xz(pose)
        R2_inv = R2.T
        out = []
        for e in self._cones:
            cam  = R2_inv @ (np.array([e[0], e[1]]) - np.array([tx, tz]))
            X_c, Z_c = float(cam[0]), float(cam[1])
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
# ─────────────────────────────────────────────────────────────────────────────

def localise_cones(
    detections: List[Tuple],
    depth_arr:  np.ndarray,
    H: int, W: int, fx: float,
) -> List[Tuple[float, float]]:
    out = []
    for x1, y1, x2, y2, _ in detections:
        cx  = (x1 + x2) / 2.0
        cy  = min(y2, H - 1)
        xi  = int(np.clip(cx, 0, W - 1))
        yi  = int(np.clip(cy, 0, H - 1))
        Z_m = _patch_depth(depth_arr, yi, xi, H, W, pad=CONE_DEPTH_PAD)
        if not (CONE_Z_MIN_M < Z_m < CONE_Z_MAX_M):
            continue
        X_m = (cx - W / 2.0) * Z_m / fx
        out.append((X_m, Z_m))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Lane boundary extraction from Segformer road_mask
# ─────────────────────────────────────────────────────────────────────────────

def lane_bounds_m(
    road_mask: Optional[np.ndarray],
    depth_arr: np.ndarray,
    H: int, W: int, fx: float,
) -> Tuple[float, float]:
    """
    Left/right drivable edge in camera X (metres).
    Samples a band of rows between SEG_FAR_FRAC and SEG_NEAR_FRAC and takes
    the median left/right column so a single occluded row doesn't corrupt bounds.
    Falls back to ±1.2 m when mask is unavailable or too sparse.
    """
    if road_mask is None or not road_mask.any():
        return -1.2, 1.2

    y_near   = int(H * SEG_NEAR_FRAC)
    y_far    = int(H * SEG_FAR_FRAC)
    safe_top = int(H * ROI_TOP_FRACTION)

    left_cols: List[int]  = []
    right_cols: List[int] = []
    step = max(1, (y_near - y_far) // 6)
    for y in range(y_far, y_near + 1, step):
        row = road_mask[y].copy()
        row[:safe_top] = False
        cols = np.where(row)[0]
        if len(cols) >= 4:
            left_cols.append(int(cols[0]))
            right_cols.append(int(cols[-1]))

    if not left_cols:
        return -1.2, 1.2

    left_col  = int(np.median(left_cols))
    right_col = int(np.median(right_cols))
    mid       = (left_col + right_col) // 2
    Z_near    = _patch_depth(depth_arr, y_near, mid, H, W)
    left_X    = (float(left_col)  - W / 2.0) * Z_near / fx
    right_X   = (float(right_col) - W / 2.0) * Z_near / fx
    log.debug("[lane_bounds] rows sampled=%d  left=%.2f m  right=%.2f m",
              len(left_cols), left_X, right_X)
    return left_X, right_X


# ─────────────────────────────────────────────────────────────────────────────
# Path-blocking check
# ─────────────────────────────────────────────────────────────────────────────

def path_blocking_cones(
    cones_cam: List[Tuple[float, float]],
    left_X_m:  float = -PATH_WIDTH_M,
    right_X_m: float =  PATH_WIDTH_M,
) -> List[Tuple[float, float]]:
    """
    Return only cones within the actual drivable road area and avoidance range.
    Uses real segformer road edges when available; falls back to ±PATH_WIDTH_M.
    """
    return [
        (X, Z) for (X, Z) in cones_cam
        if CONE_Z_BLOCKING_MIN_M < Z < AVOIDANCE_TRIGGER_M and left_X_m <= X <= right_X_m
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Gap planner
# ─────────────────────────────────────────────────────────────────────────────

def find_best_gap(
    blocking_cones: List[Tuple[float, float]],
    left_X_m: float,
    right_X_m: float,
) -> Tuple[float, float]:
    """
    Build a list of passable gaps from the lane structure and cone positions,
    then return the centre of the best gap as (X_target_m, GAP_LOOKAHEAD_M).

    Gap construction
    ────────────────
    Treat each cone as blocking the interval [X_cone - CONE_RADIUS_M,
                                               X_cone + CONE_RADIUS_M].
    The passable segments of [left_X_m + margin, right_X_m - margin]
    are the intervals NOT covered by any cone exclusion zone.

    Merge overlapping exclusion zones first so adjacent cones are treated
    as a single combined obstacle.

    Gap scoring  (higher = better)
    ─────────────────────────────
    score = gap_width - GAP_CENTER_WEIGHT × |gap_centre|

    Prefer wider gaps; tie-break toward lane centre (X=0).

    Handles:
      1 cone at centre      → two gaps, picks the wider side.
      2 cones forming slot  → threads through the slot gap.
      2 cones side-by-side  → merged into one obstacle, goes around.
      No cones (fallback)   → returns (0.0, GAP_LOOKAHEAD_M) — stay centre.
    """
    safe_left  = left_X_m  + LANE_MARGIN_M
    safe_right = right_X_m - LANE_MARGIN_M

    if not blocking_cones or safe_right - safe_left < MIN_GAP_M:
        return 0.0, GAP_LOOKAHEAD_M, [], [], safe_left, safe_right

    # Build exclusion intervals and merge overlaps
    raw_excl = sorted(
        (X - CONE_RADIUS_M, X + CONE_RADIUS_M)
        for X, _ in blocking_cones
    )
    merged: List[Tuple[float, float]] = []
    for lo, hi in raw_excl:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append([lo, hi])

    # Derive passable gaps: intervals of [safe_left, safe_right] not blocked
    gaps: List[Tuple[float, float, float]] = []  # (left, right, centre)
    cursor = safe_left
    for lo, hi in merged:
        gap_r = lo
        if gap_r - cursor >= MIN_GAP_M:
            centre = (cursor + gap_r) / 2.0
            gaps.append((cursor, gap_r, centre))
        cursor = max(cursor, hi)
    if safe_right - cursor >= MIN_GAP_M:
        centre = (cursor + safe_right) / 2.0
        gaps.append((cursor, safe_right, centre))

    if not gaps:
        cursor = safe_left
        all_gaps_fb = []
        for lo, hi in merged:
            if lo > cursor:
                all_gaps_fb.append((cursor, lo, (cursor + lo) / 2.0))
            cursor = max(cursor, hi)
        if safe_right > cursor:
            all_gaps_fb.append((cursor, safe_right, (cursor + safe_right) / 2.0))
        gaps = all_gaps_fb if all_gaps_fb else [(safe_left, safe_right, 0.0)]

    # Score gaps: wider = better; prefer centre
    best_centre = 0.0
    best_score  = float("-inf")
    for gl, gr, gc in gaps:
        width = gr - gl
        score = width - GAP_CENTER_WEIGHT * abs(gc)
        if score > best_score:
            best_score  = score
            best_centre = gc

    best_centre = float(np.clip(best_centre, safe_left, safe_right))
    return best_centre, GAP_LOOKAHEAD_M, gaps, merged, safe_left, safe_right


# ─────────────────────────────────────────────────────────────────────────────
# Pure Pursuit
# ─────────────────────────────────────────────────────────────────────────────

def pure_pursuit(X_m: float, Z_m: float) -> float:
    ld = math.hypot(X_m, Z_m)
    if ld < 0.1:
        return 0.0
    raw = math.atan2(2.0 * WHEELBASE_M * X_m, ld * ld)
    return max(-STEER_MAX_DEG, min(STEER_MAX_DEG, math.degrees(raw)))


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
    gap_z:        float = GAP_LOOKAHEAD_M,
) -> np.ndarray:
    out = frame.copy()
    all_gaps   = all_gaps   or []
    exclusions = exclusions or []

    bot_y = H - 10
    top_y = int(H * 0.38)
    mid_x = W // 2

    def project(X_m: float, Z_m: float):
        """Map a world point to image pixel using actual depth."""
        Z_m = max(Z_m, 0.1)
        py  = int(bot_y - (min(Z_m, AVOIDANCE_TRIGGER_M) / AVOIDANCE_TRIGGER_M) * (bot_y - top_y))
        px  = int(mid_x + X_m * fx / Z_m)
        return px, py

    def xm_to_px(xm: float, z: float = 1.8) -> int:
        return int(mid_x + xm * fx / z)

    # ── Segformer road mask — bright green semi-transparent overlay ───────────
    if road_mask is not None and road_mask.any():
        seg_layer = np.zeros_like(out)
        seg_layer[road_mask] = [0, 220, 60]
        out = cv2.addWeighted(out, 0.65, seg_layer, 0.35, 0)
        mask_u8 = road_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 255, 80), 1)

    # ── YOLO bboxes ───────────────────────────────────────────────────────────
    for x1, y1, x2, y2, conf in detections:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), CLR_CONE, 2)
        cv2.putText(out, f"{conf:.2f}", (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, CLR_CONE, 1)

    # ── Cone circles projected at actual depth ────────────────────────────────
    for X_c, Z_c in blocking:
        cx, cy = project(X_c, Z_c)
        r_cone = max(4, int(CONE_RADIUS_M * fx / Z_c))
        r_safe = max(6, int((CONE_RADIUS_M + GAP_SAFETY_MARGIN_M) * fx / Z_c))
        cv2.circle(out, (cx, cy), r_cone, CLR_CONE, -1)
        cv2.circle(out, (cx, cy), r_safe, (80, 80, 255), 1)
        cv2.putText(out, f"Z={Z_c:.1f}", (cx + r_safe + 2, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, CLR_CONE, 1)

    # ── Lane edges ────────────────────────────────────────────────────────────
    lx = xm_to_px(left_X_m)
    rx = xm_to_px(right_X_m)
    cv2.line(out, (lx, bot_y), (lx, top_y), CLR_LANE_L, 3)
    cv2.line(out, (rx, bot_y), (rx, top_y), CLR_LANE_R, 3)
    cv2.putText(out, f"L:{left_X_m:+.2f}m",  (lx + 4, top_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, CLR_LANE_L, 1)
    cv2.putText(out, f"R:{right_X_m:+.2f}m", (rx + 4, top_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, CLR_LANE_R, 1)

    # ── Gap decision bar — horizontal strip above top_y ───────────────────────
    if state == "AVOIDING" and (all_gaps or exclusions):
        bar_t  = top_y - 22
        bar_b  = top_y - 8
        # Road background
        sl_px = xm_to_px(safe_l_m)
        sr_px = xm_to_px(safe_r_m)
        cv2.rectangle(out, (sl_px, bar_t), (sr_px, bar_b), (60, 60, 60), -1)
        # Exclusion zones in red
        for lo, hi in exclusions:
            cv2.rectangle(out, (xm_to_px(lo), bar_t), (xm_to_px(hi), bar_b), (40, 40, 200), -1)
        # Candidate gaps
        for gl, gr, gc in all_gaps:
            is_best = abs(gc - gap_target_X) < 0.05
            col = CLR_GAP_SEL if is_best else CLR_GAP_CAND
            cv2.rectangle(out, (xm_to_px(gl), bar_t), (xm_to_px(gr), bar_b), col, -1)
            if is_best:
                cv2.circle(out, (xm_to_px(gc), (bar_t + bar_b) // 2), 4, (255, 255, 255), -1)
        cv2.rectangle(out, (sl_px, bar_t), (sr_px, bar_b), (180, 180, 180), 1)

    # ── Planned path (bezier: car → gap waypoint → return to centre) ──────────
    if state == "AVOIDING":
        car_pt = np.array([mid_x, bot_y], dtype=float)
        gap_pt = np.array(project(gap_target_X, gap_z), dtype=float)
        ret_pt = np.array(project(0.0, gap_z * 2.5), dtype=float)

        pts = []
        for t in np.linspace(0.0, 1.0, 30):
            p = (1 - t)**2 * car_pt + 2 * (1 - t) * t * gap_pt + t**2 * ret_pt
            pts.append(p.astype(int))
        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(out, [pts], False, CLR_GAP_SEL, 3, cv2.LINE_AA)
        cv2.circle(out, tuple(gap_pt.astype(int)), 8, CLR_GAP_SEL, -1)
        cv2.circle(out, tuple(ret_pt.astype(int)), 6, (200, 255, 200), 2)
        cv2.putText(out, f"gap={gap_target_X:+.2f}m",
                    (int(gap_pt[0]) + 10, int(gap_pt[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_GAP_SEL, 1)

    # ── Centre path (lane following) ──────────────────────────────────────────
    if state == "LANE_FOLLOW":
        cv2.line(out, (mid_x, bot_y), (mid_x, top_y), CLR_PATH, 1)

    # ── Blocking cone text list ───────────────────────────────────────────────
    for i, (X_c, Z_c) in enumerate(blocking):
        cv2.putText(out, f"cone X={X_c:+.2f} Z={Z_c:.1f}m",
                    (10, 58 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_CONE, 1)

    # ── Steering direction ────────────────────────────────────────────────────
    if steer_deg > 2.0:
        direction, dir_color = "RIGHT >>", (0, 140, 255)
    elif steer_deg < -2.0:
        direction, dir_color = "<< LEFT",  (255, 140, 0)
    else:
        direction, dir_color = "STRAIGHT", (200, 200, 200)

    # ── Status bar ────────────────────────────────────────────────────────────
    color = CLR_WARN if state == "AVOIDING" else CLR_OK
    line1 = (f"AVOIDING  gap={gap_target_X:+.2f}m  steer={steer_deg:+.1f}deg"
             if state == "AVOIDING"
             else f"LANE FOLLOW  steer={steer_deg:+.1f}deg")
    cv2.rectangle(out, (0, 0), (W, 52), (0, 0, 0), -1)
    cv2.putText(out, line1, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
    cv2.putText(out, direction, (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.70, dir_color, 2)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main system
# ─────────────────────────────────────────────────────────────────────────────

class ConeAvoidanceSystem:
    """
    Self-contained avoidance + lane-following system.

    External integration (from main.py):
        sys = ConeAvoidanceSystem()
        sys.init()
        steer_deg, state = sys.process_frame(frame, depth_arr, pose, fx, H, W)
        # state: "LANE_FOLLOW" | "AVOIDING" | "LOST"
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

        self._state:      str   = "LANE_FOLLOW"
        self._gap_target: float = 0.0
        self._last_gap_z: float = GAP_LOOKAHEAD_M
        self._last_gaps:  list  = []
        self._last_excl:  list  = []
        self._last_safe:  tuple = (0.0, 0.0)
        self._frame_cnt:  int   = 0

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

        log.info("[ConeAvoidance] Camera  %dx%d @ %d fps", self.W, self.H, CAM_FPS)

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
                frame = self.image_mat.get_data()[:, :, :3].copy()

                self._depth_age += 1
                if self._depth_cache is None or self._depth_age >= DEPTH_REFRESH_EVERY:
                    self.cam.retrieve_measure(self.depth_mat, sl.MEASURE.DEPTH)
                    self._depth_cache = self.depth_mat.get_data().squeeze().copy()
                    self._depth_age   = 0

                self.cam.get_position(self.pose, sl.REFERENCE_FRAME.WORLD)

                steer_deg, state = self.process_frame(
                    frame, self._depth_cache, self.pose,
                    self.cal.fx, self.H, self.W,
                )

                target_kmh = SPEED_AVOID_KMH if state == "AVOIDING" else SPEED_STRAIGHT_KMH

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

                # Retrieve latest segformer result for display and cone filtering
                lf, rf, road_mask, *_ = self.seg_lane.get_result()
                left_X, right_X = lane_bounds_m(
                    road_mask, self._depth_cache, self.H, self.W, self.cal.fx)

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
                    dets = self.cone_det.get_result()
                    vis  = draw_overlay(
                        frame, dets, blocking, road_mask,
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

    # ── Core frame processor  (also callable from main.py) ────────────────────

    def process_frame(
        self,
        frame:     np.ndarray,
        depth_arr: np.ndarray,
        pose:      sl.Pose,
        fx:        float,
        H:         int,
        W:         int,
    ) -> Tuple[float, str]:
        """
        Returns (steer_deg, state).
        state: "LANE_FOLLOW" | "AVOIDING" | "LOST"
        """
        frame_norm = self._apply_clahe(frame)

        # ── Segformer ─────────────────────────────────────────────────────────
        self.seg_lane.submit(frame_norm, None, H, W, fx)
        lf, rf, road_mask, dev_m, wid_m, lc, rc, source = self.seg_lane.get_result()
        if road_mask is None:
            log.debug("[SegformerLane] road_mask is None — using zero mask")
            road_mask = np.zeros((H, W), dtype=bool)
        else:
            log.debug("[SegformerLane] road_mask active, road pixels=%d", road_mask.sum())

        # ── Cone detection + world map ────────────────────────────────────────
        if self._frame_cnt % YOLO_SKIP_FRAMES == 0:
            self.cone_det.submit(frame_norm)
        detections      = self.cone_det.get_result()
        cones_cam_fresh = localise_cones(detections, depth_arr, H, W, fx)
        if cones_cam_fresh:
            self.cone_map.update(cones_cam_fresh, pose)
        cones_cam = self.cone_map.in_camera_frame(pose)

        # ── Lane boundaries ───────────────────────────────────────────────────
        left_X, right_X = lane_bounds_m(road_mask, depth_arr, H, W, fx)

        # ── Path-blocking cones only — use actual road edges from segformer ────
        blocking = path_blocking_cones(cones_cam, left_X, right_X)

        # ── State machine ─────────────────────────────────────────────────────
        if blocking:
            closest_z = min(Z for _, Z in blocking)
        else:
            closest_z = float("inf")

        if self._state == "LANE_FOLLOW" and closest_z < AVOIDANCE_TRIGGER_M:
            self._state = "AVOIDING"
            log.info("[ConeAvoidance] Cone at %.2f m — engaging avoidance", closest_z)

        if self._state == "AVOIDING":
            all_clear     = closest_z > AVOIDANCE_RELEASE_M
            back_to_centre = abs(self._gap_target) < RETURN_BAND_M
            if all_clear and back_to_centre:
                self._state = "LANE_FOLLOW"
                log.info("[ConeAvoidance] Path clear — resuming lane follow")

        # ── Steering decision ─────────────────────────────────────────────────
        if self._state == "AVOIDING":
            gap_X, _, all_gaps, excl, safe_l, safe_r = find_best_gap(blocking, left_X, right_X)
            # Use actual cone Z for pure pursuit — steer so you arrive at gap_X when you reach the cone
            gap_Z            = closest_z if closest_z < AVOIDANCE_TRIGGER_M else GAP_LOOKAHEAD_M
            self._gap_target = gap_X
            self._last_gap_z = gap_Z
            self._last_gaps  = all_gaps
            self._last_excl  = excl
            self._last_safe  = (safe_l, safe_r)
            raw_deg = 0.0 if abs(gap_X) < GAP_DEADBAND_M else pure_pursuit(gap_X, gap_Z)
            state   = "AVOIDING"

        elif source == "LOST":
            raw_deg = 0.0
            state   = "LOST"

        else:
            # Normal lane follow — pure pursuit on Segformer lookahead
            lookahead_pt, _ = compute_lookahead(lf, rf, H, W, fx, depth_arr)
            if lookahead_pt is not None:
                X_m, Z_m = lookahead_pt
                raw_deg  = 0.0 if abs(X_m) < CTRL_LANE_DEADBAND_M else pure_pursuit(X_m, Z_m)
            else:
                raw_deg = 0.0 if abs(dev_m) < CTRL_LANE_DEADBAND_M else pure_pursuit(dev_m, CTRL_LOOKAHEAD_M)
            self._gap_target = 0.0
            state = "LANE_FOLLOW"

        # ── Rate limiter + EMA ────────────────────────────────────────────────
        # Avoiding: fast rate-limit (10 deg/frame) + responsive EMA (0.50 alpha)
        # matches commander.py curve mode — consistent incremental corrections.
        # Lane follow: conservative rate-limit + slow EMA for stability.
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
