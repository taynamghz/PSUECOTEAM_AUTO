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

# YOLOv5
YOLO_MODEL_PATH  = "best (cones).pt"
YOLO_CONF_THRESH = 0.40
YOLO_IMG_SIZE    = 416
YOLO_SKIP_FRAMES = 2              # run every N frames

# Cone 3D localisation
CONE_DEPTH_PAD = 6                # patch half-size (px) for ZED depth median
CONE_Z_MIN_M   = 0.3
CONE_Z_MAX_M   = 8.0

# World-frame cone map
CONE_MERGE_RADIUS_M = 0.50
CONE_MAX_AGE_S      = 4.0

# Avoidance trigger
# A cone must be BOTH close in Z AND inside the path corridor in X.
# Cones to the side (|X| > PATH_WIDTH_M) are not a threat and are ignored.
AVOIDANCE_TRIGGER_M  = 5.0        # max forward distance to trigger
AVOIDANCE_RELEASE_M  = 6.5        # hysteresis: release when clear beyond this
PATH_WIDTH_M         = 1.0        # half-corridor width that counts as "in the path"
RETURN_BAND_M        = 0.20       # release only when gap target < this from centre

# Gap planner
CAR_WIDTH_M          = 0.90       # full vehicle width — measure physically
GAP_SAFETY_MARGIN_M  = 0.20       # extra clearance each side (total gap needed = CAR_WIDTH + 2×margin)
MIN_GAP_M            = CAR_WIDTH_M + 2 * GAP_SAFETY_MARGIN_M   # minimum passable gap
CONE_RADIUS_M        = 0.15       # treat each cone as a cylinder of this radius
GAP_LOOKAHEAD_M      = 3.0        # distance at which the gap centre waypoint is placed
GAP_CENTER_WEIGHT    = 0.40       # preference for gaps closer to lane centre (lower = prefer centre)

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
    return int(max(0, min(255, round(127.0 - deg * 127.0 / STEER_MAX_DEG))))


# ─────────────────────────────────────────────────────────────────────────────
# YOLOv5 cone detector  (background thread)
# ─────────────────────────────────────────────────────────────────────────────

class ConeDetector:
    def __init__(self):
        self._model = None
        self._in_q:  queue.Queue = queue.Queue(maxsize=1)
        self._out_q: queue.Queue = queue.Queue(maxsize=1)
        self._result: List[Tuple] = []

    def init(self) -> bool:
        try:
            self._model = torch.hub.load(
                "ultralytics/yolov5", "custom",
                path=YOLO_MODEL_PATH,
                force_reload=False,
                verbose=False,
            )
            self._model.conf = YOLO_CONF_THRESH
            self._model.iou  = 0.45
            threading.Thread(target=self._worker, daemon=True,
                             name="ConeDetector").start()
            log.info("[ConeDetector] Loaded %s", YOLO_MODEL_PATH)
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
                for *xyxy, conf, _ in results.xyxy[0].cpu().numpy():
                    dets.append((float(xyxy[0]), float(xyxy[1]),
                                 float(xyxy[2]), float(xyxy[3]), float(conf)))
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
    Reads the road_mask at SEG_NEAR_FRAC row — leftmost and rightmost road pixel.
    Falls back to ±2.0 m when mask is unavailable.
    """
    if road_mask is None:
        return -2.0, 2.0

    y_near   = int(H * SEG_NEAR_FRAC)
    safe_top = int(H * ROI_TOP_FRACTION)
    row      = road_mask[y_near].copy()
    row[:safe_top] = False        # strip sky
    cols = np.where(row)[0]

    if len(cols) < 4:
        return -2.0, 2.0

    mid      = int((cols[0] + cols[-1]) / 2)
    Z_near   = _patch_depth(depth_arr, y_near, mid, H, W)
    left_X   = (float(cols[0])  - W / 2.0) * Z_near / fx
    right_X  = (float(cols[-1]) - W / 2.0) * Z_near / fx
    return left_X, right_X


# ─────────────────────────────────────────────────────────────────────────────
# Path-blocking check
# ─────────────────────────────────────────────────────────────────────────────

def path_blocking_cones(
    cones_cam: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """
    Return only the cones that are actually in the car's forward path corridor.
    A cone to the side of the road (|X| > PATH_WIDTH_M) is irrelevant and excluded.
    """
    return [
        (X, Z) for (X, Z) in cones_cam
        if Z < AVOIDANCE_TRIGGER_M and abs(X) < PATH_WIDTH_M
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
        return 0.0, GAP_LOOKAHEAD_M

    # Build exclusion intervals and merge overlaps
    exclusions = sorted(
        (X - CONE_RADIUS_M, X + CONE_RADIUS_M)
        for X, _ in blocking_cones
    )
    merged: List[Tuple[float, float]] = []
    for lo, hi in exclusions:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append([lo, hi])

    # Derive passable gaps: intervals of [safe_left, safe_right] not blocked
    gaps: List[Tuple[float, float, float]] = []  # (left, right, centre)
    cursor = safe_left
    for lo, hi in merged:
        gap_r = lo      # right edge of gap before this obstacle
        if gap_r - cursor >= MIN_GAP_M:
            centre = (cursor + gap_r) / 2.0
            gaps.append((cursor, gap_r, centre))
        cursor = max(cursor, hi)
    # Gap after last obstacle
    if safe_right - cursor >= MIN_GAP_M:
        centre = (cursor + safe_right) / 2.0
        gaps.append((cursor, safe_right, centre))

    if not gaps:
        # No gap wide enough — pick the widest available gap anyway (best effort)
        cursor = safe_left
        all_gaps = []
        for lo, hi in merged:
            if lo > cursor:
                all_gaps.append((cursor, lo, (cursor + lo) / 2.0))
            cursor = max(cursor, hi)
        if safe_right > cursor:
            all_gaps.append((cursor, safe_right, (cursor + safe_right) / 2.0))
        gaps = all_gaps if all_gaps else [(safe_left, safe_right, 0.0)]

    # Score gaps: wider = better; prefer centre
    best_centre = 0.0
    best_score  = float("-inf")
    for gl, gr, gc in gaps:
        width = gr - gl
        score = width - GAP_CENTER_WEIGHT * abs(gc)
        if score > best_score:
            best_score  = score
            best_centre = gc

    # Clamp target to safe lane area
    best_centre = float(np.clip(best_centre, safe_left, safe_right))
    return best_centre, GAP_LOOKAHEAD_M


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
) -> np.ndarray:
    out = frame.copy()

    # Road mask tint
    if road_mask is not None:
        tint = np.zeros_like(out)
        tint[road_mask] = [30, 15, 0]
        out = cv2.addWeighted(out, 0.85, tint, 0.4, 0)

    # YOLO bboxes — orange for blocking, grey for others
    blocking_xs = {round(X, 2) for X, _ in blocking}
    for x1, y1, x2, y2, conf in detections:
        col = CLR_CONE if True else (120, 120, 120)
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        cv2.putText(out, f"{conf:.2f}", (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)

    # Reference Z for projecting metres → pixels (approximate, for visualisation only)
    Z_vis   = 1.8
    bot_y   = H - 10
    top_y   = int(H * 0.38)
    mid_x   = W // 2

    def xm_to_px(xm: float, z: float = Z_vis) -> int:
        return int(mid_x + xm * fx / z)

    # Lane edges
    lx = xm_to_px(left_X_m)
    rx = xm_to_px(right_X_m)
    cv2.line(out, (lx, bot_y), (lx, top_y), CLR_LANE_L, 2)
    cv2.line(out, (rx, bot_y), (rx, top_y), CLR_LANE_R, 2)

    # Gap target path (when avoiding)
    if state == "AVOIDING":
        tgt_px = xm_to_px(gap_target_X)
        cv2.line(out, (mid_x, bot_y), (tgt_px, top_y), CLR_GAP_SEL, 3)
        cv2.circle(out, (tgt_px, top_y), 8, CLR_GAP_SEL, -1)
        # Label gap target
        cv2.putText(out, f"gap={gap_target_X:+.2f}m",
                    (tgt_px + 10, top_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_GAP_SEL, 1)

    # Centre path (when lane following)
    if state == "LANE_FOLLOW":
        cv2.line(out, (mid_x, bot_y), (mid_x, top_y), CLR_PATH, 1)

    # Blocking cone positions with depth label
    for i, (X_c, Z_c) in enumerate(blocking):
        cv2.putText(out, f"cone X={X_c:+.2f} Z={Z_c:.1f}m",
                    (10, 58 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_CONE, 1)

    # Status bar
    color = CLR_WARN if state == "AVOIDING" else CLR_OK
    label = (f"AVOIDING  gap={gap_target_X:+.2f}m  steer={steer_deg:+.1f}°"
             if state == "AVOIDING"
             else f"LANE FOLLOW  steer={steer_deg:+.1f}°")
    cv2.rectangle(out, (0, 0), (W, 30), (0, 0, 0), -1)
    cv2.putText(out, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)

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
        self._gap_target: float = 0.0     # current gap centre X (metres)
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

                print(f"  [{state:11s}]  steer={steer_deg:+6.2f}°"
                      f"  gap_tgt={self._gap_target:+.2f}m  spd={target_kmh:.1f}km/h")

                if DISPLAY:
                    lf, rf, road_mask, *_ = self.seg_lane.get_result()
                    dets      = self.cone_det.get_result()
                    cones_cam = self.cone_map.in_camera_frame(self.pose)
                    blocking  = path_blocking_cones(cones_cam)
                    left_X, right_X = lane_bounds_m(
                        road_mask, self._depth_cache, self.H, self.W, self.cal.fx)
                    vis = draw_overlay(
                        frame, dets, blocking, road_mask,
                        self._gap_target, left_X, right_X,
                        steer_deg, state, self.H, self.W, self.cal.fx,
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
            road_mask = np.zeros((H, W), dtype=bool)

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

        # ── Path-blocking cones only ──────────────────────────────────────────
        # Cones outside PATH_WIDTH_M (to the side of the road) are irrelevant.
        blocking = path_blocking_cones(cones_cam)

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
            gap_X, gap_Z    = find_best_gap(blocking, left_X, right_X)
            self._gap_target = gap_X
            raw_deg          = pure_pursuit(gap_X, gap_Z)
            state            = "AVOIDING"

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
        delta = raw_deg - self._last_raw
        if abs(delta) > STEER_RATE_LIMIT_DEG:
            raw_deg = self._last_raw + math.copysign(STEER_RATE_LIMIT_DEG, delta)
        self._last_raw  = raw_deg
        self._steer_ema = STEER_EMA_ALPHA * raw_deg + (1.0 - STEER_EMA_ALPHA) * self._steer_ema

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
