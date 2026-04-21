"""
PSU Eco Racing — Perception Stack
detection/cone_avoider.py  |  YOLOv5 cone detector + gap-following avoidance planner.

Architecture:
  ConeDetector — background thread, YOLOv5 inference, queue(maxsize=1), drop-on-full.
                 Non-blocking: main thread never waits on GPU inference.
  ConeAvoider  — stateful gap planner, one call per frame from pipeline.py.
                 Returns an override lookahead point when AVOIDING, else None.
                 When None, Commander uses the normal Segformer lookahead unchanged.

Gap planning (exclusion-zone merging):
  Every visible cone occupies a lateral interval [X - CONE_RADIUS, X + CONE_RADIUS].
  Overlapping intervals are merged → single combined obstacle.
  Passable gaps are the intervals of the drivable corridor NOT covered by any obstacle.
  Best gap scored by width and centrality — never rejected for being narrow.

  Target within each gap: minimum-deviation clearance point, not midpoint.
    For a gap [gl, gr], the car needs at least half its width clear of each edge:
      lo = gl + GAP_CAR_WIDTH_M/2,  hi = gr - GAP_CAR_WIDTH_M/2
    If lo < hi the target is clip(0, lo, hi) — stays as close to X=0 as possible.
    If lo >= hi (gap narrower than car) the target is the midpoint — best effort.
    This prevents a single side-cone from pulling the target to the far edge of the
    corridor (which used to cause 25° steer and near-grass targets).

  Grass-awareness: lane boundaries come from the Segformer road_mask AFTER the
  road_validator has already trimmed grass pixels inward.  Additionally, lane
  bounds are EMA-smoothed across frames so a momentary narrow FOV during turning
  (car rotated right → left boundary temporarily off-screen) does not collapse
  the planning corridor — the planner remembers the real lane width.

  Gap target is also EMA-smoothed: the car makes a gentle initial turn, gains FOV,
  and refines the gap target in real time as more of the scene becomes visible.

  Handles any cone arrangement:
    single cone dead-centre  → two gaps, picks wider side, target near centre
    two cones forming a slot → threads minimum-deviation path through slot
    two cones side-by-side   → merged into one obstacle, goes around
    cone near lane edge      → gap on that side has very small clearance room;
                               target clips to minimum clearance → other side wins
    cone to the side of road → |X| > PATH_WIDTH_M → ignored, no avoidance

State machine:
  LANE_FOLLOW → AVOIDING  when a path-blocking cone enters AVOIDANCE_TRIGGER_M
  AVOIDING    → LANE_FOLLOW  when:
    (a) all blocking cones exit AVOIDANCE_RELEASE_M, AND
    (b) actual lateral deviation < RETURN_BAND_M  (car is back near centre)

Override contract:
  process() returns (lookahead_world, lookahead_px, state).
  When AVOIDING:  lookahead_world = (gap_X_m, GAP_LOOKAHEAD_M)
                  Commander's Pure Pursuit uses this identically to the normal
                  Segformer lookahead — no special avoidance steering needed.
  When LANE_FOLLOW: returns (None, None, "LANE_FOLLOW") → pipeline uses Segformer.
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
    CONE_DEPTH_PAD,
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

# EMA smoothing for lane bounds (preserves corridor width during mid-turn FOV loss)
_LANE_BOUNDS_ALPHA = 0.25   # low = slow to update → stable during turning
# EMA smoothing for gap target (gentle initial steer → refines as FOV widens)
_GAP_TARGET_ALPHA  = 0.35   # higher = more responsive; lower = smoother transitions

log = logging.getLogger(__name__)


# ── Depth helper ───────────────────────────────────────────────────────────────

def _patch_depth(
    depth_arr: np.ndarray, y: int, x: int, H: int, W: int, pad: int = 5
) -> float:
    """
    Median ZED depth at pixel (x, y).
    ZED RIGHT_HANDED_Y_UP: forward objects have NEGATIVE Z.
    Returns abs(median) in metres.  Falls back to 3.0 m on sparse patches.
    """
    r0, r1 = max(0, y - pad), min(H, y + pad + 1)
    c0, c1 = max(0, x - pad), min(W, x + pad + 1)
    patch   = depth_arr[r0:r1, c0:c1]
    valid   = patch[np.isfinite(patch) & (patch < -0.1) & (patch > -30.0)]
    return float(np.median(np.abs(valid))) if valid.size >= 3 else 3.0


# ── Background YOLO cone detector ─────────────────────────────────────────────

class ConeDetector:
    """
    YOLOv5 cone detector running in a dedicated daemon thread.

    submit(frame)      — post latest frame; non-blocking, drops stale frames.
    get_result()       — instant read of latest detection list; non-blocking.

    Each detection: (x1, y1, x2, y2, conf)  — pixel coordinates, float.
    """

    def __init__(self):
        self._model:   object        = None
        self._enabled: bool          = False
        self._in_q:    queue.Queue   = queue.Queue(maxsize=1)
        self._result:  List[Tuple]   = []
        self._lock                   = threading.Lock()

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
            pass   # worker busy; drop old frame, next submit will queue fresh one

    def get_result(self) -> List[Tuple]:
        with self._lock:
            return list(self._result)

    def _worker(self) -> None:
        while True:
            frame = self._in_q.get()   # blocks until main thread submits
            try:
                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res    = self._model(rgb, size=CONE_IMG_SIZE)
                dets   = [
                    (float(x1), float(y1), float(x2), float(y2), float(cf))
                    for *xyxy, cf, _ in res.xyxy[0].cpu().numpy()
                    for x1, y1, x2, y2 in [xyxy]
                ]
                with self._lock:
                    self._result = dets
            except Exception as exc:
                log.warning("[ConeDetector] inference error: %s", exc)


# ── 3D cone localisation ───────────────────────────────────────────────────────

def _localise_cones(
    detections: List[Tuple],
    depth_arr:  np.ndarray,
    H: int, W: int, fx: float,
) -> List[Tuple[float, float]]:
    """
    Convert YOLO bboxes to camera-frame (X_m, Z_m) positions.

    Samples depth at the cone base (bottom of bbox) — the base of a cone sits
    on the road surface, so this avoids reflective-tip depth noise.
    """
    out = []
    for x1, y1, x2, y2, _ in detections:
        cx  = (x1 + x2) / 2.0
        cy  = min(float(y2), H - 1.0)       # cone base row
        xi  = int(np.clip(cx, 0, W - 1))
        yi  = int(np.clip(cy, 0, H - 1))
        Z_m = _patch_depth(depth_arr, yi, xi, H, W, pad=CONE_DEPTH_PAD)
        if not (CONE_Z_MIN_M < Z_m < CONE_Z_MAX_M):
            continue
        X_m = (cx - W / 2.0) * Z_m / fx
        out.append((X_m, Z_m))
    return out


# ── Lane boundary extraction from grass-validated road mask ───────────────────

def _lane_bounds_m(
    road_mask:  Optional[np.ndarray],
    depth_arr:  np.ndarray,
    H: int, W: int, fx: float,
) -> Tuple[float, float]:
    """
    Left/right drivable boundary in camera X (metres) at the near row.

    Reads the road_mask at SEG_NEAR_FRAC (same row used for lateral deviation).
    The road_mask is already grass-validated by the road_validator, so these
    boundaries represent the true asphalt edges — gap targets are clamped here
    and cannot push the car onto grass.

    Returns (-2.0, 2.0) fallback when mask is unavailable or too sparse.
    """
    if road_mask is None:
        return -2.0, 2.0

    y_near = int(H * SEG_NEAR_FRAC)
    row    = road_mask[y_near].copy()
    row[:int(H * ROI_TOP_FRACTION)] = False   # strip sky / hood
    cols   = np.where(row)[0]
    if len(cols) < 4:
        return -2.0, 2.0

    mid     = int((int(cols[0]) + int(cols[-1])) / 2)
    Z_near  = _patch_depth(depth_arr, y_near, mid, H, W)
    left_X  = (float(cols[0])  - W / 2.0) * Z_near / fx
    right_X = (float(cols[-1]) - W / 2.0) * Z_near / fx
    return left_X, right_X


# ── Exclusion-zone gap planner ────────────────────────────────────────────────

def _clearance_target(gl: float, gr: float) -> float:
    """
    Minimum-deviation X inside [gl, gr] that keeps the car clear of both edges.
    Stays as close to X=0 (lane centre) as the gap geometry allows.
    Falls back to midpoint when the gap is narrower than the car (best effort).
    """
    half = GAP_CAR_WIDTH_M / 2.0
    lo   = gl + half
    hi   = gr - half
    if lo >= hi:
        return (gl + gr) / 2.0   # gap too narrow for car — best-effort midpoint
    return float(np.clip(0.0, lo, hi))


def _find_best_gap(
    blocking_cones: List[Tuple[float, float]],
    left_X_m: float,
    right_X_m: float,
) -> float:
    """
    Find the best X target (metres) through the cone field.

    Uses clearance-target (not midpoint): within each gap the car aims for the
    point closest to X=0 that still keeps it clear of both gap edges.
    This means:
      - A cone near the right lane edge leaves a large left gap → target stays
        near X=0, not at the far-left midpoint.
      - A cone near centre splits the corridor → car takes minimum-deviation
        path through whichever gap it picks, rather than over-steering to edge.

    Never rejects a gap — width is used for scoring only.
    If no open gaps exist (cones span full corridor) → widest obstacle interval.
    """
    safe_left  = left_X_m  + LANE_MARGIN_M
    safe_right = right_X_m - LANE_MARGIN_M

    if not blocking_cones:
        return 0.0

    if safe_right <= safe_left:
        return (left_X_m + right_X_m) / 2.0

    # Build and merge exclusion intervals
    exclusions = sorted(
        (X - GAP_CONE_RADIUS_M, X + GAP_CONE_RADIUS_M)
        for X, _ in blocking_cones
    )
    merged: List[List[float]] = []
    for lo, hi in exclusions:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])

    # Collect ALL open intervals with their clearance targets
    gaps: List[Tuple[float, float, float]] = []   # (gl, gr, clearance_target)
    cursor = safe_left
    for lo, hi in merged:
        if lo > cursor:
            ct = _clearance_target(cursor, lo)
            gaps.append((cursor, lo, ct))
        cursor = max(cursor, hi)
    if safe_right > cursor:
        ct = _clearance_target(cursor, safe_right)
        gaps.append((cursor, safe_right, ct))

    if not gaps:
        # All gaps blocked — pick widest interval between merged obstacles
        cursor = safe_left
        best_w = 0.0
        best_t = (safe_left + safe_right) / 2.0
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

    # Score by gap width + centrality of clearance target — pick highest
    best_target = 0.0
    best_score  = float("-inf")
    for gl, gr, ct in gaps:
        score = (gr - gl) - GAP_CENTER_WEIGHT * abs(ct)
        if score > best_score:
            best_score  = score
            best_target = ct

    return float(np.clip(best_target, safe_left, safe_right))


# ── Stateful avoidance planner ─────────────────────────────────────────────────

class ConeAvoider:
    """
    Stateful gap-following planner integrated into the main perception pipeline.

    Usage (from pipeline.py):
        avoider = ConeAvoider()
        avoider.init()                  # once — loads YOLO model

        # every frame, after seg_lane.get_result():
        lp_world, lp_px, av_state = avoider.process(
            frame_norm, depth_arr, road_mask, dev_m, H, W, fx, frame_cnt)

        if lp_world is not None:
            # override Segformer lookahead with gap waypoint
            lookahead_world = lp_world
            lookahead_px    = lp_px

    Returns:
        lp_world   — (X_m, Z_m)  gap waypoint for Pure Pursuit, or None
        lp_px      — (x, y)      approximate pixel position for display, or None
        av_state   — "LANE_FOLLOW" | "AVOIDING"
    """

    def __init__(self):
        self._detector   = ConeDetector()
        self._enabled    = False
        self._state      = "LANE_FOLLOW"
        self._gap_target: float = 0.0

        # EMA of lane boundaries — persists real corridor width when turning narrows FOV
        self._lane_left_ema:  float = -2.0
        self._lane_right_ema: float =  2.0
        self._lane_ema_ready: bool  = False

        # Debug state — populated each AVOIDING frame for overlay / logging
        self._last_cones_cam:  list = []
        self._last_blocking:   list = []
        self._last_detections: list = []
        self._last_gap_X_raw:  float = 0.0
        self._last_gap_X_ema:  float = 0.0
        self._last_lane_ema:   tuple = (-2.0, 2.0)

    def init(self) -> bool:
        ok = self._detector.init()
        self._enabled = ok
        return ok

    def process(
        self,
        frame_norm: np.ndarray,
        depth_arr:  np.ndarray,
        road_mask:  Optional[np.ndarray],   # bool (H,W), already grass-validated
        dev_m:      float,                  # Segformer lateral deviation (metres)
        H: int, W: int, fx: float,
        frame_cnt:  int,
    ) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[int, int]], str]:

        if not self._enabled:
            return None, None, "LANE_FOLLOW"

        # Submit to YOLO worker every CONE_SKIP_FRAMES
        if frame_cnt % CONE_SKIP_FRAMES == 0:
            self._detector.submit(frame_norm)
        detections = self._detector.get_result()

        # 3D localisation in camera frame — no world map, always fresh detections
        cones_cam = _localise_cones(detections, depth_arr, H, W, fx)

        # Filter to only cones inside the forward path corridor
        blocking  = [
            (X, Z) for X, Z in cones_cam
            if Z < AVOIDANCE_TRIGGER_M and abs(X) < PATH_WIDTH_M
        ]
        closest_z = min((Z for _, Z in blocking), default=float("inf"))

        # ── Per-frame debug log (all visible cones, not just blocking) ─────────
        if cones_cam:
            cone_strs = ["  X={:+.2f}m Z={:.2f}m{}".format(
                X, Z,
                " [BLOCKING]" if (Z < AVOIDANCE_TRIGGER_M and abs(X) < PATH_WIDTH_M)
                else " [side/far]"
            ) for X, Z in cones_cam]
            log.debug("[ConeAvoider] frame=%d  %d cone(s) localised:\n%s",
                      frame_cnt, len(cones_cam), "\n".join(cone_strs))
        else:
            log.debug("[ConeAvoider] frame=%d  no cones localised  (dets=%d)",
                      frame_cnt, len(detections))

        # Store last localised cones for debug_overlay
        self._last_cones_cam  = cones_cam
        self._last_blocking   = blocking
        self._last_detections = detections

        # ── State machine ─────────────────────────────────────────────────────
        if self._state == "LANE_FOLLOW" and closest_z < AVOIDANCE_TRIGGER_M:
            self._state = "AVOIDING"
            log.info("[ConeAvoider] Cone at %.2f m — engaging avoidance", closest_z)

        if self._state == "AVOIDING":
            all_clear      = closest_z > AVOIDANCE_RELEASE_M
            back_to_centre = abs(dev_m) < RETURN_BAND_M   # actual car position, not target
            if all_clear and back_to_centre:
                self._state          = "LANE_FOLLOW"
                self._gap_target     = 0.0
                self._lane_ema_ready = False   # re-seed lane bounds fresh next avoidance
                log.info("[ConeAvoider] Path clear — resuming lane follow")

        if self._state != "AVOIDING":
            return None, None, "LANE_FOLLOW"

        # ── Lane bounds — EMA-smoothed ────────────────────────────────────────
        # Fresh reading from grass-validated Segformer mask.
        # When the car turns mid-avoidance the opposite boundary can leave FOV
        # (reads -2.0 / +2.0 fallback).  The EMA keeps the remembered real width
        # so the gap planner doesn't suddenly think the corridor expanded to ±2 m
        # OR collapsed to a narrow slice — both cause wrong gap targets.
        raw_left, raw_right = _lane_bounds_m(road_mask, depth_arr, H, W, fx)
        fallback_left  = -2.0
        fallback_right =  2.0
        if not self._lane_ema_ready:
            # First frame: seed EMA with whatever we have (prefer real reading)
            self._lane_left_ema  = raw_left  if raw_left  != fallback_left  else -1.5
            self._lane_right_ema = raw_right if raw_right != fallback_right else  1.5
            self._lane_ema_ready = True
        else:
            # Only blend in the fresh reading if it looks like a real measurement
            # (not the -2/+2 fallback that fires when the mask row is empty)
            if raw_left != fallback_left:
                self._lane_left_ema = (_LANE_BOUNDS_ALPHA * raw_left
                                       + (1.0 - _LANE_BOUNDS_ALPHA) * self._lane_left_ema)
            if raw_right != fallback_right:
                self._lane_right_ema = (_LANE_BOUNDS_ALPHA * raw_right
                                        + (1.0 - _LANE_BOUNDS_ALPHA) * self._lane_right_ema)

        # ── Lane bounds debug ─────────────────────────────────────────────────
        log.debug("[ConeAvoider] lane bounds  raw=(%.2f, %.2f)  ema=(%.2f, %.2f)  "
                  "width_ema=%.2f m",
                  raw_left, raw_right,
                  self._lane_left_ema, self._lane_right_ema,
                  self._lane_right_ema - self._lane_left_ema)

        # ── Gap planning ──────────────────────────────────────────────────────
        gap_X_raw = _find_best_gap(blocking, self._lane_left_ema, self._lane_right_ema)

        # EMA on gap target — smooths same-side jitter but snaps immediately
        # when the best gap switches sides.  Blending through zero would steer
        # toward the wrong side for several frames while the EMA crossed over.
        prev_gap = self._gap_target
        side_switched = (gap_X_raw * self._gap_target < 0.0
                         and abs(gap_X_raw) > 0.10)
        if side_switched:
            self._gap_target = gap_X_raw   # snap — never blend across zero
        else:
            self._gap_target = (_GAP_TARGET_ALPHA * gap_X_raw
                                + (1.0 - _GAP_TARGET_ALPHA) * self._gap_target)
        gap_X = self._gap_target

        log.debug("[ConeAvoider] gap target  raw=%+.3f m  ema=%+.3f m  "
                  "(prev_ema=%+.3f)  blocking=%d cone(s)  closest_z=%.2f m",
                  gap_X_raw, gap_X, prev_gap, len(blocking), closest_z)

        # Synthetic lookahead point — same format as compute_lookahead() output
        Z_gap    = GAP_LOOKAHEAD_M
        x_px     = int(np.clip(W / 2.0 + gap_X * fx / Z_gap, 0, W - 1))
        y_px     = int(H * SEG_NEAR_FRAC)
        world_pt = (gap_X, Z_gap)
        pixel_pt = (x_px, y_px)

        # Store for overlay
        self._last_gap_X_raw = gap_X_raw
        self._last_gap_X_ema = gap_X
        self._last_lane_ema  = (self._lane_left_ema, self._lane_right_ema)

        return world_pt, pixel_pt, "AVOIDING"

    @property
    def gap_target(self) -> float:
        """Last selected gap centre X (metres) — for display/telemetry."""
        return self._gap_target

    def debug_overlay(
        self,
        frame: np.ndarray,
        detections: Optional[list] = None,
        H: int = 0, W: int = 0, fx: float = 700.0,
    ) -> np.ndarray:
        """
        Draw full avoidance debug info onto frame (BGR).  Call every frame from
        the display path — safe when state is LANE_FOLLOW (draws cones only).

        Draws:
          • Each YOLO bbox — cyan (side/far) or red (blocking)
          • Cone label: X, Z distance in metres
          • Lane boundary lines (EMA) — yellow dashed
          • All gaps as semi-transparent green bands
          • Gap target raw X — white dashed vertical
          • Gap target EMA X — solid green vertical  (the actual lookahead X)
          • State label top-left
        """
        out  = frame.copy()
        dets = detections if detections is not None else self._last_detections
        if H == 0:
            H, W = out.shape[:2]
        y_near = int(H * SEG_NEAR_FRAC)

        # ── YOLO bboxes + labels ──────────────────────────────────────────────
        for i, (x1, y1, x2, y2, conf) in enumerate(dets):
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
            # Find matching 3D position if available
            cam_pos = self._last_cones_cam[i] if i < len(self._last_cones_cam) else None
            is_blocking = cam_pos is not None and any(
                abs(cam_pos[0] - bx) < 0.01 and abs(cam_pos[1] - bz) < 0.01
                for bx, bz in self._last_blocking
            )
            colour = (0, 0, 220) if is_blocking else (200, 200, 0)   # red / cyan
            cv2.rectangle(out, (ix1, iy1), (ix2, iy2), colour, 2)
            if cam_pos:
                label = "X{:+.2f} Z{:.2f}m".format(cam_pos[0], cam_pos[1])
            else:
                label = "conf={:.2f} (no depth)".format(conf)
            cv2.putText(out, label, (ix1, max(iy1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)

        if self._state != "AVOIDING":
            cv2.putText(out, "LANE_FOLLOW", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2, cv2.LINE_AA)
            return out

        # ── Lane boundary lines (EMA) ─────────────────────────────────────────
        left_ema, right_ema = self._last_lane_ema
        for X_m, col in [(left_ema, (0, 220, 220)), (right_ema, (0, 220, 220))]:
            x_px = int(np.clip(W / 2.0 + X_m * fx / GAP_LOOKAHEAD_M, 0, W - 1))
            for y in range(0, H, 16):
                cv2.line(out, (x_px, y), (x_px, min(y + 8, H - 1)), col, 1)

        # ── Gap bands (semi-transparent green) ───────────────────────────────
        safe_left  = left_ema  + LANE_MARGIN_M
        safe_right = right_ema - LANE_MARGIN_M
        excl = sorted(
            (X - GAP_CONE_RADIUS_M, X + GAP_CONE_RADIUS_M)
            for X, _ in self._last_blocking
        )
        merged: List[List[float]] = []
        for lo, hi in excl:
            if merged and lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        cursor = safe_left
        overlay = out.copy()
        for lo, hi in merged:
            if lo > cursor:
                xl = int(np.clip(W / 2.0 + cursor * fx / GAP_LOOKAHEAD_M, 0, W - 1))
                xr = int(np.clip(W / 2.0 + lo     * fx / GAP_LOOKAHEAD_M, 0, W - 1))
                cv2.rectangle(overlay, (xl, y_near - 20), (xr, y_near + 20),
                              (0, 180, 0), -1)
            # Draw exclusion zone in red
            xl_e = int(np.clip(W / 2.0 + lo * fx / GAP_LOOKAHEAD_M, 0, W - 1))
            xr_e = int(np.clip(W / 2.0 + hi * fx / GAP_LOOKAHEAD_M, 0, W - 1))
            cv2.rectangle(overlay, (xl_e, y_near - 20), (xr_e, y_near + 20),
                          (0, 0, 180), -1)
            cursor = max(cursor, hi)
        if safe_right > cursor:
            xl = int(np.clip(W / 2.0 + cursor    * fx / GAP_LOOKAHEAD_M, 0, W - 1))
            xr = int(np.clip(W / 2.0 + safe_right * fx / GAP_LOOKAHEAD_M, 0, W - 1))
            cv2.rectangle(overlay, (xl, y_near - 20), (xr, y_near + 20),
                          (0, 180, 0), -1)
        cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)

        # ── Gap target lines ──────────────────────────────────────────────────
        raw_x_px = int(np.clip(W / 2.0 + self._last_gap_X_raw * fx / GAP_LOOKAHEAD_M,
                               0, W - 1))
        ema_x_px = int(np.clip(W / 2.0 + self._last_gap_X_ema * fx / GAP_LOOKAHEAD_M,
                               0, W - 1))
        # Raw target — white dashed
        for y in range(0, H, 16):
            cv2.line(out, (raw_x_px, y), (raw_x_px, min(y + 8, H - 1)),
                     (255, 255, 255), 1)
        # EMA target — solid bright green
        cv2.line(out, (ema_x_px, 0), (ema_x_px, H - 1), (0, 255, 80), 2)
        cv2.circle(out, (ema_x_px, y_near), 8, (0, 255, 80), -1)

        # ── HUD text ──────────────────────────────────────────────────────────
        lines = [
            "AVOIDING",
            "blocking: {}  closest: {:.2f}m".format(
                len(self._last_blocking),
                min((Z for _, Z in self._last_blocking), default=0.0)),
            "lane EMA: [{:.2f}, {:.2f}]m".format(left_ema, right_ema),
            "gap raw: {:+.3f}m  ema: {:+.3f}m".format(
                self._last_gap_X_raw, self._last_gap_X_ema),
        ]
        for i, txt in enumerate(lines):
            col = (0, 80, 255) if i == 0 else (230, 230, 230)
            cv2.putText(out, txt, (10, 28 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2 if i == 0 else 1,
                        cv2.LINE_AA)

        return out
