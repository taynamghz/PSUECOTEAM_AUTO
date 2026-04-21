"""
PSU Eco Racing — Perception Stack
perception/pipeline.py  |  LanePerception — main camera + processing pipeline.

Threading model:
  Main thread  — camera grab, CLAHE, Segformer lane detection, stop-line,
                 heading/curvature/lookahead.  Targets 15-20 FPS on Jetson.
  Sign thread  — YOLOv8 stop-sign inference inside StopSignDetector.
                 Decoupled via a queue; never blocks the main thread.

Call init() once, then process() every frame.
"""

import collections
import time
import numpy as np
import cv2
import pyzed.sl as sl
from typing import Optional

from perception_stack.config import (
    CAM_RES, CAM_FPS, CAM_DEPTH_MODE,
    ROI_TOP_FRACTION,
    STOP_VOTE_NEEDED,
    SPEED_CURVE_THRESH,
    SEG_SKIP_STRAIGHT, SEG_SKIP_CURVE,
    SEG_FIT_TOP_FRAC, SEG_NEAR_FRAC,
    PROFILE_ENABLED, PROFILE_PRINT_EVERY,
    CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE,
    FPS_WARN_BELOW,
    PC_REFRESH_EVERY,
    LANE_ENABLED,
    LOST_BRAKE_ENABLED,
    LOST_BRAKE_FRAMES,
    STOP_SIGN_ENABLED,
    CONE_AVOIDANCE_ENABLED,
)
from perception_stack.models import PerceptionResult
from perception_stack.lane.fitting import eval_x
from perception_stack.lane.control import (
    compute_heading, compute_curvature, compute_lookahead, ControlSmoother,
)
from perception_stack.detection.stop_line import detect_stop_line
from perception_stack.detection.stop_sign import StopSignDetector
from perception_stack.detection.cone_avoider import ConeAvoider
from perception_stack.perception.segformer_lane import SegformerLane


class LanePerception:

    def __init__(self):
        self.cam      = sl.Camera()
        self.frame_cnt = 0

        self.ctrl_smoother = ControlSmoother()

        self.image_mat = sl.Mat()
        self.pc_mat    = sl.Mat()   # full XYZ — only for stop-line/sign distance
        self.depth_mat = sl.Mat()   # single-channel Z — for steering lookahead
        self.runtime   = sl.RuntimeParameters()

        self.cal = None
        self.W = self.H = None

        # Last known good values — carried between frames when source is LOST
        self._last_deviation: float = 0.0
        self._last_heading:   float = 0.0
        self._last_curvature: float = 0.0
        self._last_source:    str   = "LOST"
        self._last_lc:        float = 0.0
        self._last_rc:        float = 0.0

        # Adaptive Segformer submission rate (straight vs. curve)
        self._seg_skip_ctr: int = 0

        # Stop-line temporal vote gate
        self._stop_votes:     int           = 0
        self._last_stop_dist: float         = 0.0
        self._last_stop_y:    Optional[int] = None

        # Detectors
        self.sign_detector = StopSignDetector() if STOP_SIGN_ENABLED else None
        self.cone_avoider  = ConeAvoider()      if CONE_AVOIDANCE_ENABLED else None
        self.seg_lane      = SegformerLane()

        # CLAHE for lighting normalisation (one instance reused every frame)
        self._clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_SIZE)

        # Depth cache (single-channel Z) — refreshed every PC_REFRESH_EVERY frames.
        # Used by compute_lookahead for accurate steering: 3.7 MB vs 11 MB for XYZ.
        self._depth_cache: Optional[np.ndarray] = None
        self._depth_age:   int                  = PC_REFRESH_EVERY  # force fetch frame 1
        self.depth_arr:    Optional[np.ndarray] = None  # public: latest depth, for cone avoidance

        # Full XYZ cache — only retrieved when stop-line/sign detection is active.
        self._pc_cache: Optional[np.ndarray] = None

        # Consecutive LOST-source frame counter — triggers emergency_stop after threshold
        self._lost_frame_cnt: int = 0

        # Pose — updated every frame; public so cone avoidance can read it.
        self.pose = sl.Pose()

        # Rolling frame-time buffer for FPS monitoring
        self._frame_times: collections.deque = collections.deque(maxlen=30)

        # Per-step timing accumulators (reset every PROFILE_PRINT_EVERY frames)
        self._prof: dict = {}

    # ── Init ───────────────────────────────────────────────────────────────────

    def init(self) -> bool:
        print("[Perception] Rebooting camera...")
        sl.Camera.reboot(0)
        time.sleep(3)

        init_p = sl.InitParameters()
        init_p.camera_resolution = CAM_RES
        init_p.camera_fps        = CAM_FPS
        init_p.depth_mode        = CAM_DEPTH_MODE
        init_p.coordinate_units  = sl.UNIT.METER
        init_p.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

        if self.cam.open(init_p) != sl.ERROR_CODE.SUCCESS:
            print("[Perception] Camera open failed")
            return False

        tp = sl.PositionalTrackingParameters()
        tp.set_floor_as_origin = True
        self.cam.enable_positional_tracking(tp)
        self.runtime.measure3D_reference_frame = sl.REFERENCE_FRAME.WORLD

        info     = self.cam.get_camera_information()
        self.cal = info.camera_configuration.calibration_parameters.left_cam
        self.W   = info.camera_configuration.resolution.width
        self.H   = info.camera_configuration.resolution.height

        print(f"[Perception] OK  {self.W}×{self.H} @ {CAM_FPS} fps  "
              f"depth={CAM_DEPTH_MODE.name}")

        if LANE_ENABLED:
            if not self.seg_lane.init():
                print("[Perception] SegformerLane init failed — lane detection disabled")

        if CONE_AVOIDANCE_ENABLED and self.cone_avoider is not None:
            if not self.cone_avoider.init():
                print("[Perception] ConeAvoider init failed — cone avoidance disabled")

        return True

    # ── CLAHE normalisation ────────────────────────────────────────────────────

    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """Equalise the L channel (LAB) to reduce auto-exposure / shadow effects."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # ── Profiling ──────────────────────────────────────────────────────────────

    def _tick(self, key: str, t_start: float) -> float:
        now = time.perf_counter()
        self._prof[key] = self._prof.get(key, 0.0) + (now - t_start) * 1000.0
        return now

    def _print_profile(self) -> None:
        n     = PROFILE_PRINT_EVERY
        total = sum(self._prof.values()) / n
        lines = [f"\n[Profile] avg over {n} frames  "
                 f"(total={total:.1f} ms → {1000.0/total:.1f} fps est.)"]
        for k, v in sorted(self._prof.items(), key=lambda x: -x[1]):
            lines.append(f"  {k:<22s} {v/n:6.1f} ms")
        print("\n".join(lines))
        self._prof = {}

    # ── Main processing loop ───────────────────────────────────────────────────

    def process(self):
        t = time.perf_counter()

        # ── Camera grab ───────────────────────────────────────────────────────
        if self.cam.grab(self.runtime) != sl.ERROR_CODE.SUCCESS:
            return None
        self.frame_cnt += 1

        self.cam.retrieve_image(self.image_mat, sl.VIEW.LEFT)
        frame      = self.image_mat.get_data()[:, :, :3].copy()
        frame_norm = self._apply_clahe(frame)

        # ── Depth retrieval (single-channel Z, for steering) ──────────────────
        # MEASURE.DEPTH = Z only = 3.7 MB vs 11 MB for full XYZ.
        # Refreshed every PC_REFRESH_EVERY frames. Cached between refreshes.
        # compute_lookahead uses real ZED Z (handles slopes/tilts) + pinhole X.
        self._depth_age += 1
        if self._depth_cache is None or self._depth_age >= PC_REFRESH_EVERY:
            self.cam.retrieve_measure(self.depth_mat, sl.MEASURE.DEPTH)
            raw = self.depth_mat.get_data()
            self._depth_cache = raw.squeeze().copy()   # (H, W) float32
            self._depth_age   = 0

        # ── Full XYZ point-cloud (for stop-line/sign distance only) ───────────
        # Only retrieved when a detection vote is accumulating — rare during race.
        # Skipped entirely when STOP_SIGN_ENABLED=False and no stop-line votes.
        sign_active = (STOP_SIGN_ENABLED and
                       self.sign_detector.get_result()[0])
        need_pc = (
            self._pc_cache is None
            or sign_active
            or self._stop_votes > 0
        )
        if need_pc:
            self.cam.retrieve_measure(self.pc_mat, sl.MEASURE.XYZ, sl.MEM.CPU)
            self._pc_cache = self.pc_mat.get_data()[:, :, :3].copy()
            self._pc_age   = 0
        pc = self._pc_cache

        if PROFILE_ENABLED: t = self._tick("grab+retrieve", t)

        # ── Stop-sign detector (non-blocking — runs in its own thread) ────────
        if STOP_SIGN_ENABLED:
            self.sign_detector.submit(frame_norm, pc, self.H, self.W)
            sign_confirmed, sign_dist, sign_bbox = self.sign_detector.get_result()
        else:
            sign_confirmed, sign_dist, sign_bbox = False, 0.0, None

        if PROFILE_ENABLED: t = self._tick("sign_detect", t)

        # ── Default outputs (used when LANE_ENABLED=False) ────────────────────
        fm    = None
        lf    = rf = None
        lc    = self._last_lc
        rc    = self._last_rc
        dev   = wid = 0.0
        stop_confirmed = False
        out_y = None
        out_dist = 0.0
        heading_sm      = self._last_heading
        curv_sm         = self._last_curvature
        lookahead_world = lookahead_px = None
        source          = "DISABLED"
        avoidance_state = "LANE_FOLLOW"
        cone_dets       = []

        if LANE_ENABLED:
            # ── Segformer — adaptive submission rate ──────────────────────────
            # On straights the road mask barely changes, so we only feed Segformer
            # a new frame every SEG_SKIP_STRAIGHT frames (~6 Hz).
            # On curves we feed it every SEG_SKIP_CURVE frames (up to 30 Hz) so
            # steering corrections stay fresh through the turn.
            # get_result() always returns the most recently completed inference
            # (non-blocking, ≈1 inference-cycle stale — negligible at 15 km/h).
            on_curve   = abs(self._last_curvature) > SPEED_CURVE_THRESH
            skip_every = SEG_SKIP_CURVE if on_curve else SEG_SKIP_STRAIGHT
            self._seg_skip_ctr += 1
            if self._seg_skip_ctr >= skip_every:
                self.seg_lane.submit(frame_norm, pc, self.H, self.W, self.cal.fx)
                self._seg_skip_ctr = 0

            lf, rf, road_mask, dev, wid, lc, rc, source = \
                self.seg_lane.get_result()

            # If Segformer hasn't produced its first result yet, treat as LOST
            if road_mask is None:
                road_mask = np.zeros((self.H, self.W), dtype=np.uint8)
                source = "LOST"

            self._last_lc     = lc
            self._last_rc     = rc
            self._last_source = source

            # Convert road mask → uint8 floor-mask for stop-line detector.
            # Zero out the top ROI strip (sky / bonnet pixels are not road).
            fm = road_mask.astype(np.uint8) * 255
            fm[:int(self.H * ROI_TOP_FRACTION), :] = 0

            if source != "LOST":
                self._last_deviation = dev
            else:
                dev = self._last_deviation

            if PROFILE_ENABLED: t = self._tick("segformer_lane", t)

            # ── Stop-line (orange stripe) — colour-based ──────────────────────
            # HLS/HSV are only needed for stop-line; compute inline.
            hls = cv2.cvtColor(frame_norm, cv2.COLOR_BGR2HLS)
            hsv = cv2.cvtColor(frame_norm, cv2.COLOR_BGR2HSV)

            raw_stop, raw_y, raw_dist = detect_stop_line(
                frame, fm, lf, rf, pc, self.H, self.W, hls, hsv)
            MAX_VOTES = STOP_VOTE_NEEDED + 5
            self._stop_votes = (min(MAX_VOTES, self._stop_votes + 1) if raw_stop
                                else max(0, self._stop_votes - 1))
            if raw_stop and raw_dist > 0:
                self._last_stop_dist = raw_dist
            if raw_stop and raw_y is not None:
                self._last_stop_y = raw_y

            if PROFILE_ENABLED: t = self._tick("stop_line", t)

            sign_prearmed = sign_confirmed and 3.0 <= sign_dist <= 8.0
            effective_stop_thresh = max(2, STOP_VOTE_NEEDED - (2 if sign_prearmed else 0))
            stop_confirmed = self._stop_votes >= effective_stop_thresh
            out_y    = self._last_stop_y    if stop_confirmed else None
            out_dist = self._last_stop_dist if stop_confirmed else 0.0

            # ── Heading, curvature, lookahead ─────────────────────────────────
            # Evaluate at the top of the fitted window (lookahead point), not
            # CTRL_EVAL_Y_FRAC which falls outside the fitted range.
            y_ctrl = int(self.H * SEG_FIT_TOP_FRAC)
            # wid_px from road mask directly — lf==rf==centerline so diff is 0
            wid_px = 0.0
            if fm is not None:
                y_wid = int(self.H * SEG_NEAR_FRAC)
                cols  = np.where(fm[y_wid] > 0)[0]
                if len(cols) >= 2:
                    wid_px = float(cols[-1] - cols[0])
            have_valid_lane = (lf is not None or rf is not None) and wid > 0.0
            if have_valid_lane:
                heading_raw = compute_heading(lf, rf, y_ctrl)
                curv_raw    = compute_curvature(lf, rf, y_ctrl, wid_px, wid)
                h_sm, k_sm  = self.ctrl_smoother.update(heading_raw, curv_raw)
                self._last_heading   = h_sm
                self._last_curvature = k_sm
            heading_sm = self._last_heading
            curv_sm    = self._last_curvature
            lookahead_world, lookahead_px = compute_lookahead(
                lf, rf, self.H, self.W, self.cal.fx, self._depth_cache)

            # ── Cone avoidance override ───────────────────────────────────────
            # Runs post-Segformer so it receives the grass-validated road_mask.
            # When a blocking cone is detected, returns a synthetic lookahead
            # point at the best gap centre — Commander's Pure Pursuit uses it
            # identically to the normal Segformer lookahead.
            if CONE_AVOIDANCE_ENABLED and self.cone_avoider is not None:
                road_mask_bool = road_mask if isinstance(road_mask, np.ndarray) and road_mask.dtype == bool \
                                 else (road_mask.astype(bool) if road_mask is not None else None)
                av_world, av_px, avoidance_state = self.cone_avoider.process(
                    frame_norm, self._depth_cache, road_mask_bool,
                    dev, self.H, self.W, self.cal.fx, self.frame_cnt,
                )
                cone_dets = self.cone_avoider._detector.get_result()
                if av_world is not None:
                    lookahead_world = av_world
                    lookahead_px    = av_px

            if PROFILE_ENABLED: t = self._tick("control", t)

        # ── LOST-state safety counter ─────────────────────────────────────────
        # If Segformer loses the road for LOST_BRAKE_FRAMES consecutive frames,
        # set emergency_stop so Commander applies the brake immediately.
        if LANE_ENABLED:
            if source == "LOST":
                self._lost_frame_cnt += 1
            else:
                self._lost_frame_cnt = 0
        else:
            self._lost_frame_cnt = 0

        # ── Profiling dump ────────────────────────────────────────────────────
        if PROFILE_ENABLED and self.frame_cnt % PROFILE_PRINT_EVERY == 0:
            self._print_profile()

        # ── FPS monitoring ────────────────────────────────────────────────────
        now_t = time.perf_counter()
        self._frame_times.append(now_t)
        if len(self._frame_times) == self._frame_times.maxlen:
            span = self._frame_times[-1] - self._frame_times[0]
            if span > 0:
                fps = (len(self._frame_times) - 1) / span
                if fps < FPS_WARN_BELOW:
                    print(f"[WARNING] FPS = {fps:.1f}  (target ≥ {FPS_WARN_BELOW:.0f})")

        # speed_kmh is populated by Commander (from LLC UART) after this returns.
        # Pipeline sets it to 0.0 here; Commander.update() writes the real value
        # into result before telemetry logging and display.
        return PerceptionResult(
            deviation_m      = dev,
            confidence       = min(0.99, (lc + rc) / 2.0),
            lane_width_m     = wid,
            source           = source,
            left_fit         = lf,
            right_fit        = rf,
            left_conf        = min(0.99, lc),
            right_conf       = min(0.99, rc),
            stop_line        = stop_confirmed,
            stop_line_y      = out_y,
            stop_line_dist   = out_dist,
            stop_sign        = sign_confirmed,
            stop_sign_dist_m = sign_dist,
            stop_sign_bbox   = sign_bbox,
            heading_angle    = heading_sm,
            curvature        = curv_sm,
            lookahead_point  = lookahead_world,
            lookahead_pixel  = lookahead_px,
            speed_kmh        = 0.0,   # filled by Commander after UART read
            emergency_stop   = (LANE_ENABLED and LOST_BRAKE_ENABLED
                                and self._lost_frame_cnt >= LOST_BRAKE_FRAMES),
            avoidance_state  = avoidance_state,
            cone_detections  = cone_dets,
        ), frame, fm

    def close(self):
        self.cam.disable_positional_tracking()
        self.cam.close()
