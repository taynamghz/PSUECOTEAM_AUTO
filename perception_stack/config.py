"""
PSU Eco Racing — Perception Stack
config.py  |  All tunable parameters in one place.
Edit this file to tune thresholds without touching algorithm logic.
"""

import pyzed.sl as sl

# ── Camera ─────────────────────────────────────────────────────────────────────
# Hardware: Jetson Orin Nano 8GB Super + ZED 2i
CAM_RES        = sl.RESOLUTION.HD720
CAM_FPS        = 30
# PERFORMANCE = fast, low memory — appropriate for Orin Nano 8GB Super
# NEURAL      = higher quality but heavier GPU load
CAM_DEPTH_MODE = sl.DEPTH_MODE.PERFORMANCE

# ── Region of interest ─────────────────────────────────────────────────────────
# Fraction of frame height to ignore from the top (sky, bonnet).
# Used by: Segformer floor-mask cutoff, lane/control lookahead scan.
ROI_TOP_FRACTION = 0.35

# ── Stop-line (orange horizontal stripe painted on road) ──────────────────────
STOP_ORANGE_H_MIN   = 5         # HSV hue lower  (orange)
STOP_ORANGE_H_MAX   = 20        # HSV hue upper  (orange)
STOP_ORANGE_S_MIN   = 150       # vivid orange only
STOP_ORANGE_V_MIN   = 100       # reject dark / shadowed patches
STOP_ROW_THRESH     = 0.08      # fraction of row width that must be orange
STOP_COVERAGE_MIN   = 0.60      # fraction of lane interior that must be lit
STOP_PERP_MAX_DEG   = 20.0      # cluster must be within ±20° of horizontal
STOP_DIST_MIN_M     = 0.3
STOP_DIST_MAX_M     = 10.0
STOP_DIST_N_PTS     = 10        # sample points for median distance
STOP_DIST_MIN_VALID = 4         # min valid ZED samples required
STOP_VOTE_NEEDED    = 5         # consecutive positive frames before triggering

# ── Stop line — physical stripe-width gate ─────────────────────────────────────
# Reject orange detections narrower than this fraction of the measured lane width.
# SEM stop stripe spans full track width (≥ 2 m); cones and debris are narrower.
STOP_WIDTH_MIN_FRAC = 0.70

# ── Feature flags ─────────────────────────────────────────────────────────────
# False → skip Segformer, stop-line, and control outputs; only stop-sign runs.
LANE_ENABLED = True

# ── Stop-sign detection (YOLOv8) ──────────────────────────────────────────────
# Train:  python scripts/train_stop_sign.py --api-key YOUR_KEY
# Export: python scripts/export_trt.py  (TensorRT FP16 for Jetson)
# Use .engine path after export; .pt works for development without TRT
SIGN_MODEL_PATH      = "perception_stack/weights/stop_sign.engine"  # TensorRT FP16 — built on this Jetson
SIGN_CONF_THRESH     = 0.60
SIGN_IMG_SIZE        = 416
SIGN_ACCEPT_CLASSES  = {0, 2}    # 0=stop-sign  2=stop-sign-vandalized
SIGN_SKIP_FRAMES     = 3         # run YOLO every N frames; cache between
SIGN_DIST_MIN_M      = 0.5
SIGN_DIST_MAX_M      = 15.0
SIGN_VOTE_NEEDED     = 3         # consecutive detections before confirming
# SEM-specific: sign sits on a yellow rectangular board
SIGN_YELLOW_H_MIN    = 18        # HSV hue range for SEM yellow board
SIGN_YELLOW_H_MAX    = 38
SIGN_YELLOW_S_MIN    = 120
SIGN_YELLOW_V_MIN    = 150
SIGN_YELLOW_ROI_FRAC  = 1.3      # expand bbox by this factor when sampling yellow
SIGN_YELLOW_AREA_FRAC = 0.12     # minimum yellow fraction in expanded roi
SIGN_FY_APPROX        = 730      # approx. vertical focal length at 720p (px)
SIGN_HEIGHT_M         = 0.65     # assumed sign height (m)
SIGN_BBOX_MIN_FRAC    = 0.35     # bbox height must be ≥ this fraction of expected px height

# ── UART / low-level controller ────────────────────────────────────────────────
UART_ENABLED       = True
UART_PORT          = "/dev/ttyTHS1"   # Jetson hardware UART; /dev/ttyUSB0 on PC
UART_BAUD          = 115200
UART_TIMEOUT_S     = 0.01
UART_ACK_TIMEOUT_S = 0.05
UART_HEARTBEAT_S   = 0.080   # force retransmit every 80ms — keeps Nucleo watchdog alive

# ── Vehicle commands ────────────────────────────────────────────────────────────
# The Nucleo runs a PID controller internally.
# Jetson sends ONLY the setpoints; Nucleo handles throttle, braking, and PWM.
STOP_BRAKE_DIST_M = 1.0     # stop-line/sign within this distance → send CMD_BRAKE
BRAKE_VALUE       = 255     # brake intensity byte sent with CMD_BRAKE

# ── Target speed setpoints ────────────────────────────────────────────────────
# Sent as CMD_THROTTLE DATA byte = int(kmh * 10)  →  e.g. 150 = 15.0 km/h
SPEED_TARGET_STRAIGHT_KMH = 3.0    # nominal speed on straight sections
SPEED_TARGET_CURVE_KMH    = 3.0    # reduced speed through corners
SPEED_CURVE_THRESH        = 0.15   # |κ| (m⁻¹) above which we slow to curve speed

# ── Lane-following control (Pure Pursuit) ─────────────────────────────────────
CTRL_LOOKAHEAD_M     = 2.2   # lookahead distance for Pure Pursuit (metres)
CTRL_HEADING_ALPHA   = 0.20  # EMA alpha for heading angle  (lower = smoother)
CTRL_CURVATURE_ALPHA = 0.15  # EMA alpha for curvature      (extra-smooth)
CTRL_EVAL_Y_FRAC     = 0.60  # image-row fraction to evaluate heading/curvature

# Lateral tolerance corridor — only apply lateral correction when the car drifts
# beyond this distance from the lane centre.  Inside the band, deviation is zeroed
# and the car is steered purely by road heading (curvature feed-forward).
# Goal: stay on track and take curves safely, not chase the exact centreline.
# Adaptive: scales with measured lane width so the corridor stays proportional.
# e.g. 3 m lane → ±0.45 m corridor (middle 70%).  1.5 m lane → ±0.23 m corridor.
# Falls back to CTRL_LATERAL_DEADBAND_M when lane width is unknown.
CTRL_LATERAL_DEADBAND_FRAC = 0.085  # fraction of lane width — tighter centering
CTRL_LATERAL_DEADBAND_M    = 0.15   # fallback when lane_width_m is unavailable

# ── Steering output (anti-jitter stack) ───────────────────────────────────────
# Data flow every frame:
#   Pure Pursuit → clamp → dead-band → rate-limit → EMA → UART byte (0-255)
#
# 0   = full left  (-STEER_MAX_DEG)
# 127 = straight   (0°)
# 255 = full right (+STEER_MAX_DEG)
STEER_MAX_DEG         = 25.0  # maximum steering angle — increase once tracking is stable
STEER_MIN_DEG         = 1.0   # minimum output for any correction outside deadband (soft entry)
STEER_DEADBAND_DEG    = 3.0   # ignore corrections smaller than this (mask noise)
STEER_RATE_DEG        = 8.0   # maximum output for sharp corrections (was 15 — too aggressive)
STEER_EMA_ALPHA       = 0.10  # EMA weight — lower = smoother/slower response (was 0.15)
STEER_TX_DEADBAND_DEG = 2.0   # only transmit CMD_STEER if angle changed by more than
                               # this from the last SENT value.  Suppresses rapid
                               # micro-corrections from mask noise reaching the motor.

# ── Display ────────────────────────────────────────────────────────────────────
DISPLAY = True

# ── Profiling ──────────────────────────────────────────────────────────────────
PROFILE_ENABLED      = True
PROFILE_PRINT_EVERY  = 30   # frames between profile dumps

# ── CLAHE — lighting normalisation applied before colour thresholds ────────────
CLAHE_CLIP_LIMIT  = 2.0
CLAHE_TILE_SIZE   = (8, 8)

# ── Depth / point-cloud refresh ───────────────────────────────────────────────
# Retrieve full XYZ point cloud at most every N frames.
# At 15 km/h the car moves ~0.14 m per frame — stale ≤4 frames = ≤0.56 m,
# acceptable for vote-gated stop decisions.
PC_REFRESH_EVERY   = 4

# ── FPS monitoring ─────────────────────────────────────────────────────────────
FPS_WARN_BELOW     = 20.0

# ── Telemetry logging ──────────────────────────────────────────────────────────
LOG_TELEMETRY      = True
LOG_DIR            = "logs"

# ── Adaptive Segformer submission rate ────────────────────────────────────────
# On straights the road mask barely changes frame-to-frame, so we only submit
# a new frame every SEG_SKIP_STRAIGHT frames — the EMA polynomial carries between
# submissions without noticeable error.
# On curves we submit every SEG_SKIP_CURVE frames for maximum steering freshness.
# Detection of straight vs. curve uses the smoothed curvature from the last result.
# (This reduces GPU usage and power draw on straights — critical for eco-marathon.)
SEG_SKIP_STRAIGHT = 1   # submit every frame — always fresh lane data
SEG_SKIP_CURVE    = 1   # submit every frame on curves (full inference rate)

# ── Segformer drivable-area lane detection ─────────────────────────────────────
# Priority: TRT engine → ONNX Runtime → HuggingFace (slowest fallback)
SEG_ENGINE_PATH = "segformer_road.engine"  # TensorRT FP16 (built on this Jetson)
SEG_ONNX_PATH   = "segformer_road.onnx"   # tuned 2-class ONNX (relative to working dir)
SEG_INPUT_H     = 640
SEG_INPUT_W     = 640
SEG_MODEL_ID    = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"  # HF fallback
# Tuned model: class 0 = background, class 1 = road.  HF model: class 0 = road.
SEG_ROAD_CLASSES   = [1]        # 2-class tuned model: 1 = road
SEG_ROI_TOP_FRAC   = 0.35       # ignore top fraction of frame (sky / hood)
SEG_MIN_ROAD_FRAC  = 0.02       # min road fraction per row to count as valid boundary
SEG_BOUNDARY_ROWS  = 30         # rows scanned top→bottom for left/right boundary
SEG_POLY_DEG       = 2          # quadratic fit  x = a·y² + b·y + c
SEG_CONF_THRESHOLD      = 0.35  # min valid-row fraction to accept fresh fit vs hold EMA
SEG_NEAR_FRAC           = 0.85  # image-row fraction for near point (lateral deviation)
SEG_FAR_FRAC            = 0.65  # image-row fraction for far  point (heading angle)
SEG_MAX_LANE_WIDTH_FRAC = 0.70  # max believable lane width as fraction of frame width
SEG_FIT_TOP_FRAC        = 0.72  # only fit boundary rows BELOW this fraction of frame height
                                 # = approx. lookahead distance (2.5 m) in image space
                                 # rows above this are far/wide/noisy and blow up the polynomial
SEG_CENTERLINE_ALPHA    = 0.30  # EMA weight for centerline polynomial — lower = smoother/slower
