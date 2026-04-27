# Cone Avoidance — Testing & Tuning Guide

**PSU Eco Racing — Shell Eco-Marathon Autonomous Division**

---

## 1. What It Does

The car drives autonomously using Segformer lane detection and pure pursuit (identical to the main pipeline). When YOLOv5 detects a cone inside the forward path corridor, the system switches to **gap-based avoidance**:

1. Finds all passable gaps in the road (intervals not blocked by cones and wide enough for the car)
2. Scores gaps by width and closeness to lane centre
3. Steers the car through the best gap using pure pursuit
4. Returns to normal lane following once the path is clear

Cone positions are tracked persistently in world coordinates using ZED odometry — the car knows where cones are even as it moves past them.

---

## 2. Full Pipeline

```
ZED 2i camera
  │
  ├─ CLAHE lighting normalisation
  │
  ├─ SegformerLane [background thread]
  │     → road_mask → centerline polynomial
  │     → lane_bounds_m(): left_X_m, right_X_m at near row
  │     → compute_lookahead() → X_m, Z_m (pure pursuit target)
  │
  ├─ ConeDetector / YOLOv5 [background thread, every 2 frames]
  │     → bboxes (x1,y1,x2,y2,conf)
  │     → localise_cones(): ZED depth at bbox bottom-centre → (X_cam, Z_cam)
  │
  ├─ ConeWorldMap [ZED positional tracking]
  │     → project camera-frame cones → world frame via ZED pose
  │     → merge detections within 0.5 m radius (EMA position)
  │     → expire entries older than 4 s
  │     → convert all map entries back to current camera frame
  │
  ├─ path_blocking_cones()
  │     → filter: keep only cones where Z < 5 m AND |X| < 1.0 m
  │     → cones to the side of the road are ignored entirely
  │
  ├─ State machine
  │     LANE_FOLLOW → AVOIDING  when closest blocking cone < AVOIDANCE_TRIGGER_M
  │     AVOIDING    → LANE_FOLLOW  when all blocking cones > AVOIDANCE_RELEASE_M
  │                              AND gap target returned within ±0.20 m of centre
  │
  ├─ Steering decision
  │     LANE_FOLLOW:  pure pursuit on Segformer lookahead (same as main pipeline)
  │     AVOIDING:     find_best_gap() → pure pursuit to gap centre
  │     LOST:         steer 0°
  │
  ├─ Rate limiter  (5°/frame max change)
  ├─ EMA smoother  (α = 0.20)
  └─ UART TX deadband (1.5°) → Nucleo
```

---

## 3. Gap Detection — How It Works

This is the core of the avoidance logic. It handles every cone arrangement without special cases.

### Step 1 — Build exclusion zones

Each cone blocks an interval on the X axis:

```
[X_cone - CONE_RADIUS_M,  X_cone + CONE_RADIUS_M]
```

Overlapping zones (adjacent cones) are merged into one combined obstacle.

### Step 2 — Find passable gaps

The drivable space is `[left_X_m + LANE_MARGIN_M, right_X_m - LANE_MARGIN_M]`.

Passable gaps are the intervals within that space NOT covered by any exclusion zone and wide enough for the car:

```
MIN_GAP_M = CAR_WIDTH_M + 2 × GAP_SAFETY_MARGIN_M
```

### Step 3 — Score and select

```
score = gap_width - GAP_CENTER_WEIGHT × |gap_centre_X|
```

Wider gaps score higher. The centering term breaks ties by preferring the gap closest to lane centre (X = 0).

### How each scenario resolves

| Scenario | What happens |
|---|---|
| **1 cone dead centre** (X ≈ 0) | Two gaps, one left and one right. Wider side wins. If equal, GAP_CENTER_WEIGHT is symmetric — add a small bias or the car will consistently pick right (whichever float comparison favours). |
| **2 cones forming a slot** | Gap between them is scored. If slot ≥ MIN_GAP_M, the car threads through. If too narrow, both cones merge into one obstacle and the car goes around. |
| **2 cones side by side** | Exclusion zones merge → treated as one wide obstacle → car goes around. |
| **Cone far to the side** | `path_blocking_cones()` filters it out (|X| > PATH_WIDTH_M). Avoidance never triggers. |

---

## 4. All Tuning Parameters

### Group 1 — Measure before first run

| Parameter | Default | What it is |
|---|---|---|
| `WHEELBASE_M` | `1.6` | Axle-to-axle in metres. Wrong value = systematic over/under-steer on every curve. |
| `CAR_WIDTH_M` | `0.90` | Full vehicle width including bodywork. Used to calculate minimum gap size. |

### Group 2 — Avoidance trigger

| Parameter | Default | Effect |
|---|---|---|
| `PATH_WIDTH_M` | `1.0` | Half-width of the path corridor. Cones outside `±1.0 m` are ignored. Increase if cones near the lane edge are triggering avoidance unnecessarily. Decrease if the car is not reacting to cones close to the edge. |
| `AVOIDANCE_TRIGGER_M` | `5.0` | Engage avoidance when a path-blocking cone is closer than this. Increase to start planning earlier; decrease if it starts avoiding cones that are still far away. |
| `AVOIDANCE_RELEASE_M` | `6.5` | Release avoidance when all blocking cones are beyond this distance. Always higher than `AVOIDANCE_TRIGGER_M` to prevent chatter at the boundary. |
| `RETURN_BAND_M` | `0.20` | Only release avoidance once the gap target is within ±0.20 m of lane centre. Prevents releasing mid-manoeuvre before the car has actually returned to centre. |

### Group 3 — Gap planner (most important for avoidance quality)

| Parameter | Default | Effect |
|---|---|---|
| `CONE_RADIUS_M` | `0.15` | Exclusion half-width per cone. Increase if the car is getting too close to cones. Decrease if it is avoiding too aggressively. |
| `GAP_SAFETY_MARGIN_M` | `0.20` | Extra clearance each side of the car (total gap needed = CAR_WIDTH + 2×margin). Increase for more conservative avoidance. |
| `GAP_LOOKAHEAD_M` | `3.0` | Distance ahead where the gap centre waypoint is placed. Larger = smoother but slower reaction. Smaller = quicker reaction but may overshoot. |
| `GAP_CENTER_WEIGHT` | `0.40` | How strongly the planner prefers gaps near lane centre. Higher = stronger pull to centre (may force narrower gaps). Lower = more willing to use side gaps. |
| `LANE_MARGIN_M` | `0.25` | Minimum clearance from lane edge that any gap target must respect. Prevents the planner from targeting a waypoint right at the road boundary. |

### Group 4 — Steering output (same as main pipeline)

| Parameter | Default | Effect |
|---|---|---|
| `CTRL_LANE_DEADBAND_M` | `0.15` | In LANE_FOLLOW mode: ignore lateral offsets < ±15 cm. |
| `STEER_RATE_LIMIT_DEG` | `5.0` | Max steering change per frame. Clips jolts from bad Segformer frames. |
| `STEER_EMA_ALPHA` | `0.20` | Steering smoothing. Lower = smoother/slower. Higher = more responsive. |
| `STEER_TX_DEADBAND_DEG` | `1.5` | Only send UART if angle changed by this much. Reduces servo hunting. |

### Group 5 — YOLOv5 detector

| Parameter | Default | Effect |
|---|---|---|
| `YOLO_CONF_THRESH` | `0.40` | Detection confidence threshold. Lower if cones are missed; raise if getting false positives (debris, shadows). |
| `YOLO_SKIP_FRAMES` | `2` | Run YOLO every N frames. Increase to save GPU; decrease if cone tracking is too stale at speed. |
| `CONE_Z_MIN_M` | `0.3` | Ignore cones closer than this (likely false positives from bumper/ground clutter). |
| `CONE_Z_MAX_M` | `8.0` | Ignore cones farther than this (depth unreliable). |

### Group 6 — World-frame cone map

| Parameter | Default | Effect |
|---|---|---|
| `CONE_MERGE_RADIUS_M` | `0.50` | Detections within this radius are merged into one map entry. Increase if the same cone keeps spawning duplicates. |
| `CONE_MAX_AGE_S` | `4.0` | Map entries expire if not refreshed. Increase if map entries are disappearing before the car clears them. |

---

## 5. Speed

| Parameter | Default | Effect |
|---|---|---|
| `SPEED_STRAIGHT_KMH` | `3.0` | Normal lane-following speed. |
| `SPEED_AVOID_KMH` | `2.0` | Speed while avoidance is active. Slower = more time to compute and react. |

---

## 6. Before First Run — Checklist

### Step 0 — Measure physical dimensions

- Measure `WHEELBASE_M` (axle centre to axle centre in metres).
- Measure `CAR_WIDTH_M` (widest point of bodywork including mirrors/fairings).
- Update both in the CONFIG block at the top of `obstacle_avoidance.py`.

### Step 1 — Dry run, no motor output

Set `UART_ENABLED = False`. Run:

```bash
cd ~/Documents/Projects/Embedded/shell\ autonomous/PSUECOTEAM_AUTO
python obstacle_avoidance.py
```

Watch the display window. Walk slowly in front of the camera:

- Verify `LANE FOLLOW` appears in the status bar when centred.
- Verify `steer` increases as you move off-centre.
- Place a cone directly in front within 5 m — status bar should switch to `AVOIDING`.
- Remove the cone — should return to `LANE FOLLOW`.

If the status bar always shows `AVOIDING`: `PATH_WIDTH_M` may be too large and a lane-edge marker is triggering the detector. Lower to `0.7`.

If avoidance never triggers with a cone close in front: the cone is not being detected. Check `YOLO_CONF_THRESH` — try lowering to `0.30`.

### Step 2 — Check cone 3D localisation

With `UART_ENABLED = False`, place a cone exactly **2 m ahead** and centred. The console should print:

```
cone X=+0.00 Z=2.0m
```

`Z` should read close to **2.0**. If consistently off (e.g. 1.4 or 2.6), the depth patch around the cone base is landing on a bad ZED region. Try raising `CONE_DEPTH_PAD` to `10` for a wider patch.

`X` should be near **0.0** when the cone is centred. If it is consistently offset, the camera is not mounted on the vehicle centreline — note the offset and compensate later.

### Step 3 — Verify gap planner output

Place one cone centred, 3 m ahead. Observe the green path line in the display — it should swing left or right (not straight through the cone). The console will print:

```
[AVOIDING    ]  steer=+7.20°  gap_tgt=+0.55m
```

A positive `gap_tgt` means the planner selected a gap to the right of centre. Negative = left.

Place two cones at approximately ±0.5 m, 3 m ahead. The green path should aim between them at approximately X = 0.

### Step 4 — Verify steer sign (wheels raised off ground)

Set `UART_ENABLED = True`. Raise the front wheels. With the car stationary, trigger avoidance by placing a cone:

- Cone slightly **to the left** of centre → planner selects gap on the right → front wheels should steer **right**.
- Cone slightly **to the right** of centre → gap on the left → wheels steer **left**.

If wheels turn the wrong way, flip the sign in `pure_pursuit()`:

```python
raw = math.atan2(2.0 * WHEELBASE_M * (-X_m), ld * ld)  # negate X_m
```

---

## 7. Live Testing — Step by Step

### Phase 1 — Single cone, low speed

Start with:
```python
SPEED_STRAIGHT_KMH = 2.0
SPEED_AVOID_KMH    = 1.5
AVOIDANCE_TRIGGER_M = 5.0
```

Drive toward a single cone placed at lane centre. Expected: car begins steering away from cone at ~5 m, passes the cone with comfortable clearance, returns to centre.

| What you see | Cause | Fix |
|---|---|---|
| Car does not react to cone | Cone outside `PATH_WIDTH_M` corridor | Verify cone is centred; lower `PATH_WIDTH_M` slightly |
| Car reacts but not enough clearance | `CONE_RADIUS_M` too small | Raise to `0.20` |
| Car avoids but does not return to centre | `RETURN_BAND_M` too tight or `AVOIDANCE_RELEASE_M` too close | Raise `AVOIDANCE_RELEASE_M` to `7.0` |
| Car oscillates during avoidance | Steer EMA too fast | Lower `STEER_EMA_ALPHA` to `0.15` |
| Car avoids then immediately re-triggers | Cone still in map on return path | Raise `CONE_MAX_AGE_S` — or lower `AVOIDANCE_RELEASE_M` |

### Phase 2 — Two cones, threading

Place two cones forming a slot wider than `MIN_GAP_M` (default 1.30 m). Drive through.

Expected: car threads through the centre of the slot without triggering a wide avoidance manoeuvre.

| What you see | Cause | Fix |
|---|---|---|
| Car goes around instead of through | Slot is narrower than `MIN_GAP_M` | Either widen the slot or reduce `GAP_SAFETY_MARGIN_M` |
| Car clips a cone | `CONE_RADIUS_M` too small for the slot width | Raise `CONE_RADIUS_M` |
| Car threads but barely fits | Good — increase `GAP_SAFETY_MARGIN_M` for more comfort | `GAP_SAFETY_MARGIN_M` 0.20 → 0.30 |

### Phase 3 — Two cones, side by side (go around)

Place two cones 0.3 m apart (combined width < `MIN_GAP_M`). Car should treat them as one obstacle and go around.

Expected: their exclusion zones merge, single gap computed on the wider side.

### Phase 4 — Cone to the side (should not trigger)

Place a cone 1.5 m to the right of lane centre, 3 m ahead. Car should drive straight past it without triggering avoidance.

If avoidance does trigger: `PATH_WIDTH_M` is too wide. Lower to `0.80`.

---

## 8. Quick Symptom → Fix

| Symptom | Most Likely Cause | Fix |
|---|---|---|
| Avoidance never triggers | Cone not in path corridor | Lower `PATH_WIDTH_M`; check `YOLO_CONF_THRESH` |
| Avoidance triggers for side cones | Corridor too wide | Lower `PATH_WIDTH_M` |
| Car threads slot but clips cone | Exclusion zone too small | Raise `CONE_RADIUS_M` |
| Car won't thread slot, goes wide | Slot narrower than MIN_GAP | Reduce `GAP_SAFETY_MARGIN_M` |
| Car avoids then oscillates back | EMA / rate limit too aggressive | Lower `STEER_EMA_ALPHA` or `STEER_RATE_LIMIT_DEG` |
| Car does not return to centre | Release conditions too strict | Lower `AVOIDANCE_RELEASE_M` or `RETURN_BAND_M` |
| Ghost cones from previous run | Map entries outlasting the cone | Lower `CONE_MAX_AGE_S` |
| Same cone appears as 2 entries | Merge radius too small | Raise `CONE_MERGE_RADIUS_M` |
| Avoidance too jerky | Gap target jumping frame-to-frame | Raise `STEER_EMA_ALPHA` slightly or `STEER_RATE_LIMIT_DEG` ↓ |
| Car slows too much during avoid | `SPEED_AVOID_KMH` too conservative | Raise to `2.5` |
| FPS drops during avoidance | YOLO + Segformer both heavy | Raise `YOLO_SKIP_FRAMES` to `3` |

---

## 9. Display Overlay Guide

| Element | Meaning |
|---|---|
| **Orange box** | Detected cone bounding box with confidence score |
| **Blue vertical line (left)** | Left lane boundary from Segformer road_mask |
| **Cyan vertical line (right)** | Right lane boundary |
| **Grey road tint** | Segformer drivable area |
| **Green angled line** | Selected gap path (AVOIDING mode) |
| **Green dot** | Gap centre waypoint |
| **Thin white vertical** | Lane centre (LANE_FOLLOW mode) |
| **Status bar (green)** | LANE FOLLOW — normal driving |
| **Status bar (orange/red)** | AVOIDING — gap planner active |

---

## 10. Running as Standalone

```bash
# From PSUECOTEAM_AUTO directory
cd ~/Documents/Projects/Embedded/shell\ autonomous/PSUECOTEAM_AUTO

# Dry run (no motor output, display only)
# Set UART_ENABLED = False in config block first
python obstacle_avoidance.py

# Live run (UART enabled)
python obstacle_avoidance.py

# Press 'q' in the display window to quit cleanly
# On quit: speed → 0, steer → centre, camera closed
```

---

## 11. Integrating into main.py

When the standalone is validated, replace the lane-only `Commander` with the avoidance system:

```python
# In main.py
from obstacle_avoidance import ConeAvoidanceSystem

avoidance = ConeAvoidanceSystem()
avoidance.init()

# In the main loop, replace commander.update(result) with:
steer_deg, avoid_state = avoidance.process_frame(
    frame, depth_arr, pose, pipeline.cal.fx, pipeline.H, pipeline.W
)
```

At that point, swap `YOLO_MODEL_PATH` to your YOLOv8 model path and update `ConeDetector` to use the `ultralytics` API instead of `torch.hub`.

---

## 12. Key Parameters at a Glance

```python
# Physical — measure once, lock
WHEELBASE_M           = 1.6     # axle to axle (m)
CAR_WIDTH_M           = 0.90    # full body width (m)

# Trigger — when to engage avoidance
AVOIDANCE_TRIGGER_M   = 5.0     # cone must be closer than this
PATH_WIDTH_M          = 1.0     # cone must be within ±this in X

# Gap planner — quality of avoidance
CONE_RADIUS_M         = 0.15    # exclusion half-width per cone
GAP_SAFETY_MARGIN_M   = 0.20    # clearance buffer each side of car
GAP_LOOKAHEAD_M       = 3.0     # waypoint distance ahead

# Release — when to return to lane follow
AVOIDANCE_RELEASE_M   = 6.5     # all cones must be beyond this
RETURN_BAND_M         = 0.20    # gap target must be within ±this of centre
```

---

*PSU Eco Racing — Shell Eco-Marathon Autonomous Division*
