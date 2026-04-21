# PSU Eco Racing — Perception Stack Tuning Guide

All tunable parameters live in `perception_stack/config.py`.  
Nothing else needs to be touched to change behaviour.

---

## Testing order

1. Lane only — `CONE_AVOIDANCE_ENABLED = False`, `STOP_SIGN_ENABLED = False`
2. Cone avoidance — `CONE_AVOIDANCE_ENABLED = True`, `STOP_SIGN_ENABLED = False`
3. Stop sign — `CONE_AVOIDANCE_ENABLED = False`, `STOP_SIGN_ENABLED = True`
4. Full stack — all enabled

---

## Phase 1 — Lane following

### Pure Pursuit lookahead

```python
CTRL_LOOKAHEAD_M = 4.0
```

The single most important parameter. Controls how far ahead the car aims.

| Symptom | Direction |
|---|---|
| Slow S-curves, reacts late to bends | decrease → 3.0 |
| Rapid left-right oscillation on straights | increase → 5.0 |
| Takes corners wide (clips outer edge) | decrease → 3.0 |
| Violent steering when slightly off-centre | increase → 5.0 |

**Hard rule:** must stay ≥ 2 × wheelbase (3.2 m). Below that the geometry forces near-maximum steer for any lateral offset — instant oscillation.

---

### Heading feed-forward

```python
HEADING_FF_GAIN = 0.40
```

Pre-steers into curves before the car has drifted laterally. Eliminates the "wait for drift → hard correction → overshoot" S-swerve cycle.

| Symptom | Direction |
|---|---|
| Undershoots curves (runs wide before correcting) | increase → 0.55–0.70 |
| Overshoot on corner entry, wobbles through bend | decrease → 0.20–0.25 |
| Fine on straights, only bad on curves | increase |

Set to 0.0 to disable feed-forward entirely and run pure deviation-only pursuit.

---

### Lane deadband

```python
CTRL_LANE_DEADBAND_M = 0.05
```

Offsets smaller than this are ignored. Prevents micro-corrections from mask noise.

| Symptom | Direction |
|---|---|
| Constant micro-corrections, looks nervous | increase → 0.08–0.10 |
| Slow drift to one side, never corrects until far off | decrease → 0.03 |
| Corrects then overshoots repeatedly | increase → 0.07 |

---

### Steering EMA (output smoother)

```python
STEER_EMA_ALPHA = 0.20
```

Exponential moving average on the final steering output. Lower = heavier smoothing = slower response.

| Symptom | Direction |
|---|---|
| Steering feels jerky, mechanical noise | decrease → 0.12–0.15 |
| Too slow to respond, trails behind curves | increase → 0.30–0.35 |

Formula: `steer = alpha * new + (1 - alpha) * previous`

---

### Steering rate limiter

```python
STEER_RATE_LIMIT_DEG = 5.0
```

Maximum steering angle change allowed per frame (~30 Hz). Prevents a single bad Segformer frame from yanking the servo.

| Symptom | Direction |
|---|---|
| Occasional violent jerk from bad mask frame | decrease → 3.0 |
| Transition from avoidance back to lane is abrupt | decrease → 3.0 |
| Too slow to steer around sharp curves | increase → 7.0 |

---

### TX deadband (UART transmission gate)

```python
STEER_TX_DEADBAND_DEG = 1.5
```

Only transmits a new steering command if the angle changed by more than this from the last sent value. Reduces UART traffic and servo buzz.

| Symptom | Direction |
|---|---|
| Servo buzzing/hunting at constant angle | increase → 2.5 |
| Delayed steering response on gentle curves | decrease → 0.8 |

---

### Heading and curvature EMA

```python
CTRL_HEADING_ALPHA   = 0.20
CTRL_CURVATURE_ALPHA = 0.15
```

Smooth the derived heading angle and curvature. These amplify high-frequency noise so they are kept lower than the polynomial EMA.

| Symptom | Direction |
|---|---|
| Curvature reading jumps, speed fluctuates on straights | decrease both → 0.10 / 0.08 |
| Curve speed reduction kicks in/out repeatedly | decrease `CTRL_CURVATURE_ALPHA` → 0.08 |
| Heading feed-forward feels laggy on corner entry | increase `CTRL_HEADING_ALPHA` → 0.30 |

---

### Segformer centerline EMA

```python
SEG_CENTERLINE_ALPHA = 0.30
```

Smooths the quadratic polynomial fit between frames. Lower = more stable but slower to update.

| Symptom | Direction |
|---|---|
| Centerline jumps frame to frame, steering is jittery | decrease → 0.15–0.20 |
| Slow to update when road curves sharply | increase → 0.40–0.50 |

---

### Segformer confidence threshold

```python
SEG_CONF_THRESHOLD = 0.35
```

Minimum fraction of valid boundary rows required to accept a new Segformer fit (vs. holding the EMA from last frame).

| Symptom | Direction |
|---|---|
| Source = LOST frequently, even on clear road | decrease → 0.20 |
| Bad/noisy fits accepted, steering wanders | increase → 0.50 |

---

### Segformer ROI and fit window

```python
ROI_TOP_FRACTION = 0.35    # ignore top 35% of frame (sky / bonnet)
SEG_ROI_TOP_FRAC = 0.35    # same for Segformer internal pass
SEG_FIT_TOP_FRAC = 0.72    # only fit rows below this fraction of frame height
SEG_NEAR_FRAC    = 0.85    # row for near-point lateral deviation
SEG_FAR_FRAC     = 0.65    # row for far-point heading angle
```

| Symptom | Direction |
|---|---|
| Bonnet or sky bleeds into road mask | increase `ROI_TOP_FRACTION` → 0.40 |
| Lookahead row is in noisy far region | increase `SEG_FIT_TOP_FRAC` → 0.78 |
| Near-point deviation is measured too far ahead | decrease `SEG_NEAR_FRAC` → 0.80 |

---

### Grass boundary validator

```python
GRASS_H_MIN         = 35    # HSV hue lower (green start)
GRASS_H_MAX         = 85    # HSV hue upper (covers yellow-green to blue-green)
GRASS_S_MIN         = 55    # min saturation — rejects pale/dry/dead grass
GRASS_V_MIN         = 40    # min value — rejects shadow grass
GRASS_INNER_PAD     = 8     # px strip sampled just inside boundary
GRASS_FRAC_THRESH   = 0.50  # fraction of strip that must be grass to trigger trim
GRASS_MAX_TRIM_FRAC = 0.20  # max inward trim as fraction of lane width
```

Runs post-Segformer in the worker thread. Detects when the road mask has bled onto grass and trims it back, shifting the centerline away from the edge.

| Symptom | Direction |
|---|---|
| Car hugs grass edge, never corrects inward | decrease `GRASS_FRAC_THRESH` → 0.35 |
| Trims too aggressively on light/dry patches | increase `GRASS_S_MIN` → 75, `GRASS_V_MIN` → 60 |
| Grass bleed is wide, trim doesn't reach | increase `GRASS_MAX_TRIM_FRAC` → 0.30, `GRASS_INNER_PAD` → 12 |
| Trim triggering on road markings | increase `GRASS_H_MIN` → 45 (excludes yellow) |

---

### LOST-state emergency brake

```python
LOST_BRAKE_ENABLED = True
LOST_BRAKE_FRAMES  = 15    # ~500 ms at 30 fps
```

If Segformer reports LOST for this many consecutive frames, `emergency_stop = True` and Commander brakes immediately.

Set `LOST_BRAKE_ENABLED = False` during initial testing so a momentary LOST doesn't stop the car. Re-enable before race.

| Symptom | Direction |
|---|---|
| Car brakes on every shadow / tight bend | increase → 25–30, or disable temporarily |
| Car drives too long on stale data | decrease → 10 |

---

## Phase 2 — Cone avoidance

### Enable / disable

```python
CONE_AVOIDANCE_ENABLED = True
```

Set `False` for lane-only testing. No YOLO thread starts, no GPU budget used.

---

### Engagement distance

```python
AVOIDANCE_TRIGGER_M = 5.0    # engage when blocking cone enters this range
AVOIDANCE_RELEASE_M = 6.5    # release only when all blocking cones exit this range
PATH_WIDTH_M        = 1.0    # lateral half-corridor — cones outside this ignored
```

`RELEASE > TRIGGER` is the hysteresis that prevents rapid toggling.

| Symptom | Direction |
|---|---|
| Starts reacting too late, car is almost on the cone | increase `AVOIDANCE_TRIGGER_M` → 6.5–7.0 |
| Triggers on cones clearly off to the side | decrease `PATH_WIDTH_M` → 0.7 |
| Toggles in/out rapidly after passing a cone | increase `AVOIDANCE_RELEASE_M` → 8.0 |
| Stays in avoidance mode long after cones are gone | decrease `AVOIDANCE_RELEASE_M` → 5.5, decrease `RETURN_BAND_M` → 0.10 |

---

### Return-to-centre condition

```python
RETURN_BAND_M = 0.20
```

Avoidance only exits when the **actual** lateral deviation (from Segformer) is within this band. Prevents premature release while the car is still offset.

| Symptom | Direction |
|---|---|
| Never exits avoidance even on clear road | increase → 0.35 |
| Exits too early, re-triggers immediately | decrease → 0.10 |

---

### Gap waypoint distance

```python
GAP_LOOKAHEAD_M = 1.8
```

The Z distance of the synthetic waypoint the car aims at during avoidance. Shorter = tighter, more aggressive turn into the gap.

| Symptom | Direction |
|---|---|
| Not steering hard enough, drifts through cones | decrease → 1.2–1.4 |
| Overshoots gap, oscillates through it | increase → 2.5–3.0 |

Do not go below 1.0 — below that the Pure Pursuit formula produces near-maximum steer for any non-zero offset.

---

### Gap geometry

```python
GAP_CAR_WIDTH_M     = 1.20   # your physical vehicle width — measure this
GAP_CONE_RADIUS_M   = 0.15   # cone exclusion zone radius
GAP_CENTER_WEIGHT   = 0.40   # score penalty per metre from lane centre
LANE_MARGIN_M       = 0.10   # min distance from grass edge for gap target
```

`GAP_CAR_WIDTH_M` must be measured physically. It doesn't gate gap selection (no gaps are rejected) but is used for scoring — wider gaps score proportionally higher.

`GAP_CENTER_WEIGHT`: higher values push the gap selection toward the lane centre when multiple gaps exist.

| Symptom | Direction |
|---|---|
| Always picks the side gap instead of threading through slot | increase `GAP_CENTER_WEIGHT` → 0.70 |
| Always goes centre even when side is clearly better | decrease `GAP_CENTER_WEIGHT` → 0.20 |
| Gap target drifts to grass edge | increase `LANE_MARGIN_M` → 0.20 |

---

### Avoidance speed

```python
SPEED_AVOID_KMH = 1.0
```

Speed setpoint sent to Nucleo during active cone avoidance. Lower gives more time per metre for the planner to converge.

---

### Cone detection

```python
CONE_CONF_THRESH  = 0.40    # YOLO confidence threshold
CONE_SKIP_FRAMES  = 2       # run YOLO every N frames
CONE_Z_MIN_M      = 0.3     # reject cones closer than this (noise)
CONE_Z_MAX_M      = 8.0     # reject cones further than this
CONE_DEPTH_PAD    = 6       # depth patch half-size in pixels
```

| Symptom | Direction |
|---|---|
| Misses some cones | decrease `CONE_CONF_THRESH` → 0.30 |
| False positive cones (triggers on track markings) | increase `CONE_CONF_THRESH` → 0.55 |
| Cone positions are noisy / jumping | increase `CONE_DEPTH_PAD` → 10 |
| YOLO too slow, drop rate is high | increase `CONE_SKIP_FRAMES` → 3 |

---

## Phase 3 — Stop sign

### Enable / disable

```python
STOP_SIGN_ENABLED = True
```

Set `False` to disable entirely — no YOLO thread, no GPU budget, no sign-triggered brake.

---

### Brake distance

```python
STOP_BRAKE_DIST_M = 3.5
```

Brake fires when a confirmed sign is within this distance. At 1–3 km/h, 3.5 m gives comfortable stopping margin.

| Symptom | Direction |
|---|---|
| Brakes too abruptly, not enough room | increase → 4.5–5.0 |
| Brakes too far from sign, stops unnecessarily early | decrease → 2.5 |

---

### Detection quality

```python
SIGN_CONF_THRESH   = 0.60    # YOLO confidence threshold
SIGN_VOTE_NEEDED   = 3       # consecutive positive frames to confirm
SIGN_SKIP_FRAMES   = 3       # run YOLO every N frames
SIGN_DIST_MIN_M    = 0.5
SIGN_DIST_MAX_M    = 15.0
SIGN_BBOX_MIN_FRAC = 0.35    # bbox must be ≥ this fraction of expected size
```

| Symptom | Direction |
|---|---|
| Misses the sign | decrease `SIGN_CONF_THRESH` → 0.45, decrease `SIGN_VOTE_NEEDED` → 2 |
| False positives (brakes on random objects) | increase `SIGN_CONF_THRESH` → 0.72, increase `SIGN_VOTE_NEEDED` → 5 |
| Detects at wrong distances | adjust `SIGN_DIST_MIN_M` / `SIGN_DIST_MAX_M` |
| Tiny bbox triggers false positive at claimed close distance | increase `SIGN_BBOX_MIN_FRAC` → 0.50 |

---

### Stop-line detection (orange stripe)

```python
STOP_VOTE_NEEDED    = 5      # consecutive positive frames
STOP_COVERAGE_MIN   = 0.60   # orange must cover this fraction of lane interior
STOP_WIDTH_MIN_FRAC = 0.70   # stripe must span this fraction of lane width
STOP_ORANGE_H_MIN   = 5      # HSV hue lower bound
STOP_ORANGE_H_MAX   = 20     # HSV hue upper bound
STOP_ORANGE_S_MIN   = 150    # min saturation
STOP_ORANGE_V_MIN   = 100    # min value
```

| Symptom | Direction |
|---|---|
| Misses the stop line | decrease `STOP_COVERAGE_MIN` → 0.45, widen hue range, lower `STOP_ORANGE_S_MIN` → 120 |
| False triggers on orange cones / track markings | increase `STOP_WIDTH_MIN_FRAC` → 0.80, increase `STOP_VOTE_NEEDED` → 8 |
| Triggers under shadow | lower `STOP_ORANGE_V_MIN` → 70 |

---

## EMA chain overview

Every control signal passes through multiple EMA stages. This is intentional — each stage targets a different noise source.

```
Segformer polynomial fit
        ↓  SEG_CENTERLINE_ALPHA = 0.30   (frame-to-frame mask noise)
Centerline polynomial (smoothed)
        ↓
compute_heading / compute_curvature
        ↓  CTRL_HEADING_ALPHA   = 0.20   (derivative amplifies HF noise)
        ↓  CTRL_CURVATURE_ALPHA = 0.15   (extra smooth — used for speed selection)
Heading angle, curvature (smoothed)
        ↓
Pure Pursuit → raw_deg
        ↓  STEER_RATE_LIMIT_DEG = 5.0   (hard cap per frame — blocks bad frames)
Rate-limited raw_deg
        ↓  STEER_EMA_ALPHA      = 0.20   (mechanical jitter, residual noise)
Final steer_deg → UART byte
```

**General rule:** if the problem is at a specific stage, tune the alpha for that stage. If it affects multiple stages, reduce `STEER_RATE_LIMIT_DEG` first — it's the most aggressive filter.

Lowering all alphas simultaneously makes the car sluggish. Lower only the stage that is noisy.

---

## Speed setpoints

```python
SPEED_TARGET_STRAIGHT_KMH = 3.0
SPEED_TARGET_CURVE_KMH    = 3.0
SPEED_CURVE_THRESH        = 0.15   # |κ| above which curve speed applies
SPEED_AVOID_KMH           = 1.0
```

Currently straight and curve speed are equal. Separate them once lane following is stable:

```python
SPEED_TARGET_STRAIGHT_KMH = 4.0
SPEED_TARGET_CURVE_KMH    = 2.5
SPEED_CURVE_THRESH        = 0.10   # lower threshold = slows sooner entering curves
```

Speed priority: `AVOIDING (1.0) < CURVE (2.5) < STRAIGHT (4.0)` — avoidance always wins.

---

## Point-cloud and depth cache

```python
PC_REFRESH_EVERY = 4    # full XYZ retrieved at most every N frames
```

At 30 fps, every 4 frames = 133 ms stale max. At 3 km/h the car moves ~11 cm between refreshes — acceptable for vote-gated stop decisions.

Lowering to 1 retrieves XYZ every frame but adds ~3 ms per frame of DMA cost.

---

## Profiling

```python
PROFILE_ENABLED     = True
PROFILE_PRINT_EVERY = 30
```

With `PROFILE_ENABLED = True`, every 30 frames the pipeline prints per-step average timing:

```
[Profile] avg over 30 frames  (total=48.3 ms → 20.7 fps est.)
  grab+retrieve          12.1 ms
  segformer_lane         28.4 ms
  stop_line               4.2 ms
  control                 2.1 ms
  sign_detect             1.5 ms
```

If `segformer_lane` dominates: try TensorRT engine (`SEG_ENGINE_PATH`).  
If `grab+retrieve` is high: `PC_REFRESH_EVERY` may be too low.

Set `PROFILE_ENABLED = False` for race — removes print overhead.

---

## Quick reference table

| Parameter | Default | Effect of increasing |
|---|---|---|
| `CTRL_LOOKAHEAD_M` | 4.0 | smoother, slower to react |
| `HEADING_FF_GAIN` | 0.40 | pre-steers earlier into curves |
| `CTRL_LANE_DEADBAND_M` | 0.05 | ignores more drift near centre |
| `STEER_EMA_ALPHA` | 0.20 | more responsive, less smooth |
| `STEER_RATE_LIMIT_DEG` | 5.0 | allows faster steering changes |
| `STEER_TX_DEADBAND_DEG` | 1.5 | less UART traffic, more latency |
| `CTRL_HEADING_ALPHA` | 0.20 | heading reacts faster, more noise |
| `CTRL_CURVATURE_ALPHA` | 0.15 | curvature reacts faster, more noise |
| `SEG_CENTERLINE_ALPHA` | 0.30 | polynomial updates faster |
| `SEG_CONF_THRESHOLD` | 0.35 | accepts weaker Segformer fits |
| `LOST_BRAKE_FRAMES` | 15 | tolerates longer LOST periods |
| `AVOIDANCE_TRIGGER_M` | 5.0 | reacts to cones from further away |
| `AVOIDANCE_RELEASE_M` | 6.5 | holds avoidance mode longer |
| `RETURN_BAND_M` | 0.20 | easier to exit avoidance |
| `GAP_LOOKAHEAD_M` | 1.8 | gentler turn-in toward gap |
| `GAP_CENTER_WEIGHT` | 0.40 | stronger preference for centred gaps |
| `LANE_MARGIN_M` | 0.10 | keeps gap target further from edge |
| `SPEED_AVOID_KMH` | 1.0 | faster during avoidance |
| `STOP_BRAKE_DIST_M` | 3.5 | brakes earlier from sign/line |
| `SIGN_CONF_THRESH` | 0.60 | accepts weaker sign detections |
| `SIGN_VOTE_NEEDED` | 3 | requires more frames to confirm |
| `GRASS_FRAC_THRESH` | 0.50 | trims grass boundary more readily |
| `GRASS_MAX_TRIM_FRAC` | 0.20 | allows wider grass trim |
