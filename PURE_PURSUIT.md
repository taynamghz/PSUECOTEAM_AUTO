# Pure Pursuit Controller — Testing & Tuning Guide

**PSU Eco Racing — Shell Eco-Marathon Autonomous Division**

---

## 1. What the Controller Does

The goal is **safe lane keeping**, not exact centerline tracking. The car stays comfortably within the lane on straights and turns correctly through curves. Small wandering in the middle of the lane is ignored — the controller only activates when the car drifts meaningfully toward the edge.

---

## 2. How It Works — Full Pipeline

Every frame goes through this chain:

```
ZED 2i camera
  │
  ├─ CLAHE lighting normalisation
  │
  ├─ Segformer-B2 [background thread]
  │     → road mask → centerline polynomial x = f(y)
  │
  ├─ compute_lookahead()
  │     → evaluate polynomial at far row (SEG_FAR_FRAC = 0.65H)
  │     → read ZED depth (single-channel, cached every 4 frames) at that pixel
  │     → X_m = (cx_pixel - W/2) × Z_m / fx    [pinhole, real Z]
  │     → Z_m = actual ZED depth at that row     [handles slopes, tilts]
  │
  ├─ Lane deadband gate
  │     → if |X_m| < CTRL_LANE_DEADBAND_M (0.15m)  →  steer 0°, go straight
  │
  ├─ Pure Pursuit formula
  │     → delta = atan2(2 × L × X_m,  ld²)
  │     → ld = hypot(X_m, Z_m)
  │     → L = WHEELBASE_M (1.6m)
  │
  ├─ Rate limiter
  │     → max STEER_RATE_LIMIT_DEG (5°) change per frame
  │
  ├─ EMA smoother
  │     → STEER_EMA_ALPHA = 0.20
  │
  ├─ Hardware clamp
  │     → ±STEER_MAX_DEG (25°)
  │
  └─ TX deadband
        → only sends UART if angle changed > STEER_TX_DEADBAND_DEG (1.5°)
        → CMD_STEER byte (0=full-left, 127=centre, 255=full-right) → Nucleo
```

**Depth strategy:**
- `MEASURE.DEPTH` (Z only, 3.7 MB) — retrieved every 4 frames for steering
- `MEASURE.XYZ` (full 3D, 11 MB) — retrieved only when stop-sign or stop-line votes are active

---

## 3. The Formula

```
delta = atan2(2 × L × X_m,  ld²)
```

Derived from standard pure pursuit `delta = atan(2L sin(α) / ld)` with `sin(α) = X_m / ld`.

- **X_m** — lateral offset of road center at lookahead distance. Positive = road is to the right (vehicle is left of center). Negative = road is to the left.
- **ld** — true Euclidean distance to that point = `hypot(X_m, Z_m)`
- **L** — wheelbase in metres

The formula is proportional — small offsets give small angles, large offsets give large angles:

| X_m at lookahead | Steering angle (L=1.6m, Z=2.2m) |
|---|---|
| 5 cm | ~1.6° |
| 10 cm | ~3.2° |
| 15 cm (deadband edge) | ~4.7° |
| 25 cm | ~7.7° |
| 50 cm | ~14° |
| 80 cm | ~21° |
| 120 cm+ | clamped at 25° |

---

## 4. All Tuning Parameters

All parameters live in `perception_stack/config.py`. Only touch the ones listed here.

### Group 1 — Must Verify Before Any Test

| Parameter | Default | What it is |
|---|---|---|
| `WHEELBASE_M` | `1.6` | Axle-to-axle distance in metres. **Measure physically.** Wrong value = systematic over/under-steer on every curve. |
| `UART_ENABLED` | `True` | Set `False` for dry runs (no motor output). Always start here. |

### Group 2 — Primary Tuning (these matter most)

| Parameter | Default | Effect |
|---|---|---|
| `CTRL_LANE_DEADBAND_M` | `0.15` | Lateral tolerance in metres. Car ignores offsets smaller than this. **0.15 = ±15cm corridor**. Increase for more relaxed driving, decrease for tighter tracking. |
| `CTRL_LOOKAHEAD_M` | `2.2` | Assumed depth to the far row. Affects how strongly the formula responds. Larger = smoother but slower to react to curves. Smaller = tighter but more reactive. Calibrate once on your track. |
| `STEER_EMA_ALPHA` | `0.20` | How fast the steering responds. Lower = smoother/slower. Higher = more responsive but more chatter. Range: 0.10–0.35. |

### Group 3 — Smoothing & Safety

| Parameter | Default | Effect |
|---|---|---|
| `STEER_RATE_LIMIT_DEG` | `5.0` | Max steering change per frame at 30fps. Clips jolts from single bad Segformer frames. Lower = smoother but may slow curve entry. |
| `STEER_MAX_DEG` | `25.0` | Hardware clamp. Must match your servo's physical limit. |
| `STEER_TX_DEADBAND_DEG` | `1.5` | Only sends a new UART command if angle changed by this much. Reduces servo hunting on straights. |
| `PC_REFRESH_EVERY` | `4` | Depth map refresh rate in frames. Every 4 frames at 30fps = 7.5 Hz. Lower = fresher depth, more CPU. Higher = staler depth, less CPU. |

### Group 4 — Do Not Touch During Testing

| Parameter | Reason |
|---|---|
| `SEG_FAR_FRAC` | Defines which image row is the lookahead row. Changing this changes the geometry of the entire formula. |
| `WHEELBASE_M` | Verify once, then lock. |
| `STEER_MAX_DEG` | Hardware limit — never exceed the servo spec. |

---

## 5. Before the First Test — Checklist

Work through this in order. Do not skip steps.

### Step 0 — Measure wheelbase

Measure axle center to axle center in metres. Update `WHEELBASE_M`. This is the most important physical calibration.

### Step 1 — Dry run (no motor output)

Set `UART_ENABLED = False` in `config.py`. Run:

```bash
python -m perception_stack.main
```

Watch the console output:
```
Frame |    Source |  Dev(m) |   Steer | Spd | Tgt | Cmd | Stop | Sign
```

Walk slowly in front of the camera, side to side. Verify:
- `Source` shows `SEGFORMER` (not `LOST`)
- `Steer` is near `0.0` when you are centered
- `Steer` grows in magnitude as you move off-center
- `Steer` sign flips when you cross center

If `Source` is always `LOST` — Segformer is not detecting the road. Check lighting and model path.

### Step 2 — Verify steering sign (vehicle raised off ground)

Set `UART_ENABLED = True`. Raise the front of the vehicle so wheels spin freely.

Stand to the **right** of the camera (road center appears to the left from camera's perspective):
- `X_m` should be **negative**
- Front wheels should steer **left**

Stand to the **left** of the camera (road center appears to the right):
- `X_m` should be **positive**
- Front wheels should steer **right**

**If wheels turn the wrong direction:** open `commander.py`, find `_compute_steer()`, change:
```python
raw_rad = math.atan2(2.0 * WHEELBASE_M * X_m, ld * ld)
```
to:
```python
raw_rad = math.atan2(2.0 * WHEELBASE_M * (-X_m), ld * ld)
```

### Step 3 — Verify lookahead depth (calibration)

With the vehicle stationary and `UART_ENABLED = False`, temporarily add this print inside `_compute_steer` in `commander.py`:

```python
if result.lookahead_point is not None:
    X_m, Z_m = result.lookahead_point
    print(f"  lookahead  X={X_m:+.3f}m  Z={Z_m:.3f}m")
```

Place a mark on the ground directly ahead, exactly **2.2m** from the camera. Point the camera at it centered. `Z_m` should read close to **2.2**. If it reads consistently 1.8 or 2.6, update `CTRL_LOOKAHEAD_M` to match what ZED actually measures at that row.

Remove the print before live testing.

### Step 4 — Check lookahead is not always None

In the dry run console, if `Steer` is always exactly `0.0` even when you are far off center, the lookahead_point may be returning `None` constantly (falling back to deviation_m path).

Temporarily add:
```python
print("lookahead:", result.lookahead_point)
```

in `main.py` after `result, frame, fm = out`. If it is always `None`:
- ZED depth map is not returning valid values at the far row
- Try switching `CAM_DEPTH_MODE` from `PERFORMANCE` to `NEURAL` in config
- Or reduce `CTRL_LOOKAHEAD_M` to 1.8 (ZED is more reliable at shorter range)

---

## 6. Live Testing — Step by Step

### Phase 1 — Straight line (first drive)

Start with the most conservative settings:
```python
CTRL_LANE_DEADBAND_M = 0.20   # very relaxed
STEER_EMA_ALPHA      = 0.15   # smooth
STEER_RATE_LIMIT_DEG = 4.0    # gentle
SPEED_TARGET_STRAIGHT_KMH = 3.0
```

Drive a straight section. Expected: servo is quiet, car holds a comfortable straight line.

| What you see | Cause | Fix |
|---|---|---|
| Servo constantly active, small left-right movements | Deadband too tight | `CTRL_LANE_DEADBAND_M` 0.20 → 0.25 |
| Car slowly drifts to one side and stays | Camera off-center OR wrong steer sign | Check camera alignment; verify Step 2 sign test |
| Car oscillates left-right (weaving) | EMA too fast or rate limit too high | `STEER_EMA_ALPHA` ↓ to 0.12, `STEER_RATE_LIMIT_DEG` ↓ to 3.0 |
| Car barely corrects when drifting to edge | Deadband too wide | `CTRL_LANE_DEADBAND_M` ↓ to 0.12 |
| Sudden servo jolt, then normal | Bad Segformer frame getting through | `STEER_RATE_LIMIT_DEG` ↓ to 3.0 |

### Phase 2 — Curves

Once straights are stable, approach a curve.

| What you see | Cause | Fix |
|---|---|---|
| Car takes curve smoothly, stays in lane | Good — done | — |
| Car runs wide (exits on outside) | Lookahead too long, reacts too late | `CTRL_LOOKAHEAD_M` ↓ (2.2 → 1.8) |
| Car cuts inside the curve | Lookahead too short, over-corrects | `CTRL_LOOKAHEAD_M` ↑ (2.2 → 2.6) |
| Car oscillates through the curve | Rate limit or EMA too aggressive | `STEER_RATE_LIMIT_DEG` ↓ to 3.5, `STEER_EMA_ALPHA` ↓ to 0.15 |
| Car starts turning too late | EMA lag building up | `STEER_EMA_ALPHA` ↑ to 0.25 |
| Car can't complete sharp turn | Angle clamped at 25° — servo at limit | Verify servo can physically reach needed angle. Reduce speed into turn. |

### Phase 3 — Tighten or relax

Once stable on both straights and curves, adjust to your preferred balance:

**Relaxed (SEM race — eco focus, minimal servo activity):**
```python
CTRL_LANE_DEADBAND_M  = 0.18
STEER_EMA_ALPHA       = 0.15
STEER_RATE_LIMIT_DEG  = 4.0
STEER_TX_DEADBAND_DEG = 2.0
```

**Balanced (default — good for most tracks):**
```python
CTRL_LANE_DEADBAND_M  = 0.15
STEER_EMA_ALPHA       = 0.20
STEER_RATE_LIMIT_DEG  = 5.0
STEER_TX_DEADBAND_DEG = 1.5
```

**Tighter (if you want more active centering):**
```python
CTRL_LANE_DEADBAND_M  = 0.08
STEER_EMA_ALPHA       = 0.25
STEER_RATE_LIMIT_DEG  = 6.0
STEER_TX_DEADBAND_DEG = 1.0
```

---

## 7. Quick Symptom → Fix Table

| Symptom | Most Likely Cause | Fix |
|---|---|---|
| Servo always quiet, car drifts to edge | Deadband too wide | `CTRL_LANE_DEADBAND_M` ↓ |
| Servo constantly active on straight | Deadband too tight | `CTRL_LANE_DEADBAND_M` ↑ |
| Car drifts permanently to one side | Wrong steer sign or camera off-center | Re-run Step 2 sign test; check mount |
| Runs wide on every curve | Lookahead too long | `CTRL_LOOKAHEAD_M` ↓ |
| Cuts inside every curve | Lookahead too short | `CTRL_LOOKAHEAD_M` ↑ |
| Oscillates left-right on straight | EMA too fast | `STEER_EMA_ALPHA` ↓ |
| Slow to enter curves | EMA too slow | `STEER_EMA_ALPHA` ↑ |
| Sudden single jolt then normal | Bad Segformer frame | `STEER_RATE_LIMIT_DEG` ↓ |
| Steering computed but nothing sent to servo | TX deadband too wide | `STEER_TX_DEADBAND_DEG` ↓ |
| Steer always 0° even far off-center | lookahead_point always None | See Step 4 checklist |
| Over-steer on all curves systematically | Wheelbase too small | Measure and update `WHEELBASE_M` |
| Under-steer on all curves systematically | Wheelbase too large | Measure and update `WHEELBASE_M` |
| FPS drops on straights | `PC_REFRESH_EVERY` too low | Raise to 6 or 8 |

---

## 8. Telemetry — Reading the Logs

With `LOG_TELEMETRY = True`, each run writes a `.jsonl` file to `logs/`. Each line is one frame:

```json
{
  "t": 1.234,         // time since start (seconds)
  "f": 37,            // frame number
  "dev": 0.032,       // lateral deviation in metres (near point)
  "conf": 0.87,       // Segformer confidence (0–1)
  "src": "SEGFORMER", // detection source
  "head": 0.021,      // heading angle (radians)
  "curv": 0.004,      // road curvature (m⁻¹)
  "steer_deg": 3.2,   // steering angle sent (degrees)
  "speed_kmh": 3.0,   // measured vehicle speed
  "target_kmh": 3.0,  // commanded speed
  "stop_line": false,
  "sl_dist": 0.0,
  "stop_sign": false,
  "ss_dist": 0.0,
  "cmd": "RUN"
}
```

**What to look for after a test run:**

- `steer_deg` on straights — should be near 0 most of the time. If frequently ±3–8° you have oscillation.
- `conf` — should be > 0.5 on clean road. Frequent drops below 0.3 = Segformer struggling with lighting/surface.
- `src` — `LOST` events during driving = road detection failures. Count them.
- `dev` — the lateral error trend. Steady drift in one direction = camera alignment issue.
- `cmd` — `BRAKE` should only appear near stop-line/stop-sign. Unexpected `BRAKE` = false positive detection.

---

## 9. Profiling — Checking FPS

With `PROFILE_ENABLED = True`, every 30 frames the console prints timing per stage:

```
[Profile] avg over 30 frames  (total=47.3 ms → 21.1 fps est.)
  segformer_lane         31.2 ms   ← Segformer inference (background thread, doesn't block)
  grab+retrieve          10.4 ms   ← Camera grab + depth retrieval
  stop_line               3.1 ms
  sign_detect             1.8 ms
  control                 0.8 ms
```

If total is above 50ms (< 20 fps):
- Increase `SEG_SKIP_STRAIGHT` to 2 or 3 — Segformer runs less often on straights
- Increase `PC_REFRESH_EVERY` to 6 — depth retrieved less often
- Switch `CAM_DEPTH_MODE` to `PERFORMANCE` (already default)

---

## 10. Fallback Path

When the ZED depth map has no valid pixels in the patch around the far-row centerline, `compute_lookahead()` returns `None`. The controller falls back to:

```python
dev = result.deviation_m          # lateral offset at near point (~1m ahead)
raw_rad = atan2(2×L×dev, ld²)    # same formula, assumed depth for ld
```

This is less accurate geometrically but keeps the car correcting. The lane deadband applies here too.

**If fallback triggers often** (you see `lookahead_point` is frequently `None`):
- ZED depth is unreliable at the far row distance
- Lower `CTRL_LOOKAHEAD_M` to 1.8m — depth is better at shorter range
- Or switch to `CAM_DEPTH_MODE = sl.DEPTH_MODE.NEURAL` for better depth quality

---

*PSU Eco Racing — Shell Eco-Marathon Autonomous Division*
