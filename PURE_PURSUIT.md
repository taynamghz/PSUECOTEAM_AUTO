# Pure Pursuit Controller — Full Explanation & Tuning Guide

**PSU Eco Racing — Shell Eco-Marathon Autonomous Division**

---

## 1. The Core Idea

Pure Pursuit asks one simple question every frame:

> **"If I draw a straight line from the car to a point on the road ahead, what angle do I need to steer to follow that line?"**

That point ahead is called the **lookahead point**. The further ahead you look, the smoother but slower the response. The closer you look, the more reactive but twitchier the steering.

```
         lookahead point (2.2m ahead on centerline)
              ●
             /
            /  ← steer angle = atan(deviation / lookahead_distance)
           /
          ● ← car position (camera center)
```

That's literally the entire algorithm. Everything else in the code is perception, smoothing, and safety.

---

## 2. How It Flows Frame by Frame

```
Camera frame
    ↓
CLAHE lighting normalisation
    ↓
Segformer ONNX → road mask (True/False per pixel)
    ↓
Scan road pixels row by row → find centerline
    ↓
Fit quadratic polynomial to centerline
    ↓
Evaluate at 85% of frame height → deviation_m (how far left/right of center)
Evaluate polynomial slope at 72% → heading_angle (which way road is turning)
    ↓
Pure Pursuit formula:
    raw = atan(deviation_m / 2.2) − heading_angle
    ↓
Deadband → magnitude scaling → EMA smoothing
    ↓
UART byte → Nucleo → servo
```

---

## 3. The Pure Pursuit Formula in Detail

From `commander.py`:

```python
raw_rad = math.atan(dev / max(CTRL_LOOKAHEAD_M, 0.1)) - result.heading_angle
```

### Part 1: `atan(dev / lookahead)`

`dev` = how many metres the car is from the road center (positive = car is left of center).
`CTRL_LOOKAHEAD_M = 2.2` = how far ahead on the road you are aiming at.

Think of it as a triangle:
- The base is `dev` — how far sideways you are
- The height is `2.2m` — how far ahead the target is
- The angle at the bottom is what you need to steer

```
         target
           ●
           |
    2.2m   |
           |
   car ●───┘  dev = 0.3m

   steer angle = atan(0.3 / 2.2) = 7.8°
```

If `dev = 0` (perfectly centered), `atan(0) = 0` — steer straight.
If `dev = 0.5m` off center with 2.2m lookahead → steer `atan(0.5/2.2) = 12.8°`.

Clean, intuitive, no magic.

### Part 2: `- result.heading_angle`

`heading_angle` is the polynomial slope at the lookahead row — it tells you which direction the road is curving ahead. Subtracting it means: if the road curves left, pre-steer left **before you even drift**. This is the feedforward that makes turns smooth.

Without this term, the car waits until it physically drifts before correcting — always late into corners. With it, the car starts turning the moment it sees the road turning ahead.

---

## 4. What Happens After the Raw Angle

```python
# Step 1 — Deadband: ignore tiny corrections (noise)
if abs(raw_deg) < STEER_DEADBAND_DEG:   # 3°
    return 0.0

# Step 2 — Magnitude scaling: map error size to output size
t = (|raw_deg| - deadband) / (max_deg - deadband)   # 0.0 to 1.0
magnitude = STEER_MIN_DEG + t × (STEER_RATE_DEG - STEER_MIN_DEG)
output = sign(raw_deg) × magnitude

# Step 3 — EMA smoothing (in commander.update)
steer_ema = STEER_EMA_ALPHA × output + (1 - STEER_EMA_ALPHA) × previous_ema
```

**Step 1 — Deadband:**
Corrections under 3° are road mask noise. Zeroing them out stops the servo jittering constantly on straights.

**Step 2 — Magnitude scaling:**
Instead of sending the raw angle directly, it maps the error size onto a range from `STEER_MIN_DEG` to `STEER_RATE_DEG`:
- Small error → gentle correction near `1°`
- Large error → sharp correction up to `8°`

The **direction** always comes from Pure Pursuit. Only the **magnitude** is scaled. This prevents small road mask noise from causing large servo commands.

**Step 3 — EMA:**
With `STEER_EMA_ALPHA = 0.10`, the output is very heavily smoothed. Each new frame contributes only 10%, the previous history contributes 90%. Steering changes gradually — never suddenly.

---

## 5. Every Config Parameter Explained

### `CTRL_LOOKAHEAD_M = 2.2`
**The most important parameter.**
How far ahead on the road the car is aiming at, in metres.
- **Larger (3.0m+):** Smoother, slower to react — good for straights and wide gentle curves
- **Smaller (1.5m):** Reacts faster, tighter line through curves — risks oscillation

---

### `SPEED_TARGET_STRAIGHT_KMH = 3.0` / `SPEED_TARGET_CURVE_KMH = 3.0`
Target speed sent to the Nucleo in km/h. Both are currently identical — the car runs at 3 km/h everywhere. When you're confident in the lane keeping, set `SPEED_TARGET_STRAIGHT_KMH` higher and keep `SPEED_TARGET_CURVE_KMH` conservative.

### `SPEED_CURVE_THRESH = 0.15`
When the measured road curvature exceeds 0.15 m⁻¹ (a real turn), the car switches from straight speed to curve speed. A value of 0.15 m⁻¹ corresponds to roughly a 6.5m radius corner.

---

### `CTRL_LATERAL_DEADBAND_FRAC = 0.085`
The car ignores lateral error when it's within `8.5% × lane_width` of center. On a 1.5m track that's ±13cm. Inside this corridor only the heading term runs — the car follows the road direction without trying to correct tiny lateral drift. This prevents constant micro-corrections on straights.

### `CTRL_LATERAL_DEADBAND_M = 0.15`
Fallback corridor in metres when lane width measurement is unavailable. ±15cm.

---

### `STEER_MAX_DEG = 25.0`
Hardware clamp. The servo physically cannot go beyond ±25°. Any larger computed angle is clamped here.

### `STEER_MIN_DEG = 1.0`
Minimum output magnitude for any correction that makes it past the deadband. Ensures corrections have a soft, smooth entry rather than jumping from 0 to a large angle instantly.

### `STEER_DEADBAND_DEG = 3.0`
Corrections smaller than 3° are zeroed out completely. Mask noise, vibration, and minor polynomial jitter all produce sub-3° signals — this eliminates them.

### `STEER_RATE_DEG = 8.0`
Maximum correction output per update. Even if the car is severely off-center, the servo only gets commanded 8° per frame. Prevents aggressive overcorrection.

### `STEER_EMA_ALPHA = 0.10`
How much each new frame contributes to the smoothed steering output.
- `0.10` = very smooth, slow response (~20 frames to reach full correction)
- `0.25` = faster response, more reactive
- `0.40` = very responsive, risk of oscillation

### `STEER_TX_DEADBAND_DEG = 2.0`
The UART command is only transmitted when the steering setpoint changes by more than 2° from the last sent value. Prevents flooding the bus with redundant identical commands on straights.

---

### `SEG_NEAR_FRAC = 0.85`
Where on the image (as a fraction of frame height, from the top) the lateral deviation is measured. At 85% height the road is roughly 1–2m in front of the car — close, stable, reliable pixels.

### `SEG_FAR_FRAC = 0.65`
Where heading angle is evaluated. At 65% height the road is further ahead — gives early warning of upcoming curves.

### `SEG_FIT_TOP_FRAC = 0.72`
The polynomial is only fitted to rows **below** 72% of frame height. Rows above this are far away, pixels are small and noisy — fitting them causes the polynomial to blow up. 72% corresponds to roughly 2.5m lookahead.

### `SEG_CENTERLINE_ALPHA = 0.30`
The polynomial coefficients themselves are EMA-smoothed across frames. Each new frame contributes 30%, history contributes 70%. Makes the centerline estimate stable across noisy mask frames.

### `SEG_BOUNDARY_ROWS = 30`
Number of rows scanned per frame to build the centerline. 30 rows evenly distributed across the fitted zone. More rows = more stable polynomial, slightly more CPU.

### `SEG_SKIP_STRAIGHT = 1` / `SEG_SKIP_CURVE = 1`
How often a new frame is submitted to Segformer for inference. Both are 1 = every frame. Previously on straights this was 5 (every 5th frame) to save power, but it's been set to 1 for maximum freshness. On the Jetson with TensorRT this is fine. On CPU only, you may want to raise `SEG_SKIP_STRAIGHT` back to 3–5 to reduce thermal load.

---

## 6. The Single Most Important Thing to Understand

**Pure Pursuit has no speed compensation.**

At 3 km/h and `CTRL_LOOKAHEAD_M = 2.2`, the car looks 2.2m ahead and steers accordingly. If you run at 6 km/h with the same lookahead, the car covers that 2.2m in half the time — it reacts much later in the turn. Curves feel wider and later.

**Rule of thumb: if you double the speed, shorten the lookahead by ~20–30%.**

| Speed | Recommended Lookahead |
|---|---|
| 3 km/h | 2.0 – 2.5 m |
| 5 km/h | 1.8 – 2.2 m |
| 8 km/h | 1.5 – 1.8 m |
| 10 km/h | 1.3 – 1.6 m |

---

## 7. Tuning Guide — Step by Step

### Step 1: Straight-line stability first

Before touching curves, get the car to hold a straight line cleanly.

Set `CTRL_LOOKAHEAD_M = 2.2`, `STEER_EMA_ALPHA = 0.10`, drive straight.

| What you see | Cause | Fix |
|---|---|---|
| Car weaves left-right constantly | Deadband too small | `STEER_DEADBAND_DEG` 3 → 5 |
| Car drifts slowly to one side and stays | Camera not aligned or polynomial bias | Check camera mount is centered on vehicle axis |
| Car corrects but overshoots, oscillates | `STEER_RATE_DEG` too high | Lower to 5° |
| Car barely corrects, drifts far | `STEER_RATE_DEG` too low or EMA too slow | Raise to 12°, raise alpha to 0.20 |

---

### Step 2: Tune lookahead for curves

Drive through a known curve at your target speed.

| What you see | Cause | Fix |
|---|---|---|
| Car cuts inside the curve | Lookahead too short, over-steering | `CTRL_LOOKAHEAD_M` ↑ (2.2 → 2.8) |
| Car runs wide, exits the curve late | Lookahead too long, under-steering | `CTRL_LOOKAHEAD_M` ↓ (2.2 → 1.8) |
| Car oscillates through the curve | `STEER_RATE_DEG` too high | Lower to 6° |
| Car takes wide smooth arc, barely corrects | Heading term working, lateral correction weak | Lower `CTRL_LATERAL_DEADBAND_FRAC` (0.085 → 0.06) |

---

### Step 3: Tune responsiveness

`STEER_RATE_DEG` and `STEER_EMA_ALPHA` control how fast and aggressively the car responds.

```
Slow, smooth, stable   ←──────────────────→   Fast, reactive, risky of oscillation
STEER_RATE_DEG:  5°                                                          15°
STEER_EMA_ALPHA: 0.07                                                        0.35
```

- Start at `STEER_RATE_DEG = 8`, `STEER_EMA_ALPHA = 0.10`
- If too sluggish: raise both — `STEER_RATE_DEG = 12`, `STEER_EMA_ALPHA = 0.20`
- If oscillating: lower both — `STEER_RATE_DEG = 5`, `STEER_EMA_ALPHA = 0.07`

---

### Step 4: Increase speed

Once stable at 3 km/h, push the speed up in 1–2 km/h steps.

At each new speed:
1. If curves feel late and wide → reduce `CTRL_LOOKAHEAD_M` by 0.2m
2. If oscillation increases → lower `STEER_EMA_ALPHA` slightly
3. Differentiate speeds: set `SPEED_TARGET_STRAIGHT_KMH` higher, keep `SPEED_TARGET_CURVE_KMH` at 3.0

---

## 8. Quick Reference — Symptom to Fix

| Symptom | Most Likely Cause | Parameter |
|---|---|---|
| Weaves on straights | Deadband too small | `STEER_DEADBAND_DEG` ↑ |
| Drifts to one side permanently | Camera misalignment | Check mount |
| Slow to correct lateral drift | EMA too slow or rate too low | `STEER_EMA_ALPHA` ↑, `STEER_RATE_DEG` ↑ |
| Cuts inside every curve | Lookahead too short | `CTRL_LOOKAHEAD_M` ↑ |
| Runs wide on every curve | Lookahead too long | `CTRL_LOOKAHEAD_M` ↓ |
| Jerky/twitchy on curves | Rate too high | `STEER_RATE_DEG` ↓ |
| Barely steers into curves | Rate too low | `STEER_RATE_DEG` ↑ |
| Oscillates and overshoots | EMA too fast | `STEER_EMA_ALPHA` ↓ |
| Steering lags behind road | EMA too slow | `STEER_EMA_ALPHA` ↑ |
| Polynomial unstable far rows | Fit zone too wide | `SEG_FIT_TOP_FRAC` ↑ (0.72 → 0.80) |
| Jitters at standstill | TX deadband too small | `STEER_TX_DEADBAND_DEG` ↑ |

---

*PSU Eco Racing — Shell Eco-Marathon Autonomous Division*
