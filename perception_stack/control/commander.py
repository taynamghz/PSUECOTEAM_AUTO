"""
PSU Eco Racing — Perception Stack
control/commander.py  |  High-level command decision layer.

The Nucleo LLC runs PID control internally.  The Jetson sends only setpoints:

  CMD_THROTTLE  DATA = target speed in tenths of km/h  (150 → 15.0 km/h)
  CMD_BRAKE     DATA = brake intensity (emergency stop at stop-line / stop-sign)
  CMD_STEER     DATA = steering angle byte (0=full-left, 127=centre, 255=full-right)

Command update policy:
  CMD_THROTTLE — sent every frame as a speed setpoint (km/h × 10 as byte).
  CMD_STEER    — sent only when the new angle differs from the last transmitted
                 value by ≥ STEER_TX_DEADBAND_DEG (5°).  This suppresses rapid
                 micro-corrections caused by mask noise — the motor only moves
                 when a real steering change is needed.
  CMD_BRAKE    — sent immediately every frame when a stop-line or stop-sign is
                 within STOP_BRAKE_DIST_M.  Never gated by the steer dead-band.
"""

import math
import logging

from perception_stack.config import (
    UART_ENABLED,
    STOP_BRAKE_DIST_M, BRAKE_VALUE,
    SPEED_TARGET_STRAIGHT_KMH, SPEED_TARGET_CURVE_KMH, SPEED_CURVE_THRESH,
    SPEED_AVOID_KMH,
    WHEELBASE_M, CTRL_LOOKAHEAD_M, CTRL_LANE_DEADBAND_M,
    STEER_MAX_DEG, STEER_DEADBAND_DEG, STEER_RATE_LIMIT_DEG,
    STEER_EMA_ALPHA, STEER_TX_DEADBAND_DEG,
    HEADING_FF_GAIN,
)
from perception_stack.models import PerceptionResult
from perception_stack.control.uart import UARTController

log = logging.getLogger(__name__)


def _deg_to_steer_byte(deg: float) -> int:
    """Map ±STEER_MAX_DEG → [0, 255] with 127 = straight.
    positive deg = steer right → byte > 127
    negative deg = steer left  → byte < 127
    """
    return int(max(0, min(255, round(127.0 - deg * 127.0 / STEER_MAX_DEG))))


class Commander:
    """
    Usage:
        cmd = Commander()
        cmd.open()
        cmd.update(result)      # call every frame
        cmd.close()

    Public attributes (updated each frame):
        cmd.target_kmh   — speed setpoint sent to Nucleo (km/h)
        cmd.steer_deg    — steering angle sent to Nucleo (degrees)
        cmd.speed_kmh    — current speed received from Nucleo (km/h)
    """

    def __init__(self):
        self.uart = UARTController()

        self._state:          str   = "RUN"
        self.idle_requested:  bool  = False  # toggled externally by 'i' key

        # Steering state
        self._steer_ema:           float = 0.0
        self._last_raw_steer:      float = 0.0   # for rate limiter
        self._last_sent_steer_deg: float = 0.0

        # Public state (for display / telemetry)
        self.target_kmh: float = SPEED_TARGET_STRAIGHT_KMH
        self.steer_deg:  float = 0.0
        self.speed_kmh:  float = 0.0

    # ── Lifecycle ───────────────────────────────────────────────────────────────

    def open(self) -> bool:
        if not UART_ENABLED:
            log.info("[Commander] UART disabled — dry-run mode")
            return True
        ok = self.uart.open()
        if ok:
            self.uart.idle()
            self.uart.steer(127)   # centre steering on start-up
        return ok

    def close(self):
        self.uart.set_speed(0.0)   # tell Nucleo to stop
        self.uart.steer(127)       # return to centre
        self.uart.close()

    # ── Main update ─────────────────────────────────────────────────────────────

    def update(self, result: PerceptionResult) -> str:
        """
        Send UART setpoints for this frame.
        Returns "BRAKE" or "RUN".
        """
        # Read current speed from the LLC UART reader thread (non-blocking)
        self.speed_kmh = self.uart.speed_kmh

        # Manual idle — 'i' key toggles; hold until pressed again to resume
        if self.idle_requested:
            if UART_ENABLED:
                self.uart.set_speed(0.0)
                self.uart.steer(127)
            return "IDLE"

        brake  = self._should_brake(result)
        target = self._target_speed(result)

        raw_steer = self._compute_steer(result)

        # Rate limiter — cap how fast the commanded angle can change per frame.
        # Prevents a single bad Segformer frame from yanking the EMA significantly.
        delta = raw_steer - self._last_raw_steer
        if abs(delta) > STEER_RATE_LIMIT_DEG:
            raw_steer = self._last_raw_steer + math.copysign(STEER_RATE_LIMIT_DEG, delta)
        self._last_raw_steer = raw_steer

        # EMA — smooths mechanical jitter and residual frame-to-frame noise
        self._steer_ema = (STEER_EMA_ALPHA * raw_steer
                           + (1.0 - STEER_EMA_ALPHA) * self._steer_ema)
        steer = self._steer_ema

        self.target_kmh = target
        self.steer_deg  = steer
        state = "BRAKE" if brake else "RUN"

        if UART_ENABLED:
            if brake:
                self.uart.brake(BRAKE_VALUE)
            else:
                self.uart.set_speed(target)
                # TX deadband — only send when angle changed meaningfully
                if abs(steer - self._last_sent_steer_deg) >= STEER_TX_DEADBAND_DEG:
                    self.uart.steer(_deg_to_steer_byte(steer))
                    self._last_sent_steer_deg = steer

        if state != self._state:
            log.info("[Commander] %s → %s  src=%s  spd=%.1f km/h  steer=%.1f deg",
                     self._state, state, result.source, self.speed_kmh, steer)
            self._state = state

        return state

    # ── Brake decision ───────────────────────────────────────────────────────────

    @staticmethod
    def _should_brake(result: PerceptionResult) -> bool:
        if result.emergency_stop:
            return True
        if result.stop_line and 0 < result.stop_line_dist <= STOP_BRAKE_DIST_M:
            return True
        if result.stop_sign and 0 < result.stop_sign_dist_m <= STOP_BRAKE_DIST_M:
            return True
        return False

    # ── Target speed ─────────────────────────────────────────────────────────────

    @staticmethod
    def _target_speed(result: PerceptionResult) -> float:
        """Avoidance < curve < straight — slowest always wins during manoeuvring."""
        if result.avoidance_state == "AVOIDING":
            return SPEED_AVOID_KMH
        if abs(result.curvature) > SPEED_CURVE_THRESH:
            return SPEED_TARGET_CURVE_KMH
        return SPEED_TARGET_STRAIGHT_KMH

    # ── Pure Pursuit steering ────────────────────────────────────────────────────

    def _compute_steer(self, result: PerceptionResult) -> float:
        """
        Geometric Pure Pursuit:
            delta = atan2(2 * L * X_m,  ld²)

        Where:
            X_m = lateral offset of the 3D lookahead point (positive = right of camera)
            ld  = Euclidean distance to the lookahead point  = hypot(X_m, Z_m)
            L   = vehicle wheelbase (WHEELBASE_M)

        This is derived from the standard formula  delta = atan(2L sin(α) / ld)
        with α = atan2(X_m, Z_m),  sin(α) = X_m / ld  →  atan2(2L·X_m, ld²).

        Sign: X_m > 0 (lookahead right of camera) → steer right (+).
              Vehicle left of centre → road centre is to the right → X_m > 0. Correct.

        Fallback (no ZED depth at lookahead): approximate with near-point deviation.
        """
        # During active cone avoidance the gap waypoint is valid regardless of
        # Segformer confidence — skip the lane-quality guard so the car steers.
        avoiding = result.avoidance_state == "AVOIDING" and result.lookahead_point is not None
        if not avoiding:
            if result.source in ("DISABLED", "NONE", "LOST") or result.confidence < 0.15:
                return 0.0

        if result.lookahead_point is not None:
            X_m, Z_m = result.lookahead_point
            # Lane deadband — don't correct small wandering near centre.
            # Only steer when the car has drifted meaningfully toward the edge.
            if abs(X_m) < CTRL_LANE_DEADBAND_M:
                return 0.0
            ld = math.hypot(X_m, Z_m)
            if ld < 0.1:
                return 0.0
            raw_rad = math.atan2(2.0 * WHEELBASE_M * X_m, ld * ld)
        else:
            # ZED had no valid depth at the lookahead row — fall back to deviation.
            dev = result.deviation_m
            if abs(dev) < CTRL_LANE_DEADBAND_M:
                return 0.0
            ld  = CTRL_LOOKAHEAD_M
            raw_rad = math.atan2(2.0 * WHEELBASE_M * dev, ld * ld)

        raw_deg = math.degrees(raw_rad)

        # Heading feed-forward: pre-steer into curves before lateral deviation builds.
        # heading_angle > 0 means the lane curves left → we need left steer (negative).
        # This eliminates the "wait until drift" cycle that causes S-swerves on curves.
        ff_deg  = -math.degrees(result.heading_angle) * HEADING_FF_GAIN
        raw_deg += ff_deg

        # Hardware clamp — only limit at servo physical limit
        return max(-STEER_MAX_DEG, min(STEER_MAX_DEG, raw_deg))
