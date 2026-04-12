"""
PSU Eco Racing — Perception Stack
main.py  |  Entry point.  Run with:  python -m perception_stack.main
"""

import json
import os
import time
import cv2
from collections import deque

from perception_stack.config import DISPLAY, LOG_TELEMETRY, LOG_DIR, LANE_ENABLED
from perception_stack.perception.pipeline import LanePerception
from perception_stack.control.commander import Commander
from perception_stack.visualization import draw


class TelemetryLogger:
    """
    Writes one JSON line per frame to a .jsonl file.
    Enable / disable via LOG_TELEMETRY in config.py.
    Files land in LOG_DIR/run_YYYYMMDD_HHMMSS.jsonl.
    """

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        ts   = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(log_dir, f"run_{ts}.jsonl")
        self._f  = open(path, 'w')
        self._t0 = time.monotonic()
        print(f"[Telemetry] Logging to {path}")

    def log(self, frame_idx: int, result, cmd_state: str,
            steer_deg: float, target_kmh: float, speed_kmh: float) -> None:
        record = {
            't':          round(time.monotonic() - self._t0, 3),
            'f':          frame_idx,
            'dev':        round(result.deviation_m,    4),
            'conf':       round(result.confidence,     3),
            'src':        result.source,
            'head':       round(result.heading_angle,  4),
            'curv':       round(result.curvature,      4),
            'steer_deg':  round(steer_deg,             2),
            'speed_kmh':  round(speed_kmh,             2),
            'target_kmh': round(target_kmh,            1),
            'stop_line':  result.stop_line,
            'sl_dist':    round(result.stop_line_dist, 2),
            'stop_sign':  result.stop_sign,
            'ss_dist':    round(result.stop_sign_dist_m, 2),
            'cmd':        cmd_state,
        }
        self._f.write(json.dumps(record) + '\n')

    def close(self) -> None:
        self._f.flush()
        self._f.close()
        print("[Telemetry] Log closed.")


def main():
    perc = LanePerception()
    if not perc.init():
        exit(1)

    cmd = Commander()
    if not cmd.open():
        print("[main] WARNING: UART unavailable — continuing without control output")

    logger = TelemetryLogger(LOG_DIR) if LOG_TELEMETRY else None

    if DISPLAY:
        cv2.namedWindow("PSU Eco Racing", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("PSU Eco Racing", 1280, 720)

    print("\nPerception v5 — press Q to quit, I to toggle idle\n")
    if LANE_ENABLED:
        print(f"{'Frame':>6} | {'Source':>12} | {'Dev(m)':>8} | {'Steer':>7} | "
              f"{'Spd':>6} | {'Tgt':>6} | {'Cmd':>6} | Stop       | Sign")
        print("-" * 105)
    else:
        print(f"{'Frame':>6} | {'FPS':>6} | {'Sign':>8} | {'Dist(m)':>8} | {'Cmd':>8}")
        print("-" * 50)

    fc = 0
    _disp_counter = 0
    _fps_times: deque = deque(maxlen=30)
    _fps_val: float = 0.0

    try:
        while True:
            _t0 = time.perf_counter()
            out = perc.process()
            if out is None:
                continue

            result, frame, fm = out
            fc += 1

            # Rolling FPS over last 30 frames
            _fps_times.append(time.perf_counter() - _t0)
            if len(_fps_times) >= 2:
                _fps_val = 1.0 / (sum(_fps_times) / len(_fps_times))

            # Send commands to low-level controller
            control_cmd = cmd.update(result)

            if logger:
                logger.log(fc, result, control_cmd,
                           cmd.steer_deg, cmd.target_kmh, cmd.speed_kmh)

            if DISPLAY:
                _disp_counter += 1
                if _disp_counter % 3 == 0:   # ~10 fps display to reduce CPU load
                    vis = draw(frame, result, fm, perc.H, perc.W,
                               control_cmd, cmd.steer_deg,
                               cmd.target_kmh, cmd.speed_kmh, _fps_val)
                    cv2.imshow("PSU Eco Racing", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('i'):
                    cmd.idle_requested = not cmd.idle_requested
                    state = "IDLE" if cmd.idle_requested else "RESUMED"
                    print(f"[main] {'IDLE — motors stopped' if cmd.idle_requested else 'RESUMED — motors active'}")

            if fc % 30 == 0:
                if LANE_ENABLED:
                    cs = ("CENTER" if abs(result.deviation_m) < 0.1
                          else "LEFT" if result.deviation_m > 0 else "RIGHT")
                    stop_str = (f"STOP@{result.stop_line_dist:.1f}m"
                                if result.stop_line else "-")
                    sign_str = (f"SIGN@{result.stop_sign_dist_m:.2f}m"
                                if result.stop_sign else "-")
                    print(f"{fc:>6} | {result.source:>12} | "
                          f"{result.deviation_m:>+8.3f} | {cmd.steer_deg:>+6.1f}d | "
                          f"{cmd.speed_kmh:>5.1f}k | {cmd.target_kmh:>5.1f}k | "
                          f"{control_cmd:>6} | {stop_str:<10} | {sign_str}")
                else:
                    sign_str  = f"{result.stop_sign_dist_m:.2f}m" if result.stop_sign else "--"
                    detected  = "DETECTED" if result.stop_sign else "       -"
                    print(f"{fc:>6} | {_fps_val:>6.1f} | {detected:>8} | "
                          f"{sign_str:>8} | {control_cmd:>8}")

    finally:
        if logger:
            logger.close()
        cmd.close()
        perc.close()
        cv2.destroyAllWindows()
        print("Done")


if __name__ == "__main__":
    main()
