"""
PSU Eco Racing — Perception Stack
visualization.py  |  OpenCV debug overlay for the perception pipeline.
"""

import numpy as np
import cv2

from perception_stack.config import ROI_TOP_FRACTION, LANE_ENABLED, STEER_MAX_DEG
from perception_stack.models import PerceptionResult
from perception_stack.lane.fitting import eval_x

_STATE_COLOUR = {
    "RUN":   (0, 220, 0),
    "BRAKE": (0, 60, 255),
    "IDLE":  (120, 120, 120),
}

_SOURCE_COLOUR = {
    "SEGFORMER":   (0, 220, 0),      # full detection — bright green
    "SEG_PARTIAL": (0, 165, 255),    # one side only — orange
    "LOST":        (0, 0, 255),      # no detection   — red
    "DISABLED":    (120, 120, 120),
}


def draw(frame: np.ndarray, result: PerceptionResult,
         fm, H: int, W: int,
         cmd_state: str = "IDLE",
         steer_deg: float = 0.0,
         target_kmh: float = 0.0,
         speed_kmh: float = 0.0,
         fps: float = 0.0) -> np.ndarray:
    vis = frame.copy()

    if LANE_ENABLED:
        # ── Road mask contour (Segformer output) ──────────────────────────────
        if fm is not None:
            cnts, _ = cv2.findContours(fm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (0, 255, 255), 1)

        # ── Lane polynomial overlays ──────────────────────────────────────────
        lf, rf = result.left_fit, result.right_fit
        if lf is not None or rf is not None:
            ys  = np.arange(int(H * ROI_TOP_FRACTION), H, 4)
            ovl = np.zeros_like(vis)
            if lf is not None and rf is not None:
                lxs = np.clip(np.polyval(lf, ys).astype(int), 0, W - 1)
                rxs = np.clip(np.polyval(rf, ys).astype(int), 0, W - 1)
                pts_l = np.stack([lxs, ys], axis=1)
                pts_r = np.stack([rxs, ys], axis=1)[::-1]
                cv2.fillPoly(ovl, [np.vstack([pts_l, pts_r])], (0, 55, 0))
                mxs = (lxs + rxs) // 2
                for i in range(len(ys) - 1):
                    cv2.line(ovl, (mxs[i], ys[i]), (mxs[i+1], ys[i+1]),
                             (0, 255, 200), 2)
            if lf is not None:
                lxs = np.clip(np.polyval(lf, ys).astype(int), 0, W - 1)
                for i in range(len(ys) - 1):
                    cv2.line(ovl, (lxs[i], ys[i]), (lxs[i+1], ys[i+1]),
                             (255, 80, 0), 3)
            if rf is not None:
                rxs = np.clip(np.polyval(rf, ys).astype(int), 0, W - 1)
                for i in range(len(ys) - 1):
                    cv2.line(ovl, (rxs[i], ys[i]), (rxs[i+1], ys[i+1]),
                             (0, 80, 255), 3)
            cv2.addWeighted(vis, 1.0, ovl, 0.55, 0, vis)

        # ── Lane width arrow ──────────────────────────────────────────────────
        if result.lane_width_m > 0.1 and lf is not None and rf is not None:
            ay = int(H * 0.78)
            lx = int(np.clip(eval_x(lf, ay), 0, W - 1))
            rx = int(np.clip(eval_x(rf, ay), 0, W - 1))
            cv2.arrowedLine(vis, (lx, ay), (rx, ay), (0, 255, 255), 2, tipLength=0.03)
            cv2.arrowedLine(vis, (rx, ay), (lx, ay), (0, 255, 255), 2, tipLength=0.03)
            cv2.putText(vis, f"{result.lane_width_m:.2f}m",
                        ((lx + rx) // 2 - 35, ay - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # ── Deviation bar ─────────────────────────────────────────────────────
        thresh = max(result.lane_width_m * 0.08, 0.05)
        bw, by = 500, H - 25
        bx = (W - bw) // 2
        cv2.rectangle(vis, (bx, by - 15), (bx + bw, by + 15), (40, 40, 40), -1)
        mid = bx + bw // 2
        cv2.line(vis, (mid, by - 15), (mid, by + 15), (255, 255, 255), 2)
        half = max(result.lane_width_m / 2.0, 0.3)
        dn   = np.clip(result.deviation_m / half, -1.0, 1.0)
        ind  = int(mid + dn * (bw // 2))
        col  = (0, 255, 0) if abs(result.deviation_m) < thresh else (0, 165, 255)
        cv2.circle(vis, (ind, by), 14, col, -1)
        cv2.putText(vis, "L", (bx - 20, by + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(vis, "R", (bx + bw + 5, by + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # ── Steering wheel indicator ──────────────────────────────────────────
        sw_cx, sw_cy, sw_r = W - 70, H - 70, 40
        cv2.circle(vis, (sw_cx, sw_cy), sw_r, (60, 60, 60), -1)
        cv2.circle(vis, (sw_cx, sw_cy), sw_r, (200, 200, 200), 2)
        # Needle rotates with steer_deg (positive = right)
        ang_rad = np.radians(-steer_deg)   # negate: right=clockwise on screen
        nx = int(sw_cx + (sw_r - 6) * np.sin(ang_rad))
        ny = int(sw_cy - (sw_r - 6) * np.cos(ang_rad))
        cv2.line(vis, (sw_cx, sw_cy), (nx, ny), (0, 255, 0), 3)
        steer_str = f"{steer_deg:+.1f}d"
        cv2.putText(vis, steer_str, (sw_cx - 28, sw_cy + sw_r + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # ── Road mask thumbnail (top-right corner) ────────────────────────────
        PW, PH = 200, 112
        px, py = W - PW - 6, 8
        if fm is not None:
            thumb = cv2.cvtColor(cv2.resize(fm, (PW, PH)), cv2.COLOR_GRAY2BGR)
        else:
            thumb = np.zeros((PH, PW, 3), dtype=np.uint8)
        vis[py:py + PH, px:px + PW] = thumb
        cv2.rectangle(vis, (px, py), (px + PW, py + PH), (80, 80, 80), 1)
        cv2.putText(vis, "ROAD MASK", (px + 4, py + PH - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # ── Cone detections ───────────────────────────────────────────────────────
    if result.cone_detections:
        avoiding = result.avoidance_state == "AVOIDING"
        box_col  = (0, 60, 255) if avoiding else (0, 165, 255)
        for x1, y1, x2, y2, conf in result.cone_detections:
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(vis, (ix1, iy1), (ix2, iy2), box_col, 2)
            cv2.putText(vis, f"{conf:.2f}", (ix1, iy1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_col, 2)
    if result.avoidance_state == "AVOIDING":
        av_label = "AVOIDING CONE"
        lsz = cv2.getTextSize(av_label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        tx = W // 2 - lsz[0] // 2
        cv2.rectangle(vis, (tx - 8, 95), (tx + lsz[0] + 8, 95 + lsz[1] + 12),
                      (0, 0, 180), -1)
        cv2.putText(vis, av_label, (tx, 95 + lsz[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 60, 255), 2)

    # ── Gap target waypoint ───────────────────────────────────────────────────
    if result.avoidance_state == "AVOIDING" and result.lookahead_pixel is not None:
        gx, gy = result.lookahead_pixel
        cv2.drawMarker(vis, (gx, gy), (0, 60, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.circle(vis, (gx, gy), 10, (0, 60, 255), 2)

    # ── Stop line ─────────────────────────────────────────────────────────────
    if result.stop_line and result.stop_line_y is not None:
        sy = result.stop_line_y
        cv2.line(vis, (0, sy), (W, sy), (0, 0, 255), 4)
        dist_str = f" {result.stop_line_dist:.1f}m" if result.stop_line_dist > 0 else ""
        label = f"STOP LINE{dist_str}"
        lsz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        tx = W // 2 - lsz[0] // 2
        cv2.rectangle(vis, (tx - 5, sy - lsz[1] - 14),
                      (tx + lsz[0] + 5, sy - 2), (0, 0, 160), -1)
        cv2.putText(vis, label, (tx, sy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # ── Stop sign bbox ────────────────────────────────────────────────────────
    if result.stop_sign_bbox is not None:
        sx, sy, sw, sh = result.stop_sign_bbox
        box_col = (0, 0, 255) if result.stop_sign else (0, 165, 255)
        cv2.rectangle(vis, (sx, sy), (sx + sw, sy + sh), box_col, 3)
        dist_str = f" {result.stop_sign_dist_m:.1f}m" if result.stop_sign_dist_m > 0 else ""
        label = f"STOP SIGN{dist_str}"
        lsz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        tx = sx + sw // 2 - lsz[0] // 2
        ty = max(sy - 10, lsz[1] + 4)
        cv2.rectangle(vis, (tx - 4, ty - lsz[1] - 4),
                      (tx + lsz[0] + 4, ty + 4), (0, 0, 160), -1)
        cv2.putText(vis, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # ── HUD (top bar) ─────────────────────────────────────────────────────────
    cv2.rectangle(vis, (0, 0), (W, 90), (0, 0, 0), -1)

    state_col = _STATE_COLOUR.get(cmd_state, (180, 180, 180))
    fps_col   = ((0, 255, 0) if fps >= 20 else
                 (0, 165, 255) if fps >= 10 else (0, 60, 255))
    fps_str = f"{fps:.1f} FPS"
    fps_sz  = cv2.getTextSize(fps_str, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)[0]

    if LANE_ENABLED:
        thresh = max(result.lane_width_m * 0.08, 0.05)
        cs     = ("CENTER" if abs(result.deviation_m) < thresh
                  else "LEFT" if result.deviation_m > 0 else "RIGHT")
        src_col = _SOURCE_COLOUR.get(result.source, (180, 180, 180))
        cv2.putText(vis,
            f"Dev: {result.deviation_m:+.3f}m  "
            f"Steer: {steer_deg:+.1f}d  "
            f"Spd: {speed_kmh:.1f}/{target_kmh:.0f}km/h  "
            f"Src: {result.source}  Conf: {result.confidence:.0%}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, src_col, 2)
        sign_info = (f"STOP SIGN: {result.stop_sign_dist_m:.2f}m  [CONFIRMED]"
                     if result.stop_sign else "Stop Sign: searching...")
        cv2.putText(vis, sign_info, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                    (0, 0, 255) if result.stop_sign else (100, 100, 100), 2)
    else:
        sign_info = (f"STOP SIGN DETECTED  {result.stop_sign_dist_m:.2f} m"
                     if result.stop_sign else "Stop Sign: searching...")
        sign_col = (0, 0, 255) if result.stop_sign else (100, 100, 100)
        cv2.putText(vis, sign_info, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, sign_col, 2)
        cv2.putText(vis, f"State: {cmd_state}",
                    (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_col, 2)

    cv2.putText(vis, fps_str, (W - fps_sz[0] - 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, fps_col, 2)

    # ── UART command panel ────────────────────────────────────────────────────
    steer_byte    = int(np.clip(127 + round(steer_deg / STEER_MAX_DEG * 127), 0, 255))
    throttle_byte = int(round(target_kmh * 10))

    if abs(steer_deg) < 2.0:
        steer_label = "STRAIGHT"
    elif steer_deg > 0:
        steer_label = f"RIGHT  {abs(steer_deg):.1f}deg"
    else:
        steer_label = f"LEFT   {abs(steer_deg):.1f}deg"

    braking = cmd_state == "BRAKE"
    panel_x, panel_y = 10, 95
    cv2.rectangle(vis, (panel_x - 4, panel_y - 4),
                  (panel_x + 460, panel_y + 62), (20, 20, 20), -1)
    cv2.rectangle(vis, (panel_x - 4, panel_y - 4),
                  (panel_x + 460, panel_y + 62), (80, 80, 80), 1)

    if braking:
        cv2.putText(vis, "UART  >>  BRAKE  (STOP LINE / SIGN)",
                    (panel_x, panel_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 60, 255), 2)
        cv2.putText(vis, "THROTTLE suspended",
                    (panel_x, panel_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)
    else:
        cv2.putText(vis,
            f"UART  >>  STEER    {steer_label:<18s}  [byte {steer_byte}]",
            (panel_x, panel_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2)
        cv2.putText(vis,
            f"UART  >>  THROTTLE {target_kmh:.1f} km/h              [byte {throttle_byte}]",
            (panel_x, panel_y + 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)

    return vis
