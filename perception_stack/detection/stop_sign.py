"""
PSU Eco Racing — Perception Stack
detection/stop_sign.py  |  Threaded YOLOv8 stop-sign detector.

Architecture:
  - submit(frame, pc, H, W) — called by the main pipeline every frame.
    Non-blocking: posts the latest frame to a maxsize-1 queue and returns
    immediately.  Stale frames are dropped so the worker always processes fresh.
  - get_result() — returns the latest confirmed (detected, dist_m, bbox).
    Non-blocking read; safe to call every frame from the main thread.

The worker thread:
  1. Pulls the latest frame from the queue.
  2. Runs YOLO inference every SIGN_SKIP_FRAMES frames (skips between runs to
     keep GPU load in budget; cached result carries through skipped frames).
  3. Applies distance gate via ZED point cloud.
  4. Runs the temporal vote gate (SIGN_VOTE_NEEDED consecutive detections).
  5. Writes confirmed result to _result under a lock.

This means YOLO inference (50–100 ms on Jetson Nano with .pt, ~20 ms with
TensorRT .engine) never blocks the main camera/lane thread.
"""

import queue
import threading
import numpy as np
from typing import Tuple, Optional

from perception_stack.config import (
    SIGN_MODEL_PATH,
    SIGN_CONF_THRESH,
    SIGN_IMG_SIZE,
    SIGN_ACCEPT_CLASSES,
    SIGN_SKIP_FRAMES,
    SIGN_DIST_MIN_M,
    SIGN_DIST_MAX_M,
    SIGN_VOTE_NEEDED,
    SIGN_FY_APPROX, SIGN_HEIGHT_M, SIGN_BBOX_MIN_FRAC,
)

_Result = Tuple[bool, float, Optional[Tuple[int, int, int, int]]]


class StopSignDetector:

    def __init__(self):
        import os
        from ultralytics import YOLO

        # ── Model selection ───────────────────────────────────────────────
        # Priority 1: custom-trained TRT engine or .pt on disk
        # Priority 2: YOLOv8n pretrained on COCO (class 11 = stop sign)
        #             ultralytics auto-downloads ~6 MB on first run
        if os.path.isfile(SIGN_MODEL_PATH):
            model_path           = SIGN_MODEL_PATH
            self._accept_classes = SIGN_ACCEPT_CLASSES
            print(f"[StopSign] Loading custom model {model_path}")
        else:
            model_path           = "yolov8n.pt"   # COCO pretrained fallback
            self._accept_classes = {11}            # COCO class 11 = stop sign
            print(f"[StopSign] '{SIGN_MODEL_PATH}' not found — "
                  f"falling back to YOLOv8n COCO (class 11 = stop sign).\n"
                  f"           Train a custom model: python scripts/train_stop_sign.py --api-key KEY")

        try:
            self._model   = YOLO(model_path)
            self._enabled = True
        except Exception as e:
            print(f"[StopSign] Model load failed ({e}) — sign detection disabled")
            self._enabled = False
            return

        # Warm-up: try CUDA first, fall back to CPU on OOM
        _dummy = np.zeros((SIGN_IMG_SIZE, SIGN_IMG_SIZE, 3), dtype=np.uint8)
        try:
            self._model.predict(_dummy, imgsz=SIGN_IMG_SIZE, verbose=False, device=0, half=True)
            self._device = 0
            self._half   = True
            print("[StopSign] Running on CUDA (FP16)")
        except Exception as e:
            print(f"[StopSign] CUDA unavailable ({e.__class__.__name__}), falling back to CPU")
            self._model.predict(_dummy, imgsz=SIGN_IMG_SIZE, verbose=False, device="cpu")
            self._device = "cpu"
            self._half   = False

        # Thread-safe result store
        self._result: _Result = (False, 0.0, None)
        self._lock = threading.Lock()

        # Frame queue: maxsize=1 so worker always processes the latest frame
        self._queue: queue.Queue = queue.Queue(maxsize=1)

        # Vote gate state (lives in worker thread — no lock needed)
        self._votes:      int            = 0
        self._last_dist:  float          = 0.0
        self._last_bbox:  Optional[tuple] = None

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    # ── Public API ─────────────────────────────────────────────────────────────

    def submit(self, frame: np.ndarray, pc: np.ndarray, H: int, W: int) -> None:
        """
        Post the latest frame for processing.  Non-blocking — drops the pending
        frame if the worker hasn't consumed it yet (always keep the freshest).
        frame is already a .copy() from the pipeline; pc is a ZED SDK view so
        we copy it here before handing off to the worker thread.
        """
        if not self._enabled:
            return
        try:
            self._queue.put_nowait((frame, pc.copy(), H, W))
        except queue.Full:
            pass  # worker is busy; this frame is dropped, next will be submitted

    def get_result(self) -> _Result:
        """Non-blocking read of the latest confirmed detection."""
        if not self._enabled:
            return (False, 0.0, None)
        with self._lock:
            return self._result

    # ── Worker ─────────────────────────────────────────────────────────────────

    def _worker(self) -> None:
        frame_idx = 0
        MAX_VOTES = SIGN_VOTE_NEEDED + 5

        while True:
            frame, pc, H, W = self._queue.get()   # blocks until main submits
            frame_idx += 1

            # Skip YOLO on non-scheduled frames — GPU budget control
            if frame_idx % SIGN_SKIP_FRAMES != 0:
                continue

            raw_detected, dist_m, bbox = self._run_yolo(frame, pc, H, W)

            # Vote gate
            if raw_detected:
                self._votes = min(MAX_VOTES, self._votes + 1)
                if dist_m > 0:
                    self._last_dist = dist_m
                if bbox is not None:
                    self._last_bbox = bbox
            else:
                self._votes = max(0, self._votes - 1)

            confirmed = self._votes >= SIGN_VOTE_NEEDED
            with self._lock:
                self._result = (
                    confirmed,
                    self._last_dist if confirmed else 0.0,
                    self._last_bbox if confirmed else None,
                )

    def _run_yolo(
        self, frame: np.ndarray, pc: np.ndarray, H: int, W: int
    ) -> _Result:
        results = self._model.predict(
            frame,
            imgsz=SIGN_IMG_SIZE,
            conf=SIGN_CONF_THRESH,
            verbose=False,
            device=self._device,
            half=self._half,
        )

        best_conf = -1.0
        best_bbox: Optional[Tuple[int, int, int, int]] = None
        best_dist = 0.0

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls not in self._accept_classes:
                    continue
                conf = float(box.conf[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Distance from point cloud — sample patch around centroid
                r_patch = max(4, (y2 - y1) // 8)
                py0 = max(0, cy - r_patch)
                py1 = min(H, cy + r_patch)
                px0 = max(0, cx - r_patch)
                px1 = min(W, cx + r_patch)
                patch_z = pc[py0:py1, px0:px1, 2]      # Z (negative = ahead)
                finite  = patch_z[np.isfinite(patch_z)]
                finite  = finite[finite < 0]
                dist_m  = float(np.median(np.abs(finite))) if finite.size >= 4 else 0.0

                if dist_m > 0 and not (SIGN_DIST_MIN_M <= dist_m <= SIGN_DIST_MAX_M):
                    continue

                # ── Bbox height sanity check ──────────────────────────────────
                # At distance dist_m, the sign should subtend approximately
                # (SIGN_HEIGHT_M / dist_m) * SIGN_FY_APPROX pixels in height.
                # Detections that are far too small for their reported depth
                # are likely noise or unrelated objects.
                if dist_m > 0:
                    expected_h_px = (SIGN_HEIGHT_M / dist_m) * SIGN_FY_APPROX
                    if (y2 - y1) < SIGN_BBOX_MIN_FRAC * expected_h_px:
                        continue

                if conf > best_conf:
                    best_conf = conf
                    best_bbox = (x1, y1, x2 - x1, y2 - y1)
                    best_dist = dist_m

        return (best_bbox is not None), best_dist, best_bbox

