"""
PSU Eco Racing — Perception Stack
perception/segformer_lane.py  |  Drivable-area lane detection via Segformer-B2.

Replaces the RANSAC + colour-threshold lane detector.
Works without white lane markings — detects asphalt/grass boundaries directly.

Pipeline per frame:
  1. Run Segformer (Cityscapes) → road mask (H×W bool)
  2. Scan SEG_BOUNDARY_ROWS evenly-spaced rows in the lookahead window
     → at each row: road_center_x = (leftmost_road_px + rightmost_road_px) / 2
  3. Polyfit degree-2: x = f(y) to those center points → centerline polynomial
  4. Evaluate centerline at y_near for lateral deviation
  5. Convert px offset to metres using ZED point-cloud depth at eval row
  6. Return (cf, cf, road_mask, dev_m, wid_m, conf, conf, source)
     cf is the centerline polynomial — passed as both lf and rf so that
     downstream compute_heading / compute_curvature (which average lf+rf)
     receive the correct centerline without modification.

Threading model
───────────────
Inference runs in a dedicated background thread so the main camera loop
is never blocked.  The main thread calls:

    seg_lane.submit(frame_bgr, pc, H, W, fx)   # non-blocking — drops stale frames
    lf, rf, ... = seg_lane.get_result()         # instant — returns latest result

The result is 1 camera-frame stale on average (≈33 ms at 30 fps, ≈14 cm at 15 km/h).
That latency is negligible for lane-following at the speeds used in SEM.
"""

import os
import queue
import threading
import numpy as np
import cv2

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import tensorrt as trt
    import torch
    _TRT_AVAILABLE = True
except ImportError:
    trt   = None
    _TRT_AVAILABLE = False

if not _TRT_AVAILABLE:
    try:
        import torch
    except ImportError:
        torch = None

import torch.nn.functional as F

# ImageNet normalisation (matches SegformerImageProcessor defaults)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _ort_providers() -> list:
    available = ort.get_available_providers() if ort else []
    for ep in ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"):
        if ep in available:
            return [ep]
    return ["CPUExecutionProvider"]


class _TRTSession:
    """TensorRT inference via PyTorch CUDA tensors (no pycuda required)."""
    def __init__(self, engine_path: str):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self._engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self._context  = self._engine.create_execution_context()
        self._tensors  = {}
        self._bindings = []
        for i in range(self._engine.num_io_tensors):
            name  = self._engine.get_tensor_name(i)
            shape = tuple(self._engine.get_tensor_shape(name))
            dtype = trt.nptype(self._engine.get_tensor_dtype(name))
            td    = torch.float16 if dtype == np.float16 else torch.float32
            t     = torch.zeros(shape, dtype=td, device="cuda")
            self._tensors[name]  = t
            self._bindings.append(t.data_ptr())

    def run(self, input_array: np.ndarray) -> np.ndarray:
        name_in  = self._engine.get_tensor_name(0)
        name_out = self._engine.get_tensor_name(1)
        self._tensors[name_in].copy_(
            torch.from_numpy(input_array).to(
                dtype=self._tensors[name_in].dtype, device="cuda"))
        self._context.execute_v2(self._bindings)
        return self._tensors[name_out].cpu().numpy()


from perception_stack.config import (
    SEG_ENGINE_PATH, SEG_ONNX_PATH, SEG_INPUT_H, SEG_INPUT_W,
    SEG_MODEL_ID,
    SEG_ROAD_CLASSES,
    SEG_ROI_TOP_FRAC,
    SEG_MIN_ROAD_FRAC,
    SEG_BOUNDARY_ROWS,
    SEG_POLY_DEG,
    SEG_CONF_THRESHOLD,
    SEG_NEAR_FRAC,
    SEG_FAR_FRAC,
    SEG_FIT_TOP_FRAC,
    SEG_CENTERLINE_ALPHA,
)
from perception_stack.detection.road_validator import validate_boundaries


def _best_device() -> str:
    if torch.cuda.is_available():         return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


# Default result returned before the first inference completes
_NULL_RESULT = (None, None,
                None,   # road_mask — None signals "not ready yet"
                0.0, 0.0, 0.0, 0.0, "LOST")


class SegformerLane:
    """
    Wraps Segformer-B2 (Cityscapes) for drivable-area centerline detection.

    Deviation and curvature are computed from the road mask directly:
    scan the road-pixel center per row → fit a quadratic centerline.
    No left/right boundary polynomial fitting — avoids polynomial explosion
    when the road mask is wide or noisy.

    Public async interface (used by pipeline.py):
        init()         — load model, start worker thread
        submit(...)    — non-blocking: post latest frame to worker
        get_result()   — non-blocking: read latest computed result
    """

    def __init__(self):
        self._model      = None
        self._processor  = None
        self._session    = None   # ort.InferenceSession or _TRTSession
        self._trt_mode   = False
        self._onnx_mode  = False
        self._device     = _best_device()

        # EMA on polynomial coefficients (worker thread only — no lock)
        self._smoother_cf = None   # centerline  → deviation
        self._smoother_l  = None   # left boundary  → heading/curvature
        self._smoother_r  = None   # right boundary → heading/curvature
        self._alpha       = SEG_CENTERLINE_ALPHA

        # Async infrastructure
        self._queue:  queue.Queue = queue.Queue(maxsize=1)
        self._result              = _NULL_RESULT
        self._lock                = threading.Lock()
        self._thread: threading.Thread | None = None

    # ── Initialisation ─────────────────────────────────────────────────────────

    def init(self) -> bool:
        # ── 1. TensorRT engine (fastest) ────────────────────────────────────
        if _TRT_AVAILABLE and os.path.exists(SEG_ENGINE_PATH):
            try:
                print(f"[SegformerLane] Loading TRT engine {SEG_ENGINE_PATH} ...")
                self._session  = _TRTSession(SEG_ENGINE_PATH)
                self._trt_mode = True
                print("[SegformerLane] TRT FP16 ready.")
            except Exception as e:
                print(f"[SegformerLane] TRT failed: {e}")

        # ── 2. ONNX Runtime with tuned model ────────────────────────────────
        if not self._trt_mode and ort is not None and os.path.exists(SEG_ONNX_PATH):
            try:
                providers = _ort_providers()
                print(f"[SegformerLane] Loading ONNX {SEG_ONNX_PATH} ({providers}) ...")
                self._session   = ort.InferenceSession(SEG_ONNX_PATH, providers=providers)
                self._onnx_mode = True
                print(f"[SegformerLane] ONNX ready.")
            except Exception as e:
                print(f"[SegformerLane] ONNX failed: {e}")

        # ── 3. HuggingFace model (slowest fallback) ──────────────────────────
        if not self._trt_mode and not self._onnx_mode:
            try:
                from transformers import (SegformerForSemanticSegmentation,
                                          SegformerImageProcessor)
                print(f"[SegformerLane] Loading HF model {SEG_MODEL_ID} on {self._device.upper()} ...")
                self._processor = SegformerImageProcessor.from_pretrained(SEG_MODEL_ID)
                self._model     = SegformerForSemanticSegmentation.from_pretrained(SEG_MODEL_ID)
                self._model     = self._model.to(self._device).eval()
                print("[SegformerLane] HF model ready.")
            except Exception as e:
                print(f"[SegformerLane] init failed: {e}")
                return False

        self._thread = threading.Thread(target=self._worker, daemon=True, name="SegformerWorker")
        self._thread.start()
        return True

    # ── Async public interface ─────────────────────────────────────────────────

    def submit(self, frame_bgr: np.ndarray, pc: np.ndarray,
               H: int, W: int, fx: float) -> None:
        """Non-blocking: post latest frame to worker, dropping stale frames."""
        try:
            self._queue.put_nowait((frame_bgr, pc, H, W, fx))
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait((frame_bgr, pc, H, W, fx))
            except queue.Full:
                pass

    def get_result(self):
        """
        Non-blocking read of the latest computed result.
        Returns (lf, rf, road_mask, dev_m, wid_m, lc, rc, source).
        lf == rf == centerline polynomial (so downstream heading/curvature work unchanged).
        road_mask is None until the first inference completes.
        """
        with self._lock:
            return self._result

    # ── Worker thread ──────────────────────────────────────────────────────────

    def _worker(self) -> None:
        while True:
            frame_bgr, pc, H, W, fx = self._queue.get()
            result = self._infer(frame_bgr, pc, H, W, fx)
            with self._lock:
                self._result = result

    # ── Road mask ──────────────────────────────────────────────────────────────

    def _road_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Run Segformer; return bool mask (H,W) — True = road/drivable."""
        h, w = frame_bgr.shape[:2]

        if self._trt_mode or self._onnx_mode:
            # Preprocess: BGR→RGB, resize, ImageNet normalise → (1,3,H,W)
            resized = cv2.resize(frame_bgr, (SEG_INPUT_W, SEG_INPUT_H),
                                 interpolation=cv2.INTER_LINEAR)
            rgb    = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tensor = ((rgb - _MEAN) / _STD).transpose(2, 0, 1)[np.newaxis]  # (1,3,H,W)

            if self._trt_mode:
                logits = self._session.run(tensor.astype(np.float16))  # (1,2,H/4,W/4)
            else:
                inp_name = self._session.get_inputs()[0].name
                out_name = self._session.get_outputs()[0].name
                logits = self._session.run([out_name], {inp_name: tensor})[0]

            pred_small = logits[0].argmax(axis=0).astype(np.uint8)
            pred = cv2.resize(pred_small, (w, h), interpolation=cv2.INTER_NEAREST)

        else:
            # HuggingFace path
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            inp = self._processor(images=rgb, return_tensors="pt").to(self._device)
            with torch.no_grad():
                logits = self._model(**inp).logits          # (1, C, H/4, W/4)
            logits_up = F.interpolate(logits, size=(h, w),
                                      mode="bilinear", align_corners=False)
            pred = logits_up.argmax(dim=1).squeeze(0).cpu().numpy()

        mask = np.zeros((h, w), dtype=bool)
        for cls in SEG_ROAD_CLASSES:
            mask |= (pred == cls)
        return mask

    # ── Dual-zone centerline scan ─────────────────────────────────────────────
    #
    # Zone 1 — near  [fit_top, H]:   centerline → dev_m        (precise, stable)
    # Zone 2 — full  [roi_top, H]:   centerline → heading/curv (sees full curve)
    #
    # We use the CENTRE (left+right)/2 per row for both zones — not the raw
    # boundary.  The centre averages out left/right noise, so it is reliable
    # even in far rows where individual boundary pixels are noisy.  Boundary-
    # only polynomials in far rows blow up because a single misclassified pixel
    # on one side shifts that edge wildly; the centre is immune to that.

    @staticmethod
    def _scan_roads(mask: np.ndarray, roi_top: int, fit_top: int, y_near: int):
        """
        Scan road-centre x at two row densities.

        near_pts : centre in [fit_top, H]  → deviation polynomial
        full_pts : centre in [roi_top, H]  → heading/curvature polynomial
                   (denser scan so far rows contribute meaningfully)

        Returns
        -------
        near_pts    : (N,2) float (y, cx) or None
        full_pts    : (M,2) float (y, cx) or None
        wid_px_near : float  road width in px at y_near
        """
        h, w      = mask.shape
        near_rows = np.linspace(fit_top,  h - 1, SEG_BOUNDARY_ROWS,      dtype=int)
        full_rows = np.linspace(roi_top,  h - 1, SEG_BOUNDARY_ROWS * 2,  dtype=int)

        near_pts    = []
        full_pts    = []
        wid_px_near = 0.0
        best_dist   = h

        for r in near_rows:
            cols = np.where(mask[r])[0]
            if len(cols) < int(w * SEG_MIN_ROAD_FRAC):
                continue
            lx = float(cols.min())
            rx = float(cols.max())
            cx = (lx + rx) / 2.0
            near_pts.append((float(r), cx))
            dist = abs(int(r) - y_near)
            if dist < best_dist:
                best_dist   = dist
                wid_px_near = rx - lx

        for r in full_rows:
            cols = np.where(mask[r])[0]
            if len(cols) < int(w * SEG_MIN_ROAD_FRAC):
                continue
            cx = (float(cols.min()) + float(cols.max())) / 2.0
            full_pts.append((float(r), cx))

        return (
            np.array(near_pts, dtype=float) if len(near_pts) >= 3 else None,
            np.array(full_pts, dtype=float) if len(full_pts) >= 3 else None,
            wid_px_near,
        )

    # ── Pixel → metres using ZED point cloud ──────────────────────────────────

    @staticmethod
    def _px_to_m(px_val: float, y_row: int, x_col: int,
                 pc: np.ndarray, fx: float) -> float:
        if pc is None or fx <= 0:
            return px_val / 400.0
        pad  = 5
        h, w = pc.shape[:2]
        r0, r1 = max(0, y_row - pad), min(h, y_row + pad)
        c0, c1 = max(0, x_col - pad), min(w, x_col + pad)
        patch  = pc[r0:r1, c0:c1, 2]
        # ZED RIGHT_HANDED_Y_UP: forward = −Z  (objects ahead have negative Z)
        valid  = patch[np.isfinite(patch) & (patch < -0.1) & (patch > -30.0)]
        Z      = float(np.median(np.abs(valid))) if len(valid) > 0 else 3.0
        return px_val * Z / fx

    # ── Core inference (called only from worker thread) ────────────────────────

    def _infer(self, frame_bgr: np.ndarray, pc: np.ndarray,
               H: int, W: int, fx: float):
        """
        Run one frame of Segformer lane detection.

        Returns (lf, rf, road_mask, dev_m, wid_m, lc, rc, source).

        dev_m  — from near-zone centerline (most stable)
        lf, rf — boundary polynomials from full ROI (prominent side only)
                 used by compute_heading / compute_curvature downstream
        """
        roi_top  = int(H * SEG_ROI_TOP_FRAC)
        fit_top  = int(H * SEG_FIT_TOP_FRAC)
        y_near   = int(H * SEG_NEAR_FRAC)
        cx_frame = W / 2.0

        road_mask = self._road_mask(frame_bgr)

        # Trim grass pixels from road boundaries before polynomial fitting.
        # If the right (or left) inner strip is grass, the mask is trimmed inward
        # to the first asphalt pixel → centerline shifts away from grass naturally.
        road_mask = validate_boundaries(frame_bgr, road_mask, roi_top)

        near_pts, full_pts, wid_px_near = self._scan_roads(
            road_mask, roi_top, fit_top, y_near)

        # ── Near-zone centerline → deviation ─────────────────────────────────
        cf_near_raw = None
        conf        = 0.0
        if near_pts is not None:
            cf_near_raw = np.polyfit(near_pts[:, 0], near_pts[:, 1], deg=SEG_POLY_DEG)
            x_test = float(np.polyval(cf_near_raw, y_near))
            if not (0 <= x_test <= W):
                cf_near_raw = None
            else:
                conf = len(near_pts) / SEG_BOUNDARY_ROWS

        if cf_near_raw is not None:
            self._smoother_cf = (cf_near_raw if self._smoother_cf is None
                                 else self._alpha * cf_near_raw + (1 - self._alpha) * self._smoother_cf)
        cf_near = self._smoother_cf

        dev_m = wid_m = 0.0
        if cf_near is not None:
            cx_road = float(np.clip(np.polyval(cf_near, y_near), 0, W - 1))
            dev_px  = cx_road - cx_frame
            cx_col  = int(np.clip(cx_road, 0, W - 1))
            dev_m   = self._px_to_m(dev_px,      y_near, cx_col, pc, fx)
            wid_m   = self._px_to_m(wid_px_near, y_near, cx_col, pc, fx)

        # ── Full-range centerline → heading / curvature ───────────────────────
        # Fitted over [roi_top, H] so far rows contribute their heading signal.
        # Using the centre (not boundary) means far-row noise averages out —
        # a misclassified pixel on one side is balanced by the opposite side.
        cf_full_raw = None
        if full_pts is not None:
            cf_full_raw = np.polyfit(full_pts[:, 0], full_pts[:, 1], deg=SEG_POLY_DEG)
            x_test = float(np.polyval(cf_full_raw, y_near))
            if not (0 <= x_test <= W):
                cf_full_raw = None

        if cf_full_raw is not None:
            self._smoother_l = (cf_full_raw if self._smoother_l is None
                                else self._alpha * cf_full_raw + (1 - self._alpha) * self._smoother_l)
        cf_full = self._smoother_l

        # lf = rf = full-range poly so compute_heading/curvature get the curve signal
        # while dev_m stays anchored to the stable near-zone measurement
        lf = cf_full
        rf = cf_full

        lc = rc = conf

        if cf_near is not None:
            source = "SEGFORMER"
        else:
            source = "LOST"

        return lf, rf, road_mask, dev_m, wid_m, lc, rc, source
