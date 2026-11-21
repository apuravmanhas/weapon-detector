"""
Pose estimation (YOLO11-pose) and deterministic zero-dependency person tracking.

Uses a clean velocity-predicted IoU + centroid matcher so ByteTrack-style ID stability
works reliably even without optional C/Cython dependencies like `lap` or `scipy`.
"""

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.states import PersonTrack

YOLO = None
_CUDA_DLLS_REGISTERED = False


def ensure_torch_cuda_dlls() -> None:
    """Register PyTorch's bundled CUDA/cuDNN DLL directory once on Windows for ONNX Runtime GPU."""
    global _CUDA_DLLS_REGISTERED
    if _CUDA_DLLS_REGISTERED or os.name != "nt":
        return
    try:
        import torch

        tlib = Path(torch.__file__).resolve().parent / "lib"
        if tlib.is_dir():
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(tlib))
            cur_path = os.environ.get("PATH", "")
            if str(tlib) not in cur_path:
                os.environ["PATH"] = str(tlib) + os.pathsep + cur_path
    except Exception:
        pass
    _CUDA_DLLS_REGISTERED = True


def _get_yolo_class():
    if YOLO is not None:
        return YOLO
    from ultralytics import YOLO as ultralytics_yolo
    return ultralytics_yolo


def _supports_quantize_arg() -> bool:
    from ultralytics.utils import DEFAULT_CFG_DICT
    return "quantize" in DEFAULT_CFG_DICT


def _box_area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def compute_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ix1 = max(float(box_a[0]), float(box_b[0]))
    iy1 = max(float(box_a[1]), float(box_b[1]))
    ix2 = min(float(box_a[2]), float(box_b[2]))
    iy2 = min(float(box_a[3]), float(box_b[3]))
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = _box_area(box_a) + _box_area(box_b) - inter
    return inter / union if union > 0.0 else 0.0


@dataclass
class _InternalPersonTrack:
    person_id: int
    box: list[int]
    cx: float
    cy: float
    w: float
    h: float
    confidence: float
    keypoints: np.ndarray
    vx: float = 0.0
    vy: float = 0.0
    age: int = 1
    misses: int = 0


class PersonTrackManager:
    """
    Lightweight, deterministic multi-person tracker matching person boxes by velocity-predicted
    IoU and normalized center distance, with optional keypoint EMA smoothing for jitter control.
    """

    def __init__(
        self,
        match_iou: float = 0.25,
        max_center_dist: float = 0.75,
        max_age: int = 12,
        kpt_ema_alpha: float = 0.75,
        kpt_conf_thresh: float = 0.20,
    ):
        if not 0.0 <= match_iou <= 1.0:
            raise ValueError("match_iou must be in [0.0, 1.0].")
        if max_center_dist < 0.0:
            raise ValueError("max_center_dist must be >= 0.0.")
        if max_age < 0:
            raise ValueError("max_age must be >= 0.")
        if not 0.0 < kpt_ema_alpha <= 1.0:
            raise ValueError("kpt_ema_alpha must be in (0.0, 1.0].")

        self.match_iou = match_iou
        self.max_center_dist = max_center_dist
        self.max_age = max_age
        self.kpt_ema_alpha = kpt_ema_alpha
        self.kpt_conf_thresh = kpt_conf_thresh

        self._next_id = 1
        self._tracks: dict[int, _InternalPersonTrack] = {}

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def _smooth_keypoints(self, prev_kpts: np.ndarray, curr_kpts: np.ndarray) -> np.ndarray:
        """
        Smooth visible keypoints slightly while preserving rapid draw motion (high alpha).
        If a joint briefly drops below threshold, keep its last valid location with decayed conf.
        """
        out = curr_kpts.copy().astype(np.float32)
        alpha = self.kpt_ema_alpha
        for i in range(min(len(out), len(prev_kpts))):
            c_conf = float(curr_kpts[i, 2])
            p_conf = float(prev_kpts[i, 2])
            if c_conf >= self.kpt_conf_thresh and p_conf >= self.kpt_conf_thresh:
                out[i, 0] = alpha * curr_kpts[i, 0] + (1.0 - alpha) * prev_kpts[i, 0]
                out[i, 1] = alpha * curr_kpts[i, 1] + (1.0 - alpha) * prev_kpts[i, 1]
                out[i, 2] = c_conf
            elif c_conf < self.kpt_conf_thresh and p_conf >= self.kpt_conf_thresh:
                # Brief 1-2 frame occlusion carry-forward with confidence decay
                out[i, 0] = prev_kpts[i, 0]
                out[i, 1] = prev_kpts[i, 1]
                out[i, 2] = p_conf * 0.65
        return out

    def update(self, raw_persons: list[dict[str, Any]]) -> list[PersonTrack]:
        """
        Match raw per-frame person pose detections against active person tracks and return
        the active matched tracks for the current frame.
        """
        track_ids = list(self._tracks.keys())
        candidates: list[tuple[float, int, int]] = []

        for pid in track_ids:
            tr = self._tracks[pid]
            pred_cx = tr.cx + tr.vx
            pred_cy = tr.cy + tr.vy
            pred_box = [
                pred_cx - tr.w * 0.5,
                pred_cy - tr.h * 0.5,
                pred_cx + tr.w * 0.5,
                pred_cy + tr.h * 0.5,
            ]
            tr_diag = max(1.0, math.hypot(tr.w, tr.h))
            tr_area = max(1.0, tr.w * tr.h)

            for d_idx, det in enumerate(raw_persons):
                d_box = det["box"]
                d_w = max(1.0, float(d_box[2]) - float(d_box[0]))
                d_h = max(1.0, float(d_box[3]) - float(d_box[1]))
                d_cx = 0.5 * (float(d_box[0]) + float(d_box[2]))
                d_cy = 0.5 * (float(d_box[1]) + float(d_box[3]))

                best_iou = max(compute_iou(pred_box, d_box), compute_iou(tr.box, d_box))
                mean_diag = max(1.0, 0.5 * (tr_diag + math.hypot(d_w, d_h)))
                norm_dist = math.hypot(d_cx - pred_cx, d_cy - pred_cy) / mean_diag
                scale_ratio = (d_w * d_h) / tr_area

                if best_iou >= self.match_iou or (
                    norm_dist <= self.max_center_dist and 0.35 <= scale_ratio <= 2.8
                ):
                    score = best_iou + max(0.0, 1.0 - norm_dist)
                    candidates.append((score, pid, d_idx))

        candidates.sort(key=lambda item: item[0], reverse=True)
        matched_pids: set[int] = set()
        matched_didxs: set[int] = set()
        active_outputs: list[PersonTrack] = []

        for _, pid, d_idx in candidates:
            if pid in matched_pids or d_idx in matched_didxs:
                continue
            matched_pids.add(pid)
            matched_didxs.add(d_idx)

            tr = self._tracks[pid]
            det = raw_persons[d_idx]
            d_box = [int(v) for v in det["box"]]
            d_w = float(max(1, d_box[2] - d_box[0]))
            d_h = float(max(1, d_box[3] - d_box[1]))
            d_cx = 0.5 * (d_box[0] + d_box[2])
            d_cy = 0.5 * (d_box[1] + d_box[3])

            step = tr.misses + 1
            inst_vx = (d_cx - tr.cx) / step
            inst_vy = (d_cy - tr.cy) / step
            tr.vx = 0.65 * inst_vx + 0.35 * tr.vx
            tr.vy = 0.65 * inst_vy + 0.35 * tr.vy
            tr.box = d_box
            tr.cx = d_cx
            tr.cy = d_cy
            tr.w = d_w
            tr.h = d_h
            tr.confidence = float(det["confidence"])
            tr.keypoints = self._smooth_keypoints(tr.keypoints, np.asarray(det["keypoints"], dtype=np.float32))
            tr.misses = 0
            tr.age += 1

            active_outputs.append(
                PersonTrack(
                    person_id=tr.person_id,
                    box=list(tr.box),
                    confidence=tr.confidence,
                    keypoints=tr.keypoints.copy(),
                    vx=tr.vx,
                    vy=tr.vy,
                    age=tr.age,
                    misses=0,
                )
            )

        for pid in track_ids:
            if pid in matched_pids:
                continue
            tr = self._tracks[pid]
            tr.misses += 1
            tr.age += 1
            tr.vx *= 0.8
            tr.vy *= 0.8
            tr.cx += tr.vx
            tr.cy += tr.vy
            if tr.misses > self.max_age:
                del self._tracks[pid]

        for d_idx, det in enumerate(raw_persons):
            if d_idx in matched_didxs:
                continue
            d_box = [int(v) for v in det["box"]]
            d_w = float(max(1, d_box[2] - d_box[0]))
            d_h = float(max(1, d_box[3] - d_box[1]))
            d_cx = 0.5 * (d_box[0] + d_box[2])
            d_cy = 0.5 * (d_box[1] + d_box[3])
            kpts = np.asarray(det["keypoints"], dtype=np.float32).copy()

            pid = self._next_id
            self._next_id += 1
            tr = _InternalPersonTrack(
                person_id=pid,
                box=d_box,
                cx=d_cx,
                cy=d_cy,
                w=d_w,
                h=d_h,
                confidence=float(det["confidence"]),
                keypoints=kpts,
            )
            self._tracks[pid] = tr
            active_outputs.append(
                PersonTrack(
                    person_id=pid,
                    box=list(d_box),
                    confidence=tr.confidence,
                    keypoints=kpts.copy(),
                    vx=0.0,
                    vy=0.0,
                    age=1,
                    misses=0,
                )
            )

        return active_outputs


class PoseEstimator:
    """
    Real-time 17-keypoint COCO pose estimator backed by Ultralytics YOLO11-pose (FP16 on CUDA).
    """

    def __init__(
        self,
        weights: str = "yolo11s-pose.pt",
        conf_threshold: float = 0.30,
        iou_threshold: float = 0.50,
        imgsz: int = 640,
        device: str | None = None,
        fp16: bool | None = None,
    ):
        if not 0.0 < conf_threshold <= 1.0:
            raise ValueError("conf_threshold must be in (0.0, 1.0].")
        if imgsz < 32:
            raise ValueError("imgsz must be >= 32.")

        import torch

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.fp16 = bool(fp16 if fp16 is not None else device.startswith("cuda")) and device.startswith("cuda")
        self.weights = weights
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self._use_quantize: bool | None = None
        self.is_compiled = Path(str(weights)).suffix.lower() in {".engine", ".onnx"}
        if self.is_compiled:
            ensure_torch_cuda_dlls()

        yolo_cls = _get_yolo_class()
        self.model = yolo_cls(weights, task="pose")

    def predict(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Run pose inference on a BGR frame and return parsed person dicts with (17, 3) keypoints."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Input frame must be a non-empty numpy array.")

        kwargs: dict[str, Any] = {
            "source": frame,
            "conf": self.conf_threshold,
            "iou": self.iou_threshold,
            "imgsz": self.imgsz,
            "device": self.device,
            "verbose": False,
        }
        if self.fp16 and not self.is_compiled:
            if self._use_quantize is None:
                self._use_quantize = _supports_quantize_arg()
            if self._use_quantize:
                kwargs["quantize"] = 16
            else:
                kwargs["half"] = True

        results = self.model.predict(**kwargs)
        fh, fw = frame.shape[:2]
        parsed: list[dict[str, Any]] = []

        for r in results:
            boxes = getattr(r, "boxes", None)
            kpts_obj = getattr(r, "keypoints", None)
            if boxes is None or kpts_obj is None or len(boxes) == 0:
                continue

            xyxy_arr = boxes.xyxy.cpu().numpy()
            conf_arr = boxes.conf.cpu().numpy()
            kpt_data = kpts_obj.data.cpu().numpy()  # (N, 17, 3)

            for i in range(len(xyxy_arr)):
                x1 = max(0, min(fw, int(round(float(xyxy_arr[i, 0])))))
                y1 = max(0, min(fh, int(round(float(xyxy_arr[i, 1])))))
                x2 = max(0, min(fw, int(round(float(xyxy_arr[i, 2])))))
                y2 = max(0, min(fh, int(round(float(xyxy_arr[i, 3])))))
                if x2 <= x1 or y2 <= y1:
                    continue

                kpts = kpt_data[i].astype(np.float32)
                if kpts.shape[0] < 17:
                    padded = np.zeros((17, 3), dtype=np.float32)
                    padded[: kpts.shape[0]] = kpts
                    kpts = padded
                parsed.append(
                    {
                        "box": [x1, y1, x2, y2],
                        "confidence": float(conf_arr[i]),
                        "keypoints": kpts[:17, :3],
                    }
                )
        return parsed
