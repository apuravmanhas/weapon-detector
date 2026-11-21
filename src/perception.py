"""
Stage-1 weapon detector, Stage-2 ConvNeXt crop verifier, and velocity-aware
weapon tracker with static-background suppression.

Designed for self-contained execution against weights under `weights/`.
"""

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models

from src.pose import compute_iou, ensure_torch_cuda_dlls
from src.states import WeaponObservation

REPO_ROOT = Path(__file__).resolve().parent.parent
_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
DEFAULT_IMAGENET_MEAN = (0.485, 0.456, 0.406)
DEFAULT_IMAGENET_STD = (0.229, 0.224, 0.225)

DISTRACTOR_CLASS_NAMES = {
    "phone",
    "cell phone",
    "remote",
    "umbrella",
    "power_tool",
    "drill",
    "backpack",
    "handbag",
    "table",
    "dining table",
    "table_bg",
    "other_bg",
    "person",
    "hand",
}
WEAPON_CLASS_KEYWORDS = (
    "weapon",
    "gun",
    "pistol",
    "handgun",
    "rifle",
    "shotgun",
    "knife",
    "blade",
    "revolver",
    "smg",
)

YOLO = None


def _get_yolo_class():
    if YOLO is not None:
        return YOLO
    from ultralytics import YOLO as ultralytics_yolo
    return ultralytics_yolo


def _supports_quantize_arg() -> bool:
    from ultralytics.utils import DEFAULT_CFG_DICT
    return "quantize" in DEFAULT_CFG_DICT


def is_lfs_pointer(path: str | Path) -> bool:
    p = Path(path)
    if not p.is_file() or p.stat().st_size >= 1024:
        return False
    with p.open("rb") as fh:
        return fh.read(len(_LFS_POINTER_PREFIX)).startswith(_LFS_POINTER_PREFIX)


def resolve_checkpoint(path: str | Path | None, default_rel_candidates: Sequence[str]) -> Path:
    """Resolve checkpoint from an explicit path, current working directory, or repository root."""
    search_bases = (Path.cwd(), REPO_ROOT)

    if path is not None:
        raw = Path(path)
        candidates = [raw] if raw.is_absolute() else [base / raw for base in search_bases]
        for cand in candidates:
            if cand.is_file():
                if is_lfs_pointer(cand):
                    raise RuntimeError(
                        f"Checkpoint '{cand}' is a Git LFS pointer, not binary weights. Run 'git lfs pull'."
                    )
                return cand
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    for rel in default_rel_candidates:
        for base in search_bases:
            cand = base / rel
            if cand.is_file() and not is_lfs_pointer(cand):
                return cand

    raise FileNotFoundError(
        f"Could not locate checkpoint among {list(default_rel_candidates)} in {[str(b) for b in search_bases]}."
    )


def _box_area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def _intersection_area(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ix1 = max(float(box_a[0]), float(box_b[0]))
    iy1 = max(float(box_a[1]), float(box_b[1]))
    ix2 = min(float(box_a[2]), float(box_b[2]))
    iy2 = min(float(box_a[3]), float(box_b[3]))
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def filter_weapon_boxes(
    detections: list[dict[str, Any]],
    frame_shape: tuple[int, ...],
    max_area_ratio: float = 0.75,
    nested_threshold: float = 0.85,
) -> list[dict[str, Any]]:
    frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
    frame_area = float(max(1, frame_h * frame_w))
    valid: list[dict[str, Any]] = []

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cx1 = max(0, min(frame_w, int(x1)))
        cy1 = max(0, min(frame_h, int(y1)))
        cx2 = max(0, min(frame_w, int(x2)))
        cy2 = max(0, min(frame_h, int(y2)))
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        if ((cx2 - cx1) * (cy2 - cy1)) / frame_area > max_area_ratio:
            continue
        item = dict(det)
        item["box"] = [cx1, cy1, cx2, cy2]
        item["confidence"] = float(det["confidence"])
        item["class"] = str(det.get("class", "weapon"))
        valid.append(item)

    if nested_threshold >= 1.0 or len(valid) <= 1:
        return valid

    valid.sort(key=lambda d: d["confidence"], reverse=True)
    kept: list[dict[str, Any]] = []
    for cand in valid:
        cand_area = _box_area(cand["box"])
        if any(
            (min_a := min(cand_area, _box_area(prev["box"]))) > 0.0
            and (_intersection_area(cand["box"], prev["box"]) / min_a) >= nested_threshold
            for prev in kept
        ):
            continue
        kept.append(cand)
    return kept


class Stage1WeaponDetector:
    """Stage-1 YOLO11 weapon detector (FP16 on CUDA, Ultralytics 8.4 quantize-compatible)."""

    def __init__(
        self,
        weights_path: str | Path | None = None,
        conf_threshold: float = 0.20,
        iou_threshold: float = 0.45,
        imgsz: int | None = None,
        device: str | None = None,
        fp16: bool | None = None,
    ):
        resolved = resolve_checkpoint(
            weights_path,
            ("weights/yolo11x_weapon_production.pt", "weights/best.pt"),
        )
        self.weights_path = str(resolved)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.fp16 = bool(fp16 if fp16 is not None else device.startswith("cuda")) and device.startswith("cuda")
        self.is_compiled = resolved.suffix.lower() in {".engine", ".onnx"}
        if self.is_compiled:
            ensure_torch_cuda_dlls()

        yolo_cls = _get_yolo_class()
        self.model = yolo_cls(self.weights_path, task="detect")
        self._use_quantize: bool | None = None

        raw_names = getattr(self.model, "names", None)
        if isinstance(raw_names, dict):
            name_map = {int(k): str(v) for k, v in raw_names.items()}
        elif isinstance(raw_names, (list, tuple)):
            name_map = {i: str(v) for i, v in enumerate(raw_names)}
        else:
            name_map = {0: "weapon"}

        if len(name_map) <= 1:
            self.weapon_class_ids: set[int] = {0}
            self.distractor_class_ids: set[int] = set()
        else:
            matched = {
                cid
                for cid, cname in name_map.items()
                if cname.strip().lower() not in DISTRACTOR_CLASS_NAMES
                and any(kw in cname.strip().lower() for kw in WEAPON_CLASS_KEYWORDS)
            }
            self.weapon_class_ids = matched if matched else {0}
            self.distractor_class_ids = set(name_map.keys()) - self.weapon_class_ids

        if imgsz is not None:
            self.imgsz = imgsz
        elif self.is_compiled:
            lname = resolved.name.lower()
            if "yolo11x" in lname or "_800" in lname:
                self.imgsz = 800
            elif "yolo11m" in lname or "best" in lname or "_640" in lname:
                self.imgsz = 640
            else:
                model_args = getattr(self.model, "args", None)
                trained = model_args.get("imgsz") if isinstance(model_args, dict) else None
                self.imgsz = trained if isinstance(trained, int) and trained >= 32 else 640
        else:
            model_args = getattr(self.model, "args", None)
            trained = model_args.get("imgsz") if isinstance(model_args, dict) else None
            self.imgsz = trained if isinstance(trained, int) and trained >= 32 else (
                800 if "yolo11x" in resolved.name.lower() else 640
            )

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "source": frame,
            "conf": self.conf_threshold,
            "iou": self.iou_threshold,
            "imgsz": self.imgsz,
            "device": self.device,
            "verbose": False,
        }
        if self.distractor_class_ids and self.weapon_class_ids:
            kwargs["classes"] = sorted(self.weapon_class_ids)
        if self.fp16 and not self.is_compiled:
            if self._use_quantize is None:
                self._use_quantize = _supports_quantize_arg()
            if self._use_quantize:
                kwargs["quantize"] = 16
            else:
                kwargs["half"] = True

        results = self.model.predict(**kwargs)
        raw: list[dict[str, Any]] = []
        for r in results:
            if getattr(r, "boxes", None) is None:
                continue
            for box in r.boxes:
                if self.distractor_class_ids and getattr(box, "cls", None) is not None:
                    if int(float(box.cls[0])) not in self.weapon_class_ids:
                        continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                raw.append({"box": [x1, y1, x2, y2], "confidence": float(box.conf[0]), "class": "weapon"})
        return filter_weapon_boxes(raw, frame.shape)


def _extract_padded_crop(frame: np.ndarray, box: Sequence[int | float], pad_ratio: float = 0.30) -> np.ndarray:
    fh, fw = frame.shape[:2]
    x1 = max(0, min(fw, int(round(float(box[0])))))
    y1 = max(0, min(fh, int(round(float(box[1])))))
    x2 = max(x1 + 1, min(fw, int(round(float(box[2]))))) if x1 < fw else fw
    y2 = max(y1 + 1, min(fh, int(round(float(box[3]))))) if y1 < fh else fh
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    px1 = max(0, x1 - int(round(bw * pad_ratio)))
    py1 = max(0, y1 - int(round(bh * pad_ratio)))
    px2 = min(fw, x2 + int(round(bw * pad_ratio)))
    py2 = min(fh, y2 + int(round(bh * pad_ratio)))
    crop = frame[py1:py2, px1:px2]
    if crop.ndim == 2 or crop.shape[2] == 1:
        return cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def _build_classifier(arch: str, num_classes: int) -> nn.Module:
    key = arch.lower().strip()
    if key == "convnext_tiny":
        m = tv_models.convnext_tiny(weights=None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
        return m
    if key == "convnext_small":
        m = tv_models.convnext_small(weights=None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
        return m
    if key == "mobilenet_v3_large":
        m = tv_models.mobilenet_v3_large(weights=None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
        return m
    raise ValueError(f"Unsupported verifier architecture: {arch}")


class Stage2CropVerifier:
    """Stage-2 ConvNeXt crop verifier eliminating phone, umbrella, and table-edge false alarms."""

    def __init__(
        self,
        weights_path: str | Path | None = None,
        device: str | None = None,
        fp16: bool | None = None,
        pad_ratio: float = 0.30,
    ):
        resolved = resolve_checkpoint(
            weights_path,
            ("weights/crop_verifier_convnext_tiny_v2.pt", "weights/crop_verifier_convnext_tiny.pt"),
        )
        self.weights_path = str(resolved)
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.fp16 = bool(fp16 if fp16 is not None else device.startswith("cuda")) and device.startswith("cuda")
        self.pad_ratio = pad_ratio

        ckpt = torch.load(resolved, map_location="cpu", weights_only=True)
        self.arch = str(ckpt.get("arch", "convnext_tiny"))
        self.classes = [str(c) for c in ckpt["classes"]]
        raw_pos = ckpt.get("weapon_classes")
        if raw_pos:
            weapon_classes = {str(c) for c in raw_pos}
        else:
            weapon_classes = {c for c in self.classes if "weapon" in c.lower()} or {self.classes[0]}
        self._weapon_indices = [i for i, c in enumerate(self.classes) if c in weapon_classes]
        self.img_size = int(ckpt.get("img_size", 224))
        mean = tuple(float(v) for v in ckpt.get("mean", DEFAULT_IMAGENET_MEAN))
        std = tuple(float(v) for v in ckpt.get("std", DEFAULT_IMAGENET_STD))

        self.model = _build_classifier(self.arch, len(self.classes))
        self.model.load_state_dict(ckpt["state_dict"], strict=True)
        self.model.to(self.device).eval()
        if self.fp16:
            self.model.half()
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True

        dtype = torch.float16 if self.fp16 else torch.float32
        self._mean = torch.tensor(mean, dtype=dtype, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(std, dtype=dtype, device=self.device).view(1, 3, 1, 1)

    def score_crops(self, crops: list[np.ndarray]) -> list[tuple[float, str]]:
        """Score a batch of RGB crops and return (p_weapon, top_class_label) per crop."""
        if not crops:
            return []
        resized = [cv2.resize(c, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR) for c in crops]
        batch = np.stack(resized, axis=0)
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(self.device, non_blocking=True)
        tensor = tensor.to(dtype=torch.float16 if self.fp16 else torch.float32) / 255.0
        tensor = (tensor - self._mean) / self._std
        with torch.inference_mode():
            probs = torch.softmax(self.model(tensor).float(), dim=-1).cpu().numpy()
        out: list[tuple[float, str]] = []
        for row in probs:
            p_weapon = float(np.clip(np.sum(row[self._weapon_indices]), 0.0, 1.0))
            out.append((p_weapon, self.classes[int(np.argmax(row))]))
        return out

    def verify(
        self,
        frame: np.ndarray,
        detections: list[dict[str, Any]],
        verify_below_conf: float = 0.72,
        threshold: float = 0.52,
        mid_conf: float = 0.55,
        mid_threshold: float = 0.42,
        low_conf: float = 0.35,
        low_threshold: float = 0.62,
        cached_tracks: list[dict[str, Any]] | None = None,
        cache_iou: float = 0.45,
    ) -> list[dict[str, Any]]:
        if not detections:
            return []
        cloned = [dict(d) for d in detections]
        eval_idxs: list[int] = []
        crops: list[np.ndarray] = []
        for idx, det in enumerate(cloned):
            conf = float(det["confidence"])
            if conf >= verify_below_conf:
                continue

            # Track-level cache avoids redundant ConvNeXt inference on locked weapon tracks
            matched_cache = None
            if conf >= low_conf and cached_tracks:
                best_iou = cache_iou
                for tr in cached_tracks:
                    iou = compute_iou(det["box"], tr["box"])
                    if iou >= best_iou:
                        best_iou = iou
                        matched_cache = tr

            if matched_cache is not None:
                det["verifier_prob"] = round(float(matched_cache["verifier_prob"]), 4)
                det["verifier_label"] = matched_cache.get("verifier_label")
                det["verifier_cached"] = True
            else:
                eval_idxs.append(idx)
                crops.append(_extract_padded_crop(frame, det["box"], pad_ratio=self.pad_ratio))

        if eval_idxs:
            scores = self.score_crops(crops)
            for idx, (p_weapon, top_label) in zip(eval_idxs, scores):
                cloned[idx]["verifier_prob"] = round(float(p_weapon), 4)
                cloned[idx]["verifier_label"] = top_label
                cloned[idx]["verifier_fresh"] = True

        kept: list[dict[str, Any]] = []
        for det in cloned:
            conf = float(det["confidence"])
            if conf >= verify_below_conf:
                kept.append(det)
                continue
            vp = float(det.get("verifier_prob", 0.0))
            if conf >= mid_conf:
                req = mid_threshold
            elif conf >= low_conf:
                req = threshold
            else:
                req = low_threshold
            if vp >= req:
                kept.append(det)
        return kept


@dataclass
class _WeaponTrackState:
    track_id: int
    box: list[int]
    cx: float
    cy: float
    w: float
    h: float
    ema_conf: float
    last_conf: float
    cls_name: str
    hit_history: deque[int]
    anchor_box: list[int]
    vx: float = 0.0
    vy: float = 0.0
    verifier_ema: float | None = None
    verifier_label: str | None = None
    age: int = 1
    misses: int = 0
    motionless_frames: int = 0
    quarantined: bool = False
    frames_since_verified: int = 0


class WeaponTrackManager:
    """Velocity-aware weapon tracker with persistence, EMA smoothing, and static confounder quarantine."""

    def __init__(
        self,
        min_hits: int = 2,
        window_size: int = 5,
        match_iou: float = 0.20,
        max_center_dist: float = 0.65,
        max_age: int = 6,
        ema_alpha: float = 0.45,
        high_conf_bypass: float = 0.80,
        static_suppress: bool = True,
        static_iou: float = 0.90,
        static_mad: float = 4.0,
        static_frames: int = 8,
        static_max_conf: float = 0.58,
    ):
        self.min_hits = min_hits
        self.window_size = window_size
        self.match_iou = match_iou
        self.max_center_dist = max_center_dist
        self.max_age = max_age
        self.ema_alpha = ema_alpha
        self.high_conf_bypass = high_conf_bypass
        self.static_suppress = static_suppress
        self.static_iou = static_iou
        self.static_mad = static_mad
        self.static_frames = max(1, static_frames)
        self.static_max_conf = static_max_conf
        self._next_id = 1
        self._tracks: dict[int, _WeaponTrackState] = {}
        self._prev_gray: np.ndarray | None = None

    def reset(self) -> None:
        self._tracks.clear()
        self._prev_gray = None
        self._next_id = 1

    def _to_gray(self, frame: np.ndarray | None) -> np.ndarray | None:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None
        if frame.ndim == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _crop_mad(self, prev_g: np.ndarray | None, curr_g: np.ndarray | None, box: Sequence[int]) -> float | None:
        if prev_g is None or curr_g is None or prev_g.shape != curr_g.shape:
            return None
        h, w = curr_g.shape[:2]
        x1, y1 = max(0, min(w, int(box[0]))), max(0, min(h, int(box[1])))
        x2, y2 = max(0, min(w, int(box[2]))), max(0, min(h, int(box[3])))
        if x2 <= x1 or y2 <= y1:
            return None
        return float(np.mean(np.abs(curr_g[y1:y2, x1:x2].astype(np.int16) - prev_g[y1:y2, x1:x2].astype(np.int16))))

    def update(self, detections: list[dict[str, Any]], frame: np.ndarray | None = None) -> list[WeaponObservation]:
        curr_gray = self._to_gray(frame) if self.static_suppress else None
        track_ids = list(self._tracks.keys())
        candidates: list[tuple[float, int, int]] = []

        for tid in track_ids:
            tr = self._tracks[tid]
            pred_cx, pred_cy = tr.cx + tr.vx, tr.cy + tr.vy
            pred_box = [pred_cx - tr.w * 0.5, pred_cy - tr.h * 0.5, pred_cx + tr.w * 0.5, pred_cy + tr.h * 0.5]
            tr_area = max(1.0, tr.w * tr.h)
            tr_diag = math.hypot(tr.w, tr.h)
            for d_idx, det in enumerate(detections):
                d_box = det["box"]
                d_w, d_h = max(1.0, float(d_box[2] - d_box[0])), max(1.0, float(d_box[3] - d_box[1]))
                d_cx, d_cy = 0.5 * (d_box[0] + d_box[2]), 0.5 * (d_box[1] + d_box[3])
                best_iou = max(compute_iou(pred_box, d_box), compute_iou(tr.box, d_box))
                norm_dist = math.hypot(d_cx - pred_cx, d_cy - pred_cy) / max(1.0, 0.5 * (tr_diag + math.hypot(d_w, d_h)))
                scale_ratio = (d_w * d_h) / tr_area
                if best_iou >= self.match_iou or (norm_dist <= self.max_center_dist and 0.4 <= scale_ratio <= 2.5):
                    candidates.append((best_iou + max(0.0, 1.0 - norm_dist), tid, d_idx))

        candidates.sort(key=lambda item: item[0], reverse=True)
        matched_tids: set[int] = set()
        matched_didxs: set[int] = set()
        active_tracks: list[_WeaponTrackState] = []

        for _, tid, d_idx in candidates:
            if tid in matched_tids or d_idx in matched_didxs:
                continue
            matched_tids.add(tid)
            matched_didxs.add(d_idx)
            tr = self._tracks[tid]
            det = detections[d_idx]
            d_box = [int(v) for v in det["box"]]
            conf = float(det["confidence"])
            d_w, d_h = float(max(1, d_box[2] - d_box[0])), float(max(1, d_box[3] - d_box[1]))
            d_cx, d_cy = 0.5 * (d_box[0] + d_box[2]), 0.5 * (d_box[1] + d_box[3])

            step = tr.misses + 1
            tr.vx = 0.65 * ((d_cx - tr.cx) / step) + 0.35 * tr.vx
            tr.vy = 0.65 * ((d_cy - tr.cy) / step) + 0.35 * tr.vy
            tr.box, tr.cx, tr.cy, tr.w, tr.h = d_box, d_cx, d_cy, d_w, d_h
            tr.last_conf = conf
            tr.ema_conf = self.ema_alpha * conf + (1.0 - self.ema_alpha) * tr.ema_conf
            tr.misses = 0
            tr.age += 1
            tr.hit_history.append(1)
            if det.get("verifier_fresh"):
                tr.frames_since_verified = 0
            else:
                tr.frames_since_verified += 1
            if det.get("verifier_prob") is not None and not det.get("verifier_cached"):
                vp = float(det["verifier_prob"])
                tr.verifier_ema = vp if tr.verifier_ema is None else 0.45 * vp + 0.55 * tr.verifier_ema
            if det.get("verifier_label") is not None:
                tr.verifier_label = str(det["verifier_label"])

            if self.static_suppress:
                anchor_iou = compute_iou(tr.box, tr.anchor_box)
                mad = self._crop_mad(self._prev_gray, curr_gray, tr.box)
                if anchor_iou >= self.static_iou and mad is not None and mad < self.static_mad:
                    tr.motionless_frames += 1
                else:
                    tr.anchor_box = list(tr.box)
                    tr.motionless_frames = 0
                verifier_strong = tr.verifier_ema is not None and tr.verifier_ema >= 0.80
                tr.quarantined = (
                    tr.motionless_frames >= self.static_frames
                    and tr.ema_conf < self.static_max_conf
                    and not verifier_strong
                )
            active_tracks.append(tr)

        for tid in track_ids:
            if tid in matched_tids:
                continue
            tr = self._tracks[tid]
            tr.misses += 1
            tr.age += 1
            tr.vx *= 0.8
            tr.vy *= 0.8
            tr.cx += tr.vx
            tr.cy += tr.vy
            tr.hit_history.append(0)
            if tr.misses > self.max_age:
                del self._tracks[tid]

        for d_idx, det in enumerate(detections):
            if d_idx in matched_didxs:
                continue
            d_box = [int(v) for v in det["box"]]
            conf = float(det["confidence"])
            tid = self._next_id
            self._next_id += 1
            tr = _WeaponTrackState(
                track_id=tid,
                box=d_box,
                cx=0.5 * (d_box[0] + d_box[2]),
                cy=0.5 * (d_box[1] + d_box[3]),
                w=float(max(1, d_box[2] - d_box[0])),
                h=float(max(1, d_box[3] - d_box[1])),
                ema_conf=conf,
                last_conf=conf,
                cls_name=str(det.get("class", "weapon")),
                hit_history=deque([1], maxlen=self.window_size),
                anchor_box=list(d_box),
                verifier_ema=float(det["verifier_prob"]) if det.get("verifier_prob") is not None else None,
                verifier_label=str(det["verifier_label"]) if det.get("verifier_label") is not None else None,
            )
            self._tracks[tid] = tr
            active_tracks.append(tr)

        if self.static_suppress:
            self._prev_gray = curr_gray

        confirmed: list[WeaponObservation] = []
        for tr in active_tracks:
            if tr.quarantined:
                continue
            if sum(tr.hit_history) < self.min_hits and tr.last_conf < self.high_conf_bypass:
                continue
            confirmed.append(
                WeaponObservation(
                    track_id=tr.track_id,
                    box=list(tr.box),
                    confidence=tr.last_conf,
                    cls_name=tr.cls_name,
                    ema_conf=round(tr.ema_conf, 4),
                    verifier_prob=round(tr.verifier_ema, 4) if tr.verifier_ema is not None else None,
                    verifier_label=tr.verifier_label,
                    vx=tr.vx,
                    vy=tr.vy,
                )
            )
        return confirmed

    def get_verifier_cache_candidates(
        self,
        max_ttl_frames: int = 4,
        min_verifier_ema: float = 0.75,
    ) -> list[dict[str, Any]]:
        """Return recently verified active non-quarantined tracks eligible to skip redundant Stage-2 checks."""
        out: list[dict[str, Any]] = []
        for tr in self._tracks.values():
            if tr.quarantined or tr.misses != 0:
                continue
            if sum(tr.hit_history) < self.min_hits and tr.last_conf < self.high_conf_bypass:
                continue
            if tr.frames_since_verified >= max_ttl_frames:
                continue
            if tr.verifier_ema is None or tr.verifier_ema < min_verifier_ema:
                continue
            out.append(
                {
                    "box": list(tr.box),
                    "verifier_prob": float(tr.verifier_ema),
                    "verifier_label": tr.verifier_label,
                }
            )
        return out
