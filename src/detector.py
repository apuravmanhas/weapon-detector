from collections import deque
from pathlib import Path
from typing import Any, Sequence
import warnings

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHT_CANDIDATES = (
    "weights/yolo11x_weapon_production.pt",
    "weights/best.pt",
)
_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
DISTRACTOR_CLASS_NAMES = {
    "phone",
    "cell phone",
    "remote",
    "umbrella",
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
    "larga",
    "curta",
)

# Lazily imported on first WeaponDetector instantiation so lightweight utilities stay fast.
YOLO = None


def _get_yolo_class():
    if YOLO is not None:
        return YOLO
    from ultralytics import YOLO as ultralytics_yolo
    return ultralytics_yolo


def _supports_quantize_arg() -> bool:
    """Detect Ultralytics >=8.4 'quantize' arg vs legacy 'half' arg."""
    from ultralytics.utils import DEFAULT_CFG_DICT
    return "quantize" in DEFAULT_CFG_DICT


def is_lfs_pointer(path: str | Path) -> bool:
    """Return True if path is an un-pulled Git LFS text pointer file."""
    p = Path(path)
    if not p.is_file() or p.stat().st_size >= 1024:
        return False
    with p.open("rb") as fh:
        header = fh.read(len(_LFS_POINTER_PREFIX))
    return header.startswith(_LFS_POINTER_PREFIX)


def resolve_weights_path(weights_path: str | Path | None = None) -> Path:
    """
    Resolve model weights path relative to CWD or repository root.

    Rejects missing paths and un-pulled Git LFS pointer files with clear diagnostics.
    """
    if weights_path is not None:
        candidate = Path(weights_path)
        if not candidate.is_file():
            repo_candidate = REPO_ROOT / candidate
            if repo_candidate.is_file():
                candidate = repo_candidate
            else:
                raise FileNotFoundError(f"Model weights not found at: {weights_path}")

        if is_lfs_pointer(candidate):
            raise RuntimeError(
                f"Weights file '{candidate}' is a Git LFS pointer, not binary weights. "
                "Run 'git lfs pull' to download the model."
            )
        return candidate

    saw_lfs_pointer = None
    for rel in DEFAULT_WEIGHT_CANDIDATES:
        for base in (Path.cwd(), REPO_ROOT):
            candidate = base / rel
            if not candidate.is_file():
                continue
            if is_lfs_pointer(candidate):
                saw_lfs_pointer = candidate
                continue
            if saw_lfs_pointer is not None:
                warnings.warn(
                    f"'{saw_lfs_pointer}' is an un-pulled Git LFS pointer; "
                    f"falling back to '{candidate}'. Run 'git lfs pull' for YOLO11x.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return candidate

    if saw_lfs_pointer is not None:
        raise RuntimeError(
            f"Weights file '{saw_lfs_pointer}' is a Git LFS pointer, not binary weights. "
            "Run 'git lfs pull' to download the model."
        )
    raise FileNotFoundError("No valid model weights found in weights/ directory.")


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


def compute_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Compute Intersection-over-Union between two [x1, y1, x2, y2] boxes."""
    inter = _intersection_area(box_a, box_b)
    if inter <= 0.0:
        return 0.0
    union = _box_area(box_a) + _box_area(box_b) - inter
    return inter / union if union > 0.0 else 0.0


def apply_lab_clahe(
    frame: np.ndarray,
    clip_limit: float = 2.5,
    tile_grid: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply contrast-limited adaptive histogram equalization to luminance (L channel) in LAB space."""
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("Input frame must be a non-empty numpy array.")
    if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
        return frame.copy()
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    merged = cv2.merge((clahe.apply(l_ch), a_ch, b_ch))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def nms_merge(
    detections: list[dict[str, Any]],
    iou_threshold: float = 0.58,
) -> list[dict[str, Any]]:
    """Merge overlapping multi-scale/tiled detections via greedy confidence NMS."""
    ordered = sorted(detections, key=lambda d: float(d["confidence"]), reverse=True)
    kept: list[dict[str, Any]] = []
    for cand in ordered:
        if any(compute_iou(cand["box"], prev["box"]) > iou_threshold for prev in kept):
            continue
        kept.append(cand)
    return kept


def filter_detections(
    detections: list[dict[str, Any]],
    frame_shape: tuple[int, ...],
    max_area_ratio: float = 0.75,
    nested_threshold: float = 0.85,
) -> list[dict[str, Any]]:
    """
    Sanitize and deduplicate single-class weapon detections.

    - Clamps box coordinates within image dimensions and removes degenerate boxes.
    - Drops full-frame/vignette hallucinations exceeding max_area_ratio of frame area.
    - Suppresses redundant lower-confidence detections heavily nested inside another box.
    """
    if len(frame_shape) < 2 or frame_shape[0] <= 0 or frame_shape[1] <= 0:
        raise ValueError("Frame dimensions must be positive.")

    frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
    frame_area = float(frame_h * frame_w)

    valid: list[dict[str, Any]] = []
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cx1 = max(0, min(frame_w, int(x1)))
        cy1 = max(0, min(frame_h, int(y1)))
        cx2 = max(0, min(frame_w, int(x2)))
        cy2 = max(0, min(frame_h, int(y2)))
        if cx2 <= cx1 or cy2 <= cy1:
            continue

        area = (cx2 - cx1) * (cy2 - cy1)
        if area / frame_area > max_area_ratio:
            continue

        sanitized = dict(det)
        sanitized["box"] = [cx1, cy1, cx2, cy2]
        sanitized["confidence"] = float(det["confidence"])
        sanitized["class"] = det.get("class", "weapon")
        valid.append(sanitized)

    if nested_threshold >= 1.0 or len(valid) <= 1:
        return valid

    valid.sort(key=lambda d: d["confidence"], reverse=True)
    kept: list[dict[str, Any]] = []
    for cand in valid:
        cand_area = _box_area(cand["box"])
        is_nested = False
        for existing in kept:
            min_area = min(cand_area, _box_area(existing["box"]))
            if min_area > 0.0 and (_intersection_area(cand["box"], existing["box"]) / min_area) >= nested_threshold:
                is_nested = True
                break
        if not is_nested:
            kept.append(cand)
    return kept


class TemporalPersistenceFilter:
    """
    Sliding-window IoU persistence filter to reject transient single-frame false positives
    in continuous video or camera streams.
    """

    def __init__(self, min_hits: int = 1, window_size: int = 5, match_iou: float = 0.25):
        if min_hits < 1:
            raise ValueError("min_hits must be >= 1.")
        if window_size < min_hits:
            raise ValueError("window_size must be >= min_hits.")
        if not 0.0 <= match_iou <= 1.0:
            raise ValueError("match_iou must be between 0.0 and 1.0.")

        self.min_hits = min_hits
        self.window_size = window_size
        self.match_iou = match_iou
        self._history: deque[list[Sequence[int]]] = deque(maxlen=window_size)

    def reset(self) -> None:
        self._history.clear()

    def update(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Record current frame detections and return only temporally confirmed ones."""
        boxes = [d["box"] for d in detections]
        self._history.append(boxes)

        if self.min_hits <= 1:
            return list(detections)

        confirmed: list[dict[str, Any]] = []
        for det in detections:
            hits = sum(
                1
                for frame_boxes in self._history
                if any(compute_iou(det["box"], prev) >= self.match_iou for prev in frame_boxes)
            )
            if hits >= self.min_hits:
                confirmed.append(det)
        return confirmed


class WeaponDetector:
    """
    Real-time weapon detection engine powered by fine-tuned YOLO11 weights.
    Supports both YOLO11x (production, native 800x800) and YOLO11m (baseline, 640x640).
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: str | None = None,
        fp16: bool | None = None,
        max_area_ratio: float = 0.75,
        nested_threshold: float = 0.85,
    ):
        """
        Initialize the weapon detector.

        Args:
            weights_path: Path to model weights (.pt). Defaults to YOLO11x if present, else YOLO11m.
            conf_threshold: Minimum confidence score in (0.0, 1.0] to trigger detection.
            iou_threshold: NMS IoU threshold in (0.0, 1.0] (default: 0.45).
            device: Computation device ('cuda:0', 'cpu', or None for auto-selection).
            fp16: Use FP16 half precision on CUDA. None auto-enables on CUDA GPUs.
            max_area_ratio: Reject bounding boxes covering more than this fraction of the frame.
            nested_threshold: Suppress lower-confidence sub-boxes nested inside larger boxes.
        """
        if not 0.0 < conf_threshold <= 1.0:
            raise ValueError("conf_threshold must be in (0.0, 1.0].")
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in (0.0, 1.0].")

        resolved_path = resolve_weights_path(weights_path)
        self.weights_path = str(resolved_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_area_ratio = max_area_ratio
        self.nested_threshold = nested_threshold

        import torch
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.fp16 = (fp16 if fp16 is not None else device.startswith("cuda")) and device.startswith("cuda")

        yolo_cls = _get_yolo_class()
        self.model = yolo_cls(self.weights_path)

        raw_names = getattr(self.model, "names", None)
        if isinstance(raw_names, dict):
            name_map = {int(k): str(v) for k, v in raw_names.items()}
        elif isinstance(raw_names, (list, tuple)):
            name_map = {idx: str(v) for idx, v in enumerate(raw_names)}
        else:
            name_map = {}

        if len(name_map) <= 1:
            # Enforce unified single-class surveillance mapping regardless of legacy class label
            if isinstance(raw_names, dict):
                self.model.names[0] = "weapon"
            self.class_names: dict[int, str] = {0: "weapon"}
            self.weapon_class_ids: set[int] = {0}
            self.distractor_class_ids: set[int] = set()
        else:
            self.class_names = dict(name_map)
            matched = {
                cid
                for cid, cname in name_map.items()
                if cname.strip().lower() not in DISTRACTOR_CLASS_NAMES
                and any(kw in cname.strip().lower() for kw in WEAPON_CLASS_KEYWORDS)
            }
            if not matched and 0 in name_map and name_map[0].strip().lower() not in DISTRACTOR_CLASS_NAMES:
                matched = {0}
            self.weapon_class_ids = matched
            self.distractor_class_ids = set(name_map.keys()) - self.weapon_class_ids

        self.default_imgsz = self._infer_default_imgsz(resolved_path)
        self._use_quantize: bool | None = None

    def _infer_default_imgsz(self, weights_path: Path) -> int:
        model_args = getattr(self.model, "args", None)
        if isinstance(model_args, dict):
            trained_sz = model_args.get("imgsz")
            if isinstance(trained_sz, int) and trained_sz >= 32:
                return trained_sz
        return 800 if "yolo11x" in weights_path.name.lower() else 640

    def predict(self, frame: np.ndarray, imgsz: int | None = None, iou: float | None = None):
        """
        Run inference on a single BGR numpy image frame.

        Args:
            frame: Non-empty BGR numpy array (H, W, C).
            imgsz: Spatial resolution (defaults to checkpoint's native resolution: 800 or 640).
            iou: Optional override for NMS IoU threshold.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Input frame must be a non-empty numpy array.")

        target_imgsz = imgsz if imgsz is not None else self.default_imgsz
        if target_imgsz < 32:
            raise ValueError("imgsz must be >= 32.")

        target_iou = iou if iou is not None else self.iou_threshold
        kwargs: dict[str, Any] = {
            "source": frame,
            "conf": self.conf_threshold,
            "iou": target_iou,
            "imgsz": target_imgsz,
            "device": self.device,
            "verbose": False,
        }
        if self.distractor_class_ids and self.weapon_class_ids:
            kwargs["classes"] = sorted(self.weapon_class_ids)
        if self.fp16:
            if self._use_quantize is None:
                self._use_quantize = _supports_quantize_arg()
            if self._use_quantize:
                kwargs["quantize"] = 16
            else:
                kwargs["half"] = True

        return self.model.predict(**kwargs)

    def _extract_raw_from_results(self, results) -> list[dict[str, Any]]:
        raw: list[dict[str, Any]] = []
        for r in results:
            if getattr(r, "boxes", None) is None:
                continue
            for box in r.boxes:
                if self.distractor_class_ids and getattr(box, "cls", None) is not None:
                    cls_id = int(float(box.cls[0]))
                    if cls_id not in self.weapon_class_ids:
                        continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                raw.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": float(box.conf[0]),
                    "class": "weapon",
                })
        return raw

    def extract_detections(self, results, frame_shape: tuple[int, ...]) -> list[dict[str, Any]]:
        """Parse and filter Ultralytics Results objects into clean detection dictionaries."""
        raw = self._extract_raw_from_results(results)
        return filter_detections(
            raw,
            frame_shape=frame_shape,
            max_area_ratio=self.max_area_ratio,
            nested_threshold=self.nested_threshold,
        )

    def predict_tiled(
        self,
        frame: np.ndarray,
        imgsz: int | None = None,
        iou: float | None = None,
        overlap: float = 0.25,
        merge_iou: float = 0.58,
        use_clahe: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Run full-frame + 2x2 overlapping tiled inference (SAHI) to recover densely
        piled or small distant weapons, followed by cross-tile NMS merge and filtering.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Input frame must be a non-empty numpy array.")

        infer_frame = apply_lab_clahe(frame) if use_clahe else frame
        fh, fw = frame.shape[:2]

        all_boxes: list[dict[str, Any]] = []
        all_boxes.extend(self._extract_raw_from_results(self.predict(infer_frame, imgsz=imgsz, iou=iou)))

        tile_w = min(fw, max(64, int(round(fw * (0.5 + 0.5 * overlap)))))
        tile_h = min(fh, max(64, int(round(fh * (0.5 + 0.5 * overlap)))))
        xs = [0] if tile_w >= fw else [0, fw - tile_w]
        ys = [0] if tile_h >= fh else [0, fh - tile_h]

        if len(xs) > 1 or len(ys) > 1:
            for y0 in ys:
                for x0 in xs:
                    crop = infer_frame[y0 : y0 + tile_h, x0 : x0 + tile_w]
                    for det_box in self._extract_raw_from_results(self.predict(crop, imgsz=imgsz, iou=iou)):
                        bx1, by1, bx2, by2 = det_box["box"]
                        all_boxes.append({
                            "box": [bx1 + x0, by1 + y0, bx2 + x0, by2 + y0],
                            "confidence": det_box["confidence"],
                            "class": det_box["class"],
                        })

        merged = nms_merge(all_boxes, iou_threshold=merge_iou)
        return filter_detections(
            merged,
            frame_shape=frame.shape,
            max_area_ratio=max(self.max_area_ratio, 0.85),
            nested_threshold=self.nested_threshold,
        )

    def annotate_frame(
        self,
        frame: np.ndarray,
        results_or_detections,
        box_color: tuple[int, int, int] = (0, 0, 255),
        show_conf: bool = False,
        show_track_id: bool = False,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """
        Draw detection bounding boxes and labels onto the frame.

        Accepts either Ultralytics Results or pre-filtered detection dictionaries.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Input frame must be a non-empty numpy array.")

        if isinstance(results_or_detections, list) and (
            not results_or_detections or isinstance(results_or_detections[0], dict)
        ):
            detections = filter_detections(
                results_or_detections,
                frame_shape=frame.shape,
                max_area_ratio=self.max_area_ratio,
                nested_threshold=self.nested_threshold,
            )
        else:
            detections = self.extract_detections(results_or_detections, frame.shape)

        annotated = frame.copy()
        frame_h, frame_w = annotated.shape[:2]
        long_edge = max(frame_h, frame_w)
        thickness = max(2, min(5, round(0.0028 * long_edge)))
        font_scale = max(0.50, min(1.0, long_edge / 1400.0))
        font_thickness = max(1, min(2, round(font_scale * 2.2)))

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            conf = det["confidence"]
            cls_name = det["class"]
            prefix = f"#{det['track_id']} " if show_track_id and det.get("track_id") is not None else ""

            label = f"{prefix}{cls_name.upper()} {conf:.2f}" if show_conf else f"{prefix}{cls_name.upper()}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)

            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            pad_y = 8
            pad_x = 6
            # Place badge above box when room permits; otherwise tuck cleanly inside top edge
            if y1 >= label_h + pad_y + 2:
                badge_top = y1 - label_h - pad_y
                badge_bottom = y1
            else:
                badge_top = y1
                badge_bottom = min(frame_h, y1 + label_h + pad_y)

            badge_right = min(frame_w, x1 + label_w + pad_x)
            cv2.rectangle(annotated, (x1, badge_top), (badge_right, badge_bottom), box_color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 3, badge_bottom - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA,
            )

        return annotated, detections
