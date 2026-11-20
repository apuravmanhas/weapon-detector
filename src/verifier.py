from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models

from src.detector import is_lfs_pointer

DEFAULT_IMAGENET_MEAN = (0.485, 0.456, 0.406)
DEFAULT_IMAGENET_STD = (0.229, 0.224, 0.225)


def extract_padded_crop(
    frame: np.ndarray,
    box: Sequence[int | float],
    pad_ratio: float = 0.30,
) -> np.ndarray:
    """
    Extract an RGB crop from a full-resolution frame with symmetric context padding,
    clamped safely to image boundaries.
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("Input frame must be a non-empty numpy array.")
    if pad_ratio < 0.0:
        raise ValueError("pad_ratio must be >= 0.0.")

    frame_h, frame_w = frame.shape[:2]
    x1 = max(0, min(frame_w, int(round(float(box[0])))))
    y1 = max(0, min(frame_h, int(round(float(box[1])))))
    x2 = max(0, min(frame_w, int(round(float(box[2])))))
    y2 = max(0, min(frame_h, int(round(float(box[3])))))

    if x2 <= x1:
        x1 = max(0, min(frame_w - 1, x1))
        x2 = min(frame_w, x1 + 1)
    if y2 <= y1:
        y1 = max(0, min(frame_h - 1, y1))
        y2 = min(frame_h, y1 + 1)

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = int(round(bw * pad_ratio))
    pad_y = int(round(bh * pad_ratio))

    px1 = max(0, x1 - pad_x)
    py1 = max(0, y1 - pad_y)
    px2 = min(frame_w, x2 + pad_x)
    py2 = min(frame_h, y2 + pad_y)

    crop = frame[py1:py2, px1:px2]
    if crop.ndim == 2:
        return cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    if crop.shape[2] == 1:
        return cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def build_classifier(arch: str, num_classes: int) -> nn.Module:
    """Instantiate a torchvision backbone with weights=None and an adapted linear head."""
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2.")

    arch_key = arch.lower().strip()
    if arch_key == "convnext_tiny":
        model = tv_models.convnext_tiny(weights=None)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
        return model
    if arch_key == "convnext_small":
        model = tv_models.convnext_small(weights=None)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
        return model
    if arch_key == "mobilenet_v3_large":
        model = tv_models.mobilenet_v3_large(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(
        f"Unsupported verifier architecture '{arch}'. "
        "Supported: convnext_tiny, convnext_small, mobilenet_v3_large."
    )


class CropVerifier:
    """
    Stage-2 native-resolution crop classifier that verifies candidate Stage-1 bounding
    boxes against trained weapon and hard-distractor classes (e.g., phone, umbrella, table).
    """

    def __init__(
        self,
        weights_path: str | Path,
        device: str | None = None,
        fp16: bool | None = None,
        pad_ratio: float = 0.30,
    ):
        ckpt_path = Path(weights_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Verifier checkpoint not found at: {weights_path}")
        if is_lfs_pointer(ckpt_path):
            raise RuntimeError(
                f"Verifier checkpoint '{ckpt_path}' is a Git LFS pointer, not binary weights. "
                "Run 'git lfs pull' to download the model."
            )

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.fp16 = bool(fp16 if fp16 is not None else device.startswith("cuda")) and device.startswith("cuda")
        self.pad_ratio = pad_ratio
        self.weights_path = str(ckpt_path)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if not isinstance(ckpt, dict) or "state_dict" not in ckpt or "classes" not in ckpt:
            raise ValueError(
                "Invalid verifier checkpoint format: expected dict containing 'arch', "
                "'classes', and 'state_dict'."
            )

        self.arch = str(ckpt.get("arch", "convnext_tiny"))
        self.classes = [str(c) for c in ckpt["classes"]]
        if len(self.classes) < 2:
            raise ValueError("Verifier checkpoint must define at least 2 classes.")

        raw_weapon_classes = ckpt.get("weapon_classes")
        if raw_weapon_classes:
            self.weapon_classes = {str(c) for c in raw_weapon_classes}
        else:
            default_pos = [c for c in self.classes if "weapon" in c.lower() or c.lower() in {"handgun", "long_gun", "rifle", "pistol", "knife", "blade"}]
            self.weapon_classes = set(default_pos) if default_pos else {self.classes[0]}

        self._weapon_indices = [i for i, c in enumerate(self.classes) if c in self.weapon_classes]
        if not self._weapon_indices:
            raise ValueError(
                f"None of weapon_classes={sorted(self.weapon_classes)} found in classes={self.classes}."
            )

        self.img_size = int(ckpt.get("img_size", 224))
        mean = tuple(float(v) for v in ckpt.get("mean", DEFAULT_IMAGENET_MEAN))
        std = tuple(float(v) for v in ckpt.get("std", DEFAULT_IMAGENET_STD))

        self.model = build_classifier(self.arch, len(self.classes))
        self.model.load_state_dict(ckpt["state_dict"], strict=True)
        self.model.to(self.device).eval()
        if self.fp16:
            self.model.half()

        dtype = torch.float16 if self.fp16 else torch.float32
        self._mean = torch.tensor(mean, dtype=dtype, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(std, dtype=dtype, device=self.device).view(1, 3, 1, 1)

    def score_crops(self, crops: list[np.ndarray]) -> list[tuple[float, str]]:
        """
        Score a batch of RGB numpy crops.

        Returns list of (p_weapon, top_class_label) tuples.
        """
        if not crops:
            return []

        resized = [
            cv2.resize(crop, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            for crop in crops
        ]
        batch_np = np.stack(resized, axis=0)  # (N, H, W, 3) uint8 RGB
        tensor = torch.from_numpy(batch_np).permute(0, 3, 1, 2).to(self.device, non_blocking=True)
        tensor = tensor.to(dtype=torch.float16 if self.fp16 else torch.float32) / 255.0
        tensor = (tensor - self._mean) / self._std

        with torch.no_grad():
            logits = self.model(tensor).float()
            probs = torch.softmax(logits, dim=-1)

        probs_np = probs.cpu().numpy()
        results: list[tuple[float, str]] = []
        for row in probs_np:
            p_weapon = float(np.clip(np.sum(row[self._weapon_indices]), 0.0, 1.0))
            top_idx = int(np.argmax(row))
            results.append((p_weapon, self.classes[top_idx]))
        return results

    def verify_detections(
        self,
        frame: np.ndarray,
        detections: list[dict[str, Any]],
        verify_below_conf: float = 0.85,
        threshold: float = 0.50,
        filter_out: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Run Stage-2 verification only on Stage-1 detections with confidence < verify_below_conf.
        Attaches 'verifier_prob' and 'verifier_label' to scored detections and optionally
        filters out candidates scoring below threshold.
        """
        if not detections:
            return []

        cloned = [dict(d) for d in detections]
        eval_indices: list[int] = []
        crops: list[np.ndarray] = []

        for idx, det in enumerate(cloned):
            if float(det["confidence"]) < verify_below_conf:
                eval_indices.append(idx)
                crops.append(extract_padded_crop(frame, det["box"], pad_ratio=self.pad_ratio))

        if eval_indices:
            scores = self.score_crops(crops)
            for idx, (p_weapon, top_label) in zip(eval_indices, scores):
                cloned[idx]["verifier_prob"] = round(p_weapon, 4)
                cloned[idx]["verifier_label"] = top_label

        if not filter_out:
            return cloned

        kept: list[dict[str, Any]] = []
        for det in cloned:
            if float(det["confidence"]) >= verify_below_conf:
                kept.append(det)
            elif float(det.get("verifier_prob", 0.0)) >= threshold:
                kept.append(det)
        return kept
