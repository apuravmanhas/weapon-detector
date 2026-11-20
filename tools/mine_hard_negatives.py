#!/usr/bin/env python3
"""
Storage-Safe Hard-Negative Crop Miner.

Runs Stage-1 detection on weapon-free footage (webcam or video file) at a low
confidence threshold, extracts padded 256x256 JPEG false-positive crops, enforces
anti-duplicate throttling and strict crop/megabyte budgets, writes manifest.json,
and creates a single lightweight ZIP bundle ready for Google Colab training.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import zipfile
from typing import Any, Sequence

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.detector import WeaponDetector, compute_iou
from src.verifier import extract_padded_crop


@dataclass
class RecordedCrop:
    frame_idx: int
    label: str
    filename: str
    box: list[int]
    confidence: float
    bytes_written: int


class HardNegativeCropRecorder:
    """Manages duplicate suppression, 256x256 JPEG encoding, manifest, and disk budgets."""

    def __init__(
        self,
        out_dir: str | Path,
        label: str = "other",
        crop_size: int = 256,
        pad_ratio: float = 0.30,
        jpeg_quality: int = 88,
        min_gap: int = 5,
        dup_iou: float = 0.75,
        max_crops: int = 400,
        max_mb: float = 25.0,
    ):
        clean_label = "".join(ch if (ch.isalnum() or ch in ("_", "-")) else "_" for ch in label.strip())
        if not clean_label:
            raise ValueError("label must contain at least one alphanumeric character.")
        if crop_size < 32:
            raise ValueError("crop_size must be >= 32.")
        if max_crops < 1:
            raise ValueError("max_crops must be >= 1.")
        if max_mb <= 0.0:
            raise ValueError("max_mb must be > 0.")

        self.base_dir = Path(out_dir)
        self.label = clean_label
        self.label_dir = self.base_dir / self.label
        self.label_dir.mkdir(parents=True, exist_ok=True)

        self.crop_size = crop_size
        self.pad_ratio = pad_ratio
        self.jpeg_quality = jpeg_quality
        self.min_gap = max(1, min_gap)
        self.dup_iou = dup_iou
        self.max_crops = max_crops
        self.max_bytes = int(max_mb * 1024 * 1024)

        self.saved_count = 0
        self.total_bytes = 0
        self.skipped_duplicates = 0
        self._budget_hit = False
        self._recent_saved: list[tuple[int, Sequence[int]]] = []
        self.records: list[RecordedCrop] = []

    @property
    def budget_exhausted(self) -> bool:
        return self._budget_hit or self.saved_count >= self.max_crops or self.total_bytes >= self.max_bytes

    def _is_duplicate(self, frame_idx: int, box: Sequence[int]) -> bool:
        for prev_idx, prev_box in self._recent_saved:
            if (frame_idx - prev_idx) < self.min_gap and compute_iou(box, prev_box) > self.dup_iou:
                return True
        return False

    def record_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        detections: list[dict[str, Any]],
    ) -> list[Path]:
        """Extract and save non-duplicate crops for one frame until disk/count budget hits."""
        saved_paths: list[Path] = []
        if self.budget_exhausted:
            return saved_paths

        self._recent_saved = [
            (fidx, pbox) for (fidx, pbox) in self._recent_saved if (frame_idx - fidx) < self.min_gap
        ]

        for det in detections:
            if self.budget_exhausted:
                break

            box = [int(v) for v in det["box"]]
            if self._is_duplicate(frame_idx, box):
                self.skipped_duplicates += 1
                continue

            rgb_crop = extract_padded_crop(frame, box, pad_ratio=self.pad_ratio)
            bgr_crop = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2BGR)
            resized = cv2.resize(bgr_crop, (self.crop_size, self.crop_size), interpolation=cv2.INTER_AREA)

            ok, encoded = cv2.imencode(
                ".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, int(self.jpeg_quality)]
            )
            if not ok:
                continue

            payload = encoded.tobytes()
            if self.total_bytes + len(payload) > self.max_bytes:
                self._budget_hit = True
                break

            filename = f"{self.label}_f{frame_idx:06d}_{self.saved_count:04d}.jpg"
            dst = self.label_dir / filename
            dst.write_bytes(payload)

            self.saved_count += 1
            self.total_bytes += len(payload)
            self._recent_saved.append((frame_idx, box))
            self.records.append(
                RecordedCrop(
                    frame_idx=frame_idx,
                    label=self.label,
                    filename=f"{self.label}/{filename}",
                    box=box,
                    confidence=round(float(det["confidence"]), 4),
                    bytes_written=len(payload),
                )
            )
            saved_paths.append(dst)

            if (self.max_bytes - self.total_bytes) < 2048:
                break

        return saved_paths

    def finalize(self, zip_path: str | Path | None = None) -> tuple[Path, Path]:
        """Write manifest.json and package harvested crops into a Colab-ready zip archive."""
        manifest_path = self.base_dir / "manifest.json"
        manifest = {
            "label": self.label,
            "crop_size": self.crop_size,
            "pad_ratio": self.pad_ratio,
            "saved_count": self.saved_count,
            "skipped_duplicates": self.skipped_duplicates,
            "total_bytes": self.total_bytes,
            "records": [
                {
                    "frame_idx": r.frame_idx,
                    "label": r.label,
                    "file": r.filename,
                    "box": r.box,
                    "confidence": r.confidence,
                    "bytes": r.bytes_written,
                }
                for r in self.records
            ],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        bundle_path = Path(zip_path) if zip_path else (self.base_dir / "hard_negatives_upload.zip")
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(manifest_path, arcname="manifest.json")
            for jpg_file in sorted(self.base_dir.rglob("*.jpg")):
                zf.write(jpg_file, arcname=str(jpg_file.relative_to(self.base_dir)).replace("\\", "/"))

        return manifest_path, bundle_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Harvest hard false-positive crops from weapon-free camera or video footage."
    )
    parser.add_argument("--source", type=str, default="0", help="Webcam index ('0') or weapon-free video path.")
    parser.add_argument("--label", type=str, default="phone", help="Distractor class label (e.g., phone, umbrella, table_bg).")
    parser.add_argument("--weights", type=str, default=None, help="Stage-1 YOLO11 weights path.")
    parser.add_argument("--conf", type=float, default=0.20, help="Low confidence threshold to expose hard false positives.")
    parser.add_argument("--out-dir", type=str, default="mined_hard_negatives", help="Directory to store harvested crops.")
    parser.add_argument("--zip-path", type=str, default=None, help="Output ZIP path for Colab upload.")
    parser.add_argument("--min-gap", type=int, default=5, help="Minimum frame gap before saving an overlapping crop.")
    parser.add_argument("--dup-iou", type=float, default=0.75, help="IoU threshold for duplicate suppression within min-gap.")
    parser.add_argument("--max-crops", type=int, default=400, help="Maximum number of crops to save (default: 400).")
    parser.add_argument("--max-mb", type=float, default=25.0, help="Maximum disk budget in MB (default: 25.0).")
    parser.add_argument("--no-show", action="store_true", help="Disable live preview window.")
    parser.add_argument("--device", type=str, default=None, help="Computation device ('cuda:0', 'cpu', or None).")
    return parser.parse_args()


def main():
    args = parse_args()

    from detect import open_capture, sanitize_source_label

    safe_source = sanitize_source_label(args.source)
    if not args.source.isdigit() and "://" not in args.source and not Path(args.source).is_file():
        print(f"[Error] Video source not found: {safe_source}")
        sys.exit(1)

    detector = WeaponDetector(
        weights_path=args.weights,
        conf_threshold=args.conf,
        device=args.device,
    )
    recorder = HardNegativeCropRecorder(
        out_dir=args.out_dir,
        label=args.label,
        min_gap=args.min_gap,
        dup_iou=args.dup_iou,
        max_crops=args.max_crops,
        max_mb=args.max_mb,
    )

    cap = open_capture(args.source)
    if not cap or not cap.isOpened():
        print(f"[Error] Could not open source: {safe_source}")
        sys.exit(1)

    print(
        f"[Miner] Harvesting '{recorder.label}' false positives from {safe_source} "
        f"(conf>={args.conf}, max_crops={args.max_crops}, max_mb={args.max_mb} MB)"
    )
    print("[Miner] Hold/move distractors (phones, umbrellas, tables) — press 'q' to finish early.")

    frame_idx = 0
    try:
        while not recorder.budget_exhausted:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            results = detector.predict(frame)
            detections = detector.extract_detections(results, frame.shape)
            recorder.record_frame(frame, frame_idx, detections)
            frame_idx += 1

            if not args.no_show:
                annotated, _ = detector.annotate_frame(frame, detections, show_conf=True)
                hud = (
                    f"Mined [{recorder.label}]: {recorder.saved_count}/{args.max_crops} "
                    f"| {recorder.total_bytes / 1e6:.2f}/{args.max_mb:.1f} MB"
                )
                cv2.putText(annotated, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2, cv2.LINE_AA)
                cv2.imshow("Hard-Negative Miner (Press q to stop)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if not args.no_show:
            cv2.destroyAllWindows()

    manifest_path, bundle_path = recorder.finalize(zip_path=args.zip_path)
    print("\n[Miner Summary]")
    print(f"  Frames scanned:     {frame_idx}")
    print(f"  Saved crops:        {recorder.saved_count}")
    print(f"  Skipped duplicates: {recorder.skipped_duplicates}")
    print(f"  Total size:         {recorder.total_bytes / 1e6:.2f} MB")
    print(f"  Manifest:           {manifest_path}")
    print(f"  Colab Upload ZIP:   {bundle_path}")


if __name__ == "__main__":
    main()
