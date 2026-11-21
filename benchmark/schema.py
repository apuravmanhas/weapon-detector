"""Strict manifest schema for pre-firing weapon anticipation benchmarks."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

BucketType = Literal["draw_to_fire", "passive_carry", "non_weapon_mimic"]
VALID_BUCKETS: set[str] = {"draw_to_fire", "passive_carry", "non_weapon_mimic"}


@dataclass
class ClipAnnotation:
    """Ground-truth temporal annotation for one evaluation or training video clip."""

    clip_id: str
    video_path: str
    bucket: BucketType
    split: str = "eval"  # "train" | "val" | "eval"
    camera_domain: str = "cctv"  # "cctv" | "range_eye_level" | "webcam" | "elevated"
    t_draw_start_ms: float | None = None
    t_aim_start_ms: float | None = None
    t_first_shot_ms: float | None = None
    notes: str = ""

    def validate(self) -> None:
        if not self.clip_id.strip():
            raise ValueError("clip_id must be non-empty.")
        if self.bucket not in VALID_BUCKETS:
            raise ValueError(f"Invalid bucket '{self.bucket}' for '{self.clip_id}'. Expected one of {sorted(VALID_BUCKETS)}.")
        if self.bucket == "draw_to_fire":
            if self.t_first_shot_ms is None or self.t_first_shot_ms <= 0.0:
                raise ValueError(f"draw_to_fire clip '{self.clip_id}' requires t_first_shot_ms > 0.")
            if self.t_draw_start_ms is not None and self.t_draw_start_ms > self.t_first_shot_ms:
                raise ValueError(f"Clip '{self.clip_id}': t_draw_start_ms cannot exceed t_first_shot_ms.")
            if self.t_aim_start_ms is not None and self.t_aim_start_ms > self.t_first_shot_ms:
                raise ValueError(f"Clip '{self.clip_id}': t_aim_start_ms cannot exceed t_first_shot_ms.")


def load_manifest(manifest_path: str | Path) -> list[ClipAnnotation]:
    p = Path(manifest_path)
    if not p.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    data = json.loads(p.read_text(encoding="utf-8"))
    raw_clips = data["clips"] if isinstance(data, dict) and "clips" in data else data
    if not isinstance(raw_clips, list):
        raise ValueError("Manifest JSON must contain a list under 'clips' or be a top-level list.")

    seen_ids: set[str] = set()
    clips: list[ClipAnnotation] = []
    for item in raw_clips:
        ann = ClipAnnotation(
            clip_id=str(item["clip_id"]),
            video_path=str(item["video_path"]),
            bucket=item["bucket"],
            split=str(item.get("split", "eval")),
            camera_domain=str(item.get("camera_domain", "cctv")),
            t_draw_start_ms=float(item["t_draw_start_ms"]) if item.get("t_draw_start_ms") is not None else None,
            t_aim_start_ms=float(item["t_aim_start_ms"]) if item.get("t_aim_start_ms") is not None else None,
            t_first_shot_ms=float(item["t_first_shot_ms"]) if item.get("t_first_shot_ms") is not None else None,
            notes=str(item.get("notes", "")),
        )
        ann.validate()
        if ann.clip_id in seen_ids:
            raise ValueError(f"Duplicate clip_id in manifest: '{ann.clip_id}'.")
        seen_ids.add(ann.clip_id)
        clips.append(ann)
    return clips
