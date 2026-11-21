"""
Semi-automated temporal feature extractor for Causal TCN / GRU training on Google Colab.

Runs Stage-1 + Stage-2 + Pose + Association + 30-D KinematicFeatureExtractor locally on every
annotated clip in a manifest and saves lightweight per-pair `.npz` sequences (<10 MB total
instead of gigabytes of raw video) with frame-aligned ground-truth labels:
  - state_targets:   0=PASSIVE_PRESENT, 1=DRAWING_READYING, 2=AIMING_IMMINENT
  - risk_targets:    smooth monotonic target ramping toward 1.0 as t -> t_first_shot
  - time_to_shot_s:  seconds remaining until t_first_shot (large value for passive clips)
"""

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.schema import ClipAnnotation, load_manifest
from src.features import FEATURE_NAMES
from src.pipeline import PreFireIntentPipeline


def _label_timestamp(ann: ClipAnnotation, ts_ms: float) -> tuple[int, float, float]:
    """Return (state_target in {0,1,2}, risk_target in [0,1], time_to_shot_s)."""
    if ann.bucket != "draw_to_fire" or ann.t_first_shot_ms is None:
        return 0, 0.05, 99.0

    t_shot = ann.t_first_shot_ms
    t_aim = ann.t_aim_start_ms if ann.t_aim_start_ms is not None else max(0.0, t_shot - 450.0)
    t_draw = ann.t_draw_start_ms if ann.t_draw_start_ms is not None else max(0.0, t_aim - 500.0)

    tts_s = (t_shot - ts_ms) / 1000.0

    if ts_ms < t_draw:
        return 0, 0.08, max(0.0, tts_s)
    if ts_ms < t_aim:
        frac = (ts_ms - t_draw) / max(1.0, t_aim - t_draw)
        return 1, float(0.45 + 0.25 * frac), max(0.0, tts_s)
    if ts_ms <= t_shot + 400.0:
        frac = min(1.0, max(0.0, (ts_ms - t_aim) / max(1.0, t_shot - t_aim)))
        return 2, float(0.78 + 0.22 * frac), max(0.0, tts_s)
    # Post-shot frames: still elevated / active firing window
    return 2, 0.90, -1.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract compact .npz kinematic sequences for Colab training")
    p.add_argument("--manifest", type=str, required=True, help="Path to annotated manifest JSON")
    p.add_argument("--out-dir", type=str, default="data/sequences", help="Output folder for .npz sequence files")
    p.add_argument("--weapon-weights", type=str, default=None)
    p.add_argument("--verifier-weights", type=str, default=None)
    p.add_argument("--pose-weights", type=str, default="yolo11s-pose.pt")
    p.add_argument("--min-seq-len", type=int, default=8, help="Minimum frames required to export a pair sequence")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    clips = load_manifest(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = PreFireIntentPipeline(
        weapon_weights=args.weapon_weights,
        verifier_weights=args.verifier_weights,
        pose_weights=args.pose_weights,
        device=args.device,
        sync_timing=False,
    )

    exported = 0
    for idx, ann in enumerate(clips, start=1):
        vid_path = Path(ann.video_path)
        if not vid_path.is_absolute() and not vid_path.is_file():
            vid_path = REPO_ROOT / ann.video_path
        if not vid_path.is_file():
            raise FileNotFoundError(f"Missing video for '{ann.clip_id}': {ann.video_path}")

        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {vid_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if fps <= 1.0 or fps > 240.0:
            fps = 30.0

        pipeline.reset()
        per_pair: dict[str, dict[str, list]] = {}
        f_idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                ts_s = f_idx / fps
                tel = pipeline.process_frame(frame, timestamp_s=ts_s, fps_fallback=fps)
                for a in tel.assessments:
                    if a.person_id is None:
                        continue
                    vec = np.array([a.features[name] for name in FEATURE_NAMES], dtype=np.float32)
                    st_tgt, rk_tgt, tts_s = _label_timestamp(ann, tel.timestamp_ms)
                    rec = per_pair.setdefault(
                        a.pair_key,
                        {"features": [], "timestamps_ms": [], "state_targets": [], "risk_targets": [], "time_to_shot_s": []},
                    )
                    rec["features"].append(vec)
                    rec["timestamps_ms"].append(tel.timestamp_ms)
                    rec["state_targets"].append(st_tgt)
                    rec["risk_targets"].append(rk_tgt)
                    rec["time_to_shot_s"].append(tts_s)
                f_idx += 1
        finally:
            cap.release()

        clip_saved = 0
        for pair_key, rec in per_pair.items():
            if len(rec["features"]) < args.min_seq_len:
                continue
            save_path = out_dir / f"{ann.clip_id}__{pair_key}.npz"
            np.savez_compressed(
                save_path,
                features=np.stack(rec["features"], axis=0).astype(np.float32),
                timestamps_ms=np.asarray(rec["timestamps_ms"], dtype=np.float32),
                state_targets=np.asarray(rec["state_targets"], dtype=np.int64),
                risk_targets=np.asarray(rec["risk_targets"], dtype=np.float32),
                time_to_shot_s=np.asarray(rec["time_to_shot_s"], dtype=np.float32),
                clip_id=np.asarray(ann.clip_id),
                split=np.asarray(ann.split),
                bucket=np.asarray(ann.bucket),
                feature_names=np.asarray(FEATURE_NAMES),
            )
            exported += 1
            clip_saved += 1

        print(f"[{idx}/{len(clips)}] {ann.clip_id} ({ann.bucket}, split={ann.split}) -> exported {clip_saved} sequence(s)")

    print(f"\n[Done] Exported {exported} sequence file(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
