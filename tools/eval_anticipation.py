"""
Automated benchmark runner for Pre-Firing Weapon Intent & Action Anticipation.

Evaluates a manifest JSON of timestamped clips across all three buckets:
  - draw_to_fire     -> Pre-shot alert rate & Net Effective Lead Time (minus inference latency)
  - passive_carry    -> Passive-weapon false escalation rate
  - non_weapon_mimic -> Non-weapon distractor false alarm rate (FAR/hr)
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.metrics import evaluate_clip_telemetry, format_markdown_report, summarize_benchmark
from benchmark.schema import load_manifest
from src.pipeline import PreFireIntentPipeline
from src.states import FrameTelemetry


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate pre-firing anticipation benchmark manifest")
    p.add_argument("--manifest", type=str, required=True, help="Path to manifest JSON")
    p.add_argument("--split", type=str, default=None, help="Optional split filter ('eval', 'val', 'train')")
    p.add_argument("--weapon-weights", type=str, default=None)
    p.add_argument("--weapon-imgsz", type=int, default=None)
    p.add_argument("--verifier-weights", type=str, default=None)
    p.add_argument("--no-verifier", action="store_true")
    p.add_argument("--pose-weights", type=str, default="yolo11s-pose.pt")
    p.add_argument("--pose-imgsz", type=int, default=640)
    p.add_argument("--temporal-weights", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="outputs/eval")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    clips = load_manifest(manifest_path)
    if args.split:
        clips = [c for c in clips if c.split == args.split]
    if not clips:
        raise RuntimeError(f"No clips matched in {manifest_path} (split={args.split}).")

    pipeline = PreFireIntentPipeline(
        weapon_weights=args.weapon_weights,
        weapon_imgsz=args.weapon_imgsz,
        verifier_weights=None if args.no_verifier else args.verifier_weights,
        pose_weights=args.pose_weights,
        pose_imgsz=args.pose_imgsz,
        temporal_weights=args.temporal_weights,
        device=args.device,
    )
    if args.no_verifier:
        pipeline.verifier = None

    clip_results = []
    all_latencies: list[float] = []

    for idx, ann in enumerate(clips, start=1):
        vid_path = Path(ann.video_path)
        if not vid_path.is_absolute() and not vid_path.is_file():
            vid_path = REPO_ROOT / ann.video_path
        if not vid_path.is_file():
            raise FileNotFoundError(f"Video file missing for clip '{ann.clip_id}': {ann.video_path}")

        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open clip '{ann.clip_id}': {vid_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if fps <= 1.0 or fps > 240.0:
            fps = 30.0

        pipeline.reset()
        frames_tel: list[FrameTelemetry] = []
        f_idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                tel = pipeline.process_frame(frame, timestamp_s=f_idx / fps, fps_fallback=fps)
                frames_tel.append(tel)
                if f_idx > 0:
                    all_latencies.append(tel.latencies.total_ms)
                f_idx += 1
        finally:
            cap.release()

        res = evaluate_clip_telemetry(ann, frames_tel)
        clip_results.append(res)
        print(
            f"[{idx}/{len(clips)}] {ann.clip_id} ({ann.bucket}) | frames={res.num_frames} | "
            f"net_lead_draw={res.net_lead_draw_ms} ms | net_lead_aim={res.net_lead_aim_ms} ms | "
            f"false_aim_frames={res.false_aim_frames} | p95={res.p95_latency_ms:.1f} ms"
        )

    summary = summarize_benchmark(clip_results, all_frame_latencies_ms=all_latencies)
    md_report = format_markdown_report(summary)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "intent_mode": pipeline.intent_mode,
                "summary": summary,
                "clips": [asdict(r) for r in clip_results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text(md_report + "\n", encoding="utf-8")

    print("\n" + md_report)
    print(f"\n[Saved] {out_dir / 'summary.json'} and {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
