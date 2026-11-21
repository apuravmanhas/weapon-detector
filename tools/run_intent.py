"""
CLI for running real-time Pre-Firing Weapon Intent & Action Anticipation on a webcam or video.

Outputs:
  - Optional annotated MP4 video (clean color-coded readiness boxes + bearer skeleton + latency bar)
  - Optional per-frame JSON telemetry log
  - Exact CUDA-synchronized per-stage latency summary (Mean / P50 / P95 / P99 ms)
"""

import argparse
import json
from pathlib import Path
import sys
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline import PreFireIntentPipeline
from src.states import ThreatState
from src.visualizer import render_telemetry_frame


def sanitize_source_label(source: str) -> str:
    """Mask embedded credentials in RTSP/HTTP stream URLs before logging."""
    if "://" not in source:
        return source
    parsed = urlsplit(source)
    if parsed.username is None and parsed.password is None:
        return source
    host = parsed.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    port = f":{parsed.port}" if parsed.port else ""
    masked_netloc = f"***:***@{host}{port}"
    return urlunsplit((parsed.scheme, masked_netloc, parsed.path, parsed.query, parsed.fragment))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-time Pre-Firing Weapon Intent & Action Anticipation")
    p.add_argument("--source", type=str, default="0", help="Camera index ('0') or path to video file")
    p.add_argument("--weapon-weights", type=str, default=None, help="Stage-1 YOLO weapon weights (.pt)")
    p.add_argument("--verifier-weights", type=str, default=None, help="Stage-2 ConvNeXt crop verifier (.pt)")
    p.add_argument("--no-verifier", action="store_true", help="Disable Stage-2 crop verifier")
    p.add_argument("--pose-weights", type=str, default="yolo11s-pose.pt", help="YOLO11 pose model weights")
    p.add_argument("--temporal-weights", type=str, default=None, help="Optional trained Causal TCN/GRU checkpoint")
    p.add_argument("--weapon-conf", type=float, default=0.20, help="Stage-1 weapon confidence threshold")
    p.add_argument("--weapon-imgsz", type=int, default=None, help="Stage-1 input resolution (default: model native)")
    p.add_argument("--pose-imgsz", type=int, default=640, help="Pose model input resolution")
    p.add_argument("--max-dim", type=int, default=None, help="Optional max long-edge resolution (even dimensions)")
    p.add_argument("--stride", type=int, default=1, help="Process every N-th frame (adjusts output FPS cleanly)")
    p.add_argument("--min-hits", type=int, default=2, help="Weapon tracker temporal persistence min hits")
    p.add_argument("--window", type=int, default=5, help="Weapon tracker sliding window")
    p.add_argument("--no-static-suppress", action="store_true", help="Disable static background suppression")
    p.add_argument("--save", type=str, default=None, help="Output annotated MP4 path")
    p.add_argument("--json-out", type=str, default=None, help="Optional JSON telemetry output path")
    p.add_argument("--no-show", action="store_true", help="Run headless without OpenCV display window")
    p.add_argument("--no-skeleton", action="store_true", help="Hide bearer upper-body skeleton overlay")
    p.add_argument("--no-hud", action="store_true", help="Hide bottom telemetry bar")
    p.add_argument("--pose-always", action="store_true", help="Run pose on every frame even when no weapon is detected")
    p.add_argument("--start-frame", type=int, default=0, help="Optional starting frame index for video sources")
    p.add_argument("--shot-ms", type=float, default=None, help="Optional known first-shot timestamp (ms) for T-minus HUD")
    p.add_argument("--max-frames", type=int, default=None, help="Optional cap on processed frames")
    p.add_argument("--device", type=str, default=None, help="Device override ('cuda:0' or 'cpu')")
    return p.parse_args()


def _fmt_pcts(vals: list[float]) -> str:
    arr = np.asarray(vals, dtype=np.float64)
    return (
        f"mean={np.mean(arr):5.2f} ms | "
        f"p50={np.percentile(arr, 50):5.2f} ms | "
        f"p95={np.percentile(arr, 95):5.2f} ms | "
        f"p99={np.percentile(arr, 99):5.2f} ms"
    )


def main() -> int:
    args = parse_args()
    source: int | str = int(args.source) if args.source.isdigit() else args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {sanitize_source_label(args.source)}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1.0 or fps > 240.0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if isinstance(source, str) else -1

    stride = max(1, int(args.stride))
    out_w, out_h = width, height
    if args.max_dim is not None and args.max_dim > 0 and max(width, height) > args.max_dim:
        scale = float(args.max_dim) / float(max(width, height))
        out_w = max(2, (int(round(width * scale)) // 2) * 2)
        out_h = max(2, (int(round(height * scale)) // 2) * 2)
    out_fps = fps / stride

    pipeline = PreFireIntentPipeline(
        weapon_weights=args.weapon_weights,
        verifier_weights=None if args.no_verifier else args.verifier_weights,
        pose_weights=args.pose_weights,
        temporal_weights=args.temporal_weights,
        weapon_conf=args.weapon_conf,
        weapon_imgsz=args.weapon_imgsz,
        pose_imgsz=args.pose_imgsz,
        min_weapon_hits=args.min_hits,
        weapon_window=args.window,
        static_suppress=not args.no_static_suppress,
        device=args.device,
        pose_gated=not args.pose_always,
    )
    if args.no_verifier:
        pipeline.verifier = None

    if isinstance(source, str) and args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    print(
        f"[Info] Weapon Stage-1: {Path(pipeline.weapon_detector.weights_path).name} "
        f"(imgsz={pipeline.weapon_detector.imgsz}, device={pipeline.device}, fp16={pipeline.weapon_detector.fp16})"
    )
    print(
        f"[Info] Stage-2 Verifier: {Path(pipeline.verifier.weights_path).name if pipeline.verifier else 'DISABLED'}"
    )
    print(f"[Info] Pose Backbone: {args.pose_weights} (imgsz={args.pose_imgsz}) | Intent Mode: {pipeline.intent_mode}")
    print(
        f"[Info] Source: {sanitize_source_label(args.source)} ({width}x{height} @ {fps:.2f} FPS, frames={total_frames}) -> "
        f"proc={out_w}x{out_h} @ {out_fps:.2f} FPS (stride={stride})",
        flush=True,
    )

    writer = None
    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (out_w, out_h))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Failed to create VideoWriter at: {args.save}")

    stage_logs: dict[str, list[float]] = {
        "weapon_stage1": [],
        "verifier_stage2": [],
        "weapon_track": [],
        "pose": [],
        "association": [],
        "features": [],
        "intent": [],
        "total": [],
    }
    state_counts = {s.label: 0 for s in ThreatState}
    telemetry_rows: list[dict] = []
    first_draw_ms: float | None = None
    first_aim_ms: float | None = None
    first_draw_lat_ms: float = 0.0
    first_aim_lat_ms: float = 0.0

    raw_idx = 0
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            cur_raw = raw_idx
            raw_idx += 1
            if stride > 1 and (cur_raw % stride) != 0:
                continue
            if (out_w, out_h) != (width, height):
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

            ts_s = (args.start_frame + cur_raw) / fps
            tel = pipeline.process_frame(frame, timestamp_s=ts_s, fps_fallback=out_fps)

            state_counts[tel.max_state.label] += 1
            stage_logs["weapon_stage1"].append(tel.latencies.weapon_stage1_ms)
            stage_logs["verifier_stage2"].append(tel.latencies.verifier_stage2_ms)
            stage_logs["weapon_track"].append(tel.latencies.weapon_track_ms)
            stage_logs["pose"].append(tel.latencies.pose_ms)
            stage_logs["association"].append(tel.latencies.association_ms)
            stage_logs["features"].append(tel.latencies.features_ms)
            stage_logs["intent"].append(tel.latencies.intent_ms)
            stage_logs["total"].append(tel.latencies.total_ms)

            if first_draw_ms is None and tel.max_state >= ThreatState.DRAWING_READYING:
                first_draw_ms = tel.timestamp_ms
                first_draw_lat_ms = tel.latencies.total_ms
            if first_aim_ms is None and tel.max_state >= ThreatState.AIMING_IMMINENT:
                first_aim_ms = tel.timestamp_ms
                first_aim_lat_ms = tel.latencies.total_ms

            if args.json_out:
                telemetry_rows.append(
                    {
                        "frame_index": tel.frame_index,
                        "timestamp_ms": round(tel.timestamp_ms, 2),
                        "max_state": tel.max_state.label,
                        "max_risk_score": tel.max_risk_score,
                        "total_latency_ms": round(tel.latencies.total_ms, 3),
                        "assessments": [
                            {
                                "pair_key": a.pair_key,
                                "state": a.state.label,
                                "risk_score": a.risk_score,
                                "reasons": a.reasons,
                                "weapon_box": a.weapon_box,
                                "person_id": a.person_id,
                            }
                            for a in tel.assessments
                        ],
                    }
                )

            if writer is not None or not args.no_show:
                countdown = (args.shot_ms - tel.timestamp_ms) if args.shot_ms is not None else None
                vis = render_telemetry_frame(
                    frame,
                    tel,
                    show_skeleton=not args.no_skeleton,
                    show_hud=not args.no_hud,
                    lead_countdown_ms=countdown,
                )
                if writer is not None:
                    writer.write(vis)
                if not args.no_show:
                    cv2.imshow("Pre-Firing Intent Anticipation", vis)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break

            frame_idx += 1
            if frame_idx % 300 == 0:
                lat_slice = stage_logs["total"][1:] if len(stage_logs["total"]) > 1 else stage_logs["total"]
                print(
                    f"[Progress] processed={frame_idx} (t={ts_s:.1f}s) | state={tel.max_state.label} | "
                    f"mean_lat={np.mean(lat_slice):.1f} ms",
                    flush=True,
                )
            if args.max_frames is not None and frame_idx >= args.max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_show:
            cv2.destroyAllWindows()

    if args.json_out:
        jp = Path(args.json_out)
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps({"frames": telemetry_rows}, indent=2), encoding="utf-8")
        print(f"[Saved] Telemetry JSON: {jp}")
    if args.save:
        print(f"[Saved] Annotated video: {args.save}")

    print("\n[Summary]")
    print(f"  Processed frames: {frame_idx}")
    print(f"  State counts:     {state_counts}")
    if stage_logs["total"]:
        sl = slice(1, None) if len(stage_logs["total"]) > 5 else slice(None)
        print("\n[Steady-State Latency Breakdown (CUDA-synced)]")
        for key in ("weapon_stage1", "verifier_stage2", "pose", "weapon_track", "association", "features", "intent", "total"):
            print(f"  {key:16s}: {_fmt_pcts(stage_logs[key][sl])}")
        avg_fps = 1000.0 / max(1e-3, float(np.mean(stage_logs["total"][sl])))
        print(f"  Steady-State FPS: {avg_fps:.1f}")

    if args.shot_ms is not None:
        print(f"\n[Lead Time vs t_first_shot={args.shot_ms:.1f} ms]")
        if first_draw_ms is not None:
            net_d = args.shot_ms - (first_draw_ms + first_draw_lat_ms)
            print(f"  First DRAWING+: t={first_draw_ms:.1f} ms | Net Lead Time = {net_d:.1f} ms")
        else:
            print("  First DRAWING+: NOT TRIGGERED")
        if first_aim_ms is not None:
            net_a = args.shot_ms - (first_aim_ms + first_aim_lat_ms)
            print(f"  First AIMING:   t={first_aim_ms:.1f} ms | Net Lead Time = {net_a:.1f} ms")
        else:
            print("  First AIMING:   NOT TRIGGERED")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
