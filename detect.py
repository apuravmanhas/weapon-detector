#!/usr/bin/env python3
"""
Real-Time Weapon Detection CLI
Runs inference on webcam feeds, RTSP streams, video files, or images using YOLO11 weights.
"""

import argparse
import os
import sys
import time
from urllib.parse import urlsplit, urlunsplit

import cv2

from src.detector import TemporalPersistenceFilter, WeaponDetector

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-Time Weapon Detection in Surveillance Feeds (YOLO11x / YOLO11m)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Input source: camera index ('0'), stream URL, video path, or image path.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to .pt model weights (defaults to YOLO11x if present, else YOLO11m).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold for weapon detection (default: 0.35).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold to merge overlapping boxes (default: 0.45).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Inference spatial resolution (auto-detects 800 for YOLO11x, 640 for YOLO11m).",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable automatic FP16 half precision on CUDA GPUs.",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=1,
        help="Minimum matching frames in sliding window to confirm a stream threat (default: 1).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Sliding window length (frames) for --min-frames temporal filter (default: 5).",
    )
    parser.add_argument(
        "--show-conf",
        action="store_true",
        help="Display confidence score alongside detection labels.",
    )
    parser.add_argument(
        "--show-fps",
        action="store_true",
        help="Render live throughput (FPS) overlay on video frames.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save annotated output image or video.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Run headless without displaying preview GUI window.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device ('cuda:0', 'cpu', or None for auto-selection).",
    )
    return parser.parse_args()


def _ensure_parent_dir(file_path: str) -> None:
    parent = os.path.dirname(os.path.abspath(file_path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def open_capture(source: str):
    """Open video capture with multi-backend fallback for camera streams."""
    if source.isdigit():
        cam_idx = int(source)
        # DirectShow avoids MSMF deadlocks on Windows UVC webcams
        if sys.platform.startswith("win"):
            for api in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
                cap = cv2.VideoCapture(cam_idx, api)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        return cap
                    cap.release()
        return cv2.VideoCapture(cam_idx)
    return cv2.VideoCapture(source)


def run_image_inference(detector: WeaponDetector, args) -> None:
    frame = cv2.imread(args.source)
    if frame is None:
        print(f"[Error] Could not read image: {args.source}")
        sys.exit(1)

    results = detector.predict(frame, imgsz=args.imgsz)
    annotated, detections = detector.annotate_frame(
        frame, results, show_conf=args.show_conf
    )

    print(f"[Result] Detected {len(detections)} weapon instance(s).")
    for i, det in enumerate(detections, 1):
        print(f"  [{i}] Bounding Box: {det['box']} | Conf: {det['confidence']:.2f}")

    if args.save:
        _ensure_parent_dir(args.save)
        if not cv2.imwrite(args.save, annotated):
            print(f"[Error] Failed to write output image to: {args.save}")
            sys.exit(1)
        print(f"[Saved] Output written to: {args.save}")

    if not args.no_show:
        cv2.imshow("Weapon Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_stream_inference(detector: WeaponDetector, args) -> None:
    safe_source = sanitize_source_label(args.source)
    is_local_path = not args.source.isdigit() and "://" not in args.source
    if is_local_path and not os.path.isfile(args.source):
        print(f"[Error] Video source not found: {safe_source}")
        sys.exit(1)

    cap = open_capture(args.source)
    if not cap or not cap.isOpened():
        print(f"[Error] Failed to open video source: {safe_source}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = raw_fps if raw_fps > 0 else 30.0

    writer = None
    if args.save:
        _ensure_parent_dir(args.save)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            print(f"[Error] Failed to initialize video writer for: {args.save}")
            sys.exit(1)

    temporal_filter = TemporalPersistenceFilter(
        min_hits=args.min_frames,
        window_size=max(args.window, args.min_frames),
    )

    print(f"[Info] Active source: {safe_source} ({width}x{height} @ {fps:.1f} FPS)")
    print("[Info] Controls: Press 'q' to stop.")

    frame_count = 0
    total_detections = 0
    start_time = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            results = detector.predict(frame, imgsz=args.imgsz)
            raw_detections = detector.extract_detections(results, frame.shape)
            confirmed = temporal_filter.update(raw_detections)
            annotated, _ = detector.annotate_frame(
                frame, confirmed, show_conf=args.show_conf
            )

            total_detections += len(confirmed)
            frame_count += 1

            if args.show_fps:
                elapsed_now = max(time.perf_counter() - start_time, 1e-4)
                fps_text = f"{frame_count / elapsed_now:.1f} FPS"
                cv2.putText(
                    annotated,
                    fps_text,
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            if writer:
                writer.write(annotated)

            if not args.no_show:
                cv2.imshow("Weapon Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        if writer:
            writer.release()
            print(f"[Saved] Annotated video saved to: {args.save}")
        if not args.no_show:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start_time
    avg_fps = frame_count / max(elapsed, 0.001)
    print("\n[Summary]")
    print(f"  Processed Frames: {frame_count}")
    print(f"  Total Detections: {total_detections}")
    print(f"  Average FPS:      {avg_fps:.1f}")


def main():
    args = parse_args()

    try:
        detector = WeaponDetector(
            weights_path=args.weights,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            device=args.device,
            fp16=False if args.no_fp16 else None,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[Error] {exc}")
        sys.exit(1)

    target_imgsz = args.imgsz if args.imgsz is not None else detector.default_imgsz
    precision = "FP16" if detector.fp16 else "FP32"
    print(
        f"[Info] Loaded weights: {os.path.basename(detector.weights_path)} "
        f"(device={detector.device}, {precision}, imgsz={target_imgsz})"
    )

    if os.path.isfile(args.source) and args.source.lower().endswith(IMAGE_EXTENSIONS):
        run_image_inference(detector, args)
    else:
        run_stream_inference(detector, args)


if __name__ == "__main__":
    main()
