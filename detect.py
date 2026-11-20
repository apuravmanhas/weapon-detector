#!/usr/bin/env python3
"""
Real-Time Weapon Detection CLI
Runs inference on webcam feeds, RTSP streams, video files, or images using YOLO11 weights.
"""

import argparse
import os
import queue
import sys
import threading
import time
from urllib.parse import urlsplit, urlunsplit

import cv2

from src.detector import TemporalPersistenceFilter, WeaponDetector, apply_lab_clahe
from src.tracker import WeaponTrackManager
from src.verifier import CropVerifier

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
        default=None,
        help="Confidence threshold for weapon detection (default: 0.35, or 0.20 with --dense).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=None,
        help="NMS IoU threshold to merge overlapping boxes (default: 0.45, or 0.65 with --dense).",
    )
    parser.add_argument(
        "--clahe",
        action="store_true",
        help="Apply LAB-luminance CLAHE contrast lift before Stage-1 inference for dark/piled weapons.",
    )
    parser.add_argument(
        "--dense",
        action="store_true",
        help="Preset for closely packed weapons (defaults conf=0.20, iou=0.65, nested=0.95).",
    )
    parser.add_argument(
        "--tiled",
        action="store_true",
        help="Enable full-frame + 2x2 overlapping tiled (SAHI) inference for piled/small weapons.",
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
        "--tracker",
        choices=("smart", "legacy"),
        default="smart",
        help="Temporal tracker mode: 'smart' (velocity + EMA + static/verifier rules) or 'legacy'.",
    )
    parser.add_argument(
        "--static-suppress",
        action="store_true",
        help="Quarantine persistent motionless low/moderate-confidence false positives on fixed feeds.",
    )
    parser.add_argument(
        "--static-max-conf",
        type=float,
        default=0.58,
        help="Maximum EMA confidence eligible for static-background quarantine (default: 0.58).",
    )
    parser.add_argument(
        "--verifier",
        type=str,
        default=None,
        help="Optional Stage-2 CropVerifier checkpoint (.pt) for hard false-positive rejection.",
    )
    parser.add_argument(
        "--verify-thresh",
        type=float,
        default=0.50,
        help="Minimum Stage-2 weapon probability required to keep a verified detection (default: 0.50).",
    )
    parser.add_argument(
        "--verify-below",
        type=float,
        default=0.85,
        help="Run Stage-2 CropVerifier only on Stage-1 boxes below this confidence (default: 0.85).",
    )
    parser.add_argument(
        "--show-conf",
        action="store_true",
        help="Display confidence score alongside detection labels.",
    )
    parser.add_argument(
        "--show-track-id",
        action="store_true",
        help="Display track IDs alongside detection labels when using the smart tracker.",
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
        "--async-io",
        action="store_true",
        help="Use bounded background threads for video file decoding and encoding.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device ('cuda:0', 'cpu', or None for auto-selection).",
    )
    return parser.parse_args()


class ThreadedVideoReader:
    """Bounded background frame reader to overlap video decoding with GPU inference."""

    def __init__(self, cap: cv2.VideoCapture, queue_size: int = 16):
        self._cap = cap
        self._queue: queue.Queue = queue.Queue(maxsize=max(2, queue_size))
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        try:
            while not self._stopped.is_set():
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    self._put_with_stop(None)
                    break
                if not self._put_with_stop(frame):
                    break
        finally:
            self._stopped.set()

    def _put_with_stop(self, item) -> bool:
        while not self._stopped.is_set():
            try:
                self._queue.put(item, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def read(self):
        item = self._queue.get()
        if item is None:
            return False, None
        return True, item

    def release(self) -> None:
        self._stopped.set()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._thread.join(timeout=2.0)
        self._cap.release()


class ThreadedVideoWriter:
    """Bounded background frame writer to avoid blocking inference on disk video encoding."""

    _STOP = object()

    def __init__(self, writer: cv2.VideoWriter, queue_size: int = 16):
        self._writer = writer
        self._queue: queue.Queue = queue.Queue(maxsize=max(2, queue_size))
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is self._STOP:
                break
            self._writer.write(item)

    def write(self, frame) -> None:
        self._queue.put(frame)

    def release(self) -> None:
        self._queue.put(self._STOP)
        self._thread.join(timeout=10.0)
        self._writer.release()


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


def _detect_frame(detector: WeaponDetector, frame, args) -> list[dict]:
    """Run Stage-1 detection (single-frame or 2x2 tiled SAHI) with optional LAB-CLAHE."""
    use_clahe = getattr(args, "clahe", False)
    if getattr(args, "tiled", False):
        return detector.predict_tiled(
            frame,
            imgsz=args.imgsz,
            use_clahe=use_clahe,
        )
    infer_frame = apply_lab_clahe(frame) if use_clahe else frame
    results = detector.predict(infer_frame, imgsz=args.imgsz)
    return detector.extract_detections(results, frame.shape)


def run_image_inference(
    detector: WeaponDetector,
    args,
    verifier: CropVerifier | None = None,
) -> None:
    frame = cv2.imread(args.source)
    if frame is None:
        print(f"[Error] Could not read image: {args.source}")
        sys.exit(1)

    detections = _detect_frame(detector, frame, args)
    if verifier is not None and detections:
        detections = verifier.verify_detections(
            frame,
            detections,
            verify_below_conf=getattr(args, "verify_below", 0.85),
            threshold=getattr(args, "verify_thresh", 0.50),
            filter_out=True,
        )
    annotated, detections = detector.annotate_frame(
        frame,
        detections,
        show_conf=args.show_conf,
        show_track_id=getattr(args, "show_track_id", False),
    )

    print(f"[Result] Detected {len(detections)} weapon instance(s).")
    for i, det in enumerate(detections, 1):
        extra = f" | Verifier: {det['verifier_prob']:.2f} ({det.get('verifier_label')})" if "verifier_prob" in det else ""
        print(f"  [{i}] Bounding Box: {det['box']} | Conf: {det['confidence']:.2f}{extra}")

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


def run_stream_inference(
    detector: WeaponDetector,
    args,
    verifier: CropVerifier | None = None,
) -> None:
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

    use_async = getattr(args, "async_io", False)
    reader = ThreadedVideoReader(cap) if (use_async and is_local_path) else cap

    writer = None
    if args.save:
        _ensure_parent_dir(args.save)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        raw_writer = cv2.VideoWriter(args.save, fourcc, fps, (width, height))
        if not raw_writer.isOpened():
            reader.release()
            print(f"[Error] Failed to initialize video writer for: {args.save}")
            sys.exit(1)
        writer = ThreadedVideoWriter(raw_writer) if use_async else raw_writer

    use_smart = getattr(args, "tracker", "smart") == "smart"
    static_suppress = getattr(args, "static_suppress", False)
    if use_smart:
        temporal_filter = WeaponTrackManager(
            min_hits=args.min_frames,
            window_size=max(args.window, args.min_frames),
            static_suppress=static_suppress,
            static_max_conf=getattr(args, "static_max_conf", 0.58),
            verify_thresh=getattr(args, "verify_thresh", 0.50),
        )
    else:
        temporal_filter = TemporalPersistenceFilter(
            min_hits=args.min_frames,
            window_size=max(args.window, args.min_frames),
        )

    print(
        f"[Info] Active source: {safe_source} ({width}x{height} @ {fps:.1f} FPS) "
        f"| tracker={getattr(args, 'tracker', 'smart')} "
        f"| static_suppress={static_suppress}"
    )
    print("[Info] Controls: Press 'q' to stop.")

    frame_count = 0
    total_detections = 0
    start_time = time.perf_counter()

    try:
        while True:
            ret, frame = reader.read()
            if not ret or frame is None:
                break

            raw_detections = _detect_frame(detector, frame, args)
            if verifier is not None and raw_detections:
                raw_detections = verifier.verify_detections(
                    frame,
                    raw_detections,
                    verify_below_conf=getattr(args, "verify_below", 0.85),
                    threshold=getattr(args, "verify_thresh", 0.50),
                    filter_out=not use_smart,
                )

            if use_smart:
                confirmed = temporal_filter.update(
                    raw_detections,
                    frame=frame if static_suppress else None,
                )
            else:
                confirmed = temporal_filter.update(raw_detections)

            annotated, _ = detector.annotate_frame(
                frame,
                confirmed,
                show_conf=args.show_conf,
                show_track_id=getattr(args, "show_track_id", False),
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
        reader.release()
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

    dense_mode = getattr(args, "dense", False)
    tiled_mode = getattr(args, "tiled", False)
    clahe_mode = getattr(args, "clahe", False)
    effective_conf = args.conf if args.conf is not None else (0.20 if dense_mode else 0.35)
    effective_iou = args.iou if args.iou is not None else (0.65 if dense_mode else 0.45)
    effective_nested = 0.95 if (dense_mode or tiled_mode) else 0.85

    try:
        detector = WeaponDetector(
            weights_path=args.weights,
            conf_threshold=effective_conf,
            iou_threshold=effective_iou,
            nested_threshold=effective_nested,
            device=args.device,
            fp16=False if args.no_fp16 else None,
        )
        verifier = (
            CropVerifier(
                weights_path=args.verifier,
                device=detector.device,
                fp16=detector.fp16,
            )
            if args.verifier
            else None
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[Error] {exc}")
        sys.exit(1)

    target_imgsz = args.imgsz if args.imgsz is not None else detector.default_imgsz
    precision = "FP16" if detector.fp16 else "FP32"
    print(
        f"[Info] Loaded weights: {os.path.basename(detector.weights_path)} "
        f"(device={detector.device}, {precision}, imgsz={target_imgsz}, "
        f"conf={effective_conf}, iou={effective_iou})"
    )
    if dense_mode or clahe_mode or tiled_mode:
        print(
            f"[Info] Stacked/dense options: dense={dense_mode}, clahe={clahe_mode}, tiled={tiled_mode}"
        )
    if getattr(detector, "distractor_class_ids", None):
        ignored = [detector.class_names.get(cid, str(cid)) for cid in sorted(detector.distractor_class_ids)]
        print(f"[Info] Auto-ignoring explicit distractor classes: {ignored}")
    if verifier is not None:
        print(
            f"[Info] Loaded Stage-2 verifier: {os.path.basename(verifier.weights_path)} "
            f"(arch={verifier.arch}, classes={verifier.classes}, "
            f"verify_below={args.verify_below}, thresh={args.verify_thresh})"
        )

    if os.path.isfile(args.source) and args.source.lower().endswith(IMAGE_EXTENSIONS):
        run_image_inference(detector, args, verifier=verifier)
    else:
        run_stream_inference(detector, args, verifier=verifier)


if __name__ == "__main__":
    main()
