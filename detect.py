#!/usr/bin/env python3
"""
Real-Time Weapon Detection CLI
Runs inference on webcam, video files, or images using fine-tuned YOLO11m weights.
"""

import argparse
import os
import sys
import time
import cv2

from src.detector import WeaponDetector


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-Time Weapon Detection in Surveillance Feeds (YOLO11m)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Input source: camera index (e.g. '0'), path to video file, or path to image file."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/best.pt",
        help="Path to trained model weights (.pt)."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.40,
        help="Confidence threshold for weapon detection (default: 0.40)."
    )
    parser.add_argument(
        "--show-conf",
        action="store_true",
        help="Display confidence score alongside detection labels (default: False)."
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save annotated output (image or video)."
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display live preview window (useful for headless servers)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device ('cuda:0', 'cpu', or None for automatic selection)."
    )
    return parser.parse_args()


def open_capture(source: str):
    """Open video capture with DirectShow backend support for Windows webcams."""
    if source.isdigit():
        cam_idx = int(source)
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                return cap
        return cv2.VideoCapture(cam_idx)
    return cv2.VideoCapture(source)


def main():
    args = parse_args()

    if not os.path.exists(args.weights):
        print(f"[Error] Model weights not found at: {args.weights}")
        print("Ensure 'weights/best.pt' is present or specify via --weights.")
        sys.exit(1)

    print(f"[Info] Loading model from: {args.weights}")
    detector = WeaponDetector(
        weights_path=args.weights,
        conf_threshold=args.conf,
        device=args.device
    )

    # 1. Single Image Processing
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    if os.path.isfile(args.source) and args.source.lower().endswith(image_extensions):
        frame = cv2.imread(args.source)
        if frame is None:
            print(f"[Error] Could not read image: {args.source}")
            sys.exit(1)

        results = detector.predict(frame)
        annotated, detections = detector.annotate_frame(
            frame, results, show_conf=args.show_conf
        )

        print(f"[Result] Detected {len(detections)} weapon instance(s).")
        for i, det in enumerate(detections, 1):
            print(f"  [{i}] Box: {det['box']} | Conf: {det['confidence']:.2f}")

        if args.save:
            cv2.imwrite(args.save, annotated)
            print(f"[Saved] Output written to: {args.save}")

        if not args.no_show:
            cv2.imshow("Weapon Detection", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    # 2. Video / Camera Stream Processing
    cap = open_capture(args.source)
    if not cap or not cap.isOpened():
        print(f"[Error] Failed to open video source: {args.source}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = int(fps) if fps > 0 else 30

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps, (width, height))

    print(f"[Info] Starting detection on: {args.source}")
    print("[Info] Press 'q' in preview window to exit.")

    frame_count = 0
    total_detections = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = detector.predict(frame)
            annotated, detections = detector.annotate_frame(
                frame, results, show_conf=args.show_conf
            )

            total_detections += len(detections)
            frame_count += 1

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

    elapsed = time.time() - start_time
    avg_fps = frame_count / max(elapsed, 0.001)
    print("\n[Summary]")
    print(f"  Processed Frames: {frame_count}")
    print(f"  Total Detections: {total_detections}")
    print(f"  Average FPS:      {avg_fps:.1f}")


if __name__ == "__main__":
    main()
