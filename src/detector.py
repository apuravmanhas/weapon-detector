import os
import cv2
import numpy as np
from ultralytics import YOLO


class WeaponDetector:
    """
    Real-time weapon detection engine using fine-tuned YOLO11m weights.
    """

    def __init__(self, weights_path: str = "weights/best.pt", conf_threshold: float = 0.40, device: str = None):
        """
        Initialize the detector.

        Args:
            weights_path: Path to the trained .pt model weights.
            conf_threshold: Minimum confidence score to trigger a detection.
            device: Computation device (e.g., 'cuda:0', 'cpu', or None for auto).
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights not found at: {weights_path}")

        self.conf_threshold = conf_threshold
        self.device = device
        self.model = YOLO(weights_path)

    def predict(self, frame: np.ndarray, imgsz: int = 640):
        """
        Run inference on a single frame.

        Args:
            frame: BGR numpy image frame.
            imgsz: Input resolution (default: 640).

        Returns:
            Ultralytics Results object.
        """
        return self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            imgsz=imgsz,
            device=self.device,
            verbose=False
        )

    def annotate_frame(
        self,
        frame: np.ndarray,
        results,
        box_color=(0, 0, 255),
        show_conf: bool = False
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Draw detection bounding boxes and labels onto the frame.

        Args:
            frame: Input BGR image.
            results: Results object from self.predict.
            box_color: BGR tuple for bounding box (default: Red).
            show_conf: Whether to append confidence percentage (e.g. '0.85').

        Returns:
            annotated_frame: Frame with drawn detections.
            detections: List of parsed detection dictionaries.
        """
        annotated = frame.copy()
        detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.model.names.get(cls_id, "weapon")

                # Format label
                label = f"{cls_name.upper()} {conf:.2f}" if show_conf else cls_name.upper()

                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

                # Label background padding
                (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                top = max(y1, label_h + 10)
                cv2.rectangle(
                    annotated,
                    (x1, top - label_h - 10),
                    (x1 + label_w + 6, top),
                    box_color,
                    -1
                )

                # Label text (white)
                cv2.putText(
                    annotated,
                    label,
                    (x1 + 3, top - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                detections.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class": cls_name
                })

        return annotated, detections
