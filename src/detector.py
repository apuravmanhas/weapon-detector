import os
import cv2
import numpy as np
from ultralytics import YOLO


class WeaponDetector:
    """
    Real-time weapon detection engine powered by fine-tuned YOLO11 weights.
    Supports both YOLO11x (Extra-Large, 56.9M params) and YOLO11m models.
    """

    def __init__(self, weights_path: str = None, conf_threshold: float = 0.35, device: str = None):
        """
        Initialize the detector.

        Args:
            weights_path: Path to model weights (.pt). If None, defaults to YOLO11x production model.
            conf_threshold: Minimum confidence score to trigger detection (default: 0.35).
            device: Computation device ('cuda:0', 'cpu', or None for auto).
        """
        if weights_path is None:
            if os.path.exists("weights/yolo11x_weapon_production.pt"):
                weights_path = "weights/yolo11x_weapon_production.pt"
            elif os.path.exists("weights/best.pt"):
                weights_path = "weights/best.pt"
            else:
                raise FileNotFoundError("No model weights found in weights/ directory.")
        elif not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights not found at: {weights_path}")

        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.device = device
        self.model = YOLO(weights_path)
        # Enforce unified single-class surveillance mapping
        self.model.names[0] = "weapon"

    def predict(self, frame: np.ndarray, imgsz: int = 640):
        """
        Run inference on a single frame.

        Args:
            frame: BGR numpy image frame.
            imgsz: Input resolution (default: 640; 800 for maximum spatial precision).

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
                cls_name = "weapon"

                # Format label
                label = f"{cls_name.upper()} {conf:.2f}" if show_conf else cls_name.upper()

                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 3)

                # Label background badge
                (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                top = max(y1, label_h + 10)
                cv2.rectangle(
                    annotated,
                    (x1, top - label_h - 10),
                    (x1 + label_w + 8, top),
                    box_color,
                    -1
                )

                # Label text (white)
                cv2.putText(
                    annotated,
                    label,
                    (x1 + 4, top - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
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
