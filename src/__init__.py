"""
Weapon Detector Package
Core detection engine powered by fine-tuned YOLO11 models (YOLO11x / YOLO11m).
"""

from .detector import TemporalPersistenceFilter, WeaponDetector, apply_lab_clahe, nms_merge
from .tracker import TrackState, WeaponTrackManager
from .verifier import CropVerifier, extract_padded_crop

__all__ = [
    "WeaponDetector",
    "TemporalPersistenceFilter",
    "WeaponTrackManager",
    "TrackState",
    "CropVerifier",
    "extract_padded_crop",
    "apply_lab_clahe",
    "nms_merge",
]
