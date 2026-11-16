"""
Weapon Detector Package
Core detection engine powered by fine-tuned YOLO11 models (YOLO11x / YOLO11m).
"""

from .detector import TemporalPersistenceFilter, WeaponDetector

__all__ = ["WeaponDetector", "TemporalPersistenceFilter"]
