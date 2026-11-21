"""Canonical data structures and threat readiness states for causal pre-firing anticipation."""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class ThreatState(IntEnum):
    """Ordered defensive readiness states for tracked (person, weapon) pairs."""

    NO_WEAPON = 0
    PASSIVE_PRESENT = 1
    DRAWING_READYING = 2
    AIMING_IMMINENT = 3

    @property
    def label(self) -> str:
        return self.name


# Standard 17-keypoint COCO pose layout used by YOLO11-pose
COCO_KEYPOINTS: dict[str, int] = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}


@dataclass
class WeaponObservation:
    """Single verified and tracked weapon observation on a frame."""

    track_id: int
    box: list[int]  # [x1, y1, x2, y2]
    confidence: float
    cls_name: str = "weapon"
    ema_conf: float | None = None
    verifier_prob: float | None = None
    verifier_label: str | None = None
    vx: float = 0.0
    vy: float = 0.0
    # Optional 2-keypoint orientation estimate in image pixel coords
    grip_xy: tuple[float, float] | None = None
    muzzle_xy: tuple[float, float] | None = None
    orientation_conf: float = 0.0
    is_coasted: bool = False


@dataclass
class PersonTrack:
    """Single tracked person with 17 COCO keypoints of shape (17, 3) -> [x, y, conf]."""

    person_id: int
    box: list[int]  # [x1, y1, x2, y2]
    confidence: float
    keypoints: Any  # np.ndarray shape (17, 3), float32
    vx: float = 0.0
    vy: float = 0.0
    age: int = 1
    misses: int = 0


@dataclass
class AssociatedPair:
    """Weapon track paired with its most likely bearer (or None if unowned/table weapon)."""

    pair_key: str
    weapon: WeaponObservation
    person: PersonTrack | None
    min_wrist_dist_norm: float
    left_wrist_dist_norm: float
    right_wrist_dist_norm: float
    torso_scale: float
    ownership_confidence: float


@dataclass
class IntentAssessment:
    """Causal per-pair threat readiness assessment for the current frame."""

    pair_key: str
    weapon_track_id: int
    person_id: int | None
    state: ThreatState
    risk_score: float  # Calibrated continuous [0.0, 1.0]
    state_probs: dict[str, float]
    reasons: list[str]
    weapon_box: list[int]
    person_box: list[int] | None = None
    features: dict[str, float] = field(default_factory=dict)


@dataclass
class StageLatenciesMs:
    """Wall-clock (CUDA-synchronized when on GPU) per-stage latency breakdown in ms."""

    weapon_stage1_ms: float = 0.0
    verifier_stage2_ms: float = 0.0
    weapon_track_ms: float = 0.0
    pose_ms: float = 0.0
    association_ms: float = 0.0
    features_ms: float = 0.0
    intent_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class FrameTelemetry:
    """Complete structured output for one processed video frame."""

    frame_index: int
    timestamp_ms: float
    max_state: ThreatState
    max_risk_score: float
    assessments: list[IntentAssessment]
    persons: list[PersonTrack]
    weapons: list[WeaponObservation]
    latencies: StageLatenciesMs
