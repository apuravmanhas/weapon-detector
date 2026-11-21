"""
End-to-end Pre-Firing Weapon Intent & Action Anticipation Pipeline with exact
CUDA-synchronized per-stage latency profiling.
"""

from pathlib import Path
import time

import numpy as np
import torch

from src.association import WeaponPersonAssociator
from src.features import KinematicFeatureExtractor
from src.intent_fsm import BiomechanicalIntentFSM
from src.perception import Stage1WeaponDetector, Stage2CropVerifier, WeaponTrackManager
from src.pose import PersonTrackManager, PoseEstimator
from src.states import FrameTelemetry, IntentAssessment, StageLatenciesMs, ThreatState
from src.temporal_model import CausalTemporalInferenceEngine


def _sync_if_cuda(use_cuda: bool) -> None:
    if use_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()


class PreFireIntentPipeline:
    """
    Orchestrates the full causal anticipation stack:
      1. Stage-1 Weapon Detection (YOLO11x FP16)
      2. Stage-2 Distractor Verification (ConvNeXt-Tiny, conf < 0.72)
      3. Velocity + Static-Suppression Weapon Tracking
      4. Parallel Pose Estimation (YOLO11n/s-pose) + Person Tracking
      5. Scale-Invariant Weapon-to-Person Association
      6. 30-D FPS-Invariant Biomechanical & Temporal Feature Extraction
      7. Causal Intent Engine (Biomechanical Hysteresis FSM or Causal TCN/GRU)
    """

    def __init__(
        self,
        weapon_weights: str | Path | None = None,
        verifier_weights: str | Path | None = None,
        pose_weights: str = "yolo11s-pose.pt",
        temporal_weights: str | Path | None = None,
        weapon_conf: float = 0.20,
        weapon_iou: float = 0.45,
        weapon_imgsz: int | None = None,
        pose_conf: float = 0.30,
        pose_imgsz: int = 640,
        verify_below_conf: float = 0.72,
        verify_thresh: float = 0.52,
        verifier_cache_ttl: int = 4,
        enable_coasting: bool = True,
        min_weapon_hits: int = 2,
        weapon_window: int = 5,
        static_suppress: bool = True,
        device: str | None = None,
        fp16: bool | None = None,
        sync_timing: bool = True,
        pose_gated: bool = True,
        pose_cooldown_frames: int = 15,
    ):
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.use_cuda = device.startswith("cuda") and torch.cuda.is_available()
        self.sync_timing = sync_timing
        self.verify_below_conf = verify_below_conf
        self.verify_thresh = verify_thresh
        self.verifier_cache_ttl = verifier_cache_ttl
        self.pose_gated = pose_gated
        self.pose_cooldown_frames = max(0, pose_cooldown_frames)
        self._pose_active_countdown = 0

        self.weapon_detector = Stage1WeaponDetector(
            weights_path=weapon_weights,
            conf_threshold=weapon_conf,
            iou_threshold=weapon_iou,
            imgsz=weapon_imgsz,
            device=device,
            fp16=fp16,
        )
        self.verifier = (
            Stage2CropVerifier(weights_path=verifier_weights, device=device, fp16=fp16)
            if verifier_weights is not None
            else self._try_default_verifier(device, fp16)
        )
        self.weapon_tracker = WeaponTrackManager(
            min_hits=min_weapon_hits,
            window_size=weapon_window,
            static_suppress=static_suppress,
        )
        self.pose_estimator = PoseEstimator(
            weights=pose_weights,
            conf_threshold=pose_conf,
            imgsz=pose_imgsz,
            device=device,
            fp16=fp16,
        )
        self.person_tracker = PersonTrackManager()
        self.associator = WeaponPersonAssociator(enable_coasting=enable_coasting)
        self.feature_extractor = KinematicFeatureExtractor()

        if temporal_weights is not None:
            self.intent_engine = CausalTemporalInferenceEngine(
                checkpoint_path=temporal_weights,
                device=device,
            )
            self.intent_mode = "causal_temporal"
        else:
            self.intent_engine = BiomechanicalIntentFSM()
            self.intent_mode = "biomechanical_fsm"

        self._frame_counter = 0

    def _try_default_verifier(self, device: str, fp16: bool | None) -> Stage2CropVerifier | None:
        try:
            return Stage2CropVerifier(weights_path=None, device=device, fp16=fp16)
        except FileNotFoundError:
            return None

    def reset(self) -> None:
        self.weapon_tracker.reset()
        self.person_tracker.reset()
        self.associator.reset()
        self.feature_extractor.reset()
        self.intent_engine.reset()
        self._pose_active_countdown = 0
        self._frame_counter = 0

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp_s: float | None = None,
        fps_fallback: float = 30.0,
    ) -> FrameTelemetry:
        frame_idx = self._frame_counter
        self._frame_counter += 1
        ts_s = float(timestamp_s) if timestamp_s is not None else (frame_idx / max(1.0, fps_fallback))

        lat = StageLatenciesMs()
        if self.sync_timing:
            _sync_if_cuda(self.use_cuda)
        t_total_0 = time.perf_counter()

        # 1. Stage-1 Weapon Detection
        t0 = time.perf_counter()
        raw_weapons = self.weapon_detector.detect(frame)
        if self.sync_timing:
            _sync_if_cuda(self.use_cuda)
        lat.weapon_stage1_ms = (time.perf_counter() - t0) * 1000.0

        # 2. Stage-2 Crop Verifier (with track-level caching)
        t0 = time.perf_counter()
        if self.verifier is not None and raw_weapons:
            cached_tracks = self.weapon_tracker.get_verifier_cache_candidates(
                max_ttl_frames=self.verifier_cache_ttl,
            )
            verified_weapons = self.verifier.verify(
                frame,
                raw_weapons,
                verify_below_conf=self.verify_below_conf,
                threshold=self.verify_thresh,
                cached_tracks=cached_tracks,
            )
        else:
            verified_weapons = raw_weapons
        if self.sync_timing:
            _sync_if_cuda(self.use_cuda)
        lat.verifier_stage2_ms = (time.perf_counter() - t0) * 1000.0

        # 3. Weapon Tracking & Static Suppression
        t0 = time.perf_counter()
        weapons = self.weapon_tracker.update(verified_weapons, frame=frame)
        lat.weapon_track_ms = (time.perf_counter() - t0) * 1000.0

        # 4. Pose Estimation + Person Tracking (gated on weapon presence or active coasting)
        t0 = time.perf_counter()
        weapon_present_or_pending = bool(verified_weapons or weapons or self.associator.has_active_coasts)
        if weapon_present_or_pending:
            self._pose_active_countdown = self.pose_cooldown_frames
        run_pose = (not self.pose_gated) or weapon_present_or_pending or (self._pose_active_countdown > 0)
        if run_pose:
            if not weapon_present_or_pending and self._pose_active_countdown > 0:
                self._pose_active_countdown -= 1
            raw_persons = self.pose_estimator.predict(frame)
            persons = self.person_tracker.update(raw_persons)
        else:
            persons = []
        if self.sync_timing:
            _sync_if_cuda(self.use_cuda)
        lat.pose_ms = (time.perf_counter() - t0) * 1000.0

        # 5. Weapon-Person Association
        t0 = time.perf_counter()
        pairs = self.associator.associate(weapons, persons)
        lat.association_ms = (time.perf_counter() - t0) * 1000.0

        # 6. Feature Extraction + 7. Causal Intent Estimation
        t_feat = 0.0
        t_int = 0.0
        assessments: list[IntentAssessment] = []
        for pair in pairs:
            tf0 = time.perf_counter()
            vec, feat_dict = self.feature_extractor.extract(pair, timestamp_s=ts_s)
            t_feat += (time.perf_counter() - tf0) * 1000.0

            ti0 = time.perf_counter()
            assessment = self.intent_engine.update(pair, vec, feat_dict, timestamp_s=ts_s)
            t_int += (time.perf_counter() - ti0) * 1000.0
            assessments.append(assessment)

        lat.features_ms = t_feat
        lat.intent_ms = t_int
        if self.sync_timing:
            _sync_if_cuda(self.use_cuda)
        lat.total_ms = (time.perf_counter() - t_total_0) * 1000.0

        if assessments:
            max_state = max(a.state for a in assessments)
            max_risk = max(a.risk_score for a in assessments)
        else:
            max_state = ThreatState.NO_WEAPON
            max_risk = 0.0

        return FrameTelemetry(
            frame_index=frame_idx,
            timestamp_ms=ts_s * 1000.0,
            max_state=max_state,
            max_risk_score=max_risk,
            assessments=assessments,
            persons=persons,
            weapons=weapons,
            latencies=lat,
        )
