"""
Interpretable Biomechanical Hysteresis State Machine for causal pre-firing anticipation.

Zero training data required, sub-millisecond inference, deterministic, and completely auditable:
every state transition logs the exact biomechanical reason codes that fired.
"""

from collections import deque
from dataclasses import dataclass, field
import math

import numpy as np

from src.features import FEATURE_NAMES
from src.states import AssociatedPair, IntentAssessment, ThreatState

_FEAT_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _ramp(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    return _clip01((x - lo) / (hi - lo))


@dataclass
class _PairFSMState:
    state: ThreatState = ThreatState.PASSIVE_PRESENT
    risk_ema: float = 0.10
    aim_votes: deque[int] = field(default_factory=lambda: deque(maxlen=4))
    draw_votes: deque[int] = field(default_factory=lambda: deque(maxlen=4))
    aim_hold: int = 0
    draw_hold: int = 0
    last_seen_s: float = 0.0


class BiomechanicalIntentFSM:
    """
    Causal hysteresis state machine mapping 28-D kinematic features to:
      - PASSIVE_PRESENT  (table, holstered, slung, relaxed carry)
      - DRAWING_READYING (hand-locked upward draw or torso-zone readying)
      - AIMING_IMMINENT  (extended/two-hand/shouldered firing stance or draw-to-lock)
    """

    def __init__(
        self,
        aim_confirm_frames: int = 2,
        draw_confirm_frames: int = 2,
        aim_hold_frames: int = 6,
        draw_hold_frames: int = 5,
        risk_ema_alpha: float = 0.55,
    ):
        self.aim_confirm_frames = max(1, aim_confirm_frames)
        self.draw_confirm_frames = max(1, draw_confirm_frames)
        self.aim_hold_frames = max(0, aim_hold_frames)
        self.draw_hold_frames = max(0, draw_hold_frames)
        self.risk_ema_alpha = risk_ema_alpha
        self._states: dict[str, _PairFSMState] = {}

    def reset(self) -> None:
        self._states.clear()

    def _prune(self, now_s: float) -> None:
        stale = [k for k, st in self._states.items() if (now_s - st.last_seen_s) > 3.5]
        for k in stale:
            self._states.pop(k, None)

    def update(
        self,
        pair: AssociatedPair,
        feature_vec: np.ndarray,
        feature_dict: dict[str, float],
        timestamp_s: float,
    ) -> IntentAssessment:
        self._prune(timestamp_s)

        f = feature_vec
        has_owner = float(f[_FEAT_IDX["has_owner"]])
        own_conf = float(f[_FEAT_IDX["ownership_conf"]])
        min_wd = float(f[_FEAT_IDX["min_wrist_dist"]])
        second_wd = float(f[_FEAT_IDX["second_wrist_dist"]])
        wrist_pair_dist = float(f[_FEAT_IDX["wrist_pair_dist"]])
        a_elbow_deg = float(f[_FEAT_IDX["active_elbow_angle_norm"]]) * 180.0
        a_pitch_deg = float(f[_FEAT_IDX["active_shoulder_pitch_norm"]]) * 180.0
        s_pitch_deg = float(f[_FEAT_IDX["support_shoulder_pitch_norm"]]) * 180.0
        horiz_reach = float(f[_FEAT_IDX["active_horiz_reach"]])
        wrist_elev = float(f[_FEAT_IDX["active_wrist_elev"]])
        weapon_elev = float(f[_FEAT_IDX["weapon_elev"]])
        aspect_log = float(f[_FEAT_IDX["weapon_aspect_log"]])
        muzzle_horiz = float(f[_FEAT_IDX["muzzle_horiz_align"]])
        head_dist = float(f[_FEAT_IDX["head_to_weapon_dist"]])
        weapon_up_vel = float(f[_FEAT_IDX["weapon_up_vel"]])
        wrist_up_vel = float(f[_FEAT_IDX["active_wrist_up_vel"]])
        recent_peak_up = float(f[_FEAT_IDX["recent_peak_up_vel"]])
        stab_score = float(f[_FEAT_IDX["stabilization_score"]])
        upper_vis = float(f[_FEAT_IDX["upper_body_kpt_vis"]])

        st = self._states.setdefault(pair.pair_key, _PairFSMState())
        st.last_seen_s = timestamp_s

        # 1. Unowned weapon (table / rack / ground) -> always PASSIVE_PRESENT
        if has_owner < 0.5:
            st.state = ThreatState.PASSIVE_PRESENT
            st.aim_votes.append(0)
            st.draw_votes.append(0)
            st.aim_hold = 0
            st.draw_hold = 0
            st.risk_ema = 0.35 * 0.12 + 0.65 * st.risk_ema
            return IntentAssessment(
                pair_key=pair.pair_key,
                weapon_track_id=pair.weapon.track_id,
                person_id=None,
                state=ThreatState.PASSIVE_PRESENT,
                risk_score=round(min(0.18, st.risk_ema), 4),
                state_probs={
                    ThreatState.NO_WEAPON.label: 0.0,
                    ThreatState.PASSIVE_PRESENT.label: 0.92,
                    ThreatState.DRAWING_READYING.label: 0.06,
                    ThreatState.AIMING_IMMINENT.label: 0.02,
                },
                reasons=["unowned_static_or_table"],
                weapon_box=list(pair.weapon.box),
                person_box=None,
                features=feature_dict,
            )

        # 2. Biomechanical sub-scores for owned weapons
        hand_coupled = (min_wd <= 0.55) or (upper_vis < 0.25 and own_conf >= 0.30)
        best_elev = max(weapon_elev, wrist_elev)
        best_up_vel = max(weapon_up_vel, wrist_up_vel)

        elev_score = _ramp(best_elev, -0.82, -0.08)
        draw_vel_score = _ramp(best_up_vel, 0.35, 1.75)

        reasons: list[str] = []

        # Firing stance #1: Extended handgun / side-profile aim (works cleanly for single-hand or coasted draws)
        ext_arm_score = (
            _ramp(a_elbow_deg, 118.0, 155.0)
            * _ramp(a_pitch_deg, 48.0, 85.0)
            * _ramp(horiz_reach, 0.34, 0.70)
        )
        if ext_arm_score >= 0.55:
            reasons.append(f"extended_arm(el={a_elbow_deg:.0f}deg,pitch={a_pitch_deg:.0f}deg)")

        # Firing stance #2: Two-hand handgun (front-facing punch-out) or shouldered cheek-weld rifle.
        # Strictly gates out vertical slung rifles (aspect_log <= -0.45 -> not_vertical == 0).
        both_hands_locked = (min_wd <= 0.48) and (second_wd <= 0.68)
        not_vertical = _ramp(aspect_log, -0.45, -0.05)
        cheek_weld_cue = _ramp(0.42 - head_dist, 0.0, 0.27) * _ramp(weapon_elev, -0.25, -0.02) * not_vertical
        raised_arm_cue = _ramp(max(a_pitch_deg, s_pitch_deg), 42.0, 78.0) * _ramp(best_elev, -0.32, 0.0) * not_vertical
        tight_pistol_cue = _ramp(0.42 - wrist_pair_dist, 0.0, 0.25) * _ramp(best_elev, -0.25, 0.02) * not_vertical
        two_hand_aim_score = (1.0 if both_hands_locked else 0.0) * max(cheek_weld_cue, raised_arm_cue, tight_pistol_cue)
        if two_hand_aim_score >= 0.55:
            if cheek_weld_cue >= max(raised_arm_cue, tight_pistol_cue):
                reasons.append(f"cheek_weld_aim(head_d={head_dist:.2f})")
            else:
                reasons.append("two_hand_elevated_stance")

        # Firing stance #3: Rapid upward draw followed by elevated non-vertical sight-lock stabilization
        draw_lock_score = (
            _ramp(recent_peak_up, 0.45, 1.35)
            * _ramp(best_elev, -0.28, 0.0)
            * _ramp(stab_score, 0.30, 0.75)
            * max(_ramp(max(a_pitch_deg, s_pitch_deg), 35.0, 75.0), cheek_weld_cue)
            * not_vertical
        )
        if draw_lock_score >= 0.55:
            reasons.append(f"draw_to_lock(peak_v={recent_peak_up:.2f},stab={stab_score:.2f})")

        # Firing stance #4: Heavy pose occlusion fallback on blurry CCTV
        occluded_aim_score = 0.0
        if upper_vis < 0.35 and own_conf >= 0.35:
            occluded_aim_score = (
                _ramp(weapon_elev, -0.30, 0.05)
                * max(_ramp(aspect_log, 0.30, 0.90), _ramp(recent_peak_up, 0.55, 1.40))
                * 0.85
            )
            if occluded_aim_score >= 0.55:
                reasons.append("cctv_occluded_elevated_horizontal")

        stance_score = max(ext_arm_score, two_hand_aim_score, draw_lock_score, occluded_aim_score)

        aim_candidate = hand_coupled and (best_elev >= -0.40) and (stance_score >= 0.56)
        draw_candidate = hand_coupled and (
            (best_up_vel >= 0.45 and best_elev >= -0.95)
            or (
                best_elev >= -0.55
                and (
                    a_pitch_deg >= 38.0
                    or both_hands_locked
                    or recent_peak_up >= 0.42
                )
            )
        )

        if draw_candidate and not aim_candidate:
            if best_up_vel >= 0.45:
                reasons.append(f"upward_draw(v={best_up_vel:.2f})")
            else:
                reasons.append(f"low_ready_elev(elev={best_elev:.2f})")

        st.aim_votes.append(1 if aim_candidate else 0)
        st.draw_votes.append(1 if (draw_candidate or aim_candidate) else 0)

        # 3. Hysteresis state transitions
        aim_confirmed = sum(st.aim_votes) >= self.aim_confirm_frames
        fast_promote = (
            st.state == ThreatState.DRAWING_READYING
            and aim_candidate
            and stance_score >= 0.72
        )
        draw_confirmed = sum(st.draw_votes) >= self.draw_confirm_frames

        if aim_confirmed or fast_promote:
            st.state = ThreatState.AIMING_IMMINENT
            st.aim_hold = self.aim_hold_frames
            st.draw_hold = self.draw_hold_frames
        elif st.aim_hold > 0 and hand_coupled and best_elev >= -0.55:
            st.aim_hold -= 1
            st.state = ThreatState.AIMING_IMMINENT
            reasons.append("aim_hysteresis_hold")
        elif draw_confirmed:
            st.state = ThreatState.DRAWING_READYING
            st.aim_hold = 0
            st.draw_hold = self.draw_hold_frames
        elif st.draw_hold > 0 and hand_coupled and best_elev >= -0.75:
            st.draw_hold -= 1
            st.state = ThreatState.DRAWING_READYING
            reasons.append("draw_hysteresis_hold")
        else:
            st.state = ThreatState.PASSIVE_PRESENT
            st.aim_hold = 0
            st.draw_hold = 0
            if not reasons:
                reasons.append("low_passive_carry" if hand_coupled else "weapon_not_gripped")

        # 4. Calibrated continuous pre-fire risk score in [0, 1]
        raw_risk = 0.12 + 0.28 * elev_score + 0.25 * draw_vel_score + 0.45 * stance_score
        if not hand_coupled:
            raw_risk *= 0.45
        raw_risk = _clip01(raw_risk)
        if st.state == ThreatState.AIMING_IMMINENT:
            raw_risk = max(raw_risk, 0.75)
        elif st.state == ThreatState.DRAWING_READYING:
            raw_risk = float(np.clip(raw_risk, 0.42, 0.78))
        else:
            raw_risk = min(raw_risk, 0.38)

        st.risk_ema = self.risk_ema_alpha * raw_risk + (1.0 - self.risk_ema_alpha) * st.risk_ema
        risk = round(_clip01(st.risk_ema), 4)

        # Soft probability breakdown aligned with risk score
        p_aim = _clip01(1.0 / (1.0 + math.exp(-7.5 * (risk - 0.68))))
        p_passive = _clip01(1.0 / (1.0 + math.exp(7.5 * (risk - 0.34))))
        p_draw = max(0.0, 1.0 - p_aim - p_passive)
        total_p = max(1e-6, p_passive + p_draw + p_aim)
        probs = {
            ThreatState.NO_WEAPON.label: 0.0,
            ThreatState.PASSIVE_PRESENT.label: round(p_passive / total_p, 4),
            ThreatState.DRAWING_READYING.label: round(p_draw / total_p, 4),
            ThreatState.AIMING_IMMINENT.label: round(p_aim / total_p, 4),
        }

        return IntentAssessment(
            pair_key=pair.pair_key,
            weapon_track_id=pair.weapon.track_id,
            person_id=pair.person.person_id if pair.person is not None else None,
            state=st.state,
            risk_score=risk,
            state_probs=probs,
            reasons=reasons,
            weapon_box=list(pair.weapon.box),
            person_box=list(pair.person.box) if pair.person is not None else None,
            features=feature_dict,
        )
