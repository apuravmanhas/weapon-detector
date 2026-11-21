"""
Scale-invariant and FPS-invariant biomechanical + kinematic feature extractor.

Computes a fixed-length 30-dimensional feature vector per tracked (person, weapon) pair
from strictly causal history (current frame + past frames only), ready for both the
Biomechanical Hysteresis FSM and trained Causal 1D-TCN / GRU models.
"""

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from src.association import point_to_box_distance
from src.states import AssociatedPair, COCO_KEYPOINTS

FEATURE_NAMES: tuple[str, ...] = (
    # Ownership & hand coupling (0..4)
    "has_owner",
    "ownership_conf",
    "min_wrist_dist",
    "second_wrist_dist",
    "wrist_pair_dist",
    # Active & support arm biomechanics (5..10)
    "active_elbow_angle_norm",    # deg / 180.0
    "support_elbow_angle_norm",   # deg / 180.0
    "active_shoulder_pitch_norm", # deg / 180.0 (0 = arm hanging down, 0.5 = horizontal)
    "support_shoulder_pitch_norm",
    "active_horiz_reach",         # |wrist_x - shoulder_x| / torso
    "active_wrist_elev",          # (shoulder_y - wrist_y) / torso  (0 = shoulder level, -1 = hip)
    # Weapon geometry & relative placement (11..19)
    "weapon_elev",                # (shoulder_y - weapon_cy) / torso
    "weapon_horiz_offset",        # |weapon_cx - shoulder_x| / torso
    "weapon_aspect_log",          # log(w / h) clipped to [-2, 2]
    "weapon_rel_size",            # weapon_diag / torso
    "weapon_conf",
    "weapon_verifier_prob",
    "muzzle_horiz_align",         # Optional 2-kpt orientation hook: cos(pitch) in [0, 1], else 0.5
    "head_to_weapon_dist",        # min distance from head/eye/nose to weapon box / torso (cheek weld!)
    "weapon_eye_elev",            # (head_y - weapon_cy) / torso (0 = weapon right at eye/sight line)
    # 1st & 2nd order FPS-invariant temporal kinematics (20..27)
    "weapon_up_vel",              # d(weapon_elev)/dt  (torso-lengths / sec, + = rising)
    "active_wrist_up_vel",        # d(active_wrist_elev)/dt
    "active_elbow_vel",           # d(elbow_norm)/dt
    "active_reach_vel",           # d(horiz_reach)/dt
    "weapon_up_accel",            # d2(weapon_elev)/dt2
    "weapon_speed_2d",            # 2D torso-normalized speed / sec
    "recent_peak_up_vel",         # max upward velocity over last ~0.8s (draw memory!)
    "stabilization_score",        # exp(-k * recent_elev_std): high right before trigger pull
    # Visibility / data quality cues (26..27)
    "upper_body_kpt_vis",         # fraction of upper-body keypoints visible
    "track_age_norm",             # min(1.0, history_len / 15.0)
)

FEATURE_DIM = len(FEATURE_NAMES)

_NOSE = COCO_KEYPOINTS["nose"]
_L_EYE, _R_EYE = COCO_KEYPOINTS["left_eye"], COCO_KEYPOINTS["right_eye"]
_L_EAR, _R_EAR = COCO_KEYPOINTS["left_ear"], COCO_KEYPOINTS["right_ear"]
_L_SH, _R_SH = COCO_KEYPOINTS["left_shoulder"], COCO_KEYPOINTS["right_shoulder"]
_L_EL, _R_EL = COCO_KEYPOINTS["left_elbow"], COCO_KEYPOINTS["right_elbow"]
_L_WR, _R_WR = COCO_KEYPOINTS["left_wrist"], COCO_KEYPOINTS["right_wrist"]
_L_HP, _R_HP = COCO_KEYPOINTS["left_hip"], COCO_KEYPOINTS["right_hip"]


def _kpt(kpts: np.ndarray, idx: int, min_conf: float = 0.25) -> tuple[float, float] | None:
    if kpts is None or idx >= len(kpts) or float(kpts[idx, 2]) < min_conf:
        return None
    return float(kpts[idx, 0]), float(kpts[idx, 1])


def _angle_deg(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    """Interior angle ABC in degrees in [0, 180]."""
    ux, uy = a[0] - b[0], a[1] - b[1]
    vx, vy = c[0] - b[0], c[1] - b[1]
    nu = math.hypot(ux, uy)
    nv = math.hypot(vx, vy)
    if nu < 1e-6 or nv < 1e-6:
        return 0.0
    cos_val = max(-1.0, min(1.0, (ux * vx + uy * vy) / (nu * nv)))
    return math.degrees(math.acos(cos_val))


def _arm_pitch_deg(shoulder: tuple[float, float], elbow: tuple[float, float], down_vec: tuple[float, float]) -> float:
    """Angle between torso downward axis and upper arm (0 = hanging down, 90 = horizontal)."""
    ax, ay = elbow[0] - shoulder[0], elbow[1] - shoulder[1]
    na = math.hypot(ax, ay)
    nd = math.hypot(down_vec[0], down_vec[1])
    if na < 1e-6 or nd < 1e-6:
        return 0.0
    cos_val = max(-1.0, min(1.0, (ax * down_vec[0] + ay * down_vec[1]) / (na * nd)))
    return math.degrees(math.acos(cos_val))


@dataclass
class _KinematicSnapshot:
    timestamp_s: float
    weapon_elev: float
    weapon_horiz: float
    wrist_elev: float
    elbow_norm: float
    horiz_reach: float
    weapon_up_vel: float


class KinematicFeatureExtractor:
    """
    Maintains strictly causal per-pair kinematic buffers and emits 28-D feature vectors
    plus human-readable feature dictionaries for explainability and logging.
    """

    def __init__(
        self,
        history_seconds: float = 1.2,
        kpt_conf_thresh: float = 0.25,
        max_buffer_len: int = 90,
    ):
        self.history_seconds = history_seconds
        self.kpt_conf_thresh = kpt_conf_thresh
        self.max_buffer_len = max_buffer_len
        self._buffers: dict[str, deque[_KinematicSnapshot]] = {}
        self._last_seen_s: dict[str, float] = {}

    def reset(self) -> None:
        self._buffers.clear()
        self._last_seen_s.clear()

    def _prune_stale(self, now_s: float) -> None:
        stale = [k for k, ts in self._last_seen_s.items() if (now_s - ts) > 3.0]
        for k in stale:
            self._buffers.pop(k, None)
            self._last_seen_s.pop(k, None)

    def extract(self, pair: AssociatedPair, timestamp_s: float) -> tuple[np.ndarray, dict[str, float]]:
        self._prune_stale(timestamp_s)

        w = pair.weapon
        bw = max(1.0, float(w.box[2] - w.box[0]))
        bh = max(1.0, float(w.box[3] - w.box[1]))
        wcx = 0.5 * (w.box[0] + w.box[2])
        wcy = 0.5 * (w.box[1] + w.box[3])
        torso = max(15.0, float(pair.torso_scale))

        aspect_log = float(np.clip(math.log(bw / bh), -2.0, 2.0))
        rel_size = float(np.clip(math.hypot(bw, bh) / torso, 0.0, 4.0))
        verifier_prob = float(w.verifier_prob) if w.verifier_prob is not None else float(w.confidence)

        # 2-keypoint grip->muzzle horizontal alignment hook (1.0 = perfectly horizontal, 0.0 = vertical)
        if w.grip_xy is not None and w.muzzle_xy is not None and w.orientation_conf >= 0.25:
            dx = w.muzzle_xy[0] - w.grip_xy[0]
            dy = w.muzzle_xy[1] - w.grip_xy[1]
            mag = math.hypot(dx, dy)
            muzzle_horiz_align = abs(dx) / mag if mag > 1e-3 else 0.5
        else:
            # Weak geometric proxy when an external 2-keypoint orientation head is not attached
            muzzle_horiz_align = float(np.clip(0.5 + 0.25 * math.tanh(aspect_log), 0.1, 0.9))

        has_owner = 1.0 if pair.person is not None else 0.0
        min_wd = float(np.clip(pair.min_wrist_dist_norm, 0.0, 3.0))
        second_wd = float(np.clip(max(pair.left_wrist_dist_norm, pair.right_wrist_dist_norm), 0.0, 3.0))
        wrist_pair_dist = 2.0

        active_elbow_norm = 0.0
        support_elbow_norm = 0.0
        active_pitch_norm = 0.0
        support_pitch_norm = 0.0
        active_horiz_reach = 0.0
        active_wrist_elev = -1.0
        weapon_elev = -1.0
        weapon_horiz_offset = 0.0
        head_to_weapon_dist = 2.0
        weapon_eye_elev = -1.5
        upper_vis = 0.0

        if pair.person is not None:
            kpts = pair.person.keypoints
            head_pts = [p for idx in (_NOSE, _L_EYE, _R_EYE, _L_EAR, _R_EAR) if (p := _kpt(kpts, idx, self.kpt_conf_thresh)) is not None]
            ls = _kpt(kpts, _L_SH, self.kpt_conf_thresh)
            rs = _kpt(kpts, _R_SH, self.kpt_conf_thresh)
            le = _kpt(kpts, _L_EL, self.kpt_conf_thresh)
            re = _kpt(kpts, _R_EL, self.kpt_conf_thresh)
            lw = _kpt(kpts, _L_WR, self.kpt_conf_thresh)
            rw = _kpt(kpts, _R_WR, self.kpt_conf_thresh)
            lh = _kpt(kpts, _L_HP, self.kpt_conf_thresh)
            rh = _kpt(kpts, _R_HP, self.kpt_conf_thresh)

            upper_vis = sum(1 for p in (ls, rs, le, re, lw, rw) if p is not None) / 6.0

            # Shoulder reference center (fallback to top-quarter of person box if occluded)
            sh_pts = [p for p in (ls, rs) if p is not None]
            if sh_pts:
                sh_cx = sum(p[0] for p in sh_pts) / len(sh_pts)
                sh_cy = sum(p[1] for p in sh_pts) / len(sh_pts)
            else:
                sh_cx = 0.5 * (pair.person.box[0] + pair.person.box[2])
                sh_cy = pair.person.box[1] + 0.22 * (pair.person.box[3] - pair.person.box[1])

            hip_pts = [p for p in (lh, rh) if p is not None]
            if hip_pts:
                hp_cx = sum(p[0] for p in hip_pts) / len(hip_pts)
                hp_cy = sum(p[1] for p in hip_pts) / len(hip_pts)
                down_vec = (hp_cx - sh_cx, hp_cy - sh_cy)
                if math.hypot(down_vec[0], down_vec[1]) < 5.0:
                    down_vec = (0.0, 1.0)
            else:
                down_vec = (0.0, 1.0)

            weapon_elev = float(np.clip((sh_cy - wcy) / torso, -2.5, 1.5))
            weapon_horiz_offset = float(np.clip(abs(wcx - sh_cx) / torso, 0.0, 3.0))
            if head_pts:
                head_to_weapon_dist = float(
                    np.clip(min(point_to_box_distance(hx, hy, w.box) for hx, hy in head_pts) / torso, 0.0, 3.0)
                )
                head_cy = sum(p[1] for p in head_pts) / len(head_pts)
                weapon_eye_elev = float(np.clip((head_cy - wcy) / torso, -2.5, 1.5))
            else:
                # Approximate head line ~0.32 torso above mid-shoulder when face keypoints are occluded
                est_head_y = sh_cy - 0.32 * torso
                head_to_weapon_dist = float(np.clip(point_to_box_distance(sh_cx, est_head_y, w.box) / torso, 0.0, 3.0))
                weapon_eye_elev = float(np.clip((est_head_y - wcy) / torso, -2.5, 1.5))

            if lw is not None and rw is not None:
                wrist_pair_dist = float(np.clip(math.hypot(lw[0] - rw[0], lw[1] - rw[1]) / torso, 0.0, 3.0))

            # Choose active arm (closer wrist to weapon) and support arm
            left_is_active = pair.left_wrist_dist_norm <= pair.right_wrist_dist_norm
            arms = (
                (ls, le, lw),
                (rs, re, rw),
            ) if left_is_active else (
                (rs, re, rw),
                (ls, le, lw),
            )

            a_sh, a_el, a_wr = arms[0]
            s_sh, s_el, s_wr = arms[1]

            if a_sh is not None and a_el is not None and a_wr is not None:
                active_elbow_norm = _angle_deg(a_sh, a_el, a_wr) / 180.0
            if s_sh is not None and s_el is not None and s_wr is not None:
                support_elbow_norm = _angle_deg(s_sh, s_el, s_wr) / 180.0

            if a_sh is not None and a_el is not None:
                active_pitch_norm = _arm_pitch_deg(a_sh, a_el, down_vec) / 180.0
            elif a_sh is not None and a_wr is not None:
                active_pitch_norm = _arm_pitch_deg(a_sh, a_wr, down_vec) / 180.0

            if s_sh is not None and s_el is not None:
                support_pitch_norm = _arm_pitch_deg(s_sh, s_el, down_vec) / 180.0
            elif s_sh is not None and s_wr is not None:
                support_pitch_norm = _arm_pitch_deg(s_sh, s_wr, down_vec) / 180.0

            if a_wr is not None:
                ref_sx = a_sh[0] if a_sh is not None else sh_cx
                ref_sy = a_sh[1] if a_sh is not None else sh_cy
                active_horiz_reach = float(np.clip(abs(a_wr[0] - ref_sx) / torso, 0.0, 3.0))
                active_wrist_elev = float(np.clip((ref_sy - a_wr[1]) / torso, -2.5, 1.5))
            else:
                active_wrist_elev = weapon_elev
                active_horiz_reach = weapon_horiz_offset

        # Strictly causal temporal derivatives over past snapshots
        buf = self._buffers.setdefault(pair.pair_key, deque(maxlen=self.max_buffer_len))
        self._last_seen_s[pair.pair_key] = timestamp_s

        # Drop snapshots older than history_seconds
        while buf and (timestamp_s - buf[0].timestamp_s) > self.history_seconds:
            buf.popleft()

        weapon_up_vel = 0.0
        wrist_up_vel = 0.0
        elbow_vel = 0.0
        reach_vel = 0.0
        weapon_up_accel = 0.0
        weapon_speed_2d = 0.0

        if buf:
            # Pick a reference snapshot ~0.10–0.20s ago when available for noise-resistant velocity,
            # falling back to the most recent snapshot on early frames.
            ref = buf[-1]
            for cand in reversed(buf):
                if (timestamp_s - cand.timestamp_s) >= 0.08:
                    ref = cand
                    break
            dt = max(1e-3, timestamp_s - ref.timestamp_s)
            weapon_up_vel = float(np.clip((weapon_elev - ref.weapon_elev) / dt, -8.0, 8.0))
            wrist_up_vel = float(np.clip((active_wrist_elev - ref.wrist_elev) / dt, -8.0, 8.0))
            elbow_vel = float(np.clip((active_elbow_norm - ref.elbow_norm) / dt, -5.0, 5.0))
            reach_vel = float(np.clip((active_horiz_reach - ref.horiz_reach) / dt, -8.0, 8.0))
            weapon_up_accel = float(np.clip((weapon_up_vel - ref.weapon_up_vel) / dt, -25.0, 25.0))
            weapon_speed_2d = float(
                np.clip(
                    math.hypot(weapon_elev - ref.weapon_elev, weapon_horiz_offset - ref.weapon_horiz) / dt,
                    0.0,
                    10.0,
                )
            )

        snap = _KinematicSnapshot(
            timestamp_s=timestamp_s,
            weapon_elev=weapon_elev,
            weapon_horiz=weapon_horiz_offset,
            wrist_elev=active_wrist_elev,
            elbow_norm=active_elbow_norm,
            horiz_reach=active_horiz_reach,
            weapon_up_vel=weapon_up_vel,
        )
        buf.append(snap)

        # Peak upward draw velocity within the recent ~0.85s memory window
        recent_snaps = [s for s in buf if (timestamp_s - s.timestamp_s) <= 0.85]
        recent_peak_up_vel = max((s.weapon_up_vel for s in recent_snaps), default=0.0)

        # Stabilization score over last ~0.20s (high when held steady right after raising)
        short_snaps = [s for s in buf if (timestamp_s - s.timestamp_s) <= 0.22]
        if len(short_snaps) >= 3:
            elev_std = float(np.std([s.weapon_elev for s in short_snaps]))
            horiz_std = float(np.std([s.weapon_horiz for s in short_snaps]))
            stabilization_score = float(math.exp(-8.0 * (elev_std + horiz_std)))
        else:
            stabilization_score = 0.0

        track_age_norm = float(min(1.0, len(buf) / 15.0))

        vec = np.array(
            [
                has_owner,
                pair.ownership_confidence,
                min_wd,
                second_wd,
                wrist_pair_dist,
                active_elbow_norm,
                support_elbow_norm,
                active_pitch_norm,
                support_pitch_norm,
                active_horiz_reach,
                active_wrist_elev,
                weapon_elev,
                weapon_horiz_offset,
                aspect_log,
                rel_size,
                float(w.confidence),
                verifier_prob,
                muzzle_horiz_align,
                head_to_weapon_dist,
                weapon_eye_elev,
                weapon_up_vel,
                wrist_up_vel,
                elbow_vel,
                reach_vel,
                weapon_up_accel,
                weapon_speed_2d,
                recent_peak_up_vel,
                stabilization_score,
                upper_vis,
                track_age_norm,
            ],
            dtype=np.float32,
        )

        feat_dict = {name: round(float(val), 4) for name, val in zip(FEATURE_NAMES, vec)}
        return vec, feat_dict
