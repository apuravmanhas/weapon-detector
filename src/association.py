"""
Scale-invariant weapon-to-person owner association.

Crucially separates weapons physically held/reached for by a person from unowned passive
weapons (e.g., resting on a counter/table or mounted on a wall).
"""

import math
from typing import Any, Sequence

import numpy as np

from src.states import AssociatedPair, COCO_KEYPOINTS, PersonTrack, WeaponObservation

_L_SHOULDER = COCO_KEYPOINTS["left_shoulder"]
_R_SHOULDER = COCO_KEYPOINTS["right_shoulder"]
_L_WRIST = COCO_KEYPOINTS["left_wrist"]
_R_WRIST = COCO_KEYPOINTS["right_wrist"]
_L_HIP = COCO_KEYPOINTS["left_hip"]
_R_HIP = COCO_KEYPOINTS["right_hip"]


def _valid_kpt(kpts: np.ndarray, idx: int, min_conf: float = 0.25) -> tuple[float, float] | None:
    if kpts is None or idx >= len(kpts):
        return None
    if float(kpts[idx, 2]) < min_conf:
        return None
    return float(kpts[idx, 0]), float(kpts[idx, 1])


def estimate_torso_scale(person: PersonTrack, min_conf: float = 0.25) -> float:
    """
    Estimate camera-scale-invariant torso height in pixels using hierarchical fallbacks:
      1. Distance from mid-shoulder to mid-hip (most accurate)
      2. 1.45 * shoulder width (waist-up CCTV where hips are occluded)
      3. 0.34 * person bounding box height (extreme occlusion fallback)
    """
    kpts = person.keypoints
    ls = _valid_kpt(kpts, _L_SHOULDER, min_conf)
    rs = _valid_kpt(kpts, _R_SHOULDER, min_conf)
    lh = _valid_kpt(kpts, _L_HIP, min_conf)
    rh = _valid_kpt(kpts, _R_HIP, min_conf)

    shoulder_pts = [p for p in (ls, rs) if p is not None]
    hip_pts = [p for p in (lh, rh) if p is not None]

    if shoulder_pts and hip_pts:
        sx = sum(p[0] for p in shoulder_pts) / len(shoulder_pts)
        sy = sum(p[1] for p in shoulder_pts) / len(shoulder_pts)
        hx = sum(p[0] for p in hip_pts) / len(hip_pts)
        hy = sum(p[1] for p in hip_pts) / len(hip_pts)
        dist = math.hypot(hx - sx, hy - sy)
        if dist >= 10.0:
            return dist

    if ls is not None and rs is not None:
        shoulder_w = math.hypot(rs[0] - ls[0], rs[1] - ls[1])
        if shoulder_w >= 8.0:
            return max(15.0, 1.45 * shoulder_w)

    box_h = max(1.0, float(person.box[3] - person.box[1]))
    return max(15.0, 0.34 * box_h)


def point_to_box_distance(px: float, py: float, box: Sequence[float]) -> float:
    """Euclidean distance from point (px, py) to nearest point on [x1, y1, x2, y2] (0 if inside)."""
    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return math.hypot(dx, dy)


def box_expanded_contains_center(person_box: Sequence[float], weapon_box: Sequence[float], pad_ratio: float = 0.25) -> bool:
    px1, py1, px2, py2 = map(float, person_box)
    pw = max(1.0, px2 - px1)
    ph = max(1.0, py2 - py1)
    wcx = 0.5 * (float(weapon_box[0]) + float(weapon_box[2]))
    wcy = 0.5 * (float(weapon_box[1]) + float(weapon_box[3]))
    return (px1 - pad_ratio * pw) <= wcx <= (px2 + pad_ratio * pw) and (
        py1 - pad_ratio * ph
    ) <= wcy <= (py2 + pad_ratio * ph)


class WeaponPersonAssociator:
    """
    Associates each confirmed weapon track to its most likely bearer using normalized
    wrist-to-weapon distance, expanded person-box context, temporal owner stickiness,
    and bounded pose-anchored weapon coasting during brief detector dropouts.
    """

    def __init__(
        self,
        max_wrist_dist_norm: float = 0.90,
        kpt_conf_thresh: float = 0.25,
        owner_stickiness_bonus: float = 0.15,
        enable_coasting: bool = True,
        max_coast_frames: int = 12,
        coast_min_own_conf: float = 0.50,
        coast_max_wrist_dist: float = 0.35,
    ):
        self.max_wrist_dist_norm = max_wrist_dist_norm
        self.kpt_conf_thresh = kpt_conf_thresh
        self.owner_stickiness_bonus = owner_stickiness_bonus
        self.enable_coasting = enable_coasting
        self.max_coast_frames = max(0, max_coast_frames)
        self.coast_min_own_conf = coast_min_own_conf
        self.coast_max_wrist_dist = coast_max_wrist_dist
        self._last_owner_by_weapon: dict[int, int] = {}
        self._bearer_memory: dict[int, dict[str, Any]] = {}

    @property
    def has_active_coasts(self) -> bool:
        return self.enable_coasting and any(
            mem["coast_age"] < self.max_coast_frames for mem in self._bearer_memory.values()
        )

    def reset(self) -> None:
        self._last_owner_by_weapon.clear()
        self._bearer_memory.clear()

    def associate(
        self,
        weapons: list[WeaponObservation],
        persons: list[PersonTrack],
    ) -> list[AssociatedPair]:
        pairs: list[AssociatedPair] = []
        matched_person_ids: set[int] = set()
        persons_by_id = {p.person_id: p for p in persons}
        active_wids = {w.track_id for w in weapons}
        # Keep memory bounded to recent weapon tracks
        if len(self._last_owner_by_weapon) > 128:
            self._last_owner_by_weapon = {
                wid: pid for wid, pid in self._last_owner_by_weapon.items() if wid in active_wids
            }

        for weapon in weapons:
            best_person: PersonTrack | None = None
            best_cost = float("inf")
            best_stats: tuple[float, float, float, float] = (9.99, 9.99, 9.99, 100.0)
            prev_owner_id = self._last_owner_by_weapon.get(weapon.track_id)

            wcx = 0.5 * (weapon.box[0] + weapon.box[2])
            wcy = 0.5 * (weapon.box[1] + weapon.box[3])

            for person in persons:
                torso = estimate_torso_scale(person, min_conf=self.kpt_conf_thresh)
                lw = _valid_kpt(person.keypoints, _L_WRIST, self.kpt_conf_thresh)
                rw = _valid_kpt(person.keypoints, _R_WRIST, self.kpt_conf_thresh)

                dl = point_to_box_distance(lw[0], lw[1], weapon.box) / torso if lw is not None else 9.99
                dr = point_to_box_distance(rw[0], rw[1], weapon.box) / torso if rw is not None else 9.99
                min_wd = min(dl, dr)

                in_context = box_expanded_contains_center(person.box, weapon.box, pad_ratio=0.35)
                if min_wd <= self.max_wrist_dist_norm:
                    cost = min_wd
                elif lw is None and rw is None and in_context:
                    # Fallback when both wrists are occluded in rough CCTV: check distance to upper-body center
                    pcx = 0.5 * (person.box[0] + person.box[2])
                    pcy = person.box[1] + 0.38 * (person.box[3] - person.box[1])
                    fallback_d = math.hypot(wcx - pcx, wcy - pcy) / torso
                    if fallback_d > 1.10:
                        continue
                    cost = 0.70 + 0.25 * fallback_d
                    min_wd = min(min_wd, fallback_d)
                else:
                    continue

                if prev_owner_id is not None and person.person_id == prev_owner_id:
                    cost = max(0.0, cost - self.owner_stickiness_bonus)

                if cost < best_cost:
                    best_cost = cost
                    best_person = person
                    best_stats = (min_wd, dl, dr, torso)

            if best_person is not None:
                self._last_owner_by_weapon[weapon.track_id] = best_person.person_id
                matched_person_ids.add(best_person.person_id)
                min_wd, dl, dr, torso = best_stats
                own_conf = float(np.clip(1.0 - (min_wd / max(0.01, self.max_wrist_dist_norm + 0.20)), 0.05, 1.0))

                if (
                    self.enable_coasting
                    and not weapon.is_coasted
                    and min_wd <= self.coast_max_wrist_dist
                    and own_conf >= self.coast_min_own_conf
                    and min(dl, dr) < 9.0
                ):
                    active_hand = "left" if dl <= dr else "right"
                    self._bearer_memory[best_person.person_id] = {
                        "weapon_track_id": weapon.track_id,
                        "w": float(max(12, weapon.box[2] - weapon.box[0])),
                        "h": float(max(12, weapon.box[3] - weapon.box[1])),
                        "active_hand": active_hand,
                        "confidence": float(weapon.confidence),
                        "verifier_prob": weapon.verifier_prob,
                        "cls_name": weapon.cls_name,
                        "coast_age": 0,
                        "person_misses": 0,
                    }

                pairs.append(
                    AssociatedPair(
                        pair_key=f"w{weapon.track_id}_p{best_person.person_id}",
                        weapon=weapon,
                        person=best_person,
                        min_wrist_dist_norm=min_wd,
                        left_wrist_dist_norm=dl,
                        right_wrist_dist_norm=dr,
                        torso_scale=torso,
                        ownership_confidence=own_conf,
                    )
                )
            else:
                pairs.append(
                    AssociatedPair(
                        pair_key=f"w{weapon.track_id}_unowned",
                        weapon=weapon,
                        person=None,
                        min_wrist_dist_norm=9.99,
                        left_wrist_dist_norm=9.99,
                        right_wrist_dist_norm=9.99,
                        torso_scale=max(15.0, float(weapon.box[3] - weapon.box[1]) * 2.0),
                        ownership_confidence=0.0,
                    )
                )

        if self.enable_coasting:
            for pid in list(self._bearer_memory.keys()):
                if pid in matched_person_ids:
                    continue
                mem = self._bearer_memory[pid]
                person = persons_by_id.get(pid)
                if person is None:
                    mem["person_misses"] = mem.get("person_misses", 0) + 1
                    if mem["person_misses"] > 2:
                        del self._bearer_memory[pid]
                    continue

                mem["person_misses"] = 0
                wrist_idx = _L_WRIST if mem["active_hand"] == "left" else _R_WRIST
                wrist_xy = _valid_kpt(person.keypoints, wrist_idx, self.kpt_conf_thresh)
                if wrist_xy is None or mem["coast_age"] >= self.max_coast_frames:
                    del self._bearer_memory[pid]
                    continue

                mem["coast_age"] += 1
                age = mem["coast_age"]
                half_w = 0.5 * mem["w"]
                half_h = 0.5 * mem["h"]
                wx, wy = wrist_xy
                coast_box = [
                    int(round(wx - half_w)),
                    int(round(wy - half_h)),
                    int(round(wx + half_w)),
                    int(round(wy + half_h)),
                ]
                coasted_weapon = WeaponObservation(
                    track_id=mem["weapon_track_id"],
                    box=coast_box,
                    confidence=round(float(mem["confidence"] * (0.92 ** age)), 4),
                    cls_name=mem["cls_name"],
                    verifier_prob=mem["verifier_prob"],
                    is_coasted=True,
                )

                torso = estimate_torso_scale(person, min_conf=self.kpt_conf_thresh)
                lw = _valid_kpt(person.keypoints, _L_WRIST, self.kpt_conf_thresh)
                rw = _valid_kpt(person.keypoints, _R_WRIST, self.kpt_conf_thresh)
                if mem["active_hand"] == "left":
                    dl = 0.0
                    dr = point_to_box_distance(rw[0], rw[1], coast_box) / torso if rw is not None else 9.99
                else:
                    dl = point_to_box_distance(lw[0], lw[1], coast_box) / torso if lw is not None else 9.99
                    dr = 0.0

                pairs.append(
                    AssociatedPair(
                        pair_key=f"w{mem['weapon_track_id']}_p{pid}",
                        weapon=coasted_weapon,
                        person=person,
                        min_wrist_dist_norm=0.0,
                        left_wrist_dist_norm=dl,
                        right_wrist_dist_norm=dr,
                        torso_scale=torso,
                        ownership_confidence=round(max(0.45, 0.85 * (0.94 ** age)), 4),
                    )
                )
                if age >= self.max_coast_frames:
                    del self._bearer_memory[pid]

        return pairs
