import unittest

import numpy as np

from src.association import WeaponPersonAssociator
from src.features import FEATURE_DIM, FEATURE_NAMES, KinematicFeatureExtractor
from src.states import COCO_KEYPOINTS, PersonTrack, WeaponObservation


def _build_pose(
    scale: float,
    r_elbow_offset: tuple[float, float],
    r_wrist_offset: tuple[float, float],
    l_wrist_offset: tuple[float, float] = (-0.40, 0.85),
) -> PersonTrack:
    cx, sh_y = 300.0 * scale, 200.0 * scale
    torso = 100.0 * scale
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[COCO_KEYPOINTS["left_shoulder"]] = [cx - 0.35 * torso, sh_y, 0.95]
    kpts[COCO_KEYPOINTS["right_shoulder"]] = [cx + 0.35 * torso, sh_y, 0.95]
    kpts[COCO_KEYPOINTS["left_elbow"]] = [cx - 0.38 * torso, sh_y + 0.45 * torso, 0.95]
    kpts[COCO_KEYPOINTS["left_wrist"]] = [cx + l_wrist_offset[0] * torso, sh_y + l_wrist_offset[1] * torso, 0.95]
    kpts[COCO_KEYPOINTS["right_elbow"]] = [cx + r_elbow_offset[0] * torso, sh_y + r_elbow_offset[1] * torso, 0.95]
    kpts[COCO_KEYPOINTS["right_wrist"]] = [cx + r_wrist_offset[0] * torso, sh_y + r_wrist_offset[1] * torso, 0.95]
    kpts[COCO_KEYPOINTS["left_hip"]] = [cx - 0.30 * torso, sh_y + torso, 0.95]
    kpts[COCO_KEYPOINTS["right_hip"]] = [cx + 0.30 * torso, sh_y + torso, 0.95]
    box = [int(cx - torso), int(sh_y - 0.5 * torso), int(cx + 1.4 * torso), int(sh_y + 2.2 * torso)]
    return PersonTrack(person_id=1, box=box, confidence=0.95, keypoints=kpts)


class TestKinematicFeatures(unittest.TestCase):
    def test_feature_dim_and_scale_invariance(self) -> None:
        # Fully extended horizontal right arm (180 deg elbow, 90 deg shoulder pitch)
        p1 = _build_pose(1.0, r_elbow_offset=(0.80, 0.0), r_wrist_offset=(1.25, 0.0))
        w1 = WeaponObservation(track_id=1, box=[425, 190, 485, 215], confidence=0.86)

        p2 = _build_pose(2.0, r_elbow_offset=(0.80, 0.0), r_wrist_offset=(1.25, 0.0))
        w2 = WeaponObservation(track_id=1, box=[850, 380, 970, 430], confidence=0.86)

        pair1 = WeaponPersonAssociator().associate([w1], [p1])[0]
        pair2 = WeaponPersonAssociator().associate([w2], [p2])[0]

        v1, d1 = KinematicFeatureExtractor().extract(pair1, timestamp_s=0.0)
        v2, d2 = KinematicFeatureExtractor().extract(pair2, timestamp_s=0.0)

        self.assertEqual(len(v1), FEATURE_DIM)
        self.assertEqual(len(FEATURE_NAMES), FEATURE_DIM)
        self.assertAlmostEqual(d1["active_elbow_angle_norm"], 1.0, places=3)
        self.assertAlmostEqual(d1["active_shoulder_pitch_norm"], 0.5, places=2)
        for key in ("weapon_elev", "active_wrist_elev", "active_horiz_reach", "active_elbow_angle_norm"):
            self.assertAlmostEqual(d1[key], d2[key], places=3, msg=f"Scale invariance failed for {key}")

    def test_fps_invariant_upward_draw_velocity(self) -> None:
        # Rise smoothly from elev=-1.0 (hip) to elev=0.0 (shoulder) over 0.5s -> expected vel ~ +2.0 torso/s
        def run_at_fps(fps: float) -> float:
            assoc = WeaponPersonAssociator()
            ext = KinematicFeatureExtractor()
            n_steps = int(round(0.5 * fps))
            last_vel = 0.0
            for i in range(n_steps + 1):
                t = i / fps
                frac = t / 0.5
                wy_off = 1.0 - frac  # 1.0 -> 0.0 below shoulder
                p = _build_pose(1.0, r_elbow_offset=(0.55, 0.5 * wy_off), r_wrist_offset=(0.85, wy_off))
                wcy = 200.0 + wy_off * 100.0
                w = WeaponObservation(track_id=1, box=[380, int(wcy - 12), 435, int(wcy + 12)], confidence=0.85)
                pair = assoc.associate([w], [p])[0]
                _, fd = ext.extract(pair, timestamp_s=t)
                last_vel = fd["weapon_up_vel"]
            return last_vel

        vel_30 = run_at_fps(30.0)
        vel_60 = run_at_fps(60.0)
        self.assertGreater(vel_30, 1.5)
        self.assertAlmostEqual(vel_30, vel_60, delta=0.20)


if __name__ == "__main__":
    unittest.main()
