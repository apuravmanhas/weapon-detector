import unittest

import numpy as np

from src.association import WeaponPersonAssociator, estimate_torso_scale
from src.states import COCO_KEYPOINTS, PersonTrack, WeaponObservation


def _make_person(
    person_id: int,
    cx: float,
    sh_y: float,
    torso: float = 100.0,
    lw_xy: tuple[float, float] | None = None,
    rw_xy: tuple[float, float] | None = None,
    hide_hips: bool = False,
) -> PersonTrack:
    kpts = np.zeros((17, 3), dtype=np.float32)
    sh_half = 0.35 * torso
    kpts[COCO_KEYPOINTS["left_shoulder"]] = [cx - sh_half, sh_y, 0.95]
    kpts[COCO_KEYPOINTS["right_shoulder"]] = [cx + sh_half, sh_y, 0.95]
    if not hide_hips:
        kpts[COCO_KEYPOINTS["left_hip"]] = [cx - 0.3 * torso, sh_y + torso, 0.95]
        kpts[COCO_KEYPOINTS["right_hip"]] = [cx + 0.3 * torso, sh_y + torso, 0.95]
    if lw_xy is not None:
        kpts[COCO_KEYPOINTS["left_wrist"]] = [lw_xy[0], lw_xy[1], 0.95]
    if rw_xy is not None:
        kpts[COCO_KEYPOINTS["right_wrist"]] = [rw_xy[0], rw_xy[1], 0.95]

    box = [
        int(cx - 0.8 * torso),
        int(sh_y - 0.5 * torso),
        int(cx + 0.8 * torso),
        int(sh_y + 2.2 * torso),
    ]
    return PersonTrack(person_id=person_id, box=box, confidence=0.9, keypoints=kpts)


class TestWeaponPersonAssociation(unittest.TestCase):
    def test_table_weapon_is_unowned(self) -> None:
        person = _make_person(1, cx=200.0, sh_y=200.0, torso=100.0, lw_xy=(165.0, 295.0), rw_xy=(235.0, 295.0))
        table_gun = WeaponObservation(track_id=10, box=[560, 420, 620, 455], confidence=0.82)

        assoc = WeaponPersonAssociator()
        pairs = assoc.associate([table_gun], [person])
        self.assertEqual(len(pairs), 1)
        self.assertIsNone(pairs[0].person)
        self.assertEqual(pairs[0].ownership_confidence, 0.0)

    def test_multi_person_assigns_actual_bearer(self) -> None:
        bystander = _make_person(1, cx=180.0, sh_y=200.0, torso=100.0, lw_xy=(145.0, 300.0), rw_xy=(215.0, 300.0))
        bearer = _make_person(2, cx=420.0, sh_y=200.0, torso=100.0, lw_xy=(385.0, 260.0), rw_xy=(485.0, 210.0))
        gun = WeaponObservation(track_id=7, box=[480, 195, 535, 225], confidence=0.88)

        assoc = WeaponPersonAssociator()
        pairs = assoc.associate([gun], [bystander, bearer])
        self.assertEqual(len(pairs), 1)
        self.assertIsNotNone(pairs[0].person)
        self.assertEqual(pairs[0].person.person_id, 2)
        self.assertLess(pairs[0].right_wrist_dist_norm, 0.10)

    def test_waist_up_torso_fallback_and_scale_invariance(self) -> None:
        p1 = _make_person(1, cx=300.0, sh_y=200.0, torso=100.0, rw_xy=(360.0, 200.0), hide_hips=True)
        w1 = WeaponObservation(track_id=1, box=[375, 190, 425, 215], confidence=0.85)
        scale1 = estimate_torso_scale(p1)
        self.assertGreater(scale1, 80.0)

        # Scale entire scene by 2x
        p2 = _make_person(1, cx=600.0, sh_y=400.0, torso=200.0, rw_xy=(720.0, 400.0), hide_hips=True)
        w2 = WeaponObservation(track_id=1, box=[750, 380, 850, 430], confidence=0.85)

        assoc1 = WeaponPersonAssociator()
        assoc2 = WeaponPersonAssociator()
        pair1 = assoc1.associate([w1], [p1])[0]
        pair2 = assoc2.associate([w2], [p2])[0]
        self.assertAlmostEqual(pair1.min_wrist_dist_norm, pair2.min_wrist_dist_norm, places=4)

    def test_pose_anchored_weapon_coasting_and_expiry(self) -> None:
        assoc = WeaponPersonAssociator(enable_coasting=True, max_coast_frames=4)
        p = _make_person(1, cx=300.0, sh_y=200.0, torso=100.0, rw_xy=(380.0, 200.0))
        gun = WeaponObservation(track_id=5, box=[370, 188, 420, 212], confidence=0.82)

        # Real frame locks bearer memory
        real_pairs = assoc.associate([gun], [p])
        self.assertEqual(len(real_pairs), 1)
        self.assertFalse(real_pairs[0].weapon.is_coasted)
        self.assertTrue(assoc.has_active_coasts)

        # Drop detector completely for 4 frames while wrist extends further forward
        for step in range(4):
            moved = _make_person(1, cx=300.0, sh_y=200.0, torso=100.0, rw_xy=(400.0 + step * 5.0, 195.0))
            coasted = assoc.associate([], [moved])
            self.assertEqual(len(coasted), 1)
            self.assertTrue(coasted[0].weapon.is_coasted)
            self.assertEqual(coasted[0].person.person_id, 1)
            self.assertGreater(coasted[0].ownership_confidence, 0.4)

        # Frame 5 must expire strictly after max_coast_frames=4
        self.assertFalse(assoc.has_active_coasts)
        self.assertEqual(assoc.associate([], [p]), [])


if __name__ == "__main__":
    unittest.main()
