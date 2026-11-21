import unittest

import numpy as np

from src.association import WeaponPersonAssociator
from src.features import KinematicFeatureExtractor
from src.intent_fsm import BiomechanicalIntentFSM
from src.states import COCO_KEYPOINTS, PersonTrack, ThreatState, WeaponObservation


def _pose_with_arms(
    r_el: tuple[float, float],
    r_wr: tuple[float, float],
    l_el: tuple[float, float] = (-0.40, 0.48),
    l_wr: tuple[float, float] = (-0.42, 0.92),
    nose_offset: tuple[float, float] = (0.0, -0.36),
) -> PersonTrack:
    cx, sh_y, torso = 320.0, 180.0, 100.0
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[COCO_KEYPOINTS["nose"]] = [cx + nose_offset[0] * torso, sh_y + nose_offset[1] * torso, 0.95]
    kpts[COCO_KEYPOINTS["left_shoulder"]] = [cx - 35.0, sh_y, 0.95]
    kpts[COCO_KEYPOINTS["right_shoulder"]] = [cx + 35.0, sh_y, 0.95]
    kpts[COCO_KEYPOINTS["left_elbow"]] = [cx + l_el[0] * torso, sh_y + l_el[1] * torso, 0.95]
    kpts[COCO_KEYPOINTS["left_wrist"]] = [cx + l_wr[0] * torso, sh_y + l_wr[1] * torso, 0.95]
    kpts[COCO_KEYPOINTS["right_elbow"]] = [cx + r_el[0] * torso, sh_y + r_el[1] * torso, 0.95]
    kpts[COCO_KEYPOINTS["right_wrist"]] = [cx + r_wr[0] * torso, sh_y + r_wr[1] * torso, 0.95]
    kpts[COCO_KEYPOINTS["left_hip"]] = [cx - 30.0, sh_y + torso, 0.95]
    kpts[COCO_KEYPOINTS["right_hip"]] = [cx + 30.0, sh_y + torso, 0.95]
    return PersonTrack(person_id=1, box=[220, 120, 440, 400], confidence=0.95, keypoints=kpts)


class TestBiomechanicalIntentFSM(unittest.TestCase):
    def test_unowned_table_weapon_never_escalates(self) -> None:
        assoc = WeaponPersonAssociator()
        ext = KinematicFeatureExtractor()
        fsm = BiomechanicalIntentFSM()

        person = _pose_with_arms(r_el=(0.40, 0.45), r_wr=(0.42, 0.92))
        table_gun = WeaponObservation(track_id=1, box=[620, 350, 700, 385], confidence=0.90)

        for i in range(25):
            ts = i / 30.0
            pair = assoc.associate([table_gun], [person])[0]
            vec, fd = ext.extract(pair, timestamp_s=ts)
            out = fsm.update(pair, vec, fd, timestamp_s=ts)
            self.assertEqual(out.state, ThreatState.PASSIVE_PRESENT)
            self.assertLessEqual(out.risk_score, 0.20)

    def test_passive_low_carry_stays_passive(self) -> None:
        assoc = WeaponPersonAssociator()
        ext = KinematicFeatureExtractor()
        fsm = BiomechanicalIntentFSM()

        # Weapon held relaxed by right hip (muzzle down, arm hanging straight down)
        person = _pose_with_arms(r_el=(0.38, 0.50), r_wr=(0.40, 0.98))
        gun = WeaponObservation(track_id=2, box=[350, 275, 382, 330], confidence=0.87)

        for i in range(30):
            ts = i / 30.0
            pair = assoc.associate([gun], [person])[0]
            vec, fd = ext.extract(pair, timestamp_s=ts)
            out = fsm.update(pair, vec, fd, timestamp_s=ts)
            self.assertEqual(out.state, ThreatState.PASSIVE_PRESENT)

    def test_draw_raise_aim_triggers_drawing_then_aiming_before_shot(self) -> None:
        assoc = WeaponPersonAssociator()
        ext = KinematicFeatureExtractor()
        fsm = BiomechanicalIntentFSM()

        fps = 30.0
        t_shot_s = 1.40
        states: list[tuple[float, ThreatState]] = []

        # 0.00 - 0.40s: passive at hip
        # 0.40 - 0.90s: rapid upward draw + arm extension
        # 0.90 - 1.50s: extended horizontal aiming stance before shot at 1.40s
        for i in range(46):
            ts = i / fps
            if ts < 0.40:
                frac = 0.0
            elif ts < 0.90:
                frac = (ts - 0.40) / 0.50
            else:
                frac = 1.0

            r_el = (0.38 + 0.42 * frac, 0.48 * (1.0 - frac))
            r_wr = (0.40 + 0.85 * frac, 0.95 * (1.0 - frac))
            person = _pose_with_arms(r_el=r_el, r_wr=r_wr)
            wcx = 320.0 + r_wr[0] * 100.0 + 18.0
            wcy = 180.0 + r_wr[1] * 100.0
            gun = WeaponObservation(
                track_id=3,
                box=[int(wcx - 25), int(wcy - 12), int(wcx + 25), int(wcy + 12)],
                confidence=0.89,
            )
            pair = assoc.associate([gun], [person])[0]
            vec, fd = ext.extract(pair, timestamp_s=ts)
            out = fsm.update(pair, vec, fd, timestamp_s=ts)
            states.append((ts, out.state))

        # Initially passive
        self.assertEqual(states[5][1], ThreatState.PASSIVE_PRESENT)

        draw_times = [t for t, s in states if s >= ThreatState.DRAWING_READYING]
        aim_times = [t for t, s in states if s == ThreatState.AIMING_IMMINENT]

        self.assertTrue(draw_times, "Expected DRAWING_READYING trigger during draw.")
        self.assertTrue(aim_times, "Expected AIMING_IMMINENT trigger during aiming.")
        self.assertLess(draw_times[0], aim_times[0], "DRAWING should precede AIMING.")
        self.assertLess(aim_times[0], t_shot_s - 0.30, "AIMING should fire with >300ms lead time before shot.")

    def test_two_hand_shouldered_rifle_triggers_aiming_despite_bent_trigger_elbow(self) -> None:
        assoc = WeaponPersonAssociator()
        ext = KinematicFeatureExtractor()
        fsm = BiomechanicalIntentFSM()

        # Classic shouldered rifle stance: trigger elbow bent (~75 deg), support hand forward on barrel,
        # cheek weld near top of receiver, wide horizontal rifle box right across shoulder/cheek plane.
        person = _pose_with_arms(
            r_el=(0.50, 0.35),
            r_wr=(0.45, 0.06),
            l_el=(0.65, 0.25),
            l_wr=(0.95, 0.04),
            nose_offset=(0.35, -0.15),
        )
        rifle = WeaponObservation(track_id=9, box=[345, 168, 470, 202], confidence=0.91)

        final_state = ThreatState.PASSIVE_PRESENT
        for i in range(6):
            ts = i / 30.0
            pair = assoc.associate([rifle], [person])[0]
            vec, fd = ext.extract(pair, timestamp_s=ts)
            final_state = fsm.update(pair, vec, fd, timestamp_s=ts).state

        self.assertEqual(final_state, ThreatState.AIMING_IMMINENT)

    def test_port_arms_chest_carry_stays_drawing_not_aiming(self) -> None:
        assoc = WeaponPersonAssociator()
        ext = KinematicFeatureExtractor()
        fsm = BiomechanicalIntentFSM()

        # Holding rifle horizontally across chest with upper arms hanging down and head upright
        # (e.g. talking to camera / inspecting weapon) -> must NOT fire AIMING_IMMINENT!
        person = _pose_with_arms(
            r_el=(0.38, 0.48),
            r_wr=(0.15, 0.36),
            l_el=(-0.38, 0.48),
            l_wr=(0.45, 0.34),
            nose_offset=(0.0, -0.45),
        )
        rifle = WeaponObservation(track_id=10, box=[305, 200, 440, 232], confidence=0.90)

        states = []
        for i in range(10):
            ts = i / 30.0
            pair = assoc.associate([rifle], [person])[0]
            vec, fd = ext.extract(pair, timestamp_s=ts)
            states.append(fsm.update(pair, vec, fd, timestamp_s=ts).state)

        self.assertNotIn(ThreatState.AIMING_IMMINENT, states)
        self.assertEqual(states[-1], ThreatState.DRAWING_READYING)

    def test_vertical_slung_rifle_never_triggers_aiming_even_at_chest(self) -> None:
        assoc = WeaponPersonAssociator()
        ext = KinematicFeatureExtractor()
        fsm = BiomechanicalIntentFSM()

        # Reproduces p01_sling_indoor: vertical rifle box (120x401, asp=-1.21) close to chest/chin
        person = _pose_with_arms(
            r_el=(0.25, 0.42),
            r_wr=(0.15, -0.05),
            l_el=(-0.25, 0.40),
            l_wr=(0.22, -0.02),
            nose_offset=(0.0, -0.35),
        )
        vert_rifle = WeaponObservation(track_id=11, box=[260, 120, 380, 521], confidence=0.88)

        states = []
        for i in range(12):
            ts = i / 30.0
            pair = assoc.associate([vert_rifle], [person])[0]
            vec, fd = ext.extract(pair, timestamp_s=ts)
            states.append(fsm.update(pair, vec, fd, timestamp_s=ts).state)

        self.assertNotIn(ThreatState.AIMING_IMMINENT, states)

    def test_front_facing_cheek_weld_rifle_escalates_to_aiming(self) -> None:
        assoc = WeaponPersonAssociator()
        ext = KinematicFeatureExtractor()
        fsm = BiomechanicalIntentFSM()

        # Reproduces d02_snow_rifle_front: elbows tucked (el=61 deg), both hands close at shoulder plane,
        # horizontal foreshortened rifle box (asp=0.31), cheek-welded right to nose/eye line (hd=0.00).
        person = _pose_with_arms(
            r_el=(0.28, 0.25),
            r_wr=(0.08, 0.12),
            l_el=(-0.28, 0.25),
            l_wr=(0.12, 0.10),
            nose_offset=(0.0, -0.25),
        )
        front_rifle = WeaponObservation(track_id=12, box=[270, 130, 420, 234], confidence=0.81)

        states = []
        for i in range(6):
            ts = i / 30.0
            pair = assoc.associate([front_rifle], [person])[0]
            vec, fd = ext.extract(pair, timestamp_s=ts)
            states.append(fsm.update(pair, vec, fd, timestamp_s=ts).state)

        self.assertEqual(states[-1], ThreatState.AIMING_IMMINENT)


if __name__ == "__main__":
    unittest.main()
