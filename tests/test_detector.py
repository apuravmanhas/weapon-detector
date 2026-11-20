import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from detect import sanitize_source_label
from src.detector import (
    REPO_ROOT,
    TemporalPersistenceFilter,
    WeaponDetector,
    apply_lab_clahe,
    filter_detections,
    is_lfs_pointer,
    nms_merge,
    resolve_weights_path,
)
from src.tracker import WeaponTrackManager
from src.verifier import CropVerifier, build_classifier, extract_padded_crop
from tools.mine_hard_negatives import HardNegativeCropRecorder


class TestSecurityAndResolution(unittest.TestCase):
    def test_sanitize_source_label_masks_rtsp_credentials(self):
        masked = sanitize_source_label("rtsp://sec_cam:P@ssw0rd123@10.20.30.40:554/stream1")
        self.assertEqual(masked, "rtsp://***:***@10.20.30.40:554/stream1")
        self.assertNotIn("sec_cam", masked)
        self.assertNotIn("P@ssw0rd123", masked)

    def test_sanitize_source_label_preserves_plain_inputs(self):
        self.assertEqual(sanitize_source_label("rtsp://10.0.0.5:554/live"), "rtsp://10.0.0.5:554/live")
        self.assertEqual(sanitize_source_label("assets/sample.jpg"), "assets/sample.jpg")
        self.assertEqual(sanitize_source_label("0"), "0")

    def test_lfs_pointer_detection_raises_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ptr = Path(tmpdir) / "fake_model.pt"
            ptr.write_text(
                "version https://git-lfs.github.com/spec/v1\n"
                "oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
                "size 114474313\n",
                encoding="utf-8",
            )
            self.assertTrue(is_lfs_pointer(ptr))
            with self.assertRaises(RuntimeError) as ctx:
                resolve_weights_path(ptr)
            self.assertIn("git lfs pull", str(ctx.exception))

            binary_stub = Path(tmpdir) / "real.bin"
            binary_stub.write_bytes(b"\x80\x02" * 600)
            self.assertFalse(is_lfs_pointer(binary_stub))


class TestDetectionFiltering(unittest.TestCase):
    def test_drops_oversized_and_nested_sub_boxes(self):
        detections = [
            {"box": [0, 0, 1112, 1108], "confidence": 0.90, "class": "weapon"},  # full-frame artifact
            {"box": [450, 736, 582, 912], "confidence": 0.67, "class": "weapon"},  # primary rifle
            {"box": [523, 839, 579, 907], "confidence": 0.37, "class": "weapon"},  # nested barrel tip
            {"box": [50, 100, 120, 180], "confidence": 0.55, "class": "weapon"},   # separate weapon
            {"box": [200, 200, 200, 250], "confidence": 0.80, "class": "weapon"},  # zero width
        ]
        kept = filter_detections(detections, frame_shape=(1108, 1112, 3))
        self.assertEqual(len(kept), 2)
        self.assertEqual(kept[0]["box"], [450, 736, 582, 912])
        self.assertEqual(kept[1]["box"], [50, 100, 120, 180])

    def test_clamps_negative_and_overflow_bounds(self):
        detections = [{"box": [-15, -5, 120, 140], "confidence": 0.80, "class": "weapon"}]
        kept = filter_detections(detections, frame_shape=(480, 640, 3))
        self.assertEqual(kept[0]["box"], [0, 0, 120, 140])

    def test_temporal_persistence_filter_debounces_transient_spikes(self):
        tfilter = TemporalPersistenceFilter(min_hits=3, window_size=5, match_iou=0.25)
        box_a = [{"box": [100, 100, 200, 200], "confidence": 0.70, "class": "weapon"}]
        transient = [{"box": [400, 300, 480, 360], "confidence": 0.40, "class": "weapon"}]

        self.assertEqual(tfilter.update(transient), [])  # Frame 1: transient flash rejected
        self.assertEqual(tfilter.update(box_a), [])      # Frame 2: 1st hit
        self.assertEqual(tfilter.update(box_a), [])      # Frame 3: 2nd hit
        confirmed = tfilter.update(box_a)                # Frame 4: 3rd hit confirms
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0]["box"], [100, 100, 200, 200])

    def test_temporal_filter_validation(self):
        with self.assertRaises(ValueError):
            TemporalPersistenceFilter(min_hits=0)
        with self.assertRaises(ValueError):
            TemporalPersistenceFilter(min_hits=4, window_size=2)
        with self.assertRaises(ValueError):
            TemporalPersistenceFilter(match_iou=1.5)


class TestDetectorAPI(unittest.TestCase):
    def test_invalid_thresholds_raise(self):
        with self.assertRaises(ValueError):
            WeaponDetector(conf_threshold=0.0)
        with self.assertRaises(ValueError):
            WeaponDetector(iou_threshold=1.5)
        with self.assertRaises(FileNotFoundError):
            WeaponDetector(weights_path="nonexistent_model_xyz.pt")

    def test_annotate_frame_handles_top_edge_box_without_mutating_input(self):
        class DummyModel:
            names = {0: "weapon"}
            args = {"imgsz": 640}

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_pt = Path(tmpdir) / "best.pt"
            fake_pt.write_bytes(b"stub-weights-content" * 100)
            with patch("src.detector._get_yolo_class", return_value=lambda _: DummyModel()):
                det = WeaponDetector(weights_path=fake_pt, device="cpu")

        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        annotated, out_dets = det.annotate_frame(
            frame,
            [{"box": [10, 0, 90, 60], "confidence": 0.85, "class": "weapon", "track_id": 7}],
            show_conf=True,
            show_track_id=True,
        )
        self.assertEqual(len(out_dets), 1)
        self.assertEqual(out_dets[0]["track_id"], 7)
        self.assertTrue(np.array_equal(frame, np.zeros((240, 320, 3), dtype=np.uint8)))
        self.assertGreater(int(annotated.sum()), 0)

    def test_auto_imgsz_matches_model_variant(self):
        class DummyX:
            names = {0: "weapon"}
            args = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            x_pt = Path(tmpdir) / "yolo11x_weapon_production.pt"
            x_pt.write_bytes(b"stub-weights-content" * 100)
            m_pt = Path(tmpdir) / "best.pt"
            m_pt.write_bytes(b"stub-weights-content" * 100)
            with patch("src.detector._get_yolo_class", return_value=lambda _: DummyX()):
                det_x = WeaponDetector(weights_path=x_pt, device="cpu")
                det_m = WeaponDetector(weights_path=m_pt, device="cpu")
        self.assertEqual(det_x.default_imgsz, 800)
        self.assertEqual(det_m.default_imgsz, 640)

    def test_multiclass_checkpoint_ignores_distractors(self):
        class FakeBox:
            def __init__(self, xyxy, conf, cls_id):
                self.xyxy = [xyxy]
                self.conf = [conf]
                self.cls = [cls_id]

        class FakeResult:
            def __init__(self, boxes):
                self.boxes = boxes

        class DummyMultiModel:
            def __init__(self):
                self.names = {0: "weapon", 1: "phone", 2: "umbrella"}
                self.args = {"imgsz": 800}
                self.last_kwargs = None

            def predict(self, **kwargs):
                self.last_kwargs = kwargs
                return []

        class DummySingleModel:
            def __init__(self):
                self.names = {0: "gun"}
                self.args = {"imgsz": 640}
                self.last_kwargs = None

            def predict(self, **kwargs):
                self.last_kwargs = kwargs
                return []

        with tempfile.TemporaryDirectory() as tmpdir:
            multi_pt = Path(tmpdir) / "yolo11x_weapon_v2_ep13.pt"
            multi_pt.write_bytes(b"stub-weights-content" * 100)
            single_pt = Path(tmpdir) / "best.pt"
            single_pt.write_bytes(b"stub-weights-content" * 100)

            multi_instance = DummyMultiModel()
            single_instance = DummySingleModel()
            models_by_path = {str(multi_pt): multi_instance, str(single_pt): single_instance}

            with patch("src.detector._get_yolo_class", return_value=lambda p: models_by_path[str(p)]):
                det_multi = WeaponDetector(weights_path=multi_pt, device="cpu")
                det_single = WeaponDetector(weights_path=single_pt, device="cpu")

        self.assertEqual(det_multi.weapon_class_ids, {0})
        self.assertEqual(det_multi.distractor_class_ids, {1, 2})
        self.assertEqual(det_multi.class_names, {0: "weapon", 1: "phone", 2: "umbrella"})
        self.assertEqual(det_single.weapon_class_ids, {0})
        self.assertEqual(det_single.distractor_class_ids, set())

        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        det_multi.predict(frame)
        self.assertEqual(multi_instance.last_kwargs.get("classes"), [0])

        det_single.predict(frame)
        self.assertNotIn("classes", single_instance.last_kwargs)

        fake_results = [
            FakeResult([
                FakeBox([10, 10, 80, 70], 0.78, 0),    # weapon -> keep
                FakeBox([100, 30, 150, 90], 0.88, 1),  # phone -> drop
                FakeBox([160, 20, 260, 150], 0.82, 2), # umbrella -> drop
            ])
        ]
        filtered = det_multi.extract_detections(fake_results, frame.shape)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["box"], [10, 10, 80, 70])
        self.assertEqual(filtered[0]["class"], "weapon")

    def test_sample_image_inference_smoke(self):
        sample_path = REPO_ROOT / "assets" / "sample.jpg"
        weights_path = REPO_ROOT / "weights" / "yolo11x_weapon_production.pt"
        if not weights_path.is_file() or is_lfs_pointer(weights_path):
            self.skipTest("Production Git LFS weights not present locally.")

        import cv2
        try:
            detector = WeaponDetector(weights_path=weights_path, conf_threshold=0.35)
        except PermissionError as exc:
            self.skipTest(f"Restricted sandbox environment blocked Ultralytics git check: {exc}")

        self.assertEqual(detector.default_imgsz, 800)
        frame = cv2.imread(str(sample_path))
        self.assertIsNotNone(frame)
        results = detector.predict(frame)
        detections = detector.extract_detections(results, frame.shape)
        self.assertEqual(len(detections), 1)
        self.assertGreater(detections[0]["confidence"], 0.50)


class TestSmartTracker(unittest.TestCase):
    def test_fast_motion_velocity_continuity_and_high_conf_bypass(self):
        tracker = WeaponTrackManager(min_hits=2, window_size=5, high_conf_bypass=0.80)
        # Rapid lateral swing: Frame 1 -> 2 has IoU 0.14 (< 0.20) and norm_dist 0.60 (<= 0.65)
        out1 = tracker.update([{"box": [100, 100, 180, 160], "confidence": 0.60, "class": "weapon"}])
        self.assertEqual(out1, [])  # 1st hit < min_hits and < high_conf_bypass

        out2 = tracker.update([{"box": [160, 100, 240, 160], "confidence": 0.65, "class": "weapon"}])
        self.assertEqual(len(out2), 1)
        self.assertEqual(out2[0]["track_id"], 1)

        # Frame 3 jumps 85px (norm_dist 0.85 without velocity, ~0.46 with learned vx)
        # plus a second new high-confidence track >= 0.80 that bypasses min_hits delay
        out3 = tracker.update([
            {"box": [245, 100, 325, 160], "confidence": 0.65, "class": "weapon"},
            {"box": [400, 200, 500, 280], "confidence": 0.88, "class": "weapon"},
        ])
        self.assertEqual(len(out3), 2)
        tids = sorted(d["track_id"] for d in out3)
        self.assertEqual(tids, [1, 2])

    def test_static_confounder_quarantine_preserves_real_targets(self):
        tracker = WeaponTrackManager(
            min_hits=1,
            static_suppress=True,
            static_frames=4,
            static_mad=4.0,
            static_max_conf=0.58,
        )
        rng = np.random.default_rng(123)
        static_frame = rng.integers(30, 200, size=(360, 640, 3), dtype=np.uint8)

        table_box = {"box": [420, 280, 580, 345], "confidence": 0.41, "class": "weapon"}
        strong_rifle = {"box": [120, 90, 340, 170], "confidence": 0.79, "class": "weapon"}

        quarantined_step = None
        for step in range(8):
            f_step = static_frame.copy()
            res = tracker.update([table_box, strong_rifle], frame=f_step)
            confs = [round(d["confidence"], 2) for d in res]
            if 0.41 not in confs and quarantined_step is None:
                quarantined_step = step
            self.assertIn(0.79, confs)  # High-confidence weapon never suppressed

        self.assertIsNotNone(quarantined_step)
        self.assertGreaterEqual(quarantined_step, 4)

        # If pixel texture inside the low-conf box starts moving, quarantine lifts
        active_frame = static_frame.copy()
        active_frame[280:345, 420:580] = 255 - active_frame[280:345, 420:580]
        res_moved = tracker.update([table_box, strong_rifle], frame=active_frame)
        self.assertEqual(len(res_moved), 2)

    def test_verifier_ema_suppresses_oblique_distractor_tracks(self):
        tracker = WeaponTrackManager(min_hits=1, verify_thresh=0.50, high_conf_bypass=0.85)
        res1 = tracker.update([{"box": [50, 50, 130, 130], "confidence": 0.55, "class": "weapon", "verifier_prob": 0.20}])
        self.assertEqual(res1, [])  # 0.20 < 0.50

        res2 = tracker.update([{"box": [55, 50, 135, 130], "confidence": 0.55, "class": "weapon", "verifier_prob": 0.25}])
        self.assertEqual(res2, [])  # EMA stays ~0.22 < 0.50

        res3 = tracker.update([{"box": [55, 50, 135, 130], "confidence": 0.55, "class": "weapon", "verifier_prob": 0.92}])
        # EMA jumps to 0.45*0.92 + 0.55*0.2225 = 0.536 >= 0.50
        self.assertEqual(len(res3), 1)
        self.assertGreaterEqual(res3[0]["verifier_ema"], 0.50)


class TestStage2CropVerifier(unittest.TestCase):
    def test_extract_padded_crop_clamps_borders(self):
        frame = np.full((100, 120, 3), 180, dtype=np.uint8)
        crop = extract_padded_crop(frame, [-10, 0, 20, 30], pad_ratio=0.30)
        self.assertEqual(crop.ndim, 3)
        self.assertEqual(crop.shape[2], 3)
        self.assertGreater(crop.shape[0], 30)
        self.assertGreater(crop.shape[1], 20)

    def test_checkpoint_roundtrip_and_selective_verification(self):
        classes = ["weapon", "phone", "other_bg"]
        model = build_classifier("mobilenet_v3_large", num_classes=len(classes))
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "verifier_mbv3.pt"
            torch.save(
                {
                    "arch": "mobilenet_v3_large",
                    "classes": classes,
                    "weapon_classes": ["weapon"],
                    "img_size": 96,
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                    "state_dict": model.state_dict(),
                },
                ckpt_path,
            )

            verifier = CropVerifier(ckpt_path, device="cpu", fp16=False)
            frame = np.full((240, 320, 3), 110, dtype=np.uint8)
            dets = [
                {"box": [20, 20, 90, 90], "confidence": 0.92, "class": "weapon"},    # bypasses stage 2
                {"box": [120, 80, 190, 160], "confidence": 0.45, "class": "weapon"},  # scored by stage 2
            ]
            scored = verifier.verify_detections(frame, dets, verify_below_conf=0.85, threshold=0.0, filter_out=True)
            self.assertEqual(len(scored), 2)
            self.assertNotIn("verifier_prob", scored[0])
            self.assertIn("verifier_prob", scored[1])
            self.assertIn(scored[1]["verifier_label"], classes)

            # Force strict threshold=1.01 so uncertain box drops while >=0.85 bypass stays
            strict_out = verifier.verify_detections(frame, dets, verify_below_conf=0.85, threshold=1.01, filter_out=True)
            self.assertEqual(len(strict_out), 1)
            self.assertEqual(strict_out[0]["confidence"], 0.92)

    def test_lfs_pointer_verifier_raises_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ptr = Path(tmpdir) / "crop_verifier_convnext_tiny.pt"
            ptr.write_text(
                "version https://git-lfs.github.com/spec/v1\n"
                "oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
                "size 111356023\n",
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError) as ctx:
                CropVerifier(ptr)
            self.assertIn("git lfs pull", str(ctx.exception))


class TestDenseAndTiledInference(unittest.TestCase):
    def test_apply_lab_clahe_preserves_shape_and_lifts_dark_contrast(self):
        dark = np.full((160, 200, 3), 25, dtype=np.uint8)
        dark[40:120, 40:160] = 15
        dark[60:100, 65:135] = 35
        enhanced = apply_lab_clahe(dark)
        self.assertEqual(enhanced.shape, dark.shape)
        self.assertEqual(enhanced.dtype, np.uint8)
        raw_range = int(dark.max()) - int(dark.min())
        enh_range = int(enhanced.max()) - int(enhanced.min())
        self.assertGreater(enh_range, raw_range)

    def test_nms_merge_and_predict_tiled_translation(self):
        merged = nms_merge([
            {"box": [10, 10, 100, 100], "confidence": 0.90, "class": "weapon"},
            {"box": [12, 12, 102, 102], "confidence": 0.60, "class": "weapon"},  # high IoU dup
            {"box": [200, 200, 280, 280], "confidence": 0.70, "class": "weapon"},
        ], iou_threshold=0.58)
        self.assertEqual(len(merged), 2)

        class FakeBox:
            def __init__(self, xyxy, conf, cls_id):
                self.xyxy = [xyxy]
                self.conf = [conf]
                self.cls = [cls_id]

        class FakeResult:
            def __init__(self, boxes):
                self.boxes = boxes

        class DummyTiledModel:
            def __init__(self):
                self.names = {0: "weapon", 1: "phone"}
                self.args = {"imgsz": 800}
                self.calls = 0

            def predict(self, source, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    # Full frame (800x1000) returns lower-conf box in bottom-right area
                    return [FakeResult([FakeBox([760, 620, 880, 710], 0.55, 0)])]
                elif self.calls == 5:
                    # Bottom-right tile (x0=375, y0=300) detects same rifle higher + a distractor phone
                    return [FakeResult([
                        FakeBox([385, 320, 505, 410], 0.85, 0),  # shifts to [760, 620, 880, 710]
                        FakeBox([20, 20, 80, 100], 0.90, 1),     # phone -> ignored
                    ])]
                return [FakeResult([])]

        with tempfile.TemporaryDirectory() as tmpdir:
            pt = Path(tmpdir) / "yolo11x_test.pt"
            pt.write_bytes(b"stub" * 400)
            dummy = DummyTiledModel()
            with patch("src.detector._get_yolo_class", return_value=lambda _: dummy):
                det = WeaponDetector(weights_path=pt, conf_threshold=0.20, nested_threshold=0.95, device="cpu")

            frame = np.full((800, 1000, 3), 64, dtype=np.uint8)
            out = det.predict_tiled(frame, use_clahe=True)

        self.assertEqual(dummy.calls, 5)  # 1 full + 4 overlapping tiles
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["box"], [760, 620, 880, 710])
        self.assertAlmostEqual(out[0]["confidence"], 0.85, places=3)


class TestHardNegativeMiner(unittest.TestCase):
    def test_duplicate_throttling_budget_and_zip_packaging(self):
        rng = np.random.default_rng(7)
        frame = rng.integers(0, 255, size=(300, 400, 3), dtype=np.uint8)
        det_a = [{"box": [40, 40, 140, 140], "confidence": 0.42, "class": "weapon"}]
        det_b = [{"box": [200, 100, 300, 220], "confidence": 0.35, "class": "weapon"}]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "mined"
            recorder = HardNegativeCropRecorder(
                out_dir=out_dir,
                label="phone",
                min_gap=5,
                dup_iou=0.75,
                max_crops=3,
                max_mb=1.0,
            )
            self.assertEqual(len(recorder.record_frame(frame, 0, det_a)), 1)   # Saved #1
            self.assertEqual(len(recorder.record_frame(frame, 2, det_a)), 0)   # Duplicate skipped (< min_gap)
            self.assertEqual(recorder.skipped_duplicates, 1)
            self.assertEqual(len(recorder.record_frame(frame, 2, det_b)), 1)   # Non-overlapping box saved #2
            self.assertEqual(len(recorder.record_frame(frame, 6, det_a)), 1)   # Beyond min_gap saved #3 -> hits max_crops
            self.assertTrue(recorder.budget_exhausted)
            self.assertEqual(len(recorder.record_frame(frame, 20, det_b)), 0)  # Stopped by max_crops

            manifest_path, bundle_path = recorder.finalize()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["saved_count"], 3)
            self.assertTrue(bundle_path.is_file())
            with zipfile.ZipFile(bundle_path, "r") as zf:
                names = zf.namelist()
                self.assertIn("manifest.json", names)
                self.assertEqual(sum(1 for n in names if n.endswith(".jpg")), 3)


if __name__ == "__main__":
    unittest.main()
