import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from detect import sanitize_source_label
from src.detector import (
    REPO_ROOT,
    TemporalPersistenceFilter,
    WeaponDetector,
    filter_detections,
    is_lfs_pointer,
    resolve_weights_path,
)


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
            [{"box": [10, 0, 90, 60], "confidence": 0.85, "class": "weapon"}],
            show_conf=True,
        )
        self.assertEqual(len(out_dets), 1)
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


if __name__ == "__main__":
    unittest.main()
