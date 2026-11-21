from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

import src.perception as perception_mod
import src.pose as pose_mod
from src.perception import Stage1WeaponDetector, Stage2CropVerifier, WeaponTrackManager
from src.pose import PoseEstimator


def _mock_verifier(score_map: list[tuple[float, str]]) -> tuple[Stage2CropVerifier, dict[str, int]]:
    ver = Stage2CropVerifier.__new__(Stage2CropVerifier)
    ver.pad_ratio = 0.30
    calls = {"count": 0, "crops": 0}

    def _fake_score(crops: list[np.ndarray]) -> list[tuple[float, str]]:
        calls["count"] += 1
        calls["crops"] += len(crops)
        return score_map[: len(crops)]

    ver.score_crops = _fake_score  # type: ignore[method-assign]
    return ver, calls


class TestStage2TieredAndCache(unittest.TestCase):
    def test_tiered_confidence_thresholds(self) -> None:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        # High confidence >= 0.72 bypasses Stage-2 completely
        ver, calls = _mock_verifier([(0.05, "other_bg")])
        kept = ver.verify(frame, [{"box": [10, 10, 60, 40], "confidence": 0.76, "class": "weapon"}])
        self.assertEqual(len(kept), 1)
        self.assertEqual(calls["count"], 0)

        # Mid band [0.55, 0.72): requires >= 0.42
        ver_mid_ok, _ = _mock_verifier([(0.45, "weapon")])
        self.assertEqual(len(ver_mid_ok.verify(frame, [{"box": [10, 10, 60, 40], "confidence": 0.60}])), 1)
        ver_mid_no, _ = _mock_verifier([(0.39, "umbrella")])
        self.assertEqual(len(ver_mid_no.verify(frame, [{"box": [10, 10, 60, 40], "confidence": 0.60}])), 0)

        # Low-mid band [0.35, 0.55): requires >= 0.52
        ver_std_no, _ = _mock_verifier([(0.48, "other_bg")])
        self.assertEqual(len(ver_std_no.verify(frame, [{"box": [10, 10, 60, 40], "confidence": 0.44}])), 0)
        ver_std_ok, _ = _mock_verifier([(0.55, "weapon")])
        self.assertEqual(len(ver_std_ok.verify(frame, [{"box": [10, 10, 60, 40], "confidence": 0.44}])), 1)

        # Ultra-low band [0.20, 0.35): requires >= 0.62
        ver_low_no, _ = _mock_verifier([(0.58, "weapon")])
        self.assertEqual(len(ver_low_no.verify(frame, [{"box": [10, 10, 60, 40], "confidence": 0.25}])), 0)
        ver_low_ok, _ = _mock_verifier([(0.65, "weapon")])
        self.assertEqual(len(ver_low_ok.verify(frame, [{"box": [10, 10, 60, 40], "confidence": 0.25}])), 1)

    def test_track_verifier_cache_hit_and_ttl_expiry(self) -> None:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        tracker = WeaponTrackManager(min_hits=2, window_size=5, static_suppress=False)
        ver, calls = _mock_verifier([(0.88, "weapon")])
        det = {"box": [80, 80, 140, 120], "confidence": 0.48, "class": "weapon"}

        # Frame 0 & Frame 1: run Stage-2 fresh until track is confirmed (min_hits=2)
        v0 = ver.verify(frame, [det], cached_tracks=tracker.get_verifier_cache_candidates(max_ttl_frames=3))
        tracker.update(v0, frame=frame)
        v1 = ver.verify(frame, [det], cached_tracks=tracker.get_verifier_cache_candidates(max_ttl_frames=3))
        tracker.update(v1, frame=frame)
        self.assertEqual(calls["crops"], 2)

        # Frames 2 & 3: track is confirmed with frames_since_verified < 3 -> Stage-2 MUST hit cache (0 new crops!)
        v2 = ver.verify(frame, [det], cached_tracks=tracker.get_verifier_cache_candidates(max_ttl_frames=3))
        self.assertTrue(v2[0].get("verifier_cached"))
        tracker.update(v2, frame=frame)

        v3 = ver.verify(frame, [det], cached_tracks=tracker.get_verifier_cache_candidates(max_ttl_frames=3))
        self.assertTrue(v3[0].get("verifier_cached"))
        tracker.update(v3, frame=frame)
        v4 = ver.verify(frame, [det], cached_tracks=tracker.get_verifier_cache_candidates(max_ttl_frames=3))
        self.assertTrue(v4[0].get("verifier_cached"))
        tracker.update(v4, frame=frame)
        self.assertEqual(calls["crops"], 2)

        # Very low conf (< 0.35) MUST NOT bypass via cache even when IoU matches
        low_det = {"box": [80, 80, 140, 120], "confidence": 0.24, "class": "weapon"}
        v_low = ver.verify(frame, [low_det], cached_tracks=tracker.get_verifier_cache_candidates(max_ttl_frames=10))
        self.assertFalse(v_low[0].get("verifier_cached", False))
        self.assertEqual(calls["crops"], 3)

        # After 3 unverified updates (v2, v3, v4), TTL=3 expires -> cache empty -> forces fresh check
        cands = tracker.get_verifier_cache_candidates(max_ttl_frames=3)
        self.assertEqual(cands, [])

    def test_compiled_onnx_and_engine_backends_omit_quantize_and_half(self) -> None:
        recorded: list[dict] = []

        class _FakeYOLO:
            def __init__(self, weights_path: str, task: str | None = None):
                self.names = {0: "weapon"}
                self.args = {}
                recorded.append({"init_path": str(weights_path), "task": task})

            def predict(self, **kwargs):
                recorded.append({"predict_kwargs": dict(kwargs)})
                return []

        orig_det_yolo = perception_mod.YOLO
        orig_pose_yolo = pose_mod.YOLO
        try:
            perception_mod.YOLO = _FakeYOLO  # type: ignore[assignment]
            pose_mod.YOLO = _FakeYOLO  # type: ignore[assignment]

            frame = np.zeros((120, 160, 3), dtype=np.uint8)
            with tempfile.TemporaryDirectory() as tmp:
                onnx_det = Path(tmp) / "yolo11x_weapon_800.onnx"
                onnx_det.write_bytes(b"0" * 2048)
                det = Stage1WeaponDetector(weights_path=onnx_det, device="cuda:0", fp16=True)
                self.assertTrue(det.is_compiled)
                self.assertEqual(det.imgsz, 800)
                det.detect(frame)

                eng_pose = Path(tmp) / "yolo11n_pose_512.engine"
                eng_pose.write_bytes(b"0" * 2048)
                pose = PoseEstimator(weights=str(eng_pose), imgsz=512, device="cuda:0", fp16=True)
                self.assertTrue(pose.is_compiled)
                pose.predict(frame)

            self.assertEqual(recorded[0]["task"], "detect")
            self.assertNotIn("quantize", recorded[1]["predict_kwargs"])
            self.assertNotIn("half", recorded[1]["predict_kwargs"])
            self.assertEqual(recorded[2]["task"], "pose")
            self.assertNotIn("quantize", recorded[3]["predict_kwargs"])
            self.assertNotIn("half", recorded[3]["predict_kwargs"])
        finally:
            perception_mod.YOLO = orig_det_yolo  # type: ignore[assignment]
            pose_mod.YOLO = orig_pose_yolo  # type: ignore[assignment]

    def test_six_class_power_tool_checkpoint_loads_weights_only(self) -> None:
        classes = ["other_bg", "phone", "power_tool", "table_bg", "umbrella", "weapon"]
        model = perception_mod._build_classifier("mobilenet_v3_large", len(classes))
        with torch.no_grad():
            head = model.classifier[3]
            head.weight.zero_()
            head.bias.fill_(-4.0)
            head.bias[classes.index("power_tool")] = 6.0

        self.assertIn("power_tool", perception_mod.DISTRACTOR_CLASS_NAMES)
        self.assertIn("drill", perception_mod.DISTRACTOR_CLASS_NAMES)

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "crop_verifier_convnext_tiny_v2.pt"
            torch.save(
                {
                    "arch": "mobilenet_v3_large",
                    "classes": classes,
                    "weapon_classes": ["weapon"],
                    "img_size": 224,
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                    "state_dict": model.state_dict(),
                    "val_weapon_auc": 0.992,
                    "val_power_tool_fp_rate": 0.015,
                    "val_umbrella_fp_rate": 0.012,
                },
                ckpt_path,
            )

            ver = Stage2CropVerifier(weights_path=ckpt_path, device="cpu", fp16=False)
            self.assertEqual(ver.classes, classes)
            self.assertEqual(ver._weapon_indices, [5])

            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            kept = ver.verify(frame, [{"box": [10, 10, 80, 60], "confidence": 0.48}])
            self.assertEqual(kept, [])

    def test_run_intent_sanitize_source_label_masks_rtsp_credentials(self) -> None:
        from tools.run_intent import sanitize_source_label

        masked = sanitize_source_label("rtsp://admin:s3cr3t@10.0.0.5:554/stream")
        self.assertEqual(masked, "rtsp://***:***@10.0.0.5:554/stream")
        self.assertEqual(sanitize_source_label("0"), "0")
        self.assertEqual(sanitize_source_label("assets/sample.jpg"), "assets/sample.jpg")


if __name__ == "__main__":
    unittest.main()
