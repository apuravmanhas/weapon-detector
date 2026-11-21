from pathlib import Path
import tempfile
import unittest
import wave

import numpy as np
import torch

from src.features import FEATURE_DIM
from src.temporal_model import LeadTimeAnticipationLoss, build_temporal_model
from tools.detect_audio_shots import detect_shot_peaks_wav


class TestCausalTemporalModels(unittest.TestCase):
    def test_strict_causality_tcn_and_gru(self) -> None:
        """Perturbing future frames t > split_idx MUST produce bit-identical outputs at t <= split_idx."""
        torch.manual_seed(0)
        seq_len = 24
        split_idx = 11

        for arch in ("tcn", "gru"):
            model = build_temporal_model(arch=arch, input_dim=FEATURE_DIM, hidden_dim=32, dropout=0.0)
            model.eval()

            x_clean = torch.randn(2, seq_len, FEATURE_DIM)
            x_future_corrupted = x_clean.clone()
            x_future_corrupted[:, split_idx + 1 :, :] = 999.0 * torch.randn(2, seq_len - split_idx - 1, FEATURE_DIM)

            with torch.no_grad():
                st_a, rk_a = model(x_clean)
                st_b, rk_b = model(x_future_corrupted)

            self.assertTrue(
                torch.equal(st_a[:, : split_idx + 1, :], st_b[:, : split_idx + 1, :]),
                f"{arch.upper()} leaked future information into state_logits!",
            )
            self.assertTrue(
                torch.equal(rk_a[:, : split_idx + 1], rk_b[:, : split_idx + 1]),
                f"{arch.upper()} leaked future information into risk_logit!",
            )

    def test_lead_time_loss_weights_pre_shot_higher_and_backprops(self) -> None:
        torch.manual_seed(1)
        model = build_temporal_model(arch="tcn", input_dim=FEATURE_DIM, hidden_dim=32)
        criterion = LeadTimeAnticipationLoss()

        x = torch.randn(2, 16, FEATURE_DIM)
        st_logits, rk_logit = model(x)
        st_targets = torch.randint(0, 3, (2, 16))
        rk_targets = torch.full((2, 16), 0.9)

        # Same predictions, but closer time-to-shot must yield higher loss weight
        loss_near = criterion(
            st_logits,
            rk_logit,
            st_targets,
            rk_targets,
            time_to_shot_s=torch.full((2, 16), 0.10),
        )
        loss_far = criterion(
            st_logits,
            rk_logit,
            st_targets,
            rk_targets,
            time_to_shot_s=torch.full((2, 16), 10.0),
        )
        self.assertGreater(float(loss_near.item()), float(loss_far.item()))
        loss_near.backward()
        grad_norm = sum(float(p.grad.abs().sum().item()) for p in model.parameters() if p.grad is not None)
        self.assertGreater(grad_norm, 0.0)

    def test_audio_shot_impulse_detector_on_synthetic_wav(self) -> None:
        sr = 16000
        dur_s = 2.5
        rng = np.random.default_rng(42)
        audio = rng.normal(0.0, 0.008, size=int(sr * dur_s)).astype(np.float32)
        # Inject sharp impulsive gunshot peak at t = 1250 ms
        shot_idx = int(1.250 * sr)
        audio[shot_idx : shot_idx + 160] += 0.85 * np.exp(-np.linspace(0.0, 5.0, 160))
        pcm16 = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)

        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "shot_test.wav"
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(pcm16.tobytes())

            peaks = detect_shot_peaks_wav(wav_path, window_ms=10.0, sigma_threshold=6.0)
            self.assertTrue(peaks, "Expected synthetic gunshot impulse peak to be detected.")
            self.assertAlmostEqual(peaks[0]["timestamp_ms"], 1250.0, delta=20.0)


if __name__ == "__main__":
    unittest.main()
