"""
Zero-dependency acoustic impulse peak detector for WAV tracks exported from draw-to-fire videos.

Helps locate candidate `t_first_shot_ms` timestamps quickly so human verification against
muzzle flash / slide recoil takes seconds per clip.
"""

import argparse
from pathlib import Path
import wave

import numpy as np


def detect_shot_peaks_wav(
    wav_path: str | Path,
    window_ms: float = 10.0,
    min_separation_ms: float = 200.0,
    sigma_threshold: float = 6.0,
    top_k: int = 8,
) -> list[dict[str, float]]:
    """Detect sharp high-energy acoustic impulse peaks in an uncompressed PCM WAV file."""
    p = Path(wav_path)
    if not p.is_file():
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    with wave.open(str(p), "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sampwidth == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes.")

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    win_samples = max(1, int(round(sample_rate * (window_ms / 1000.0))))
    n_wins = len(samples) // win_samples
    if n_wins < 3:
        return []

    blocks = samples[: n_wins * win_samples].reshape(n_wins, win_samples)
    rms = np.sqrt(np.mean(blocks * blocks, axis=1) + 1e-12)

    # First-order positive onset flux emphasizes sharp impulsive gunshots over steady speech/wind.
    # Measure robust scale on raw signed diff BEFORE clamping negatives to zero so MAD never collapses.
    raw_diff = np.diff(rms, prepend=rms[0])
    onset = np.maximum(0.0, raw_diff)
    diff_med = float(np.median(raw_diff))
    diff_mad = float(np.median(np.abs(raw_diff - diff_med)))
    scale = max(1.4826 * diff_mad, float(np.std(raw_diff)) * 0.25, 1e-6)
    robust_z = (onset - max(0.0, diff_med)) / scale

    rms_med = float(np.median(rms))
    rms_mad = float(np.median(np.abs(rms - rms_med)))
    rms_scale = max(1.4826 * rms_mad, float(np.std(rms)) * 0.25, 1e-6)
    rms_z = (rms - rms_med) / rms_scale
    min_rms_floor = max(rms_med * 3.0, float(np.max(rms)) * 0.25)

    cand_idxs = np.where((robust_z >= sigma_threshold) & (rms_z >= sigma_threshold) & (rms >= min_rms_floor))[0]
    if len(cand_idxs) == 0:
        cand_idxs = np.argsort(robust_z)[-min(top_k, len(robust_z)) :]

    # Greedy non-maximum suppression by min_separation_ms
    ordered = sorted(cand_idxs.tolist(), key=lambda i: float(robust_z[i]), reverse=True)
    min_sep_wins = max(1, int(round(min_separation_ms / window_ms)))
    selected: list[int] = []
    for idx in ordered:
        if any(abs(idx - prev) < min_sep_wins for prev in selected):
            continue
        selected.append(idx)
        if len(selected) >= top_k:
            break

    selected.sort()
    return [
        {
            "timestamp_ms": round(idx * window_ms, 1),
            "rms": round(float(rms[idx]), 4),
            "onset_zscore": round(float(robust_z[idx]), 2),
        }
        for idx in selected
    ]


def main() -> int:
    p = argparse.ArgumentParser(description="Find candidate gunshot acoustic peaks in a WAV file")
    p.add_argument("wav", type=str, help="Path to uncompressed PCM WAV file")
    p.add_argument("--sigma", type=float, default=6.0, help="Robust onset Z-score threshold")
    p.add_argument("--top-k", type=int, default=8, help="Max peaks to report")
    args = p.parse_args()

    peaks = detect_shot_peaks_wav(args.wav, sigma_threshold=args.sigma, top_k=args.top_k)
    print(f"[Peaks] {args.wav} ({len(peaks)} candidate(s)):")
    for pk in peaks:
        print(f"  t = {pk['timestamp_ms']:8.1f} ms | onset_z = {pk['onset_zscore']:6.2f} | rms = {pk['rms']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
