# weapon-detector

Real-time automated weapon detection pipeline for CCTV and surveillance video streams, powered by fine-tuned Ultralytics YOLO11 models.

![CCTV Detection Preview](docs/cctv_surveillance_preview.jpg)

---

## Overview

Surveillance weapon detection requires high recall on small, partially occluded objects while suppressing false alarms from everyday handheld items (phones, umbrellas, wallets, power tools).

This repository ships a complete surveillance detection + pre-firing action anticipation stack:
1. **Stage-1 — YOLO11x (Production / Default):** 56.87M parameters, trained at `800×800` on 37,641 surveillance frames (high recall on blurry CCTV, fast motion, and oblique angles).
2. **Stage-2 — ConvNeXt-Tiny Crop Verifiers (Git LFS):**
   - `weights/crop_verifier_convnext_tiny_v2.pt` (**Default, 6-Class**): Trained across `weapon`, `phone`, `power_tool`, `umbrella`, `table_bg`, and `other_bg`, achieving **99.98% Weapon-vs-All ROC-AUC**, **0.00% `power_tool` FP@0.52**, and **0.29% `umbrella` FP@0.52** on the held-out 1,517-crop validation split.
   - `weights/crop_verifier_convnext_tiny.pt` (**5-Class Legacy**): Trained across `weapon`, `phone`, `umbrella`, `table_bg`, and `other_bg` (**99.82% weapon ROC-AUC**).
3. **Stage-1 Light — YOLO11m (Baseline):** 20.05M parameters, trained at `640×640` for lower-compute environments and FP16 ONNX edge deployment.
4. **Pre-Firing Intent & Action Anticipation Engine:** Couples weapon detections + Stage-2 verification with weapon-gated 17-keypoint COCO pose (`YOLO11n-pose`), scale-invariant owner association with pose-anchored coasting, 30-D FPS-invariant hand-weapon kinematics, an auditable Biomechanical Hysteresis FSM, and a Causal 1D-TCN / GRU training pipeline (`PASSIVE_PRESENT` → `DRAWING_READYING` → `AIMING_IMMINENT`).

---

## Model Zoo & Benchmarks

Validation metrics (held-out validation split) and end-to-end inference speed measured on an NVIDIA GeForce RTX 3050 Laptop GPU (`720p` input frames):

| Model Variant | Weights File | Parameters | GFLOPs | Native `imgsz` | Precision | Recall | mAP@50 | mAP@50-95 | End-to-End GPU Speed |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLO11x (Production)** | `weights/yolo11x_weapon_production.pt` | **56.87M** | 195.8 | `800` | 88.6% | **83.2%** | **89.5%** | **63.3%** | ~51.0 ms (~19.6 FPS, FP16) |
| **YOLO11m (Baseline)** | `weights/best.pt` | 20.05M | 68.3 | `640` | **94.3%** | 66.0% | 75.9% | 47.7% | ~33.7 ms (~29.7 FPS, FP32) |

![Model Benchmark Comparison](docs/model_comparison.png)

### Key Engineering Improvements
- **Native Resolution Auto-Matching:** Automatically selects `800×800` for YOLO11x and `640×640` for YOLO11m based on checkpoint metadata.
- **FP16 GPU Acceleration:** Enables half precision (`quantize=16` / `half=True`) automatically on CUDA devices, increasing YOLO11x throughput at `800×800` from ~15.8 FPS (63.2 ms) to ~19.6 FPS (51.0 ms).
- **Geometric & Temporal False-Alarm Filtering:** Rejects degenerate full-frame boxes (`>75%` frame area), suppresses heavily nested single-class sub-boxes (`>=85%` overlap), supports velocity-predicted tracking (`WeaponTrackManager`) with motionless low-confidence confounder quarantine (`--static-suppress`), and supports Stage-2 native-resolution crop verification (`--verifier`).
- **Async Video I/O (`--async-io`):** Overlaps video frame decoding/encoding on bounded worker threads for higher local throughput.
- **Windows Camera Resilience:** Probes `DirectShow` (`CAP_DSHOW`), `MSMF`, and default OpenCV capture backends in sequence.
- **Dense & Piled Weapon Modes (`--clahe`, `--dense`, `--tiled`):** Lifts dark-on-dark metal contrast via LAB-space CLAHE, relaxes NMS overlap thresholds for parallel weapons, and runs optional `2×2` overlapping SAHI tiling for dense weapon piles.
- **Git LFS Pointer Guard:** Detects un-pulled LFS stubs in both Stage-1 and Stage-2 weights and exits with an actionable `git lfs pull` error instead of crashing inside `torch.load`.

---

## Repository Structure

```text
weapon-detector/
├── .gitattributes                                # Git LFS tracking for production weights
├── .gitignore                                    # Excludes secrets, caches, raw clips, and media outputs
├── LICENSE                                       # MIT License
├── README.md
├── detect.py                                     # Object-detection CLI (image, video, RTSP, webcam)
├── requirements.txt                              # Python dependencies
├── assets/
│   └── sample.jpg                                # Unannotated surveillance frame for testing
├── benchmark/
│   ├── __init__.py
│   ├── manifest.json                             # Curated 37-clip anticipation benchmark manifest
│   ├── manifest_template.json                    # Template manifest schema for custom clip buckets
│   ├── metrics.py                                # Net Effective Lead Time & FAR evaluation metrics
│   └── schema.py                                 # Typed ClipAnnotation dataclasses & validator
├── docs/
│   ├── cctv_surveillance_preview.jpg
│   ├── confusion_matrix.png
│   ├── model_comparison.png
│   ├── pr_curve.png
│   └── training_results.png
├── colab/
│   ├── Train_Weapon_V2_Colab.ipynb               # Turnkey Colab training (Stage-1 + 5-class verifier)
│   ├── Train_Stage2_Verifier_6Class_Colab.ipynb  # Zero-leakage 6-class Stage-2 verifier training (OpenImages V7)
│   └── train_causal_tcn.py                       # Strictly causal 1D-TCN / GRU trainer with anticipation loss
├── src/
│   ├── __init__.py
│   ├── detector.py                               # WeaponDetector & TemporalPersistenceFilter
│   ├── tracker.py                                # WeaponTrackManager (velocity + static quarantine)
│   ├── verifier.py                               # Stage-2 CropVerifier (for detect.py)
│   ├── states.py                                 # Readiness ThreatState enum & tracking dataclasses
│   ├── perception.py                             # Unified Stage-1 (.pt/.onnx/.engine) + 6-class Stage-2 TTL cache
│   ├── pose.py                                   # Weapon-gated YOLO11-pose estimation + PersonTrackManager
│   ├── association.py                            # Scale-invariant wrist-to-weapon matching & pose coasting
│   ├── features.py                               # 30-D scale- & FPS-invariant biomechanical + kinematic extractor
│   ├── intent_fsm.py                             # Auditable Biomechanical Hysteresis State Machine
│   ├── temporal_model.py                         # CausalWeaponIntentTCN, CausalWeaponIntentGRU & LeadTimeLoss
│   ├── pipeline.py                               # End-to-end PreFireIntentPipeline with CUDA-synced telemetry
│   └── visualizer.py                             # Color-coded readiness HUD + bearer upper-body skeleton overlay
├── tests/
│   ├── test_association.py
│   ├── test_benchmark_metrics.py
│   ├── test_detector.py
│   ├── test_features.py
│   ├── test_intent_fsm.py
│   ├── test_perception_cache.py
│   └── test_temporal_model.py
├── tools/
│   ├── __init__.py
│   ├── detect_audio_shots.py                     # Gunshot audio impulse candidate finder for WAV tracks
│   ├── eval_anticipation.py                      # 37-clip Net Effective Lead Time & latency evaluator
│   ├── export_onnx.py                            # Lightweight FP16 ONNX exporter for CUDA edge inference
│   ├── extract_sequences.py                      # Compact 30-D .npz feature sequence extractor for Colab
│   ├── mine_hard_negatives.py                    # Storage-capped local false-positive crop miner
│   └── run_intent.py                             # Real-time pre-firing anticipation CLI + profiler
└── weights/
    ├── yolo11x_weapon_production.pt              # Production YOLO11x weights (~109 MiB, Git LFS)
    ├── crop_verifier_convnext_tiny_v2.pt         # 6-class ConvNeXt-Tiny v2 verifier (~111 MB, Git LFS)
    ├── crop_verifier_convnext_tiny.pt            # 5-class ConvNeXt-Tiny v1 verifier (~111 MB, Git LFS)
    └── best.pt                                   # Baseline YOLO11m weights (~38.6 MiB)
```

---

## Installation

### Prerequisites
- Python 3.10+
- [Git LFS](https://git-lfs.com/) (required to download `weights/yolo11x_weapon_production.pt` and the `ConvNeXt-Tiny` verifier checkpoints)

### 1. Clone and pull Git LFS weights
```bash
git clone https://github.com/apuravmanhas/weapon-detector.git
cd weapon-detector
git lfs pull
```

### 2. Create a virtual environment and install dependencies
```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

---

## Usage

### Single Image
```bash
python detect.py --source assets/sample.jpg --show-conf --save result.jpg
```

### Live Webcam
```bash
python detect.py --source 0 --show-fps
```

### Video / RTSP CCTV Stream with Smart Tracking & Static Confounder Quarantine
Require sustained multi-frame tracks while suppressing motionless low-confidence background confounders (e.g., table edges):
```bash
python detect.py --source input_cctv.mp4 --min-frames 3 --window 5 --static-suppress --async-io --show-track-id --save output.mp4 --no-show
```

### Two-Stage Production Inference (Recommended)
Pair high-recall `YOLO11x` with the included Stage-2 `ConvNeXt-Tiny` verifier (`git lfs pull`) and Smart Tracker to block oblique smartphone and static table-edge false positives:
```bash
python detect.py --source 0 --verifier weights/crop_verifier_convnext_tiny.pt --static-suppress --min-frames 2 --window 5 --show-conf --show-fps
```

### Dense / Piled Matte-Black Weapons (`--clahe`, `--dense`, `--tiled`)
```bash
# Fast real-time contrast lift + dense NMS (ideal for parallel stacked rifles)
python detect.py --source input_cctv.mp4 --verifier weights/crop_verifier_convnext_tiny.pt --clahe --dense --static-suppress --save output.mp4 --no-show

# Deep-scan 2x2 overlapping SAHI tiling (ideal for forensic photos / dense piles)
python detect.py --source assets/sample.jpg --verifier weights/crop_verifier_convnext_tiny.pt --clahe --dense --tiled --show-conf --save result.jpg
```

### Stage-2 Training & Optional Hard-Negative Mining (Google Colab)
1. *(Optional)* Harvest local false-positive crops (e.g., oblique phones) into a disk-capped ZIP (`<=25 MB`), or skip straight to Step 2 where Colab harvests public COCO 2017 distractors automatically:
```bash
python tools/mine_hard_negatives.py --source 0 --label phone --conf 0.20 --max-crops 300 --max-mb 25
```
2. Open [colab/Train_Weapon_V2_Colab.ipynb](colab/Train_Weapon_V2_Colab.ipynb) in Google Colab (`T4 GPU`, zero laptop storage used) to train or customize your Stage-2 verifier and Stage-1 models with per-epoch Google Drive auto-resume.

### Selecting Weights or Running on CPU
```bash
# Lightweight YOLO11m baseline
python detect.py --weights weights/best.pt --source 0

# Explicit CPU FP32 run
python detect.py --source assets/sample.jpg --device cpu --no-show
```

### Running Tests
```bash
python -m unittest discover -s tests -v
```

---

## Pre-Firing Weapon Intent & Action Anticipation

> **Defensive Scope:** Designed exclusively for human-in-the-loop public-safety and security operator early warning. Not intended for autonomous kinetic action or automated targeting.

Static weapon detectors cannot distinguish a firearm resting on a counter or slung passively from one raised into a firing stance half a second before trigger pull. The anticipation pipeline (`tools/run_intent.py`, `src/pipeline.py`) extends Stage-1/Stage-2 detection with causal human pose biomechanics and latency-deducted evaluation:

### Anticipation Architecture
1. **Stage 1 (`src/perception.py`):** `YOLO11x` (`FP16`, `800×800`, `conf=0.20`) with unified PyTorch (`.pt`), ONNX Runtime CUDA FP16 (`.onnx`), and TensorRT (`.engine`) support.
2. **Stage 2 (`src/perception.py`):** Auto-prefers 6-class `crop_verifier_convnext_tiny_v2.pt` (falling back to 5-class v1) on lower-confidence detections (`< 0.72`) with tiered thresholds (`verify_thresh=0.52`) and track-level TTL caching (`verifier_cache_ttl=4` frames).
3. **Weapon-Gated Pose & Coasting (`src/pose.py`, `src/association.py`):** Triggers 17-keypoint `YOLO11n-pose` (`512×512` ONNX Runtime CUDA FP16 or PyTorch) only when a weapon track is active or cooling down, associates each firearm via torso-normalized wrist-to-box edge distance, and applies pose-anchored coasting (`max_coast_frames=12`) across brief draw-occlusion or motion-blur dropouts. Unowned firearms stay locked to `PASSIVE_PRESENT`.
4. **30-D Kinematics & Causal Intent Engine (`src/features.py`, `src/intent_fsm.py`, `src/temporal_model.py`):** Strictly causal, torso- and FPS-normalized biomechanics (arm extension, shoulder pitch, head/eye cheek-weld sight alignment, upward velocity/acceleration, post-raise stabilization variance, and an optional 2-keypoint `grip -> muzzle` orientation hook) feeding an auditable Biomechanical Hysteresis FSM (`NO_WEAPON`, `PASSIVE_PRESENT`, `DRAWING_READYING`, `AIMING_IMMINENT`) or a trained strictly causal `1D-TCN` / `GRU`.

### Core Metric: Net Effective Lead Time
Video-timestamp lead time overstates warning margins if processing delay is ignored. The evaluator (`benchmark/metrics.py`) deducts real measured per-frame inference latency before scoring hits:
```text
delta_t_net = t_first_shot_ms - (t_alert_frame_ms + inference_latency_ms)
```
An alert counts as a valid pre-shot hit only when `delta_t_net > 0` across all three required buckets (`benchmark/schema.py`):
- `draw_to_fire`: Timestamped draw-to-shot events (`t_draw_start_ms`, `t_aim_start_ms`, `t_first_shot_ms`).
- `passive_carry`: Real firearms visibly present (slung, holstered, table/counter, walking) without firing intent.
- `non_weapon_mimic`: Rapid arm-raising distractors without a firearm (smartphones, cordless drills, umbrellas, water bottles, TV remotes).

### 37-Clip End-to-End Speed–Accuracy Pareto Benchmark (RTX 3050 Laptop GPU)
Evaluated end-to-end across **37 curated clips (12,507 frames)** (`13 draw_to_fire`, `8 passive_carry`, `16 non_weapon_mimic`) with `4 GB VRAM` and `onnxruntime-gpu==1.22.0` (`CUDA 12` + `cuDNN 9`):

| Configuration | Pre-Shot Alert (`DRAWING+`) | Median Net Lead (`DRAW+`) | Imminent Alert (`AIMING`) | Median Net Lead (`AIM`) | Passive FAR | Non-Weapon FAR (`/hr`) | Latency Mean / P50 / P95 / P99 (`FPS`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **v1 Baseline** (`YOLO11x` PyTorch + Pose 640 PyTorch) | 76.9% | 1090.4 ms | 53.8% | 328.8 ms | 2.9% | 288.95 | 82.6 / 83.74 / 124.85 / 143.92 ms (12.1) |
| **v1.5 Coasting + Verifier Cache** (`YOLO11x` PyTorch + Pose PyTorch) | **100.0%** | 890.8 ms | 69.2% | 354.1 ms | 2.4% | 177.81 | 81.6 / 83.19 / 116.85 / 138.95 ms (12.3) |
| **Hybrid v1** (`YOLO11x` PyTorch + Verifier v1 + Pose ONNX 512) | **100.0%** | 895.4 ms | **76.9%** | 347.8 ms | **2.0%** | 266.72 | 71.32 / 69.1 / 99.73 / 119.67 ms (14.0) |
| **Hybrid + Verifier v2** (`YOLO11x` PyTorch + 6-Class Verifier v2 + Pose ONNX 512) | **100.0%** | 905.5 ms | **76.9%** | 365.9 ms | 2.1% | 222.27 | 69.11 / 69.36 / 92.44 / 103.06 ms (14.5) |
| **Full-ONNX Edge** (`YOLO11m` 640 ONNX + Verifier v1 + Pose ONNX 512) | 53.8% | 618.9 ms | 38.5% | 475.3 ms | 2.4% | **133.36** | **36.9 / 36.71 / 64.59 / 73.75 ms (27.1)** |

- **Isolated 30-Frame CUDA-Synced Micro-Benchmark:** Compute-bound `YOLO11x @ 800px` runs faster in PyTorch cuDNN FP16 (`66.45 ms/frame`) than ORT CUDA FP16 (`74.32 ms/frame`, `0.89x`), whereas kernel-launch-bound `YOLO11m @ 640px` (`33.65 -> 23.25 ms/frame`, **1.45x**) and `YOLO11n-pose @ 512px` (`29.20 -> 19.40 ms/frame`, **1.51x**) accelerate cleanly under ORT CUDA FP16 (`46.22 MB` total ONNX weight footprint).
- **Stage-2 Verifier v2 Impact:** Adding OpenImages V7 `power_tool` hard negatives eliminated `m03_cordless_drill_1` false alarms (`47 -> 0` frames) and cut `m09_cordless_drill_2` (`49 -> 21` frames) and `m02_concert_phone_raise` (`9 -> 7` frames), keeping **13/16** mimic clips at **0** false-aim frames while improving median net lead times to **905.5 ms** (`DRAWING+`) and **365.9 ms** (`AIMING`) at **92.44 ms P95 (`14.5 FPS`)**.

### Single-Clip Active-Frame Stage Breakdown (PyTorch Eager FP16 Baseline)
Measured with `torch.cuda.synchronize()` across 250 active-weapon frames (`854×480` video, `YOLO11x @ 800px` + `ConvNeXt-Tiny` verifier + `YOLO11n-pose @ 512px` + Biomechanical FSM):

| Stage | Mean | P50 | P95 | P99 |
| :--- | :---: | :---: | :---: | :---: |
| Stage 1 Weapon Detector (`YOLO11x`, 800px) | 54.21 ms | 54.41 ms | 62.83 ms | 68.34 ms |
| Stage 2 Crop Verifier (`ConvNeXt-Tiny`, uncached) | 17.76 ms | 17.81 ms | 21.35 ms | 23.55 ms |
| Pose Backbone (`YOLO11n-pose`, 512px, gated) | 33.87 ms | 33.88 ms | 39.44 ms | 43.04 ms |
| Weapon Tracker + Owner Assoc + 30-D Features + Intent FSM | 1.16 ms | 1.10 ms | — | — |
| **Total End-to-End (PyTorch Eager Active Frame)** | **107.05 ms** | **107.79 ms** | **121.55 ms** | **134.15 ms** |

### Pre-Firing Anticipation Usage
```bash
# Optional: export lightweight YOLO11m (640) + YOLO11n-pose (512) FP16 ONNX models
python tools/export_onnx.py --device 0 --out-dir weights

# Real-time hybrid anticipation inference on video / RTSP / webcam
python tools/run_intent.py \
  --source path/to/video.mp4 \
  --pose-weights weights/yolo11n_pose_512.onnx \
  --pose-imgsz 512 \
  --max-dim 1280 \
  --save outputs/intent_demo.mp4 \
  --no-show

# Automated 37-clip Net Effective Lead Time benchmark
python tools/eval_anticipation.py \
  --manifest benchmark/manifest.json \
  --pose-weights weights/yolo11n_pose_512.onnx \
  --pose-imgsz 512 \
  --out-dir outputs/eval_hybrid_v2

# Optional: extract compact 30-D .npz sequences and train Causal 1D-TCN / GRU
python tools/extract_sequences.py --manifest benchmark/manifest.json --out-dir data/sequences
python colab/train_causal_tcn.py --seq-dir data/sequences --arch tcn --epochs 45 --out weights/causal_intent_tcn.pt
```
- **Google Colab 6-Class Verifier Training:** Open [colab/Train_Stage2_Verifier_6Class_Colab.ipynb](colab/Train_Stage2_Verifier_6Class_Colab.ipynb) on a free Colab `T4 GPU` to retrain `crop_verifier_convnext_tiny_v2.pt` from public OpenImages V7 hard negatives with image-grouped splits and atomic Drive checkpoints.
- **Audio-Assisted Shot Timestamp Helper:** `python tools/detect_audio_shots.py path/to/clip.wav`

### Honest Physical & Engineering Limits
No vision-only model can beat physical observability or frame-rate limits:
- **Lookalike Dark Cylinders Shouldered Like Rifles:** A closed black umbrella intentionally shouldered in a firing stance (`m08_umbrella_rifle_aim`, `95` false aim frames in the Hybrid v2 benchmark) closely mimics a matte barrel in tight crops.
- **Occluded / Back-Turned Draws:** If the torso completely blocks the draw and firearm until trigger pull, optical lead time approaches zero.
- **Ultra-Fast Hip Draws (< 350 ms):** When camera frame intervals plus persistence and inference consume most of the motion window, net lead time shrinks sharply.

All benchmark numbers in this repository come from recorded video frames and CUDA-synchronized latencies.

---

## CLI Options (`detect.py`)

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--source` | `str` | `0` | Camera index (`0`), stream URL, video path, or image path. |
| `--weights` | `str` | Auto-detect | `.pt` weights path (prefers YOLO11x if pulled, else YOLO11m). |
| `--conf` | `float` | `0.35` (`0.20` w/ `--dense`) | Minimum confidence threshold in `(0.0, 1.0]`. |
| `--iou` | `float` | `0.45` (`0.65` w/ `--dense`) | NMS IoU threshold to merge overlapping detections. |
| `--clahe` | `flag` | `False` | Apply LAB-luminance CLAHE contrast lift before Stage-1 inference. |
| `--dense` | `flag` | `False` | Preset for closely packed weapons (`conf=0.20`, `iou=0.65`, `nested=0.95`). |
| `--tiled` | `flag` | `False` | Enable full-frame + `2×2` overlapping SAHI tiled inference. |
| `--imgsz` | `int` | Auto | Spatial resolution (defaults to checkpoint's trained size: `800` or `640`). |
| `--no-fp16` | `flag` | `False` | Force FP32 inference even on CUDA devices. |
| `--min-frames` | `int` | `1` | Minimum matching frames in `--window` required to confirm stream threats. |
| `--window` | `int` | `5` | Sliding window size (frames) used with `--min-frames`. |
| `--tracker` | `str` | `smart` | Tracker backend (`smart` = velocity + EMA + rules, `legacy` = raw IoU deque). |
| `--static-suppress` | `flag` | `False` | Quarantine persistent motionless low-confidence confounders. |
| `--static-max-conf` | `float` | `0.58` | Maximum EMA confidence eligible for static-background quarantine. |
| `--verifier` | `str` | `None` | Path to Stage-2 `CropVerifier` `.pt` weights (`ConvNeXt` / `MobileNetV3`). |
| `--verify-thresh` | `float` | `0.50` | Minimum Stage-2 weapon probability to retain uncertain detections. |
| `--verify-below` | `float` | `0.85` | Run Stage-2 verifier only on Stage-1 detections below this confidence. |
| `--show-conf` | `flag` | `False` | Render confidence scores on detection labels. |
| `--show-track-id` | `flag` | `False` | Render persistent track IDs (`#id`) when using `--tracker smart`. |
| `--show-fps` | `flag` | `False` | Overlay live throughput (FPS) on video frames. |
| `--save` | `str` | `None` | Destination path to write annotated image or `.mp4` video. |
| `--no-show` | `flag` | `False` | Headless mode without GUI preview window. |
| `--async-io` | `flag` | `False` | Use background worker threads for video file decoding/encoding. |
| `--device` | `str` | Auto | Computation device (`cuda:0`, `cpu`, or auto-detect). |

---

## Security & Privacy Safeguards

- **Local Execution Only:** All inference runs on host hardware with zero outbound network telemetry.
- **Credential-Safe Stream Logging:** Embedded user credentials in `rtsp://user:pass@host` URLs are masked (`***:***@host`) before stdout/exception logging across both `detect.py` and `tools/run_intent.py`.
- **Sanitized Model Checkpoints:** Production weights (`weights_only=True` compatible) strip local training paths and timestamps.
- **Repository Hygiene:** `.gitignore` excludes `.env`, keyfiles, test artifacts, virtual environments, raw video/sequence caches, and output videos/exports.

---

## License

Licensed under the [MIT License](LICENSE).
