# weapon-detector

Real-time automated weapon detection pipeline for CCTV and surveillance video streams, powered by fine-tuned Ultralytics YOLO11 models.

![CCTV Detection Preview](docs/cctv_surveillance_preview.jpg)

---

## Overview

Surveillance weapon detection requires high recall on small, partially occluded objects while suppressing false alarms from everyday handheld items (phones, umbrellas, wallets, power tools).

This repository ships two pre-trained single-class (`weapon`) model variants:
1. **YOLO11x (Production / Default):** 56.87M parameters, trained at `800×800` on 37,641 surveillance frames including hard-negative background samples.
2. **YOLO11m (Baseline):** 20.05M parameters, trained at `640×640` for lower-compute environments.

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
- **Geometric & Temporal False-Alarm Filtering:** Rejects degenerate full-frame boxes (`>75%` frame area), suppresses heavily nested single-class sub-boxes (`>=85%` overlap), and supports optional multi-frame IoU debouncing (`--min-frames`).
- **Windows Camera Resilience:** Probes `DirectShow` (`CAP_DSHOW`), `MSMF`, and default OpenCV capture backends in sequence.
- **Git LFS Pointer Guard:** Detects un-pulled LFS stubs and exits with an actionable `git lfs pull` error instead of crashing inside `torch.load`.

---

## Repository Structure

```text
weapon-detector/
├── .gitattributes                         # Git LFS tracking for production weights
├── .gitignore                             # Excludes secrets, caches, and media outputs
├── LICENSE                                # MIT License
├── README.md
├── detect.py                              # CLI interface (image, video, RTSP, webcam)
├── requirements.txt                       # Python dependencies
├── assets/
│   └── sample.jpg                         # Unannotated surveillance frame for testing
├── docs/
│   ├── cctv_surveillance_preview.jpg
│   ├── confusion_matrix.png
│   ├── model_comparison.png
│   ├── pr_curve.png
│   └── training_results.png
├── src/
│   ├── __init__.py
│   └── detector.py                        # WeaponDetector & TemporalPersistenceFilter
├── tests/
│   └── test_detector.py                   # Unit & smoke tests
└── weights/
    ├── yolo11x_weapon_production.pt       # Production YOLO11x weights (~109 MiB, Git LFS)
    └── best.pt                            # Baseline YOLO11m weights (~38.6 MiB)
```

---

## Installation

### Prerequisites
- Python 3.10+
- [Git LFS](https://git-lfs.com/) (required to download `weights/yolo11x_weapon_production.pt`)

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

### Video / RTSP CCTV Stream with Temporal Debouncing
Require detections to persist across at least 3 of the last 5 frames before displaying alerts:
```bash
python detect.py --source input_cctv.mp4 --min-frames 3 --window 5 --save output.mp4 --no-show
```

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

## CLI Options

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--source` | `str` | `0` | Camera index (`0`), stream URL, video path, or image path. |
| `--weights` | `str` | Auto-detect | `.pt` weights path (prefers YOLO11x if pulled, else YOLO11m). |
| `--conf` | `float` | `0.35` | Minimum confidence threshold in `(0.0, 1.0]`. |
| `--iou` | `float` | `0.45` | NMS IoU threshold to merge overlapping detections. |
| `--imgsz` | `int` | Auto | Spatial resolution (defaults to checkpoint's trained size: `800` or `640`). |
| `--no-fp16` | `flag` | `False` | Force FP32 inference even on CUDA devices. |
| `--min-frames` | `int` | `1` | Minimum matching frames in `--window` required to confirm stream threats. |
| `--window` | `int` | `5` | Sliding window size (frames) used with `--min-frames`. |
| `--show-conf` | `flag` | `False` | Render confidence scores on detection labels. |
| `--show-fps` | `flag` | `False` | Overlay live throughput (FPS) on video frames. |
| `--save` | `str` | `None` | Destination path to write annotated image or `.mp4` video. |
| `--no-show` | `flag` | `False` | Headless mode without GUI preview window. |
| `--device` | `str` | Auto | Computation device (`cuda:0`, `cpu`, or auto-detect). |

---

## Security & Privacy Safeguards

- **Local Execution Only:** All inference runs on host hardware with zero outbound network telemetry.
- **Credential-Safe Stream Logging:** Embedded user credentials in `rtsp://user:pass@host` URLs are masked (`***:***@host`) before stdout logging.
- **Sanitized Model Checkpoints:** Production weights contain unified class names (`weapon`) with local training paths and timestamps stripped.
- **Repository Hygiene:** `.gitignore` excludes `.env`, keyfiles, test artifacts, virtual environments, and output videos/exports.

---

## License

Licensed under the [MIT License](LICENSE).
