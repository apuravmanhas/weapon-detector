"""
Export YOLO11 weapon & pose PyTorch checkpoints into lightweight FP16 ONNX models under weights/.

Keeps source `.pt` directories clean (moves exported `.onnx` artifacts cleanly into `weights/`).
"""

import argparse
from pathlib import Path
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pose import ensure_torch_cuda_dlls


def _resolve_pose_source(weights_dir: Path) -> str | Path:
    for cand in (weights_dir / "yolo11n-pose.pt", REPO_ROOT / "yolo11n-pose.pt"):
        if cand.is_file():
            return cand
    return "yolo11n-pose.pt"


def export_model(src_pt: str | Path, dst_onnx: Path, imgsz: int, device: str = "0") -> Path:
    from ultralytics import YOLO

    if isinstance(src_pt, Path) and not src_pt.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {src_pt}")

    print(f"[Export] {src_pt} -> {dst_onnx.name} (imgsz={imgsz}, FP16 ONNX, opset=17)...")
    model = YOLO(str(src_pt))
    exported = Path(
        model.export(
            format="onnx",
            imgsz=imgsz,
            half=True,
            simplify=True,
            opset=17,
            device=device,
        )
    )
    if not exported.is_file():
        raise RuntimeError(f"Ultralytics export did not produce expected file: {exported}")

    dst_onnx.parent.mkdir(parents=True, exist_ok=True)
    if dst_onnx.exists():
        dst_onnx.unlink()
    if exported.resolve() != dst_onnx.resolve():
        shutil.move(str(exported), str(dst_onnx))
    size_mb = dst_onnx.stat().st_size / 1e6
    print(f"[Saved] {dst_onnx} ({size_mb:.2f} MB)")
    return dst_onnx


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export YOLO11 weapon & pose models to FP16 ONNX")
    p.add_argument("--device", type=str, default="0", help="CUDA device for FP16 ONNX export ('0' or 'cpu')")
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(REPO_ROOT / "weights"),
        help="Destination folder for exported .onnx files",
    )
    p.add_argument(
        "--include-yolo11x",
        action="store_true",
        help="Also export YOLO11x 800px FP16 ONNX (PyTorch FP16 cuDNN is faster for YOLO11x)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ensure_torch_cuda_dlls()

    out_dir = Path(args.out_dir)
    weights_dir = REPO_ROOT / "weights"

    specs: list[tuple[str | Path, Path, int]] = [
        (weights_dir / "best.pt", out_dir / "yolo11m_weapon_baseline_640.onnx", 640),
        (_resolve_pose_source(weights_dir), out_dir / "yolo11n_pose_512.onnx", 512),
    ]
    if args.include_yolo11x:
        specs.insert(
            0,
            (weights_dir / "yolo11x_weapon_production.pt", out_dir / "yolo11x_weapon_production_800.onnx", 800),
        )

    total_mb = 0.0
    for src, dst, imgsz in specs:
        out_path = export_model(src, dst, imgsz=imgsz, device=args.device)
        total_mb += out_path.stat().st_size / 1e6

    print(f"\n[Done] Exported {len(specs)} FP16 ONNX models ({total_mb:.2f} MB total) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
