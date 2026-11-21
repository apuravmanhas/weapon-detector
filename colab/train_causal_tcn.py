"""
Colab / Local Trainer for Causal 1D-TCN and Causal GRU Pre-Firing Anticipation.

Strictly enforces clip-level train/val separation (zero frame-level data leakage).
Trains in <15 minutes on a free Google Colab T4 GPU on compact `.npz` sequences exported
by `tools/extract_sequences.py`.
"""

import argparse
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features import FEATURE_DIM
from src.temporal_model import LeadTimeAnticipationLoss, build_temporal_model


class SequenceDataset(Dataset):
    """Loads `.npz` sequences with optional random start cropping and Gaussian keypoint jitter."""

    def __init__(self, files: list[Path], max_len: int = 120, augment: bool = False):
        self.samples: list[dict[str, np.ndarray]] = []
        self.max_len = max_len
        self.augment = augment
        for f in files:
            d = np.load(f, allow_pickle=False)
            self.samples.append(
                {
                    "features": d["features"].astype(np.float32),
                    "state_targets": d["state_targets"].astype(np.int64),
                    "risk_targets": d["risk_targets"].astype(np.float32),
                    "time_to_shot_s": d["time_to_shot_s"].astype(np.float32),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.samples[idx]
        feats = s["features"].copy()
        st = s["state_targets"].copy()
        rk = s["risk_targets"].copy()
        tts = s["time_to_shot_s"].copy()

        t_len = len(feats)
        if t_len > self.max_len:
            start = random.randint(0, t_len - self.max_len) if self.augment else 0
            feats = feats[start : start + self.max_len]
            st = st[start : start + self.max_len]
            rk = rk[start : start + self.max_len]
            tts = tts[start : start + self.max_len]

        if self.augment:
            # Mild sensor/pose jitter noise on kinematic features only (preserve binary has_owner col 0)
            noise = np.random.normal(0.0, 0.015, size=feats.shape).astype(np.float32)
            noise[:, 0] = 0.0
            feats = feats + noise

        return {
            "features": torch.from_numpy(feats),
            "state_targets": torch.from_numpy(st),
            "risk_targets": torch.from_numpy(rk),
            "time_to_shot_s": torch.from_numpy(tts),
        }


def collate_padded(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_t = max(item["features"].shape[0] for item in batch)
    f_dim = batch[0]["features"].shape[1]
    bsz = len(batch)

    feats = torch.zeros((bsz, max_t, f_dim), dtype=torch.float32)
    st = torch.full((bsz, max_t), -100, dtype=torch.int64)
    rk = torch.zeros((bsz, max_t), dtype=torch.float32)
    tts = torch.full((bsz, max_t), 99.0, dtype=torch.float32)
    mask = torch.zeros((bsz, max_t), dtype=torch.bool)

    for i, item in enumerate(batch):
        t = item["features"].shape[0]
        feats[i, :t] = item["features"]
        st[i, :t] = item["state_targets"]
        rk[i, :t] = item["risk_targets"]
        tts[i, :t] = item["time_to_shot_s"]
        mask[i, :t] = True

    return {
        "features": feats,
        "state_targets": st,
        "risk_targets": rk,
        "time_to_shot_s": tts,
        "valid_mask": mask,
    }


def split_by_clip(seq_files: list[Path], val_ratio: float = 0.2, seed: int = 42) -> tuple[list[Path], list[Path]]:
    """Respect explicit `split` metadata if present, else split strictly by unique `clip_id`."""
    explicit_train: list[Path] = []
    explicit_val: list[Path] = []
    by_clip: dict[str, list[Path]] = {}

    for p in seq_files:
        d = np.load(p, allow_pickle=False)
        sp = str(d["split"].item()) if "split" in d else ""
        cid = str(d["clip_id"].item()) if "clip_id" in d else p.stem.split("__")[0]
        if sp == "train":
            explicit_train.append(p)
        elif sp in ("val", "eval"):
            explicit_val.append(p)
        else:
            by_clip.setdefault(cid, []).append(p)

    if explicit_train and explicit_val:
        return explicit_train, explicit_val

    all_by_clip: dict[str, list[Path]] = {}
    for p in seq_files:
        d = np.load(p, allow_pickle=False)
        cid = str(d["clip_id"].item()) if "clip_id" in d else p.stem.split("__")[0]
        all_by_clip.setdefault(cid, []).append(p)

    clip_ids = sorted(all_by_clip.keys())
    rng = random.Random(seed)
    rng.shuffle(clip_ids)
    n_val = max(1, int(round(len(clip_ids) * val_ratio))) if len(clip_ids) > 1 else 0
    val_clips = set(clip_ids[:n_val])

    train_files = [p for cid, ps in all_by_clip.items() if cid not in val_clips for p in ps]
    val_files = [p for cid, ps in all_by_clip.items() if cid in val_clips for p in ps]
    return train_files, val_files


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Causal TCN/GRU on extracted kinematic sequences")
    p.add_argument("--seq-dir", type=str, required=True, help="Directory of .npz sequence files")
    p.add_argument("--arch", type=str, choices=("tcn", "gru"), default="tcn")
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=45)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="weights/causal_intent_tcn.pt")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    seq_files = sorted(Path(args.seq_dir).glob("*.npz"))
    if not seq_files:
        raise FileNotFoundError(f"No .npz sequence files found in: {args.seq_dir}")

    train_files, val_files = split_by_clip(seq_files, val_ratio=args.val_ratio, seed=args.seed)
    if not train_files:
        raise RuntimeError("Need at least 1 training sequence after clip-level split.")

    train_ds = SequenceDataset(train_files, augment=True)
    val_ds = SequenceDataset(val_files, augment=False) if val_files else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_padded)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_padded) if val_ds else None

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = build_temporal_model(arch=args.arch, input_dim=FEATURE_DIM, hidden_dim=args.hidden_dim).to(device)
    criterion = LeadTimeAnticipationLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_val_loss = float("inf")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[Train] arch={args.arch} hidden={args.hidden_dim} device={device} | "
        f"train_seqs={len(train_files)} val_seqs={len(val_files)}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            st_logits, rk_logit = model(batch["features"].to(device))
            loss = criterion(
                st_logits,
                rk_logit,
                batch["state_targets"].to(device),
                batch["risk_targets"].to(device),
                batch["time_to_shot_s"].to(device),
                batch["valid_mask"].to(device),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            tr_loss += float(loss.item())
        tr_loss /= max(1, len(train_loader))
        scheduler.step()

        val_loss = tr_loss
        val_acc = 0.0
        if val_loader is not None:
            model.eval()
            v_sum, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    st_logits, rk_logit = model(batch["features"].to(device))
                    targets = batch["state_targets"].to(device)
                    mask = batch["valid_mask"].to(device)
                    loss = criterion(
                        st_logits,
                        rk_logit,
                        targets,
                        batch["risk_targets"].to(device),
                        batch["time_to_shot_s"].to(device),
                        mask,
                    )
                    v_sum += float(loss.item())
                    preds = st_logits.argmax(dim=-1)
                    correct += int(((preds == targets) & mask).sum().item())
                    total += int(mask.sum().item())
            val_loss = v_sum / max(1, len(val_loader))
            val_acc = correct / max(1, total)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "arch": args.arch,
                    "input_dim": FEATURE_DIM,
                    "hidden_dim": args.hidden_dim,
                    "best_val_loss": round(best_val_loss, 5),
                    "state_dict": model.state_dict(),
                },
                out_path,
            )

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"  Epoch {epoch:03d}/{args.epochs} | train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc*100:.1f}%")

    print(f"[Saved] Best checkpoint ({best_val_loss:.4f}) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
