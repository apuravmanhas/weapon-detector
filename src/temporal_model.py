"""
Strictly Causal Temporal Action Anticipation models (Causal 1D-TCN and Causal GRU)
plus Lead-Time Exponential Anticipation Loss.

Mathematical guarantee: output at timestep t depends ONLY on inputs at <= t (zero lookahead).
Designed for <1 ms GPU inference per active weapon track and <15 min training on Colab T4.
"""

from collections import deque
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.features import FEATURE_DIM
from src.states import AssociatedPair, IntentAssessment, ThreatState


class CausalConv1d(nn.Module):
    """1D convolution strictly padded on the left only so future frames are never observed."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        if self.left_pad > 0:
            x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)


class CausalChannelNorm(nn.Module):
    """Per-timestep LayerNorm across channels only (strictly causal: zero temporal pooling)."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> normalize across C at each timestep t independently
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class CausalTCNBlock(nn.Module):
    """Residual dilated causal temporal block with strictly per-timestep channel normalization."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.15):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation)
        self.norm1 = CausalChannelNorm(out_ch)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size=kernel_size, dilation=dilation)
        self.norm2 = CausalChannelNorm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dropout(F.gelu(self.norm1(self.conv1(x))))
        h = self.dropout(F.gelu(self.norm2(self.conv2(h))))
        return F.gelu(h + self.skip(x))


class CausalWeaponIntentTCN(nn.Module):
    """
    Strictly causal dilated 1D-TCN operating on (B, T, F) biomechanical/kinematic sequences.

    Receptive field with default dilations (1, 2, 4) and kernel=3 is 29 frames (~0.5–1.0 s).
    Outputs:
      - state_logits: (B, T, 3) over [PASSIVE_PRESENT, DRAWING_READYING, AIMING_IMMINENT]
      - risk_logit:   (B, T)    continuous pre-fire risk logit (sigmoid -> [0, 1])
    """

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dim: int = 64,
        dilations: tuple[int, ...] = (1, 2, 4),
        kernel_size: int = 3,
        dropout: float = 0.15,
        num_states: int = 3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_states = num_states

        self.in_norm = nn.LayerNorm(input_dim)
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                CausalTCNBlock(hidden_dim, hidden_dim, kernel_size=kernel_size, dilation=d, dropout=dropout)
                for d in dilations
            ]
        )
        self.state_head = nn.Linear(hidden_dim, num_states)
        self.risk_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, F)
        h = self.in_proj(self.in_norm(x)).transpose(1, 2)  # (B, H, T)
        for blk in self.blocks:
            h = blk(h)
        h = h.transpose(1, 2)  # (B, T, H)
        state_logits = self.state_head(h)
        risk_logit = self.risk_head(h).squeeze(-1)
        return state_logits, risk_logit


class CausalWeaponIntentGRU(nn.Module):
    """Unidirectional (strictly causal) 2-layer GRU baseline for head-to-head ablation vs TCN."""

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.15,
        num_states: int = 3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_states = num_states

        self.in_norm = nn.LayerNorm(input_dim)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.state_head = nn.Linear(hidden_dim, num_states)
        self.risk_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, _ = self.gru(self.in_norm(x))
        return self.state_head(h), self.risk_head(h).squeeze(-1)


class LeadTimeAnticipationLoss(nn.Module):
    """
    Joint multi-task loss combining:
      1. Focal cross-entropy on readiness state [PASSIVE, DRAWING, AIMING]
      2. Exponentially time-weighted binary cross-entropy on pre-fire risk score,
         penalizing missed/late alarms more heavily as time-to-shot approaches 0.
    """

    def __init__(
        self,
        class_weights: tuple[float, float, float] = (1.0, 1.5, 2.2),
        focal_gamma: float = 1.5,
        risk_loss_weight: float = 1.2,
        exp_decay_sec: float = 0.75,
    ):
        super().__init__()
        self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        self.focal_gamma = focal_gamma
        self.risk_loss_weight = risk_loss_weight
        self.exp_decay_sec = max(1e-3, exp_decay_sec)

    def forward(
        self,
        state_logits: torch.Tensor,  # (B, T, 3)
        risk_logit: torch.Tensor,    # (B, T)
        state_targets: torch.Tensor, # (B, T) int64 in {0, 1, 2}, -100 ignored
        risk_targets: torch.Tensor,  # (B, T) float32 in [0, 1]
        time_to_shot_s: torch.Tensor | None = None,  # (B, T) seconds until t_first_shot (<0 or large if passive)
        valid_mask: torch.Tensor | None = None,      # (B, T) bool
    ) -> torch.Tensor:
        bsz, seq_len, n_cls = state_logits.shape
        if valid_mask is None:
            valid_mask = state_targets != -100

        # 1. Focal Cross-Entropy over valid timesteps
        flat_logits = state_logits.reshape(-1, n_cls)
        flat_targets = state_targets.reshape(-1)
        ce = F.cross_entropy(
            flat_logits,
            flat_targets,
            weight=self.class_weights.to(flat_logits.device),
            ignore_index=-100,
            reduction="none",
        ).reshape(bsz, seq_len)

        pt = torch.exp(-ce)
        focal_ce = ((1.0 - pt) ** self.focal_gamma) * ce

        # 2. Lead-time exponential weighting on risk BCE
        bce = F.binary_cross_entropy_with_logits(risk_logit, risk_targets, reduction="none")
        if time_to_shot_s is not None:
            # Frames within [0, 2.0s] before first shot get boosted exponentially as t -> t_shot
            pre_shot_zone = (time_to_shot_s >= 0.0) & (time_to_shot_s <= 2.0)
            lead_weight = torch.where(
                pre_shot_zone,
                1.0 + 2.5 * torch.exp(-time_to_shot_s / self.exp_decay_sec),
                torch.ones_like(time_to_shot_s),
            )
            bce = bce * lead_weight

        mask_f = valid_mask.float()
        denom = mask_f.sum().clamp_min(1.0)
        total = ((focal_ce + self.risk_loss_weight * bce) * mask_f).sum() / denom
        return total


def build_temporal_model(
    arch: Literal["tcn", "gru"] = "tcn",
    input_dim: int = FEATURE_DIM,
    hidden_dim: int = 64,
    dropout: float = 0.15,
) -> nn.Module:
    if arch == "tcn":
        return CausalWeaponIntentTCN(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    if arch == "gru":
        return CausalWeaponIntentGRU(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    raise ValueError(f"Unsupported temporal arch '{arch}'. Use 'tcn' or 'gru'.")


class CausalTemporalInferenceEngine:
    """
    Streaming inference wrapper around a trained Causal TCN/GRU checkpoint.
    Maintains a fixed sliding window of past feature vectors per pair_key (<1 ms per step).
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | None = None,
        window_len: int = 32,
        aim_threshold: float = 0.60,
        draw_threshold: float = 0.40,
    ):
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Temporal model checkpoint not found: {checkpoint_path}")

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.window_len = window_len
        self.aim_threshold = aim_threshold
        self.draw_threshold = draw_threshold

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        arch = str(ckpt.get("arch", "tcn"))
        input_dim = int(ckpt.get("input_dim", FEATURE_DIM))
        hidden_dim = int(ckpt.get("hidden_dim", 64))

        self.model = build_temporal_model(arch=arch, input_dim=input_dim, hidden_dim=hidden_dim, dropout=0.0)
        self.model.load_state_dict(ckpt["state_dict"], strict=True)
        self.model.to(self.device).eval()

        self._windows: dict[str, deque[np.ndarray]] = {}

    def reset(self) -> None:
        self._windows.clear()

    def update(
        self,
        pair: AssociatedPair,
        feature_vec: np.ndarray,
        feature_dict: dict[str, float],
        timestamp_s: float,
    ) -> IntentAssessment:
        del timestamp_s
        if pair.person is None:
            return IntentAssessment(
                pair_key=pair.pair_key,
                weapon_track_id=pair.weapon.track_id,
                person_id=None,
                state=ThreatState.PASSIVE_PRESENT,
                risk_score=0.10,
                state_probs={
                    ThreatState.NO_WEAPON.label: 0.0,
                    ThreatState.PASSIVE_PRESENT.label: 0.94,
                    ThreatState.DRAWING_READYING.label: 0.04,
                    ThreatState.AIMING_IMMINENT.label: 0.02,
                },
                reasons=["unowned_static_or_table"],
                weapon_box=list(pair.weapon.box),
                person_box=None,
                features=feature_dict,
            )

        win = self._windows.setdefault(pair.pair_key, deque(maxlen=self.window_len))
        win.append(feature_vec.astype(np.float32))
        seq = np.stack(list(win), axis=0)[None, :, :]  # (1, T, F)

        with torch.no_grad():
            x = torch.from_numpy(seq).to(self.device)
            state_logits, risk_logit = self.model(x)
            probs = torch.softmax(state_logits[0, -1], dim=-1).cpu().numpy()
            risk = float(torch.sigmoid(risk_logit[0, -1]).item())

        p_passive, p_draw, p_aim = float(probs[0]), float(probs[1]), float(probs[2])
        if risk >= self.aim_threshold or p_aim >= 0.55:
            state = ThreatState.AIMING_IMMINENT
            reason = f"causal_tcn_aim(risk={risk:.2f},p_aim={p_aim:.2f})"
        elif risk >= self.draw_threshold or p_draw >= 0.45:
            state = ThreatState.DRAWING_READYING
            reason = f"causal_tcn_draw(risk={risk:.2f},p_draw={p_draw:.2f})"
        else:
            state = ThreatState.PASSIVE_PRESENT
            reason = f"causal_tcn_passive(risk={risk:.2f})"

        return IntentAssessment(
            pair_key=pair.pair_key,
            weapon_track_id=pair.weapon.track_id,
            person_id=pair.person.person_id,
            state=state,
            risk_score=round(risk, 4),
            state_probs={
                ThreatState.NO_WEAPON.label: 0.0,
                ThreatState.PASSIVE_PRESENT.label: round(p_passive, 4),
                ThreatState.DRAWING_READYING.label: round(p_draw, 4),
                ThreatState.AIMING_IMMINENT.label: round(p_aim, 4),
            },
            reasons=[reason],
            weapon_box=list(pair.weapon.box),
            person_box=list(pair.person.box),
            features=feature_dict,
        )
