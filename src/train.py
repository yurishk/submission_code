"""
Unified Training Script for ASD Speech Prediction.

- SA: Recording-level MIL + CCCLoss (+ tiny MSE stabilization)
- RRB: Recording-level MIL + Class-weighted CORAL (+ auxiliary regression)

Usage:
    python train.py -c config.yaml --seed 42
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import yaml
import argparse
import random
import os
import math
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
from scipy import stats
import copy
from pathlib import Path
import datetime
import matplotlib.pyplot as plt
from typing import Optional

from feature_groups import FEATURE_GROUPS, validate_feature_groups

validate_feature_groups()


# ============================================================
# Random Seed
# ============================================================
def set_seed(seed=42):
    """Fix ALL random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# ============================================================
# Loss Functions
# ============================================================
class CCCLoss(nn.Module):
    """Concordance Correlation Coefficient Loss for SA regression."""
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x = x.view(-1)
        y = y.view(-1)
        if len(x) < 2:
            return torch.tensor(1.0, requires_grad=True, device=x.device)
        mx, my = torch.mean(x), torch.mean(y)
        sx2 = torch.var(x, unbiased=False)
        sy2 = torch.var(y, unbiased=False)
        sxy = torch.mean((x - mx) * (y - my))
        ccc = (2 * sxy) / (sx2 + sy2 + (mx - my)**2 + 1e-8)
        return 1.0 - ccc


class CORALLoss(nn.Module):
    """CORAL Loss for RRB ordinal classification."""
    def __init__(self, num_classes=9):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, labels):
        levels = torch.arange(self.num_classes - 1, device=logits.device)
        targets = (labels.unsqueeze(1) > levels.unsqueeze(0)).float()
        return nn.functional.binary_cross_entropy_with_logits(logits, targets)


class WeightedCORALLoss(nn.Module):
    """Class-weighted CORAL to handle severe class imbalance in RRB (0-8)."""
    def __init__(self, num_classes=9):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, labels, class_weights: Optional[torch.Tensor] = None,
                sample_weights: Optional[torch.Tensor] = None):
        levels = torch.arange(self.num_classes - 1, device=logits.device)
        targets = (labels.unsqueeze(1) > levels.unsqueeze(0)).float()
        loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        if class_weights is not None:
            w = class_weights[labels].unsqueeze(1)
            loss = loss * w
        if sample_weights is not None:
            loss = loss * sample_weights.unsqueeze(1)
        return loss.mean()


class FocalOrdinalLoss(nn.Module):
    """Focal-weighted CORAL: down-weight easy ordinal thresholds, focus on hard ones."""
    def __init__(self, num_classes=9, gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma

    def forward(self, logits, labels, class_weights: Optional[torch.Tensor] = None,
                sample_weights: Optional[torch.Tensor] = None):
        levels = torch.arange(self.num_classes - 1, device=logits.device)
        targets = (labels.unsqueeze(1) > levels.unsqueeze(0)).float()
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        focal = ((1 - pt) ** self.gamma) * bce
        if class_weights is not None:
            w = class_weights[labels].unsqueeze(1)
            focal = focal * w
        if sample_weights is not None:
            focal = focal * sample_weights.unsqueeze(1)
        return focal.mean()


def pairwise_ordinal_rank_loss(predictions, targets, temperature=1.0, min_gap=1.0):
    """Logistic pairwise ranking loss for distinct ordinal scores in a batch."""
    predictions = predictions.view(-1)
    targets = targets.float().view(-1)
    target_delta = targets[:, None] - targets[None, :]
    pred_delta = predictions[:, None] - predictions[None, :]
    mask = torch.triu(target_delta.abs() >= float(min_gap), diagonal=1)
    if not mask.any():
        return predictions.sum() * 0.0
    signed_margin = torch.sign(target_delta[mask]) * pred_delta[mask]
    return nn.functional.softplus(-signed_margin / max(float(temperature), 1e-6)).mean()


def ordinal_moment_loss(predictions, targets, score_range=8.0):
    """Match batch location and scale without rewarding unbounded variance."""
    predictions = predictions.float().view(-1)
    targets = targets.float().view(-1)
    scale = max(float(score_range), 1e-6)
    mean_error = (predictions.mean() - targets.mean()) / scale
    pred_std = torch.std(predictions, unbiased=False)
    target_std = torch.std(targets, unbiased=False)
    std_error = (pred_std - target_std) / scale
    return mean_error.square() + std_error.square()


class FeatureGroupDropout(nn.Module):
    """Randomly zero-out entire feature groups during training for robustness."""
    GROUPS = list(FEATURE_GROUPS.values())

    def __init__(self, p_group=0.15):
        super().__init__()
        self.p_group = p_group

    def forward(self, x):
        if not self.training or self.p_group <= 0:
            return x
        x = x.clone()
        for g in self.GROUPS:
            if torch.rand(1).item() < self.p_group:
                x[..., g] = 0.0
        return x


class OrderedCumulativeHead(nn.Module):
    """Cumulative-link ordinal head with strictly ordered learned thresholds."""

    def __init__(self, input_dim: int, num_classes: int, dropout: float):
        super().__init__()
        hidden_dim = max(1, input_dim // 2)
        self.features = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.severity = nn.Linear(hidden_dim, 1)
        self.threshold_start = nn.Parameter(torch.tensor(-2.0))
        initial_gap = 4.0 / max(num_classes - 2, 1)
        raw_gap = math.log(math.expm1(initial_gap))
        self.threshold_gaps = nn.Parameter(torch.full((num_classes - 2,), raw_gap))

    def thresholds(self) -> torch.Tensor:
        if self.threshold_gaps.numel() == 0:
            return self.threshold_start.view(1)
        gaps = nn.functional.softplus(self.threshold_gaps) + 1e-4
        return torch.cat([self.threshold_start.view(1), self.threshold_start + torch.cumsum(gaps, dim=0)])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        severity = self.severity(self.features(inputs))
        return severity - self.thresholds().unsqueeze(0)


# ============================================================
# Models (NEW): Hierarchical MIL (recording-level, uses all matrices jointly)
# ============================================================
class _SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, t = x.shape
        s = x.mean(dim=2)
        w = self.fc(s).view(b, c, 1)
        return x * w


class _TemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel_size - 1) // 2 * dilation
        self.norm1 = nn.GroupNorm(1, channels)
        self.act = nn.GELU()
        self.conv_dw = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=pad,
            dilation=dilation,
            groups=channels,
        )
        self.conv_pw = nn.Conv1d(channels, channels, kernel_size=1)
        self.se = _SEBlock(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv_dw(h)
        h = self.conv_pw(h)
        h = self.se(h)
        h = self.drop(h)
        return x + h


class _AttentiveStatsPool(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, 1, kernel_size=1),
        )

    def forward(self, x):
        a = torch.softmax(self.attn(x).squeeze(1), dim=1).unsqueeze(1)
        mu = torch.sum(a * x, dim=2)
        var = torch.sum(a * (x - mu.unsqueeze(2)) ** 2, dim=2).clamp_min(1e-8)
        std = torch.sqrt(var)
        return torch.cat([mu, std], dim=1)


class _MultiScaleTemporalEncoder(nn.Module):
    """Multi-scale temporal encoder for signal-derived acoustic matrices."""
    def __init__(self, input_dim: int = 49, emb_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.in_norm = nn.LayerNorm(input_dim)
        branch_dim = max(16, emb_dim // 4)
        specs = [(3, 1), (5, 1), (7, 2), (9, 3)]
        self.branches = nn.ModuleList()
        for kernel, dilation in specs:
            pad = ((kernel - 1) // 2) * dilation
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(input_dim, branch_dim, kernel_size=kernel, padding=pad, dilation=dilation),
                    nn.GroupNorm(1, branch_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        merged_dim = branch_dim * len(specs)
        self.proj = nn.Conv1d(merged_dim, emb_dim, kernel_size=1)
        self.blocks = nn.Sequential(
            _TemporalBlock(emb_dim, kernel_size=3, dilation=1, dropout=dropout),
            _TemporalBlock(emb_dim, kernel_size=3, dilation=2, dropout=dropout),
        )
        self.pool = _AttentiveStatsPool(emb_dim, dropout=dropout)
        self.out_dim = emb_dim * 2

    def forward(self, x):
        # x: (B, T, F)
        x = self.in_norm(x).transpose(1, 2)
        h = torch.cat([branch(x) for branch in self.branches], dim=1)
        h = self.proj(h)
        h = self.blocks(h)
        return self.pool(h)


class _FeatureGroupTemporalEncoder(nn.Module):
    """Encode canonical acoustic feature groups before fusing them by attention."""
    def __init__(self, emb_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.groups = list(FEATURE_GROUPS.items())
        group_dim = max(16, emb_dim // 4)
        group_out = group_dim * 2
        self.group_dim = group_dim
        self.group_out = group_out
        self.norms = nn.ModuleList([nn.LayerNorm(len(indices)) for _, indices in self.groups])
        self.stems = nn.ModuleList([
            nn.Conv1d(len(indices), group_dim, kernel_size=1) for _, indices in self.groups
        ])
        self.blocks = nn.ModuleList([
            nn.Sequential(
                _TemporalBlock(group_dim, kernel_size=3, dilation=1, dropout=dropout),
                _TemporalBlock(group_dim, kernel_size=3, dilation=2, dropout=dropout),
            )
            for _ in self.groups
        ])
        self.pools = nn.ModuleList([
            _AttentiveStatsPool(group_dim, dropout=dropout) for _ in self.groups
        ])
        self.group_score = nn.Sequential(
            nn.Linear(group_out, max(1, group_out // 2)),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(max(1, group_out // 2), 1),
        )
        self.out_norm = nn.LayerNorm(group_out)
        self.out_dim = group_out

    def forward(self, x):
        # x: (B, T, F)
        zs = []
        for idx, (_, indices) in enumerate(self.groups):
            g = x[..., indices]
            g = self.norms[idx](g).transpose(1, 2)
            h = self.stems[idx](g)
            h = self.blocks[idx](h)
            zs.append(self.pools[idx](h))
        z = torch.stack(zs, dim=1)  # (B, G, D)
        w = torch.softmax(self.group_score(z).squeeze(-1), dim=1)
        pooled = torch.sum(z * w.unsqueeze(-1), dim=1)
        return self.out_norm(pooled)


class _AcousticGroupInteractionEncoder(nn.Module):
    """Parameter-efficient temporal statistics with learned acoustic-group interactions."""

    def __init__(self, emb_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.groups = list(FEATURE_GROUPS.items())
        token_dim = max(24, emb_dim // 4)
        self.group_projections = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(len(indices) * 4),
                nn.Linear(len(indices) * 4, token_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for _, indices in self.groups
        ])
        self.group_embeddings = nn.Parameter(torch.zeros(1, len(self.groups), token_dim))
        nn.init.normal_(self.group_embeddings, std=0.02)
        for nhead in (4, 3, 2, 1):
            if token_dim % nhead == 0:
                break
        self.interaction = nn.MultiheadAttention(
            token_dim, nhead, dropout=dropout, batch_first=True
        )
        self.interaction_norm = nn.LayerNorm(token_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * 2, token_dim),
        )
        self.feed_forward_norm = nn.LayerNorm(token_dim)
        self.group_score = nn.Sequential(
            nn.Linear(token_dim, max(8, token_dim // 2)),
            nn.Tanh(),
            nn.Linear(max(8, token_dim // 2), 1),
        )
        self.global_projection = nn.Sequential(
            nn.LayerNorm(49 * 4),
            nn.Linear(49 * 4, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.Linear(token_dim * 2, token_dim),
            nn.GELU(),
            nn.LayerNorm(token_dim),
        )
        self.out_dim = token_dim

    @staticmethod
    def _temporal_statistics(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1)
        std = x.std(dim=1, unbiased=False)
        if x.size(1) > 1:
            mean_abs_delta = (x[:, 1:] - x[:, :-1]).abs().mean(dim=1)
        else:
            mean_abs_delta = torch.zeros_like(mean)
        positions = torch.linspace(-1.0, 1.0, x.size(1), device=x.device, dtype=x.dtype)
        denominator = positions.square().mean().clamp_min(1e-6)
        trend = (x * positions.view(1, -1, 1)).mean(dim=1) / denominator
        return torch.cat([mean, std, mean_abs_delta, trend], dim=-1)

    def forward(self, x):
        group_tokens = []
        for projection, (_, indices) in zip(self.group_projections, self.groups):
            group_tokens.append(projection(self._temporal_statistics(x[..., indices])))
        tokens = torch.stack(group_tokens, dim=1) + self.group_embeddings
        interacted, _ = self.interaction(tokens, tokens, tokens, need_weights=False)
        tokens = self.interaction_norm(tokens + interacted)
        tokens = self.feed_forward_norm(tokens + self.feed_forward(tokens))
        weights = torch.softmax(self.group_score(tokens).squeeze(-1), dim=1)
        grouped = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        global_summary = self.global_projection(self._temporal_statistics(x))
        return self.fusion(torch.cat([grouped, global_summary], dim=-1))


class _VocalizationDistributionEncoder(nn.Module):
    """Permutation-invariant encoder for sampled vocalization descriptor rows."""

    def __init__(self, input_dim: int = 49, emb_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        token_dim = max(32, emb_dim)
        self.input_norm = nn.LayerNorm(input_dim)
        self.token_encoder = nn.Sequential(
            nn.Linear(input_dim, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
        )
        self.token_score = nn.Sequential(
            nn.Linear(token_dim, max(16, token_dim // 2)),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(max(16, token_dim // 2), 1),
        )
        self.register_buffer("quantile_levels", torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90]))
        self.quantile_projection = nn.Sequential(
            nn.LayerNorm(input_dim * 5),
            nn.Linear(input_dim * 5, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.Linear(token_dim * 4, token_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(token_dim * 2),
        )
        self.out_dim = token_dim * 2

    def forward(self, x):
        tokens = self.token_encoder(self.input_norm(x))
        mean = tokens.mean(dim=1)
        std = tokens.std(dim=1, unbiased=False)
        weights = torch.softmax(self.token_score(tokens).squeeze(-1), dim=1)
        attended = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        quantiles = torch.quantile(x, self.quantile_levels.to(dtype=x.dtype), dim=1)
        quantiles = quantiles.permute(1, 0, 2).reshape(x.size(0), -1)
        distribution = self.quantile_projection(quantiles)
        return self.fusion(torch.cat([mean, std, attended, distribution], dim=-1))


class _TemporalDistributionResidualEncoder(nn.Module):
    """TCN backbone with a near-zero residual vocalization-distribution path."""

    def __init__(self, input_dim: int = 49, emb_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.temporal = _MatrixEncoder(
            input_dim=input_dim, emb_dim=emb_dim, dropout=dropout,
            encoder_variant="tcn_stats",
        )
        distribution_dim = max(64, emb_dim // 2)
        self.distribution = _VocalizationDistributionEncoder(
            input_dim=input_dim, emb_dim=distribution_dim, dropout=dropout
        )
        self.residual_projection = nn.Linear(self.distribution.out_dim, self.temporal.out_dim)
        nn.init.normal_(self.residual_projection.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.residual_projection.bias)
        self.residual_gate = nn.Sequential(
            nn.LayerNorm(self.temporal.out_dim),
            nn.Linear(self.temporal.out_dim, self.temporal.out_dim),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.residual_gate[1].weight)
        nn.init.constant_(self.residual_gate[1].bias, -3.0)
        self.out_dim = self.temporal.out_dim

    def forward(self, x):
        temporal = self.temporal(x)
        residual = self.residual_projection(self.distribution(x))
        return temporal + self.residual_gate(temporal) * residual


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        import math

        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])


class _MatrixEncoder(nn.Module):
    """Encode one 100x49 matrix into an embedding (strong inductive bias, small params)."""
    def __init__(self, input_dim=49, emb_dim=128, dropout=0.2, encoder_variant: str = "tcn_stats"):
        super().__init__()
        self.encoder_variant = str(encoder_variant).lower()
        if self.encoder_variant in ("ms_tcn_stats", "multiscale_tcn", "multi_scale_tcn"):
            self.ms = _MultiScaleTemporalEncoder(input_dim=input_dim, emb_dim=emb_dim, dropout=dropout)
            self.out_dim = self.ms.out_dim
        elif self.encoder_variant in ("group_tcn", "feature_group_tcn", "clinical_group_tcn"):
            self.group_enc = _FeatureGroupTemporalEncoder(emb_dim=emb_dim, dropout=dropout)
            self.out_dim = self.group_enc.out_dim
        elif self.encoder_variant in ("group_interaction", "acoustic_group_interaction", "agi"):
            self.group_interaction = _AcousticGroupInteractionEncoder(emb_dim=emb_dim, dropout=dropout)
            self.out_dim = self.group_interaction.out_dim
        elif self.encoder_variant in ("vocalization_set", "vocal_set", "distribution_set"):
            self.vocalization_set = _VocalizationDistributionEncoder(
                input_dim=input_dim, emb_dim=emb_dim, dropout=dropout
            )
            self.out_dim = self.vocalization_set.out_dim
        elif self.encoder_variant in ("temporal_distribution_residual", "tdr", "hybrid_tdr"):
            self.temporal_distribution = _TemporalDistributionResidualEncoder(
                input_dim=input_dim, emb_dim=emb_dim, dropout=dropout
            )
            self.out_dim = self.temporal_distribution.out_dim
        elif self.encoder_variant == "cnn_attn":
            self.conv = nn.Sequential(
                nn.Conv1d(input_dim, 128, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(128, emb_dim, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.attn = nn.Sequential(
                nn.Linear(emb_dim, emb_dim // 2),
                nn.Tanh(),
                nn.Linear(emb_dim // 2, 1),
            )
            self.out_dim = emb_dim
        elif self.encoder_variant in ("transformer_cls", "tfm_cls", "cls_transformer"):
            self.in_norm = nn.LayerNorm(input_dim)
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, emb_dim),
                nn.LayerNorm(emb_dim),
                nn.Dropout(dropout),
            )
            self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
            nn.init.normal_(self.cls_token, std=0.02)
            self.pos_encoder = _PositionalEncoding(emb_dim, max_len=101, dropout=dropout)

            for nhead in (8, 6, 4, 3, 2, 1):
                if emb_dim % nhead == 0:
                    self.nhead = nhead
                    break
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=emb_dim,
                nhead=self.nhead,
                dim_feedforward=emb_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation='gelu',
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.out_norm = nn.LayerNorm(emb_dim)
            self.out_dim = emb_dim
        else:
            self.in_norm = nn.LayerNorm(input_dim)
            self.stem = nn.Conv1d(input_dim, emb_dim, kernel_size=1)
            self.blocks = nn.Sequential(
                _TemporalBlock(emb_dim, kernel_size=3, dilation=1, dropout=dropout),
                _TemporalBlock(emb_dim, kernel_size=3, dilation=2, dropout=dropout),
                _TemporalBlock(emb_dim, kernel_size=3, dilation=4, dropout=dropout),
            )
            self.pool = _AttentiveStatsPool(emb_dim, dropout=dropout)
            self.out_dim = emb_dim * 2

    def forward(self, x):
        # x: (B*M, 100, 49)
        if self.encoder_variant in ("ms_tcn_stats", "multiscale_tcn", "multi_scale_tcn"):
            return self.ms(x)
        if self.encoder_variant in ("group_tcn", "feature_group_tcn", "clinical_group_tcn"):
            return self.group_enc(x)
        if self.encoder_variant in ("group_interaction", "acoustic_group_interaction", "agi"):
            return self.group_interaction(x)
        if self.encoder_variant in ("vocalization_set", "vocal_set", "distribution_set"):
            return self.vocalization_set(x)
        if self.encoder_variant in ("temporal_distribution_residual", "tdr", "hybrid_tdr"):
            return self.temporal_distribution(x)
        if self.encoder_variant == "cnn_attn":
            x = x.transpose(1, 2)
            h = self.conv(x).transpose(1, 2)
            a = torch.softmax(self.attn(h).squeeze(-1), dim=1)
            z = torch.sum(h * a.unsqueeze(-1), dim=1)
            return z
        if self.encoder_variant in ("transformer_cls", "tfm_cls", "cls_transformer"):
            x = self.in_norm(x)
            h = self.input_proj(x)
            cls = self.cls_token.expand(h.size(0), 1, h.size(2))
            h = torch.cat([cls, h], dim=1)
            h = self.pos_encoder(h)
            h = self.transformer(h)
            return self.out_norm(h[:, 0, :])

        x = self.in_norm(x).transpose(1, 2)
        h = self.stem(x)
        h = self.blocks(h)
        z = self.pool(h)
        return z


class _BagAttentionPool(nn.Module):
    """Attention pooling over matrices within a recording."""
    def __init__(self, emb_dim=128, dropout=0.2, pool_variant: str = "gated"):
        super().__init__()
        self.pool_variant = str(pool_variant).lower()
        if self.pool_variant in ("mean", "avg", "max"):
            return
        if self.pool_variant in ("bag_transformer", "transformer_pool"):
            self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
            nn.init.normal_(self.cls_token, std=0.02)
            self.pos_encoder = _PositionalEncoding(emb_dim, max_len=32, dropout=dropout)
            for nhead in (8, 6, 4, 3, 2, 1):
                if emb_dim % nhead == 0:
                    self.nhead = nhead
                    break
            layer = nn.TransformerEncoderLayer(
                d_model=emb_dim,
                nhead=self.nhead,
                dim_feedforward=emb_dim * 2,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=1)
            self.out_norm = nn.LayerNorm(emb_dim)
            return
        if self.pool_variant in ("pma", "set_transformer", "context_gated"):
            for nhead in (8, 6, 4, 3, 2, 1):
                if emb_dim % nhead == 0:
                    self.nhead = nhead
                    break
            self.instance_norm = nn.LayerNorm(emb_dim)
            if self.pool_variant in ("set_transformer", "context_gated"):
                layer = nn.TransformerEncoderLayer(
                    d_model=emb_dim,
                    nhead=self.nhead,
                    dim_feedforward=emb_dim * 2,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                )
                self.instance_transformer = nn.TransformerEncoder(layer, num_layers=1)
            if self.pool_variant in ("pma", "set_transformer"):
                self.seed_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
                nn.init.normal_(self.seed_token, std=0.02)
                self.cross_attn = nn.MultiheadAttention(
                    embed_dim=emb_dim,
                    num_heads=self.nhead,
                    dropout=dropout,
                    batch_first=True,
                )
                self.seed_norm = nn.LayerNorm(emb_dim)
                self.seed_ff = nn.Sequential(
                    nn.Linear(emb_dim, emb_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(emb_dim * 2, emb_dim),
                )
                self.out_norm = nn.LayerNorm(emb_dim)
            else:
                h = max(1, emb_dim // 2)
                self.v = nn.Linear(emb_dim, h)
                self.u = nn.Linear(emb_dim, h)
                self.w = nn.Linear(h, 1)
                self.drop = nn.Dropout(dropout)
            return
        if self.pool_variant == "plain":
            self.score = nn.Sequential(
                nn.Linear(emb_dim, emb_dim // 2),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(emb_dim // 2, 1),
            )
        else:
            h = max(1, emb_dim // 2)
            self.v = nn.Linear(emb_dim, h)
            self.u = nn.Linear(emb_dim, h)
            self.w = nn.Linear(h, 1)
            self.drop = nn.Dropout(dropout)

    def forward(self, z):
        # z: (B, M, D)
        if self.pool_variant in ("mean", "avg"):
            w = torch.full(z.shape[:2], 1.0 / z.size(1), device=z.device, dtype=z.dtype)
            return z.mean(dim=1), w
        if self.pool_variant == "max":
            idx = z.norm(dim=-1).argmax(dim=1)
            pooled = z[torch.arange(z.size(0), device=z.device), idx]
            w = torch.zeros(z.shape[:2], device=z.device, dtype=z.dtype)
            w[torch.arange(z.size(0), device=z.device), idx] = 1.0
            return pooled, w
        if self.pool_variant in ("bag_transformer", "transformer_pool"):
            cls = self.cls_token.expand(z.size(0), 1, z.size(2))
            h = torch.cat([cls, z], dim=1)
            h = self.pos_encoder(h)
            h = self.transformer(h)
            w = torch.full(z.shape[:2], 1.0 / z.size(1), device=z.device, dtype=z.dtype)
            return self.out_norm(h[:, 0, :]), w
        if self.pool_variant in ("pma", "set_transformer", "context_gated"):
            h = self.instance_norm(z)
            if self.pool_variant in ("set_transformer", "context_gated"):
                h = self.instance_transformer(h)
            if self.pool_variant in ("pma", "set_transformer"):
                seed = self.seed_token.expand(h.size(0), 1, h.size(2))
                attended, weights = self.cross_attn(
                    seed,
                    h,
                    h,
                    need_weights=True,
                    average_attn_weights=True,
                )
                pooled = self.seed_norm(seed + attended)
                pooled = self.out_norm(pooled + self.seed_ff(pooled))
                return pooled[:, 0, :], weights[:, 0, :]
            a = torch.tanh(self.v(h)) * torch.sigmoid(self.u(h))
            a = self.drop(a)
            weights = torch.softmax(self.w(a).squeeze(-1), dim=1)
            return torch.sum(h * weights.unsqueeze(-1), dim=1), weights
        if self.pool_variant == "plain":
            w = torch.softmax(self.score(z).squeeze(-1), dim=1)
            pooled = torch.sum(z * w.unsqueeze(-1), dim=1)
            return pooled, w

        a = torch.tanh(self.v(z)) * torch.sigmoid(self.u(z))
        a = self.drop(a)
        w = torch.softmax(self.w(a).squeeze(-1), dim=1)
        pooled = torch.sum(z * w.unsqueeze(-1), dim=1)
        return pooled, w


class _PaperCNNScalar(nn.Module):
    def __init__(self, input_dim: int = 49, seq_len: int = 100, dropout: float = 0.5):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3)

        self.fc1 = nn.Linear(256, 1024)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(1024, 512)
        self.drop2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)

        self.flatten_dim = self._infer_flatten_dim()
        self.out = nn.Linear(self.flatten_dim, 1)

    def _infer_flatten_dim(self):
        with torch.no_grad():
            x = torch.zeros(1, self.input_dim, self.seq_len)
            x = torch.relu(self.conv1(x))
            x = self.pool(x)
            x = torch.relu(self.conv2(x))
            x = x.transpose(1, 2)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = torch.relu(self.fc4(x))
            x = torch.flatten(x, 1)
            return int(x.shape[1])

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = x.transpose(1, 2)
        x = torch.relu(self.fc1(x))
        x = self.drop1(x)
        x = torch.relu(self.fc2(x))
        x = self.drop2(x)
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.flatten(x, 1)
        return self.out(x)


class _PaperCNNSameScalar(nn.Module):
    def __init__(self, input_dim: int = 49, seq_len: int = 100, dropout: float = 0.5):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.seq_after_pool = self.seq_len // 3
        self.flatten_dim = 256 * self.seq_after_pool

        self.fc_shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.out = nn.Linear(128, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.fc_shared(x)
        return self.out(x)


class SA_Model(nn.Module):
    """SA estimator with either a scalar or cumulative-threshold readout."""
    def __init__(self, input_dim=49, emb_dim=128, dropout=0.2,
                 encoder_variant: str = "tcn_stats", pool_variant: str = "gated",
                 sa_head_variant: str = "regression", sa_num_classes: int = 23,
                 sa_score_min: float = 0.0):
        super().__init__()
        self.encoder_variant = str(encoder_variant).lower()
        self.sa_head_variant = str(sa_head_variant).lower()
        self.sa_num_classes = int(sa_num_classes)
        self.sa_score_min = float(sa_score_min)
        self.is_ordinal = self.sa_head_variant in ("ordinal", "coral", "cumulative_link", "ordered")
        if self.sa_num_classes < 2:
            raise ValueError("sa_num_classes must be at least 2")
        if self.is_ordinal and self.encoder_variant in ("paper_avg", "paper_same_avg"):
            raise ValueError("The SA ordinal comparison requires a recording-level encoder")
        if self.encoder_variant == "paper_avg":
            self.paper = _PaperCNNScalar(input_dim=input_dim, seq_len=100, dropout=0.5)
            self.enc = None
            self.pool = None
            self.head = None
        elif self.encoder_variant == "paper_same_avg":
            self.paper = _PaperCNNSameScalar(input_dim=input_dim, seq_len=100, dropout=0.5)
            self.enc = None
            self.pool = None
            self.head = None
        else:
            self.paper = None
            self.enc = _MatrixEncoder(input_dim=input_dim, emb_dim=emb_dim, dropout=dropout, encoder_variant=encoder_variant)
            self.pool = _BagAttentionPool(emb_dim=self.enc.out_dim, dropout=dropout, pool_variant=pool_variant)
            if self.is_ordinal:
                self.head = OrderedCumulativeHead(self.enc.out_dim, self.sa_num_classes, dropout)
            else:
                self.head = nn.Sequential(
                    nn.LayerNorm(self.enc.out_dim),
                    nn.Linear(self.enc.out_dim, max(1, self.enc.out_dim // 2)),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(max(1, self.enc.out_dim // 2), 1),
                )

    def forward_logits_with_attention(self, X):
        if X.dim() == 3:
            X = X.unsqueeze(1)
        B, M, T, F = X.shape
        if self.paper is not None:
            y = self.paper(X.reshape(B * M, T, F)).reshape(B, M)
            weights = torch.full((B, M), 1.0 / M, device=X.device, dtype=X.dtype)
            return y.mean(dim=1, keepdim=True), weights

        z = self.enc(X.reshape(B * M, T, F)).reshape(B, M, -1)
        pooled, weights = self.pool(z)
        return self.head(pooled), weights

    def predict_continuous_from_logits(self, output):
        if not self.is_ordinal:
            return output
        return self.sa_score_min + torch.sigmoid(output).sum(dim=1, keepdim=True)

    def forward_logits(self, X):
        output, _ = self.forward_logits_with_attention(X)
        return output

    def forward_with_attention(self, X):
        output, weights = self.forward_logits_with_attention(X)
        return self.predict_continuous_from_logits(output), weights

    def forward(self, X):
        prediction, _ = self.forward_with_attention(X)
        return prediction


class RRB_Model(nn.Module):
    """RRB ordinal head on top of MIL encoder; also outputs expected value for auxiliary regression."""
    def __init__(self, input_dim=49, emb_dim=128, dropout=0.2, num_classes=9,
                 encoder_variant: str = "tcn_stats", pool_variant: str = "gated",
                 rrb_pool_variant: str = "", feat_group_dropout: float = 0.0,
                 rrb_head_variant: str = "independent"):
        super().__init__()
        self.num_classes = num_classes
        self.feat_drop = FeatureGroupDropout(p_group=feat_group_dropout) if feat_group_dropout > 0 else None
        self.enc = _MatrixEncoder(input_dim=input_dim, emb_dim=emb_dim, dropout=dropout, encoder_variant=encoder_variant)

        rpv = (rrb_pool_variant or pool_variant).lower()
        self._rpv = rpv
        if rpv == "topk":
            self.pool = None
            self.topk_k = 3
            head_in = self.enc.out_dim
        elif rpv == "meanmax":
            self.pool = None
            head_in = self.enc.out_dim * 2
        else:
            self.pool = _BagAttentionPool(emb_dim=self.enc.out_dim, dropout=dropout, pool_variant=pool_variant)
            head_in = self.enc.out_dim

        self.rrb_head_variant = str(rrb_head_variant).lower()
        if self.rrb_head_variant in ("cumulative_link", "ordered"):
            self.head = OrderedCumulativeHead(head_in, num_classes, dropout)
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(head_in),
                nn.Linear(head_in, max(1, head_in // 2)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(max(1, head_in // 2), num_classes - 1),
            )

    def _bag_pool(self, z):
        # z: (B, M, D)
        if self._rpv == "topk":
            norms = z.norm(dim=-1)  # (B, M)
            k = min(self.topk_k, z.size(1))
            _, idx = norms.topk(k, dim=1)
            idx_exp = idx.unsqueeze(-1).expand(-1, -1, z.size(-1))
            top = z.gather(1, idx_exp)
            return top.mean(dim=1)
        elif self._rpv == "meanmax":
            return torch.cat([z.mean(dim=1), z.max(dim=1).values], dim=-1)
        else:
            pooled, _ = self.pool(z)
            return pooled

    def forward_with_attention(self, X):
        if X.dim() == 3:
            X = X.unsqueeze(1)
        B, M, T, F = X.shape
        x = X.reshape(B * M, T, F)
        if self.feat_drop is not None:
            x = self.feat_drop(x)
        z = self.enc(x).reshape(B, M, -1)
        if self._rpv == "topk":
            norms = z.norm(dim=-1)
            k = min(self.topk_k, M)
            _, idx = norms.topk(k, dim=1)
            idx_exp = idx.unsqueeze(-1).expand(-1, -1, z.size(-1))
            pooled = z.gather(1, idx_exp).mean(dim=1)
            weights = torch.zeros(B, M, device=z.device, dtype=z.dtype)
            weights.scatter_(1, idx, 1.0 / k)
        elif self._rpv == "meanmax":
            pooled = torch.cat([z.mean(dim=1), z.max(dim=1).values], dim=-1)
            weights = torch.full((B, M), 1.0 / M, device=z.device, dtype=z.dtype)
        else:
            pooled, weights = self.pool(z)
        return self.head(pooled), weights

    def forward(self, X):
        logits, _ = self.forward_with_attention(X)
        return logits

    def probs_per_class(self, X):
        # Convert CORAL cumulative probs -> per-class probs (B, K)
        logits = self.forward(X)
        return self.probs_per_class_from_logits(logits)

    def probs_per_class_from_logits(self, logits):
        """Convert one CORAL forward pass into per-class probabilities."""
        cum = torch.sigmoid(logits)  # P(y > k)
        ones = torch.ones(cum.size(0), 1, device=cum.device)
        zeros = torch.zeros(cum.size(0), 1, device=cum.device)
        cum_ext = torch.cat([ones, cum, zeros], dim=1)
        probs = cum_ext[:, :-1] - cum_ext[:, 1:]
        return probs

    def predict_continuous(self, X):
        return self.predict_continuous_from_logits(self.forward(X))

    def predict_continuous_from_logits(self, logits):
        """Compute the expected ordinal score from an existing forward pass."""
        probs = self.probs_per_class_from_logits(logits)
        classes = torch.arange(self.num_classes, device=probs.device, dtype=torch.float32)
        return (probs * classes).sum(dim=1)


class MultiTaskMILModel(nn.Module):
    """
    Multi-task MIL model with a shared recording-level encoder.
    Heads:
      - SA: regression
      - RRB: ordinal (CORAL)
    Total is NOT a separate head to avoid shortcut learning.
    We always define: Total_pred = SA_pred + E[RRB_pred]
    """
    def __init__(self, input_dim=49, emb_dim=128, dropout=0.2, num_classes=9, encoder_variant: str = "tcn_stats", pool_variant: str = "gated"):
        super().__init__()
        self.num_classes = num_classes
        self.enc = _MatrixEncoder(input_dim=input_dim, emb_dim=emb_dim, dropout=dropout, encoder_variant=encoder_variant)
        # Task-specific bag pooling: SA and RRB can attend to different matrices
        self.sa_pool = _BagAttentionPool(emb_dim=self.enc.out_dim, dropout=dropout, pool_variant=pool_variant)
        self.rrb_pool = _BagAttentionPool(emb_dim=self.enc.out_dim, dropout=dropout, pool_variant=pool_variant)

        self.sa_head = nn.Sequential(
            nn.LayerNorm(self.enc.out_dim),
            nn.Linear(self.enc.out_dim, max(1, self.enc.out_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(1, self.enc.out_dim // 2), 1),
        )
        self.rrb_head = nn.Sequential(
            nn.LayerNorm(self.enc.out_dim),
            nn.Linear(self.enc.out_dim, max(1, self.enc.out_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(1, self.enc.out_dim // 2), num_classes - 1),
        )

    def forward(self, X):
        if X.dim() == 3:
            X = X.unsqueeze(1)
        B, M, T, F = X.shape
        z = self.enc(X.reshape(B * M, T, F)).reshape(B, M, -1)
        sa_pooled, _ = self.sa_pool(z)
        rrb_pooled, _ = self.rrb_pool(z)
        sa = self.sa_head(sa_pooled)              # (B, 1)
        rrb_logits = self.rrb_head(rrb_pooled)    # (B, K-1)
        return sa, rrb_logits

    def rrb_predict_continuous(self, rrb_logits):
        # logits -> per-class probs -> expected value
        cum = torch.sigmoid(rrb_logits)
        ones = torch.ones(cum.size(0), 1, device=cum.device)
        zeros = torch.zeros(cum.size(0), 1, device=cum.device)
        cum_ext = torch.cat([ones, cum, zeros], dim=1)
        probs = cum_ext[:, :-1] - cum_ext[:, 1:]
        classes = torch.arange(self.num_classes, device=probs.device, dtype=torch.float32)
        return (probs * classes).sum(dim=1)  # (B,)


# ============================================================
# Dataset
# ============================================================
class ASDDataset(Dataset):
    """(Legacy) matrix-level dataset kept for compatibility; NOT used by default anymore."""
    def __init__(self, data_file_path, target='sa', rec_indices=None, 
                 scaler=None, fit_scaler=True):
        self.data = loadmat(data_file_path)
        self.target = target.lower()
        
        self.num_mats = int(np.squeeze(self.data.get('num_matrices', 1)))
        
        # Features
        raw_features = self.data['features']
        if raw_features.dtype == 'O':
            feats_list = []
            for i in range(raw_features.shape[0]):
                feat = raw_features[i, 0] if raw_features.ndim == 2 else raw_features[i]
                feats_list.append(np.array(feat))
            self.all_features = np.stack(feats_list)
        else:
            self.all_features = raw_features
        self.all_features = np.nan_to_num(self.all_features)
        
        # Labels
        key = self.target if self.target in self.data else self.target.upper()
        self.labels = np.squeeze(self.data[key])
        if self.target == 'rrb':
            self.labels = self.labels.astype(np.int64)
        
        # Recording indices
        n_recordings = self.all_features.shape[0] // self.num_mats
        self.rec_indices = rec_indices if rec_indices is not None else np.arange(n_recordings)
        
        # Build samples
        self.samples = []
        for rec_idx in self.rec_indices:
            for mat_idx in range(self.num_mats):
                feat_idx = rec_idx * self.num_mats + mat_idx
                if feat_idx < len(self.all_features):
                    self.samples.append({'feat_idx': feat_idx, 'rec_idx': rec_idx})
        
        # Scaler
        self.scaler = scaler
        if fit_scaler and scaler is None:
            self._fit_scaler()

    def _fit_scaler(self):
        feat_indices = [s['feat_idx'] for s in self.samples]
        X = self.all_features[feat_indices]
        X_flat = X.reshape(-1, X.shape[-1])
        self.scaler = StandardScaler()
        self.scaler.fit(X_flat)

    def transform(self, X):
        if self.scaler is None:
            return X
        orig_shape = X.shape
        return self.scaler.transform(X.reshape(-1, orig_shape[-1])).reshape(orig_shape)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        X = self.all_features[sample['feat_idx']]
        X = self.transform(X[np.newaxis, ...])[0]
        y = self.labels[sample['rec_idx']]
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        if self.target == 'rrb':
            y_tensor = torch.tensor(y, dtype=torch.long)
        else:
            y_tensor = torch.tensor(y, dtype=torch.float32)
        
        return X_tensor, y_tensor, sample['rec_idx']


class ASDRecordDataset(Dataset):
    """Recording-level dataset: one sample = all matrices of a recording (M, 100, 49)."""
    def __init__(self, data_file_path, target='sa', rec_indices=None, scaler=None, fit_scaler=True):
        self.data = loadmat(data_file_path)
        self.target = target.lower()
        self.num_mats = int(np.squeeze(self.data.get('num_matrices', 1)))

        raw_features = self.data['features']
        if raw_features.dtype == 'O':
            feats_list = []
            for i in range(raw_features.shape[0]):
                feat = raw_features[i, 0] if raw_features.ndim == 2 else raw_features[i]
                feats_list.append(np.array(feat))
            all_features = np.stack(feats_list)
        else:
            all_features = raw_features
        self.all_features = np.nan_to_num(all_features)

        key = self.target if self.target in self.data else self.target.upper()
        self.labels = np.squeeze(self.data[key])
        if self.target == 'rrb':
            self.labels = self.labels.astype(np.int64)

        n_recordings = self.all_features.shape[0] // self.num_mats
        self.rec_indices = rec_indices if rec_indices is not None else np.arange(n_recordings)

        self.scaler = scaler
        if fit_scaler and scaler is None:
            self._fit_scaler()

    def _fit_scaler(self):
        # fit on TRAIN recordings only (no leakage)
        idxs = []
        for rec_idx in self.rec_indices:
            base = rec_idx * self.num_mats
            idxs.extend(list(range(base, base + self.num_mats)))
        X = self.all_features[idxs]  # (N*M, 100, 49)
        self.scaler = StandardScaler()
        self.scaler.fit(X.reshape(-1, X.shape[-1]))

    def _transform(self, X):
        if self.scaler is None:
            return X
        shape = X.shape
        return self.scaler.transform(X.reshape(-1, shape[-1])).reshape(shape)

    def __len__(self):
        return len(self.rec_indices)

    def __getitem__(self, idx):
        rec_idx = int(self.rec_indices[idx])
        base = rec_idx * self.num_mats
        mats = self.all_features[base: base + self.num_mats]  # (M, 100, 49)
        mats = self._transform(mats)
        y = self.labels[rec_idx]

        X_tensor = torch.tensor(mats, dtype=torch.float32)
        if self.target == 'rrb':
            y_tensor = torch.tensor(int(y), dtype=torch.long)
        else:
            y_tensor = torch.tensor(float(y), dtype=torch.float32)
        return X_tensor, y_tensor, rec_idx


def permute_recording_labels(dataset, seed):
    """Permute labels only among the recording indices present in a training fold."""
    indices = np.asarray(dataset.rec_indices, dtype=int)
    values = np.asarray(dataset.labels[indices]).copy()
    rng = np.random.default_rng(int(seed))
    rng.shuffle(values)
    dataset.labels = np.asarray(dataset.labels).copy()
    dataset.labels[indices] = values


# ============================================================
# Training
# ============================================================
class EarlyStopping:
    def __init__(self, patience=30):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, score, model):
        if self.best_score is None or score < self.best_score:
            self.best_score = score
            self.best_model = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def calculate_metrics(y_true, y_pred, is_ordinal=False, y_range_override=None):
    """Calculate evaluation metrics."""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    if y_range_override is None:
        y_range = np.max(y_true) - np.min(y_true)
    else:
        y_range = float(y_range_override)
    nrmse = rmse / y_range if y_range > 0 else 0
    
    if np.std(y_pred) < 1e-8 or np.std(y_true) < 1e-8:
        r, p = 0.0, 1.0
        r_spear, p_spear = 0.0, 1.0
    else:
        r, p = stats.pearsonr(y_pred, y_true)
        r_spear, p_spear = stats.spearmanr(y_pred, y_true)
    
    mx, my = np.mean(y_pred), np.mean(y_true)
    sx2, sy2 = np.var(y_pred), np.var(y_true)
    sxy = np.mean((y_pred - mx) * (y_true - my))
    ccc = (2 * sxy) / (sx2 + sy2 + (mx - my)**2 + 1e-8)
    
    metrics = {
        'RMSE': float(rmse), 'NRMSE': float(nrmse),
        'R': float(r) if not np.isnan(r) else 0.0,
        'R_spear': float(r_spear) if not np.isnan(r_spear) else 0.0,
        'CCC': float(ccc) if not np.isnan(ccc) else 0.0,
        'p': float(p) if not np.isnan(p) else 1.0,
    }
    
    if is_ordinal:
        metrics['Accuracy'] = float(np.mean(np.round(y_pred) == y_true))
    
    return metrics


def train_sa_fold(train_dataset, val_dataset, params, device):
    """Train a matched scalar-regression or cumulative-threshold SA model."""
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
    
    score_min = float(params.get('sa_score_min', 0.0))
    score_max = float(params.get('sa_score_max', 22.0))
    num_classes = int(round(score_max - score_min)) + 1
    model = SA_Model(
        emb_dim=params.get('emb_dim', 128),
        dropout=params.get('dropout', 0.2),
        encoder_variant=params.get('encoder_variant', 'tcn_stats'),
        pool_variant=params.get('pool_variant', 'gated'),
        sa_head_variant=params.get('sa_head_variant', 'regression'),
        sa_num_classes=num_classes,
        sa_score_min=score_min,
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'],
                            weight_decay=params.get('weight_decay', 1e-4))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion_mse = nn.MSELoss()
    criterion_huber = nn.SmoothL1Loss()
    criterion_ccc = CCCLoss()
    criterion_ordinal = WeightedCORALLoss(num_classes=num_classes)
    use_mse_only = str(params.get('encoder_variant', '')).lower() in ('paper_avg', 'paper_same_avg')
    sa_loss_type = str(params.get('sa_loss_type', 'ccc_mse')).lower()
    w_ordinal = float(params.get('sa_ordinal_weight', 1.0))
    default_point_weight = 0.0 if sa_loss_type in ('ordinal', 'coral') else 1.0
    w_point = float(params.get('sa_point_weight', default_point_weight))
    early_stopping = EarlyStopping(patience=params.get('patience', 30))

    train_labels = []
    if model.is_ordinal:
        for i in range(len(train_dataset)):
            _, target, _ = train_dataset[i]
            shifted = int(round(float(target))) - int(round(score_min))
            if shifted < 0 or shifted >= num_classes:
                raise ValueError(f"SA label {float(target)} is outside [{score_min}, {score_max}]")
            train_labels.append(shifted)
        counts = np.bincount(np.asarray(train_labels), minlength=num_classes)
        counts = np.maximum(counts, 1)
        inv = 1.0 / counts
        class_weights = torch.tensor(inv / inv.mean(), dtype=torch.float32, device=device)
    else:
        class_weights = None

    def _sa_loss(preds, targets):
        if use_mse_only or sa_loss_type == 'mse':
            return criterion_mse(preds, targets)
        if sa_loss_type == 'huber':
            return criterion_huber(preds, targets)
        if sa_loss_type in ('ccc_huber', 'ccc_smoothl1'):
            return criterion_ccc(preds, targets) + params.get('sa_huber_weight', 0.10) * criterion_huber(preds, targets)
        return criterion_ccc(preds, targets) + params.get('sa_mse_weight', 0.05) * criterion_mse(preds, targets)

    def _forward_and_loss(inputs, targets):
        if not model.is_ordinal:
            predictions = model(inputs)
            return predictions, _sa_loss(predictions, targets)

        logits = model.forward_logits(inputs)
        predictions = model.predict_continuous_from_logits(logits)
        ordinal_targets = torch.round(targets.view(-1) - score_min).long()
        ordinal_loss = criterion_ordinal(logits, ordinal_targets, class_weights=class_weights)
        if sa_loss_type in ('ordinal', 'coral'):
            point_loss = predictions.sum() * 0.0
        elif sa_loss_type in ('ordinal_huber', 'coral_huber'):
            point_loss = criterion_huber(predictions, targets)
        elif sa_loss_type in ('ordinal_ccc_huber', 'coral_ccc_huber'):
            point_loss = criterion_ccc(predictions, targets) + params.get('sa_huber_weight', 0.10) * criterion_huber(predictions, targets)
        else:
            point_loss = criterion_ccc(predictions, targets) + params.get('sa_mse_weight', 0.05) * criterion_mse(predictions, targets)
        return predictions, w_ordinal * ordinal_loss + w_point * point_loss
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(params['epochs']):
        model.train()
        train_loss = 0
        for inputs, targets, _ in train_loader:
            inputs, targets = inputs.to(device), targets.to(device).view(-1, 1)
            optimizer.zero_grad()
            _, loss = _forward_and_loss(inputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets, _ in val_loader:
                inputs, targets = inputs.to(device), targets.to(device).view(-1, 1)
                _, loss = _forward_and_loss(inputs, targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)
        
        if early_stopping(val_loss, model):
            print(f"  [SA] Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 10 == 0:
            print(f"  [SA] Epoch {epoch+1}: Train {train_loss:.4f} | Val {val_loss:.4f}")
    
    if early_stopping.best_model:
        model.load_state_dict(early_stopping.best_model)
    
    # Evaluate
    model.eval()
    all_preds, all_true, all_recs = [], [], []
    with torch.no_grad():
        for inputs, targets, rec_ids in val_loader:
            preds = model(inputs.to(device)).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_true.extend(targets.numpy().flatten())
            all_recs.extend(rec_ids.tolist())
    
    # Recording-level dataset => already 1 pred per rec, but keep robust
    unique_recs = np.unique(all_recs)
    agg_preds = [np.mean([all_preds[i] for i, r in enumerate(all_recs) if r == rec]) for rec in unique_recs]
    agg_true = [all_true[all_recs.index(rec)] for rec in unique_recs]
    
    return {
        'model': model,
        'scaler': train_dataset.scaler,
        'history': history,
        'metrics': calculate_metrics(
            agg_true,
            agg_preds,
            is_ordinal=model.is_ordinal,
            y_range_override=(float(params.get('sa_score_max', 22.0)) - float(params.get('sa_score_min', 0.0))),
        ),
        'preds': np.array(agg_preds),
        'true': np.array(agg_true),
        'rec_ids': np.array(unique_recs),
        'num_classes': num_classes,
    }


def train_rrb_fold(train_dataset, val_dataset, params, device, num_classes):
    """Train RRB ordinal model."""
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
    
    model = RRB_Model(
        emb_dim=params.get('emb_dim', 128),
        dropout=params.get('dropout', 0.2),
        num_classes=num_classes,
        encoder_variant=params.get('encoder_variant', 'tcn_stats'),
        pool_variant=params.get('pool_variant', 'gated'),
        rrb_pool_variant=params.get('rrb_pool_variant', ''),
        feat_group_dropout=float(params.get('feat_group_dropout', 0.0)),
        rrb_head_variant=params.get('rrb_head_variant', 'independent'),
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'],
                            weight_decay=params.get('weight_decay', 1e-4))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    rrb_loss_type = str(params.get('rrb_loss_type', 'coral')).lower()
    if rrb_loss_type == 'focal':
        criterion_coral = FocalOrdinalLoss(num_classes=num_classes, gamma=float(params.get('focal_gamma', 2.0)))
    else:
        criterion_coral = WeightedCORALLoss(num_classes=num_classes)
    reg_loss_type = str(params.get('rrb_reg_loss_type', 'huber')).lower()
    criterion_reg = nn.MSELoss() if reg_loss_type == 'mse' else nn.SmoothL1Loss()
    criterion_ccc = CCCLoss()
    early_stopping = EarlyStopping(patience=params.get('patience', 30))
    
    history = {'train_loss': [], 'val_loss': []}

    # class weights computed from TRAIN fold labels (recording-level)
    train_labels = []
    for i in range(len(train_dataset)):
        _, y, _ = train_dataset[i]
        train_labels.append(int(y))
    counts = np.bincount(np.array(train_labels), minlength=num_classes)
    counts = np.maximum(counts, 1)
    inv = 1.0 / counts
    class_weights = torch.tensor(inv / inv.mean(), dtype=torch.float32, device=device)

    w_ordinal = float(params.get('rrb_ordinal_weight', 1.0))
    w_rrb_reg = float(params.get('rrb_reg_weight', 0.3))
    w_anti = float(params.get('rrb_anti_collapse_weight', 0.0))
    w_ccc = float(params.get('rrb_ccc_weight', 0.0))
    w_rank = float(params.get('rrb_rank_weight', 0.0))
    w_moment = float(params.get('rrb_moment_weight', 0.0))
    rank_temperature = float(params.get('rrb_rank_temperature', 1.0))
    rank_min_gap = float(params.get('rrb_rank_min_gap', 1.0))
    score_range = float(params.get('rrb_score_max', 8.0)) - float(params.get('rrb_score_min', 0.0))
    w_low_score = float(params.get('low_score_weight', 1.0))
    w_low_over = float(params.get('low_over_weight', 0.0))
    low_score_thresh = int(params.get('low_score_thresh', 2))
    
    for epoch in range(params['epochs']):
        model.train()
        train_loss = 0
        for inputs, targets, _ in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)

            sample_w = None
            if w_low_score != 1.0:
                sample_w = torch.ones(targets.size(0), device=device)
                sample_w[targets <= low_score_thresh] = w_low_score

            loss_ord = criterion_coral(logits, targets, class_weights=class_weights,
                                       sample_weights=sample_w)
            pred_cont = model.predict_continuous_from_logits(logits)
            loss_reg = criterion_reg(pred_cont, targets.float())
            anti = -torch.var(pred_cont)
            loss_ccc = criterion_ccc(pred_cont, targets.float())
            loss_rank = pairwise_ordinal_rank_loss(
                pred_cont, targets, temperature=rank_temperature, min_gap=rank_min_gap
            )
            loss_moment = ordinal_moment_loss(pred_cont, targets, score_range=score_range)
            loss = (
                w_ordinal * loss_ord + w_rrb_reg * loss_reg + w_anti * anti
                + w_ccc * loss_ccc + w_rank * loss_rank + w_moment * loss_moment
            )

            if w_low_over > 0:
                low_mask = targets <= low_score_thresh
                if low_mask.any():
                    over = torch.relu(pred_cont.view(-1) - targets.float())
                    loss = loss + w_low_over * (over[low_mask] ** 2).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets, _ in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                loss_ord = criterion_coral(logits, targets, class_weights=class_weights)
                pred_cont = model.predict_continuous_from_logits(logits)
                loss_reg = criterion_reg(pred_cont, targets.float())
                anti = -torch.var(pred_cont)
                loss_ccc = criterion_ccc(pred_cont, targets.float())
                loss_rank = pairwise_ordinal_rank_loss(
                    pred_cont, targets, temperature=rank_temperature, min_gap=rank_min_gap
                )
                loss_moment = ordinal_moment_loss(pred_cont, targets, score_range=score_range)
                val_loss += (
                    w_ordinal * loss_ord + w_rrb_reg * loss_reg + w_anti * anti
                    + w_ccc * loss_ccc + w_rank * loss_rank + w_moment * loss_moment
                ).item()
        val_loss /= len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)
        
        if early_stopping(val_loss, model):
            print(f"  [RRB] Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 10 == 0:
            print(f"  [RRB] Epoch {epoch+1}: Train {train_loss:.4f} | Val {val_loss:.4f}")
    
    if early_stopping.best_model:
        model.load_state_dict(early_stopping.best_model)
    
    # Evaluate
    model.eval()
    all_preds, all_true, all_recs = [], [], []
    with torch.no_grad():
        for inputs, targets, rec_ids in val_loader:
            preds = model.predict_continuous(inputs.to(device)).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_true.extend(targets.numpy().flatten())
            all_recs.extend(rec_ids.tolist())
    
    # Recording-level dataset => already 1 pred per rec, but keep robust
    unique_recs = np.unique(all_recs)
    agg_preds = [np.mean([all_preds[i] for i, r in enumerate(all_recs) if r == rec]) for rec in unique_recs]
    agg_true = [all_true[all_recs.index(rec)] for rec in unique_recs]
    
    return {
        'model': model,
        'scaler': train_dataset.scaler,
        'history': history,
        'metrics': calculate_metrics(
            agg_true,
            agg_preds,
            is_ordinal=True,
            y_range_override=(float(params.get('rrb_score_max', 8.0)) - float(params.get('rrb_score_min', 0.0))),
        ),
        'preds': np.array(agg_preds),
        'true': np.array(agg_true),
        'num_classes': num_classes
    }


def train_multitask_fold(train_dataset_sa, train_dataset_rrb, val_dataset_sa, val_dataset_rrb, params, device, num_classes):
    """
    Train a single MultiTaskMILModel on shared X with both labels.
    Assumes datasets share the same rec_indices ordering for train/val.
    """
    # Build loaders from a lightweight paired dataset
    class _Paired(Dataset):
        def __init__(self, ds_sa, ds_rrb):
            assert len(ds_sa) == len(ds_rrb)
            self.ds_sa = ds_sa
            self.ds_rrb = ds_rrb
        def __len__(self):
            return len(self.ds_sa)
        def __getitem__(self, idx):
            X_sa, y_sa, rec = self.ds_sa[idx]
            X_rrb, y_rrb, rec2 = self.ds_rrb[idx]
            # same X after scaling because scaler identical source; keep X_sa
            assert rec == rec2
            return X_sa, y_sa, y_rrb, rec

    train_loader = DataLoader(_Paired(train_dataset_sa, train_dataset_rrb),
                              batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(_Paired(val_dataset_sa, val_dataset_rrb),
                            batch_size=params['batch_size'], shuffle=False)

    model = MultiTaskMILModel(
        emb_dim=params.get('emb_dim', 128),
        dropout=params.get('dropout', 0.2),
        num_classes=num_classes,
        encoder_variant=params.get('encoder_variant', 'tcn_stats'),
        pool_variant=params.get('pool_variant', 'gated'),
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'],
                            weight_decay=params.get('weight_decay', 1e-4))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # losses
    ccc = CCCLoss()
    mse = nn.MSELoss()
    huber = nn.SmoothL1Loss()
    coral = WeightedCORALLoss(num_classes=num_classes)
    rrb_reg = nn.SmoothL1Loss()

    early_stopping = EarlyStopping(patience=params.get('patience', 30))

    # class weights from train RRB labels
    train_rrb_labels = []
    for i in range(len(train_dataset_rrb)):
        _, y, _ = train_dataset_rrb[i]
        train_rrb_labels.append(int(y))
    counts = np.bincount(np.array(train_rrb_labels), minlength=num_classes)
    counts = np.maximum(counts, 1)
    inv = 1.0 / counts
    class_weights = torch.tensor(inv / inv.mean(), dtype=torch.float32, device=device)

    w_sa = float(params.get('sa_weight', 1.0))
    w_rrb = float(params.get('rrb_weight', 1.0))
    w_total = float(params.get('total_weight', 1.0))
    w_cons = float(params.get('consistency_weight', 0.0))  # deprecated (no total head)
    w_sa_mse = float(params.get('sa_mse_weight', 0.05))
    w_rrb_reg = float(params.get('rrb_reg_weight', 0.3))
    w_anti = float(params.get('anti_collapse_weight', 0.05))

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(params['epochs']):
        model.train()
        train_loss = 0.0
        for X, y_sa, y_rrb, _ in train_loader:
            X = X.to(device)
            y_sa = y_sa.to(device).view(-1, 1)
            y_rrb = y_rrb.to(device)
        
            optimizer.zero_grad()
            pred_sa, rrb_logits = model(X)
            pred_rrb_cont = model.rrb_predict_continuous(rrb_logits)

            true_total = (y_sa.view(-1) + y_rrb.float()).view(-1, 1)
            pred_total = (pred_sa.view(-1) + pred_rrb_cont).view(-1, 1)

            loss_sa = ccc(pred_sa, y_sa) + w_sa_mse * mse(pred_sa, y_sa)
            loss_rrb_ord = coral(rrb_logits, y_rrb, class_weights=class_weights)
            loss_rrb_reg = rrb_reg(pred_rrb_cont, y_rrb.float())
            loss_total = huber(pred_total, true_total)
            # Anti-collapse: reward variance of predictions within a batch (prevents constant ~4)
            anti = -torch.var(pred_rrb_cont)  # maximize variance => negative in loss

            loss = (
                w_sa * loss_sa
                + w_rrb * (loss_rrb_ord + w_rrb_reg * loss_rrb_reg)
                + w_total * loss_total
                + w_anti * anti
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.item())
        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y_sa, y_rrb, _ in val_loader:
                X = X.to(device)
                y_sa = y_sa.to(device).view(-1, 1)
                y_rrb = y_rrb.to(device)
                pred_sa, rrb_logits = model(X)
                pred_rrb_cont = model.rrb_predict_continuous(rrb_logits)
                true_total = (y_sa.view(-1) + y_rrb.float()).view(-1, 1)
                pred_total = (pred_sa.view(-1) + pred_rrb_cont).view(-1, 1)

                loss_sa = ccc(pred_sa, y_sa) + w_sa_mse * mse(pred_sa, y_sa)
                loss_rrb_ord = coral(rrb_logits, y_rrb, class_weights=class_weights)
                loss_rrb_reg = rrb_reg(pred_rrb_cont, y_rrb.float())
                loss_total = huber(pred_total, true_total)
                anti = -torch.var(pred_rrb_cont)
                loss = (
                    w_sa * loss_sa
                    + w_rrb * (loss_rrb_ord + w_rrb_reg * loss_rrb_reg)
                    + w_total * loss_total
                    + w_anti * anti
                )
                val_loss += float(loss.item())
        val_loss /= max(1, len(val_loader))

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)

        if early_stopping(val_loss, model):
            print(f"  [MTL] Early stopping at epoch {epoch+1}")
            break
        if (epoch + 1) % 10 == 0:
            print(f"  [MTL] Epoch {epoch+1}: Train {train_loss:.4f} | Val {val_loss:.4f}")

    if early_stopping.best_model:
        model.load_state_dict(early_stopping.best_model)

    # Evaluate on val: collect rec-level preds
    model.eval()
    sa_preds, sa_true = [], []
    rrb_preds, rrb_true = [], []
    total_preds, total_true = [], []
    recs = []
    with torch.no_grad():
        for X, y_sa, y_rrb, rec in val_loader:
            X = X.to(device)
            pred_sa, rrb_logits = model(X)
            pred_rrb_cont = model.rrb_predict_continuous(rrb_logits)

            sa_preds.extend(pred_sa.cpu().numpy().flatten().tolist())
            sa_true.extend(y_sa.numpy().flatten().tolist())
            rrb_preds.extend(pred_rrb_cont.cpu().numpy().flatten().tolist())
            rrb_true.extend(y_rrb.numpy().flatten().tolist())
            total_preds.extend((pred_sa.view(-1) + pred_rrb_cont).cpu().numpy().flatten().tolist())  # paper definition
            total_true.extend((y_sa.view(-1) + y_rrb.float()).numpy().flatten().tolist())
            recs.extend(rec.tolist())

    # aggregate (robust)
    unique = np.unique(recs)
    def _agg(vals):
        return [np.mean([vals[i] for i, r in enumerate(recs) if r == rec]) for rec in unique]
    def _first(vals):
        return [vals[recs.index(rec)] for rec in unique]

    sa_p, sa_t = _agg(sa_preds), _first(sa_true)
    rrb_p, rrb_t = _agg(rrb_preds), _first(rrb_true)
    total_p, total_t = _agg(total_preds), _first(total_true)

    sa_range = float(params.get('sa_score_max', 22.0)) - float(params.get('sa_score_min', 0.0))
    rrb_range = float(params.get('rrb_score_max', 8.0)) - float(params.get('rrb_score_min', 0.0))
    total_range = sa_range + rrb_range

    return {
        'model': model,
        'history': history,
        'metrics_sa': calculate_metrics(sa_t, sa_p, y_range_override=sa_range),
        'metrics_rrb': calculate_metrics(rrb_t, rrb_p, is_ordinal=True, y_range_override=rrb_range),
        'metrics_total': calculate_metrics(total_t, total_p, y_range_override=total_range),
        'preds_sa': np.array(sa_p),
        'true_sa': np.array(sa_t),
        'preds_rrb': np.array(rrb_p),
        'true_rrb': np.array(rrb_t),
        'preds_total': np.array(total_p),
        'true_total': np.array(total_t),
        'num_classes': num_classes,
    }


def plot_loss(output_dir, train_loss, val_loss, title):
    """Plot training loss."""
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss, label='Train')
    plt.plot(val_loss, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / f'{title.replace(" ", "_")}_loss.png', dpi=150)
    plt.close()


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='config.yaml')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    print(f"Random seed: {args.seed}")
    
    config = load_config(args.config)
    params = config['params_config']
    paths = config['paths_config']
    
    # Mode selection:
    # - SA_ONLY: only train SA_Model
    # - DUAL: train two separate branches in ONE experiment dir: SA_Model + RRB_Model, then Total=SA+RRB
    # - MIL/MTL: train MultiTaskMILModel (single model predicts SA and RRB)
    model_type = str(params.get('model_type', 'MIL')).upper()
    sa_only = (model_type == 'SA_ONLY') or (
        float(params.get('rrb_weight', 1.0)) == 0.0 and float(params.get('total_weight', 1.0)) == 0.0
    )
    dual = model_type in ("DUAL", "DUAL_BRANCH", "SEPARATE", "SEPARATE_BRANCHES")

    def _branch_params(branch_key: str):
        """Allow per-branch overrides via params_config.{sa_params, rrb_params} (optional)."""
        base = dict(params)
        overrides = params.get(branch_key, None)
        if isinstance(overrides, dict):
            base.update(overrides)
        return base
    
    device = torch.device(f"cuda:{params['gpu_id']}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent
    data_path = root_dir / paths['data_file_path']
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = Path(paths.get('output_dir', 'outputs'))
    if not out_base.is_absolute():
        out_base = root_dir / out_base
    output_dir = out_base / f"experiment_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")
    
    # Load data info
    full_data = loadmat(data_path)
    num_mats = int(np.squeeze(full_data.get('num_matrices', 1)))
    n_recordings = full_data['features'].shape[0] // num_mats
    rrb = np.squeeze(full_data.get('rrb', full_data.get('RRB')))
    num_classes = int(rrb.max()) + 1
    
    print(f"Recordings: {n_recordings}, Matrices/rec: {num_mats}, RRB classes: {num_classes}")
    
    k_folds = params['k_folds']
    # RRB is discrete and imbalanced; stratify to keep folds comparable
    try:
        splitter = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=args.seed)
        split_iter = splitter.split(np.arange(n_recordings), rrb[:n_recordings])
        print("Using StratifiedKFold by RRB.")
    except Exception as e:
        print(f"Warning: StratifiedKFold unavailable ({e}); falling back to KFold.")
        splitter = KFold(n_splits=k_folds, shuffle=True, random_state=args.seed)
        split_iter = splitter.split(range(n_recordings))
    
    all_sa_results = []
    all_rrb_results = []
    all_total_results = []
    
    for fold, (train_idx, val_idx) in enumerate(split_iter):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{k_folds} (Train: {len(train_idx)}, Val: {len(val_idx)})")
        print(f"{'='*60}")
        
        fold_dir = output_dir / f'fold_{fold+1}'
        fold_dir.mkdir(exist_ok=True)
        
        set_seed(args.seed + fold)

        if sa_only:
            print("\n--- Training SA ONLY (Recording-level MIL) ---")
            sa_params = _branch_params('sa_params')
            if (
                bool(sa_params.get('matrix_level_supervision', False))
                or str(sa_params.get('encoder_variant', params.get('encoder_variant', 'tcn_stats'))).lower()
                in ('paper_avg', 'paper_same_avg')
            ):
                train_sa = ASDDataset(data_path, target='sa', rec_indices=train_idx, fit_scaler=True)
                val_sa = ASDDataset(
                    data_path, target='sa', rec_indices=val_idx,
                    scaler=train_sa.scaler, fit_scaler=False
                )
            else:
                train_sa = ASDRecordDataset(data_path, target='sa', rec_indices=train_idx, fit_scaler=True)
                val_sa = ASDRecordDataset(
                    data_path, target='sa', rec_indices=val_idx,
                    scaler=train_sa.scaler, fit_scaler=False
                )
            fold_res = train_sa_fold(train_sa, val_sa, sa_params, device)
            all_sa_results.append({'metrics': fold_res['metrics']})

            print(f"  SA: R={fold_res['metrics']['R']:.4f}, CCC={fold_res['metrics']['CCC']:.4f}")

            # Save SA-only model
            torch.save({
                'model_state_dict': fold_res['model'].state_dict(),
                'scaler_mean': train_sa.scaler.mean_,
                'scaler_scale': train_sa.scaler.scale_,
                'model_type': 'SA_Model',
                'model_kwargs': {
                    'emb_dim': sa_params.get('emb_dim', params.get('emb_dim', 128)),
                    'dropout': sa_params.get('dropout', params.get('dropout', 0.2)),
                    'encoder_variant': sa_params.get('encoder_variant', params.get('encoder_variant', 'tcn_stats')),
                    'pool_variant': sa_params.get('pool_variant', params.get('pool_variant', 'gated')),
                    'sa_head_variant': sa_params.get('sa_head_variant', 'regression'),
                    'sa_num_classes': fold_res.get('num_classes', 23),
                    'sa_score_min': float(sa_params.get('sa_score_min', params.get('sa_score_min', 0.0))),
                },
                'metrics': fold_res['metrics'],
            }, fold_dir / 'model_SA.pth')

            # Save predictions
            np.savetxt(
                fold_dir / 'val_pred_SA.txt',
                np.column_stack([fold_res['rec_ids'], fold_res['true'], fold_res['preds']]),
                header='rec_id true_SA pred_SA',
                fmt='%.4f'
            )

            # Plot loss
            plot_loss(
                fold_dir,
                fold_res['history']['train_loss'],
                fold_res['history']['val_loss'],
                f'Fold_{fold+1}_SA'
            )
        elif dual:
            print("\n--- Training DUAL branches (SA-only + RRB-only) in one experiment ---")

            # Fit ONE scaler on train fold and reuse for both targets (avoid mismatch)
            sa_params = _branch_params('sa_params')
            rrb_params = _branch_params('rrb_params')

            if (
                bool(sa_params.get('matrix_level_supervision', False))
                or str(sa_params.get('encoder_variant', params.get('encoder_variant', 'tcn_stats'))).lower()
                in ('paper_avg', 'paper_same_avg')
            ):
                train_sa = ASDDataset(data_path, target='sa', rec_indices=train_idx, fit_scaler=True)
                val_sa = ASDDataset(
                    data_path, target='sa', rec_indices=val_idx,
                    scaler=train_sa.scaler, fit_scaler=False
                )
            else:
                train_sa = ASDRecordDataset(data_path, target='sa', rec_indices=train_idx, fit_scaler=True)
                val_sa = ASDRecordDataset(
                    data_path, target='sa', rec_indices=val_idx,
                    scaler=train_sa.scaler, fit_scaler=False
                )
            rrb_dataset_cls = ASDDataset if bool(rrb_params.get('matrix_level_supervision', False)) else ASDRecordDataset
            train_rrb = rrb_dataset_cls(
                data_path, target='rrb', rec_indices=train_idx,
                scaler=train_sa.scaler, fit_scaler=False
            )
            val_rrb = rrb_dataset_cls(
                data_path, target='rrb', rec_indices=val_idx,
                scaler=train_sa.scaler, fit_scaler=False
            )
            if rrb_dataset_cls is ASDDataset:
                print("  RRB ablation: matrix-level duplicated labels; recording prediction averages matrix outputs.")
            if bool(rrb_params.get('permute_train_labels', False)):
                permute_recording_labels(train_rrb, args.seed + fold + 700000)
                print("  RRB negative control: permuted training-fold recording labels.")

            # Train SA branch
            if params.get('isolate_branch_rng', False):
                set_seed(args.seed + fold + int(params.get('sa_seed_offset', 0)))
            sa_res = train_sa_fold(train_sa, val_sa, sa_params, device)
            all_sa_results.append({'metrics': sa_res['metrics']})

            # Train RRB branch
            if params.get('isolate_branch_rng', False):
                set_seed(args.seed + fold + int(params.get('rrb_seed_offset', 100000)))
            rrb_res = train_rrb_fold(train_rrb, val_rrb, rrb_params, device, num_classes)
            all_rrb_results.append({'metrics': rrb_res['metrics']})

            # Total = SA + RRB (paper definition)
            # Both are aggregated by unique rec order (np.unique -> sorted), so alignment is stable.
            total_true = sa_res['true'].astype(np.float32) + rrb_res['true'].astype(np.float32)
            total_pred = sa_res['preds'].astype(np.float32) + rrb_res['preds'].astype(np.float32)
            total_metrics = calculate_metrics(
                total_true,
                total_pred,
                y_range_override=(
                    (float(params.get('sa_score_max', 22.0)) - float(params.get('sa_score_min', 0.0)))
                    + (float(params.get('rrb_score_max', 8.0)) - float(params.get('rrb_score_min', 0.0)))
                ),
            )
            all_total_results.append({'metrics': total_metrics})

            print(f"  SA:    R={sa_res['metrics']['R']:.4f}, CCC={sa_res['metrics']['CCC']:.4f}")
            print(f"  RRB:   R={rrb_res['metrics']['R']:.4f}, CCC={rrb_res['metrics']['CCC']:.4f}, Acc={rrb_res['metrics']['Accuracy']:.2%}")
            print(f"  Total: R={total_metrics['R']:.4f}, CCC={total_metrics['CCC']:.4f}")

            # Save models (same fold dir, single experiment)
            torch.save({
                'model_state_dict': sa_res['model'].state_dict(),
                'scaler_mean': train_sa.scaler.mean_,
                'scaler_scale': train_sa.scaler.scale_,
                'model_type': 'SA_Model',
                'model_kwargs': {
                    'emb_dim': sa_params.get('emb_dim', params.get('emb_dim', 128)),
                    'dropout': sa_params.get('dropout', params.get('dropout', 0.2)),
                    'encoder_variant': sa_params.get('encoder_variant', params.get('encoder_variant', 'tcn_stats')),
                    'pool_variant': sa_params.get('pool_variant', params.get('pool_variant', 'gated')),
                    'matrix_level_supervision': bool(sa_params.get('matrix_level_supervision', False)),
                    'sa_head_variant': sa_params.get('sa_head_variant', 'regression'),
                    'sa_num_classes': sa_res.get('num_classes', 23),
                    'sa_score_min': float(sa_params.get('sa_score_min', params.get('sa_score_min', 0.0))),
                },
                'metrics': sa_res['metrics'],
            }, fold_dir / 'model_SA.pth')

            torch.save({
                'model_state_dict': rrb_res['model'].state_dict(),
                'scaler_mean': train_sa.scaler.mean_,
                'scaler_scale': train_sa.scaler.scale_,
                'model_type': 'RRB_Model',
                'num_classes': num_classes,
                'model_kwargs': {
                    'emb_dim': rrb_params.get('emb_dim', params.get('emb_dim', 128)),
                    'dropout': rrb_params.get('dropout', params.get('dropout', 0.2)),
                    'encoder_variant': rrb_params.get('encoder_variant', params.get('encoder_variant', 'tcn_stats')),
                    'pool_variant': rrb_params.get('pool_variant', params.get('pool_variant', 'gated')),
                    'rrb_pool_variant': rrb_params.get('rrb_pool_variant', ''),
                    'rrb_head_variant': rrb_params.get('rrb_head_variant', 'independent'),
                    'matrix_level_supervision': bool(rrb_params.get('matrix_level_supervision', False)),
                },
                'metrics': rrb_res['metrics'],
            }, fold_dir / 'model_RRB.pth')

            # Save predictions
            np.savetxt(
                fold_dir / 'val_pred_SA.txt',
                np.column_stack([sa_res['true'], sa_res['preds']]),
                header='true_SA pred_SA',
                fmt='%.4f'
            )
            np.savetxt(
                fold_dir / 'val_pred_RRB.txt',
                np.column_stack([rrb_res['true'], rrb_res['preds']]),
                header='true_RRB pred_RRB',
                fmt='%.4f'
            )
            np.savetxt(
                fold_dir / 'val_pred_Total.txt',
                np.column_stack([total_true, total_pred]),
                header='true_Total pred_Total',
                fmt='%.4f'
            )

            # Plot loss
            plot_loss(
                fold_dir,
                sa_res['history']['train_loss'],
                sa_res['history']['val_loss'],
                f'Fold_{fold+1}_SA'
            )
            plot_loss(
                fold_dir,
                rrb_res['history']['train_loss'],
                rrb_res['history']['val_loss'],
                f'Fold_{fold+1}_RRB'
            )
        else:
            # Train MultiTask MIL
            print("\n--- Training MultiTask MIL Model (SA + RRB + Total) ---")

            # Fit ONE scaler on train fold and reuse for both targets (avoid mismatch)
            train_sa = ASDRecordDataset(data_path, target='sa', rec_indices=train_idx, fit_scaler=True)
            val_sa = ASDRecordDataset(
                data_path, target='sa', rec_indices=val_idx,
                scaler=train_sa.scaler, fit_scaler=False
            )
            train_rrb = ASDRecordDataset(
                data_path, target='rrb', rec_indices=train_idx,
                scaler=train_sa.scaler, fit_scaler=False
            )
            val_rrb = ASDRecordDataset(
                data_path, target='rrb', rec_indices=val_idx,
                scaler=train_sa.scaler, fit_scaler=False
            )
            if bool(params.get('permute_train_labels', False)):
                permute_recording_labels(train_rrb, args.seed + fold + 700000)
                print("  RRB negative control: permuted training-fold recording labels.")

            fold_res = train_multitask_fold(train_sa, train_rrb, val_sa, val_rrb, params, device, num_classes)
            all_sa_results.append({'metrics': fold_res['metrics_sa']})
            all_rrb_results.append({'metrics': fold_res['metrics_rrb']})
            all_total_results.append({'metrics': fold_res['metrics_total']})

            print(f"  SA:    R={fold_res['metrics_sa']['R']:.4f}, CCC={fold_res['metrics_sa']['CCC']:.4f}")
            print(f"  RRB:   R={fold_res['metrics_rrb']['R']:.4f}, CCC={fold_res['metrics_rrb']['CCC']:.4f}, Acc={fold_res['metrics_rrb']['Accuracy']:.2%}")
            print(f"  Total: R={fold_res['metrics_total']['R']:.4f}, CCC={fold_res['metrics_total']['CCC']:.4f}")

            # Save multitask model
            torch.save({
                'model_state_dict': fold_res['model'].state_dict(),
                'scaler_mean': train_sa.scaler.mean_,
                'scaler_scale': train_sa.scaler.scale_,
                'model_type': 'MultiTaskMILModel',
                'num_classes': num_classes,
                'model_kwargs': {
                    'emb_dim': params.get('emb_dim', 128),
                    'dropout': params.get('dropout', 0.2),
                    'encoder_variant': params.get('encoder_variant', 'tcn_stats'),
                    'pool_variant': params.get('pool_variant', 'gated'),
                },
                'metrics_sa': fold_res['metrics_sa'],
                'metrics_rrb': fold_res['metrics_rrb'],
                'metrics_total': fold_res['metrics_total'],
            }, fold_dir / 'model_MTL.pth')

            # Save predictions
            np.savetxt(
                fold_dir / 'val_pred_SA.txt',
                np.column_stack([fold_res['true_sa'], fold_res['preds_sa']]),
                header='true_SA pred_SA',
                fmt='%.4f'
            )
            np.savetxt(
                fold_dir / 'val_pred_RRB.txt',
                np.column_stack([fold_res['true_rrb'], fold_res['preds_rrb']]),
                header='true_RRB pred_RRB',
                fmt='%.4f'
            )
            np.savetxt(
                fold_dir / 'val_pred_Total.txt',
                np.column_stack([fold_res['true_total'], fold_res['preds_total']]),
                header='true_Total pred_Total',
                fmt='%.4f'
            )

            # Plot loss
            plot_loss(
                fold_dir,
                fold_res['history']['train_loss'],
                fold_res['history']['val_loss'],
                f'Fold_{fold+1}_MTL'
            )
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    sa_metrics = {k: np.mean([r['metrics'][k] for r in all_sa_results])
                  for k in all_sa_results[0]['metrics'].keys()}
    rrb_metrics = None
    total_metrics = None
    if not sa_only:
        rrb_metrics = {k: np.mean([r['metrics'][k] for r in all_rrb_results])
                       for k in all_rrb_results[0]['metrics'].keys()}
        total_metrics = {k: np.mean([r['metrics'][k] for r in all_total_results])
                         for k in all_total_results[0]['metrics'].keys()}
    
    print(f"\nSA Model ({k_folds}-fold average):")
    print(f"  R:      {sa_metrics['R']:.4f}")
    print(f"  CCC:    {sa_metrics['CCC']:.4f}")
    print(f"  RMSE:   {sa_metrics['RMSE']:.2f}")
    
    if not sa_only and rrb_metrics is not None and total_metrics is not None:
        print(f"\nRRB Model ({k_folds}-fold average, Ordinal):")
        print(f"  R:      {rrb_metrics['R']:.4f}")
        print(f"  CCC:    {rrb_metrics['CCC']:.4f}")
        print(f"  RMSE:   {rrb_metrics['RMSE']:.2f}")
        print(f"  Acc:    {rrb_metrics['Accuracy']:.2%}")

        print(f"\nTotal Model ({k_folds}-fold average):")
        print(f"  R:      {total_metrics['R']:.4f}")
        print(f"  CCC:    {total_metrics['CCC']:.4f}")
        print(f"  RMSE:   {total_metrics['RMSE']:.2f}")
    
    # Save summary
    with open(output_dir / 'summary.txt', 'w', encoding='utf-8') as f:
        f.write("ASD Speech Prediction - Training Summary\n")
        f.write("="*60 + "\n\n")
        if sa_only:
            mode_str = "SA_ONLY"
        elif dual:
            mode_str = "DUAL_BRANCHES"
        else:
            mode_str = "MTL"
        f.write(f"Mode: {mode_str}\n\n")
        f.write("Methods:\n")
        f.write("  Shared: Recording-level MIL (matrix encoder + bag attention)\n")
        f.write("  SA:     CCC (+ MSE stabilizer)\n")
        if not sa_only:
            f.write("  RRB:    Weighted CORAL (+ reg auxiliary)\n")
            if dual:
                f.write("  Total:  SA_pred + E[RRB_pred] (from two separately-trained branches)\n")
            else:
                f.write("  Total:  Huber on (SA+RRB)\n")
                f.write("  Total def: Total_pred = SA_pred + E[RRB_pred]\n")
        f.write("\n")
        f.write(f"Random seed: {args.seed}\n")
        f.write(f"K-folds: {k_folds}\n\n")

        f.write(f"SA Model ({k_folds}-fold average):\n")
        for k, v in sa_metrics.items():
            if k == 'p':
                continue
            f.write(f"  {k}: {v:.4f}\n")

        if not sa_only and rrb_metrics is not None and total_metrics is not None:
            f.write(f"\nRRB Model ({k_folds}-fold average):\n")
            for k, v in rrb_metrics.items():
                if k == 'p':
                    continue
                f.write(f"  {k}: {v:.4f}\n")

            f.write(f"\nTotal Model ({k_folds}-fold average):\n")
            for k, v in total_metrics.items():
                if k == 'p':
                    continue
                f.write(f"  {k}: {v:.4f}\n")
    
    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
