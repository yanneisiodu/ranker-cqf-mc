"""Listwise neural ranker for options using a Transformer encoder.

Architecture:
    1. Per-option MLP encodes raw features into embeddings
    2. Transformer encoder layers learn cross-option attention
    3. Ranking head produces scalar scores
    4. ListMLE loss optimizes the full permutation likelihood

Designed to re-rank top-K candidates from an upstream XGBoost ranker.
Trains on Apple Silicon via PyTorch MPS.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class NeuralRankerConfig:
    input_dim: int = 50
    embed_dim: int = 128
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.1
    mlp_hidden: int = 256
    top_k_candidates: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_dates: int = 16
    epochs: int = 50
    patience: int = 8
    seed: int = 42

    @classmethod
    def from_config(cls, config: Dict) -> "NeuralRankerConfig":
        cfg = config.get("neural_ranker", {})
        return cls(**{k: cfg[k] for k in cfg if k in cls.__dataclass_fields__})


class OptionEncoder(nn.Module):
    """MLP that encodes per-option raw features into embeddings."""

    def __init__(self, input_dim: int, embed_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ChainTransformer(nn.Module):
    """Transformer encoder over the daily option chain.

    Each option attends to all other options in the same day's chain,
    learning cross-strike, cross-tenor, and cross-type relationships.
    """

    def __init__(self, config: NeuralRankerConfig):
        super().__init__()
        self.config = config

        # Per-option feature encoder
        self.encoder = OptionEncoder(
            input_dim=config.input_dim,
            embed_dim=config.embed_dim,
            hidden_dim=config.mlp_hidden,
            dropout=config.dropout,
        )

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.n_heads,
            dim_feedforward=config.mlp_hidden,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
            enable_nested_tensor=False,
        )

        # Ranking head: embed_dim -> 1 scalar score
        self.ranking_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim // 2, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            features: (batch, seq_len, input_dim) — option features per chain
            padding_mask: (batch, seq_len) — True for padded positions

        Returns:
            scores: (batch, seq_len) — ranking scores per option
        """
        # Encode each option independently
        embeddings = self.encoder(features)  # (B, S, embed_dim)

        # Cross-option attention via transformer
        embeddings = self.transformer(
            embeddings,
            src_key_padding_mask=padding_mask,
        )  # (B, S, embed_dim)

        # Score each option
        scores = self.ranking_head(embeddings).squeeze(-1)  # (B, S)

        # Mask padded positions to -inf so they don't affect ranking
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask, float("-inf"))

        return scores


def listmle_loss(
    scores: torch.Tensor,
    relevance: torch.Tensor,
    padding_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-10,
) -> torch.Tensor:
    """ListMLE: negative log-likelihood of the ground-truth permutation.

    Sorts items by true relevance (descending), then computes the likelihood
    of that ordering under the model's scores using Plackett-Luce.

    Args:
        scores: (B, S) predicted scores
        relevance: (B, S) ground truth relevance labels
        padding_mask: (B, S) True for padded positions
        eps: numerical stability constant

    Returns:
        Scalar loss (mean over batch)
    """
    if padding_mask is not None:
        # Set padded relevance to -inf so they sort last
        relevance = relevance.clone()
        relevance[padding_mask] = -float("inf")
        scores = scores.clone()
        scores[padding_mask] = -float("inf")

    # Sort by true relevance (descending) to get ground-truth permutation
    _, indices = relevance.sort(descending=True, dim=-1)
    sorted_scores = scores.gather(1, indices)  # (B, S)

    # Plackett-Luce log-likelihood
    # For each position i, P(item_i | remaining) = exp(s_i) / sum(exp(s_j) for j >= i)
    # Use cumulative logsumexp from the end for numerical stability
    max_score = sorted_scores.max(dim=-1, keepdim=True).values
    shifted = sorted_scores - max_score  # (B, S)

    # Cumulative logsumexp from right to left
    # Note: logcumsumexp is not supported on MPS, so compute on CPU and move back
    orig_device = shifted.device
    cumsums = torch.logcumsumexp(shifted.flip(dims=[-1]).cpu(), dim=-1).flip(dims=[-1]).to(orig_device)

    # Log-likelihood = sum of (score_i - logsumexp(scores[i:]))
    log_likelihood = shifted - cumsums  # (B, S)

    # Only sum over non-padded, valid positions
    if padding_mask is not None:
        sorted_mask = padding_mask.gather(1, indices)
        log_likelihood = log_likelihood.masked_fill(sorted_mask, 0.0)
        # Mean over valid items per sample, then mean over batch
        valid_counts = (~sorted_mask).float().sum(dim=-1).clamp(min=1)
        loss = -(log_likelihood.sum(dim=-1) / valid_counts).mean()
    else:
        loss = -log_likelihood.mean()

    return loss


def ndcg_at_k(scores: np.ndarray, relevance: np.ndarray, k: int = 20) -> float:
    """Compute NDCG@k for a single query (one day's chain)."""
    if len(scores) < 2:
        return float("nan")
    # Sort by predicted scores descending
    ranked_idx = np.argsort(-scores)[:k]
    dcg = np.sum(relevance[ranked_idx] / np.log2(np.arange(2, len(ranked_idx) + 2)))
    # Ideal DCG
    ideal_idx = np.argsort(-relevance)[:k]
    idcg = np.sum(relevance[ideal_idx] / np.log2(np.arange(2, len(ideal_idx) + 2)))
    if idcg < 1e-10:
        return float("nan")
    return float(dcg / idcg)


def get_device() -> torch.device:
    """Select best available device: MPS (Apple Silicon) > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
