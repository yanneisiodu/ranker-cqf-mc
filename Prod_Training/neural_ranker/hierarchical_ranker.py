"""Hierarchical attention ranker for options.

Architecture:
    1. Group options by cluster (strike bucket × expiry bucket × type)
    2. Intra-cluster attention: options attend to neighbors within same cluster
    3. Cluster summary: pool each cluster into a single representation
    4. Inter-cluster attention: cluster summaries attend to each other
    5. Broadcast back: each option gets its cluster's global context
    6. Ranking head: score each option using local + global features

This is more efficient than flat attention (O(n²)) because:
    - Intra-cluster: O(c × k²) where k = options per cluster (~100-200)
    - Inter-cluster: O(c²) where c = number of clusters (~30-50)
    - Total: O(c × k² + c²) << O(n²) for n = 6,500

And more expressive because it explicitly models the chain structure:
    - Within a cluster: "is this 580 call mispriced vs the 582 call?"
    - Across clusters: "are short-dated puts more attractive than long-dated calls today?"
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_ranker import NeuralRankerConfig, get_device, listmle_loss, ndcg_at_k


@dataclass(frozen=True)
class HierarchicalConfig:
    input_dim: int = 63
    embed_dim: int = 256
    n_heads: int = 4
    n_intra_layers: int = 2
    n_inter_layers: int = 1
    dropout: float = 0.25
    mlp_hidden: int = 512
    # Clustering
    n_strike_buckets: int = 10  # deciles of moneyness
    n_expiry_buckets: int = 4   # weekly, monthly, quarterly, leaps
    # These get multiplied: 10 × 4 × 2 (call/put) = 80 max clusters
    learning_rate: float = 4.717e-4
    weight_decay: float = 2.267e-3
    warmup_epochs: int = 1
    feature_noise: float = 0.06
    epochs: int = 50
    patience: int = 8
    seed: int = 42

    @classmethod
    def from_config(cls, config: Dict) -> "HierarchicalConfig":
        cfg = config.get("hierarchical_ranker", config.get("neural_ranker", {}))
        return cls(**{k: cfg[k] for k in cfg if k in cls.__dataclass_fields__})


def assign_clusters(
    features: np.ndarray,
    feature_names: List[str],
    n_strike_buckets: int = 10,
    n_expiry_buckets: int = 4,
) -> np.ndarray:
    """Assign each option to a cluster based on moneyness, expiry, and type.

    Returns cluster_ids: (n_options,) integer array.
    """
    n = features.shape[0]

    # Find feature indices
    def get_idx(name):
        try:
            return feature_names.index(name)
        except ValueError:
            return None

    # Moneyness buckets
    moneyness_idx = get_idx("moneyness")
    if moneyness_idx is not None:
        moneyness = features[:, moneyness_idx]
        strike_bucket = np.clip(
            np.digitize(moneyness, np.linspace(moneyness.min() - 1e-8, moneyness.max() + 1e-8, n_strike_buckets + 1)) - 1,
            0, n_strike_buckets - 1,
        )
    else:
        strike_bucket = np.zeros(n, dtype=int)

    # Expiry buckets (days_to_exp: 0-7, 7-30, 30-90, 90+)
    dte_idx = get_idx("days_to_exp")
    if dte_idx is not None:
        dte = features[:, dte_idx]
        expiry_bucket = np.digitize(dte, [7, 30, 90]) # 0=weekly, 1=monthly, 2=quarterly, 3=leaps
    else:
        expiry_bucket = np.zeros(n, dtype=int)

    # Type bucket (call=0, put=1)
    type_idx = get_idx("type_numeric")
    if type_idx is not None:
        type_bucket = (features[:, type_idx] < 0.5).astype(int)  # put=1 if type_numeric=0
    else:
        type_bucket = np.zeros(n, dtype=int)

    # Combine: cluster_id = strike * (n_expiry * 2) + expiry * 2 + type
    cluster_ids = strike_bucket * (n_expiry_buckets * 2) + expiry_bucket * 2 + type_bucket
    return cluster_ids.astype(np.int64)


class IntraClusterAttention(nn.Module):
    """Attention within a cluster — options attend to their neighbors."""

    def __init__(self, embed_dim: int, n_heads: int, n_layers: int, dropout: float, mlp_hidden: int):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=mlp_hidden,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, enable_nested_tensor=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encoder(x, src_key_padding_mask=mask)


class InterClusterAttention(nn.Module):
    """Attention across cluster summaries — global context."""

    def __init__(self, embed_dim: int, n_heads: int, n_layers: int, dropout: float, mlp_hidden: int):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=mlp_hidden,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, enable_nested_tensor=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encoder(x, src_key_padding_mask=mask)


class HierarchicalChainRanker(nn.Module):
    """Hierarchical attention ranker.

    Step 1: Encode each option's features
    Step 2: Group by cluster, run intra-cluster attention
    Step 3: Pool clusters, run inter-cluster attention
    Step 4: Broadcast global context back, score each option
    """

    def __init__(self, config: HierarchicalConfig):
        super().__init__()
        self.config = config

        # Per-option encoder
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.mlp_hidden),
            nn.LayerNorm(config.mlp_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_hidden, config.embed_dim),
            nn.LayerNorm(config.embed_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Intra-cluster attention (local context)
        self.intra_attention = IntraClusterAttention(
            config.embed_dim, config.n_heads, config.n_intra_layers,
            config.dropout, config.mlp_hidden,
        )

        # Inter-cluster attention (global context)
        self.inter_attention = InterClusterAttention(
            config.embed_dim, config.n_heads, config.n_inter_layers,
            config.dropout, config.mlp_hidden,
        )

        # Fusion: combine local (intra) + global (inter) representations
        self.fusion = nn.Sequential(
            nn.Linear(config.embed_dim * 2, config.embed_dim),
            nn.LayerNorm(config.embed_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Ranking head
        self.ranking_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim // 2, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        cluster_ids: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            features: (B, S, input_dim)
            cluster_ids: (B, S) integer cluster assignments
            padding_mask: (B, S) True for padded positions

        Returns:
            scores: (B, S) ranking scores
        """
        B, S, D = features.shape
        device = features.device

        # Step 1: Encode all options
        embeddings = self.encoder(features)  # (B, S, embed_dim)

        # Step 2: Intra-cluster attention
        # Process each cluster independently for each batch
        intra_output = torch.zeros_like(embeddings)

        for b in range(B):
            valid = ~padding_mask[b] if padding_mask is not None else torch.ones(S, dtype=torch.bool, device=device)
            cids = cluster_ids[b][valid]
            emb = embeddings[b][valid]

            unique_clusters = cids.unique()
            for cid in unique_clusters:
                mask = cids == cid
                cluster_emb = emb[mask].unsqueeze(0)  # (1, k, embed_dim)
                if cluster_emb.shape[1] < 2:
                    intra_output[b][valid][mask] = cluster_emb.squeeze(0)
                    continue
                attended = self.intra_attention(cluster_emb)  # (1, k, embed_dim)
                intra_output[b][valid][mask] = attended.squeeze(0)

        # Step 3: Pool clusters and run inter-cluster attention
        # Collect cluster summaries (mean pool)
        max_clusters = self.config.n_strike_buckets * self.config.n_expiry_buckets * 2
        cluster_summaries = torch.zeros(B, max_clusters, self.config.embed_dim, device=device)
        cluster_mask = torch.ones(B, max_clusters, dtype=torch.bool, device=device)

        for b in range(B):
            valid = ~padding_mask[b] if padding_mask is not None else torch.ones(S, dtype=torch.bool, device=device)
            cids = cluster_ids[b][valid]
            for cid in cids.unique():
                mask = cids == cid
                cluster_summaries[b, cid] = intra_output[b][valid][mask].mean(dim=0)
                cluster_mask[b, cid] = False

        # Inter-cluster attention
        global_context = self.inter_attention(cluster_summaries, mask=cluster_mask)  # (B, max_clusters, embed_dim)

        # Step 4: Broadcast global context back to each option
        global_per_option = torch.zeros_like(embeddings)
        for b in range(B):
            valid = ~padding_mask[b] if padding_mask is not None else torch.ones(S, dtype=torch.bool, device=device)
            cids = cluster_ids[b][valid]
            for cid in cids.unique():
                mask = cids == cid
                global_per_option[b][valid][mask] = global_context[b, cid].unsqueeze(0).expand(mask.sum(), -1)

        # Step 5: Fuse local + global
        fused = self.fusion(torch.cat([intra_output, global_per_option], dim=-1))  # (B, S, embed_dim)

        # Step 6: Score
        scores = self.ranking_head(fused).squeeze(-1)  # (B, S)

        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask, float("-inf"))

        return scores
