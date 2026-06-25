"""HuggingFace modeling code for Orthrus RNA foundation models.

This file is the single source of truth across every Orthrus HF repo:

  antichronology/orthrus-4-track            (Nature Methods, 10M, 4-track)
  antichronology/orthrus-6-track            (Nature Methods, 10M, 6-track)
  antichronology/orthrus-small-6-track      (Nature Methods, 1.3M, 6-track)
  antichronology/orthrus-mlm-6-track        (Nature Methods, 10M, 6-track, MLM head)
  quietflamingo/orthrus-base-4-track        (Pre-publication, ~1M, 4-track)
  quietflamingo/orthrus-large-4-track       (Pre-publication, 10M, 4-track)
  quietflamingo/orthrus-large-6-track       (Pre-publication, 10M, 6-track)

Architecture: unidirectional Mamba backbone (state-space) + optional MLM head.

Every model exposes exactly three inference methods plus the one-hot helper:

  model.representation(x, lengths, channel_last)         -> (B, D)     mean-pooled
  model.representation_unpooled(x, channel_last)         -> (B, L, D)  per-position
  model.predict_tokens(x, lengths, channel_last)         -> (B, L, 4)  MLM logits, MLM models only
  model.seq_to_oh(seq)                                   -> (L, 4)     one-hot encoding

`predict_tokens` raises NotImplementedError on contrastive-only checkpoints
(config.has_mlm_head == False).
"""

import json
import os
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.modules.mamba_simple import Mamba
    from mamba_ssm.modules.block import Block
except ImportError:
    from mamba_ssm.modules.mamba_simple import Block, Mamba

from transformers import PretrainedConfig, PreTrainedModel


HAS_MLP = "mlp_cls" in Block.__init__.__code__.co_varnames


def _return_norm_layer(norm_type: str, num_features: int) -> nn.Module:
    """Construct a normalization layer keyed by string name."""
    if norm_type == "batchnorm":
        return nn.BatchNorm1d(num_features)
    if norm_type == "layernorm":
        return nn.LayerNorm(num_features)
    return nn.Identity()


class _SequenceHead(nn.Module):
    """Per-position MLM head matching the trained Orthrus-MLM checkpoint.

    Structure: fc1 -> norm1 -> ReLU -> fc2 -> norm2 -> ReLU -> fc3 (no bias).
    BatchNorm1d is applied across the feature dimension; input is permuted
    to (B, C, L) for the BN call and back to (B, L, C) after.
    """

    def __init__(
        self,
        input_features: int,
        body: int,
        output_dim: int,
        norm_type: str = "batchnorm",
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_features, body)
        self.norm1 = _return_norm_layer(norm_type, body)
        self.fc2 = nn.Linear(body, body)
        self.norm2 = _return_norm_layer(norm_type, body)
        self.fc3 = nn.Linear(body, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, H)
        x = self.fc1(x)
        x = x.permute(0, 2, 1)
        x = self.norm1(x)
        x = x.permute(0, 2, 1)
        x = F.relu(x)

        x = self.fc2(x)
        x = x.permute(0, 2, 1)
        x = self.norm2(x)
        x = x.permute(0, 2, 1)
        x = F.relu(x)
        return self.fc3(x)


class OrthrusConfig(PretrainedConfig):
    """HuggingFace config for pre-trained Orthrus model."""

    model_type = "orthrus"

    def __init__(
        self,
        n_tracks: int = 4,
        ssm_model_dim: int = 256,
        ssm_n_layers: int = 3,
        has_mlm_head: bool = False,
        mlm_head_body: int = 512,
        mlm_head_dim: int = 4,
        mlm_head_norm_type: str = "batchnorm",
        **kwargs,
    ):
        """Initialize OrthrusConfig.

        Args:
            n_tracks: Number of input data tracks (4 or 6).
            ssm_model_dim: Hidden dimension of Mamba backbone.
            ssm_n_layers: Number of layers in Mamba backbone.
            has_mlm_head: If True, the checkpoint has a trained MLM head and
                supports `predict_tokens`. False for contrastive-only models.
            mlm_head_body: Hidden width of the MLM head MLP.
            mlm_head_dim: Number of output classes (4 for [A, C, G, T]).
            mlm_head_norm_type: Normalization layer type inside the MLM head.
        """
        self.n_tracks = n_tracks
        self.ssm_model_dim = ssm_model_dim
        self.ssm_n_layers = ssm_n_layers
        self.has_mlm_head = has_mlm_head
        self.mlm_head_body = mlm_head_body
        self.mlm_head_dim = mlm_head_dim
        self.mlm_head_norm_type = mlm_head_norm_type
        super().__init__(**kwargs)

    @classmethod
    def init_from_config(cls, config_dir_path: str) -> "OrthrusConfig":
        """Load config from pretraining config files."""
        model_config_path = os.path.join(config_dir_path, "model_config.json")
        data_config_path = os.path.join(config_dir_path, "data_config.json")

        with open(model_config_path, "r") as f:
            model_params = json.load(f)

        if "n_tracks" not in model_params:
            with open(data_config_path, "r") as f:
                data_params = json.load(f)
            n_tracks = data_params["n_tracks"]
        else:
            n_tracks = model_params["n_tracks"]

        return cls(
            n_tracks=n_tracks,
            ssm_model_dim=model_params["ssm_model_dim"],
            ssm_n_layers=model_params["ssm_n_layers"],
            has_mlm_head=model_params.get("has_mlm_head", False),
            mlm_head_body=model_params.get("mlm_head_body", 512),
            mlm_head_dim=model_params.get("mlm_head_dim", 4),
            mlm_head_norm_type=model_params.get("mlm_head_norm_type", "batchnorm"),
        )


class OrthrusPretrainedModel(PreTrainedModel):
    """HuggingFace wrapper for a pretrained Orthrus model."""

    config_class = OrthrusConfig
    base_model_prefix = "orthrus"

    def __init__(self, config: OrthrusConfig, **kwargs):
        super().__init__(config, **kwargs)

        self.config = config
        self.embedding = nn.Linear(config.n_tracks, config.ssm_model_dim)

        self.layers = nn.ModuleList(
            [
                self.create_block(config.ssm_model_dim, layer_idx=i)
                for i in range(config.ssm_n_layers)
            ]
        )

        self.norm_f = nn.LayerNorm(config.ssm_model_dim)

        # MLM head only built when the checkpoint declares it; contrastive
        # checkpoints stay byte-compatible with the original published weights.
        if getattr(config, "has_mlm_head", False):
            self.sequence_head = _SequenceHead(
                input_features=config.ssm_model_dim,
                body=config.mlm_head_body,
                output_dim=config.mlm_head_dim,
                norm_type=config.mlm_head_norm_type,
            )
        else:
            self.sequence_head = None

    def create_block(
        self,
        d_model: int,
        layer_idx: int | None = None,
    ) -> Block:
        """Create a Mamba Block, compatible with old and new mamba_ssm APIs."""
        mix_cls = partial(Mamba, layer_idx=layer_idx)
        norm_cls = nn.LayerNorm

        if HAS_MLP:
            block = Block(
                d_model,
                mix_cls,
                norm_cls=norm_cls,
                mlp_cls=nn.Identity,
            )
        else:
            block = Block(d_model, mix_cls, norm_cls=norm_cls)

        block.layer_idx = layer_idx
        return block

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
        channel_last: bool = False,
    ) -> torch.Tensor:
        """Per-position Orthrus forward pass.

        Args:
            x: Input. Shape (B, C, L) by default, or (B, L, C) if channel_last.
            lengths: Optional unpadded lengths per sequence. Ignored here;
                accepted for signature parity with pooled methods so callers
                can pass the same args interchangeably.
            channel_last: Whether channel dimension is last.

        Returns:
            Per-position hidden states with shape (B, L, D).
        """
        del lengths  # unused; signature parity only.

        if not channel_last:
            x = x.transpose(1, 2)

        hidden_states = self.embedding(x)
        res = None
        for layer in self.layers:
            hidden_states, res = layer(hidden_states, res)

        res = (hidden_states + res) if res is not None else hidden_states
        hidden_states = self.norm_f(res.to(dtype=self.norm_f.weight.dtype))

        return hidden_states

    # ------------------------------------------------------------------
    # Standardized inference interface
    # ------------------------------------------------------------------

    def representation(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        channel_last: bool = False,
    ) -> torch.Tensor:
        """Mean-pooled global representation, masking padding.

        Args:
            x: Input. Shape (B, C, L) or (B, L, C) if channel_last.
            lengths: Unpadded length of each sequence, shape (B,).
            channel_last: Whether channel dimension is last.

        Returns:
            Global representation, shape (B, D).
        """
        out = self.forward(x, channel_last=channel_last)
        return self.mean_unpadded(out, lengths)

    def representation_unpooled(
        self,
        x: torch.Tensor,
        channel_last: bool = False,
    ) -> torch.Tensor:
        """Per-position representation (no pooling).

        Args:
            x: Input. Shape (B, C, L) or (B, L, C) if channel_last.
            channel_last: Whether channel dimension is last.

        Returns:
            Per-position embeddings, shape (B, L, D). Padding positions
            are not masked; callers requiring masking should slice by
            their own length tensor.
        """
        return self.forward(x, channel_last=channel_last)

    def predict_tokens(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
        channel_last: bool = False,
    ) -> torch.Tensor:
        """MLM token logits over [A, C, G, T] at every position.

        Only available on MLM-pretrained checkpoints
        (config.has_mlm_head == True).

        Args:
            x: Input. Shape (B, C, L) or (B, L, C) if channel_last.
                Positions to score should be masked (nucleotide channels
                set to zero) before calling.
            lengths: Optional unpadded lengths, accepted for interface
                parity with `representation`.
            channel_last: Whether channel dimension is last.

        Returns:
            Logits over [A, C, G, T], shape (B, L, 4). Apply log_softmax
            on the last dim for log-probabilities.

        Raises:
            NotImplementedError: if this checkpoint was not trained with
                the masked-LM objective.
        """
        if self.sequence_head is None:
            raise NotImplementedError(
                "predict_tokens requires an MLM-pretrained Orthrus model "
                "(config.has_mlm_head == True). This checkpoint was "
                "trained with the contrastive objective only. Load an "
                "orthrus-mlm-* repo instead."
            )
        hidden = self.forward(x, lengths=lengths, channel_last=channel_last)
        return self.sequence_head(hidden)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def seq_to_oh(self, seq: str) -> torch.Tensor:
        """One-hot encode a nucleotide string with ordering [A, C, G, T].

        Args:
            seq: Sequence over A/C/G/T (and U, treated as T for RNA inputs).
                Other characters become all-zero rows.

        Returns:
            One-hot encoded tensor of shape (L, 4).
        """
        oh = torch.zeros((len(seq), 4), dtype=torch.float32)
        for i, base in enumerate(seq):
            if base == "A":
                oh[i, 0] = 1
            elif base == "C":
                oh[i, 1] = 1
            elif base == "G":
                oh[i, 2] = 1
            elif base == "T" or base == "U":
                oh[i, 3] = 1
        return oh

    def mean_unpadded(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Mean across length dim, masking padding positions."""
        mask = (
            torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None]
        )
        masked_tensor = x * mask.unsqueeze(-1)
        sum_tensor = masked_tensor.sum(dim=1)
        mean_tensor = sum_tensor / lengths.unsqueeze(-1).float()
        return mean_tensor
