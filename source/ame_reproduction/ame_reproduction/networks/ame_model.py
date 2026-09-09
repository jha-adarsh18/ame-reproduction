"""Attention-based map encoder for legged locomotion.

Reproduces the policy architecture of "Attention-Based Map Encoding for Learning Generalized
Legged Locomotion", Figure 8B.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from rsl_rl.modules import HiddenState
from rsl_rl.utils import resolve_nn_activation


class AMEModel(MLPModel):
    """Multi-head-attention map encoder feeding an MLP head.

    The observation group is a single flat vector whose tail is the map scan produced by
    :func:`~ame_reproduction...mdp.observations.height_scan_xyz` (``L * W * 3`` values, point-major)
    and whose head is the proprioception. The map's z-channel is embedded by a two-layer CNN,
    the 3-D point coordinates are concatenated back onto those features to give the permutation-
    invariant attention module its positional encoding, and the proprioception embedding is used as
    a single query over the resulting ``L * W`` key/value tokens. The attention output is
    concatenated with the proprioception and passed to the inherited MLP head.

    .. attention::
        This assumes the map scan is the last observation term in the group. In the AME env
        config that holds because ``height_scan`` is declared last in ``PolicyCfg``/``CriticCfg``.
        Reordering the observation terms silently scrambles the input.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        map_scan_dim: tuple[int, int, int] = (26, 16, 3),
        mha_dim: int = 64,
        num_heads: int = 16,
        cnn_hidden_channels: int = 16,
        cnn_kernel_size: int = 5,
        cnn_activation: str = "elu",
        store_attention: bool = False,
        cnns: nn.ModuleDict | dict[str, nn.Module] | None = None,
    ) -> None:
        """Initialize the attention-based map encoding model.

        Args:
            obs: Observation dictionary.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
            obs_set: Observation set to use for this model ("actor" or "critic").
            output_dim: Dimension of the output (number of actions for the actor, 1 for the critic).
            hidden_dims: Hidden dimensions of the MLP head.
            activation: Activation function of the MLP head.
            obs_normalization: Whether to normalize the proprioception before the MLP head. The map
                scan is never normalized: its x/y channels are a fixed grid.
            distribution_cfg: Configuration dictionary for the output distribution.
            map_scan_dim: ``(L, W, 3)`` of the map scan. ``(26, 16, 3)`` for ANYmal-D, ``(17, 11, 3)``
                for GR-1 (paper, "Training").
            mha_dim: Feature dimension ``d`` of the attention module. The paper uses 64.
            num_heads: Number of attention heads ``h``. The paper uses 16.
            cnn_hidden_channels: Channels of the first CNN layer. The paper uses 16.
            cnn_kernel_size: Kernel size of both CNN layers, zero-padded to preserve the grid
                dimensions. The paper uses 5.
            cnn_activation: Activation of the CNN layers. The paper does not state one.
            store_attention: Whether to keep the attention weights on :attr:`attention_weights`
                for visualization. Costs an extra softmax reduction, so keep it off for training.
            cnns: Encoder modules to reuse, supplied by the ``share_cnn_encoders`` hook in
                ``rsl_rl.algorithms.ppo.PPO.construct_algorithm``. If None, new modules are created.
        """
        # -- map geometry (plain attributes, must be set before nn.Module.__init__ runs)
        self.map_length, self.map_width, self.coord_dim = map_scan_dim
        self.num_points = self.map_length * self.map_width
        self.map_size = self.num_points * self.coord_dim
        self.mha_dim = mha_dim
        self.num_heads = num_heads
        self.store_attention = store_attention
        self.attention_weights: torch.Tensor | None = None

        if mha_dim % num_heads != 0:
            raise ValueError(f"mha_dim ({mha_dim}) must be divisible by num_heads ({num_heads}).")

        # -- build (or adopt) the encoder shared between actor and critic
        if cnns is not None:
            missing = {"map_cnn", "mha"} - set(cnns.keys())
            if missing:
                raise ValueError(f"Shared encoder is missing the modules {sorted(missing)}.")
            encoder = cnns
        else:
            # The paper embeds the z-values only; the 3-D coordinates are concatenated afterwards,
            # so the CNN produces d - 3 channels.
            cnn_out_channels = mha_dim - self.coord_dim
            if cnn_out_channels <= 0:
                raise ValueError(f"mha_dim ({mha_dim}) must exceed the coordinate dimension ({self.coord_dim}).")
            padding = cnn_kernel_size // 2
            encoder = {
                "map_cnn": nn.Sequential(
                    nn.Conv2d(1, cnn_hidden_channels, kernel_size=cnn_kernel_size, padding=padding),
                    resolve_nn_activation(cnn_activation),
                    nn.Conv2d(cnn_hidden_channels, cnn_out_channels, kernel_size=cnn_kernel_size, padding=padding),
                ),
                "mha": nn.MultiheadAttention(embed_dim=mha_dim, num_heads=num_heads, batch_first=True),
            }

        # -- initialize the parent MLP model (resolves obs dims and builds the head)
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
        )

        # -- register submodules (only valid once nn.Module.__init__ has run)
        self.cnns = encoder if isinstance(encoder, nn.ModuleDict) else nn.ModuleDict(encoder)
        # The query embedding is deliberately not shared: the paper's actor and critic share the
        # encoder but embed their own (differently sized, in stage 2) proprioception.
        self.proprio_embedding = nn.Linear(self.obs_dim, mha_dim)

        print(f"Encoder CNN: {self.cnns['map_cnn']}")
        print(f"Encoder MHA: {self.cnns['mha']}")
        print(f"Encoder proprioception embedding ({obs_set}): {self.proprio_embedding}")

    """
    Operations.
    """

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Build the MLP input by concatenating proprioception with the attention map encoding."""
        flat_obs = torch.cat([obs[obs_group] for obs_group in self.obs_groups], dim=-1)
        # Proprioception occupies the head of the vector, the map scan the tail.
        proprio = self.obs_normalizer(flat_obs[:, : self.obs_dim])
        map_scan = flat_obs[:, self.obs_dim :]
        return torch.cat([proprio, self._encode_map(map_scan, proprio)], dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update the proprioception normalizer, ignoring the map scan."""
        if self.obs_normalization:
            flat_obs = torch.cat([obs[obs_group] for obs_group in self.obs_groups], dim=-1)
            self.obs_normalizer.update(flat_obs[:, : self.obs_dim])  # type: ignore

    """
    Internal helpers.
    """

    def _encode_map(self, map_scan: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """Encode the flat map scan into a single ``mha_dim`` vector conditioned on proprioception."""
        batch_size = map_scan.shape[0]
        # ``GridPatternCfg(ordering="xy")`` emits torch.meshgrid(..., indexing="xy"), which makes the
        # y-axis (width) the outer axis and the x-axis (length) the inner one. The grid must therefore
        # be reshaped as (W, L), not (L, W), to stay spatially aligned.
        grid = map_scan.reshape(batch_size, self.map_width, self.map_length, self.coord_dim)

        # Point-wise local features: CNN over the heights, then the coordinates concatenated back on.
        heights = grid[..., 2:3].permute(0, 3, 1, 2)
        features = self.cnns["map_cnn"](heights)
        features = features.permute(0, 2, 3, 1).reshape(batch_size, self.num_points, -1)
        coords = map_scan.reshape(batch_size, self.num_points, self.coord_dim)
        local_features = torch.cat([coords, features], dim=-1)

        # Proprioception conditions the map encoding as a single (n = 1) attention query.
        query = self.proprio_embedding(proprio).unsqueeze(1)
        encoding, weights = self.cnns["mha"](
            query, local_features, local_features, need_weights=self.store_attention
        )
        if self.store_attention:
            self.attention_weights = weights.detach()
        return encoding.squeeze(1)

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        """Select active observation groups and split their width into proprioception and map scan."""
        active_obs_groups = obs_groups[obs_set]
        total_dim = 0
        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    f"The AME model only supports 1D observations, got shape {obs[obs_group].shape}"
                    f" for '{obs_group}'."
                )
            total_dim += obs[obs_group].shape[-1]

        proprio_dim = total_dim - self.map_size
        if proprio_dim <= 0:
            raise ValueError(
                f"Observation set '{obs_set}' has width {total_dim}, which leaves no room for"
                f" proprioception after the {self.map_size}-value map scan"
                f" ({self.map_length} x {self.map_width} x {self.coord_dim})."
            )
        # The parent stores this as ``self.obs_dim`` and sizes the observation normalizer with it.
        return active_obs_groups, proprio_dim

    def _get_latent_dim(self) -> int:
        """Return the latent dimensionality consumed by the MLP head."""
        return self.obs_dim + self.mha_dim
