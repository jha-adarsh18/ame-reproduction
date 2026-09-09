"""Configuration for the attention-based map encoding model."""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg


@configclass
class RslRlAMEModelCfg(RslRlMLPModelCfg):
    """Configuration for :class:`~ame_reproduction.networks.ame_model.AMEModel`.

    Inherits :attr:`hidden_dims`, :attr:`activation`, :attr:`obs_normalization` and
    :attr:`distribution_cfg` from :class:`RslRlMLPModelCfg`, which configure the MLP head. The
    fields below configure the map encoder and mirror the constructor of the model.

    The values follow "Attention-Based Map Encoding for Learning Generalized Legged Locomotion", 
    section "Multi-Head Attention-Based Map Encoding" and "Training".
    """

    class_name: str = "ame_reproduction.networks.ame_model:AMEModel"
    """The model class. Resolved by ``rsl_rl.utils.resolve_callable``, hence the qualified path."""

    map_scan_dim: tuple[int, int, int] = (26, 16, 3)
    """The ``(L, W, 3)`` shape of the map scan. Defaults to ANYmal-D's 26 x 16 grid.

    This must agree with the height scanner: a :class:`~isaaclab.sensors.patterns.GridPatternCfg`
    with ``resolution=0.1`` and ``size=[2.5, 1.5]`` produces 26 x 16 points. The paper uses
    17 x 11 for GR-1.
    """

    mha_dim: int = 64
    """The feature dimension ``d`` of the attention module. Defaults to 64, as in the paper."""

    num_heads: int = 16
    """The number of attention heads ``h``. Defaults to 16, as in the paper.

    Must divide :attr:`mha_dim`, since each head processes ``d / h`` dimensions of the inputs.
    """

    cnn_hidden_channels: int = 16
    """The number of channels of the first CNN layer. Defaults to 16, as in the paper.

    The second layer's channel count is not configurable: the paper fixes it at ``d - 3`` so that
    concatenating the 3-D point coordinates yields tokens of exactly :attr:`mha_dim`.
    """

    cnn_kernel_size: int = 5
    """The kernel size of both CNN layers. Defaults to 5, as in the paper.

    The layers are zero-padded by ``cnn_kernel_size // 2`` to preserve the L x W grid, so this
    should stay odd.
    """

    cnn_activation: str = "elu"
    """The activation of the CNN layers. Defaults to ``"elu"``.

    The paper does not state an activation for the encoder.
    """

    store_attention: bool = False
    """Whether to keep the attention weights on the model for visualization. Defaults to False.

    Enabling this makes the attention module materialize its weight matrix on every forward pass,
    so it should stay off during training and be enabled only for play/analysis.
    """
