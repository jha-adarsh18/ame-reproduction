"""Configurations for the AME terrains generated as height fields."""

from dataclasses import MISSING

from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.utils.configclass import configclass

from . import ame_hf_terrains


@configclass
class HfRoughTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a rough height field terrain with a difficulty-dependent amplitude."""

    function = ame_hf_terrains.rough_terrain

    noise_height_range: tuple[float, float] = MISSING
    """The minimum and maximum amplitude of the height noise (in m).

    The heights are sampled symmetrically around the zero level, i.e. within the interval given by the
    negated and the positive amplitude. The amplitude is interpolated from the minimum to the maximum
    value as the difficulty increases.
    """

    noise_step: float = MISSING
    """The minimum height (in m) change between two points."""

    downsampled_scale: float | None = None
    """The distance between two randomly sampled points on the terrain. Defaults to None,
    in which case the :obj:`horizontal scale` is used.

    The heights are sampled at this resolution and interpolation is performed for intermediate points.
    This must be larger than or equal to the :obj:`horizontal scale`.
    """
