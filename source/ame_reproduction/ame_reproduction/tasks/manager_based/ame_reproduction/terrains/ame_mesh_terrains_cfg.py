"""Configurations for the AME terrains generated using the ``trimesh`` library."""

from dataclasses import MISSING

from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.utils.configclass import configclass

from . import ame_mesh_terrains


@configclass
class MeshPalletsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with concentric pallets separated by gaps."""

    function = ame_mesh_terrains.pallets_terrain

    border_width: float = 0.0
    """The width of the border around the terrain (in m). Defaults to 0.0.

    The border is generated at zero height so that the terrain is flush with the neighboring terrains.
    """

    plank_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the pallets (in m).

    The width is interpolated from the maximum to the minimum value as the difficulty increases.
    """

    gap_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the gaps between two consecutive pallets (in m).

    The width of each gap is sampled uniformly between the minimum value and the interpolated value. The
    latter is interpolated from the minimum to the maximum value as the difficulty increases.
    """

    plank_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height deviation of the pallets (in m).

    The height of each pallet is sampled uniformly from the symmetric interval given by the interpolated
    value. The deviation is interpolated from the minimum to the maximum value as the difficulty increases.
    """

    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""
