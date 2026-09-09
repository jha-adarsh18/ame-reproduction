"""Functions to generate the AME terrains using the ``trimesh`` library."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import trimesh

from isaaclab.terrains.trimesh.utils import make_border

if TYPE_CHECKING:
    from . import ame_mesh_terrains_cfg


def pallets_terrain(
    difficulty: float, cfg: ame_mesh_terrains_cfg.MeshPalletsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with concentric pallets separated by gaps.

    The terrain is a set of concentric rectangular pallets that extend from the border of the terrain
    towards a flat platform at its center. The distance between two consecutive pallets and the height of
    each pallet are sampled independently, so that the robot has to step over gaps of varying width and
    onto surfaces at varying heights.

    As the difficulty increases, the pallets become narrower, the gaps become wider, and the height
    deviation between consecutive pallets becomes larger.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # resolve the terrain configuration
    plank_width = cfg.plank_width_range[1] - difficulty * (cfg.plank_width_range[1] - cfg.plank_width_range[0])
    max_gap_width = cfg.gap_width_range[0] + difficulty * (cfg.gap_width_range[1] - cfg.gap_width_range[0])
    height_range = cfg.plank_height_range[0] + difficulty * (cfg.plank_height_range[1] - cfg.plank_height_range[0])

    # initialize list of meshes
    meshes_list = list()
    # constants for terrain generation
    terrain_height = 1.0
    terrain_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1])

    # generate the border if needed
    # note: the border is kept at zero height so that the terrain is flush with the neighboring terrains
    if cfg.border_width > 0.0:
        border_center = (terrain_center[0], terrain_center[1], -terrain_height / 2.0)
        border_inner_size = (cfg.size[0] - 2.0 * cfg.border_width, cfg.size[1] - 2.0 * cfg.border_width)
        meshes_list += make_border(cfg.size, border_inner_size, terrain_height, border_center)

    # compute the size of the terrain inside the border
    terrain_size = (cfg.size[0] - 2.0 * cfg.border_width, cfg.size[1] - 2.0 * cfg.border_width)

    # sample the distance between the pallets until the span between the border and the platform is filled
    span = 0.5 * (min(terrain_size) - cfg.platform_width)
    gap_widths = list()
    occupied_span = 0.0
    while True:
        gap_width = np.random.uniform(cfg.gap_width_range[0], max_gap_width)
        if occupied_span + plank_width + gap_width > span:
            break
        gap_widths.append(gap_width)
        occupied_span += plank_width + gap_width
    # the outermost pallet absorbs the remaining space so that the platform keeps its configured size
    outer_plank_width = plank_width + span - occupied_span

    # generate the pallets from the border of the terrain towards its center
    outer_size = terrain_size
    for k, gap_width in enumerate(gap_widths):
        # compute the inner size of the pallet
        width = outer_plank_width if k == 0 else plank_width
        inner_size = (outer_size[0] - 2.0 * width, outer_size[1] - 2.0 * width)
        # sample the height of the pallet
        # note: the boxes are extruded downwards so that the top surface lies at the sampled height
        plank_height = np.random.uniform(-height_range, height_range)
        plank_center = (terrain_center[0], terrain_center[1], plank_height - terrain_height / 2.0)
        # generate the pallet as a rectangular border
        meshes_list += make_border(outer_size, inner_size, terrain_height, plank_center)
        # leave a gap before the next pallet
        outer_size = (inner_size[0] - 2.0 * gap_width, inner_size[1] - 2.0 * gap_width)

    # generate the platform at the center of the terrain
    box_dims = (outer_size[0], outer_size[1], terrain_height)
    box_pos = (terrain_center[0], terrain_center[1], -terrain_height / 2.0)
    meshes_list.append(trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos)))

    # specify the origin of the terrain
    origin = np.array([terrain_center[0], terrain_center[1], 0.0])

    return meshes_list, origin
