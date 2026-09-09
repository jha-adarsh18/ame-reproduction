"""
Configuration for stage 1 training as in the paper. These include:
    - stairs
    - pits: stages with a certain height
    - rough ground with ± 8 cm height noise
    - pallets: horizontally placed pallets with random distances and heights
    - gaps
    - grid stones: randomly placed stepping stones with random height differences
"""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

from .ame_mesh_terrains_cfg import *
from .ame_hf_terrains_cfg import *

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    seed=42,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pits": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "rough_ground": HfRoughTerrainCfg(
            proportion=0.2,
            noise_height_range=(0.02, 0.08), # ± 8 cm at difficulty = 1.0
            noise_step=0.02,
            border_width=0.25,
        ),
        "pallets": MeshPalletsTerrainCfg(
            proportion=0.2,
            plank_width_range=(0.25, 0.60),
            gap_width_range=(0.10, 0.30),
            plank_height_range=(0.02, 0.10),
            platform_width=2.0,
            border_width=0.5,
        ),
        "gaps": terrain_gen.MeshGapTerrainCfg(
            proportion=0.2,
            platform_width=2.0,
            gap_width_range=(0.10, 0.50),
        ),
        "grid_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.2,
            platform_width=2.0,
            stone_height_max=0.05,
            stone_distance_range=(0.10,0.30),
            stone_width_range=(0.30, 0.60),
            border_width=0.5,
        ),
    },
)