"""Functions to generate the AME terrains as height fields."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from isaaclab.terrains.height_field.utils import height_field_to_mesh

if TYPE_CHECKING:
    from . import ame_hf_terrains_cfg


@height_field_to_mesh
def rough_terrain(difficulty: float, cfg: ame_hf_terrains_cfg.HfRoughTerrainCfg) -> np.ndarray:
    """Generate a terrain with height sampled uniformly around the zero level.

    The terrain is a ground with a uniform height noise. The amplitude of the noise is symmetric around
    the zero level, so that the ground deviates by at most the sampled amplitude in either direction.

    As the difficulty increases, the amplitude of the noise becomes larger.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.

    Raises:
        ValueError: When the downsampled scale is smaller than the horizontal scale.
    """
    # check parameters
    # -- horizontal scale
    if cfg.downsampled_scale is None:
        cfg.downsampled_scale = cfg.horizontal_scale
    elif cfg.downsampled_scale < cfg.horizontal_scale:
        raise ValueError(
            "Downsampled scale must be larger than or equal to the horizontal scale:"
            f" {cfg.downsampled_scale} < {cfg.horizontal_scale}."
        )

    # resolve the terrain configuration
    noise_height = cfg.noise_height_range[0] + difficulty * (cfg.noise_height_range[1] - cfg.noise_height_range[0])

    # switch parameters to discrete units
    # -- horizontal scale
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- downsampled scale
    width_downsampled = int(cfg.size[0] / cfg.downsampled_scale)
    length_downsampled = int(cfg.size[1] / cfg.downsampled_scale)
    # -- height
    # note: the noise is symmetric around the zero level, i.e. the ground deviates in both directions
    height_min = int(-noise_height / cfg.vertical_scale)
    height_max = int(noise_height / cfg.vertical_scale)
    height_step = int(cfg.noise_step / cfg.vertical_scale)

    # create range of heights possible
    height_range = np.arange(height_min, height_max + height_step, height_step)
    # sample heights randomly from the range along a grid
    height_field_downsampled = np.random.choice(height_range, size=(width_downsampled, length_downsampled))
    # create interpolation function for the sampled heights
    x = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_downsampled)
    y = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_downsampled)

    # note: imported lazily so that the module can be imported before the simulation app starts
    import scipy.interpolate as interpolate
    func = interpolate.RectBivariateSpline(x, y, height_field_downsampled)

    # interpolate the sampled heights to obtain the height field
    x_upsampled = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_pixels)
    y_upsampled = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_pixels)
    z_upsampled = func(x_upsampled, y_upsampled)
    # round off the interpolated heights to the nearest vertical step
    return np.rint(z_upsampled).astype(np.int16)
