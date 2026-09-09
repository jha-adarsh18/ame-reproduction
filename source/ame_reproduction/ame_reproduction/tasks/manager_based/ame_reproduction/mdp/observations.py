"""Common functions that can be used to define observations for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to
specify the observation function and its parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.sensors import RayCaster


def height_scan_xyz(env: ManagerBasedEnv,
                    sensor_cfg: SceneEntityCfg,
                    offset: float = 0.5, 
                    clip_height: tuple[float, float] = (-1.0, 1.0)) -> torch.Tensor:
    """Vector map scans from the given sensor w.r.t. the sensor's frame.

    Same as :func:`isaaclab.envs.mdp.height_scan`, but returns the full 3-D coordinate of each
    scan point (a total of L x W x 3 values) instead of its height alone. This is the exteroceptive
    observation of "Attention-Based Map Encoding for Learning Generalized Legged Locomotion"
    (arXiv:2506.09588), whose encoder concatenates the coordinates back onto the CNN features to
    give the permutation-invariant attention module its positional encoding.

    The provided offset (Defaults to 0.5) is subtracted from the height values.

    .. attention:: term-level clip or noise should not be set on this term as it would end up
    corrupting x and y coordinates too. 
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # (x, y) of each scan point in the sensor frame, taken from the ray pattern
    xy = sensor.ray_starts.torch[..., :2]
    # height scan: height = sensor_height - hit_point_z - offset
    z = sensor.data.pos_w.torch[:, 2].unsqueeze(1) - sensor.data.ray_hits_w.torch[..., 2] - offset
    # rays that hit nothing keep the +inf the sensor pre-fills them with, which lands here as -inf.
    # a missing return means "no support", so it must read as the deepest value, not the highest.
    z = torch.nan_to_num(z, nan=clip_height[1], posinf=clip_height[1], neginf=clip_height[1])
    z = z.clamp(clip_height[0], clip_height[1])

    return torch.cat([xy, z.unsqueeze(-1)], dim=-1).flatten(start_dim=1)
