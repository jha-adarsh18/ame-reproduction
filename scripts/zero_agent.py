# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

import argparse
import contextlib
import os
import sys

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import (
    add_launcher_args,
    launch_simulation,
    resolve_task_config,
    setup_preset_cli,
)

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video of the rollout.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--video_folder", type=str, default="logs/zero_agent/videos", help="Directory to write recorded videos to."
)
parser.add_argument("--video_env", type=int, default=0, help="Environment index the recording camera follows.")
# append AppLauncher cli args
add_launcher_args(parser)
# simple agents should open Kit visualizer by default
parser.set_defaults(visualizer=["kit"])
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

# recording requires the rendering pipeline, which is off by default on headless machines
if args_cli.video:
    args_cli.enable_cameras = True

import ame_reproduction.tasks  # noqa: F401
MAX_STEPS = 100


def main():
    """Zero actions agent with Isaac Lab environment."""

    torch.manual_seed(42)

    # parse configuration via Hydra (supports preset selection, e.g. env.sim.physics=newton_mjwarp)
    env_cfg, _ = resolve_task_config(args_cli.task, "")

    with launch_simulation(env_cfg, args_cli):
        # override with CLI arguments
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        if args_cli.disable_fabric:
            env_cfg.sim.use_fabric = False

        # follow a robot with the camera, otherwise generated terrain puts the
        # subject far away from the default viewer pose at the world origin
        if args_cli.video:
            env_cfg.viewer.origin_type = "asset_root"
            env_cfg.viewer.asset_name = "robot"
            env_cfg.viewer.env_index = args_cli.video_env
            env_cfg.viewer.eye = (2.5, 2.5, 1.5)
            env_cfg.viewer.lookat = (0.0, 0.0, 0.0)

        # create environment
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        # wrap for video recording
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.abspath(args_cli.video_folder),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording video of the zero-action rollout.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        # print info (this is vectorized environment)
        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")
        # reset environment
        env.reset()
        # simulate environment
        # keep running while any visualizer is open, otherwise fall back to a step budget
        sim = env.unwrapped.sim
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        # headless runs have no window to close, and recordings should stop once the clip is complete
        bounded = args_cli.video or not sim.visualizers
        max_steps = args_cli.video_length if args_cli.video else MAX_STEPS
        step_count = 0
        while True:
            if sim.visualizers:
                # visualizer mode: run until the visualizer window is closed
                if not any(v.is_running() and not v.is_closed for v in sim.visualizers):
                    break
            if bounded and step_count >= max_steps:
                break
            # run everything in inference mode
            with torch.inference_mode():
                # apply actions
                env.step(actions)
            step_count += 1
        # close the simulator
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
