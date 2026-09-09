# AME Stage-1 Reproduction (Isaac Lab)

A reproduction of **stage 1** of *"Attention-based map encoding for learning generalized legged
locomotion"* (He, Zhang, Jenelten, Grandia, Bächer, Hutter — *Science Robotics* 10(105), eadv3604,
2025; [arXiv:2506.09588](https://arxiv.org/abs/2506.09588)), on ANYmal-D in Isaac Lab 3.0.

**This repository was built to learn the Isaac Lab workflow.** The environment, the reward set, the
attention encoder and the PPO wiring are all implemented and the pipeline trains end to end, but the
policy has **not** been trained to convergence and nothing here has been deployed on hardware. Treat
it as a working reference implementation, not as a result.

## What "stage 1" means

The paper trains in two stages. Stage 1 learns on six base terrains with *perfect perception* and
privileged observations for both actor and critic. Stage 2 fine-tunes on six harder terrains with
observation noise and per-terrain map drift. **Only stage 1 is implemented here.**

## Architecture

The policy is the paper's attention-based map encoder (Figure 8B):

```
map scan (26 x 16 x 3)  ──► z only ──► CNN (k=5, pad=2, 1→16→61 ch, shape-preserving)
                          │                                  │
                          └────────── xyz coords ────────────┴──► tokens (416 x 64)
                                                                        │  K, V
proprioception (48) ──► Linear(48, 64) ─────────────────────────────────┴──► MHA (d=64, h=16, n=1)
                     │                                                            │
                     └──────────────────────────────────────────────► concat ◄────┘
                                                                        │
                                                              MLP [512, 256, 128] ──► 12 joint targets
```

The CNN and the attention module are **shared between actor and critic** (`share_cnn_encoders=True`),
as the paper describes; the proprioception embedding and the MLP heads are separate.

## Layout

| Path | Contents |
|---|---|
| `source/ame_reproduction/ame_reproduction/networks/` | `AMEModel` (the encoder) and its `RslRlAMEModelCfg` |
| `.../tasks/manager_based/ame_reproduction/configs/quadruped/` | `AmeReproductionEnvCfg` — scene, rewards, terminations, events, curriculum |
| `.../tasks/manager_based/ame_reproduction/terrains/` | The six stage-1 terrains and their generators |
| `.../tasks/manager_based/ame_reproduction/mdp/` | `height_scan_xyz` observation, `joint_torque_limits` reward |
| `.../tasks/manager_based/ame_reproduction/agents/` | `PPORunnerCfg` |
| `AMEModel` subclasses its `MLPModel`. |

The model is bound by string, not import — `class_name = "ame_reproduction.networks.ame_model:AMEModel"`
is resolved by `rsl_rl.utils.resolve_callable` at runner construction.

## What matches the paper

**Rewards** — all 14 stage-1 ANYmal-D terms at the Table 2 weights: linear (5.0) and angular (3.0)
velocity tracking, termination (200), shank collision (1), action rate (5e-3), joint acceleration
(2.5e-7), joint torques (2e-5), joint position limits (1.0), joint velocity limits (1.0), joint
torque limits (0.2), vertical linear velocity (1.0), roll/pitch angular velocity (5e-2), foot contact
forces above 700 N (2.5e-5), foot slippage (0.5).

Two of those terms had no stock Isaac Lab equivalent and are implemented here. `joint_torque_limits`
(`mdp/rewards.py`) is the paper's soft limit at 80% of the actuator effort limit — Isaac Lab's
`applied_torque_limits` measures how much torque the actuator model clipped away instead, which only
fires at 100%. `height_scan_xyz` (`mdp/observations.py`) returns each scan point's full 3-D
coordinate rather than its height alone.

**Terminations** — torso contact with the terrain, or bad torso orientation. The paper's two, exactly.

**Terrain** — the six stage-1 types (stairs, pits, ±8 cm rough ground, pallets, gaps, grid stones),
10 difficulty levels, with the terrain-level curriculum from Rudin et al.

**Observation** — 26 × 16 map scan at 10 cm resolution in the base-yaw frame, each cell carrying its
3-D coordinate as positional encoding (paper: *"heightmap cell coordinates in the base-yaw frame as
positional embedding"*). Policy observation is 1296 = 48 proprioception + 1248 map.

**PPO** — 4096 environments, 24 steps per environment, 5 learning epochs, clip 0.2, entropy 0.005,
γ 0.99, λ 0.95, target KL 0.01, adaptive learning rate, 18000 iterations.

## Deviations and assumptions

Things the paper does not specify, chosen here:

- MLP head `[512, 256, 128]`, ELU throughout, no normalization layers in the CNN.
- Height-scan clamp of ±1.0 m. Rays that hit nothing (gaps) return `+inf` from the sensor, which
  becomes `-inf` after the height subtraction — the same end of the scale as a tall obstacle. They
  are mapped to the **deep** bound instead, so a gap reads as "no support" rather than "wall".
- Velocity command ranges ±1.0, 20 s episodes, 50 Hz control over 200 Hz physics.
- Domain-randomization magnitudes (the paper names torso mass, per-foot friction and pushes, but
  gives no values).

Known deviations from the paper:

- **Actuators.** Isaac Lab has no public ANYmal-D actuator network, so ANYmal-C's LSTM net is used.
- **Joint position limits.** Isaac Lab shrinks the soft limit band about the midpoint of the hard
  limits, while the paper scales the bound from zero. For ANYmal-D's asymmetric HAA joints
  (−45°…+35° left, mirrored right) that differs by 0.5°. Negligible, but not identical.
- **Joint velocity limit** is measured against the actuator's 7.5 rad/s rather than the URDF's 8.5.

## Running it

```bash
python -m pip install -e source/ame_reproduction
python -m pip install -e rsl_rl          # after Isaac Lab: same distribution name, later install wins

python scripts/zero_agent.py --task Template-Ame-Reproduction-v0 --num_envs 32 --headless
python scripts/rsl_rl/train.py --task Template-Ame-Reproduction-v0 --headless
```

Sanity checks worth doing on a fresh machine, in order: the observation space must be 1296; the
model's proprioception embedding must print `Linear(in_features=48, out_features=64)` (if it says
1296, `map_scan_dim` disagrees with the height scanner); `.*SHANK` and `.*FOOT` must each resolve to
four bodies.

## Observed behaviour

Verified to train, then stopped deliberately — there was no hardware target and no reason to pay for
convergence. From iteration 3 on an RTX 8000 (48 GB), 4096 environments:

```
Steps per second: 890          Collection time: 14.79s
Iteration time:   110.40s      Learning time:   95.60s
Mean reward:      -2.17        Mean episode length: 89.44
```

Every reward term is active and signed as expected, and the three terminations all fire. Terrain
level sits at 4.81, which is the mean of the random initial assignment (`max_init_terrain_level=9`),
not learning — 3 iterations is far too early to read anything into it.

The one practical finding: **87% of iteration time is the PPO update, not simulation** (95.6 s versus
14.8 s). On this hardware the attention encoder at a 32768-sample minibatch dominates everything else,
so `num_mini_batches` and GPU choice matter far more than environment count. The paper reports 24 s
per iteration for stage 1 on an A100-40GB; a Turing-generation card is not the tool for this job.

## Not implemented

- **Stage 2** — the six fine-tuning terrains, observation corruption, per-terrain map drift, and the
  two standing rewards.
- **Symmetry augmentation.** Absent from the paper, but the author's tuning guide below places it in the AME-1
  section and says it *"helps improve the motion style a lot"*, with the map mirrored by flipping the
  z values only and keeping x and y. Deliberately left out of the baseline.

## References

- He et al., [*Attention-based map encoding for learning generalized legged locomotion*](https://www.science.org/doi/10.1126/scirobotics.adv3604),
  Science Robotics 10(105), eadv3604 (2025). [arXiv:2506.09588](https://arxiv.org/abs/2506.09588)
- Chong Zhang, [*Attention-based Map Encoding: a Practical Tuning Guide*](https://github.com/zita-ch/techblogs/blob/main/2026-03-28-AME%20Tuning%20Guide.md)
- [ANYbotics/anymal_d_simple_description](https://github.com/ANYbotics/anymal_d_simple_description) — joint limits
- [SII-FUSC/AME_Locomotion](https://github.com/SII-FUSC/AME_Locomotion) — an independent reimplementation on Unitree G1
