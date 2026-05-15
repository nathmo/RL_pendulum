# Inverted Pendulum RL Plan

## Goal
Train a policy that can:

1. Swing the pendulum from arbitrary initial states into the upright region.
2. Keep it balanced there.
3. Be robust to a real hardware loop that has jitter, small latency, sensor noise, and occasional dropped commands.
4. Export cleanly to ONNX and run on a Raspberry Pi that talks to a moteus controller over CAN.

The system should be explicit, visually inspectable, and easy to iterate on.

## Given Hardware Assumptions

- Direct-drive motor.
- Torque limit: `+- 1.0 N·m`.
- Rod: massless.
- Tip mass: `0.03 kg`.
- Rod length: `0.75 m`.
- Position is reported in absolute turns and does not wrap at each revolution.
- Available telemetry: position, velocity, torque.
- Control input: torque command.
- Hardware control loop target: `50 Hz` from the Raspberry Pi.
- Real controller timing is jittery, so training must model a few milliseconds of variability in command application.

## Core Design Choice

Use a single PPO policy trained in MuJoCo, with domain randomization and a short observation history.

Why this choice:

- PPO is stable, easy to debug, and works well for continuous control with shaped rewards.
- MuJoCo gives fast and physically credible pendulum dynamics for training.
- A short history window helps the policy infer jitter, delay, and velocity trends without requiring a recurrent model at the start.
- One policy is simpler than a two-policy swing-up + balance stack, but still supports both behaviors if the reward and initial-state distribution are designed correctly.

## Training Loop Overview

The policy will learn from a simulation that runs the plant at a higher internal rate than the policy rate.

Suggested timing model:

- Physics integration rate: `1000 Hz`.
- Policy decision rate: configurable, default `50 Hz`.
- One policy action is held for the duration of the control interval, but that interval is jittered during training.

This matches the real setup better than assuming perfectly periodic actuation.

## Observation Design

Use the current sample plus the last 4 samples, for a total history length of 5.

For each sample, include:

- Position in turns.
- Velocity in turns per second.
- Applied or last commanded torque in N·m.

Explicit tensor layout:

- Raw per-step observation: `o_t ∈ R^3`.
- History buffer: `O_t ∈ R^{5×3}` where rows are `[o_{t-4}, o_{t-3}, o_{t-2}, o_{t-1}, o_t]`.
- Preprocessed per-step feature vector: `x_t ∈ R^4 = [sin(theta_t), cos(theta_t), vel_norm_t, torque_norm_t]`.
- Preprocessed history tensor: `X_t ∈ R^{5×4}`.
- Flattened policy input: `z_t ∈ R^{20}`.

Recommended preprocessing before the policy sees the data:

- Convert turns to radians internally for learning convenience.
- Normalize angles with `sin(angle)` and `cos(angle)` so the model does not have to learn periodicity from scratch.
- Scale velocity and torque to roughly `[-1, 1]` ranges.
- Keep the raw-turn representation available in the environment and the hardware bridge so that logging stays interpretable.

Why this matters:

- Absolute turns are fine for telemetry, but raw angle values grow without bound and are awkward for the network.
- `sin` and `cos` avoid discontinuities and make swing-up around `2π` easier.
- A 5-sample stack lets the policy see short-term motion and infer delay/jitter effects.

Recommended normalization targets:

- `theta_t = 2π * turns_t`.
- `vel_norm_t = clip(velocity_turns_per_s / 2.0, -1, 1)` where `2.0 turns/s = 120 RPM`.
- `torque_norm_t = clip(torque_nm / 1.0, -1, 1)`.

If you want the policy to see timing directly, an optional extra scalar can be appended per sample:

- `dt_norm_t ∈ R`.

That would change the per-step vector to `R^5` and the flattened input to `R^25`. The default plan below does not include it, because the 5-step history already carries most of the timing information.

## Action Design

The policy outputs one scalar in `[-1, 1]`.

That scalar is mapped linearly to a torque command in `[-1.0, 1.0] N·m`.

Additional runtime logic:

- Clamp torque to the hardware limit.
- Clamp simulated speed to `+- 120 RPM` for safety and to keep the motion inside realistic bounds.

## Reward Design

Use a dense shaped reward so the agent can discover swing-up without sparse exploration problems.

Recommended reward terms:

- Upright alignment reward: high when the pendulum is near the inverted position.
- Angular velocity penalty: discourage fast oscillation near the target.
- Torque penalty: prefer efficient control.
- Torque change penalty: reduce chattering.
- Small success bonus when the pendulum stays near upright with low velocity for a sustained period.

A practical structure is:

- Early learning: reward energy gain and movement toward upright.
- Mid learning: reward proximity to the upright angle more strongly.
- Late learning: emphasize balance precision and smooth torque.

This can be implemented as a single reward with fixed weights, or with a mild curriculum that gradually increases the upright/balance term.

## Termination and Episode Setup

Episode length:

- Default `30 s`.
- Must be configurable, because faster experiments may want shorter episodes.

Initial state distribution:

- Position randomized in a wide range, for example `[-5, 5]` turns.
- Initial velocity randomized as well, so the policy learns from both moving and static starts.

Termination rules:

- Keep episodes running unless a configured safety boundary is exceeded.
- Since the real mechanism can spin freely, avoid hard termination on angle alone.
- Use termination mainly for numerical blow-up, invalid sim state, or extreme velocity beyond the configured `120 RPM` limit.

## Jitter and Delay Randomization

This is important for sim-to-real.

During training, randomize:

- Command application delay by a few milliseconds.
- Control-period jitter around the nominal policy period.
- Small packet drop probability.
- Observation noise.
- Latency between the plant state and what the policy sees.

Suggested defaults:

- Jitter mean near zero.
- Jitter standard deviation tuned to match the Pi and CAN path.
- Rare but small command-delay spikes, capped to a few milliseconds.

The exact values should stay configurable from the training script so you can retrain faster or slower loops without rewriting the environment.

## Domain Randomization

Randomize the following per episode:

- Mass: around `+- 20%`.
- Length: around `+- 5%`.
- Friction: wide range, since none was measured.
- Sensor noise: tunable Gaussian noise.
- Timing delay and jitter: small random variation.
- Optional small variation in gravity if you want extra robustness across setups.

Why this matters:

- The real plant will never match the simulation exactly.
- Friction and latency are often the biggest reasons a controller that works in sim fails on hardware.
- Randomizing them early forces the policy to learn behavior that is less brittle.

## Simulator Choice

Use MuJoCo for training.

Why MuJoCo:

- Fast enough for PPO training on CPU.
- Accurate enough for a simple pendulum and direct-drive torque control.
- Better long-term fit for repeated sim-to-real experiments than a throwaway toy simulator.

For visualization, use a separate `visualize.py` script that loads the same environment and renders the policy rollout in 3D.

## Training Stack

Recommended stack:

- Python + Gymnasium environment.
- MuJoCo physics.
- Stable-Baselines3 PPO.
- ONNX export for deployment.
- ONNX Runtime on the Raspberry Pi.

## Configuration File

Keep all tunable parameters in one reusable config module, for example `config.py`.

This config should contain:

- Physics constants.
- Observation and action scaling.
- Reward weights.
- Jitter and latency distributions.
- Domain randomization ranges.
- Episode length and control rate.
- PPO hyperparameters.
- Export settings.

Why this matters:

- It keeps `train.py`, `visualize.py`, and `run.py` synchronized.
- It lets you retune reward and jitter without rewriting code.
- It makes the same ONNX policy usable on Windows and Raspberry Pi because the preprocessing stays identical.

## Network Architecture

Use a compact feedforward actor-critic network with a shared encoder.

Dimensions and layers:

- Input tensor: `z_t ∈ R^{20}`.
- Shared encoder layer 1: `Linear(20 -> 128)` followed by `Tanh`.
- Shared encoder layer 2: `Linear(128 -> 128)` followed by `Tanh`.
- Policy trunk: `Linear(128 -> 64)` followed by `Tanh`.
- Value trunk: `Linear(128 -> 64)` followed by `Tanh`.
- Policy head: `Linear(64 -> 1)`.
- Value head: `Linear(64 -> 1)`.

Tensor flow:

- `z_t ∈ R^{20}`.
- Shared hidden representation `h_t ∈ R^{128}`.
- Policy latent `p_t ∈ R^{64}`.
- Value latent `v_t ∈ R^{64}`.
- Policy output `a_t ∈ R^{1}` in `[-1, 1]` after `tanh`.
- Value output `V_t ∈ R^{1}`.

Why this shape:

- The input is small, so a shallow MLP is enough and exports cleanly to ONNX.
- `128 -> 128` hidden layers are large enough to absorb randomization and jitter, but still light for Raspberry Pi inference.
- Separate policy and value heads keep PPO training straightforward.

If training under Stable-Baselines3, the equivalent policy config is:

- `net_arch = dict(pi=[64], vf=[64])`
- shared feature extractor output size `128`

That preserves the same effective tensor sizes while fitting SB3's actor-critic structure.

Why this stack:

- It is straightforward to debug.
- It has a clean training/export/deploy path.
- It avoids custom infrastructure until the task is already working.

## `train.py`

Purpose:

- Run headless training on the server (ubuntu based).
- Create the environment with randomization and jitter.
- Train PPO.
- Save checkpoints.
- Export the final policy to ONNX.

Expected behavior:

- Configurable via CLI flags.
- No GUI.
- Deterministic seed support.
- Save model, normalization stats, and export metadata.

Artifacts to produce:

- PPO checkpoint.
- Final ONNX model.
- A JSON or YAML config snapshot describing the training run.

## `visualize.py`

Purpose:

- Load the exported policy.
- Run the same environment with 3D rendering.
- Show telemetry live so you can inspect what the policy is doing.

Platform note:

- This script should be easy to run on Windows, with no Linux-only assumptions in the visualization path.
- The ONNX policy must load with the same runtime and preprocessing on Windows and on Raspberry Pi.

Recommended features:

- Render the pendulum state.
- Plot position, velocity, torque, and reward over time.
- Display the applied jitter and any latency samples.
- Optionally let you pause, reset, and switch seeds.

This script should be the main sanity-check tool before hardware testing.

## `run.py`

Purpose:

- Run the exported ONNX policy on the Raspberry Pi.
- Read live moteus telemetry.
- Apply the same preprocessing used in training.
- Send a torque command back to the controller.

Important runtime notes:

- Keep the control loop rate configurable, default `50 Hz`.
- Use the same history window as training.
- Match the exact normalization and stacking used by the simulator policy.
- Keep the command path small and deterministic.
- Log timings so you can measure actual loop jitter.

The current draft is fine as a hardware bridge, but it should eventually share preprocessing code with training and visualization so there is no train/deploy mismatch.

## Sim-to-Real Strategy

Recommended rollout plan:

1. Train in simulation with moderate randomization.
2. Validate in `visualize.py` and make sure the swing-up behavior is qualitatively correct.
3. Run on hardware with conservative logging and low duty-cycle tests.
4. Compare hardware trajectories against simulation.
5. Expand randomization only after the first policy behaves reasonably.

This is more reliable than trying to train the hardest possible policy from the start.

## Safety Approach

You said the motor can spin freely and there is no emergency kill requirement.

Even so, keep software limits:

- Torque clamp at `+- 1.0 N·m`.
- Speed clamp at `+- 120 RPM` in sim and monitoring logic.
- Sanity checks for NaNs and invalid telemetry.

## What Will Be Implemented

The first implementation pass should create:

- `train.py` for headless PPO training and ONNX export.
- `visualize.py` for 3D rollout inspection.
- `run.py` for Raspberry Pi deployment with moteus.
- A shared environment module so training, visualization, and deployment use the same observation and timing logic.
- A configuration file or dataclass for all tunable parameters.

## Open Implementation Questions

These are the remaining choices to lock down before coding:

- Exact reward weights.
- Exact jitter and latency distributions.
- The preferred file layout for shared code.

Decisions already fixed:

- Observation stack length: `5`.
- Per-step feature vector size: `4`.
- Flattened policy input size: `20`.
- Policy output size: `1`.
- Network hidden width: `128`.
- Policy/value head width: `64`.

Default tuning policy:

- Use sensible starting reward weights in the config, but keep them editable.
- Use the current jitter proposal as a default, but keep its mean, std, and max spike configurable.
- Keep the ONNX export and preprocessing identical across Windows and Raspberry Pi.

Once those are fixed, the code can be built without rework.
