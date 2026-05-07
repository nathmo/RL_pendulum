# Inverted Pendulum RL

This project trains a policy for a direct-drive inverted pendulum with a 0.5 m massless rod and a 0.5 kg tip mass, using a maximum torque of 1.0 N·m.

The implementation plan is documented in [PENDULUM_RL_PLAN.md](PENDULUM_RL_PLAN.md).

## What is implemented

- `pendulum_rl/config.py`: all tunable parameters.
- `pendulum_rl/preprocess.py`: shared observation building and scaling.
- `pendulum_rl/env.py`: MuJoCo swing-up environment with jitter and randomization.
- `pendulum_rl/model.py`: SB3 feature extractor and ONNX export helper.
- `pendulum_rl/runtime.py`: ONNX policy loader usable on Windows and Raspberry Pi.
- `train.py`: headless PPO training and ONNX export.
- `visualize.py`: Windows-friendly 3D simulation viewer.
- `run.py`: hardware runner for moteus, plus a dry-run simulator mode.

## Design choices

- Training simulator: MuJoCo.
- RL algorithm: PPO.
- Policy type: deterministic at runtime, stochastic during training.
- Control rate: default 50 Hz, configurable.
- Observation history: last 5 samples.
- Policy input size: 20 features after preprocessing.
- Policy output size: 1 normalized torque command in `[-1, 1]`.
- Network: compact MLP actor-critic with a 128-unit shared encoder and 64-unit policy/value heads.

## Tunable defaults

The following values are sensible defaults, but they stay configurable in `pendulum_rl/config.py`:

- Reward weights: defaults are set, but tune them later.
- Jitter: default small mean-zero jitter with a few milliseconds of variation.
- Observation noise: tunable Gaussian noise.
- Domain randomization ranges: tunable per episode.
- Episode length: default 30 seconds, configurable.
- Control rate: default 50 Hz, configurable.

## Portability

The same exported ONNX policy runs on:

- Windows, for visualization and dry-run testing.
- Raspberry Pi, for hardware deployment with moteus.

The preprocessing, observation stacking, normalization, and action scaling are shared between training, visualization, and runtime.

## Install

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

If you prefer editable package installs:

```bash
.venv\Scripts\pip install -e .
```

## Train

```bash
.venv\Scripts\python train.py --output-dir artifacts --total-timesteps 1000000
```

Outputs:

- `artifacts/pendulum_ppo.zip`
- `artifacts/pendulum_policy.onnx`
- `artifacts/config.json`
- `artifacts/train_summary.json`
- `artifacts/checkpoints/pendulum_ppo_*`
- `artifacts/best_model/`

Training behavior:

- Checkpoints are written periodically during training.
- The best eval model is saved separately.
- Training only stops early if you pass `--stop-reward-threshold` and the evaluation reward reaches it.

Example early-stop run:

```bash
.venv\Scripts\python train.py --output-dir artifacts --total-timesteps 1000000 --stop-reward-threshold 4500
```

## Visualize

```bash
.venv\Scripts\python visualize.py --model artifacts/pendulum_policy.onnx
```

## Run dry on Windows

```bash
.venv\Scripts\python run.py --model artifacts/pendulum_policy.onnx --dry-run --debug
```

## Run on Raspberry Pi

```bash
python run.py --model /path/to/pendulum_policy.onnx --debug
```

For Raspberry Pi hardware use, you do not need the full training stack. `requirements.txt` includes `torch` and `stable-baselines3`, which are only needed for training on a PC and can be large to download on the Pi. Install only the runtime dependencies instead:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip

cd ~/RL_pendulum
python3 -m venv .venv
source .venv/bin/activate

export PIP_NO_CACHE_DIR=1
export TMPDIR=$HOME/tmp
mkdir -p "$TMPDIR"

python -m pip install setuptools wheel
python -m pip install numpy onnx onnxruntime onnxscript moteus
```

If the Pi cannot resolve `pypi.org` or `www.piwheels.org`, the install will fail before it reaches the package downloads. In that case, skip the network install step and use a local wheelhouse or a machine with working DNS to pre-download the wheels.

Then run the hardware loop:

```bash
source .venv/bin/activate
python run.py --model /home/pi/RL_pendulum/artifacts/pendulum_policy.onnx --debug
```

### Raspberry Pi 3B stability notes

Pi 3B uses a specific onnxruntime version (1.20.1) to avoid C++ memory access bugs in older/newer versions. Use the dedicated Pi3 requirements file:

```bash
# on Pi 3B
cd ~/RL_pendulum
python3 -m venv .venv_pi3
source .venv_pi3/bin/activate

export PIP_NO_CACHE_DIR=1
export TMPDIR=$HOME/tmp
mkdir -p "$TMPDIR"

python -m pip install setuptools wheel
python -m pip install -r requirements.pi3.txt
python run.py --model /home/pi/RL_pendulum/artifacts/pendulum_policy.onnx --debug
```

The opset export defaults to 13 in config, which is compatible with onnxruntime 1.20.1 on Pi 3B.

## Next step

Run `train.py` after installing dependencies, then use `visualize.py` to inspect the learned swing-up behavior before moving to hardware.