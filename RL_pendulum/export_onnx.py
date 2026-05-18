from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from pendulum_rl.config import PendulumConfig
from pendulum_rl.model import export_policy_to_onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-export ONNX from an existing SB3 PPO zip model")
    parser.add_argument("--model-zip", type=Path, required=True, help="Path to SB3 model zip (e.g. artifacts/pendulum_ppo.zip)")
    parser.add_argument("--out", type=Path, required=True, help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=None, help="ONNX opset version")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PendulumConfig()
    opset = cfg.export_opset if args.opset is None else args.opset

    model = PPO.load(str(args.model_zip), device="cpu")
    sample_obs = np.zeros((cfg.observation_dim,), dtype=np.float32)
    export_policy_to_onnx(model.policy, args.out, sample_obs, opset)

    print(f"Exported ONNX to {args.out}")
    print(f"Expected input features: {cfg.observation_dim}")


if __name__ == "__main__":
    main()
