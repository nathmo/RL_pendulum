from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pendulum_rl.config import PendulumConfig
from pendulum_rl.env import PendulumSwingUpEnv
from pendulum_rl.runtime import OnnxPendulumPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a pendulum policy in simulation")
    parser.add_argument("--model", type=Path, required=True, help="Path to the exported ONNX policy")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to display")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional step cap per episode")
    parser.add_argument("--sleep", type=float, default=None, help="Optional real-time delay per step")
    return parser.parse_args()


def draw_pendulum(ax, theta: float, length: float) -> None:
    x = length * np.sin(theta)
    z = -length * np.cos(theta)
    ax.cla()
    ax.set_xlim(-length * 1.1, length * 1.1)
    ax.set_ylim(-length * 1.1, length * 1.1)
    ax.set_zlim(-length * 1.1, length * 1.1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.plot([0.0, x], [0.0, 0.0], [0.0, z], linewidth=3)
    ax.scatter([x], [0.0], [z], s=80)
    ax.view_init(elev=20, azim=40)


def main() -> None:
    args = parse_args()
    config = PendulumConfig()
    env = PendulumSwingUpEnv(config=config)
    policy = OnnxPendulumPolicy(args.model)

    plt.ion()
    fig = plt.figure(figsize=(13, 8))
    ax_3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax_angle = fig.add_subplot(2, 2, 2)
    ax_velocity = fig.add_subplot(2, 2, 3)
    ax_torque = fig.add_subplot(2, 2, 4)

    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed)
        history_time: list[float] = []
        history_theta: list[float] = []
        history_theta_dot: list[float] = []
        history_torque: list[float] = []
        history_reward: list[float] = []

        done = False
        step_index = 0
        while not done:
            start = time.perf_counter()
            action = policy.predict(obs)
            obs, reward, terminated, truncated, info = env.step(np.array([action], dtype=np.float32))
            done = terminated or truncated

            history_time.append(info["time_s"])
            history_theta.append(info["theta_turns"] * 2.0 * np.pi)
            history_theta_dot.append(info["theta_dot_turns_per_s"] * 2.0 * np.pi)
            history_torque.append(info["applied_torque_nm"])
            history_reward.append(reward)

            theta = info["theta_turns"] * 2.0 * np.pi
            length_m = info.get("randomization", {}).get("length_m", config.length_m)
            draw_pendulum(ax_3d, theta, length_m)

            ax_angle.cla()
            ax_angle.plot(history_time, history_theta, label="theta (rad)")
            ax_angle.plot(history_time, np.full(len(history_time), np.pi), linestyle="--", label="upright")
            ax_angle.set_title("Angle")
            ax_angle.legend(loc="best")

            ax_velocity.cla()
            ax_velocity.plot(history_time, history_theta_dot, label="theta dot (rad/s)")
            ax_velocity.set_title("Velocity")
            ax_velocity.legend(loc="best")

            ax_torque.cla()
            ax_torque.plot(history_time, history_torque, label="torque (Nm)")
            ax_torque.plot(history_time, history_reward, label="reward")
            ax_torque.set_title("Torque and Reward")
            ax_torque.legend(loc="best")

            fig.suptitle(f"Episode {episode + 1} step {step_index} reward {reward:.3f}")
            fig.tight_layout()
            plt.pause(0.001)

            if args.sleep is not None:
                elapsed = time.perf_counter() - start
                time.sleep(max(0.0, args.sleep - elapsed))
            step_index += 1
            if args.max_steps is not None and step_index >= args.max_steps:
                break

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
