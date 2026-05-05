from __future__ import annotations

import argparse
import asyncio
import math
import time
from pathlib import Path

import numpy as np

from pendulum_rl.config import PendulumConfig
from pendulum_rl.preprocess import ObservationBuilder
from pendulum_rl.runtime import OnnxPendulumPolicy


def _read_register(values: object, register: object, default: float = 0.0) -> float:
    index = int(register)
    try:
        if hasattr(values, "__len__") and index < len(values):
            value = values[index]
            if value is not None:
                return float(value)
    except Exception:
        pass
    return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an exported ONNX pendulum policy")
    parser.add_argument("--model", required=True, help="Path to the exported .onnx policy")
    parser.add_argument("--rate", type=float, default=50.0, help="Control rate in Hz")
    parser.add_argument("--max-torque", type=float, default=1.0, help="Maximum torque to command in Nm")
    parser.add_argument("--watchdog", type=float, default=0.1, help="Watchdog timeout in seconds")
    parser.add_argument("--debug", action="store_true", help="Print telemetry every cycle")
    parser.add_argument("--device", default="/dev/ttyACM0", help="moteus device path placeholder")
    parser.add_argument("--dry-run", action="store_true", help="Run against the local simulator instead of hardware")
    parser.add_argument("--steps", type=int, default=500, help="Dry-run step count")
    return parser.parse_args()


async def run_hardware(model_path: Path, rate_hz: float, max_torque: float, watchdog_timeout: float, debug: bool) -> None:
    try:
        import moteus
    except Exception as exc:  # pragma: no cover - hardware only
        raise RuntimeError("moteus is required for hardware mode") from exc

    policy = OnnxPendulumPolicy(model_path)
    config = PendulumConfig()
    tracker = ObservationBuilder(
        history_length=config.history_length,
        velocity_scale_turns_per_s=config.max_speed_turns_per_s,
        torque_scale_nm=max_torque,
    )

    controller = moteus.Controller()
    result = await controller.set_stop(query=True)
    values = result.values
    position_turns = _read_register(values, moteus.Register.POSITION)
    velocity_turns_per_s = _read_register(values, moteus.Register.VELOCITY)
    torque_nm = _read_register(values, moteus.Register.TORQUE)
    obs = tracker.reset(position_turns, velocity_turns_per_s, torque_nm)

    period_s = 1.0 / rate_hz
    try:
        while True:
            loop_start = time.perf_counter()
            action = policy.predict(obs)
            command_torque = float(np.clip(action * max_torque, -max_torque, max_torque))

            result = await controller.set_position(
                position=math.nan,
                velocity=math.nan,
                feedforward_torque=command_torque,
                maximum_torque=max_torque,
                watchdog_timeout=watchdog_timeout,
                kp_scale=0.0,
                kd_scale=0.0,
                ignore_position_bounds=1,
                query=True,
            )
            values = result.values
            position_turns = _read_register(values, moteus.Register.POSITION, position_turns)
            velocity_turns_per_s = _read_register(values, moteus.Register.VELOCITY, velocity_turns_per_s)
            torque_nm = _read_register(values, moteus.Register.TORQUE, torque_nm)
            obs = tracker.push(position_turns, velocity_turns_per_s, torque_nm)

            if debug:
                print(
                    f"pos={position_turns: .3f} turns vel={velocity_turns_per_s: .3f} turns/s "
                    f"torque={torque_nm: .3f} Nm cmd={command_torque: .3f} Nm"
                )

            elapsed = time.perf_counter() - loop_start
            if elapsed < period_s:
                await asyncio.sleep(period_s - elapsed)
    finally:
        try:
            await controller.set_stop(query=True)
        except Exception:
            pass


def run_dry(model_path: Path, steps: int, rate_hz: float, debug: bool) -> None:
    from pendulum_rl.env import PendulumSwingUpEnv

    config = PendulumConfig()
    env = PendulumSwingUpEnv(config=config)
    policy = OnnxPendulumPolicy(model_path)
    obs, info = env.reset(seed=config.seed)

    period_s = 1.0 / rate_hz
    for step in range(steps):
        start = time.perf_counter()
        action = policy.predict(obs)
        obs, reward, terminated, truncated, info = env.step(np.array([action], dtype=np.float32))
        if debug:
            print(
                f"step={step} time={info['time_s']:.3f}s theta={info['theta_turns']:.3f} turns "
                f"vel={info['theta_dot_turns_per_s']:.3f} turns/s torque={info['applied_torque_nm']:.3f} Nm "
                f"reward={reward:.3f}"
            )
        if terminated or truncated:
            obs, info = env.reset(seed=config.seed)
        elapsed = time.perf_counter() - start
        if elapsed < period_s:
            time.sleep(period_s - elapsed)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if args.dry_run:
        run_dry(model_path, args.steps, args.rate, args.debug)
        return
    asyncio.run(run_hardware(model_path, args.rate, args.max_torque, args.watchdog, args.debug))


if __name__ == "__main__":
    main()
