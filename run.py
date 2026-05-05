"""Hardware runner: ONNX policy + moteus controller using 20-dim obs."""

from __future__ import annotations

import argparse
import asyncio
import math
import time
from pathlib import Path

import numpy as np
import moteus

from pendulum_rl.config import PendulumConfig
from pendulum_rl.preprocess import ObservationBuilder
from pendulum_rl.runtime import OnnxPendulumPolicy


def _read_register(values: object, register: object, default: float = 0.0) -> float:
    try:
        idx = int(register)
        if hasattr(values, "__len__") and idx < len(values):
            v = values[idx]
            if v is not None:
                return float(v)
    except Exception:
        pass
    return float(default)


async def run_hardware(model_path: Path, rate_hz: float, max_torque: float, watchdog_timeout: float, debug: bool) -> None:
    # Load ONNX policy that was exported from training (expects 20-dim stacked obs)
    policy = OnnxPendulumPolicy(model_path)

    # Print ONNX I/O metadata for debugging
    try:
        sess = policy.session
        inputs = sess.get_inputs()
        outputs = sess.get_outputs()
        print(f"[DEBUG] ONNX inputs: {[(i.name, i.shape, i.type) for i in inputs]}")
        print(f"[DEBUG] ONNX outputs: {[(o.name, o.shape, o.type) for o in outputs]}")
    except Exception as e:
        print(f"[DEBUG] Failed to read ONNX metadata: {e}")

    config = PendulumConfig()
    tracker = ObservationBuilder(
        history_length=config.history_length,
        velocity_scale_turns_per_s=config.max_speed_turns_per_s,
        torque_scale_nm=config.max_torque_nm,
    )

    # Request position, velocity, torque explicitly so result.values is predictable
    qr = moteus.QueryResolution()
    qr.position = moteus.F32
    qr.velocity = moteus.F32
    qr.torque = moteus.F32
    controller = moteus.Controller(id=1, query_resolution=qr)

    await controller.set_stop(query=False)
    res = await controller.query()
    vals = res.values
    print(f"[DEBUG] initial result.values length={len(vals) if hasattr(vals,'__len__') else 'unknown'}")

    pos = _read_register(vals, moteus.Register.POSITION)
    vel = _read_register(vals, moteus.Register.VELOCITY)
    torque = _read_register(vals, moteus.Register.TORQUE)
    obs = tracker.reset(pos, vel, torque)

    period_s = 1.0 / rate_hz
    step = 0
    try:
        while True:
            loop_start = time.perf_counter()

            if debug:
                a = np.asarray(obs)
                print(f"[DEBUG] loop={step} obs.shape={a.shape} obs[:8]={a.ravel()[:8]}")

            action = policy.predict(obs)
            cmd_torque = float(np.clip(action * max_torque, -max_torque, max_torque))

            res = await controller.set_position(
                position=math.nan,
                velocity=math.nan,
                feedforward_torque=cmd_torque,
                maximum_torque=max_torque,
                watchdog_timeout=watchdog_timeout,
                kp_scale=0.0,
                kd_scale=0.0,
                ignore_position_bounds=1,
                query=True,
            )

            vals = res.values
            print(f"[DEBUG] result.values length={len(vals) if hasattr(vals,'__len__') else 'unknown'}")
            pos = _read_register(vals, moteus.Register.POSITION, pos)
            vel = _read_register(vals, moteus.Register.VELOCITY, vel)
            torque = _read_register(vals, moteus.Register.TORQUE, torque)

            obs = tracker.push(pos, vel, torque)

            if debug:
                print(f"pos={pos: .3f} turns vel={vel: .3f} turns/s torque={torque: .3f} Nm cmd={cmd_torque: .3f} Nm")

            elapsed = time.perf_counter() - loop_start
            if elapsed < period_s:
                await asyncio.sleep(period_s - elapsed)
            step += 1
    finally:
        try:
            await controller.set_stop(query=True)
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a pendulum ONNX policy on moteus hardware (20-dim obs)")
    parser.add_argument("--model", required=True, help="Path to the exported .onnx policy")
    parser.add_argument("--rate", type=float, default=50.0, help="Control rate in Hz")
    parser.add_argument("--max-torque", type=float, default=1.0, help="Maximum torque to command (Nm)")
    parser.add_argument("--watchdog", type=float, default=0.1, help="Watchdog timeout (s)")
    parser.add_argument("--debug", action="store_true", help="Print telemetry every cycle")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_hardware(Path(args.model), args.rate, args.max_torque, args.watchdog, args.debug))
