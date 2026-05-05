from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from pendulum_rl.config import PendulumConfig
from pendulum_rl.env import PendulumSwingUpEnv
from pendulum_rl.model import build_policy_kwargs, export_policy_to_onnx


def make_env(config: PendulumConfig, seed: int):
    def _init():
        env = PendulumSwingUpEnv(config=config)
        env.reset(seed=seed)
        return Monitor(env)

    return _init


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a pendulum PPO policy and export ONNX")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"), help="Directory for models and metadata")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Training timesteps")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", default="cpu", help="Torch device")
    parser.add_argument("--n-envs", type=int, default=None, help="Number of vectorized environments")
    parser.add_argument("--progress-bar", action="store_true", help="Show the PPO progress bar if tqdm and rich are installed")
    parser.add_argument("--checkpoint-freq", type=int, default=50_000, help="Save a checkpoint every N environment steps")
    parser.add_argument("--eval-freq", type=int, default=10_000, help="Evaluate every N environment steps")
    parser.add_argument("--stop-reward-threshold", type=float, default=None, help="Stop training early when mean eval reward reaches this threshold")
    return parser.parse_args()


def make_eval_env(config: PendulumConfig):
    def _init():
        env = PendulumSwingUpEnv(config=config)
        env.reset(seed=config.seed + 10_000)
        return Monitor(env)

    return _init


class CheckpointOnnxExportCallback(CheckpointCallback):
    """Extend CheckpointCallback to export ONNX whenever a checkpoint is saved."""

    def __init__(self, save_freq: int, save_path: str, config: PendulumConfig, vec_env, **kwargs):
        super().__init__(save_freq=save_freq, save_path=save_path, **kwargs)
        self.config = config
        self.vec_env = vec_env

    def _on_step(self) -> bool:
        result = super()._on_step()
        if result is False:
            return False

        # If a checkpoint was just saved, also export ONNX
        if self.n_calls % self.save_freq == 0:
            checkpoint_num = self.n_calls // self.save_freq
            onnx_path = Path(self.save_path) / f"{self.name_prefix}_ckpt_{checkpoint_num}_policy.onnx"
            sample_obs = self.vec_env.reset()
            if isinstance(sample_obs, tuple):
                sample_obs = sample_obs[0]
            export_policy_to_onnx(self.model.policy, onnx_path, sample_obs[0], self.config.export_opset)
            print(f"[CheckpointOnnxExport] Exported ONNX for checkpoint {checkpoint_num} to {onnx_path}")

        return result


class EpochEtaCallback(BaseCallback):
    """Log ETA per epoch and total training time remaining."""

    def __init__(self, config: PendulumConfig, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.config = config
        self.start_time = time.time()
        self.epoch_start_time = time.time()
        self.last_logged_timestep = 0

    def _on_step(self) -> bool:
        # Log at every epoch transition (every n_steps timesteps)
        n_steps = self.config.ppo_n_steps * self.config.ppo_n_envs
        timesteps_so_far = self.model.num_timesteps
        total_steps = self.config.ppo_total_timesteps

        # Check if we've completed at least one epoch
        if timesteps_so_far >= n_steps and timesteps_so_far - self.last_logged_timestep >= n_steps:
            elapsed_time = time.time() - self.epoch_start_time
            epochs_completed = timesteps_so_far // n_steps
            total_epochs = (total_steps + n_steps - 1) // n_steps

            # ETA for current epoch
            if elapsed_time > 0:
                eta_epoch_s = elapsed_time
                total_elapsed = time.time() - self.start_time
                eta_total_s = (total_elapsed / timesteps_so_far) * (total_steps - timesteps_so_far)

                eta_epoch_m = eta_epoch_s / 60.0
                eta_total_m = eta_total_s / 60.0

                print(
                    f"[ETA] Epoch {epochs_completed}/{total_epochs} | "
                    f"Progress: {timesteps_so_far}/{total_steps} timesteps | "
                    f"Epoch time: {eta_epoch_m:.1f}min | "
                    f"ETA total: {eta_total_m:.1f}min"
                )
                self.epoch_start_time = time.time()

            self.last_logged_timestep = timesteps_so_far

        return True


def main() -> None:
    args = parse_args()
    config = PendulumConfig()
    if args.total_timesteps is not None:
        config.ppo_total_timesteps = args.total_timesteps
    if args.seed is not None:
        config.seed = args.seed
    if args.n_envs is not None:
        config.ppo_n_envs = args.n_envs

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "config.json"
    metadata_path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")

    env_fns = [make_env(config, config.seed + idx) for idx in range(config.ppo_n_envs)]
    vec_env = DummyVecEnv(env_fns)
    eval_env = DummyVecEnv([make_eval_env(config)])

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        policy_kwargs=build_policy_kwargs(),
        verbose=1,
        seed=config.seed,
        device=args.device,
        n_steps=config.ppo_n_steps,
        batch_size=config.ppo_batch_size,
        learning_rate=config.ppo_learning_rate,
        gamma=config.ppo_gamma,
        gae_lambda=config.ppo_gae_lambda,
        clip_range=config.ppo_clip_range,
        ent_coef=config.ppo_ent_coef,
        vf_coef=config.ppo_vf_coef,
        max_grad_norm=config.ppo_max_grad_norm,
        n_epochs=config.ppo_n_epochs,
    )

    callbacks = []
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    callbacks.append(
        CheckpointOnnxExportCallback(
            save_freq=max(1, args.checkpoint_freq // config.ppo_n_envs),
            save_path=str(checkpoint_dir),
            config=config,
            vec_env=vec_env,
            name_prefix="pendulum_ppo",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
    )

    # Add ETA logging callback
    callbacks.append(EpochEtaCallback(config=config, verbose=1))

    stop_training_callback = None
    if args.stop_reward_threshold is not None:
        stop_training_callback = StopTrainingOnRewardThreshold(
            reward_threshold=args.stop_reward_threshold,
            verbose=1,
        )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=max(1, args.eval_freq // config.ppo_n_envs),
        deterministic=True,
        render=False,
        callback_on_new_best=stop_training_callback,
        verbose=1,
    )
    callbacks.append(eval_callback)

    model.learn(total_timesteps=config.ppo_total_timesteps, progress_bar=args.progress_bar, callback=callbacks)

    model_path = output_dir / "pendulum_ppo.zip"
    onnx_path = output_dir / "pendulum_policy.onnx"
    model.save(model_path)

    sample_obs = vec_env.reset()
    if isinstance(sample_obs, tuple):
        sample_obs = sample_obs[0]
    export_policy_to_onnx(model.policy, onnx_path, sample_obs[0], config.export_opset)

    summary = {
        "model_path": str(model_path),
        "onnx_path": str(onnx_path),
        "total_timesteps": config.ppo_total_timesteps,
        "seed": config.seed,
        "n_envs": config.ppo_n_envs,
        "checkpoint_dir": str(checkpoint_dir),
        "best_model_dir": str(output_dir / "best_model"),
        "eval_log_dir": str(output_dir / "eval_logs"),
        "stop_reward_threshold": args.stop_reward_threshold,
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
