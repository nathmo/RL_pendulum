from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class RandomizationConfig:
    mass_scale_min: float = 0.5
    mass_scale_max: float = 2.0
    length_scale_min: float = 0.5
    length_scale_max: float = 2.0
    viscous_friction_min: float = 0.0
    viscous_friction_max: float = 0.2
    coulomb_friction_min: float = 0.0
    coulomb_friction_max: float = 0.2
    gravity_min: float = 9.0
    gravity_max: float = 11.0
    observation_position_sigma_turns: float = 0.2
    observation_velocity_sigma_turns_per_s: float = 0.5
    observation_torque_sigma_nm: float = 0.1
    control_jitter_std_s: float = 0.002
    control_jitter_max_s: float = 0.005
    command_delay_mean_s: float = 0.005
    command_delay_std_s: float = 0.001
    command_delay_max_s: float = 0.005
    packet_drop_prob: float = 0.002


@dataclass(slots=True)
class RewardConfig:
    upright_weight: float = 5
    energy_weight: float = 3
    velocity_penalty_weight: float = 0.02
    torque_penalty_weight: float = 0.05
    torque_saturation_penalty_weight: float = 0.2
    # Normalized torque above which saturation is considered (fraction of max torque)
    torque_saturation_threshold: float = 0.95
    # Time constant (seconds) for the saturation integrator (thermal-like)
    torque_saturation_time_constant_s: float = 2.0
    # Exponent applied to normalized saturation when accumulating (larger -> harsher)
    torque_saturation_integrator_exponent: float = 2.0
    delta_torque_penalty_weight: float = 0.01
    target_phase_turns: float = 0.5
    success_bonus: float = 8.0
    success_hold_steps: int = 12
    success_upright_threshold: float = 0.95
    success_velocity_threshold_turns_per_s: float = 0.08
    energy_scale: float = 0.35
    velocity_scale_turns_per_s: float = 2.0
    # Rolling-average penalty (seconds)
    rolling_window_s: float = 5.0
    # Weight applied to rolling average speed penalty
    rolling_penalty_weight: float = 0.2
    # Minimum cumulative revolutions over the window to apply penalty
    rolling_rev_threshold_turns: float = 0.1


@dataclass(slots=True)
class PendulumConfig:
    mass_kg: float = 0.2
    length_m: float = 0.75
    max_torque_nm: float = 1.0
    control_hz: float = 50.0
    physics_hz: float = 1000.0
    history_length: int = 5
    observation_features_per_step: int = 4
    initial_position_turns_min: float = -5.0
    initial_position_turns_max: float = 5.0
    initial_velocity_turns_per_s_min: float = -1.0
    initial_velocity_turns_per_s_max: float = 1.0
    max_speed_turns_per_s: float = 2.0
    episode_seconds: float = 30.0
    seed: int = 7
    export_opset: int = 13
    ppo_total_timesteps: int = 1_000_000
    ppo_n_steps: int = 2048
    ppo_batch_size: int = 256
    ppo_learning_rate: float = 3e-4
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95
    ppo_clip_range: float = 0.2
    ppo_ent_coef: float = 0.0
    ppo_vf_coef: float = 0.5
    ppo_max_grad_norm: float = 0.5
    ppo_n_epochs: int = 10
    ppo_n_envs: int = 1
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    @property
    def control_dt(self) -> float:
        return 1.0 / self.control_hz

    @property
    def physics_dt(self) -> float:
        return 1.0 / self.physics_hz

    @property
    def observation_dim(self) -> int:
        return self.history_length * self.observation_features_per_step

    @property
    def episode_steps(self) -> int:
        return int(round(self.episode_seconds * self.control_hz))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
