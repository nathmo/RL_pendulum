from __future__ import annotations

from math import pi
from collections import deque
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .config import PendulumConfig
from .preprocess import ObservationBuilder, radians_to_turns, turns_to_radians, wrap_angle


def _make_xml(config: PendulumConfig) -> str:
    timestep = 1.0 / config.physics_hz
    return f"""
<mujoco model="inverted_pendulum">
  <compiler angle="radian" coordinate="local"/>
  <option timestep="{timestep}" integrator="Euler" gravity="0 0 -9.81"/>
  <default>
    <joint damping="0" frictionloss="0" armature="0"/>
    <geom contype="0" conaffinity="0" rgba="0.5 0.5 0.5 1"/>
  </default>
  <worldbody>
    <body name="pivot" pos="0 0 0">
      <joint name="hinge" type="hinge" axis="0 1 0" limited="false"/>
            <geom name="rod" type="capsule" fromto="0 0 0 0 0 {config.length_m}" size="0.01" density="0"/>
      <body name="tip" pos="0 0 {config.length_m}">
                <geom name="tip_mass" type="sphere" size="{config.tip_radius_m}" mass="{config.tip_mass_kg}"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="hinge" gear="1" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""


class PendulumSwingUpEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self, config: PendulumConfig | None = None) -> None:
        super().__init__()
        self.config = config or PendulumConfig()
        self.np_random = np.random.default_rng(self.config.seed)
        self.model = mujoco.MjModel.from_xml_string(_make_xml(self.config))
        self.data = mujoco.MjData(self.model)

        self._hinge_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hinge")
        self._hinge_qposadr = int(self.model.jnt_qposadr[self._hinge_joint_id])
        self._hinge_dofadr = int(self.model.jnt_dofadr[self._hinge_joint_id])
        self._tip_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "tip")

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.config.observation_dim,),
            dtype=np.float32,
        )

        self._observation_builder = ObservationBuilder(
            history_length=self.config.history_length,
            velocity_scale_turns_per_s=self.config.max_speed_turns_per_s,
            torque_scale_nm=self.config.max_torque_nm,
        )
        self._active_torque_nm = 0.0
        self._pending_torque_nm = 0.0
        self._pending_apply_substep = 0
        self._substep_index = 0
        self._episode_step = 0
        self._success_counter = 0
        self._last_commanded_torque_nm = 0.0
        self._torque_saturation_integrator = 0.0
        # thermal-style integrator for torque saturation (normalized 0..1)
        self._torque_saturation_integrator = 0.0
        self._steady_state_error_time_s = 0.0
        self._length_m = self.config.length_m
        self._tip_mass_kg = self.config.tip_mass_kg
        self._viscous_friction = 0.0
        self._coulomb_friction = 0.0
        self._gravity = 9.81
        # rotation history: deque of (time_s, theta_turns)
        self._rotation_history: deque[tuple[float, float]] = deque()
        
        # Perturbation tracking
        self._injected_torque_nm = 0.0  # External torque being injected
        self._perturbation_end_substep = 0  # When current perturbation ends
        self._tip_force_n = 0.0  # Tangential force at pendulum tip (N)
        self._injected_force_history: deque[tuple[float, float]] = deque()  # (time_s, force_magnitude)

    def _apply_randomization(self) -> dict[str, float]:
        rand = self.config.randomization
        self._tip_mass_kg = self.config.tip_mass_kg * self.np_random.uniform(rand.mass_scale_min, rand.mass_scale_max)
        self._length_m = self.config.length_m * self.np_random.uniform(rand.length_scale_min, rand.length_scale_max)
        self._viscous_friction = self.np_random.uniform(rand.viscous_friction_min, rand.viscous_friction_max)
        self._coulomb_friction = self.np_random.uniform(rand.coulomb_friction_min, rand.coulomb_friction_max)
        self._gravity = self.np_random.uniform(rand.gravity_min, rand.gravity_max)

        self.model.dof_damping[self._hinge_dofadr] = self._viscous_friction
        self.model.dof_frictionloss[self._hinge_dofadr] = self._coulomb_friction
        self.model.opt.gravity[:] = np.array([0.0, 0.0, -self._gravity], dtype=np.float64)
        return {
            "tip_mass_kg": self._tip_mass_kg,
            "length_m": self._length_m,
            "pivot_inertia_kgm2": self._tip_mass_kg * self._length_m**2,
            "viscous_friction": self._viscous_friction,
            "coulomb_friction": self._coulomb_friction,
            "gravity": self._gravity,
        }

    def _maybe_inject_perturbation(self) -> None:
        """Inject random external torque or force with configured probability."""
        pert_cfg = self.config.perturbation
        
        # Check if current perturbation has ended
        if self._substep_index >= self._perturbation_end_substep:
            # Decide if we should inject a new perturbation
            if float(self.np_random.random()) < pert_cfg.injection_probability:
                # Inject torque at the hinge joint
                magnitude = float(self.np_random.uniform(-pert_cfg.max_torque_nm, pert_cfg.max_torque_nm))
                self._injected_torque_nm = magnitude
                
                # Optionally also inject tangential force at tip
                if float(self.np_random.random()) < pert_cfg.tip_force_probability:
                    self._tip_force_n = float(self.np_random.uniform(-pert_cfg.max_tip_force_n, pert_cfg.max_tip_force_n))
                else:
                    self._tip_force_n = 0.0
                
                # Set when this perturbation should end
                duration_substeps = max(1, int(round(pert_cfg.injection_duration_s / self.config.physics_dt)))
                self._perturbation_end_substep = self._substep_index + duration_substeps
            else:
                self._injected_torque_nm = 0.0
                self._tip_force_n = 0.0

    def _raw_measurement(self) -> tuple[float, float, float]:
        theta = float(self.data.qpos[self._hinge_qposadr])
        theta_turns = radians_to_turns(theta)
        theta_dot_turns_per_s = radians_to_turns(float(self.data.qvel[self._hinge_dofadr]))
        torque_nm = float(self._active_torque_nm)

        rand = self.config.randomization
        theta_turns += float(self.np_random.normal(0.0, rand.observation_position_sigma_turns))
        theta_dot_turns_per_s += float(self.np_random.normal(0.0, rand.observation_velocity_sigma_turns_per_s))
        torque_nm += float(self.np_random.normal(0.0, rand.observation_torque_sigma_nm))
        return theta_turns, theta_dot_turns_per_s, torque_nm

    def _build_observation(self) -> np.ndarray:
        theta_turns, theta_dot_turns_per_s, torque_nm = self._raw_measurement()
        return self._observation_builder.push(theta_turns, theta_dot_turns_per_s, torque_nm)

    @staticmethod
    def _shape_reward(error_ratio: float, mode: str) -> float:
        if mode == "quadratic":
            return max(0.0, 1.0 - error_ratio**2)
        return float(np.exp(-(error_ratio**2)))

    def _compute_reward(self, commanded_torque_nm: float, previous_commanded_torque_nm: float) -> tuple[float, dict[str, float]]:
        cfg = self.config
        reward_cfg = cfg.reward
        theta = float(self.data.qpos[self._hinge_qposadr])
        theta_dot = float(self.data.qvel[self._hinge_dofadr])
        theta_turns = radians_to_turns(theta)
        phase_error_turns = ((theta_turns - reward_cfg.target_phase_turns + 0.5) % 1.0) - 0.5
        abs_phase_error_turns = abs(phase_error_turns)
        theta_dot_turns_per_s = radians_to_turns(theta_dot)
        torque_norm = commanded_torque_nm / max(self.config.max_torque_nm, 1e-6)
        position_cost = phase_error_turns**2
        speed_cost = theta_dot_turns_per_s**2
        effort_cost = torque_norm**2

        reward = (
            -reward_cfg.upright_weight * position_cost
            -reward_cfg.velocity_penalty_weight * speed_cost
            -reward_cfg.torque_penalty_weight * effort_cost
        )

        dt = float(self.config.control_dt)
        self._steady_state_error_time_s = 0.0
        self._success_counter = 0

        rolling_penalty = 0.0
        rolling_rev = 0.0
        steady_state_error_penalty = 0.0
        torque_saturation_penalty = 0.0
        gravity_torque_nm = -self._gravity * (self._tip_mass_kg * self._length_m) * np.sin(theta)
        gravity_torque_abs_nm = abs(gravity_torque_nm)
        pivot_inertia_kgm2 = self._tip_mass_kg * self._length_m**2
        energy_reward = 0.0

        metrics = {
            "reward_mode": "quadratic",
            "upright": max(0.0, 1.0 - position_cost),
            "phase_error_turns": phase_error_turns,
            "abs_phase_error_turns": abs_phase_error_turns,
            "energy_reward": energy_reward,
            "gravity_torque_nm": gravity_torque_nm,
            "gravity_torque_abs_nm": gravity_torque_abs_nm,
            "pivot_inertia_kgm2": pivot_inertia_kgm2,
            "theta_turns": theta_turns,
            "theta_dot_turns_per_s": theta_dot_turns_per_s,
            "commanded_torque_nm": commanded_torque_nm,
            "torque_norm": torque_norm,
            "position_cost": position_cost,
            "speed_cost": speed_cost,
            "effort_cost": effort_cost,
            "total_cost": -reward,
            "torque_saturation_penalty": torque_saturation_penalty,
            "torque_saturation_integrator": 0.0,
            "steady_state_error_time_s": float(self._steady_state_error_time_s),
            "steady_state_error_penalty": steady_state_error_penalty,
            "rolling_rev_turns": rolling_rev,
            "rolling_penalty": rolling_penalty,
        }
        return float(reward), metrics

    def _simulate_control_interval(self, commanded_torque_nm: float) -> None:
        rand = self.config.randomization
        hold_dt = self.config.control_dt + float(self.np_random.normal(0.0, rand.control_jitter_std_s))
        hold_dt = float(np.clip(hold_dt, self.config.physics_dt, self.config.control_dt + rand.control_jitter_max_s))
        hold_steps = max(1, int(round(hold_dt / self.config.physics_dt)))

        if float(self.np_random.random()) >= rand.packet_drop_prob:
            delay_s = float(self.np_random.normal(rand.command_delay_mean_s, rand.command_delay_std_s))
            delay_s = float(np.clip(delay_s, 0.0, rand.command_delay_max_s))
            delay_steps = int(round(delay_s / self.config.physics_dt))
            self._pending_torque_nm = commanded_torque_nm
            self._pending_apply_substep = self._substep_index + delay_steps

        for _ in range(hold_steps):
            # Check if we need to inject a new perturbation
            if self._substep_index % max(1, int(round(self.config.control_dt / self.config.physics_dt))) == 0:
                self._maybe_inject_perturbation()
            
            if self._substep_index >= self._pending_apply_substep:
                self._active_torque_nm = self._pending_torque_nm

            # Apply commanded torque + injected perturbation torque
            total_torque = self._active_torque_nm + self._injected_torque_nm
            self.data.ctrl[0] = np.clip(total_torque, -self.config.max_torque_nm * 2, self.config.max_torque_nm * 2)
            
            # Apply tangential force at pendulum tip if present
            if abs(self._tip_force_n) > 1e-6:
                # Get pendulum angle to compute force direction
                theta = float(self.data.qpos[self._hinge_qposadr])
                # Tangential (perpendicular to rod) direction in the plane of rotation
                # For a 2D pendulum rotating about y-axis, force is in x-z plane
                # Perpendicular to rod direction is: (-sin(theta), 0, -cos(theta)) normalized
                force_x = -self._tip_force_n * np.sin(theta)
                force_z = -self._tip_force_n * np.cos(theta)
                
                # Apply force at tip body
                self.data.xfrc_applied[self._tip_body_id, 0] = force_x  # x component
                self.data.xfrc_applied[self._tip_body_id, 2] = force_z  # z component (gravity is in -z)
            
            mujoco.mj_step(self.model, self.data)
            self._substep_index += 1

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        randomization = self._apply_randomization()
        mujoco.mj_resetData(self.model, self.data)
        # Clear external forces
        self.data.xfrc_applied[:] = 0.0

        position_turns = float(
            self.np_random.uniform(
                self.config.initial_position_turns_min,
                self.config.initial_position_turns_max,
            )
        )
        velocity_turns_per_s = float(
            self.np_random.uniform(
                self.config.initial_velocity_turns_per_s_min,
                self.config.initial_velocity_turns_per_s_max,
            )
        )
        self.data.qpos[self._hinge_qposadr] = turns_to_radians(position_turns)
        self.data.qvel[self._hinge_dofadr] = turns_to_radians(velocity_turns_per_s)
        self._active_torque_nm = 0.0
        self._pending_torque_nm = 0.0
        self._pending_apply_substep = 0
        self._substep_index = 0
        self._episode_step = 0
        self._success_counter = 0
        self._last_commanded_torque_nm = 0.0
        # Reset perturbation state
        self._injected_torque_nm = 0.0
        self._tip_force_n = 0.0
        self._perturbation_end_substep = 0
        self._steady_state_error_time_s = 0.0
        mujoco.mj_forward(self.model, self.data)

        observation = self._observation_builder.reset(position_turns, velocity_turns_per_s, 0.0)
        info = {
            "randomization": randomization,
            "time_s": 0.0,
            "position_turns": position_turns,
            "velocity_turns_per_s": velocity_turns_per_s,
            "pivot_inertia_kgm2": self.config.pivot_inertia_kgm2,
            "tip_mass_kg": self._tip_mass_kg,
        }
        # initialize rotation history with the starting state at t=0
        self._rotation_history.clear()
        self._rotation_history.append((0.0, float(position_turns)))
        return observation, info

    def step(self, action: np.ndarray):
        self._episode_step += 1
        requested = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        requested = float(np.clip(requested, -1.0, 1.0))
        commanded_torque_nm = requested * self.config.max_torque_nm
        previous_commanded_torque_nm = self._last_commanded_torque_nm
        self._last_commanded_torque_nm = commanded_torque_nm
        
        # Clear external forces at start of step (will be reapplied if needed during simulation)
        self.data.xfrc_applied[:] = 0.0

        self._simulate_control_interval(commanded_torque_nm)
        reward, metrics = self._compute_reward(commanded_torque_nm, previous_commanded_torque_nm)
        observation = self._build_observation()

        theta_dot_turns_per_s = metrics["theta_dot_turns_per_s"]
        terminated = False
        truncated = self._episode_step >= self.config.episode_steps
        if not np.isfinite(observation).all():
            truncated = True
        if abs(theta_dot_turns_per_s) > self.config.max_speed_turns_per_s * 4.0:
            truncated = True

        info = {
            **metrics,
            "time_s": self._substep_index * self.config.physics_dt,
            "episode_step": self._episode_step,
            "applied_torque_nm": self._active_torque_nm,
            "previous_commanded_torque_nm": previous_commanded_torque_nm,
        }
        return observation, reward, terminated, truncated, info
