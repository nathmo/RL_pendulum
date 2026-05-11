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
        <geom name="tip_mass" type="sphere" size="0.03" mass="{config.mass_kg}"/>
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
        self._length_m = self.config.length_m
        self._mass_kg = self.config.mass_kg
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
        self._mass_kg = self.config.mass_kg * self.np_random.uniform(rand.mass_scale_min, rand.mass_scale_max)
        self._length_m = self.config.length_m * self.np_random.uniform(rand.length_scale_min, rand.length_scale_max)
        self._viscous_friction = self.np_random.uniform(rand.viscous_friction_min, rand.viscous_friction_max)
        self._coulomb_friction = self.np_random.uniform(rand.coulomb_friction_min, rand.coulomb_friction_max)
        self._gravity = self.np_random.uniform(rand.gravity_min, rand.gravity_max)

        tip_mass = self._mass_kg
        radius = 0.03
        sphere_inertia = 0.4 * tip_mass * radius * radius
        self.model.body_mass[self._tip_body_id] = tip_mass
        self.model.body_inertia[self._tip_body_id] = np.array([sphere_inertia, sphere_inertia, sphere_inertia], dtype=np.float64)
        self.model.body_pos[self._tip_body_id] = np.array([0.0, 0.0, self._length_m], dtype=np.float64)
        self.model.dof_damping[self._hinge_dofadr] = self._viscous_friction
        self.model.dof_frictionloss[self._hinge_dofadr] = self._coulomb_friction
        self.model.opt.gravity[:] = np.array([0.0, 0.0, -self._gravity], dtype=np.float64)
        return {
            "mass_kg": self._mass_kg,
            "length_m": self._length_m,
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

    def _compute_reward(self, commanded_torque_nm: float, previous_commanded_torque_nm: float) -> tuple[float, dict[str, float]]:
        cfg = self.config
        reward_cfg = cfg.reward

        theta = float(self.data.qpos[self._hinge_qposadr])
        theta_dot = float(self.data.qvel[self._hinge_dofadr])
        theta_turns = radians_to_turns(theta)
        phase_error_turns = ((theta_turns - reward_cfg.target_phase_turns + 0.5) % 1.0) - 0.5

        upright = 0.5 * (1.0 + np.cos(2.0 * np.pi * phase_error_turns))
        potential = self._mass_kg * self._gravity * self._length_m * (1.0 - np.cos(theta))
        kinetic = 0.5 * self._mass_kg * (self._length_m * theta_dot) ** 2
        energy = potential + kinetic
        target_energy = 2.0 * self._mass_kg * self._gravity * self._length_m
        energy_error = abs(energy - target_energy)
        energy_reward = np.exp(-energy_error / max(reward_cfg.energy_scale, 1e-6))
        gravity_torque_nm = -self._mass_kg * self._gravity * self._length_m * np.sin(theta)
        gravity_torque_abs_nm = abs(gravity_torque_nm)

        theta_dot_turns_per_s = radians_to_turns(theta_dot)
        vel_penalty = (theta_dot_turns_per_s / max(reward_cfg.velocity_scale_turns_per_s, 1e-6)) ** 2
        
        # --- Smart torque penalty aware of perturbations ---
        # If external torque/force is being injected, don't penalize the control effort as harshly
        # Instead, reward the agent for recovering/stabilizing the system
        external_disturbance_magnitude = abs(self._injected_torque_nm) + abs(self._tip_force_n)
        
        torque_norm = commanded_torque_nm / max(self.config.max_torque_nm, 1e-6)
        
        # Base torque penalty
        base_torque_penalty = torque_norm ** 2
        
        # If there's significant external disturbance, modulate the penalty
        if external_disturbance_magnitude > 1e-6:
            # Normalize disturbance magnitude
            max_possible_disturbance = self.config.perturbation.max_torque_nm + self.config.perturbation.max_tip_force_n
            disturbance_norm = min(1.0, external_disturbance_magnitude / max(max_possible_disturbance, 1e-6))
            
            # When disturbance is present, reduce the torque penalty (allow higher control effort to counter it)
            # but only if the control is actually counter-acting the disturbance
            # This is a simple heuristic: if commanded torque is in opposite direction to injected torque, reward it
            if abs(self._injected_torque_nm) > 1e-6:
                torque_alignment = (self._injected_torque_nm * commanded_torque_nm) / (abs(self._injected_torque_nm) * abs(commanded_torque_nm) + 1e-6)
                # If controller is opposing the disturbance (negative alignment), reduce penalty
                if torque_alignment < -0.3:  # Somewhat opposed
                    base_torque_penalty *= max(0.2, 1.0 - 0.8 * disturbance_norm)
            else:
                # No opposing torque disturbance, just reduce penalty due to presence of disturbance
                base_torque_penalty *= max(0.3, 1.0 - 0.7 * disturbance_norm)
        
        torque_penalty = base_torque_penalty
        # --- Duration-sensitive (thermal-like) torque saturation penalty ---
        # Compute normalized excess above the saturation threshold
        threshold = float(getattr(reward_cfg, "torque_saturation_threshold", 0.95))
        excess = max(0.0, abs(torque_norm) - threshold) / max(1e-6, (1.0 - threshold))
        scaled = min(1.0, excess)
        expnt = float(getattr(reward_cfg, "torque_saturation_integrator_exponent", 2.0))
        # integrate with exponential decay using control_dt as timestep
        dt = float(self.config.control_dt)
        tau = max(1e-6, float(getattr(reward_cfg, "torque_saturation_time_constant_s", 2.0)))
        alpha = float(np.exp(-dt / tau))
        self._torque_saturation_integrator = alpha * self._torque_saturation_integrator + (1.0 - alpha) * (scaled ** expnt)
        torque_saturation_penalty = float(reward_cfg.torque_saturation_penalty_weight) * (self._torque_saturation_integrator ** 2)
        delta_torque_penalty = ((commanded_torque_nm - previous_commanded_torque_nm) / max(self.config.max_torque_nm, 1e-6)) ** 2

        reward = (
            reward_cfg.upright_weight * upright
            + reward_cfg.energy_weight * energy_reward
            - reward_cfg.velocity_penalty_weight * vel_penalty
            - reward_cfg.torque_penalty_weight * torque_penalty
            - torque_saturation_penalty
            - reward_cfg.delta_torque_penalty_weight * delta_torque_penalty
        )

        # --- Rolling-average rotation penalty ---
        # Track rotation (in turns) over time and compute net revolutions
        current_time_s = float(self._substep_index * self.config.physics_dt)
        current_turns = radians_to_turns(theta)
        # append current sample
        self._rotation_history.append((current_time_s, current_turns))
        # purge old samples outside the window
        window_s = float(reward_cfg.rolling_window_s)
        while self._rotation_history and (current_time_s - self._rotation_history[0][0]) > window_s:
            self._rotation_history.popleft()

        rolling_penalty = 0.0
        rolling_rev = 0.0
        if current_time_s >= window_s and len(self._rotation_history) >= 2:
            first_time, first_turns = self._rotation_history[0]
            last_time, last_turns = self._rotation_history[-1]
            # net accumulated revolutions over the window
            rolling_rev = float(last_turns - first_turns)
            # if net revolutions exceed threshold, penalize
            if abs(rolling_rev) > float(reward_cfg.rolling_rev_threshold_turns):
                rev_per_s = rolling_rev / window_s
                # normalize by velocity scale then square
                norm = (abs(rev_per_s) / max(reward_cfg.velocity_scale_turns_per_s, 1e-6)) ** 2
                rolling_penalty = float(reward_cfg.rolling_penalty_weight) * norm
                reward -= rolling_penalty


        success = upright >= reward_cfg.success_upright_threshold and abs(theta_dot_turns_per_s) <= reward_cfg.success_velocity_threshold_turns_per_s
        self._success_counter = self._success_counter + 1 if success else 0
        if self._success_counter >= reward_cfg.success_hold_steps:
            reward += reward_cfg.success_bonus

        metrics = {
            "upright": upright,
            "phase_error_turns": phase_error_turns,
            "energy_reward": energy_reward,
            "gravity_torque_nm": gravity_torque_nm,
            "gravity_torque_abs_nm": gravity_torque_abs_nm,
            "theta_turns": theta_turns,
            "theta_dot_turns_per_s": theta_dot_turns_per_s,
            "commanded_torque_nm": commanded_torque_nm,
            "injected_torque_nm": self._injected_torque_nm,
            "injected_force_n": self._tip_force_n,
            "external_disturbance_magnitude": external_disturbance_magnitude,
            "torque_norm": torque_norm,
            "torque_saturation_penalty": torque_saturation_penalty,
            "torque_saturation_integrator": float(self._torque_saturation_integrator),
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
        mujoco.mj_forward(self.model, self.data)

        observation = self._observation_builder.reset(position_turns, velocity_turns_per_s, 0.0)
        info = {
            "randomization": randomization,
            "time_s": 0.0,
            "position_turns": position_turns,
            "velocity_turns_per_s": velocity_turns_per_s,
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
