import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from pendulum_rl.config import PendulumConfig
from pendulum_rl.env import PendulumSwingUpEnv
from pendulum_rl.preprocess import turns_to_radians

cfg = PendulumConfig()
env = PendulumSwingUpEnv(cfg)

print("model.opt.gravity:", env.model.opt.gravity)
print("config.gravity (applied via randomization) default:", cfg.randomization.gravity_min, "to", cfg.randomization.gravity_max)
print("nominal pivot inertia (kg m^2):", cfg.pivot_inertia_kgm2)

# target phase (turns)
target_turns = cfg.reward.target_phase_turns
theta = turns_to_radians(target_turns)
print(f"target_phase_turns: {target_turns} turns -> theta rad: {theta}")

m = cfg.tip_mass_kg
l = cfg.length_m
g = env._gravity
# gravity torque for simple pendulum (approx): -m * g * l * sin(theta)
torque_gravity = -m * g * l * __import__('math').sin(theta)
print(f"analytical gravity torque at target (Nm): {torque_gravity}")

# Show what happens if we set qpos to the target and forward
env.data.qpos[env._hinge_qposadr] = theta
env.data.qvel[env._hinge_dofadr] = 0.0
import mujoco
mujoco.mj_forward(env.model, env.data)
# Read generalized force due to gravity: use qfrc_bias (coriolis + gravity + constraint)
# To isolate gravity, zero velocities and controls and compute qfrc_bias
env.data.ctrl[0] = 0.0
mujoco.mj_forward(env.model, env.data)
print('data.qfrc_bias (first dof):', env.data.qfrc_bias[env._hinge_dofadr])

# Print observation for clarity
obs = env._raw_measurement()
print('raw measurement (turns, vel, torque):', obs)
print('\nPerturbation test around inverted position:')
for delta in [ -0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02 ]:
	tt = theta + delta
	env.data.qpos[env._hinge_qposadr] = tt
	env.data.qvel[env._hinge_dofadr] = 0.0
	mujoco.mj_forward(env.model, env.data)
	gtorque = float(env.data.qfrc_bias[env._hinge_dofadr])
	print(f" delta={delta:+.5f} rad -> qfrc_bias: {gtorque:+.6f} Nm")
print('Done')
