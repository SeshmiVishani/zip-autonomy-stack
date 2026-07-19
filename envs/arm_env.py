"""
arm_env.py

Reaching-task environment for a MuJoCo planar arm, built specifically to support
the Stage 1 goal: measuring how control-autonomy level (what the policy actually
outputs) and control frequency (how often it outputs) trade off against latency
and task performance.

Every autonomy level bottoms out in a torque command sent to MuJoCo -- that is
the one thing the physics engine actually understands. The three levels differ
in *how much of the low-level control problem is delegated to the policy* vs.
handled by a fixed lower-level controller:

  - "torque":         policy output IS the torque. Maximum policy authority,
                       zero abstraction. Highest potential reactivity, hardest
                       to learn, most sensitive to control frequency.
  - "joint_position":  policy outputs a target joint angle; a PD loop (running
                       at the *physics* rate, not the policy rate) converts
                       that into torque every physics step. Classic robotics
                       abstraction -- easier to learn, but the PD loop's fixed
                       gains impose a latency floor the policy can't reach below.
  - "pwm":             policy outputs a duty cycle in [-1, 1]; a simplified
                       DC-motor model (deadband + saturation + first-order
                       torque response) converts that into torque. Sits between
                       the other two: more abstraction than raw torque, but
                       models actuator nonlinearity that "torque" ignores.

decimation = physics_steps_per_control_step lets you sweep control frequency
independently of the 500Hz physics timestep (see configs/*.yaml).
"""

import os
from typing import Optional

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

ASSET_PATH = os.path.join(os.path.dirname(__file__), "assets", "arm3dof.xml")


class ArmReachEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        control_level: str = "joint_position",
        control_hz: float = 50.0,
        physics_hz: float = 500.0,
        pd_kp: float = 8.0,
        pd_kd: float = 0.8,
        pwm_deadband: float = 0.03,
        pwm_torque_gain: Optional[np.ndarray] = None,
        pwm_time_const: float = 0.02,
        reward_dist_weight: float = 1.0,
        reward_effort_weight: float = 0.001,
        success_thresh: float = 0.03,
        max_episode_steps: int = 300,
        render_mode: Optional[str] = None,
    ):
        assert control_level in ("torque", "joint_position", "pwm")
        self.control_level = control_level
        self.control_hz = control_hz
        self.physics_hz = physics_hz
        self.decimation = max(1, round(physics_hz / control_hz))

        self.model = mujoco.MjModel.from_xml_path(ASSET_PATH)
        self.model.opt.timestep = 1.0 / physics_hz
        self.data = mujoco.MjData(self.model)

        self.n_joints = self.model.nu  # actuator count == joint count here
        self.torque_limits = self.model.actuator_ctrlrange.copy()

        self.pd_kp = np.full(self.n_joints, pd_kp)
        self.pd_kd = np.full(self.n_joints, pd_kd)
        self.pwm_deadband = pwm_deadband
        self.pwm_torque_gain = (
            pwm_torque_gain
            if pwm_torque_gain is not None
            else self.torque_limits[:, 1]
        )
        self.pwm_time_const = pwm_time_const
        self._pwm_applied_torque = np.zeros(self.n_joints)  # first-order motor state

        self.reward_dist_weight = reward_dist_weight
        self.reward_effort_weight = reward_effort_weight
        self.success_thresh = success_thresh
        self.max_episode_steps = max_episode_steps
        self._step_count = 0
        self.render_mode = render_mode
        self._viewer = None
        self._renderer = None

        # Action space differs by autonomy level; observation space is fixed.
        if control_level == "torque":
            low, high = self.torque_limits[:, 0], self.torque_limits[:, 1]
        elif control_level == "joint_position":
            low = self.model.jnt_range[:, 0]
            high = self.model.jnt_range[:, 1]
        else:  # pwm
            low = -np.ones(self.n_joints)
            high = np.ones(self.n_joints)
        self.action_space = spaces.Box(low=low.astype(np.float32), high=high.astype(np.float32))

        obs_dim = self.n_joints * 2 + 3  # qpos, qvel, target_xy_rel + ee_z (kept flat)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.target_id = self.model.body("target").mocapid[0]
        self.ee_site_id = self.model.site("end_effector").id

    # ---------- core Gym API ----------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._pwm_applied_torque[:] = 0.0
        self._step_count = 0

        # Randomize start joint config slightly and target position within reach.
        self.data.qpos[: self.n_joints] = self.np_random.uniform(-0.2, 0.2, self.n_joints)
        mujoco.mj_forward(self.model, self.data)

        r = self.np_random.uniform(0.25, 0.65)
        theta = self.np_random.uniform(-np.pi, np.pi)
        target_xy = np.array([r * np.cos(theta), r * np.sin(theta)])
        self.data.mocap_pos[self.target_id, :2] = target_xy
        self.data.mocap_pos[self.target_id, 2] = 0.05

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Run `decimation` physics steps per control step -- this is what lets
        # control_hz be swept independently of physics_hz in the config sweep.
        for _ in range(self.decimation):
            torque = self._action_to_torque(action)
            self.data.ctrl[:] = np.clip(
                torque, self.torque_limits[:, 0], self.torque_limits[:, 1]
            )
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs = self._get_obs()
        dist = np.linalg.norm(self._ee_pos()[:2] - self.data.mocap_pos[self.target_id, :2])
        effort = float(np.sum(np.square(self.data.ctrl)))

        reward = (
            -self.reward_dist_weight * dist
            - self.reward_effort_weight * effort
        )
        terminated = bool(dist < self.success_thresh)
        if terminated:
            reward += 5.0
        truncated = self._step_count >= self.max_episode_steps

        info = {"dist": dist, "effort": effort, "control_level": self.control_level}
        return obs, reward, terminated, truncated, info

    # ---------- autonomy-level action mapping ----------

    def _action_to_torque(self, action: np.ndarray) -> np.ndarray:
        """
        This is the one function that changes meaning across autonomy levels --
        everything upstream (RL algorithm, replay buffer, action space) is
        agnostic to which of these branches runs.
        """
        if self.control_level == "torque":
            return action

        elif self.control_level == "joint_position":
            qpos = self.data.qpos[: self.n_joints]
            qvel = self.data.qvel[: self.n_joints]
            return self.pd_kp * (action - qpos) - self.pd_kd * qvel

        else:  # pwm
            duty = np.where(np.abs(action) < self.pwm_deadband, 0.0, action)
            target_torque = duty * self.pwm_torque_gain
            # First-order lag models winding inductance / commutation delay --
            # the policy can't get instantaneous torque even if it wants to,
            # which is exactly the kind of actuator-level limit Stage 1 is
            # meant to expose.
            dt = 1.0 / self.control_hz
            alpha = dt / (self.pwm_time_const + dt)
            self._pwm_applied_torque += alpha * (target_torque - self._pwm_applied_torque)
            return self._pwm_applied_torque

    # ---------- helpers ----------

    def _ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.ee_site_id].copy()

    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos[: self.n_joints]
        qvel = self.data.qvel[: self.n_joints]
        rel_target = self.data.mocap_pos[self.target_id, :2] - self._ee_pos()[:2]
        ee_z = self._ee_pos()[2:3]
        return np.concatenate([qpos, qvel, rel_target, ee_z]).astype(np.float32)

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=480)
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self):
        pass
