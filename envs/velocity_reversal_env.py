"""
velocity_reversal_env.py
 
The Stage 1 case study: does the target actually need a low-latency policy?
 
ArmReachEnv (static target) doesn't answer this -- a static target has no
notion of "reaction time." This env makes the target oscillate back and forth
along a line, reversing direction periodically. The oscillation frequency
directly controls how much velocity/acceleration the arm has to track, and is
driven purely by simulated wall-clock time (self.data.time), NOT by the
control loop -- so the physical scenario is identical regardless of
control_hz. Only the policy's ability to *keep up* with it differs.
 
This is the mechanism that should reveal control-frequency / autonomy-level
sensitivity: a policy that only gets to act 10 times a second cannot react to
a target that reverses direction 8 times a second, no matter how well it's
trained, because it's fundamentally blind to what happens between its control
steps. A 200Hz policy can. That gap -- and where it starts to bite -- is
exactly the evidence Stage 1 is supposed to produce.
 
Domain randomization: oscillation_hz is resampled each episode within
[hz_min, hz_max] rather than fixed, so a single trained policy has to learn to
handle a *range* of speeds. This is what makes the later evaluation
(scripts/run_stage1_sweep.py) meaningful -- we evaluate the same policy across
a sweep of test oscillation frequencies and watch tracking error grow once the
frequency exceeds what that policy's control_hz can handle.
"""
 
import numpy as np
 
from envs.arm_env import ArmReachEnv
 
 
class VelocityReversalArmEnv(ArmReachEnv):
    def __init__(
        self,
        oscillation_hz_min: float = 0.3,
        oscillation_hz_max: float = 2.0,
        oscillation_amplitude: float = 0.22,
        target_radius: float = 0.45,
        eval_oscillation_hz: float = None,  # if set, overrides randomization (for eval sweeps)
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.oscillation_hz_min = oscillation_hz_min
        self.oscillation_hz_max = oscillation_hz_max
        self.oscillation_amplitude = oscillation_amplitude
        self.target_radius = target_radius
        self.eval_oscillation_hz = eval_oscillation_hz
 
        self._osc_hz = None
        self._osc_center = None
        self._osc_axis = None  # unit vector, tangential to the reach circle
        self._osc_phase0 = 0.0
 
        # extra observation channels: target velocity (2,) so the policy can
        # at least in principle anticipate the reversal, on top of position.
        # (ArmReachEnv's obs already includes qpos, qvel, rel_target, ee_z --
        # we append target velocity here.)
        low = np.concatenate([self.observation_space.low, [-np.inf, -np.inf]])
        high = np.concatenate([self.observation_space.high, [np.inf, np.inf]])
        from gymnasium import spaces
        self.observation_space = spaces.Box(low=low.astype(np.float32), high=high.astype(np.float32))
 
        self._episode_dists = []  # for reporting mean tracking error via info at episode end
 
    def reset(self, *, seed=None, options=None):
        # Reuse ArmReachEnv's reset for mujoco/model reset + randomized start
        # pose + RNG seeding, then immediately overwrite its static-target
        # logic with our oscillating target.
        _, _ = super().reset(seed=seed, options=options)
 
        self._osc_hz = (
            self.eval_oscillation_hz
            if self.eval_oscillation_hz is not None
            else self.np_random.uniform(self.oscillation_hz_min, self.oscillation_hz_max)
        )
        r = self.np_random.uniform(0.3, self.target_radius)
        theta = self.np_random.uniform(-np.pi, np.pi)
        self._osc_center = np.array([r * np.cos(theta), r * np.sin(theta)])
        # oscillate tangentially (perpendicular to the radius) so the motion
        # stays roughly within the arm's reachable annulus instead of moving
        # radially in/out of reach.
        tangent = np.array([-np.sin(theta), np.cos(theta)])
        self._osc_axis = tangent
        self._osc_phase0 = self.np_random.uniform(0, 2 * np.pi)
 
        self._episode_dists = []
        self._update_target()
        return self._get_obs(), {}
 
    def _target_pos_vel(self, t: float):
        """Analytic position/velocity of the oscillating target at sim time t."""
        omega = 2 * np.pi * self._osc_hz
        phase = omega * t + self._osc_phase0
        offset = self.oscillation_amplitude * np.sin(phase)
        vel_scalar = self.oscillation_amplitude * omega * np.cos(phase)
        pos = self._osc_center + offset * self._osc_axis
        vel = vel_scalar * self._osc_axis
        return pos, vel
 
    def _update_target(self):
        pos, vel = self._target_pos_vel(self.data.time)
        self.data.mocap_pos[self.target_id, :2] = pos
        self.data.mocap_pos[self.target_id, 2] = 0.05
        self._target_vel = vel
 
    def step(self, action: np.ndarray):
        action = np.clip(action, self.action_space.low, self.action_space.high)
 
        # Target position is updated every physics substep, driven by
        # self.data.time -- NOT by the control loop. This is what makes the
        # comparison across control_hz values fair: the physical scenario
        # (how fast the target actually moves) is identical, only the
        # policy's sampling rate of it differs.
        for _ in range(self.decimation):
            torque = self._action_to_torque(action)
            self.data.ctrl[:] = np.clip(
                torque, self.torque_limits[:, 0], self.torque_limits[:, 1]
            )
            import mujoco
            mujoco.mj_step(self.model, self.data)
            self._update_target()
 
        self._step_count += 1
        obs = self._get_obs()
        dist = np.linalg.norm(self._ee_pos()[:2] - self.data.mocap_pos[self.target_id, :2])
        effort = float(np.sum(np.square(self.data.ctrl)))
        self._episode_dists.append(dist)
 
        reward = (
            -self.reward_dist_weight * dist
            - self.reward_effort_weight * effort
        )
        terminated = False  # no early termination -- tracking is continuous
        truncated = self._step_count >= self.max_episode_steps
 
        info = {
            "dist": dist,
            "effort": effort,
            "control_level": self.control_level,
            "oscillation_hz": self._osc_hz,
        }
        if truncated:
            info["mean_tracking_dist"] = float(np.mean(self._episode_dists))
 
        return obs, reward, terminated, truncated, info
 
    def _get_obs(self) -> np.ndarray:
        base_obs = super()._get_obs()
        vel = getattr(self, "_target_vel", np.zeros(2))
        return np.concatenate([base_obs, vel]).astype(np.float32)
