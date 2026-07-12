"""
ArmTrackReverseEnv -- Stage 1 case study environment.

Target moves at a FIXED speed and reverses direction periodically. Unlike a
naive triangle wave, amplitude is NOT coupled to period here -- track_speed
(m/s) is held constant across all reversal_period_s conditions, so period
alone controls "how often does the policy have to react to a reversal,"
without also changing how far the target travels. This isolates reactivity
from task difficulty.
"""

import numpy as np
from envs.arm_env import ArmReachEnv


class ArmTrackReverseEnv(ArmReachEnv):
    def __init__(self, reversal_period_s: float = 1.0, track_speed: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.reversal_period_s = reversal_period_s
        self.track_speed = track_speed  # m/s, constant regardless of period
        self._track_center = np.zeros(2)
        self._track_axis = np.array([1.0, 0.0])
        self._t = 0.0

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._track_center = self.data.mocap_pos[self.target_id, :2].copy()
        theta = self.np_random.uniform(0, 2 * np.pi)
        self._track_axis = np.array([np.cos(theta), np.sin(theta)])
        self._t = 0.0
        return obs, info

    def _target_offset(self, t: float) -> np.ndarray:
        phase = (t / self.reversal_period_s) % 1.0
        tri = 4 * abs(phase - 0.5) - 1.0  # in [-1, 1]
        # amplitude fixed by speed alone: half a period of travel at track_speed
        amp = self.track_speed * (self.reversal_period_s / 2.0)
        return self._track_axis * tri * amp

    def step(self, action):
        self._t += 1.0 / self.control_hz
        self.data.mocap_pos[self.target_id, :2] = (
            self._track_center + self._target_offset(self._t)
        )
        return super().step(action)
