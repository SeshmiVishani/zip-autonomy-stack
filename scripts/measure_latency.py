"""
measure_latency.py

Stage 1 deliverable: "Measure compute bottleneck and how much is needed to
close the gap."

Measures, per autonomy level, the wall-clock cost of a single control step
(policy forward pass + action-to-torque mapping), decomposed so you can see
which part dominates. This is the number that eventually gets compared against
your target control frequency (and, later, the FPGA-deployed ternary/binary
network's latency in Stage 2) -- the "gap" the project outline refers to.

Deliberately kept independent of SB3's rollout loop, since VecEnv batches and
hides per-step timing in ways that would obscure exactly the number we want.

Usage:
    python scripts/measure_latency.py --model logs/joint_position_50hz/final_model.zip \\
        --config configs/joint_position.yaml --n-steps 2000
"""

import argparse
import os
import sys
import time

import numpy as np
import yaml
from stable_baselines3 import PPO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs.arm_env import ArmReachEnv


def measure(env, model, n_steps: int):
    obs, _ = env.reset(seed=0)

    inference_times = []
    action_map_times = []
    physics_times = []
    total_times = []

    for _ in range(n_steps):
        t0 = time.perf_counter()
        action, _ = model.predict(obs, deterministic=True)
        t1 = time.perf_counter()

        # Replicate env.step()'s internals with fine-grained timing instead of
        # calling env.step() directly, so action-to-torque mapping and physics
        # stepping are timed separately.
        action = np.clip(action, env.action_space.low, env.action_space.high)
        t2 = time.perf_counter()
        for _ in range(env.decimation):
            torque = env._action_to_torque(action)
            env.data.ctrl[:] = np.clip(
                torque, env.torque_limits[:, 0], env.torque_limits[:, 1]
            )
            import mujoco
            mujoco.mj_step(env.model, env.data)
        t3 = time.perf_counter()

        obs = env._get_obs()

        inference_times.append(t1 - t0)
        action_map_times.append(t2 - t1)
        physics_times.append(t3 - t2)
        total_times.append(t3 - t0)

    return {
        "inference_ms": np.array(inference_times) * 1000,
        "action_map_ms": np.array(action_map_times) * 1000,
        "physics_ms": np.array(physics_times) * 1000,
        "total_ms": np.array(total_times) * 1000,
    }


def summarize(name: str, times_ms: np.ndarray, target_hz: float):
    budget_ms = 1000.0 / target_hz
    mean, p50, p99 = times_ms.mean(), np.percentile(times_ms, 50), np.percentile(times_ms, 99)
    max_achievable_hz = 1000.0 / mean if mean > 0 else float("inf")
    headroom = "OK" if p99 < budget_ms else "OVER BUDGET"
    print(f"  {name:14s} mean={mean:7.3f}ms  p50={p50:7.3f}ms  p99={p99:7.3f}ms  "
          f"max_hz={max_achievable_hz:8.1f}  budget={budget_ms:.3f}ms [{headroom}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--n-steps", type=int, default=2000)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    env_cfg = cfg["env"]

    env = ArmReachEnv(**env_cfg)
    model = PPO.load(args.model)

    print(f"[measure_latency] control_level={env_cfg['control_level']} "
          f"target control_hz={env_cfg['control_hz']} n_steps={args.n_steps}\n")

    results = measure(env, model, args.n_steps)
    target_hz = env_cfg["control_hz"]

    summarize("inference", results["inference_ms"], target_hz)
    summarize("action_map", results["action_map_ms"], target_hz)
    summarize("physics_step", results["physics_ms"], target_hz)
    summarize("TOTAL", results["total_ms"], target_hz)

    out_path = os.path.join(
        cfg["logging"]["log_dir"], cfg["logging"]["run_name"], "latency_results.npz"
    )
    np.savez(out_path, **results)
    print(f"\n[measure_latency] raw timings saved to {out_path}")


if __name__ == "__main__":
    main()
