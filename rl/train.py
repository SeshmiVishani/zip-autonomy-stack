"""
train.py

Trains a PPO policy on a registered env (see envs/__init__.py: ENV_REGISTRY)
at a given autonomy level / control frequency, as specified by a config YAML
(see configs/*.yaml).

Usage:
    python rl/train.py --config configs/joint_position.yaml
    python rl/train.py --config configs/torque.yaml --total-timesteps 100000  # override

train_policy() is factored out separately from the CLI so scripts/run_stage1_sweep.py
can call it directly across a grid of configs without shelling out per run.

This is intentionally SB3/PPO for now -- it's the fastest path to "does the
training pipeline work end to end" for Stage 1. It is not the final algorithm
choice; swap in a custom PPO/SAC implementation later if you need tighter
control over the rollout loop (e.g. to instrument per-step latency directly,
which SB3's VecEnv abstraction partially hides -- see scripts/measure_latency.py
for latency profiling done outside of SB3 for that reason).
"""

import argparse
import os
import sys

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs import make_env_from_cfg


def make_env(env_cfg: dict, seed: int, rank: int):
    def _init():
        env = make_env_from_cfg(env_cfg)
        env.reset(seed=seed + rank)
        return Monitor(env)
    return _init


def train_policy(env_cfg: dict, train_cfg: dict, run_dir: str,
                  total_timesteps: int = None, n_envs: int = None,
                  verbose: int = 1, progress_bar: bool = True) -> str:
    """Trains one PPO policy and returns the path to the saved model.zip.
    Shared by the CLI entrypoint (main()) and scripts/run_stage1_sweep.py."""
    total_timesteps = total_timesteps or train_cfg["total_timesteps"]
    n_envs = n_envs or train_cfg["n_envs"]
    seed = train_cfg.get("seed", 0)

    os.makedirs(run_dir, exist_ok=True)

    # SubprocVecEnv parallelizes rollout collection across CPU cores -- important
    # here because MuJoCo stepping, not GPU inference, is the rollout bottleneck
    # at this scale. Falls back to DummyVecEnv for n_envs=1 (easier debugging).
    if n_envs > 1:
        vec_env = SubprocVecEnv([make_env(env_cfg, seed, i) for i in range(n_envs)])
    else:
        vec_env = DummyVecEnv([make_env(env_cfg, seed, 0)])

    model = PPO(
        train_cfg.get("policy", "MlpPolicy"),
        vec_env,
        learning_rate=train_cfg.get("learning_rate", 3e-4),
        n_steps=train_cfg.get("n_steps", 512),
        batch_size=train_cfg.get("batch_size", 256),
        seed=seed,
        verbose=verbose,
        tensorboard_log=run_dir,
    )

    print(f"[train] env_id={env_cfg.get('env_id', 'arm_reach')} "
          f"control_level={env_cfg['control_level']} "
          f"control_hz={env_cfg['control_hz']} n_envs={n_envs} "
          f"total_timesteps={total_timesteps}")

    model.learn(total_timesteps=total_timesteps, progress_bar=progress_bar)

    save_path = os.path.join(run_dir, "final_model.zip")
    model.save(save_path)
    print(f"[train] saved model to {save_path}")
    vec_env.close()
    return save_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--total-timesteps", type=int, default=None, help="override config")
    parser.add_argument("--n-envs", type=int, default=None, help="override config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    env_cfg = cfg["env"]
    train_cfg = cfg["train"]
    log_cfg = cfg["logging"]
    run_dir = os.path.join(log_cfg["log_dir"], log_cfg["run_name"])

    train_policy(env_cfg, train_cfg, run_dir,
                 total_timesteps=args.total_timesteps, n_envs=args.n_envs)


if __name__ == "__main__":
    main()
