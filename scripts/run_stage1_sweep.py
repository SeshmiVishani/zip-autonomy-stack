"""
run_stage1_sweep.py

This is the script that actually produces Stage 1's headline evidence:
"low-latency policies are necessary for reactive control." It ties together
everything built so far:

  1. TRAIN one PPO policy per (control_level, control_hz) grid point on the
     velocity-reversal case study, with oscillation_hz domain-randomized
     during training so each policy has to learn to handle a *range* of
     target speeds, not just one.
  2. MEASURE per-step compute latency for that policy (inference + action
     mapping + physics), same methodology as measure_latency.py.
  3. EVALUATE tracking error at a sweep of *test* oscillation frequencies
     (deterministic policy, no further learning) -- this is what reveals a
     performance cliff once the target reverses faster than a given
     control_hz / autonomy level can keep up with.

Output: results/sweep_results.csv (one row per grid point, one column per
test oscillation frequency) and results/sweep_plot.png (tracking error vs.
oscillation frequency, one line per grid point) -- the plot is the actual
case-study artifact.

Usage:
    # full sweep (slow -- see --quick-test for a fast smoke test first)
    python scripts/run_stage1_sweep.py

    # fast smoke test to confirm the whole pipeline runs end to end
    python scripts/run_stage1_sweep.py --quick-test

    # custom grid
    python scripts/run_stage1_sweep.py \\
        --control-levels torque joint_position pwm \\
        --control-hz-list 10 25 50 100 200 \\
        --timesteps 200000
"""

import argparse
import csv
import os
import sys

import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs import make_env_from_cfg
from rl.train import train_policy
from scripts.measure_latency import measure


def evaluate_tracking(env_cfg: dict, model, test_hz: float, n_episodes: int, seed0: int = 10_000):
    """Runs n_episodes deterministic episodes with oscillation_hz fixed at
    test_hz (no training happening here) and returns mean tracking distance
    across all of them. This is the number that should degrade once test_hz
    exceeds what the policy's control_hz can react to."""
    cfg = dict(env_cfg)
    cfg["eval_oscillation_hz"] = test_hz
    env = make_env_from_cfg(cfg)

    ep_means = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed0 + ep)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        ep_means.append(info["mean_tracking_dist"])
    env.close()
    return float(np.mean(ep_means))


def run_sweep(control_levels, control_hz_list, eval_hz_list, timesteps, n_envs,
              eval_episodes, latency_steps, osc_hz_min, osc_hz_max, log_dir, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    rows = []

    total_runs = len(control_levels) * len(control_hz_list)
    run_idx = 0

    for level in control_levels:
        for hz in control_hz_list:
            run_idx += 1
            print(f"\n{'=' * 70}\n[sweep] run {run_idx}/{total_runs}: "
                  f"control_level={level} control_hz={hz}\n{'=' * 70}")

            env_cfg = {
                "env_id": "velocity_reversal",
                "control_level": level,
                "control_hz": hz,
                "physics_hz": 500,
                "oscillation_hz_min": osc_hz_min,
                "oscillation_hz_max": osc_hz_max,
                "oscillation_amplitude": 0.22,
                "reward_dist_weight": 1.0,
                "reward_effort_weight": 0.001,
                "max_episode_steps": 300,
            }
            if level == "joint_position":
                env_cfg.update(pd_kp=8.0, pd_kd=0.8)
            elif level == "pwm":
                env_cfg.update(pwm_deadband=0.03, pwm_time_const=0.02)

            train_cfg = {
                "total_timesteps": timesteps,
                "n_envs": n_envs,
                "seed": 0,
                "policy": "MlpPolicy",
                "learning_rate": 3e-4,
                "n_steps": 512,
                "batch_size": 256,
            }

            run_dir = os.path.join(log_dir, f"sweep_{level}_{hz}hz")
            model_path = train_policy(
                env_cfg, train_cfg, run_dir,
                verbose=0, progress_bar=False,
            )
            model = PPO.load(model_path)

            # --- compute latency, same methodology as measure_latency.py ---
            latency_env = make_env_from_cfg(env_cfg)
            latency = measure(latency_env, model, latency_steps)
            latency_env.close()

            row = {
                "control_level": level,
                "control_hz": hz,
                "latency_mean_ms": float(latency["total_ms"].mean()),
                "latency_p99_ms": float(np.percentile(latency["total_ms"], 99)),
                "latency_max_hz": float(1000.0 / latency["total_ms"].mean()),
                "budget_ms": 1000.0 / hz,
            }
            print(f"[sweep] latency: mean={row['latency_mean_ms']:.3f}ms "
                  f"budget={row['budget_ms']:.3f}ms "
                  f"({'OK' if row['latency_mean_ms'] < row['budget_ms'] else 'OVER BUDGET'})")

            # --- tracking error across test oscillation frequencies ---
            for test_hz in eval_hz_list:
                err = evaluate_tracking(env_cfg, model, test_hz, eval_episodes)
                row[f"track_err_hz{test_hz}"] = err
                print(f"[sweep]   test_osc_hz={test_hz:5.2f}  mean_tracking_err={err:.4f}")

            rows.append(row)

    # --- write CSV ---
    csv_path = os.path.join(results_dir, "sweep_results.csv")
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[sweep] results written to {csv_path}")

    # --- plot: tracking error vs oscillation frequency, one line per grid point ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5.5))
        for row in rows:
            xs = eval_hz_list
            ys = [row[f"track_err_hz{h}"] for h in eval_hz_list]
            label = f"{row['control_level']} @ {row['control_hz']}Hz"
            ax.plot(xs, ys, marker="o", label=label)
        ax.set_xlabel("Test oscillation frequency (Hz) -- how fast the target reverses")
        ax.set_ylabel("Mean tracking error (m)")
        ax.set_title("Stage 1 case study: tracking error vs. target speed,\n"
                      "by control autonomy level and control frequency")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plot_path = os.path.join(results_dir, "sweep_plot.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"[sweep] plot written to {plot_path}")
    except ImportError:
        print("[sweep] matplotlib not installed -- skipping plot (CSV still written)")

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-levels", nargs="+", default=["torque", "joint_position", "pwm"])
    parser.add_argument("--control-hz-list", nargs="+", type=float, default=[10, 25, 50, 100])
    parser.add_argument("--eval-hz-list", nargs="+", type=float, default=[0.25, 0.5, 1, 2, 4, 8])
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--latency-steps", type=int, default=1000)
    parser.add_argument("--osc-hz-min", type=float, default=0.3)
    parser.add_argument("--osc-hz-max", type=float, default=2.0)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--quick-test", action="store_true",
                         help="tiny grid + timesteps, to smoke-test the pipeline in minutes")
    args = parser.parse_args()

    if args.quick_test:
        args.control_levels = ["joint_position"]
        args.control_hz_list = [20, 100]
        args.eval_hz_list = [0.5, 4]
        args.timesteps = 3000
        args.n_envs = 1
        args.eval_episodes = 1
        args.latency_steps = 200
        print("[sweep] --quick-test: using a tiny grid/timestep budget to verify "
              "the pipeline runs end to end. Results will NOT be scientifically "
              "meaningful -- rerun without --quick-test for real numbers.")

    run_sweep(
        control_levels=args.control_levels,
        control_hz_list=[int(h) for h in args.control_hz_list],
        eval_hz_list=args.eval_hz_list,
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        eval_episodes=args.eval_episodes,
        latency_steps=args.latency_steps,
        osc_hz_min=args.osc_hz_min,
        osc_hz_max=args.osc_hz_max,
        log_dir=args.log_dir,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
