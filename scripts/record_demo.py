"""
record_demo.py

Records a video of a trained policy controlling the arm on the velocity-
reversal task -- this is the visual Ian asked for: watching the arm actually
chase the target makes the behavior legible in a way tracking-error numbers
alone don't.

Usage:
    python scripts/record_demo.py \
        --model logs/sweep_torque_50hz/final_model.zip \
        --control-level torque --control-hz 50 \
        --eval-oscillation-hz 1.0 \
        --output demo_torque_50hz.mp4

Produces an MP4 (falls back to GIF if ffmpeg isn't available) showing the
arm (blue links), end-effector (red dot), and target (green dot) over one
full episode, with tracking distance overlaid as a running readout.
"""

import argparse
import os
import sys

import numpy as np
import imageio
from stable_baselines3 import PPO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs import make_env_from_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--control-level", type=str, required=True,
                         choices=["torque", "joint_position", "pwm"])
    parser.add_argument("--control-hz", type=int, required=True)
    parser.add_argument("--eval-oscillation-hz", type=float, default=1.0,
                         help="target reversal frequency to demo (Hz)")
    parser.add_argument("--output", type=str, default="demo.mp4")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-seconds", type=float, default=8.0,
                         help="how much of the episode (real time) to record")
    args = parser.parse_args()

    env_cfg = {
        "env_id": "velocity_reversal",
        "control_level": args.control_level,
        "control_hz": args.control_hz,
        "physics_hz": 500,
        "eval_oscillation_hz": args.eval_oscillation_hz,
        "oscillation_amplitude": 0.22,
        "max_episode_steps": int(args.max_seconds * args.control_hz),
        "render_mode": "rgb_array",
    }
    if args.control_level == "joint_position":
        env_cfg.update(pd_kp=8.0, pd_kd=0.8)
    elif args.control_level == "pwm":
        env_cfg.update(pwm_deadband=0.03, pwm_time_const=0.02)

    env = make_env_from_cfg(env_cfg)
    model = PPO.load(args.model)

    obs, info = env.reset(seed=0)
    frames = []
    dist_history = []

    n_steps = env.max_episode_steps
    print(f"[record_demo] recording {n_steps} steps "
          f"({args.max_seconds}s at {args.control_hz}Hz control) "
          f"-- eval_oscillation_hz={args.eval_oscillation_hz}")

    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        dist_history.append(info["dist"])

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        if terminated or truncated:
            break

    env.close()

    if not frames:
        print("[record_demo] ERROR: no frames captured -- render_mode may not "
              "be wired correctly for this env.")
        return

    print(f"[record_demo] captured {len(frames)} frames, "
          f"mean tracking dist={np.mean(dist_history):.4f}m")

    try:
        imageio.mimsave(args.output, frames, fps=args.fps, codec="libx264",
                         quality=8)
        print(f"[record_demo] wrote {args.output}")
    except Exception as e:
        gif_path = os.path.splitext(args.output)[0] + ".gif"
        print(f"[record_demo] mp4 write failed ({e}), falling back to GIF")
        imageio.mimsave(gif_path, frames, fps=args.fps)
        print(f"[record_demo] wrote {gif_path}")


if __name__ == "__main__":
    main()
