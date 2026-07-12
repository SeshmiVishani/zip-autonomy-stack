import pathlib
import csv
import numpy as np
from stable_baselines3 import PPO
from envs.arm_track_env import ArmTrackReverseEnv

LEVELS = ["torque", "joint_position", "pwm"]
HZS = [10, 20, 50, 100, 200]
REVERSAL_PERIODS = [2.0, 1.0, 0.5, 0.25]
TRACK_SPEED = 0.3
N_EPISODES = 5
LOOKAHEAD_S = 0.1  # measure error at exactly this many seconds after each reversal

rows = []
for level in LEVELS:
    for hz in HZS:
        run_name = f"{level}_{hz}hz"
        model_path = pathlib.Path(f"logs/{run_name}/final_model.zip")
        if not model_path.exists():
            print(f"skip {run_name}: no model")
            continue
        model = PPO.load(model_path)
        lookahead_steps = max(1, round(LOOKAHEAD_S * hz))

        for period in REVERSAL_PERIODS:
            env = ArmTrackReverseEnv(
                control_level=level, control_hz=hz, physics_hz=500,
                reversal_period_s=period, track_speed=TRACK_SPEED,
            )
            post_reversal_errors = []
            for ep in range(N_EPISODES):
                obs, _ = env.reset(seed=ep)
                prev_phase_sign = 1
                for step in range(env.max_episode_steps):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, term, trunc, info = env.step(action)
                    phase = (env._t / period) % 1.0
                    sign = 1 if phase < 0.5 else -1
                    just_reversed = sign != prev_phase_sign
                    prev_phase_sign = sign
                    if just_reversed:
                        n_steps = min(lookahead_steps, env.max_episode_steps - step - 1)
                        if n_steps == lookahead_steps:
                            for _ in range(n_steps - 1):
                                a2, _ = model.predict(obs, deterministic=True)
                                obs, r2, t2, tr2, i2 = env.step(a2)
                            a2, _ = model.predict(obs, deterministic=True)
                            obs, r2, t2, tr2, i2 = env.step(a2)
                            post_reversal_errors.append(i2["dist"])
                    if term or trunc:
                        break

            mean_err = float(np.mean(post_reversal_errors)) if post_reversal_errors else float("nan")
            rows.append({
                "control_level": level, "control_hz": hz,
                "reversal_period_s": period,
                "lookahead_s": LOOKAHEAD_S,
                "error_at_lookahead_mean_m": mean_err,
                "n_reversals_sampled": len(post_reversal_errors),
            })
            print(f"{level:15s} {hz:4d}Hz period={period:.2f}s  "
                  f"err(t+{LOOKAHEAD_S*1000:.0f}ms)={mean_err:.4f}m  n={len(post_reversal_errors)}")

out = pathlib.Path("logs/case_study_reversal.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\nwrote {out}")
