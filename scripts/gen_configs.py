import yaml, pathlib, itertools

BASE = pathlib.Path("configs")
LEVELS = ["torque", "joint_position", "pwm"]
HZS = [10, 20, 50, 100, 200]

for level, hz in itertools.product(LEVELS, HZS):
    src = BASE / f"{level}.yaml"
    cfg = yaml.safe_load(src.read_text())
    cfg["env"]["control_hz"] = hz
    run_name = f"{level}_{hz}hz"
    cfg["logging"]["run_name"] = run_name
    cfg["train"]["total_timesteps"] = 20000  # short — sweep is about latency, not final policy quality
    out = BASE / f"{run_name}.yaml"
    out.write_text(yaml.dump(cfg, sort_keys=False))
    print(f"wrote {out}")
