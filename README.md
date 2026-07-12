# zip-autonomy-stack — Stage 1

Stage 1 of the autonomy-stack project: demonstrate the need for low-latency
control policies by measuring how control-autonomy level (torque / joint
position / PWM) and control frequency trade off against task performance and
compute cost, on a MuJoCo reaching-arm testbed.

## Repo layout

```
envs/
  assets/arm3dof.xml   3-DOF planar arm MJCF (dependency-free placeholder;
                        swap for a real manipulator model, e.g. MuJoCo
                        Menagerie's Franka Panda / UR5e, once validated)
  arm_env.py            Gymnasium env with switchable control_level
                        (torque / joint_position / pwm) and configurable
                        control_hz (decimated relative to physics_hz)
rl/
  train.py              PPO training entrypoint, config-driven
configs/
  torque.yaml
  joint_position.yaml
  pwm.yaml
scripts/
  measure_latency.py    Per-autonomy-level compute breakdown: inference vs.
                        action-mapping vs. physics-step time, checked against
                        the control-frequency budget
logs/                    Training runs + latency results land here
```

## Setup on beartic-srv (seshmi-dev container)

Two things need fixing in the container before this will run correctly on the
RTX 5080:

**1. PyTorch doesn't support sm_120 (Blackwell) yet on the 2.4.0 build in this
image.** `torch.cuda.is_available()` returns `True` but kernels will fail.
Install a nightly build against CUDA 12.8+:

```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

Verify with an actual kernel launch (not just `is_available()`):

```python
import torch
x = torch.randn(1000, 1000, device='cuda')
print((x @ x).sum())
```

**2. `git` isn't installed in the container:**

```bash
apt-get update && apt-get install -y git
```

Then install this repo's deps:

```bash
pip install -r requirements.txt
```

Note: `/workspace` in the container is bind-mounted from
`/home/seshmi/workspace` on the host but currently owned by `root` inside the
container. If you create files there as root you'll lose host-side write
access — either `chown -R seshmi:seshmi /home/seshmi/workspace` from the host,
or make sure your container user matches.

## Running

Train at a given autonomy level:

```bash
python rl/train.py --config configs/joint_position.yaml
python rl/train.py --config configs/torque.yaml
python rl/train.py --config configs/pwm.yaml
```

Each writes to `logs/<run_name>/` (model checkpoint + tensorboard log).

Measure the compute bottleneck for a trained policy:

```bash
python scripts/measure_latency.py \
  --model logs/joint_position_50hz/final_model.zip \
  --config configs/joint_position.yaml \
  --n-steps 2000
```

This reports mean/p50/p99 latency for inference, action-to-torque mapping, and
physics stepping separately, plus the max achievable control frequency implied
by each — checked against the `control_hz` budget in the config. This is the
number Stage 2's distillation work will need to beat.

## Case study: does this task actually need low-latency control?

`envs/velocity_reversal_env.py` adds a second task on top of the static
reaching env: the target now oscillates back and forth along a line, with its
oscillation frequency domain-randomized during training (so a policy learns
to handle a *range* of speeds, not one fixed speed). Critically, the target's
motion is driven by simulated wall-clock time (`self.data.time`), not by the
control loop -- so the physical scenario is identical across different
`control_hz` values. Only the policy's ability to keep up with it differs.

`scripts/run_stage1_sweep.py` is the script that actually produces Stage 1's
headline evidence. For each (control_level, control_hz) grid point it:

1. Trains a PPO policy on the velocity-reversal task.
2. Measures compute latency (same methodology as `measure_latency.py`).
3. Evaluates tracking error at a sweep of *test* oscillation frequencies
   (deterministic, no further learning) -- this is what should reveal a
   performance cliff once the target reverses faster than that control_hz /
   autonomy level can react to.

Output: `results/sweep_results.csv` and `results/sweep_plot.png` (tracking
error vs. target oscillation frequency, one line per grid point). The plot is
the actual case-study artifact -- if the lines for low control_hz start
diverging upward (worse tracking) at high oscillation frequencies while
high control_hz stays flat, that's the evidence motivating the rest of the
project.

```bash
# fast smoke test first (~1-2 min) -- confirms the pipeline runs, numbers
# will be meaningless (way too few training steps)
python scripts/run_stage1_sweep.py --quick-test

# real sweep -- this trains 12 policies (3 levels x 4 control_hz values) at
# 200k timesteps each by default. Expect a while even on the RTX 5080;
# start with a smaller grid/timesteps and scale up once you've confirmed
# the trend looks right.
python scripts/run_stage1_sweep.py \
  --control-levels torque joint_position pwm \
  --control-hz-list 10 25 50 100 \
  --timesteps 200000
```

## What's not built yet (next steps)

- **Real manipulator MJCF**: swap the placeholder 3-DOF planar arm for
  something like MuJoCo Menagerie's Franka Panda or UR5e once the sweep
  trend looks right on the toy arm -- `arm_env.py` doesn't hardcode link
  count, so this should mostly be a config change plus re-tuning PD gains.
- **Push the compute-bottleneck finding further**: the original latency
  numbers on this small 3-DOF arm + MLP policy showed huge headroom (36x
  margin at 50Hz on the RTX 5080) -- physics stepping dominated over
  inference. That headroom will shrink with a bigger policy / more DOF /
  higher control_hz; worth explicitly sweeping policy size too, since that's
  the actual proxy for "how much compute the eventual FPGA target needs to
  beat" in Stage 2.
- **CPU vs GPU latency comparison**: SB3 warns that PPO with a small MLP is
  "primarily intended to run on CPU" -- worth explicitly comparing
  `device='cpu'` vs `device='cuda'` in `measure_latency.py`/the sweep once
  policy sizes grow, rather than assuming GPU is always faster at this scale.
