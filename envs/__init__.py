"""
envs/__init__.py

Env registry: config files pick an env by string ID (env_cfg["env_id"]) rather
than hardcoding an import, so rl/train.py and the sweep script don't need to
know about specific env classes.
"""

from envs.arm_env import ArmReachEnv
from envs.velocity_reversal_env import VelocityReversalArmEnv

ENV_REGISTRY = {
    "arm_reach": ArmReachEnv,
    "velocity_reversal": VelocityReversalArmEnv,
}


def make_env_from_cfg(env_cfg: dict):
    """env_cfg is a dict as loaded from config YAML's `env:` block. Pops
    'env_id' (defaulting to 'arm_reach' for backward compatibility with the
    Stage 1 configs that predate this registry) and forwards the rest as
    kwargs to the selected env class."""
    cfg = dict(env_cfg)  # don't mutate caller's dict
    env_id = cfg.pop("env_id", "arm_reach")
    if env_id not in ENV_REGISTRY:
        raise ValueError(f"Unknown env_id '{env_id}'. Options: {list(ENV_REGISTRY)}")
    return ENV_REGISTRY[env_id](**cfg)
