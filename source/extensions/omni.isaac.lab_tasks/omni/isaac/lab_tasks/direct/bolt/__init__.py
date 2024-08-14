"""
Bolt locomotion environment.
"""

import gymnasium as gym

from . import agents
from .bolt_env import BoltEnv, BoltEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Bolt-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.bolt:BoltEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BoltEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.BoltPPORunnerCfg,
    },
)
