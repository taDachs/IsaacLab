import gymnasium as gym

from . import agents, bipedal_env_cfg
##
# Register Gym environments.
##

gym.register(
    id="Isaac-Bipedal-Flat-Bolt-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": bipedal_env_cfg.BoltFlatEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.BoltFlatPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Bipedal-Flat-Bolt-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": bipedal_env_cfg.BoltFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.BoltFlatPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Bipedal-Rough-Bolt-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": bipedal_env_cfg.BoltRoughEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.BoltRoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Bipedal-Rough-Bolt-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": bipedal_env_cfg.BoltRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.BoltRoughPPORunnerCfg,
    },
)
