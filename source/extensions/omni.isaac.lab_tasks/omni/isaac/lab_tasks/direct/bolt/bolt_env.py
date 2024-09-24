from omni.isaac.lab_assets.bolt import BOLT_CFG  # isort: skip

import torch
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.sensors import (
    ContactSensor,
    ContactSensorCfg,
    RayCaster,
    RayCasterCfg,
    patterns,
)
from omni.isaac.core.utils.torch.rotations import (
    compute_heading_and_up,
    compute_rot,
    quat_conjugate,
    get_euler_xyz,
)
from omni.isaac.lab_tasks.direct.locomotion.locomotion_env import LocomotionEnv
from omni.isaac.lab.utils.math import euler_xyz_from_quat


OBSERVATION_DIMS = {
    "joint_pos": 6,
    "joint_vel": 6,
    "actions": 6,
    "projected_gravity": 2,
    "height": 1,
    "rotation": 3,
    "linear_velocity": 3,
    "angular_velocity": 3,
    "commands": 3,
}


def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


@configclass
class BoltEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 2
    action_scale = 1.0
    num_actions = 6
    num_observations = sum(OBSERVATION_DIMS.values())
    num_states = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16384, env_spacing=4.0, replicate_physics=True)
    robot: ArticulationCfg = BOLT_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    joint_gears: list = [
        9.0,
        9.0,
        9.0,
        9.0,
        9.0,
        9.0,
    ]

    lin_vel_reward_weight = 2.0
    yaw_rate_reward_weight = 0.5
    yaw_tracking_weight = 1.0
    angular_velocity_error = -0.5
    z_vel_reward_weight: float = -2.0
    energy_cost_weight: float = -0.00001
    action_rate_weight: float = -0.01
    alive_reward_weight: float = +2.0
    joint_accel_weight: float = -2.5e-6
    flat_orientation_weight: float = -1.0
    target_height_error_weight: float = -0.1
    feet_air_time_reward_scale = 2.0
    hip_deviation_weight = -0.2

    death_cost: float = -1.0
    termination_height: float = 0.2
    target_height = 0.35

    angular_velocity_scale: float = 0.25
    contact_force_scale: float = 0.01
    step_threshold: float = 0.6


class BoltEnv(DirectRLEnv):
    cfg: BoltEnvCfg

    def __init__(self, cfg: BoltEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # joint effort command
        self._actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        self._base_id, _ = self._contact_sensor.find_bodies("base_link")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*_FOOT")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*(?<!_FOOT)")

        self.joint_gears = torch.tensor(self.cfg.joint_gears, dtype=torch.float32, device=self.sim.device)
        self._joint_dof_idx, _ = self._robot.find_joints(".*")
        self._hip_joint_idx, _ = self._robot.find_joints(".*_HAA")

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                # "track_ang_vel_z_exp",
                "track_yaw_angle_l2",
                "lin_vel_z_l2",
                "energy_cost",
                "joint_accel",
                "action_rate",
                "flat_orientation",
                "alive_reward",
                "target_height_error_l2",
                "feet_air_time",
                "hip_deviation_l1",
                "minimize_angular_velocity",
            ]
        }

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()

    def _apply_action(self):
        forces = self.cfg.action_scale * self.joint_gears * self._actions
        self._robot.set_joint_effort_target(forces, joint_ids=self._joint_dof_idx)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        q = self._robot.data.root_quat_w
        roll, pitch, yaw = get_euler_xyz(q)

        height = self._robot.data.root_pos_w[:, 2].view(-1, 1)
        obs = torch.cat(
            [
                self._robot.data.joint_pos,
                self._robot.data.joint_vel,
                self._actions,
                self._robot.data.projected_gravity_b,
                height,
                normalize_angle(roll).unsqueeze(-1),
                normalize_angle(pitch).unsqueeze(-1),
                normalize_angle(yaw).unsqueeze(-1),
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b * self.cfg.angular_velocity_scale,
                self._commands,
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # linear velocity tracking
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)

        # yaw rate tracking
        # yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        # yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        yaw = euler_xyz_from_quat(self._robot.data.root_quat_w)[2]
        yaw_error = torch.square(self._commands[:, 2] - yaw)
        yaw_error_mapped = torch.exp(-yaw_error / 0.25)

        # minimize angular velocity
        angular_velocity = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)

        height = self._robot.data.root_pos_w[:, 2]
        target_height_error = torch.square(self.cfg.target_height - height)

        # z velocity tracking
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])

        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        # action rate (basically jerk)
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)

        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

        # feet air time
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        air_time = torch.sum((last_air_time - self.cfg.step_threshold) * first_contact, dim=1) * (
            torch.norm(self._commands[:, :2], dim=1) > 0.1
        )

        # hip deviation
        joint_deviation = torch.sum(
            torch.abs(
                self._robot.data.joint_pos[:, self._hip_joint_idx]
                - self._robot.data.default_joint_pos[:, self._hip_joint_idx]
            ),
            dim=1,
        )

        rewards = {
            "track_lin_vel_xy_exp": self.cfg.lin_vel_reward_weight * lin_vel_error_mapped * self.step_dt,
            # "track_ang_vel_z_exp": self.cfg.yaw_rate_reward_weight * yaw_rate_error_mapped * self.step_dt,
            "track_yaw_angle_l2": self.cfg.yaw_tracking_weight * yaw_error_mapped * self.step_dt,
            "lin_vel_z_l2": self.cfg.z_vel_reward_weight * z_vel_error * self.step_dt,
            "energy_cost": self.cfg.energy_cost_weight * joint_torques * self.step_dt,
            "joint_accel": self.cfg.joint_accel_weight * joint_accel * self.step_dt,
            "action_rate": self.cfg.action_rate_weight * action_rate * self.step_dt,
            "flat_orientation": self.cfg.flat_orientation_weight * flat_orientation * self.step_dt,
            "alive_reward": self.cfg.alive_reward_weight
            * torch.ones(self.num_envs, dtype=torch.float, device=self.device)
            * self.step_dt,
            "target_height_error_l2": self.cfg.target_height_error_weight * target_height_error * self.step_dt,
            "feet_air_time": self.cfg.feet_air_time_reward_scale * air_time * self.step_dt,
            "hip_deviation_l1": self.cfg.hip_deviation_weight * joint_deviation * self.step_dt,
            "minimize_angular_velocity": self.cfg.angular_velocity_error * angular_velocity * self.step_dt
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        height = self._robot.data.root_pos_w[:, 2]
        died = height < self.cfg.termination_height
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode Termination/height"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)
