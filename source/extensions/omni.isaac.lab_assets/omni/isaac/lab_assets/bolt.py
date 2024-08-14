"""Configuration for bolt robots.

The following configurations are available:

* :obj:`BOLT_CFG`: Bolt robot
"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

BOLT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/max/studium/current_course/project_ws/IsaacLab/source/extensions/omni.isaac.lab_assets/data/Robots/Bolt/bolt.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            "FL_HAA": 0.0,
            "FL_HFE": 0.5,
            "FL_KFE": -1.0,
            "FR_HAA": 0.0,
            "FR_HFE": 0.5,
            "FR_KFE": -1.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_HAA", ".*_HFE", ".*_KFE"],
            stiffness={
                ".*_HAA": 10.0,
                ".*_HFE": 20.0,
                ".*_KFE": 20.0,
            },
            damping={
                ".*_HAA": 3.0,
                ".*_HFE": 6.0,
                ".*_KFE": 6.0,
            },
        ),
    },
)
