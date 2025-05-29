# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from typing import Any, Dict, Union

import numpy as np
import torch
from depth_camera_filtering import filter_depth
from frontier_exploration.base_explorer import BaseExplorer
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.config.default_structured_configs import (
    PolicyConfig,
)
from habitat_baselines.rl.ppo.policy import PolicyActionData
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from torch import Tensor
from omegaconf import OmegaConf ## 
from vlfm.utils.geometry_utils import xyz_yaw_to_tf_matrix
from vlfm.vlm.grounding_dino import ObjectDetections

# from ..mapping.obstacle_map import ObstacleMap
from .base_objectnav_policy import BaseObjectNavPolicy, VLFMConfig
from .itm_policy import ITMPolicy, ITMPolicyV2, ITMPolicyV3

HM3D_ID_TO_NAME = ["chair", "bed", "potted plant", "toilet", "tv", "couch"]
MP3D_ID_TO_NAME = [
    "chair",
    "table|dining table|coffee table|side table|desk",  # "table",
    "framed photograph",  # "picture",
    "cabinet",
    "pillow",  # "cushion",
    "couch",  # "sofa",
    "bed",
    "nightstand",  # "chest of drawers",
    "potted plant",  # "plant",
    "sink",
    "toilet",
    "stool",
    "towel",
    "tv",  # "tv monitor",
    "shower",
    "bathtub",
    "counter",
    "fireplace",
    "gym equipment",
    "seating",
    "clothes",
]


class TorchActionIDs:
    STOP = torch.tensor([[0]], dtype=torch.long)
    MOVE_FORWARD = torch.tensor([[1]], dtype=torch.long)
    TURN_LEFT = torch.tensor([[2]], dtype=torch.long)
    TURN_RIGHT = torch.tensor([[3]], dtype=torch.long)


class HabitatMixin:
    """This Python mixin only contains code relevant for running a BaseObjectNavPolicy
    explicitly within Habitat (vs. the real world, etc.) and will endow any parent class
    (that is a subclass of BaseObjectNavPolicy) with the necessary methods to run in
    Habitat.
    """

    _stop_action: Tensor = TorchActionIDs.STOP
    _start_yaw: Union[float, None] = None  # must be set by _reset() method
    # _observations_cache: Dict[str, Any] = {}
    _policy_info: Dict[str, Any] = {}
    _compute_frontiers: bool = False

    def __init__(
        self,
        camera_height: float,
        min_depth: float,
        max_depth: float,
        camera_fov: float,
        image_width: int,
        dataset_type: str = "hm3d",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._camera_height = camera_height
        self._min_depth = min_depth
        self._max_depth = max_depth
        camera_fov_rad = np.deg2rad(camera_fov)
        self._camera_fov = camera_fov_rad
        self._fx = self._fy = image_width / (2 * np.tan(camera_fov_rad / 2))
        self._dataset_type = dataset_type

        self._observations_cache = [{} for _ in range(self._num_envs)]

    @classmethod
    def from_config(cls, config: DictConfig, *args_unused: Any, **kwargs_unused: Any) -> "HabitatMixin":
        policy_config: VLFMPolicyConfig = config.habitat_baselines.rl.policy[kwargs_unused['agent_name']] # For Habitat 3.0
        kwargs = {k: policy_config[k] for k in VLFMPolicyConfig.kwaarg_names}  # type: ignore
        ## adapt to multi envs 
        kwargs["num_envs"] = config.habitat_baselines.num_environments
        # In habitat, we need the height of the camera to generate the camera transform

        # sim_sensors_cfg = config.habitat.simulator.agents.main_agent.sim_sensors
        if 'main_agent' in config.habitat.simulator.agents:
            sim_sensors_cfg = config.habitat.simulator.agents['main_agent'].sim_sensors
            kwargs["camera_height"] = sim_sensors_cfg.rgb_sensor.position[1]

            # Synchronize the mapping min/max depth values with the habitat config
            kwargs["min_depth"] = sim_sensors_cfg.depth_sensor.min_depth
            kwargs["max_depth"] = sim_sensors_cfg.depth_sensor.max_depth
            kwargs["camera_fov"] = sim_sensors_cfg.depth_sensor.hfov
            kwargs["image_width"] = sim_sensors_cfg.depth_sensor.width
            kwargs["image_height"] = sim_sensors_cfg.depth_sensor.height # added
        elif 'agent_0' in config.habitat.simulator.agents:
            if config.habitat.simulator.agents['agent_0'].articulated_agent_type == "SpotRobot":
                sim_sensors_cfg = config.habitat.simulator.agents['agent_0'].sim_sensors
                kwargs["camera_height"] = sim_sensors_cfg.jaw_rgb_sensor.position[1]

                # Synchronize the mapping min/max depth values with the habitat config
                kwargs["min_depth"] = sim_sensors_cfg.jaw_depth_sensor.min_depth
                kwargs["max_depth"] = sim_sensors_cfg.jaw_depth_sensor.max_depth
                kwargs["camera_fov"] = sim_sensors_cfg.jaw_depth_sensor.hfov
                kwargs["image_width"] = sim_sensors_cfg.jaw_depth_sensor.width
                kwargs["image_height"] = sim_sensors_cfg.jaw_depth_sensor.height # added
            else:
                sim_sensors_cfg = config.habitat.simulator.agents['agent_0'].sim_sensors
                kwargs["camera_height"] = sim_sensors_cfg.rgb_sensor.position[1]

                # Synchronize the mapping min/max depth values with the habitat config
                kwargs["min_depth"] = sim_sensors_cfg.depth_sensor.min_depth
                kwargs["max_depth"] = sim_sensors_cfg.depth_sensor.max_depth
                kwargs["camera_fov"] = sim_sensors_cfg.depth_sensor.hfov
                kwargs["image_width"] = sim_sensors_cfg.depth_sensor.width
                kwargs["image_height"] = sim_sensors_cfg.depth_sensor.height # added
            # for spot
            kwargs["agent_radius"] = config.habitat.simulator.agents['agent_0'].radius
        else:
            raise ValueError("No agent found in configuration.")

        # Only bother visualizing if we're actually going to save the video
        kwargs["visualize"] = len(config.habitat_baselines.eval.video_option) > 0

        # For Habitat 3.0
        kwargs["action_space"]= args_unused[-1]

        if "hm3d" in config.habitat.dataset.data_path:
            kwargs["dataset_type"] = "hm3d"
        elif "mp3d" in config.habitat.dataset.data_path:
            kwargs["dataset_type"] = "mp3d"
        else:
            raise ValueError("Dataset type could not be inferred from habitat config")
        # config added
        kwargs["full_config"] = config
        return cls(**kwargs)

    def act(
        self: Union["HabitatMixin", BaseObjectNavPolicy],
        observations: TensorDict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> PolicyActionData:
        """Converts object ID to string name, returns action as PolicyActionData"""
        # Extract the object_ids, assuming observations[ObjectGoalSensor.cls_uuid] contains multiple values
        object_ids = observations[ObjectGoalSensor.cls_uuid] # .cpu().numpy().flatten()

        # Convert observations to dictionary format
        obs_dict = observations.to_tree()

        # Loop through each object_id and replace the goal IDs with corresponding names
        if self._dataset_type == "hm3d":
            obs_dict[ObjectGoalSensor.cls_uuid] = [HM3D_ID_TO_NAME[oid.item()] for oid in object_ids]
        elif self._dataset_type == "mp3d":
            obs_dict[ObjectGoalSensor.cls_uuid] = [MP3D_ID_TO_NAME[oid.item()] for oid in object_ids]
            self._non_coco_caption = " . ".join(MP3D_ID_TO_NAME).replace("|", " . ") + " ."
        else:
            raise ValueError(f"Dataset type {self._dataset_type} not recognized")
        
        parent_cls: BaseObjectNavPolicy = super()  # type: ignore
        # try:
        action, rnn_hidden_states = parent_cls.act(obs_dict, rnn_hidden_states, prev_actions, masks, deterministic)
        # except StopIteration:
        #     print("Check it!")
        return PolicyActionData(
            actions=action,
            rnn_hidden_states=rnn_hidden_states,
            policy_info=self._policy_info, # [self._policy_info],
        )

    def _initialize(self, env: int, masks: Tensor) -> Tensor:
        """Turn left 30 degrees 12 times to get a 360 view at the beginning"""
        self._done_initializing[env] = not self._num_steps[env] < 11  # type: ignore
        return TorchActionIDs.TURN_LEFT.to(masks.device)

    def _reset(self, env: int) -> None:
        parent_cls: BaseObjectNavPolicy = super()  # type: ignore
        parent_cls._reset(env)  # pass env
        # self._start_yaw = None

    def _get_policy_info(self, detections: ObjectDetections, env: int = 0) -> Dict[str, Any]:
        """Get policy info for logging"""
        parent_cls: BaseObjectNavPolicy = super()  # type: ignore
        info = parent_cls._get_policy_info(detections, env)

        if not self._visualize:  # type: ignore
            return info

        # if self._start_yaw is None:
        #     self._start_yaw = self._observations_cache["habitat_start_yaw"]
        info["start_yaw"] = self._observations_cache[env]["habitat_start_yaw"] # self._start_yaw
        return info

    def _cache_observations(self: Union["HabitatMixin", BaseObjectNavPolicy], observations: TensorDict, env: int) -> None:
        """Caches the rgb, depth, and camera transform from the observations.

        Args:
           observations (TensorDict): The observations from the current timestep.
        """
        if len(self._observations_cache[env]) > 0:
            return
        if "articulated_agent_jaw_depth" in observations:
            rgb = observations["articulated_agent_jaw_rgb"][env].cpu().numpy()
            depth = observations["articulated_agent_jaw_depth"][env].cpu().numpy()
            x, y = observations["gps"][env].cpu().numpy()
            camera_yaw = observations["compass"][env].cpu().item()
            depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
            # Habitat GPS makes west negative, so flip y
            camera_position = np.array([x, -y, self._camera_height])
            robot_xy = camera_position[:2]
            tf_camera_to_episodic = xyz_yaw_to_tf_matrix(camera_position, camera_yaw)

        else:
            rgb = observations["rgb"][env].cpu().numpy() ## modify this to fit on multiple environments
            depth = observations["depth"][env].cpu().numpy()
            x, y = observations["gps"][env].cpu().numpy()
            camera_yaw = observations["compass"][env].cpu().item()
            depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
            # Habitat GPS makes west negative, so flip y
            camera_position = np.array([x, -y, self._camera_height])
            robot_xy = camera_position[:2]
            tf_camera_to_episodic = xyz_yaw_to_tf_matrix(camera_position, camera_yaw)

        # self._obstacle_map: ObstacleMap # original obstacle map place

        self._observations_cache[env] = {
            # "frontier_sensor": frontiers,
            # "nav_depth": observations["depth"],  # for general depth
            "robot_xy": robot_xy,
            "robot_heading": camera_yaw,
            "object_map_rgbd": [
                (
                    rgb,
                    depth,
                    tf_camera_to_episodic,
                    self._min_depth,
                    self._max_depth,
                    self._fx,
                    self._fy,
                )
            ],
            "value_map_rgbd": [
                (
                    rgb,
                    depth,
                    tf_camera_to_episodic,
                    self._min_depth,
                    self._max_depth,
                    self._camera_fov,
                )
            ],
            "habitat_start_yaw": observations["heading"][env].item(),
            
        }

        if "articulated_agent_jaw_depth" in observations:
            self._observations_cache[env]["nav_depth"]=torch.unsqueeze(observations["articulated_agent_jaw_depth"][env], dim=0)
        else:
            self._observations_cache[env]["nav_depth"]=torch.unsqueeze(observations["depth"][env], dim=0)

        if "third_rgb" in observations:
            self._observations_cache[env]["third_rgb"]=observations["third_rgb"][env].cpu().numpy()


@baseline_registry.register_policy
class OracleFBEPolicy(HabitatMixin, BaseObjectNavPolicy):
    def _explore(self, observations: TensorDict) -> Tensor:
        explorer_key = [k for k in observations.keys() if k.endswith("_explorer")][0]
        pointnav_action = observations[explorer_key]
        return pointnav_action


@baseline_registry.register_policy
class SuperOracleFBEPolicy(HabitatMixin, BaseObjectNavPolicy):
    def act(
        self,
        observations: TensorDict,
        rnn_hidden_states: Any,  # can be anything because it is not used
        *args: Any,
        **kwargs: Any,
    ) -> PolicyActionData:
        return PolicyActionData(
            actions=observations[BaseExplorer.cls_uuid],
            rnn_hidden_states=rnn_hidden_states,
            policy_info=[self._policy_info],
        )


@baseline_registry.register_policy
class HabitatITMPolicy(HabitatMixin, ITMPolicy):
    pass


@baseline_registry.register_policy
class HabitatITMPolicyV2(HabitatMixin, ITMPolicyV2):
    pass

@baseline_registry.register_policy
class HabitatITMPolicyV3(HabitatMixin, ITMPolicyV3):
    pass


@dataclass
class VLFMPolicyConfig(VLFMConfig, PolicyConfig):
    pass


cs = ConfigStore.instance()
cs.store(group="habitat_baselines/rl/policy", name="vlfm_policy", node=VLFMPolicyConfig)

# For adapt to Habitat 3.0 (Multi-agent setup)
main_agent_vlfm_policy = {
    'main_agent': OmegaConf.structured(VLFMPolicyConfig())
}
agent_0_vlfm_policy = {
    'agent_0': OmegaConf.structured(VLFMPolicyConfig())
}

# For Habitat 3.0, replace vlfm_policy with main_agent_vlfm_policy, 
# the internal config actually remains
cs.store(group="habitat_baselines/rl/policy", name="main_agent_vlfm_policy", node=main_agent_vlfm_policy)
cs.store(group="habitat_baselines/rl/policy", name="agent_0_vlfm_policy", node=agent_0_vlfm_policy)
