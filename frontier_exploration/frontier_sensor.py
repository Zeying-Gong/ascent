from dataclasses import dataclass
from typing import Any

import numpy as np
from gym import Space, spaces
from habitat import EmbodiedTask, Sensor, SensorTypes, registry
from habitat.config.default_structured_configs import (
    HeadingSensorConfig,
    LabSensorConfig,
)
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.tasks.nav.nav import HeadingSensor
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from frontier_exploration.base_explorer import BaseExplorer

# from frontier_exploration.utils.path_utils import path_dist_cost
from frontier_exploration.utils.path_utils import path_time_cost


@registry.register_sensor
class FrontierSensor(Sensor):
    cls_uuid: str = "frontier_sensor"

    def __init__(
        self, sim: HabitatSim, config: "DictConfig", *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(sim, config, *args, **kwargs)
        self._curr_ep_id = None
        self.episodic_yaw = 0.0

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        return spaces.Box(
            low=-float("inf"), high=float("inf"), shape=(1, 2), dtype=np.float32
        )

    def get_observation(
        self, task: EmbodiedTask, episode, *args: Any, **kwargs: Any
    ) -> np.ndarray:
        """Return the 3D coordinates of each frontier."""
        if self._curr_ep_id != episode.episode_id:
            self._curr_ep_id = episode.episode_id
            heading_sensor: HeadingSensor = task.sensor_suite.sensors[  # type: ignore
                "heading"
            ]
            self.episodic_yaw = heading_sensor.get_observation(None, episode)[0]

        explorer_key = [
            k for k in task.sensor_suite.sensors.keys() if k.endswith("_explorer")
        ][0]

        explorer: BaseExplorer = task.sensor_suite.sensors[explorer_key]  # type: ignore

        if len(explorer.frontier_waypoints) == 0:
            return np.zeros((1, 2), dtype=np.float32)

        global_frontiers = explorer._pixel_to_map_coors(explorer.frontier_waypoints)

        # Sort the frontiers by completion time cost
        completion_times = []

        for frontier in global_frontiers:
            completion_times.append(
                path_time_cost(
                    frontier,
                    explorer.agent_position,
                    explorer.agent_heading,
                    explorer._lin_vel,
                    explorer._ang_vel,
                    explorer._sim,
                )
            )
            # completion_times.append(
            #     path_dist_cost(frontier, explorer.agent_position, explorer._sim)
            # )
        global_frontiers = global_frontiers[np.argsort(completion_times)]

        episode_origin = np.array(episode.start_position)

        episodic_frontiers = []
        for g_frontier in global_frontiers:
            pt = global_to_episodic_xy(episode_origin, self.episodic_yaw, g_frontier)
            episodic_frontiers.append(pt)
        episodic_frontiers = np.array(episodic_frontiers)

        return episodic_frontiers


def global_to_episodic_xy(episodic_start, episodic_yaw, pt):
    """
    All args are in Habitat format.
    """
    # Habitat to xy
    pt = np.array([pt[2], pt[0]])
    episodic_start = np.array([episodic_start[2], episodic_start[0]])

    rotation_matrix = np.array(
        [
            [np.cos(-episodic_yaw), -np.sin(-episodic_yaw)],
            [np.sin(-episodic_yaw), np.cos(-episodic_yaw)],
        ]
    )
    episodic_xy = -np.matmul(rotation_matrix, pt - episodic_start)

    return episodic_xy


@dataclass
class FrontierSensorConfig(LabSensorConfig):
    type: str = FrontierSensor.__name__


cs = ConfigStore.instance()
cs.store(
    package="habitat.task.lab_sensors.heading_sensor",
    group="habitat/task/lab_sensors",
    name="heading_sensor",
    node=HeadingSensorConfig,
)
cs.store(
    package=f"habitat.task.lab_sensors.{FrontierSensor.cls_uuid}",
    group="habitat/task/lab_sensors",
    name=f"{FrontierSensor.cls_uuid}",
    node=FrontierSensorConfig,
)
