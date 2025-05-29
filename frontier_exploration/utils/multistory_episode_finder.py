import hashlib
import os
import os.path as osp
from dataclasses import dataclass
from typing import Any

import numpy as np
from gym import Space, spaces
from habitat import EmbodiedTask, Sensor, SensorTypes, registry
from habitat.config.default_structured_configs import LabSensorConfig
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

HASH_KEYS = ["scene_id", "start_position", "start_rotation"]


@registry.register_sensor
class MultistoryEpisodeFinder(Sensor):

    cls_uuid: str = "multistory_episode_finder"

    def __init__(
        self, sim: HabitatSim, config: "DictConfig", *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(sim, config, *args, **kwargs)
        self._sim = sim
        self._config = config
        self._output_dir = config.output_dir
        for i in ["multi_story", "single_story"]:
            os.makedirs(osp.join(self._output_dir, i), exist_ok=True)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        return spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8)

    def get_observation(
        self, task: EmbodiedTask, episode, *args: Any, **kwargs: Any
    ) -> np.ndarray:
        hash_values = [getattr(episode, k) for k in HASH_KEYS]
        hash_values[0] = osp.basename(hash_values[0])
        hash_str = ":".join([str(i) for i in hash_values])
        basename = hashlib.sha224(hash_str.encode("ASCII")).hexdigest()

        story_type = (
            "multi_story" if self.episode_is_multistory(episode) else "single_story"
        )
        filename = f"{self._output_dir}/{story_type}/{basename}.txt"
        with open(filename, "w") as f:
            f.write(hash_str)

        task.is_stop_called = True
        return np.array([0], dtype=np.uint8)

    def episode_is_multistory(self, episode):
        """Generate a path to every view point in the episode and check if there aren't
        any paths on the same floor as the start point. If so, return True. Otherwise,
        return False.
        """
        goal_positions = [
            view_point.agent_state.position
            for goal in episode.goals
            for view_point in goal.view_points
        ]
        agent_position = self._sim.get_agent_state().position
        for g in goal_positions:
            # Skip this goal if it isn't even on the same floor as the agent
            if not self._is_on_same_floor(self._sim.pathfinder.snap_point(g)[1]):
                continue

            self._sim.geodesic_distance(agent_position, g, episode)
            pts = episode._shortest_path_cache.points
            if len(pts) >= 2:
                # Last point hasn't been snapepd yet
                pts = pts[:-1] + [self._sim.pathfinder.snap_point(pts[-1])]
                is_single_story = True
                for p in pts:
                    if not self._is_on_same_floor(p[1]):
                        is_single_story = False
                        break
                if is_single_story:
                    return False
        return True

    def _is_on_same_floor(self, height, ceiling_height=0.5):
        ref_floor_height = self._sim.get_agent(0).state.position[1]
        return ref_floor_height <= height < ref_floor_height + ceiling_height


@registry.register_sensor
class DummyExplorer(Sensor):
    cls_uuid: str = "base_explorer"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        return spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8)

    def get_observation(
        self, task: EmbodiedTask, episode, *args: Any, **kwargs: Any
    ) -> np.ndarray:
        return np.array([0], dtype=np.uint8)


@dataclass
class MultistoryEpisodeFinderSensorConfig(LabSensorConfig):
    type: str = MultistoryEpisodeFinder.__name__
    output_dir: str = "data/multistory_episodes"

@dataclass
class DummyExplorerSensorConfig(LabSensorConfig):
    type: str = DummyExplorer.__name__


cs = ConfigStore.instance()
cs.store(
    package=f"habitat.task.lab_sensors.{MultistoryEpisodeFinder.cls_uuid}",
    group="habitat/task/lab_sensors",
    name=f"{MultistoryEpisodeFinder.cls_uuid}",
    node=MultistoryEpisodeFinderSensorConfig,
)


cs = ConfigStore.instance()
cs.store(
    package=f"habitat.task.lab_sensors.dummy_explorer",
    group="habitat/task/lab_sensors",
    name=f"dummy_explorer",
    node=DummyExplorerSensorConfig,
)
