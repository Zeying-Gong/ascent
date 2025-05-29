from dataclasses import dataclass
from typing import Any

import numpy as np
from gym import Space, spaces
from habitat import EmbodiedTask, Sensor, SensorTypes, registry
from habitat.config.default_structured_configs import LabSensorConfig
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from frontier_exploration.base_explorer import BaseExplorer
from frontier_exploration.objnav_explorer import GreedyObjNavExplorer, ObjNavExplorer


@registry.register_sensor
class InflectionSensor(Sensor):
    cls_uuid = "inflection"

    def __init__(
        self, sim: HabitatSim, config: "DictConfig", *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(sim, config, *args, **kwargs)
        self.explorer = None

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        return spaces.Box(low=0, high=1, shape=(1,), dtype=np.bool)

    def get_observation(self, task: EmbodiedTask, *args: Any, **kwargs: Any) -> Any:
        if self.explorer is None:
            self.explorer = self._get_explorer(task)
        return np.array([self.explorer.inflection], dtype=np.bool)

    @staticmethod
    def _get_explorer(task: EmbodiedTask):
        for sensor in task.sensor_suite.sensors.values():
            if isinstance(sensor, (BaseExplorer, ObjNavExplorer, GreedyObjNavExplorer)):
                return sensor
        raise RuntimeError("No explorer found in sensor suite!")


@dataclass
class InflectionSensorConfig(LabSensorConfig):
    type: str = InflectionSensor.__name__


cs = ConfigStore.instance()
cs.store(
    package=f"habitat.task.lab_sensors.{InflectionSensor.cls_uuid}",
    group="habitat/task/lab_sensors",
    name=f"{InflectionSensor.cls_uuid}",
    node=InflectionSensorConfig,
)
