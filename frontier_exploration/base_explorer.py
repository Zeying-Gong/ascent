import random
from dataclasses import dataclass
from typing import Any

import habitat_sim
import numpy as np
from gym import Space, spaces
from habitat import EmbodiedTask, Sensor, SensorTypes, registry
from habitat.config.default_structured_configs import LabSensorConfig
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.tasks.nav.nav import TopDownMap
from habitat.utils.visualizations import maps
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from frontier_exploration.frontier_detection import detect_frontier_waypoints
from frontier_exploration.utils.fog_of_war import reveal_fog_of_war
from frontier_exploration.utils.path_utils import (
    a_star_search,
    completion_time_heuristic,
    euclidean_heuristic,
    heading_error,
    path_dist_cost,
    path_time_cost,
)


class ActionIDs:
    STOP = np.array([0])
    MOVE_FORWARD = np.array([1])
    TURN_LEFT = np.array([2])
    TURN_RIGHT = np.array([3])


@registry.register_sensor
class BaseExplorer(Sensor):
    """Returns the action that moves the robot towards the closest frontier"""

    cls_uuid: str = "base_explorer"

    def __init__(
        self, sim: HabitatSim, config: "DictConfig", *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(sim, config, *args, **kwargs)
        self._sim = sim

        # Extract information from config
        self._config = config
        self._ang_vel = np.deg2rad(config.ang_vel)
        self._area_thresh = config.area_thresh
        self._forward_step_size = config.forward_step_size
        self._fov = config.fov
        self._lin_vel = config.lin_vel
        self._map_resolution = config.map_resolution
        self._minimize_time = config.minimize_time
        self._success_distance = config.success_distance
        self._turn_angle = np.deg2rad(config.turn_angle)
        self._visibility_dist = config.visibility_dist

        # Public attributes are used by the FrontierExplorationMap measurement
        self.closest_frontier_waypoint = None
        self.top_down_map = None
        self.fog_of_war_mask = None
        self.frontier_waypoints = np.array([])
        # Inflection is used by action inflection sensor for IL
        self.inflection = False
        self._prev_action = None

        self._area_thresh_in_pixels = None
        self._visibility_dist_in_pixels = None
        self._agent_position = None
        self._agent_heading = None
        self._curr_ep_id = None
        self._next_waypoint = None
        self._default_dir = None
        self._first_frontier = False  # whether frontiers have been found yet

    def _reset(self, *args, **kwargs):
        self.top_down_map = maps.get_topdown_map_from_sim(
            self._sim,
            map_resolution=self._map_resolution,
            draw_border=False,
        )
        self.fog_of_war_mask = np.zeros_like(self.top_down_map)
        self._area_thresh_in_pixels = self._convert_meters_to_pixel(
            self._area_thresh**2
        )
        self._visibility_dist_in_pixels = self._convert_meters_to_pixel(
            self._visibility_dist
        )
        self.closest_frontier_waypoint = None
        self._next_waypoint = None
        self._default_dir = None
        self._first_frontier = False
        self.inflection = False
        self._prev_action = None

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        return spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8)

    @property
    def agent_position(self):
        if self._agent_position is None:
            self._agent_position = self._sim.get_agent_state().position
        return self._agent_position

    @property
    def agent_heading(self):
        if self._agent_heading is None:
            try:
                # hablab v0.2.3
                self._agent_heading = TopDownMap.get_polar_angle(self)
            except AttributeError:
                # hablab v0.2.4
                self._agent_heading = TopDownMap.get_polar_angle(
                    self._sim.get_agent_state()
                )
        return self._agent_heading

    @property
    def next_waypoint_pixels(self):
        # This property is used by the FrontierExplorationMap measurement
        if self._next_waypoint is None:
            return None
        return self._map_coors_to_pixel(self._next_waypoint)

    def get_observation(
        self, task: EmbodiedTask, episode, *args: Any, **kwargs: Any
    ) -> np.ndarray:
        self._pre_step(episode)
        self._update_frontiers()
        self.closest_frontier_waypoint = self._get_closest_waypoint()
        action = self._decide_action(self.closest_frontier_waypoint)

        # Inflection is used by action inflection sensor for IL
        if self._prev_action is not None:
            self.inflection = self._prev_action != action
        self._prev_action = action

        return action

    def _pre_step(self, episode):
        self._agent_position, self._agent_heading = None, None
        if self._curr_ep_id != episode.episode_id:
            self._curr_ep_id = episode.episode_id
            self._reset(episode)  # New episode, reset maps

    def _update_fog_of_war_mask(self):
        orig = self.fog_of_war_mask.copy()
        self.fog_of_war_mask = reveal_fog_of_war(
            self.top_down_map,
            self.fog_of_war_mask,
            self._get_agent_pixel_coords(),
            self.agent_heading,
            fov=self._fov,
            max_line_len=self._visibility_dist_in_pixels,
        )
        updated = not np.array_equal(orig, self.fog_of_war_mask)
        return updated

    def _update_frontiers(self):
        updated = self._update_fog_of_war_mask()
        if updated:
            self.frontier_waypoints = detect_frontier_waypoints(
                self.top_down_map,
                self.fog_of_war_mask,
                self._area_thresh_in_pixels,
                # xy=self._get_agent_pixel_coords(),
            )
            if len(self.frontier_waypoints) > 0:
                # frontiers are in (y, x) format, need to do some swapping
                self.frontier_waypoints = self.frontier_waypoints[:, ::-1]

    def _get_next_waypoint(self, goal: np.ndarray):
        goal_3d = self._pixel_to_map_coors(goal) if len(goal) == 2 else goal
        next_waypoint = get_next_waypoint(
            self.agent_position, goal_3d, self._sim.pathfinder
        )
        return next_waypoint

    def _get_closest_waypoint(self):
        if len(self.frontier_waypoints) == 0:
            return None
        sim_waypoints = self._pixel_to_map_coors(self.frontier_waypoints)
        idx, _ = self._astar_search(sim_waypoints)
        if idx is None:
            return None

        return self.frontier_waypoints[idx]

    def _astar_search(self, sim_waypoints, start_position=None):
        if start_position is None:
            minimize_time = self._minimize_time
            start_position = self.agent_position
        else:
            minimize_time = False

        if minimize_time:

            def heuristic_fn(x):
                return completion_time_heuristic(
                    x,
                    self.agent_position,
                    self.agent_heading,
                    self._lin_vel,
                    self._ang_vel,
                )

            def cost_fn(x):
                return path_time_cost(
                    x,
                    self.agent_position,
                    self.agent_heading,
                    self._lin_vel,
                    self._ang_vel,
                    self._sim,
                )

        else:

            def heuristic_fn(x):
                return euclidean_heuristic(x, start_position)

            def cost_fn(x):
                return path_dist_cost(x, start_position, self._sim)

        return a_star_search(sim_waypoints, heuristic_fn, cost_fn)

    def _decide_action(self, target: np.ndarray) -> np.ndarray:
        if target is None:
            if not self._first_frontier:
                # If no frontiers have ever been found, likely just need to spin around
                if self._default_dir is None:
                    # Randomly select between LEFT or RIGHT
                    self._default_dir = bool(random.getrandbits(1))
                if self._default_dir:
                    return ActionIDs.TURN_LEFT
                else:
                    return ActionIDs.TURN_RIGHT
            # If frontiers have been found but now there are no more, stop
            return ActionIDs.STOP
        self._first_frontier = True

        # Get next waypoint along the shortest path towards the selected frontier
        # (target)
        self._next_waypoint = self._get_next_waypoint(target)

        # Determine which action is most suitable for reaching the next waypoint
        action = determine_pointturn_action(
            self.agent_position,
            self._next_waypoint,
            self.agent_heading,
            self._turn_angle,
        )

        return action

    def _get_agent_pixel_coords(self) -> np.ndarray:
        return self._map_coors_to_pixel(self.agent_position)

    def _convert_meters_to_pixel(self, meters: float) -> int:
        return int(
            meters
            / maps.calculate_meters_per_pixel(self._map_resolution, sim=self._sim)
        )

    def _pixel_to_map_coors(self, pixel: np.ndarray) -> np.ndarray:
        if pixel.ndim == 1:
            x, y = pixel
        else:
            x, y = pixel[:, 0], pixel[:, 1]
        realworld_x, realworld_y = maps.from_grid(
            x,
            y,
            (self.top_down_map.shape[0], self.top_down_map.shape[1]),
            self._sim,
        )
        if pixel.ndim == 1:
            return self._sim.pathfinder.snap_point(
                [realworld_y, self.agent_position[1], realworld_x]
            )
        snapped = [
            self._sim.pathfinder.snap_point([y, self.agent_position[1], x])
            for y, x in zip(realworld_y, realworld_x)  # noqa
        ]
        return np.array(snapped)

    def _map_coors_to_pixel(self, position) -> np.ndarray:
        a_x, a_y = maps.to_grid(
            position[2],
            position[0],
            (self.top_down_map.shape[0], self.top_down_map.shape[1]),
            sim=self._sim,
        )
        return np.array([a_x, a_y])


def get_next_waypoint(
    start: np.ndarray, goal: np.ndarray, pathfinder: "PathFinder"
) -> np.ndarray:
    shortest_path = habitat_sim.nav.ShortestPath()
    shortest_path.requested_start = start
    shortest_path.requested_end = goal
    assert pathfinder.find_path(shortest_path), "No path found!"
    next_waypoint = shortest_path.points[1]
    return next_waypoint


def determine_pointturn_action(
    start: np.ndarray,
    next_waypoint: np.ndarray,
    agent_heading: np.ndarray,
    turn_angle: float,
) -> np.ndarray:
    heading_err = heading_error(start, next_waypoint, agent_heading)
    if heading_err > turn_angle:
        return ActionIDs.TURN_RIGHT
    elif heading_err < -turn_angle:
        return ActionIDs.TURN_LEFT
    return ActionIDs.MOVE_FORWARD


@dataclass
class BaseExplorerSensorConfig(LabSensorConfig):
    type: str = BaseExplorer.__name__
    # minimum unexplored area (in meters) needed adjacent to a frontier for that
    # frontier to be valid
    ang_vel: float = 10.0  # degrees per second
    area_thresh: float = 3.0  # square meters
    forward_step_size: float = 0.25  # meters
    fov: int = 79
    lin_vel: float = 0.25  # meters per second
    map_resolution: int = 256
    minimize_time: bool = True
    success_distance: float = 0.18  # meters
    turn_angle: float = 10.0  # degrees
    visibility_dist: float = 4.5  # in meters


cs = ConfigStore.instance()
cs.store(
    package=f"habitat.task.lab_sensors.{BaseExplorer.cls_uuid}",
    group="habitat/task/lab_sensors",
    name=f"{BaseExplorer.cls_uuid}",
    node=BaseExplorerSensorConfig,
)
