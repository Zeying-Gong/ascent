#!/usr/bin/env python3

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union
import math
import numpy as np
from gym import Space, spaces

from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import (
    AgentState,
    RGBSensor,
    Sensor,
    SensorTypes,
    ShortestPathPoint,
    Simulator,
)
from habitat.tasks.nav.nav import PointGoalSensor
from hydra.core.config_store import ConfigStore
import habitat_sim

from dataclasses import dataclass
from habitat.config.default_structured_configs import LabSensorConfig
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
if TYPE_CHECKING:
    from omegaconf import DictConfig

from habitat.tasks.rearrange.utils import UsesArticulatedAgentInterface

from frontier_exploration.base_explorer import (
    ActionIDs,
    BaseExplorer,
    BaseExplorerSensorConfig,
    determine_pointturn_action,
    get_next_waypoint,

)
from frontier_exploration.utils.path_utils import (
    a_star_search,
    completion_time_heuristic,
    euclidean_heuristic,
    heading_error,
    path_dist_cost,
    path_time_cost,
)
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.utils.visualizations import maps
import random
from habitat.tasks.nav.nav import TopDownMap
from habitat import EmbodiedTask
from frontier_exploration.frontier_detection import detect_frontier_waypoints
from frontier_exploration.utils.fog_of_war import reveal_fog_of_war

@dataclass
class OracleShortestPathSensorConfig(LabSensorConfig):
    
    type: str = "OracleShortestPathSensor"

@dataclass
class OracleShortestPathFollowerSensorConfig(LabSensorConfig):
    
    type: str = "OracleShortestPathFollowerSensor"

@dataclass
class DistanceToGoalSensorConfig(LabSensorConfig):
    
    type: str = "DistanceToGoalSensor"

@dataclass
class OracleFollowerSensorConfig(LabSensorConfig):
    
    type: str = "OracleFollowerSensor"


@dataclass
class HumanVelocitySensorConfig(LabSensorConfig):
    type: str = "HumanVelocitySensor"

@dataclass
class HumanNumSensorConfig(LabSensorConfig):
    type: str = "HumanNumSensor"
    max_num: int = 6

@dataclass
class RiskSensorConfig(LabSensorConfig):
    type: str = "RiskSensor"
    thres: float = 3.0
    use_geo_distance: bool = True

@dataclass
class SocialCompassSensorConfig(LabSensorConfig):
    type: str = "SocialCompassSensor"
    thres: float = 9.0
    num_bins: int = 8

@dataclass
class OracleHumanoidFutureTrajectorySensorConfig(LabSensorConfig):
    type: str = "OracleHumanoidFutureTrajectorySensor"
    future_step: int = 5

@dataclass
class MultiFloorExplorerSensorConfig(BaseExplorerSensorConfig):
    type: str = "MultiFloorExplorer"

@registry.register_sensor(name="OracleShortestPathSensor")
class OracleShortestPathSensor(Sensor):
    r"""Sensor that used for A* and ORCA
    """
    cls_uuid: str = "oracle_shortest_path_sensor"

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid
    
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (2,3)

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )
    
    def _path_to_point_2(self, point_a, point_b):
        """Get the shortest path between two points."""
        path = habitat_sim.ShortestPath()  # habitat_sim
        path.requested_start = point_a 
        path.requested_end = point_b
        found_path = self._sim.pathfinder.find_path(path)
        if found_path and len(path.points) >= 2:
            # Return the first two points of the path
            return np.array(path.points[:2], dtype=np.float32)
        else:
            # No valid path, return fallback
            return np.array([point_a, point_b], dtype=np.float32)
    
    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        agent_position = np.array(agent_state.position, dtype=np.float32)
        # rotation_world_agent = agent_state.rotation
        goal_position = np.array(episode.goals[0].position, dtype=np.float32)

        # return [agent_position, goal_position]
        path = self._path_to_point_2(
            agent_position, goal_position
        )
        return path

@registry.register_sensor(name="OracleShortestPathFollowerSensor")
class OracleShortestPathFollowerSensor(Sensor):
    r"""Sensor that used for A* and ORCA
    """
    cls_uuid: str = "oracle_shortest_path_follower_sensor"

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self.follower = ShortestPathFollower(
        sim=sim,  # Placeholder, will be set later
        return_one_hot = False,
        goal_radius=0.1,
    )
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid
    
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (1,)

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )
    
    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):

        return self.follower.get_next_action(episode.goals[0].position)
    
@registry.register_sensor(name="DistanceToGoalSensor")
class DistanceToGoalSensor(Sensor):
    r"""Sensor that used for A* and ORCA
    """
    cls_uuid: str = "distance_to_goal_sensor"

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._episode_view_points: Optional[
            List[Tuple[float, float, float]]
        ] = None
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid
    
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (1,)

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )
    
    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        agent_position = np.array(agent_state.position, dtype=np.float32)
        # rotation_world_agent = agent_state.rotation
        if hasattr(episode.goals[0], "view_points"): # objectnav
            self._episode_view_points = [
                    view_point.agent_state.position
                    for view_point in episode.goals[0].view_points
                ]
            distance_to_target = self._sim.geodesic_distance(
                agent_position, self._episode_view_points, episode
            )
        
        else: # pointnav
            goal_position = np.array(episode.goals[0].position, dtype=np.float32)
            distance_to_target = self._sim.geodesic_distance(
                agent_position,
                goal_position,
                episode,
            )
        if np.isnan(distance_to_target) or np.isinf(distance_to_target):
            distance_to_target = 100
        # return [agent_position, goal_position]
        return distance_to_target

@registry.register_sensor(name="OracleFollowerSensor")
class OracleFollowerSensor(PointGoalSensor):
    r"""Sensor that used for A* and ORCA
    """
    cls_uuid: str = "oracle_follower_sensor"
        
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid
    
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (2,)

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )
    
    def _path_to_point_1(self, point_a, point_b):
        """Get the shortest path between two points."""
        path = habitat_sim.ShortestPath()  # habitat_sim
        path.requested_start = point_a 
        path.requested_end = point_b
        found_path = self._sim.pathfinder.find_path(path)
        return path.points[1] if found_path else [point_b]
    
    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position
        rotation_world_agent = agent_state.rotation
        goal_position = np.array(episode.goals[0].position, dtype=np.float32)

        return self._compute_pointgoal(
            agent_position, rotation_world_agent, self._path_to_point_1(agent_position,goal_position)
        )

@registry.register_sensor
class HumanVelocitySensor(UsesArticulatedAgentInterface, Sensor):
    """
    The position and angle of the articulated_agent in world coordinates.
    """

    cls_uuid = "human_velocity_sensor"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim
        self.value = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 6, dtype=np.float64)

    def _get_uuid(self, *args, **kwargs):
        return HumanVelocitySensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(6,6),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        # human_num = kwargs["task"]._human_num
        for i in range(self._sim.num_articulated_agents-1):
            articulated_agent = self._sim.get_agent_data(i+1).articulated_agent
            human_pos = np.array(articulated_agent.base_pos, dtype=np.float64)
            human_rot = np.array([float(articulated_agent.base_rot)], dtype=np.float64)
            human_vel = np.array(kwargs['task'].measurements.measures['human_velocity_measure']._metric[i],dtype=np.float64)
            self.value[i] = np.concatenate((human_pos, human_rot, human_vel))
        return self.value
    
@registry.register_sensor
class HumanNumSensor(UsesArticulatedAgentInterface, Sensor):
    """
    The num of the other agent in world.
    (in our setup, agents except agent_0 are humanoids)
    """

    cls_uuid = "human_num_sensor"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args, **kwargs):
        return HumanNumSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(1,), low=0, high=6, dtype=np.int32
        )

    def get_observation(self, observations, episode, *args, **kwargs):    
        if "human_num" in episode.info:
            human_num = min(episode.info['human_num'], 6)
        else:
            human_num = min(self._sim.num_articulated_agents - 1, 6)
        # Ensure the returned value is a tensor with shape (1,)
        return np.array([human_num], dtype=np.int32)

@registry.register_sensor
class RiskSensor(UsesArticulatedAgentInterface, Sensor):
    r"""Sensor for observing social risk to which the agent is subjected".

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """
    cls_uuid: str = "risk_sensor"

    def __init__(
        self, sim, config, *args, **kwargs
    ):
        self._sim = sim
        self._robot_idx = 0
        self.thres = config.thres
        self._use_geo_distance = config.use_geo_distance
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.MEASUREMENT

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

    def get_observation(
        self, observations, episode, *args, **kwargs
    ):
        self._human_nums = min(episode.info['human_num'], self._sim.num_articulated_agents - 1)
        if self._human_nums == 0:
            return np.array([0], dtype=np.float32)
        else:
            robot_pos = self._sim.get_agent_state(0).position

            human_pos = []
            human_dis = []

            for i in range(self._human_nums):
                human_position = self._sim.get_agent_state(i+1).position
                human_pos.append(human_position)

                if self._use_geo_distance:
                    path = habitat_sim.ShortestPath()
                    path.requested_start = robot_pos
                    path.requested_end = human_position
                    found_path = self._sim.pathfinder.find_path(path)

                    if found_path:
                        distance = self._sim.geodesic_distance(robot_pos, human_position)
                    else:
                        distance = np.linalg.norm(human_position - robot_pos, ord=2)
                else:
                    distance = np.linalg.norm(human_position - robot_pos, ord=2)

                human_dis.append(distance)

            return np.array([max(1 - min(human_dis) / self.thres, 0)],
                            dtype=np.float32)

@registry.register_sensor
class SocialCompassSensor(UsesArticulatedAgentInterface, Sensor):
    r"""
    Implementation of people relative position sensor
    """

    cls_uuid: str = "social_compass_sensor"

    def __init__(
        self, sim, config, *args, **kwargs
    ):
        self._sim = sim
        # parameters
        self.thres = config.thres
        self.num_bins = config.num_bins
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        return self.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.MEASUREMENT

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(low=0, high=np.inf, shape=(self.num_bins,),
                          dtype=np.float32)

    def get_polar_angle(self, agent_id = 0):
        agent_state = self._sim.get_agent_state(agent_id)
        # quaternion is in x, y, z, w format
        ref_rotation = agent_state.rotation

        heading_vector = quaternion_rotate_vector(
            ref_rotation.inverse(), np.array([0, 0, -1])
        )

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        z_neg_z_flip = np.pi
        return np.array(phi) + z_neg_z_flip
    
    def get_heading_error(self, source, target):
        r"""Computes the difference between two headings (radians); can be negative
        or positive.
        """
        diff = target - source
        if diff > np.pi:
            diff -= np.pi*2
        elif diff < -np.pi:
            diff += np.pi*2
        return diff
    
    def get_observation(self, observations, episode, *args, **kwargs):
        self._human_nums = min(episode.info['human_num'], self._sim.num_articulated_agents - 1)
        angles = [0] * self.num_bins
        if self._human_nums == 0:
            return np.array(angles, dtype=np.float32)
        else:
            a_pos = self._sim.get_agent_state(0).position
            a_head = self._sim.get_agent_state(0).rotation  # 2*np.arccos(self._sim.get_agent_state().rotation.w)

            a_head = -self.get_polar_angle(0) + np.pi / 2  # -quat_to_rad(a_head) + np.pi / 2

            for i in range(self._human_nums):
                pos = self._sim.get_agent_state(i+1).position
                theta = math.atan2(pos[2] - a_pos[2], pos[0] - a_pos[0])
                theta = self.get_heading_error(a_head, theta)
                theta = theta if theta > 0 else 2 * np.pi + theta

                bin = int(theta / (2 * np.pi / self.num_bins))

                dist = np.sqrt((pos[2] - a_pos[2]) ** 2 + (pos[0] - a_pos[
                    0]) ** 2)  # self._sim.geodesic_distance(a_pos, pos)
                norm_dist = max(1 - dist / self.thres, 0)
                if norm_dist > angles[bin]:
                    angles[bin] = norm_dist

            return np.array(angles, dtype=np.float32)

@registry.register_sensor
class OracleHumanoidFutureTrajectorySensor(UsesArticulatedAgentInterface, Sensor):
    """
    Assumed Oracle Humanoid Future Trajectory Sensor.
    """

    cls_uuid: str = "oracle_humanoid_future_trajectory"

    def __init__(self, *args, sim, task, **kwargs):
        self._sim = sim
        self._task = task
        self.future_step = kwargs['config']['future_step'] 
        self.max_human_num = 6
        self.human_num = task._human_num
        self.result_list = None  

        super().__init__(*args, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return OracleHumanoidFutureTrajectorySensor.cls_uuid

    @staticmethod
    def _get_sensor_type(*args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(self.max_human_num, self.future_step, 2),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    @staticmethod
    def _initialize_result_list(human_num, future_step, max_human_num):
        """Initialize the result list with default values."""
        result = np.full((max_human_num, future_step, 2), -100, dtype=np.float32)
        return result

    def get_observation(self, task, *args, **kwargs):
        human_num = self._task._human_num

        if self.result_list is None or human_num != self.human_num:
            self.result_list = self._initialize_result_list(human_num, self.future_step, self.max_human_num)
            self.human_num = human_num
        
        if self.human_num == 0:
            return self.result_list
        
        human_future_trajectory = task.measurements.measures.get("human_future_trajectory")._metric
        if not human_future_trajectory:
            return self.result_list

        robot_pos = np.array(self._sim.get_agent_data(0).articulated_agent.base_pos)[[0, 2]]

        for key, trajectories in human_future_trajectory.items():
            trajectories = np.array(trajectories)
            trajectories = trajectories.astype('float32')
            self.result_list[key - 1, :len(trajectories), :] = (trajectories[:, [0, 2]] - robot_pos)

        return self.result_list.tolist()

@registry.register_sensor
class MultiFloorExplorer(BaseExplorer):
    """Returns the action that moves the robot towards the closest frontier"""

    cls_uuid: str = "multi_floor_explorer"

    def __init__(
        self, sim: HabitatSim, config: "DictConfig", *args: Any, **kwargs: Any
    ) -> None:
        ### For Sensor
        self.config = kwargs["config"] if "config" in kwargs else None
        if hasattr(self.config, "uuid"):
            # We allow any sensor config to override the uuid
            self.uuid = self.config.uuid
        else:
            self.uuid = self._get_uuid(*args, **kwargs)
        self.sensor_type = self._get_sensor_type(*args, **kwargs)
        self.observation_space = self._get_observation_space(*args, **kwargs)

        ### Adjust by BaseExplorer
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

cs = ConfigStore.instance()

cs.store(
    package="habitat.task.lab_sensors.distance_to_goal_sensor",
    group="habitat/task/lab_sensors",
    name="distance_to_goal_sensor",
    node=DistanceToGoalSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.oracle_shortest_path_sensor",
    group="habitat/task/lab_sensors",
    name="oracle_shortest_path_sensor",
    node=OracleShortestPathSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.oracle_shortest_path_follower_sensor",
    group="habitat/task/lab_sensors",
    name="oracle_shortest_path_follower_sensor",
    node=OracleShortestPathFollowerSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.oracle_follower_sensor",
    group="habitat/task/lab_sensors",
    name="oracle_follower_sensor",
    node=OracleFollowerSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.human_velocity_sensor",
    group="habitat/task/lab_sensors",
    name="human_velocity_sensor",
    node=HumanVelocitySensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.human_num_sensor",
    group="habitat/task/lab_sensors",
    name="human_num_sensor",
    node=HumanNumSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.risk_sensor",
    group="habitat/task/lab_sensors",
    name="risk_sensor",
    node=RiskSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.social_compass_sensor",
    group="habitat/task/lab_sensors",
    name="social_compass_sensor",
    node=SocialCompassSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.oracle_humanoid_future_trajectory",
    group="habitat/task/lab_sensors",
    name="oracle_humanoid_future_trajectory",
    node=OracleHumanoidFutureTrajectorySensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.multi_floor_explorer",
    group="habitat/task/lab_sensors",
    name="multi_floor_explorer",
    node=MultiFloorExplorerSensorConfig,
)