#!/usr/bin/env python3

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, List, Tuple, Dict


import numpy as np
from gym import spaces

from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat.core.dataset import Dataset, Episode

from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.tasks.rearrange.utils import UsesArticulatedAgentInterface
from habitat.tasks.nav.nav import Success, TopDownMap, HeadingSensor, NavigationEpisode
from habitat.tasks.nav.object_nav_task import ObjectGoalNavEpisode
from hydra.core.config_store import ConfigStore
import habitat_sim
from habitat.tasks.rearrange.rearrange_sensors import NumStepsMeasure
from dataclasses import dataclass
from habitat.config.default_structured_configs import MeasurementConfig, TopDownMapMeasurementConfig

from habitat.tasks.rearrange.utils import rearrange_collision
from habitat.core.embodied_task import Measure
from habitat.tasks.rearrange.social_nav.utils import (
    robot_human_vec_dot_product,
)
from habitat.tasks.nav.nav import DistanceToGoalReward, DistanceToGoal
from habitat.tasks.rearrange.utils import coll_name_matches
try:
    import magnum as mn
except ImportError:
    pass

# if TYPE_CHECKING:
from omegaconf import DictConfig

from habitat.tasks.rearrange.articulated_agent_manager_v2 import ArticulatedAgentData
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.utils.visualizations import fog_of_war, maps
import matplotlib.pyplot as plt
from collections import Counter
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from habitat.core.utils import not_none_validator, try_cv2_import
cv2 = try_cv2_import()

from frontier_exploration.base_explorer import BaseExplorer
from frontier_exploration.objnav_explorer import GreedyObjNavExplorer, ObjNavExplorer
from habitat import EmbodiedTask
from frontier_exploration.utils.general_utils import habitat_to_xyz
from frontier_exploration.measurements import FrontierExplorationMap, FrontierExplorationMapMeasurementConfig
from habitat.utils.visualizations.utils import observations_to_image
import os 
from habitat.core.simulator import AgentState

@registry.register_measure
class DidMultiAgentsCollide(Measure):
    """
    Detects if the multi-agent ( more than 1 humanoids agents) in the scene 
    are colliding with each other at the current step. 
    """

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return "did_multi_agents_collide"

    def reset_metric(self, *args, **kwargs):
        self.update_metric(
            *args,
            **kwargs,
        )

    def update_metric(self, *args, task, **kwargs):
        sim = task._sim
        human_num = task._human_num
        if isinstance(sim.agents_mgr._all_agent_data[0], ArticulatedAgentData):
            sim.perform_discrete_collision_detection()
            contact_points = sim.get_physics_contact_points()
            found_contact = False

            agent_ids = [
                articulated_agent.sim_obj.object_id
                for articulated_agent in sim.agents_mgr.articulated_agents_iter
            ]
            main_agent_id = agent_ids[0]
            other_agent_ids = set(agent_ids[1:human_num+1])  
            for cp in contact_points:
                if coll_name_matches(cp, main_agent_id):
                    if any(coll_name_matches(cp, agent_id) for agent_id in other_agent_ids):
                        found_contact = True
                        break  

            self._metric = found_contact
        else:
            robot_position = np.array(sim.get_agent_state(0).position)  # 机器人坐标
            robot_radius = sim.agents_mgr._all_agent_data[0].agent_config.radius  # 机器人半径
            collision_detected = False
            for i in range(1, human_num + 1):
                # 获取人的位置和半径
                human_position = np.array(sim.get_agent_state(i).position)
                human_radius = sim.agents_mgr._all_agent_data[i].cfg.radius  # 人的半径

                # 计算机器人和人的距离
                distance = np.linalg.norm(robot_position - human_position)

                # 判断是否碰撞
                if distance < (robot_radius + human_radius):
                    collision_detected = True
                    # print(f"Collision detected with human {i} at position {human_position}")
                    break  # 找到碰撞的情况，跳出循环
            # if not collision_detected:
                # print("No collision detected.")
            self._metric = collision_detected

@registry.register_measure
class HumanCollision(Measure):

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        self._ever_collide = False
        super().__init__()

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return "human_collision"

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid, [DidMultiAgentsCollide._get_uuid()]
        )
        self._metric = 0.0
        self._ever_collide = False

    def update_metric(self, *args, episode, task, observations, **kwargs):
        collid = task.measurements.measures[DidMultiAgentsCollide._get_uuid()].get_metric()
        if collid or self._ever_collide:
            self._metric = 1.0
            self._ever_collide = True
            task.should_end = True
        else:
            self._metric = 0.0
@registry.register_measure
class STL(Measure):
    r"""Success weighted by Completion Time
    """
    cls_uuid: str = "stl"
    
    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__()

    def _get_uuid(self, *args, **kwargs):
        return self.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid, Success.cls_uuid, NumStepsMeasure.cls_uuid]
        )

        self._num_steps_taken = 0
        self._start_end_episode_distance = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self.update_metric(episode=episode, task=task, observations=observations, *args, **kwargs)

    def update_metric(self, *args, episode, task, observations, **kwargs):
        ep_success = task.measurements.measures[Success.cls_uuid].get_metric() 
        self._num_steps_taken = task.measurements.measures[NumStepsMeasure.cls_uuid].get_metric()

        oracle_time = (
            self._start_end_episode_distance / (0.25 / 10)
        )
        oracle_time = max(oracle_time, 1e-6)
        agent_time = max(self._num_steps_taken, 1e-6)
        self._metric = ep_success * (oracle_time / max(oracle_time, agent_time))

@registry.register_measure
class PersonalSpaceCompliance(Measure):

    cls_uuid: str = "psc"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        self._use_geo_distance = config.use_geo_distance
        super().__init__()
        
    def _get_uuid(self, *args, **kwargs):
        return self.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid, [NumStepsMeasure.cls_uuid]
        )
        self._compliant_steps = 0
        self._num_steps = 0

    def update_metric(self, *args, episode, task, observations, **kwargs):
        self._human_nums = min(episode.info['human_num'], self._sim.num_articulated_agents - 1)
        if self._human_nums == 0:
            self._metric = 1.0
        else:
            robot_pos = self._sim.get_agent_state(0).position
            self._num_steps = task.measurements.measures[NumStepsMeasure.cls_uuid].get_metric()
            compliance = True
            for i in range(self._human_nums):
                human_position = self._sim.get_agent_state(i+1).position

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

                if distance < 1.0:
                    compliance = False
                    break                    

            if compliance:
                self._compliant_steps += 1
            self._metric = (self._compliant_steps / self._num_steps)

@registry.register_measure
class SuccessfulPersonalSpaceCompliance(Measure):

    cls_uuid: str = "suc_psc"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__()
        
    def _get_uuid(self, *args, **kwargs):
        return self.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid, [PersonalSpaceCompliance.cls_uuid, Success.cls_uuid]
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        self._human_nums = min(episode.info['human_num'], len(self._sim.agents_mgr._all_agent_data) - 1) 
        self._success = task.measurements.measures[Success.cls_uuid].get_metric()
        if self._success:
            if self._human_nums == 0:
                self._metric = 1.0
            else:
                psc = task.measurements.measures[
                PersonalSpaceCompliance.cls_uuid
            ].get_metric()
                self._metric = psc
        else:
            self._metric = 0.0

@registry.register_measure
class SocialEtiquetteCompliance_1(Measure):
    """
    Social Etiquette Compliance metric for 1 second trajectory.
    """
    cls_uuid: str = "sec_1"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        self._threshold_squared = config.cover_future_dis_thre ** 2
        super().__init__()
        
    def _get_uuid(self, *args, **kwargs):
        return self.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self._metric = 1.0
        task.measurements.check_measure_dependencies(
            self.uuid, [HumanFutureTrajectory.cls_uuid, NumStepsMeasure.cls_uuid]
        )
        self._compliant_steps = 0
        self._num_steps = 0

    def update_metric(self, *args, episode, task, observations, **kwargs):
        robot_pos = self._sim.get_agent_state(0).position
        self._num_steps = task.measurements.measures[NumStepsMeasure.cls_uuid].get_metric()
        
        human_future_trajectory = task.measurements.measures[HumanFutureTrajectory.cls_uuid].get_metric()
        compliance_1 = True

        for trajectory in human_future_trajectory.values():
            for t, point in enumerate(trajectory):
                if t < 2:  # 1秒内的重合判断
                    if np.sum((robot_pos - point) ** 2) < self._threshold_squared:
                        compliance_1 = False
                        break
            if not compliance_1:
                break

        if compliance_1:
            self._compliant_steps += 1

        self._metric = (self._compliant_steps / self._num_steps)


@registry.register_measure
class SocialEtiquetteCompliance_2(Measure):
    """
    Social Etiquette Compliance metric for 2 second trajectory.
    """
    cls_uuid: str = "sec_2"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        self._threshold_squared = config.cover_future_dis_thre ** 2
        super().__init__()
        
    def _get_uuid(self, *args, **kwargs):
        return self.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self._metric = 1.0
        task.measurements.check_measure_dependencies(
            self.uuid, [HumanFutureTrajectory.cls_uuid, NumStepsMeasure.cls_uuid]
        )
        self._compliant_steps = 0
        self._num_steps = 0

    def update_metric(self, *args, episode, task, observations, **kwargs):
        robot_pos = self._sim.get_agent_state(0).position
        self._num_steps = task.measurements.measures[NumStepsMeasure.cls_uuid].get_metric()
        
        human_future_trajectory = task.measurements.measures[HumanFutureTrajectory.cls_uuid].get_metric()
        compliance_2 = True

        for trajectory in human_future_trajectory.values():
            for t, point in enumerate(trajectory):
                if t < 4:  # 2秒内的重合判断
                    if np.sum((robot_pos - point) ** 2) < self._threshold_squared:
                        compliance_2 = False
                        break
            if not compliance_2:
                break

        if compliance_2:
            self._compliant_steps += 1

        self._metric = (self._compliant_steps / self._num_steps)

@registry.register_measure
class SuccessfulSocialEtiquetteCompliance_1(Measure):

    cls_uuid: str = "suc_sec_1"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__()
        
    def _get_uuid(self, *args, **kwargs):
        return self.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid, [SocialEtiquetteCompliance_1.cls_uuid, Success.cls_uuid]
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        self._human_nums = min(episode.info['human_num'], len(self._sim.agents_mgr._all_agent_data) - 1) 
        self._success = task.measurements.measures[Success.cls_uuid].get_metric()
        if self._success:
            if self._human_nums == 0:
                self._metric = 1.0
            else:
                sec = task.measurements.measures[
                SocialEtiquetteCompliance_1.cls_uuid
            ].get_metric()
                self._metric = sec
        else:
            self._metric = 0.0

@registry.register_measure
class SuccessfulSocialEtiquetteCompliance_2(Measure):

    cls_uuid: str = "suc_sec_2"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__()
        
    def _get_uuid(self, *args, **kwargs):
        return self.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid, [SocialEtiquetteCompliance_2.cls_uuid, Success.cls_uuid]
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        self._human_nums = min(episode.info['human_num'], len(self._sim.agents_mgr._all_agent_data) - 1) 
        self._success = task.measurements.measures[Success.cls_uuid].get_metric()
        if self._success:
            if self._human_nums == 0:
                self._metric = 1.0
            else:
                sec = task.measurements.measures[
                SocialEtiquetteCompliance_2.cls_uuid
            ].get_metric()
                self._metric = sec
        else:
            self._metric = 0.0

@registry.register_measure
class MultiAgentNavReward(Measure):
    """
    Reward that gives a continuous reward for the social navigation task.
    """

    cls_uuid: str = "multi_agent_nav_reward"
        
    # @staticmethod
    # def _get_uuid(*args, **kwargs):
    #     return MultiAgentNavReward.cls_uuid
    def _get_uuid(self,*args, **kwargs):
        return self.cls_uuid

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metric = 0.0
        config = kwargs["config"]
        # Get the config and setup the hyperparameters
        self._config = config
        self._sim = kwargs["sim"]

        self._use_geo_distance = config.use_geo_distance
        self._allow_distance = config.allow_distance
        self._collide_scene_penalty = config.collide_scene_penalty
        self._collide_human_penalty = config.collide_human_penalty
        self._trajectory_cover_penalty = config.trajectory_cover_penalty
        self._threshold_squared = config.cover_future_dis_thre ** 2
        self._robot_idx = config.robot_idx
        self._close_to_human_penalty = config.close_to_human_penalty
        self._facing_human_dis = config.facing_human_dis

        self._human_nums = 0

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        if "human_num" in episode.info:
            self._human_nums = min(episode.info['human_num'], self._sim.num_articulated_agents - 1)
        else: 
            self._human_nums = 0
        self._metric = 0.0
        
    def _check_human_facing_robot(self, human_pos, robot_pos, human_idx):
        base_T = self._sim.get_agent_data(
            human_idx
        ).articulated_agent.sim_obj.transformation
        facing = (
            robot_human_vec_dot_product(human_pos, robot_pos, base_T)
            > self._config.human_face_robot_threshold
        )
        return facing
    
    def update_metric(self, *args, episode, task, observations, **kwargs):

        # Start social nav reward
        social_nav_reward = 0.0

        # Component 1: Goal distance reward (strengthened by multiplying by 1.5)
        distance_to_goal_reward = task.measurements.measures[
            DistanceToGoalReward.cls_uuid
        ].get_metric()
        social_nav_reward += 1.5 * distance_to_goal_reward  # Slightly reduced reward multiplier

        # Component 2: Penalize being too close to humans
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        use_k_robot = f"agent_{self._robot_idx}_localization_sensor"
        robot_pos = np.array(observations[use_k_robot][:3])

        if distance_to_target > self._allow_distance:
            human_dis = []
            for i in range(self._human_nums):
                use_k_human = f"agent_{i+1}_localization_sensor"
                human_position = observations[use_k_human][:3]

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
            
            # Apply penalties for being too close to humans
            for distance in human_dis:
                if distance < self._facing_human_dis:
                    penalty = self._close_to_human_penalty * np.exp(-distance / self._facing_human_dis)
                    social_nav_reward += penalty

        # Component 3: Collision detection for two agents
        did_agents_collide = task.measurements.measures[
            DidMultiAgentsCollide._get_uuid()
        ].get_metric()
        if did_agents_collide:
            task.should_end = True
            social_nav_reward += self._collide_human_penalty

        # Component 4: Collision detection for the main agent and the scene 
        did_rearrange_collide, collision_detail = rearrange_collision(
            self._sim, True, ignore_base=False, agent_idx=self._robot_idx
        )
        if did_rearrange_collide:
            social_nav_reward += self._collide_scene_penalty
        
        # Component 5: Trajectory overlap penalty with time-based weighting
        if distance_to_target > self._allow_distance and "human_future_trajectory" in task.measurements.measures:
            human_future_trajectory_temp = task.measurements.measures['human_future_trajectory']._metric
            for trajectory in human_future_trajectory_temp.values():
                for t, point in enumerate(trajectory):
                    time_weight = 1.0 / (1 + t)  # Time-weighted penalty
                    if np.sum((robot_pos - point) ** 2) < self._threshold_squared:
                        social_nav_reward += self._trajectory_cover_penalty * time_weight
                        break

        self._metric = social_nav_reward

@registry.register_measure
class HumanVelocityMeasure(UsesArticulatedAgentInterface, Measure):
    """
    The measure for ORCA
    """

    cls_uuid: str = "human_velocity_measure"

    def __init__(self, *args, sim, **kwargs):
        self._sim = sim
        self.human_num = kwargs['task']._human_num
        self.velo_coff = np.array([[0, 1]] * 6)
        self.velo_base = np.array([[0.25, np.deg2rad(10)]] * 6)
        
        super().__init__(*args, sim=sim, **kwargs)
        self._metric = self.velo_base * self.velo_coff 

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return HumanVelocityMeasure.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self.human_num = task._human_num
        self.velo_coff = np.array([[0.0, 0.0]] * 6)
        self.velo_base = np.array([[0.25, np.deg2rad(10)]] * 6)
        self._metric = self.velo_base * self.velo_coff 

    def update_metric(self, *args, episode, task, observations, **kwargs):
        self._metric = self.velo_base * self.velo_coff 

def merge_paths(paths):
    merged_path = []
    for i, path in enumerate(paths):
        if i > 0:
            path = path[1:]
        merged_path.extend(path)
    return merged_path


@registry.register_measure
class HumanFutureTrajectory(UsesArticulatedAgentInterface, Measure):
    """
    The measure for future prediction of social crowd navigation.
    """

    cls_uuid: str = "human_future_trajectory"

    def __init__(self, *args, sim, **kwargs):
        self._sim = sim
        self.human_num = kwargs['task']._human_num
        self.output_length = 5
        self.target_dict = self._initialize_target_dict(self.human_num)
        self.path_dict = {}
        super().__init__(*args, sim=sim, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return HumanFutureTrajectory.cls_uuid

    def _initialize_target_dict(self, human_num):
        """Initialize the target dictionary with default values."""
        # Initialize with a list of shape (human_num, 2, 3) filled with [-100, -100, -100]
        return np.full((human_num, 2, 3), -100, dtype=np.float32).tolist()

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self.human_num = task._human_num
        self.target_dict = self._initialize_target_dict(self.human_num)
        self.path_dict = {}
        self._metric = {}

    def _path_to_point(self, point_a, point_b):
        """Get the shortest path between two points."""
        path = habitat_sim.ShortestPath()
        path.requested_start = point_a
        path.requested_end = point_b
        found_path = self._sim.pathfinder.find_path(path)
        return path.points if found_path else [point_a, point_b]

    def _process_path(self, path, length):
        """Process the path by merging and padding/truncating to the desired length."""
        temp_merged_path = np.array(merge_paths(path), dtype=np.float32) # merge_paths(path)
        merged_length = len(temp_merged_path)
        
        if merged_length < length:
            padding = np.tile(temp_merged_path[-1], (length - merged_length, 1))
            temp_merged_path = np.concatenate([temp_merged_path, padding], axis=0)
        else:
            temp_merged_path = temp_merged_path[:length]
        
        return temp_merged_path.tolist()

    def update_metric(self, *args, episode, task, observations, **kwargs):
        for agent_idx, target in enumerate(self.target_dict):
            path = []
            agent_pos = np.array(self._sim.get_agent_data(agent_idx + 1).articulated_agent.base_pos)

            # Use only valid targets to reduce unnecessary path computations
            valid_targets = [np.array(point) for point in target if not np.allclose(point, [-100, -100, -100])]
            prev_point = agent_pos

            for path_point in valid_targets:
                temp_path = self._path_to_point(prev_point, path_point)
                path.extend(temp_path)  # Directly add the path between points
                prev_point = path_point

            self.path_dict[agent_idx + 1] = self._process_path(path, self.output_length)
            
        self._metric = self.path_dict

@registry.register_measure
class MultiFloorTopDownMap(FrontierExplorationMap):
    def __init__(
        self,
        sim: HabitatSim,
        config: DictConfig,
        task: EmbodiedTask,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(sim, config, task, *args, **kwargs)
        self._floor_heights = None
        self._saved_maps = None
        self._saved_fogs = None
        self._cur_floor = 0

    def reset_metric(
        self, episode: NavigationEpisode, *args: Any, **kwargs: Any
    ) -> None:
        self._floor_heights, self._saved_maps, self._saved_fogs = self.detect_floors(sample_points=100)
        if hasattr(self._sim,"agents_mgr") and isinstance(self._sim.agents_mgr._all_agent_data[0], ArticulatedAgentData):
            agent_position = self._sim.get_agent_data().articulated_agent.base_pos
        else:
            agent_position = self._sim.get_agent_state().position
        agent_height = agent_position[1]
        # Find the floor corresponding to the agent's height
        flag = True  # 标记是否找到匹配楼层
        for ithfloor, floor_height in enumerate(self._floor_heights):
            if self._is_on_same_floor(agent_height, floor_height):
                self._top_down_map = self._saved_maps[ithfloor]
                self._fog_of_war_mask = self._saved_fogs[ithfloor]
                self._cur_floor = ithfloor
                flag = False  # 找到匹配楼层，标记为 False
                break
        if flag:  # 未找到匹配楼层，选择最近的楼层
            closest_floor = min(
                range(len(self._floor_heights)),
                key=lambda idx: abs(self._floor_heights[idx] - agent_height),
            )
            self._top_down_map = self._saved_maps[closest_floor]
            self._fog_of_war_mask = self._saved_fogs[closest_floor]
            self._cur_floor = closest_floor
            
        if self._top_down_map is None:
            self._top_down_map = self._saved_maps[0] # if update, use the previous one

        self._previous_xy_location = [
            None for _ in range(len(self._sim.habitat_config.agents))
        ]

        if hasattr(episode, "goals"):
            # draw source and target parts last to avoid overlap
            self._draw_goals_view_points(episode)
            self._draw_goals_aabb(episode)
            self._draw_goals_positions(episode)
            self._draw_shortest_path(episode, agent_position)

        if self._config.draw_source and self._is_on_same_floor(agent_height, episode.start_position[1]):
            self._draw_point(
                episode.start_position, maps.MAP_SOURCE_POINT_INDICATOR
            )

        ### For frontier exploration
        assert "task" in kwargs, "task must be passed to reset_metric!"
        self._explorer_sensor = kwargs["task"].sensor_suite.sensors[self._explorer_uuid]
        self._static_metrics = {}

        self.update_metric(episode, None) # need _explorer_sensor to provide fog_mask # 

        self._draw_target_bbox_mask(episode)

        # Expose sufficient info for drawing 3D points on the map
        lower_bound, upper_bound = self._sim.pathfinder.get_bounds()
        episodic_start_yaw = HeadingSensor._quat_to_xy_heading(
            None,  # type: ignore
            quaternion_from_coeff(episode.start_rotation).inverse(),
        )[0]
        x, y, z = habitat_to_xyz(np.array(episode.start_position))
        self._static_metrics["upper_bound"] = (upper_bound[0], upper_bound[2])
        self._static_metrics["lower_bound"] = (lower_bound[0], lower_bound[2])
        self._static_metrics["grid_resolution"] = self._metric["map"].shape[:2]
        self._static_metrics["tf_episodic_to_global"] = np.array(
            [
                [np.cos(episodic_start_yaw), -np.sin(episodic_start_yaw), 0, x],
                [np.sin(episodic_start_yaw), np.cos(episodic_start_yaw), 0, y],
                [0, 0, 1, z],
                [0, 0, 0, 1],
            ]
        )

    def _is_on_same_floor(
        self, height, ref_floor_height=None, ceiling_height=0.5, floor_tolerance = 0.2, # ceiling_height=2.0, floor_tolerance = 0.02
    ):
        if ref_floor_height is None:
            ref_floor_height = self._sim.get_agent(0).state.position[1]
        return ref_floor_height - floor_tolerance <= height < ref_floor_height + ceiling_height

    def detect_floors(self, output_dir="./", sample_points=100, floor_tolerance=0.5):
        """
        Detect the number of floors and their heights in a scene.

        :param pathfinder: Habitat-sim's PathFinder instance.
        :param sample_points: Number of points to sample per island to calculate its height.
        :param floor_tolerance: Minimum height difference to distinguish different floors.

        :return: List of detected floor heights (sorted) and the total number of floors.
        """
        island_heights = []

        # Iterate through all navigable islands
        for island_index in range(self._sim.pathfinder.num_islands):
            heights = []
            
            # Randomly sample points from the island
            for _ in range(sample_points):
                point = self._sim.pathfinder.get_random_navigable_point(island_index=island_index)
                if point is not None:
                    heights.append(point[1])  # Store the height (y-coordinate)

            # 计算该岛屿的高度
            if heights:
                height_counts = Counter(heights)  # 统计每个高度的出现次数
                
                for height, count in height_counts.items():
                    # 如果某高度的出现次数超过三分之一，记录该高度
                    if count > sample_points // 4:
                        island_heights.append(height)

        # Sort the heights and group them by floor tolerance
        island_heights = sorted(island_heights, reverse=False)
        
        floor_heights = []
        for height in island_heights:
            if not floor_heights or abs(height - floor_heights[-1]) >= floor_tolerance:
                floor_heights.append(height)

        # floor_heights=island_heights

        # Generate and save top-down maps
        saved_maps = []
        saved_fogs = []        
        for idx, height in enumerate(floor_heights):
            # try:
                # Generate top-down view
                # topdown_map = self._sim.pathfinder.get_topdown_view(
                #     meters_per_pixel=1.0 / self._map_resolution, height=height
                # )
            topdown_map = maps.get_topdown_map(
            pathfinder=self._sim.pathfinder,
            height=height,
            map_resolution=self._map_resolution,
            draw_border=self._config.draw_border,
        )
            saved_maps.append(topdown_map)
            if self._config.fog_of_war.draw:
                saved_fogs.append(np.zeros_like(topdown_map)) # 
            else:
                saved_fogs.append(None)
            
            DEBUG = False
            if DEBUG: # defalut = False
                if not os.path.exists("map_debug"):
                    os.mkdir("map_debug")
                # img = observations_to_image(
                #     {}, {f"top_down_map.{k}": v for k, v in self._metric.items()}
                # )

                # Save map as an image
                plt.figure(figsize=(10, 10))
                plt.imshow(topdown_map, cmap="gray")
                plt.axis("off")
                filename = f"map_debug/top_down_map_floor_{idx}_height_{height:.2f}.png"
                plt.savefig(filename, bbox_inches="tight", pad_inches=0)
                plt.close()
                print(f"Saved top-down map for height {height:.2f} as {filename}")

        return floor_heights, saved_maps, saved_fogs
    
    def update_map(self, agent_state: AgentState, agent_index: int):
        agent_position = agent_state.position
        a_x, a_y = maps.to_grid(
            agent_position[2],
            agent_position[0],
            (self._top_down_map.shape[0], self._top_down_map.shape[1]),
            sim=self._sim,
        )
        if (a_x < self._top_down_map.shape[0] and a_x >= 0) and (a_y < self._top_down_map.shape[1] and a_y >= 0):
            pass
        else:
            return a_x, a_y
        # Don't draw over the source point
        if self._top_down_map[a_x, a_y] != maps.MAP_SOURCE_POINT_INDICATOR:
            color = maps.MAP_SOURCE_POINT_INDICATOR + agent_index * 10 
            # color = 10 + min(
            #     self._step_count * 245 // self._config.max_episode_steps, 245
            # )
            thickness = self.line_thickness
            if self._previous_xy_location[agent_index] is not None:
                cv2.line(
                    self._top_down_map,
                    self._previous_xy_location[agent_index],
                    (a_y, a_x),
                    color,
                    thickness=thickness,
                )
        angle = TopDownMap.get_polar_angle(agent_state)
        if self._fog_of_war_mask.shape == self._explorer_sensor.fog_of_war_mask.shape: # the 0th step can cause error
            self.update_fog_of_war_mask(np.array([a_x, a_y]), angle)

        self._previous_xy_location[agent_index] = (a_y, a_x)
        return a_x, a_y

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        if hasattr(self._sim,"agents_mgr") and isinstance(self._sim.agents_mgr._all_agent_data[0], ArticulatedAgentData):
            agent_position = self._sim.get_agent_data().articulated_agent.base_pos
        else:
            agent_position = self._sim.get_agent_state().position
        agent_height = agent_position[1]
        if self._is_on_same_floor(agent_height, self._floor_heights[self._cur_floor]):
            pass
        else:
            flag = True # do not match any floor
            for ithfloor, floor_height in enumerate(self._floor_heights):
                if self._is_on_same_floor(agent_height, floor_height):
                    self._top_down_map = self._saved_maps[ithfloor]
                    self._fog_of_war_mask = self._saved_fogs[ithfloor]
                    self._cur_floor = ithfloor
                    flag = False
                    break
            if flag: # maybe at stair
                pass
            else:
                if hasattr(episode, "goals"):
                # draw source and target parts last to avoid overlap
                    self._draw_goals_view_points(episode)
                    self._draw_goals_aabb(episode)
                    self._draw_goals_positions(episode)
                    self._draw_shortest_path(episode, agent_position)
                if self._config.draw_source and self._is_on_same_floor(episode.start_position[1], self._floor_heights[self._cur_floor]):
                    self._draw_point(
                        episode.start_position, maps.MAP_SOURCE_POINT_INDICATOR
                    )
        map_positions: List[Tuple[float]] = []
        map_angles = []
        if 'human_num' in episode.info:
            for agent_index in range(episode.info['human_num']):
                agent_state = self._sim.get_agent_state(agent_index)
                if self._is_on_same_floor(agent_state.position[1], self._floor_heights[self._cur_floor]) : # for human, filter the one in other floor
                    map_positions.append(self.update_map(agent_state, agent_index))
                    map_angles.append(MultiFloorTopDownMap.get_polar_angle(agent_state))
        else:
            for agent_index in range(len(self._sim.habitat_config.agents)):
                agent_state = self._sim.get_agent_state(agent_index)
                if self._is_on_same_floor(agent_state.position[1], self._floor_heights[self._cur_floor]) : # for human, filter the one in other floor
                    map_positions.append(self.update_map(agent_state, agent_index))
                    map_angles.append(MultiFloorTopDownMap.get_polar_angle(agent_state))
        self._metric = {
            "map": self._top_down_map,
            "fog_of_war_mask": self._fog_of_war_mask,
            "agent_map_coord": map_positions,
            "agent_angle": map_angles,
        }

        ### For frontier exploration
        # Update the map with visualizations of the frontier waypoints
        new_map = self._metric["map"].copy()
        circle_size = 20 * self._map_resolution // 1024
        thickness = max(int(round(3 * self._map_resolution / 1024)), 1)
        selected_frontier = self._explorer_sensor.closest_frontier_waypoint

        if self._draw_waypoints:
            next_waypoint = self._explorer_sensor.next_waypoint_pixels
            if next_waypoint is not None:
                cv2.circle(
                    new_map,
                    tuple(next_waypoint[::-1].astype(np.int32)),
                    circle_size,
                    maps.MAP_INVALID_POINT,
                    1,
                )

        for waypoint in self._explorer_sensor.frontier_waypoints:
            if np.array_equal(waypoint, selected_frontier):
                color = maps.MAP_TARGET_POINT_INDICATOR
            else:
                color = maps.MAP_SOURCE_POINT_INDICATOR
            cv2.circle(
                new_map,
                waypoint[::-1].astype(np.int32),
                circle_size,
                color,
                1,
            )

        beeline_target = getattr(self._explorer_sensor, "beeline_target_pixels", None)
        if beeline_target is not None:
            cv2.circle(
                new_map,
                tuple(beeline_target[::-1].astype(np.int32)),
                circle_size * 2,
                maps.MAP_SOURCE_POINT_INDICATOR,
                thickness,
            )
        self._metric["map"] = new_map
        self._metric["is_feasible"] = self._is_feasible
        # if not self._is_feasible:
        #     self._task._is_episode_active = False

        # Update self._metric with the static metrics
        self._metric.update(self._static_metrics)

        # if DEBUG:
        #     import time

        #     if not os.path.exists("map_debug"):
        #         os.mkdir("map_debug")
        #     img = observations_to_image(
        #         {}, {f"top_down_map.{k}": v for k, v in self._metric.items()}
        #     )
        #     cv2.imwrite(
        #         f"map_debug/{int(time.time())}_full.png",
        #         cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
        #     )

    def _draw_goals_view_points(self, episode):
        super()._draw_goals_view_points(episode)

        # Use this opportunity to determine whether this episode is feasible to complete
        # without climbing stairs

        # Compute the pixel location of the start position
        t_x, t_y = maps.to_grid(
            episode.start_position[2],
            episode.start_position[0],
            (self._top_down_map.shape[0], self._top_down_map.shape[1]),
            sim=self._sim,
        )
        # The start position would be here: self._top_down_map[t_x,t_y]

        # Compute contours that contain MAP_VALID_POINT and/or MAP_VIEW_POINT_INDICATOR
        valid_with_viewpoints = self._top_down_map.copy()
        valid_with_viewpoints[
            valid_with_viewpoints == maps.MAP_VIEW_POINT_INDICATOR
        ] = maps.MAP_VALID_POINT
        # Dilate valid_with_viewpoints by 2 pixels to ensure that the contour is not
        # broken by pinching obstacles
        valid_with_viewpoints = cv2.dilate(
            valid_with_viewpoints, np.ones((3, 3), dtype=np.uint8)
        )
        contours, _ = cv2.findContours(
            valid_with_viewpoints, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            # No contours found, mark as infeasible and return
            self._is_feasible = False
            return
        # Identify the contour that is closest to the start position
        min_dist = np.inf
        best_idx = 0
        for idx, cnt in enumerate(contours):
            dist = cv2.pointPolygonTest(cnt, (t_y, t_x), True)
            if dist >= 0:
                best_idx = idx
                break
            elif abs(dist) < min_dist:
                min_dist = abs(dist)
                best_idx = idx

        try:
            # Access the best contour
            best_cnt = contours[best_idx]
            mask = np.zeros_like(valid_with_viewpoints)
            mask = cv2.drawContours(mask, [best_cnt], 0, 1, -1)  # type: ignore
            masked_values = self._top_down_map[mask.astype(bool)]
            values = set(masked_values.tolist())
            is_feasible = maps.MAP_VALID_POINT in values and maps.MAP_VIEW_POINT_INDICATOR in values

            self._is_feasible = is_feasible
        except IndexError as e:
            # Log the error and mark the episode as infeasible
            print(f"Error accessing contours: {e}")
            self._is_feasible = False

    def _draw_target_bbox_mask(self, episode: NavigationEpisode):
        """Save a mask that is the same size as self._top_down_map, and draw a filled
        rectangle for each bounding box of each target in the episode"""
        if not isinstance(episode, ObjectGoalNavEpisode):
            return

        bbox_mask = np.zeros_like(self._top_down_map)
        for goal in episode.goals:
            sem_scene = self._sim.semantic_annotations()
            object_id = goal.object_id  # type: ignore
            assert int(sem_scene.objects[object_id].id.split("_")[-1]) == int(
                object_id
            ), (
                f"Object_id doesn't correspond to id in semantic scene objects"
                f"dictionary for episode: {episode}"
            )

            center = sem_scene.objects[object_id].aabb.center
            x_len, _, z_len = sem_scene.objects[object_id].aabb.sizes / 2.0

            # Nodes to draw rectangle
            corners = [
                center + np.array([x, 0, z])
                for x, z in [(-x_len, -z_len), (x_len, z_len)]
                if self._is_on_same_floor(center[1])
            ]

            if not corners:
                continue

            map_corners = [
                maps.to_grid(
                    p[2],
                    p[0],
                    (self._top_down_map.shape[0], self._top_down_map.shape[1]),
                    sim=self._sim,
                )
                for p in corners
            ]
            (y1, x1), (y2, x2) = map_corners
            bbox_mask[y1:y2, x1:x2] = 1

        self._static_metrics["target_bboxes_mask"] = bbox_mask

@dataclass
class MultiAgentNavReward(MeasurementConfig):
    r"""
    The reward for the multi agent navigation tasks.
    """
    type: str = "MultiAgentNavReward"
    
    # If we want to use geo distance to measure the distance
    # between the robot and the human
    use_geo_distance: bool = True
    # discomfort for multi agents
    allow_distance: float = 0.5 
    collide_scene_penalty: float = -0.25 
    collide_human_penalty: float = -0.5  
    facing_human_dis: float = 1.0
    human_face_robot_threshold: float = 0.5
    close_to_human_penalty: float = -0.025
    trajectory_cover_penalty: float = -0.025 
    cover_future_dis_thre: float = -0.05  
    # Set the id of the agent
    robot_idx: int = 0

@dataclass
class DidMultiAgentsCollideConfig(MeasurementConfig):
    type: str = "DidMultiAgentsCollide"
    
@dataclass
class STLMeasurementConfig(MeasurementConfig):
    type: str = "STL"

@dataclass
class PersonalSpaceComplianceMeasurementConfig(MeasurementConfig):
    type: str = "PersonalSpaceCompliance"
    use_geo_distance: bool = True

@dataclass
class SuccessfulPersonalSpaceComplianceMeasurementConfig(MeasurementConfig):
    type: str = "SuccessfulPersonalSpaceCompliance"

@dataclass
class SocialEtiquetteComplianceMeasurementConfig_1(MeasurementConfig):
    type: str = "SocialEtiquetteCompliance_1"
    cover_future_dis_thre: float = 0.18 # agent's radius 0.05

@dataclass
class SocialEtiquetteComplianceMeasurementConfig_2(MeasurementConfig):
    type: str = "SocialEtiquetteCompliance_2"
    cover_future_dis_thre: float = 0.18 # agent's radius 0.05

@dataclass
class SuccessfulSocialEtiquetteComplianceMeasurementConfig_1(MeasurementConfig):
    type: str = "SuccessfulSocialEtiquetteCompliance_1"

@dataclass
class SuccessfulSocialEtiquetteComplianceMeasurementConfig_2(MeasurementConfig):
    type: str = "SuccessfulSocialEtiquetteCompliance_2"

@dataclass
class HumanCollisionMeasurementConfig(MeasurementConfig):
    type: str = "HumanCollision"

@dataclass
class HumanVelocityMeasurementConfig(MeasurementConfig):
    type: str = "HumanVelocityMeasure"

@dataclass
class HumanFutureTrajectoryMeasurementConfig(MeasurementConfig):
    type: str = "HumanFutureTrajectory"

@dataclass
class MultiFloorTopDownMapMeasurementConfig(FrontierExplorationMapMeasurementConfig):
    type: str = "MultiFloorTopDownMap"

cs = ConfigStore.instance()

cs.store(
    package="habitat.task.measurements.multi_agent_nav_reward",
    group="habitat/task/measurements",
    name="multi_agent_nav_reward",
    node=MultiAgentNavReward,
)
cs.store(
    package="habitat.task.measurements.stl",
    group="habitat/task/measurements",
    name="stl",
    node=STLMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.psc",
    group="habitat/task/measurements",
    name="psc",
    node=PersonalSpaceComplianceMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.suc_psc",
    group="habitat/task/measurements",
    name="suc_psc",
    node=SuccessfulPersonalSpaceComplianceMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.sec_1",
    group="habitat/task/measurements",
    name="sec_1",
    node=SocialEtiquetteComplianceMeasurementConfig_1,
)
cs.store(
    package="habitat.task.measurements.sec_2",
    group="habitat/task/measurements",
    name="sec_2",
    node=SocialEtiquetteComplianceMeasurementConfig_2,
)
cs.store(
    package="habitat.task.measurements.suc_sec_1",
    group="habitat/task/measurements",
    name="suc_sec_1",
    node=SuccessfulSocialEtiquetteComplianceMeasurementConfig_1,
)
cs.store(
    package="habitat.task.measurements.suc_sec_2",
    group="habitat/task/measurements",
    name="suc_sec_2",
    node=SuccessfulSocialEtiquetteComplianceMeasurementConfig_2,
)
cs.store(
    package="habitat.task.measurements.human_collision",
    group="habitat/task/measurements",
    name="human_collision",
    node=HumanCollisionMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.did_multi_agents_collide",
    group="habitat/task/measurements",
    name="did_multi_agents_collide",
    node=DidMultiAgentsCollideConfig,
)
cs.store(
    package="habitat.task.measurements.human_velocity_measure",
    group="habitat/task/measurements",
    name="human_velocity_measure",
    node=HumanVelocityMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.human_future_trajectory",
    group="habitat/task/measurements",
    name="human_future_trajectory",
    node=HumanFutureTrajectoryMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.multi_floor_map",
    group="habitat/task/measurements",
    name="multi_floor_map",
    node=MultiFloorTopDownMapMeasurementConfig,
)