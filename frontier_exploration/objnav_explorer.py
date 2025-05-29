import random
from dataclasses import dataclass
from typing import Any

import numpy as np
from habitat import EmbodiedTask, registry
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.tasks.nav.nav import DistanceToGoal
from habitat.tasks.nav.object_nav_task import ObjectGoal, ObjectViewLocation
from habitat_sim import AgentState
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from frontier_exploration.base_explorer import (
    ActionIDs,
    BaseExplorer,
    BaseExplorerSensorConfig,
)
from frontier_exploration.utils.general_utils import wrap_heading


class State:
    EXPLORE = 0
    BEELINE = 1
    PIVOT = 2


@registry.register_sensor
class ObjNavExplorer(BaseExplorer):
    cls_uuid: str = "objnav_explorer"

    def __init__(
        self,
        sim: HabitatSim,
        config: "DictConfig",
        task: EmbodiedTask,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(sim, config, *args, **kwargs)
        self._beeline_dist_thresh = config.beeline_dist_thresh
        self._success_distance = config.success_distance

        self._episode = None
        self._task = task
        self._state = State.EXPLORE
        self._beeline_target = None
        self._target_yaw = None
        self._goal_dist_measure = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ]
        self._step_count = 0

    def _reset(self, episode):
        super()._reset(episode)
        self._episode = episode
        self._state = State.EXPLORE
        self._beeline_target = None
        self._target_yaw = None
        self._goal_dist_measure.reset_metric(episode, task=self._task)
        self._step_count = 0

    @property
    def beeline_target_pixels(self):
        # This property is used by the FrontierExplorationMap measurement
        if self._beeline_target is None:
            return None
        return self._map_coors_to_pixel(self._beeline_target)

    def get_observation(
        self, task: EmbodiedTask, episode, *args: Any, **kwargs: Any
    ) -> np.ndarray:
        super()._pre_step(episode)
        if self._state == State.EXPLORE:
            action = super().get_observation(task, episode, *args, **kwargs)
            if np.array_equal(action, ActionIDs.STOP):
                # This is undefined behavior. The explorer shouldn't run out of
                # frontiers, but episode is set up incorrectly.
                task.is_stop_called = True
                # Randomly re-assign the action to forward or turn left/right
                action = random.choice(
                    [
                        ActionIDs.MOVE_FORWARD,
                        ActionIDs.TURN_LEFT,
                        ActionIDs.TURN_RIGHT,
                    ]
                )
        else:
            # Transition to another state if necessary
            min_dist = self._get_min_dist()
            if (
                self._state == State.BEELINE
                and min_dist < self._success_distance
            ):
                # Change to PIVOT state if already within success distance
                closest_point = self.identify_closest_viewpoint()
                closest_rot = closest_point.agent_state.rotation
                self._target_yaw = 2 * np.arctan2(
                    closest_rot[1], closest_rot[3]
                )
                self._state = State.PIVOT
            elif (
                self._state == State.PIVOT
                and min_dist > self._success_distance
            ):
                # Change to BEELINE state if now out of the success distance
                self._state = State.BEELINE

            # Execute the appropriate behavior for the current state
            if self._state == State.BEELINE:
                self._beeline_target = (
                    self._episode._shortest_path_cache.points[-1]
                )
                action = self._decide_action(self._beeline_target)
            elif self._state == State.PIVOT:
                action = self._pivot()
            else:
                raise ValueError("Invalid state")

            # Inflection is used by action inflection sensor for IL. This is
            # already done by BaseExplorer when in EXPLORE state.
            if self._prev_action is not None:
                self.inflection = self._prev_action != action
            self._prev_action = action

        self._step_count += 1

        return action

    def _pivot(self):
        """Returns LEFT or RIGHT action to pivot the agent towards the target,
        or STOP if the agent is already facing the target as best it can."""
        agent_rot = self._sim.get_agent_state().rotation
        agent_yaw = 2 * np.arctan2(agent_rot.y, agent_rot.w)
        heading_err = -wrap_heading(self._target_yaw - agent_yaw)
        if heading_err > self._turn_angle / 2:
            return ActionIDs.TURN_RIGHT
        elif heading_err < -self._turn_angle / 2:
            return ActionIDs.TURN_LEFT
        return ActionIDs.STOP

    def identify_closest_viewpoint(self):
        """Returns the viewpoint closest to the agent"""
        if len(self._episode.goals) > 0:
            goals = self._episode.goals
        else:
            goals = self._task._dataset.goals_by_category[  # type: ignore
                self._episode.goals_key
            ]
        if isinstance(goals[0], ObjectGoal):
            view_points = [vp for goal in goals for vp in goal.view_points]
        else:
            agent_state = AgentState(goals[0].position, goals[0].rotation, {})
            view_points = [ObjectViewLocation(agent_state, None)]
        min_dist, min_idx = float("inf"), None
        for i, view_point in enumerate(view_points):
            dist = np.linalg.norm(
                view_point.agent_state.position - self.agent_position
            )
            if dist < min_dist:
                min_dist, min_idx = dist, i
        return view_points[min_idx]

    def _get_min_dist(self):
        """Returns the minimum distance to the closest target"""
        self._goal_dist_measure.update_metric(self._episode, task=self._task)
        dist = self._goal_dist_measure.get_metric()
        if (
            self._episode._shortest_path_cache is None
            or len(self._episode._shortest_path_cache.points) == 0
        ):
            return float("inf")
        return dist

    def _update_fog_of_war_mask(self):
        updated = (
            super()._update_fog_of_war_mask()
            if self._state == State.EXPLORE
            else False
        )

        min_dist = self._get_min_dist()

        if self._state == State.EXPLORE:
            # Start beelining if the minimum distance to the target is less than the
            # set threshold
            if min_dist < self._beeline_dist_thresh:
                self._state = State.BEELINE
                self._beeline_target = (
                    self._episode._shortest_path_cache.points[-1]
                )

        return updated


@registry.register_sensor
class GreedyObjNavExplorer(ObjNavExplorer):
    cls_uuid: str = "greedy_objnav_explorer"

    def _get_closest_waypoint(self):
        if len(self.frontier_waypoints) == 0:
            return None
        # Identify the closest target object
        self._goal_dist_measure.update_metric(self._episode, task=self._task)
        pts_cache = self._episode._shortest_path_cache
        if hasattr(pts_cache, "closest_end_point_index"):
            if pts_cache.closest_end_point_index == -1:
                return None
            closest_point = pts_cache.requested_ends[
                pts_cache.closest_end_point_index
            ]
        else:
            idx, _ = self._astar_search(self._episode_view_points)
            if idx is None:
                return None
            closest_point = self._episode_view_points[idx]
        # Identify the frontier waypoint closest to this object
        sim_waypoints = self._pixel_to_map_coors(self.frontier_waypoints)
        idx, _ = self._astar_search(
            sim_waypoints, start_position=closest_point
        )

        return self.frontier_waypoints[idx]


@dataclass
class ObjNavExplorerSensorConfig(BaseExplorerSensorConfig):
    type: str = ObjNavExplorer.__name__
    turn_angle: float = 30.0  # degrees
    beeline_dist_thresh: float = 8  # meters
    success_distance: float = 0.1  # meters


@dataclass
class GreedyObjNavExplorerSensorConfig(ObjNavExplorerSensorConfig):
    type: str = GreedyObjNavExplorer.__name__


cs = ConfigStore.instance()
cs.store(
    package=f"habitat.task.lab_sensors.{ObjNavExplorer.cls_uuid}",
    group="habitat/task/lab_sensors",
    name=f"{ObjNavExplorer.cls_uuid}",
    node=ObjNavExplorerSensorConfig,
)
cs.store(
    package=f"habitat.task.lab_sensors.{GreedyObjNavExplorer.cls_uuid}",
    group="habitat/task/lab_sensors",
    name=f"{GreedyObjNavExplorer.cls_uuid}",
    node=GreedyObjNavExplorerSensorConfig,
)
