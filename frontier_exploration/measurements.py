import os
from dataclasses import dataclass
from typing import Any, Dict

import cv2
import numpy as np
from habitat import EmbodiedTask, registry
from habitat.config import read_write
from habitat.config.default_structured_configs import TopDownMapMeasurementConfig
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.tasks.nav.nav import HeadingSensor, NavigationEpisode, TopDownMap
from habitat.tasks.nav.object_nav_task import ObjectGoalNavEpisode
from habitat.utils.geometry_utils import quaternion_from_coeff
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.maps import (
    MAP_INVALID_POINT,
    MAP_SOURCE_POINT_INDICATOR,
    MAP_TARGET_POINT_INDICATOR,
    MAP_VALID_POINT,
    MAP_VIEW_POINT_INDICATOR,
)
from habitat.utils.visualizations.utils import observations_to_image
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from frontier_exploration.base_explorer import BaseExplorer
from frontier_exploration.objnav_explorer import GreedyObjNavExplorer, ObjNavExplorer
from frontier_exploration.utils.general_utils import habitat_to_xyz

DEBUG = os.environ.get("MAP_DEBUG", "False").lower() == "true"
if DEBUG:
    print(f"[{os.path.basename(__file__)}]: WARNING: MAP_DEBUG is True")


@registry.register_measure
class FrontierExplorationMap(TopDownMap):
    def __init__(
        self,
        sim: HabitatSim,
        config: DictConfig,
        task: EmbodiedTask,
        *args: Any,
        **kwargs: Any,
    ):
        self._explorer_uuid = None
        for i in [BaseExplorer, ObjNavExplorer, GreedyObjNavExplorer]:
            if i.cls_uuid in task._config.lab_sensors:
                assert (
                    self._explorer_uuid is None
                ), "FrontierExplorationMap only supports 1 explorer sensor at a time!"
                self._explorer_uuid = i.cls_uuid

        if self._explorer_uuid is None:
            raise RuntimeError("FrontierExplorationMap needs an exploration sensor!")
        explorer_config = task._config.lab_sensors[self._explorer_uuid]
        with read_write(config):
            config.map_resolution = explorer_config.map_resolution

        super().__init__(sim, config, *args, **kwargs)

        self._explorer_sensor = None
        self._draw_waypoints: bool = config.draw_waypoints
        self._is_feasible: bool = True
        self._static_metrics: Dict[str, Any] = {}  # only updated once per episode
        self._task = task

    def reset_metric(
        self, episode: NavigationEpisode, *args: Any, **kwargs: Any
    ) -> None:
        assert "task" in kwargs, "task must be passed to reset_metric!"
        self._explorer_sensor = kwargs["task"].sensor_suite.sensors[self._explorer_uuid]
        self._static_metrics = {}
        super().reset_metric(episode, *args, **kwargs)
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

    def get_original_map(self):
        return self._explorer_sensor.top_down_map.copy()

    def update_fog_of_war_mask(self, *args, **kwargs):
        self._fog_of_war_mask = self._explorer_sensor.fog_of_war_mask.copy()

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        super().update_metric(episode, action, *args, **kwargs)
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
                    MAP_INVALID_POINT,
                    1,
                )

        for waypoint in self._explorer_sensor.frontier_waypoints:
            if np.array_equal(waypoint, selected_frontier):
                color = MAP_TARGET_POINT_INDICATOR
            else:
                color = MAP_SOURCE_POINT_INDICATOR
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
                MAP_SOURCE_POINT_INDICATOR,
                thickness,
            )
        self._metric["map"] = new_map
        self._metric["is_feasible"] = self._is_feasible
        # if not self._is_feasible:
        #     self._task._is_episode_active = False

        # Update self._metric with the static metrics
        self._metric.update(self._static_metrics)

        if DEBUG:
            import time

            if not os.path.exists("map_debug"):
                os.mkdir("map_debug")
            img = observations_to_image(
                {}, {f"top_down_map.{k}": v for k, v in self._metric.items()}
            )
            cv2.imwrite(
                f"map_debug/{int(time.time())}_full.png",
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            )

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
            valid_with_viewpoints == MAP_VIEW_POINT_INDICATOR
        ] = MAP_VALID_POINT
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
            is_feasible = MAP_VALID_POINT in values and MAP_VIEW_POINT_INDICATOR in values

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
class FrontierExplorationMapMeasurementConfig(TopDownMapMeasurementConfig):
    type: str = FrontierExplorationMap.__name__
    draw_waypoints: bool = True


cs = ConfigStore.instance()
cs.store(
    package="habitat.task.measurements.frontier_exploration_map",
    group="habitat/task/measurements",
    name="frontier_exploration_map",
    node=FrontierExplorationMapMeasurementConfig,
)
