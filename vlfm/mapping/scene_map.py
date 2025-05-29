# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Union, Dict, Optional, List

import cv2
import numpy as np
from frontier_exploration.frontier_detection import detect_frontier_waypoints
from frontier_exploration.utils.fog_of_war import get_two_farthest_points, vectorize_get_line_points

from vlfm.mapping.base_map import BaseMap
from vlfm.utils.geometry_utils import extract_yaw, get_point_cloud, transform_points
from vlfm.utils.img_utils import fill_small_holes

from vlfm.vlm.detections import ObjectDetections
from vlfm.mapping.object_point_cloud_map import ObjectPointCloudMap

from collections import deque

import os
from frontier_exploration.frontier_detection import contour_to_frontiers, interpolate_contour, get_frontier_midpoint, get_closest_frontier_point
import open3d as o3d
import matplotlib.pyplot as plt
from frontier_exploration.utils.general_utils import wrap_heading

class SceneMap(BaseMap):
    """Generates two maps; one representing the area that the robot has explored so far,
    and another representing the obstacles that the robot has seen so far.
    """

    def __init__(
        self,
        min_height: float,
        max_height: float,
        agent_radius: float,
        area_thresh: float = 3.0,  # square meters
        hole_area_thresh: int = 100000,  # square pixels
        size: int = 1000,
        pixels_per_meter: int = 20,
    ):
        super().__init__(size, pixels_per_meter)

        # initialize class variable
        self._map_dtype = np.dtype(np.uint8)
        self._map = np.zeros((size, size), dtype=np.uint8)
        
    def reset(self) -> None:
        super().reset()
        # initialize class variable
        self._map_dtype = np.dtype(np.uint8)
        
        
    def update_map(
        self,
        depth: Union[np.ndarray, Any],
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
        topdown_fov: float,
        navigable_map: np.ndarray,
        new_value: int, # np.uint8,
    ) -> None:

        # Update the explored area
        agent_xy_location = tf_camera_to_episodic[:2, 3]
        agent_pixel_location = self._xy_to_px(agent_xy_location.reshape(1, 2))[0]
        new_explored_area = reveal_fog_of_war(
            top_down_map=navigable_map.astype(np.uint8),
            current_fog_of_war_mask=np.zeros_like(self._map, dtype=np.uint8),
            current_point=agent_pixel_location[::-1],
            current_angle=-extract_yaw(tf_camera_to_episodic),
            fov=np.rad2deg(topdown_fov),
            max_line_len=max_depth * self.pixels_per_meter,
        )
        new_explored_area = cv2.dilate(new_explored_area, np.ones((3, 3), np.uint8), iterations=1)
        self._map[new_explored_area > 0] = new_value + 1 # 0 is background
        # 考虑要不要加
        # self._map[navigable_map == 0] = 0
         
        # 考虑连通性的，可能无意义
        # contours, _ = cv2.findContours(
        #     self._map.astype(np.uint8),
        #     cv2.RETR_EXTERNAL,
        #     cv2.CHAIN_APPROX_SIMPLE,
        # )
        # if len(contours) > 1:
        #     min_dist = np.inf
        #     best_idx = 0
        #     for idx, cnt in enumerate(contours):
        #         dist = cv2.pointPolygonTest(cnt, tuple([int(i) for i in agent_pixel_location]), True)
        #         if dist >= 0:
        #             best_idx = idx
        #             break
        #         elif abs(dist) < min_dist:
        #             min_dist = abs(dist)
        #             best_idx = idx
        #     new_area = np.zeros_like(self.explored_area, dtype=np.uint8)
        #     cv2.drawContours(new_area, contours, best_idx, 1, -1)  # type: ignore
        #     self.explored_area = new_area.astype(bool)
            
    def visualize(self, color_map:np.ndarray) -> np.ndarray:
        
        vis_img = color_map[self._map]
        vis_img = vis_img.astype(np.uint8)
        vis_img = cv2.flip(vis_img, 0)

        if len(self._camera_positions) > 0:
            self._traj_vis.draw_trajectory(
                vis_img,
                self._camera_positions,
                self._last_camera_yaw,
            )

        return vis_img

# for scene classfication, make a new one 
def reveal_fog_of_war(
    top_down_map: np.ndarray,
    current_fog_of_war_mask: np.ndarray,
    current_point: np.ndarray,
    current_angle: float,
    fov: float = 90,
    max_line_len: float = 100,
) -> np.ndarray:
    curr_pt_cv2 = current_point[::-1].astype(int)
    angle_cv2 = np.rad2deg(wrap_heading(-current_angle + np.pi / 2))

    cone_mask = cv2.ellipse(
        np.zeros_like(top_down_map),
        curr_pt_cv2,
        (int(max_line_len), int(max_line_len)),
        0,
        angle_cv2 - fov / 2,
        angle_cv2 + fov / 2,
        1,
        -1,
    )

    # Create a mask of pixels that are both in the cone and NOT in the top_down_map, actually the obstacle map
    obstacles_in_cone = cv2.bitwise_and(cone_mask, 1 - top_down_map)

    # Find the contours of the obstacles in the cone
    obstacle_contours, _ = cv2.findContours(
        obstacles_in_cone, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(obstacle_contours) == 0:
        return current_fog_of_war_mask  # there were no obstacles in the cone

    # Find the two points in each contour that form the smallest and largest angles
    # from the current position
    points = []
    for cnt in obstacle_contours:
        if cv2.isContourConvex(cnt):
            pt1, pt2 = get_two_farthest_points(curr_pt_cv2, cnt, angle_cv2)
            points.append(pt1.reshape(-1, 2))
            points.append(pt2.reshape(-1, 2))
        else:
            # Just add every point in the contour
            points.append(cnt.reshape(-1, 2))
    points = np.concatenate(points, axis=0)

    # Fragment the cone using obstacles and two lines per obstacle in the cone
    visible_cone_mask = cv2.bitwise_and(cone_mask, top_down_map)
    line_points = vectorize_get_line_points(curr_pt_cv2, points, max_line_len * 1.05)
    # Draw all lines simultaneously using cv2.polylines
    cv2.polylines(visible_cone_mask, line_points, isClosed=False, color=0, thickness=2)

    # Identify the contour that is closest to the current position
    final_contours, _ = cv2.findContours(
        visible_cone_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    visible_area = None
    min_dist = np.inf
    for cnt in final_contours:
        pt = tuple([int(i) for i in curr_pt_cv2])
        dist = abs(cv2.pointPolygonTest(cnt, pt, True))
        if dist < min_dist:
            min_dist = dist
            visible_area = cnt

    if min_dist > 3:
        return current_fog_of_war_mask  # the closest contour was too far away

    new_fog = cv2.drawContours(current_fog_of_war_mask, [visible_area], 0, 1, -1)

    return new_fog