# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
from torch import Tensor

from vlfm.mapping.frontier_map import FrontierMap
from vlfm.mapping.value_map import ValueMap
from vlfm.policy.base_objectnav_policy import BaseObjectNavPolicy
from vlfm.policy.utils.acyclic_enforcer import AcyclicEnforcer
from vlfm.utils.geometry_utils import closest_point_within_threshold
from vlfm.vlm.blip2itm import BLIP2ITMClient
from vlfm.vlm.detections import ObjectDetections
from ..mapping.obstacle_map import ObstacleMap
try:
    from habitat_baselines.common.tensor_dict import TensorDict
except Exception:
    pass

from vlfm.utils.geometry_utils import get_fov
from vlfm.mapping.object_point_cloud_map import ObjectPointCloudMap

PROMPT_SEPARATOR = "|"


class BaseITMPolicy(BaseObjectNavPolicy):
    _target_object_color: Tuple[int, int, int] = (0, 255, 0)
    _selected__frontier_color: Tuple[int, int, int] = (0, 255, 255)
    _frontier_color: Tuple[int, int, int] = (0, 0, 255)
    _circle_marker_thickness: int = 2
    _circle_marker_radius: int = 5
    # _last_value: float = float("-inf")
    # _last_frontier: np.ndarray = np.zeros(2)

    @staticmethod
    def _vis_reduce_fn(i: np.ndarray) -> np.ndarray:
        return np.max(i, axis=-1)

    def __init__(
        self,
        text_prompt: str,
        use_max_confidence: bool = True,
        sync_explored_areas: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._itm = BLIP2ITMClient(port=int(os.environ.get("BLIP2ITM_PORT", "12182")))
        self._text_prompt = text_prompt

        # parent class already defines obstacle map, use it to filter moving obstacles
        self._value_map=[ValueMap(
            value_channels=len(text_prompt.split(PROMPT_SEPARATOR)),
            use_max_confidence=use_max_confidence,
            obstacle_map=self._obstacle_map if sync_explored_areas else None,
        )
        for _ in range(self._num_envs)]
        self._acyclic_enforcer = [AcyclicEnforcer() for _ in range(self._num_envs)]

        self._last_value = [float("-inf") for _ in range(self._num_envs)]
        self._last_frontier = [np.zeros(2) for _ in range(self._num_envs)]

        self._object_masks = []

    def _reset(self, env: int) -> None:
        super()._reset(env)  
        self._value_map[env].reset()
        self._acyclic_enforcer[env] = AcyclicEnforcer()
        self._last_value[env] = float("-inf")
        self._last_frontier[env] = np.zeros(2)


    def _explore(self, observations: Union[Dict[str, Tensor], "TensorDict"], env: int, masks: Tensor) -> Tensor:
        frontiers = self._observations_cache[env]["frontier_sensor"]
        if np.array_equal(frontiers, np.zeros((1, 2))) or len(frontiers) == 0:
            print("No frontiers found during exploration, stopping.")
            return self._stop_action.to(masks.device)
        best_frontier, best_value = self._get_best_frontier(observations, frontiers, env)
        os.environ["DEBUG_INFO"] = f"Best value: {best_value*100:.2f}%"
        # print(f"Best value: {best_value*100:.2f}%")
        pointnav_action = self._pointnav(best_frontier, stop=False, env=env)

        return pointnav_action

    def _get_best_frontier(
        self,
        observations: Union[Dict[str, Tensor], "TensorDict"],
        frontiers: np.ndarray,
        env: int = 0,
    ) -> Tuple[np.ndarray, float]:
        """Returns the best frontier and its value based on self._value_map.

        Args:
            observations (Union[Dict[str, Tensor], "TensorDict"]): The observations from
                the environment.
            frontiers (np.ndarray): The frontiers to choose from, array of 2D points.

        Returns:
            Tuple[np.ndarray, float]: The best frontier and its value.
        """
        # The points and values will be sorted in descending order
        sorted_pts, sorted_values = self._sort_frontiers_by_value(observations, frontiers, env)
        robot_xy = self._observations_cache[env]["robot_xy"]
        best_frontier_idx = None
        top_two_values = tuple(sorted_values[:2])

        os.environ["DEBUG_INFO"] = ""
        # If there is a last point pursued, then we consider sticking to pursuing it
        # if it is still in the list of frontiers and its current value is not much
        # worse than self._last_value.
        if not np.array_equal(self._last_frontier[env], np.zeros(2)):
            curr_index = None

            for idx, p in enumerate(sorted_pts):
                if np.array_equal(p, self._last_frontier[env]):
                    # Last point is still in the list of frontiers
                    curr_index = idx
                    break

            if curr_index is None:
                closest_index = closest_point_within_threshold(sorted_pts, self._last_frontier[env], threshold=0.5)

                if closest_index != -1:
                    # There is a point close to the last point pursued
                    curr_index = closest_index

            if curr_index is not None:
                curr_value = sorted_values[curr_index]
                if curr_value + 0.01 > self._last_value[env]:
                    # The last point pursued is still in the list of frontiers and its
                    # value is not much worse than self._last_value
                    # print("Sticking to last point.")
                    os.environ["DEBUG_INFO"] += "Sticking to last point. "
                    best_frontier_idx = curr_index

        # If there is no last point pursued, then just take the best point, given that
        # it is not cyclic.
        if best_frontier_idx is None:
            for idx, frontier in enumerate(sorted_pts):
                cyclic = self._acyclic_enforcer[env].check_cyclic(robot_xy, frontier, top_two_values)
                if cyclic:
                    print("Suppressed cyclic frontier.")
                    continue
                best_frontier_idx = idx
                break

        if best_frontier_idx is None:
            print("All frontiers are cyclic. Just choosing the closest one.")
            os.environ["DEBUG_INFO"] += "All frontiers are cyclic. "
            best_frontier_idx = max(
                range(len(frontiers)),
                key=lambda i: np.linalg.norm(frontiers[i] - robot_xy),
            )

        best_frontier = sorted_pts[best_frontier_idx]
        best_value = sorted_values[best_frontier_idx]
        self._acyclic_enforcer[env].add_state_action(robot_xy, best_frontier, top_two_values)
        self._last_value[env] = best_value
        self._last_frontier[env] = best_frontier
        os.environ["DEBUG_INFO"] += f" Best value: {best_value*100:.2f}%"
        
        return best_frontier, best_value

    def _get_policy_info(self, detections: ObjectDetections, env: int = 0) -> Dict[str, Any]:
        policy_info = super()._get_policy_info(detections, env)

        if not self._visualize:
            return policy_info

        markers = []

        # Draw frontiers on to the cost map
        frontiers = self._observations_cache[env]["frontier_sensor"]
        for frontier in frontiers:
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": self._frontier_color,
            }
            markers.append((frontier[:2], marker_kwargs))

        if not np.array_equal(self._last_goal[env], np.zeros(2)):
            # Draw the pointnav goal on to the cost map
            if any(np.array_equal(self._last_goal[env], frontier) for frontier in frontiers):
                color = self._selected__frontier_color
            else:
                color = self._target_object_color
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": color,
            }
            markers.append((self._last_goal[env], marker_kwargs))
        policy_info["value_map"] = cv2.cvtColor(
            self._value_map[env].visualize(markers, reduce_fn=self._vis_reduce_fn),
            cv2.COLOR_BGR2RGB,
        )

        return policy_info

    def _update_obstacle_map(self, observations: "TensorDict", human_mask_list: List[np.ndarray]) -> None:
        for env in range(self._num_envs):
            if self._compute_frontiers:
                self._obstacle_map[env].update_map(
                    self._observations_cache[env]["object_map_rgbd"][0][1],
                    self._observations_cache[env]["object_map_rgbd"][0][2],
                    self._min_depth,
                    self._max_depth,
                    self._fx,
                    self._fy,
                    self._camera_fov,
                    self._object_map[env].movable_clouds,
                    human_mask_list[env],
                )
                frontiers = self._obstacle_map[env].frontiers
                self._obstacle_map[env].update_agent_traj(self._observations_cache[env]["robot_xy"], self._observations_cache[env]["robot_heading"])
                self._observations_cache[env]["frontier_sensor"] = frontiers
            else:
                if "frontier_sensor" in observations:
                    frontiers = observations["frontier_sensor"][env].cpu().numpy()
                else:
                    frontiers = np.array([])
                self._observations_cache[env]["frontier_sensor"] = frontiers

    def _update_value_map(self) -> None:
        for env in range(self._num_envs):
            all_rgb = [i[0] for i in self._observations_cache[env]["value_map_rgbd"]]            
            cosines = [
                [
                    self._itm.cosine(
                        all_rgb[0],
                        p.replace("target_object", self._target_object[env].replace("|", "/")),
                    )
                    for p in self._text_prompt.split(PROMPT_SEPARATOR)
                ]
                # for rgb in all_rgb
            ]
            # for cosine, (rgb, depth, tf, min_depth, max_depth, fov) in zip(
            #     cosines, self._observations_cache[env]["value_map_rgbd"]
            # ):
            self._value_map[env].update_map(np.array(cosines[0]), 
                                            self._observations_cache[env]["value_map_rgbd"][0][1], 
                                            self._observations_cache[env]["value_map_rgbd"][0][2],
                                            self._observations_cache[env]["value_map_rgbd"][0][3],
                                            self._observations_cache[env]["value_map_rgbd"][0][4],
                                            self._observations_cache[env]["value_map_rgbd"][0][5])

            self._value_map[env].update_agent_traj(
                self._observations_cache[env]["robot_xy"],
                self._observations_cache[env]["robot_heading"],
            )

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        raise NotImplementedError


class ITMPolicy(BaseITMPolicy):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._frontier_map: FrontierMap = FrontierMap()

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        self._pre_step(observations, masks)
        if self._visualize:
            self._update_value_map()
        return super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic)

    def _reset(self) -> None:
        super()._reset()
        self._frontier_map.reset()

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        rgb = self._observations_cache["object_map_rgbd"][0][0]
        text = self._text_prompt.replace("target_object", self._target_object)
        self._frontier_map.update(frontiers, rgb, text)  # type: ignore
        return self._frontier_map.sort_waypoints()


class ITMPolicyV2(BaseITMPolicy):
    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Any:
        self._pre_step(observations, masks)
        img_height, img_width = observations["rgb"].shape[1:3]
        detections_list, human_mask_list = self._update_object_map(img_height, img_width)
        self._update_obstacle_map(observations, human_mask_list)
        self._update_value_map()
        return super().act(observations, rnn_hidden_states, prev_actions, masks, detections_list, deterministic)

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray, env: int = 0,
    ) -> Tuple[np.ndarray, List[float]]:
        sorted_frontiers, sorted_values = self._value_map[env].sort_waypoints(frontiers, 0.5)
        return sorted_frontiers, sorted_values

    def _update_object_map(self, height: int, width: int) -> Tuple[ ObjectDetections, List[np.ndarray] ]:
        """
        Updates the object map with the given rgb and depth images, and the given
        transformation matrix from the camera to the episodic coordinate frame.

        Args:
            rgb (np.ndarray): The rgb image to use for updating the object map. Used for
                object detection and Mobile SAM segmentation to extract better object
                point clouds.
            depth (np.ndarray): The depth image to use for updating the object map. It
                is normalized to the range [0, 1] and has a shape of (height, width).
            tf_camera_to_episodic (np.ndarray): The transformation matrix from the
                camera to the episodic coordinate frame.
            min_depth (float): The minimum depth value (in meters) of the depth image.
            max_depth (float): The maximum depth value (in meters) of the depth image.
            fx (float): The focal length of the camera in the x direction.
            fy (float): The focal length of the camera in the y direction.

        Returns:
            ObjectDetections: The object detections from the object detector.
        """
        detections_list = []
        human_mask_list = np.zeros((self._num_envs,height, width), dtype=np.uint8)
        for env in range(self._num_envs):
            object_map_rgbd = self._observations_cache[env]["object_map_rgbd"]
            rgb, depth, tf_camera_to_episodic, min_depth, max_depth, fx, fy = object_map_rgbd[0]
            detections = self._get_object_detections(rgb, env)
            self._object_masks = np.zeros((self._num_envs,height, width), dtype=np.uint8)
            if np.array_equal(depth, np.ones_like(depth)) and detections.num_detections > 0:
                depth = self._infer_depth(rgb, min_depth, max_depth)
                obs = list(self._observations_cache[env]["object_map_rgbd"][0])
                obs[1] = depth
                self._observations_cache["object_map_rgbd"][0] = tuple(obs)
            for idx, phrase in enumerate(detections.phrases):
                bbox_denorm = detections.boxes[idx] * np.array([width, height, width, height])
                object_mask = self._mobile_sam.segment_bbox(rgb, bbox_denorm.tolist())

                # If we are using vqa, then use the BLIP2 model to visually confirm whether
                # the contours are actually correct.
                if self._use_vqa:
                    contours, _ = cv2.findContours(object_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    annotated_rgb = cv2.drawContours(rgb.copy(), contours, -1, (255, 0, 0), 2)
                    question = f"Question: {self._vqa_prompt}"
                    if not detections.phrases[idx].endswith("ing"):
                        question += "a "
                    question += detections.phrases[idx] + "? Answer:"
                    answer = self._vqa.ask(annotated_rgb, question)
                    if not answer.lower().startswith("yes"):
                        continue

                self._object_masks[env][object_mask > 0] = 1 # for drawing
                
                if phrase == "person":
                    human_mask_list[env][object_mask > 0] = 1
                    self._object_map[env].update_movable_clouds(
                        phrase, # no phrase, because object_map only record target (especially after update_explored)
                        depth,
                        object_mask,
                        tf_camera_to_episodic,
                        min_depth,
                        max_depth,
                        fx,
                        fy,
                    )
                else:
                    self._object_map[env].update_map(
                        self._target_object[env], # no phrase, because object_map only record target (especially after update_explored)
                        depth,
                        object_mask,
                        tf_camera_to_episodic,
                        min_depth,
                        max_depth,
                        fx,
                        fy,
                    )

            cone_fov = get_fov(fx, depth.shape[1])
            self._object_map[env].update_explored(tf_camera_to_episodic, max_depth, cone_fov)
            detections_list.append(detections)
        return detections_list, human_mask_list

class ITMPolicyV3(ITMPolicyV2):
    def __init__(self, exploration_thresh: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._exploration_thresh = exploration_thresh

        def visualize_value_map(arr: np.ndarray) -> np.ndarray:
            # Get the values in the first channel
            first_channel = arr[:, :, 0]
            # Get the max values across the two channels
            max_values = np.max(arr, axis=2)
            # Create a boolean mask where the first channel is above the threshold
            mask = first_channel > exploration_thresh
            # Use the mask to select from the first channel or max values
            result = np.where(mask, first_channel, max_values)

            return result

        self._vis_reduce_fn = visualize_value_map  # type: ignore

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        sorted_frontiers, sorted_values = self._value_map.sort_waypoints(frontiers, 0.5, reduce_fn=self._reduce_values)

        return sorted_frontiers, sorted_values

    def _reduce_values(self, values: List[Tuple[float, float]]) -> List[float]:
        """
        Reduce the values to a single value per frontier

        Args:
            values: A list of tuples of the form (target_value, exploration_value). If
                the highest target_value of all the value tuples is below the threshold,
                then we return the second element (exploration_value) of each tuple.
                Otherwise, we return the first element (target_value) of each tuple.

        Returns:
            A list of values, one per frontier.
        """
        target_values = [v[0] for v in values]
        max_target_value = max(target_values)

        if max_target_value < self._exploration_thresh:
            explore_values = [v[1] for v in values]
            return explore_values
        else:
            return [v[0] for v in values]
