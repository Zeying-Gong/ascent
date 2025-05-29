# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from torch import Tensor

from vlfm.mapping.object_point_cloud_map import ObjectPointCloudMap
from vlfm.mapping.obstacle_map import ObstacleMap
from vlfm.obs_transformers.utils import image_resize
from vlfm.policy.utils.pointnav_policy import WrappedPointNavResNetPolicy
from vlfm.utils.geometry_utils import get_fov, rho_theta
from vlfm.vlm.blip2 import BLIP2Client
from vlfm.vlm.coco_classes import COCO_CLASSES
from vlfm.vlm.grounding_dino_test import GroundingDINOClient_MF, ObjectDetections
from vlfm.vlm.sam import MobileSAMClient
from vlfm.vlm.yolov7 import YOLOv7Client

try:
    from habitat_baselines.common.tensor_dict import TensorDict

    from vlfm.policy.base_policy import BasePolicy
except Exception:

    class BasePolicy:  # type: ignore
        pass


class BaseObjectNavPolicy(BasePolicy):
    # _target_object: List[str] = [] # str = ""
    _policy_info: Dict[str, Any] = {}
    # _object_masks: Union[np.ndarray, Any] = None  # set by ._update_object_map()
    _stop_action: Union[Tensor, Any] = None  # MUST BE SET BY SUBCLASS
    _observations_cache: Dict[str, Any] = {}
    _non_coco_caption = ""
    _load_yolo: bool = True

    def __init__(
        self,
        pointnav_policy_path: str,
        depth_image_shape: Tuple[int, int],
        pointnav_stop_radius: float,
        object_map_erosion_size: float,
        visualize: bool = True,
        compute_frontiers: bool = True,
        min_obstacle_height: float = 0.15,
        max_obstacle_height: float = 0.88,
        agent_radius: float = 0.18,
        obstacle_map_area_threshold: float = 1.5,
        hole_area_thresh: int = 100000,
        use_vqa: bool = False,
        vqa_prompt: str = "Is this ",
        coco_threshold: float = 0.8,
        non_coco_threshold: float = 0.4,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._object_detector = GroundingDINOClient_MF(port=int(os.environ.get("GROUNDING_DINO_PORT", "12181")))
        self._coco_object_detector = YOLOv7Client(port=int(os.environ.get("YOLOV7_PORT", "12184")))
        self._mobile_sam = MobileSAMClient(port=int(os.environ.get("SAM_PORT", "12183")))
        self._use_vqa = use_vqa
        if use_vqa:
            self._vqa = BLIP2Client(port=int(os.environ.get("BLIP2_PORT", "12185")))
        
        
        self._depth_image_shape = tuple(depth_image_shape)
        self._pointnav_stop_radius = pointnav_stop_radius
        self._visualize = visualize
        self._vqa_prompt = vqa_prompt
        self._coco_threshold = coco_threshold
        self._non_coco_threshold = non_coco_threshold

        ## num_envs
        self._num_envs = kwargs['num_envs']
        self._object_map=[ObjectPointCloudMap(erosion_size=object_map_erosion_size) for _ in range(self._num_envs)]
        self._pointnav_policy = [WrappedPointNavResNetPolicy(pointnav_policy_path) for _ in range(self._num_envs)]
        self._num_steps = [0 for _ in range(self._num_envs)]
        self._did_reset = [False for _ in range(self._num_envs)]
        self._last_goal = [np.zeros(2) for _ in range(self._num_envs)]
        self._done_initializing = [False for _ in range(self._num_envs)]
        self._called_stop = [False for _ in range(self._num_envs)]
        self._compute_frontiers = compute_frontiers
        if compute_frontiers:
            self._obstacle_map = [
                ObstacleMap(
                min_height=min_obstacle_height,
                max_height=max_obstacle_height,
                area_thresh=obstacle_map_area_threshold,
                agent_radius=agent_radius,
                hole_area_thresh=hole_area_thresh,
            )
            for _ in range(self._num_envs)
            ]
        self._target_object = ["" for _ in range(self._num_envs)]
    def _reset(self, env: int=0) -> None:
        self._target_object[env] = ""
        self._pointnav_policy[env].reset()
        self._object_map[env].reset()
        self._last_goal[env] = np.zeros(2)
        self._num_steps[env] = 0
        self._done_initializing[env] = False
        self._called_stop[env] = False
        if self._compute_frontiers:
            self._obstacle_map[env].reset()
        self._did_reset[env] = True

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        detections_list: ObjectDetections,
        deterministic: bool = False,
    ) -> Any:
        """
        Starts the episode by 'initializing' and allowing robot to get its bearings
        (e.g., spinning in place to get a good view of the scene).
        Then, explores the scene until it finds the target object.
        Once the target object is found, it navigates to the object.
        """
        self._pre_step(observations, masks)

        pointnav_action_env_list = []

        for env in range(self._num_envs):
            # object_map_rgbd = self._observations_cache[env]["object_map_rgbd"]
            # detections = [
            #     self._update_object_map(rgb, depth, tf, min_depth, max_depth, fx, fy, env, ori_detections[env])
            #     for (rgb, depth, tf, min_depth, max_depth, fx, fy) in object_map_rgbd
            # ]
            robot_xy = self._observations_cache[env]["robot_xy"]
            goal = self._get_target_object_location(robot_xy, env)

            if not self._done_initializing[env]:  # Initialize
                mode = "initialize"
                pointnav_action = self._initialize(env,masks)
            elif goal is None:  # Haven't found target object yet
                mode = "explore"
                pointnav_action = self._explore(observations,env,masks)
            else:
                mode = "navigate"
                pointnav_action = self._pointnav(goal[:2], stop=True, env=env, ori_masks=masks)

            pointnav_action_env_list.append(pointnav_action)

            action_numpy = pointnav_action.detach().cpu().numpy()[0]

            self._num_steps[env] += 1 # start from 1, 0 means do not start

            if len(action_numpy) == 1:
                action_numpy = action_numpy[0]

            print(f"Env: {env} | Step: {self._num_steps} | Mode: {mode} | Action: {action_numpy}")
            
            self._policy_info[env].update(self._get_policy_info(detections_list[env],env))

            self._observations_cache[env] = {}
            self._did_reset[env] = False

        pointnav_action_tensor = torch.cat(pointnav_action_env_list, dim=0)
        return pointnav_action_tensor, rnn_hidden_states
    def _pre_step(self, observations: "TensorDict", masks: Tensor) -> None:
        assert masks.shape[1] == 1, "Currently only supporting one env at a time"
        if not self._did_reset and masks[0] == 0:
            self._reset()
            self._target_object = observations["objectgoal"]
        try:
            self._cache_observations(observations)
        except IndexError as e:
            print(e)
            print("Reached edge of map, stopping.")
            raise StopIteration
        self._policy_info = {}
    def _initialize(self, env: int, masks: Tensor) -> Tensor:
        raise NotImplementedError

    def _explore(self, observations: "TensorDict", env: int, masks: Tensor) -> Tensor:
        raise NotImplementedError

    def _get_target_object_location(self, position: np.ndarray, env: int = 0) -> Union[None, np.ndarray]:
        if self._object_map[env].has_object(self._target_object[env]):
            return self._object_map[env].get_best_object(self._target_object[env], position)
        else:
            return None

    def _get_policy_info(self, detections: ObjectDetections, env: int = 0) -> Dict[str, Any]:
        if self._object_map[env].has_object(self._target_object[env]):
            target_point_cloud = self._object_map[env].get_target_cloud(self._target_object[env])
        else:
            target_point_cloud = np.array([])
        policy_info = {
            "target_object": self._target_object[env].split("|")[0],
            "gps": str(self._observations_cache[env]["robot_xy"] * np.array([1, -1])),
            "yaw": np.rad2deg(self._observations_cache[env]["robot_heading"]),
            "target_detected": self._object_map[env].has_object(self._target_object[env]),
            "target_point_cloud": target_point_cloud,
            "nav_goal": self._last_goal[env],
            "stop_called": self._called_stop[env],
            # don't render these on egocentric images when making videos:
            "render_below_images": [
                "target_object",
            ],
        }

        if not self._visualize:
            return policy_info

        annotated_depth = self._observations_cache[env]["object_map_rgbd"][0][1] * 255
        annotated_depth = cv2.cvtColor(annotated_depth.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        if self._object_masks[env].sum() > 0:
            # If self._object_masks isn't all zero, get the object segmentations and
            # draw them on the rgb and depth images
            contours, _ = cv2.findContours(self._object_masks[env], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            annotated_rgb = cv2.drawContours(detections.annotated_frame, contours, -1, (255, 0, 0), 2)
            annotated_depth = cv2.drawContours(annotated_depth, contours, -1, (255, 0, 0), 2)
        else:
            annotated_rgb = self._observations_cache[env]["object_map_rgbd"][0][0]
        policy_info["annotated_rgb"] = annotated_rgb
        policy_info["annotated_depth"] = annotated_depth

        # add third_view
        if "third_rgb" in self._observations_cache[env]:
            policy_info["third_rgb"] = self._observations_cache[env]["third_rgb"]
            
        if self._compute_frontiers:
            policy_info["obstacle_map"] = cv2.cvtColor(self._obstacle_map[env].visualize(), cv2.COLOR_BGR2RGB)

        if "DEBUG_INFO" in os.environ:
            policy_info["render_below_images"].append("debug")
            policy_info["debug"] = "debug: " + os.environ["DEBUG_INFO"]

        return policy_info

    def _get_object_detections(self, img: np.ndarray, env: int) -> ObjectDetections:
        target_classes = self._target_object[env].split("|")
        # target_classes.append('stair')
        # self._non_coco_caption += 'stair'
        has_coco = any(c in COCO_CLASSES for c in target_classes) and self._load_yolo
        has_non_coco = any(c not in COCO_CLASSES for c in target_classes)

        detections = (
            self._coco_object_detector.predict(img)
            if has_coco
            else self._object_detector.predict(img, caption=self._non_coco_caption)
        )
        detections.filter_by_class(target_classes) # (target_classes + ["person"])
        det_conf_threshold = self._coco_threshold if has_coco else self._non_coco_threshold
        detections.filter_by_conf(det_conf_threshold)

        if has_coco and has_non_coco and detections.num_detections == 0:
            # Retry with non-coco object detector
            detections = self._object_detector.predict(img, caption=self._non_coco_caption)
            detections.filter_by_class(target_classes) # (target_classes + ["person"])
            detections.filter_by_conf(self._non_coco_threshold)

        return detections

    # def _get_object_detections_with_person_and_stair(self, img: np.ndarray, env: int) -> ObjectDetections:
    #     target_classes = self._target_object[env].split("|")
    #     target_classes.append('stair')
    #     self._non_coco_caption += 'stair'
    #     has_coco = any(c in COCO_CLASSES for c in target_classes) and self._load_yolo
    #     has_non_coco = any(c not in COCO_CLASSES for c in target_classes)

    #     detections = (
    #         self._coco_object_detector.predict(img)
    #         if has_coco
    #         else self._object_detector.predict(img, caption=self._non_coco_caption)
    #     )
    #     detections.filter_by_class(target_classes + ["person"]) # (target_classes)
    #     det_conf_threshold = self._coco_threshold if has_coco else self._non_coco_threshold
    #     detections.filter_by_conf(det_conf_threshold)

    #     if has_coco and has_non_coco and detections.num_detections == 0:
    #         # Retry with non-coco object detector
    #         detections = self._object_detector.predict(img, caption=self._non_coco_caption)
    #         detections.filter_by_class(target_classes + ["person"])
    #         detections.filter_by_conf(self._non_coco_threshold)

    #     return detections
    
    def _pointnav(self, goal: np.ndarray, stop: bool = False, env: int = 0, ori_masks: Tensor = None) -> Tensor:
        """
        Calculates rho and theta from the robot's current position to the goal using the
        gps and heading sensors within the observations and the given goal, then uses
        it to determine the next action to take using the pre-trained pointnav policy.

        Args:
            goal (np.ndarray): The goal to navigate to as (x, y), where x and y are in
                meters.
            stop (bool): Whether to stop if we are close enough to the goal.

        """

        masks = torch.tensor([self._num_steps[env] != 0], dtype=torch.bool, device="cuda")
        if not np.array_equal(goal, self._last_goal[env]):
            if np.linalg.norm(goal - self._last_goal[env]) > 0.1:
                self._pointnav_policy[env].reset()
                masks = torch.zeros_like(masks)
            self._last_goal[env] = goal
        robot_xy = self._observations_cache[env]["robot_xy"]
        heading = self._observations_cache[env]["robot_heading"]
        rho, theta = rho_theta(robot_xy, heading, goal)
        rho_theta_tensor = torch.tensor([[rho, theta]], device="cuda", dtype=torch.float32)
        obs_pointnav = {
            "depth": image_resize(
                self._observations_cache[env]["nav_depth"],
                (self._depth_image_shape[0], self._depth_image_shape[1]),
                channels_last=True,
                interpolation_mode="area",
            ),
            "pointgoal_with_gps_compass": rho_theta_tensor,
        }
        self._policy_info[env]["rho_theta"] = np.array([rho, theta])
        if rho < self._pointnav_stop_radius and stop:
            self._called_stop[env] = True
            return self._stop_action.to(ori_masks.device)
        action = self._pointnav_policy[env].act(obs_pointnav, masks, deterministic=True)
        return action
    
    def _update_object_map(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
        env: int,
        detections: ObjectDetections,
        object_mask_list: List[np.ndarray],
    ) -> ObjectDetections:
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
        # detections = self._get_object_detections(rgb, env) # original detections
        # target_classes = self._target_object[env].split("|")
        # detections.filter_by_class(target_classes) 

        # height, width = rgb.shape[:2]
        # self._object_masks = np.zeros((self._num_envs,height, width), dtype=np.uint8)
        if np.array_equal(depth, np.ones_like(depth)) and detections.num_detections > 0:
            depth = self._infer_depth(rgb, min_depth, max_depth)
            obs = list(self._observations_cache[env]["object_map_rgbd"][0])
            obs[1] = depth
            self._observations_cache[env]["object_map_rgbd"][0] = tuple(obs)

        # modify to filter other (is not goals)
        # for idx in range(len(detections.logits)):
        #     bbox_denorm = detections.boxes[idx] * np.array([width, height, width, height])
        #     object_mask = self._mobile_sam.segment_bbox(rgb, bbox_denorm.tolist())

        # for idx, phrase in enumerate(detections.phrases):
        #     if phrase == self._target_object[env]:
        #         bbox_denorm = detections.boxes[idx] * np.array([width, height, width, height]) # 假设 box 是 [x1, y1, x2, y2] 格式
        #         object_mask = self._mobile_sam.segment_bbox(rgb, bbox_denorm.tolist())

            # If we are using vqa, then use the BLIP2 model to visually confirm whether
            # the contours are actually correct.

            # if self._use_vqa:
            #     contours, _ = cv2.findContours(object_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #     annotated_rgb = cv2.drawContours(rgb.copy(), contours, -1, (255, 0, 0), 2)
            #     question = f"Question: {self._vqa_prompt}"
            #     if not detections.phrases[idx].endswith("ing"):
            #         question += "a "
            #     question += detections.phrases[idx] + "? Answer:"
            #     answer = self._vqa.ask(annotated_rgb, question)
            #     if not answer.lower().startswith("yes"):
            #         continue

            # self._object_masks[env][object_mask > 0] = 1
        for object_mask in object_mask_list[env]:
            self._object_map[env].update_map(
                self._target_object[env],
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

        return detections

    def _cache_observations(self, observations: "TensorDict") -> None:
        """Extracts the rgb, depth, and camera transform from the observations.

        Args:
            observations ("TensorDict"): The observations from the current timestep.
        """
        raise NotImplementedError

    def _infer_depth(self, rgb: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
        """Infers the depth image from the rgb image.

        Args:
            rgb (np.ndarray): The rgb image to infer the depth from.

        Returns:
            np.ndarray: The inferred depth image.
        """
        raise NotImplementedError


@dataclass
class VLFMConfig:
    name: str = "HabitatITMPolicy"
    text_prompt: str = "Seems like there is a target_object ahead."
    pointnav_policy_path: str = "data/pointnav_weights.pth"
    depth_image_shape: Tuple[int, int] = (224, 224)
    pointnav_stop_radius: float = 0.9
    use_max_confidence: bool = False
    object_map_erosion_size: int = 5
    exploration_thresh: float = 0.0
    obstacle_map_area_threshold: float = 1.5  # in square meters
    min_obstacle_height: float = 0.61
    max_obstacle_height: float = 0.88
    hole_area_thresh: int = 100000
    use_vqa: bool = False
    vqa_prompt: str = "Is this "
    coco_threshold: float = 0.8
    non_coco_threshold: float = 0.4
    agent_radius: float = 0.18

    @classmethod  # type: ignore
    @property
    def kwaarg_names(cls) -> List[str]:
        # This returns all the fields listed above, except the name field
        return [f.name for f in fields(VLFMConfig) if f.name != "name"]


cs = ConfigStore.instance()
cs.store(group="policy", name="vlfm_config_base", node=VLFMConfig())
