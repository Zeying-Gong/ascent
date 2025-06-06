from vlfm.policy.itm_policy import ITMPolicy, ITMPolicyV2, ITMPolicyV3
from vlfm.policy.habitat_policies import HabitatMixin
from habitat_baselines.common.baseline_registry import baseline_registry
from vlfm.vlm.grounding_dino import ObjectDetections
from gym import spaces
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Union, List
from vlfm.policy.base_objectnav_policy import BaseObjectNavPolicy, VLFMConfig
from habitat_baselines.common.tensor_dict import DictTree, TensorDict
from depth_camera_filtering import filter_depth
from vlfm.utils.geometry_utils import xyz_yaw_to_tf_matrix
import numpy as np
import torch
import cv2
import os
from RedNet.RedNet_model import load_rednet
from constants import MPCAT40_RGB_COLORS, MPCAT40_NAMES
from torch import Tensor
from vlfm.utils.geometry_utils import closest_point_within_threshold
# from vlfm.policy.habitat_policies import TorchActionIDs
from habitat_baselines.rl.ppo.policy import PolicyActionData
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from vlfm.policy.habitat_policies import HM3D_ID_TO_NAME, MP3D_ID_TO_NAME
from vlfm.vlm.coco_classes import COCO_CLASSES
import matplotlib.pyplot as plt
from vlfm.utils.geometry_utils import get_fov, rho_theta
from copy import deepcopy
from vlfm.obs_transformers.utils import image_resize
from vlfm.vlm.blip2 import BLIP2Client
from vlfm.vlm.coco_classes import COCO_CLASSES
from vlfm.vlm.grounding_dino_test import GroundingDINOClient_MF, ObjectDetections
from vlfm.vlm.sam import MobileSAMClient
from vlfm.vlm.yolov7 import YOLOv7Client
from vlfm.vlm.blip2itm import BLIP2ITMClient
from vlfm.policy.utils.pointnav_policy import WrappedPointNavResNetPolicy
from vlfm.mapping.object_point_cloud_map import ObjectPointCloudMap
from vlfm.mapping.obstacle_map import ObstacleMap
from vlfm.mapping.value_map import ValueMap
from vlfm.policy.utils.acyclic_enforcer import AcyclicEnforcer

PROMPT_SEPARATOR = "|"
STAIR_CLASS_ID = 17  # MPCAT40中 楼梯的类别编号是 16 + 1
class TorchActionIDs_plook:
    STOP = torch.tensor([[0]], dtype=torch.long)
    MOVE_FORWARD = torch.tensor([[1]], dtype=torch.long)
    TURN_LEFT = torch.tensor([[2]], dtype=torch.long)
    TURN_RIGHT = torch.tensor([[3]], dtype=torch.long)
    LOOK_UP = torch.tensor([[4]], dtype=torch.long)
    LOOK_DOWN = torch.tensor([[5]], dtype=torch.long)

def xyz_yaw_pitch_roll_to_tf_matrix(xyz: np.ndarray, yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Converts a given position and yaw, pitch, roll angles to a 4x4 transformation matrix.

    Args:
        xyz (np.ndarray): A 3D vector representing the position.
        yaw (float): The yaw angle in radians (rotation around Z-axis).
        pitch (float): The pitch angle in radians (rotation around Y-axis).
        roll (float): The roll angle in radians (rotation around X-axis).

    Returns:
        np.ndarray: A 4x4 transformation matrix.
    """
    x, y, z = xyz
    
    # Rotation matrices for yaw, pitch, roll
    R_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1],
    ])
    R_pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)],
    ])
    R_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)],
    ])
    
    # Combined rotation matrix
    R = R_yaw @ R_pitch @ R_roll
    
    # Construct 4x4 transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R  # Rotation
    transformation_matrix[:3, 3] = [x, y, z]  # Translation

    return transformation_matrix

def check_stairs_in_upper_50_percent(mask):
    """
    检查在图像的上方30%区域是否有STAIR_CLASS_ID的标记
    参数：
    - mask: 布尔值数组，表示各像素是否属于STARR_CLASS_ID
    
    返回：
    - 如果上方30%区域有True，则返回True，否则返回False
    """
    # 获取图像的高度
    height = mask.shape[0]
    
    # 计算上方50%的区域的高度范围
    upper_50_height = int(height * 0.5)
    
    # 获取上方50%的区域的掩码
    upper_50_mask = mask[:upper_50_height, :]
    
    print(f"Stair upper 50% points: {np.sum(upper_50_mask)}")
    # 检查该区域内是否有True
    if np.sum(upper_50_mask) > 50:  # 如果上方50%区域内有True
        return True
    return False

@baseline_registry.register_policy
class HabitatITMPolicy_MF(HabitatMixin, ITMPolicyV2):
    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        # super().__init__(*args, **kwargs)
        # Policy
        self._action_space = kwargs["action_space"]

        # BaseObjectNavPolicy
        self._policy_info = {}
        self._stop_action = TorchActionIDs_plook.STOP  # MUST BE SET BY SUBCLASS
        self._observations_cache = {}
        self._non_coco_caption = ""
        self._load_yolo: bool = True
        self._object_detector = GroundingDINOClient_MF(port=int(os.environ.get("GROUNDING_DINO_PORT", "12181")))
        self._coco_object_detector = YOLOv7Client(port=int(os.environ.get("YOLOV7_PORT", "12184")))
        self._mobile_sam = MobileSAMClient(port=int(os.environ.get("SAM_PORT", "12183")))
        self._use_vqa = kwargs["use_vqa"]
        if self._use_vqa:
            self._vqa = BLIP2Client(port=int(os.environ.get("BLIP2_PORT", "12185")))
        self._depth_image_shape = tuple(kwargs["depth_image_shape"])
        self._pointnav_stop_radius = kwargs["pointnav_stop_radius"]
        self._visualize = kwargs["visualize"]
        self._vqa_prompt = kwargs["vqa_prompt"]
        self._coco_threshold = kwargs["coco_threshold"]
        self._non_coco_threshold = kwargs["non_coco_threshold"]
        ## num_envs
        self._num_envs = kwargs['num_envs']
        self._object_map_erosion_size = kwargs["object_map_erosion_size"]
        # self._object_map=[ObjectPointCloudMap(erosion_size=kwargs["object_map_erosion_size"]) for _ in range(self._num_envs)] # 
        self._object_map_list = [[ObjectPointCloudMap(erosion_size=self._object_map_erosion_size)] for _ in range(self._num_envs)]
        self._pointnav_policy = [WrappedPointNavResNetPolicy(kwargs["pointnav_policy_path"]) for _ in range(self._num_envs)]
        self._num_steps = [0 for _ in range(self._num_envs)]
        self._did_reset = [False for _ in range(self._num_envs)]
        self._last_goal = [np.zeros(2) for _ in range(self._num_envs)]
        self._done_initializing = [False for _ in range(self._num_envs)]
        self._called_stop = [False for _ in range(self._num_envs)]
        self._compute_frontiers = True # kwargs["compute_frontiers"]
        self.min_obstacle_height = kwargs["min_obstacle_height"]
        self.max_obstacle_height = kwargs["max_obstacle_height"]
        self.obstacle_map_area_threshold = kwargs["obstacle_map_area_threshold"]
        self.agent_radius = kwargs["agent_radius"]
        self.hole_area_thresh = kwargs["hole_area_thresh"]
        if self._compute_frontiers:
            self._obstacle_map_list = [
                [ObstacleMap(
                min_height=self.min_obstacle_height,
                max_height=self.max_obstacle_height,
                area_thresh=self.obstacle_map_area_threshold,
                agent_radius=self.agent_radius,
                hole_area_thresh=self.hole_area_thresh,
            )]
            for _ in range(self._num_envs)
            ]
        self._target_object = ["" for _ in range(self._num_envs)]

        # BaseITMPolicy
        self._target_object_color = (0, 255, 0)
        self._selected__frontier_color = (0, 255, 255)
        self._frontier_color = (0, 0, 255)
        self._circle_marker_thickness = 2
        self._circle_marker_radius = 5

        self._itm = BLIP2ITMClient(port=int(os.environ.get("BLIP2ITM_PORT", "13182")))
        self._text_prompt = kwargs["text_prompt"]

        self.use_max_confidence = kwargs["use_max_confidence"]
        self._value_map_list = [ [ValueMap(
            value_channels=len(self._text_prompt.split(PROMPT_SEPARATOR)),
            use_max_confidence = self.use_max_confidence,
            obstacle_map=None,  # self._obstacle_map if kwargs["sync_explored_areas"] else None,
        )]
        for _ in range(self._num_envs)]
        self._acyclic_enforcer = [AcyclicEnforcer() for _ in range(self._num_envs)]

        self._last_value = [float("-inf") for _ in range(self._num_envs)]
        self._last_frontier = [np.zeros(2) for _ in range(self._num_envs)]

        self._object_masks = [] # do not know
        
        # HabitatMixin
        self._camera_height = kwargs["camera_height"]
        self._min_depth = kwargs["min_depth"]
        self._max_depth = kwargs["max_depth"]
        camera_fov_rad = np.deg2rad(kwargs["camera_fov"])
        self._camera_fov = camera_fov_rad
        self._fx = self._fy = kwargs["image_width"] / (2 * np.tan(camera_fov_rad / 2))
        self._cx, self._cy = kwargs["image_width"] // 2, kwargs['full_config'].habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height // 2,
        self._dataset_type = kwargs["dataset_type"]
        self._observations_cache = [{} for _ in range(self._num_envs)]

        if "full_config" in kwargs:
            self.device = (
                torch.device("cuda:{}".format(kwargs["full_config"].habitat_baselines.torch_gpu_id))
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            self._pitch_angle_offset = kwargs["full_config"].habitat.task.actions.look_down.tilt_angle
        else:
            self.device = (
                torch.device("cuda:0")  # {}".format(full_config.habitat_baselines.torch_gpu_id))
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            self._pitch_angle_offset = 30
        
        # To find stair step
        self.red_sem_pred = load_rednet(
            self.device, ckpt='RedNet/model/rednet_semmap_mp3d_40.pth', resize=True, # since we train on half-vision
        )
        self.red_sem_pred.eval()
        
        self._pitch_angle = [0 for _ in range(self._num_envs)]
        self._person_masks = []
        self._stair_masks = []
        self._climb_stair_over = [True for _ in range(self._num_envs)]
        self._reach_stair = [False for _ in range(self._num_envs)]
        self._reach_stair_centroid = [False for _ in range(self._num_envs)]
        self._stair_frontier = [None for _ in range(self._num_envs)]
        # self.step_count = 0 # 设置为 DEBUG 时启用计数
        
        # add to manage the maps of each floor
        self._cur_floor_index = [0 for _ in range(self._num_envs)]
        self._object_map = [self._object_map_list[env][self._cur_floor_index[env]] for env in range(self._num_envs)]
        self._obstacle_map = [self._obstacle_map_list[env][self._cur_floor_index[env]] for env in range(self._num_envs)]
        self._value_map = [self._value_map_list[env][self._cur_floor_index[env]] for env in range(self._num_envs)]

        self.target_detection_list = [None for _ in range(self._num_envs)]
        self.coco_detection_list = [None for _ in range(self._num_envs)]
        self.non_coco_detection_list = [None for _ in range(self._num_envs)]

        self._climb_stair_flag = [0 for _ in range(self._num_envs)]
        self._stair_dilate_flag = [False for _ in range(self._num_envs)]
        # self._start_initial_step = [0 for _ in range(self._num_envs)]
        self.target_might_detected = [False for _ in range(self._num_envs)]

        self._frontier_stick_step = [0 for _ in range(self._num_envs)]
        self._last_frontier_distance = [0 for _ in range(self._num_envs)]

        # RedNet
        self.red_semantic_pred_list = [[] for _ in range(self._num_envs)]
        self.seg_map_color_list = [[] for _ in range(self._num_envs)]

        # self._last_carrot_dist = [[] for _ in range(self._num_envs)]
        self._last_carrot_xy = [[] for _ in range(self._num_envs)]
        self._last_carrot_px = [[] for _ in range(self._num_envs)]
        # self._last_carrot_xy = [[] for _ in range(self._num_envs)]
        self._carrot_goal_xy = [[] for _ in range(self._num_envs)]

        self._temp_stair_map = [[] for _ in range(self._num_envs)]
        self.history_action = [[] for _ in range(self._num_envs)] 

        ## double_check
        self._try_to_navigate = [False for _ in range(self._num_envs)]
        # self._double_check_goal = [False for _ in range(self._num_envs)]
        self._initialize_step = [0 for _ in range(self._num_envs)]

        ## stop distance
        self._might_close_to_goal = [False for _ in range(self._num_envs)]
        self.min_distance_px = [np.inf for _ in range(self._num_envs)]
    def _reset(self, env: int) -> None:

        self._target_object[env] = ""
        self._pointnav_policy[env].reset()
        self._last_goal[env] = np.zeros(2)
        self._num_steps[env] = 0

        self._done_initializing[env] = False
        # self._start_initial_step[env] = 0 # 
        self._called_stop[env] = False
        self._did_reset[env] = True
        self._acyclic_enforcer[env] = AcyclicEnforcer()
        self._last_value[env] = float("-inf")
        self._last_frontier[env] = np.zeros(2)

        self._cur_floor_index[env] = 0
        self._object_map[env] = self._object_map_list[env][self._cur_floor_index[env]]
        self._value_map[env] = self._value_map_list[env][self._cur_floor_index[env]]
        self._object_map[env].reset()
        self._value_map[env].reset()
        del self._object_map_list[env][1:]  
        del self._value_map_list[env][1:]

        if self._compute_frontiers:
            self._obstacle_map[env] = self._obstacle_map_list[env][self._cur_floor_index[env]]
            self._obstacle_map[env].reset()
            del self._obstacle_map_list[env][1:]

        self._initialize_step[env] = 0
            
        # 防止之前episode爬楼梯异常退出
        self._climb_stair_over[env] = True
        self._reach_stair[env] = False
        self._reach_stair_centroid[env] = False
        self._stair_dilate_flag[env] = False
        self._pitch_angle[env] = 0

        # 防止识别正确之后造成误识别
        self.target_might_detected[env] = False
        self._frontier_stick_step[env] = 0
        self._last_frontier_distance[env] = 0

        self._person_masks = []
        self._stair_masks = []

        self._last_carrot_xy[env] = []
        self._last_carrot_px[env] = []
        self._carrot_goal_xy[env] = []
        # RedNet
        # self.red_semantic_pred_list[env] = []
        # self.seg_map_color_list[env] = []
        self._temp_stair_map[env] = []
        self.history_action[env] = []

        ## double_check
        self._try_to_navigate[env] = False
        # self._double_check_goal[env] = False

        self._might_close_to_goal[env] = False
        self.min_distance_px[env] = np.inf
    def _cache_observations(self: Union["HabitatMixin", BaseObjectNavPolicy], observations: TensorDict, env: int) -> None:
        """Caches the rgb, depth, and camera transform from the observations.

        Args:
           observations (TensorDict): The observations from the current timestep.
        """
        if len(self._observations_cache[env]) > 0:
            return
        if "articulated_agent_jaw_depth" in observations:
            rgb = observations["articulated_agent_jaw_rgb"][env].cpu().numpy()
            depth = observations["articulated_agent_jaw_depth"][env].cpu().numpy()
            x, y = observations["gps"][env].cpu().numpy()
            camera_yaw = observations["compass"][env].cpu().item()
            depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
            # Habitat GPS makes west negative, so flip y
            camera_position = np.array([x, -y, self._camera_height])
            robot_xy = camera_position[:2]
            tf_camera_to_episodic = xyz_yaw_to_tf_matrix(camera_position, camera_yaw)

        else:
            rgb = observations["rgb"][env].cpu().numpy() ## modify this to fit on multiple environments
            depth = observations["depth"][env].cpu().numpy()
            x, y = observations["gps"][env].cpu().numpy()
            camera_yaw = observations["compass"][env].cpu().item()
            depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
            # Habitat GPS makes west negative, so flip y
            camera_position = np.array([x, -y, self._camera_height])
            robot_xy = camera_position[:2]
            camera_pitch = np.radians(-self._pitch_angle[env]) # 应该是弧度制 -
            camera_roll = 0
            # tf_camera_to_episodic = xyz_yaw_to_tf_matrix(camera_position, camera_yaw) # add pitch 
            tf_camera_to_episodic = xyz_yaw_pitch_roll_to_tf_matrix(camera_position, camera_yaw, camera_pitch, camera_roll)
        # self._obstacle_map: ObstacleMap # original obstacle map place

        self._observations_cache[env] = {
            # "frontier_sensor": frontiers,
            # "nav_depth": observations["depth"],  # for general depth
            "robot_xy": robot_xy,
            "robot_heading": camera_yaw,
            "tf_camera_to_episodic": tf_camera_to_episodic,
            "object_map_rgbd": [
                (
                    rgb,
                    depth,
                    tf_camera_to_episodic,
                    self._min_depth,
                    self._max_depth,
                    self._fx,
                    self._fy,
                )
            ],
            "value_map_rgbd": [
                (
                    rgb,
                    depth,
                    tf_camera_to_episodic,
                    self._min_depth,
                    self._max_depth,
                    self._camera_fov,
                )
            ],
            "habitat_start_yaw": observations["heading"][env].item(),
            
        }

        ## add for rednet
        if "articulated_agent_jaw_rgb" in observations:
            self._observations_cache[env]["nav_rgb"]=torch.unsqueeze(observations["articulated_agent_jaw_rgb"][env], dim=0)
        else:
            self._observations_cache[env]["nav_rgb"]=torch.unsqueeze(observations["rgb"][env], dim=0)
        
        if "articulated_agent_jaw_depth" in observations:
            self._observations_cache[env]["nav_depth"]=torch.unsqueeze(observations["articulated_agent_jaw_depth"][env], dim=0)
        else:
            self._observations_cache[env]["nav_depth"]=torch.unsqueeze(observations["depth"][env], dim=0)

        if "third_rgb" in observations:
            self._observations_cache[env]["third_rgb"]=observations["third_rgb"][env].cpu().numpy()

    def _get_policy_info(self, detections: ObjectDetections,  env: int = 0) -> Dict[str, Any]: # seg_map_color:np.ndarray,
        """Get policy info for logging, especially, we add rednet to add seg_map"""
        # 获取目标点云信息
        if self._object_map[env].has_object(self._target_object[env]):
            target_point_cloud = self._object_map[env].get_target_cloud(self._target_object[env])
        else:
            target_point_cloud = np.array([])

        # 初始化 policy_info
        policy_info = {
            "target_object": self._target_object[env].split("|")[0],
            "gps": str(self._observations_cache[env]["robot_xy"] * np.array([1, -1])),
            "yaw": np.rad2deg(self._observations_cache[env]["robot_heading"]),
            "target_detected": self._object_map[env].has_object(self._target_object[env]),
            "target_point_cloud": target_point_cloud,
            "nav_goal": self._last_goal[env],
            "stop_called": self._called_stop[env],
            "render_below_images": ["target_object"],
            "seg_map": self.seg_map_color_list[env], # seg_map_color,
            "num_steps": self._num_steps[env],
            # "floor_num_steps": self._obstacle_map[env]._floor_num_steps,
        }

        # 若不需要可视化,直接返回
        if not self._visualize:
            return policy_info

        # 处理注释深度图和 RGB 图
        annotated_depth = self._observations_cache[env]["object_map_rgbd"][0][1] * 255
        annotated_depth = cv2.cvtColor(annotated_depth.astype(np.uint8), cv2.COLOR_GRAY2RGB)

        if self._object_masks[env].sum() > 0:
            contours, _ = cv2.findContours(self._object_masks[env], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            annotated_rgb = cv2.drawContours(detections.annotated_frame, contours, -1, (255, 0, 0), 2)
            annotated_depth = cv2.drawContours(annotated_depth, contours, -1, (255, 0, 0), 2)
        elif self._person_masks[env].sum() > 0:
            contours, _ = cv2.findContours(self._person_masks[env].astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            annotated_rgb = cv2.drawContours(self.coco_detection_list[env].annotated_frame, contours, -1, (255, 0, 0), 2)
            annotated_depth = cv2.drawContours(annotated_depth, contours, -1, (255, 0, 0), 2)
        elif self._stair_masks[env].sum() > 0:
            contours, _ = cv2.findContours(self._stair_masks[env].astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            annotated_rgb = cv2.drawContours(self.non_coco_detection_list[env].annotated_frame, contours, -1, (255, 0, 0), 2)
            annotated_depth = cv2.drawContours(annotated_depth, contours, -1, (255, 0, 0), 2)
        else:
            annotated_rgb = self._observations_cache[env]["object_map_rgbd"][0][0]

        policy_info["annotated_rgb"] = annotated_rgb
        policy_info["annotated_depth"] = annotated_depth

        # 添加第三视角 RGB
        if "third_rgb" in self._observations_cache[env]:
            policy_info["third_rgb"] = self._observations_cache[env]["third_rgb"]

        # 绘制 frontiers
        if self._compute_frontiers:
            policy_info["obstacle_map"] = cv2.cvtColor(self._obstacle_map[env].visualize(), cv2.COLOR_BGR2RGB)

        policy_info["object_map"] = cv2.cvtColor(self._object_map[env].visualize(), cv2.COLOR_BGR2RGB)

        markers = []
        frontiers = self._observations_cache[env]["frontier_sensor"]
        for frontier in frontiers:
            markers.append((frontier[:2], {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": self._frontier_color,
            }))

        if not np.array_equal(self._last_goal[env], np.zeros(2)):
            goal_color = (self._selected__frontier_color
                        if any(np.array_equal(self._last_goal[env], frontier) for frontier in frontiers)
                        else self._target_object_color)
            markers.append((self._last_goal[env], {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": goal_color,
            }))

        policy_info["value_map"] = cv2.cvtColor(
            self._value_map[env].visualize(markers, reduce_fn=self._vis_reduce_fn),
            cv2.COLOR_BGR2RGB,
        )

        if "DEBUG_INFO" in os.environ:
            policy_info["render_below_images"].append("debug")
            policy_info["debug"] = "debug: " + os.environ["DEBUG_INFO"]

        policy_info["start_yaw"] = self._observations_cache[env]["habitat_start_yaw"]

        DEBUG = False
        if DEBUG:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            debug_dir = "debug/20241208/obstacle_map_debug" # seg_debug_up_climb_stair
            os.makedirs(debug_dir, exist_ok=True)  # 确保调试目录存在
            if not hasattr(self, "step_count"):
                self.step_count = 0  # 添加 step_count 属性
            # 将 step 计数器用于文件名
            filename = os.path.join(debug_dir, f"Step_{self.step_count}.png")
            self.step_count += 1  # 每调用一次,计数器加一

            # 创建子图
            fig, ax = plt.subplots(1, 3, figsize=(12, 6))

            # 绘制 Depth 子图
            # draw_depth = ( #.squeeze(0).cpu().numpy() * 255.0).astype(np.uint8)
            ax[0].imshow(annotated_depth)
            ax[0].set_title("Depth Image")
            ax[0].axis("off")

            # 绘制 RGB 子图
            ax[1].imshow(annotated_rgb) # .squeeze(0).cpu().numpy().astype(np.uint8))
            ax[1].set_title("RGB Image")
            ax[1].axis("off")

            # 绘制obstacle map子图
            ax[2].imshow(policy_info["obstacle_map"])
            ax[2].set_title(f"Obstacle Map")
            ax[2].axis("off")

            # 保存子图
            plt.tight_layout()
            plt.savefig(filename, bbox_inches="tight", pad_inches=0)
            plt.close()
            print(f"Saved debug image to {filename}")
        
        return policy_info

    def is_robot_in_stair_map_fast(self, env: int, robot_px:np.ndarray, stair_map: np.ndarray):
        """
        高效判断以机器人质心为圆心、指定半径的圆是否覆盖 stair_map 中值为 1 的点。

        Args:
            env: 当前环境标识。
            stair_map (np.ndarray): 地图的 _stair_map。
            robot_xy_2d (np.ndarray): 机器人质心在相机坐标系下的 (x, y) 坐标。
            agent_radius (float): 机器人在相机坐标系中的半径。
            obstacle_map: 包含坐标转换功能和地图信息的对象。

        Returns:
            bool: 如果范围内有值为 1,则返回 True,否则返回 False。
        """
        x, y = robot_px[0, 0], robot_px[0, 1]

        # 转换半径到地图坐标系
        radius_px = self.agent_radius * self._obstacle_map[env].pixels_per_meter

        # 获取地图边界
        rows, cols = stair_map.shape
        x_min = max(0, int(x - radius_px))
        x_max = min(cols - 1, int(x + radius_px))
        y_min = max(0, int(y - radius_px))
        y_max = min(rows - 1, int(y + radius_px))

        # 提取感兴趣的子矩阵
        sub_matrix = stair_map[y_min:y_max + 1, x_min:x_max + 1]

        # 创建圆形掩码
        # y_indices, x_indices = np.ogrid[0:sub_matrix.shape[0], 0:sub_matrix.shape[1]]
        y_indices, x_indices = np.ogrid[y_min:y_max + 1, x_min:x_max + 1]
        mask = (y_indices - y) ** 2 + (x_indices - x) ** 2 <= radius_px ** 2

        # 检查掩码范围内是否有值为 1
        # 获取子矩阵中圆形区域为 True 的坐标
        # if np.any(sub_matrix[mask]):
        #     true_coords = np.column_stack(np.where(mask))  # 获取相对于子矩阵的坐标
            
        #     # 将相对坐标转换为 stair_map 中的坐标
        #     true_coords_in_stair_map = true_coords + [y_min, x_min]
            
        #     return True, true_coords_in_stair_map
        # else:
        #     return False, None

        # 获取sub_matrix中为 True 的坐标
        if np.any(sub_matrix[mask]):  # 在圆形区域内有值为True的元素
            # 找出sub_matrix中值为 True 的位置
            true_coords_in_sub_matrix = np.column_stack(np.where(sub_matrix))  # 获取相对于sub_matrix的坐标

            # 通过mask过滤,只留下圆形区域内为 True 的坐标
            true_coords_filtered = true_coords_in_sub_matrix[mask[true_coords_in_sub_matrix[:, 0], true_coords_in_sub_matrix[:, 1]]]

            # 将相对坐标转换为 stair_map 中的坐标
            true_coords_in_stair_map = true_coords_filtered + [y_min, x_min]
            
            return True, true_coords_in_stair_map
        else:
            return False, None
        
    def is_point_within_distance_of_area(
        self, 
        env: int, 
        point_px: np.ndarray, 
        area_map: np.ndarray, 
        distance_threshold: float
    ) -> bool:
        """
        判断一个点到二维区域的距离是否小于等于指定的阈值。

        Args:
            env: 当前环境标识。
            point_px (np.ndarray): 二维点坐标 (x, y)。
            area_map (np.ndarray): 二维区域地图,非零值表示有效区域。
            distance_threshold (float): 距离阈值。

        Returns:
            bool: 如果点到区域的最小距离小于等于阈值,返回 True；否则返回 False。
        """
        # 获取点的坐标
        x, y = point_px[0, 0], point_px[0, 1]

        # 转换距离阈值到像素坐标系
        radius_px = distance_threshold * self._obstacle_map[env].pixels_per_meter

        # 获取地图边界
        rows, cols = area_map.shape
        x_min = max(0, int(x - radius_px))
        x_max = min(cols - 1, int(x + radius_px))
        y_min = max(0, int(y - radius_px))
        y_max = min(rows - 1, int(y + radius_px))

        # 提取感兴趣的子矩阵
        sub_matrix = area_map[y_min:y_max + 1, x_min:x_max + 1]

        # 创建圆形掩码(局部区域的半径掩码)
        y_indices, x_indices = np.ogrid[y_min:y_max + 1, x_min:x_max + 1]
        mask = (y_indices - y) ** 2 + (x_indices - x) ** 2 <= radius_px ** 2

        # 检查掩码范围内是否有值为非零
        return np.any(sub_matrix[mask])

    def _update_obstacle_map(self, observations: "TensorDict") -> None: #  depth_for_stair_list: List[bool],
        for env in range(self._num_envs):
            if self._compute_frontiers:
                if self._climb_stair_over[env] == False:
                    if self._climb_stair_flag[env] == 1: # 0不是,1是上,2是下
                        self._temp_stair_map[env] = self._obstacle_map[env]._up_stair_map
                    elif self._climb_stair_flag[env] == 2: # 0不是,1是上,2是下
                        self._temp_stair_map[env] = self._obstacle_map[env]._down_stair_map
                    if self._stair_dilate_flag[env] == False:
                        self._temp_stair_map[env] = cv2.dilate(
                        self._temp_stair_map[env].astype(np.uint8),
                        (7,7), # (14,14), # 
                        iterations=1, # 1,
                        )
                        self._stair_dilate_flag[env] = True
                    robot_xy = self._observations_cache[env]["robot_xy"]
                    robot_xy_2d = np.atleast_2d(robot_xy) 
                    robot_px = self._obstacle_map[env]._xy_to_px(robot_xy_2d)
                    x, y = robot_px[0, 0], robot_px[0, 1]
                    if self._reach_stair[env] == False: 
                        # 边缘已经上了楼梯
                        already_reach_stair, reach_yx = self.is_robot_in_stair_map_fast(env, robot_px, self._temp_stair_map[env]) 
                        if already_reach_stair:
                            self._reach_stair[env] = True
                            if self._stair_masks[env].sum() == 0 and self._climb_stair_flag[env] == 1 and len(self._obstacle_map[env]._down_stair_frontiers) > 0: 
                                # 视野里没楼梯却要上楼,那就应该是下楼的楼梯
                                # 还需要确认,语义图是不是也没楼梯,防止vlm漏检
                                # 还需要防止楼梯间离得太近看不到
                                if np.sum(self.red_semantic_pred_list[env] == STAIR_CLASS_ID) <= 50 and  self._pitch_angle[env] >= 0: # self._obstacle_map[env]._floor_num_steps >= 20 and
                                    self._climb_stair_flag[env] = 2
                                    # 所处的位置的(原来上楼的楼梯的)连通域应该变成下楼的
                                    for y_upstair, x_upstair in reach_yx:
                                        if 0 <= x_upstair < self._obstacle_map[env]._up_stair_map.shape[1] and 0 <= y_upstair < self._obstacle_map[env]._up_stair_map.shape[0]:
                                            self._obstacle_map[env].upstair_to_downstair(y_upstair, x_upstair)
                                    self._stair_frontier[env] = self._obstacle_map[env]._down_stair_frontiers
                                # self._stair_frontier[env] = self._obstacle_map[env]._down_stair_frontiers
                            if self._climb_stair_flag[env] == 1: # 1是上,2是下
                                self._obstacle_map[env]._up_stair_start = robot_px[0].copy()
                            elif self._climb_stair_flag[env] == 2: # 1是上,2是下
                                self._obstacle_map[env]._down_stair_start = robot_px[0].copy()
                    # robot_at_stair = self.is_robot_in_stair_map_fast(env, robot_xy_2d)
                    if self._reach_stair[env] == True and self._reach_stair_centroid[env] == False:
                        # 原先是判断代理的质心是否已处于楼梯,但发现这样代理还是容易转身又下楼梯,
                        # if self._climb_stair_flag[env] == 1 and self._obstacle_map[env]._up_stair_map[y,x] == 1:
                        #     self._reach_stair_centroid[env] = True
                        # elif self._climb_stair_flag[env] == 2 and self._obstacle_map[env]._down_stair_map[y,x] == 1:
                        #     self._reach_stair_centroid[env] = True

                        # 所以改成代理的质心是否距离楼梯的质心很近
                        if self._stair_frontier[env] is not None and np.linalg.norm(self._stair_frontier[env] - robot_xy_2d) <= 0.3:
                            self._reach_stair_centroid[env] = True
                            # 记录该楼梯质心px坐标,上楼转下楼或者下楼转上楼时,保留该点所在的楼梯连通域

                    # 如果视野中没有楼梯并且质心已经走出楼梯,结束各种阶段。
                    # 对下楼梯似乎不够,得整个边缘都走出
                    if self._reach_stair_centroid[env] == True:
                        # if np.any(self._stair_masks[env]) > 0: # 看到楼梯也不一定在爬
                        #     pass
                        if self.is_robot_in_stair_map_fast(env, robot_px, self._temp_stair_map[env])[0]: # (self._climb_stair_flag[env] == 1 and  # 
                        #     self._obstacle_map[env]._up_stair_map[y,x] == 1
                        #     ) or (
                        #         self._climb_stair_flag[env] == 2 and 
                        #         self._obstacle_map[env]._down_stair_map[y,x] == 1
                        #     ): # 
                            pass
                            # # 搞不通,楼梯间直接失效
                            # 如果没有记录过终点,那么假设终点在某处,通过obstacle_map来记录
                            # 先导航到终点,接近终点的时候换用萝卜点
                            
                            # temp_stair_map_now = self._temp_stair_map[env].copy()
                            # kernel_size = 14 # 7
                            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                            # # for i in range(20):
                            # temp_stair_map_now = cv2.dilate(
                            #     temp_stair_map_now.astype(np.uint8),
                            #     kernel, # (14,14), #(7,7), #  
                            #     iterations=1, # 1,
                            #     )
                            # # 先得到当前所在的楼梯的连通域
                            # # 使用连通域分析来获得所有连通区域
                            # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(temp_stair_map_now.astype(np.uint8), connectivity=8)
                            # # 找到质心所在的连通域
                            # closest_label = -1
                            # min_distance = float('inf')
                            # for i in range(1, num_labels):  # 从1开始,0是背景
                            #     centroid_px = centroids[i]  # 获取当前连通区域的质心坐标
                            #     centroid = self._obstacle_map[env]._px_to_xy(np.atleast_2d(centroid_px))
                            #     # 计算质心与保存的爬楼梯质心的距离(欧氏距离)
                            #     distance = np.linalg.norm(self._stair_frontier[env] - centroid)
                            #     if distance < min_distance:
                            #         min_distance = distance
                            #         closest_label = i
                            #     # 如果找到了质心所在的连通域
                            # if closest_label != -1:
                            #     temp_stair_map_now[labels != closest_label] = 0 
                            # stair_boundary = temp_stair_map_now.astype(bool) &  self._obstacle_map[env]._navigable_map.astype(bool) #  (~self._obstacle_map[env]._map).astype(bool) #  

                            # DEBUG = False # True
                            # if DEBUG:
                            #     import matplotlib.pyplot as plt
                            #     import matplotlib.patches as mpatches
                            #     debug_dir = "debug/20250103/stair_test_v2" # seg_debug_up_climb_stair
                            #     os.makedirs(debug_dir, exist_ok=True)  # 确保调试目录存在
                            #     if not hasattr(self, "step_count"):
                            #         self.step_count = 0  # 添加 step_count 属性
                            #     # 将 step 计数器用于文件名
                            #     filename = os.path.join(debug_dir, f"Step_{self.step_count}.png")
                            #     self.step_count += 1  # 每调用一次,计数器加一

                            #     # 创建子图
                            #     fig, ax = plt.subplots(1, 5, figsize=(12, 6))
                            #     ax[0].imshow(self._temp_stair_map[env], cmap="gray")
                            #     ax[0].set_title("Stair_not_dilated")
                            #     ax[0].axis("off")

                            #     ax[1].imshow(temp_stair_map_now, cmap="gray")
                            #     ax[1].set_title("Stair_dilated")
                            #     ax[1].axis("off")

                            #     # 绘制 RGB 子图
                            #     ax[2].imshow(self._obstacle_map[env]._navigable_map.astype(bool), cmap="gray")
                            #     ax[2].set_title("Navigable bool")
                            #     ax[2].axis("off")

                            #     # 绘制分割图子图
                            #     ax[3].imshow(stair_boundary, cmap="gray")
                            #     ax[3].set_title(f"Stair_boundary")
                            #     ax[3].axis("off")

                            #     # 绘制 RGB 子图
                            #     ax[4].imshow((~self._obstacle_map[env]._map).astype(bool), cmap="gray")
                            #     ax[4].set_title("_map Bool")
                            #     ax[4].axis("off")

                            #     # 保存子图
                            #     plt.tight_layout()
                            #     plt.savefig(filename, bbox_inches="tight", pad_inches=0)
                            #     plt.close()
                            #     print(f"Saved debug image to {filename}")
                            # self._obstacle_map[env].stair_boundary = temp_stair_map_now.copy()
                            # self._obstacle_map[env].stair_boundary_goal = stair_boundary.copy()
                            # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(stair_boundary.astype(np.uint8), connectivity=8)
                            # # 找到质心所在的连通域
                            # farthest_label = -1
                            # max_distance = 0.0
                            # if self._climb_stair_flag[env] == 1 and len(self._obstacle_map[env]._up_stair_start) > 0 and self._obstacle_map[env]._disable_end == False: # and self._obstacle_map[env]._explored_up_stair == False
                            #     # 1是上, 2是下,  如果还没爬过这条楼梯
                            #     for i in range(1, num_labels):  # 从1开始,0是背景
                            #         centroid_px = centroids[i]  # 获取当前连通区域的质心坐标
                            #         # centroid = self._obstacle_map[env]._px_to_xy(np.atleast_2d(centroid_px))
                            #         # 计算质心与保存的起点的距离(欧氏距离)
                            #         distance = np.linalg.norm(self._obstacle_map[env]._up_stair_start - centroid_px)
                            #         if distance > max_distance:
                            #             max_distance = distance
                            #             farthest_label = i
                            #         # 如果找到了质心所在的连通域
                            #     if farthest_label != -1:
                            #         self._obstacle_map[env]._up_stair_end = centroids[farthest_label]
                            # elif self._climb_stair_flag[env] == 2 and len(self._obstacle_map[env]._down_stair_start) > 0 and self._obstacle_map[env]._disable_end == False:  #  and self._obstacle_map[env]._explored_down_stair == False
                            #     # 如果还没爬过这条楼梯
                            #     for i in range(1, num_labels):  # 从1开始,0是背景
                            #         centroid_px = centroids[i]  # 获取当前连通区域的质心坐标
                            #         # centroid = self._obstacle_map[env]._px_to_xy(np.atleast_2d(centroid_px))
                            #         # 计算质心与保存的起点的距离(欧氏距离)
                            #         distance = np.linalg.norm(self._obstacle_map[env]._down_stair_start - centroid_px)
                            #         if distance > max_distance:
                            #             max_distance = distance
                            #             farthest_label = i
                            #         # 如果找到了质心所在的连通域
                            #     if farthest_label != -1:
                            #         self._obstacle_map[env]._down_stair_end = centroids[farthest_label]
                        elif self._obstacle_map[env]._climb_stair_paused_step >= 30:
                            self._obstacle_map[env]._climb_stair_paused_step = 0
                            self._last_carrot_xy[env] = []
                            self._last_carrot_px[env] = []
                            self._reach_stair[env] = False
                            self._reach_stair_centroid[env] = False
                            self._stair_dilate_flag[env] = False
                            self._climb_stair_over[env] = True
                            self._obstacle_map[env]._disabled_frontiers.add(tuple(self._stair_frontier[env][0]))
                            print(f"Frontier {self._stair_frontier[env]} is disabled due to no movement.")
                            if  self._climb_stair_flag[env] == 1:
                                self._obstacle_map[env]._disabled_stair_map[self._obstacle_map[env]._up_stair_map == 1] = 1
                                self._obstacle_map[env]._up_stair_map.fill(0)
                                self._obstacle_map[env]._has_up_stair = False
                            elif  self._climb_stair_flag[env] == 2:
                                self._obstacle_map[env]._disabled_stair_map[self._obstacle_map[env]._down_stair_map == 1] = 1
                                self._obstacle_map[env]._down_stair_frontiers.fill(0)
                                self._obstacle_map[env]._has_down_stair = False
                            self._climb_stair_flag[env] = 0
                        else:
                            self._climb_stair_over[env] = True
                            self._reach_stair[env] = False
                            self._reach_stair_centroid[env] = False
                            self._stair_dilate_flag[env] = False
                            if self._climb_stair_flag[env] == 1: # 1是上,2是下
                                # 原来的地图记录楼梯终点
                                self._obstacle_map[env]._up_stair_end = robot_px[0].copy()
                                # 检查是否爬过这条楼梯(是否初始化过新楼层)
                                if self._obstacle_map_list[env][self._cur_floor_index[env]+1]._done_initializing == False:
                                    # 重新初始化以确定方向
                                    self._done_initializing[env] = False
                                    self._initialize_step[env] = 0
                                    self._obstacle_map[env]._explored_up_stair = True
                                    # 更新当前楼层索引
                                    self._cur_floor_index[env] += 1
                                    # 设置当前楼层的地图
                                    self._object_map[env] = self._object_map_list[env][self._cur_floor_index[env]]
                                    self._obstacle_map[env] = self._obstacle_map_list[env][self._cur_floor_index[env]]
                                    self._value_map[env] = self._value_map_list[env][self._cur_floor_index[env]]
                                    # 新楼层的步数
                                    # self._obstacle_map[env]._floor_num_steps = 0
                                    # 新楼层(向上一层)的下楼的楼梯是刚才上楼的楼梯,起止点互换一下
                                    # 可能中间有平坦楼梯间,而提前看到了再上去的楼梯,这时候应该只保留爬过的楼梯
                                    # 获取当前楼层的 _up_stair_map
                                    ori_up_stair_map = self._obstacle_map_list[env][self._cur_floor_index[env]-1]._up_stair_map.copy()
                                    
                                    # 使用连通域分析来获得所有连通区域
                                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ori_up_stair_map.astype(np.uint8), connectivity=8)
                                    # 找到质心所在的连通域
                                    closest_label = -1
                                    min_distance = float('inf')
                                    for i in range(1, num_labels):  # 从1开始,0是背景
                                        centroid_px = centroids[i]  # 获取当前连通区域的质心坐标
                                        centroid = self._obstacle_map[env]._px_to_xy(np.atleast_2d(centroid_px))
                                        # 计算质心与保存的爬楼梯质心的距离(欧氏距离)
                                        # distance = np.linalg.norm(self._stair_frontier[env] - centroid)
                                        distance = np.abs(self._obstacle_map_list[env][self._cur_floor_index[env]-1]._up_stair_frontiers[0][0] - centroid[0][0]) + np.abs(self._obstacle_map_list[env][self._cur_floor_index[env]-1]._up_stair_frontiers[0][1] - centroid[0][1])
                                        if distance < min_distance:
                                            min_distance = distance
                                            closest_label = i
                                        # 如果找到了质心所在的连通域
                                    if closest_label != -1:
                                        ori_up_stair_map[labels != closest_label] = 0 
                                    
                                    # 将更新后的 _up_stair_map 赋值回去
                                    self._obstacle_map_list[env][self._cur_floor_index[env]]._down_stair_map = ori_up_stair_map
                                    # self._obstacle_map_list[env][self._cur_floor_index[env]]._down_stair_map = self._obstacle_map_list[env][self._cur_floor_index[env] - 1]._up_stair_map.copy()
                                    self._obstacle_map_list[env][self._cur_floor_index[env]]._down_stair_start = self._obstacle_map_list[env][self._cur_floor_index[env] - 1]._up_stair_end.copy()
                                    self._obstacle_map_list[env][self._cur_floor_index[env]]._down_stair_end = self._obstacle_map_list[env][self._cur_floor_index[env] - 1]._up_stair_start.copy()
                                    self._obstacle_map_list[env][self._cur_floor_index[env]]._down_stair_frontiers = self._obstacle_map_list[env][self._cur_floor_index[env] - 1]._up_stair_frontiers.copy()
                                else:
                                    # 只更新当前楼层索引
                                    self._cur_floor_index[env] += 1
                                    # 设置当前楼层的地图
                                    self._object_map[env] = self._object_map_list[env][self._cur_floor_index[env]]
                                    self._obstacle_map[env] = self._obstacle_map_list[env][self._cur_floor_index[env]]
                                    self._value_map[env] = self._value_map_list[env][self._cur_floor_index[env]]
                            elif self._climb_stair_flag[env] == 2:
                                # 原来的地图记录楼梯终点
                                self._obstacle_map[env]._down_stair_end = robot_px[0].copy()
                                # 检查是否爬过这条楼梯(是否初始化过新楼层)
                                if self._obstacle_map_list[env][self._cur_floor_index[env]-1]._done_initializing == False:

                                    # 重新初始化以确定方向
                                    self._done_initializing[env] = False
                                    self._initialize_step[env] = 0
                                    self._obstacle_map[env]._explored_down_stair = True
                                    # 更新当前楼层索引
                                    self._cur_floor_index[env] -= 1 # 当前是0,不需要更新
                                    # 设置当前楼层的地图
                                    self._object_map[env] = self._object_map_list[env][self._cur_floor_index[env]]
                                    self._obstacle_map[env] = self._obstacle_map_list[env][self._cur_floor_index[env]]
                                    self._value_map[env] = self._value_map_list[env][self._cur_floor_index[env]]
                                    # 新楼层的步数
                                    # self._obstacle_map[env]._floor_num_steps = 0
                                    # 新楼层(向下一层)的上楼的楼梯是刚才下楼的楼梯,起止点互换一下
                                    # 可能中间有平坦楼梯间,而提前看到了再下去的楼梯,这时候应该只保留爬过的楼梯
                                    # 获取当前楼层的 _down_stair_map
                                    ori_down_stair_map = self._obstacle_map_list[env][self._cur_floor_index[env]+1]._down_stair_map.copy()
                                    
                                    # 使用连通域分析来获得所有连通区域
                                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ori_down_stair_map.astype(np.uint8), connectivity=8)
                                    # 找到质心所在的连通域 
                                    closest_label = -1
                                    min_distance = float('inf')
                                    for i in range(1, num_labels):  # 从1开始,0是背景
                                        centroid_px = centroids[i]  # 获取当前连通区域的质心坐标
                                        centroid = self._obstacle_map[env]._px_to_xy(np.atleast_2d(centroid_px))
                                        # 计算质心与保存的爬楼梯质心的距离(欧氏距离) # 改为起点周围的
                                        # distance = np.linalg.norm(self._stair_frontier[env] - centroid)
                                        distance = np.abs(self._obstacle_map_list[env][self._cur_floor_index[env]+1]._down_stair_frontiers[0][0] - centroid[0][0]) + np.abs(self._obstacle_map_list[env][self._cur_floor_index[env]+1]._down_stair_frontiers[0][1] - centroid[0][1])
                                        if distance < min_distance:
                                            min_distance = distance
                                            closest_label = i
                                        # 如果找到了质心所在的连通域
                                    if closest_label != -1:
                                        ori_down_stair_map[labels != closest_label] = 0 
                                    
                                    # 将更新后的 _up_stair_map 赋值回去
                                    self._obstacle_map_list[env][self._cur_floor_index[env]]._up_stair_map = ori_down_stair_map
                                    # 新楼层(向下一层)的上楼的楼梯是刚才下楼的楼梯,起止点互换一下
                                    # self._obstacle_map_list[env][self._cur_floor_index[env]]._up_stair_map = self._obstacle_map_list[env][self._cur_floor_index[env] + 1]._down_stair_map.copy()
                                    self._obstacle_map_list[env][self._cur_floor_index[env]]._up_stair_start = self._obstacle_map_list[env][self._cur_floor_index[env] + 1]._down_stair_end.copy()
                                    self._obstacle_map_list[env][self._cur_floor_index[env]]._up_stair_end = self._obstacle_map_list[env][self._cur_floor_index[env] + 1]._down_stair_start.copy()
                                    self._obstacle_map_list[env][self._cur_floor_index[env]]._up_stair_frontiers = self._obstacle_map_list[env][self._cur_floor_index[env] + 1]._down_stair_frontiers.copy()
                                else:
                                    # 只更新当前楼层索引
                                    self._cur_floor_index[env] -= 1
                                    self._object_map[env] = self._object_map_list[env][self._cur_floor_index[env]]
                                    self._obstacle_map[env] = self._obstacle_map_list[env][self._cur_floor_index[env]]
                                    self._value_map[env] = self._value_map_list[env][self._cur_floor_index[env]]
                            self._climb_stair_flag[env] = 0
                            self._obstacle_map[env]._climb_stair_paused_step = 0
                            self._last_carrot_xy[env] = []
                            self._last_carrot_px[env] = [] 
                            
                            print("climb stair success!!!!")
                # update obstacle map with stairs and persons # update_map_with_stair_and_person_wo_rednet
                self._obstacle_map[env].update_map_with_stair_and_person(
                    self._observations_cache[env]["object_map_rgbd"][0][1],
                    self._observations_cache[env]["object_map_rgbd"][0][2],
                    self._min_depth,
                    self._max_depth,
                    self._fx,
                    self._fy,
                    self._camera_fov,
                    self._object_map[env].movable_clouds,
                    # self._object_map[env].stair_clouds,
                    self._person_masks[env],
                    self._stair_masks[env],
                    self.red_semantic_pred_list[env],
                    self._pitch_angle[env],
                    self._climb_stair_over[env],
                    self._reach_stair[env],
                    self._climb_stair_flag[env],
                    # self._reach_stair_centroid[env],
                )
                frontiers = self._obstacle_map[env].frontiers
                self._obstacle_map[env].update_agent_traj(self._observations_cache[env]["robot_xy"], self._observations_cache[env]["robot_heading"])

            else:
                if "frontier_sensor" in observations:
                    frontiers = observations["frontier_sensor"][env].cpu().numpy()
                else:
                    frontiers = np.array([])
            self._observations_cache[env]["frontier_sensor"] = frontiers

            # 附加
            # 如果发现了楼梯,那就先把楼梯对应的楼层搞定
            if self._obstacle_map[env]._has_up_stair and self._cur_floor_index[env] + 1 >= len(self._object_map_list[env]):
                # 添加新的地图
                self._object_map_list[env].append(ObjectPointCloudMap(erosion_size=self._object_map_erosion_size)) 
                self._obstacle_map_list[env].append(ObstacleMap(
                    min_height=self.min_obstacle_height,
                    max_height=self.max_obstacle_height,
                    area_thresh=self.obstacle_map_area_threshold,
                    agent_radius=self.agent_radius,
                    hole_area_thresh=self.hole_area_thresh,
                ))
                self._value_map_list[env].append(ValueMap(
                    value_channels=len(self._text_prompt.split(PROMPT_SEPARATOR)),
                    use_max_confidence=self.use_max_confidence,
                    obstacle_map=None,
                ))
            if self._obstacle_map[env]._has_down_stair and self._cur_floor_index[env] == 0:
                # 如果当前楼层索引为0,说明需要向前插入新的地图
                self._object_map_list[env].insert(0, ObjectPointCloudMap(erosion_size=self._object_map_erosion_size))
                self._obstacle_map_list[env].insert(0, ObstacleMap(
                    min_height=self.min_obstacle_height,
                    max_height=self.max_obstacle_height,
                    area_thresh=self.obstacle_map_area_threshold,
                    agent_radius=self.agent_radius,
                    hole_area_thresh=self.hole_area_thresh,
                ))
                self._value_map_list[env].insert(0, ValueMap(
                    value_channels=len(self._text_prompt.split(PROMPT_SEPARATOR)),
                    use_max_confidence=self.use_max_confidence,
                    obstacle_map=None,
                ))
                self._cur_floor_index[env] += 1 # 当前不是最底层了

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

    def _update_object_map(self) -> None:
        for env in range(self._num_envs):
            self._object_map[env].update_agent_traj(
                self._observations_cache[env]["robot_xy"],
                self._observations_cache[env]["robot_heading"],
            )
        if np.argwhere(self._object_map[env]._map).size > 0:
            # camera_position_2d = np.atleast_2d(self._object_map[env]._camera_positions[-1])
            # camera_position_px = self._object_map[env]._xy_to_px(camera_position_2d)

            # up_stair_points = np.argwhere(self._obstacle_map[env]._up_stair_map)
            robot_xy = self._observations_cache[env]["robot_xy"]
            robot_xy_2d = np.atleast_2d(robot_xy) 
            robot_px = self._obstacle_map[env]._xy_to_px(robot_xy_2d)
            # distances = np.abs(up_stair_points[:, 0] - robot_px[0][0]) + np.abs(up_stair_points[:, 1] - robot_px[0][1])
            # min_dis_to_upstair = np.min(distances)
            
            distances_px = np.linalg.norm(np.argwhere(self._object_map[env]._map) - robot_px[0], axis=1)
            # distances_in_meters = distances * self._object_map[env].pixels_per_meter
            self.min_distance_px[env] = np.min(distances_px) # px
            if self.min_distance_px[env] < 2 * self._object_map[env].pixels_per_meter:
                self._might_close_to_goal[env] = True
            else:
                self._might_close_to_goal[env] = False
                
    # # Useless
    def _get_target_object_location_with_seg(self, position: np.ndarray, env: int = 0,) -> Union[None, np.ndarray]:
        if self._object_map[env].has_object(self._target_object[env]): 
            if self._target_object[env] != "bed" and self._target_object[env] != "tv":
                return self._object_map[env].get_best_object(self._target_object[env], position)
            # add RedNet to help localization
            # "chair",
            # "sofa",
            # "plant",
            # "bed",
            # "toilet",
            # "tv_monitor",
            if  self._object_masks[env].sum() > 0 and self.target_might_detected[env] == False: # 刚发现目标,需要判定
                # if self._target_object[env] == "chair":
                #     target_class_id = CHAIR_CLASS_ID
                # elif self._target_object[env] == "couch":
                #     target_class_id = SOFA_CLASS_ID
                # if self._target_object[env] == "plant":
                #     target_class_id = PLANT_CLASS_ID
                if self._target_object[env] == "bed":
                    target_class_id = BED_CLASS_ID
                # elif self._target_object[env] == "toilet":
                #     target_class_id = TOILET_CLASS_ID
                # 其他不需要,电视/床需要(vlm会误识别)
                elif self._target_object[env] == "tv":
                    target_class_id = TV_CLASS_ID
        
                # 获取物体区域
                object_region = self._object_masks[env].astype(bool)

                # 获取seg_mask_region对应的部分
                seg_mask_region = self.red_semantic_pred_list[env][object_region]

                # 计算 seg_mask_region 中为 target_class_id 的部分
                target_region = (seg_mask_region == target_class_id)

                # 计算 object_region 中为 True 的像素数量
                object_region_count = np.sum(object_region)

                # 如果 object_region 的像素数量为零,直接设置 target_exists 为 False
                if object_region_count == 0:
                    self.target_might_detected[env] = False
                else:
                    # 计算在 object_region 为 True 的区域内,target_region 为 True 的比例
                    target_ratio = np.sum(target_region) / object_region_count

                    # 如果 target_region 中至少有 50% 为 True,则 target_exists 为 True
                    self.target_might_detected[env] = target_ratio >= 0.5

                if self.target_might_detected[env]:
                    return self._object_map[env].get_best_object(self._target_object[env], position)
                else:
                     # 有问题,如果没有的话,点云部分得标识为错
                    if 'false' in self._object_map[env].clouds:
                        self._object_map[env].clouds['false'] = np.concatenate((self._object_map[env].clouds['false'], self._object_map[env].clouds[self._target_object[env]].copy()), axis=0)
                    else:
                        self._object_map[env].clouds['false'] = self._object_map[env].clouds[self._target_object[env]].copy()
                    del self._object_map[env].clouds[self._target_object[env]]
                    return None # self._object_map[env].get_best_object(self._target_object[env], position)
            elif self.target_might_detected[env] == True: # 判定过应该是,保持原判
                return self._object_map[env].get_best_object(self._target_object[env], position)
            else:
                return None
        else:
            return None
    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Any:

        # Extract the object_ids, assuming observations[ObjectGoalSensor.cls_uuid] contains multiple values
        object_ids = observations[ObjectGoalSensor.cls_uuid] # .cpu().numpy().flatten()

        # Convert observations to dictionary format
        obs_dict = observations.to_tree()

        # Loop through each object_id and replace the goal IDs with corresponding names
        if self._dataset_type == "hm3d":
            obs_dict[ObjectGoalSensor.cls_uuid] = [HM3D_ID_TO_NAME[oid.item()] for oid in object_ids]
        elif self._dataset_type == "mp3d":
            obs_dict[ObjectGoalSensor.cls_uuid] = [MP3D_ID_TO_NAME[oid.item()] for oid in object_ids]
            # self._non_coco_caption = " . ".join(MP3D_ID_TO_NAME).replace("|", " . ") + " ."
        else:
            raise ValueError(f"Dataset type {self._dataset_type} not recognized")
        
        self._pre_step(obs_dict, masks)
        img_height, img_width = observations["rgb"].shape[1:3]
        self._update_object_map_with_stair_and_person(img_height, img_width)

        self.red_semantic_pred_list = [] # 每个元素对应当前环境的图片
        self.seg_map_color_list = []
        # move it forward to detect the stairs
        ### For RedNet
        for env in range(self._num_envs):
            rgb = torch.unsqueeze(observations["rgb"][env], dim=0).float()
            depth = torch.unsqueeze(observations["depth"][env], dim=0).float()

            # seg_map 是类别索引的二维数组,color_palette 是固定颜色表
            with torch.no_grad():
                red_semantic_pred = self.red_sem_pred(rgb, depth)
                red_semantic_pred = red_semantic_pred.squeeze().cpu().detach().numpy().astype(np.uint8)
            self.red_semantic_pred_list.append(red_semantic_pred)
            # 创建颜色查找表
            color_map = np.array(MPCAT40_RGB_COLORS, dtype=np.uint8)
            seg_map_color = color_map[red_semantic_pred]
            self.seg_map_color_list.append(seg_map_color)

            DEBUG = False # True
            if DEBUG:
                import matplotlib.pyplot as plt
                import matplotlib.patches as mpatches
                debug_dir = "debug/20241231/seg_obj_mask" # seg_debug_up_climb_stair
                os.makedirs(debug_dir, exist_ok=True)  # 确保调试目录存在
                if not hasattr(self, "step_count"):
                    self.step_count = 0  # 添加 step_count 属性
                # 将 step 计数器用于文件名
                filename = os.path.join(debug_dir, f"Step_{self.step_count}.png")
                self.step_count += 1  # 每调用一次,计数器加一

                # 确定包含的类别
                unique_classes = np.unique(red_semantic_pred)
                detected_categories = [MPCAT40_NAMES[c] for c in unique_classes if c < len(MPCAT40_NAMES)]
                categories_title = ", ".join(detected_categories)

                # 创建子图
                fig, ax = plt.subplots(1, 3, figsize=(12, 6))

                # 绘制 Depth 子图
                draw_depth = (depth.squeeze(0).cpu().numpy() * 255.0).astype(np.uint8)
                draw_depth = cv2.cvtColor(draw_depth, cv2.COLOR_GRAY2RGB)
                ax[0].imshow(draw_depth)
                ax[0].set_title("Depth Image")
                ax[0].axis("off")

                # 绘制 RGB 子图
                ax[1].imshow(rgb.squeeze(0).cpu().numpy().astype(np.uint8))
                ax[1].set_title("RGB Image")
                ax[1].axis("off")

                # 绘制分割图子图
                ax[2].imshow(seg_map_color)
                ax[2].set_title(f"Segmentation Map\nDetected: {categories_title}")
                ax[2].axis("off")

                # 创建图例
                legend_handles = []
                for i, color in enumerate(MPCAT40_RGB_COLORS):
                    # 如果该类别存在于当前分割图中
                    if i in unique_classes:
                        color_patch = mpatches.Patch(color=color/255.0, label=MPCAT40_NAMES[i])
                        legend_handles.append(color_patch)

                # 添加图例到右侧
                ax[2].legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.2, 1))

                # 保存子图
                plt.tight_layout()
                plt.savefig(filename, bbox_inches="tight", pad_inches=0)
                plt.close()
                print(f"Saved debug image to {filename}")

        self._update_obstacle_map(observations)
        self._update_value_map()
        self._update_object_map()
        # self._pre_step(observations, masks) # duplicated one, consider to get rid of this
        
        pointnav_action_env_list = []

        for env in range(self._num_envs):

            robot_xy = self._observations_cache[env]["robot_xy"]
            goal = self._get_target_object_location(robot_xy, env) #  self._get_target_object_location_with_seg(robot_xy, red_semantic_pred_list, env, ) #
            robot_xy_2d = np.atleast_2d(robot_xy) 
            robot_px = self._obstacle_map[env]._xy_to_px(robot_xy_2d)
            x, y = robot_px[0, 0], robot_px[0, 1]
            # 不知不觉到了下楼的楼梯,且不是刚刚上楼的
            if self._climb_stair_over[env] == True and self._obstacle_map[env]._down_stair_map[y,x] == 1 and len(self._obstacle_map[env]._down_stair_frontiers) > 0 and self._obstacle_map_list[env][self._cur_floor_index[env] - 1]._explored_up_stair == False:  # and self._obstacle_map_list[env][self._cur_floor_index[env] - 1]._explored_up_stair == False 
                self._reach_stair[env] = True
                self._climb_stair_over[env] = False
                self._climb_stair_flag[env] = 2
                self._obstacle_map[env]._down_stair_start = robot_px[0].copy()
                # self._reach_stair_centroid[env] = True
            # 不知不觉到了上楼的楼梯,且不是刚刚下楼的
            elif self._climb_stair_over[env] == True and self._obstacle_map[env]._up_stair_map[y,x] == 1 and len(self._obstacle_map[env]._up_stair_frontiers) > 0 and self._obstacle_map_list[env][self._cur_floor_index[env] + 1]._explored_down_stair == False: # and self._obstacle_map_list[env][self._cur_floor_index[env] + 1]._explored_down_stair == False 
                self._reach_stair[env] = True
                self._climb_stair_over[env] = False
                self._climb_stair_flag[env] = 1
                self._obstacle_map[env]._up_stair_start = robot_px[0].copy()
            if self._climb_stair_over[env] == False:
                if self._reach_stair[env] == True:
                    # if self._pitch_angle[env] == 0:
                             
                        # if self._climb_stair_flag[env] == 1: # up
                        #     self._pitch_angle[env] += self._pitch_angle_offset
                        #     mode = "look_up"
                        #     pointnav_action = TorchActionIDs_plook.LOOK_UP.to(masks.device)
                            
                        # elif self._climb_stair_flag[env] == 2: # down
                        #     self._pitch_angle[env] -= self._pitch_angle_offset
                        #     mode = "look_down"
                        #     pointnav_action = TorchActionIDs_plook.LOOK_DOWN.to(masks.device)
                    if self._pitch_angle[env] == 0 and self._climb_stair_flag[env] == 2: 
                            self._pitch_angle[env] -= self._pitch_angle_offset
                            mode = "look_down"
                            pointnav_action = TorchActionIDs_plook.LOOK_DOWN.to(masks.device)
                    elif self._climb_stair_flag[env] == 2 and self._pitch_angle[env] >= -30 and self._reach_stair_centroid[env] == False: 
                            # 更好地下楼梯 
                            self._pitch_angle[env] -= self._pitch_angle_offset
                            mode = "look_down_twice"
                            pointnav_action = TorchActionIDs_plook.LOOK_DOWN.to(masks.device)
                    else:
                        if self._obstacle_map[env]._climb_stair_paused_step < 30:
                            # if self._climb_stair_flag[env] == 1 and self._pitch_angle[env] < 30 and check_stairs_in_upper_50_percent(self.red_semantic_pred_list[env] == STAIR_CLASS_ID):
                            #     self._pitch_angle[env] += self._pitch_angle_offset
                            #     mode = "look_up"
                            #     pointnav_action = TorchActionIDs_plook.LOOK_UP.to(masks.device)
                            # else:
                            mode = "climb_stair"
                            pointnav_action = self._climb_stair(observations, env, masks)
                        else:
                            if self._climb_stair_flag[env] == 1 and self._obstacle_map_list[env][self._cur_floor_index[env]+1]._done_initializing == False:
                                # 重新初始化以确定方向
                                self._done_initializing[env] = False
                                self._initialize_step[env] = 0
                                self._obstacle_map[env]._explored_up_stair = True
                                # 更新当前楼层索引
                                self._cur_floor_index[env] += 1
                                # 设置当前楼层的地图
                                self._object_map[env] = self._object_map_list[env][self._cur_floor_index[env]]
                                self._obstacle_map[env] = self._obstacle_map_list[env][self._cur_floor_index[env]]
                                self._value_map[env] = self._value_map_list[env][self._cur_floor_index[env]]
                                # 新楼层的步数
                                # self._obstacle_map[env]._floor_num_steps = 0
                                # 新楼层(向上一层)的下楼的楼梯是刚才上楼的楼梯,起止点互换一下
                                # 可能中间有平坦楼梯间,而提前看到了再上去的楼梯,这时候应该只保留爬过的楼梯
                                # 获取当前楼层的 _up_stair_map
                                ori_up_stair_map = self._obstacle_map_list[env][self._cur_floor_index[env]-1]._up_stair_map.copy()
                                
                                # 使用连通域分析来获得所有连通区域
                                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ori_up_stair_map.astype(np.uint8), connectivity=8)
                                # 找到质心所在的连通域
                                closest_label = -1
                                min_distance = float('inf')
                                for i in range(1, num_labels):  # 从1开始,0是背景
                                    centroid_px = centroids[i]  # 获取当前连通区域的质心坐标
                                    centroid = self._obstacle_map[env]._px_to_xy(np.atleast_2d(centroid_px))
                                    # 计算质心与保存的爬楼梯质心的距离(欧氏距离)
                                    distance = np.linalg.norm(self._stair_frontier[env] - centroid)
                                    if distance < min_distance:
                                        min_distance = distance
                                        closest_label = i
                                    # 如果找到了质心所在的连通域
                                if closest_label != -1:
                                    ori_up_stair_map[labels != closest_label] = 0 
                                
                                # 将更新后的 _up_stair_map 赋值回去
                                self._obstacle_map_list[env][self._cur_floor_index[env]]._down_stair_map = ori_up_stair_map
                                # self._obstacle_map_list[env][self._cur_floor_index[env]]._down_stair_map = self._obstacle_map_list[env][self._cur_floor_index[env] - 1]._up_stair_map.copy()
                                self._obstacle_map_list[env][self._cur_floor_index[env]]._down_stair_start = self._obstacle_map_list[env][self._cur_floor_index[env] - 1]._up_stair_end.copy()
                                self._obstacle_map_list[env][self._cur_floor_index[env]]._down_stair_end = self._obstacle_map_list[env][self._cur_floor_index[env] - 1]._up_stair_start.copy()
                                self._obstacle_map_list[env][self._cur_floor_index[env]]._down_stair_frontiers = self._obstacle_map_list[env][self._cur_floor_index[env] - 1]._up_stair_frontiers.copy()

                            # 可能有坑，存疑
                            elif self._climb_stair_flag[env] == 2 and self._obstacle_map_list[env][self._cur_floor_index[env]-1]._done_initializing == False:

                                # 重新初始化以确定方向
                                self._done_initializing[env] = False
                                self._initialize_step[env] = 0 
                                self._obstacle_map[env]._explored_down_stair = True
                                # 更新当前楼层索引
                                self._cur_floor_index[env] -= 1 # 当前是0,不需要更新
                                # 设置当前楼层的地图
                                self._object_map[env] = self._object_map_list[env][self._cur_floor_index[env]]
                                self._obstacle_map[env] = self._obstacle_map_list[env][self._cur_floor_index[env]]
                                self._value_map[env] = self._value_map_list[env][self._cur_floor_index[env]]
                                # 新楼层的步数
                                # self._obstacle_map[env]._floor_num_steps = 0
                                # 新楼层(向下一层)的上楼的楼梯是刚才下楼的楼梯,起止点互换一下
                                # 可能中间有平坦楼梯间,而提前看到了再下去的楼梯,这时候应该只保留爬过的楼梯
                                # 获取当前楼层的 _down_stair_map
                                ori_down_stair_map = self._obstacle_map_list[env][self._cur_floor_index[env]+1]._down_stair_map.copy()
                                
                                # 使用连通域分析来获得所有连通区域
                                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ori_down_stair_map.astype(np.uint8), connectivity=8)
                                # 找到质心所在的连通域
                                closest_label = -1
                                min_distance = float('inf')
                                for i in range(1, num_labels):  # 从1开始,0是背景
                                    centroid_px = centroids[i]  # 获取当前连通区域的质心坐标
                                    centroid = self._obstacle_map[env]._px_to_xy(np.atleast_2d(centroid_px))
                                    # 计算质心与保存的爬楼梯质心的距离(欧氏距离)
                                    distance = np.linalg.norm(self._stair_frontier[env] - centroid)
                                    if distance < min_distance:
                                        min_distance = distance
                                        closest_label = i
                                    # 如果找到了质心所在的连通域
                                if closest_label != -1:
                                    ori_down_stair_map[labels != closest_label] = 0 
                                
                                # 将更新后的 _up_stair_map 赋值回去
                                self._obstacle_map_list[env][self._cur_floor_index[env]]._up_stair_map = ori_down_stair_map
                                # 新楼层(向下一层)的上楼的楼梯是刚才下楼的楼梯,起止点互换一下
                                # self._obstacle_map_list[env][self._cur_floor_index[env]]._up_stair_map = self._obstacle_map_list[env][self._cur_floor_index[env] + 1]._down_stair_map.copy()
                                self._obstacle_map_list[env][self._cur_floor_index[env]]._up_stair_start = self._obstacle_map_list[env][self._cur_floor_index[env] + 1]._down_stair_end.copy()
                                self._obstacle_map_list[env][self._cur_floor_index[env]]._up_stair_end = self._obstacle_map_list[env][self._cur_floor_index[env] + 1]._down_stair_start.copy()
                                self._obstacle_map_list[env][self._cur_floor_index[env]]._up_stair_frontiers = self._obstacle_map_list[env][self._cur_floor_index[env] + 1]._down_stair_frontiers.copy()
                                
                            mode = "climb_stair_initialize"
                        
                            # pointnav_action = self._climb_stair(observations, env, masks)
                            if self._pitch_angle[env] > 0: 
                                # mode = "look_down_back"
                                self._pitch_angle[env] -= self._pitch_angle_offset
                                pointnav_action = TorchActionIDs_plook.LOOK_DOWN.to(masks.device)
                            elif self._pitch_angle[env] < 0:
                                # mode = "look_up_back"
                                self._pitch_angle[env] += self._pitch_angle_offset
                                pointnav_action = TorchActionIDs_plook.LOOK_UP.to(masks.device)
                            else:  # Initialize for 12 steps
                                self._obstacle_map[env]._done_initializing = False # add, for the initial floor and new floor
                                # mode = "initialize"
                                self._initialize_step[env] = 0
                                pointnav_action = self._initialize(env,masks)
                            self._obstacle_map[env]._climb_stair_paused_step = 0
                            self._climb_stair_over[env] = True
                            self._reach_stair[env] = False
                            self._reach_stair_centroid[env] = False
                            self._stair_dilate_flag[env] = False
                            # mode = "reverse_climb_stair"
                            # pointnav_action = self._reverse_climb_stair(observations, env, masks)
                else:
                    # 打印离楼梯最近点的距离
                    # 如果很近，且镜头上半部分有楼梯语义，那么就抬头。
                    # 主要是有些楼梯点导航预训练模型太笨了不往前走。
                    # 为了防止楼梯间，还要找一下楼梯防止上楼变下楼
                    if self._obstacle_map[env]._look_for_downstair_flag == True:
                        mode = "look_for_downstair"
                        pointnav_action = self._look_for_downstair(observations, env, masks)
                    elif self._climb_stair_flag[env] == 1 and self._pitch_angle[env] == 0 and np.sum(self._obstacle_map[env]._up_stair_map)>0: # up
                        up_stair_points = np.argwhere(self._obstacle_map[env]._up_stair_map)
                        robot_xy = self._observations_cache[env]["robot_xy"]
                        robot_xy_2d = np.atleast_2d(robot_xy) 
                        robot_px = self._obstacle_map[env]._xy_to_px(robot_xy_2d)
                        distances = np.abs(up_stair_points[:, 0] - robot_px[0][0]) + np.abs(up_stair_points[:, 1] - robot_px[0][1])
                        min_dis_to_upstair = np.min(distances)
                        print(f"min_dis_to_upstair: {min_dis_to_upstair}")
                        if min_dis_to_upstair <= 2.0 * self._obstacle_map[env].pixels_per_meter and check_stairs_in_upper_50_percent(self.red_semantic_pred_list[env] == STAIR_CLASS_ID):
                            self._pitch_angle[env] += self._pitch_angle_offset
                            mode = "look_up"
                            pointnav_action = TorchActionIDs_plook.LOOK_UP.to(masks.device)
                        else:
                            mode = "get_close_to_stair"
                            pointnav_action = self._get_close_to_stair(observations, env, masks)
                    elif self._climb_stair_flag[env] == 2 and self._pitch_angle[env] == 0 and np.sum(self._obstacle_map[env]._down_stair_map)>0 :
                        down_stair_points = np.argwhere(self._obstacle_map[env]._down_stair_map)
                        robot_xy = self._observations_cache[env]["robot_xy"]
                        robot_xy_2d = np.atleast_2d(robot_xy) 
                        robot_px = self._obstacle_map[env]._xy_to_px(robot_xy_2d)
                        distances = np.abs(down_stair_points[:, 0] - robot_px[0][0]) + np.abs(down_stair_points[:, 1] - robot_px[0][1])
                        min_dis_to_downstair = np.min(distances)
                        print(f"min_dis_to_downstair: {min_dis_to_downstair}")
                        if min_dis_to_downstair <= 2.0 * self._obstacle_map[env].pixels_per_meter:
                            self._pitch_angle[env] -= self._pitch_angle_offset
                            mode = "look_down"
                            pointnav_action = TorchActionIDs_plook.LOOK_DOWN.to(masks.device)
                        else:
                            mode = "get_close_to_stair"
                            pointnav_action = self._get_close_to_stair(observations, env, masks)
                    else:
                        mode = "get_close_to_stair"
                        pointnav_action = self._get_close_to_stair(observations, env, masks)
            # if self._climb_stair_over[env] == False:
            else:
                # elif self._obstacle_map[env]._search_down_stair == True:
                #     mode = "search_down_stair"
                #     pointnav_action = self._search_down_stair(observations, env, masks)
                if self._pitch_angle[env] > 0: 
                    mode = "look_down_back"
                    self._pitch_angle[env] -= self._pitch_angle_offset
                    pointnav_action = TorchActionIDs_plook.LOOK_DOWN.to(masks.device)
                elif self._pitch_angle[env] < 0 and self._obstacle_map[env]._look_for_downstair_flag == False:
                    mode = "look_up_back"
                    self._pitch_angle[env] += self._pitch_angle_offset
                    pointnav_action = TorchActionIDs_plook.LOOK_UP.to(masks.device)
                elif not self._done_initializing[env]:  # Initialize for 12 steps
                    self._obstacle_map[env]._done_initializing = True # add, for the initial floor and new floor
                    mode = "initialize"
                    pointnav_action = self._initialize(env,masks)
                elif goal is None:  # Haven't found target object yet
                    if self._obstacle_map[env]._look_for_downstair_flag == True:
                        mode = "look_for_downstair"
                        pointnav_action = self._look_for_downstair(observations, env, masks)
                    else:
                        mode = "explore"
                        pointnav_action = self._explore(observations, env, masks)
                else:
                    mode = "navigate"
                    self._try_to_navigate[env] = True # 显示处于导航状态
                    # pointnav_action = self._navigate(observations, goal[:2], stop=True, env=env, ori_masks=masks)
                    pointnav_action = self._pointnav(observations, goal[:2], stop=True, env=env, ori_masks=masks)

            action_numpy = pointnav_action.detach().cpu().numpy()[0]
            if len(action_numpy) == 1:
                action_numpy = action_numpy[0]
            # 更新动作历史
            if len(self.history_action[env]) > 20:
                self.history_action[env].pop(0)  # 保持历史长度为20
                # 检查最近20步是否都是2或3
                if all(action in [2, 3] for action in self.history_action[env]):
                    # 强制执行动作1
                    action_numpy = 1
                    pointnav_action = torch.tensor([action_numpy], dtype=torch.int64, device=masks.device)
                    print("Continuous turns to force forward.")
                if all(action in [1] for action in self.history_action[env]):
                    # 强制执行动作3
                    action_numpy = 3
                    pointnav_action = torch.tensor([action_numpy], dtype=torch.int64, device=masks.device)
                    print("Continuous turns to force turn right.")
            self.history_action[env].append(action_numpy)
            pointnav_action_env_list.append(pointnav_action)
            
            print(f"Env: {env} | Step: {self._num_steps[env]} | Floor_step: {self._obstacle_map[env]._floor_num_steps} | Mode: {mode} | Stair_flag: {self._climb_stair_flag[env]} | Action: {action_numpy}")
            if self._climb_stair_over[env] == False:
                print(f"Reach_stair_centroid: {self._reach_stair_centroid[env]}")
                # print(f"Stair_pixedl: {np.sum(self.red_semantic_pred_list[env] == STAIR_CLASS_ID)}")
            self._num_steps[env] += 1
            self._obstacle_map[env]._floor_num_steps += 1
            self._policy_info[env].update(self._get_policy_info(self.target_detection_list[env],env)) # self.seg_map_color_list[env],

            self._observations_cache[env] = {}
            self._did_reset[env] = False

        pointnav_action_tensor = torch.cat(pointnav_action_env_list, dim=0)

        return PolicyActionData(
            actions=pointnav_action_tensor,
            rnn_hidden_states=rnn_hidden_states,
            policy_info=self._policy_info, # [self._policy_info],
        )

    def _initialize(self, env: int, masks: Tensor) -> Tensor:
        """Turn left 30 degrees 12 times to get a 360 view at the beginning"""
        # self._done_initializing[env] = not self._num_steps[env] < 11  # type: ignore
        if self._initialize_step[env] > 11: # self._obstacle_map[env]._floor_num_steps > 11:
            self._done_initializing[env] = True
        else:
            self._initialize_step[env] += 1 
        return TorchActionIDs_plook.TURN_LEFT.to(masks.device)

    def _explore(self, observations: Union[Dict[str, Tensor], "TensorDict"], env: int, masks: Tensor) -> Tensor:
        initial_frontiers = self._observations_cache[env]["frontier_sensor"]
        frontiers = [
            frontier for frontier in initial_frontiers if tuple(frontier) not in self._obstacle_map[env]._disabled_frontiers
        ]
        temp_flag = False
        if np.array_equal(frontiers, np.zeros((1, 2))) or len(frontiers) == 0: # no frontier in this floor, check if there is stair
            if self._obstacle_map[env]._reinitialize_flag == False and self._obstacle_map[env]._floor_num_steps < 50: # 防止楼梯间状态
                # temp_floor_num_steps = self._obstacle_map[env]._floor_num_steps.copy()
                # 如果有之前的楼梯，可能还是保留一下比较好？ 
                # 如果连通域错误，可能不要保留的好 
                # temp_up_stair = self._obstacle_map[env]._up_stair_map.copy()
                # temp_down_stair = self._obstacle_map[env]._down_stair_map.copy()
                self._object_map[env].reset()
                self._value_map[env].reset()

                if self._compute_frontiers:
                    self._obstacle_map[env].reset()
                    self._obstacle_map[env]._reinitialize_flag = True
                    # self._obstacle_map[env]._up_stair_map = temp_up_stair
                    # self._obstacle_map[env]._down_stair_map = temp_down_stair
                # 防止之前episode爬楼梯异常退出
                self._climb_stair_over[env] = True
                self._reach_stair[env] = False
                self._reach_stair_centroid[env] = False
                self._stair_dilate_flag[env] = False
                self._pitch_angle[env] = 0
                self._done_initializing[env] = False
                self._initialize_step[env] = 0
                pointnav_action = self._initialize(env,masks)
                return pointnav_action
            else:
                self._obstacle_map[env]._this_floor_explored = True # 标记
                # 目前该逻辑是探索完一层,就去探索其他层,同时上楼优先。这个逻辑后面可能改掉
                if self._obstacle_map[env]._has_up_stair:
                    # 有上楼的楼梯,检查更高的楼层
                    for ith_floor in range(self._cur_floor_index[env] + 1, len(self._object_map_list[env])):
                        if not self._obstacle_map_list[env][ith_floor]._this_floor_explored:
                            temp_flag = True
                            break

                    if temp_flag:
                        self._climb_stair_over[env] = False
                        self._climb_stair_flag[env] = 1
                        self._stair_frontier[env] = self._obstacle_map[env]._up_stair_frontiers
                        pointnav_action = self._pointnav(observations, self._stair_frontier[env][0], stop=False, env=env)
                        return pointnav_action
                    elif self._obstacle_map[env]._has_down_stair:
                        # 没有上楼的楼梯,但有下楼的楼梯
                        for ith_floor in range(self._cur_floor_index[env] - 1, -1, -1):
                            if not self._obstacle_map_list[env][ith_floor]._this_floor_explored:
                                temp_flag = True
                                break
                        if temp_flag:
                            self._climb_stair_over[env] = False
                            self._climb_stair_flag[env] = 2
                            self._stair_frontier[env] = self._obstacle_map[env]._down_stair_frontiers
                            pointnav_action = self._pointnav(observations, self._stair_frontier[env][0], stop=False, env=env)
                            return pointnav_action
                        else:
                            print("In all floors, no frontiers found during exploration, stopping.")
                            return self._stop_action.to(masks.device)
                    else:
                        # if self._obstacle_map[env]._floor_num_steps < 50:
                        print("In all floors, no frontiers found during exploration, stopping.")
                        return self._stop_action.to(masks.device)
            
                elif self._obstacle_map[env]._has_down_stair:
                    # 如果只有下楼的楼梯
                    for ith_floor in range(self._cur_floor_index[env] - 1, -1, -1):
                        if not self._obstacle_map_list[env][ith_floor]._this_floor_explored:
                            temp_flag = True
                            break
                    
                    if temp_flag:
                        self._climb_stair_over[env] = False
                        self._climb_stair_flag[env] = 2
                        self._stair_frontier[env] = self._obstacle_map[env]._down_stair_frontiers
                        pointnav_action = self._pointnav(observations, self._stair_frontier[env][0], stop=False, env=env)
                        return pointnav_action
                    else:
                        print("In all floors, no frontiers found during exploration, stopping.")
                        return self._stop_action.to(masks.device)
                elif  self._obstacle_map[env]._tight_search_thresh == False: # 只有没找到楼梯的才tight
                    self._obstacle_map[env]._tight_search_thresh = True
                    return TorchActionIDs_plook.MOVE_FORWARD.to(masks.device)
                else:
                    print("No frontiers found during exploration, stopping.")
                    return self._stop_action.to(masks.device)

        # elif self._obstacle_map[env]._has_up_stair: #  and self._target_object[env] == "bed" 测试,直接上楼能力 加速上楼
        #     # 有上楼的楼梯,检查更高的楼层
        #     for ith_floor in range(self._cur_floor_index[env] + 1, len(self._object_map_list[env])):
        #         if not self._obstacle_map_list[env][ith_floor]._this_floor_explored:
        #             temp_flag = True
        #             break

        #     if temp_flag:
        #         self._climb_stair_over[env] = False
        #         self._climb_stair_flag[env] = 1
        #         self._stair_frontier[env] = self._obstacle_map[env]._up_stair_frontiers
        #         pointnav_action = self._pointnav(self._stair_frontier[env][0], stop=False, env=env)
        #         return pointnav_action
        #     else:
        #         # 如果没有楼梯
        #         print("No frontiers found during exploration, stopping.")
        #         return self._stop_action.to(masks.device)
                
        else:
            best_frontier, best_value = self._get_best_frontier(observations, frontiers, env)

            # best_frontier_2d = np.atleast_2d(best_frontier) 
            # best_frontier_px = self._obstacle_map[env]._xy_to_px(best_frontier_2d)
            # if self._obstacle_map[env]._up_stair_frontiers.shape[0] > 0 and self.is_point_within_distance_of_area(
            #     env, best_frontier_px, self._obstacle_map[env]._up_stair_map, 0.8) and len(self._obstacle_map[env]._up_stair_end) == 0: # 0.5 frontier离楼梯很近,又没爬过这个楼梯
            #     self._climb_stair_over[env] = False
            #     self._climb_stair_flag[env] = 1
            #     self._stair_frontier[env] = self._obstacle_map[env]._up_stair_frontiers
            # elif self._obstacle_map[env]._down_stair_frontiers.shape[0] > 0 and self.is_point_within_distance_of_area(
            #     env, best_frontier_px, self._obstacle_map[env]._down_stair_map, 0.8) and len(self._obstacle_map[env]._down_stair_end) == 0: # 0.5
            #     self._climb_stair_over[env] = False
            #     self._climb_stair_flag[env] = 2
            #     self._stair_frontier[env] = self._obstacle_map[env]._down_stair_frontiers

            # os.environ["DEBUG_INFO"] = f"Best value: {best_value*100:.2f}%"
            # print(f"Best value: {best_value*100:.2f}%")
            pointnav_action = self._pointnav(observations, best_frontier, stop=False, env=env, stop_radius=self._pointnav_stop_radius) # 探索的时候可以远一点停？
            if pointnav_action.item() == 0:
                print("Might stop, change to move forward.")
                pointnav_action.fill_(1)
            return pointnav_action

    def _look_for_downstair(self, observations: Union[Dict[str, Tensor], "TensorDict"], env: int, masks: Tensor) -> Tensor:
        # 如果已经有centroid就不用了
        if self._pitch_angle[env] >= 0:
            self._pitch_angle[env] -= self._pitch_angle_offset
            pointnav_action = TorchActionIDs_plook.LOOK_DOWN.to(masks.device)
        else:
            robot_xy = self._observations_cache[env]["robot_xy"]
            robot_xy_2d = np.atleast_2d(robot_xy) 
            dis_to_potential_stair = np.linalg.norm(self._obstacle_map[env]._potential_stair_centroid - robot_xy_2d)
            if dis_to_potential_stair > 0.2:
                pointnav_action = self._pointnav(observations,self._obstacle_map[env]._potential_stair_centroid[0], stop=False, env=env, stop_radius=self._pointnav_stop_radius) # 探索的时候可以远一点停？
                if pointnav_action.item() == 0:
                    print("Might false recognize down stairs, change to other mode.")
                    self._obstacle_map[env]._disabled_frontiers.add(tuple(self._obstacle_map[env]._potential_stair_centroid[0]))
                    print(f"Frontier {self._obstacle_map[env]._potential_stair_centroid[0]} is disabled due to no movement.")
                    # 需验证，一般来说，如果真有向下的楼梯，并不会执行到这里
                    self._obstacle_map[env]._disabled_stair_map[self._obstacle_map[env]._down_stair_map == 1] = 1
                    self._obstacle_map[env]._down_stair_map.fill(0)
                    self._obstacle_map[env]._has_down_stair = False
                    self._pitch_angle[env] += self._pitch_angle_offset
                    self._obstacle_map[env]._look_for_downstair_flag = False
                    pointnav_action = TorchActionIDs_plook.LOOK_UP.to(masks.device)
            else:
                print("Might false recognize down stairs, change to other mode.")
                self._obstacle_map[env]._disabled_frontiers.add(tuple(self._obstacle_map[env]._potential_stair_centroid[0]))
                print(f"Frontier {self._obstacle_map[env]._potential_stair_centroid[0]} is disabled due to no movement.")
                # 需验证，一般来说，如果真有向下的楼梯，并不会执行到这里
                self._obstacle_map[env]._disabled_stair_map[self._obstacle_map[env]._down_stair_map == 1] = 1
                self._obstacle_map[env]._down_stair_map.fill(0)
                self._obstacle_map[env]._has_down_stair = False
                self._pitch_angle[env] += self._pitch_angle_offset
                self._obstacle_map[env]._look_for_downstair_flag = False
                pointnav_action = TorchActionIDs_plook.LOOK_UP.to(masks.device)
        return pointnav_action
    
    def _pointnav(self, observations: "TensorDict", goal: np.ndarray, stop: bool = False, env: int = 0, ori_masks: Tensor = None, stop_radius: float = 0.9) -> Tensor: #, 
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
        if rho < stop_radius: # self._pointnav_stop_radius
            if stop:
                # if self._double_check_goal[env] == True:
                self._called_stop[env] = True
                    # print(f"Distance to possible goal is {self.min_distance_px[env]} m.") # For debug
                return self._stop_action.to(ori_masks.device)
                # else:
                #     print("Might false positive, change to look for the true goal.")
                #     self._object_map[env].clouds = {}
                #     self._try_to_navigate[env] = False
                #     self._object_map[env]._disabled_object_map[self._object_map[env]._map == 1] = 1
                #     self._object_map[env]._map.fill(0)
                #     action = self._explore(observations, env, ori_masks) # 果断换成探索
                #     return action
        action = self._pointnav_policy[env].act(obs_pointnav, masks, deterministic=True)
        return action

    def _navigate(self, observations: "TensorDict", goal: np.ndarray, stop: bool = False, env: int = 0, ori_masks: Tensor = None) -> Tensor:
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
        print(f"Distance to possible goal is {self.min_distance_px[env] / self._object_map[env].pixels_per_meter} m.")
        if self._might_close_to_goal[env] == True: # self._pointnav_stop_radius
            if stop:
                # if self._double_check_goal[env] == True:
                self._called_stop[env] = True
                return self._stop_action.to(ori_masks.device)
                # else:
                #     print("Might false positive, change to look for the true goal.")
                #     self._object_map[env].clouds = {}
                #     self._try_to_navigate[env] = False
                #     self._object_map[env]._disabled_object_map[self._object_map[env]._map == 1] = 1
                #     self._object_map[env]._map.fill(0)
                #     action = self._explore(observations, env, ori_masks) # 果断换成探索
                #     return action
        action = self._pointnav_policy[env].act(obs_pointnav, masks, deterministic=True)
        return action

    def astar_get_best_action(self, robot_xy, heading, goal, navigable_map):
        """
        计算机器人当前的最佳动作,并返回轨迹点
        
        参数:
        - robot_xy: 机器人当前坐标 (x, y)
        - heading: 机器人当前朝向角度,弧度制
        - goal: 目标坐标 (x, y)
        - navigable_map: 可导航地图,一个二维数组,True代表可达区域
        
        返回:
        - 最优动作(0:停止,1:前进,2:左转30度,3:右转30度)
        - 轨迹点列表,用于调试
        """
        
        def a_star(start_state, goal_state, navigable_map):
            open_list = []  # 优先队列（堆）
            open_dict = {}  # 用来存储状态与其代价的映射
            
            # 初始状态加入 open_list 和 open_dict
            heapq.heappush(open_list, (heuristic(start_state, goal_state), 0, start_state))  # (f, g, state)
            open_dict[start_state] = (0, heuristic(start_state, goal_state))  # g 和 f
            
            came_from = {}  # 路径回溯
            g_score = {start_state: 0}  # g值

            while open_list:
                _, current_g, current_state = heapq.heappop(open_list)

                # 如果当前状态就是目标状态，直接返回路径
                if heuristic(current_state, goal_state) <= 0.2 * self._obstacle_map[0].pixels_per_meter: # current_state == goal_state: # 
                    path = []
                    while current_state in came_from:
                        action, previous_state = came_from[current_state]
                        path.append((action, previous_state))
                        current_state = previous_state
                    path.reverse()  # 路径反向，返回顺序
                    return path

                # 遍历所有可能的动作：前进，左转，右转
                for action in [1, 2, 3]:  # 1:前进, 2:左转, 3:右转
                    next_state = get_next_state(current_state, action)
                    
                    if is_valid_state(next_state, navigable_map):
                        # 计算转弯和前进的代价：前进代价为1，转弯代价为0.5
                        tentative_g = current_g + (1 if action == 1 else 0.5)
                        
                        if next_state not in g_score or tentative_g < g_score[next_state]:
                            g_score[next_state] = tentative_g
                            f_score = tentative_g + heuristic(next_state, goal_state)

                            if next_state not in open_dict or open_dict[next_state][1] > f_score:
                                heapq.heappush(open_list, (f_score, tentative_g, next_state))
                                open_dict[next_state] = (tentative_g, f_score)
                                came_from[next_state] = (action, current_state)

            return []  # 无法到达目标，返回空路径

        def heuristic(state, goal_state):
            """ 计算欧几里得距离或者曼哈顿距离 """
            dx = abs(state[0] - goal_state[0])
            dy = abs(state[1] - goal_state[1])
            return dx + dy  # 曼哈顿距离：适用于网格状地图

        def get_next_state(current_state, action):
            """ 根据动作计算下一个状态 """
            if action == 1:  # 前进
                dx = math.cos(current_state[2]) * 0.25 * self._obstacle_map[0].pixels_per_meter
                dy = math.sin(current_state[2]) * 0.25 * self._obstacle_map[0].pixels_per_meter
                return (current_state[0] + dx, current_state[1] + dy, current_state[2])
            elif action == 2:  # 左转30度
                return (current_state[0], current_state[1], current_state[2] + math.radians(30))
            elif action == 3:  # 右转30度
                return (current_state[0], current_state[1], current_state[2] - math.radians(30))

        # def is_valid_state(state, navigable_map):
        #     """ 判断一个状态是否在可导航区域 """
        #     x, y = int(state[0]), int(state[1])
        #     if 0 <= x < len(navigable_map) and 0 <= y < len(navigable_map[0]):
        #         return navigable_map[x][y]
        #     return False
        def is_valid_state(state, navigable_map):
            """ 判断一个状态是否在可导航区域，并检查其周围邻近点是否可达 """
            x, y = int(state[0]), int(state[1])

            # 如果当前点不在导航区域内，直接返回 False
            if not (0 <= x < len(navigable_map) and 0 <= y < len(navigable_map[0])):
                return False
            
            # 检查当前点是否可达
            if not navigable_map[x][y]:
                return False
            
            # 检查周围的九宫格内的点是否可达
            # 上下左右和四个对角线方向
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue  # 跳过自己
                    nx, ny = x + dx, y + dy
                    # 检查邻近点是否越界
                    if 0 <= nx < len(navigable_map) and 0 <= ny < len(navigable_map[0]):
                        # 如果任何邻近点不可达，则当前点无效
                        if not navigable_map[nx][ny]:
                            return False
                    else:
                        return False  # 如果邻近点越界，认为当前点无效
            
            # 如果所有检查通过，则返回 True
            return True

        # 初始化起始状态
        start_state = (robot_xy[0], robot_xy[1], heading)  # (x, y, heading)
        goal_state = (goal[0], goal[1], 0)  # 假设目标的朝向为0

        # 执行A*算法来规划路径
        path = a_star(start_state, goal_state, navigable_map)
        
        # 如果路径为空,表示无法到达目标,返回停止动作
        if not path:
            return 0, []  # 停止并返回空路径
        
        # 获取最优的动作,轨迹点
        optimal_action, _ = path[0]  # 第一个动作
        trajectory_points = [state for _, state in path]  # 路径上的状态点
        
        return optimal_action, trajectory_points

    def _get_close_to_stair(self, observations: "TensorDict", env: int, ori_masks: Tensor) -> Tensor:

        masks = torch.tensor([self._num_steps[env] != 0], dtype=torch.bool, device="cuda")
        robot_xy = self._observations_cache[env]["robot_xy"]
        heading = self._observations_cache[env]["robot_heading"]
        # each step update (stair may be updated)
        # 防止空气墙或者动作循环,强制更换 frontier
        if  self._climb_stair_flag[env] == 1:
            self._stair_frontier[env] = self._obstacle_map[env]._up_stair_frontiers
        elif  self._climb_stair_flag[env] == 2:
            self._stair_frontier[env] = self._obstacle_map[env]._down_stair_frontiers
        if np.array_equal(self._last_frontier[env], self._stair_frontier[env][0]):
            # 检查是否是第一次选中该前沿
            if self._frontier_stick_step[env] == 0:
                # 记录初始的距离(首次选中该前沿时)
                self._last_frontier_distance[env] = np.linalg.norm(self._stair_frontier[env] - robot_xy)
                self._frontier_stick_step[env] += 1
            else:
                # 计算当前与前沿的距离
                current_distance = np.linalg.norm(self._stair_frontier[env] - robot_xy)
                print(f"Distance Change: {np.abs(self._last_frontier_distance[env] - current_distance)} and Stick Step {self._frontier_stick_step[env]}")
                # 检查距离变化是否超过 1 米
                if np.abs(self._last_frontier_distance[env] - current_distance) > 0.3:
                    # 如果距离变化超过 1 米,重置步数和更新距离
                    self._frontier_stick_step[env] = 0
                    self._last_frontier_distance[env] = current_distance  # 更新为当前距离
                else:
                    # 如果步数达到 30 且没有明显的距离变化(< 0.3 米),禁用前沿
                    if self._frontier_stick_step[env] >= 30:
                        self._obstacle_map[env]._disabled_frontiers.add(tuple(self._stair_frontier[env][0]))
                        print(f"Frontier {self._stair_frontier[env]} is disabled due to no movement.")
                        if  self._climb_stair_flag[env] == 1:
                            self._obstacle_map[env]._disabled_stair_map[self._obstacle_map[env]._up_stair_map == 1] = 1
                            self._obstacle_map[env]._up_stair_map.fill(0)
                            self._climb_stair_flag[env] = 0
                            self._climb_stair_over[env] = True
                            self._obstacle_map[env]._has_up_stair = False
                            # self._obstacle_map[env]._climb_stair_paused_step = 0
                        elif  self._climb_stair_flag[env] == 2:
                            self._obstacle_map[env]._disabled_stair_map[self._obstacle_map[env]._down_stair_map == 1] = 1
                            self._obstacle_map[env]._down_stair_frontiers.fill(0)
                            self._climb_stair_flag[env] = 0
                            self._climb_stair_over[env] = True
                            self._obstacle_map[env]._has_down_stair = False
                            # self._obstacle_map[env]._climb_stair_paused_step = 0
                        action = self._explore(observations, env, ori_masks) # 果断换成探索
                        return action
                    else:
                        # 如果没有达到 30 步,继续增加步数
                        self._frontier_stick_step[env] += 1
        else:
            # 如果选中了不同的前沿,重置步数和距离
            self._frontier_stick_step[env] = 0
            self._last_frontier_distance[env] = 0
        self._last_frontier[env] = self._stair_frontier[env][0]

        # 点导航模型算动作
        rho, theta = rho_theta(robot_xy, heading, self._stair_frontier[env][0]) # stair_frontiers[0]) # 
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
        action = self._pointnav_policy[env].act(obs_pointnav, masks, deterministic=True) 
        if action.item() == 0:
            self._obstacle_map[env]._disabled_frontiers.add(tuple(self._stair_frontier[env][0]))
            print(f"Frontier {self._stair_frontier[env]} is disabled due to no movement.")
            if  self._climb_stair_flag[env] == 1:
                self._obstacle_map[env]._disabled_stair_map[self._obstacle_map[env]._up_stair_map == 1] = 1
                self._obstacle_map[env]._up_stair_map.fill(0)
                self._obstacle_map[env]._up_stair_frontiers = np.array([])
                self._climb_stair_over[env] = True
                self._obstacle_map[env]._has_up_stair = False
            elif  self._climb_stair_flag[env] == 2:
                self._obstacle_map[env]._disabled_stair_map[self._obstacle_map[env]._down_stair_map == 1] = 1
                self._obstacle_map[env]._down_stair_map.fill(0)
                self._obstacle_map[env]._down_stair_frontiers = np.array([])
                self._climb_stair_over[env] = True
                self._obstacle_map[env]._has_down_stair = False
            self._climb_stair_flag[env] = 0
            action = self._explore(observations, env, ori_masks) # 果断换成探索
        return action

    def _climb_stair(self, observations: "TensorDict", env: int, ori_masks: Tensor) -> Tensor:
        masks = torch.tensor([self._num_steps[env] != 0], dtype=torch.bool, device="cuda")
        # if self._reach_stair[env] == True: # 进了入口之后,爬楼梯,frontier定的比较远
        robot_xy = self._observations_cache[env]["robot_xy"]
        robot_xy_2d = np.atleast_2d(robot_xy) 
        robot_px = self._obstacle_map[env]._xy_to_px(robot_xy_2d)
        heading = self._observations_cache[env]["robot_heading"]  # 以弧度为单位

        if  self._climb_stair_flag[env] == 1:
            self._stair_frontier[env] = self._obstacle_map[env]._up_stair_frontiers
        elif  self._climb_stair_flag[env] == 2:
            self._stair_frontier[env] = self._obstacle_map[env]._down_stair_frontiers
        current_distance = np.linalg.norm(self._stair_frontier[env] - robot_xy)
        print(f"Climb Stair -- Distance Change: {np.abs(self._last_frontier_distance[env] - current_distance)} and Stick Step {self._obstacle_map[env]._climb_stair_paused_step}")
        # 检查距离变化是否超过 1 米
        if np.abs(self._last_frontier_distance[env] - current_distance) > 0.2:
            # 如果距离变化超过 1 米,重置步数和更新距离
            self._obstacle_map[env]._climb_stair_paused_step = 0
            self._last_frontier_distance[env] = current_distance  # 更新为当前距离
        else:
            self._obstacle_map[env]._climb_stair_paused_step += 1
        
        if self._obstacle_map[env]._climb_stair_paused_step > 15:
            self._obstacle_map[env]._disable_end = True

        # 进了入口但没爬到质心(agent中心的点还没到楼梯),先往楼梯质心走
        if self._reach_stair_centroid[env] == False:
            stair_frontiers = self._stair_frontier[env][0] # self._observations_cache[env]["frontier_sensor"]
            rho, theta = rho_theta(robot_xy, heading, stair_frontiers)
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
            action = self._pointnav_policy[env].act(obs_pointnav, masks, deterministic=True)
            # do not allow stop
            # might near the centroid
            if action.item() == 0:
                self._reach_stair_centroid[env] = True
                print("Might close, change to move forward.") # 
                action[0] = 1
            return action

        # 爬到了楼梯质心,对于下楼梯
        elif self._climb_stair_flag[env] == 2 and self._pitch_angle[env] < -30: 
            self._pitch_angle[env] += self._pitch_angle_offset
            # mode = "look_up"
            print("Look up a little for downstair!")
            action = TorchActionIDs_plook.LOOK_UP.to(masks.device)
            return action
        
        else:
            # 距离2米的目标点,像在驴子前面吊一根萝卜
            distance = 0.8 # 1.2 # 1.2下不了6s的楼梯 # 0.3 上不去cv的楼梯 0.5上不去xb的楼梯

            # 当前位置与深度最大区域中点距离 1.2 米的位置
            # 找到深度图的最大值
            depth_map = self._observations_cache[env]["nav_depth"].squeeze(0).cpu().numpy()
            max_value = np.max(depth_map)
            # 找到所有最大值的坐标
            max_indices = np.argwhere(depth_map == max_value)  # 返回所有满足条件的 (v, u) 坐标
            # 计算这些坐标的平均值,得到中心点
            center_point = np.mean(max_indices, axis=0).astype(int)
            v, u = center_point[0], center_point[1] # 注意顺序:v对应行,u对应列
            # 定义相机的水平视场角 (以弧度为单位)
            # 计算列偏移量相对于图像中心的归一化值 (-1 到 1)
            normalized_u = (u - self._cx) / self._cx # (width / 2)

            # 限制归一化值在 -1 到 1 之间
            normalized_u = np.clip(normalized_u, -1, 1)

            # 计算角度偏差
            angle_offset = normalized_u * (self._camera_fov / 2)
            # 计算目标方向的角度
            target_heading = heading - angle_offset # 原来是加的,－试试
            target_heading = target_heading % (2 * np.pi)
            x_target = robot_xy[0] + distance * np.cos(target_heading)
            y_target = robot_xy[1] + distance * np.sin(target_heading)
            target_point = np.array([x_target, y_target])
            target_point_2d = np.atleast_2d(target_point) 
            temp_target_px = self._obstacle_map[env]._xy_to_px(target_point_2d) # self._obstacle_map[env]._carrot_goal_px
            if  self._climb_stair_flag[env] == 1:
                this_stair_end = self._obstacle_map[env]._up_stair_end
            elif  self._climb_stair_flag[env] == 2:
                this_stair_end = self._obstacle_map[env]._down_stair_end

            # 不能用存的,因为终点可能会变
            # else:
            if len(self._last_carrot_xy[env]) == 0 or len(this_stair_end) == 0: # 最开始的时候
                self._carrot_goal_xy[env] = target_point
                self._obstacle_map[env]._carrot_goal_px = temp_target_px
                self._last_carrot_xy[env] = target_point
                self._last_carrot_px[env] = temp_target_px
            elif np.linalg.norm(this_stair_end - robot_px) <= 0.5 * self._obstacle_map[env].pixels_per_meter or self._obstacle_map[env]._disable_end == True:   # 0.5上不去xb
                self._carrot_goal_xy[env] = target_point
                self._obstacle_map[env]._carrot_goal_px = temp_target_px
                self._last_carrot_xy[env] = target_point
                self._last_carrot_px[env] = temp_target_px # 离终点很近了,可能要走出楼梯(终点不一定准的)
            else: # 计算L1距离
                l1_distance = np.abs(this_stair_end[0] - temp_target_px[0][0]) + np.abs(this_stair_end[1] - temp_target_px[0][1])
                last_l1_distance = np.abs(this_stair_end[0] - self._last_carrot_px[env][0][0]) + np.abs(this_stair_end[1] - self._last_carrot_px[env][0][1])
                if last_l1_distance > l1_distance:
                    self._carrot_goal_xy[env] = target_point
                    self._obstacle_map[env]._carrot_goal_px = temp_target_px
                    self._last_carrot_xy[env] = target_point
                    self._last_carrot_px[env] = temp_target_px
                else: # 维持上一个不变
                    self._carrot_goal_xy[env] = self._last_carrot_xy[env]
                    self._obstacle_map[env]._carrot_goal_px = self._last_carrot_px[env]

            rho, theta = rho_theta(robot_xy, heading, self._carrot_goal_xy[env]) # target_point)
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
            action = self._pointnav_policy[env].act(obs_pointnav, masks, deterministic=True)
            # do not allow stop
            if action.item() == 0:
                print("Might stop, change to move forward.")
                action[0] = 1
            return action

    def _reverse_climb_stair(self, observations: "TensorDict", env: int, ori_masks: Tensor) -> Tensor:
        if self._climb_stair_flag[env] == 1 and self._pitch_angle[env] >= 0: # 如果原先上楼，那么现在要回去(下楼)，角度调到向下30度 
            # mode = "look_down_back"
            self._pitch_angle[env] -= self._pitch_angle_offset
            pointnav_action = TorchActionIDs_plook.LOOK_DOWN.to(ori_masks.device)
            return pointnav_action
        elif self._climb_stair_flag[env] == 2 and self._pitch_angle[env] <= 0: # 如果原先下楼，那么现在要回去(上楼)，角度调到向上30度 
            # mode = "look_up_back"
            self._pitch_angle[env] += self._pitch_angle_offset
            pointnav_action = TorchActionIDs_plook.LOOK_UP.to(ori_masks.device)
            return pointnav_action
        masks = torch.tensor([self._num_steps[env] != 0], dtype=torch.bool, device="cuda")
        # if self._reach_stair[env] == True: # 进了入口之后,爬楼梯,frontier定的比较远
        robot_xy = self._observations_cache[env]["robot_xy"]
        # robot_xy_2d = np.atleast_2d(robot_xy) 
        # robot_px = self._obstacle_map[env]._xy_to_px(robot_xy_2d)
        heading = self._observations_cache[env]["robot_heading"]  # 以弧度为单位
        
        # 直接往楼梯起点走，走到很近的时候换成探索，并且取消原来的楼梯
        if  self._climb_stair_flag[env] == 1:
            start_point = self._obstacle_map[env]._up_stair_start
        elif  self._climb_stair_flag[env] == 2:
            start_point = self._obstacle_map[env]._down_stair_start
        start_distance = np.linalg.norm(start_point - robot_xy)
        if start_distance > 0.3:
            rho, theta = rho_theta(robot_xy, heading, start_point)
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
            action = self._pointnav_policy[env].act(obs_pointnav, masks, deterministic=True)
            # do not allow stop
            # might near the centroid
            if action.item() == 0:
                print("Might close, change to move forward.") # 
                action[0] = 1
            return action
        else:
            self._obstacle_map[env]._climb_stair_paused_step = 0
            self._last_carrot_xy[env] = []
            self._last_carrot_px[env] = []
            self._reach_stair[env] = False
            self._reach_stair_centroid[env] = False
            self._stair_dilate_flag[env] = False
            self._climb_stair_over[env] = True
            self._obstacle_map[env]._disabled_frontiers.add(tuple(self._stair_frontier[env][0]))
            print(f"Frontier {self._stair_frontier[env]} is disabled due to no movement.")
            if  self._climb_stair_flag[env] == 1:
                self._obstacle_map[env]._disabled_stair_map[self._obstacle_map[env]._up_stair_map == 1] = 1
                self._obstacle_map[env]._up_stair_map.fill(0)
                self._obstacle_map[env]._up_stair_frontiers = np.array([])
                self._obstacle_map[env]._has_up_stair = False
            elif  self._climb_stair_flag[env] == 2:
                self._obstacle_map[env]._disabled_stair_map[self._obstacle_map[env]._down_stair_map == 1] = 1
                self._obstacle_map[env]._down_stair_map.fill(0)
                self._obstacle_map[env]._down_stair_frontiers = np.array([])
                self._obstacle_map[env]._has_down_stair = False
            self._climb_stair_flag[env] = 0
        
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
        # 计算每个前沿与机器人之间的距离
        distances = [np.linalg.norm(frontier - robot_xy) for frontier in sorted_pts]
        
        # 首先筛选出距离小于等于3米的前沿
        close_frontiers = [
            (idx, frontier, distance) 
            for idx, (frontier, distance) in enumerate(zip(sorted_pts, distances)) 
            if distance <= 3.0
        ]
        
        if close_frontiers:
            # 如果有多个前沿离机器人都很近,则选择距离最小的前沿
            closest_frontier = min(close_frontiers, key=lambda x: x[2])  # 根据距离排序
            best_frontier_idx = closest_frontier[0]
            print(f"Frontier {sorted_pts[best_frontier_idx]} is very close (distance: {distances[best_frontier_idx]:.2f}m), selecting it.")
        else:
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

        # 防止空气墙或者动作循环,强制更换 frontier
        if np.array_equal(self._last_frontier[env],best_frontier):
            # 检查是否是第一次选中该前沿
            if self._frontier_stick_step[env] == 0:
                # 记录初始的距离(首次选中该前沿时)
                self._last_frontier_distance[env] = np.linalg.norm(best_frontier - robot_xy)
                self._frontier_stick_step[env] += 1
            else:
                # 计算当前与前沿的距离
                current_distance = np.linalg.norm(best_frontier - robot_xy)
                
                # 检查距离变化是否超过 1 米
                if np.abs(self._last_frontier_distance[env] - current_distance) > 0.3:
                    # 如果距离变化超过 1 米,重置步数和更新距离
                    self._frontier_stick_step[env] = 0
                    self._last_frontier_distance[env] = current_distance  # 更新为当前距离
                else:
                    # 如果步数达到 30 且没有明显的距离变化(< 0.3 米),禁用前沿
                    if self._frontier_stick_step[env] >= 30:
                        self._obstacle_map[env]._disabled_frontiers.add(tuple(best_frontier))
                        print(f"Frontier {best_frontier} is disabled due to no movement.")
                    else:
                        # 如果没有达到 30 步,继续增加步数
                        self._frontier_stick_step[env] += 1
        else:
            # 如果选中了不同的前沿,重置步数和距离
            self._frontier_stick_step[env] = 0
            self._last_frontier_distance[env] = 0

        ## 防止来回走动
        frontier_tuple = tuple(best_frontier)
        if frontier_tuple not in self._obstacle_map[env]._best_frontier_selection_count:
            self._obstacle_map[env]._best_frontier_selection_count[frontier_tuple] = 0
        self._obstacle_map[env]._best_frontier_selection_count[frontier_tuple] += 1
        # 检查选中次数是否超过 60 次，如果超过，则禁用该前沿
        if self._obstacle_map[env]._best_frontier_selection_count[frontier_tuple] >= 40:
            self._obstacle_map[env]._disabled_frontiers.add(frontier_tuple)
            print(f"Frontier {best_frontier} has been selected {self._obstacle_map[env]._best_frontier_selection_count[frontier_tuple]} times, now disabled.")
        
        self._last_value[env] = best_value
        self._last_frontier[env] = best_frontier
        os.environ["DEBUG_INFO"] += f" Best value: {best_value*100:.2f}%"
        print(f"Now the best_frontier is {best_frontier}")
        return best_frontier, best_value

    def _get_object_detections_with_stair_and_person(self, img: np.ndarray, env: int) -> ObjectDetections:

        target_classes = self._target_object[env].split("|")
        target_in_coco = any(c in COCO_CLASSES for c in target_classes) and self._load_yolo
        target_in_non_coco = any(c not in COCO_CLASSES for c in target_classes)

        non_coco_caption_with_stair = self._non_coco_caption +  "stair ." #  stair . "step ." 基本认不出楼梯
        # 进行目标检测
        coco_detections = self._coco_object_detector.predict(img)
        non_coco_detections = self._object_detector.predict(img, caption=non_coco_caption_with_stair)
        
        # 过滤目标类别和置信度
        if target_in_coco:
            target_detections = deepcopy(coco_detections)
        elif target_in_non_coco:
            target_detections = deepcopy(non_coco_detections)
        target_detections.filter_by_class(target_classes)  
        det_conf_threshold = self._coco_threshold if target_in_coco else self._non_coco_threshold
        target_detections.filter_by_conf(det_conf_threshold)

        # 如果目标检测结果为空，尝试重新检测
        if target_in_coco and target_in_non_coco and target_detections.num_detections == 0:
            target_detections = self._object_detector.predict(img, caption=self._non_coco_caption)
            target_detections.filter_by_class(target_classes)  
            target_detections.filter_by_conf(self._non_coco_threshold)

        return target_detections, coco_detections, non_coco_detections

    def _update_object_map_with_stair_and_person(self, height: int, width: int): # -> Tuple[ ObjectDetections, List[np.ndarray] ]:
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
        # detections_list = []
        self._object_masks = np.zeros((self._num_envs, height, width), dtype=np.uint8)
        self._person_masks = np.zeros((self._num_envs, height, width), dtype=bool)
        self._stair_masks = np.zeros((self._num_envs, height, width), dtype=bool)
        for env in range(self._num_envs):
            # for target
            object_map_rgbd = self._observations_cache[env]["object_map_rgbd"]
            rgb, depth, tf_camera_to_episodic, min_depth, max_depth, fx, fy = object_map_rgbd[0]
            target_detections, coco_detections, non_coco_detections = self._get_object_detections_with_stair_and_person(rgb, env) # get three detections
        
            
            if np.array_equal(depth, np.ones_like(depth)) and target_detections.num_detections > 0:
                depth = self._infer_depth(rgb, min_depth, max_depth)
                obs = list(self._observations_cache[env]["object_map_rgbd"][0])
                obs[1] = depth
                self._observations_cache["object_map_rgbd"][0] = tuple(obs)

            for idx in range(len(target_detections.logits)):
                target_bbox_denorm = target_detections.boxes[idx] * np.array([width, height, width, height])
                object_mask = self._mobile_sam.segment_bbox(rgb, target_bbox_denorm.tolist())

                # If we are using vqa, then use the BLIP2 model to visually confirm whether
                # the contours are actually correct.
                if self._use_vqa:
                    contours, _ = cv2.findContours(object_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    annotated_rgb = cv2.drawContours(rgb.copy(), contours, -1, (255, 0, 0), 2)
                    question = f"Question: {self._vqa_prompt}"
                    if not target_detections.phrases[idx].endswith("ing"):
                        question += "a "
                    question += target_detections.phrases[idx] + "? Answer:"
                    answer = self._vqa.ask(annotated_rgb, question)
                    if not answer.lower().startswith("yes"):
                        continue

                self._object_masks[env][object_mask > 0] = 1 # for drawing
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
            
            # 多楼层需要考虑stair - gdino - non_coco，有人场景需要考虑person - yolo - coco
            # for person
            coco_detections.filter_by_class(["person"])
            coco_detections.filter_by_conf(self._coco_threshold)
            
            for idx in range(len(coco_detections.logits)):
                person_bbox_denorm = coco_detections.boxes[idx] * np.array([width, height, width, height])
                person_mask = self._mobile_sam.segment_bbox(rgb, person_bbox_denorm.tolist())
                self._person_masks[env][person_mask > 0] = 1
                self._object_map[env].update_movable_map(
                    "person", # no phrase, because object_map only record target (especially after update_explored)
                    depth,
                    person_mask,
                    tf_camera_to_episodic,
                    min_depth,
                    max_depth,
                    fx,
                    fy,
                )

            # for stair        
            non_coco_detections.filter_by_class(["stair"]) # try stair step
            non_coco_detections.filter_by_conf(0.60) # for stair, 0.80 seems a good value to filter for up, 0.60 for down
            
            for idx in range(len(non_coco_detections.logits)):
                stair_bbox_denorm = non_coco_detections.boxes[idx] * np.array([width, height, width, height])
                stair_mask = self._mobile_sam.segment_bbox(rgb, stair_bbox_denorm.tolist())
                self._stair_masks[env][stair_mask > 0] = 1         
            # final
            cone_fov = get_fov(fx, depth.shape[1])
            self._object_map[env].update_explored(tf_camera_to_episodic, max_depth, cone_fov)
            # detections_list.append(target_detections)
            self.target_detection_list[env] = target_detections
            self.coco_detection_list[env] = coco_detections
            self.non_coco_detection_list[env] = non_coco_detections
        # return detections_list