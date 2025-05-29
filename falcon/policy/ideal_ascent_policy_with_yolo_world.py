from vlfm.policy.itm_policy import ITMPolicyV2
from vlfm.policy.habitat_policies import HabitatMixin
from habitat_baselines.common.baseline_registry import baseline_registry
from typing import Dict, Tuple, Any, Union, List
from vlfm.policy.base_objectnav_policy import BaseObjectNavPolicy
from habitat_baselines.common.tensor_dict import TensorDict
from depth_camera_filtering import filter_depth
from vlfm.utils.geometry_utils import xyz_yaw_to_tf_matrix
import numpy as np
import torch
import cv2
import os
import logging
from RedNet.RedNet_model import load_rednet
from constants import MPCAT40_RGB_COLORS
from torch import Tensor
from habitat_baselines.rl.ppo.policy import PolicyActionData
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from vlfm.policy.habitat_policies import HM3D_ID_TO_NAME, MP3D_ID_TO_NAME
from vlfm.utils.geometry_utils import get_fov, rho_theta
from vlfm.obs_transformers.utils import image_resize
from vlfm.vlm.ram_test import RAMClient
from vlfm.vlm.sam import MobileSAMClient
from vlfm.vlm.blip2itm_async_server import AsyncBLIP2ITMClient
from vlfm.vlm.qwen25itm import Qwen2_5ITMClient
from vlfm.policy.utils.pointnav_policy import WrappedPointNavResNetPolicy
from vlfm.policy.utils.acyclic_enforcer import AcyclicEnforcer
from torchvision import transforms as trn
from torch.autograd import Variable as V
import torchvision.models as models
from torch.nn import functional as F
from vlfm.vlm.detections import ObjectDetections
from vlfm.vlm.yolo_world_test import YoloWorldClient_MF
from vlfm.vlm.dfine_test import DFineClient
from PIL import Image
import json
from skimage.metrics import structural_similarity as ssim
from falcon.policy.map_utils import ObstacleMapUpdater,ValueMapUpdater,ObjectMapUpdater
from falcon.policy.data_utils import (
    floor_probabilities_df,
    reference_rooms,
    direct_mapping,
    xyz_yaw_pitch_roll_to_tf_matrix,
    check_stairs_in_upper_50_percent,
    PROMPT_SEPARATOR,
    STAIR_CLASS_ID,
    INDENT_L1,
    INDENT_L2,
    TorchActionIDs_plook,
    knowledge_graph,
) 


@baseline_registry.register_policy
class HabitatITMPolicy_Qwen_YOLO_WORLD_IDEAL(HabitatMixin, ITMPolicyV2):
    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):

        self.scene_id = None
        self.object_mapping = {
            "couch": "sofa",
            "tv": "monitor",
            "potted plant": "plant",
        }


        self._action_space = kwargs["action_space"]

        # BaseObjectNavPolicy
        self._policy_info = {}
        self._stop_action = TorchActionIDs_plook.STOP 
        self._observations_cache = {}
        self._load_yolo: bool = True
        self._object_detector = YoloWorldClient_MF(port=int(os.environ.get("YOLO_WORLD_PORT", "13184")))
        self._coco_object_detector = DFineClient(port=int(os.environ.get("DFINE_PORT", "13186")))
        self._mobile_sam = MobileSAMClient(port=int(os.environ.get("SAM_PORT", "13183")))
        self._ram = RAMClient(port=int(os.environ.get("RAM_PORT", "13185")))
        
        self._pointnav_stop_radius = kwargs["pointnav_stop_radius"] 
        self._visualize = kwargs["visualize"]
        self._vqa_prompt = kwargs["vqa_prompt"]

        self._coco_threshold = kwargs["coco_threshold"] # 0.8 in base_objectnav_policy
        self._non_coco_threshold = kwargs["non_coco_threshold"] # 0.4 in base_objectnav_policy

        ## num_envs
        self._num_envs = kwargs['num_envs']
        self._object_map_erosion_size = kwargs["object_map_erosion_size"]
        self._object_map_list = [[ObjectMapUpdater(erosion_size=self._object_map_erosion_size,size=1600,)] for _ in range(self._num_envs)]
        self._depth_image_shape =  tuple(kwargs["depth_image_shape"]) # (224, 224) 
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
                [ObstacleMapUpdater(
                min_height=self.min_obstacle_height,
                max_height=self.max_obstacle_height,
                area_thresh=self.obstacle_map_area_threshold,
                agent_radius=self.agent_radius,
                hole_area_thresh=self.hole_area_thresh,
                size=1600,
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
        self._llm = Qwen2_5ITMClient(port=int(os.environ.get("QWEN2_5ITM_PORT", "13181")))
        self._itm = AsyncBLIP2ITMClient(port=int(os.environ.get("ASYNCBLIP2ITM_PORT", "13182")))
        self._text_prompt = kwargs["text_prompt"]

        self.use_max_confidence = kwargs["use_max_confidence"]
        self._value_map_list = [ [ValueMapUpdater(
            value_channels=len(self._text_prompt.split(PROMPT_SEPARATOR)),
            use_max_confidence = self.use_max_confidence,
            obstacle_map=None,  size=1600,
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
        self._image_width = kwargs["image_width"] # add later
        self._image_height = 480 # add later
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
        self._stair_masks = []
        self._climb_stair_over = [True for _ in range(self._num_envs)]
        self._reach_stair = [False for _ in range(self._num_envs)]
        self._reach_stair_centroid = [False for _ in range(self._num_envs)]
        self._stair_frontier = [None for _ in range(self._num_envs)]
        
        # add to manage the maps of each floor
        self._cur_floor_index = [0 for _ in range(self._num_envs)]
        self._object_map = [self._object_map_list[env][self._cur_floor_index[env]] for env in range(self._num_envs)]
        self._obstacle_map = [self._obstacle_map_list[env][self._cur_floor_index[env]] for env in range(self._num_envs)]
        self._value_map = [self._value_map_list[env][self._cur_floor_index[env]] for env in range(self._num_envs)]

        self.all_detection_list = [None for _ in range(self._num_envs)]
        self.target_detection_list = [None for _ in range(self._num_envs)]
        self.coco_detection_list = [None for _ in range(self._num_envs)]
        self.non_coco_detection_list = [None for _ in range(self._num_envs)]

        self._climb_stair_flag = [0 for _ in range(self._num_envs)]
        self._stair_dilate_flag = [False for _ in range(self._num_envs)]
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
        self._try_to_navigate_step = [0 for _ in range(self._num_envs)]
        self._initialize_step = [0 for _ in range(self._num_envs)]

        ## stop distance
        self.min_distance_xy = [np.inf for _ in range(self._num_envs)]
        self.cur_dis_to_goal = [np.inf for _ in range(self._num_envs)]
        self._force_frontier = [np.zeros(2) for _ in range(self._num_envs)]

        ## continuous exploration for frontiers
        self.cur_frontier = [np.array([]) for _ in range(self._num_envs)] 
        self.frontier_rgb_list = [[] for _ in range(self._num_envs)]
        self.frontier_step_list = [[] for _ in range(self._num_envs)]
        self.vlm_response = ["" for _ in range(self._num_envs)]

        # Place365
        arch = 'resnet50'
        model_file = 'place365/%s_places365.pth.tar' % arch
        self.scene_classify_model = models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        load_result = self.scene_classify_model.load_state_dict(state_dict, strict=False)

        # 检查是否有缺失或多余的键
        if load_result.missing_keys:
            print(f"Missing keys: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            print(f"Unexpected keys: {load_result.unexpected_keys}")

        # 将模型移动到指定设备
        self.scene_classify_model = self.scene_classify_model.to(self.device)
        self.scene_classify_model.eval()
        self.place365_centre_crop = trn.Compose([
                trn.Resize((256,256)),
                trn.CenterCrop(224),
                trn.ToTensor(),
                trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        place365_file_name = 'place365/categories_places365.txt'
        self.place365_classes = list()
        with open(place365_file_name) as class_file:
            for line in class_file:
                self.place365_classes.append(line.strip().split(' ')[0][3:])
        self.place365_classes = tuple(self.place365_classes)
        self.floor_num = [len(self._obstacle_map_list[env]) for env in range(self._num_envs)]
        self._blip_cosine = [0 for _ in range(self._num_envs)]
        self.multi_floor_ask_step = [0 for _ in range(self._num_envs)]
        self._get_close_to_stair_step = [0 for _ in range(self._num_envs)]
            
        self.semantic_dict =  [
            {
                "scene_id": None,
                "target_ids": None,
                "target_names": None,
            } for _ in range(self._num_envs)]
        
    def _reset(self, env: int) -> None:

        self._target_object[env] = ""
        self._pointnav_policy[env].reset()
        self._last_goal[env] = np.zeros(2)
        self._num_steps[env] = 0

        self._done_initializing[env] = False
        self._called_stop[env] = False
        self._did_reset[env] = True
        self._acyclic_enforcer[env] = AcyclicEnforcer()
        self._last_value[env] = float("-inf")
        self._last_frontier[env] = np.zeros(2)

        self._cur_floor_index[env] = 0
        self._object_map[env] = self._object_map_list[env][0]
        self._value_map[env] = self._value_map_list[env][0]
        self._object_map[env].reset()
        self._value_map[env].reset()
        del self._object_map_list[env][1:]  
        del self._value_map_list[env][1:]

        if self._compute_frontiers:
            self._obstacle_map[env] = self._obstacle_map_list[env][0]
            self._obstacle_map[env].reset()
            del self._obstacle_map_list[env][1:]

        ## floor num
        self.floor_num[env] = len(self._obstacle_map_list[env])

        self._initialize_step[env] = 0
        self._try_to_navigate_step[env] = 0
        # 防止之前episode爬楼梯异常退出
        self._climb_stair_over[env] = True
        self._reach_stair[env] = False
        self._reach_stair_centroid[env] = False
        self._stair_dilate_flag[env] = False
        self._pitch_angle[env] = 0
        self._climb_stair_flag[env] = 0 

        # 防止识别正确之后造成误识别
        self.target_might_detected[env] = False
        self._frontier_stick_step[env] = 0
        self._last_frontier_distance[env] = 0

        self._stair_masks = []

        self._last_carrot_xy[env] = []
        self._last_carrot_px[env] = []
        self._carrot_goal_xy[env] = []
        self._temp_stair_map[env] = []
        self.history_action[env] = []

        self._try_to_navigate[env] = False

        # 防止来回走动
        self._force_frontier[env] = np.zeros(2)

        # self._might_close_to_goal[env] = False
        self.all_detection_list[env] = None
        self.target_detection_list[env] = None
        self.coco_detection_list[env] = None
        self.non_coco_detection_list[env] = None
        self.min_distance_xy[env] = np.inf
        self.cur_dis_to_goal[env] = np.inf
        self.cur_frontier[env] = np.array([])

        # vlm
        self.frontier_rgb_list[env] = []
        self.vlm_response[env] = ""
        self.frontier_step_list[env] = []

        self._blip_cosine[env] = 0
        self.multi_floor_ask_step[env] = 0

        self._get_close_to_stair_step[env] = 0

        self.semantic_dict[env] = {
                "scene_id": None,
                "target_ids": None,
                "target_names": None,
            } 
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
            tf_camera_to_episodic = xyz_yaw_pitch_roll_to_tf_matrix(camera_position, camera_yaw, camera_pitch, camera_roll)

        self._observations_cache[env] = {
            "robot_xy": robot_xy,
            "robot_heading": camera_yaw,
            "tf_camera_to_episodic": tf_camera_to_episodic,
            "camera_fov": self._camera_fov,
            "map_rgbd": [
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
            "habitat_start_yaw": observations["heading"][env].item(),
            "semantic": observations["semantic"][env].cpu().numpy(),
        }

        ## add for rednet
        self._observations_cache[env]["nav_rgb"]=torch.unsqueeze(observations["rgb"][env], dim=0)
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
        }

        # 若不需要可视化,直接返回
        if not self._visualize:
            return policy_info

        # 处理注释深度图和 RGB 图
        annotated_depth = self._observations_cache[env]["map_rgbd"][0][1] * 255
        annotated_depth = cv2.cvtColor(annotated_depth.astype(np.uint8), cv2.COLOR_GRAY2RGB)

        # 定义所有需要处理的掩膜和对应的标注帧
        annotated_rgb = self._observations_cache[env]["map_rgbd"][0][0]
        masks_info = [
            (self._object_masks[env], (255, 0, 0)), # object: 红色
            (self._stair_masks[env], (0, 0, 255)) # stair: 蓝色
        ]

        for mask, color in masks_info:
            if mask.sum() == 0:
                continue
            
            # 统一转换为uint8类型（即使原类型已经是uint8也不影响）
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), 
                cv2.RETR_TREE, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # 绘制到RGB和深度图
            annotated_rgb = cv2.drawContours(annotated_rgb, contours, -1, color, 2)
            annotated_depth = cv2.drawContours(annotated_depth, contours, -1, color, 2)

        policy_info["annotated_rgb"] = annotated_rgb
        policy_info["annotated_depth"] = annotated_depth

        # 添加第三视角 RGB
        if "third_rgb" in self._observations_cache[env]:
            policy_info["third_rgb"] = self._observations_cache[env]["third_rgb"]

        # 绘制 frontiers
        if self._compute_frontiers:
            policy_info["obstacle_map"] = cv2.cvtColor(self._obstacle_map[env].visualize(), cv2.COLOR_BGR2RGB)

        policy_info["vlm_input"] = self.frontier_rgb_list[env]
        policy_info["vlm_response"] = self.vlm_response[env]
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

    def _update_obstacle_map(self, observations: "TensorDict") -> None:
        STAIR_UP = 1
        STAIR_DOWN = 2
        STAIR_CONFIG = {
            STAIR_UP: ("_up_stair_map", "_up_stair_start", "_up_stair_end", 1),
            STAIR_DOWN: ("_down_stair_map", "_down_stair_start", "_down_stair_end", -1)
        }

        for env in range(self._num_envs):
            # 处理非边界计算情况
            if not self._compute_frontiers:
                self._observations_cache[env]["frontier_sensor"] = (
                    observations["frontier_sensor"][env].cpu().numpy()
                    if "frontier_sensor" in observations 
                    else np.array([])
                )
                continue

            # 主处理逻辑
            if not self._climb_stair_over[env] and self._climb_stair_flag[env] in (STAIR_UP, STAIR_DOWN):
                stair_flag = self._climb_stair_flag[env]
                map_attr, start_attr, end_attr, floor_dir = STAIR_CONFIG[stair_flag]
                
                # 楼梯地图处理
                self._temp_stair_map[env] = getattr(self._obstacle_map[env], map_attr)
                if not self._stair_dilate_flag[env]:
                    self._temp_stair_map[env] = cv2.dilate(
                        self._temp_stair_map[env].astype(np.uint8), 
                        (7, 7), iterations=1
                    )
                    self._stair_dilate_flag[env] = True

                # 获取机器人坐标
                robot_xy = self._observations_cache[env]["robot_xy"]
                robot_px = self._obstacle_map[env]._xy_to_px(np.atleast_2d(robot_xy))

                # 楼梯到达检测
                if not self._reach_stair[env]:
                    reached, _ = self.is_robot_in_stair_map_fast(env, robot_px, self._temp_stair_map[env])
                    if reached:
                        self._reach_stair[env] = True
                        self._get_close_to_stair_step[env] = 0
                        setattr(self._obstacle_map[env], start_attr, robot_px[0].copy())

                # 质心到达检测
                if self._reach_stair[env] and not self._reach_stair_centroid[env]:
                    if self._stair_frontier[env] is not None and np.linalg.norm(
                        self._stair_frontier[env] - robot_xy
                    ) <= 0.3:
                        self._reach_stair_centroid[env] = True

                # 完成楼梯处理
                if self._reach_stair_centroid[env]:
                    if not self.is_robot_in_stair_map_fast(env, robot_px, self._temp_stair_map[env])[0]:
                        if self._obstacle_map[env]._climb_stair_paused_step >= 30:
                            # 禁用当前楼梯
                            getattr(self._obstacle_map[env], map_attr).fill(0)
                            setattr(self._obstacle_map[env], f"_has_{'up' if stair_flag == STAIR_UP else 'down'}_stair", False)
                            
                            # 更新楼层索引
                            target_floor = self._cur_floor_index[env] + floor_dir
                            if 0 <= target_floor < len(self._obstacle_map_list[env]):
                                self._cur_floor_index[env] = target_floor
                                self._object_map[env] = self._object_map_list[env][self._cur_floor_index[env]]
                                self._obstacle_map[env] = self._obstacle_map_list[env][self._cur_floor_index[env]]
                                self._value_map[env] = self._value_map_list[env][self._cur_floor_index[env]]

                            # 重置状态
                            self._climb_stair_flag[env] = 0
                            self._obstacle_map[env]._climb_stair_paused_step = 0
                            self._last_carrot_xy[env] = []
                            self._last_carrot_px[env] = []
                            self._reach_stair[env] = False
                            self._reach_stair_centroid[env] = False
                            self._stair_dilate_flag[env] = False
                            self._climb_stair_over[env] = True
                            print(f"Frontier {self._stair_frontier[env]} disabled")
                        else:
                            self._climb_stair_over[env] = True
                            setattr(self._obstacle_map[env], end_attr, robot_px[0].copy())
                            print("climb stair success!!!!")

            # 关键更新点：保持原始执行顺序
            frontiers = self._obstacle_map[env].frontiers  # 获取边界
            self._obstacle_map[env].update_agent_traj(     # 更新轨迹
                self._observations_cache[env]["robot_xy"],
                self._observations_cache[env]["robot_heading"]
            )
            self._observations_cache[env]["frontier_sensor"] = frontiers  # 存储边界

            # 更新障碍物地图
            self._obstacle_map[env].update_map_with_stair(
                self._observations_cache[env]["map_rgbd"][0][1],
                self._observations_cache[env]["map_rgbd"][0][2],
                self._min_depth, self._max_depth, self._fx, self._fy,
                self._camera_fov,  self._stair_masks[env],
                self.red_semantic_pred_list[env], self._pitch_angle[env],
                self._climb_stair_over[env], self._reach_stair[env],
                self._climb_stair_flag[env]
            )

            # 楼层地图管理
            if self._obstacle_map[env]._has_up_stair and self._cur_floor_index[env] + 1 >= len(self._obstacle_map_list[env]):
                self._object_map_list[env].append(ObjectMapUpdater(erosion_size=self._object_map_erosion_size, size=1600))
                self._obstacle_map_list[env].append(ObstacleMapUpdater(
                    min_height=self.min_obstacle_height,
                    max_height=self.max_obstacle_height,
                    area_thresh=self.obstacle_map_area_threshold,
                    agent_radius=self.agent_radius,
                    hole_area_thresh=self.hole_area_thresh,
                    size=1600
                ))
                self._value_map_list[env].append(ValueMapUpdater(
                    value_channels=len(self._text_prompt.split(PROMPT_SEPARATOR)),
                    use_max_confidence=self.use_max_confidence,
                    obstacle_map=None,
                    size=1600
                ))

            if self._obstacle_map[env]._has_down_stair and self._cur_floor_index[env] == 0:
                self._object_map_list[env].insert(0, ObjectMapUpdater(erosion_size=self._object_map_erosion_size, size=1600))
                self._obstacle_map_list[env].insert(0, ObstacleMapUpdater(
                    min_height=self.min_obstacle_height,
                    max_height=self.max_obstacle_height,
                    area_thresh=self.obstacle_map_area_threshold,
                    agent_radius=self.agent_radius,
                    hole_area_thresh=self.hole_area_thresh,
                    size=1600
                ))
                self._value_map_list[env].insert(0, ValueMapUpdater(
                    value_channels=len(self._text_prompt.split(PROMPT_SEPARATOR)),
                    use_max_confidence=self.use_max_confidence,
                    obstacle_map=None,
                    size=1600
                ))
                self._cur_floor_index[env] += 1

            # 最终更新
            self.floor_num[env] = len(self._obstacle_map_list[env])
            self._obstacle_map[env].project_frontiers_to_rgb_hush(
                self._observations_cache[env]["map_rgbd"][0][0]
            )
            
    def _update_stair_map(self, env: int) -> None:
        if self._climb_stair_flag[env] == 1:
            self._temp_stair_map[env] = self._obstacle_map[env]._up_stair_map
        elif self._climb_stair_flag[env] == 2:
            self._temp_stair_map[env] = self._obstacle_map[env]._down_stair_map

        if not self._stair_dilate_flag[env]:
            self._temp_stair_map[env] = cv2.dilate(
                self._temp_stair_map[env].astype(np.uint8),
                (7, 7),
                iterations=1,
            )
            self._stair_dilate_flag[env] = True

    def _reset_stair_state(self, env: int) -> None:
        self._obstacle_map[env]._climb_stair_paused_step = 0
        self._last_carrot_xy[env] = []
        self._last_carrot_px[env] = []
        self._reach_stair[env] = False
        self._reach_stair_centroid[env] = False
        self._stair_dilate_flag[env] = False
        self._climb_stair_over[env] = True
        self._obstacle_map[env]._disabled_frontiers.add(tuple(self._stair_frontier[env][0]))
        print(f"Frontier {self._stair_frontier[env]} is disabled due to no movement.")

        if self._climb_stair_flag[env] == 1:
            self._disable_up_stair(env)
        elif self._climb_stair_flag[env] == 2:
            self._disable_down_stair(env)

        self._climb_stair_flag[env] = 0

    def _disable_up_stair(self, env: int) -> None:
        self._obstacle_map[env]._disabled_stair_map[self._obstacle_map[env]._up_stair_map == 1] = 1
        self._obstacle_map[env]._up_stair_map.fill(0)
        self._obstacle_map[env]._has_up_stair = False
        self._remove_floor_from_map_list(env, self._cur_floor_index[env] + 1)
        self.floor_num[env] -= 1

    def _disable_down_stair(self, env: int) -> None:
        self._obstacle_map[env]._disabled_stair_map[self._obstacle_map[env]._down_stair_map == 1] = 1
        self._obstacle_map[env]._down_stair_frontiers.fill(0)
        self._obstacle_map[env]._has_down_stair = False
        self._obstacle_map[env]._look_for_downstair_flag = False
        self._remove_floor_from_map_list(env, self._cur_floor_index[env] - 1)
        self.floor_num[env] -= 1
        self._cur_floor_index[env] -= 1

    def _remove_floor_from_map_list(self, env: int, floor_index: int) -> None:
        del self._object_map_list[env][floor_index]
        del self._value_map_list[env][floor_index]
        del self._obstacle_map_list[env][floor_index]

    def _update_stair_endpoints(self, env: int, robot_px: np.ndarray) -> None:
        if self._climb_stair_flag[env] == 1:
            self._update_up_stair_endpoints(env, robot_px)
        elif self._climb_stair_flag[env] == 2:
            self._update_down_stair_endpoints(env, robot_px)

    def _update_up_stair_endpoints(self, env: int, robot_px: np.ndarray) -> None:
        self._obstacle_map[env]._up_stair_end = robot_px[0].copy()
        if not self._obstacle_map_list[env][self._cur_floor_index[env] + 1]._done_initializing:
            self._initialize_new_floor(env, self._cur_floor_index[env] + 1, "up")
        else:
            self._cur_floor_index[env] += 1
            self._update_current_floor_maps(env)

    def _update_down_stair_endpoints(self, env: int, robot_px: np.ndarray) -> None:
        self._obstacle_map[env]._down_stair_end = robot_px[0].copy()
        if not self._obstacle_map_list[env][self._cur_floor_index[env] - 1]._done_initializing:
            self._initialize_new_floor(env, self._cur_floor_index[env] - 1, "down")
        else:
            self._cur_floor_index[env] -= 1
            self._update_current_floor_maps(env)

    def _initialize_new_floor(self, env: int, floor_index: int, stair_type: str) -> None:
        self._done_initializing[env] = False
        self._initialize_step[env] = 0
        if stair_type == "up":
            self._obstacle_map[env]._explored_up_stair = True
            self._obstacle_map_list[env][floor_index]._explored_down_stair = True
        else:
            self._obstacle_map[env]._explored_down_stair = True
            self._obstacle_map_list[env][floor_index]._explored_up_stair = True

        self._cur_floor_index[env] = floor_index
        self._update_current_floor_maps(env)

    def _update_current_floor_maps(self, env: int) -> None:
        self._object_map[env] = self._object_map_list[env][self._cur_floor_index[env]]
        self._obstacle_map[env] = self._obstacle_map_list[env][self._cur_floor_index[env]]
        self._value_map[env] = self._value_map_list[env][self._cur_floor_index[env]]


    def _add_new_floor_maps(self, env: int, stair_type: str) -> None:
        if stair_type == "up":
            self._object_map_list[env].append(ObjectMapUpdater(erosion_size=self._object_map_erosion_size, size=1600))
            self._obstacle_map_list[env].append(ObstacleMapUpdater(
                min_height=self.min_obstacle_height,
                max_height=self.max_obstacle_height,
                area_thresh=self.obstacle_map_area_threshold,
                agent_radius=self.agent_radius,
                hole_area_thresh=self.hole_area_thresh,
                size=1600,
            ))
            self._value_map_list[env].append(ValueMapUpdater(
                value_channels=len(self._text_prompt.split(PROMPT_SEPARATOR)),
                use_max_confidence=self.use_max_confidence,
                obstacle_map=None,
                size=1600,
            ))
        else:
            self._object_map_list[env].insert(0, ObjectMapUpdater(erosion_size=self._object_map_erosion_size, size=1600))
            self._obstacle_map_list[env].insert(0, ObstacleMapUpdater(
                min_height=self.min_obstacle_height,
                max_height=self.max_obstacle_height,
                area_thresh=self.obstacle_map_area_threshold,
                agent_radius=self.agent_radius,
                hole_area_thresh=self.hole_area_thresh,
                size=1600,
            ))
            self._value_map_list[env].insert(0, ValueMapUpdater(
                value_channels=len(self._text_prompt.split(PROMPT_SEPARATOR)),
                use_max_confidence=self.use_max_confidence,
                obstacle_map=None,
                size=1600,
            ))
            self._cur_floor_index[env] += 1
        
    def _update_value_map(self) -> None:
        # 收集所有环境的 RGB 图像和文本
        all_rgb = []
        all_texts = []
        
        for env in range(self._num_envs):
            # 提取当前环境的 RGB 图像
            rgb = self._observations_cache[env]["map_rgbd"][0][0]
            all_rgb.append(rgb)
            
            # 准备当前环境的文本
            text = self._text_prompt.replace("target_object", self._target_object[env].replace("|", "/"))
            all_texts.append(text)
        
        # 批量计算所有环境的余弦相似度
        all_cosines = self._itm.cosine(all_rgb, all_texts)
        
        # 更新每个环境的 value map 和保存结果
        for env in range(self._num_envs):
            # 更新 value map
            self._value_map[env].update_map(
                np.array([all_cosines[env]]),  # 将余弦相似度包装为数组
                self._observations_cache[env]["map_rgbd"][0][1],
                self._observations_cache[env]["map_rgbd"][0][2],
                self._observations_cache[env]["map_rgbd"][0][3],
                self._observations_cache[env]["map_rgbd"][0][4],
                self._observations_cache[env]["camera_fov"],
            )
            
            # 更新 agent 轨迹
            self._value_map[env].update_agent_traj(
                self._observations_cache[env]["robot_xy"],
                self._observations_cache[env]["robot_heading"],
            )
            
            # 保存 BLIP 余弦相似度
            self._blip_cosine[env] = all_cosines[env]

    def _update_distance_on_object_map(self) -> None:
        for env in range(self._num_envs):
            self._object_map[env].update_agent_traj(
                self._observations_cache[env]["robot_xy"],
                self._observations_cache[env]["robot_heading"],
            )
            if np.argwhere(self._object_map[env]._map).size > 0 and self._target_object[env] in self._object_map[env].clouds and self._object_map[env].clouds[self._target_object[env]].shape[0] > 0:
                curr_position = self._observations_cache[env]["tf_camera_to_episodic"][:3, 3]
                closest_point = self._object_map[env]._get_closest_point(self._object_map[env].clouds[self._target_object[env]], curr_position)
                self.cur_dis_to_goal[env] = np.linalg.norm(closest_point[:2] - curr_position[:2])

    def _get_target_object_location(self, position: np.ndarray, env: int = 0) -> Union[None, np.ndarray]:
        if self._object_map[env].has_object(self._target_object[env]):
            return self._object_map[env].get_best_object(self._target_object[env], position)
        else:
            return None

    def extract_scene_with_target(self, scene_id: str, target_object: str) -> Dict:
        if target_object == "":
            return {
                "scene_id": None,
                "target_ids": None,
                "target_names": None,
            }
        # 构造语义标注文件的路径
        semantic_file_path = self._get_semantic_file_path(scene_id)

        # 读取语义标注文件并提取目标物体的 ID
        target_ids, target_names = self._parse_semantic_file(semantic_file_path, target_object)

        # 返回包含场景 ID 和目标物体 ID 的字典
        return {
            "scene_id": scene_id,
            "target_ids": target_ids,
            "target_names": target_names,
        }

    def _get_semantic_file_path(self, scene_id: str) -> str:
        # 从 scene_id 构造语义标注文件的路径
        scene_dir = os.path.dirname(scene_id)
        scene_name = os.path.basename(scene_id).replace(".basis.glb", "")
        semantic_file_path = os.path.join(scene_dir, f"{scene_name}.semantic.txt")
        return semantic_file_path

    def _parse_semantic_file(self, semantic_file_path: str, target_object: str) -> List[int]:
        target_ids = []
        target_names = set()
        with open(semantic_file_path, "r") as f:
            for line in f:
                # 跳过注释行
                if line.startswith("#"):
                    continue
                
                # 解析每一行
                parts = line.strip().split(",")
                if len(parts) >= 3:
                    id = int(parts[0])
                    object_name = parts[2].strip('"')
                    # 检查目标物体名称是否在映射表中
                    if target_object in self.object_mapping:
                        mapped_object_name = self.object_mapping[target_object]
                    else:
                        mapped_object_name = target_object
                    
                    if mapped_object_name.strip().lower() in object_name.strip().lower():
                        target_ids.append(id)
                        target_names.add(target_object)
        return target_ids, target_names

    def _pre_step(self, observations: "TensorDict", masks: Tensor) -> None:
        # assert masks.shape[1] == 1, "Currently only supporting one env at a time"
        self._policy_info = []
        self._num_envs = masks.shape[0]
        for env in range(self._num_envs):
            # if not self._did_reset[env] and masks[env] == 0:
            try:
                if not self._did_reset[env] and masks[env] == 0:
                    self._reset(env)
                    self._target_object[env] = observations["objectgoal"][env]
            except IndexError as e:
                print(f"Caught an IndexError: {e}")
                print(f"self._did_reset: {self._did_reset}")
                print(f"masks: {masks}")
                raise StopIteration
            try:
                self._cache_observations(observations, env)
            except IndexError as e:
                print(e)
                print("Reached edge of map, stopping.")
                raise StopIteration
            self._policy_info.append({})
            
    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if "current_episodes_info" in kwargs: 
            for env in range(self._num_envs):
                target_names = self.semantic_dict[env]["target_names"]
                if target_names:
                    target_name = next(iter(target_names))
                    if self.semantic_dict[env]["scene_id"] == kwargs["current_episodes_info"][env].scene_id and target_name == self._target_object[env]:
                        pass
                    elif self._target_object[env] == "":
                        self.semantic_dict[env] = {
                                "scene_id": None,
                                "target_ids": None,
                                "target_names": None,
                            }
                else:
                    self.semantic_dict[env] = self.extract_scene_with_target(kwargs["current_episodes_info"][env].scene_id, self._target_object[env])
                
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
        self._update_object_map_with_stair(img_height, img_width)

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

        self._update_obstacle_map(observations)
        self._update_value_map()
        self._update_distance_on_object_map()
        # self._visualize_object_map()
        # self._pre_step(observations, masks) # duplicated one, consider to get rid of this
        
        pointnav_action_env_list = []

        for env in range(self._num_envs):

            robot_xy = self._observations_cache[env]["robot_xy"]
            goal = self._get_target_object_location(robot_xy, env) #  self._get_target_object_location_with_seg(robot_xy, red_semantic_pred_list, env, ) #
            robot_xy_2d = np.atleast_2d(robot_xy) 
            robot_px = self._obstacle_map[env]._xy_to_px(robot_xy_2d)
            x, y = robot_px[0, 0], robot_px[0, 1]
            # 不知不觉到了下楼的楼梯,且不是刚刚上楼的
            if self._climb_stair_over[env] == True and self._obstacle_map[env]._down_stair_map[y,x] == 1 and len(self._obstacle_map[env]._down_stair_frontiers) > 0 and self._obstacle_map_list[env][self._cur_floor_index[env] - 1]._explored_up_stair == False:
                self._reach_stair[env] = True
                self._get_close_to_stair_step[env] = 0
                self._climb_stair_over[env] = False
                self._climb_stair_flag[env] = 2
                self._obstacle_map[env]._down_stair_start = robot_px[0].copy()
                # self._reach_stair_centroid[env] = True
            # 不知不觉到了上楼的楼梯,且不是刚刚下楼的
            elif self._climb_stair_over[env] == True and self._obstacle_map[env]._up_stair_map[y,x] == 1 and len(self._obstacle_map[env]._up_stair_frontiers) > 0 and self._obstacle_map_list[env][self._cur_floor_index[env] + 1]._explored_down_stair == False:
                self._reach_stair[env] = True
                self._get_close_to_stair_step[env] = 0
                self._climb_stair_over[env] = False
                self._climb_stair_flag[env] = 1
                self._obstacle_map[env]._up_stair_start = robot_px[0].copy()
            if self._climb_stair_over[env] == False:
                if self._reach_stair[env] == True:

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
                        
                            if self._pitch_angle[env] > 0: 
                                self._pitch_angle[env] -= self._pitch_angle_offset
                                pointnav_action = TorchActionIDs_plook.LOOK_DOWN.to(masks.device)
                            elif self._pitch_angle[env] < 0:
                                self._pitch_angle[env] += self._pitch_angle_offset
                                pointnav_action = TorchActionIDs_plook.LOOK_UP.to(masks.device)
                            else:  # Initialize for 12 steps
                                self._obstacle_map[env]._done_initializing = False # add, for the initial floor and new floor
                                self._initialize_step[env] = 0
                                pointnav_action = self._initialize(env,masks)
                            self._obstacle_map[env]._climb_stair_paused_step = 0
                            self._climb_stair_over[env] = True
                            self._climb_stair_flag[env] = 0
                            self._reach_stair[env] = False
                            self._reach_stair_centroid[env] = False
                            self._stair_dilate_flag[env] = False
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
            else:
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
                    pointnav_action = self._navigate(observations, goal[:2], stop=True, env=env, ori_masks=masks)

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
                    pointnav_action = torch.tensor([[action_numpy]], dtype=torch.int64, device=masks.device)
                    print("Continuous turns to force forward.")
                if all(action in [1] for action in self.history_action[env]):
                    # 强制执行动作3
                    action_numpy = 3
                    pointnav_action = torch.tensor([[action_numpy]], dtype=torch.int64, device=masks.device)
                    print("Continuous turns to force turn right.")
            
            if self._num_steps[env] == 500 - 1:
                # 强制执行动作0
                action_numpy = 0
                pointnav_action = torch.tensor([[action_numpy]], dtype=torch.int64, device=masks.device)
                print("Force stop.")
            self.history_action[env].append(action_numpy)
            pointnav_action_env_list.append(pointnav_action)
            
            print(f"Env: {env} | Step: {self._num_steps[env]} | Floor_step: {self._obstacle_map[env]._floor_num_steps} | Mode: {mode} | Stair_flag: {self._climb_stair_flag[env]} | Action: {action_numpy}")
            if self._climb_stair_over[env] == False:
                print(f"Reach_stair_centroid: {self._reach_stair_centroid[env]}")
                # print(f"Stair_pixedl: {np.sum(self.red_semantic_pred_list[env] == STAIR_CLASS_ID)}")
            self._num_steps[env] += 1
            self._obstacle_map[env]._floor_num_steps += 1
            self._policy_info[env].update(self._get_policy_info(self.all_detection_list[env],env)) # self.target_detection_list[env]

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
        if self._initialize_step[env] > 11: # 11, 12 step is for the first step in this floor
            self._done_initializing[env] = True
            self._obstacle_map[env]._tight_search_thresh = False 
        else:
            self._initialize_step[env] += 1 
        return TorchActionIDs_plook.TURN_LEFT.to(masks.device)

    def _explore(self, observations: Union[Dict[str, Tensor], "TensorDict"], env: int, masks: Tensor) -> Tensor:

        # self.vlm_analyze(env)

        initial_frontiers = self._observations_cache[env]["frontier_sensor"]
        # self._all_frontiers[env] =  
        frontiers = [
            frontier for frontier in initial_frontiers if tuple(frontier) not in self._obstacle_map[env]._disabled_frontiers
        ]

        # 需要修改逻辑了，比较久一次才更新frontier
        # 如果目前就有frontier在探索，那么继续探索，中间记录是否有太久探索或者停滞不前的情况

        temp_flag = False
        if np.array_equal(frontiers, np.zeros((1, 2))) or len(frontiers) == 0: # no frontier in this floor, check if there is stair
            # 如果在楼梯间，且楼上楼下任一没去过（第一次去楼梯间的话肯定没上过或者没下过楼），立刻初始化（25.2.24）
            # 防止楼梯间状态
            if self._obstacle_map[env]._reinitialize_flag == False and self._obstacle_map[env]._floor_num_steps < 50 and (self._obstacle_map[env]._explored_up_stair == False or self._obstacle_map[env]._explored_down_stair == False): 
                self._object_map[env].reset()
                self._value_map[env].reset()

                if self._compute_frontiers:
                    temp_has_up_stair = self._obstacle_map[env]._has_up_stair
                    # 如果原来有上楼的楼梯，保留
                    if temp_has_up_stair:
                        temp_up_stair_map = self._obstacle_map[env]._up_stair_map.copy()
                        temp_up_stair_start = self._obstacle_map[env]._up_stair_start.copy()
                        temp_up_stair_end = self._obstacle_map[env]._up_stair_end.copy()
                        temp_up_stair_frontiers = self._obstacle_map[env]._up_stair_frontiers.copy()
                        temp_explored_up_stair = self._obstacle_map[env]._explored_up_stair # .copy()
                    
                    temp_has_down_stair = self._obstacle_map[env]._has_down_stair
                    # 如果原来有下楼的楼梯，保留
                    if temp_has_down_stair:
                        temp_down_stair_map = self._obstacle_map[env]._down_stair_map.copy()
                        temp_down_stair_start = self._obstacle_map[env]._down_stair_start.copy()
                        temp_down_stair_end = self._obstacle_map[env]._down_stair_end.copy()
                        temp_down_stair_frontiers = self._obstacle_map[env]._down_stair_frontiers.copy()
                        temp_explored_down_stair = self._obstacle_map[env]._explored_down_stair # .copy()
                                                
                    self._obstacle_map[env].reset()

                    if temp_has_up_stair:
                        self._obstacle_map[env]._has_up_stair = temp_has_up_stair # .copy()
                        self._obstacle_map[env]._up_stair_map = temp_up_stair_map.copy()
                        self._obstacle_map[env]._up_stair_start = temp_up_stair_start.copy()
                        self._obstacle_map[env]._up_stair_end = temp_up_stair_end.copy()
                        self._obstacle_map[env]._up_stair_frontiers = temp_up_stair_frontiers.copy() 
                        self._obstacle_map[env]._explored_up_stair = temp_explored_up_stair # .copy()
                    if temp_has_down_stair:
                        self._obstacle_map[env]._has_down_stair = temp_has_down_stair # .copy()
                        self._obstacle_map[env]._down_stair_map = temp_down_stair_map.copy()
                        self._obstacle_map[env]._down_stair_start = temp_down_stair_start.copy()
                        self._obstacle_map[env]._down_stair_end = temp_down_stair_end.copy()
                        self._obstacle_map[env]._down_stair_frontiers = temp_down_stair_frontiers.copy() 
                        self._obstacle_map[env]._explored_down_stair = temp_explored_down_stair # .copy()

                    self._obstacle_map[env]._reinitialize_flag = True

                self._obstacle_map[env]._tight_search_thresh = True # 预防楼梯间
                self._climb_stair_over[env] = True
                self._reach_stair[env] = False
                self._reach_stair_centroid[env] = False
                self._stair_dilate_flag[env] = False
                self._pitch_angle[env] = 0
                self._done_initializing[env] = False
                self._initialize_step[env] = 0
                pointnav_action = self._initialize(env,masks)
                return pointnav_action
            
            self._obstacle_map[env]._this_floor_explored = True # 标记

            # 如果在楼梯间，不是立刻初始化，而是优先去找有没有去过的楼层
            # 目前该逻辑是探索完一层,就去探索其他层,同时上楼优先。这个逻辑后面可能改掉
            # (2025.02.24)改成优先探索没去过的楼层（没爬过的楼梯）
            # 判断是否有未探索的楼梯
            has_unexplored_up_stair = (self._obstacle_map[env]._has_up_stair and
                                    not self._obstacle_map[env]._explored_up_stair)
            has_unexplored_down_stair = (self._obstacle_map[env]._has_down_stair and
                                        not self._obstacle_map[env]._explored_down_stair)

            # 优先探索未探索的楼梯
            if has_unexplored_up_stair or has_unexplored_down_stair:
                stair_direction = 1 if has_unexplored_up_stair else 2
                stair_frontiers = (self._obstacle_map[env]._up_stair_frontiers if has_unexplored_up_stair
                                else self._obstacle_map[env]._down_stair_frontiers)
                self._climb_stair_over[env] = False
                self._climb_stair_flag[env] = stair_direction
                self._stair_frontier[env] = stair_frontiers

                # 检查更高或更低的楼层是否有未探索的区域
                floors_to_check = range(self._cur_floor_index[env] + 1, len(self._object_map_list[env])) if has_unexplored_up_stair else \
                                range(self._cur_floor_index[env] - 1, -1, -1)

                for ith_floor in floors_to_check:
                    if not self._obstacle_map_list[env][ith_floor]._this_floor_explored:
                        pointnav_action = self._pointnav(observations, self._stair_frontier[env][0], stop=False, env=env)
                        return pointnav_action

                # 如果所有楼梯方向的楼层都已探索完毕
                print("In all floors, no frontiers found during exploration, stopping.")
                return self._stop_action.to(masks.device)

            # 如果所有情况都不满足，停止探索
            else:
                print("No frontiers found during exploration, stopping.")
                return self._stop_action.to(masks.device)
        else:
            best_frontier, best_value = self._get_best_frontier_with_llm(observations, frontiers, env, use_multi_floor=True) 
            if best_value == -100: # llm判断上楼，加入保底机制
                if self._obstacle_map[env]._has_up_stair:
                    # 有上楼的楼梯,检查更高的楼层,直到楼上有没探索过的
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
                    else: # 上不了楼，或者楼上都去过了，直接本楼层探索。再靠后面的保底机制探索楼下
                        print("Can't go upstairs or have already fully explored upstairs, explore directly on this floor.")
                        best_frontier, best_value = self._get_best_frontier_with_llm(observations, frontiers, env, use_multi_floor=False)
            
            elif best_value == -200: # llm判断下楼，加入保底机制
                if self._obstacle_map[env]._has_down_stair:
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
                        print("Can't go downstairs or have already fully explored downstairs, explore directly on this floor.")
                        best_frontier, best_value = self._get_best_frontier_with_llm(observations, frontiers, env, use_multi_floor=False)
            
            self.cur_frontier[env] = best_frontier
            pointnav_action = self._pointnav(observations, self.cur_frontier[env], stop=False, env=env, stop_radius=self._pointnav_stop_radius)
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
                    self._called_stop[env] = True
                    return self._stop_action.to(ori_masks.device)
        action = self._pointnav_policy[env].act(obs_pointnav, masks, deterministic=True)
        return action

    def _navigate(self, observations: "TensorDict", goal: np.ndarray, stop: bool = False, env: int = 0, ori_masks: Tensor = None, stop_radius: float = 0.9) -> Tensor:
        """
        Calculates rho and theta from the robot's current position to the goal using the
        gps and heading sensors within the observations and the given goal, then uses
        it to determine the next action to take using the pre-trained pointnav policy.

        Args:
            goal (np.ndarray): The goal to navigate to as (x, y), where x and y are in
                meters.
            stop (bool): Whether to stop if we are close enough to the goal.

        """
        self._try_to_navigate_step[env] += 1
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
        print(f"Distance to goal: {self.cur_dis_to_goal[env]}")
        if self.cur_dis_to_goal[env] < 1.0: # close to the goal, but might be some noise, so get close as possible 
            if self.cur_dis_to_goal[env] <= 0.6 or np.abs(self.cur_dis_to_goal[env] - self.min_distance_xy[env]) < 0.1: # close enough or cannot move forward more #  or self._num_steps[env] == (500 - 1)
                self._called_stop[env] = True
                return self._stop_action.to(ori_masks.device)
            else:
                self.min_distance_xy[env] = self.cur_dis_to_goal[env].copy()
                return TorchActionIDs_plook.MOVE_FORWARD.to(ori_masks.device) # force to move forward
        else:        
            action = self._pointnav_policy[env].act(obs_pointnav, masks, deterministic=True)
        if self._try_to_navigate_step[env] >= 200:
            print("Might false positive, change to look for the true goal.")
            self._object_map[env].clouds = {}
            self._try_to_navigate[env] = False
            self._try_to_navigate_step[env] = 0
            self._object_map[env]._disabled_object_map[self._object_map[env]._map == 1] = 1
            self._object_map[env]._map.fill(0)
            action = self._explore(observations, env, ori_masks) # 果断换成探索
            return action
        return action

    def _get_close_to_stair(self, observations: "TensorDict", env: int, ori_masks: Tensor) -> Tensor:
        masks = torch.tensor([self._num_steps[env] != 0], dtype=torch.bool, device="cuda")
        robot_xy = self._observations_cache[env]["robot_xy"]
        heading = self._observations_cache[env]["robot_heading"]

        # 更新楼梯前沿
        if self._climb_stair_flag[env] == 1:
            self._stair_frontier[env] = self._obstacle_map[env]._up_stair_frontiers
        elif self._climb_stair_flag[env] == 2:
            self._stair_frontier[env] = self._obstacle_map[env]._down_stair_frontiers

        # 检查是否卡住或需要禁用前沿
        if np.array_equal(self._last_frontier[env], self._stair_frontier[env][0]):
            if self._frontier_stick_step[env] == 0:
                self._last_frontier_distance[env] = np.linalg.norm(self._stair_frontier[env] - robot_xy)
                self._frontier_stick_step[env] += 1
                self._get_close_to_stair_step[env] += 1
            else:
                current_distance = np.linalg.norm(self._stair_frontier[env] - robot_xy)
                print(f"Distance Change: {np.abs(self._last_frontier_distance[env] - current_distance)} and Stick Step {self._frontier_stick_step[env]}, and Get Close Step {self._get_close_to_stair_step[env]}")

                if np.abs(self._last_frontier_distance[env] - current_distance) > 0.3:
                    self._frontier_stick_step[env] = 0
                    self._last_frontier_distance[env] = current_distance
                else:
                    if self._frontier_stick_step[env] >= 30 or self._get_close_to_stair_step[env] >= 60:
                        self._disable_stair_frontier(env)
                        return self._explore(observations, env, ori_masks)
                    else:
                        self._frontier_stick_step[env] += 1
                        self._get_close_to_stair_step[env] += 1
        else:
            self._frontier_stick_step[env] = 0
            self._last_frontier_distance[env] = 0
            self._get_close_to_stair_step[env] = 0

        self._last_frontier[env] = self._stair_frontier[env][0]

        # 使用点导航模型计算动作
        rho, theta = rho_theta(robot_xy, heading, self._stair_frontier[env][0])
        action = self._get_pointnav_action(env, rho, theta, masks)

        # 如果动作是停止，则禁用前沿并切换到探索模式
        if action.item() == 0:
            self._disable_stair_frontier(env)
            return self._explore(observations, env, ori_masks)

        return action

    def _get_pointnav_action(self, env: int, rho: float, theta: float, masks: Tensor) -> Tensor:
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
        return self._pointnav_policy[env].act(obs_pointnav, masks, deterministic=True)

    def _disable_stair_frontier(self, env: int) -> None:
        """禁用楼梯前沿并更新相关状态"""
        self._obstacle_map[env]._disabled_frontiers.add(tuple(self._stair_frontier[env][0]))
        print(f"Frontier {self._stair_frontier[env]} is disabled due to no movement.")

        if self._climb_stair_flag[env] == 1:
            self._obstacle_map[env]._disabled_stair_map[self._obstacle_map[env]._up_stair_map == 1] = 1
            self._obstacle_map[env]._up_stair_map.fill(0)
            self._obstacle_map[env]._up_stair_frontiers = np.array([])
            self._climb_stair_over[env] = True
            self._obstacle_map[env]._has_up_stair = False
            self._obstacle_map[env]._look_for_downstair_flag = False
            self._remove_floor_from_map_list(env, self._cur_floor_index[env] + 1)
            self.floor_num[env] -= 1
        elif self._climb_stair_flag[env] == 2:
            self._obstacle_map[env]._disabled_stair_map[self._obstacle_map[env]._down_stair_map == 1] = 1
            self._obstacle_map[env]._down_stair_map.fill(0)
            self._obstacle_map[env]._down_stair_frontiers = np.array([])
            self._climb_stair_over[env] = True
            self._obstacle_map[env]._has_down_stair = False
            self._obstacle_map[env]._look_for_downstair_flag = False
            self._remove_floor_from_map_list(env, self._cur_floor_index[env] - 1)
            self.floor_num[env] -= 1
            self._cur_floor_index[env] -= 1

        self._climb_stair_flag[env] = 0
        self._stair_dilate_flag[env] = False

    def _remove_floor_from_map_list(self, env: int, floor_index: int) -> None:
        """从地图列表中移除指定楼层"""
        del self._object_map_list[env][floor_index]
        del self._value_map_list[env][floor_index]
        del self._obstacle_map_list[env][floor_index]

    def _climb_stair(self, observations: "TensorDict", env: int, ori_masks: Tensor) -> Tensor:
        masks = torch.tensor([self._num_steps[env] != 0], dtype=torch.bool, device="cuda")
        robot_xy = self._observations_cache[env]["robot_xy"]
        robot_xy_2d = np.atleast_2d(robot_xy)
        robot_px = self._obstacle_map[env]._xy_to_px(robot_xy_2d)
        heading = self._observations_cache[env]["robot_heading"]  # 以弧度为单位

        if self._climb_stair_flag[env] == 1:
            self._stair_frontier[env] = self._obstacle_map[env]._up_stair_frontiers
        elif self._climb_stair_flag[env] == 2:
            self._stair_frontier[env] = self._obstacle_map[env]._down_stair_frontiers

        current_distance = np.linalg.norm(self._stair_frontier[env] - robot_xy)
        print(f"Climb Stair -- Distance Change: {np.abs(self._last_frontier_distance[env] - current_distance)} and Stick Step {self._obstacle_map[env]._climb_stair_paused_step}")

        # 检查距离变化是否超过 0.2 米
        if np.abs(self._last_frontier_distance[env] - current_distance) > 0.2:
            self._obstacle_map[env]._climb_stair_paused_step = 0
            self._last_frontier_distance[env] = current_distance
        else:
            self._obstacle_map[env]._climb_stair_paused_step += 1

        if self._obstacle_map[env]._climb_stair_paused_step > 15:
            self._obstacle_map[env]._disable_end = True

        # 进了入口但没爬到质心(agent中心的点还没到楼梯),先往楼梯质心走
        if not self._reach_stair_centroid[env]:
            stair_frontiers = self._stair_frontier[env][0]
            rho, theta = rho_theta(robot_xy, heading, stair_frontiers)
            action = self._get_pointnav_action(env, rho, theta, masks)
            if action.item() == 0:
                self._reach_stair_centroid[env] = True
                print("Might close, change to move forward.")
                action[0] = 1
            return action

        # 爬到了楼梯质心,对于下楼梯
        if self._climb_stair_flag[env] == 2 and self._pitch_angle[env] < -30:
            self._pitch_angle[env] += self._pitch_angle_offset
            print("Look up a little for downstair!")
            action = TorchActionIDs_plook.LOOK_UP.to(masks.device)
            return action

        # 距离2米的目标点,像在驴子前面吊一根萝卜
        distance = 0.8

        # 找到深度图的最大值
        depth_map = self._observations_cache[env]["nav_depth"].squeeze(0).cpu().numpy()
        max_value = np.max(depth_map)
        max_indices = np.argwhere(depth_map == max_value)
        center_point = np.mean(max_indices, axis=0).astype(int)
        v, u = center_point[0], center_point[1]

        # 计算列偏移量相对于图像中心的归一化值 (-1 到 1)
        normalized_u = (u - self._cx) / self._cx
        normalized_u = np.clip(normalized_u, -1, 1)

        # 计算角度偏差
        angle_offset = normalized_u * (self._camera_fov / 2)
        target_heading = heading - angle_offset
        target_heading = target_heading % (2 * np.pi)
        x_target = robot_xy[0] + distance * np.cos(target_heading)
        y_target = robot_xy[1] + distance * np.sin(target_heading)
        target_point = np.array([x_target, y_target])
        target_point_2d = np.atleast_2d(target_point)
        temp_target_px = self._obstacle_map[env]._xy_to_px(target_point_2d)

        if self._climb_stair_flag[env] == 1:
            this_stair_end = self._obstacle_map[env]._up_stair_end
        elif self._climb_stair_flag[env] == 2:
            this_stair_end = self._obstacle_map[env]._down_stair_end

        if len(self._last_carrot_xy[env]) == 0 or len(this_stair_end) == 0:
            self._update_carrot_goal(env, target_point, temp_target_px)
        elif np.linalg.norm(this_stair_end - robot_px) <= 0.5 * self._obstacle_map[env].pixels_per_meter or self._obstacle_map[env]._disable_end:
            self._update_carrot_goal(env, target_point, temp_target_px)
        else:
            l1_distance = np.abs(this_stair_end[0] - temp_target_px[0][0]) + np.abs(this_stair_end[1] - temp_target_px[0][1])
            last_l1_distance = np.abs(this_stair_end[0] - self._last_carrot_px[env][0][0]) + np.abs(this_stair_end[1] - self._last_carrot_px[env][0][1])
            if last_l1_distance > l1_distance:
                self._update_carrot_goal(env, target_point, temp_target_px)
            else:
                self._carrot_goal_xy[env] = self._last_carrot_xy[env]
                self._obstacle_map[env]._carrot_goal_px = self._last_carrot_px[env]

        rho, theta = rho_theta(robot_xy, heading, self._carrot_goal_xy[env])
        action = self._get_pointnav_action(env, rho, theta, masks)
        if action.item() == 0:
            print("Might stop, change to move forward.")
            action[0] = 1
        return action

    def _get_pointnav_action(self, env: int, rho: float, theta: float, masks: Tensor) -> Tensor:
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
        return self._pointnav_policy[env].act(obs_pointnav, masks, deterministic=True)

    def _update_carrot_goal(self, env: int, target_point: np.ndarray, temp_target_px: np.ndarray) -> None:
        self._carrot_goal_xy[env] = target_point
        self._obstacle_map[env]._carrot_goal_px = temp_target_px
        self._last_carrot_xy[env] = target_point
        self._last_carrot_px[env] = temp_target_px

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray, env: int = 0,
    ) -> Tuple[np.ndarray, List[float]]:
        sorted_frontiers, sorted_values = self._value_map[env].sort_waypoints(frontiers, 0.5)
        sorted_frontiers = [temp for temp in sorted_frontiers if tuple(temp) not in self._obstacle_map[env]._disabled_frontiers]
        return sorted_frontiers, sorted_values

    @staticmethod
    def hamming_distance(hash1, hash2):
        """
        计算两个pHash值的汉明距离。
        """
        return np.sum(hash1 != hash2)
    
    def _get_best_frontier_with_llm(
        self,
        observations: Union[Dict[str, Tensor], "TensorDict"],
        frontiers: np.ndarray,
        env: int = 0,
        topk: int = 3,
        use_multi_floor: bool = True,
    ) -> Tuple[np.ndarray, float]:
        """Returns the best frontier and its value based on self._value_map.

        Args:
            observations (Union[Dict[str, Tensor], "TensorDict"]): The observations from
                the environment.
            frontiers (np.ndarray): The frontiers to choose from, array of 2D points.

        Returns:
            Tuple[np.ndarray, float]: The best frontier and its value.
        """
        sorted_pts, sorted_values = self._sort_frontiers_by_value(observations, frontiers, env)
        robot_xy = self._observations_cache[env]["robot_xy"]
        distances = np.linalg.norm(frontiers - robot_xy, axis=1)

        # 筛选出距离小于等于3.0米的前沿
        close_frontiers = [(idx, frontier, distance) for idx, (frontier, distance) in enumerate(zip(frontiers, distances)) if distance <= 3.0]

        if close_frontiers:
            # 选择距离最小的前沿
            closest_frontier = min(close_frontiers, key=lambda x: x[2])
            best_frontier_idx = closest_frontier[0]
            best_frontier = frontiers[best_frontier_idx]
            best_value = sorted_values[best_frontier_idx]
            print(f"Frontier {best_frontier} is very close (distance: {distances[best_frontier_idx]:.2f}m), selecting it.")
            self._obstacle_map[env]._neighbor_search = True
            return best_frontier, best_value

        self._obstacle_map[env]._finish_first_explore = False

        # 如果没有距离小于等于3.0米的前沿，开始决策
        if len(frontiers) == 1:
            best_frontier = frontiers[0]
            best_value = sorted_values[0]
        else:
            self.frontier_step_list[env] = []
            frontier_index_list = []
            seen_gray = []

            for idx, frontier in enumerate(frontiers):
                floor_num_steps, image_rgb = self._obstacle_map[env].extract_frontiers_with_image(frontier)
                image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
                is_similar = any(ssim(gray, image_gray, full=True)[0] > 0.75 for gray in seen_gray)

                if not is_similar:
                    seen_gray.append(image_gray)
                    self.frontier_step_list[env].append(floor_num_steps)
                    frontier_index_list.append(idx)

                    if len(self.frontier_step_list[env]) == topk:
                        break

            target_object_category = self._target_object[env].split("|")[0]

            if len(self.frontier_step_list[env]) <= 1:
                best_frontier_idx = 0
            elif self.floor_num[env] > 1 and self._num_steps[env] - self.multi_floor_ask_step[env] >= 60 and self._obstacle_map[env]._floor_num_steps >= 100 and use_multi_floor:
                self.multi_floor_ask_step[env] = self._num_steps[env]
                multi_floor_prompt = self._prepare_multiple_floor_prompt(target_object_category, env)
                print(f"## Multi-floor Prompt: {multi_floor_prompt}")
                multi_floor_response = self._llm.chat(multi_floor_prompt)
                if multi_floor_response == "-1":
                    best_frontier_idx = self.llm_analyze_single_floor(env, target_object_category, frontier_index_list)
                else:
                    current_floor = self._cur_floor_index[env] + 1
                    temp_frontier_index = self._extract_multiple_floor_decision(multi_floor_response, env)
                    if temp_frontier_index > current_floor:
                        return frontiers[0], -100
                    elif temp_frontier_index < current_floor:
                        return frontiers[0], -200
                    else:
                        best_frontier_idx = self.llm_analyze_single_floor(env, target_object_category, frontier_index_list)
            else:
                best_frontier_idx = self.llm_analyze_single_floor(env, target_object_category, frontier_index_list)

            best_frontier = frontiers[best_frontier_idx]
            best_value = sorted_values[best_frontier_idx]

        # 防止空气墙或者动作循环,强制更换 frontier
        if np.array_equal(self._last_frontier[env], best_frontier):
            if self._frontier_stick_step[env] == 0:
                self._last_frontier_distance[env] = np.linalg.norm(best_frontier - robot_xy)
                self._frontier_stick_step[env] += 1
            else:
                current_distance = np.linalg.norm(best_frontier - robot_xy)
                print(f"Distance Change: {np.abs(self._last_frontier_distance[env] - current_distance)} and Stick Step {self._frontier_stick_step[env]}")
                if np.abs(self._last_frontier_distance[env] - current_distance) > 0.3 and not self._obstacle_map[env]._neighbor_search:
                    self._frontier_stick_step[env] = 0
                    self._last_frontier_distance[env] = current_distance
                else:
                    if self._frontier_stick_step[env] >= 20:
                        self._obstacle_map[env]._disabled_frontiers.add(tuple(best_frontier))
                        print(f"Frontier {best_frontier} is disabled due to no movement.")
                        self._frontier_stick_step[env] = 0
                    else:
                        self._frontier_stick_step[env] += 1
        else:
            self._frontier_stick_step[env] = 0
            self._last_frontier_distance[env] = 0

        self._last_value[env] = best_value
        self._last_frontier[env] = best_frontier
        if not self._obstacle_map[env]._finish_first_explore:
            self._obstacle_map[env]._finish_first_explore = True
            self._force_frontier[env] = best_frontier.copy()

        print(f"Now the best_frontier is {best_frontier}")
        return best_frontier, best_value

    # def _get_object_detections_with_stair(self, img: np.ndarray, env: int) -> ObjectDetections:
    
    #     non_coco_caption_with_stair = self._non_coco_caption +  " stair ."
    #     non_coco_detections = self._object_detector.predict(img, captions=non_coco_caption_with_stair)

    #     return non_coco_detections 
    
    def _get_object_detections_with_stair(self, imgs: List[np.ndarray], env_indices: List[int]) -> List[ObjectDetections]:
        common_captions = [["stair"], [" "]]
        # 为每个环境生成对应的描述
        non_coco_captions_with_stair = [common_captions for _ in range(len(env_indices))]
        
        # 批量推理
        non_coco_detections_list = self._object_detector.predict(imgs, captions=non_coco_captions_with_stair)
        
        return non_coco_detections_list

    def _update_object_map_with_stair(self, height: int, width: int):
        """
        Updates the object map with the given rgb and depth images, and the given
        transformation matrix from the camera to the episodic coordinate frame.

        Args:
            height (int): The height of the image.
            width (int): The width of the image.
        """
        self._object_masks = np.zeros((self._num_envs, height, width), dtype=np.uint8)
        self._stair_masks = np.zeros((self._num_envs, height, width), dtype=bool)

        # 收集所有环境的 RGB 图像
        rgb_images = []
        env_indices = []
        for env in range(self._num_envs):
            if self._num_steps[env] == 0:
                # rgb_images.append(None)
                continue
            # rgb, depth, tf_camera_to_episodic, min_depth, max_depth, fx, fy = self._observations_cache[env]["map_rgbd"][0]
            rgb_images.append(self._observations_cache[env]["map_rgbd"][0][0])
            env_indices.append(env)

        # 批量推理
        if rgb_images:
            batch_results = self._ram.predict(rgb_images)  # 假设 RAM 支持批量推理
            self.non_coco_detection_list = self._get_object_detections_with_stair(rgb_images, env_indices) # 检测目标和楼梯
            # 处理批量推理结果
            for env_idx, result in zip(env_indices, batch_results["response"]):
                cur_objs_list = [item.strip() for item in result.split('|')]
                self._object_map[env_idx].each_step_objects[self._obstacle_map[env_idx]._floor_num_steps] = cur_objs_list
                for obj in cur_objs_list:
                    self._object_map[env_idx].this_floor_objects.add(obj)

            # 准备批量图片输入
            pil_images = [Image.fromarray(rgb_images[i]).convert("RGB") for i in range(len(env_indices))]
            if any(img.mode != "RGB" for img in pil_images):
                pil_images = [img.convert("RGB") if img.mode != "RGB" else img for img in pil_images]

            # 批量处理图片
            place365_input_imgs = torch.stack([self.place365_centre_crop(img) for img in pil_images]).to(self.device) # torch.stack([self.place365_centre_crop(img).unsqueeze(0) for img in pil_images]).to(self.device)
            logits = self.scene_classify_model.forward(place365_input_imgs)
            h_x = torch.nn.functional.softmax(logits, 1) # .squeeze()

            # 获取每个环境的top-5预测
            top_5_indices = torch.topk(h_x, 5, dim=1).indices
            top_5_classes = [[self.place365_classes[i] for i in indices] for indices in top_5_indices]
            room_candi_type = [self.extract_room_categories(classes) for classes in top_5_classes]

            # 更新房间候选类型
            for env_idx, rc_type in zip(env_indices, room_candi_type):
                self._object_map[env_idx].each_step_rooms[self._obstacle_map[env_idx]._floor_num_steps] = rc_type
                self._object_map[env_idx].this_floor_rooms.add(rc_type)

            # 处理语义数据（如果存在）
            for env in range(len(env_indices)):
                env_idx = env_indices[env]
                if self._observations_cache[env_idx]["semantic"] is not None:
                    semantic_data = self._observations_cache[env_idx]["semantic"]
                    target_object_mask = np.zeros_like(semantic_data, dtype=bool)
                    target_ids = self.semantic_dict[env_idx]["target_ids"]
                    for target_id in target_ids:
                        target_object_mask |= (semantic_data == target_id)
                    if np.any(target_object_mask > 0):
                        if target_object_mask.shape[-1] == 1:
                            target_object_mask = target_object_mask.squeeze(axis=-1)
                            target_object_mask = np.array(target_object_mask, dtype=np.uint8)
                        self._object_masks[env_idx][target_object_mask > 0] = 1
                        self._object_map[env_idx].update_map(
                            self._target_object[env_idx],
                            self._observations_cache[env_idx]["map_rgbd"][0][1], # depth
                            target_object_mask,
                            self._observations_cache[env_idx]["map_rgbd"][0][2], # tf_camera_to_episodic
                            self._observations_cache[env_idx]["map_rgbd"][0][3], # min_depth
                            self._observations_cache[env_idx]["map_rgbd"][0][4], # max_depth
                            self._observations_cache[env_idx]["map_rgbd"][0][5], # fx
                            self._observations_cache[env_idx]["map_rgbd"][0][6], # fy
                        )

                # for stair
                # 过滤检测结果
                self.non_coco_detection_list[env_idx].filter_by_class(["stair"])
                # non_coco_detections.filter_by_conf(0.60)
                for idx in range(len(self.non_coco_detection_list[env_idx].logits)):
                    stair_bbox_denorm = self.non_coco_detection_list[env_idx].boxes[idx] * np.array([width, height, width, height])
                    stair_mask = self._mobile_sam.segment_bbox(rgb_images[env], stair_bbox_denorm.tolist())
                    self._stair_masks[env_idx][stair_mask > 0] = 1

                # final
                cone_fov = get_fov(self._observations_cache[env_idx]["map_rgbd"][0][5], self._observations_cache[env_idx]["map_rgbd"][0][1].shape[1])
                self._object_map[env_idx].update_explored(self._observations_cache[env_idx]["map_rgbd"][0][2], self._observations_cache[env_idx]["map_rgbd"][0][4], cone_fov)

    def llm_analyze_single_floor(self, env, target_object_category, frontier_index_list):
        """
        Analyze the environment using the Large Language Model (LLM) to determine the best frontier to explore.

        Parameters:
        env (str): The current environment identifier.
        target_object_category (str): The category of the target object to find.
        frontier_index_list (list): A list of frontier indices.

        Returns:
        int: The index of the frontier that is most likely to lead to the target object.
        """
        prompt = self._prepare_single_floor_prompt(target_object_category, env)
        print(f"## Single-floor Prompt:\n{prompt}")
        response = self._llm.chat(prompt)

        # Extract the frontier index from the response
        if response == "-1":
            return frontier_index_list[0]  # Default to the first frontier

        try:
            cleaned_response = response.replace("\n", "").replace("\r", "")
            response_dict = json.loads(cleaned_response)
            index = response_dict.get("Index", "N/A")
            reason = response_dict.get("Reason", "N/A")

            if index == "N/A":
                logging.warning("Index not found in response")
                return frontier_index_list[0]

            # Log the response
            if reason != "N/A":
                response_str = f"Area Index: {index}. Reason: {reason}"
                self.vlm_response[env] = "## Single-floor Prompt:\n" + response_str
                print(f"## Single-floor Response:\n{response_str}")
            else:
                print(f"Index: {index}")

            # Convert index to integer and validate
            try:
                index_int = int(index)
            except ValueError:
                logging.warning(f"Index is not a valid integer: {index}")
                return frontier_index_list[0]

            if 1 <= index_int <= len(frontier_index_list):
                return frontier_index_list[index_int - 1]  # Convert to 0-based index
            else:
                logging.warning(f"Index ({index_int}) is out of valid range: 1 to {len(frontier_index_list)}")
                return frontier_index_list[0]

        except json.JSONDecodeError:
            logging.warning(f"Failed to parse JSON response: {response}")
            return frontier_index_list[0]

    def get_room_probabilities(self, target_object_category: str):
        """
        Get the probabilities of the target object category in different room types.

        Parameters:
        target_object_category (str): The category of the target object.

        Returns:
        dict: A dictionary of room probabilities.
        """
        synonym_mapping = {
            "couch": ["sofa"],
            "sofa": ["couch"],
        }

        target_categories = [target_object_category] + synonym_mapping.get(target_object_category, [])
        if not any(category in knowledge_graph for category in target_categories):
            return {}

        room_probabilities = {}
        for room in reference_rooms:
            for category in target_categories:
                if knowledge_graph.has_edge(category, room):
                    room_probabilities[room] = round(knowledge_graph[category][room]['weight'] * 100, 1)
                    break
            else:
                room_probabilities[room] = 0.0
        return room_probabilities

    def get_floor_probabilities(self, df, target_object_category, floor_num):
        """
        Get the probabilities of the target object category on different floors.

        Parameters:
        df (pd.DataFrame): The probability table.
        target_object_category (str): The category of the target object.
        floor_num (int): The total number of floors.

        Returns:
        dict: A dictionary of floor probabilities.
        """
        if df is None:
            return None

        probabilities = {}
        if floor_num > 4:
            logging.warning(f"Floor number {floor_num} exceeds the maximum supported floor number (4). Showing probabilities for maximum multi-floor scenarios.")
            floor_num = 4

        for floor in range(1, floor_num + 1):
            column_name = f"train_floor{floor_num}_{floor}"
            if column_name in df.columns:
                prob = df.set_index("category").at[target_object_category, column_name]
                probabilities[floor] = prob
            else:
                logging.warning(f"Column {column_name} not found in the probability table.")
                probabilities[floor] = 0.0
        return probabilities

    def _prepare_single_floor_prompt(self, target_object_category, env):
        """
        Prepare the prompt for the LLM in a single-floor scenario.
        """
        area_descriptions = []
        self.frontier_rgb_list[env] = []

        for i, step in enumerate(self.frontier_step_list[env]):
            try:
                room = self._object_map[env].each_step_rooms[step] or "unknown room"
                objects = self._object_map[env].each_step_objects[step] or "no visible objects"
                if isinstance(objects, list):
                    objects = ", ".join(objects)
                self.frontier_rgb_list[env].append(self._obstacle_map[env]._each_step_rgb[step])
                area_descriptions.append({
                    "area_id": i + 1,
                    "room": room,
                    "objects": objects
                })
            except (IndexError, KeyError) as e:
                logging.warning(f"Error accessing room or objects for step {step}: {e}")
                continue

        room_probabilities = self.get_room_probabilities(target_object_category)
        sorted_rooms = sorted(room_probabilities.items(), key=lambda x: (-x[1], x[0]))
        prob_entries = ',\n'.join([f'{INDENT_L2}"{room.capitalize()}": {prob:.1f}%' for room, prob in sorted_rooms])
        formatted_area_descriptions = [f'{INDENT_L2}"Area {desc["area_id"]}": "a {desc["room"].replace("_", " ")} containing objects: {desc["objects"]}"' for desc in area_descriptions]
        area_entries = ',\n'.join(formatted_area_descriptions)

        example_input = (
            'Example Input:\n'
            '{\n'
            f'{INDENT_L1}"Goal": "toilet",\n'
            f'{INDENT_L1}"Prior Probabilities between Room Type and Goal Object": [\n'
            f'{prob_entries}\n'
            f'{INDENT_L1}],\n'
            f'{INDENT_L1}"Area Descriptions": [\n'
            f'{area_entries}\n'
            f'{INDENT_L1}]\n'
            '}'
        ).strip()

        actual_input = (
            'Now answer question:\n'
            'Input:\n'
            '{\n'
            f'{INDENT_L1}"Goal": "{target_object_category}",\n'
            f'{INDENT_L1}"Prior Probabilities between Room Type and Goal Object": [\n'
            f'{prob_entries}\n'
            f'{INDENT_L1}],\n'
            f'{INDENT_L1}"Area Descriptions": [\n'
            f'{area_entries}\n'
            f'{INDENT_L1}]\n'
            '}'
        ).strip()

        prompt = "\n".join([
            "You need to select the optimal area based on prior probabilistic data and environmental context.",
            "You need to answer the question in the following JSON format:",
            example_input,
            'Example Response:\n{"Index": "1", "Reason": "Shower and towel in Bathroom indicate toilet location, with high probability (90.0%)."}',
            actual_input
        ])
        return prompt

    def _prepare_multiple_floor_prompt(self, target_object_category, env):
        """
        Prepare the prompt for the LLM in a multiple-floor scenario.
        """
        current_floor = self._cur_floor_index[env] + 1
        total_floors = self.floor_num[env]
        floor_probs = self.get_floor_probabilities(floor_probabilities_df, target_object_category, total_floors)
        floor_prob_entries = ',\n'.join([f'{INDENT_L2}"Floor {floor}": {prob:.1f}%' for floor, prob in floor_probs.items()])
        room_probabilities = self.get_room_probabilities(target_object_category)
        sorted_rooms = sorted(room_probabilities.items(), key=lambda x: (-x[1], x[0]))
        prob_entries = ',\n'.join([f'{INDENT_L2}"{room.capitalize()}": {prob:.1f}%' for room, prob in sorted_rooms])

        floor_descriptions = []
        for floor in range(total_floors):
            try:
                rooms = self._object_map_list[env][floor].this_floor_rooms or {"unknown rooms"}
                objects = self._object_map_list[env][floor].this_floor_objects or {"unknown objects"}
                rooms_str = ", ".join(rooms)
                objects_str = ", ".join(objects)
                floor_descriptions.append({
                    "floor_id": floor + 1,
                    "status": 'Current floor' if floor + 1 == current_floor else 'Other floor',
                    "fully_explored": self._obstacle_map_list[env][floor]._this_floor_explored,
                    "room": rooms_str,
                    "objects": objects_str,
                })
            except Exception as e:
                logging.error(f"Error describing floor {floor}: {e}")
                continue

        formatted_floor_descriptions = [
            f'{INDENT_L2}"Floor {desc["floor_id"]}": "{desc["status"]}. There are room types: {desc["room"]}, containing objects: {desc["objects"]}'
            + ('. You do not need to explore this floor again"' if desc.get("fully_explored", False) else '"')
            for desc in floor_descriptions
        ]
        floor_entries = ',\n'.join(formatted_floor_descriptions)

        example_input = (
            'Example Input:\n'
            '{\n'
            f'{INDENT_L1}"Goal": "bed",\n'
            f'{INDENT_L1}"Prior Probabilities between Floor and Goal Object": [\n'
            f'{floor_prob_entries}\n'
            f'{INDENT_L1}],\n'
            f'{INDENT_L1}"Prior Probabilities between Room Type and Goal Object": [\n'
            f'{prob_entries}\n'
            f'{INDENT_L1}],\n'
            f'{INDENT_L1}"Floor Descriptions": [\n'
            f'{floor_entries}\n'
            f'{INDENT_L1}]\n'
            '}'
        ).strip()

        actual_input = (
            'Now answer question:\n'
            'Input:\n'
            '{\n'
            f'{INDENT_L1}"Goal": "{target_object_category}",\n'
            f'{INDENT_L1}"Prior Probabilities between Floor and Goal Object": [\n'
            f'{floor_prob_entries}\n'
            f'{INDENT_L1}],\n'
            f'{INDENT_L1}"Prior Probabilities between Room Type and Goal Object": [\n'
            f'{prob_entries}\n'
            f'{INDENT_L1}],\n'
            f'{INDENT_L1}"Floor Descriptions": [\n'
            f'{floor_entries}\n'
            f'{INDENT_L1}]\n'
            '}'
        ).strip()

        prompt = "\n".join([
            "You need to select the optimal floor based on prior probabilistic data and environmental context.",
            "You need to answer the question in the following JSON format:",
            example_input,
            'Example Response:\n{"Index": "3", "Reason": "The bedroom is most likely to be on the Floor 3, and the room types and object types on the Floor 1 and Floor 2 are not directly related to the target object bed, especially it do not need to explore Floor 2 again."}',
            actual_input
        ])
        return prompt

    def _extract_multiple_floor_decision(self, multi_floor_response, env) -> int:
        """
        从LLM响应中提取多楼层决策
        
        参数:
            multi_floor_response (str): LLM的原始响应文本
            current_floor (int): 当前楼层索引 (0-based)
            total_floors (int): 总楼层数
            
        返回:
            int: 楼层决策 0/1/2，解析失败返回0
        """
        # 防御性输入检查
        try:
            # 解析 LLM 的回复
            cleaned_response = multi_floor_response.replace("\n", "").replace("\r", "")
            response_dict = json.loads(cleaned_response)
            target_floor_index = int(response_dict.get("Index", -1))
            current_floor = self._cur_floor_index[env] + 1  # 当前楼层（从1开始）
            reason = response_dict.get("Reason", "N/A")
            # Form the response string
            if reason != "N/A":
                response_str = f"Floor Index: {target_floor_index}. Reason: {reason}"
                self.vlm_response[env] = "## Multi-floor Prompt:\n" + response_str
                print(f"## Multi-floor Response:\n{response_str}")
            # 检查目标楼层是否合理
            if target_floor_index <= 0 or target_floor_index > self.floor_num[env]:
                logging.warning("Invalid floor index from LLM response. Returning current floor.")
                return current_floor  # 返回当前楼层

            return target_floor_index  # 返回目标楼层

        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse LLM response: {e}")
        except Exception as e:
            logging.error(f"Error extracting floor decision: {e}")

        # 如果解析失败或异常，返回当前楼层
        return self._cur_floor_index[env] + 1  # 当前楼层（从1开始）

    def extract_room_categories(self, top_5_classes):
        """
        提取目标类别或选择 top 1 类别。
        
        参数:
            top_5_classes (list of str): top 5 的类别名称列表。
        
        返回:
            list: 如果 top 5 中有direct_mapping能映射的目标类别，返回排名靠前的目标类别；否则返回 top 1 类别。
        """
        # 遍历 top_5_classes，尝试映射到目标类别
        selected_categories = ""
        for cls_name in top_5_classes:
            if cls_name in direct_mapping:  # 如果类别在 direct_mapping 中
                mapped_category = direct_mapping[cls_name]
                if mapped_category in reference_rooms:  # 如果映射后的类别是目标类别
                    selected_categories = mapped_category
                    break  # 只选择第一个匹配的目标类别

        # 如果没有匹配的目标类别，返回 top 1 类别
        return selected_categories if selected_categories else top_5_classes[0]
