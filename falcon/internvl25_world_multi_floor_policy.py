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
import logging
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
from vlfm.vlm.grounding_dino_test import ObjectDetections
from vlfm.vlm.yolo_world_test import YoloWorldClient_MF
from vlfm.vlm.ram_test import RAMClient
from vlfm.vlm.sam import MobileSAMClient
from vlfm.vlm.blip2itm import BLIP2ITMClient
from vlfm.vlm.internvl25itm import INTERNVL2_5ITMClient, load_image_array
from vlfm.policy.utils.pointnav_policy import WrappedPointNavResNetPolicy
from vlfm.mapping.object_point_cloud_map import ObjectPointCloudMap, reference_object_list
from vlfm.mapping.obstacle_map import ObstacleMap
from vlfm.mapping.value_map import ValueMap
from vlfm.policy.utils.acyclic_enforcer import AcyclicEnforcer
from collections import deque
from torchvision import transforms as trn
import math
import heapq
from torch.autograd import Variable as V
import torchvision.models as models
from torch.nn import functional as F
# from vlfm.utils.img_utils import (
#     reorient_rescale_map,
# )
from PIL import Image
import json


PROMPT_SEPARATOR = "|"
STAIR_CLASS_ID = 17  # MPCAT40中 楼梯的类别编号是 16 + 1

CHAIR_CLASS_ID = 4  # MPCAT40中 椅子的类别编号是 3 + 1
SOFA_CLASS_ID = 11  # MPCAT40中 沙发的类别编号是 10 + 1
PLANT_CLASS_ID = 15  # MPCAT40中 植物的类别编号是 14 + 1
BED_CLASS_ID = 12  # MPCAT40中 床的类别编号是 11 + 1
TOILET_CLASS_ID = 19  # MPCAT40中 马桶的类别编号是 18 + 1
TV_CLASS_ID = 23  # MPCAT40中 电视的类别编号是 22 + 1

# reference_captions = [[obj] for obj in reference_object_list]
reference_rooms = [
            "bathroom", "bedroom", "dining_room", "garage", "hall",
            "kitchen", "laundry_room", "living_room", "office", "rec_room"
        ]
# 直接映射表
direct_mapping = {
    # Bathroom 相关
    "bathroom": "bathroom",
    "shower": "bathroom",
    "jacuzzi/indoor": "bathroom",

    # Bedroom 相关
    "bedroom": "bedroom",
    "bedchamber": "bedroom",
    "dorm_room": "bedroom",
    "hotel_room": "bedroom",
    "childs_room": "bedroom",

    # Dining Room 相关
    "dining_room": "dining_room",
    "dining_hall": "dining_room",
    "banquet_hall": "dining_room",
    "restaurant": "dining_room",
    "cafeteria": "dining_room",

    # Garage 相关
    "garage/indoor": "garage",
    "garage/outdoor": "garage",
    "parking_garage/indoor": "garage",
    "parking_garage/outdoor": "garage",

    # Hall 相关
    "entrance_hall": "hall",
    "lobby": "hall",
    "corridor": "hall",
    "mezzanine": "hall",

    # Kitchen 相关
    "kitchen": "kitchen",
    "restaurant_kitchen": "kitchen",

    # Laundry Room 相关
    "laundry_room": "laundry_room",
    "laundromat": "laundry_room",

    # Living Room 相关
    "living_room": "living_room",
    "home_theater": "living_room",
    "television_room": "living_room",

    # Office 相关
    "office": "office",
    "office_cubicles": "office",
    "conference_room": "office",
    "home_office": "office",
    "computer_room": "office",

    # Rec Room 相关
    "recreation_room": "rec_room",
    "playroom": "rec_room",
    "amusement_arcade": "rec_room",
    "gymnasium/indoor": "rec_room",
    "arcade": "rec_room",
}
# 加载知识图谱
import networkx as nx
# 多楼层分析
# 加载楼层概率表格
import pandas as pd
def load_floor_probabilities(file_path):
    """
    加载楼层和物体分布概率表格。

    Parameters:
    file_path (str): 表格文件路径。

    Returns:
    pd.DataFrame: 包含物体分布概率的表格。
    """
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        logging.error(f"Failed to load floor probabilities: {e}")
        return None
floor_probabilities_df = load_floor_probabilities("falcon/floor_object_possibility.xlsx")

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
class HabitatITMPolicy_Intern(HabitatMixin, ITMPolicyV2):
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
        # self._non_coco_caption = ""
        self._load_yolo: bool = True
        self._object_detector = YoloWorldClient_MF(port=int(os.environ.get("YOLO_WORLD_PORT", "15184")))
        self._mobile_sam = MobileSAMClient(port=int(os.environ.get("SAM_PORT", "15183")))
        self._ram = RAMClient(port=int(os.environ.get("RAM_PORT", "15185")))
        
        self._use_vqa = kwargs["use_vqa"]
        # if self._use_vqa:
        #     self._vqa = BLIP2Client(port=int(os.environ.get("BLIP2_PORT", "12185")))
        self._pointnav_stop_radius = kwargs["pointnav_stop_radius"] # 看看会不会停的正常点 0.7
        self._visualize = kwargs["visualize"]
        self._vqa_prompt = kwargs["vqa_prompt"]
        self._coco_threshold = kwargs["coco_threshold"]
        # self._non_coco_threshold = kwargs["non_coco_threshold"]
        ## num_envs
        self._num_envs = kwargs['num_envs']
        self._object_map_erosion_size = kwargs["object_map_erosion_size"]
        # self._object_map=[ObjectPointCloudMap(erosion_size=kwargs["object_map_erosion_size"]) for _ in range(self._num_envs)] # 
        self._object_map_list = [[ObjectPointCloudMap(erosion_size=self._object_map_erosion_size)] for _ in range(self._num_envs)]
        self._depth_image_shape =  tuple(kwargs["depth_image_shape"]) # (224, 224) #
        self._pointnav_policy = [WrappedPointNavResNetPolicy(kwargs["pointnav_policy_path"]) for _ in range(self._num_envs)]
        # HM3D pretrained model
        # self._depth_image_shape =  (256, 256) #
        # self._pointnav_policy = [WrappedPointNavResNetPolicy('data/pretrained_models/pretrained_hm3d_habitat3_v2.pth') for _ in range(self._num_envs)]
        
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
        self.vlm_itm = INTERNVL2_5ITMClient(port=int(os.environ.get("INTERNVL2_5ITM_PORT", "15181")))
        self._itm = BLIP2ITMClient(port=int(os.environ.get("BLIP2ITM_PORT", "15182")))
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
        self._double_check_goal = [False for _ in range(self._num_envs)]
        self._initialize_step = [0 for _ in range(self._num_envs)]

        ## stop distance
        # self._might_close_to_goal = [False for _ in range(self._num_envs)]
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
        # self.scene_classify_model.load_state_dict(state_dict).to(self.device)
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

        ## knowledge graph
        with open('knowledge_graph.json', 'r') as f:
            self.knowledge_graph = nx.node_link_graph(json.load(f))
        
        ## floor num
        self.floor_num = [len(self._obstacle_map_list[env]) for env in range(self._num_envs)]
        self._blip_cosine = [0 for _ in range(self._num_envs)]
        ## true floor num
            
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
        self._double_check_goal[env] = False

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

        # if self._object_masks[env].sum() > 0:
        #     contours, _ = cv2.findContours(self._object_masks[env], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #     annotated_rgb = cv2.drawContours(detections.annotated_frame, contours, -1, (255, 0, 0), 2)
        #     annotated_depth = cv2.drawContours(annotated_depth, contours, -1, (255, 0, 0), 2)
        # elif self._person_masks[env].sum() > 0:
        #     contours, _ = cv2.findContours(self._person_masks[env].astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #     annotated_rgb = cv2.drawContours(self.coco_detection_list[env].annotated_frame, contours, -1, (255, 0, 0), 2)
        #     annotated_depth = cv2.drawContours(annotated_depth, contours, -1, (255, 0, 0), 2)
        # elif self._stair_masks[env].sum() > 0:
        #     contours, _ = cv2.findContours(self._stair_masks[env].astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #     annotated_rgb = cv2.drawContours(self.non_coco_detection_list[env].annotated_frame, contours, -1, (255, 0, 0), 2)
        #     annotated_depth = cv2.drawContours(annotated_depth, contours, -1, (255, 0, 0), 2)
        if self._all_object_masks[env].sum() > 0:
            contours, _ = cv2.findContours(self._all_object_masks[env].astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            annotated_rgb = cv2.drawContours(detections.annotated_frame, contours, -1, (255, 0, 0), 2)
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

        policy_info["vlm_input"] = self.frontier_rgb_list[env]
        policy_info["vlm_response"] = self.vlm_response[env]
        # policy_info["object_map"] = self._object_map[env].visualization # cv2.cvtColor( self._object_map[env].visualization,
                                                # cv2.COLOR_BGR2RGB) 
                                                                        # self._object_map[env].visualize(self._obstacle_map[env]._map, 
                                                                        #                                         self._obstacle_map[env]._up_stair_map, self._obstacle_map[env]._down_stair_map,
                                                                        #                                         self._obstacle_map[env]._frontiers_px,self._obstacle_map[env]._disabled_frontiers_px, 
                                                                        #                                         self._obstacle_map[env].explored_area,self._obstacle_map[env]._up_stair_frontiers_px,
                                                                        #                                         self._obstacle_map[env]._down_stair_frontiers_px)

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
                        iterations=1,
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
                            if self._climb_stair_flag[env] == 1: # 1是上,2是下
                                self._obstacle_map[env]._up_stair_start = robot_px[0].copy()
                            elif self._climb_stair_flag[env] == 2: # 1是上,2是下
                                self._obstacle_map[env]._down_stair_start = robot_px[0].copy()
                    # robot_at_stair = self.is_robot_in_stair_map_fast(env, robot_xy_2d)
                    if self._reach_stair[env] == True and self._reach_stair_centroid[env] == False:
                        # 原先是判断代理的质心是否已处于楼梯,但发现这样代理还是容易转身又下楼梯,
                        # 所以改成代理的质心是否距离楼梯的质心很近
                        if self._stair_frontier[env] is not None and np.linalg.norm(self._stair_frontier[env] - robot_xy_2d) <= 0.3:
                            self._reach_stair_centroid[env] = True
                            # 记录该楼梯质心px坐标,上楼转下楼或者下楼转上楼时,保留该点所在的楼梯连通域

                    # 如果视野中没有楼梯并且质心已经走出楼梯,结束各种阶段。
                    # 对下楼梯似乎不够,得整个边缘都走出
                    if self._reach_stair_centroid[env] == True:

                        if self.is_robot_in_stair_map_fast(env, robot_px, self._temp_stair_map[env])[0]: # (self._climb_stair_flag[env] == 1 and  # 
                            pass
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
                                self._obstacle_map[env]._look_for_downstair_flag = False
                            self._climb_stair_flag[env] = 0
                            self._stair_dilate_flag[env] = False
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
        self.floor_num[env] = len(self._obstacle_map_list[env])
        # 附加，记录当前新加的frontier对应rgb信息
        self._obstacle_map[env].project_frontiers_to_rgb_hush(self._observations_cache[env]["object_map_rgbd"][0][0])
        # self._observations_cache[env]["robot_xy"]
        

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
            self._blip_cosine[env] = cosines[0][0]

    def _update_distance_on_object_map(self) -> None:
        for env in range(self._num_envs):
            self._object_map[env].update_agent_traj(
                self._observations_cache[env]["robot_xy"],
                self._observations_cache[env]["robot_heading"],
            )
            if np.argwhere(self._object_map[env]._map).size > 0 and self._target_object[env] in self._object_map[env].clouds:
                # camera_position_2d = np.atleast_2d(self._object_map[env]._camera_positions[-1])
                # camera_position_px = self._object_map[env]._xy_to_px(camera_position_2d)

                # up_stair_points = np.argwhere(self._obstacle_map[env]._up_stair_map)
                # robot_xy = self._observations_cache[env]["robot_xy"]
                # robot_xy_2d = np.atleast_2d(robot_xy) 
                # robot_px = self._object_map[env]._xy_to_px(robot_xy_2d)
                # distances = np.abs(up_stair_points[:, 0] - robot_px[0][0]) + np.abs(up_stair_points[:, 1] - robot_px[0][1])
                # min_dis_to_upstair = np.min(distances)
                
                # distances_px = np.linalg.norm(np.argwhere(self._object_map[env]._map) - robot_px[0], axis=1)
                # self.min_distance_px[env] = np.min(distances_px) # px
                # if self.min_distance_px[env] <= (0.2+self.agent_radius) * self._object_map[env].pixels_per_meter * self._object_map[env].pixels_per_meter:
                #     self._might_close_to_goal[env] = True
                # else:
                #     self._might_close_to_goal[env] = False
                curr_position = self._observations_cache[env]["tf_camera_to_episodic"][:3, 3]
                closest_point = self._object_map[env]._get_closest_point(self._object_map[env].clouds[self._target_object[env]], curr_position)
                self.cur_dis_to_goal[env] = np.linalg.norm(closest_point[:2] - curr_position[:2])
                # self.min_distance_xy[env] = np.min(self.cur_dis_to_goal[env],self.min_distance_xy[env])
                # if self.cur_dis_to_goal[env] < 1.0 : # * self._object_map[env].pixels_per_meter * self._object_map[env].pixels_per_meter: # 0.2+self.agent_radius
                #     self._might_close_to_goal[env] = True
                # else:
                #     self._might_close_to_goal[env] = False


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
            if self._climb_stair_over[env] == True and self._obstacle_map[env]._down_stair_map[y,x] == 1 and len(self._obstacle_map[env]._down_stair_frontiers) > 0:  # and self._obstacle_map_list[env][self._cur_floor_index[env] - 1]._explored_up_stair == False 
                self._reach_stair[env] = True
                self._climb_stair_over[env] = False
                self._climb_stair_flag[env] = 2
                self._obstacle_map[env]._down_stair_start = robot_px[0].copy()
                # self._reach_stair_centroid[env] = True
            # 不知不觉到了上楼的楼梯,且不是刚刚下楼的
            elif self._climb_stair_over[env] == True and self._obstacle_map[env]._up_stair_map[y,x] == 1 and len(self._obstacle_map[env]._up_stair_frontiers) > 0: # and self._obstacle_map_list[env][self._cur_floor_index[env] + 1]._explored_down_stair == False 
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
                            self._climb_stair_flag[env] = 0
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
        # self._done_initializing[env] = not self._num_steps[env] < 11  # type: ignore
        if self._initialize_step[env] > 11: # self._obstacle_map[env]._floor_num_steps > 11:
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
            
            # 如果在楼梯间，不是立刻初始化，而是优先去找有没有去过的楼层
            self._obstacle_map[env]._this_floor_explored = True # 标记
            # 目前该逻辑是探索完一层,就去探索其他层,同时上楼优先。这个逻辑后面可能改掉
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
                        if self._obstacle_map[env]._reinitialize_flag == False and self._obstacle_map[env]._floor_num_steps < 50: # 防止楼梯间状态
                            # 如果有之前的楼梯，连通域可能错误，不要保留的好 
                            self._object_map[env].reset()
                            self._value_map[env].reset()

                            if self._compute_frontiers:
                                self._obstacle_map[env].reset()
                                self._obstacle_map[env]._reinitialize_flag = True
                            # 防止之前episode爬楼梯异常退出
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
                        else:
                            print("In all floors, no frontiers found during exploration, stopping.")
                            return self._stop_action.to(masks.device)
                else:
                    if self._obstacle_map[env]._reinitialize_flag == False and self._obstacle_map[env]._floor_num_steps < 50: # 防止楼梯间状态
                        self._object_map[env].reset()
                        self._value_map[env].reset()

                        if self._compute_frontiers:
                            self._obstacle_map[env].reset()
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
                    else:
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
                    if self._obstacle_map[env]._reinitialize_flag == False and self._obstacle_map[env]._floor_num_steps < 50: # 防止楼梯间状态

                        self._object_map[env].reset()
                        self._value_map[env].reset()

                        if self._compute_frontiers:
                            self._obstacle_map[env].reset()
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
                    else:
                        print("In all floors, no frontiers found during exploration, stopping.")
                        return self._stop_action.to(masks.device)
            elif  self._obstacle_map[env]._tight_search_thresh == False: # 只有没找到楼梯的才tight
                self._obstacle_map[env]._tight_search_thresh = True
                return TorchActionIDs_plook.MOVE_FORWARD.to(masks.device)
            else:
                print("No frontiers found during exploration, stopping.")
                return self._stop_action.to(masks.device)

        else:
            best_frontier, best_value = self._get_best_frontier_with_llm(observations, frontiers, env) # self._get_best_frontier(observations, frontiers, env)
            if best_value == 0.0:
                if best_frontier == -1:
                    self._climb_stair_over[env] = False
                    self._climb_stair_flag[env] = 1
                    self._stair_frontier[env] = self._obstacle_map[env]._up_stair_frontiers
                    pointnav_action = self._pointnav(observations, self._stair_frontier[env][0], stop=False, env=env)
                    return pointnav_action
                else:
                    self._climb_stair_over[env] = False
                    self._climb_stair_flag[env] = 2
                    self._stair_frontier[env] = self._obstacle_map[env]._down_stair_frontiers
                    pointnav_action = self._pointnav(observations, self._stair_frontier[env][0], stop=False, env=env)
                    return pointnav_action
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
            if self.cur_dis_to_goal[env] <= 0.7 or np.abs(self.cur_dis_to_goal[env] - self.min_distance_xy[env]) < 0.1: # close enough or cannot move forward more #  or self._num_steps[env] == (500 - 1)
                # if self._try_to_navigate_step[env] < 5:
                #     self._called_stop[env] = True
                #     return self._stop_action.to(ori_masks.device)
                # elif self._try_to_navigate_step[env] >= 5  and self._double_check_goal[env] == True:
                #     self._called_stop[env] = True
                #     return self._stop_action.to(ori_masks.device)
                if self._double_check_goal[env] == True: # self._try_to_navigate_step[env] < 5 or 
                    self._called_stop[env] = True
                    # self._obstacle_map[env].visualize_and_save_frontiers() ## for debug
                    return self._stop_action.to(ori_masks.device)
                else:
                    print("Might false positive, change to look for the true goal.")
                    self._object_map[env].clouds = {}
                    self._try_to_navigate[env] = False
                    self._try_to_navigate_step[env] = 0
                    self._object_map[env]._disabled_object_map[self._object_map[env]._map == 1] = 1
                    self._object_map[env]._map.fill(0)
                    action = self._explore(observations, env, ori_masks) # 果断换成探索
                    return action
            else:
                self.min_distance_xy[env] = self.cur_dis_to_goal[env].copy()
                return TorchActionIDs_plook.MOVE_FORWARD.to(ori_masks.device) # force to move forward
        else:        
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
                            self._stair_dilate_flag[env] = False
                            self._climb_stair_over[env] = True
                            self._obstacle_map[env]._has_up_stair = False
                            self._obstacle_map[env]._look_for_downstair_flag = False
                            # self._obstacle_map[env]._climb_stair_paused_step = 0
                        elif  self._climb_stair_flag[env] == 2:
                            self._obstacle_map[env]._disabled_stair_map[self._obstacle_map[env]._down_stair_map == 1] = 1
                            self._obstacle_map[env]._down_stair_frontiers.fill(0)
                            self._climb_stair_flag[env] = 0
                            self._stair_dilate_flag[env] = False
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
                self._obstacle_map[env]._look_for_downstair_flag = False
            elif  self._climb_stair_flag[env] == 2:
                self._obstacle_map[env]._disabled_stair_map[self._obstacle_map[env]._down_stair_map == 1] = 1
                self._obstacle_map[env]._down_stair_map.fill(0)
                self._obstacle_map[env]._down_stair_frontiers = np.array([])
                self._climb_stair_over[env] = True
                self._obstacle_map[env]._has_down_stair = False
                self._obstacle_map[env]._look_for_downstair_flag = False
            self._climb_stair_flag[env] = 0
            self._stair_dilate_flag[env] = False
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
            stair_frontiers = self._stair_frontier[env][0]
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
                self._obstacle_map[env]._look_for_downstair_flag = False
            elif  self._climb_stair_flag[env] == 2:
                self._obstacle_map[env]._disabled_stair_map[self._obstacle_map[env]._down_stair_map == 1] = 1
                self._obstacle_map[env]._down_stair_map.fill(0)
                self._obstacle_map[env]._down_stair_frontiers = np.array([])
                self._obstacle_map[env]._has_down_stair = False
                self._obstacle_map[env]._look_for_downstair_flag = False
            self._climb_stair_flag[env] = 0
            self._stair_dilate_flag[env] = False
        
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
       
        # 先检查是否有优先级最高的frontier
        index = None
        for i, frontier in enumerate(frontiers):
            if np.array_equal(frontier, self._force_frontier[env]):
                index = i
                break
        if index is not None:
            best_frontier = self._force_frontier[env]
            frontier_tuple = tuple(best_frontier)
            best_value = sorted_values[index]
        else:
            self._force_frontier[env] = np.zeros(2)
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

            frontier_tuple = tuple(best_frontier)

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
                ## 如果是第二次选中这个frontier，那么持续导航到这个frontier（直到导航到或者被禁用掉）
                self._frontier_stick_step[env] = 0
                self._last_frontier_distance[env] = 0
                if frontier_tuple in self._obstacle_map[env]._best_frontier_selection_count:
                    self._force_frontier[env] = best_frontier.copy()
            
        # 第一次选中
        if frontier_tuple not in self._obstacle_map[env]._best_frontier_selection_count:
            self._obstacle_map[env]._best_frontier_selection_count[frontier_tuple] = 0
        # 第二次选中
        else:
            self._obstacle_map[env]._best_frontier_selection_count[frontier_tuple] += 1
        
        # 检查选中次数是否超过 40 次，如果超过，则禁用该前沿，但可能导致正常frontier没掉
        # if self._obstacle_map[env]._best_frontier_selection_count[frontier_tuple] >= 40:
        #     self._obstacle_map[env]._disabled_frontiers.add(frontier_tuple)
        #     print(f"Frontier {best_frontier} has been selected {self._obstacle_map[env]._best_frontier_selection_count[frontier_tuple]} times, now disabled.")
        
        self._last_value[env] = best_value
        self._last_frontier[env] = best_frontier
        os.environ["DEBUG_INFO"] += f" Best value: {best_value*100:.2f}%"
        print(f"Now the best_frontier is {best_frontier}")
        return best_frontier, best_value

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray, env: int = 0,
    ) -> Tuple[np.ndarray, List[float]]:
        sorted_frontiers, sorted_values = self._value_map[env].sort_waypoints(frontiers, 0.5)
        sorted_frontiers = [temp for temp in sorted_frontiers if tuple(temp) not in self._obstacle_map[env]._disabled_frontiers]
        return sorted_frontiers, sorted_values
    
    def _get_best_frontier_with_llm(
        self,
        observations: Union[Dict[str, Tensor], "TensorDict"],
        frontiers: np.ndarray,
        env: int = 0,
        topk: int = 3,
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
        
        # 先检查是否有优先级最高的frontier

        index = None
        best_frontier = None
        for i, frontier in enumerate(frontiers):
            if np.array_equal(frontier, self._force_frontier[env]):
                index = i
                break
        if index is not None:
            best_frontier = self._force_frontier[env]
            best_value = sorted_values[index]
            print(f"Force Move.")

        if best_frontier is None:
            self._force_frontier[env] = np.zeros(2)
            # 禁用近邻导航
            # 计算每个前沿与机器人之间的距离
            distances = [np.linalg.norm(frontier - robot_xy) for frontier in sorted_pts]
            
            # 首先筛选出距离小于等于2.0米的前沿
            close_frontiers = [
                (idx, frontier, distance) 
                for idx, (frontier, distance) in enumerate(zip(sorted_pts, distances)) 
                if distance <= 2.0
            ]
        
            if close_frontiers:
                # 如果有多个前沿离机器人都很近,则选择距离最小的前沿
                closest_frontier = min(close_frontiers, key=lambda x: x[2]) # 根据距离排序
                best_frontier_idx = closest_frontier[0]
                best_frontier = sorted_pts[best_frontier_idx]
                best_value = sorted_values[best_frontier_idx]
                print(f"Frontier {best_frontier} is very close (distance: {distances[best_frontier_idx]:.2f}m), selecting it.")

        # If there is no last point pursued, then just take the best point, given that
        # it is not cyclic.
        if best_frontier is None:
        # 如果没有，那么开始决策
            best_frontier_idx = 0
            # 后续得加上楼梯的frontier
            if len(sorted_pts) == 1:
                best_frontier = sorted_pts[best_frontier_idx]
                best_value = sorted_values[best_frontier_idx]
                self._last_value[env] = best_value
                self._last_frontier[env] = best_frontier

            else:
                self.frontier_step_list[env] = []
                frontier_index_list = []
                seen_hashes = set()
                for idx, frontier in enumerate(sorted_pts):
                    floor_num_steps, rgb_hash = self._obstacle_map[env].extract_frontiers_with_image(frontier)
                    # rgb_hash = hash(visualized_rgb.tobytes())  # 计算哈希值
                    if rgb_hash not in seen_hashes: #  and floor_num_steps != 0
                        seen_hashes.add(rgb_hash)
                        self.frontier_step_list[env].append(floor_num_steps)
                        frontier_index_list.append(idx)
                        if len(self.frontier_step_list[env]) == topk:
                            break

                # 后续得加上楼梯的frontier
                target_object_category = self._target_object[env].split("|")[0] # 目前只是单目标导航
                if len(self.frontier_step_list[env]) <= 1:
                    pass
                # 如果有多楼层判断，先做判断， # 然后再判断单层的
                elif self.floor_num[env] > 1 and len(self._object_map_list[env][self._cur_floor_index[env]].this_floor_rooms) >= 5:
                    # 加一个辅助条件，当前层发现大于等于5个房间
                    multi_floor_prompt = self._prepare_multiple_floor_prompt(target_object_category, env)
                    print(multi_floor_prompt)
                    multi_floor_response = self.vlm_itm.chat(None, multi_floor_prompt)
                    self.vlm_response[env] = multi_floor_response
                    print(multi_floor_response)
                    if multi_floor_response == "-1":
                        best_frontier_idx = self.llm_analyze_single_floor(env, target_object_category, frontier_index_list)
                    else:
                        temp_frontier_index = self._extract_multiple_floor_decision(multi_floor_response, env)
                        if temp_frontier_index == 1: # 上楼
                            return -1, 0.0
                        elif temp_frontier_index == 2: # 下楼
                            return -2, 0.0
                        else:
                            best_frontier_idx = self.llm_analyze_single_floor(env, target_object_category, frontier_index_list)  
                else: 
                    best_frontier_idx = self.llm_analyze_single_floor(env, target_object_category, frontier_index_list)  
            best_frontier = sorted_pts[best_frontier_idx]
            best_value = sorted_values[best_frontier_idx]
        
        frontier_tuple = tuple(best_frontier)

        # 防止空气墙或者动作循环,强制更换 frontier
        if np.array_equal(self._last_frontier[env], best_frontier):
            # 检查是否是第一次选中该前沿
            if self._frontier_stick_step[env] == 0:
                # 记录初始的距离(首次选中该前沿时)
                self._last_frontier_distance[env] = np.linalg.norm(best_frontier - robot_xy)
                self._frontier_stick_step[env] += 1
            else:
                # 计算当前与前沿的距离
                current_distance = np.linalg.norm(best_frontier - robot_xy)
                print(f"Distance Change: {np.abs(self._last_frontier_distance[env] - current_distance)} and Stick Step {self._frontier_stick_step[env]}")
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
            ## 如果是第二次选中这个frontier，那么持续导航到这个frontier（直到导航到或者被禁用掉）
            self._frontier_stick_step[env] = 0
            self._last_frontier_distance[env] = 0
            if frontier_tuple in self._obstacle_map[env]._best_frontier_selection_count:
                self._force_frontier[env] = best_frontier.copy()
            
        # 第一次选中
        if frontier_tuple not in self._obstacle_map[env]._best_frontier_selection_count:
            self._obstacle_map[env]._best_frontier_selection_count[frontier_tuple] = 0
        # 第二次选中
        else:
            self._obstacle_map[env]._best_frontier_selection_count[frontier_tuple] += 1

        self._last_value[env] = best_value
        self._last_frontier[env] = best_frontier
        # os.environ["DEBUG_INFO"] += f" Best value: {best_value*100:.2f}%"
        print(f"Now the best_frontier is {best_frontier}")
        return best_frontier, best_value

    # main difference with original vlfm, use yolo_world to replace gdino and yolov7
    def _get_object_detections_with_stair_and_person(self, img: np.ndarray, env: int) -> ObjectDetections:

        target_classes = self._target_object[env].split("|")
        # 尝试增加大量负样本
        # "chair", 
        # "sofa", # 等价于 "couch"
        # "plant", # 等价于 "potted plant"
        # "bed", # - "table", "cabinet"
        # "toilet", # - "wastebin", "washer"
        # "tv_monitor", # "picture"
        base_caption = [["chair"], ["couch"], ["potted plant"], ["bed"], ["toilet"], ["tv"]] \
                        + [["stair"], ["person"], [" "]]          # 去掉 all_caption 中已存在的元素
                        # + [["bathroom"], ["bedroom"], ["dining room"], ["hall"], ["kitchen"], ["laundry room"], ["living room"], ["office"], ["rec room"]] 
        all_caption = base_caption # + [ref_caption for ref_caption in reference_captions if ref_caption not in base_caption]
        all_detections = self._object_detector.predict(image=img, caption=all_caption) # , score_thr=0.05) # default score_thr=0.5, 但楼梯好像有些时候0.5识别不到

        # 定义每个类别的置信度阈值
        # class_conf_thresholds = {
        #     # "stair": 0.3,
        #     "person": 0.5,
        #     **{target_class: 0.3 for target_class in target_classes},
        #     **{cls[0]: 0.3 for cls in all_caption if cls[0] not in ["stair", "person"] + target_classes}
        # }

        class_conf_thresholds = {
            "person": 0.5,
            **{target_class: 0.3 for target_class in target_classes},
            **{cls[0]: 0.3 for cls in all_caption if cls[0] not in ["stair", "person"] + target_classes} #0.05 if cls[0] in reference_rooms else
        }
        # 按类别和置信度阈值过滤检测结果
        all_detections.filter_by_class_and_conf(class_conf_thresholds)

        return all_detections

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
        self._all_object_masks = np.zeros((self._num_envs, height, width), dtype=np.uint8)
        self._object_masks = np.zeros((self._num_envs, height, width), dtype=np.uint8)
        self._person_masks = np.zeros((self._num_envs, height, width), dtype=bool)
        self._stair_masks = np.zeros((self._num_envs, height, width), dtype=bool)
        for env in range(self._num_envs):
            # the observations of the last episode might influnence the 1st step of next episode
            if self._num_steps[env] == 0:
                return
            # for target
            object_map_rgbd = self._observations_cache[env]["object_map_rgbd"]
            rgb, depth, tf_camera_to_episodic, min_depth, max_depth, fx, fy = object_map_rgbd[0]
            all_detections = self._get_object_detections_with_stair_and_person(rgb, env) # get three detections #target_detections, person_detections, stair_detections 
            
            # if np.array_equal(depth, np.ones_like(depth)) and target_detections.num_detections > 0:
            #     depth = self._infer_depth(rgb, min_depth, max_depth)
            #     obs = list(self._observations_cache[env]["object_map_rgbd"][0])
            #     obs[1] = depth
            #     self._observations_cache["object_map_rgbd"][0] = tuple(obs)

            self._object_map[env].each_step_objects[self._obstacle_map[env]._floor_num_steps] = []
            self._object_map[env].each_step_rooms[self._obstacle_map[env]._floor_num_steps] = []
            
            cur_rgb = self._observations_cache[env]["value_map_rgbd"][0][0]
            cur_objs = self._ram.predict(cur_rgb)    
            cur_objs_list = [item.strip() for item in cur_objs.split('|')]  # 分割并清理字符串
            self._object_map[env].each_step_objects[self._obstacle_map[env]._floor_num_steps] = cur_objs_list  # 记录每一步的对象列表
            # 将 list 中的每个元素逐个加入到 set 中
            for obj in cur_objs_list:
                self._object_map[env].this_floor_objects.add(obj)
            pil_image = Image.fromarray(cur_rgb).convert("RGB")
            # 如果图像是 RGBA 模式，转换为 RGB 模式
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            place365_input_img = V(self.place365_centre_crop(pil_image).unsqueeze(0)).to(self.device)
            logit = self.scene_classify_model.forward(place365_input_img)
            h_x = F.softmax(logit, 1).data.squeeze()
            probs, idx = h_x.sort(0, True)
            # scene_color_map = np.array(PLACE365_RGB_COLORS, dtype=np.uint8)
            top_5_indices = idx[:5]  # 取前 5 个索引
            top_5_classes = [self.place365_classes[i] for i in top_5_indices]
            # print(self.place365_classes[idx[0]])
            room_candi_type = self.extract_room_categories(top_5_classes)
            self._object_map[env].each_step_rooms[self._obstacle_map[env]._floor_num_steps] = room_candi_type
            self._object_map[env].this_floor_rooms.add(room_candi_type)

            for idx, class_name in enumerate(all_detections.phrases):
                all_bbox_denorm = all_detections.boxes[idx].int()
                all_object_mask = self._mobile_sam.segment_bbox(rgb, all_bbox_denorm.tolist())
                self._all_object_masks[env][all_object_mask > 0] = 1
                if class_name == self._target_object[env]:
                    # self._object_map[env].each_step_objects[self._obstacle_map[env]._floor_num_steps].append(class_name)
                    # if self._target_object[env] == "bed":
                    #     seg_mask = (self.red_semantic_pred_list[env] == BED_CLASS_ID) # target_class_id
                    #     if np.any(seg_mask):
                    #         close_kernel = np.ones((7, 7), np.uint8)
                    #         seg_mask_bool = cv2.morphologyEx(seg_mask.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel)
                    #         # 计算 object_region
                    #         object_region = all_object_mask.astype(bool)
                    #         seg_mask_bool = seg_mask_bool.astype(bool)
                    #         # 计算在 object_region 区域内 seg_mask 为 True 的位置
                    #         overlap = object_region & seg_mask_bool  # 交集
                    #         if np.any(overlap):
                    #             overlap_percentage = np.sum(overlap) / np.sum(object_region)  # 计算交集占 object_region 的比例
                    #             print(f"The overlap percentage of {self._target_object[env]} is {overlap_percentage}.")
                    #             # 如果比例达到门限(34%),才进行更新
                    #             if overlap_percentage >= 0.34: # 1/3 有点危险
                    #                 fusion_goal_mask = overlap.astype(bool)  # 交集作为融合目标 mask
                    #                 self._object_map[env].update_map(
                    #                     self._target_object[env],  # 更新目标对象
                    #                     depth,
                    #                     fusion_goal_mask.astype(np.uint8),  # 使用交集的目标 mask
                    #                     tf_camera_to_episodic,
                    #                     min_depth,
                    #                     max_depth,
                    #                     fx,
                    #                     fy,
                    #                 )
                    #                 self._object_masks[env][all_object_mask > 0] = 1 # for drawing
                    #                 if self._try_to_navigate[env] == True:
                    #                     self._double_check_goal[env] = True
                    #                     print("Double check success!!!")
                    # elif self._target_object[env] == "tv":
                    #     seg_mask = (self.red_semantic_pred_list[env] == TV_CLASS_ID) # target_class_id
                    #     if np.any(seg_mask):
                    #         close_kernel = np.ones((7, 7), np.uint8)
                    #         seg_mask_bool = cv2.morphologyEx(seg_mask.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel)
                    #         # 计算 object_region
                    #         object_region = all_object_mask.astype(bool)
                    #         seg_mask_bool = seg_mask_bool.astype(bool)
                    #         # 计算在 object_region 区域内 seg_mask 为 True 的位置
                    #         overlap = object_region & seg_mask_bool  # 交集
                    #         if np.any(overlap):
                    #             overlap_percentage = np.sum(overlap) / np.sum(object_region)  # 计算交集占 object_region 的比例
                    #             print(f"The overlap percentage of {self._target_object[env]} is {overlap_percentage}.")
                    #             # 如果比例达到门限(34%),才进行更新
                    #             if overlap_percentage >= 0.34: # 1/3 有点危险
                    #                 fusion_goal_mask = overlap.astype(bool)  # 交集作为融合目标 mask
                    #                 self._object_map[env].update_map(
                    #                     self._target_object[env],  # 更新目标对象
                    #                     depth,
                    #                     fusion_goal_mask.astype(np.uint8),  # 使用交集的目标 mask
                    #                     tf_camera_to_episodic,
                    #                     min_depth,
                    #                     max_depth,
                    #                     fx,
                    #                     fy,
                    #                 )
                    #                 self._object_masks[env][all_object_mask > 0] = 1 # for drawing
                    #                 if self._try_to_navigate[env] == True:
                    #                     self._double_check_goal[env] = True
                    #                     print("Double check success!!!")
                    # elif self._target_object[env] == "couch":
                    #     seg_mask = (self.red_semantic_pred_list[env] == SOFA_CLASS_ID) # target_class_id
                    #     if np.any(seg_mask):
                    #         close_kernel = np.ones((7, 7), np.uint8)
                    #         seg_mask_bool = cv2.morphologyEx(seg_mask.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel)
                    #         # 计算 object_region
                    #         object_region = all_object_mask.astype(bool)
                    #         seg_mask_bool = seg_mask_bool.astype(bool)
                    #         # 计算在 object_region 区域内 seg_mask 为 True 的位置
                    #         overlap = object_region & seg_mask_bool  # 交集
                    #         if np.any(overlap):
                    #             overlap_percentage = np.sum(overlap) / np.sum(object_region)  # 计算交集占 object_region 的比例
                    #             print(f"The overlap percentage of {self._target_object[env]} is {overlap_percentage}.")
                    #             # 如果比例达到门限(34%),才进行更新
                    #             if overlap_percentage >= 0.34: # 1/3 有点危险
                    #                 fusion_goal_mask = overlap.astype(bool)  # 交集作为融合目标 mask
                    #                 self._object_map[env].update_map(
                    #                     self._target_object[env],  # 更新目标对象
                    #                     depth,
                    #                     fusion_goal_mask.astype(np.uint8),  # 使用交集的目标 mask
                    #                     tf_camera_to_episodic,
                    #                     min_depth,
                    #                     max_depth,
                    #                     fx,
                    #                     fy,
                    #                 )
                    #                 self._object_masks[env][all_object_mask > 0] = 1 # for drawing
                    #                 if self._try_to_navigate[env] == True:
                    #                     self._double_check_goal[env] = True
                    #                     print("Double check success!!!")
                    # else:
                    #     self._object_masks[env][all_object_mask > 0] = 1 # for drawing
                    #     self._object_map[env].update_map(
                    #         self._target_object[env], # no phrase, because object_map only record target (especially after update_explored) 
                    #         depth,
                    #         all_object_mask,
                    #         tf_camera_to_episodic,
                    #         min_depth,
                    #         max_depth,
                    #         fx,
                    #         fy,
                    #     )
                    #     if self._try_to_navigate[env] == True:
                    #         self._double_check_goal[env] = True
                    #         print("Double check success!!!")
                    
                    # double check with ram
                    self._object_masks[env][all_object_mask > 0] = 1 # for drawing
                    self._object_map[env].update_map(
                        self._target_object[env], # no phrase, because object_map only record target (especially after update_explored) 
                        depth,
                        all_object_mask,
                        tf_camera_to_episodic,
                        min_depth,
                        max_depth,
                        fx,
                        fy,
                    )
                    if self._try_to_navigate[env] == True and self._double_check_goal[env] == False: # and  class_name in self._object_map[env].each_step_objects[self._obstacle_map[env]._floor_num_steps] ram不认识马桶
                        match_score = self._blip_cosine[env]
                        print(f"Blip2 match score: {match_score}")
                        if match_score >  0.2:
                            self._double_check_goal[env] = True
                            print("Double check success!!!")
                elif class_name == "stair":
                    self._stair_masks[env][all_object_mask > 0] = 1         
                elif class_name == "person":
                    self._person_masks[env][all_object_mask > 0] = 1
                    self._object_map[env].update_movable_map(
                        "person", # no phrase, because object_map only record target (especially after update_explored)
                        depth,
                        all_object_mask,
                        tf_camera_to_episodic,
                        min_depth,
                        max_depth,
                        fx,
                        fy,
                    )
                else:
                    pass
                    # if class_name in reference_rooms:
                    #     # blip2 match to double check
                    #      # class_name
                    #     pass
                    # else:
                    #     self._object_map[env].each_step_objects[self._obstacle_map[env]._floor_num_steps].append(class_name)
                    # self._object_map[env].update_object_map_for_others(
                    #     # self._observations_cache[env]["object_map_rgbd"][0][0],
                    #     self._observations_cache[env]["object_map_rgbd"][0][1],
                    #     self._observations_cache[env]["object_map_rgbd"][0][2],
                    #     self._min_depth,
                    #     self._max_depth,
                    #     self._fx,
                    #     self._fy,
                    #     class_name,
                    #     all_object_mask,
                    #     # self._mobile_sam,
                    #     # all_detections,
                    #     self.red_semantic_pred_list[env],
                    #     self._pitch_angle[env],
                    #     self._climb_stair_over[env],
                    #     self._reach_stair[env],
                    #     self._climb_stair_flag[env],
                    # )
            # max_score = 0
            # room_candi_type = None
            # for room_type in reference_rooms:
            #     all_rgb = [i[0] for i in self._observations_cache[env]["value_map_rgbd"]]            
            #     match_score = self._itm.match(all_rgb[0], f"This image contains {room_type}.",)
            #     if match_score > max_score:
            #         max_score = match_score
            #         room_candi_type = room_type
            # # if match_score >  0.5:
            # self._object_map[env].each_step_rooms[self._obstacle_map[env]._floor_num_steps].append(room_candi_type)

            # final
            cone_fov = get_fov(fx, depth.shape[1])
            # 我用了新的机制禁用掉假阳性，而且update_explored会错误的去掉正确的目标
            # 但还是得启用，不然会误认为远方的东西是max_depth处（实际可能更远）
            self._object_map[env].update_explored(tf_camera_to_episodic, max_depth, cone_fov)
            # detections_list.append(target_detections)
            self.all_detection_list[env] = all_detections
            # self.target_detection_list[env] = target_detections
            # self.coco_detection_list[env] = person_detections
            # self.non_coco_detection_list[env] = stair_detections

    def _visualize_object_map(self):
        for env in range(self._num_envs):
            self._object_map[env].visualize(self._obstacle_map[env]._map, 
                                            self._obstacle_map[env]._up_stair_map, self._obstacle_map[env]._down_stair_map,
                                            self._obstacle_map[env]._frontiers_px,self._obstacle_map[env]._disabled_frontiers_px, 
                                            self._obstacle_map[env].explored_area,self._obstacle_map[env]._up_stair_frontiers_px,
                                            self._obstacle_map[env]._down_stair_frontiers_px)
    
    def llm_analyze_single_floor(self, env, target_object_category, frontier_index_list):
        """
        Analyze the environment using the Vision and Language Model (VLM) to determine the best frontier to explore.

        Parameters:
        env (str): The current environment identifier.
        target_object_category (str): The category of the target object to find.
        frontier_identifiers (list): A list of frontier identifiers (e.g., ["A", "B", "C", "P"]).
        exploration_status (str): A binary string representing the exploration status of each floor.

        Returns:
        str: The identifier of the frontier that is most likely to lead to the target object.
        """
    
        # else, continue to explore on this floor
        prompt = self._prepare_single_floor_prompt(target_object_category, env)

        # Get the visualization of the current environment
        # image = reorient_rescale_map(self._object_map[env].visualization)

        # Analyze the environment using the VLM
        print(prompt)
        response = self.vlm_itm.chat(None, prompt)
        
        self.vlm_response[env] = response
        print(response)
        # Extract the frontier identifier from the response
        if response == "-1":
            temp_frontier_index = 0
        else:
            temp_frontier_index = self._extract_single_floor_decision(response, len(frontier_index_list),env)

        return frontier_index_list[temp_frontier_index]

    def get_room_probabilities(self, target_object_category: str):
        """
        获取目标对象类别在各个房间类型的概率。
        
        :param target_object_category: 目标对象类别
        :return: 房间类型概率字典
        """
        # 定义一个映射表，用于扩展某些目标对象类别的查询范围
        synonym_mapping = {
            "couch": ["sofa"],
            "sofa": ["couch"],
            # 可以根据需要添加更多映射关系
        }

        # 获取目标对象类别及其同义词
        target_categories = [target_object_category] + synonym_mapping.get(target_object_category, [])

        # 如果目标对象类别及其同义词都不在知识图谱中，直接返回空字典
        if not any(category in self.knowledge_graph for category in target_categories):
            return {}

        room_probabilities = {}
        for room in reference_rooms:
            for category in target_categories:
                if self.knowledge_graph.has_edge(category, room):
                    room_probabilities[room] = round(self.knowledge_graph[category][room]['weight'] * 100, 1)
                    break  # 找到一个有效类别后，不再检查其他类别
            else:
                room_probabilities[room] = 0.0
        return room_probabilities

    def get_floor_probabilities(self, df, target_object_category, floor_num):
        """
        获取当前楼层和场景的物体分布概率。

        Parameters:
        df (pd.DataFrame): 包含物体分布概率的表格。
        target_object_category (str): 目标物体类别。
        floor_num (int): 总楼层数。

        Returns:
        dict: 所有相关楼层的物体分布概率。
        """
        if df is None:
            return None

        # 初始化概率字典
        probabilities = {}

        # 如果检测到的楼层数超出了表格的范围，展示所有已知多楼层场景的概率
        if floor_num > 4:  # 假设表格最多支持 4 层
            logging.warning(f"Floor number {floor_num} exceeds the maximum supported floor number (4). Showing probabilities for all known multi-floor scenarios.")
            for known_floor_num in range(2, 5):  # 2 层、3 层、4 层
                for floor in range(1, known_floor_num + 1):
                    column_name = f"train_floor{known_floor_num}_{floor}"
                    if column_name in df.columns:
                        prob = df.set_index("category").at[target_object_category, column_name]
                        probabilities[f"{known_floor_num}-Floor Scenario, Floor {floor}"] = prob
                    else:
                        logging.warning(f"Column {column_name} not found in the probability table.")
                        probabilities[f"{known_floor_num}-Floor Scenario, Floor {floor}"] = 0.0  # 默认概率为 0
        else:
            # 展示当前检测到的楼层数的概率
            for floor in range(1, floor_num + 1):
                column_name = f"train_floor{floor_num}_{floor}"
                if column_name in df.columns:
                    prob = df.set_index("category").at[target_object_category, column_name]
                    probabilities[f"Floor {floor}"] = prob
                else:
                    logging.warning(f"Column {column_name} not found in the probability table.")
                    probabilities[f"Floor {floor}"] = 0.0  # 默认概率为 0

        return probabilities

    def _prepare_single_floor_prompt(self, target_object_category, env):
        """
        Prepare the prompt for the LLM in a single-floor scenario.

        Parameters:
        env (int): The current environment identifier.
        target_object_category (str): The category of the target object to find.

        Returns:
        str: The prepared prompt for the LLM.
        """
        num_areas = len(self.frontier_step_list[env])
        area_descriptions = []
        self.frontier_rgb_list[env] = []

        # 生成区域描述
        for i, step in enumerate(self.frontier_step_list[env]):
            try:
                room = self._object_map[env].each_step_rooms[step] or "unknown room"
                objects = self._object_map[env].each_step_objects[step] or "no visible objects"
                if isinstance(objects, list):
                    objects = ", ".join(objects)
                self.frontier_rgb_list[env].append(self._obstacle_map[env]._each_step_rgb[step])
                area_description = f"Area {i + 1}: a {room} containing: {objects}.\n"
                area_descriptions.append(area_description)
            except (IndexError, KeyError) as e:
                logging.warning(f"Error accessing room or objects for step {step}: {e}")
                continue

        area_descriptions_str = ''.join(area_descriptions)

        # 获取房间-对象关联概率
        room_probabilities = self.get_room_probabilities(target_object_category)

        # 1. 角色定义
        role_definition = (
            "You are an AI assistant with advanced spatial reasoning capabilities. "
            "Your task is to navigate an indoor environment to find a target object.\n"
        )

        # 2. 任务描述
        task_description = (
            "Your goal is to analyze the following areas and determine the most promising one to explore first. "
            "Your decision should be based on:\n"
            "- The target object category.\n"
            "- The environmental context of each area.\n"
            "- The probabilities of finding the target object in different room types.\n"
        )

        # 3. 步骤说明
        steps_guidance = (
            "Follow these steps to make your decision:\n"
            "1. **Analyze**: Examine the target object category and its typical locations.\n"
            "2. **Compare**: Evaluate each area's environmental context and its match with the target object.\n"
            "3. **Decide**: Use the room probabilities to select the most promising area.\n"
        )

        # 4. 示例增强（包含概率使用案例）
        examples = (
            "Examples:\n"
            "Example One Input - High Probability Case: Number of Candidate Areas: 3. Target Object Category: toilet.\n"
            "Example One Area Description:\n"
            "Area 1: a bathroom containing: shower, towels.\n"
            "Area 2: a bedroom containing: bed, nightstand.\n"
            "Area 3: a garage containing: car.\n"
            "Example One Room Probabilities:\n"
            "- Bathroom: 90.0%\n"
            "- Bedroom: 10.0%\n"
            "Example One Output: 1. Reason: Shower/towels in Bathroom indicate toilet location, with high probability (90.0%).\n\n"
            "Example Two Input - Feature Dominant Case: Number of Candidate Areas: 2. Target Object Category: book.\n"
            "Example Two Area Description:\n"
            "Area 1: a bedroom containing: bed, nightstand.\n"
            "Area 2: a living room containing: sofa, bookshelf.\n"
            "Example Two Room Probabilities:\n"
            "- Bedroom: 70.0%\n"
            "- Living room: 30.0%\n"
            "Example Two Output: 2. Reason: Bookshelf in Living Room suggests book storage, despite lower probability (30.0%).\n\n"
            "Example Three Input - Uncertain Case: Number of Candidate Areas: 2. Target Object Category: toilet.\n"
            "Example Three Area Description:\n"
            "Area 1: a hall containing: clock, coat rack, umbrella stand.\n"
            "Area 2: a kitchen containing: sink, plumbing fixtures, cleaning supplies.\n"
            "Example Three Room Probabilities:\n"
            "- Bathroom: 97.0%\n"
            "- Laundry Room: 0.6%\n"
            "Example Three Output: 1. Reason: It seems that both areas are not closely related to the target object, so the first one is selected by default.\n\n"
        )

        # 5. 真实输入
        true_input = (
            f"True Input: Number of Candidate Areas: {num_areas}. Target Object Category: {target_object_category}.\n"
        )

        # 6. 区域描述
        area_context = (
            "True Area Description:\n"
            f"{area_descriptions_str}\n"
        )

        # 7. 房间概率信息（按概率降序排列）
        true_input += "True Room Probabilities:\n"
        sorted_probs = sorted(room_probabilities.items(), key=lambda x: x[1], reverse=True)
        for room, prob in sorted_probs:
            true_input += f"- {room}: {prob:.1f}%\n"

        # 8. 输出约束
        output_constraints = (
            "True Output Requirements:\n"
            f"- Area Number must select from areas: {', '.join(map(str, range(1, num_areas + 1)))}\n"
            "- Reason must:\n"
            "  a) Reference the target object category.\n"
            "  b) Mention the room type and its environmental context.\n"
            "  c) Consider the room probability, even if it is lower, if the context strongly suggests the presence of the target object.\n"
            "  d) Follow the format: [Area Number]. Reason: [Context].\n"
        )

        # 构建完整 prompt
        prompt = (
            role_definition +
            task_description +
            steps_guidance +
            examples +
            true_input +
            area_context +
            output_constraints
        )
        return prompt

    def _prepare_multiple_floor_prompt(self, target_object_category, env):
        """
        多楼层决策提示生成（兼容单楼层风格）
        """
        # =============== 基础数据准备 ===============
        current_floor = self._cur_floor_index[env] + 1 # 从1开始
        total_floors = self.floor_num[env]
        floor_probs = self.get_floor_probabilities(floor_probabilities_df, target_object_category, total_floors)
        room_probs = self.get_room_probabilities(target_object_category)

        # =============== 楼层特征描述 ===============
        floor_descriptions = []
        for floor in range(total_floors):
            try:
                # 获取楼层特征
                rooms = self._object_map_list[env][floor].this_floor_rooms or {"Unknown"}
                objects = self._object_map_list[env][floor].this_floor_objects or {"Unknown"}
                # 将 set 转换为字符串（以逗号分隔）
                rooms_str = ", ".join(rooms)
                objects_str = ", ".join(objects)
                # 构建描述
                desc = (
                    f"Floor {floor + 1}:\n"
                    f"- Status: {'Current' if floor + 1 == current_floor else 'Other'}\n"
                    f"- Have Explored: {self._obstacle_map_list[env][floor]._done_initializing}.\n"
                    f"- Fully Explored: {self._obstacle_map_list[env][floor]._this_floor_explored}.\n"
                    f"- Detected Rooms: {rooms_str}\n"
                    f"- Detected Objects:  {objects_str}\n"
                    f"- Floor Probability: {floor_probs.get(f'Floor {floor+1}', 0.0):.1f}%\n"
                )
                floor_descriptions.append(desc)
            except Exception as e:
                logging.error(f"Error describing floor {floor}: {e}")
                continue

        # =============== 提示组件 ===============
        # 1. 角色定义（与单楼层保持风格一致）
        role_definition = (
            "You are an AI assistant with advanced vertical navigation capabilities. "
            "Your task is to decide floor transition in a multi-floor environment.\n"
        )

        # 2. 任务描述（新增楼层维度）
        task_description = (
            "Analyze floor characteristics and decide whether to:\n"
            "0. Stay on current floor\n"
            "1. Go upstairs\n"
            "2. Go downstairs\n\n"
            "Decision factors must include:\n"
            f"- Target object category\n"
            "- Floor probability distribution\n"
            "- Detected room types\n"
            "- Exploration progress\n"
        )

        # 3. 分析步骤（扩展楼层分析）
        steps_guidance = (
            "Decision steps:\n"
            "1. Compare floor probabilities vertically\n"
            "2. Check if current floor has unexplored high-probability rooms\n"
            "3. Select optimal movement direction\n\n"
        )

        # 4. 多楼层示例（完全对齐真实输入结构）
        examples = (
            "Examples:\n"
            
            "Example One - Stay Decision:\n"
            "Example One Environment:\n"
            "Example One Target Object Category: toilet.\n"
            "Example One Total Floors: 2\n"
            "Example One Current Floor: 1\n"
            "Example One Floor Characteristics:\n"
            "Floor 1:\n"
            "- Status: Current\n"
            "- Have Explored: True\n"
            "- Fully Explored: False\n"
            "- Detected Rooms: bathroom, hallway\n"
            "- Detected Objects: shower, towels, sink\n"
            "- Floor Probability: 45.3%\n"
            "Floor 2:\n"
            "- Status: Other\n"
            "- Have Explored: False\n"
            "- Fully Explored: False\n"
            "- Detected Rooms: bedroom\n"
            "- Detected Objects: bed, nightstand\n"
            "- Floor Probability: 54.7%\n"
            "Example One Room Probabilities:\n"
            "- bathroom: 90.0%\n"
            "- bedroom: 10.0%\n"
            "Example One Output: 0. Reason: Current floor contains bathroom (90% room probability) with shower/towels, "
            "despite lower floor probability (45.3% vs 54.7%).\n"
            
            "Example Two - Upstairs Decision:\n"
            "Example Two Environment:\n"
            "Example Two Target Object Category: bed.\n"
            "Example Two Total Floors: 3\n"
            "Example Two Current Floor: 2\n"
            "Example Two Floor Characteristics:\n"
            "Floor 1:\n"
            "- Status: Other\n"
            "- Have Explored: True\n"
            "- Fully Explored: False\n"
            "- Detected Rooms: bathroom, hallway\n"
            "- Detected Objects: shower, towels, sink\n"
            "- Floor Probability: 15.3%\n"
            "Floor 2:\n"
            "- Status: Current\n"
            "- Have Explored: True\n"
            "- Fully Explored: False\n"
            "- Detected Rooms: office\n"
            "- Detected Objects: desk, computer\n"
            "- Floor Probability: 22.2%\n"
            "Floor 3:\n"
            "- Status: Other\n"
            "- Have Explored: False\n"
            "- Fully Explored: False\n"
            "- Detected Rooms: Unknown\n"
            "- Detected Objects: Unknown\n"
            "- Floor Probability: 48.1%\n"
            "Example Two Room Probabilities:\n"
            "- bedroom: 85.0%\n"
            "- office: 6.0%\n"
            "- bathroom: 4.0%\n"
            "Example Two Output: 1. Reason: Upper floor shows higher floor probability (48.1% vs 22.2%) with detected bathroom on Floor 1"
            "(4.0% room probability) and detected office on Floor 2 (6.0%) showing much lower room probability.\n"
            
            "Example Three - Downstairs Decision:\n"
            "Example Three Environment:\n"
            "Example Three Target Object Category: fire_extinguisher.\n"
            "Example Three Total Floors: 2\n"
            "Example Three Current Floor: 2\n"
            "Example Three Floor Characteristics:\n"
            "Floor 1:\n"
            "- Status: Other\n"
            "- Have Explored: False\n"
            "- Fully Explored: False\n"
            "- Detected Rooms: Unknown\n"
            "- Detected Objects: Unknown\n"
            "- Floor Probability: 50.0%\n"
            "Floor 2:\n"
            "- Status: Current\n"
            "- Have Explored: True\n"
            "- Fully Explored: False\n"
            "- Detected Rooms: living_room\n"
            "- Detected Objects: sofa, tv\n"
            "- Floor Probability: 5.0%\n"
            "Example Three Room Probabilities:\n"
            "- garage: 75.0%\n"
            "- hall: 15.0%\n"
            "- living_room: 5.0%\n"
            "Example Three Output: 2. Reason: Current floor shows lower floor probability compared to the lower floor (5.0% vs 50.0%)\n\n"

            "Example Four - Uncertain Case:\n"
            "Example Four Environment:\n"
            "Example Four Target Object Category: toilet.\n"
            "Example Four Total Floors: 2\n"
            "Example Four Current Floor: 2\n"
            "Example Four Floor Characteristics:\n"
            "Floor 1:\n"
            "- Status: Other\n"
            "- Have Explored: False\n"
            "- Fully Explored: False\n"
            "- Detected Rooms: Unknown\n"
            "- Detected Objects: Unknown\n"
            "- Floor Probability: 33.3%\n"
            "Floor 2:\n"
            "- Status: Current\n"
            "- Have Explored: True\n"
            "- Fully Explored: False\n"
            "- Detected Rooms: living_room\n"
            "- Detected Objects: sofa, tv\n"
            "- Floor Probability: 33.3%\n"
            "Floor 3:\n"
            "- Status: Other\n"
            "- Have Explored: False\n"
            "- Fully Explored: False\n"
            "- Detected Rooms: Unknown\n"
            "- Detected Objects: Unknown\n"
            "- Floor Probability: 33.3%\n"
            "Example Four Room Probabilities:\n"
            "- bathroom: 75.0%\n"
            "- hall: 15.0%\n"
            "Example Four Output: 2. Reason: close floor probabilities (~33%) and irrelevant room probabilities warrant prioritizing continued exploration on current floor.\n\n"
        )

        # 5. 真实输入数据
        true_input = (
            f"True Environment:\n"
            f"Target Object Category: {target_object_category}.\n"
            f"Total Floors: {total_floors}\n"
            f"Current Floor: {current_floor}\n"
            "Floor Characteristics:\n"
            f"{''.join(floor_descriptions)}\n"
            "Room Probabilities:\n"
            f"{self._format_probs(room_probs)}\n"
        )

        # 6. 输出约束（严格数字模式）
        # =============== 动态调整输出约束 ===============
        # 根据是否有上楼或下楼的选项，动态调整允许的输出约束
        valid_options = ["0"]  # 默认总是包含 "0. Stay on current floor"
        if self._obstacle_map[env]._has_up_stair:
            valid_options.append("1")  # 如果有上楼选项，添加 "1. Go upstairs"
        if self._obstacle_map[env]._has_down_stair:
            valid_options.append("2")  # 如果有下楼选项，添加 "2. Go downstairs"

        # 根据有效选项生成输出约束
        output_constraints = (
            "True Output Requirements:\n"
            "- Option Number Must be one of the following options: {}\n".format(", ".join(valid_options)) +
            "- Reason must:\n"
            "  a) Reference floor probabilities.\n"
            "  b) Mention key detected rooms and objects.\n"
            "  c) Consider the room probability, even if it is lower, if the context strongly suggests the presence of the target object.\n"
            "- d) Follow the format: [Option Number]. Reason: [Context].\n" 
        )
        # =============== 组合完整提示 ===============
        prompt = (
            role_definition +
            task_description +
            steps_guidance +
            examples +
            true_input +
            output_constraints
        )
        
        return prompt

    def _format_probs(self, prob_dict):
        """概率格式化工具（复用单楼层风格）"""
        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        return "\n".join([f"- {k}: {v:.1f}%" for k, v in sorted_probs])

    
    def _extract_single_floor_decision(self, response, num_areas, env):
        """
        Extract the frontier identifier from the VLM response.

        Parameters:
        response (str): The response from the VLM.
        num_areas (int): The number of candidate areas.

        Returns:
        int: The identifier of the frontier that is most likely to lead to the target object.
            如果解析失败或格式不符合预期，返回默认值 0。
        """
        # 确保 response 是字符串
        if not isinstance(response, str) or len(response.strip()) == 0:
            logging.warning("Invalid response input")
            return 0

        # 分割字符串
        parts = response.split('. Reason: ')
        
        # 检查 parts 是否为空或不符合预期
        if not parts or len(parts) < 1:
            return 0

        # 提取 frontier 部分
        frontier_str = parts[0].strip()  # 去除前后空白字符

        # 尝试将 frontier_str 转换为整数
        try:
            frontier = int(frontier_str)
        except ValueError:
            # 如果转换失败，返回默认值 0
            return 0

        # 检查 frontier 是否在有效范围内
        valid_areas = list(range(1, num_areas + 1))

        if frontier not in valid_areas:
            return 0

        return frontier - 1

    def _extract_multiple_floor_decision(self, response, env) -> int:
        """
        从VLM响应中提取多楼层决策
        
        参数:
            response (str): VLM的原始响应文本
            current_floor (int): 当前楼层索引 (0-based)
            total_floors (int): 总楼层数
            
        返回:
            int: 楼层决策 0/1/2，解析失败返回0
        """
        # 防御性输入检查
        if not isinstance(response, str) or len(response.strip()) == 0:
            logging.warning("Invalid response input")
            return 0
        
        # 分割字符串
        parts = response.split('. Reason: ')
        
        # 检查 parts 是否为空或不符合预期
        if not parts or len(parts) < 1:
            return 0
        
                # 提取 frontier 部分
        frontier_str = parts[0].strip()  # 去除前后空白字符

        # 尝试将 frontier_str 转换为整数
        try:
            frontier = int(frontier_str)
        except ValueError:
            # 如果转换失败，返回默认值 0
            return 0

        # 检查 frontier 是否在有效范围内
        valid_areas = [0, 1, 2] 
        if self._obstacle_map[env]._has_up_stair == False:
            valid_areas.remove(1)
        elif self._obstacle_map[env]._has_down_stair == False:
            valid_areas.remove(2)

        if frontier not in valid_areas:
            return 0

        return frontier

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
