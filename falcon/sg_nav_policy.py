from typing import Any, Dict, List, Optional, Tuple

import gym.spaces as spaces
from habitat.core.spaces import ActionSpace
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo.policy import Policy, PolicyActionData
import torch.nn as nn
import torch
import numpy as np

import copy
import math
import os
from matplotlib import colors
import cv2
import pandas
import skimage
import sys
sys.path.insert(0, '/home/zeyingg/github/habitat-lab-vlfm/falcon/GLIP')

from maskrcnn_benchmark.config import cfg as glip_cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo

from pslpython.model import Model as PSLModel
from pslpython.partition import Partition
from pslpython.predicate import Predicate
from pslpython.rule import Rule

sys.path.insert(0, '/home/zeyingg/github/habitat-lab-vlfm/falcon')
from scenegraph import SceneGraph

sys.path.insert(0, '/home/zeyingg/github/habitat-lab-vlfm/falcon/utils')
# import utils.utils_fmm.control_helper as CH
# import utils.utils_fmm.pose_utils as pu
from .utils.utils_fmm import control_helper as CH
from .utils.utils_fmm import pose_utils as pu

from .utils.utils_fmm.fmm_planner import FMMPlanner    
from .utils.utils_fmm.mapping import Semantic_Mapping
from .utils.utils_glip import *
from .utils.image_process import (
    add_resized_image,
    add_rectangle,
    add_text,
    add_text_list,
    crop_around_point,
    draw_agent,
    draw_goal,
    line_list
)

from omegaconf import DictConfig
from vlfm.policy.habitat_policies import VLFMPolicyConfig, HabitatMixin # BaseObjectNavPolicy base_objectnav_policy

from vlfm.policy.base_policy import BasePolicy
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from vlfm.policy.habitat_policies import HM3D_ID_TO_NAME, MP3D_ID_TO_NAME
import imageio

class TorchActionIDs_plook:
    STOP = torch.tensor([[0]], dtype=torch.long)
    MOVE_FORWARD = torch.tensor([[1]], dtype=torch.long)
    TURN_LEFT = torch.tensor([[2]], dtype=torch.long)
    TURN_RIGHT = torch.tensor([[3]], dtype=torch.long)
    LOOK_UP = torch.tensor([[4]], dtype=torch.long)
    LOOK_DOWN = torch.tensor([[5]], dtype=torch.long)
    
@baseline_registry.register_policy
class SG_Nav_Policy(BasePolicy):
    def __init__(
        self,
        # config,
        # full_config,
        # observation_space: spaces.Space,
        # action_space: ActionSpace,
        # orig_action_space: ActionSpace,
        # num_envs: int,
        # aux_loss_config,
        # agent_name: Optional[str],
        # **kwargs: Any,
        *args: Any,
        **kwargs: Any,
    ):
        # Policy.__init__(self, action_space)
        # nn.Module.__init__(self)
        # if "action_space" not in kwargs:
        #     raise ValueError("Missing required argument: action_space")
        # nn.Module.__init__(self)
        # Policy.__init__(self, kwargs["action_space"])
        super().__init__(*args, **kwargs)
        self.panoramic = []
        self.panoramic_depth = []
        self.turn_angles = 0
        self.device = (
            torch.device("cuda:{}".format(0))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.prev_action = 0
        self.navigate_steps = 0
        self.move_steps = 0
        self.total_steps = 0
        self.found_goal = False
        self.found_goal_times = 0
        self.threshold_list = {'bathtub': 3, 'bed': 3, 'cabinet': 2, 'chair': 1, 'chest_of_drawers': 3, 'clothes': 2, 'counter': 1, 'cushion': 3, 'fireplace': 3, 'gym_equipment': 2, 'picture': 3, 'plant': 3, 'seating': 0, 'shower': 2, 'sink': 2, 'sofa': 2, 'stool': 2, 'table': 1, 'toilet': 3, 'towel': 2, 'tv_monitor': 0}
        self.found_goal_times_threshold = 3
        self.distance_threshold = 5
        self.correct_room = False
        self.changing_room = False
        self.changing_room_steps = 0
        self.move_after_new_goal = False
        self.former_check_step = -10
        self.goal_disappear_step = 100
        self.force_change_room = False
        self.current_room_search_step = 0
        self.target_room = ''
        self.current_rooms = []
        self.nav_without_goal_step = 0
        self.former_collide = 0
        self.history_pose = []
        self.visualize_image_list = []
        self.count_episodes = -1
        self.loop_time = 0
        self.stuck_time = 0
        self.rooms = rooms
        self.rooms_captions = rooms_captions
        # self.split = (self.args.split_l >= 0)
        self.metrics = {'distance_to_goal': 0., 'spl': 0., 'softspl': 0.}

        ### ------ init glip model ------ ###
        config_file = "falcon/GLIP/configs/pretrain/glip_Swin_L.yaml" 
        weight_file = "falcon/GLIP/MODEL/glip_large_model.pth"
        # config_file = "GLIP/configs/pretrain/glip_Swin_T_O365_GoldG.yaml"
        # weight_file = "GLIP/MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth"
        glip_cfg.local_rank = 0
        glip_cfg.num_gpus = 1
        glip_cfg.merge_from_file(config_file) 
        glip_cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
        glip_cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
        self.glip_demo = GLIPDemo(
            glip_cfg,
            min_image_size=800,
            confidence_threshold=0.61,
            show_mask_heatmaps=False
        )
        print('glip init finish!!!')

        ### ----- init some static variables ----- ###
        self.map_size_cm = 4000
        self.resolution = self.map_resolution = 5
        self.camera_horizon = 0
        self.dilation_deg = 0
        self.collision_threshold = 0.08
        self.col_width = 5
        self.selem = skimage.morphology.square(1)
        self.explanation = ''
        
        ### ----- init maps ----- ###
        self.init_map()

        ##
        self._image_height = kwargs["image_height"]
        self._image_width = kwargs["image_width"]
        self._camera_height = kwargs["camera_height"]
        camera_fov_rad = np.deg2rad(kwargs["camera_fov"])
        self._camera_fov = camera_fov_rad
        self._camera_fov_ori = kwargs["camera_fov"]
        self._num_envs = kwargs['num_envs']
        self._did_reset = [False for _ in range(self._num_envs)]
        self._target_object = ["" for _ in range(self._num_envs)]
        self._dataset_type = kwargs["dataset_type"]
        self._policy_info = []
        self.obj_locations = [[] for i in range(21)] # length equal to all the objects in reference matrix 
        self.found_long_goal = False
        self.ever_long_goal = False
        self.goal_gps = np.array([0.,0.])
        self.last_gps = np.array([11100.,11100.])
        self.using_random_goal = False
        self.move_since_random = 0
        self.not_move_steps = 0
        self.goal_loc = None
        self.min_depth = kwargs["min_depth"]
        self.max_depth = kwargs["max_depth"]
        self.first_fbe = False
        self.fronter_this_ex = 0
        self.random_this_ex = 0
        self.long_goal_temp_gps = np.array([0.,0.])
        self.has_panarama = False
        self.last_loc = self.full_pose
        self.dist_to_frontier_goal = 10
        self.current_obj_predictions = []
        self.detect_true = False
        self.goal_appear = False
        self.frontiers_gps = []
        self.last_location = np.array([0.,0.])
        self.current_stuck_steps = 0
        self.total_stuck_steps = 0
        ##

        self.sem_map_module = Semantic_Mapping(self).to(self.device) 
        self.free_map_module = Semantic_Mapping(self, max_height=10,min_height=-150).to(self.device)
        self.room_map_module = Semantic_Mapping(self, max_height=200,min_height=-10, num_cats=9).to(self.device)
        
        self.free_map_module.eval()
        self.free_map_module.set_view_angles(self.camera_horizon)
        self.sem_map_module.eval()
        self.sem_map_module.set_view_angles(self.camera_horizon)
        self.room_map_module.eval()
        self.room_map_module.set_view_angles(self.camera_horizon)

        self.camera_matrix = self.free_map_module.camera_matrix
        
        print('FMM navigate map init finish!!!')
        
        ### ----- load commonsense from LLMs ----- ###
        self.goal_idx = {}
        for key in projection:
            self.goal_idx[projection[key]] = categories_21.index(projection[key]) # each goal corresponding to which column in co-orrcurance matrix 
        self.co_occur_mtx = np.load('tools/obj.npy')
        self.co_occur_mtx -= self.co_occur_mtx.min()
        self.co_occur_mtx /= self.co_occur_mtx.max() 
        
        self.co_occur_room_mtx = np.load('tools/room.npy')
        self.co_occur_room_mtx -= self.co_occur_room_mtx.min()
        self.co_occur_room_mtx /= self.co_occur_room_mtx.max()
        
        ### ----- option: using PSL optimization ADMM ----- ###
        # if self.args.PSL_infer:
        self.psl_model = PSLModel('objnav1')  ## important: please use different name here for different process in the same machine. eg. objnav, objnav2, ...
        # Add Predicates
        self.add_predicates(self.psl_model)

        # Add Rules
        self.add_rules(self.psl_model)

        ### ----- load scene graph module ----- ###
        self.scenegraph = SceneGraph(map_resolution=self.map_resolution, map_size_cm=self.map_size_cm, map_size=self.map_size, camera_matrix=self.camera_matrix)

        # self.experiment_name = 'test_2'

        # if self.split:
        #     self.experiment_name = self.experiment_name + f'/[{self.args.split_l}:{self.args.split_r}]'
        self.visualize_option = kwargs["visualize"]
        self.visualization_dir = f'debug/20250211/v5' # {self.experiment_name}/

        print('scene graph module init finish!!!')


    def add_predicates(self, model):
        """
        add predicates for ADMM PSL inference
        """
        # if self.args.reasoning in ['both', 'obj']:

        predicate = Predicate('IsNearObj', closed = True, size = 2)
        model.add_predicate(predicate)
        
        predicate = Predicate('ObjCooccur', closed = True, size = 1)
        model.add_predicate(predicate)
        # if self.args.reasoning in ['both', 'room']:

        predicate = Predicate('IsNearRoom', closed = True, size = 2)
        model.add_predicate(predicate)
        
        predicate = Predicate('RoomCooccur', closed = True, size = 1)
        model.add_predicate(predicate)
        
        predicate = Predicate('Choose', closed = False, size = 1)
        model.add_predicate(predicate)
        
        predicate = Predicate('ShortDist', closed = True, size = 1)
        model.add_predicate(predicate)
        
    def add_rules(self, model):
        """
        add rules for ADMM PSL inference
        """
        # if self.args.reasoning in ['both', 'obj']:
        model.add_rule(Rule('2: ObjCooccur(O) & IsNearObj(O,F)  -> Choose(F)^2'))
        model.add_rule(Rule('2: !ObjCooccur(O) & IsNearObj(O,F) -> !Choose(F)^2'))
        # if self.args.reasoning in ['both', 'room']:
        model.add_rule(Rule('2: RoomCooccur(R) & IsNearRoom(R,F) -> Choose(F)^2'))
        model.add_rule(Rule('2: !RoomCooccur(R) & IsNearRoom(R,F) -> !Choose(F)^2'))
        model.add_rule(Rule('2: ShortDist(F) -> Choose(F)^2'))
        model.add_rule(Rule('Choose(+F) = 1 .'))
    
    def _reset(self, env = 0):
        """
        reset variables for each episodes
        """
        self.navigate_steps = 0
        self.turn_angles = 0
        self.move_steps = 0
        self.total_steps = 0
        self.current_room_search_step = 0
        self.found_goal = False
        self.found_goal_times = 0
        self.ever_long_goal = False
        self.correct_room = False
        self.changing_room = False
        self.goal_loc = None
        self.changing_room_steps = 0
        self.move_after_new_goal = False
        self.former_check_step = -10
        self.goal_disappear_step = 100
        self.prev_action = 0
        self.col_width = 5
        self.former_collide = 0
        self.goal_gps = np.array([0.,0.])
        self.long_goal_temp_gps = np.array([0.,0.])
        self.last_gps = np.array([11100.,11100.])
        self.has_panarama = False
        self.init_map()
        self.last_loc = self.full_pose
        self.panoramic = []
        self.panoramic_depth = []
        self.current_rooms = []
        self.dist_to_frontier_goal = 10
        self.first_fbe = False
        self.goal_map = np.zeros(self.full_map.shape[-2:])
        self.found_long_goal = False
        self.history_pose = []
        self.visualize_image_list = []
        self.count_episodes = self.count_episodes + 1
        self.loop_time = 0
        self.stuck_time = 0
        self.metrics = {'distance_to_goal': 0., 'spl': 0., 'softspl': 0.}
        # self.obj_goal = self.simulator._env.current_episode.object_category
        ###########
        self.current_obj_predictions = []
        self.obj_locations = [[] for i in range(21)] # length equal to all the objects in reference matrix 
        self.not_move_steps = 0
        self.move_since_random = 0
        self.using_random_goal = False
        
        self.fronter_this_ex = 0
        self.random_this_ex = 0
        ########### error analysis
        self.detect_true = False
        self.goal_appear = False
        self.frontiers_gps = []
        
        self.last_location = np.array([0.,0.])
        self.current_stuck_steps = 0
        self.total_stuck_steps = 0
        self.explanation = ''
        self.text_node = ''
        self.text_edge = ''

        self.scenegraph.reset()

        ##
        self.obj_locations = [[] for i in range(21)] # length equal to all the objects in reference matrix 
        self.found_long_goal = False
        self.found_goal_times = 0
        self.ever_long_goal = False
        self.last_gps = np.array([11100.,11100.])
        self.using_random_goal = False
        self._did_reset[env] = True
        ##
        
    def detect_objects(self, observations, env = 0):
        """
        detect objects from current observations and update semantic map.
        """
        self.current_obj_predictions = self.glip_demo.inference(observations["rgb"][:,:,[2,1,0]], object_captions) # GLIP object detection, time cosuming
        new_labels = self.get_glip_real_label(self.current_obj_predictions) # transfer int labels to string labels
        self.current_obj_predictions.add_field("labels", new_labels)

        
        shortest_distance = 120
        shortest_distance_angle = 0
        goal_prediction = copy.deepcopy(self.current_obj_predictions)
        obj_labels = self.current_obj_predictions.get_field("labels")
        goal_bbox = []
        ### save the bounding boxes if there is a goal object
        for j, label in enumerate(obj_labels):
            if self._target_object[env] in label:
                goal_bbox.append(self.current_obj_predictions.bbox[j])
            elif self._target_object[env] == 'gym_equipment' and (label in ['treadmill', 'exercise machine']):
                goal_bbox.append(self.current_obj_predictions.bbox[j])
        
        ### record the location of object center in the semantic map for object reasoning.
        # if self.args.reasoning == 'both' or 'obj':
        for j, label in enumerate(obj_labels):
            if label in categories_21_origin:
                confidence = self.current_obj_predictions.get_field("scores")[j]
                bbox = self.current_obj_predictions.bbox[j].to(torch.int64)
                center_point = (bbox[:2] + bbox[2:]) // 2
                temp_direction = (center_point[0] - 320) * 79 / 640
                temp_distance = self.depth[center_point[1],center_point[0],0]
                if temp_distance >= 4.999:
                    continue
                obj_gps = self.get_goal_gps(observations, temp_direction, temp_distance)
                x = int(self.map_size_cm/10-obj_gps[1]*100/self.resolution)
                y = int(self.map_size_cm/10+obj_gps[0]*100/self.resolution)
                self.obj_locations[categories_21_origin.index(label)].append([confidence, x, y])
        
        ### if detect a goal object, determine if it's beyond 5 meters or not. 
        if len(goal_bbox) > 0:
            long_goal_detected_before = copy.deepcopy(self.found_long_goal)
            goal_prediction.bbox = torch.stack(goal_bbox)
            for box in goal_prediction.bbox:  ## select the closest goal as the detected goal
                box = box.to(torch.int64)
                center_point = (box[:2] + box[2:]) // 2
                temp_direction = (center_point[0] - 320) * 79 / 640
                temp_distance = self.depth[center_point[1],center_point[0],0]
                k = 0
                pos_neg = 1
                ## case that a detected goal is within 0.5 meters, maybe it's because the image is corrupted, let's find another points in the image instead of the center point
                while temp_distance >= 100 and 0<center_point[1]+int(pos_neg*k)<479 and 0<center_point[0]+int(pos_neg*k)<639:
                    pos_neg *= -1
                    k += 0.5
                    temp_distance = max(self.depth[center_point[1]+int(pos_neg*k),center_point[0],0],
                    self.depth[center_point[1],center_point[0]+int(pos_neg*k),0])
                    
                if temp_distance >= 4.999:
                    self.found_long_goal = True
                    self.ever_long_goal = True
                else:
                    if self.found_goal:
                        if temp_distance < self.distance_threshold:
                            self.found_goal_times = self.found_goal_times + 1
                    self.found_goal = True
                    self.found_long_goal = False
                
                ## select the closest goal
                direction = temp_direction
                distance = temp_distance
                if distance < shortest_distance:
                    shortest_distance = distance
                    shortest_distance_angle = direction
                    box_shortest = copy.deepcopy(box)
            
            if self.found_goal:
                self.goal_gps = self.get_goal_gps(observations, shortest_distance_angle, shortest_distance)
            elif not long_goal_detected_before:
                # if detected a long goal before, then don't change it until see a goal within 5 meters
                self.long_goal_temp_gps = self.get_goal_gps(observations, shortest_distance_angle, shortest_distance)
            # if self.args.error_analysis and self.found_goal:
            #     if (observations['semantic'][box_shortest[0]:box_shortest[2],box_shortest[1]:box_shortest[3]] == self.goal_mp3d_idx).sum() > min(300, 0.2 * (box_shortest[2]-box_shortest[0])*(box_shortest[3]-box_shortest[1])):
            #          self.detect_true = True
        else:
            if self.found_goal:
                self.found_goal = False
                self.found_goal_times = 0

    def _get_policy_info(self, env: int = 0) -> Dict[str, Any]: # seg_map_color:np.ndarray, # 
        """Get policy info for logging, especially, we add rednet to add seg_map"""
        # 获取目标点云信息

        # 初始化 policy_info
        policy_info = {
            "target_object": self._target_object[env].split("|")[0], # 
        }
        
        return policy_info

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
        **kwargs,
    ): # (self, observations):
        self._num_envs = masks.shape[0]
        self._policy_info = []
        # 创建一个新的字典 observations_numpy
        observations_numpy = {}

        # 遍历 observations 字典，将每个 Tensor 转换为 NumPy 数组
        for key, value in observations.items():
            if isinstance(value, torch.Tensor):  # 确保值是 Tensor
                observations_numpy[key] = value[0].cpu().numpy().copy()  # 转换为 NumPy 数组并创建副本
            else:
                observations_numpy[key] = value[0]  # 如果不是 Tensor，直接赋值

        object_ids = observations_numpy[ObjectGoalSensor.cls_uuid] # .cpu().numpy().flatten()

        # Convert observations to dictionary format
        # obs_dict = observations.to_tree()

        # Loop through each object_id and replace the goal IDs with corresponding names
        
        if self._dataset_type == "hm3d":
            observations_numpy[ObjectGoalSensor.cls_uuid] = [HM3D_ID_TO_NAME[oid.item()] for oid in object_ids]
        elif self._dataset_type == "mp3d":
            observations_numpy[ObjectGoalSensor.cls_uuid] = [MP3D_ID_TO_NAME[oid.item()] for oid in object_ids]
            self._non_coco_caption = " . ".join(MP3D_ID_TO_NAME).replace("|", " . ") + " ."
        else:
            raise ValueError(f"Dataset type {self._dataset_type} not recognized")

        for env in range(self._num_envs):
            if not self._did_reset[env] and masks[env] == 0:
                if len(self.visualize_image_list) > 0:
                    self.save_video()
                self._reset(env)
            if observations_numpy["objectgoal"][env] == "couch":
                self._target_object[env] = "sofa"
            elif observations_numpy["objectgoal"][env] == "potted plant":
                self._target_object[env] = "plant"
            elif observations_numpy["objectgoal"][env] == "tv":
                self._target_object[env] = "tv_monitor" 
            else:
                self._target_object[env] = observations_numpy["objectgoal"][env] 
            self._policy_info.append({})
        # batch_size = masks.shape[0]
            self._policy_info[env].update(self._get_policy_info(env))
            self._did_reset[env] = False
        if self.total_steps >= 500:
            # return {"action": 0}
            actions = torch.tensor([[0]],device=masks.device,)
            use_action = actions
            return PolicyActionData(
                take_actions=actions,
                actions=use_action,
                rnn_hidden_states=None, # rnn_hidden_states,
                policy_info=self._policy_info,
            )
        
        self.total_steps += 1
        if self.navigate_steps == 0:
            # self.obj_goal = projection[int(observations["objectgoal"])]

            self.prob_array_room = self.co_occur_room_mtx[self.goal_idx[self._target_object[env]]]
            self.prob_array_obj = self.co_occur_mtx[self.goal_idx[self._target_object[env]]]

        observations_numpy["depth"] = observations_numpy["depth"] * (self.max_depth - self.min_depth) + self.min_depth
        observations_numpy["depth"][observations_numpy["depth"]==0.5] = 100 # don't construct unprecise map with distance less than 0.5 m
        self.depth = observations_numpy["depth"] # observations_numpy["depth"]
        self.rgb = observations_numpy["rgb"][:,:,[2,1,0]]
        # observations_numpy["rgb_annotated"] = observations_numpy["rgb"]

        self.scenegraph.set_agent(self)
        self.scenegraph.set_navigate_steps(self.navigate_steps)
        self.scenegraph.set_obj_goal(self._target_object[env])
        self.scenegraph.set_room_map(self.room_map)
        self.scenegraph.set_fbe_free_map(self.fbe_free_map)
        self.scenegraph.set_observations(observations_numpy)
        self.scenegraph.set_full_map(self.full_map)
        self.scenegraph.set_full_pose(self.full_pose)
        self.scenegraph.update_scenegraph()
        
        self.update_map(observations_numpy)
        self.update_free_map(observations_numpy)
        
        # look down twice and look around at first to initialize map
        if self.total_steps == 1:
            # look down
            self.sem_map_module.set_view_angles(30)
            self.free_map_module.set_view_angles(30)
            # self.observed_map_module.set_view_angles(30)
            # return {"action": 5}
            actions = torch.tensor([[5]],device=masks.device,)
            use_action = actions
            return PolicyActionData(
                take_actions=actions,
                actions=use_action,
                rnn_hidden_states=None, # rnn_hidden_states,
                policy_info=self._policy_info,
            )
        elif self.total_steps <= 13: # <= 7: # 
            # return {"action": 6}
            actions = torch.tensor([[3]],device=masks.device,) # 3
            use_action = actions
            return PolicyActionData(
                take_actions=actions,
                actions=use_action,
                rnn_hidden_states=None, # rnn_hidden_states,
                policy_info=self._policy_info,
            )
        elif self.total_steps == 14: # == 8: # 
            # look down
            self.sem_map_module.set_view_angles(60)
            self.free_map_module.set_view_angles(60)
            # self.observed_map_module.set_view_angles(60)
            # return {"action": 5}
            actions = torch.tensor([[5]],device=masks.device,)
            use_action = actions
            return PolicyActionData(
                take_actions=actions,
                actions=use_action,
                rnn_hidden_states=None, # rnn_hidden_states,
                policy_info=self._policy_info,
            )
        elif self.total_steps <= 26: # <= 14: # 
            # return {"action": 6}
            actions = torch.tensor([[3]],device=masks.device,) # 3
            use_action = actions
            return PolicyActionData(
                take_actions=actions,
                actions=use_action,
                rnn_hidden_states=None, # rnn_hidden_states,
                policy_info=self._policy_info,
            )
        elif self.total_steps == 27 : # == 15: # 
            self.sem_map_module.set_view_angles(30)
            self.free_map_module.set_view_angles(30)
            # self.observed_map_module.set_view_angles(30)
            # return {"action": 4}
            actions = torch.tensor([[4]],device=masks.device,)
            use_action = actions
            return PolicyActionData(
                take_actions=actions,
                actions=use_action,
                rnn_hidden_states=None, # rnn_hidden_states,
                policy_info=self._policy_info,
            )
        elif self.total_steps == 28 : # == 16: # 
            self.sem_map_module.set_view_angles(0)
            self.free_map_module.set_view_angles(0)
            # self.observed_map_module.set_view_angles(0)
            # return {"action": 4}
            actions = torch.tensor([[4]],device=masks.device,)
            use_action = actions
            return PolicyActionData(
                take_actions=actions,
                actions=use_action,
                rnn_hidden_states=None, # rnn_hidden_states,
                policy_info=self._policy_info,
            )
        # get panoramic view at first
        if self.total_steps <= 40 and not self.found_goal: #   # <= 22
            self.panoramic.append(observations_numpy["rgb"][:,:,[2,1,0]])
            self.panoramic_depth.append(observations_numpy["depth"]) # observations_numpy["depth"]
            self.detect_objects(observations_numpy)
            room_detection_result = self.glip_demo.inference(observations_numpy["rgb"][:,:,[2,1,0]], rooms_captions)
            self.update_room_map(observations_numpy, room_detection_result)
            if not self.found_goal: # if found a goal, directly go to it
                # return {"action": 6}
                actions = torch.tensor([[3]],device=masks.device,) # 3
                use_action = actions
                return PolicyActionData(
                    take_actions=actions,
                    actions=use_action,
                    rnn_hidden_states=None, # rnn_hidden_states,
                    policy_info=self._policy_info,
                )

        if not (observations_numpy["gps"] == self.last_gps).all():
            self.move_steps += 1
            self.not_move_steps = 0
            if self.using_random_goal:
                self.move_since_random += 1
        else:
            self.not_move_steps += 1
            
        self.last_gps = observations_numpy["gps"]
        
        self.scenegraph.perception()
          
        ### ------ generate action using FMM ------ ###
        ## update pose and map
        self.history_pose.append(self.full_pose.cpu().detach().clone())
        input_pose = np.zeros(7)
        input_pose[:3] = self.full_pose.cpu().numpy()
        input_pose[1] = self.map_size_cm/100 - input_pose[1]
        input_pose[2] = -input_pose[2]
        input_pose[4] = self.full_map.shape[-2]
        input_pose[6] = self.full_map.shape[-1]
        traversible, cur_start, cur_start_o = self.get_traversible(self.full_map.cpu().numpy()[0,0,::-1], input_pose)
        
        if self.found_goal: 
            ## directly go to goal
            self.not_use_random_goal()
            self.goal_map = np.zeros(self.full_map.shape[-2:])
            self.goal_map[max(0,min(self.map_size - 1,int(self.map_size_cm/10+self.goal_gps[1]*100/self.resolution))), max(0,min(self.map_size - 1,int(self.map_size_cm/10+self.goal_gps[0]*100/self.resolution)))] = 1
        elif self.found_long_goal: 
            ## go to long goal
            self.not_use_random_goal()
            self.goal_map = np.zeros(self.full_map.shape[-2:])
            self.goal_map[max(0,min(self.map_size - 1,int(self.map_size_cm/10+self.long_goal_temp_gps[1]*100/self.resolution))), max(0,min(self.map_size - 1,int(self.map_size_cm/10+self.long_goal_temp_gps[0]*100/self.resolution)))] = 1
        elif not self.first_fbe: # first FBE process
            self.goal_loc = self.fbe(traversible, cur_start)
            self.not_use_random_goal()
            self.first_fbe = True
            self.goal_map = np.zeros(self.full_map.shape[-2:])
            if self.goal_loc is None:
                self.random_this_ex += 1
                self.goal_map = self.set_random_goal()
                self.using_random_goal = True
            else:
                self.fronter_this_ex += 1
                self.goal_map[self.goal_loc[0], self.goal_loc[1]] = 1
                self.goal_map = self.goal_map[::-1]
        
        stg_y, stg_x, number_action = self._plan(traversible, self.goal_map, self.full_pose, cur_start, cur_start_o, self.found_goal)
        if self.found_long_goal and number_action == 0: # didn't detect goal when arrive at long goal, start over FBE. 
            self.found_long_goal = False
        
        if (not self.found_goal and not self.found_long_goal and number_action == 0) or (self.using_random_goal and self.move_since_random > 20): 
            # FBE if arrive at a selected frontier, or randomly explore for some steps
            self.goal_loc = self.fbe(traversible, cur_start)
            self.not_use_random_goal()
            self.goal_map = np.zeros(self.full_map.shape[-2:])
            if self.goal_loc is None:
                self.random_this_ex += 1
                self.goal_map = self.set_random_goal()
                self.using_random_goal = True
            else:
                self.fronter_this_ex += 1
                self.goal_map[self.goal_loc[0], self.goal_loc[1]] = 1
                self.goal_map = self.goal_map[::-1]
            stg_y, stg_x, number_action = self._plan(traversible, self.goal_map, self.full_pose, cur_start, cur_start_o, self.found_goal)
        
        self.loop_time = 0
        while (not self.found_goal and number_action == 0) or self.not_move_steps >= 13: # 7, 6 * 2 + 1
            # the agent is stuck, then random explore
            self.loop_time += 1
            self.random_this_ex += 1
            self.stuck_time += 1
            if self.loop_time > 20 or self.stuck_time == 5:
                # return {"action": 0}
                actions = torch.tensor([[0]],device=masks.device,)
                use_action = actions
                return PolicyActionData(
                    take_actions=actions,
                    actions=use_action,
                    rnn_hidden_states=None, # rnn_hidden_states,
                    policy_info=self._policy_info,
                )
            self.not_move_steps = 0
            self.goal_map = self.set_random_goal()
            self.using_random_goal = True
            stg_y, stg_x, number_action = self._plan(traversible, self.goal_map, self.full_pose, cur_start, cur_start_o, self.found_goal)
        
        if self.visualize_option:
            self.visualize(traversible, observations_numpy, number_action)

        observations_numpy["pointgoal_with_gps_compass"] = self.get_relative_goal_gps(observations_numpy)

        ###-----------------------------------###

        self.last_loc = copy.deepcopy(self.full_pose)
        self.prev_action = number_action
        self.navigate_steps += 1
        torch.cuda.empty_cache()
        
        actions = torch.tensor([[number_action]],device=masks.device,)
        use_action = actions
        return PolicyActionData(
            take_actions=actions,
            actions=use_action,
            rnn_hidden_states=None, # rnn_hidden_states,
            policy_info=self._policy_info,
        )
    
    def not_use_random_goal(self):
        self.move_since_random = 0
        self.using_random_goal = False
        
    def get_glip_real_label(self, prediction):
        labels = prediction.get_field("labels").tolist()
        new_labels = []
        if self.glip_demo.entities and self.glip_demo.plus:
            for i in labels:
                if i <= len(self.glip_demo.entities):
                    new_labels.append(self.glip_demo.entities[i - self.glip_demo.plus])
                else:
                    new_labels.append('object')
        else:
            new_labels = ['object' for i in labels]
        return new_labels
    
    def fbe(self, traversible, start):
        """
        fontier: unknown area and free area 
        unknown area: not free and not obstacle 
        select a frontier using commonsense and PSL and return a GPS
        """
        fbe_map = torch.zeros_like(self.full_map[0,0])
        fbe_map[self.fbe_free_map[0,0]>0] = 1 # first free 
        fbe_map[skimage.morphology.binary_dilation(self.full_map[0,0].cpu().numpy(), skimage.morphology.disk(4))] = 3 # then dialte obstacle

        fbe_cp = copy.deepcopy(fbe_map)
        fbe_cpp = copy.deepcopy(fbe_map)
        fbe_cp[fbe_cp==0] = 4 # don't know space is 4
        fbe_cp[fbe_cp<4] = 0 # free and obstacle
        selem = skimage.morphology.disk(1)
        fbe_cpp[skimage.morphology.binary_dilation(fbe_cp.cpu().numpy(), selem)] = 0 # don't know space is 0 dialate unknown space
        
        diff = fbe_map - fbe_cpp # intersection between unknown area and free area 
        frontier_map = diff == 1
        frontier_locations = torch.stack([torch.where(frontier_map)[0], torch.where(frontier_map)[1]]).T
        num_frontiers = len(torch.where(frontier_map)[0])
        if num_frontiers == 0:
            return None
        
        # for each frontier, calculate the inverse of distance
        planner = FMMPlanner(traversible, None)
        state = [start[0] + 1, start[1] + 1]
        planner.set_goal(state)
        fmm_dist = planner.fmm_dist[::-1]
        frontier_locations += 1
        frontier_locations = frontier_locations.cpu().numpy()
        distances = fmm_dist[frontier_locations[:,0],frontier_locations[:,1]] / 20
        
        ## use the threshold of 1.6 to filter close frontiers to encourage exploration
        idx_16 = np.where(distances>=1.6)
        distances_16 = distances[idx_16]
        distances_16_inverse = 1 - (np.clip(distances_16,0,11.6)-1.6) / (11.6-1.6)
        frontier_locations_16 = frontier_locations[idx_16]
        self.frontier_locations = frontier_locations
        self.frontier_locations_16 = frontier_locations_16
        if len(distances_16) == 0:
            return None
        num_16_frontiers = len(idx_16[0])  # 175
        # scores = np.zeros((num_16_frontiers))

        scores = self.scenegraph.score(frontier_locations_16, num_16_frontiers)
                

        # select the frontier with highest score
        # if self.args.reasoning == 'both':  # True
        scores += 2 * distances_16_inverse
        # else:
        #     scores += 1 * distances_16_inverse
        idx_16_max = idx_16[0][np.argmax(scores)]
        goal = frontier_locations[idx_16_max] - 1
        self.scores = scores
        return goal
        
    def get_goal_gps(self, observations, angle, distance):
        ### return goal gps in the original agent coordinates
        if type(angle) is torch.Tensor:
            angle = angle.cpu().numpy()
        agent_gps = observations['gps']
        agent_compass = observations['compass']
        goal_direction = agent_compass - angle/180*np.pi
        goal_gps = np.array([(agent_gps[0]+np.cos(goal_direction)*distance).item(),
         (agent_gps[1]-np.sin(goal_direction)*distance).item()])
        return goal_gps

    def get_relative_goal_gps(self, observations, goal_gps=None):
        if goal_gps is None:
            goal_gps = self.goal_gps
        direction_vector = goal_gps - np.array([observations['gps'][0].item(),observations['gps'][1].item()])
        rho = np.sqrt(direction_vector[0]**2 + direction_vector[1]**2)
        phi_world = np.arctan2(direction_vector[1], direction_vector[0])
        agent_compass = observations['compass']
        phi = phi_world - agent_compass
        return np.array([rho, phi.item()], dtype=np.float32)
   
    def init_map(self):
        self.map_size = self.map_size_cm // self.map_resolution
        full_w, full_h = self.map_size, self.map_size
        self.full_map = torch.zeros(1,1 ,full_w, full_h).float().to(self.device)
        self.room_map = torch.zeros(1,9 ,full_w, full_h).float().to(self.device)
        self.visited = self.full_map[0,0].cpu().numpy()
        self.collision_map = self.full_map[0,0].cpu().numpy()
        self.fbe_free_map = copy.deepcopy(self.full_map).to(self.device) # 0 is unknown, 1 is free
        self.full_pose = torch.zeros(3).float().to(self.device)
        # Origin of local map
        self.origins = np.zeros((2))
        
        def init_map_and_pose():
            self.full_map.fill_(0.)
            self.full_pose.fill_(0.)
            # full_pose[:, 2] = 90
            self.full_pose[:2] = self.map_size_cm / 100.0 / 2.0  # put the agent in the middle of the map

        init_map_and_pose()

    def update_map(self, observations):
        """
        full pose: gps and angle in the initial coordinate system, where 0 is towards the x axis
        """
        self.full_pose[0] = self.map_size_cm / 100.0 / 2.0+torch.from_numpy(observations['gps']).to(self.device)[0]
        self.full_pose[1] = self.map_size_cm / 100.0 / 2.0-torch.from_numpy(observations['gps']).to(self.device)[1]
        self.full_pose[2:] = torch.from_numpy(observations['compass'] * 57.29577951308232).to(self.device) # input degrees and meters
        self.full_map = self.sem_map_module(torch.squeeze(torch.from_numpy(observations['depth']), dim=-1).to(self.device), self.full_pose, self.full_map)
    
    def update_free_map(self, observations):
        """
        update free map using visual projection
        """
        self.full_pose[0] = self.map_size_cm / 100.0 / 2.0+torch.from_numpy(observations['gps']).to(self.device)[0]
        self.full_pose[1] = self.map_size_cm / 100.0 / 2.0-torch.from_numpy(observations['gps']).to(self.device)[1]
        self.full_pose[2:] = torch.from_numpy(observations['compass'] * 57.29577951308232).to(self.device) # input degrees and meters
        self.fbe_free_map = self.free_map_module(torch.squeeze(torch.from_numpy(observations['depth']), dim=-1).to(self.device), self.full_pose, self.fbe_free_map)
        self.fbe_free_map[int(self.map_size_cm / 10) - 3:int(self.map_size_cm / 10) + 4, int(self.map_size_cm / 10) - 3:int(self.map_size_cm / 10) + 4] = 1
    
    def update_room_map(self, observations, room_prediction_result):
        new_room_labels = self.get_glip_real_label(room_prediction_result)
        type_mask = np.zeros((9,480,640)) # self.config.SIMULATOR.DEPTH_SENSOR.HEIGHT, self.config.SIMULATOR.DEPTH_SENSOR.WIDTH
        bboxs = room_prediction_result.bbox
        score_vec = torch.zeros((9)).to(self.device)
        for i, box in enumerate(bboxs):
            box = box.to(torch.int64)
            idx = rooms.index(new_room_labels[i])
            type_mask[idx,box[1]:box[3],box[0]:box[2]] = 1
            score_vec[idx] = room_prediction_result.get_field("scores")[i]
        self.room_map = self.room_map_module(torch.squeeze(torch.from_numpy(observations['depth']), dim=-1).to(self.device), self.full_pose, self.room_map, torch.from_numpy(type_mask).to(self.device).type(torch.float32), score_vec)
        # self.room_map_refine = copy.deepcopy(self.room_map)
        # other_room_map_sum = self.room_map_refine[0,0] + torch.sum(self.room_map_refine[0,2:],axis=0)
        # self.room_map_refine[0,1][other_room_map_sum>0] = 0
    
    def get_traversible(self, map_pred, pose_pred):
        """
        update traversible map
        """
        grid = np.rint(map_pred)

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = pose_pred
        gx1, gx2, gy1, gy2  = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        # Get curr loc
        r, c = start_y, start_x
        start = [int(r*100/self.map_resolution - gy1),
                 int(c*100/self.map_resolution - gx1)]
        # start = [int(start_x), int(start_y)]
        start = pu.threshold_poses(start, grid.shape)
        self.visited[gy1:gy2, gx1:gx2][start[0]-2:start[0]+3,
                                       start[1]-2:start[1]+3] = 1
        #Get traversible
        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h+2,w+2)) + value
            new_mat[1:h+1,1:w+1] = mat
            return new_mat
        
        def delete_boundary(mat):
            new_mat = copy.deepcopy(mat)
            return new_mat[1:-1,1:-1]
        
        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = 0, 0
        x2, y2 = grid.shape

        traversible = skimage.morphology.binary_dilation(
                    grid[y1:y2, x1:x2],
                    self.selem) != True

        if not(traversible[start[0], start[1]]):
            print("Not traversible, step is  ", self.navigate_steps)

        # obstacle dilation do not dilate collision
        traversible = 1 - traversible
        selem = skimage.morphology.disk(4)
        traversible = skimage.morphology.binary_dilation(
                        traversible, selem) != True
        
        traversible[int(start[0]-y1)-1:int(start[0]-y1)+2,
            int(start[1]-x1)-1:int(start[1]-x1)+2] = 1
        traversible = traversible * 1.
        
        traversible[self.visited[gy1:gy2, gx1:gx2][y1:y2, x1:x2] == 1] = 1
        traversible[self.collision_map[gy1:gy2, gx1:gx2][y1:y2, x1:x2] == 1] = 0
        traversible = add_boundary(traversible)
        return traversible, start, start_o
    
    def _plan(self, traversible, goal_map, agent_pose, start, start_o, goal_found):
        """Function responsible for planning

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) goal locations
                    'pose_pred' (ndarray): (7,) array  denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                    'found_goal' (bool): whether the goal object is found

        Returns:
            action (int): action id
        """
        # if newly_goal_set:
        #     self.action_5_count = 0

        if self.prev_action == 1:
            x1, y1, t1 = self.last_loc.cpu().numpy()
            x2, y2, t2 = self.full_pose.cpu()
            y1 = self.map_size_cm/100 - y1
            y2 = self.map_size_cm/100 - y2
            t1 = -t1
            t2 = -t2
            buf = 4
            length = 5

            if abs(x1 - x2)< 0.05 and abs(y1 - y2) < 0.05:
                self.col_width += 1
                self.col_width = min(self.col_width, 3)
            else:
                self.col_width = 1
            # self.col_width = 4
            dist = pu.get_l2_distance(x1, x2, y1, y2)
            col_threshold = self.collision_threshold

            if dist < col_threshold: # Collision
                self.former_collide += 1
                width = self.col_width
                for i in range(length):
                    for j in range(width):
                        wx = x1 + 0.05*((i+buf) * np.cos(np.deg2rad(t1)) + \
                                        (j-width//2) * np.sin(np.deg2rad(t1)))
                        wy = y1 + 0.05*((i+buf) * np.sin(np.deg2rad(t1)) - \
                                        (j-width//2) * np.cos(np.deg2rad(t1)))
                        r, c = wy, wx
                        r, c = int(round(r*100/self.map_resolution)), \
                               int(round(c*100/self.map_resolution))
                        [r, c] = pu.threshold_poses([r, c],
                                    self.collision_map.shape)
                        self.collision_map[r,c] = 1
            else:
                self.former_collide = 0

        stg, stop, = self._get_stg(traversible, start, np.copy(goal_map), goal_found)

        # Deterministic Local Policy
        if stop:
            action = 0
            (stg_y, stg_x) = stg

        else:
            (stg_y, stg_x) = stg
            angle_st_goal = math.degrees(math.atan2(stg_y - start[0],
                                                stg_x - start[1]))
            angle_agent = (start_o)%360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_st_goal- angle_agent)%360.0
            if relative_angle > 180:
                relative_angle -= 360
            if self.former_collide < 10:
                if relative_angle > 16:
                    action = 3 # Right
                elif relative_angle < -16:
                    action = 2 # Left
                else:
                    action = 1
            elif self.prev_action == 1:
                if relative_angle > 0:
                    action = 3 # Right
                else:
                    action = 2 # Left
            else:
                action = 1
            if self.former_collide >= 10 and self.prev_action != 1:
                self.former_collide  = 0
            if stg_y == start[0] and stg_x == start[1]:
                action = 1

        return stg_y, stg_x, action
    
    def _get_stg(self, traversible, start, goal, goal_found):
        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h+2,w+2)) + value
            new_mat[1:h+1,1:w+1] = mat
            return new_mat
        
        def delete_boundary(mat):
            new_mat = copy.deepcopy(mat)
            return new_mat[1:-1,1:-1]
        
        goal = add_boundary(goal, value=0)
        original_goal = copy.deepcopy(goal)
        
            
        centers = []
        if len(np.where(goal !=0)[0]) > 1:
            goal, centers = CH._get_center_goal(goal)
        state = [start[0] + 1, start[1] + 1]
        self.planner = FMMPlanner(traversible, None)
            
        if self.dilation_deg!=0: 
            #if self.args.debug_local:
            #    self.print_log("dilation added")
            goal = CH._add_cross_dilation(goal, self.dilation_deg, 3)
            
        if goal_found:
            # if self.args.debug_local:
            #     self.print_log("goal found!")
            try:
                goal = CH._block_goal(centers, goal, original_goal, goal_found)
            except:
                goal = self.set_random_goal(goal)

        self.planner.set_multi_goal(goal, state) # time cosuming 

        decrease_stop_cond =0
        if self.dilation_deg >= 6:
            decrease_stop_cond = 0.2 #decrease to 0.2 (7 grids until closest goal)
        stg_y, stg_x, replan, stop = self.planner.get_short_term_goal(state, found_goal = goal_found, decrease_stop_cond=decrease_stop_cond)
        stg_x, stg_y = stg_x - 1, stg_y - 1
        if stop:
            a = 1
        
        # self.closest_goal = CH._get_closest_goal(start, goal)
        
        return (stg_y, stg_x), stop
    
    def set_random_goal(self):
        """
        return a random goal in the map
        """
        obstacle_map = self.full_map.cpu().numpy()[0,0,::-1]
        goal = np.zeros_like(obstacle_map)
        goal_index = np.where((obstacle_map<1))
        np.random.seed(self.total_steps)
        if len(goal_index[0]) != 0:
            i = np.random.choice(len(goal_index[0]), 1)[0]
            h_goal = goal_index[0][i]
            w_goal = goal_index[1][i]
        else:
            h_goal = np.random.choice(goal.shape[0], 1)[0]
            w_goal = np.random.choice(goal.shape[1], 1)[0]
        goal[h_goal, w_goal] = 1
        return goal
    
    def update_metrics(self, metrics):
        self.metrics['distance_to_goal'] = metrics['distance_to_goal']
        self.metrics['spl'] = metrics['spl']
        self.metrics['softspl'] = metrics['softspl']
        # if self.args.visualize:
        #     if self.simulator._env.episode_over or self.total_steps == 500:
        # if self.total_steps == 40:
        #     self.save_video()

    def visualize(self, traversible, observations, number_action, env = 0):
        # if self.args.visualize:
        save_map = copy.deepcopy(torch.from_numpy(traversible))
        gray_map = torch.stack((save_map, save_map, save_map))
        paper_obstacle_map = copy.deepcopy(gray_map)[:,1:-1,1:-1]
        paper_map = torch.zeros_like(paper_obstacle_map)
        paper_map_trans = paper_map.permute(1,2,0)
        unknown_rgb = colors.to_rgb('#FFFFFF')
        paper_map_trans[:,:,:] = torch.tensor( unknown_rgb)
        free_rgb = colors.to_rgb('#E7E7E7')
        paper_map_trans[self.fbe_free_map.cpu().numpy()[0,0,::-1]>0.5,:] = torch.tensor( free_rgb).double()
        obstacle_rgb = colors.to_rgb('#A2A2A2')
        paper_map_trans[skimage.morphology.binary_dilation(self.full_map.cpu().numpy()[0,0,::-1]>0.5,skimage.morphology.disk(1)),:] = torch.tensor(obstacle_rgb).double()
        paper_map_trans = paper_map_trans.permute(2,0,1)
        self.visualize_agent_and_goal(paper_map_trans)
        agent_coordinate = (int(self.history_pose[-1][0]*100/self.resolution), int((self.map_size_cm/100-self.history_pose[-1][1])*100/self.resolution))
        occupancy_map = crop_around_point((paper_map_trans.permute(1, 2, 0) * 255).numpy().astype(np.uint8), agent_coordinate, (150, 200))
        visualize_image = np.full((450, 800, 3), 255, dtype=np.uint8)
        visualize_image = add_resized_image(visualize_image, observations["rgb"], (10, 60), (320, 240))
        visualize_image = add_resized_image(visualize_image, occupancy_map, (340, 60), (180, 240))
        visualize_image = add_rectangle(visualize_image, (10, 60), (330, 300), (128, 128, 128), thickness=1)
        visualize_image = add_rectangle(visualize_image, (340, 60), (520, 300), (128, 128, 128), thickness=1)
        visualize_image = add_rectangle(visualize_image, (540, 60), (790, 165), (128, 128, 128), thickness=1)
        visualize_image = add_rectangle(visualize_image, (540, 195), (790, 300), (128, 128, 128), thickness=1)
        visualize_image = add_rectangle(visualize_image, (10, 350), (790, 400), (128, 128, 128), thickness=1)
        visualize_image = add_text(visualize_image, f"Observation (Goal: {self._target_object[env]})", (70, 50), font_scale=0.5, thickness=1)
        visualize_image = add_text(visualize_image, "Occupancy Map", (370, 50), font_scale=0.5, thickness=1)
        visualize_image = add_text(visualize_image, "Scene Graph Nodes", (580, 50), font_scale=0.5, thickness=1)
        visualize_image = add_text(visualize_image, "Scene Graph Edges", (580, 185), font_scale=0.5, thickness=1)
        visualize_image = add_text(visualize_image, "LLM Explanation", (330, 340), font_scale=0.5, thickness=1)
        visualize_image = add_text_list(visualize_image, line_list(self.text_node, 40), (550, 80), font_scale=0.3, thickness=1)
        visualize_image = add_text_list(visualize_image, line_list(self.text_edge, 40), (550, 215), font_scale=0.3, thickness=1)
        visualize_image = add_text_list(visualize_image, line_list(self.explanation, 150), (20, 370), font_scale=0.3, thickness=1)
        visualize_image = visualize_image[:, :, ::-1]
        visualize_image_rgb = cv2.cvtColor(visualize_image, cv2.COLOR_BGR2RGB) # for imageio
        self.visualize_image_list.append(visualize_image_rgb) # visualize_image

    def save_video(self):
        save_video_dir = os.path.join(self.visualization_dir, 'video')
        save_video_path = f'{save_video_dir}/vid_{self.count_episodes:06d}.mp4'
        if not os.path.exists(save_video_dir):
            os.makedirs(save_video_dir)
        # 使用 imageio 保存视频
        writer = imageio.get_writer(save_video_path, fps=4.0)
        for visualize_image in self.visualize_image_list:
            writer.append_data(visualize_image)
        writer.close()
        # height, width, layers = self.visualize_image_list[0].shape
        # fourcc = cv2.VideoWriter_fourcc(*'avc1')# mp4v
        # video = cv2.VideoWriter(save_video_path, fourcc, 4.0, (width, height))
        # for visualize_image in self.visualize_image_list:  
        #     video.write(visualize_image)
        # video.release()

    def visualize_agent_and_goal(self, map):
        for idx, pose in enumerate(self.history_pose):
            draw_step_num = 30
            alpha = max(0, 1 - (len(self.history_pose) - idx) / draw_step_num)
            agent_size = 1
            if idx == len(self.history_pose) - 1:
                agent_size = 2
            draw_agent(agent=self, map=map, pose=pose, agent_size=agent_size, color_index=0, alpha=alpha)
        draw_goal(agent=self, map=map, goal_size=2, color_index=1)
        return map

    @classmethod
    def from_config(cls, config: DictConfig, *args_unused: Any, **kwargs_unused: Any) -> "HabitatMixin":
        policy_config: VLFMPolicyConfig = config.habitat_baselines.rl.policy[kwargs_unused['agent_name']] # For Habitat 3.0
        kwargs = {k: policy_config[k] for k in VLFMPolicyConfig.kwaarg_names}  # type: ignore
        ## adapt to multi envs 
        kwargs["num_envs"] = config.habitat_baselines.num_environments
        # In habitat, we need the height of the camera to generate the camera transform

        # sim_sensors_cfg = config.habitat.simulator.agents.main_agent.sim_sensors
        if 'main_agent' in config.habitat.simulator.agents:
            sim_sensors_cfg = config.habitat.simulator.agents['main_agent'].sim_sensors
            kwargs["camera_height"] = sim_sensors_cfg.rgb_sensor.position[1]

            # Synchronize the mapping min/max depth values with the habitat config
            kwargs["min_depth"] = sim_sensors_cfg.depth_sensor.min_depth
            kwargs["max_depth"] = sim_sensors_cfg.depth_sensor.max_depth
            kwargs["camera_fov"] = sim_sensors_cfg.depth_sensor.hfov
            kwargs["image_width"] = sim_sensors_cfg.depth_sensor.width
            kwargs["image_height"] = sim_sensors_cfg.depth_sensor.height # added
        elif 'agent_0' in config.habitat.simulator.agents:
            if config.habitat.simulator.agents['agent_0'].articulated_agent_type == "SpotRobot":
                sim_sensors_cfg = config.habitat.simulator.agents['agent_0'].sim_sensors
                kwargs["camera_height"] = sim_sensors_cfg.jaw_rgb_sensor.position[1]

                # Synchronize the mapping min/max depth values with the habitat config
                kwargs["min_depth"] = sim_sensors_cfg.jaw_depth_sensor.min_depth
                kwargs["max_depth"] = sim_sensors_cfg.jaw_depth_sensor.max_depth
                kwargs["camera_fov"] = sim_sensors_cfg.jaw_depth_sensor.hfov
                kwargs["image_width"] = sim_sensors_cfg.jaw_depth_sensor.width
                kwargs["image_height"] = sim_sensors_cfg.jaw_depth_sensor.height # added
            else:
                sim_sensors_cfg = config.habitat.simulator.agents['agent_0'].sim_sensors
                kwargs["camera_height"] = sim_sensors_cfg.rgb_sensor.position[1]

                # Synchronize the mapping min/max depth values with the habitat config
                kwargs["min_depth"] = sim_sensors_cfg.depth_sensor.min_depth
                kwargs["max_depth"] = sim_sensors_cfg.depth_sensor.max_depth
                kwargs["camera_fov"] = sim_sensors_cfg.depth_sensor.hfov
                kwargs["image_width"] = sim_sensors_cfg.depth_sensor.width
                kwargs["image_height"] = sim_sensors_cfg.depth_sensor.height # added
            # for spot
            kwargs["agent_radius"] = config.habitat.simulator.agents['agent_0'].radius
        else:
            raise ValueError("No agent found in configuration.")

        # Only bother visualizing if we're actually going to save the video
        kwargs["visualize"] = len(config.habitat_baselines.eval.video_option) > 0

        # For Habitat 3.0
        kwargs["action_space"]= args_unused[-1]

        if "hm3d" in config.habitat.dataset.data_path:
            kwargs["dataset_type"] = "hm3d"
        elif "mp3d" in config.habitat.dataset.data_path:
            kwargs["dataset_type"] = "mp3d"
        else:
            raise ValueError("Dataset type could not be inferred from habitat config")
        # config added
        kwargs["full_config"] = config
        return cls(**kwargs)